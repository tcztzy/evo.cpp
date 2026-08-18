#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile pinned NT/Agro-NT HF checkpoints for the native GENEB ESM runtime."""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if not (_SCRIPT_DIRECTORY / "evo").is_dir():
    _INSTALLED_PYTHON = _SCRIPT_DIRECTORY.parent / "share" / "evo" / "python"
    if _INSTALLED_PYTHON.is_dir():
        sys.path.insert(0, str(_INSTALLED_PYTHON))

from evo.hf_checkpoint import (  # noqa: E402
    CheckpointError,
    TensorSpec,
    load_hf_torch_checkpoints,
    load_json,
    load_json_bytes,
    normalized_relative_path,
    read_hf_safetensors,
    select_hf_safetensors_paths,
    select_hf_torch_paths,
    validate_tensor_manifest,
)
from evo.geneb_artifact import (  # noqa: E402
    GenebArtifactError,
    build_geneb_artifact_metadata,
    catalog_contract_sha256,
    converter_profile_contract_sha256,
    validate_hf_fetch_receipt_provenance,
)


ARTIFACT_PROFILE = "geneb-esm-runtime-v1"
RUNTIME_ABI = "geneb-esm-safetensors-v1"
RUNTIME_ARCHITECTURE = "GenebEsmEncoder"
PROFILE_FORMAT = "geneb-esm-converter-v1"
TOKENIZER_PROFILE = "evo-tokenizer-v1"
OMIT_POLICY = "validate-heads-and-unused-buffers-then-omit"
AGRO_LAYER_NORM_KERNEL = "torch-2.1.2-apple-arm64-exact-v1"
NT_2_5B_RUNTIME_ID = "geneb-nt-2-5b-ms"
NT_2_5B_TOKENIZER_PINS = {
    "tokenizer_manifest_sha256": "a19b30c4284c792be67a79dba336156d0cdebd2b2f4d57ea4d1a7cc62975cb26",
    "tokenizer_source_receipt_contract_sha256": "3c9bc95e63930dccd733d81e51e75522475127b08ea080468575742882c22662",
    "tokenizer_asset_sha256": "f574aad91e382548e0bc6f87fefd0b2022fcfeaf612c7c90be5c133f74a75794",
    "tokenizer_asset_size": 118259,
}
MAX_HEADER_SIZE = 16 * 1024 * 1024
CHUNK_SIZE = 16 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
KEY_RE = re.compile(r"[A-Za-z0-9._-]+")

PROFILE_KEYS = {
    "runtime_id",
    "geneb_model_id",
    "paper_name",
    "catalog_architecture",
    "repo",
    "revision",
    "source_format",
    "config_sha256",
    "source_variant",
    "source_position_ids",
    "source_rotary_inv_freq",
    "omit_policy",
    "tokenizer_k",
    "config_required",
    "topology",
}
PROFILE_OPTIONAL_KEYS = {
    "config_duplicate_keys",
    "layer_norm_kernel",
    "source_tensor_layout",
    "tokenizer_manifest_sha256",
    "tokenizer_source_receipt_contract_sha256",
    "tokenizer_asset_sha256",
    "tokenizer_asset_size",
}
TOPOLOGY_KEYS = {
    "vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "head_dim",
    "intermediate_size",
    "max_seqlen",
    "max_position_embeddings",
    "layer_norm_epsilon",
    "rope_base",
    "position_embedding_type",
    "mlp_activation",
    "attention_bias",
    "feed_forward_bias",
    "token_dropout",
    "pad_token_id",
    "mask_token_id",
    "cls_token_id",
}
TOKENIZER_DESCRIPTOR_KEYS = {
    "converter.schema",
    "converter.version",
    "compiler_manifest_sha256",
    "source_receipt_contract_sha256",
    "tokenizer.profile",
    "tokenizer.path",
    "tokenizer.sha256",
    "tokenizer.size",
}


class ConversionError(ValueError):
    """Raised when conversion input differs from the frozen ESM contract."""


class _JsonObjectPairs(list):
    """Distinguish JSON objects from arrays while auditing duplicate keys."""


def exact_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str],
    label: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    actual = set(value)
    if not required_set <= actual or not actual <= allowed:
        raise ConversionError(
            "%s fields differ: missing=%s extra=%s"
            % (label, sorted(required_set - actual), sorted(actual - allowed))
        )


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ConversionError("%s must be a nonempty string" % label)
    return value


def uint64_value(value: Any, label: str, positive: bool) -> int:
    minimum = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > UINT64_MAX
    ):
        raise ConversionError(
            "%s must be a%s uint64" % (label, " positive" if positive else "")
        )
    return value


def finite_float(value: Any, label: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionError("%s must be finite numeric metadata" % label)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ConversionError(
            "%s must be finite%s" % (label, " and positive" if positive else "")
        )
    return result


def validate_config_duplicate_keys(raw: Any, label: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict) or not raw or len(raw) > 8:
        raise ConversionError("%s must be a nonempty bounded object" % label)
    result = {}  # type: Dict[str, Dict[str, Any]]
    for raw_key, raw_rule in raw.items():
        key = nonempty_string(raw_key, label + " key")
        if not isinstance(raw_rule, dict) or set(raw_rule) != {"count", "value"}:
            raise ConversionError(
                "%s[%s] fields must be exactly count/value" % (label, key)
            )
        count = uint64_value(raw_rule["count"], "%s[%s].count" % (label, key), True)
        if count < 2 or count > 16:
            raise ConversionError("%s[%s].count must be in [2,16]" % (label, key))
        value = raw_rule["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConversionError("%s[%s].value must be finite numeric" % (label, key))
        expected = finite_float(value, "%s[%s].value" % (label, key), False)
        result[key] = {"count": count, "value": expected}
    return result


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ConversionError("cannot hash %s: %s" % (path, error))
    return digest.hexdigest()


def default_config_path(name: str) -> Path:
    source = _SCRIPT_DIRECTORY.parent / "configs" / name
    if source.is_file():
        return source
    return _SCRIPT_DIRECTORY.parent / "share" / "evo" / "configs" / name


def validate_topology(raw: Any, label: str) -> Dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != TOPOLOGY_KEYS:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise ConversionError(
            "%s fields differ: missing=%s extra=%s"
            % (label, sorted(TOPOLOGY_KEYS - actual), sorted(actual - TOPOLOGY_KEYS))
        )
    topology = dict(raw)
    for key in (
        "vocab_size",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "head_dim",
        "intermediate_size",
        "max_seqlen",
        "max_position_embeddings",
    ):
        topology[key] = uint64_value(topology[key], "%s.%s" % (label, key), True)
    for key in ("pad_token_id", "mask_token_id", "cls_token_id"):
        topology[key] = uint64_value(topology[key], "%s.%s" % (label, key), False)
    topology["layer_norm_epsilon"] = finite_float(
        topology["layer_norm_epsilon"], label + ".layer_norm_epsilon", True
    )
    topology["rope_base"] = finite_float(
        topology["rope_base"], label + ".rope_base", True
    )
    if topology["position_embedding_type"] not in ("absolute", "rotary"):
        raise ConversionError("%s.position_embedding_type is unsupported" % label)
    if topology["mlp_activation"] not in ("gelu", "swiglu"):
        raise ConversionError("%s.mlp_activation is unsupported" % label)
    for key in ("attention_bias", "feed_forward_bias", "token_dropout"):
        if not isinstance(topology[key], bool):
            raise ConversionError("%s.%s must be boolean" % (label, key))
    if topology["num_attention_heads"] * topology["head_dim"] != topology["hidden_size"]:
        raise ConversionError("%s attention width differs from hidden size" % label)
    if topology["position_embedding_type"] == "rotary" and topology["head_dim"] % 2:
        raise ConversionError("%s rotary head_dim must be even" % label)
    if topology["max_position_embeddings"] <= topology["pad_token_id"] + topology["max_seqlen"]:
        raise ConversionError("%s absolute position cumsum range is too small" % label)
    special_ids = {
        topology["pad_token_id"],
        topology["mask_token_id"],
        topology["cls_token_id"],
    }
    if len(special_ids) != 3 or any(item >= topology["vocab_size"] for item in special_ids):
        raise ConversionError("%s special IDs are invalid" % label)
    if topology["rope_base"] <= 1.0:
        raise ConversionError("%s.rope_base must exceed one" % label)
    return topology


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    root, payload = load_json(path, "GENEB ESM profile manifest")
    exact_keys(root, ["schema_version", "format", "models"], [], "profile manifest")
    if root["schema_version"] != 1 or root["format"] != PROFILE_FORMAT:
        raise ConversionError("GENEB ESM profile manifest schema is unsupported")
    if not isinstance(root["models"], list) or not root["models"]:
        raise ConversionError("profile manifest models must be nonempty")
    profiles = {}  # type: Dict[str, Dict[str, Any]]
    identities = set()  # type: set
    for index, raw in enumerate(root["models"]):
        label = "profile manifest models[%d]" % index
        if (
            not isinstance(raw, dict)
            or not PROFILE_KEYS <= set(raw)
            or not set(raw) <= PROFILE_KEYS | PROFILE_OPTIONAL_KEYS
        ):
            actual = set(raw) if isinstance(raw, dict) else set()
            raise ConversionError(
                "%s fields differ: missing=%s extra=%s"
                % (
                    label,
                    sorted(PROFILE_KEYS - actual),
                    sorted(actual - PROFILE_KEYS - PROFILE_OPTIONAL_KEYS),
                )
            )
        profile = dict(raw)
        runtime_id = nonempty_string(profile["runtime_id"], label + ".runtime_id")
        if runtime_id in profiles:
            raise ConversionError("profile manifest contains duplicate runtime_id")
        for key in (
            "geneb_model_id",
            "paper_name",
            "catalog_architecture",
            "repo",
            "revision",
            "config_sha256",
            "source_variant",
            "omit_policy",
        ):
            profile[key] = nonempty_string(profile[key], "%s.%s" % (label, key))
        if not REPO_RE.fullmatch(profile["repo"]):
            raise ConversionError("%s.repo is not OWNER/NAME" % label)
        if not COMMIT_RE.fullmatch(profile["revision"]):
            raise ConversionError("%s.revision is not an immutable commit" % label)
        if not SHA256_RE.fullmatch(profile["config_sha256"]):
            raise ConversionError("%s.config_sha256 is not lowercase SHA256" % label)
        if "tokenizer_manifest_sha256" in profile and (
            not isinstance(profile["tokenizer_manifest_sha256"], str)
            or not SHA256_RE.fullmatch(profile["tokenizer_manifest_sha256"])
        ):
            raise ConversionError(
                "%s.tokenizer_manifest_sha256 is not lowercase SHA256" % label
            )
        for key in (
            "tokenizer_source_receipt_contract_sha256",
            "tokenizer_asset_sha256",
        ):
            if key in profile and (
                not isinstance(profile[key], str)
                or not SHA256_RE.fullmatch(profile[key])
            ):
                raise ConversionError("%s.%s is not lowercase SHA256" % (label, key))
        if "tokenizer_asset_size" in profile:
            profile["tokenizer_asset_size"] = uint64_value(
                profile["tokenizer_asset_size"], label + ".tokenizer_asset_size", True
            )
        if "config_duplicate_keys" in profile:
            profile["config_duplicate_keys"] = validate_config_duplicate_keys(
                profile["config_duplicate_keys"], label + ".config_duplicate_keys"
            )
        source_tensor_layout = profile.get("source_tensor_layout", "contiguous")
        if source_tensor_layout not in (
            "contiguous",
            "contiguous-or-exact-column-major",
        ):
            raise ConversionError("%s.source_tensor_layout is unsupported" % label)
        profile["source_tensor_layout"] = source_tensor_layout
        pinned_descriptor_keys = set(NT_2_5B_TOKENIZER_PINS)
        if runtime_id == NT_2_5B_RUNTIME_ID:
            actual_pins = {
                key: profile.get(key) for key in NT_2_5B_TOKENIZER_PINS
            }
            if actual_pins != NT_2_5B_TOKENIZER_PINS:
                raise ConversionError(
                    "%s tokenizer descriptor pins differ from the exact NT-2.5B contract"
                    % label
                )
            if source_tensor_layout != "contiguous-or-exact-column-major":
                raise ConversionError(
                    "%s must opt into the audited NT-2.5B column-major layout"
                    % label
                )
        elif any(
            key in profile
            for key in pinned_descriptor_keys - {"tokenizer_manifest_sha256"}
        ):
            raise ConversionError(
                "%s exact tokenizer descriptor pins are only valid for %s"
                % (label, NT_2_5B_RUNTIME_ID)
            )
        if profile["source_format"] not in (
            "safetensors",
            "pytorch-bin",
            "pytorch-bin-sharded",
        ):
            raise ConversionError("%s.source_format is unsupported" % label)
        if profile["source_variant"] not in (
            "hf-esm-gelu",
            "instadeep-esm-swiglu",
        ):
            raise ConversionError("%s.source_variant is unsupported" % label)
        if profile["omit_policy"] != OMIT_POLICY:
            raise ConversionError("%s omit policy differs" % label)
        for key in ("source_position_ids", "source_rotary_inv_freq"):
            if not isinstance(profile[key], bool):
                raise ConversionError("%s.%s must be boolean" % (label, key))
        profile["tokenizer_k"] = uint64_value(
            profile["tokenizer_k"], label + ".tokenizer_k", True
        )
        if profile["tokenizer_k"] not in (3, 6):
            raise ConversionError("%s tokenizer_k must be 3 or 6" % label)
        if not isinstance(profile["config_required"], dict) or not profile["config_required"]:
            raise ConversionError("%s.config_required must be nonempty" % label)
        profile["topology"] = validate_topology(profile["topology"], label + ".topology")
        topology = profile["topology"]
        layer_norm_kernel = profile.get("layer_norm_kernel")
        if runtime_id == "geneb-agro-nt-1b":
            if layer_norm_kernel != AGRO_LAYER_NORM_KERNEL:
                raise ConversionError(
                    "%s.layer_norm_kernel must be the exact Agro-NT-1B contract"
                    % label
                )
            if (
                topology["hidden_size"] != 1500
                or topology["layer_norm_epsilon"] != 1.0e-12
            ):
                raise ConversionError(
                    "%s exact LayerNorm topology must be width=1500/eps=1e-12"
                    % label
                )
        elif layer_norm_kernel is not None:
            raise ConversionError(
                "%s.layer_norm_kernel is only valid for geneb-agro-nt-1b"
                % label
            )
        is_v2 = profile["source_variant"] == "instadeep-esm-swiglu"
        if (
            is_v2 != (topology["mlp_activation"] == "swiglu")
            or is_v2 != (topology["position_embedding_type"] == "rotary")
            or is_v2 != profile["source_rotary_inv_freq"]
            or is_v2 == topology["feed_forward_bias"]
            or profile["source_position_ids"]
            and profile["source_variant"] != "hf-esm-gelu"
        ):
            raise ConversionError("%s source variant/topology flags disagree" % label)
        identity = (profile["repo"], profile["revision"])
        if identity in identities:
            raise ConversionError("profile manifest duplicates repo/revision")
        identities.add(identity)
        profiles[runtime_id] = profile
    return profiles, payload


def load_catalog_entry(
    path: Path, profile: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    root, payload = load_json(path, "GENEB catalog")
    if root.get("schema_version") != 1 or not isinstance(root.get("models"), list):
        raise ConversionError("GENEB catalog schema is unsupported")
    matches = [
        item
        for item in root["models"]
        if isinstance(item, dict) and item.get("runtime_id") == profile["runtime_id"]
    ]
    if len(matches) != 1:
        raise ConversionError("GENEB catalog must contain exactly one runtime profile")
    entry = matches[0]
    expected = {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "architecture": profile["catalog_architecture"],
        "family": "transformer-encoder",
    }
    wrong = {
        key: (entry.get(key), value)
        for key, value in expected.items()
        if entry.get(key) != value
    }
    if wrong:
        raise ConversionError("GENEB catalog ESM identity differs: %s" % wrong)
    source = entry.get("source")
    if (
        not isinstance(source, dict)
        or source.get("kind") != "huggingface"
        or source.get("repo") != profile["repo"]
        or source.get("revision") != profile["revision"]
        or not isinstance(source.get("requested_revision"), str)
        or not source.get("requested_revision")
        or source.get("immutable") is not True
    ):
        raise ConversionError("GENEB catalog repo/revision differs from ESM profile")
    topology = profile["topology"]
    context = entry.get("context")
    tokenizer = entry.get("tokenizer")
    presets = entry.get("embedding_presets")
    fixed_pad = entry.get("input_transform", {}).get("fixed_pad")
    if (
        not isinstance(context, dict)
        or context.get("declared_max_tokens") != topology["max_position_embeddings"]
        or context.get("reference_max_tokens") != topology["max_seqlen"]
        or context.get("length_policy") != "tokenizer-truncate"
        or not isinstance(tokenizer, dict)
        or tokenizer.get("kind") != "k-mer"
        or tokenizer.get("max_tokens") != topology["max_seqlen"]
        or tokenizer.get("add_special_tokens") is not True
        or tokenizer.get("padding_side") != "right"
        or tokenizer.get("pad_to") != "model-max"
        or not isinstance(fixed_pad, dict)
        or fixed_pad.get("length") != topology["max_seqlen"]
        or fixed_pad.get("side") != "right"
        or fixed_pad.get("value") != "tokenizer-pad"
        or not isinstance(presets, dict)
    ):
        raise ConversionError("GENEB catalog context/tokenizer transform differs")
    catalog_k = tokenizer.get("k", 6)
    if catalog_k != profile["tokenizer_k"]:
        raise ConversionError("GENEB catalog k-mer size differs from ESM profile")
    for preset_name in ("reference", "normalized"):
        preset = presets.get(preset_name)
        if (
            not isinstance(preset, dict)
            or preset.get("hidden_tap") != "last-hidden-state"
            or preset.get("pooling") != "attention-mask-mean"
            or preset.get("special_tokens") != "include"
            or preset.get("mask_domain") != "attention-mask"
            or preset.get("output_width") != topology["hidden_size"]
        ):
            raise ConversionError("GENEB catalog embedding preset differs")
    return entry, root, payload


def validate_receipt(
    path: Path,
    profile: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
    catalog_path: Path,
    catalog_payload: bytes,
) -> Tuple[Dict[str, Path], bytes]:
    receipt, payload = load_json(path, "source receipt")
    exact_keys(
        receipt,
        [
            "schema_version",
            "kind",
            "model_id",
            "repo",
            "requested_revision",
            "resolved_revision",
            "files",
        ],
        ["load_path", "source_kind", "catalog_path", "catalog_sha256", "catalog_contract_sha256"],
        "source receipt",
    )
    validate_hf_fetch_receipt_provenance(
        receipt, catalog_path, catalog_payload, catalog_entry
    )
    catalog_source = catalog_entry.get("source")
    requested_revision = (
        catalog_source.get("requested_revision")
        if isinstance(catalog_source, dict)
        else None
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != profile["runtime_id"]
        or receipt["repo"] != profile["repo"]
        or receipt["requested_revision"] != requested_revision
        or receipt["resolved_revision"] != profile["revision"]
        or ("load_path" in receipt and receipt["load_path"] is not None)
    ):
        raise ConversionError("source receipt model/repository/revision is not pinned")
    files = receipt["files"]
    if not isinstance(files, list) or not files:
        raise ConversionError("source receipt files must be nonempty")
    paths = {}  # type: Dict[str, Path]
    for index, raw in enumerate(files):
        label = "source receipt files[%d]" % index
        if not isinstance(raw, dict):
            raise ConversionError("%s must be an object" % label)
        exact_keys(raw, ["name", "size", "sha256", "path"], [], label)
        name = normalized_relative_path(raw["name"], label + ".name")
        if name in paths:
            raise ConversionError("%s name must be unique" % label)
        size = uint64_value(raw["size"], label + ".size", False)
        digest = nonempty_string(raw["sha256"], label + ".sha256")
        if not SHA256_RE.fullmatch(digest):
            raise ConversionError("%s.sha256 must be lowercase SHA256" % label)
        source_path = Path(nonempty_string(raw["path"], label + ".path")).resolve()
        try:
            actual_size = source_path.stat().st_size
        except OSError as error:
            raise ConversionError("cannot stat receipt asset %s: %s" % (name, error))
        actual_digest = sha256_file(source_path)
        if actual_size != size or actual_digest != digest:
            raise ConversionError(
                "source receipt integrity mismatch for %s: size=%d sha256=%s"
                % (name, actual_size, actual_digest)
            )
        paths[name] = source_path
    return paths, payload


def validate_tokenizer_descriptor(
    descriptor_path: Path,
    tokenizer_root: Optional[Path],
    artifact_root: Path,
    profile: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    descriptor, payload = load_json(descriptor_path, "tokenizer descriptor")
    if set(descriptor) != TOKENIZER_DESCRIPTOR_KEYS:
        raise ConversionError(
            "tokenizer descriptor fields differ: missing=%s extra=%s"
            % (
                sorted(TOKENIZER_DESCRIPTOR_KEYS - set(descriptor)),
                sorted(set(descriptor) - TOKENIZER_DESCRIPTOR_KEYS),
            )
        )
    if (
        descriptor["converter.schema"] != "evo-tokenizer-conversion-receipt"
        or descriptor["converter.version"] != 1
        or descriptor["tokenizer.profile"] != TOKENIZER_PROFILE
    ):
        raise ConversionError("tokenizer descriptor schema/profile is unsupported")
    for key in (
        "compiler_manifest_sha256",
        "source_receipt_contract_sha256",
        "tokenizer.sha256",
    ):
        if not isinstance(descriptor[key], str) or not SHA256_RE.fullmatch(descriptor[key]):
            raise ConversionError("tokenizer descriptor %s is not lowercase SHA256" % key)
    expected_compiler_manifest_sha256 = profile.get("tokenizer_manifest_sha256")
    if expected_compiler_manifest_sha256 is not None and (
        descriptor["compiler_manifest_sha256"]
        != expected_compiler_manifest_sha256
    ):
        raise ConversionError("tokenizer compiler manifest SHA256 differs")
    if profile["runtime_id"] == NT_2_5B_RUNTIME_ID:
        expected_descriptor = {
            "compiler_manifest_sha256": profile["tokenizer_manifest_sha256"],
            "source_receipt_contract_sha256": profile[
                "tokenizer_source_receipt_contract_sha256"
            ],
            "tokenizer.sha256": profile["tokenizer_asset_sha256"],
            "tokenizer.size": profile["tokenizer_asset_size"],
        }
        wrong = {
            key: (descriptor.get(key), value)
            for key, value in expected_descriptor.items()
            if descriptor.get(key) != value
        }
        if wrong:
            raise ConversionError(
                "tokenizer descriptor differs from pinned NT-2.5B profile: %s"
                % wrong
            )
    relative = normalized_relative_path(descriptor["tokenizer.path"], "tokenizer.path")
    size = uint64_value(descriptor["tokenizer.size"], "tokenizer.size", False)
    root = (tokenizer_root if tokenizer_root is not None else descriptor_path.parent).resolve()
    asset = (root / relative).resolve()
    try:
        asset.relative_to(root)
    except ValueError:
        raise ConversionError("tokenizer.path escapes tokenizer root")
    if (artifact_root.resolve() / relative).resolve() != asset:
        raise ConversionError(
            "tokenizer asset must already be staged relative to runtime artifact"
        )
    try:
        actual_size = asset.stat().st_size
    except OSError as error:
        raise ConversionError("cannot stat tokenizer asset: %s" % error)
    actual_digest = sha256_file(asset)
    if actual_size != size or actual_digest != descriptor["tokenizer.sha256"]:
        raise ConversionError(
            "tokenizer asset integrity mismatch: size=%d sha256=%s"
            % (actual_size, actual_digest)
        )
    projected = {
        key: descriptor[key]
        for key in (
            "tokenizer.profile",
            "tokenizer.path",
            "tokenizer.sha256",
            "tokenizer.size",
        )
    }
    return projected, sha256_bytes(payload)


def _materialize_config_json(
    value: Any,
    duplicate_policy: Mapping[str, Mapping[str, Any]],
    label: str,
    root: bool,
) -> Any:
    if isinstance(value, _JsonObjectPairs):
        grouped = {}  # type: Dict[str, List[Any]]
        for key, item in value:
            grouped.setdefault(key, []).append(
                _materialize_config_json(
                    item, duplicate_policy, "%s.%s" % (label, key), False
                )
            )
        result = {}  # type: Dict[str, Any]
        for key, items in grouped.items():
            if len(items) > 1:
                rule = duplicate_policy.get(key) if root else None
                if rule is None:
                    raise ConversionError(
                        "%s contains undeclared duplicate key %r" % (label, key)
                    )
                expected = rule["value"]
                if len(items) != rule["count"] or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or float(item) != expected
                    for item in items
                ):
                    raise ConversionError(
                        "%s duplicate key %r count/value differs from profile"
                        % (label, key)
                    )
            result[key] = items[-1]
        if root:
            missing = sorted(
                key
                for key, rule in duplicate_policy.items()
                if len(grouped.get(key, [])) != rule["count"]
            )
            if missing:
                raise ConversionError(
                    "%s declared duplicate keys differ: %s" % (label, missing)
                )
        return result
    if isinstance(value, list):
        return [
            _materialize_config_json(
                item, duplicate_policy, "%s[%d]" % (label, index), False
            )
            for index, item in enumerate(value)
        ]
    return value


def load_profile_config_json(
    payload: bytes, duplicate_policy: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    try:
        parsed = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_JsonObjectPairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversionError("cannot parse HF config: %s" % error)
    if not isinstance(parsed, _JsonObjectPairs):
        raise ConversionError("HF config root must be an object")
    value = _materialize_config_json(parsed, duplicate_policy, "HF config", True)
    if not isinstance(value, dict):
        raise ConversionError("HF config root must be an object")
    return value


def validate_config(path: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ConversionError("cannot read HF config %s: %s" % (path, error))
    digest = sha256_bytes(payload)
    if digest != profile["config_sha256"]:
        raise ConversionError("HF config SHA256 differs from pinned profile: got %s" % digest)
    duplicate_policy = profile.get("config_duplicate_keys")
    if duplicate_policy is None:
        config = load_json_bytes(payload, "HF config")
    else:
        config = load_profile_config_json(payload, duplicate_policy)
    required = profile["config_required"]
    missing = sorted(set(required) - set(config))
    wrong = {
        key: (config.get(key), value)
        for key, value in required.items()
        if key in config and config[key] != value
    }
    if missing or wrong:
        raise ConversionError(
            "HF config topology gate failed: missing=%s wrong=%s" % (missing, wrong)
        )
    topology = profile["topology"]
    semantic = {
        "vocab_size": config.get("vocab_size"),
        "hidden_size": config.get("hidden_size"),
        "num_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "head_dim": config.get("hidden_size") // config.get("num_attention_heads"),
        "intermediate_size": config.get("intermediate_size"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "layer_norm_epsilon": config.get("layer_norm_eps"),
        "position_embedding_type": config.get("position_embedding_type"),
        "attention_bias": True,
        "feed_forward_bias": profile["source_variant"] == "hf-esm-gelu",
        "token_dropout": config.get("token_dropout"),
        "pad_token_id": config.get("pad_token_id"),
        "mask_token_id": config.get("mask_token_id"),
        "mlp_activation": (
            "gelu" if profile["source_variant"] == "hf-esm-gelu" else "swiglu"
        ),
        "rope_base": 10000.0,
    }
    mismatch = {
        key: (value, topology[key])
        for key, value in semantic.items()
        if value != topology[key]
    }
    if mismatch:
        raise ConversionError("HF config semantic topology mismatch: %s" % mismatch)
    if topology["max_position_embeddings"] != topology["max_seqlen"] + 2:
        raise ConversionError("ESM max_position_embeddings must equal tokenizer max+2")
    if config.get("emb_layer_norm_before") is not False or config.get("torch_dtype") != "float32":
        raise ConversionError("ESM embedding norm/dtype differs from F32 runtime")
    if profile["source_variant"] == "instadeep-esm-swiglu" and config.get("add_bias_fnn") is not False:
        raise ConversionError("NT-v2 add_bias_fnn must be false")
    return config


def canonical_tensor_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    hidden = topology["hidden_size"]
    inner = topology["intermediate_size"]
    specs = [
        TensorSpec(
            "esm.embeddings.word_embeddings.weight",
            "F32",
            (topology["vocab_size"], hidden),
        )
    ]
    if topology["position_embedding_type"] == "absolute":
        specs.append(
            TensorSpec(
                "esm.embeddings.position_embeddings.weight",
                "F32",
                (topology["max_position_embeddings"], hidden),
            )
        )
    for layer in range(topology["num_layers"]):
        prefix = "esm.encoder.layer.%d." % layer
        specs.extend(
            [
                TensorSpec(prefix + "attention.LayerNorm.weight", "F32", (hidden,)),
                TensorSpec(prefix + "attention.LayerNorm.bias", "F32", (hidden,)),
            ]
        )
        for projection in ("query", "key", "value"):
            specs.append(
                TensorSpec(
                    prefix + "attention.self." + projection + ".weight",
                    "F32",
                    (hidden, hidden),
                )
            )
            if topology["attention_bias"]:
                specs.append(
                    TensorSpec(
                        prefix + "attention.self." + projection + ".bias",
                        "F32",
                        (hidden,),
                    )
                )
        specs.append(
            TensorSpec(prefix + "attention.output.dense.weight", "F32", (hidden, hidden))
        )
        if topology["attention_bias"]:
            specs.append(
                TensorSpec(prefix + "attention.output.dense.bias", "F32", (hidden,))
            )
        specs.extend(
            [
                TensorSpec(prefix + "LayerNorm.weight", "F32", (hidden,)),
                TensorSpec(prefix + "LayerNorm.bias", "F32", (hidden,)),
            ]
        )
        fused_inner = inner * (2 if topology["mlp_activation"] == "swiglu" else 1)
        specs.append(
            TensorSpec(prefix + "intermediate.dense.weight", "F32", (fused_inner, hidden))
        )
        if topology["feed_forward_bias"]:
            specs.append(
                TensorSpec(prefix + "intermediate.dense.bias", "F32", (fused_inner,))
            )
        specs.append(
            TensorSpec(prefix + "output.dense.weight", "F32", (hidden, inner))
        )
        if topology["feed_forward_bias"]:
            specs.append(TensorSpec(prefix + "output.dense.bias", "F32", (hidden,)))
    specs.extend(
        [
            TensorSpec("esm.encoder.emb_layer_norm_after.weight", "F32", (hidden,)),
            TensorSpec("esm.encoder.emb_layer_norm_after.bias", "F32", (hidden,)),
        ]
    )
    return specs


def source_tensor_specs(profile: Mapping[str, Any]) -> Tuple[List[TensorSpec], int]:
    topology = profile["topology"]
    runtime = canonical_tensor_specs(topology)
    omitted = []  # type: List[TensorSpec]
    if topology["position_embedding_type"] == "rotary":
        omitted.append(
            TensorSpec(
                "esm.embeddings.position_embeddings.weight",
                "F32",
                (topology["max_position_embeddings"], topology["hidden_size"]),
            )
        )
    if profile["source_position_ids"]:
        omitted.append(
            TensorSpec(
                "esm.embeddings.position_ids",
                "I64",
                (1, topology["max_position_embeddings"]),
            )
        )
    if profile["source_rotary_inv_freq"]:
        for layer in range(topology["num_layers"]):
            omitted.append(
                TensorSpec(
                    "esm.encoder.layer.%d.attention.self.rotary_embeddings.inv_freq"
                    % layer,
                    "F32",
                    (topology["head_dim"] // 2,),
                )
            )
    omitted.extend(
        [
            TensorSpec(
                "esm.contact_head.regression.weight",
                "F32",
                (1, topology["num_layers"] * topology["num_attention_heads"]),
            ),
            TensorSpec("esm.contact_head.regression.bias", "F32", (1,)),
            TensorSpec(
                "lm_head.bias", "F32", (topology["vocab_size"],)
            ),
            TensorSpec(
                "lm_head.dense.weight",
                "F32",
                (topology["hidden_size"], topology["hidden_size"]),
            ),
            TensorSpec("lm_head.dense.bias", "F32", (topology["hidden_size"],)),
            TensorSpec("lm_head.layer_norm.weight", "F32", (topology["hidden_size"],)),
            TensorSpec("lm_head.layer_norm.bias", "F32", (topology["hidden_size"],)),
            TensorSpec(
                "lm_head.decoder.weight",
                "F32",
                (topology["vocab_size"], topology["hidden_size"]),
            ),
        ]
    )
    return runtime + omitted, len(runtime)


def read_source_tensors(paths: Mapping[str, Path], profile: Mapping[str, Any]) -> List[Any]:
    if "config.json" not in paths:
        raise ConversionError("HF source receipt is missing config.json")
    source_format = profile["source_format"]
    allowed_dtypes = (
        ("F32", "BF16", "I64")
        if profile["source_position_ids"]
        else ("F32", "BF16")
    )
    if source_format == "safetensors":
        return read_hf_safetensors(
            select_hf_safetensors_paths(paths),
            allowed_dtypes=allowed_dtypes,
        )
    return load_hf_torch_checkpoints(
        select_hf_torch_paths(paths),
        allowed_dtypes=allowed_dtypes,
        allow_exact_column_major=profile.get("source_tensor_layout")
        == "contiguous-or-exact-column-major",
    )


def _encode_metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "b:%d" % int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or value > UINT64_MAX:
            raise ConversionError("metadata integer is outside uint64")
        return "u:%d" % value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConversionError("metadata float is not finite")
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        return "f:%016x" % bits
    if isinstance(value, str):
        return "s:" + value
    raise ConversionError("unsupported metadata value type: %s" % type(value).__name__)


def encoded_header(metadata: Mapping[str, Any], tensors: Sequence[Any]) -> bytes:
    encoded_metadata = {}  # type: Dict[str, str]
    for key in sorted(metadata):
        if not KEY_RE.fullmatch(key) or len(key.encode("ascii")) > 255:
            raise ConversionError("invalid artifact metadata key %r" % key)
        if key == "runtime.profile":
            raise ConversionError("runtime.profile is reserved")
        encoded_metadata[key] = _encode_metadata_value(metadata[key])
    encoded_metadata["runtime.profile"] = "s:" + ARTIFACT_PROFILE
    root = {"__metadata__": dict(sorted(encoded_metadata.items()))}  # type: Dict[str, Any]
    offset = 0
    for tensor in tensors:
        if tensor.dtype not in ("F32", "BF16"):
            raise ConversionError("runtime artifact cannot emit dtype %s" % tensor.dtype)
        end = offset + tensor.nbytes
        if end > UINT64_MAX:
            raise ConversionError("artifact tensor data exceeds uint64")
        root[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    raw = json.dumps(
        root, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    if not header or len(header) > MAX_HEADER_SIZE:
        raise ConversionError("runtime Safetensors header exceeds 16 MiB")
    return header


def write_artifact(
    output_path: Path,
    metadata: Mapping[str, Any],
    tensors: Sequence[Any],
    force: bool,
) -> None:
    if output_path.suffix != ".safetensors":
        raise ConversionError("output path must end in .safetensors")
    if not tensors:
        raise ConversionError("runtime tensor manifest is empty")
    header = encoded_header(metadata, tensors)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError("output already exists: %s" % output_path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % output_path.name,
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as output:
            output.truncate(8 + len(header) + sum(item.nbytes for item in tensors))
            output.seek(0)
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for index, tensor in enumerate(tensors, start=1):
                print("[%d/%d] %s" % (index, len(tensors), tensor.name), file=sys.stderr)
                written = 0
                for raw_chunk in tensor.iter_chunks(CHUNK_SIZE):
                    chunk = memoryview(raw_chunk).cast("B")
                    if written + len(chunk) > tensor.nbytes:
                        raise ConversionError("tensor %r yielded too many bytes" % tensor.name)
                    output.write(chunk)
                    written += len(chunk)
                if written != tensor.nbytes:
                    raise ConversionError(
                        "tensor %r yielded %d bytes; expected %d"
                        % (tensor.name, written, tensor.nbytes)
                    )
            output.flush()
            os.fsync(output.fileno())
        if force:
            os.replace(str(temporary), str(output_path))
        else:
            os.link(str(temporary), str(output_path))
            temporary.unlink()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_metadata(
    profile: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    receipt_sha256: str,
    catalog_sha256: str,
    profile_sha256: str,
    tokenizer_descriptor_sha256: str,
    geneb_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    topology = profile["topology"]
    metadata = {
        "runtime.abi": RUNTIME_ABI,
        "runtime.embedding_layer_count": topology["num_layers"] + 1,
        "model.id": profile["runtime_id"],
        "model.architecture": RUNTIME_ARCHITECTURE,
        "config.vocab_size": topology["vocab_size"],
        "config.hidden_size": topology["hidden_size"],
        "config.num_layers": topology["num_layers"],
        "config.max_seqlen": topology["max_seqlen"],
        "source.repo": profile["repo"],
        "source.revision": profile["revision"],
        "source.receipt_sha256": receipt_sha256,
        "source.catalog_contract_sha256": catalog_sha256,
        "source.converter_profile_contract_sha256": profile_sha256,
        "source.config_sha256": profile["config_sha256"],
        "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        "source.variant": profile["source_variant"],
        "source.omit_policy": profile["omit_policy"],
        "esm.vocab_size": topology["vocab_size"],
        "esm.hidden_size": topology["hidden_size"],
        "esm.num_layers": topology["num_layers"],
        "esm.num_attention_heads": topology["num_attention_heads"],
        "esm.head_dim": topology["head_dim"],
        "esm.intermediate_size": topology["intermediate_size"],
        "esm.max_seqlen": topology["max_seqlen"],
        "esm.max_position_embeddings": topology["max_position_embeddings"],
        "esm.layer_norm_epsilon": topology["layer_norm_epsilon"],
        "esm.rope_base": topology["rope_base"],
        "esm.position_embedding_type": topology["position_embedding_type"],
        "esm.rope_layout": "split-half",
        "esm.mlp_activation": topology["mlp_activation"],
        "esm.attention_bias": topology["attention_bias"],
        "esm.feed_forward_bias": topology["feed_forward_bias"],
        "esm.embedding_layer_norm_before": False,
        "esm.final_layer_norm": True,
        "esm.token_dropout": topology["token_dropout"],
        "esm.token_dropout_training_ratio": 0.12,
        "esm.pad_token_id": topology["pad_token_id"],
        "esm.mask_token_id": topology["mask_token_id"],
        "esm.cls_token_id": topology["cls_token_id"],
        "esm.special_token_policy": "cls-only",
        "esm.position_id_mode": "padding-cumsum",
        "esm.hidden_tap": "last-hidden-state",
        "esm.pooling": "attention-mask-mean",
        "esm.weight_dtype": "F32",
        "esm.q_scale_placement": "before-rope",
        "esm.attention_mask": "bidirectional-key-mask",
        "esm.source_unused_position_embeddings": topology["position_embedding_type"] == "rotary",
    }  # type: Dict[str, Any]
    if "layer_norm_kernel" in profile:
        metadata["esm.layer_norm_kernel"] = profile["layer_norm_kernel"]
    metadata.update(geneb_metadata)
    metadata.update(tokenizer)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--catalog", type=Path, default=default_config_path("geneb-models.json")
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=default_config_path("geneb-esm-models.json"),
    )
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, _ = load_profiles(args.profiles.resolve())
        receipt_root, _ = load_json(args.receipt.resolve(), "source receipt identity")
        model_id = receipt_root.get("model_id")
        profile = profiles.get(model_id) if isinstance(model_id, str) else None
        if profile is None:
            raise ConversionError("source receipt does not identify an ESM profile")
        catalog_entry, catalog_root, catalog_payload = load_catalog_entry(
            args.catalog.resolve(), profile
        )
        paths, receipt_payload = validate_receipt(
            args.receipt.resolve(),
            profile,
            catalog_entry,
            args.catalog.resolve(),
            catalog_payload,
        )
        if "config.json" not in paths:
            raise ConversionError("source receipt is missing config.json")
        validate_config(paths["config.json"], profile)
        tokenizer, tokenizer_descriptor_sha256 = validate_tokenizer_descriptor(
            args.tokenizer_descriptor.resolve(),
            args.tokenizer_root.resolve() if args.tokenizer_root is not None else None,
            args.output.resolve().parent,
            profile,
        )
        source_tensors = read_source_tensors(paths, profile)
        source_specs, runtime_count = source_tensor_specs(profile)
        ordered = validate_tensor_manifest(source_tensors, source_specs)
        runtime_tensors = ordered[:runtime_count]
        try:
            geneb_metadata = build_geneb_artifact_metadata(
                catalog_root, catalog_entry, catalog_payload
            )
        except GenebArtifactError as error:
            raise ConversionError(str(error)) from error
        metadata = build_metadata(
            profile,
            tokenizer,
            sha256_bytes(receipt_payload),
            catalog_contract_sha256(catalog_root, catalog_entry),
            converter_profile_contract_sha256(1, PROFILE_FORMAT, profile),
            tokenizer_descriptor_sha256,
            geneb_metadata,
        )
        write_artifact(args.output, metadata, runtime_tensors, args.force)
        print("wrote %s" % args.output)
        print("source_receipt_sha256=%s" % sha256_bytes(receipt_payload))
        print("omit_policy=%s" % profile["omit_policy"])
        return 0
    except (
        CheckpointError,
        ConversionError,
        FileExistsError,
        OSError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
        print("convert_geneb_esm_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
