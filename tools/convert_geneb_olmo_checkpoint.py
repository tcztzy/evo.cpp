#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile pinned Omni-DNA OLMo-GFM checkpoints for the portable CPU ABI."""

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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if not (_SCRIPT_DIRECTORY / "evo").is_dir():
    _INSTALLED_PYTHON = _SCRIPT_DIRECTORY.parent / "share" / "evo" / "python"
    if _INSTALLED_PYTHON.is_dir():
        sys.path.insert(0, str(_INSTALLED_PYTHON))

from evo.hf_checkpoint import (  # noqa: E402
    CheckpointError,
    TensorSpec,
    load_json,
    normalized_relative_path,
    read_hf_safetensors,
    validate_tensor_manifest,
)
from evo.geneb_artifact import (  # noqa: E402
    GenebArtifactError,
    build_geneb_artifact_metadata,
    catalog_contract_sha256,
    converter_profile_contract_sha256,
    validate_hf_fetch_receipt_provenance,
)


ARTIFACT_PROFILE = "geneb-olmo-runtime-v1"
RUNTIME_ABI = "geneb-olmo-safetensors-v1"
RUNTIME_ARCHITECTURE = "GenebOlmoDecoder"
PROFILE_FORMAT = "geneb-olmo-converter-v1"
TOKENIZER_PROFILE = "evo-tokenizer-v1"
OMNI1B_LAYER_NORM_KERNEL = "torch-2.1.2-apple-arm64-exact-v1"
OLMO_PACKAGE = "ai2-olmo"
OLMO_VERSION = "0.6.0"
OLMO_REPO = "allenai/OLMo"
OLMO_REVISION = "6c3373fa182af2d57fe3c390ffc8420d5c5b325a"
OLMO_MODEL_SHA256 = "1986566f15ceaa1177604fc7404a3f51942b7c818b5ae3d3afb553e7dc2504bc"
OLMO_CONFIG_SHA256 = "84a5265454458b5151a2ca0adfdc0536ba93451228605f96c4b3d89b8604cd74"
REMOTE_MODELING_SHA256 = "8e54a1c85cffe7eb9049549e2213d087cdfe56db4f7b0ae211171f98aeec1c21"
REMOTE_CONFIGURATION_SHA256 = "6e23d2c8ae0d9420670ee9a00de279d04e93c86d1a6b555dd62c1b49ecae12a5"
MAX_HEADER_SIZE = 16 * 1024 * 1024
CHUNK_SIZE = 16 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
REPO_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
KEY_RE = re.compile(r"[A-Za-z0-9._-]+")

PROFILE_KEYS = {
    "runtime_id",
    "geneb_model_id",
    "paper_name",
    "catalog_architecture",
    "repo",
    "revision",
    "config_sha256",
    "weights_sha256",
    "weights_size",
    "code_provenance",
    "remote_code",
    "config_required",
    "config_defaults",
    "topology",
}
TOKENIZER_PROFILE_KEYS = {
    "tokenizer_manifest_sha256",
    "tokenizer_source_receipt_contract_sha256",
    "tokenizer_asset_sha256",
    "tokenizer_asset_size",
}
PROFILE_OPTIONAL_KEYS = TOKENIZER_PROFILE_KEYS | {"layer_norm_kernel"}
CODE_PROVENANCE_KEYS = {
    "package",
    "version",
    "repo",
    "revision",
    "model_py_sha256",
    "config_py_sha256",
}
REMOTE_CODE_KEYS = {"modeling_olmo_sha256", "configuration_olmo_sha256"}
TOPOLOGY_KEYS = {
    "vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "fused_mlp_width",
    "max_seqlen",
    "norm_epsilon",
    "rope_theta",
    "norm_type",
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
CONFIG_SEMANTIC_KEYS = {
    "activation_type",
    "alibi",
    "attention_dropout",
    "attention_layer_norm",
    "attention_layer_norm_with_affine",
    "bias_for_layer_norm",
    "block_group_size",
    "block_type",
    "clip_qkv",
    "d_model",
    "embedding_dropout",
    "embedding_layer_norm",
    "embedding_size",
    "flash_attention",
    "include_bias",
    "layer_norm_eps",
    "layer_norm_type",
    "layer_norm_with_affine",
    "max_sequence_length",
    "mlp_hidden_size",
    "mlp_ratio",
    "model_type",
    "multi_query_attention",
    "n_heads",
    "n_kv_heads",
    "n_layers",
    "norm_after",
    "residual_dropout",
    "rope",
    "rope_full_precision",
    "rope_theta",
    "scale_logits",
    "vocab_size",
    "weight_tying",
}


class ConversionError(ValueError):
    """Raised when an input falls outside the closed conversion contract."""


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
        "fused_mlp_width",
        "max_seqlen",
    ):
        topology[key] = uint64_value(topology[key], "%s.%s" % (label, key), True)
    topology["norm_epsilon"] = finite_float(
        topology["norm_epsilon"], "%s.norm_epsilon" % label, True
    )
    topology["rope_theta"] = finite_float(
        topology["rope_theta"], "%s.rope_theta" % label, True
    )
    if topology["norm_type"] not in (
        "layernorm-no-affine",
        "rmsnorm-affine",
    ):
        raise ConversionError("%s.norm_type is unsupported" % label)
    hidden = topology["hidden_size"]
    heads = topology["num_attention_heads"]
    if (
        hidden % heads != 0
        or (hidden // heads) % 2 != 0
        or topology["fused_mlp_width"] % 2 != 0
        or topology["rope_theta"] <= 1.0
    ):
        raise ConversionError("%s dimensions/RoPE are invalid" % label)
    return topology


def validate_code_provenance(raw: Any, label: str) -> Dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != CODE_PROVENANCE_KEYS:
        raise ConversionError("%s fields differ from the pinned ai2-olmo receipt" % label)
    expected = {
        "package": OLMO_PACKAGE,
        "version": OLMO_VERSION,
        "repo": OLMO_REPO,
        "revision": OLMO_REVISION,
        "model_py_sha256": OLMO_MODEL_SHA256,
        "config_py_sha256": OLMO_CONFIG_SHA256,
    }
    if raw != expected:
        raise ConversionError("%s must pin ai2-olmo v0.6.0@%s" % (label, OLMO_REVISION))
    return dict(raw)


def validate_remote_code(raw: Any, label: str) -> Dict[str, str]:
    expected = {
        "modeling_olmo_sha256": REMOTE_MODELING_SHA256,
        "configuration_olmo_sha256": REMOTE_CONFIGURATION_SHA256,
    }
    if not isinstance(raw, dict) or set(raw) != REMOTE_CODE_KEYS or raw != expected:
        raise ConversionError("%s differs from the pinned model-repo wrapper code" % label)
    return dict(raw)


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    root, payload = load_json(path, "GENEB OLMo converter profiles")
    if set(root) != {"schema_version", "format", "models"}:
        raise ConversionError("OLMo profile document fields differ")
    if root["schema_version"] != 1 or root["format"] != PROFILE_FORMAT:
        raise ConversionError("OLMo profile schema/format is unsupported")
    if not isinstance(root["models"], list) or not root["models"]:
        raise ConversionError("OLMo profiles must be a nonempty array")
    profiles = {}  # type: Dict[str, Dict[str, Any]]
    identities = set()  # type: Set[Tuple[str, str]]
    for index, raw in enumerate(root["models"]):
        label = "models[%d]" % index
        actual = set(raw) if isinstance(raw, dict) else set()
        if (
            not isinstance(raw, dict)
            or not PROFILE_KEYS <= actual
            or not actual <= PROFILE_KEYS | PROFILE_OPTIONAL_KEYS
        ):
            raise ConversionError(
                "%s fields differ: missing=%s extra=%s"
                % (
                    label,
                    sorted(PROFILE_KEYS - actual),
                    sorted(actual - PROFILE_KEYS - PROFILE_OPTIONAL_KEYS),
                )
            )
        profile = dict(raw)
        for key in (
            "runtime_id",
            "geneb_model_id",
            "paper_name",
            "catalog_architecture",
            "repo",
            "revision",
            "config_sha256",
            "weights_sha256",
        ):
            profile[key] = nonempty_string(profile[key], "%s.%s" % (label, key))
        if not REPO_RE.fullmatch(profile["repo"]):
            raise ConversionError("%s.repo is not OWNER/NAME" % label)
        if not COMMIT_RE.fullmatch(profile["revision"]):
            raise ConversionError("%s.revision is not an immutable commit" % label)
        for key in ("config_sha256", "weights_sha256"):
            if not SHA256_RE.fullmatch(profile[key]):
                raise ConversionError("%s.%s is not lowercase SHA256" % (label, key))
        profile["weights_size"] = uint64_value(
            profile["weights_size"], "%s.weights_size" % label, True
        )
        tokenizer_fields = TOKENIZER_PROFILE_KEYS & set(profile)
        if tokenizer_fields and tokenizer_fields != TOKENIZER_PROFILE_KEYS:
            raise ConversionError(
                "%s tokenizer profile pins must appear together" % label
            )
        if tokenizer_fields:
            for key in (
                "tokenizer_manifest_sha256",
                "tokenizer_source_receipt_contract_sha256",
                "tokenizer_asset_sha256",
            ):
                value = profile[key]
                if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                    raise ConversionError(
                        "%s.%s is not lowercase SHA256" % (label, key)
                    )
            profile["tokenizer_asset_size"] = uint64_value(
                profile["tokenizer_asset_size"],
                "%s.tokenizer_asset_size" % label,
                True,
            )
        profile["code_provenance"] = validate_code_provenance(
            profile["code_provenance"], label + ".code_provenance"
        )
        profile["remote_code"] = validate_remote_code(
            profile["remote_code"], label + ".remote_code"
        )
        required = profile["config_required"]
        defaults = profile["config_defaults"]
        if not isinstance(required, dict) or not isinstance(defaults, dict):
            raise ConversionError("%s config gates must be objects" % label)
        if set(required) & set(defaults):
            raise ConversionError("%s config required/default keys overlap" % label)
        missing_semantics = CONFIG_SEMANTIC_KEYS - (set(required) | set(defaults))
        if missing_semantics:
            raise ConversionError(
                "%s config gates omit required OLMo semantics: %s"
                % (label, sorted(missing_semantics))
            )
        profile["topology"] = validate_topology(profile["topology"], label + ".topology")
        runtime_id = profile["runtime_id"]
        layer_norm_kernel = profile.get("layer_norm_kernel")
        if (
            runtime_id == "geneb-omni-dna-1b"
            and layer_norm_kernel != OMNI1B_LAYER_NORM_KERNEL
        ) or (
            runtime_id != "geneb-omni-dna-1b" and layer_norm_kernel is not None
        ):
            raise ConversionError(
                "%s layer_norm_kernel is not the exact Omni-DNA-1B contract"
                % label
            )
        if layer_norm_kernel is not None:
            topology = profile["topology"]
            expected_exact_topology = {
                "norm_type": "layernorm-no-affine",
                "hidden_size": 2048,
                "num_layers": 16,
                "num_attention_heads": 16,
                "fused_mlp_width": 16384,
                "norm_epsilon": 0.00001,
            }
            wrong_exact_topology = {
                key: (topology.get(key), expected)
                for key, expected in expected_exact_topology.items()
                if topology.get(key) != expected
            }
            if wrong_exact_topology:
                raise ConversionError(
                    "%s exact LayerNorm kernel topology differs: %s"
                    % (label, wrong_exact_topology)
                )
        identity = (profile["repo"], profile["revision"])
        if runtime_id in profiles or identity in identities:
            raise ConversionError("OLMo profile duplicates runtime or source identity")
        profiles[runtime_id] = profile
        identities.add(identity)
    return profiles, payload


def load_catalog_entry(
    path: Path, profile: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    root, payload = load_json(path, "GENEB catalog")
    if root.get("schema_version") != 1 or not isinstance(root.get("models"), list):
        raise ConversionError("GENEB catalog schema is unsupported")
    matches = [
        value
        for value in root["models"]
        if isinstance(value, dict) and value.get("runtime_id") == profile["runtime_id"]
    ]
    if len(matches) != 1:
        raise ConversionError(
            "GENEB catalog must contain exactly one %s" % profile["runtime_id"]
        )
    entry = matches[0]
    expected = {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "architecture": profile["catalog_architecture"],
        "family": "transformer-decoder",
    }
    wrong = {
        key: (entry.get(key), value)
        for key, value in expected.items()
        if entry.get(key) != value
    }
    if wrong:
        raise ConversionError("GENEB catalog OLMo identity differs: %s" % wrong)
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
        raise ConversionError("GENEB catalog repo/revision differs from OLMo profile")
    topology = profile["topology"]
    context = entry.get("context")
    presets = entry.get("embedding_presets")
    if (
        not isinstance(context, dict)
        or context.get("declared_max_tokens") != topology["max_seqlen"]
        or not isinstance(presets, dict)
        or any(
            not isinstance(presets.get(name), dict)
            or presets[name].get("output_width") != topology["hidden_size"]
            for name in ("reference", "normalized")
        )
    ):
        raise ConversionError("GENEB catalog context/embedding width differs from topology")
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
        raise ConversionError("source receipt files must be a nonempty array")
    paths = {}  # type: Dict[str, Path]
    verified = {}  # type: Dict[str, Tuple[int, str]]
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
        verified[name] = (size, digest)
    expected = {"config.json", "model.safetensors"}
    if not expected.issubset(paths):
        raise ConversionError(
            "OLMo source assets are missing: %s"
            % sorted(expected - set(paths))
        )
    if verified["model.safetensors"] != (
        profile["weights_size"],
        profile["weights_sha256"],
    ):
        raise ConversionError("source weights size/SHA256 differ from the pinned profile")
    return paths, payload


def validate_tokenizer_descriptor(
    descriptor_path: Path,
    tokenizer_root: Optional[Path],
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
    if TOKENIZER_PROFILE_KEYS <= set(profile):
        expected = {
            "compiler_manifest_sha256": profile["tokenizer_manifest_sha256"],
            "source_receipt_contract_sha256": profile[
                "tokenizer_source_receipt_contract_sha256"
            ],
            "tokenizer.sha256": profile["tokenizer_asset_sha256"],
            "tokenizer.size": profile["tokenizer_asset_size"],
        }
        wrong = {
            key: (descriptor.get(key), value)
            for key, value in expected.items()
            if descriptor.get(key) != value
        }
        if wrong:
            raise ConversionError(
                "tokenizer descriptor differs from pinned profile: %s" % wrong
            )
    relative = normalized_relative_path(descriptor["tokenizer.path"], "tokenizer.path")
    size = uint64_value(descriptor["tokenizer.size"], "tokenizer.size", True)
    root = (tokenizer_root if tokenizer_root is not None else descriptor_path.parent).resolve()
    asset = (root / relative).resolve()
    try:
        asset.relative_to(root)
    except ValueError:
        raise ConversionError("tokenizer.path escapes tokenizer root")
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


def validate_config(path: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
    config, payload = load_json(path, "HF config")
    digest = sha256_bytes(payload)
    if digest != profile["config_sha256"]:
        raise ConversionError("HF config SHA256 differs from pinned profile: got %s" % digest)
    required = profile["config_required"]
    defaults = profile["config_defaults"]
    missing = sorted(set(required) - set(config))
    wrong = {
        key: (config.get(key), value)
        for key, value in required.items()
        if key in config and config[key] != value
    }
    wrong.update(
        {
            key: (config[key], value)
            for key, value in defaults.items()
            if key in config and config[key] != value
        }
    )
    if missing or wrong:
        raise ConversionError(
            "HF config topology gate failed: missing=%s wrong=%s" % (missing, wrong)
        )

    def effective(key: str) -> Any:
        return config[key] if key in config else defaults.get(key)

    closed_semantics = {
        "activation_type": "swiglu",
        "alibi": False,
        "attention_dropout": 0.0,
        "attention_layer_norm": False,
        "attention_layer_norm_with_affine": False,
        "bias_for_layer_norm": False,
        "block_group_size": 1,
        "block_type": "sequential",
        "clip_qkv": None,
        "embedding_dropout": 0.0,
        "embedding_layer_norm": False,
        "flash_attention": False,
        "include_bias": False,
        "model_type": "olmo-gfm",
        "multi_query_attention": False,
        "n_kv_heads": None,
        "norm_after": False,
        "residual_dropout": 0.0,
        "rope": True,
        "rope_full_precision": True,
        "scale_logits": False,
        "weight_tying": True,
    }
    closed_wrong = {
        key: (effective(key), expected)
        for key, expected in closed_semantics.items()
        if effective(key) != expected
    }
    if effective("embedding_size") != effective("vocab_size"):
        closed_wrong["embedding_size"] = (
            effective("embedding_size"),
            effective("vocab_size"),
        )
    if closed_wrong:
        raise ConversionError("HF config is not the closed OLMo-GFM ABI: %s" % closed_wrong)

    hidden = effective("d_model")
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
        raise ConversionError("HF config d_model must be a positive integer")
    fused_mlp = effective("mlp_hidden_size")
    if fused_mlp is None:
        ratio = effective("mlp_ratio")
        if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio <= 0:
            raise ConversionError("HF config mlp_ratio must be a positive integer")
        fused_mlp = ratio * hidden
    elif isinstance(fused_mlp, bool) or not isinstance(fused_mlp, int):
        raise ConversionError("HF config mlp_hidden_size must be an integer or null")
    if effective("layer_norm_type") == "default" and effective("layer_norm_with_affine") is False:
        norm_type = "layernorm-no-affine"
    elif effective("layer_norm_type") == "rms" and effective("layer_norm_with_affine") is True:
        norm_type = "rmsnorm-affine"
    else:
        norm_type = None
    semantic = {
        "vocab_size": effective("vocab_size"),
        "hidden_size": hidden,
        "num_layers": effective("n_layers"),
        "num_attention_heads": effective("n_heads"),
        "fused_mlp_width": fused_mlp,
        "max_seqlen": effective("max_sequence_length"),
        "norm_epsilon": effective("layer_norm_eps"),
        "rope_theta": effective("rope_theta"),
        "norm_type": norm_type,
    }
    mismatch = {
        key: (semantic.get(key), profile["topology"][key])
        for key in sorted(TOPOLOGY_KEYS)
        if semantic.get(key) != profile["topology"][key]
    }
    if mismatch:
        raise ConversionError("HF config semantic topology mismatch: %s" % mismatch)
    return config


def canonical_tensor_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    hidden = topology["hidden_size"]
    fused_mlp = topology["fused_mlp_width"]
    specs = [
        TensorSpec("model.transformer.wte.weight", "F32", (topology["vocab_size"], hidden))
    ]
    for layer in range(topology["num_layers"]):
        prefix = "model.transformer.blocks.%d." % layer
        specs.append(TensorSpec(prefix + "att_proj.weight", "F32", (hidden * 3, hidden)))
        if topology["norm_type"] == "rmsnorm-affine":
            specs.append(TensorSpec(prefix + "attn_norm.weight", "F32", (hidden,)))
        specs.append(TensorSpec(prefix + "attn_out.weight", "F32", (hidden, hidden)))
        if topology["norm_type"] == "rmsnorm-affine":
            specs.append(TensorSpec(prefix + "ff_norm.weight", "F32", (hidden,)))
        specs.extend(
            [
                TensorSpec(prefix + "ff_out.weight", "F32", (hidden, fused_mlp // 2)),
                TensorSpec(prefix + "ff_proj.weight", "F32", (fused_mlp, hidden)),
            ]
        )
    if topology["norm_type"] == "rmsnorm-affine":
        specs.append(TensorSpec("model.transformer.ln_f.weight", "F32", (hidden,)))
    return specs


def read_source_tensors(paths: Mapping[str, Path]) -> List[Any]:
    return read_hf_safetensors({"model.safetensors": paths["model.safetensors"]})


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
            raise ConversionError("runtime.profile is reserved by converter")
        encoded_metadata[key] = _encode_metadata_value(metadata[key])
    encoded_metadata["runtime.profile"] = "s:" + ARTIFACT_PROFILE
    root = {"__metadata__": dict(sorted(encoded_metadata.items()))}  # type: Dict[str, Any]
    offset = 0
    for tensor in tensors:
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
            total_size = 8 + len(header) + sum(tensor.nbytes for tensor in tensors)
            if total_size > UINT64_MAX:
                raise ConversionError("runtime Safetensors file size exceeds uint64")
            output.truncate(total_size)
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
        "olmo.num_attention_heads": topology["num_attention_heads"],
        "olmo.fused_mlp_width": topology["fused_mlp_width"],
        "olmo.norm_epsilon": topology["norm_epsilon"],
        "olmo.rope_theta": topology["rope_theta"],
        "olmo.norm_type": topology["norm_type"],
        "olmo.block_type": "sequential",
        "olmo.activation": "swiglu",
        "olmo.qkv_layout": "q-k-v",
        "olmo.swiglu_layout": "x-gate",
        "olmo.rope_layout": "split-half",
        "olmo.weight_dtype": "F32",
        "olmo.norm_after": False,
        "olmo.attention_layer_norm": False,
        "olmo.include_bias": False,
        "olmo.weight_tying": True,
        "olmo.rope_full_precision": True,
        "source.repo": profile["repo"],
        "source.revision": profile["revision"],
        "source.weights.sha256": profile["weights_sha256"],
        "source.weights.size": profile["weights_size"],
        "source.receipt_sha256": receipt_sha256,
        "source.catalog_contract_sha256": catalog_sha256,
        "source.converter_profile_contract_sha256": profile_sha256,
        "source.config_sha256": profile["config_sha256"],
        "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        "source.olmo.package": OLMO_PACKAGE,
        "source.olmo.version": OLMO_VERSION,
        "source.olmo.repo": OLMO_REPO,
        "source.olmo.revision": OLMO_REVISION,
        "source.olmo.model_py_sha256": OLMO_MODEL_SHA256,
        "source.olmo.config_py_sha256": OLMO_CONFIG_SHA256,
        "source.remote_modeling_sha256": REMOTE_MODELING_SHA256,
        "source.remote_configuration_sha256": REMOTE_CONFIGURATION_SHA256,
    }  # type: Dict[str, Any]
    if "layer_norm_kernel" in profile:
        metadata["olmo.layer_norm_kernel"] = profile["layer_norm_kernel"]
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
        "--profiles", type=Path, default=default_config_path("geneb-olmo-models.json")
    )
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, _profile_payload = load_profiles(args.profiles.resolve())
        receipt_identity, _ = load_json(args.receipt.resolve(), "source receipt identity")
        model_id = receipt_identity.get("model_id")
        profile = profiles.get(model_id) if isinstance(model_id, str) else None
        if profile is None:
            raise ConversionError("source receipt does not identify an OLMo profile")
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
        validate_config(paths["config.json"], profile)
        tokenizer, tokenizer_descriptor_sha256 = validate_tokenizer_descriptor(
            args.tokenizer_descriptor.resolve(),
            args.tokenizer_root.resolve() if args.tokenizer_root is not None else None,
            profile,
        )
        source_tensors = read_source_tensors(paths)
        runtime_specs = canonical_tensor_specs(profile["topology"])
        runtime_tensors = validate_tensor_manifest(source_tensors, runtime_specs)
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
        print("olmo_code_revision=%s" % OLMO_REVISION)
        return 0
    except (
        CheckpointError,
        ConversionError,
        FileExistsError,
        OSError,
        ValueError,
    ) as error:
        print("convert_geneb_olmo_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
