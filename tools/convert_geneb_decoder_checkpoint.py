#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile verified HF Llama/Mistral checkpoints for the GENEB CPU decoder."""

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
    normalized_relative_path,
    read_hf_safetensors,
    select_hf_safetensors_paths,
    validate_tensor_manifest,
)
from evo.geneb_artifact import (  # noqa: E402
    GenebArtifactError,
    build_geneb_artifact_metadata,
    catalog_contract_sha256,
    converter_profile_contract_sha256,
    validate_hf_fetch_receipt_provenance,
)


ARTIFACT_PROFILE = "geneb-decoder-runtime-v1"
RUNTIME_ABI = "geneb-decoder-safetensors-v1"
RUNTIME_ARCHITECTURE = "GenebTransformerDecoder"
PROFILE_FORMAT = "geneb-decoder-converter-v1"
TOKENIZER_PROFILE = "evo-tokenizer-v1"
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
    "source_format",
    "config_sha256",
    "config_required",
    "config_defaults",
    "lm_head_policy",
    "topology",
}
TOKENIZER_PROFILE_KEYS = {
    "tokenizer_manifest_sha256",
    "tokenizer_source_receipt_contract_sha256",
    "tokenizer_asset_sha256",
    "tokenizer_asset_size",
}
PROFILE_OPTIONAL_KEYS = {
    "attention_kernel",
    "f32_math_kernel",
    "legacy_rotary_inv_freq",
} | TOKENIZER_PROFILE_KEYS
BIOFM_ATTENTION_KERNEL = "torch-cpu-flash-bf16-portable"
OMNINA_F32_MATH_KERNEL = "torch-2.7.1-apple-arm64-exact-v1"
OMNINA_LEGACY_ROTARY_INV_FREQ = {
    "policy": "validate-and-omit",
    "layer_count": 16,
    "dtype": "BF16",
    "shape": [32],
    "tensor_sha256": (
        "8743397b60ff5e2c99e396f6342c654eb02f988fd35b6c725d6a5fc341dbed6d"
    ),
}
TOPOLOGY_KEYS = {
    "vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rotary_dim",
    "inner_mlp_size",
    "max_seqlen",
    "sliding_window",
    "rms_norm_epsilon",
    "rms_epsilon_placement",
    "rope_base",
    "rope_position_scale",
    "rope_layout",
    "mlp_activation",
    "attention_bias",
    "mlp_bias",
    "embedding_dtype",
    "projection_dtype",
    "norm_dtype",
    "activation_dtype",
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
    """Raised when an input does not match the frozen converter contract."""


def exact_keys(
    value: Mapping[str, Any], required: Iterable[str], optional: Iterable[str], label: str
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
        raise ConversionError("%s must be a%s uint64" % (label, " positive" if positive else ""))
    return value


def finite_float(value: Any, label: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionError("%s must be finite numeric metadata" % label)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ConversionError("%s must be finite%s" % (label, " and positive" if positive else ""))
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
        "num_key_value_heads",
        "head_dim",
        "rotary_dim",
        "inner_mlp_size",
        "max_seqlen",
    ):
        topology[key] = uint64_value(topology[key], "%s.%s" % (label, key), True)
    topology["sliding_window"] = uint64_value(
        topology["sliding_window"], "%s.sliding_window" % label, False
    )
    for key in ("rms_norm_epsilon", "rope_base", "rope_position_scale"):
        topology[key] = finite_float(topology[key], "%s.%s" % (label, key), True)
    if topology["rms_epsilon_placement"] not in ("inside-sqrt", "after-sqrt"):
        raise ConversionError("%s.rms_epsilon_placement is unsupported" % label)
    if topology["rope_layout"] not in ("split-half", "adjacent-pairs"):
        raise ConversionError("%s.rope_layout is unsupported" % label)
    if topology["mlp_activation"] not in ("swiglu", "gelu"):
        raise ConversionError("%s.mlp_activation is unsupported" % label)
    for key in ("attention_bias", "mlp_bias"):
        if not isinstance(topology[key], bool):
            raise ConversionError("%s.%s must be boolean" % (label, key))
    for key in (
        "embedding_dtype",
        "projection_dtype",
        "norm_dtype",
        "activation_dtype",
    ):
        if topology[key] not in ("F32", "BF16"):
            raise ConversionError("%s.%s must be F32 or BF16" % (label, key))

    q_heads = topology["num_attention_heads"]
    kv_heads = topology["num_key_value_heads"]
    head_dim = topology["head_dim"]
    rotary_dim = topology["rotary_dim"]
    if q_heads % kv_heads != 0:
        raise ConversionError("%s query heads are not divisible by KV heads" % label)
    if head_dim % 2 != 0 or rotary_dim % 2 != 0 or rotary_dim > head_dim:
        raise ConversionError("%s head/rotary dimensions are invalid" % label)
    if q_heads > UINT64_MAX // head_dim or kv_heads > UINT64_MAX // head_dim:
        raise ConversionError("%s attention width exceeds uint64" % label)
    if topology["sliding_window"] > topology["max_seqlen"]:
        raise ConversionError("%s sliding window exceeds context" % label)
    return topology


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    root, payload = load_json(path, "GENEB decoder profile manifest")
    exact_keys(root, ["schema_version", "format", "models"], [], "profile manifest")
    if root["schema_version"] != 1 or root["format"] != PROFILE_FORMAT:
        raise ConversionError("GENEB decoder profile manifest schema is unsupported")
    if not isinstance(root["models"], list) or not root["models"]:
        raise ConversionError("profile manifest models must be a nonempty array")
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
        ):
            profile[key] = nonempty_string(profile[key], "%s.%s" % (label, key))
        if not REPO_RE.fullmatch(profile["repo"]):
            raise ConversionError("%s.repo is not OWNER/NAME" % label)
        if not COMMIT_RE.fullmatch(profile["revision"]):
            raise ConversionError("%s.revision is not an immutable commit" % label)
        if not SHA256_RE.fullmatch(profile["config_sha256"]):
            raise ConversionError("%s.config_sha256 is not lowercase SHA256" % label)
        if profile["source_format"] not in ("safetensors", "pytorch-bin"):
            raise ConversionError("%s.source_format is unsupported" % label)
        if profile["lm_head_policy"] != "validate-and-omit":
            raise ConversionError("%s must use validate-and-omit lm_head policy" % label)
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
        legacy_rotary = profile.get("legacy_rotary_inv_freq")
        if legacy_rotary is not None and (
            runtime_id != "geneb-omnina-220m"
            or legacy_rotary != OMNINA_LEGACY_ROTARY_INV_FREQ
        ):
            raise ConversionError(
                "%s legacy_rotary_inv_freq is not the exact OmniNA contract"
                % label
            )
        attention_kernel = profile.get("attention_kernel")
        if (
            runtime_id == "geneb-biofm-265m"
            and attention_kernel != BIOFM_ATTENTION_KERNEL
        ) or (
            runtime_id != "geneb-biofm-265m" and attention_kernel is not None
        ):
            raise ConversionError(
                "%s attention_kernel is not the exact BioFM contract" % label
            )
        f32_math_kernel = profile.get("f32_math_kernel")
        if (
            runtime_id == "geneb-omnina-220m"
            and f32_math_kernel != OMNINA_F32_MATH_KERNEL
        ) or (
            runtime_id != "geneb-omnina-220m" and f32_math_kernel is not None
        ):
            raise ConversionError(
                "%s f32_math_kernel is not the exact OmniNA contract" % label
            )
        required = profile["config_required"]
        defaults = profile["config_defaults"]
        if not isinstance(required, dict) or not isinstance(defaults, dict):
            raise ConversionError("%s config gates must be objects" % label)
        if set(required) & set(defaults):
            raise ConversionError("%s config required/default keys overlap" % label)
        profile["topology"] = validate_topology(profile["topology"], label + ".topology")
        identity = (profile["repo"], profile["revision"])
        if identity in identities:
            raise ConversionError("profile manifest duplicates repo/revision identity")
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
        value
        for value in root["models"]
        if isinstance(value, dict) and value.get("runtime_id") == profile["runtime_id"]
    ]
    if len(matches) != 1:
        raise ConversionError(
            "GENEB catalog must contain exactly one %s" % profile["runtime_id"]
        )
    entry = matches[0]
    source = entry.get("source")
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
        raise ConversionError("GENEB catalog decoder identity differs: %s" % wrong)
    if (
        not isinstance(source, dict)
        or source.get("kind") != "huggingface"
        or source.get("repo") != profile["repo"]
        or source.get("revision") != profile["revision"]
        or not isinstance(source.get("requested_revision"), str)
        or not source.get("requested_revision")
        or source.get("immutable") is not True
    ):
        raise ConversionError("GENEB catalog repo/revision differs from decoder profile")
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
    size = uint64_value(descriptor["tokenizer.size"], "tokenizer.size", False)
    root = (tokenizer_root if tokenizer_root is not None else descriptor_path.parent).resolve()
    asset = (root / relative).resolve()
    try:
        asset.relative_to(root)
    except ValueError:
        raise ConversionError("tokenizer.path escapes tokenizer root")
    artifact_asset = (artifact_root.resolve() / relative).resolve()
    if artifact_asset != asset:
        raise ConversionError(
            "tokenizer asset must already be staged at tokenizer.path relative "
            "to the runtime artifact"
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


def validate_config(
    path: Path, profile: Mapping[str, Any]
) -> Dict[str, Any]:
    config, payload = load_json(path, "HF config")
    digest = sha256_bytes(payload)
    if digest != profile["config_sha256"]:
        raise ConversionError(
            "HF config SHA256 differs from pinned profile: got %s" % digest
        )
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

    topology = profile["topology"]
    dtype = {"float32": "F32", "bfloat16": "BF16"}.get(effective("torch_dtype"))
    activation_dtype = dtype
    if profile["runtime_id"] in {
        "geneb-genome-ocean-4b",
        "geneb-genome-ocean-500m",
    }:
        if dtype != "BF16":
            raise ConversionError(
                "GenomeOcean source dtype must remain BF16 for CPU-F32 activation"
            )
        activation_dtype = "F32"
    semantic = {
        "vocab_size": effective("vocab_size"),
        "hidden_size": effective("hidden_size"),
        "num_layers": effective("num_hidden_layers"),
        "num_attention_heads": effective("num_attention_heads"),
        "num_key_value_heads": effective("num_key_value_heads"),
        "head_dim": effective("head_dim"),
        "rotary_dim": effective("head_dim"),
        "inner_mlp_size": effective("intermediate_size"),
        "max_seqlen": effective("max_position_embeddings"),
        "sliding_window": 0 if effective("sliding_window") is None else effective("sliding_window"),
        "rms_norm_epsilon": effective("rms_norm_eps"),
        "rms_epsilon_placement": "inside-sqrt",
        "rope_base": effective("rope_theta"),
        "rope_position_scale": 1.0 if effective("rope_scaling") is None else None,
        "rope_layout": "split-half",
        "mlp_activation": "swiglu" if effective("hidden_act") == "silu" else None,
        "attention_bias": effective("attention_bias"),
        "mlp_bias": effective("mlp_bias"),
        "embedding_dtype": dtype,
        "projection_dtype": dtype,
        "norm_dtype": dtype,
        "activation_dtype": activation_dtype,
    }
    mismatch = {
        key: (semantic.get(key), topology[key])
        for key in sorted(TOPOLOGY_KEYS)
        if semantic.get(key) != topology[key]
    }
    if mismatch:
        raise ConversionError("HF config semantic topology mismatch: %s" % mismatch)
    return config


def canonical_tensor_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    hidden = topology["hidden_size"]
    inner = topology["inner_mlp_size"]
    query_width = topology["num_attention_heads"] * topology["head_dim"]
    kv_width = topology["num_key_value_heads"] * topology["head_dim"]
    projection = topology["projection_dtype"]
    norm = topology["norm_dtype"]
    specs = [
        TensorSpec(
            "model.embed_tokens.weight",
            topology["embedding_dtype"],
            (topology["vocab_size"], hidden),
        )
    ]
    for layer in range(topology["num_layers"]):
        prefix = "model.layers.%d." % layer
        specs.extend(
            [
                TensorSpec(prefix + "input_layernorm.weight", norm, (hidden,)),
                TensorSpec(prefix + "self_attn.q_proj.weight", projection, (query_width, hidden)),
                TensorSpec(prefix + "self_attn.k_proj.weight", projection, (kv_width, hidden)),
                TensorSpec(prefix + "self_attn.v_proj.weight", projection, (kv_width, hidden)),
                TensorSpec(prefix + "self_attn.o_proj.weight", projection, (hidden, query_width)),
            ]
        )
        if topology["attention_bias"]:
            specs.extend(
                [
                    TensorSpec(prefix + "self_attn.q_proj.bias", projection, (query_width,)),
                    TensorSpec(prefix + "self_attn.k_proj.bias", projection, (kv_width,)),
                    TensorSpec(prefix + "self_attn.v_proj.bias", projection, (kv_width,)),
                    TensorSpec(prefix + "self_attn.o_proj.bias", projection, (hidden,)),
                ]
            )
        specs.append(TensorSpec(prefix + "post_attention_layernorm.weight", norm, (hidden,)))
        if topology["mlp_activation"] == "swiglu":
            specs.append(TensorSpec(prefix + "mlp.gate_proj.weight", projection, (inner, hidden)))
        specs.extend(
            [
                TensorSpec(prefix + "mlp.up_proj.weight", projection, (inner, hidden)),
                TensorSpec(prefix + "mlp.down_proj.weight", projection, (hidden, inner)),
            ]
        )
        if topology["mlp_bias"]:
            if topology["mlp_activation"] == "swiglu":
                specs.append(TensorSpec(prefix + "mlp.gate_proj.bias", projection, (inner,)))
            specs.extend(
                [
                    TensorSpec(prefix + "mlp.up_proj.bias", projection, (inner,)),
                    TensorSpec(prefix + "mlp.down_proj.bias", projection, (hidden,)),
                ]
            )
    specs.append(TensorSpec("model.norm.weight", norm, (hidden,)))
    return specs


def tensor_source_sha256(source: Any) -> str:
    digest = hashlib.sha256()
    size = 0
    for chunk in source.iter_chunks(CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    if size != source.nbytes:
        raise ConversionError(
            "tensor %r exposed %d bytes; expected %d"
            % (source.name, size, source.nbytes)
        )
    return digest.hexdigest()


def validate_and_omit_legacy_rotary_inv_freq(
    sources: Sequence[Any], profile: Mapping[str, Any]
) -> List[Any]:
    contract = profile.get("legacy_rotary_inv_freq")
    if contract is None:
        return list(sources)
    expected_names = {
        "model.layers.%d.self_attn.rotary_emb.inv_freq" % layer
        for layer in range(contract["layer_count"])
    }
    by_name = {}  # type: Dict[str, Any]
    for source in sources:
        if source.name in by_name:
            raise ConversionError("source tensor %r is duplicated" % source.name)
        by_name[source.name] = source
    missing = sorted(expected_names - set(by_name))
    if missing:
        raise ConversionError(
            "OmniNA legacy rotary inv_freq tensors are missing: %s" % missing
        )
    expected_shape = tuple(contract["shape"])
    for name in sorted(expected_names):
        source = by_name[name]
        if source.dtype != contract["dtype"]:
            raise ConversionError(
                "OmniNA legacy rotary inv_freq tensor %r dtype differs" % name
            )
        if tuple(source.shape) != expected_shape:
            raise ConversionError(
                "OmniNA legacy rotary inv_freq tensor %r shape differs" % name
            )
        if tensor_source_sha256(source) != contract["tensor_sha256"]:
            raise ConversionError(
                "OmniNA legacy rotary inv_freq tensor %r content differs" % name
            )
    return [source for source in sources if source.name not in expected_names]


def read_source_tensors(
    paths: Mapping[str, Path], profile: Mapping[str, Any]
) -> List[Any]:
    source_format = profile["source_format"]
    if "config.json" not in paths:
        raise ConversionError("HF source receipt is missing config.json")
    if source_format == "safetensors":
        return read_hf_safetensors(select_hf_safetensors_paths(paths))
    expected = {"pytorch_model.bin"}
    checkpoint_names = {
        name
        for name in paths
        if name.endswith(".bin") or name.endswith(".bin.index.json")
    }
    if checkpoint_names != expected:
        raise ConversionError(
            "OmniNA PyTorch checkpoint assets differ: missing=%s extra=%s"
            % (
                sorted(expected - checkpoint_names),
                sorted(checkpoint_names - expected),
            )
        )
    # Receipt size/SHA256 verification completed before this optional import.
    return load_hf_torch_checkpoints(
        {"pytorch_model.bin": paths["pytorch_model.bin"]}
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
            raise ConversionError("runtime.profile is reserved by converter")
        encoded_metadata[key] = _encode_metadata_value(metadata[key])
    encoded_metadata["runtime.profile"] = "s:" + ARTIFACT_PROFILE
    root = {"__metadata__": dict(sorted(encoded_metadata.items()))}  # type: Dict[str, Any]
    offset = 0
    dtype_names = {"F32": "F32", "BF16": "BF16"}
    for tensor in tensors:
        end = offset + tensor.nbytes
        if end > UINT64_MAX:
            raise ConversionError("artifact tensor data exceeds uint64")
        root[tensor.name] = {
            "dtype": dtype_names[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    raw = json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
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
        "source.repo": profile["repo"],
        "source.revision": profile["revision"],
        "source.receipt_sha256": receipt_sha256,
        "source.catalog_contract_sha256": catalog_sha256,
        "source.converter_profile_contract_sha256": profile_sha256,
        "source.config_sha256": profile["config_sha256"],
        "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        "source.lm_head_policy": profile["lm_head_policy"],
    }  # type: Dict[str, Any]
    for key in sorted(TOPOLOGY_KEYS):
        metadata["decoder." + key] = topology[key]
    if "attention_kernel" in profile:
        metadata["decoder.attention_kernel"] = profile["attention_kernel"]
    if "f32_math_kernel" in profile:
        metadata["decoder.f32_math_kernel"] = profile["f32_math_kernel"]
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
        default=default_config_path("geneb-decoder-models.json"),
    )
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, _profile_payload = load_profiles(args.profiles.resolve())
        receipt_root, _ = load_json(args.receipt.resolve(), "source receipt identity")
        model_id = receipt_root.get("model_id")
        profile = profiles.get(model_id) if isinstance(model_id, str) else None
        if profile is None:
            raise ConversionError("source receipt does not identify a decoder profile")
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
        source_tensors = validate_and_omit_legacy_rotary_inv_freq(
            read_source_tensors(paths, profile), profile
        )
        runtime_specs = canonical_tensor_specs(profile["topology"])
        source_specs = runtime_specs + [
            TensorSpec(
                "lm_head.weight",
                profile["topology"]["projection_dtype"],
                (
                    profile["topology"]["vocab_size"],
                    profile["topology"]["hidden_size"],
                ),
            )
        ]
        ordered = validate_tensor_manifest(source_tensors, source_specs)
        if ordered[-1].name != "lm_head.weight":
            raise ConversionError("internal lm_head validation order is invalid")
        runtime_tensors = ordered[:-1]
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
        print("lm_head_policy=validate-and-omit")
        return 0
    except (
        CheckpointError,
        ConversionError,
        FileExistsError,
        OSError,
        ValueError,
    ) as error:
        print("convert_geneb_decoder_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
