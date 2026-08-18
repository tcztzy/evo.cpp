#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile verified HF BERT/Mosaic/MutBERT checkpoints for GENEB CPU."""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if not (_SCRIPT_DIRECTORY / "evo").is_dir():
    _INSTALLED_PYTHON = _SCRIPT_DIRECTORY.parent / "share" / "evo" / "python"
    if _INSTALLED_PYTHON.is_dir():
        sys.path.insert(0, str(_INSTALLED_PYTHON))

from evo.geneb_artifact import (  # noqa: E402
    GenebArtifactError,
    build_geneb_artifact_metadata,
    catalog_contract_sha256,
    converter_profile_contract_sha256,
    validate_hf_fetch_receipt_provenance,
)
from evo.hf_checkpoint import (  # noqa: E402
    CheckpointError,
    TensorSpec,
    load_hf_torch_checkpoints,
    load_json,
    normalized_relative_path,
    read_hf_safetensors,
    select_hf_safetensors_paths,
    select_hf_torch_paths,
    validate_tensor_manifest,
)


ARTIFACT_PROFILE = "geneb-bert-runtime-v1"
RUNTIME_ABI = "geneb-bert-safetensors-v1"
RUNTIME_ARCHITECTURE = "GenebBertEncoder"
PROFILE_FORMAT = "geneb-bert-converter-v1"
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
    "source_prefix",
    "source_omit_policy",
    "topology",
}
TOPOLOGY_KEYS = {
    "vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "head_dim",
    "inner_mlp_size",
    "max_seqlen",
    "type_vocab_size",
    "layer_norm_epsilon",
    "rope_base",
    "position_encoding",
    "norm_placement",
    "final_layer_norm",
    "unpad_masked_tokens",
    "mlp_kind",
    "qkv_layout",
    "input_kind",
    "pooling",
    "attention_bias",
    "mlp_input_bias",
    "mlp_output_bias",
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
OMIT_POLICIES = {
    "gena-mlm-position-ids",
    "gena-pretraining-position-ids",
    "mosaic-pooler",
    "mosaic-mlm",
    "bert-mlm-pooler",
    "mutbert-mlm",
}


class ConversionError(ValueError):
    """Raised when input is outside the frozen GENEB BERT contract."""


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
        "head_dim",
        "inner_mlp_size",
        "max_seqlen",
        "type_vocab_size",
    ):
        topology[key] = uint64_value(topology[key], "%s.%s" % (label, key), True)
    topology["layer_norm_epsilon"] = finite_float(
        topology["layer_norm_epsilon"], label + ".layer_norm_epsilon", True
    )
    topology["rope_base"] = finite_float(
        topology["rope_base"], label + ".rope_base", False
    )
    if topology["hidden_size"] != (
        topology["num_attention_heads"] * topology["head_dim"]
    ):
        raise ConversionError("%s attention geometry differs from hidden width" % label)
    if topology["head_dim"] % 2 != 0:
        raise ConversionError("%s head_dim must be even" % label)
    enum_values = {
        "position_encoding": ("absolute", "alibi-symmetric", "rope-split-half"),
        "norm_placement": ("pre", "post"),
        "mlp_kind": ("gelu", "gated-gelu"),
        "qkv_layout": ("separate", "fused-qkv"),
        "input_kind": ("token-ids", "soft-vocabulary"),
        "pooling": ("attention-mask-mean", "cls-token"),
    }
    for key, values in enum_values.items():
        if topology[key] not in values:
            raise ConversionError("%s.%s is unsupported" % (label, key))
    for key in (
        "final_layer_norm",
        "unpad_masked_tokens",
        "attention_bias",
        "mlp_input_bias",
        "mlp_output_bias",
    ):
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

    position = topology["position_encoding"]
    if position == "absolute":
        exact = (
            topology["rope_base"] == 0.0
            and topology["mlp_kind"] == "gelu"
            and topology["qkv_layout"] == "separate"
            and topology["input_kind"] == "token-ids"
            and not topology["unpad_masked_tokens"]
        )
    elif position == "alibi-symmetric":
        exact = (
            topology["rope_base"] == 0.0
            and topology["norm_placement"] == "post"
            and topology["mlp_kind"] == "gated-gelu"
            and topology["qkv_layout"] == "fused-qkv"
            and topology["input_kind"] == "token-ids"
            and topology["unpad_masked_tokens"]
            and not topology["mlp_input_bias"]
            and topology["mlp_output_bias"]
        )
    else:
        exact = (
            topology["rope_base"] > 0.0
            and topology["norm_placement"] == "post"
            and topology["mlp_kind"] == "gelu"
            and topology["qkv_layout"] == "separate"
            and topology["input_kind"] == "soft-vocabulary"
            and not topology["unpad_masked_tokens"]
        )
    if not exact:
        raise ConversionError("%s mixes unsupported BERT family semantics" % label)
    if topology["final_layer_norm"] and topology["norm_placement"] != "pre":
        raise ConversionError("%s final_layer_norm requires pre norm" % label)
    return topology


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    root, payload = load_json(path, "GENEB BERT profile manifest")
    exact_keys(root, ["schema_version", "format", "models"], [], "profile manifest")
    if root["schema_version"] != 1 or root["format"] != PROFILE_FORMAT:
        raise ConversionError("GENEB BERT profile manifest schema is unsupported")
    if not isinstance(root["models"], list) or not root["models"]:
        raise ConversionError("profile manifest models must be a nonempty array")
    profiles = {}  # type: Dict[str, Dict[str, Any]]
    identities = set()  # type: set
    for index, raw in enumerate(root["models"]):
        label = "profile manifest models[%d]" % index
        if not isinstance(raw, dict) or set(raw) != PROFILE_KEYS:
            actual = set(raw) if isinstance(raw, dict) else set()
            raise ConversionError(
                "%s fields differ: missing=%s extra=%s"
                % (label, sorted(PROFILE_KEYS - actual), sorted(actual - PROFILE_KEYS))
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
        if profile["source_prefix"] not in ("", "bert."):
            raise ConversionError("%s.source_prefix is unsupported" % label)
        if profile["source_omit_policy"] not in OMIT_POLICIES:
            raise ConversionError("%s.source_omit_policy is unsupported" % label)
        required = profile["config_required"]
        defaults = profile["config_defaults"]
        if not isinstance(required, dict) or not isinstance(defaults, dict):
            raise ConversionError("%s config gates must be objects" % label)
        if set(required) & set(defaults):
            raise ConversionError("%s config required/default keys overlap" % label)
        profile["topology"] = validate_topology(profile["topology"], label + ".topology")
        if (profile["source_prefix"] == "") != (
            profile["source_omit_policy"] == "mosaic-pooler"
        ):
            raise ConversionError("%s bare source prefix is reserved for DNABERT-S" % label)
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
        "family": "transformer-encoder",
    }
    wrong = {
        key: (entry.get(key), value)
        for key, value in expected.items()
        if entry.get(key) != value
    }
    if wrong:
        raise ConversionError("GENEB catalog BERT identity differs: %s" % wrong)
    if (
        not isinstance(source, dict)
        or source.get("kind") != "huggingface"
        or source.get("repo") != profile["repo"]
        or source.get("revision") != profile["revision"]
        or not isinstance(source.get("requested_revision"), str)
        or not source.get("requested_revision")
        or source.get("immutable") is not True
    ):
        raise ConversionError("GENEB catalog repo/revision differs from BERT profile")
    topology = profile["topology"]
    context = entry.get("context")
    presets = entry.get("embedding_presets")
    normalized = presets.get("normalized") if isinstance(presets, dict) else None
    expected_mask = "cls-row" if topology["pooling"] == "cls-token" else "attention-mask"
    if (
        not isinstance(context, dict)
        or context.get("declared_max_tokens") != topology["max_seqlen"]
        or not isinstance(normalized, dict)
        or normalized.get("output_width") != topology["hidden_size"]
        or normalized.get("hidden_tap") != "last-hidden-state"
        or normalized.get("pooling") != topology["pooling"]
        or normalized.get("special_tokens") != "include"
        or normalized.get("mask_domain") != expected_mask
    ):
        raise ConversionError("GENEB catalog normalized embedding contract differs")
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
    descriptor_path: Path, tokenizer_root: Optional[Path], artifact_root: Path
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
            "tokenizer asset must already be staged relative to the runtime artifact"
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


def validate_config(path: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
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

    def effective(key: str, fallback: Any = None) -> Any:
        if key in config:
            return config[key]
        if key in defaults:
            return defaults[key]
        return fallback

    topology = profile["topology"]
    torch_dtype = effective("torch_dtype")
    if torch_dtype in (None, "float32"):
        dtype = "F32"
    elif torch_dtype == "bfloat16":
        dtype = "BF16"
    else:
        dtype = None
    policy = profile["source_omit_policy"]
    semantic = {
        "vocab_size": effective("vocab_size"),
        "hidden_size": effective("hidden_size"),
        "num_layers": effective("num_hidden_layers"),
        "num_attention_heads": effective("num_attention_heads"),
        "head_dim": (
            effective("hidden_size") // effective("num_attention_heads")
            if isinstance(effective("hidden_size"), int)
            and isinstance(effective("num_attention_heads"), int)
            and effective("num_attention_heads") > 0
            else None
        ),
        "inner_mlp_size": effective("intermediate_size"),
        "max_seqlen": effective("max_position_embeddings"),
        "type_vocab_size": effective("type_vocab_size"),
        "layer_norm_epsilon": effective("layer_norm_eps"),
        "pooling": topology["pooling"],
        "attention_bias": True,
        "mlp_output_bias": True,
        "embedding_dtype": dtype,
        "projection_dtype": dtype,
        "norm_dtype": dtype,
        "activation_dtype": dtype,
    }
    if policy.startswith("gena-"):
        semantic.update(
            {
                "rope_base": 0.0,
                "position_encoding": "absolute",
                "norm_placement": "pre" if effective("pre_layer_norm", False) else "post",
                "final_layer_norm": bool(effective("last_layer_norm", False)),
                "unpad_masked_tokens": False,
                "mlp_kind": "gelu" if effective("hidden_act") == "gelu" else None,
                "qkv_layout": "separate",
                "input_kind": "token-ids",
                "mlp_input_bias": True,
            }
        )
    elif policy in ("mosaic-pooler", "mosaic-mlm"):
        semantic.update(
            {
                "rope_base": 0.0,
                "position_encoding": (
                    "alibi-symmetric"
                    if effective("alibi_starting_size") == effective("max_position_embeddings")
                    else None
                ),
                "norm_placement": "post",
                "final_layer_norm": False,
                "unpad_masked_tokens": True,
                "mlp_kind": "gated-gelu" if effective("hidden_act") == "gelu" else None,
                "qkv_layout": "fused-qkv",
                "input_kind": "token-ids",
                "mlp_input_bias": False,
            }
        )
    elif policy == "mutbert-mlm":
        semantic.update(
            {
                "rope_base": effective("rope_theta"),
                "position_encoding": (
                    "rope-split-half" if effective("rope_scaling") is None else None
                ),
                "norm_placement": "post",
                "final_layer_norm": False,
                "unpad_masked_tokens": False,
                "mlp_kind": "gelu" if effective("hidden_act") == "gelu" else None,
                "qkv_layout": "separate",
                "input_kind": "soft-vocabulary",
                "mlp_input_bias": True,
            }
        )
    else:
        semantic.update(
            {
                "rope_base": 0.0,
                "position_encoding": (
                    "absolute"
                    if effective("position_embedding_type", "absolute") == "absolute"
                    else None
                ),
                "norm_placement": "post",
                "final_layer_norm": False,
                "unpad_masked_tokens": False,
                "mlp_kind": "gelu" if effective("hidden_act") == "gelu" else None,
                "qkv_layout": "separate",
                "input_kind": "token-ids",
                "mlp_input_bias": True,
            }
        )
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
    projection = topology["projection_dtype"]
    norm = topology["norm_dtype"]
    specs = [
        TensorSpec(
            "bert.embeddings.word_embeddings.weight",
            topology["embedding_dtype"],
            (topology["vocab_size"], hidden),
        )
    ]
    if topology["position_encoding"] == "absolute":
        specs.append(
            TensorSpec(
                "bert.embeddings.position_embeddings.weight",
                topology["embedding_dtype"],
                (topology["max_seqlen"], hidden),
            )
        )
    specs.extend(
        [
            TensorSpec(
                "bert.embeddings.token_type_embeddings.weight",
                topology["embedding_dtype"],
                (topology["type_vocab_size"], hidden),
            ),
            TensorSpec("bert.embeddings.LayerNorm.weight", norm, (hidden,)),
            TensorSpec("bert.embeddings.LayerNorm.bias", norm, (hidden,)),
        ]
    )
    for layer in range(topology["num_layers"]):
        prefix = "bert.encoder.layer.%d." % layer
        if topology["norm_placement"] == "pre":
            specs.extend(
                [
                    TensorSpec(prefix + "pre_attention_ln.weight", norm, (hidden,)),
                    TensorSpec(prefix + "pre_attention_ln.bias", norm, (hidden,)),
                ]
            )
        if topology["qkv_layout"] == "separate":
            for name in ("query", "key", "value"):
                specs.append(
                    TensorSpec(
                        prefix + "attention.self.%s.weight" % name,
                        projection,
                        (hidden, hidden),
                    )
                )
                if topology["attention_bias"]:
                    specs.append(
                        TensorSpec(
                            prefix + "attention.self.%s.bias" % name,
                            projection,
                            (hidden,),
                        )
                    )
        else:
            specs.append(
                TensorSpec(
                    prefix + "attention.self.Wqkv.weight",
                    projection,
                    (3 * hidden, hidden),
                )
            )
            if topology["attention_bias"]:
                specs.append(
                    TensorSpec(
                        prefix + "attention.self.Wqkv.bias",
                        projection,
                        (3 * hidden,),
                    )
                )
        specs.append(
            TensorSpec(
                prefix + "attention.output.dense.weight",
                projection,
                (hidden, hidden),
            )
        )
        if topology["attention_bias"]:
            specs.append(
                TensorSpec(
                    prefix + "attention.output.dense.bias", projection, (hidden,)
                )
            )
        if topology["norm_placement"] == "post":
            specs.extend(
                [
                    TensorSpec(
                        prefix + "attention.output.LayerNorm.weight", norm, (hidden,)
                    ),
                    TensorSpec(
                        prefix + "attention.output.LayerNorm.bias", norm, (hidden,)
                    ),
                ]
            )
        if topology["mlp_kind"] == "gelu":
            specs.append(
                TensorSpec(
                    prefix + "intermediate.dense.weight",
                    projection,
                    (inner, hidden),
                )
            )
            if topology["mlp_input_bias"]:
                specs.append(
                    TensorSpec(
                        prefix + "intermediate.dense.bias", projection, (inner,)
                    )
                )
            specs.append(
                TensorSpec(
                    prefix + "output.dense.weight", projection, (hidden, inner)
                )
            )
            if topology["mlp_output_bias"]:
                specs.append(
                    TensorSpec(prefix + "output.dense.bias", projection, (hidden,))
                )
        else:
            specs.append(
                TensorSpec(
                    prefix + "mlp.gated_layers.weight",
                    projection,
                    (2 * inner, hidden),
                )
            )
            if topology["mlp_input_bias"]:
                specs.append(
                    TensorSpec(
                        prefix + "mlp.gated_layers.bias", projection, (2 * inner,)
                    )
                )
            specs.append(
                TensorSpec(prefix + "mlp.wo.weight", projection, (hidden, inner))
            )
            if topology["mlp_output_bias"]:
                specs.append(
                    TensorSpec(prefix + "mlp.wo.bias", projection, (hidden,))
                )
        if topology["norm_placement"] == "pre":
            specs.extend(
                [
                    TensorSpec(prefix + "post_attention_ln.weight", norm, (hidden,)),
                    TensorSpec(prefix + "post_attention_ln.bias", norm, (hidden,)),
                ]
            )
        elif topology["mlp_kind"] == "gated-gelu":
            specs.extend(
                [
                    TensorSpec(prefix + "mlp.layernorm.weight", norm, (hidden,)),
                    TensorSpec(prefix + "mlp.layernorm.bias", norm, (hidden,)),
                ]
            )
        else:
            specs.extend(
                [
                    TensorSpec(prefix + "output.LayerNorm.weight", norm, (hidden,)),
                    TensorSpec(prefix + "output.LayerNorm.bias", norm, (hidden,)),
                ]
            )
    if topology["final_layer_norm"]:
        specs.extend(
            [
                TensorSpec("bert.encoder.last_layer_ln.weight", norm, (hidden,)),
                TensorSpec("bert.encoder.last_layer_ln.bias", norm, (hidden,)),
            ]
        )
    return specs


def _source_name(canonical_name: str, source_prefix: str) -> str:
    if source_prefix == "":
        if not canonical_name.startswith("bert."):
            raise ConversionError("internal canonical BERT prefix is missing")
        return canonical_name[len("bert.") :]
    return canonical_name


def _standard_mlm_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    hidden = topology["hidden_size"]
    vocab = topology["vocab_size"]
    dtype = topology["projection_dtype"]
    norm = topology["norm_dtype"]
    return [
        TensorSpec("cls.predictions.bias", dtype, (vocab,)),
        TensorSpec("cls.predictions.transform.dense.weight", dtype, (hidden, hidden)),
        TensorSpec("cls.predictions.transform.dense.bias", dtype, (hidden,)),
        TensorSpec("cls.predictions.transform.LayerNorm.weight", norm, (hidden,)),
        TensorSpec("cls.predictions.transform.LayerNorm.bias", norm, (hidden,)),
        TensorSpec("cls.predictions.decoder.weight", dtype, (vocab, hidden)),
        TensorSpec("cls.predictions.decoder.bias", dtype, (vocab,)),
    ]


def source_tensor_specs(
    profile: Mapping[str, Any]
) -> Tuple[List[TensorSpec], List[TensorSpec]]:
    topology = profile["topology"]
    runtime = canonical_tensor_specs(topology)
    source_prefix = profile["source_prefix"]
    source = [
        TensorSpec(_source_name(spec.name, source_prefix), spec.dtype, spec.shape)
        for spec in runtime
    ]
    hidden = topology["hidden_size"]
    vocab = topology["vocab_size"]
    dtype = topology["projection_dtype"]
    norm = topology["norm_dtype"]
    policy = profile["source_omit_policy"]
    if policy.startswith("gena-"):
        source.append(
            TensorSpec(
                "bert.embeddings.position_ids",
                "I64",
                (1, topology["max_seqlen"]),
            )
        )
    if policy in ("gena-mlm-position-ids", "bert-mlm-pooler", "mutbert-mlm"):
        source.extend(_standard_mlm_specs(topology))
    elif policy == "gena-pretraining-position-ids":
        source.extend(
            [
                TensorSpec("bert.pooler.dense.weight", dtype, (hidden, hidden)),
                TensorSpec("bert.pooler.dense.bias", dtype, (hidden,)),
            ]
        )
        source.extend(_standard_mlm_specs(topology))
        source.extend(
            [
                TensorSpec("cls.seq_relationship.weight", dtype, (2, hidden)),
                TensorSpec("cls.seq_relationship.bias", dtype, (2,)),
            ]
        )
    elif policy == "mosaic-pooler":
        source.extend(
            [
                TensorSpec("pooler.dense.weight", dtype, (hidden, hidden)),
                TensorSpec("pooler.dense.bias", dtype, (hidden,)),
            ]
        )
    elif policy == "mosaic-mlm":
        source.extend(
            [
                TensorSpec(
                    "cls.predictions.transform.dense.weight", dtype, (hidden, hidden)
                ),
                TensorSpec("cls.predictions.transform.dense.bias", dtype, (hidden,)),
                TensorSpec(
                    "cls.predictions.transform.LayerNorm.weight", norm, (hidden,)
                ),
                TensorSpec(
                    "cls.predictions.transform.LayerNorm.bias", norm, (hidden,)
                ),
                TensorSpec("cls.predictions.decoder.weight", dtype, (vocab, hidden)),
                TensorSpec("cls.predictions.decoder.bias", dtype, (vocab,)),
            ]
        )
    if policy == "bert-mlm-pooler":
        source.extend(
            [
                TensorSpec("bert.pooler.dense.weight", dtype, (hidden, hidden)),
                TensorSpec("bert.pooler.dense.bias", dtype, (hidden,)),
            ]
        )
    return source, runtime


@dataclasses.dataclass(frozen=True)
class AliasedTensorSource:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    nbytes: int
    source: Any

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        return self.source.iter_chunks(chunk_size)


def read_source_tensors(
    paths: Mapping[str, Path], profile: Mapping[str, Any]
) -> List[Any]:
    if "config.json" not in paths:
        raise ConversionError("HF source receipt is missing config.json")
    allowed_dtypes = (
        ("F32", "BF16", "I64")
        if profile["source_omit_policy"].startswith("gena-")
        else ("F32", "BF16")
    )
    if profile["source_format"] == "safetensors":
        return read_hf_safetensors(
            select_hf_safetensors_paths(paths),
            allowed_dtypes=allowed_dtypes,
        )
    # Receipt verification happens before this optional offline torch import.
    return load_hf_torch_checkpoints(
        select_hf_torch_paths(paths), allowed_dtypes=allowed_dtypes
    )


def validated_runtime_tensors(
    source_tensors: Sequence[Any], profile: Mapping[str, Any]
) -> List[AliasedTensorSource]:
    expected_source, runtime_specs = source_tensor_specs(profile)
    ordered = validate_tensor_manifest(source_tensors, expected_source)
    if len(ordered) < len(runtime_specs):
        raise ConversionError("internal runtime/source tensor manifest order is invalid")
    runtime = []  # type: List[AliasedTensorSource]
    for canonical, actual in zip(runtime_specs, ordered[: len(runtime_specs)]):
        runtime.append(
            AliasedTensorSource(
                canonical.name,
                actual.dtype,
                tuple(actual.shape),
                actual.nbytes,
                actual,
            )
        )
    return runtime


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
        if tensor.dtype not in ("F32", "BF16"):
            raise ConversionError(
                "runtime tensor %r has forbidden dtype %s" % (tensor.name, tensor.dtype)
            )
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
        "source.tensor_omit_policy": profile["source_omit_policy"],
        "encoder.hidden_tap": "last-hidden-state",
        "encoder.special_tokens": "include",
        "encoder.mask_domain": (
            "cls-row" if topology["pooling"] == "cls-token" else "attention-mask"
        ),
    }  # type: Dict[str, Any]
    for key in sorted(TOPOLOGY_KEYS):
        metadata["encoder." + key] = topology[key]
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
        "--profiles", type=Path, default=default_config_path("geneb-bert-models.json")
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
            raise ConversionError("source receipt does not identify a BERT profile")
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
        )
        source_tensors = read_source_tensors(paths, profile)
        runtime_tensors = validated_runtime_tensors(source_tensors, profile)
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
        print("tensor_omit_policy=%s" % profile["source_omit_policy"])
        return 0
    except (
        CheckpointError,
        ConversionError,
        FileExistsError,
        GenebArtifactError,
        OSError,
        ValueError,
    ) as error:
        print("convert_geneb_bert_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
