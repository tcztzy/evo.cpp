#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile pinned LucaOne/Genomics-FM checkpoints for the native CPU runtime."""

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
    validate_tensor_manifest,
)


ARTIFACT_PROFILE = "geneb-custom-encoder-runtime-v1"
RUNTIME_ABI = "geneb-custom-encoder-safetensors-v1"
RUNTIME_ARCHITECTURE = "GenebCustomEncoder"
PROFILE_FORMAT = "geneb-custom-encoder-converter-v1"
TOKENIZER_PROFILE = "evo-tokenizer-v1"
MAX_HEADER_SIZE = 16 * 1024 * 1024
CHUNK_SIZE = 16 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
KEY_RE = re.compile(r"[A-Za-z0-9._-]+")

PROFILE_KEYS = {
    "runtime_id",
    "geneb_model_id",
    "paper_name",
    "catalog_architecture",
    "source_kind",
    "repo",
    "requested_revision",
    "revision",
    "source_url",
    "source_format",
    "source_files",
    "config_sha256",
    "source_prefix",
    "source_omit_policy",
    "tokenizer_manifest",
    "topology",
}
TOPOLOGY_KEYS = {
    "variant",
    "vocab_size",
    "tokenizer_vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "head_dim",
    "inner_size",
    "max_seqlen",
    "type_vocab_size",
    "pad_token_id",
    "cls_token_id",
    "sep_token_id",
    "layer_norm_epsilon",
    "rope_base",
    "position_encoding",
    "norm_placement",
    "qkv_layout",
    "mlp_kind",
    "pooling",
    "attention_bias",
    "mlp_input_bias",
    "mlp_output_bias",
    "embedding_layer_norm",
    "final_layer_norm",
    "unpad_masked_tokens",
    "token_type_embeddings",
    "weight_dtype",
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
    """Raised when source data differs from the frozen conversion contract."""


def exact_keys(value: Mapping[str, Any], required: Iterable[str], optional: Iterable[str], label: str) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    actual = set(value)
    if not required_set <= actual or not actual <= allowed:
        raise ConversionError(
            "%s fields differ: missing=%s extra=%s"
            % (label, sorted(required_set - actual), sorted(actual - allowed))
        )


def text(value: Any, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value) or "\x00" in value:
        raise ConversionError("%s must be a %sstring" % (label, "nonempty " if not allow_empty else ""))
    return value


def uint(value: Any, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0) or value > UINT64_MAX:
        raise ConversionError("%s must be a%s uint64" % (label, " positive" if positive else ""))
    return value


def number(value: Any, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ConversionError("%s must be finite%s" % (label, " and positive" if positive else ""))
    return result


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
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
        "tokenizer_vocab_size",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "head_dim",
        "inner_size",
        "max_seqlen",
        "type_vocab_size",
    ):
        topology[key] = uint(topology[key], label + "." + key, True)
    for key in ("pad_token_id", "cls_token_id", "sep_token_id"):
        topology[key] = uint(topology[key], label + "." + key)
    topology["layer_norm_epsilon"] = number(topology["layer_norm_epsilon"], label + ".layer_norm_epsilon", True)
    topology["rope_base"] = number(topology["rope_base"], label + ".rope_base")
    if topology["hidden_size"] != topology["num_attention_heads"] * topology["head_dim"]:
        raise ConversionError(label + " attention geometry differs")
    if topology["tokenizer_vocab_size"] > topology["vocab_size"]:
        raise ConversionError(label + " tokenizer vocabulary exceeds model vocabulary")
    if len({topology["pad_token_id"], topology["cls_token_id"], topology["sep_token_id"]}) != 3:
        raise ConversionError(label + " special token IDs must be distinct")
    if any(
        topology[key] >= topology["tokenizer_vocab_size"]
        for key in ("pad_token_id", "cls_token_id", "sep_token_id")
    ):
        raise ConversionError(label + " special token ID exceeds tokenizer vocabulary")
    for key in (
        "attention_bias",
        "mlp_input_bias",
        "mlp_output_bias",
        "embedding_layer_norm",
        "final_layer_norm",
        "unpad_masked_tokens",
        "token_type_embeddings",
    ):
        if not isinstance(topology[key], bool):
            raise ConversionError(label + "." + key + " must be boolean")
    if topology["weight_dtype"] != "F32":
        raise ConversionError(label + " only admits F32")
    if topology["variant"] == "lucaone":
        exact = (
            topology["tokenizer_vocab_size"] == topology["vocab_size"]
            and topology["type_vocab_size"] == 2
            and (topology["pad_token_id"], topology["cls_token_id"], topology["sep_token_id"]) == (0, 2, 3)
            and topology["layer_norm_epsilon"] == 1.0e-5
            and topology["rope_base"] == 10000.0
            and topology["position_encoding"] == "rope-split-half"
            and topology["norm_placement"] == "pre"
            and topology["qkv_layout"] == "separate"
            and topology["mlp_kind"] == "gelu"
            and topology["pooling"] == "attention-mask-mean"
            and topology["attention_bias"]
            and topology["mlp_input_bias"]
            and topology["mlp_output_bias"]
            and not topology["embedding_layer_norm"]
            and topology["final_layer_norm"]
            and not topology["unpad_masked_tokens"]
            and topology["token_type_embeddings"]
            and topology["head_dim"] % 2 == 0
        )
    elif topology["variant"] == "genomics-fm":
        exact = (
            topology["type_vocab_size"] == 2
            and (topology["pad_token_id"], topology["cls_token_id"], topology["sep_token_id"]) == (3, 1, 2)
            and topology["layer_norm_epsilon"] == 1.0e-12
            and topology["rope_base"] == 0.0
            and topology["position_encoding"] == "absolute"
            and topology["norm_placement"] == "post"
            and topology["qkv_layout"] == "fused-qkv"
            and topology["mlp_kind"] == "gated-gelu"
            and topology["pooling"] == "cls-token"
            and topology["attention_bias"]
            and not topology["mlp_input_bias"]
            and topology["mlp_output_bias"]
            and topology["embedding_layer_norm"]
            and not topology["final_layer_norm"]
            and topology["unpad_masked_tokens"]
            and topology["token_type_embeddings"]
        )
    else:
        raise ConversionError(label + ".variant is unsupported")
    if not exact:
        raise ConversionError(label + " mixes unsupported custom-encoder semantics")
    return topology


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    root, payload = load_json(path, "custom encoder profile manifest")
    exact_keys(root, ["schema_version", "format", "models"], [], "profile manifest")
    if root["schema_version"] != 1 or root["format"] != PROFILE_FORMAT or not isinstance(root["models"], list):
        raise ConversionError("custom encoder profile manifest schema is unsupported")
    result = {}  # type: Dict[str, Dict[str, Any]]
    for index, raw in enumerate(root["models"]):
        label = "profile models[%d]" % index
        if not isinstance(raw, dict) or set(raw) != PROFILE_KEYS:
            raise ConversionError(label + " keys differ")
        profile = dict(raw)
        runtime_id = text(profile["runtime_id"], label + ".runtime_id")
        if runtime_id in result:
            raise ConversionError("duplicate custom encoder runtime_id")
        for key in ("geneb_model_id", "paper_name", "catalog_architecture", "source_kind", "source_url", "source_format", "config_sha256", "source_prefix", "source_omit_policy", "tokenizer_manifest"):
            profile[key] = text(profile[key], label + "." + key)
        for key in ("repo", "requested_revision", "revision"):
            profile[key] = text(profile[key], label + "." + key, True)
        if profile["source_kind"] not in ("huggingface", "google-drive") or profile["source_format"] not in ("safetensors", "pytorch-bin"):
            raise ConversionError(label + " source kind/format is unsupported")
        if profile["source_kind"] == "huggingface":
            if not profile["repo"] or not profile["requested_revision"] or not COMMIT_RE.fullmatch(profile["revision"]):
                raise ConversionError(label + " Hugging Face identity is not pinned")
        elif profile["repo"] or profile["requested_revision"] or profile["revision"]:
            raise ConversionError(label + " mutable provider must not invent a revision")
        if not SHA256_RE.fullmatch(profile["config_sha256"]):
            raise ConversionError(label + ".config_sha256 is invalid")
        expected_files = {}  # type: Dict[str, Dict[str, Any]]
        if not isinstance(profile["source_files"], list) or not profile["source_files"]:
            raise ConversionError(label + ".source_files must be nonempty")
        for file_index, item in enumerate(profile["source_files"]):
            file_label = label + ".source_files[%d]" % file_index
            if not isinstance(item, dict):
                raise ConversionError(file_label + " must be an object")
            exact_keys(item, ["name", "size", "sha256"], [], file_label)
            name = normalized_relative_path(item["name"], file_label + ".name")
            size = uint(item["size"], file_label + ".size")
            digest = text(item["sha256"], file_label + ".sha256")
            if name in expected_files or not SHA256_RE.fullmatch(digest):
                raise ConversionError(file_label + " duplicates a name or has invalid SHA256")
            expected_files[name] = {"name": name, "size": size, "sha256": digest}
        if "config.json" not in expected_files or expected_files["config.json"]["sha256"] != profile["config_sha256"]:
            raise ConversionError(label + " config source file disagrees")
        profile["source_files_by_name"] = expected_files
        profile["topology"] = validate_topology(profile["topology"], label + ".topology")
        manifest_relative = normalized_relative_path(
            profile["tokenizer_manifest"], label + ".tokenizer_manifest"
        )
        manifest_path = (
            path.parent.parent / manifest_relative
            if manifest_relative.startswith("configs/")
            else path.parent / manifest_relative
        )
        try:
            profile["tokenizer_manifest_sha256"] = sha256_file(
                manifest_path.resolve()
            )
        except OSError as error:
            raise ConversionError(
                label + " tokenizer manifest is unavailable: %s" % error
            ) from error
        variant = profile["topology"]["variant"]
        if (variant == "lucaone") != (profile["source_prefix"] == "lucaone." and profile["source_omit_policy"] == "lucaone-mlm-head"):
            raise ConversionError(label + " LucaOne source mapping differs")
        if (variant == "genomics-fm") != (profile["source_prefix"] == "bert." and profile["source_omit_policy"] == "genomics-fm-mlm-head"):
            raise ConversionError(label + " Genomics-FM source mapping differs")
        result[runtime_id] = profile
    if set(result) != {"geneb-lucaone", "geneb-genomics-fm"}:
        raise ConversionError("profile manifest must contain exactly both custom encoders")
    return result, payload


def load_catalog(path: Path, profile: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    root, payload = load_json(path, "GENEB catalog")
    matches = [item for item in root.get("models", []) if isinstance(item, dict) and item.get("runtime_id") == profile["runtime_id"]]
    if root.get("schema_version") != 1 or len(matches) != 1:
        raise ConversionError("GENEB catalog custom encoder identity is missing")
    entry = matches[0]
    expected = {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "architecture": profile["catalog_architecture"],
        "family": "transformer-encoder",
    }
    if any(entry.get(key) != value for key, value in expected.items()):
        raise ConversionError("GENEB catalog custom encoder identity differs")
    source = entry.get("source")
    if not isinstance(source, dict) or source.get("kind") != profile["source_kind"] or source.get("url") != profile["source_url"]:
        raise ConversionError("GENEB catalog source provider differs")
    if profile["source_kind"] == "huggingface":
        if source.get("repo") != profile["repo"] or source.get("requested_revision") != profile["requested_revision"] or source.get("revision") != profile["revision"] or source.get("immutable") is not True:
            raise ConversionError("GENEB catalog pinned source differs")
    elif source.get("immutable") is not False or source.get("revision") is not None:
        raise ConversionError("GENEB mutable source contract differs")
    topology = profile["topology"]
    tokenizer = entry.get("tokenizer")
    context = entry.get("context")
    presets = entry.get("embedding_presets")
    normalized = presets.get("normalized") if isinstance(presets, dict) else None
    expected_kind = "character" if topology["variant"] == "lucaone" else "bpe"
    expected_mask = "attention-mask" if topology["variant"] == "lucaone" else "cls-row"
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("kind") != expected_kind
        or tokenizer.get("padding_side") != "right"
        or tokenizer.get("max_tokens") != topology["max_seqlen"]
        or not isinstance(context, dict)
        or context.get("declared_max_tokens") != topology["max_seqlen"]
        or not isinstance(normalized, dict)
        or normalized.get("output_width") != topology["hidden_size"]
        or normalized.get("pooling") != topology["pooling"]
        or normalized.get("special_tokens") != "include"
        or normalized.get("mask_domain") != expected_mask
    ):
        raise ConversionError("GENEB catalog tokenizer/context/pooling contract differs")
    return entry, root, payload


def validate_receipt(
    path: Path,
    profile: Mapping[str, Any],
    catalog_path: Path,
    catalog_payload: bytes,
    catalog_entry: Mapping[str, Any],
) -> Tuple[Dict[str, Path], bytes]:
    receipt, payload = load_json(path, "source receipt")
    if profile["source_kind"] == "huggingface":
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
                "source_kind",
            ],
            [
                "catalog_path",
                "catalog_sha256",
                "catalog_contract_sha256",
                "load_path",
            ],
            "source receipt",
        )
        if (
            receipt["schema_version"] != 1
            or receipt["kind"] != "source-checkpoint"
            or receipt["model_id"] != profile["runtime_id"]
            or receipt["repo"] != profile["repo"]
            or receipt["requested_revision"] != profile["requested_revision"]
            or receipt["resolved_revision"] != profile["revision"]
            or receipt["source_kind"] != "huggingface"
            or ("load_path" in receipt and receipt["load_path"] is not None)
        ):
            raise ConversionError("source receipt provider/model/revision differs")
        validate_hf_fetch_receipt_provenance(
            receipt, catalog_path, catalog_payload, catalog_entry
        )
    else:
        exact_keys(
            receipt,
            [
                "schema_version",
                "kind",
                "model_id",
                "source_kind",
                "source_url",
                "files",
            ],
            [],
            "source receipt",
        )
        if (
            receipt["schema_version"] != 1
            or receipt["kind"] != "source-checkpoint"
            or receipt["model_id"] != profile["runtime_id"]
            or receipt["source_kind"] != profile["source_kind"]
            or receipt["source_url"] != profile["source_url"]
        ):
            raise ConversionError("source receipt provider/model/revision differs")
    expected = profile["source_files_by_name"]
    files = receipt["files"]
    if not isinstance(files, list):
        raise ConversionError("source receipt files must be an array")
    paths = {}  # type: Dict[str, Path]
    for index, raw in enumerate(files):
        label = "source receipt files[%d]" % index
        if not isinstance(raw, dict):
            raise ConversionError(label + " must be an object")
        exact_keys(raw, ["name", "size", "sha256", "path"], [], label)
        name = normalized_relative_path(raw["name"], label + ".name")
        if name in paths or name not in expected:
            raise ConversionError(label + " is duplicate or unexpected")
        registered = expected[name]
        if raw["size"] != registered["size"] or raw["sha256"] != registered["sha256"]:
            raise ConversionError("source receipt pinned size/SHA256 differs for " + name)
        source_path = Path(text(raw["path"], label + ".path")).resolve()
        if source_path.stat().st_size != registered["size"] or sha256_file(source_path) != registered["sha256"]:
            raise ConversionError("source receipt payload integrity differs for " + name)
        paths[name] = source_path
    if set(paths) != set(expected):
        raise ConversionError("source receipt has missing files: %s" % sorted(set(expected) - set(paths)))
    return paths, payload


def validate_config(path: Path, profile: Mapping[str, Any]) -> None:
    config, payload = load_json(path, "source config")
    if sha256_bytes(payload) != profile["config_sha256"]:
        raise ConversionError("source config SHA256 differs from profile")
    topology = profile["topology"]
    common = {
        "vocab_size": topology["vocab_size"],
        "hidden_size": topology["hidden_size"],
        "num_hidden_layers": topology["num_layers"],
        "num_attention_heads": topology["num_attention_heads"],
        "max_position_embeddings": topology["max_seqlen"],
        "type_vocab_size": topology["type_vocab_size"],
        "torch_dtype": "float32",
    }
    if topology["variant"] == "lucaone":
        semantic = dict(common)
        semantic.update(
            {
                "ffn_dim": topology["inner_size"],
                "pad_token_id": 0,
                "bos_token_id": 2,
                "sep_token_id": 3,
                "no_position_embeddings": True,
                "no_token_type_embeddings": False,
                "use_embed_layer_norm": False,
                "use_last_layer_norm": True,
                "embed_scale": 1.0,
                "token_dropout": False,
                "hidden_act": "gelu",
                # The source constructs torch/apex LayerNorm without passing this
                # config field, so the executable epsilon is the framework 1e-5.
                "layer_norm_eps": 1.0e-12,
            }
        )
    else:
        semantic = dict(common)
        semantic.update(
            {
                "intermediate_size": topology["inner_size"],
                "pad_token_id": 0,
                "position_embedding_type": "absolute",
                "layer_norm_eps": topology["layer_norm_epsilon"],
                "hidden_act": "gelu",
                "attention_probs_dropout_prob": 0.0,
            }
        )
    wrong = {key: (config.get(key), value) for key, value in semantic.items() if config.get(key) != value}
    if wrong:
        raise ConversionError("source config semantic topology differs: %s" % wrong)


def canonical_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    variant = topology["variant"]
    hidden = topology["hidden_size"]
    inner = topology["inner_size"]
    if variant == "genomics-fm":
        result = [
            TensorSpec("bert.embeddings.word_embeddings.weight", "F32", (topology["vocab_size"], hidden)),
            TensorSpec("bert.embeddings.position_embeddings.weight", "F32", (topology["max_seqlen"], hidden)),
            TensorSpec("bert.embeddings.token_type_embeddings.weight", "F32", (topology["type_vocab_size"], hidden)),
            TensorSpec("bert.embeddings.LayerNorm.weight", "F32", (hidden,)),
            TensorSpec("bert.embeddings.LayerNorm.bias", "F32", (hidden,)),
        ]
        for layer in range(topology["num_layers"]):
            prefix = "bert.encoder.layer.%d." % layer
            result.extend(
                [
                    TensorSpec(prefix + "attention.self.Wqkv.weight", "F32", (3 * hidden, hidden)),
                    TensorSpec(prefix + "attention.self.Wqkv.bias", "F32", (3 * hidden,)),
                    TensorSpec(prefix + "attention.output.dense.weight", "F32", (hidden, hidden)),
                    TensorSpec(prefix + "attention.output.dense.bias", "F32", (hidden,)),
                    TensorSpec(prefix + "attention.output.LayerNorm.weight", "F32", (hidden,)),
                    TensorSpec(prefix + "attention.output.LayerNorm.bias", "F32", (hidden,)),
                    TensorSpec(prefix + "mlp.gated_layers.weight", "F32", (2 * inner, hidden)),
                    TensorSpec(prefix + "mlp.wo.weight", "F32", (hidden, inner)),
                    TensorSpec(prefix + "mlp.wo.bias", "F32", (hidden,)),
                    TensorSpec(prefix + "mlp.layernorm.weight", "F32", (hidden,)),
                    TensorSpec(prefix + "mlp.layernorm.bias", "F32", (hidden,)),
                ]
            )
        return result
    result = [
        TensorSpec("lucaone.embeddings.embed_tokens.weight", "F32", (topology["vocab_size"], hidden)),
        TensorSpec("lucaone.embeddings.embed_type.weight", "F32", (topology["type_vocab_size"], hidden)),
    ]
    for layer in range(topology["num_layers"]):
        prefix = "lucaone.encoder.layers.%d." % layer
        result.extend(
            [
                TensorSpec(prefix + "pre_layer_norm.weight", "F32", (hidden,)),
                TensorSpec(prefix + "pre_layer_norm.bias", "F32", (hidden,)),
            ]
        )
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            result.extend(
                [
                    TensorSpec(prefix + "self_attn." + projection + ".weight", "F32", (hidden, hidden)),
                    TensorSpec(prefix + "self_attn." + projection + ".bias", "F32", (hidden,)),
                ]
            )
        result.extend(
            [
                TensorSpec(prefix + "self_attn.rot_emb.inv_freq", "F32", (topology["head_dim"] // 2,)),
                TensorSpec(prefix + "post_layer_norm.weight", "F32", (hidden,)),
                TensorSpec(prefix + "post_layer_norm.bias", "F32", (hidden,)),
                TensorSpec(prefix + "fc1.weight", "F32", (inner, hidden)),
                TensorSpec(prefix + "fc1.bias", "F32", (inner,)),
                TensorSpec(prefix + "fc2.weight", "F32", (hidden, inner)),
                TensorSpec(prefix + "fc2.bias", "F32", (hidden,)),
            ]
        )
    result.extend(
        [
            TensorSpec("lucaone.encoder.last_layer_norm.weight", "F32", (hidden,)),
            TensorSpec("lucaone.encoder.last_layer_norm.bias", "F32", (hidden,)),
        ]
    )
    return result


def source_specs(profile: Mapping[str, Any]) -> Tuple[List[TensorSpec], List[TensorSpec]]:
    topology = profile["topology"]
    runtime = canonical_specs(topology)
    source = list(runtime)
    hidden = topology["hidden_size"]
    vocab = topology["vocab_size"]
    if topology["variant"] == "lucaone":
        source.extend(
            [
                TensorSpec("lm_head.bias", "F32", (vocab,)),
                TensorSpec("lm_head.dense.weight", "F32", (hidden, hidden)),
                TensorSpec("lm_head.dense.bias", "F32", (hidden,)),
                TensorSpec("lm_head.layer_norm.weight", "F32", (hidden,)),
                TensorSpec("lm_head.layer_norm.bias", "F32", (hidden,)),
                TensorSpec("lm_head.decoder.weight", "F32", (vocab, hidden)),
            ]
        )
    else:
        source.extend(
            [
                TensorSpec("cls.predictions.decoder.bias", "F32", (vocab,)),
                TensorSpec("cls.predictions.transform.dense.weight", "F32", (hidden, hidden)),
                TensorSpec("cls.predictions.transform.dense.bias", "F32", (hidden,)),
                TensorSpec("cls.predictions.transform.LayerNorm.weight", "F32", (hidden,)),
                TensorSpec("cls.predictions.transform.LayerNorm.bias", "F32", (hidden,)),
                TensorSpec("cls.predictions.decoder.weight", "F32", (vocab, hidden)),
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


def read_source(paths: Mapping[str, Path], profile: Mapping[str, Any]) -> List[Any]:
    if profile["source_format"] == "safetensors":
        weights = {name: path for name, path in paths.items() if name.endswith(".safetensors") or name == "model.safetensors.index.json"}
        if not any(name.endswith(".safetensors") for name in weights):
            raise ConversionError("Safetensors source has no weight file")
        return read_hf_safetensors(weights, allowed_dtypes=("F32",))
    weights = {name: path for name, path in paths.items() if name == "pytorch_model.bin" or name.endswith(".bin") or name.endswith(".bin.index.json")}
    if not weights:
        raise ConversionError("PyTorch source has no weight file")
    return load_hf_torch_checkpoints(weights, allowed_dtypes=("F32",))


def runtime_tensors(source: Sequence[Any], profile: Mapping[str, Any]) -> List[AliasedTensorSource]:
    expected_source, runtime = source_specs(profile)
    ordered = validate_tensor_manifest(source, expected_source)
    result = []  # type: List[AliasedTensorSource]
    by_name = {tensor.name: tensor for tensor in ordered}
    if profile["topology"]["variant"] == "lucaone":
        head_dim = profile["topology"]["head_dim"]
        expected_inverse_payload = b"".join(
            struct.pack(
                "<f",
                struct.unpack(
                    "<e",
                    struct.pack(
                        "<e",
                        1.0
                        / math.pow(
                            profile["topology"]["rope_base"],
                            float(index * 2) / float(head_dim),
                        ),
                    ),
                )[0],
            )
            for index in range(head_dim // 2)
        )
        first_inverse_payload = None  # type: Optional[bytes]
        for layer in range(profile["topology"]["num_layers"]):
            name = "lucaone.encoder.layers.%d.self_attn.rot_emb.inv_freq" % layer
            payload = b"".join(bytes(chunk) for chunk in by_name[name].iter_chunks(CHUNK_SIZE))
            if payload != expected_inverse_payload or (
                first_inverse_payload is not None and payload != first_inverse_payload
            ):
                raise ConversionError("LucaOne rotary inv_freq buffer differs: " + name)
            first_inverse_payload = payload
    else:
        word = by_name["bert.embeddings.word_embeddings.weight"]
        decoder = by_name["cls.predictions.decoder.weight"]
        word_hash = hashlib.sha256()
        decoder_hash = hashlib.sha256()
        for chunk in word.iter_chunks(CHUNK_SIZE):
            word_hash.update(chunk)
        for chunk in decoder.iter_chunks(CHUNK_SIZE):
            decoder_hash.update(chunk)
        if word_hash.digest() != decoder_hash.digest():
            raise ConversionError("Genomics-FM MLM decoder is not tied to word embeddings")
    for spec in runtime:
        actual = by_name[spec.name]
        result.append(AliasedTensorSource(spec.name, actual.dtype, tuple(actual.shape), actual.nbytes, actual))
    return result


def validate_tokenizer_descriptor(
    descriptor_path: Path,
    tokenizer_root: Optional[Path],
    artifact_root: Path,
    topology: Mapping[str, Any],
    expected_manifest_sha256: str,
) -> Tuple[Dict[str, Any], str]:
    descriptor, descriptor_payload = load_json(descriptor_path, "tokenizer descriptor")
    if set(descriptor) != TOKENIZER_DESCRIPTOR_KEYS:
        raise ConversionError("tokenizer descriptor fields differ")
    if descriptor["converter.schema"] != "evo-tokenizer-conversion-receipt" or descriptor["converter.version"] != 1 or descriptor["tokenizer.profile"] != TOKENIZER_PROFILE:
        raise ConversionError("tokenizer descriptor schema/profile differs")
    for key in ("compiler_manifest_sha256", "source_receipt_contract_sha256", "tokenizer.sha256"):
        if not isinstance(descriptor[key], str) or not SHA256_RE.fullmatch(descriptor[key]):
            raise ConversionError("tokenizer descriptor digest is invalid: " + key)
    if descriptor["compiler_manifest_sha256"] != expected_manifest_sha256:
        raise ConversionError("tokenizer compiler manifest differs from profile")
    relative = normalized_relative_path(descriptor["tokenizer.path"], "tokenizer.path")
    size = uint(descriptor["tokenizer.size"], "tokenizer.size")
    root = (tokenizer_root if tokenizer_root is not None else descriptor_path.parent).resolve()
    asset_path = (root / relative).resolve()
    try:
        asset_path.relative_to(root)
    except ValueError as error:
        raise ConversionError("tokenizer path escapes root") from error
    if (artifact_root.resolve() / relative).resolve() != asset_path:
        raise ConversionError("tokenizer asset must be staged beside output")
    payload = asset_path.read_bytes()
    if len(payload) != size or sha256_bytes(payload) != descriptor["tokenizer.sha256"]:
        raise ConversionError("tokenizer asset size/SHA256 differs")
    asset, _ = load_json(asset_path, "compiled tokenizer asset")
    exact_keys(asset, ["format", "kind", "normalization", "pre_tokenizer", "model", "post_processor", "special_tokens", "vocab"], [], "compiled tokenizer")
    variant = topology["variant"]
    expected_kind = "character" if variant == "lucaone" else "bpe"
    vocab = asset["vocab"]
    special = asset["special_tokens"]
    post = asset["post_processor"]
    padding = post.get("padding") if isinstance(post, dict) else None
    if (
        asset["format"] != "evo-tokenizer-v1"
        or asset["kind"] != expected_kind
        or not isinstance(vocab, list)
        or len(vocab) != topology["tokenizer_vocab_size"]
        or not all(isinstance(item, dict) for item in vocab)
        or [item.get("id") for item in vocab if isinstance(item, dict)] != list(range(topology["tokenizer_vocab_size"]))
        or not isinstance(special, dict)
        or special.get("pad") != topology["pad_token_id"]
        or special.get("cls") != topology["cls_token_id"]
        or special.get("sep") != topology["sep_token_id"]
        or not isinstance(post, dict)
        or post.get("prefix_ids") != [topology["cls_token_id"]]
        or post.get("suffix_ids") != [topology["sep_token_id"]]
        or not isinstance(padding, dict)
        or padding.get("side") != "right"
        or padding.get("pad_id") != topology["pad_token_id"]
    ):
        raise ConversionError("compiled tokenizer kind/vocab/special/padding contract differs")
    if variant == "lucaone":
        if asset["normalization"] != [{"op": "ascii-uppercase"}, {"op": "u-to-t"}] or special.get("unk") != 9 or asset.get("pre_tokenizer") != {"kind": "none"}:
            raise ConversionError("LucaOne gene character mapping differs")
    else:
        expected_literals = ["[BPE]", "[KMER]", "[ALIBI]", "[APE]"]
        model = asset.get("model")
        if (
            asset["normalization"] != []
            or asset.get("pre_tokenizer") != {"kind": "hf-whitespace-ascii"}
            or not isinstance(model, dict)
            or model.get("literal_token_ids")
            != [0, 1, 2, 3, 4, 4096, 4097, 4098, 4099]
            or [vocab[index].get("piece") for index in range(4096, 4100)]
            != expected_literals
        ):
            raise ConversionError("Genomics-FM root BPE/literal-token contract differs")
    projected = {key: descriptor[key] for key in ("tokenizer.profile", "tokenizer.path", "tokenizer.sha256", "tokenizer.size")}
    return projected, sha256_bytes(descriptor_payload)


def encode_metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "b:%d" % int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return "u:%d" % uint(value, "metadata integer")
    if isinstance(value, float):
        return "f:%016x" % struct.unpack("<Q", struct.pack("<d", number(value, "metadata float")))[0]
    if isinstance(value, str):
        return "s:" + value
    raise ConversionError("unsupported metadata type")


def encoded_header(metadata: Mapping[str, Any], tensors: Sequence[Any]) -> bytes:
    encoded = {}  # type: Dict[str, str]
    for key in sorted(metadata):
        if not KEY_RE.fullmatch(key) or key == "runtime.profile":
            raise ConversionError("invalid/reserved metadata key: " + key)
        encoded[key] = encode_metadata_value(metadata[key])
    encoded["runtime.profile"] = "s:" + ARTIFACT_PROFILE
    root = {"__metadata__": dict(sorted(encoded.items()))}  # type: Dict[str, Any]
    offset = 0
    for tensor in tensors:
        end = offset + tensor.nbytes
        root[tensor.name] = {"dtype": tensor.dtype, "shape": list(tensor.shape), "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    if not header or len(header) > MAX_HEADER_SIZE:
        raise ConversionError("artifact header exceeds 16 MiB")
    return header


def write_artifact(output_path: Path, metadata: Mapping[str, Any], tensors: Sequence[Any], force: bool) -> None:
    if output_path.suffix != ".safetensors" or not tensors:
        raise ConversionError("output must be a nonempty .safetensors artifact")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError("output already exists: %s" % output_path)
    header = encoded_header(metadata, tensors)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + output_path.name + ".", suffix=".tmp", dir=str(output_path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as output:
            output.truncate(8 + len(header) + sum(tensor.nbytes for tensor in tensors))
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for tensor in tensors:
                written = 0
                for raw_chunk in tensor.iter_chunks(CHUNK_SIZE):
                    chunk = memoryview(raw_chunk).cast("B")
                    if written + len(chunk) > tensor.nbytes:
                        raise ConversionError("tensor yielded excess bytes: " + tensor.name)
                    output.write(chunk)
                    written += len(chunk)
                if written != tensor.nbytes:
                    raise ConversionError("tensor yielded wrong byte count: " + tensor.name)
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
    luca = topology["variant"] == "lucaone"
    metadata = {
        "runtime.abi": RUNTIME_ABI,
        "runtime.embedding_layer_count": topology["num_layers"] + 1,
        "runtime.tokenizer_vocabulary_size": topology["tokenizer_vocab_size"],
        "model.id": profile["runtime_id"],
        "model.architecture": RUNTIME_ARCHITECTURE,
        "config.vocab_size": topology["vocab_size"],
        "config.hidden_size": topology["hidden_size"],
        "config.num_layers": topology["num_layers"],
        "config.max_seqlen": topology["max_seqlen"],
        "source.kind": profile["source_kind"],
        "source.repo": profile["repo"],
        "source.revision": profile["revision"],
        "source.url": profile["source_url"],
        "source.receipt_sha256": receipt_sha256,
        "source.catalog_contract_sha256": catalog_sha256,
        "source.converter_profile_contract_sha256": profile_sha256,
        "source.config_sha256": profile["config_sha256"],
        "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        "source.tensor_omit_policy": profile["source_omit_policy"],
        "custom.hidden_tap": "last-hidden-state",
        "custom.special_tokens": "include-added-boundaries",
        "custom.mask_domain": "attention-mask",
        "custom.attention_mask": "bidirectional-key-mask",
        "custom.rope_layout": "split-half" if luca else "none",
        "custom.gelu": "exact-erf",
        "custom.tokenizer_kind": "character" if luca else "bpe",
        "custom.official_reference_device": "cpu-or-gpu" if luca else "gpu",
    }  # type: Dict[str, Any]
    for key in sorted(TOPOLOGY_KEYS):
        metadata["custom." + key] = topology[key]
    metadata.update(geneb_metadata)
    metadata.update(tokenizer)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=default_config_path("geneb-models.json"))
    parser.add_argument("--profiles", type=Path, default=default_config_path("geneb-custom-encoder-models.json"))
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
            raise ConversionError("receipt does not identify a custom encoder profile")
        catalog_entry, catalog, catalog_payload = load_catalog(args.catalog.resolve(), profile)
        paths, receipt_payload = validate_receipt(
            args.receipt.resolve(),
            profile,
            args.catalog.resolve(),
            catalog_payload,
            catalog_entry,
        )
        validate_config(paths["config.json"], profile)
        tokenizer, tokenizer_descriptor_sha256 = validate_tokenizer_descriptor(
            args.tokenizer_descriptor.resolve(),
            args.tokenizer_root.resolve() if args.tokenizer_root is not None else None,
            args.output.resolve().parent,
            profile["topology"],
            profile["tokenizer_manifest_sha256"],
        )
        tensors = runtime_tensors(read_source(paths, profile), profile)
        geneb_metadata = build_geneb_artifact_metadata(catalog, catalog_entry, catalog_payload)
        metadata = build_metadata(
            profile,
            tokenizer,
            sha256_bytes(receipt_payload),
            catalog_contract_sha256(catalog, catalog_entry),
            converter_profile_contract_sha256(1, PROFILE_FORMAT, profile),
            tokenizer_descriptor_sha256,
            geneb_metadata,
        )
        write_artifact(args.output, metadata, tensors, args.force)
        print("wrote %s" % args.output)
        print("source_receipt_sha256=%s" % sha256_bytes(receipt_payload))
        print("variant=%s" % profile["topology"]["variant"])
        return 0
    except (CheckpointError, ConversionError, FileExistsError, GenebArtifactError, OSError, ValueError) as error:
        print("convert_geneb_custom_encoder_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
