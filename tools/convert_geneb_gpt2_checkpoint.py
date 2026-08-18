#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile pinned GPT2-Gene checkpoints into the portable GENEB GPT-2 ABI."""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


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
from evo.geneb_gpt_artifact import (  # noqa: E402
    CanonicalF32TensorSource,
    ConversionError,
    SHA256_RE,
    default_config_path,
    exact_keys,
    finite_float,
    nonempty_string,
    object_value,
    sha256_bytes,
    validate_common_tokenizer_asset,
    validate_profile_manifest_identity,
    validate_receipt_file_entry,
    validate_tokenizer_descriptor,
    validate_tokenizer_output_binding,
    uint64_value,
    write_artifact,
)
from evo.hf_checkpoint import (  # noqa: E402
    CheckpointError,
    TensorSpec,
    load_json,
    read_hf_safetensors,
    validate_tensor_manifest,
)


ARTIFACT_PROFILE = "geneb-gpt2-runtime-v1"
RUNTIME_ABI = "geneb-gpt2-safetensors-v1"
RUNTIME_ARCHITECTURE = "GenebGpt2Decoder"
PROFILE_FORMAT = "geneb-gpt2-converter-v1"
TOKENIZER_PROFILE = "evo-tokenizer-v1"
EXPECTED_VOCAB_SIZE = 115000
TRANSFORMERS_VERSION = "4.45.2"

PROFILE_KEYS = {
    "runtime_id",
    "geneb_model_id",
    "paper_name",
    "catalog_architecture",
    "repo",
    "requested_revision",
    "revision",
    "config_sha256",
    "weights_sha256",
    "weights_size",
    "source_manifest_sha256",
    "source_tensor_count",
    "source_tensor_bytes",
    "tokenizer",
    "topology",
}
TOPOLOGY_KEYS = {
    "vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "inner_width",
    "max_seqlen",
    "norm_epsilon",
}
TOKENIZER_KEYS = {
    "tokenizer_json_sha256",
    "tokenizer_json_size",
    "tokenizer_config_sha256",
    "tokenizer_config_size",
    "special_tokens_map_sha256",
    "special_tokens_map_size",
    "compiled_asset_sha256",
    "compiled_asset_size",
    "compiler_manifest_sha256",
    "vocab_size",
    "merge_count",
    "pad_id",
}
CODE_KEYS = {
    "package",
    "version",
    "modeling_gpt2_sha256",
    "configuration_gpt2_sha256",
    "activations_sha256",
    "extractor_commit",
    "extractor_sha256",
}


def validate_topology(raw: Any, label: str) -> Dict[str, Any]:
    value = object_value(raw, label)
    if set(value) != TOPOLOGY_KEYS:
        raise ConversionError("%s fields differ" % label)
    topology = dict(value)
    for key in (
        "vocab_size",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "inner_width",
        "max_seqlen",
    ):
        topology[key] = uint64_value(topology[key], label + "." + key)
    topology["norm_epsilon"] = finite_float(
        topology["norm_epsilon"], label + ".norm_epsilon"
    )
    if (
        topology["vocab_size"] != EXPECTED_VOCAB_SIZE
        or topology["hidden_size"] % topology["num_attention_heads"] != 0
        or topology["inner_width"] != topology["hidden_size"] * 4
    ):
        raise ConversionError("%s differs from pinned GPT2-Gene topology" % label)
    return topology


def validate_tokenizer_profile(raw: Any, label: str) -> Dict[str, Any]:
    value = object_value(raw, label)
    if set(value) != TOKENIZER_KEYS:
        raise ConversionError("%s fields differ" % label)
    result = dict(value)
    for key in (
        "tokenizer_json_sha256",
        "tokenizer_config_sha256",
        "special_tokens_map_sha256",
        "compiled_asset_sha256",
        "compiler_manifest_sha256",
    ):
        result[key] = nonempty_string(result[key], label + "." + key)
        if not SHA256_RE.fullmatch(result[key]):
            raise ConversionError("%s.%s must be lowercase SHA256" % (label, key))
    for key in (
        "tokenizer_json_size",
        "tokenizer_config_size",
        "special_tokens_map_size",
        "compiled_asset_size",
        "vocab_size",
        "merge_count",
    ):
        result[key] = uint64_value(result[key], label + "." + key)
    result["pad_id"] = uint64_value(result["pad_id"], label + ".pad_id", False)
    if result["vocab_size"] != EXPECTED_VOCAB_SIZE or result["pad_id"] not in (0, 1):
        raise ConversionError("%s vocabulary/pad contract differs" % label)
    return result


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], bytes]:
    root, payload = load_json(path, "GENEB GPT-2 converter profiles")
    exact_keys(root, ["schema_version", "format", "models", "code_provenance"], [], "profiles")
    if root["schema_version"] != 1 or root["format"] != PROFILE_FORMAT:
        raise ConversionError("GPT-2 profile schema/format is unsupported")
    raw_code = object_value(root["code_provenance"], "code_provenance")
    if set(raw_code) != CODE_KEYS:
        raise ConversionError("GPT-2 code provenance fields differ")
    code = {}  # type: Dict[str, str]
    for key in CODE_KEYS:
        code[key] = nonempty_string(raw_code[key], "code_provenance." + key)
    if code["package"] != "transformers" or code["version"] != TRANSFORMERS_VERSION:
        raise ConversionError("GPT-2 Transformers provenance differs")
    for key in CODE_KEYS - {"package", "version"}:
        expected = 40 if key == "extractor_commit" else 64
        if len(code[key]) != expected or not re.fullmatch("[0-9a-f]+", code[key]):
            raise ConversionError("GPT-2 code provenance digest is invalid")
    models = root["models"]
    if not isinstance(models, list) or not models:
        raise ConversionError("GPT-2 profiles must be a nonempty array")
    profiles = {}  # type: Dict[str, Dict[str, Any]]
    identities = set()  # type: Set[Tuple[str, str]]
    for index, raw in enumerate(models):
        label = "models[%d]" % index
        profile = dict(object_value(raw, label))
        if set(profile) != PROFILE_KEYS:
            raise ConversionError("%s fields differ" % label)
        for key in (
            "runtime_id",
            "geneb_model_id",
            "paper_name",
            "catalog_architecture",
            "repo",
            "requested_revision",
            "revision",
            "config_sha256",
            "weights_sha256",
            "source_manifest_sha256",
        ):
            profile[key] = nonempty_string(profile[key], label + "." + key)
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", profile["repo"]):
            raise ConversionError("%s.repo is invalid" % label)
        if not re.fullmatch(r"[0-9a-f]{40}", profile["revision"]):
            raise ConversionError("%s.revision must be a commit" % label)
        for key in ("config_sha256", "weights_sha256", "source_manifest_sha256"):
            if not SHA256_RE.fullmatch(profile[key]):
                raise ConversionError("%s.%s must be lowercase SHA256" % (label, key))
        for key in ("weights_size", "source_tensor_count", "source_tensor_bytes"):
            profile[key] = uint64_value(profile[key], label + "." + key)
        profile["topology"] = validate_topology(profile["topology"], label + ".topology")
        profile["tokenizer"] = validate_tokenizer_profile(
            profile["tokenizer"], label + ".tokenizer"
        )
        identity = (profile["repo"], profile["revision"])
        if profile["runtime_id"] in profiles or identity in identities:
            raise ConversionError("GPT-2 profile duplicates runtime/source identity")
        profiles[profile["runtime_id"]] = profile
        identities.add(identity)
    return profiles, code, payload


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
        raise ConversionError("GENEB catalog must contain one GPT-2 profile row")
    entry = matches[0]
    expected = {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "family": "transformer-decoder",
        "architecture": profile["catalog_architecture"],
    }
    if any(entry.get(key) != value for key, value in expected.items()):
        raise ConversionError("GENEB catalog GPT-2 identity differs")
    source = object_value(entry.get("source"), "catalog source")
    if (
        source.get("kind") != "huggingface"
        or source.get("repo") != profile["repo"]
        or source.get("requested_revision") != profile["requested_revision"]
        or source.get("revision") != profile["revision"]
        or source.get("immutable") is not True
    ):
        raise ConversionError("GENEB catalog GPT-2 source identity differs")
    topology = profile["topology"]
    tokenizer = object_value(entry.get("tokenizer"), "catalog tokenizer")
    context = object_value(entry.get("context"), "catalog context")
    presets = object_value(entry.get("embedding_presets"), "catalog presets")
    if (
        tokenizer.get("kind") != "bpe"
        or tokenizer.get("max_tokens") != topology["max_seqlen"]
        or tokenizer.get("padding_side") != "right"
        or tokenizer.get("add_special_tokens") is not False
        or context.get("declared_max_tokens") != topology["max_seqlen"]
        or any(
            not isinstance(presets.get(name), dict)
            or presets[name].get("output_width") != topology["hidden_size"]
            or presets[name].get("pooling") != "attention-mask-mean"
            for name in ("reference", "normalized")
        )
    ):
        raise ConversionError("GENEB catalog GPT-2 tokenizer/context/preset differs")
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
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != profile["runtime_id"]
        or receipt["repo"] != profile["repo"]
        or receipt["requested_revision"] != profile["requested_revision"]
        or receipt["resolved_revision"] != profile["revision"]
        or ("load_path" in receipt and receipt["load_path"] is not None)
    ):
        raise ConversionError("source receipt GPT-2 identity/revision is not pinned")
    files = receipt["files"]
    if not isinstance(files, list) or not files:
        raise ConversionError("source receipt files must be nonempty")
    paths = {}  # type: Dict[str, Path]
    identities = {}  # type: Dict[str, Tuple[int, str]]
    for index, raw in enumerate(files):
        name, size, digest, source_path = validate_receipt_file_entry(
            raw, "source receipt files[%d]" % index, allow_nested=True
        )
        if name in paths:
            raise ConversionError("source receipt has duplicate file names")
        paths[name] = source_path
        identities[name] = (size, digest)
    expected = {"config.json", "model.safetensors"}
    if not expected.issubset(paths):
        raise ConversionError("GPT-2 source assets are missing files")
    if identities["model.safetensors"] != (
        profile["weights_size"],
        profile["weights_sha256"],
    ):
        raise ConversionError("GPT-2 weights size/SHA differ from pinned profile")
    if identities["config.json"][1] != profile["config_sha256"]:
        raise ConversionError("GPT-2 config SHA differs from pinned profile")
    return paths, payload


def validate_config(path: Path, profile: Mapping[str, Any]) -> None:
    config, _ = load_json(path, "GPT-2 config")
    topology = profile["topology"]
    expected = {
        "model_type": "gpt2",
        "vocab_size": topology["vocab_size"],
        "n_embd": topology["hidden_size"],
        "n_layer": topology["num_layers"],
        "n_head": topology["num_attention_heads"],
        "n_positions": topology["max_seqlen"],
        "n_ctx": topology["max_seqlen"],
        "n_inner": None,
        "activation_function": "gelu_new",
        "layer_norm_epsilon": topology["norm_epsilon"],
        "scale_attn_weights": True,
        "scale_attn_by_inverse_layer_idx": False,
        "reorder_and_upcast_attn": False,
        "torch_dtype": "float32",
        "transformers_version": TRANSFORMERS_VERSION,
    }
    wrong = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if config.get("architectures") != ["GPT2LMHeadModel"] or wrong:
        raise ConversionError("GPT-2 config semantics differ: %s" % wrong)
    for key in ("attn_pdrop", "embd_pdrop", "resid_pdrop"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != 0.1:
            raise ConversionError("GPT-2 source dropout metadata differs")


def canonical_tensor_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    vocab = topology["vocab_size"]
    width = topology["hidden_size"]
    layers = topology["num_layers"]
    inner = topology["inner_width"]
    result = [
        TensorSpec("transformer.wte.weight", "F32", (vocab, width)),
        TensorSpec(
            "transformer.wpe.weight", "F32", (topology["max_seqlen"], width)
        ),
    ]
    for layer in range(layers):
        prefix = "transformer.h.%d." % layer
        result.extend(
            [
                TensorSpec(prefix + "attn.c_attn.weight", "F32", (width, width * 3)),
                TensorSpec(prefix + "attn.c_attn.bias", "F32", (width * 3,)),
                TensorSpec(prefix + "attn.c_proj.weight", "F32", (width, width)),
                TensorSpec(prefix + "attn.c_proj.bias", "F32", (width,)),
                TensorSpec(prefix + "ln_1.weight", "F32", (width,)),
                TensorSpec(prefix + "ln_1.bias", "F32", (width,)),
                TensorSpec(prefix + "ln_2.weight", "F32", (width,)),
                TensorSpec(prefix + "ln_2.bias", "F32", (width,)),
                TensorSpec(prefix + "mlp.c_fc.weight", "F32", (width, inner)),
                TensorSpec(prefix + "mlp.c_fc.bias", "F32", (inner,)),
                TensorSpec(prefix + "mlp.c_proj.weight", "F32", (inner, width)),
                TensorSpec(prefix + "mlp.c_proj.bias", "F32", (width,)),
            ]
        )
    result.extend(
        [
            TensorSpec("transformer.ln_f.weight", "F32", (width,)),
            TensorSpec("transformer.ln_f.bias", "F32", (width,)),
        ]
    )
    return result


def validate_tokenizer_asset(asset: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    topology = profile["topology"]
    tokenizer_profile = profile["tokenizer"]
    by_id, _ = validate_common_tokenizer_asset(asset, topology["vocab_size"])
    if asset.get("kind") != "bpe" or asset.get("normalization") != []:
        raise ConversionError("GPT-2 runtime tokenizer kind/normalization differs")
    if asset.get("pre_tokenizer") != {"kind": "hf-whitespace-ascii"}:
        raise ConversionError("GPT-2 runtime tokenizer pretokenizer differs")
    model = object_value(asset.get("model"), "runtime tokenizer model")
    if set(model) != {"merges"} or not isinstance(model["merges"], list) or len(model["merges"]) != tokenizer_profile["merge_count"]:
        raise ConversionError("GPT-2 runtime tokenizer merge contract differs")
    post = object_value(asset.get("post_processor"), "runtime tokenizer post_processor")
    specials = object_value(asset.get("special_tokens"), "runtime tokenizer special_tokens")
    expected_padding = {"side": "right", "pad_id": tokenizer_profile["pad_id"]}
    if (
        set(post) != {"prefix_ids", "suffix_ids", "padding"}
        or post["prefix_ids"] != []
        or post["suffix_ids"] != []
        or post["padding"] != expected_padding
        or specials.get("unk") is not None
        or specials.get("pad") != tokenizer_profile["pad_id"]
        or specials.get("eos") != 0
        or by_id.get(0) != "<|endoftext|>"
        or by_id.get(1) != "<pad>"
        or by_id.get(2) != "<unk>"
    ):
        raise ConversionError("GPT-2 runtime tokenizer special/padding contract differs")


def build_metadata(
    profile: Mapping[str, Any],
    code: Mapping[str, str],
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
        "gpt2.num_attention_heads": topology["num_attention_heads"],
        "gpt2.inner_width": topology["inner_width"],
        "gpt2.norm_epsilon": topology["norm_epsilon"],
        "gpt2.activation": "gelu-new",
        "gpt2.qkv_layout": "q-k-v",
        "gpt2.weight_layout": "conv1d-in-out",
        "gpt2.weight_dtype": "F32",
        "gpt2.hidden_tap": "post-final-layernorm",
        "gpt2.pooling": "attention-mask-mean",
        "gpt2.linear_bias": True,
        "gpt2.layer_norm_bias": True,
        "gpt2.absolute_position_embeddings": True,
        "gpt2.causal_attention": True,
        "gpt2.attention_uses_mask": True,
        "gpt2.eval_dropout_disabled": True,
        "source.kind": "huggingface",
        "source.immutable": True,
        "source.repo": profile["repo"],
        "source.requested_revision": profile["requested_revision"],
        "source.revision": profile["revision"],
        "source.weights.sha256": profile["weights_sha256"],
        "source.weights.size": profile["weights_size"],
        "source.receipt_sha256": receipt_sha256,
        "source.catalog_contract_sha256": catalog_sha256,
        "source.converter_profile_contract_sha256": profile_sha256,
        "source.config_sha256": profile["config_sha256"],
        "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        "source.transformers.version": code["version"],
        "source.transformers.modeling_gpt2_sha256": code["modeling_gpt2_sha256"],
        "source.transformers.configuration_gpt2_sha256": code["configuration_gpt2_sha256"],
        "source.transformers.activations_sha256": code["activations_sha256"],
        "source.geneb.extractor_sha256": code["extractor_sha256"],
    }  # type: Dict[str, Any]
    metadata.update(geneb_metadata)
    metadata.update(tokenizer)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--catalog", type=Path, default=default_config_path(Path(__file__), "geneb-models.json")
    )
    parser.add_argument(
        "--profiles", type=Path, default=default_config_path(Path(__file__), "geneb-gpt2-models.json")
    )
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, code, _profile_payload = load_profiles(args.profiles.resolve())
        receipt_identity, _ = load_json(args.receipt.resolve(), "source receipt identity")
        model_id = receipt_identity.get("model_id")
        profile = profiles.get(model_id) if isinstance(model_id, str) else None
        if profile is None:
            raise ConversionError("source receipt does not identify a GPT2-Gene profile")
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
        validate_tokenizer_output_binding(
            args.tokenizer_descriptor,
            args.tokenizer_root,
            args.output,
        )
        tokenizer, tokenizer_descriptor_sha256, tokenizer_asset, tokenizer_payload = validate_tokenizer_descriptor(
            args.tokenizer_descriptor.resolve(),
            args.tokenizer_root.resolve() if args.tokenizer_root is not None else None,
            profile["tokenizer"]["compiler_manifest_sha256"],
        )
        if (
            sha256_bytes(tokenizer_payload) != profile["tokenizer"]["compiled_asset_sha256"]
            or len(tokenizer_payload) != profile["tokenizer"]["compiled_asset_size"]
        ):
            raise ConversionError("compiled GPT-2 tokenizer asset differs from pinned profile")
        validate_tokenizer_asset(tokenizer_asset, profile)
        sources = read_hf_safetensors({"model.safetensors": paths["model.safetensors"]})
        validate_profile_manifest_identity(sources, profile)
        runtime_sources = validate_tensor_manifest(
            sources, canonical_tensor_specs(profile["topology"])
        )
        canonical = [CanonicalF32TensorSource(source) for source in runtime_sources]
        try:
            geneb_metadata = build_geneb_artifact_metadata(
                catalog_root, catalog_entry, catalog_payload
            )
        except GenebArtifactError as error:
            raise ConversionError(str(error)) from error
        metadata = build_metadata(
            profile,
            code,
            tokenizer,
            sha256_bytes(receipt_payload),
            catalog_contract_sha256(catalog_root, catalog_entry),
            converter_profile_contract_sha256(
                1, PROFILE_FORMAT, profile, {"code_provenance": code}
            ),
            tokenizer_descriptor_sha256,
            geneb_metadata,
        )
        write_artifact(
            args.output, metadata, canonical, ARTIFACT_PROFILE, args.force
        )
        print("wrote %s" % args.output)
        print("source_receipt_sha256=%s" % sha256_bytes(receipt_payload))
        return 0
    except (
        CheckpointError,
        ConversionError,
        FileExistsError,
        OSError,
        ValueError,
    ) as error:
        print("convert_geneb_gpt2_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
