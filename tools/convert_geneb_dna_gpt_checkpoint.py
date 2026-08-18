#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile manually acquired DNA-GPT checkpoints into canonical F32 artifacts."""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple


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
)
from evo.geneb_gpt_artifact import (  # noqa: E402
    CanonicalF32TensorSource,
    ConversionError,
    DNA_GPT_BASE_ALPHABET,
    DNA_GPT_RESERVED_TOKENS,
    SHA256_RE,
    default_config_path,
    dna_gpt_vocab_pieces,
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
    load_torch_checkpoint,
    validate_tensor_manifest,
)


ARTIFACT_PROFILE = "geneb-dna-gpt-runtime-v1"
RUNTIME_ABI = "geneb-dna-gpt-torch-pth-v1"
RUNTIME_ARCHITECTURE = "GenebDnaGptDecoder"
PROFILE_FORMAT = "geneb-dna-gpt-converter-v1"
TOKENIZER_PROFILE = "evo-tokenizer-v1"
BASE_ALPHABET = DNA_GPT_BASE_ALPHABET
RESERVED_TOKENS = DNA_GPT_RESERVED_TOKENS

PROFILE_KEYS = {
    "runtime_id",
    "geneb_model_id",
    "paper_name",
    "catalog_architecture",
    "source_filename",
    "google_drive_file_id",
    "source_dtype",
    "source_manifest_sha256",
    "source_tensor_count",
    "source_tensor_bytes",
    "topology",
    "tokenizer",
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
    "kind",
    "alphabet",
    "reserved_count",
    "unk_id",
    "pad_id",
    "prefix_id",
    "prefix_literal",
    "compiled_asset_sha256",
    "compiled_asset_size",
    "compiler_manifest_sha256",
}
CODE_KEYS = {
    "official_repo",
    "official_revision",
    "extractor_commit",
    "extractor_sha256",
    "gpt_py_sha256",
    "dna_gpt_py_sha256",
    "tokenizer_py_sha256",
}


def tokenizer_vocabulary_size(kind: str) -> int:
    return len(dna_gpt_vocab_pieces(kind))


def validate_topology(raw: Any, label: str) -> Dict[str, Any]:
    value = object_value(raw, label)
    if set(value) != TOPOLOGY_KEYS:
        raise ConversionError("%s fields differ" % label)
    result = dict(value)
    for key in (
        "vocab_size",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "inner_width",
        "max_seqlen",
    ):
        result[key] = uint64_value(result[key], label + "." + key)
    result["norm_epsilon"] = finite_float(
        result["norm_epsilon"], label + ".norm_epsilon"
    )
    if (
        result["hidden_size"] % result["num_attention_heads"] != 0
        or result["inner_width"] != result["hidden_size"] * 4
    ):
        raise ConversionError("%s hidden/head/MLP contract differs" % label)
    return result


def validate_tokenizer_profile(raw: Any, topology: Mapping[str, Any], label: str) -> Dict[str, Any]:
    value = object_value(raw, label)
    if set(value) != TOKENIZER_KEYS:
        raise ConversionError("%s fields differ" % label)
    result = dict(value)
    for key in ("kind", "alphabet", "prefix_literal"):
        result[key] = nonempty_string(result[key], label + "." + key)
    for key in ("compiled_asset_sha256", "compiler_manifest_sha256"):
        result[key] = nonempty_string(result[key], label + "." + key)
        if not SHA256_RE.fullmatch(result[key]):
            raise ConversionError("%s.%s must be lowercase SHA256" % (label, key))
    for key in ("reserved_count", "unk_id", "pad_id", "prefix_id"):
        result[key] = uint64_value(result[key], label + "." + key, False)
    result["compiled_asset_size"] = uint64_value(
        result["compiled_asset_size"], label + ".compiled_asset_size"
    )
    if (
        result["kind"] not in ("static-sixmer", "dynamic-sixmer")
        or result["alphabet"] != BASE_ALPHABET
        or result["reserved_count"] != len(RESERVED_TOKENS)
        or result["unk_id"] != 0
        or result["pad_id"] != 20
        or result["prefix_id"] != 21
        or result["prefix_literal"] != "<R>"
        or topology["vocab_size"] != tokenizer_vocabulary_size(result["kind"])
    ):
        raise ConversionError("%s differs from pinned KmerTokenizer" % label)
    return result


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], bytes]:
    root, payload = load_json(path, "GENEB DNA-GPT converter profiles")
    exact_keys(root, ["schema_version", "format", "models", "code_provenance"], [], "profiles")
    if root["schema_version"] != 1 or root["format"] != PROFILE_FORMAT:
        raise ConversionError("DNA-GPT profile schema/format is unsupported")
    raw_code = object_value(root["code_provenance"], "code_provenance")
    if set(raw_code) != CODE_KEYS:
        raise ConversionError("DNA-GPT code provenance fields differ")
    code = {}  # type: Dict[str, str]
    for key in CODE_KEYS:
        code[key] = nonempty_string(raw_code[key], "code_provenance." + key)
    for key in ("official_revision", "extractor_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", code[key]):
            raise ConversionError("DNA-GPT code revision is invalid")
    for key in CODE_KEYS - {"official_repo", "official_revision", "extractor_commit"}:
        if not SHA256_RE.fullmatch(code[key]):
            raise ConversionError("DNA-GPT code digest is invalid")
    models = root["models"]
    if not isinstance(models, list) or not models:
        raise ConversionError("DNA-GPT profiles must be a nonempty array")
    profiles = {}  # type: Dict[str, Dict[str, Any]]
    files = set()  # type: Set[str]
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
            "source_filename",
            "google_drive_file_id",
            "source_dtype",
            "source_manifest_sha256",
        ):
            profile[key] = nonempty_string(profile[key], label + "." + key)
        if (
            Path(profile["source_filename"]).name != profile["source_filename"]
            or not profile["source_filename"].endswith(".pth")
            or profile["source_dtype"] not in ("F32", "BF16")
            or not SHA256_RE.fullmatch(profile["source_manifest_sha256"])
        ):
            raise ConversionError("%s source identity/dtype is invalid" % label)
        for key in ("source_tensor_count", "source_tensor_bytes"):
            profile[key] = uint64_value(profile[key], label + "." + key)
        profile["topology"] = validate_topology(profile["topology"], label + ".topology")
        profile["tokenizer"] = validate_tokenizer_profile(
            profile["tokenizer"], profile["topology"], label + ".tokenizer"
        )
        if profile["runtime_id"] in profiles or profile["source_filename"] in files:
            raise ConversionError("DNA-GPT profile duplicates runtime/source filename")
        profiles[profile["runtime_id"]] = profile
        files.add(profile["source_filename"])
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
        raise ConversionError("GENEB catalog must contain one DNA-GPT profile row")
    entry = matches[0]
    expected = {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "family": "transformer-decoder",
        "architecture": profile["catalog_architecture"],
    }
    if any(entry.get(key) != value for key, value in expected.items()):
        raise ConversionError("GENEB catalog DNA-GPT identity differs")
    source = object_value(entry.get("source"), "catalog source")
    required_files = source.get("required_files")
    required_file = (
        required_files[0]
        if isinstance(required_files, list) and len(required_files) == 1
        else None
    )
    if (
        source.get("kind") != "google-drive"
        or source.get("repo") is not None
        or source.get("requested_revision") is not None
        or source.get("revision") is not None
        or source.get("immutable") is not False
        or not isinstance(required_file, dict)
        or set(required_file) != {"path", "size", "sha256"}
        or required_file.get("path") != profile["source_filename"]
        or not isinstance(required_file.get("size"), int)
        or isinstance(required_file.get("size"), bool)
        or required_file["size"] <= 0
        or not isinstance(required_file.get("sha256"), str)
        or not SHA256_RE.fullmatch(required_file["sha256"])
    ):
        raise ConversionError("GENEB manual DNA-GPT source contract differs")
    topology = profile["topology"]
    tokenizer = object_value(entry.get("tokenizer"), "catalog tokenizer")
    context = object_value(entry.get("context"), "catalog context")
    presets = object_value(entry.get("embedding_presets"), "catalog presets")
    dynamic = profile["tokenizer"]["kind"] == "dynamic-sixmer"
    if (
        tokenizer.get("kind") != "k-mer"
        or tokenizer.get("k") != 6
        or tokenizer.get("dynamic") is not dynamic
        or tokenizer.get("add_special_tokens") is not False
        or tokenizer.get("max_tokens") != topology["max_seqlen"]
        or tokenizer.get("padding_side") != "right"
        or context.get("declared_max_tokens") != topology["max_seqlen"]
        or any(
            not isinstance(presets.get(name), dict)
            or presets[name].get("output_width") != topology["hidden_size"]
            or presets[name].get("hidden_tap") != "post-final-layernorm"
            or presets[name].get("pooling") != "attention-mask-mean"
            or presets[name].get("special_tokens") != "include-prefix"
            or presets[name].get("mask_domain") != "non-pad-token-rows"
            for name in ("reference", "normalized")
        )
    ):
        raise ConversionError("GENEB catalog DNA-GPT tokenizer/context/preset differs")
    return entry, root, payload


def validate_receipt(
    path: Path,
    profile: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
) -> Tuple[Path, int, str, bytes]:
    receipt, payload = load_json(path, "manual source receipt")
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
        "manual source receipt",
    )
    source = object_value(catalog_entry.get("source"), "catalog source")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != profile["runtime_id"]
        or receipt["source_kind"] != "google-drive"
        or receipt["source_url"] != source.get("url")
    ):
        raise ConversionError("manual DNA-GPT receipt provider/model differs")
    files = receipt["files"]
    if not isinstance(files, list) or len(files) != 1:
        raise ConversionError("manual DNA-GPT receipt must contain exactly one file")
    name, size, digest, source_path = validate_receipt_file_entry(
        files[0], "manual source receipt files[0]"
    )
    if name != profile["source_filename"]:
        raise ConversionError("manual DNA-GPT receipt filename differs from profile")
    required = source["required_files"][0]
    if size != required["size"] or digest != required["sha256"]:
        raise ConversionError("manual DNA-GPT receipt differs from catalog file pin")
    return source_path, size, digest, payload


def backbone_tensor_specs(
    topology: Mapping[str, Any], dtype: str
) -> List[TensorSpec]:
    vocab = topology["vocab_size"]
    width = topology["hidden_size"]
    inner = topology["inner_width"]
    result = [
        TensorSpec("transformer.wte.weight", dtype, (vocab, width)),
        TensorSpec(
            "transformer.wpe.weight", dtype, (topology["max_seqlen"], width)
        ),
    ]
    for layer in range(topology["num_layers"]):
        prefix = "transformer.h.%d." % layer
        result.extend(
            [
                TensorSpec(prefix + "attn.c_attn.weight", dtype, (width * 3, width)),
                TensorSpec(prefix + "attn.c_proj.weight", dtype, (width, width)),
                TensorSpec(prefix + "mlp.c_fc.weight", dtype, (inner, width)),
                TensorSpec(prefix + "mlp.c_proj.weight", dtype, (width, inner)),
                TensorSpec(prefix + "ln_1.weight", dtype, (width,)),
                TensorSpec(prefix + "ln_2.weight", dtype, (width,)),
            ]
        )
    result.append(TensorSpec("transformer.ln_f.weight", dtype, (width,)))
    return result


def source_tensor_specs(profile: Mapping[str, Any]) -> List[TensorSpec]:
    topology = profile["topology"]
    dtype = profile["source_dtype"]
    width = topology["hidden_size"]
    vocab = topology["vocab_size"]
    result = backbone_tensor_specs(topology, dtype)
    result.extend(
        [
            TensorSpec("number_embedding.0.weight", dtype, (width, 1)),
            TensorSpec("number_embedding.2.weight", dtype, (width,)),
            TensorSpec("number_embedding.3.weight", dtype, (width, width)),
            TensorSpec("num_regression.0.weight", dtype, (width, width)),
            TensorSpec("num_regression.2.weight", dtype, (width,)),
            TensorSpec("num_regression.3.weight", dtype, (1, width)),
            TensorSpec("mlm_head.0.weight", dtype, (width, width)),
            TensorSpec("mlm_head.2.weight", dtype, (width,)),
            TensorSpec("mlm_head.3.weight", dtype, (vocab, width)),
        ]
    )
    return result


def expected_vocab_pieces(kind: str) -> List[str]:
    return dna_gpt_vocab_pieces(kind)


def validate_tokenizer_asset(asset: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    topology = profile["topology"]
    tokenizer = profile["tokenizer"]
    by_id, _ = validate_common_tokenizer_asset(asset, topology["vocab_size"])
    expected = expected_vocab_pieces(tokenizer["kind"])
    if any(by_id[index] != piece for index, piece in enumerate(expected)):
        raise ConversionError("DNA-GPT runtime tokenizer vocabulary order differs")
    if asset.get("normalization") != []:
        raise ConversionError("DNA-GPT runtime tokenizer normalization must be empty")
    post = object_value(asset.get("post_processor"), "runtime tokenizer post_processor")
    specials = object_value(asset.get("special_tokens"), "runtime tokenizer special_tokens")
    if (
        post
        != {
            "prefix_ids": [tokenizer["prefix_id"]],
            "suffix_ids": [],
            "padding": {"side": "right", "pad_id": tokenizer["pad_id"]},
        }
        or specials.get("unk") != tokenizer["unk_id"]
        or specials.get("pad") != tokenizer["pad_id"]
        or specials.get("bos") != tokenizer["prefix_id"]
        or any(specials.get(name) is not None for name in ("eos", "cls", "sep", "mask"))
    ):
        raise ConversionError("DNA-GPT runtime tokenizer special/padding contract differs")
    model = object_value(asset.get("model"), "runtime tokenizer model")
    if tokenizer["kind"] == "static-sixmer":
        expected_model = {
            "k": 6,
            "stride": 6,
            "tail": "lookup",
            "unknown_policy": "unk",
            "match_special_literals": True,
        }
        if (
            asset.get("kind") != "kmer"
            or asset.get("pre_tokenizer") != {"kind": "none"}
            or model != expected_model
        ):
            raise ConversionError("DNA-GPT static six-mer runtime contract differs")
    else:
        expected_model = {
            "k": 6,
            "stride": 6,
            "tail": "lookup",
            "unknown_policy": "unk",
            "match_special_literals": True,
        }
        if (
            asset.get("kind") != "kmer"
            or asset.get("pre_tokenizer") != {"kind": "none"}
            or model != expected_model
        ):
            raise ConversionError("DNA-GPT dynamic six-mer runtime contract differs")


def build_metadata(
    profile: Mapping[str, Any],
    code: Mapping[str, str],
    tokenizer_metadata: Mapping[str, Any],
    source_size: int,
    source_sha256: str,
    receipt_sha256: str,
    catalog_sha256: str,
    profile_sha256: str,
    tokenizer_descriptor_sha256: str,
    geneb_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    topology = profile["topology"]
    tokenizer = profile["tokenizer"]
    metadata = {
        "runtime.abi": RUNTIME_ABI,
        "runtime.embedding_layer_count": topology["num_layers"] + 1,
        "model.id": profile["runtime_id"],
        "model.architecture": RUNTIME_ARCHITECTURE,
        "config.vocab_size": topology["vocab_size"],
        "config.hidden_size": topology["hidden_size"],
        "config.num_layers": topology["num_layers"],
        "config.max_seqlen": topology["max_seqlen"],
        "dna_gpt.num_attention_heads": topology["num_attention_heads"],
        "dna_gpt.inner_width": topology["inner_width"],
        "dna_gpt.norm_epsilon": topology["norm_epsilon"],
        "dna_gpt.activation": "gelu-tanh",
        "dna_gpt.qkv_layout": "q-k-v",
        "dna_gpt.weight_layout": "linear-out-in",
        "dna_gpt.weight_dtype": "F32",
        "dna_gpt.source_weight_dtype": profile["source_dtype"],
        "dna_gpt.layer_norm": "affine-weight-only",
        "dna_gpt.hidden_tap": "post-final-layernorm",
        "dna_gpt.pooling": "non-pad-token-mean",
        "dna_gpt.linear_bias": False,
        "dna_gpt.layer_norm_bias": False,
        "dna_gpt.absolute_position_embeddings": True,
        "dna_gpt.causal_attention": True,
        "dna_gpt.attention_uses_mask": False,
        "dna_gpt.eval_dropout_disabled": True,
        "dna_gpt.tokenizer_kind": tokenizer["kind"],
        "dna_gpt.tokenizer_prefix_id": tokenizer["prefix_id"],
        "dna_gpt.tokenizer_pad_id": tokenizer["pad_id"],
        "source.kind": "google-drive",
        "source.immutable": False,
        "source.requested_revision": "",
        "source.revision": "",
        "source.filename": profile["source_filename"],
        "source.google_drive_file_id": profile["google_drive_file_id"],
        "source.weights.sha256": source_sha256,
        "source.weights.size": source_size,
        "source.tensor_manifest_sha256": profile["source_manifest_sha256"],
        "source.receipt_sha256": receipt_sha256,
        "source.catalog_contract_sha256": catalog_sha256,
        "source.converter_profile_contract_sha256": profile_sha256,
        "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        "source.dnagpt.repo": code["official_repo"],
        "source.dnagpt.revision": code["official_revision"],
        "source.dnagpt.gpt_py_sha256": code["gpt_py_sha256"],
        "source.dnagpt.dna_gpt_py_sha256": code["dna_gpt_py_sha256"],
        "source.dnagpt.tokenizer_py_sha256": code["tokenizer_py_sha256"],
        "source.geneb.extractor_sha256": code["extractor_sha256"],
    }  # type: Dict[str, Any]
    metadata.update(geneb_metadata)
    metadata.update(tokenizer_metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--catalog", type=Path, default=default_config_path(Path(__file__), "geneb-models.json")
    )
    parser.add_argument(
        "--profiles", type=Path, default=default_config_path(Path(__file__), "geneb-dna-gpt-models.json")
    )
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, code, _profile_payload = load_profiles(args.profiles.resolve())
        receipt_identity, _ = load_json(args.receipt.resolve(), "manual source receipt identity")
        model_id = receipt_identity.get("model_id")
        profile = profiles.get(model_id) if isinstance(model_id, str) else None
        if profile is None:
            raise ConversionError("manual receipt does not identify a DNA-GPT profile")
        catalog_entry, catalog_root, catalog_payload = load_catalog_entry(
            args.catalog.resolve(), profile
        )
        source_path, source_size, source_sha256, receipt_payload = validate_receipt(
            args.receipt.resolve(), profile, catalog_entry
        )
        validate_tokenizer_output_binding(
            args.tokenizer_descriptor,
            args.tokenizer_root,
            args.output,
        )
        tokenizer_metadata, tokenizer_descriptor_sha256, tokenizer_asset, tokenizer_payload = validate_tokenizer_descriptor(
            args.tokenizer_descriptor.resolve(),
            args.tokenizer_root.resolve() if args.tokenizer_root is not None else None,
            profile["tokenizer"]["compiler_manifest_sha256"],
        )
        if (
            sha256_bytes(tokenizer_payload)
            != profile["tokenizer"]["compiled_asset_sha256"]
            or len(tokenizer_payload)
            != profile["tokenizer"]["compiled_asset_size"]
        ):
            raise ConversionError(
                "compiled DNA-GPT tokenizer asset differs from pinned profile"
            )
        validate_tokenizer_asset(tokenizer_asset, profile)
        # This call occurs only after the complete operator receipt size/SHA gate.
        sources = load_torch_checkpoint(source_path)
        validate_profile_manifest_identity(sources, profile)
        all_sources = validate_tensor_manifest(sources, source_tensor_specs(profile))
        by_name = {source.name: source for source in all_sources}
        runtime_specs = backbone_tensor_specs(profile["topology"], profile["source_dtype"])
        runtime_sources = [by_name[spec.name] for spec in runtime_specs]
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
            tokenizer_metadata,
            source_size,
            source_sha256,
            sha256_bytes(receipt_payload),
            catalog_contract_sha256(catalog_root, catalog_entry),
            converter_profile_contract_sha256(
                1, PROFILE_FORMAT, profile, {"code_provenance": code}
            ),
            tokenizer_descriptor_sha256,
            geneb_metadata,
        )
        write_artifact(args.output, metadata, canonical, ARTIFACT_PROFILE, args.force)
        print("wrote %s" % args.output)
        print("source_receipt_sha256=%s" % sha256_bytes(receipt_payload))
        print("source_revision=null")
        return 0
    except (
        CheckpointError,
        ConversionError,
        FileExistsError,
        OSError,
        ValueError,
    ) as error:
        print("convert_geneb_dna_gpt_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
