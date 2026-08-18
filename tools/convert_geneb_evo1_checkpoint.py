#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert pinned GENEB Evo-1 StripedHyena-v1 to its independent ABI."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if not (_SCRIPT_DIRECTORY / "evo").is_dir():
    _INSTALLED_PYTHON = _SCRIPT_DIRECTORY.parent / "share" / "evo" / "python"
    if _INSTALLED_PYTHON.is_dir():
        sys.path.insert(0, str(_INSTALLED_PYTHON))

from evo.hf_checkpoint import load_json, read_hf_safetensors  # noqa: E402
from evo.geneb_long_hyena_artifact import (  # noqa: E402
    CheckpointError,
    LongHyenaConversionError,
    TensorSpec,
    common_metadata,
    load_catalog_entry,
    load_profile_document,
    number,
    tensor_manifest_sha256,
    uint,
    validate_receipt,
    validate_catalog_runtime_contract,
    validate_tensor_manifest,
    validate_tokenizer_descriptor,
    write_artifact,
)


ARTIFACT_PROFILE = "geneb-evo1-runtime-v1"
RUNTIME_ABI = "geneb-evo1-safetensors-v1"
RUNTIME_ARCHITECTURE = "GenebStripedHyenaV1"
RUNTIME_ID = "geneb-evo-1-131k"
PROFILE_FORMAT = "geneb-evo1-converter-v1"
PINNED_REVISION = "c206aab77ae5967a069c4200ecb1858588528c9d"
PINNED_CACHE_SHA = "4aeaaf482d1da04ee9c31d47fe85bdce8e9c9e03a5cd3e183843e902c9114b73"
PINNED_MODEL_SHA = "6a03a9c28bd7e282cb253d051ae7ec39159d14d3a1b2272913c8ff1747f3d87a"
PINNED_ENGINE_SHA = "8fe4cd97ec6b6be43807fe2d33ec16b53d3ff831e5d33c25802e701041221639"
PINNED_LAYERS_SHA = "52c7909406acccd07d872f839cd22f3f494c8555ac96d51ce7eb2629cc8a98da"
PINNED_POSITION_SHA = "fc22bdb7447ae3c0adea17c00442c3ec040f0dbb61e54c4486d14bbc5217c3ae"
PINNED_TOKENIZER_SHA = "15e37ca8a1994a1bb4a9ac526d746808ef77a1b68e26ea9fe8c03f823d25c7be"
PINNED_EXTRACTOR_SHA = "edc260d061f332360a38669418517184e848fe88ffbccf1aea3376006068f256"
PINNED_EXTRACTOR_COMMIT = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
PROFILE_KEYS = {
    "runtime_id",
    "geneb_model_id",
    "paper_name",
    "catalog_architecture",
    "repo",
    "requested_revision",
    "revision",
    "checkpoint_manifest_sha256",
    "tokenizer",
    "source_files",
    "conversion_assets",
    "topology",
}
TOPOLOGY_KEYS = {
    "vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "inner_width",
    "state_width",
    "short_filter_width",
    "max_seqlen",
    "norm_epsilon",
    "rope_theta",
    "rope_scaling_factor",
    "attention_layers",
}


def default_config_path(name: str) -> Path:
    source = _SCRIPT_DIRECTORY.parent / "configs" / name
    if source.is_file():
        return source
    return _SCRIPT_DIRECTORY.parent / "share" / "evo" / "configs" / name


def validate_topology(raw: Dict[str, Any], label: str) -> Dict[str, Any]:
    for key in TOPOLOGY_KEYS - {
        "norm_epsilon",
        "rope_theta",
        "rope_scaling_factor",
        "attention_layers",
    }:
        raw[key] = uint(raw[key], label + "." + key)
    for key in ("norm_epsilon", "rope_theta", "rope_scaling_factor"):
        raw[key] = number(raw[key], label + "." + key)
    layers = raw["attention_layers"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers)
    ):
        raise LongHyenaConversionError("%s.attention_layers is invalid" % label)
    raw["attention_layers"] = [
        uint(layer, label + ".attention_layers", False) for layer in layers
    ]
    if (
        raw["attention_layers"] != sorted(set(raw["attention_layers"]))
        or raw["attention_layers"][-1] >= raw["num_layers"]
        or raw["hidden_size"] % raw["num_attention_heads"] != 0
        or (raw["hidden_size"] // raw["num_attention_heads"]) % 2 != 0
    ):
        raise LongHyenaConversionError("%s geometry/attention layout is invalid" % label)
    return raw


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    profiles, payload = load_profile_document(
        path,
        PROFILE_FORMAT,
        PROFILE_KEYS,
        TOPOLOGY_KEYS,
        validate_topology,
    )
    required = {
        "cache.py",
        "config.json",
        "configuration_hyena.py",
        "engine.py",
        "layers.py",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "model.py",
        "model.safetensors.index.json",
        "modeling_hyena.py",
        "positional_embeddings.py",
        "special_tokens_map.json",
        "tokenizer.py",
        "tokenizer_config.json",
        "utils.py",
    }
    for profile in profiles.values():
        files = profile["source_files"]
        if set(files) != required:
            raise LongHyenaConversionError("Evo-1 source file set differs")
        expected_hashes = {
            "cache.py": PINNED_CACHE_SHA,
            "model.py": PINNED_MODEL_SHA,
            "engine.py": PINNED_ENGINE_SHA,
            "layers.py": PINNED_LAYERS_SHA,
            "positional_embeddings.py": PINNED_POSITION_SHA,
            "tokenizer.py": PINNED_TOKENIZER_SHA,
        }
        wrong = {
            name: (files[name]["sha256"], digest)
            for name, digest in expected_hashes.items()
            if files[name]["sha256"] != digest
        }
        if profile["revision"] != PINNED_REVISION or wrong:
            raise LongHyenaConversionError("Evo-1 source code/revision is not pinned: %s" % wrong)
    return profiles, payload


def validate_catalog_source_files(
    entry: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    source = entry.get("source")
    raw_files = source.get("required_files") if isinstance(source, dict) else None
    if not isinstance(raw_files, list):
        raise LongHyenaConversionError("catalog Evo-1 source file set differs")
    files = {}  # type: Dict[str, Dict[str, Any]]
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
            raise LongHyenaConversionError(
                "catalog Evo-1 source descriptor %d differs" % index
            )
        name = raw["path"]
        size = raw["size"]
        digest = raw["sha256"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in files
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise LongHyenaConversionError(
                "catalog Evo-1 source descriptor %d differs" % index
            )
        files[name] = {"size": size, "sha256": digest}
    if files != profile["source_files"]:
        raise LongHyenaConversionError("catalog Evo-1 source file set differs")


def validate_production_receipt_file_set(
    receipt: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    if profile.get("runtime_id") != RUNTIME_ID:
        return
    raw_files = receipt.get("files")
    if not isinstance(raw_files, list):
        raise LongHyenaConversionError("Evo-1 source receipt file set differs")
    names = []  # type: List[str]
    for raw in raw_files:
        name = raw.get("name") if isinstance(raw, dict) else None
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise LongHyenaConversionError("Evo-1 source receipt file set differs")
        names.append(name)
    if set(names) != set(profile["source_files"]):
        raise LongHyenaConversionError("Evo-1 source receipt file set differs")


def validate_config(path: Path, profile: Mapping[str, Any]) -> None:
    config, payload = load_json(path, "Evo-1 config")
    if hashlib.sha256(payload).hexdigest() != profile["source_files"]["config.json"]["sha256"]:
        raise LongHyenaConversionError("Evo-1 config SHA256 differs")
    topology = profile["topology"]
    expected = {
        "vocab_size": topology["vocab_size"],
        "hidden_size": topology["hidden_size"],
        "num_layers": topology["num_layers"],
        "num_attention_heads": topology["num_attention_heads"],
        "inner_mlp_size": topology["inner_width"],
        "state_size": topology["state_width"],
        "short_filter_length": topology["short_filter_width"],
        "max_seqlen": topology["max_seqlen"],
        "eps": topology["norm_epsilon"],
        "rotary_emb_base": topology["rope_theta"],
        "rotary_emb_scaling_factor": topology["rope_scaling_factor"],
        "attn_layer_idxs": topology["attention_layers"],
        "column_split_hyena": True,
        "hyena_filter_groups": 1,
        "mlp_activation": "gelu",
        "prefill_style": "fft",
        "torch_dtype": "bfloat16",
        "use_flashfft": False,
        "use_interpolated_rotary_pos_emb": True,
        "qkv_proj_bias": True,
        "mha_out_proj_bias": True,
        "short_filter_bias": True,
        "final_norm": True,
    }
    wrong = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if wrong:
        raise LongHyenaConversionError("Evo-1 config semantics differ: %s" % wrong)
    hyena_layers = config.get("hyena_layer_idxs")
    complement = [
        index for index in range(topology["num_layers"])
        if index not in topology["attention_layers"]
    ]
    if hyena_layers != complement:
        raise LongHyenaConversionError("Evo-1 Hyena layer complement differs")


def validate_tokenizer_config(path: Path, profile: Mapping[str, Any]) -> None:
    config, payload = load_json(path, "Evo-1 tokenizer config")
    if hashlib.sha256(payload).hexdigest() != profile["source_files"]["tokenizer_config.json"]["sha256"]:
        raise LongHyenaConversionError("Evo-1 tokenizer config SHA256 differs")
    auto_map = config.get("auto_map")
    if config.get("tokenizer_class") not in (None, "ByteTokenizer") and auto_map is None:
        raise LongHyenaConversionError("Evo-1 tokenizer class is not byte tokenizer")


def canonical_tensor_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    width = topology["hidden_size"]
    inner = topology["inner_width"]
    state = topology["state_width"]
    short = topology["short_filter_width"]
    attention = set(topology["attention_layers"])
    specs = []  # type: List[TensorSpec]
    for layer in range(topology["num_layers"]):
        prefix = "backbone.blocks.%d." % layer
        if layer in attention:
            specs.extend(
                [
                    TensorSpec(prefix + "inner_mha_cls.Wqkv.bias", "BF16", (width * 3,)),
                    TensorSpec(prefix + "inner_mha_cls.Wqkv.weight", "BF16", (width * 3, width)),
                    TensorSpec(prefix + "inner_mha_cls.out_proj.bias", "BF16", (width,)),
                    TensorSpec(prefix + "inner_mha_cls.out_proj.weight", "BF16", (width, width)),
                    TensorSpec(prefix + "inner_mha_cls.rotary_emb.inv_freq", "BF16", (width // topology["num_attention_heads"] // 2,)),
                ]
            )
        else:
            specs.extend(
                [
                    TensorSpec(prefix + "filter.D", "BF16", (width,)),
                    TensorSpec(prefix + "filter.poles", "F32", (width, state, 1, 2)),
                    TensorSpec(prefix + "filter.residues", "F32", (width, state, 1, 2)),
                    TensorSpec(prefix + "filter.short_filter_bias", "BF16", (width * 3,)),
                    TensorSpec(prefix + "filter.short_filter_weight", "BF16", (width * 3, 1, short)),
                    TensorSpec(prefix + "out_filter_dense.bias", "BF16", (width,)),
                    TensorSpec(prefix + "out_filter_dense.weight", "BF16", (width, width)),
                    TensorSpec(prefix + "projections.bias", "BF16", (width * 3,)),
                    TensorSpec(prefix + "projections.weight", "BF16", (width * 3, width)),
                ]
            )
        specs.extend(
            [
                TensorSpec(prefix + "mlp.l1.weight", "BF16", (inner, width)),
                TensorSpec(prefix + "mlp.l2.weight", "BF16", (inner, width)),
                TensorSpec(prefix + "mlp.l3.weight", "BF16", (width, inner)),
                TensorSpec(prefix + "post_norm.scale", "BF16", (width,)),
                TensorSpec(prefix + "pre_norm.scale", "BF16", (width,)),
            ]
        )
    specs.extend(
        [
            TensorSpec("backbone.embedding_layer.weight", "BF16", (topology["vocab_size"], width)),
            TensorSpec("backbone.norm.scale", "BF16", (width,)),
        ]
    )
    return specs


def build_metadata(profile: Mapping[str, Any], base: Mapping[str, Any]) -> Dict[str, Any]:
    topology = profile["topology"]
    metadata = dict(base)
    metadata.update(
        {
            "runtime.abi": RUNTIME_ABI,
            "runtime.embedding_layer_count": topology["num_layers"] + 1,
            "runtime.tokenizer_vocabulary_size": topology["vocab_size"],
            "model.architecture": RUNTIME_ARCHITECTURE,
            "config.vocab_size": topology["vocab_size"],
            "config.hidden_size": topology["hidden_size"],
            "config.num_layers": topology["num_layers"],
            "config.max_seqlen": topology["max_seqlen"],
            "evo1.num_attention_heads": topology["num_attention_heads"],
            "evo1.inner_width": topology["inner_width"],
            "evo1.state_width": topology["state_width"],
            "evo1.short_filter_width": topology["short_filter_width"],
            "evo1.norm_epsilon": topology["norm_epsilon"],
            "evo1.rope_theta": topology["rope_theta"],
            "evo1.rope_scaling_factor": topology["rope_scaling_factor"],
            "evo1.attention_layers": topology["attention_layers"],
            "evo1.weight_dtype": "mixed-bf16-f32",
            "evo1.norm_denominator": "sqrt-mean-plus-epsilon",
            "evo1.mlp_activation": "gelu-exact",
            "evo1.hyena_projection_layout": "per-head-x2-x1-v",
            "evo1.long_convolution": "exact-2l-modal-fft",
            "evo1.attention_memory": "streaming-linear",
            "evo1.hidden_tap": "post-final-rmsnorm",
            "evo1.pooling": "all-token-mean",
            "evo1.causal_attention": True,
            "evo1.add_special_tokens": False,
            "source.model_py_sha256": PINNED_MODEL_SHA,
            "source.model_py_path": "model.py",
            "source.model_py_lines": "29-84,87-318,333-383",
            "source.engine_py_sha256": PINNED_ENGINE_SHA,
            "source.engine_py_path": "engine.py",
            "source.engine_py_lines": "66-226",
            "source.layers_py_sha256": PINNED_LAYERS_SHA,
            "source.layers_py_path": "layers.py",
            "source.layers_py_lines": "17-40,43-86",
            "source.positional_embeddings_py_sha256": PINNED_POSITION_SHA,
            "source.positional_embeddings_py_path": "positional_embeddings.py",
            "source.positional_embeddings_py_lines": "11-112",
            "source.tokenizer_py_sha256": PINNED_TOKENIZER_SHA,
            "source.tokenizer_py_path": "tokenizer.py",
            "source.tokenizer_config_path": "tokenizer_config.json",
            "source.geneb.extractor_sha256": PINNED_EXTRACTOR_SHA,
            "source.geneb.extractor_path": "embedding_pipeline/extractors/evo.py",
            "source.geneb.extractor_lines": "29-35",
            "geneb.provenance.extractor_commit": PINNED_EXTRACTOR_COMMIT,
        }
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=default_config_path("geneb-models.json"))
    parser.add_argument("--profiles", type=Path, default=default_config_path("geneb-evo1-models.json"))
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, _profile_payload = load_profiles(args.profiles.resolve())
        identity, _ = load_json(args.receipt.resolve(), "source receipt identity")
        model_id = identity.get("model_id")
        profile = profiles.get(model_id) if isinstance(model_id, str) else None
        if profile is None:
            raise LongHyenaConversionError("receipt does not identify an Evo-1 profile")
        validate_production_receipt_file_set(identity, profile)
        entry, catalog_root, catalog_payload = load_catalog_entry(
            args.catalog.resolve(), profile, "left"
        )
        validate_catalog_source_files(entry, profile)
        validate_catalog_runtime_contract(
            entry, profile["topology"], "evo", PINNED_EXTRACTOR_COMMIT
        )
        paths, receipt_payload = validate_receipt(
            args.receipt.resolve(),
            profile,
            args.catalog.resolve(),
            catalog_payload,
            catalog_root,
            entry,
        )
        validate_config(paths["config.json"], profile)
        validate_tokenizer_config(paths["tokenizer_config.json"], profile)
        tokenizer, descriptor_sha = validate_tokenizer_descriptor(
            args.tokenizer_descriptor.resolve(),
            args.tokenizer_root.resolve() if args.tokenizer_root is not None else None,
            args.output,
            profile["tokenizer"]["compiler_manifest_sha256"],
        )
        specs = canonical_tensor_specs(profile["topology"])
        if tensor_manifest_sha256(specs) != profile["checkpoint_manifest_sha256"]:
            raise LongHyenaConversionError("canonical tensor manifest hash differs")
        weight_paths = {
            name: path
            for name, path in paths.items()
            if name == "model.safetensors.index.json" or name.endswith(".safetensors")
        }
        sources = read_hf_safetensors(weight_paths)
        tensors = validate_tensor_manifest(sources, specs)
        base = common_metadata(
            catalog_root,
            entry,
            catalog_payload,
            profile,
            PROFILE_FORMAT,
            receipt_payload,
            tokenizer,
            descriptor_sha,
        )
        write_artifact(args.output, ARTIFACT_PROFILE, build_metadata(profile, base), tensors, args.force)
        print("wrote %s" % args.output)
        return 0
    except (
        CheckpointError,
        LongHyenaConversionError,
        FileExistsError,
        OSError,
        ValueError,
    ) as error:
        print("convert_geneb_evo1_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
