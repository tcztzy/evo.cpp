#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert pinned GENEB HyenaDNA checkpoints to the strict portable ABI."""

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


ARTIFACT_PROFILE = "geneb-hyenadna-runtime-v1"
RUNTIME_ABI = "geneb-hyenadna-safetensors-v1"
RUNTIME_ARCHITECTURE = "GenebHyenaDnaDecoder"
PROFILE_FORMAT = "geneb-hyenadna-converter-v1"
PINNED_MODELING_SHA = "d78029cacea2c9259f1293fdb50b60f90872089469185e27d5c769086c475607"
PINNED_TOKENIZATION_SHA = "d424da2f794958b0360e033bbf5edc9ee4e3e3126e883093621348f07761e408"
PINNED_EXTRACTOR_SHA = "b449660cc7c2f0efb06e771ca0206f890b7cd7a523bbf24b877e7494449f8941"
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
    "embedding_rows",
    "hidden_size",
    "num_layers",
    "inner_width",
    "filter_width",
    "positional_width",
    "short_filter_width",
    "max_seqlen",
    "norm_epsilon",
}


def default_config_path(name: str) -> Path:
    source = _SCRIPT_DIRECTORY.parent / "configs" / name
    if source.is_file():
        return source
    return _SCRIPT_DIRECTORY.parent / "share" / "evo" / "configs" / name


def validate_topology(raw: Dict[str, Any], label: str) -> Dict[str, Any]:
    for key in TOPOLOGY_KEYS - {"norm_epsilon"}:
        raw[key] = uint(raw[key], label + "." + key)
    raw["norm_epsilon"] = number(raw["norm_epsilon"], label + ".norm_epsilon")
    if (
        raw["embedding_rows"] < raw["vocab_size"]
        or raw["positional_width"] < 3
        or raw["positional_width"] % 2 == 0
        or raw["short_filter_width"] > 4096
    ):
        raise LongHyenaConversionError("%s geometry is invalid" % label)
    return raw


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    profiles, payload = load_profile_document(
        path,
        PROFILE_FORMAT,
        PROFILE_KEYS,
        TOPOLOGY_KEYS,
        validate_topology,
    )
    for profile in profiles.values():
        files = profile["source_files"]
        required = {
            "config.json",
            "configuration_hyena.py",
            "model.safetensors",
            "modeling_hyena.py",
            "special_tokens_map.json",
            "tokenization_hyena.py",
            "tokenizer_config.json",
        }
        if set(files) != required:
            raise LongHyenaConversionError("HyenaDNA source file set differs")
        if files["modeling_hyena.py"]["sha256"] != PINNED_MODELING_SHA:
            raise LongHyenaConversionError("HyenaDNA modeling source is not pinned")
        if files["tokenization_hyena.py"]["sha256"] != PINNED_TOKENIZATION_SHA:
            raise LongHyenaConversionError("HyenaDNA tokenizer source is not pinned")
    return profiles, payload


def validate_config(path: Path, profile: Mapping[str, Any]) -> None:
    config, payload = load_json(path, "HyenaDNA config")
    if hashlib.sha256(payload).hexdigest() != profile["source_files"]["config.json"]["sha256"]:
        raise LongHyenaConversionError("HyenaDNA config SHA256 differs")
    topology = profile["topology"]
    expected = {
        "vocab_size": topology["vocab_size"],
        "d_model": topology["hidden_size"],
        "n_layer": topology["num_layers"],
        "d_inner": topology["inner_width"],
        "filter_order": topology["filter_width"],
        "emb_dim": topology["positional_width"],
        "short_filter_order": topology["short_filter_width"],
        "max_seq_len": topology["max_seqlen"],
        "layer_norm_epsilon": topology["norm_epsilon"],
        "hyena_order": 2,
        "num_inner_mlps": 2,
        "torch_dtype": "float32",
        "use_bias": True,
        "hyena_dropout": 0.0,
        "hyena_filter_dropout": 0.0,
        "pad_token_id": 4,
        "pad_vocab_size_multiple": 8,
    }
    wrong = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if wrong:
        raise LongHyenaConversionError("HyenaDNA config semantics differ: %s" % wrong)
    padded = ((topology["vocab_size"] + 7) // 8) * 8
    if padded != topology["embedding_rows"]:
        raise LongHyenaConversionError("HyenaDNA physical embedding rows differ")


def validate_tokenizer_config(path: Path, profile: Mapping[str, Any]) -> None:
    config, payload = load_json(path, "HyenaDNA tokenizer config")
    if hashlib.sha256(payload).hexdigest() != profile["source_files"]["tokenizer_config.json"]["sha256"]:
        raise LongHyenaConversionError("HyenaDNA tokenizer config SHA256 differs")
    expected = {
        "model_max_length": profile["topology"]["max_seqlen"],
        "padding_side": "left",
        "tokenizer_class": "HyenaDNATokenizer",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
    }
    wrong = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if wrong:
        raise LongHyenaConversionError("HyenaDNA tokenizer semantics differ: %s" % wrong)


def canonical_tensor_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    width = topology["hidden_size"]
    inner = topology["inner_width"]
    filt = topology["filter_width"]
    positional = topology["positional_width"]
    maximum = topology["max_seqlen"]
    short = topology["short_filter_width"]
    specs = [
        TensorSpec(
            "hyena.backbone.embeddings.word_embeddings.weight",
            "F32",
            (topology["embedding_rows"], width),
        )
    ]  # type: List[TensorSpec]
    for layer in range(topology["num_layers"]):
        prefix = "hyena.backbone.layers.%d." % layer
        specs.extend(
            [
                TensorSpec(prefix + "mixer.filter_fn.bias", "F32", (width,)),
                TensorSpec(prefix + "mixer.filter_fn.implicit_filter.0.bias", "F32", (filt,)),
                TensorSpec(prefix + "mixer.filter_fn.implicit_filter.0.weight", "F32", (filt, positional)),
                TensorSpec(prefix + "mixer.filter_fn.implicit_filter.1.freq", "F32", (1, filt)),
                TensorSpec(prefix + "mixer.filter_fn.implicit_filter.2.bias", "F32", (filt,)),
                TensorSpec(prefix + "mixer.filter_fn.implicit_filter.2.weight", "F32", (filt, filt)),
                TensorSpec(prefix + "mixer.filter_fn.implicit_filter.4.bias", "F32", (filt,)),
                TensorSpec(prefix + "mixer.filter_fn.implicit_filter.4.weight", "F32", (filt, filt)),
                TensorSpec(prefix + "mixer.filter_fn.implicit_filter.6.weight", "F32", (width, filt)),
                TensorSpec(prefix + "mixer.filter_fn.modulation.deltas", "F32", (1, 1, width)),
                TensorSpec(prefix + "mixer.filter_fn.pos_emb.t", "F32", (1, maximum, 1)),
                TensorSpec(prefix + "mixer.filter_fn.pos_emb.z", "F32", (1, maximum, positional)),
                TensorSpec(prefix + "mixer.in_proj.bias", "F32", (width * 3,)),
                TensorSpec(prefix + "mixer.in_proj.weight", "F32", (width * 3, width)),
                TensorSpec(prefix + "mixer.out_proj.bias", "F32", (width,)),
                TensorSpec(prefix + "mixer.out_proj.weight", "F32", (width, width)),
                TensorSpec(prefix + "mixer.short_filter.bias", "F32", (width * 3,)),
                TensorSpec(prefix + "mixer.short_filter.weight", "F32", (width * 3, 1, short)),
                TensorSpec(prefix + "mlp.fc1.bias", "F32", (inner,)),
                TensorSpec(prefix + "mlp.fc1.weight", "F32", (inner, width)),
                TensorSpec(prefix + "mlp.fc2.bias", "F32", (width,)),
                TensorSpec(prefix + "mlp.fc2.weight", "F32", (width, inner)),
                TensorSpec(prefix + "norm1.bias", "F32", (width,)),
                TensorSpec(prefix + "norm1.weight", "F32", (width,)),
                TensorSpec(prefix + "norm2.bias", "F32", (width,)),
                TensorSpec(prefix + "norm2.weight", "F32", (width,)),
            ]
        )
    specs.extend(
        [
            TensorSpec("hyena.backbone.ln_f.bias", "F32", (width,)),
            TensorSpec("hyena.backbone.ln_f.weight", "F32", (width,)),
            TensorSpec("lm_head.weight", "F32", (topology["embedding_rows"], width)),
        ]
    )
    return specs


def build_metadata(
    profile: Mapping[str, Any], base: Mapping[str, Any]
) -> Dict[str, Any]:
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
            "hyenadna.embedding_rows": topology["embedding_rows"],
            "hyenadna.inner_width": topology["inner_width"],
            "hyenadna.filter_width": topology["filter_width"],
            "hyenadna.positional_width": topology["positional_width"],
            "hyenadna.short_filter_width": topology["short_filter_width"],
            "hyenadna.norm_epsilon": topology["norm_epsilon"],
            "hyenadna.weight_dtype": "F32",
            "hyenadna.hidden_tap": "post-final-layernorm",
            "hyenadna.pooling": "attention-mask-mean",
            "hyenadna.padding_side": "left",
            "hyenadna.activation": "gelu-tanh",
            "hyenadna.long_convolution": "exact-2l-fft",
            "hyenadna.model_receives_attention_mask": False,
            "hyenadna.add_special_tokens": False,
            "source.modeling_hyena_sha256": PINNED_MODELING_SHA,
            "source.modeling_hyena_path": "modeling_hyena.py",
            "source.modeling_hyena_lines": "14-20,86-160,214-235,250-348",
            "source.tokenization_hyena_sha256": PINNED_TOKENIZATION_SHA,
            "source.tokenization_hyena_path": "tokenization_hyena.py",
            "source.tokenization_hyena_lines": "8-76",
            "source.tokenizer_config_path": "tokenizer_config.json",
            "source.geneb.extractor_sha256": PINNED_EXTRACTOR_SHA,
            "source.geneb.extractor_path": "embedding_pipeline/extractors/hyenadna.py",
            "source.geneb.extractor_lines": "14,16-24,40-56",
            "geneb.provenance.extractor_commit": PINNED_EXTRACTOR_COMMIT,
        }
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=default_config_path("geneb-models.json"))
    parser.add_argument("--profiles", type=Path, default=default_config_path("geneb-hyenadna-models.json"))
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
            raise LongHyenaConversionError("receipt does not identify a HyenaDNA profile")
        entry, catalog_root, catalog_payload = load_catalog_entry(
            args.catalog.resolve(), profile, "left"
        )
        validate_catalog_runtime_contract(
            entry, profile["topology"], "hyena", PINNED_EXTRACTOR_COMMIT
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
        sources = read_hf_safetensors({"model.safetensors": paths["model.safetensors"]})
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
        print("convert_geneb_hyenadna_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
