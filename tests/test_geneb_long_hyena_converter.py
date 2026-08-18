#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict converter, corruption, and native closure for long-Hyena models."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_converter(path: Path, name: str) -> Any:
    tool_directory = str(path.resolve().parent)
    if tool_directory not in sys.path:
        sys.path.insert(0, tool_directory)
    spec = importlib.util.spec_from_file_location(name, str(path.resolve()))
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import converter %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_bytes(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    if bits & 0x7F800000 != 0x7F800000:
        bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (bits >> 16) & 0xFFFF)


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> float:
    integer = ((tensor_index + 5) * 17 + (element_index + 3) * 11) % 37 - 18
    value = f32(f32(float(integer)) / f32(41.0))
    if name == "hyena.backbone.embeddings.word_embeddings.weight":
        row = element_index // 4
        column = element_index % 4
        if row == 4:
            return f32(f32(0.35) - f32(float(column)) * f32(0.11))
        return f32(value * f32(0.24))
    if "hyena.backbone.layers" in name:
        if "norm" in name:
            return 1.0 if "bias" not in name else 0.0
        if "mixer.in_proj.weight" in name:
            return 0.55 if element_index % 4 == (element_index // 4) % 4 else 0.0
        if any(
            piece in name
            for piece in (
                "mixer.in_proj.bias",
                "mixer.short_filter.bias",
                "mixer.out_proj.bias",
                "mlp.fc1.bias",
                "mlp.fc2.bias",
            )
        ):
            return 0.0
        if "mixer.short_filter.weight" in name:
            return 0.35 if element_index % 2 == 0 else 0.65
        if "mixer.out_proj.weight" in name:
            return 0.45 if element_index % 4 == element_index // 4 else 0.0
        if "mixer.filter_fn.bias" in name:
            return 0.4
    if "norm" in name and "bias" not in name:
        return f32(f32(1.0) + f32(value * f32(0.03)))
    if name == "backbone.norm.scale" or ".scale" in name:
        return f32(f32(1.0) + f32(value * f32(0.03)))
    if "filter.poles" in name:
        return f32(f32(0.72) + f32(value * f32(0.02))) if element_index % 2 == 0 else f32(f32(0.04) + f32(value * f32(0.01)))
    if "filter.residues" in name:
        return f32(value * f32(0.08))
    if "modulation.deltas" in name:
        return f32(f32(-1.1) + f32(value * f32(0.1)))
    if "implicit_filter.1.freq" in name:
        return f32(f32(1.7) + f32(value * f32(0.05)))
    if "pos_emb.t" in name:
        return f32(f32(float(element_index)) / f32(3.0))
    if "pos_emb.z" in name:
        row = element_index // 3
        column = element_index % 3
        if column == 0:
            return f32(f32(float(row)) / f32(3.0))
        return f32(math.cos(float(row))) if column == 1 else f32(math.sin(float(row)))
    if "embedding" in name:
        return f32(value * f32(0.24))
    return f32(value * f32(0.09))


def tensor_payload(spec: Any, tensor_index: int, shape: Sequence[int], dtype: str) -> bytes:
    count = 1
    for dimension in shape:
        count *= dimension
    output = bytearray()
    for element_index in range(count):
        value = fixture_scalar(spec.name, tensor_index, element_index)
        output.extend(struct.pack("<f", value) if dtype == "F32" else bf16_bytes(value))
    return bytes(output)


def write_safetensors(
    path: Path,
    indexed_specs: Sequence[Tuple[int, Any]],
    omit: Optional[str] = None,
    extra: bool = False,
    wrong_shape: Optional[str] = None,
    wrong_dtype: Optional[str] = None,
) -> Dict[str, bytes]:
    header = {"__metadata__": {"format": "pt"}}  # type: Dict[str, Any]
    payloads = {}  # type: Dict[str, bytes]
    offset = 0
    for tensor_index, spec in indexed_specs:
        if spec.name == omit:
            continue
        shape = list(spec.shape)
        if spec.name == wrong_shape:
            shape[0] += 1
        dtype = spec.dtype
        if spec.name == wrong_dtype:
            dtype = "BF16" if dtype == "F32" else "F32"
        payload = tensor_payload(spec, tensor_index, shape, dtype)
        header[spec.name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads[spec.name] = payload
        offset += len(payload)
    if extra:
        payloads["unexpected.weight"] = struct.pack("<f", 0.25)
        header["unexpected.weight"] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + 4],
        }
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    padded = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(padded))
        + padded
        + b"".join(payloads[name] for name in header if name != "__metadata__")
    )
    return payloads


def tokenizer_asset(padding_side: str, vocabulary: int) -> Dict[str, Any]:
    pieces = ["[UNK]", "[CLS]", "[SEP]", "[BOS]", "[PAD]", "[RESERVED]", "N", "A", "C", "G", "T"]
    while len(pieces) < vocabulary:
        pieces.append("fixture-%d" % len(pieces))
    return {
        "format": "evo-tokenizer-v1",
        "kind": "character",
        "normalization": [],
        "pre_tokenizer": {"kind": "none"},
        "model": {"unknown_policy": "unk", "match_special_literals": False},
        "post_processor": {
            "prefix_ids": [],
            "suffix_ids": [],
            "padding": {"side": padding_side, "pad_id": 4},
        },
        "special_tokens": {
            "unk": 0,
            "pad": 4,
            "bos": None,
            "eos": None,
            "cls": None,
            "sep": None,
            "mask": None,
        },
        "vocab": [
            {"id": index, "piece": piece}
            for index, piece in enumerate(pieces[:vocabulary])
        ],
    }


def receipt_entry(path: Path) -> Dict[str, Any]:
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": digest(path),
        "path": str(path.resolve()),
    }


def catalog_entry(profile: Mapping[str, Any], family: str, width: int) -> Dict[str, Any]:
    hyena = family == "hyena"
    return {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "family": "hyena" if hyena else "striped-hyena",
        "architecture": profile["catalog_architecture"],
        "source": {
            "kind": "huggingface",
            "repo": profile["repo"],
            "requested_revision": profile["requested_revision"],
            "revision": profile["revision"],
            "immutable": True,
            "required_files": [
                {
                    "path": name,
                    "size": descriptor["size"],
                    "sha256": descriptor["sha256"],
                }
                for name, descriptor in sorted(profile["source_files"].items())
            ],
        },
        "tokenizer": {
            "kind": "single-nucleotide",
            "asset_source": "model-source",
            "assets": None,
            "add_special_tokens": False,
            "padding_side": "left",
            "pad_to": "batch-max" if hyena else "none",
            "max_tokens": profile["topology"]["max_seqlen"],
            "unknown_fields": ["assets"],
        },
        "context": {
            "unit": "tokens",
            "declared_max_tokens": profile["topology"]["max_seqlen"],
            "reference_max_tokens": profile["topology"]["max_seqlen"],
            "length_policy": "tokenizer-truncate" if hyena else "reject",
            "unknown_fields": [],
        },
        "input_transform": {
            "case": "preserve",
            "strip_ascii_whitespace": False,
            "u_to_t": False,
            "invalid": "tokenizer-defined",
            "frame_trim": None,
            "raw_crop": None,
            "fixed_pad": None,
            "prefix": None,
            "special_tokens": "none",
            "token_truncation": "right" if hyena else "left",
        },
        "embedding_presets": {
            "reference": {
                "hidden_tap": "last-hidden-state" if hyena else "model-final-hidden",
                "pooling": "attention-mask-mean" if hyena else "mean-first-record-only",
                "special_tokens": "none",
                "mask_domain": "attention-mask" if hyena else "all-token-rows",
                "output_width": width,
            },
            "normalized": {
                "hidden_tap": "last-hidden-state" if hyena else "model-final-hidden",
                "pooling": "attention-mask-mean" if hyena else "per-record-mean",
                "special_tokens": "none",
                "mask_domain": "attention-mask" if hyena else "record-token-rows",
                "output_width": width,
            },
        },
        "provenance": {
            "extractor": {"commit": "b465d2d6a11efbbc9a22c105e34832725ce50e05"},
            "reference_patch": {"sha256": "c" * 64},
            "normalization_patch_sha256": "d" * 64,
        },
    }


def hyena_config(topology: Mapping[str, Any]) -> Dict[str, Any]:
    return {
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


def evo_config(topology: Mapping[str, Any]) -> Dict[str, Any]:
    attention = topology["attention_layers"]
    return {
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
        "attn_layer_idxs": attention,
        "hyena_layer_idxs": [
            index for index in range(topology["num_layers"]) if index not in attention
        ],
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


def make_case(root: Path, converter: Any, production_profile: Mapping[str, Any], family: str, mutation: Optional[str] = None) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    profile = copy.deepcopy(production_profile)
    profile["runtime_id"] = "geneb-%s-tiny" % family
    profile["geneb_model_id"] = "%s-tiny" % family
    profile["paper_name"] = "%s-tiny" % family
    profile["repo"] = "fixture/%sTiny" % family
    if family == "hyena":
        profile["revision"] = "a" * 40
        topology = {
            "vocab_size": 12,
            "embedding_rows": 16,
            "hidden_size": 4,
            "num_layers": 1,
            "inner_width": 6,
            "filter_width": 3,
            "positional_width": 3,
            "short_filter_width": 2,
            "max_seqlen": 4,
            "norm_epsilon": 0.00001,
        }
        config_value = hyena_config(topology)
        tokenizer_config = {
            "model_max_length": 4,
            "padding_side": "left",
            "tokenizer_class": "HyenaDNATokenizer",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
        }
    else:
        topology = {
            "vocab_size": 8,
            "hidden_size": 4,
            "num_layers": 2,
            "num_attention_heads": 2,
            "inner_width": 6,
            "state_width": 2,
            "short_filter_width": 2,
            "max_seqlen": 8,
            "norm_epsilon": 0.0001,
            "rope_theta": 10000.0,
            "rope_scaling_factor": 2.0,
            "attention_layers": [1],
        }
        config_value = evo_config(topology)
        tokenizer_config = {"tokenizer_class": "ByteTokenizer"}
    profile["topology"] = topology
    specs = converter.canonical_tensor_specs(topology)
    profile["checkpoint_manifest_sha256"] = converter.tensor_manifest_sha256(specs)

    config_path = root / "config.json"
    tokenizer_config_path = root / "tokenizer_config.json"
    if mutation == "config":
        config_value["max_seqlen" if family == "evo" else "max_seq_len"] += 1
    write_json(config_path, config_value)
    write_json(tokenizer_config_path, tokenizer_config)
    source_paths = [config_path]  # type: List[Path]
    payloads = {}  # type: Dict[str, bytes]
    indexed = list(enumerate(specs))
    target_name = specs[0].name
    if family == "hyena":
        weights = root / "model.safetensors"
        payloads.update(
            write_safetensors(
                weights,
                indexed,
                omit=target_name if mutation == "missing" else None,
                extra=mutation == "extra",
                wrong_shape=target_name if mutation == "shape" else None,
                wrong_dtype=target_name if mutation == "dtype" else None,
            )
        )
        source_paths.append(weights)
    else:
        shard_names = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        partitions = [indexed[index::3] for index in range(3)]
        owners = {}  # type: Dict[str, str]
        for shard_index, (name, partition) in enumerate(zip(shard_names, partitions)):
            shard = root / name
            payloads.update(
                write_safetensors(
                    shard,
                    partition,
                    omit=target_name if mutation == "missing" and shard_index == 0 else None,
                    extra=mutation == "extra" and shard_index == 0,
                    wrong_shape=target_name if mutation == "shape" and shard_index == 0 else None,
                    wrong_dtype=target_name if mutation == "dtype" and shard_index == 0 else None,
                )
            )
            source_paths.append(shard)
            owners.update({spec.name: name for _, spec in partition})
        index_path = root / "model.safetensors.index.json"
        write_json(
            index_path,
            {
                "metadata": {
                    "total_size": (
                        sum(len(payload) for payload in payloads.values())
                        if mutation in ("shape", "dtype")
                        else sum(spec.nbytes for spec in specs)
                    )
                },
                "weight_map": owners,
            },
        )
        source_paths.append(index_path)
    source_paths.append(tokenizer_config_path)
    for path in source_paths:
        profile["source_files"][path.name] = {
            "size": path.stat().st_size,
            "sha256": digest(path),
        }
    profile["conversion_assets"] = [path.name for path in source_paths]
    if family == "evo" and mutation == "cache-byte":
        profile["source_files"]["cache.py"]["sha256"] = "0" * 64
    if family == "evo" and mutation == "old-source-contract":
        del profile["source_files"]["cache.py"]

    profiles = root / "profiles.json"
    write_json(
        profiles,
        {"schema_version": 1, "format": converter.PROFILE_FORMAT, "models": [profile]},
    )
    entry = catalog_entry(profile, family, topology["hidden_size"])
    if mutation == "catalog":
        entry["input_transform"]["case"] = "upper"
    if family == "evo" and mutation == "catalog-source-missing":
        entry["source"]["required_files"] = [
            item
            for item in entry["source"]["required_files"]
            if item["path"] != "cache.py"
        ]
    if family == "evo" and mutation == "catalog-cache-byte":
        for item in entry["source"]["required_files"]:
            if item["path"] == "cache.py":
                item["sha256"] = "0" * 64
    if family == "evo" and mutation == "catalog-source-extra":
        entry["source"]["required_files"].append(
            {"path": "unexpected.py", "size": 1, "sha256": "0" * 64}
        )
    catalog = root / "catalog.json"
    write_json(
        catalog,
        {
            "schema_version": 1,
            "suite": {"id": "geneb-v4", "raw_safety_cap_bytes": 16777216},
            "models": [entry],
        },
    )
    receipt = root / "source-receipt.json"
    receipt_files = [receipt_entry(path) for path in source_paths]
    extra_receipt_asset = None  # type: Optional[Path]
    if mutation in ("receipt-extra", "receipt-extra-corrupt"):
        extra_receipt_asset = root / "metadata" / "generation_config.json"
        write_json(extra_receipt_asset, {"fixture": family})
        receipt_files.append(
            {
                "name": "metadata/generation_config.json",
                "size": extra_receipt_asset.stat().st_size,
                "sha256": digest(extra_receipt_asset),
                "path": str(extra_receipt_asset.resolve()),
            }
        )
    if mutation == "receipt-sha":
        receipt_files[0]["sha256"] = "0" * 64
    write_json(
        receipt,
        {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": profile["runtime_id"],
            "repo": profile["repo"],
            "requested_revision": profile["requested_revision"],
            "resolved_revision": profile["revision"],
            "files": receipt_files,
            "load_path": None,
            "source_kind": "huggingface",
            "catalog_path": str(catalog.resolve()),
            "catalog_sha256": digest(catalog),
        },
    )
    if mutation == "receipt-source-kind":
        value = json.loads(receipt.read_text(encoding="utf-8"))
        value["source_kind"] = "google-drive"
        write_json(receipt, value)
    if mutation == "receipt-catalog-path":
        value = json.loads(receipt.read_text(encoding="utf-8"))
        value["catalog_path"] = str((root / "other-catalog.json").resolve())
        write_json(receipt, value)
    if mutation == "receipt-catalog-sha":
        value = json.loads(receipt.read_text(encoding="utf-8"))
        value["catalog_sha256"] = "0" * 64
        write_json(receipt, value)
    if mutation == "receipt-catalog-partial":
        value = json.loads(receipt.read_text(encoding="utf-8"))
        del value["catalog_sha256"]
        write_json(receipt, value)
    if mutation == "receipt-extra-corrupt":
        assert extra_receipt_asset is not None
        extra_receipt_asset.write_bytes(b"corrupt after receipt\n")
    asset = root / "assets" / "tokenizer.json"
    if mutation == "tokenizer-symlink":
        real_asset = root / "real-assets" / "tokenizer.json"
        write_json(real_asset, tokenizer_asset("left", topology["vocab_size"]))
        asset.parent.symlink_to(real_asset.parent.resolve(), target_is_directory=True)
    else:
        write_json(asset, tokenizer_asset("left", topology["vocab_size"]))
    descriptor = root / "tokenizer-descriptor.json"
    compiler_sha = profile["tokenizer"]["compiler_manifest_sha256"]
    if mutation == "tokenizer-compiler":
        compiler_sha = "0" * 64
    write_json(
        descriptor,
        {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": compiler_sha,
            "source_receipt_contract_sha256": "b" * 64,
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": "assets/tokenizer.json",
            "tokenizer.sha256": digest(asset),
            "tokenizer.size": asset.stat().st_size,
        },
    )
    return {
        "root": root,
        "profile": profile,
        "profiles": profiles,
        "catalog": catalog,
        "receipt": receipt,
        "descriptor": descriptor,
        "output": root / "runtime.safetensors",
        "specs": specs,
        "payloads": payloads,
    }


def command(args: argparse.Namespace, case: Mapping[str, Any], family: str) -> List[str]:
    converter = args.hyena_converter if family == "hyena" else args.evo_converter
    return [
        sys.executable,
        str(converter),
        "--receipt",
        str(case["receipt"]),
        "--catalog",
        str(case["catalog"]),
        "--profiles",
        str(case["profiles"]),
        "--tokenizer-descriptor",
        str(case["descriptor"]),
        "--tokenizer-root",
        str(case["root"]),
        "--output",
        str(case["output"]),
    ]


def convert(args: argparse.Namespace, case: Mapping[str, Any], family: str) -> subprocess.CompletedProcess:
    return subprocess.run(command(args, case, family), check=False, capture_output=True, text=True)


def expect_failure(args: argparse.Namespace, case: Mapping[str, Any], family: str, needle: str) -> None:
    result = convert(args, case, family)
    if result.returncode == 0 or needle not in result.stderr:
        raise AssertionError(
            "%s expected failure containing %r:\n%s\n%s"
            % (family, needle, result.stdout, result.stderr)
        )
    if case["output"].exists():
        raise AssertionError("failed conversion published a partial artifact")


def read_artifact(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        data = source.read()
    return header.pop("__metadata__"), header, data


def rewrite_artifact(source: Path, destination: Path, metadata_key: str, value: str) -> None:
    metadata, tensors, data = read_artifact(source)
    metadata[metadata_key] = value
    root = dict(tensors)
    root["__metadata__"] = metadata
    raw = json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    padded = raw + b" " * ((-len(raw)) % 8)
    destination.write_bytes(struct.pack("<Q", len(padded)) + padded + data)


def native(args: argparse.Namespace, family: str, artifact: Path, success: bool) -> subprocess.CompletedProcess:
    option = "--verify-hyena-artifact" if family == "hyena" else "--verify-evo-artifact"
    result = subprocess.run(
        [str(args.native_loader), option, str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    if success and result.returncode != 0:
        raise AssertionError("native %s artifact failed:\n%s" % (family, result.stderr))
    if not success and result.returncode == 0:
        raise AssertionError("native %s corruption was accepted" % family)
    return result


def native_cpu_adapter(
    args: argparse.Namespace, family: str, artifact: Path
) -> Mapping[str, Any]:
    result = subprocess.run(
        [str(args.native_loader), "--verify-cpu-adapter", family, str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "public CPU %s adapter failed:\n%s%s"
            % (family, result.stdout, result.stderr)
        )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError("public CPU adapter output is not JSON: %s" % error)
    expected = (
        {
            "family": "hyena",
            "tokens": [7, 8],
            "padded_tokens": [4, 7, 8],
            "mask": [0, 1, 1],
            "rows": 3,
            "width": 4,
            "pooling": "attention-mask-mean",
        }
        if family == "hyena"
        else {
            "family": "evo",
            "tokens": [7, 6],
            "mask": [1, 1],
            "rows": 2,
            "width": 4,
            "pooling": "mean-first-record-only",
        }
    )
    if document != expected:
        raise AssertionError("public CPU %s adapter JSON differs: %r" % (family, document))
    return document


def validate_production_tokenizer(
    args: argparse.Namespace,
    profile: Mapping[str, Any],
    profile_directory: Path,
    work: Path,
    input_text: str,
    expected_ids: str,
) -> None:
    manifest_path = profile_directory / profile["tokenizer"]["compiler_manifest"]
    if digest(manifest_path) != profile["tokenizer"]["compiler_manifest_sha256"]:
        raise AssertionError("production tokenizer compiler manifest SHA differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_files = []  # type: List[Dict[str, Any]]
    for entry in manifest["files"]:
        source = manifest_path.parent / entry["name"]
        if source.stat().st_size != entry["size"] or digest(source) != entry["sha256"]:
            raise AssertionError("production tokenizer source size/SHA differs")
        receipt_files.append(dict(entry, path=str(source.resolve())))
    work.mkdir(parents=True, exist_ok=True)
    receipt = work / "source-receipt.json"
    write_json(
        receipt,
        {"schema_version": 1, "kind": "tokenizer-source", "files": receipt_files},
    )
    output = work / "assets" / "tokenizer.json"
    descriptor = work / "descriptor.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(args.tokenizer_compiler),
            "--manifest",
            str(manifest_path),
            "--receipt",
            str(receipt),
            "--output",
            str(output),
            "--descriptor",
            str(descriptor),
            "--asset-path",
            "assets/tokenizer.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError("production tokenizer compilation failed:\n" + completed.stderr)
    description = json.loads(descriptor.read_text(encoding="utf-8"))
    verified = subprocess.run(
        [
            str(args.tokenizer_runtime),
            "--verify-asset",
            str(work),
            description["tokenizer.path"],
            description["tokenizer.sha256"],
            str(description["tokenizer.size"]),
            input_text,
            expected_ids,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode != 0:
        raise AssertionError("production tokenizer native closure failed:\n" + verified.stderr)


def validate_success(args: argparse.Namespace, case: Mapping[str, Any], family: str, oracle: Mapping[str, Any]) -> None:
    result = convert(args, case, family)
    if result.returncode != 0:
        raise AssertionError("valid %s conversion failed:\n%s%s" % (family, result.stdout, result.stderr))
    metadata, tensors, _ = read_artifact(case["output"])
    expected_profile = "geneb-hyenadna-runtime-v1" if family == "hyena" else "geneb-evo1-runtime-v1"
    if metadata.get("runtime.profile") != "s:" + expected_profile:
        raise AssertionError("%s runtime profile differs" % family)
    expected_vocabulary = 12 if family == "hyena" else 8
    if metadata.get("runtime.tokenizer_vocabulary_size") != "u:%d" % expected_vocabulary:
        raise AssertionError("%s tokenizer vocabulary metadata differs" % family)
    if set(tensors) != {spec.name for spec in case["specs"]}:
        raise AssertionError("%s exact tensor set differs" % family)
    if any(key.startswith("converter.") for key in metadata):
        raise AssertionError("tokenizer receipt fields leaked into model metadata")
    result = native(args, family, case["output"], True)
    document = json.loads(result.stdout)
    expected_key = "hyena.reference.pooled" if family == "hyena" else "evo.pooled"
    expected = oracle["vectors"][expected_key]
    actual = document.get("pooled")
    maximum = max(abs(float(a) - float(b)) for a, b in zip(actual, expected))
    if maximum > 1.0e-5:
        raise AssertionError("%s converter/native oracle max_abs %.9g" % (family, maximum))
    native_cpu_adapter(args, family, case["output"])
    corrupted = case["root"] / "corrupted-metadata.safetensors"
    if family == "hyena":
        rewrite_artifact(case["output"], corrupted, "hyenadna.long_convolution", "s:quadratic")
    else:
        rewrite_artifact(
            case["output"],
            corrupted,
            "evo1.norm_denominator",
            "s:sqrt-mean-plus-epsilon-inside",
        )
    native(args, family, corrupted, False)


def validate_production(args: argparse.Namespace, hyena: Any, evo: Any) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    hyena_profiles, _ = hyena.load_profiles(args.hyena_profiles.resolve())
    evo_profiles, _ = evo.load_profiles(args.evo_profiles.resolve())
    if set(hyena_profiles) != {"geneb-hyenadna-medium-160k", "geneb-hyenadna-large-1m"}:
        raise AssertionError("production HyenaDNA profiles differ")
    expected_hyena_hashes = {
        "geneb-hyenadna-medium-160k": "7f32d3cc46db00f93668dbe5c473fe7d785181743506efa754321b76957e3268",
        "geneb-hyenadna-large-1m": "8d62dda67973ee571bdabdfb0316349747cdb197a7e7f13da0fbd30916ebe358",
    }
    expected_tokenizer_configs = {
        "geneb-hyenadna-medium-160k": "dd519d81a1891aa4cb1933e4787b60fa449ff818ab4c531953899dc538633429",
        "geneb-hyenadna-large-1m": "7ceea744a19e155f056818b05e80195093ade809614964d8016ce2505c1485d4",
    }
    for name, profile in hyena_profiles.items():
        specs = hyena.canonical_tensor_specs(profile["topology"])
        if len(specs) != 212 or hyena.tensor_manifest_sha256(specs) != expected_hyena_hashes[name]:
            raise AssertionError("%s exact production manifest differs" % name)
        if profile["source_files"]["tokenization_hyena.py"]["sha256"] != hyena.PINNED_TOKENIZATION_SHA:
            raise AssertionError("HyenaDNA tokenizer source SHA differs")
        if profile["source_files"]["tokenizer_config.json"]["sha256"] != expected_tokenizer_configs[name]:
            raise AssertionError("%s tokenizer_config SHA differs" % name)
    evo_profile = evo_profiles.get("geneb-evo-1-131k")
    if evo_profile is None:
        raise AssertionError("production Evo-1 profile is missing")
    specs = evo.canonical_tensor_specs(evo_profile["topology"])
    if len(specs) != 438 or evo.tensor_manifest_sha256(specs) != "8021ede2367eeb16b6a37a27f41024efd7cede2e5b76d0cc50746ddde81cf18a":
        raise AssertionError("Evo-1 exact production manifest differs")
    if sum(spec.dtype == "F32" for spec in specs) != 58:
        raise AssertionError("Evo-1 mixed dtype manifest differs")
    if evo_profile["source_files"]["model.py"]["sha256"] != evo.PINNED_MODEL_SHA:
        raise AssertionError("Evo-1 model source path/SHA differs")
    if (
        len(evo_profile["source_files"]) != 16
        or evo_profile["source_files"].get("cache.py")
        != {"size": 1378, "sha256": evo.PINNED_CACHE_SHA}
    ):
        raise AssertionError("Evo-1 executable source closure differs")
    if evo_profile["source_files"]["tokenizer_config.json"]["sha256"] != "52fdc041aadead5cc38e72ba7db3e64965526cd974f5f9aa1185a3d9b7aca32c":
        raise AssertionError("Evo-1 tokenizer_config SHA differs")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    rows = {item.get("runtime_id"): item for item in catalog["models"]}
    for name in expected_hyena_hashes:
        row = rows[name]
        if row["tokenizer"]["padding_side"] != "left" or not row["provenance"]["known_defects"]:
            raise AssertionError("%s left-padding provenance differs" % name)
        hyena.validate_catalog_runtime_contract(
            row,
            hyena_profiles[name]["topology"],
            "hyena",
            hyena.PINNED_EXTRACTOR_COMMIT,
        )
    evo.validate_catalog_runtime_contract(
        rows["geneb-evo-1-131k"],
        evo_profile["topology"],
        "evo",
        evo.PINNED_EXTRACTOR_COMMIT,
    )
    evo.validate_catalog_source_files(rows["geneb-evo-1-131k"], evo_profile)
    exact_receipt = {
        "files": [{"name": name} for name in sorted(evo_profile["source_files"])]
    }
    evo.validate_production_receipt_file_set(exact_receipt, evo_profile)
    for label, files in (
        (
            "old-15-file",
            [
                item
                for item in exact_receipt["files"]
                if item["name"] != "cache.py"
            ],
        ),
        (
            "extra-file",
            exact_receipt["files"] + [{"name": "unexpected.py"}],
        ),
    ):
        try:
            evo.validate_production_receipt_file_set({"files": files}, evo_profile)
        except evo.LongHyenaConversionError as error:
            if "source receipt file set differs" not in str(error):
                raise AssertionError("Evo-1 %s receipt failure differs" % label)
        else:
            raise AssertionError("Evo-1 %s receipt was accepted" % label)
    return hyena_profiles["geneb-hyenadna-medium-160k"], evo_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyena-converter", required=True, type=Path)
    parser.add_argument("--evo-converter", required=True, type=Path)
    parser.add_argument("--hyena-profiles", required=True, type=Path)
    parser.add_argument("--evo-profiles", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--native-loader", required=True, type=Path)
    parser.add_argument("--tokenizer-compiler", required=True, type=Path)
    parser.add_argument("--tokenizer-runtime", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    hyena = import_converter(args.hyena_converter, "t33_hyena_converter")
    evo = import_converter(args.evo_converter, "t33_evo_converter")
    production_hyena, production_evo = validate_production(args, hyena, evo)
    validate_production_tokenizer(
        args,
        production_hyena,
        args.hyena_profiles.resolve().parent,
        args.work_dir / "hyena-production-tokenizer",
        "ACGTN",
        "7,8,9,10,11",
    )
    validate_production_tokenizer(
        args,
        production_evo,
        args.evo_profiles.resolve().parent,
        args.work_dir / "evo-production-tokenizer",
        "ACGT",
        "65,67,71,84",
    )
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    for family, converter, production in (
        ("hyena", hyena, production_hyena),
        ("evo", evo, production_evo),
    ):
        case = make_case(args.work_dir / (family + "-valid"), converter, production, family)
        validate_success(args, case, family, oracle)
        extra_receipt = make_case(
            args.work_dir / (family + "-receipt-extra"),
            converter,
            production,
            family,
            "receipt-extra",
        )
        validate_success(args, extra_receipt, family, oracle)
        corrupt_extra_receipt = make_case(
            args.work_dir / (family + "-receipt-extra-corrupt"),
            converter,
            production,
            family,
            "receipt-extra-corrupt",
        )
        expect_failure(
            args,
            corrupt_extra_receipt,
            family,
            "source receipt asset metadata/generation_config.json size/SHA differs",
        )
        failures = {
            "missing": "tensor",
            "extra": "tensor",
            "shape": "shape",
            "dtype": "dtype",
            "config": "semantics differ",
            "catalog": "catalog input transform differs",
            "receipt-sha": "size/SHA differs",
            "receipt-source-kind": "catalog provenance differs",
            "receipt-catalog-path": "catalog provenance differs",
            "receipt-catalog-sha": "catalog provenance differs",
            "receipt-catalog-partial": "catalog provenance is incomplete",
            "tokenizer-compiler": "schema/compiler differs",
            "tokenizer-symlink": "contains a symlink",
        }
        if family == "evo":
            failures.update(
                {
                    "old-source-contract": "source file set differs",
                    "cache-byte": "source code/revision is not pinned",
                    "catalog-source-missing": "catalog Evo-1 source file set differs",
                    "catalog-cache-byte": "catalog Evo-1 source file set differs",
                    "catalog-source-extra": "catalog Evo-1 source file set differs",
                }
            )
        for mutation, needle in failures.items():
            corrupted = make_case(
                args.work_dir / (family + "-" + mutation),
                converter,
                production,
                family,
                mutation,
            )
            expect_failure(args, corrupted, family, needle)
        separated = make_case(
            args.work_dir / (family + "-separated"), converter, production, family
        )
        separated["output"] = separated["root"] / "published" / "runtime.safetensors"
        expect_failure(args, separated, family, "artifact output root")
    print("GENEB long-Hyena converter/native contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
