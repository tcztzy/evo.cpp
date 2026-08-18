#!/usr/bin/env python3
"""Offline contract tests for the strict GENEB BERT checkpoint converter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CASE_KINDS = (
    "gena-final",
    "gena-no-final",
    "gena-pretraining",
    "dnabert-s",
    "dnabert-2",
    "grover",
    "mutbert",
)
PRODUCTION_IDENTITIES = {
    "geneb-gena-lm-large-t2t": (
        "AIRI-Institute/gena-lm-bert-large-t2t",
        "f997a8a7c1ba5feb6d11de46354a41c51ffe9660",
        "safetensors",
        "28aeb28e3326fe377d98ac5b624e3e81005026671e3bc278b6a5dc37df34098a",
        399,
    ),
    "geneb-gena-lm-t2t-multi": (
        "AIRI-Institute/gena-lm-bert-base-t2t-multi",
        "4633e5a1ada905bb7afee6877d71cc12578a95a5",
        "pytorch-bin",
        "2afe290d44e50890cbdb48b61ce9c8c944dbf5d07c4796bd3de0840f420129b9",
        205,
    ),
    "geneb-gena-lm": (
        "AIRI-Institute/gena-lm-bert-base",
        "416f055300346a5830ca49438daf5f4e136ed9a8",
        "pytorch-bin",
        "f4288ae1b2270570a803bb90b0c207f9927149876f333955df97f4e91eff2f67",
        209,
    ),
    "geneb-dnabert-s": (
        "zhihan1996/DNABERT-S",
        "00e47f96cdea35e4b6f5df89e5419cbe47d490c6",
        "pytorch-bin",
        "d54c142114563aa34ef7c2b54ccc921d08e50cd7c3cd69c439d5e0d2bbcfcdee",
        138,
    ),
    "geneb-dnabert-2": (
        "zhihan1996/DNABERT-2-117M",
        "7bce263b15377fc15361f52cfab88f8b586abda0",
        "pytorch-bin",
        "ba9bdafaff0cc3e30556927474d4a179519a9864012bed2628e9f1bc23c84bfd",
        142,
    ),
    "geneb-grover": (
        "PoetschLab/GROVER",
        "6b223110f0d6963e849f55bc2a2f3cff0e38c7a4",
        "pytorch-bin",
        "de60db6f5cb476d9f38630517abec87803e32a87672ace7946b5cf82f68ac5de",
        206,
    ),
    "geneb-mutbert": (
        "JadenLong/MutBERT",
        "b68d8d6c9ccd8167639b25fb979cbd39a5c5c60c",
        "pytorch-bin",
        "543cf23dba83fb917c9bec4d92db330e9609fa76c3e629675024fe5e6ccd3d33",
        203,
    ),
}


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_bytes(value))


def elements(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def tensor_payload(spec: Any) -> bytes:
    count = elements(spec.shape)
    seed = sum(spec.name.encode("ascii")) % 97
    if spec.dtype == "I64":
        return b"".join(struct.pack("<q", seed + index) for index in range(count))
    if spec.dtype == "F32":
        return b"".join(
            struct.pack("<f", (seed + index) * 0.001) for index in range(count)
        )
    if spec.dtype == "BF16":
        return b"".join(
            struct.pack("<H", (seed + index) & 0xFFFF) for index in range(count)
        )
    raise AssertionError("unsupported fixture dtype %s" % spec.dtype)


def write_safetensors(path: Path, specs: Sequence[Any]) -> None:
    root = {}  # type: Dict[str, Any]
    payloads = []  # type: List[bytes]
    offset = 0
    for spec in sorted(specs, key=lambda value: value.name):
        payload = tensor_payload(spec)
        root[spec.name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        payloads.append(payload)
    raw = json.dumps(root, sort_keys=True, separators=(",", ":")).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(payloads))


def read_safetensors(path: Path) -> Tuple[Dict[str, Any], bytes]:
    payload = path.read_bytes()
    header_size = struct.unpack("<Q", payload[:8])[0]
    header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
    return header, payload[8 + header_size :]


def rewrite_safetensors(path: Path, header: Mapping[str, Any], data: bytes) -> None:
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    padded = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + data)


def base_topology(kind: str) -> Dict[str, Any]:
    topology = {
        "vocab_size": 9,
        "hidden_size": 4,
        "num_layers": 1,
        "num_attention_heads": 2,
        "head_dim": 2,
        "inner_mlp_size": 6,
        "max_seqlen": 8,
        "type_vocab_size": 2,
        "layer_norm_epsilon": 1e-5,
        "rope_base": 0.0,
        "position_encoding": "absolute",
        "norm_placement": "post",
        "final_layer_norm": False,
        "unpad_masked_tokens": False,
        "mlp_kind": "gelu",
        "qkv_layout": "separate",
        "input_kind": "token-ids",
        "pooling": "attention-mask-mean",
        "attention_bias": True,
        "mlp_input_bias": True,
        "mlp_output_bias": True,
        "embedding_dtype": "F32",
        "projection_dtype": "F32",
        "norm_dtype": "F32",
        "activation_dtype": "F32",
    }
    if kind in ("gena-final", "gena-no-final", "gena-pretraining"):
        topology["norm_placement"] = "pre"
        topology["final_layer_norm"] = kind == "gena-final"
    elif kind in ("dnabert-s", "dnabert-2"):
        topology.update(
            {
                "position_encoding": "alibi-symmetric",
                "mlp_kind": "gated-gelu",
                "qkv_layout": "fused-qkv",
                "unpad_masked_tokens": True,
                "mlp_input_bias": False,
                "pooling": "cls-token" if kind == "dnabert-s" else "attention-mask-mean",
            }
        )
    elif kind == "mutbert":
        topology.update(
            {
                "position_encoding": "rope-split-half",
                "input_kind": "soft-vocabulary",
                "rope_base": 10000.0,
            }
        )
    elif kind != "grover":
        raise AssertionError("unknown fixture kind %s" % kind)
    return topology


def config_for(kind: str, topology: Mapping[str, Any]) -> Dict[str, Any]:
    config = {
        "vocab_size": topology["vocab_size"],
        "hidden_size": topology["hidden_size"],
        "num_hidden_layers": topology["num_layers"],
        "num_attention_heads": topology["num_attention_heads"],
        "intermediate_size": topology["inner_mlp_size"],
        "max_position_embeddings": topology["max_seqlen"],
        "type_vocab_size": topology["type_vocab_size"],
        "layer_norm_eps": topology["layer_norm_epsilon"],
        "hidden_act": "gelu",
        "torch_dtype": "float32",
    }
    if kind in ("gena-final", "gena-no-final", "gena-pretraining"):
        config.update(
            {
                "pre_layer_norm": True,
                "last_layer_norm": kind == "gena-final",
                "position_embedding_type": "absolute",
            }
        )
    elif kind in ("dnabert-s", "dnabert-2"):
        config.update(
            {
                "alibi_starting_size": topology["max_seqlen"],
                "position_embedding_type": "absolute",
            }
        )
    elif kind == "mutbert":
        config.update({"rope_theta": 10000.0, "rope_scaling": None})
    else:
        config["position_embedding_type"] = "absolute"
    return config


def policy_for(kind: str) -> str:
    return {
        "gena-final": "gena-mlm-position-ids",
        "gena-no-final": "gena-mlm-position-ids",
        "gena-pretraining": "gena-pretraining-position-ids",
        "dnabert-s": "mosaic-pooler",
        "dnabert-2": "mosaic-mlm",
        "grover": "bert-mlm-pooler",
        "mutbert": "mutbert-mlm",
    }[kind]


def make_profile(kind: str, index: int) -> Tuple[Dict[str, Any], bytes]:
    topology = base_topology(kind)
    config_payload = canonical_bytes(config_for(kind, topology))
    runtime_id = "fixture-%s" % kind
    return (
        {
            "runtime_id": runtime_id,
            "geneb_model_id": runtime_id,
            "paper_name": "Fixture %s" % kind,
            "catalog_architecture": "fixture-bert-encoder",
            "repo": "fixture/%s" % kind,
            "revision": ("%040x" % (index + 1)),
            "source_format": "safetensors",
            "config_sha256": sha256_bytes(config_payload),
            "config_required": json.loads(config_payload.decode("utf-8")),
            "config_defaults": {},
            "source_prefix": "" if kind == "dnabert-s" else "bert.",
            "source_omit_policy": policy_for(kind),
            "topology": topology,
        },
        config_payload,
    )


def catalog_entry(profile: Mapping[str, Any]) -> Dict[str, Any]:
    topology = profile["topology"]
    mask_domain = "cls-row" if topology["pooling"] == "cls-token" else "attention-mask"
    preset = {
        "hidden_tap": "last-hidden-state",
        "pooling": topology["pooling"],
        "special_tokens": "include",
        "mask_domain": mask_domain,
        "output_width": topology["hidden_size"],
    }
    return {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "architecture": profile["catalog_architecture"],
        "family": "transformer-encoder",
        "source": {
            "kind": "huggingface",
            "repo": profile["repo"],
            "requested_revision": "main",
            "revision": profile["revision"],
            "immutable": True,
        },
        "context": {
            "unit": "tokens",
            "length_policy": "tokenizer-truncate",
            "declared_max_tokens": topology["max_seqlen"],
            "reference_max_tokens": topology["max_seqlen"],
        },
        "input_transform": {
            "case": "preserve",
            "strip_ascii_whitespace": False,
            "fixed_pad": None,
            "frame_trim": None,
            "invalid": "tokenizer-defined",
            "prefix": None,
            "raw_crop": None,
            "special_tokens": (
                "tokenizer-default-then-one-hot-float"
                if profile["source_omit_policy"] == "mutbert-mlm"
                else "tokenizer-default"
            ),
            "token_truncation": "right",
            "u_to_t": False,
        },
        "embedding_presets": {"reference": dict(preset), "normalized": dict(preset)},
        "provenance": {
            "extractor": {"commit": "1" * 40},
            "reference_patch": {"sha256": "2" * 64},
            "normalization_patch_sha256": "3" * 64,
        },
    }


def receipt_file(path: Path, name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "path": str(path.resolve()),
    }


def write_receipt(root: Path, profile: Mapping[str, Any], names: Sequence[str]) -> Path:
    receipt = {
        "schema_version": 1,
        "kind": "source-checkpoint",
        "model_id": profile["runtime_id"],
        "repo": profile["repo"],
        "requested_revision": "main",
        "resolved_revision": profile["revision"],
        "files": [receipt_file(root / name, name) for name in names],
    }
    path = root / "source-receipt.json"
    write_json(path, receipt)
    return path


def refresh_receipt(root: Path) -> None:
    path = root / "source-receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["files"] = [
        receipt_file(root / item["name"], item["name"]) for item in receipt["files"]
    ]
    write_json(path, receipt)


def write_tokenizer_descriptor(root: Path) -> Path:
    asset = root / "tokenizer.json"
    write_json(
        asset,
        {
            "format": "evo-tokenizer-v1",
            "kind": "single-nucleotide",
            "normalization": [{"op": "ascii-uppercase"}, {"op": "u-to-t"}],
            "pre_tokenizer": {"kind": "none"},
            "model": {
                "unknown_policy": "unk",
                "match_special_literals": False,
            },
            "post_processor": {
                "prefix_ids": [7],
                "suffix_ids": [8],
                "padding": {"side": "right", "pad_id": 6},
            },
            "special_tokens": {
                "unk": 5,
                "pad": 6,
                "bos": None,
                "eos": 8,
                "cls": 7,
                "sep": None,
                "mask": None,
            },
            "vocab": [
                {"id": 0, "piece": "A"},
                {"id": 1, "piece": "C"},
                {"id": 2, "piece": "G"},
                {"id": 3, "piece": "T"},
                {"id": 4, "piece": "N"},
                {"id": 5, "piece": "[UNK]"},
                {"id": 6, "piece": "[PAD]"},
                {"id": 7, "piece": "[CLS]"},
                {"id": 8, "piece": "[EOS]"},
            ],
        },
    )
    descriptor = {
        "converter.schema": "evo-tokenizer-conversion-receipt",
        "converter.version": 1,
        "compiler_manifest_sha256": "4" * 64,
        "source_receipt_contract_sha256": "5" * 64,
        "tokenizer.profile": "evo-tokenizer-v1",
        "tokenizer.path": asset.name,
        "tokenizer.sha256": sha256_file(asset),
        "tokenizer.size": asset.stat().st_size,
    }
    path = root / "tokenizer-descriptor.json"
    write_json(path, descriptor)
    return path


def write_sharded_weights(root: Path, specs: Sequence[Any]) -> List[str]:
    midpoint = len(specs) // 2
    partitions = (specs[:midpoint], specs[midpoint:])
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    weight_map = {}  # type: Dict[str, str]
    for name, partition in zip(shard_names, partitions):
        write_safetensors(root / name, partition)
        for spec in partition:
            weight_map[spec.name] = name
    index = {
        "metadata": {"total_size": sum(spec.nbytes for spec in specs)},
        "weight_map": weight_map,
    }
    index_name = "model.safetensors.index.json"
    write_json(root / index_name, index)
    return [index_name] + list(shard_names)


def make_cases(work_dir: Path, converter: Any) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    profiles = []  # type: List[Dict[str, Any]]
    entries = []  # type: List[Dict[str, Any]]
    cases = []  # type: List[Dict[str, Any]]
    for index, kind in enumerate(CASE_KINDS):
        profile, config_payload = make_profile(kind, index)
        profiles.append(profile)
        entries.append(catalog_entry(profile))
        root = work_dir / kind
        root.mkdir(parents=True)
        (root / "config.json").write_bytes(config_payload)
        source_specs, runtime_specs = converter.source_tensor_specs(profile)
        if kind == "dnabert-2":
            weight_names = write_sharded_weights(root, source_specs)
        else:
            weight_names = ["model.safetensors"]
            write_safetensors(root / weight_names[0], source_specs)
        receipt = write_receipt(root, profile, ["config.json"] + weight_names)
        descriptor = write_tokenizer_descriptor(root)
        cases.append(
            {
                "kind": kind,
                "root": root,
                "profile": profile,
                "source_specs": source_specs,
                "runtime_specs": runtime_specs,
                "receipt": receipt,
                "descriptor": descriptor,
                "weight_names": weight_names,
            }
        )
    profiles_path = work_dir / "profiles.json"
    write_json(
        profiles_path,
        {"schema_version": 1, "format": "geneb-bert-converter-v1", "models": profiles},
    )
    catalog_path = work_dir / "catalog.json"
    write_json(
        catalog_path,
        {
            "schema_version": 1,
            "suite": {"id": "geneb-v4", "raw_safety_cap_bytes": 1000000},
            "models": entries,
        },
    )
    return profiles_path, catalog_path, cases


def converter_command(
    converter_path: Path,
    profiles: Path,
    catalog: Path,
    case: Mapping[str, Any],
    output: Path,
    force: bool = False,
) -> List[str]:
    command = [
        sys.executable,
        str(converter_path),
        "--receipt",
        str(case["receipt"]),
        "--catalog",
        str(catalog),
        "--profiles",
        str(profiles),
        "--tokenizer-descriptor",
        str(case["descriptor"]),
        "--output",
        str(output),
    ]
    if force:
        command.append("--force")
    return command


def require_success(command: Sequence[str], label: str) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode != 0:
        raise AssertionError(
            "%s failed (%d):\n%s%s"
            % (label, process.returncode, process.stdout, process.stderr)
        )
    return process


def require_failure(
    command: Sequence[str], output: Path, expected: Optional[str], label: str
) -> None:
    if output.exists():
        raise AssertionError("%s output already exists" % label)
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode == 0:
        raise AssertionError("%s unexpectedly succeeded" % label)
    if output.exists():
        raise AssertionError("%s left a partial output" % label)
    combined = process.stdout + process.stderr
    if expected is not None and expected not in combined:
        raise AssertionError(
            "%s did not report %r:\n%s" % (label, expected, combined)
        )


def verify_production_profiles(converter: Any, profiles_path: Path) -> None:
    profiles, _ = converter.load_profiles(profiles_path)
    if set(profiles) != set(PRODUCTION_IDENTITIES):
        raise AssertionError("production GENEB BERT profile IDs differ")
    for runtime_id, expected in PRODUCTION_IDENTITIES.items():
        profile = profiles[runtime_id]
        actual = (
            profile["repo"],
            profile["revision"],
            profile["source_format"],
            profile["config_sha256"],
            len(converter.source_tensor_specs(profile)[0]),
        )
        if actual != expected:
            raise AssertionError(
                "%s production identity/manifest differs: %r" % (runtime_id, actual)
            )
    if profiles["geneb-dnabert-s"]["source_omit_policy"] != "mosaic-pooler":
        raise AssertionError("DNABERT-S must use the intended remote Mosaic source")
    if profiles["geneb-grover"]["topology"]["vocab_size"] != 609:
        raise AssertionError("GROVER must retain the audited reachable 609-row vocab")
    if profiles["geneb-mutbert"]["topology"]["input_kind"] != "soft-vocabulary":
        raise AssertionError("MutBERT must retain its soft-vocabulary input contract")


def verify_artifact(
    output: Path, converter: Any, case: Mapping[str, Any]
) -> Dict[str, Any]:
    header, _ = read_safetensors(output)
    metadata = header.pop("__metadata__")
    expected_names = {spec.name for spec in case["runtime_specs"]}
    if set(header) != expected_names:
        raise AssertionError(
            "%s runtime tensor names differ: missing=%s extra=%s"
            % (
                case["kind"],
                sorted(expected_names - set(header)),
                sorted(set(header) - expected_names),
            )
        )
    if any(value["dtype"] == "I64" for value in header.values()):
        raise AssertionError("validation-only I64 leaked into runtime artifact")
    expected_metadata = {
        "runtime.profile": "s:geneb-bert-runtime-v1",
        "runtime.abi": "s:geneb-bert-safetensors-v1",
        "model.architecture": "s:GenebBertEncoder",
        "config.vocab_size": "u:%d" % case["profile"]["topology"]["vocab_size"],
        "runtime.embedding_layer_count": "u:%d"
        % (case["profile"]["topology"]["num_layers"] + 1),
        "tokenizer.profile": "s:evo-tokenizer-v1",
        "geneb.suite": "s:geneb-v4",
        "geneb.schema_version": "u:1",
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise AssertionError(
                "%s metadata %s differs: %r" % (case["kind"], key, metadata.get(key))
            )
    forbidden = {
        "compiler_manifest_sha256",
        "source_receipt_contract_sha256",
        "converter.schema",
        "converter.version",
    }
    if forbidden & set(metadata):
        raise AssertionError("tokenizer conversion receipt leaked into artifact metadata")
    if metadata.get("source.tensor_omit_policy") != (
        "s:" + case["profile"]["source_omit_policy"]
    ):
        raise AssertionError("source tensor omission policy is missing")
    return {"metadata": metadata, "tensors": header}


def native_verify(
    native_loader: Path,
    artifact: Path,
    success: bool,
    label: str,
    adapter: bool = False,
) -> None:
    process = subprocess.run(
        [
            str(native_loader),
            "--verify-adapter" if adapter else "--verify-artifact",
            str(artifact),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if (process.returncode == 0) != success:
        raise AssertionError(
            "%s native verification returned %d:\n%s%s"
            % (label, process.returncode, process.stdout, process.stderr)
        )


def mutate_artifact(
    source: Path, output: Path, mutation: str, tensor_name: Optional[str] = None
) -> None:
    header, data = read_safetensors(source)
    if mutation == "abi":
        header["__metadata__"]["runtime.abi"] = "s:wrong-abi"
    elif mutation == "common-config":
        header["__metadata__"]["config.vocab_size"] = "u:8"
    elif mutation == "activation-dtype":
        header["__metadata__"]["encoder.activation_dtype"] = "s:I64"
    elif mutation == "shape":
        if tensor_name is None:
            raise AssertionError("shape mutation needs tensor")
        shape = list(header[tensor_name]["shape"])
        shape[0], shape[1] = shape[1], shape[0]
        header[tensor_name]["shape"] = shape
    elif mutation == "missing":
        if tensor_name is None:
            raise AssertionError("missing mutation needs tensor")
        del header[tensor_name]
    else:
        raise AssertionError("unknown artifact mutation")
    rewrite_safetensors(output, header, data)


def test_source_corruptions(
    converter_path: Path,
    converter: Any,
    profiles: Path,
    catalog: Path,
    case: Mapping[str, Any],
) -> None:
    root = case["root"]
    weights = root / "model.safetensors"
    original = weights.read_bytes()
    specs = list(case["source_specs"])
    variants = []  # type: List[Tuple[str, List[Any], str]]
    variants.append(("missing", specs[:-1], "tensor manifest mismatch"))
    variants.append(
        (
            "extra",
            specs + [converter.TensorSpec("unexpected.weight", "F32", (1,))],
            "tensor manifest mismatch",
        )
    )
    wrong_shape = list(specs)
    wrong_shape[0] = converter.TensorSpec(
        wrong_shape[0].name,
        wrong_shape[0].dtype,
        (wrong_shape[0].shape[0] + 1,) + wrong_shape[0].shape[1:],
    )
    variants.append(("shape", wrong_shape, "wrong_shape"))
    wrong_dtype = list(specs)
    wrong_dtype[0] = converter.TensorSpec(
        wrong_dtype[0].name, "BF16", wrong_dtype[0].shape
    )
    variants.append(("dtype", wrong_dtype, "wrong_dtype"))
    for label, variant, expected in variants:
        write_safetensors(weights, variant)
        refresh_receipt(root)
        output = root / ("corrupt-%s.safetensors" % label)
        require_failure(
            converter_command(converter_path, profiles, catalog, case, output),
            output,
            expected,
            "source tensor %s corruption" % label,
        )
    weights.write_bytes(original)
    refresh_receipt(root)


def test_config_and_receipt_corruption(
    converter_path: Path,
    profiles: Path,
    catalog: Path,
    case: Mapping[str, Any],
) -> None:
    root = case["root"]
    config = root / "config.json"
    original_config = config.read_bytes()
    parsed = json.loads(original_config.decode("utf-8"))
    parsed["hidden_size"] += 1
    write_json(config, parsed)
    output = root / "bad-receipt-hash.safetensors"
    require_failure(
        converter_command(converter_path, profiles, catalog, case, output),
        output,
        "source receipt integrity mismatch",
        "stale source receipt",
    )
    refresh_receipt(root)
    output = root / "bad-config-hash.safetensors"
    require_failure(
        converter_command(converter_path, profiles, catalog, case, output),
        output,
        "HF config SHA256 differs",
        "pinned config hash",
    )
    config.write_bytes(original_config)
    refresh_receipt(root)

    receipt_path = case["receipt"]
    original_receipt = receipt_path.read_bytes()
    receipt = json.loads(original_receipt.decode("utf-8"))
    receipt["files"] = receipt["files"][:-1]
    write_json(receipt_path, receipt)
    output = root / "missing-receipt-asset.safetensors"
    require_failure(
        converter_command(converter_path, profiles, catalog, case, output),
        output,
        "HF receipt is missing model.safetensors",
        "missing receipt asset",
    )
    receipt_path.write_bytes(original_receipt)


def test_shard_index_corruption(
    converter_path: Path,
    profiles: Path,
    catalog: Path,
    case: Mapping[str, Any],
) -> None:
    root = case["root"]
    index_path = root / "model.safetensors.index.json"
    original = index_path.read_bytes()
    index = json.loads(original.decode("utf-8"))
    first = sorted(index["weight_map"])[0]
    current = index["weight_map"][first]
    index["weight_map"][first] = (
        "model-00002-of-00002.safetensors"
        if current == "model-00001-of-00002.safetensors"
        else "model-00001-of-00002.safetensors"
    )
    write_json(index_path, index)
    refresh_receipt(root)
    output = root / "bad-shard-owner.safetensors"
    require_failure(
        converter_command(converter_path, profiles, catalog, case, output),
        output,
        "index differs from shards",
        "HF shard owner corruption",
    )
    index_path.write_bytes(original)
    refresh_receipt(root)


def test_i64_opt_in(converter: Any, work_dir: Path) -> None:
    source = work_dir / "i64.safetensors"
    write_safetensors(source, [converter.TensorSpec("position_ids", "I64", (1, 8))])
    try:
        converter.read_hf_safetensors({"model.safetensors": source})
    except converter.CheckpointError as error:
        if "outside allowed set" not in str(error):
            raise
    else:
        raise AssertionError("shared HF reader accepted I64 without explicit opt-in")
    tensors = converter.read_hf_safetensors(
        {"model.safetensors": source}, allowed_dtypes=("F32", "BF16", "I64")
    )
    if len(tensors) != 1 or tensors[0].dtype != "I64":
        raise AssertionError("explicit validation-only I64 reader path differs")


def test_descriptor_and_profile_corruption(
    converter_path: Path,
    profiles: Path,
    catalog: Path,
    case: Mapping[str, Any],
) -> None:
    descriptor_path = case["descriptor"]
    original_descriptor = descriptor_path.read_bytes()
    descriptor = json.loads(original_descriptor.decode("utf-8"))
    descriptor["tokenizer.sha256"] = "0" * 64
    write_json(descriptor_path, descriptor)
    output = case["root"] / "bad-tokenizer-descriptor.safetensors"
    require_failure(
        converter_command(converter_path, profiles, catalog, case, output),
        output,
        "tokenizer asset integrity mismatch",
        "tokenizer descriptor corruption",
    )
    descriptor_path.write_bytes(original_descriptor)

    manifest = json.loads(profiles.read_text(encoding="utf-8"))
    manifest["models"][0]["unexpected"] = True
    bad_profiles = case["root"] / "bad-profiles.json"
    write_json(bad_profiles, manifest)
    output = case["root"] / "bad-profile-field.safetensors"
    require_failure(
        converter_command(converter_path, bad_profiles, catalog, case, output),
        output,
        "fields differ",
        "unknown profile field",
    )


def test_optional_torch_path(
    converter_path: Path,
    converter: Any,
    catalog: Path,
    case: Mapping[str, Any],
    work_dir: Path,
) -> None:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore
    root = work_dir / "optional-torch"
    root.mkdir()
    profile = copy.deepcopy(case["profile"])
    profile["source_format"] = "pytorch-bin"
    profiles = root / "profiles.json"
    write_json(
        profiles,
        {
            "schema_version": 1,
            "format": "geneb-bert-converter-v1",
            "models": [profile],
        },
    )
    source_root = root / "source"
    source_root.mkdir()
    config_payload = canonical_bytes(config_for(case["kind"], profile["topology"]))
    (source_root / "config.json").write_bytes(config_payload)
    source_specs, runtime_specs = converter.source_tensor_specs(profile)
    if torch is None:
        (source_root / "pytorch_model.bin").write_bytes(b"verified-offline-fixture")
    else:
        state = {}
        for spec in source_specs:
            dtype = torch.int64 if spec.dtype == "I64" else torch.float32
            state[spec.name] = torch.zeros(spec.shape, dtype=dtype)
        torch.save(state, source_root / "pytorch_model.bin")
    receipt = write_receipt(
        source_root, profile, ["config.json", "pytorch_model.bin"]
    )
    # Real Hugging Face cache receipts preserve the logical filename in
    # `name` while `path` resolves to a content-addressed blob.  The strict
    # mapping must use the verified logical name rather than the blob's
    # physical basename.
    blob_root = source_root / "blobs"
    blob_root.mkdir()
    blob_path = blob_root / ("a" * 64)
    (source_root / "pytorch_model.bin").rename(blob_path)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    for item in receipt_payload["files"]:
        if item["name"] == "pytorch_model.bin":
            item["path"] = str(blob_path.resolve())
    write_json(receipt, receipt_payload)
    descriptor = write_tokenizer_descriptor(source_root)
    torch_case = {
        "kind": case["kind"],
        "root": source_root,
        "profile": profile,
        "runtime_specs": runtime_specs,
        "receipt": receipt,
        "descriptor": descriptor,
    }
    output = source_root / "runtime.safetensors"
    if torch is None:
        require_failure(
            converter_command(converter_path, profiles, catalog, torch_case, output),
            output,
            "requires offline PyTorch",
            "actionable missing PyTorch gate",
        )
        return
    require_success(
        converter_command(converter_path, profiles, catalog, torch_case, output),
        "optional verified pytorch_model.bin conversion",
    )
    verify_artifact(output, converter, torch_case)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--native-loader", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    converter_path = args.converter.resolve()
    sys.path.insert(0, str(converter_path.parent))
    spec = importlib.util.spec_from_file_location("geneb_bert_converter", converter_path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import GENEB BERT converter")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    verify_production_profiles(converter, args.profiles.resolve())

    work_dir = args.work_dir.resolve() / "bert-converter-fixture"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    profiles, catalog, cases = make_cases(work_dir, converter)
    outputs = {}  # type: Dict[str, Path]
    for case in cases:
        output_a = case["root"] / "runtime-a.safetensors"
        output_b = case["root"] / "runtime-b.safetensors"
        require_success(
            converter_command(converter_path, profiles, catalog, case, output_a),
            "%s conversion A" % case["kind"],
        )
        require_success(
            converter_command(converter_path, profiles, catalog, case, output_b),
            "%s conversion B" % case["kind"],
        )
        if output_a.read_bytes() != output_b.read_bytes():
            raise AssertionError("%s converter output is not deterministic" % case["kind"])
        verify_artifact(output_a, converter, case)
        outputs[case["kind"]] = output_a
        if args.native_loader is not None:
            native_verify(
                args.native_loader.resolve(), output_a, True, "%s artifact" % case["kind"]
            )
            native_verify(
                args.native_loader.resolve(),
                output_a,
                True,
                "%s CPU adapter" % case["kind"],
                adapter=True,
            )

    by_kind = {case["kind"]: case for case in cases}
    test_i64_opt_in(converter, work_dir)
    test_source_corruptions(
        converter_path, converter, profiles, catalog, by_kind["grover"]
    )
    test_config_and_receipt_corruption(
        converter_path, profiles, catalog, by_kind["gena-no-final"]
    )
    test_shard_index_corruption(
        converter_path, profiles, catalog, by_kind["dnabert-2"]
    )
    test_descriptor_and_profile_corruption(
        converter_path, profiles, catalog, by_kind["gena-final"]
    )
    test_optional_torch_path(
        converter_path,
        converter,
        catalog,
        by_kind["gena-no-final"],
        work_dir,
    )

    if args.native_loader is not None:
        base = outputs["grover"]
        first_tensor = by_kind["grover"]["runtime_specs"][0].name
        for mutation in ("abi", "common-config", "activation-dtype", "shape", "missing"):
            corrupt = work_dir / ("native-%s.safetensors" % mutation)
            mutate_artifact(
                base,
                corrupt,
                mutation,
                first_tensor if mutation in ("shape", "missing") else None,
            )
            native_verify(
                args.native_loader.resolve(),
                corrupt,
                False,
                "native %s corruption" % mutation,
            )

    grammar_paths = (
        converter_path,
        converter_path.parent / "evo" / "hf_checkpoint.py",
        converter_path.parent / "evo" / "geneb_artifact.py",
        Path(__file__).resolve(),
    )
    for path in grammar_paths:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read(), "
                "filename=sys.argv[1], feature_version=(3,8))",
                str(path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            raise AssertionError("Python 3.8 grammar failed for %s: %s" % (path, process.stderr))

    print("GENEB BERT converter contract passed for seven offline model profiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
