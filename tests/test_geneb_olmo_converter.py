#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict converter, corruption, and native-loader closure for OLMo-GFM."""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> float:
    integer = ((tensor_index + 2) * 19 + (element_index + 5) * 11) % 43 - 21
    value = f32(f32(float(integer)) / f32(31.0))
    if "norm.weight" in name or name == "model.transformer.ln_f.weight":
        value = f32(f32(1.0) + f32(value * f32(0.04)))
    elif name == "model.transformer.wte.weight":
        value = f32(value * f32(0.3))
    else:
        value = f32(value * f32(0.12))
    return value


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_bytes(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (bits >> 16) & 0xFFFF)


def tensor_payload(name: str, tensor_index: int, count: int, dtype: str) -> bytes:
    output = bytearray()
    for element_index in range(count):
        value = f32(fixture_scalar(name, tensor_index, element_index))
        output.extend(struct.pack("<f", value) if dtype == "F32" else bf16_bytes(value))
    return bytes(output)


def write_safetensors(
    path: Path,
    specs: Sequence[Any],
    omit: Optional[str] = None,
    add_extra: bool = False,
    wrong_shape: Optional[str] = None,
    wrong_dtype: Optional[str] = None,
) -> Dict[str, bytes]:
    root = {"__metadata__": {"format": "pt"}}  # type: Dict[str, Any]
    payloads = {}  # type: Dict[str, bytes]
    offset = 0
    for tensor_index, spec in enumerate(specs):
        if spec.name == omit:
            continue
        shape = list(spec.shape)
        if spec.name == wrong_shape:
            shape[0] += 1
        dtype = "BF16" if spec.name == wrong_dtype else spec.dtype
        count = 1
        for dimension in shape:
            count *= dimension
        payload = tensor_payload(spec.name, tensor_index, count, dtype)
        root[spec.name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads[spec.name] = payload
        offset += len(payload)
    if add_extra:
        name = "unexpected.weight"
        payload = struct.pack("<f", 0.25)
        root[name] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads[name] = payload
        offset += len(payload)
    raw = json.dumps(root, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(header))
        + header
        + b"".join(payloads[name] for name in root if name != "__metadata__")
    )
    return payloads


def config_value(norm_type: str) -> Dict[str, Any]:
    return {
        "activation_type": "swiglu",
        "alibi": False,
        "attention_dropout": 0.0,
        "attention_layer_norm": False,
        "attention_layer_norm_with_affine": False,
        "bias_for_layer_norm": False,
        "block_group_size": 1,
        "block_type": "sequential",
        "clip_qkv": None,
        "d_model": 4,
        "embedding_dropout": 0.0,
        "embedding_layer_norm": False,
        "embedding_size": 7,
        "flash_attention": False,
        "include_bias": False,
        "layer_norm_eps": 0.00001,
        "layer_norm_type": "default" if norm_type == "layernorm" else "rms",
        "layer_norm_with_affine": norm_type == "rmsnorm",
        "max_sequence_length": 8,
        "mlp_hidden_size": None,
        "mlp_ratio": 3,
        "model_type": "olmo-gfm",
        "multi_query_attention": False,
        "n_heads": 2,
        "n_kv_heads": None,
        "n_layers": 2,
        "norm_after": False,
        "residual_dropout": 0.0,
        "rope": True,
        "rope_full_precision": True,
        "rope_theta": 10000,
        "scale_logits": False,
        "vocab_size": 7,
        "weight_tying": True,
    }


def topology_value(norm_type: str) -> Dict[str, Any]:
    return {
        "vocab_size": 7,
        "hidden_size": 4,
        "num_layers": 2,
        "num_attention_heads": 2,
        "fused_mlp_width": 12,
        "max_seqlen": 8,
        "norm_epsilon": 0.00001,
        "rope_theta": 10000.0,
        "norm_type": (
            "layernorm-no-affine" if norm_type == "layernorm" else "rmsnorm-affine"
        ),
    }


def tokenizer_value() -> Dict[str, Any]:
    return {
        "format": "evo-tokenizer-v1",
        "kind": "character",
        "normalization": [],
        "pre_tokenizer": {"kind": "none"},
        "model": {"unknown_policy": "unk", "match_special_literals": False},
        "post_processor": {
            "prefix_ids": [],
            "suffix_ids": [],
            "padding": {"side": "right", "pad_id": 6},
        },
        "special_tokens": {
            "unk": 0,
            "pad": 6,
            "bos": None,
            "eos": None,
            "cls": None,
            "sep": None,
            "mask": None,
        },
        "vocab": [
            {"id": 0, "piece": "[UNK]"},
            {"id": 1, "piece": "A"},
            {"id": 2, "piece": "G"},
            {"id": 3, "piece": "N"},
            {"id": 4, "piece": "C"},
            {"id": 5, "piece": "U"},
            {"id": 6, "piece": "[PAD]"},
        ],
    }


def receipt_entry(path: Path) -> Dict[str, Any]:
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": digest(path),
        "path": str(path.resolve()),
    }


def refresh_receipt(case: Mapping[str, Any]) -> None:
    receipt = json.loads(case["receipt"].read_text(encoding="utf-8"))
    receipt["files"] = [receipt_entry(path) for path in case["source_files"]]
    write_json(case["receipt"], receipt)


def refresh_profile_weights(case: Mapping[str, Any]) -> None:
    profiles = json.loads(case["profiles"].read_text(encoding="utf-8"))
    profile = profiles["models"][0]
    profile["weights_size"] = case["weights"].stat().st_size
    profile["weights_sha256"] = digest(case["weights"])
    write_json(case["profiles"], profiles)


def make_case(
    root: Path,
    converter: Any,
    norm_type: str,
    omit: Optional[str] = None,
    add_extra: bool = False,
    wrong_shape: Optional[str] = None,
    wrong_dtype: Optional[str] = None,
) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    runtime_id = "geneb-olmo-tiny-" + norm_type
    repo = "fixture/OlmoTiny"
    revision = "a" * 40
    config = root / "config.json"
    config_document = config_value(norm_type)
    write_json(config, config_document)
    topology = topology_value(norm_type)
    specs = converter.canonical_tensor_specs(topology)
    weights = root / "model.safetensors"
    payloads = write_safetensors(
        weights,
        specs,
        omit=omit,
        add_extra=add_extra,
        wrong_shape=wrong_shape,
        wrong_dtype=wrong_dtype,
    )
    tokenizer_payload = canonical_json(tokenizer_value())
    profile = {
        "runtime_id": runtime_id,
        "geneb_model_id": "OlmoTiny-" + norm_type,
        "paper_name": "OlmoTiny-" + norm_type,
        "catalog_architecture": "olmo-causal-lm",
        "repo": repo,
        "revision": revision,
        "config_sha256": digest(config),
        "weights_sha256": digest(weights),
        "weights_size": weights.stat().st_size,
        "tokenizer_manifest_sha256": "b" * 64,
        "tokenizer_source_receipt_contract_sha256": "c" * 64,
        "tokenizer_asset_sha256": hashlib.sha256(tokenizer_payload).hexdigest(),
        "tokenizer_asset_size": len(tokenizer_payload),
        "code_provenance": {
            "package": converter.OLMO_PACKAGE,
            "version": converter.OLMO_VERSION,
            "repo": converter.OLMO_REPO,
            "revision": converter.OLMO_REVISION,
            "model_py_sha256": converter.OLMO_MODEL_SHA256,
            "config_py_sha256": converter.OLMO_CONFIG_SHA256,
        },
        "remote_code": {
            "modeling_olmo_sha256": converter.REMOTE_MODELING_SHA256,
            "configuration_olmo_sha256": converter.REMOTE_CONFIGURATION_SHA256,
        },
        "config_required": config_document,
        "config_defaults": {},
        "topology": topology,
    }
    profiles = root / "profiles.json"
    write_json(
        profiles,
        {
            "schema_version": 1,
            "format": converter.PROFILE_FORMAT,
            "models": [profile],
        },
    )
    catalog = root / "catalog.json"
    write_json(
        catalog,
        {
            "schema_version": 1,
            "suite": {"id": "geneb-v4", "raw_safety_cap_bytes": 16777216},
            "models": [
                {
                    "runtime_id": runtime_id,
                    "geneb_model_id": profile["geneb_model_id"],
                    "paper_name": profile["paper_name"],
                    "family": "transformer-decoder",
                    "architecture": "olmo-causal-lm",
                    "source": {
                        "kind": "huggingface",
                        "repo": repo,
                        "requested_revision": "main",
                        "revision": revision,
                        "immutable": True,
                    },
                    "context": {
                        "unit": "tokens",
                        "declared_max_tokens": 8,
                        "reference_max_tokens": 8,
                        "length_policy": "tokenizer-truncate",
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
                        "special_tokens": "tokenizer-default",
                        "token_truncation": "right",
                    },
                    "embedding_presets": {
                        "reference": {
                            "hidden_tap": "post-final-norm",
                            "pooling": "masked-mean",
                            "special_tokens": "include",
                            "mask_domain": "attention-mask",
                            "output_width": 4,
                        },
                        "normalized": {
                            "hidden_tap": "post-final-norm",
                            "pooling": "masked-mean",
                            "special_tokens": "include",
                            "mask_domain": "attention-mask",
                            "output_width": 4,
                        },
                    },
                    "provenance": {
                        "extractor": {"commit": "b" * 40},
                        "reference_patch": {"sha256": "c" * 64},
                        "normalization_patch_sha256": "d" * 64,
                    },
                }
            ],
        },
    )
    receipt = root / "source-receipt.json"
    source_files = [config, weights]
    write_json(
        receipt,
        {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": runtime_id,
            "repo": repo,
            "requested_revision": "main",
            "resolved_revision": revision,
            "files": [receipt_entry(path) for path in source_files],
            "load_path": None,
        },
    )
    tokenizer = root / "assets" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.write_bytes(tokenizer_payload)
    tokenizer_descriptor = root / "tokenizer-descriptor.json"
    write_json(
        tokenizer_descriptor,
        {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": "b" * 64,
            "source_receipt_contract_sha256": "c" * 64,
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": "assets/tokenizer.json",
            "tokenizer.sha256": digest(tokenizer),
            "tokenizer.size": tokenizer.stat().st_size,
        },
    )
    return {
        "root": root,
        "profiles": profiles,
        "catalog": catalog,
        "receipt": receipt,
        "tokenizer": tokenizer,
        "tokenizer_descriptor": tokenizer_descriptor,
        "config": config,
        "weights": weights,
        "source_files": source_files,
        "output": root / "runtime.safetensors",
        "payloads": payloads,
        "specs": specs,
        "norm_type": norm_type,
    }


def converter_command(args: argparse.Namespace, case: Mapping[str, Any]) -> List[str]:
    return [
        sys.executable,
        str(args.converter),
        "--receipt",
        str(case["receipt"]),
        "--catalog",
        str(case["catalog"]),
        "--profiles",
        str(case["profiles"]),
        "--tokenizer-descriptor",
        str(case["tokenizer_descriptor"]),
        "--tokenizer-root",
        str(case["root"]),
        "--output",
        str(case["output"]),
    ]


def convert(args: argparse.Namespace, case: Mapping[str, Any]) -> subprocess.CompletedProcess:
    return subprocess.run(
        converter_command(args, case),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def expect_failure(
    args: argparse.Namespace, case: Mapping[str, Any], needle: str
) -> None:
    result = convert(args, case)
    if result.returncode == 0 or needle not in result.stderr:
        raise AssertionError(
            "expected converter failure containing %r:\n%s\n%s"
            % (needle, result.stdout, result.stderr)
        )
    if case["output"].exists():
        raise AssertionError("failed conversion published a partial artifact")


def read_artifact(path: Path) -> Tuple[Dict[str, Any], Dict[str, bytes], Dict[str, Any]]:
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        data = source.read()
    metadata = header.pop("__metadata__")
    tensors = {
        name: data[value["data_offsets"][0] : value["data_offsets"][1]]
        for name, value in header.items()
    }
    return metadata, tensors, header


def rewrite_artifact(
    source_path: Path,
    destination: Path,
    metadata_mutation: Optional[Any] = None,
    tensor_mutation: Optional[Any] = None,
) -> None:
    with source_path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        data = source.read()
    metadata = header["__metadata__"]
    if metadata_mutation is not None:
        metadata_mutation(metadata)
    if tensor_mutation is not None:
        tensor_mutation(header)
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    padded = raw + b" " * ((-len(raw)) % 8)
    destination.write_bytes(struct.pack("<Q", len(padded)) + padded + data)


def run_native(loader: Path, artifact: Path, expect_success: bool) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [str(loader), "--verify-artifact", str(artifact)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError("native artifact load failed:\n" + result.stderr)
    if not expect_success and result.returncode == 0:
        raise AssertionError("native artifact corruption was accepted")
    return result


def validate_success(
    args: argparse.Namespace, case: Mapping[str, Any], converter: Any
) -> None:
    result = convert(args, case)
    if result.returncode != 0:
        raise AssertionError("valid conversion failed:\n" + result.stdout + result.stderr)
    metadata, tensors, header = read_artifact(case["output"])
    if metadata.get("runtime.profile") != "s:" + converter.ARTIFACT_PROFILE:
        raise AssertionError("runtime profile metadata differs")
    expected_metadata = {
        "runtime.abi": "s:" + converter.RUNTIME_ABI,
        "runtime.embedding_layer_count": "u:3",
        "model.architecture": "s:" + converter.RUNTIME_ARCHITECTURE,
        "config.vocab_size": "u:7",
        "config.hidden_size": "u:4",
        "config.num_layers": "u:2",
        "config.max_seqlen": "u:8",
        "olmo.qkv_layout": "s:q-k-v",
        "olmo.swiglu_layout": "s:x-gate",
        "olmo.weight_dtype": "s:F32",
        "source.olmo.version": "s:0.6.0",
        "source.olmo.revision": "s:" + converter.OLMO_REVISION,
        "source.olmo.model_py_sha256": "s:" + converter.OLMO_MODEL_SHA256,
        "source.olmo.config_py_sha256": "s:" + converter.OLMO_CONFIG_SHA256,
        "source.remote_modeling_sha256": "s:" + converter.REMOTE_MODELING_SHA256,
        "source.remote_configuration_sha256": "s:"
        + converter.REMOTE_CONFIGURATION_SHA256,
        "tokenizer.profile": "s:evo-tokenizer-v1",
        "tokenizer.path": "s:assets/tokenizer.json",
    }
    wrong = {
        key: (metadata.get(key), value)
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if wrong:
        raise AssertionError("artifact metadata differs: %s" % wrong)
    if any(key.startswith("converter.") for key in metadata):
        raise AssertionError("tokenizer conversion receipt leaked into model metadata")
    if "olmo.layer_norm_kernel" in metadata:
        raise AssertionError("exact Omni-DNA-1B LayerNorm kernel escaped to tiny profile")
    expected_names = [spec.name for spec in case["specs"]]
    if list(tensors) != sorted(expected_names):
        # JSON headers are written with sort_keys=True, so native tensor order is lexical.
        raise AssertionError("runtime tensor names differ")
    if set(tensors) != set(case["payloads"]):
        raise AssertionError("runtime tensor set differs from verified source")
    for name, payload in case["payloads"].items():
        if tensors[name] != payload:
            raise AssertionError("runtime tensor %s bytes changed" % name)
    if set(header) != set(tensors):
        raise AssertionError("runtime tensor header differs")
    native = run_native(args.native_loader, case["output"], True)
    document = json.loads(native.stdout)
    if document.get("rows") != 4 or document.get("width") != 4:
        raise AssertionError("native converter closure returned wrong shape")
    typed = subprocess.run(
        [str(args.native_loader), "--dump-json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if typed.returncode != 0:
        raise AssertionError("native typed fixture failed:\n" + typed.stderr)
    typed_vectors = json.loads(typed.stdout).get("vectors")
    artifact_vectors = document.get("vectors")
    if not isinstance(typed_vectors, dict) or not isinstance(artifact_vectors, dict):
        raise AssertionError("native vector closure returned malformed JSON")
    expected_vectors = {
        "artifact." + name.split(".", 1)[1]: values
        for name, values in typed_vectors.items()
        if name.startswith(case["norm_type"] + ".")
    }
    if artifact_vectors != expected_vectors:
        raise AssertionError("converter artifact differs from the typed oracle fixture")
    adapter = subprocess.run(
        [str(args.native_loader), "--verify-cpu-adapter", str(case["output"])],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if adapter.returncode != 0:
        raise AssertionError("CPU Model/Context adapter closure failed:\n" + adapter.stderr)
    adapter_document = json.loads(adapter.stdout)
    if adapter_document != {"tokens": [1, 4, 2], "rows": 3, "width": 4}:
        raise AssertionError("CPU Model tokenizer/embedding adapter result differs")


def test_native_corruption(args: argparse.Namespace, case: Mapping[str, Any]) -> None:
    corrupt = case["root"] / "bad-abi.safetensors"
    rewrite_artifact(
        case["output"],
        corrupt,
        metadata_mutation=lambda metadata: metadata.__setitem__(
            "runtime.abi", "s:geneb-olmo-safetensors-v2"
        ),
    )
    run_native(args.native_loader, corrupt, False)

    corrupt = case["root"] / "bad-layer-count.safetensors"
    rewrite_artifact(
        case["output"],
        corrupt,
        metadata_mutation=lambda metadata: metadata.__setitem__(
            "runtime.embedding_layer_count", "u:4"
        ),
    )
    run_native(args.native_loader, corrupt, False)

    corrupt = case["root"] / "bad-remote-code.safetensors"
    rewrite_artifact(
        case["output"],
        corrupt,
        metadata_mutation=lambda metadata: metadata.__setitem__(
            "source.remote_modeling_sha256", "s:" + "0" * 64
        ),
    )
    run_native(args.native_loader, corrupt, False)

    corrupt = case["root"] / "partial-tokenizer.safetensors"
    rewrite_artifact(
        case["output"],
        corrupt,
        metadata_mutation=lambda metadata: metadata.pop("tokenizer.sha256"),
    )
    run_native(args.native_loader, corrupt, False)

    corrupt = case["root"] / "missing-omni1b-layer-norm-kernel.safetensors"
    rewrite_artifact(
        case["output"],
        corrupt,
        metadata_mutation=lambda metadata: metadata.__setitem__(
            "geneb.runtime_id", "s:geneb-omni-dna-1b"
        ),
    )
    run_native(args.native_loader, corrupt, False)

    def select_exact_kernel(metadata: Dict[str, Any]) -> None:
        metadata["geneb.runtime_id"] = "s:geneb-omni-dna-1b"
        metadata["olmo.layer_norm_kernel"] = (
            "s:" + "torch-2.1.2-apple-arm64-exact-v1"
        )

    corrupt = case["root"] / "exact-layer-norm-wrong-topology.safetensors"
    rewrite_artifact(
        case["output"], corrupt, metadata_mutation=select_exact_kernel
    )
    run_native(args.native_loader, corrupt, False)

    def select_unknown_kernel(metadata: Dict[str, Any]) -> None:
        metadata["geneb.runtime_id"] = "s:geneb-omni-dna-1b"
        metadata["olmo.layer_norm_kernel"] = "s:not-a-kernel"

    corrupt = case["root"] / "unknown-layer-norm-kernel.safetensors"
    rewrite_artifact(
        case["output"], corrupt, metadata_mutation=select_unknown_kernel
    )
    run_native(args.native_loader, corrupt, False)

    first_name = case["specs"][0].name
    corrupt = case["root"] / "renamed-tensor.safetensors"

    def rename_tensor(header: Dict[str, Any]) -> None:
        header["unexpected.embedding"] = header.pop(first_name)

    rewrite_artifact(case["output"], corrupt, tensor_mutation=rename_tensor)
    run_native(args.native_loader, corrupt, False)


def test_corruptions(args: argparse.Namespace, converter: Any) -> None:
    case = make_case(args.work_dir / "bad-requested-revision", converter, "layernorm")
    receipt = json.loads(case["receipt"].read_text(encoding="utf-8"))
    receipt["requested_revision"] = receipt["resolved_revision"]
    write_json(case["receipt"], receipt)
    expect_failure(args, case, "model/repository/revision is not pinned")

    case = make_case(args.work_dir / "bad-resolved-revision", converter, "layernorm")
    receipt = json.loads(case["receipt"].read_text(encoding="utf-8"))
    receipt["resolved_revision"] = "b" * 40
    write_json(case["receipt"], receipt)
    expect_failure(args, case, "model/repository/revision is not pinned")

    case = make_case(args.work_dir / "bad-receipt", converter, "layernorm")
    with case["weights"].open("ab") as output:
        output.write(b"X")
    expect_failure(args, case, "source receipt integrity mismatch")

    case = make_case(args.work_dir / "bad-pinned-weight", converter, "layernorm")
    with case["weights"].open("ab") as output:
        output.write(b"X")
    refresh_receipt(case)
    expect_failure(args, case, "differ from the pinned profile")

    base = topology_value("layernorm")
    first = converter.canonical_tensor_specs(base)[0].name
    case = make_case(args.work_dir / "missing-tensor", converter, "layernorm", omit=first)
    refresh_receipt(case)
    refresh_profile_weights(case)
    expect_failure(args, case, "tensor manifest mismatch")

    case = make_case(args.work_dir / "extra-tensor", converter, "layernorm", add_extra=True)
    refresh_receipt(case)
    refresh_profile_weights(case)
    expect_failure(args, case, "tensor manifest mismatch")

    case = make_case(
        args.work_dir / "wrong-shape", converter, "layernorm", wrong_shape=first
    )
    refresh_receipt(case)
    refresh_profile_weights(case)
    expect_failure(args, case, "tensor manifest mismatch")

    case = make_case(
        args.work_dir / "wrong-dtype", converter, "layernorm", wrong_dtype=first
    )
    refresh_receipt(case)
    refresh_profile_weights(case)
    expect_failure(args, case, "tensor manifest mismatch")

    case = make_case(args.work_dir / "bad-config", converter, "layernorm")
    config = json.loads(case["config"].read_text(encoding="utf-8"))
    config["block_type"] = "llama"
    write_json(case["config"], config)
    refresh_receipt(case)
    profiles = json.loads(case["profiles"].read_text(encoding="utf-8"))
    profiles["models"][0]["config_sha256"] = digest(case["config"])
    profiles["models"][0]["config_required"]["block_type"] = "llama"
    write_json(case["profiles"], profiles)
    expect_failure(args, case, "closed OLMo-GFM ABI")

    case = make_case(args.work_dir / "bad-tokenizer", converter, "layernorm")
    with case["tokenizer"].open("ab") as output:
        output.write(b"X")
    expect_failure(args, case, "tokenizer asset integrity mismatch")

    case = make_case(args.work_dir / "bad-descriptor", converter, "layernorm")
    descriptor = json.loads(case["tokenizer_descriptor"].read_text(encoding="utf-8"))
    descriptor["tokenizer.extra"] = "forbidden"
    write_json(case["tokenizer_descriptor"], descriptor)
    expect_failure(args, case, "tokenizer descriptor fields differ")

    case = make_case(args.work_dir / "mismatched-tokenizer-pin", converter, "layernorm")
    descriptor = json.loads(case["tokenizer_descriptor"].read_text(encoding="utf-8"))
    descriptor["compiler_manifest_sha256"] = "d" * 64
    write_json(case["tokenizer_descriptor"], descriptor)
    expect_failure(args, case, "tokenizer descriptor differs from pinned profile")

    case = make_case(args.work_dir / "partial-tokenizer-pin", converter, "layernorm")
    profiles = json.loads(case["profiles"].read_text(encoding="utf-8"))
    del profiles["models"][0]["tokenizer_asset_size"]
    write_json(case["profiles"], profiles)
    expect_failure(args, case, "tokenizer profile pins must appear together")

    case = make_case(args.work_dir / "bad-catalog", converter, "layernorm")
    catalog = json.loads(case["catalog"].read_text(encoding="utf-8"))
    catalog["models"][0]["embedding_presets"]["reference"]["output_width"] = 5
    write_json(case["catalog"], catalog)
    expect_failure(args, case, "context/embedding width differs")

    case = make_case(args.work_dir / "bad-remote-code-profile", converter, "layernorm")
    profiles = json.loads(case["profiles"].read_text(encoding="utf-8"))
    profiles["models"][0]["remote_code"]["modeling_olmo_sha256"] = "0" * 64
    write_json(case["profiles"], profiles)
    expect_failure(args, case, "pinned model-repo wrapper code")

    case = make_case(args.work_dir / "duplicate-profile-key", converter, "layernorm")
    payload = case["profiles"].read_text(encoding="utf-8")
    payload = payload.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    case["profiles"].write_text(payload, encoding="utf-8")
    expect_failure(args, case, "duplicate key")


def validate_production_profiles(converter: Any, path: Path) -> None:
    profiles, _ = converter.load_profiles(path.resolve())
    if set(profiles) != {"geneb-omni-dna-1b", "geneb-omni-dna-300m"}:
        raise AssertionError("production OLMo profile IDs differ")
    one_b = profiles["geneb-omni-dna-1b"]
    three_hundred = profiles["geneb-omni-dna-300m"]
    if one_b["topology"]["norm_type"] != "layernorm-no-affine":
        raise AssertionError("1B must use non-affine standard LayerNorm")
    if three_hundred["topology"]["norm_type"] != "rmsnorm-affine":
        raise AssertionError("300M must use affine RMSNorm")
    exact_kernel_profiles = {
        runtime_id
        for runtime_id, profile in profiles.items()
        if "layer_norm_kernel" in profile
    }
    if (
        exact_kernel_profiles != {"geneb-omni-dna-1b"}
        or one_b["layer_norm_kernel"] != converter.OMNI1B_LAYER_NORM_KERNEL
    ):
        raise AssertionError("1B exact Apple-arm64 LayerNorm selector drifted")
    if len(converter.canonical_tensor_specs(one_b["topology"])) != 65:
        raise AssertionError("1B production tensor manifest must contain 65 tensors")
    if len(converter.canonical_tensor_specs(three_hundred["topology"])) != 98:
        raise AssertionError("300M production tensor manifest must contain 98 tensors")
    expected_tokenizer = {
        "tokenizer_manifest_sha256": "a43c15ea23faa74779065971045d916c3596bce011b29a6d801cfb7e83c231b7",
        "tokenizer_source_receipt_contract_sha256": "a537d3f1a40725b11e2dbe350cb7febb980304ac48c92cacb8fec0bfda5f1250",
        "tokenizer_asset_sha256": "371c4385d340f46222e845abde5343768fcac26d432a9d97e369ad96bd2aa781",
        "tokenizer_asset_size": 182189,
    }
    if any(one_b.get(key) != value for key, value in expected_tokenizer.items()):
        raise AssertionError("1B tokenizer profile pins differ")
    if any(key in three_hundred for key in expected_tokenizer):
        raise AssertionError("1B tokenizer profile pins escaped to 300M")
    one_b_metadata = converter.build_metadata(
        one_b, {}, "a" * 64, "b" * 64, "c" * 64, "d" * 64, {}
    )
    three_hundred_metadata = converter.build_metadata(
        three_hundred, {}, "a" * 64, "b" * 64, "c" * 64, "d" * 64, {}
    )
    if one_b_metadata.get("olmo.layer_norm_kernel") != (
        converter.OMNI1B_LAYER_NORM_KERNEL
    ) or "olmo.layer_norm_kernel" in three_hundred_metadata:
        raise AssertionError("exact LayerNorm artifact metadata selection drifted")
    manifest = path.parent / "tokenizers" / "geneb-omni-dna-1b-bpe-v1.json"
    if (
        digest(manifest) != expected_tokenizer["tokenizer_manifest_sha256"]
        or manifest.stat().st_size != 515
    ):
        raise AssertionError("1B production tokenizer manifest differs")
    expected_sources = {
        "geneb-omni-dna-1b": (
            "zehui127/Omni-DNA-1B",
            "0ea9d54e356b4e7354dc40e2d980e9ebcb2ccfd5",
            "41560210c06cb75cec5f38adcd32d901829868a1fe5ade35d6dc94b3087a8ac8",
            4328529472,
            "12cba5f3b0c533801c7e05ee3ac93bf33f1557ac36cd11099e2e5ca1ca5ff5ad",
        ),
        "geneb-omni-dna-300m": (
            "zehui127/Omni-DNA-300M",
            "23587a0177a1f00d18b7079a2e81150b80e63f1c",
            "d0745a00ba32645e8f8294bdc605d8006de96ed91d5cac8b7fa97bd810cf7dfc",
            1090665568,
            "0433e1c67a5417f4995713c92cc505cf057d4056c4193a8c506231cbc91948e8",
        ),
    }
    for runtime_id, profile in profiles.items():
        identity = (
            profile["repo"],
            profile["revision"],
            profile["config_sha256"],
            profile["weights_size"],
            profile["weights_sha256"],
        )
        if identity != expected_sources[runtime_id]:
            raise AssertionError("%s pinned source identity differs" % runtime_id)
        if profile["remote_code"] != {
            "modeling_olmo_sha256": converter.REMOTE_MODELING_SHA256,
            "configuration_olmo_sha256": converter.REMOTE_CONFIGURATION_SHA256,
        }:
            raise AssertionError("%s pinned remote-code identity differs" % runtime_id)


def test_production_profile_corruptions(
    converter: Any, path: Path, work_dir: Path
) -> None:
    production = json.loads(path.read_text(encoding="utf-8"))
    cases = (
        (
            "missing-omni1b-layer-norm-kernel",
            "geneb-omni-dna-1b",
            lambda profile: profile.pop("layer_norm_kernel"),
            "layer_norm_kernel is not the exact Omni-DNA-1B contract",
        ),
        (
            "layer-norm-kernel-on-300m",
            "geneb-omni-dna-300m",
            lambda profile: profile.update(
                {"layer_norm_kernel": converter.OMNI1B_LAYER_NORM_KERNEL}
            ),
            "layer_norm_kernel is not the exact Omni-DNA-1B contract",
        ),
        (
            "unknown-omni1b-layer-norm-kernel",
            "geneb-omni-dna-1b",
            lambda profile: profile.update({"layer_norm_kernel": "not-a-kernel"}),
            "layer_norm_kernel is not the exact Omni-DNA-1B contract",
        ),
        (
            "omni1b-exact-topology-drift",
            "geneb-omni-dna-1b",
            lambda profile: profile["topology"].update({"hidden_size": 1024}),
            "exact LayerNorm kernel topology differs",
        ),
    )
    for label, runtime_id, mutate, expected in cases:
        corrupted = copy.deepcopy(production)
        profile = next(
            item for item in corrupted["models"] if item["runtime_id"] == runtime_id
        )
        mutate(profile)
        corrupt_path = work_dir / (label + ".json")
        write_json(corrupt_path, corrupted)
        try:
            converter.load_profiles(corrupt_path)
        except converter.ConversionError as error:
            if expected not in str(error):
                raise AssertionError(
                    "%s returned wrong profile error: %s" % (label, error)
                )
        else:
            raise AssertionError("%s profile corruption was accepted" % label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--native-loader", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.converter.resolve().parent))
    spec = importlib.util.spec_from_file_location("geneb_olmo_converter", args.converter)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import converter")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)

    validate_production_profiles(converter, args.profiles)
    test_production_profile_corruptions(converter, args.profiles, args.work_dir)
    layer_case = make_case(args.work_dir / "valid-layernorm", converter, "layernorm")
    validate_success(args, layer_case, converter)
    test_native_corruption(args, layer_case)
    rms_case = make_case(args.work_dir / "valid-rmsnorm", converter, "rmsnorm")
    validate_success(args, rms_case, converter)
    test_corruptions(args, converter)
    print("GENEB OLMo converter/native corruption contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
