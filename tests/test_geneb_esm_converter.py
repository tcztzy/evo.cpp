#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic conversion and corruption gates for GENEB ESM artifacts."""

import argparse
import ast
import hashlib
import http.client
import json
import os
import selectors
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


PRODUCTION_IDS = {
    "geneb-nt-2-5b-ms",
    "geneb-agro-nt-1b",
    "geneb-nt-v2-250m-ms",
    "geneb-nt-v2-100m-ms",
    "geneb-nt-v2-50m-3mer-ms",
    "geneb-nt-v2-50m-ms",
}


def wait_for_server(process: Any) -> int:
    if process.stderr is None:
        raise AssertionError("server stderr was not captured")
    selector = selectors.DefaultSelector()
    selector.register(process.stderr, selectors.EVENT_READ)
    prefix = b"evo_server listening 127.0.0.1:"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                "GENEB server exited before listening: %s"
                % process.stderr.read().decode(errors="replace")
            )
        for key, _ in selector.select(timeout=0.25):
            line = key.fileobj.readline()
            if line.startswith(prefix):
                return int(line[len(prefix) :])
    raise AssertionError("GENEB server did not announce its port")


def server_json(port: int, payload: Mapping[str, Any]) -> Tuple[int, Dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(
        "POST",
        "/v1/embeddings",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    body = response.read()
    status = response.status
    connection.close()
    return status, json.loads(body.decode("utf-8"))


def read_npy_f32(path: Path) -> Tuple[Tuple[int, ...], List[float]]:
    payload = path.read_bytes()
    if payload[:6] != b"\x93NUMPY" or payload[6:8] != b"\x01\x00":
        raise AssertionError("CLI embedding is not a v1 NPY file")
    header_size = struct.unpack("<H", payload[8:10])[0]
    header = ast.literal_eval(payload[10 : 10 + header_size].decode("ascii"))
    shape = tuple(int(value) for value in header["shape"])
    offset = 10 + header_size
    values = list(struct.unpack("<%df" % ((len(payload) - offset) // 4), payload[offset:]))
    return shape, values


def verify_cli_server_preset(cli: Path, artifact: Path, work_dir: Path) -> None:
    sequence = work_dir / "preset-input.fasta"
    sequence.write_text(">probe\nAAAAAA\n", encoding="ascii")
    output = work_dir / "preset-cli"
    subprocess.run(
        [
            str(cli),
            "embed",
            "-m",
            str(artifact),
            "--input",
            str(sequence),
            "--output",
            str(output),
            "--preset",
            "geneb",
            "--backend",
            "cpu",
        ],
        check=True,
        capture_output=True,
    )
    manifest = json.loads((output / "embeddings.jsonl").read_text(encoding="utf-8"))
    shape, values = read_npy_f32(output / manifest["file"])
    if (
        manifest.get("preset") != "geneb-v4-normalized"
        or manifest.get("pooling") != "attention-mask-mean"
        or manifest.get("pad_right") != 2
        or shape != (1, 4)
    ):
        raise AssertionError("CLI GENEB preset metadata drifted")

    process = subprocess.Popen(
        [
            str(cli),
            "serve",
            "-m",
            str(artifact),
            "--backend",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--max-queue",
            "4",
            "--max-batch",
            "2",
            "--max-request-bytes",
            "512",
            "--max-sequence-bytes",
            "24",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        port = wait_for_server(process)
        status, document = server_json(
            port,
            {"sequence": "AAAAAA", "preset": "geneb-v4-normalized"},
        )
        if (
            status != 200
            or document.get("preset") != manifest["preset"]
            or document.get("hidden_tap") != manifest["point"]
            or document.get("pooling") != manifest["pooling"]
            or document.get("shape") != [1, 4]
            or document.get("embedding") != values
            or document.get("input", {}).get("pad_right") != 2
            or document.get("input", {}).get("token_count") != 4
        ):
            raise AssertionError("CLI/server GENEB preset parity failed: %r" % document)
        rejected, rejection = server_json(
            port,
            {
                "sequence": "AAAAAA",
                "preset": "geneb-v4-normalized",
                "layer": 1,
            },
        )
        if rejected != 400 or rejection.get("error", {}).get("code") != "invalid_argument":
            raise AssertionError("server accepted preset plus explicit layer")
        if process.poll() is not None:
            raise AssertionError("server exited after a preset request")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def write_duplicate_config(
    path: Path, key: str, first_value: Any, second_value: Any
) -> None:
    value = config_value()
    value.pop(key)
    fields = [
        json.dumps(key) + ":" + json.dumps(first_value),
        json.dumps(key) + ":" + json.dumps(second_value),
    ]
    fields.extend(
        json.dumps(name) + ":" + json.dumps(value[name], separators=(",", ":"))
        for name in sorted(value)
    )
    path.write_text("{" + ",".join(fields) + "}\n", encoding="ascii")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_value() -> Dict[str, Any]:
    return {
        "architectures": ["EsmForMaskedLM"],
        "attention_probs_dropout_prob": 0,
        "emb_layer_norm_before": False,
        "hidden_dropout_prob": 0,
        "hidden_size": 4,
        "intermediate_size": 6,
        "layer_norm_eps": 0.00001,
        "mask_token_id": 2,
        "max_position_embeddings": 6,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "pad_token_id": 1,
        "position_embedding_type": "absolute",
        "tie_word_embeddings": False,
        "token_dropout": True,
        "torch_dtype": "float32",
        "vocab_size": 8,
    }


def topology_value() -> Dict[str, Any]:
    return {
        "vocab_size": 8,
        "hidden_size": 4,
        "num_layers": 1,
        "num_attention_heads": 2,
        "head_dim": 2,
        "intermediate_size": 6,
        "max_seqlen": 4,
        "max_position_embeddings": 6,
        "layer_norm_epsilon": 0.00001,
        "rope_base": 10000.0,
        "position_embedding_type": "absolute",
        "mlp_activation": "gelu",
        "attention_bias": True,
        "feed_forward_bias": True,
        "token_dropout": True,
        "pad_token_id": 1,
        "mask_token_id": 2,
        "cls_token_id": 3,
    }


def payload_for(name: str, size: int, dtype: str) -> bytes:
    seed = sum(name.encode("ascii")) % 251
    if dtype == "F32":
        values = []
        for index in range(size // 4):
            if name.endswith("LayerNorm.weight") or "layer_norm_after.weight" in name:
                value = 1.0 + ((seed + index) % 5) / 128.0
            else:
                value = (((seed + index * 7) % 19) - 9) / 64.0
            values.append(struct.pack("<f", value))
        return b"".join(values)
    if dtype == "BF16":
        return struct.pack("<H", 0x3D00) * (size // 2)
    if dtype == "I64":
        return b"".join(struct.pack("<q", index) for index in range(size // 8))
    raise AssertionError("unsupported fixture dtype %s" % dtype)


def spec_bytes(spec: Any, dtype_override: Optional[str] = None) -> int:
    dtype = dtype_override if dtype_override is not None else spec.dtype
    width = {"F32": 4, "BF16": 2, "I64": 8}[dtype]
    count = 1
    for dimension in spec.shape:
        count *= dimension
    return count * width


def write_safetensors(
    path: Path,
    specs: Sequence[Any],
    dtype_override: Optional[Tuple[str, str]] = None,
    trailing: bytes = b"",
) -> Dict[str, bytes]:
    root = {"__metadata__": {"format": "pt"}}  # type: Dict[str, Any]
    payloads = {}  # type: Dict[str, bytes]
    offset = 0
    for spec in specs:
        dtype = (
            dtype_override[1]
            if dtype_override is not None and spec.name == dtype_override[0]
            else spec.dtype
        )
        size = spec_bytes(spec, dtype)
        payload = payload_for(spec.name, size, dtype)
        root[spec.name] = {
            "dtype": dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + size],
        }
        payloads[spec.name] = payload
        offset += size
    raw = json.dumps(root, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(header))
        + header
        + b"".join(payloads[spec.name] for spec in specs)
        + trailing
    )
    return payloads


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


def read_artifact(path: Path) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        data = source.read()
    metadata = header.pop("__metadata__")
    tensors = {
        name: data[value["data_offsets"][0] : value["data_offsets"][1]]
        for name, value in header.items()
    }
    return metadata, tensors


def rewrite_artifact_header(
    source_path: Path,
    output_path: Path,
    mutate: Callable[[Dict[str, Any]], None],
) -> None:
    with source_path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        payload = source.read()
    mutate(header)
    raw = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    encoded = raw + b" " * ((-len(raw)) % 8)
    output_path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def make_case(
    root: Path, converter: Any, source_format: str = "safetensors"
) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    runtime_id = "geneb-esm-tiny"
    revision = "a" * 40
    repo = "fixture/EsmTiny"
    config = root / "config.json"
    config_required = config_value()
    write_json(config, config_required)
    topology = topology_value()
    profile = {
        "runtime_id": runtime_id,
        "geneb_model_id": "EsmTiny",
        "paper_name": "EsmTiny",
        "catalog_architecture": "esm-masked-lm",
        "repo": repo,
        "revision": revision,
        "source_format": source_format,
        "config_sha256": digest(config),
        "source_variant": "hf-esm-gelu",
        "source_position_ids": True,
        "source_rotary_inv_freq": False,
        "omit_policy": "validate-heads-and-unused-buffers-then-omit",
        "tokenizer_k": 6,
        "config_required": config_required,
        "topology": topology,
    }
    profiles = root / "profiles.json"
    write_json(
        profiles,
        {
            "schema_version": 1,
            "format": "geneb-esm-converter-v1",
            "models": [profile],
        },
    )
    catalog = root / "catalog.json"
    preset = {
        "hidden_tap": "last-hidden-state",
        "pooling": "attention-mask-mean",
        "special_tokens": "include",
        "mask_domain": "attention-mask",
        "output_width": 4,
    }
    write_json(
        catalog,
        {
            "schema_version": 1,
            "suite": {
                "id": "geneb-v4",
                "raw_safety_cap_bytes": 16777216,
            },
            "models": [
                {
                    "runtime_id": runtime_id,
                    "geneb_model_id": "EsmTiny",
                    "paper_name": "EsmTiny",
                    "family": "transformer-encoder",
                    "architecture": "esm-masked-lm",
                    "source": {
                        "kind": "huggingface",
                        "repo": repo,
                        "requested_revision": revision,
                        "revision": revision,
                        "immutable": True,
                    },
                    "context": {
                        "declared_max_tokens": 6,
                        "reference_max_tokens": 4,
                        "unit": "tokens",
                        "length_policy": "tokenizer-truncate",
                    },
                    "tokenizer": {
                        "kind": "k-mer",
                        "k": 6,
                        "max_tokens": 4,
                        "add_special_tokens": True,
                        "padding_side": "right",
                        "pad_to": "model-max",
                    },
                    "input_transform": {
                        "case": "preserve",
                        "strip_ascii_whitespace": False,
                        "u_to_t": False,
                        "invalid": "tokenizer-defined",
                        "frame_trim": None,
                        "raw_crop": None,
                        "fixed_pad": {
                            "length": 4,
                            "side": "right",
                            "value": "tokenizer-pad",
                            "balance": None,
                        },
                        "prefix": None,
                        "special_tokens": "tokenizer-default",
                        "token_truncation": "right",
                    },
                    "embedding_presets": {
                        "reference": dict(preset),
                        "normalized": dict(preset),
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

    specs, runtime_count = converter.source_tensor_specs(profile)
    payloads = {}  # type: Dict[str, bytes]
    if source_format == "safetensors":
        weights = root / "model.safetensors"
        payloads = write_safetensors(weights, specs)
        source_files = [config, weights]
    elif source_format == "pytorch-bin":
        weights = root / "pytorch_model.bin"
        weights.write_bytes(b"optional offline PyTorch fixture")
        source_files = [config, weights]
    elif source_format == "pytorch-bin-sharded":
        split = len(specs) // 2
        groups = (specs[:split], specs[split:])
        source_files = [config]
        owners = {}  # type: Dict[str, str]
        for index, group in enumerate(groups, start=1):
            shard = root / ("pytorch_model-%05d-of-00002.bin" % index)
            shard.write_bytes(b"optional offline PyTorch shard")
            source_files.append(shard)
            owners.update({spec.name: shard.name for spec in group})
        index_path = root / "pytorch_model.bin.index.json"
        write_json(
            index_path,
            {
                "metadata": {"total_size": sum(spec.nbytes for spec in specs)},
                "weight_map": owners,
            },
        )
        source_files.insert(1, index_path)
    else:
        raise AssertionError("unsupported fixture source format")
    receipt = root / "source-receipt.json"
    write_json(
        receipt,
        {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": runtime_id,
            "repo": repo,
            "requested_revision": revision,
            "resolved_revision": revision,
            "files": [receipt_entry(path) for path in source_files],
            "load_path": None,
        },
    )
    tokenizer_asset = root / "assets" / "tokenizer.json"
    tokenizer_asset.parent.mkdir(parents=True)
    write_json(
        tokenizer_asset,
        {
            "format": "evo-tokenizer-v1",
            "kind": "kmer",
            "normalization": [{"op": "ascii-uppercase"}],
            "pre_tokenizer": {"kind": "none"},
            "model": {
                "k": 6,
                "stride": 6,
                "tail": "drop",
                "unknown_policy": "unk",
            },
            "post_processor": {
                "prefix_ids": [3],
                "suffix_ids": [],
                "padding": {"side": "none", "pad_id": None},
            },
            "special_tokens": {
                "unk": 0,
                "pad": 1,
                "bos": None,
                "eos": None,
                "cls": 3,
                "sep": None,
                "mask": 2,
            },
            "vocab": [
                {"id": 0, "piece": "[UNK]"},
                {"id": 1, "piece": "[PAD]"},
                {"id": 2, "piece": "[MASK]"},
                {"id": 3, "piece": "[CLS]"},
                {"id": 4, "piece": "AAAAAA"},
                {"id": 5, "piece": "CCCCCC"},
                {"id": 6, "piece": "GGGGGG"},
                {"id": 7, "piece": "TTTTTT"},
            ],
        },
    )
    tokenizer_descriptor = root / "tokenizer-descriptor.json"
    write_json(
        tokenizer_descriptor,
        {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": "1" * 64,
            "source_receipt_contract_sha256": "2" * 64,
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": "assets/tokenizer.json",
            "tokenizer.sha256": digest(tokenizer_asset),
            "tokenizer.size": tokenizer_asset.stat().st_size,
        },
    )
    return {
        "root": root,
        "profile": profile,
        "profiles": profiles,
        "catalog": catalog,
        "config": config,
        "receipt": receipt,
        "source_files": source_files,
        "specs": specs,
        "runtime_count": runtime_count,
        "payloads": payloads,
        "tokenizer_root": root,
        "tokenizer_asset": tokenizer_asset,
        "tokenizer_descriptor": tokenizer_descriptor,
    }


def run_converter(
    converter_path: Path,
    case: Mapping[str, Any],
    output: Path,
    force: bool = False,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(converter_path),
        "--receipt",
        str(case["receipt"]),
        "--catalog",
        str(case["catalog"]),
        "--profiles",
        str(case["profiles"]),
        "--tokenizer-descriptor",
        str(case["tokenizer_descriptor"]),
        "--tokenizer-root",
        str(case["tokenizer_root"]),
        "--output",
        str(output),
    ]
    if force:
        command.append("--force")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(converter_path.parent)
    return subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment
    )


def require_success(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode != 0:
        raise AssertionError("%s failed: %s" % (label, result.stderr))


def require_rejected(
    converter_path: Path,
    case: Mapping[str, Any],
    output: Path,
    fragment: Optional[str] = None,
) -> None:
    result = run_converter(converter_path, case, output)
    if result.returncode == 0:
        raise AssertionError("corrupt case unexpectedly converted: %s" % output.parent.name)
    if fragment is not None and fragment not in result.stderr:
        raise AssertionError(
            "%s rejection omitted %r: %s" % (output.parent.name, fragment, result.stderr)
        )
    if output.exists() or list(output.parent.glob(".%s.*.tmp" % output.name)):
        raise AssertionError("corrupt conversion left partial output")


def rewrite_profiles(case: Mapping[str, Any], mutate: Callable[[Dict[str, Any]], None]) -> None:
    root = json.loads(case["profiles"].read_text(encoding="utf-8"))
    mutate(root["models"][0])
    write_json(case["profiles"], root)


def require_native_rejected(native_loader: Path, artifact: Path, label: str) -> None:
    result = subprocess.run(
        [str(native_loader), "--verify-artifact", str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        raise AssertionError("native loader accepted corrupt runtime %s" % label)


def validate_production_profiles(
    converter: Any, path: Path, catalog: Path
) -> None:
    profiles, _ = converter.load_profiles(path)
    if set(profiles) != PRODUCTION_IDS:
        raise AssertionError("production ESM profile set is not NT x5 plus Agro-NT")
    expected = {
        "geneb-nt-2-5b-ms": (2560, 32, 20, 128, 1000, "absolute", "gelu", True),
        "geneb-agro-nt-1b": (1500, 40, 20, 75, 1024, "absolute", "gelu", False),
        "geneb-nt-v2-250m-ms": (768, 24, 16, 48, 2048, "rotary", "swiglu", False),
        "geneb-nt-v2-100m-ms": (512, 22, 16, 32, 2048, "rotary", "swiglu", False),
        "geneb-nt-v2-50m-3mer-ms": (512, 12, 16, 32, 2048, "rotary", "swiglu", False),
        "geneb-nt-v2-50m-ms": (512, 12, 16, 32, 2048, "rotary", "swiglu", False),
    }
    for runtime_id, values in expected.items():
        profile = profiles[runtime_id]
        topology = profile["topology"]
        actual = (
            topology["hidden_size"],
            topology["num_layers"],
            topology["num_attention_heads"],
            topology["head_dim"],
            topology["max_seqlen"],
            topology["position_embedding_type"],
            topology["mlp_activation"],
            topology["token_dropout"],
        )
        if actual != values or profile["config_required"].get("vocab_size") != topology["vocab_size"]:
            raise AssertionError("production ESM topology drifted for %s" % runtime_id)
        entry, catalog_root, catalog_payload = converter.load_catalog_entry(
            catalog, profile
        )
        geneb_metadata = converter.build_geneb_artifact_metadata(
            catalog_root, entry, catalog_payload
        )
        if (
            geneb_metadata.get("geneb.runtime_id") != runtime_id
            or geneb_metadata.get("geneb.preset.reference.output_width")
            != topology["hidden_size"]
        ):
            raise AssertionError("GENEB evidence metadata drifted for %s" % runtime_id)
        artifact_metadata = converter.build_metadata(
            profile,
            {},
            "0" * 64,
            "1" * 64,
            "2" * 64,
            "3" * 64,
            geneb_metadata,
        )
        expected_layer_norm_kernel = (
            converter.AGRO_LAYER_NORM_KERNEL
            if runtime_id == "geneb-agro-nt-1b"
            else None
        )
        if artifact_metadata.get("esm.layer_norm_kernel") != expected_layer_norm_kernel:
            raise AssertionError(
                "profile-to-artifact LayerNorm kernel drifted for %s" % runtime_id
            )
        source, runtime_count = converter.source_tensor_specs(profile)
        runtime_names = {
            item.name for item in converter.canonical_tensor_specs(topology)
        }
        omitted_names = {item.name for item in source[runtime_count:]}
        if set(item.name for item in source[:runtime_count]) != runtime_names:
            raise AssertionError("runtime tensor prefix drifted for %s" % runtime_id)
        if "lm_head.decoder.weight" not in omitted_names:
            raise AssertionError("ESM language-model head is not validate-and-omit")
        if topology["position_embedding_type"] == "rotary":
            if "esm.embeddings.position_embeddings.weight" not in omitted_names:
                raise AssertionError("NT-v2 unused learned position table was not validated")
            inv_freq = [name for name in omitted_names if name.endswith("inv_freq")]
            if len(inv_freq) != topology["num_layers"]:
                raise AssertionError("NT-v2 rotary buffers were not strictly validated")
    if profiles["geneb-nt-v2-50m-3mer-ms"]["tokenizer_k"] != 3:
        raise AssertionError("NT-v2 50M 3-mer tokenizer contract drifted")
    if (
        profiles["geneb-agro-nt-1b"].get("layer_norm_kernel")
        != converter.AGRO_LAYER_NORM_KERNEL
        or any(
            "layer_norm_kernel" in profile
            for runtime_id, profile in profiles.items()
            if runtime_id != "geneb-agro-nt-1b"
        )
    ):
        raise AssertionError("Agro-only exact LayerNorm profile contract drifted")
    if profiles["geneb-nt-2-5b-ms"]["source_format"] != "pytorch-bin-sharded":
        raise AssertionError("NT-2.5B sharded PyTorch source contract drifted")
    expected_manifest_sha256 = {
        "geneb-nt-2-5b-ms": "a19b30c4284c792be67a79dba336156d0cdebd2b2f4d57ea4d1a7cc62975cb26",
        "geneb-agro-nt-1b": "a19b30c4284c792be67a79dba336156d0cdebd2b2f4d57ea4d1a7cc62975cb26",
        "geneb-nt-v2-50m-3mer-ms": "e2be8bab14cf7d23838c7771b2b0ff40e83dd04ae55d04041e39d1ec1937b3cf",
        "geneb-nt-v2-50m-ms": "5835cc71cc57e4f2d0d66f89549c732b9d5ff1ae6c01b3ed54776ec654dfc570",
    }
    if any(
        profile.get("tokenizer_manifest_sha256")
        != expected_manifest_sha256.get(runtime_id)
        for runtime_id, profile in profiles.items()
    ):
        raise AssertionError("NT-v2-50M tokenizer compiler binding drifted")
    nt3_manifest_path = path.parent / "tokenizers" / "geneb-nt-v2-3mer-v1.json"
    nt3_manifest = json.loads(nt3_manifest_path.read_text(encoding="utf-8"))
    nt3_vocab = ["<unk>", "<pad>", "<mask>", "<cls>", "<eos>", "<bos>"]
    nt3_vocab.extend(
        first + second + third
        for first in "ATCG"
        for second in "ATCG"
        for third in "ATCG"
    )
    nt3_vocab.extend(["A", "T", "C", "G", "N"])
    nt3_vocab_payload = "\n".join(nt3_vocab).encode("ascii")
    if (
        digest(nt3_manifest_path)
        != expected_manifest_sha256["geneb-nt-v2-50m-3mer-ms"]
        or nt3_manifest.get("source") != "vocab-text"
        or nt3_manifest.get("kind") != "longest-match"
        or nt3_manifest.get("files")
        != [
            {
                "role": "vocab",
                "name": "vocab.txt",
                "size": len(nt3_vocab_payload),
                "sha256": hashlib.sha256(nt3_vocab_payload).hexdigest(),
            }
        ]
        or len(nt3_vocab) != 75
        or nt3_vocab[11] != "ATT"
        or nt3_vocab[49] != "CCG"
        or nt3_vocab[70:] != ["A", "T", "C", "G", "N"]
    ):
        raise AssertionError("NT-v2-50M-3mer production vocabulary drifted")
    nt25 = profiles["geneb-nt-2-5b-ms"]
    if {
        key: nt25.get(key) for key in converter.NT_2_5B_TOKENIZER_PINS
    } != converter.NT_2_5B_TOKENIZER_PINS:
        raise AssertionError("NT-2.5B exact tokenizer descriptor pins drifted")
    if any(
        key in profile
        for runtime_id, profile in profiles.items()
        if runtime_id != "geneb-nt-2-5b-ms"
        for key in (
            "tokenizer_source_receipt_contract_sha256",
            "tokenizer_asset_sha256",
            "tokenizer_asset_size",
        )
    ):
        raise AssertionError("NT-2.5B exact tokenizer pins escaped its profile")
    nt3 = profiles["geneb-nt-v2-50m-3mer-ms"]
    column_major_models = {
        "geneb-nt-2-5b-ms",
        "geneb-agro-nt-1b",
        "geneb-nt-v2-50m-3mer-ms",
        "geneb-nt-v2-250m-ms",
    }
    if (
        nt3.get("config_duplicate_keys")
        != {"attention_probs_dropout_prob": {"count": 2, "value": 0.0}}
        or nt25.get("source_tensor_layout")
        != "contiguous-or-exact-column-major"
        or nt3.get("source_tensor_layout")
        != "contiguous-or-exact-column-major"
        or profiles["geneb-nt-v2-250m-ms"].get("source_tensor_layout")
        != "contiguous-or-exact-column-major"
        or profiles["geneb-agro-nt-1b"].get("source_tensor_layout")
        != "contiguous-or-exact-column-major"
        or any(
            profile.get("source_tensor_layout") != "contiguous"
            or "config_duplicate_keys" in profile
            for runtime_id, profile in profiles.items()
            if runtime_id not in column_major_models
        )
    ):
        raise AssertionError("NT-v2 audited source exceptions drifted")
    boundary_manifests = {
        "geneb-nt-2-5b-ms": "geneb-nt-2-5b-ms-6mer-v1.json",
        "geneb-agro-nt-1b": "geneb-agro-nt-1b-6mer-v1.json",
    }
    for runtime_id, filename in boundary_manifests.items():
        manifest_path = path.parent / "tokenizers" / filename
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            digest(manifest_path) != expected_manifest_sha256[runtime_id]
            or manifest.get("source") != "vocab-text"
            or manifest.get("kind") != "longest-match"
            or manifest.get("files")
            != [
                {
                    "role": "vocab",
                    "name": "vocab.txt",
                    "size": 28718,
                    "sha256": "f1b544e27897936b50bbd925850fa8a08b421c33bb9c26e3711c140c061d0d4c",
                }
            ]
            or manifest.get("options", {}).get("audited_vocab_boundary")
            != {
                "source_size": 4107,
                "compiled_size": 4105,
                "excluded_suffix": [
                    {"id": 4105, "piece": "<eos>", "input_policy": "reject"},
                    {"id": 4106, "piece": "<bos>", "input_policy": "reject"},
                ],
            }
        ):
            raise AssertionError(
                "production vocabulary boundary drifted for %s" % runtime_id
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--native-loader", type=Path)
    parser.add_argument("--c-api", type=Path)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    sys.path.insert(0, str(args.converter.parent))
    import convert_geneb_esm_checkpoint as converter

    validate_production_profiles(converter, args.profiles, args.catalog)
    production_profiles = json.loads(args.profiles.read_text(encoding="utf-8"))

    def require_bad_production_profile(label: str, mutate: Callable[[Dict[str, Any]], None]) -> None:
        document = json.loads(json.dumps(production_profiles))
        mutate(document)
        path = args.work_dir / ("bad-production-" + label + ".json")
        write_json(path, document)
        try:
            converter.load_profiles(path)
        except converter.ConversionError:
            return
        raise AssertionError("production ESM profile admitted %s LayerNorm contract" % label)

    def production_row(root: Dict[str, Any], runtime_id: str) -> Dict[str, Any]:
        matches = [
            row
            for row in root["models"]
            if isinstance(row, dict) and row.get("runtime_id") == runtime_id
        ]
        if len(matches) != 1:
            raise AssertionError("production profile fixture lost %s" % runtime_id)
        return matches[0]

    def remove_agro_kernel(root: Dict[str, Any]) -> None:
        production_row(root, "geneb-agro-nt-1b").pop("layer_norm_kernel")

    def corrupt_agro_kernel(root: Dict[str, Any]) -> None:
        production_row(root, "geneb-agro-nt-1b")[
            "layer_norm_kernel"
        ] = "portable-two-pass"

    require_bad_production_profile(
        "missing-kernel",
        remove_agro_kernel,
    )
    require_bad_production_profile(
        "wrong-kernel",
        corrupt_agro_kernel,
    )

    def move_kernel_to_wrong_runtime(root: Dict[str, Any]) -> None:
        value = production_row(root, "geneb-agro-nt-1b").pop(
            "layer_norm_kernel"
        )
        production_row(root, "geneb-nt-2-5b-ms")["layer_norm_kernel"] = value

    require_bad_production_profile("wrong-runtime", move_kernel_to_wrong_runtime)

    for pin_key in converter.NT_2_5B_TOKENIZER_PINS:
        require_bad_production_profile(
            "missing-nt25-" + pin_key.replace("_", "-"),
            lambda root, key=pin_key: production_row(
                root, converter.NT_2_5B_RUNTIME_ID
            ).pop(key),
        )

    def corrupt_nt25_tokenizer_pin(root: Dict[str, Any]) -> None:
        production_row(root, converter.NT_2_5B_RUNTIME_ID)[
            "tokenizer_asset_sha256"
        ] = "0" * 64

    require_bad_production_profile(
        "wrong-nt25-tokenizer-pin", corrupt_nt25_tokenizer_pin
    )

    def move_nt25_tokenizer_pins(root: Dict[str, Any]) -> None:
        nt25 = production_row(root, converter.NT_2_5B_RUNTIME_ID)
        sibling = production_row(root, "geneb-nt-v2-100m-ms")
        for key in converter.NT_2_5B_TOKENIZER_PINS:
            sibling[key] = nt25.pop(key)

    require_bad_production_profile(
        "wrong-nt25-tokenizer-runtime", move_nt25_tokenizer_pins
    )

    def remove_nt25_layout(root: Dict[str, Any]) -> None:
        production_row(root, converter.NT_2_5B_RUNTIME_ID).pop(
            "source_tensor_layout"
        )

    require_bad_production_profile("missing-nt25-layout", remove_nt25_layout)
    helper_profile = {"runtime_id": "fixture", "value": 1}
    helper_digest = converter.converter_profile_contract_sha256(
        1, converter.PROFILE_FORMAT, helper_profile
    )
    if helper_digest == converter.converter_profile_contract_sha256(
        1, converter.PROFILE_FORMAT, {"runtime_id": "fixture", "value": 2}
    ):
        raise AssertionError("converter profile helper ignored selected-row semantics")
    try:
        converter.converter_profile_contract_sha256(
            1,
            converter.PROFILE_FORMAT,
            helper_profile,
            {"unknown_contract": {"value": 1}},
        )
    except converter.GenebArtifactError:
        pass
    else:
        raise AssertionError("converter profile helper admitted an unknown shared field")

    valid = make_case(args.work_dir / "valid", converter)
    descriptor_payload = json.loads(
        valid["tokenizer_descriptor"].read_text(encoding="utf-8")
    )
    pinned_descriptor_profile = {
        "runtime_id": converter.NT_2_5B_RUNTIME_ID,
        "tokenizer_manifest_sha256": descriptor_payload[
            "compiler_manifest_sha256"
        ],
        "tokenizer_source_receipt_contract_sha256": descriptor_payload[
            "source_receipt_contract_sha256"
        ],
        "tokenizer_asset_sha256": descriptor_payload["tokenizer.sha256"],
        "tokenizer_asset_size": descriptor_payload["tokenizer.size"],
    }
    converter.validate_tokenizer_descriptor(
        valid["tokenizer_descriptor"],
        valid["tokenizer_root"],
        valid["root"],
        pinned_descriptor_profile,
    )
    mismatched_descriptor = valid["root"] / "mismatched-tokenizer-descriptor.json"
    mismatched_payload = json.loads(json.dumps(descriptor_payload))
    mismatched_payload["source_receipt_contract_sha256"] = "f" * 64
    write_json(mismatched_descriptor, mismatched_payload)
    try:
        converter.validate_tokenizer_descriptor(
            mismatched_descriptor,
            valid["tokenizer_root"],
            valid["root"],
            pinned_descriptor_profile,
        )
    except converter.ConversionError as error:
        if "differs from pinned NT-2.5B profile" not in str(error):
            raise
    else:
        raise AssertionError("NT-2.5B descriptor source provenance drift was accepted")
    output_a = valid["root"] / "runtime-a.safetensors"
    output_b = valid["root"] / "runtime-b.safetensors"
    require_success(run_converter(args.converter, valid, output_a), "ESM conversion")
    require_success(run_converter(args.converter, valid, output_b), "repeat conversion")
    if output_a.read_bytes() != output_b.read_bytes():
        raise AssertionError("identical ESM conversion inputs did not produce identical bytes")
    base_profiles = json.loads(valid["profiles"].read_text(encoding="utf-8"))
    sibling_profiles = json.loads(json.dumps(base_profiles))
    sibling = json.loads(json.dumps(sibling_profiles["models"][0]))
    sibling.update(
        {
            "runtime_id": "geneb-esm-sibling",
            "geneb_model_id": "EsmSibling",
            "paper_name": "EsmSibling",
            "repo": "fixture/EsmSibling",
            "revision": "e" * 40,
        }
    )
    sibling_profiles["models"].append(sibling)
    write_json(valid["profiles"], sibling_profiles)
    sibling_output = valid["root"] / "runtime-sibling-change.safetensors"
    require_success(
        run_converter(args.converter, valid, sibling_output),
        "sibling-profile-independent ESM conversion",
    )
    if output_a.read_bytes() != sibling_output.read_bytes():
        raise AssertionError("sibling ESM profile changed selected artifact bytes")

    selected_change = json.loads(json.dumps(base_profiles))
    selected_change["models"][0][
        "source_tensor_layout"
    ] = "contiguous-or-exact-column-major"
    write_json(valid["profiles"], selected_change)
    selected_output = valid["root"] / "runtime-selected-change.safetensors"
    require_success(
        run_converter(args.converter, valid, selected_output),
        "selected-profile contract change",
    )
    if output_a.read_bytes() == selected_output.read_bytes():
        raise AssertionError("selected ESM profile semantics did not change artifact bytes")
    write_json(valid["profiles"], base_profiles)
    relocated_root = args.work_dir / "relocated-tokenizer-artifact"
    relocated_asset = relocated_root / "assets" / "tokenizer.json"
    relocated_asset.parent.mkdir(parents=True)
    shutil.copyfile(valid["tokenizer_asset"], relocated_asset)
    relocated_descriptor = relocated_root / "tokenizer-descriptor.json"
    shutil.copyfile(valid["tokenizer_descriptor"], relocated_descriptor)
    relocated_case = dict(valid)
    relocated_case["tokenizer_root"] = relocated_root
    relocated_case["tokenizer_asset"] = relocated_asset
    relocated_case["tokenizer_descriptor"] = relocated_descriptor
    relocated_output = relocated_root / "runtime.safetensors"
    require_success(
        run_converter(args.converter, relocated_case, relocated_output),
        "relocated tokenizer descriptor ESM conversion",
    )
    if output_a.read_bytes() != relocated_output.read_bytes():
        raise AssertionError(
            "ESM artifact changed across absolute tokenizer staging roots"
        )
    base_catalog = json.loads(valid["catalog"].read_text(encoding="utf-8"))
    promotion_catalog = json.loads(valid["catalog"].read_text(encoding="utf-8"))
    promotion_model = promotion_catalog["models"][0]
    promotion_model["oracle"] = {"status": "passed", "evidence": {"fixture": True}}
    promotion_model["runtime_support"] = {"status": "supported"}
    promotion_model["backends"] = {
        "cpu": {"status": "promoted"},
        "cuda": {"status": "unsupported"},
        "mps": {"status": "unsupported"},
    }
    promotion_model["promotion_state"] = "runtime-supported"
    write_json(valid["catalog"], promotion_catalog)
    promotion_output = valid["root"] / "runtime-promotion-only.safetensors"
    require_success(
        run_converter(args.converter, valid, promotion_output),
        "promotion-only ESM conversion",
    )
    if output_a.read_bytes() != promotion_output.read_bytes():
        raise AssertionError("promotion-only catalog changes altered artifact bytes")
    write_json(valid["catalog"], base_catalog)
    metadata, output_tensors = read_artifact(output_a)
    runtime_specs = valid["specs"][: valid["runtime_count"]]
    runtime_names = {spec.name for spec in runtime_specs}
    if set(output_tensors) != runtime_names:
        raise AssertionError("runtime artifact tensor set differs from canonical runtime set")
    if any(output_tensors[name] != valid["payloads"][name] for name in runtime_names):
        raise AssertionError("F32 ESM tensor payload changed during conversion")
    omitted = {spec.name for spec in valid["specs"][valid["runtime_count"] :]}
    if set(output_tensors) & omitted or "esm.embeddings.position_ids" not in omitted:
        raise AssertionError("source-only ESM tensors were not validate-and-omit")
    required_metadata = {
        "runtime.profile": "s:geneb-esm-runtime-v1",
        "runtime.abi": "s:geneb-esm-safetensors-v1",
        "model.architecture": "s:GenebEsmEncoder",
        "config.vocab_size": "u:8",
        "config.hidden_size": "u:4",
        "config.num_layers": "u:1",
        "config.max_seqlen": "u:4",
        "runtime.embedding_layer_count": "u:2",
        "esm.position_embedding_type": "s:absolute",
        "esm.position_id_mode": "s:padding-cumsum",
        "esm.hidden_tap": "s:last-hidden-state",
        "esm.pooling": "s:attention-mask-mean",
        "esm.special_token_policy": "s:cls-only",
        "esm.token_dropout": "b:1",
        "geneb.schema_version": "u:1",
        "geneb.suite": "s:geneb-v4",
        "geneb.catalog_contract_sha256": "s:"
        + converter.catalog_contract_sha256(base_catalog, base_catalog["models"][0]),
        "source.catalog_contract_sha256": "s:"
        + converter.catalog_contract_sha256(base_catalog, base_catalog["models"][0]),
        "source.converter_profile_contract_sha256": "s:"
        + converter.converter_profile_contract_sha256(
            1,
            converter.PROFILE_FORMAT,
            converter.load_profiles(valid["profiles"])[0]["geneb-esm-tiny"],
        ),
        "geneb.runtime_id": "s:geneb-esm-tiny",
        "geneb.model_id": "s:EsmTiny",
        "geneb.source.immutable": "b:1",
        "geneb.raw_safety_cap_bytes": "u:16777216",
        "geneb.input.fixed_pad.enabled": "b:1",
        "geneb.input.fixed_pad.length": "u:4",
        "geneb.preset.reference.pooling": "s:attention-mask-mean",
        "geneb.provenance.extractor_commit": "s:" + "b" * 40,
        "tokenizer.profile": "s:evo-tokenizer-v1",
        "tokenizer.path": "s:assets/tokenizer.json",
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise AssertionError("runtime ESM metadata differs: %s" % metadata)
    if "esm.layer_norm_kernel" in metadata:
        raise AssertionError("non-Agro fixture emitted exact LayerNorm metadata")

    if args.native_loader is not None:
        native = subprocess.run(
            [str(args.native_loader), "--verify-artifact", str(output_a)],
            check=False,
            capture_output=True,
            text=True,
        )
        if native.returncode != 0:
            raise AssertionError(
                "native ModelFile/load_artifact rejected converter output: %s"
                % native.stderr
            )
        adapter = subprocess.run(
            [str(args.native_loader), "--verify-cpu-adapter", str(output_a)],
            check=False,
            capture_output=True,
            text=True,
        )
        if adapter.returncode != 0:
            raise AssertionError(
                "native CPU Model/Context adapter rejected converter output: %s"
                % adapter.stderr
            )
        if args.c_api is not None:
            c_api = subprocess.run(
                [str(args.c_api), "--geneb-embed", str(output_a)],
                check=False,
                capture_output=True,
                text=True,
            )
            if c_api.returncode != 0:
                raise AssertionError(
                    "C ABI GENEB preset closure failed: %s" % c_api.stderr
                )
        if args.cli is not None:
            verify_cli_server_preset(args.cli, output_a, args.work_dir)
        metadata_corruptions = {
            "profile": lambda root: root["__metadata__"].update(
                {"runtime.profile": "s:wrong-profile"}
            ),
            "abi": lambda root: root["__metadata__"].update(
                {"runtime.abi": "s:wrong-abi"}
            ),
            "architecture": lambda root: root["__metadata__"].update(
                {"model.architecture": "s:WrongEsm"}
            ),
            "position": lambda root: root["__metadata__"].update(
                {"esm.position_embedding_type": "s:rotary"}
            ),
            "common-config": lambda root: root["__metadata__"].update(
                {"config.hidden_size": "u:5"}
            ),
            "embedding-layers": lambda root: root["__metadata__"].update(
                {"runtime.embedding_layer_count": "u:1"}
            ),
            "wrong-metadata-type": lambda root: root["__metadata__"].update(
                {"esm.num_layers": "s:1"}
            ),
            "unexpected-esm-key": lambda root: root["__metadata__"].update(
                {"esm.unexpected": "s:closed-namespace"}
            ),
            "missing-esm-key": lambda root: root["__metadata__"].pop(
                "esm.pooling"
            ),
            "unexpected-layernorm-kernel": lambda root: root["__metadata__"].update(
                {
                    "esm.layer_norm_kernel": "s:"
                    + converter.AGRO_LAYER_NORM_KERNEL
                }
            ),
            "missing-agro-layernorm-kernel": lambda root: root[
                "__metadata__"
            ].update({"geneb.runtime_id": "s:geneb-agro-nt-1b"}),
            "wrong-agro-layernorm-kernel": lambda root: root[
                "__metadata__"
            ].update(
                {
                    "geneb.runtime_id": "s:geneb-agro-nt-1b",
                    "esm.layer_norm_kernel": "s:portable-two-pass",
                }
            ),
            "wrong-agro-layernorm-topology": lambda root: root[
                "__metadata__"
            ].update(
                {
                    "geneb.runtime_id": "s:geneb-agro-nt-1b",
                    "esm.layer_norm_kernel": "s:"
                    + converter.AGRO_LAYER_NORM_KERNEL,
                }
            ),
        }
        for label, mutate in metadata_corruptions.items():
            corrupt = valid["root"] / ("runtime-corrupt-%s.safetensors" % label)
            rewrite_artifact_header(output_a, corrupt, mutate)
            require_native_rejected(args.native_loader, corrupt, label)

        def corrupt_tensor_shape(root: Dict[str, Any]) -> None:
            root["esm.embeddings.word_embeddings.weight"]["shape"] = [4, 8]

        corrupt_shape = valid["root"] / "runtime-corrupt-shape.safetensors"
        rewrite_artifact_header(output_a, corrupt_shape, corrupt_tensor_shape)
        require_native_rejected(args.native_loader, corrupt_shape, "tensor shape")

    bad_hash = make_case(args.work_dir / "bad-hash", converter)
    bad_hash["source_files"][-1].write_bytes(
        bad_hash["source_files"][-1].read_bytes() + b"x"
    )
    require_rejected(
        args.converter,
        bad_hash,
        bad_hash["root"] / "runtime.safetensors",
        "integrity mismatch",
    )

    extra_receipt_asset = make_case(args.work_dir / "extra-receipt-asset", converter)
    extra = extra_receipt_asset["root"] / "generation_config.json"
    extra.write_bytes(b"{}\n")
    extra_receipt_asset["source_files"].append(extra)
    refresh_receipt(extra_receipt_asset)
    require_success(
        run_converter(
            args.converter,
            extra_receipt_asset,
            extra_receipt_asset["root"] / "runtime-with-extra.safetensors",
        ),
        "verified non-critical receipt asset",
    )
    extra.write_bytes(b"corrupt after receipt\n")
    require_rejected(
        args.converter,
        extra_receipt_asset,
        extra_receipt_asset["root"] / "runtime-corrupt-extra.safetensors",
        "integrity mismatch",
    )

    bad_config = make_case(args.work_dir / "bad-config", converter)
    changed_config = config_value()
    changed_config["hidden_size"] = 6
    write_json(bad_config["config"], changed_config)
    refresh_receipt(bad_config)
    rewrite_profiles(
        bad_config,
        lambda profile: profile.update({"config_sha256": digest(bad_config["config"])}),
    )
    require_rejected(
        args.converter,
        bad_config,
        bad_config["root"] / "runtime.safetensors",
        "topology gate",
    )

    duplicate_config = make_case(args.work_dir / "duplicate-config", converter)
    write_duplicate_config(
        duplicate_config["config"], "attention_probs_dropout_prob", 0.0, 0.0
    )
    refresh_receipt(duplicate_config)
    rewrite_profiles(
        duplicate_config,
        lambda profile: profile.update(
            {
                "config_sha256": digest(duplicate_config["config"]),
                "config_duplicate_keys": {
                    "attention_probs_dropout_prob": {"count": 2, "value": 0.0}
                },
            }
        ),
    )
    require_success(
        run_converter(
            args.converter,
            duplicate_config,
            duplicate_config["root"] / "runtime.safetensors",
        ),
        "audited equal-duplicate config conversion",
    )

    undeclared_duplicate = make_case(
        args.work_dir / "undeclared-duplicate-config", converter
    )
    write_duplicate_config(
        undeclared_duplicate["config"],
        "attention_probs_dropout_prob",
        0.0,
        0.0,
    )
    refresh_receipt(undeclared_duplicate)
    rewrite_profiles(
        undeclared_duplicate,
        lambda profile: profile.update(
            {"config_sha256": digest(undeclared_duplicate["config"])}
        ),
    )
    require_rejected(
        args.converter,
        undeclared_duplicate,
        undeclared_duplicate["root"] / "runtime.safetensors",
        "duplicate key",
    )

    wrong_duplicate_count = make_case(
        args.work_dir / "wrong-duplicate-count", converter
    )
    write_duplicate_config(
        wrong_duplicate_count["config"],
        "attention_probs_dropout_prob",
        0.0,
        0.0,
    )
    refresh_receipt(wrong_duplicate_count)
    rewrite_profiles(
        wrong_duplicate_count,
        lambda profile: profile.update(
            {
                "config_sha256": digest(wrong_duplicate_count["config"]),
                "config_duplicate_keys": {
                    "attention_probs_dropout_prob": {"count": 3, "value": 0.0}
                },
            }
        ),
    )
    require_rejected(
        args.converter,
        wrong_duplicate_count,
        wrong_duplicate_count["root"] / "runtime.safetensors",
        "count/value differs",
    )

    divergent_duplicate = make_case(
        args.work_dir / "divergent-duplicate-config", converter
    )
    write_duplicate_config(
        divergent_duplicate["config"],
        "attention_probs_dropout_prob",
        0.0,
        0.5,
    )
    refresh_receipt(divergent_duplicate)
    rewrite_profiles(
        divergent_duplicate,
        lambda profile: profile.update(
            {
                "config_sha256": digest(divergent_duplicate["config"]),
                "config_duplicate_keys": {
                    "attention_probs_dropout_prob": {"count": 2, "value": 0.0}
                },
            }
        ),
    )
    require_rejected(
        args.converter,
        divergent_duplicate,
        divergent_duplicate["root"] / "runtime.safetensors",
        "count/value differs",
    )

    for corruption in ("missing", "extra", "shape", "dtype"):
        case = make_case(args.work_dir / ("bad-tensor-" + corruption), converter)
        specs = list(case["specs"])
        dtype_override = None
        if corruption == "missing":
            specs.pop(3)
        elif corruption == "extra":
            specs.append(converter.TensorSpec("esm.extra.weight", "F32", (1,)))
        elif corruption == "shape":
            original = specs[3]
            specs[3] = converter.TensorSpec(
                original.name,
                original.dtype,
                (original.shape[0] + 1,) + original.shape[1:],
            )
        else:
            dtype_override = (specs[3].name, "BF16")
        write_safetensors(case["source_files"][-1], specs, dtype_override)
        refresh_receipt(case)
        require_rejected(
            args.converter,
            case,
            case["root"] / "runtime.safetensors",
            "tensor manifest mismatch",
        )

    bad_trailing = make_case(args.work_dir / "bad-trailing", converter)
    bad_trailing["source_files"][-1].write_bytes(
        bad_trailing["source_files"][-1].read_bytes() + b"trailing"
    )
    refresh_receipt(bad_trailing)
    require_rejected(
        args.converter,
        bad_trailing,
        bad_trailing["root"] / "runtime.safetensors",
        "complete file payload",
    )

    bad_tokenizer = make_case(args.work_dir / "bad-tokenizer", converter)
    descriptor = json.loads(
        bad_tokenizer["tokenizer_descriptor"].read_text(encoding="utf-8")
    )
    descriptor["tokenizer.sha256"] = "0" * 64
    write_json(bad_tokenizer["tokenizer_descriptor"], descriptor)
    require_rejected(
        args.converter,
        bad_tokenizer,
        bad_tokenizer["root"] / "runtime.safetensors",
        "tokenizer asset integrity mismatch",
    )

    existing = valid["root"] / "existing.safetensors"
    existing.write_bytes(b"keep-me")
    rejected_existing = run_converter(args.converter, valid, existing)
    if rejected_existing.returncode == 0 or existing.read_bytes() != b"keep-me":
        raise AssertionError("converter overwrote an existing output without --force")

    torch_bad_hash = make_case(
        args.work_dir / "pytorch-bad-hash", converter, "pytorch-bin"
    )
    torch_bad_hash["source_files"][-1].write_bytes(b"tampered before torch import")
    bad_torch_result = run_converter(
        args.converter,
        torch_bad_hash,
        torch_bad_hash["root"] / "runtime.safetensors",
    )
    if (
        bad_torch_result.returncode == 0
        or "integrity mismatch" not in bad_torch_result.stderr
        or "requires offline PyTorch" in bad_torch_result.stderr
    ):
        raise AssertionError("ESM converter imported torch before receipt validation")

    torch_cases = [
        make_case(args.work_dir / "pytorch-single", converter, "pytorch-bin"),
        make_case(
            args.work_dir / "pytorch-sharded",
            converter,
            "pytorch-bin-sharded",
        ),
    ]
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        for case in torch_cases:
            require_rejected(
                args.converter,
                case,
                case["root"] / "runtime.safetensors",
                "requires offline PyTorch",
            )
    else:
        for case in torch_cases:
            specs = case["specs"]

            def tensor(spec: Any) -> Any:
                dtype = torch.int64 if spec.dtype == "I64" else torch.float32
                return torch.zeros(spec.shape, dtype=dtype)

            if case["profile"]["source_format"] == "pytorch-bin":
                state = {spec.name: tensor(spec) for spec in specs}
                torch.save(state, case["root"] / "pytorch_model.bin")
            else:
                split = len(specs) // 2
                for index, group in enumerate(
                    (specs[:split], specs[split:]), start=1
                ):
                    state = {spec.name: tensor(spec) for spec in group}
                    torch.save(
                        state,
                        case["root"]
                        / ("pytorch_model-%05d-of-00002.bin" % index),
                    )
            refresh_receipt(case)
            output = case["root"] / "runtime.safetensors"
            require_success(
                run_converter(args.converter, case, output),
                case["profile"]["source_format"] + " conversion",
            )
            _, tensors = read_artifact(output)
            if "esm.embeddings.position_ids" in tensors:
                raise AssertionError("validation-only I64 position_ids was emitted")

        def column_major_state(case: Mapping[str, Any], mode: str) -> Tuple[Dict[str, Any], Any, bytes]:
            specs = case["specs"]
            target = next(
                spec
                for spec in specs[: case["runtime_count"]]
                if spec.dtype == "F32" and len(spec.shape) == 2
            )
            state = {
                spec.name: torch.zeros(
                    spec.shape,
                    dtype=torch.int64 if spec.dtype == "I64" else torch.float32,
                )
                for spec in specs
            }
            rows, columns = target.shape
            if mode == "exact":
                matrix = torch.arange(
                    rows * columns, dtype=torch.float32
                ).reshape(columns, rows).T
            elif mode == "offset":
                matrix = torch.arange(
                    rows * columns + 1, dtype=torch.float32
                )[1:].reshape(columns, rows).T
            elif mode == "stride":
                matrix = torch.arange(
                    rows * columns * 2, dtype=torch.float32
                ).reshape(rows, columns * 2)[:, ::2]
            else:
                raise AssertionError("unknown column-major fixture mode")
            if matrix.is_contiguous():
                raise AssertionError("column-major fixture unexpectedly contiguous")
            state[target.name] = matrix
            expected = bytes(
                matrix.contiguous().view(torch.uint8).reshape(-1).numpy()
            )
            return state, target, expected

        default_layout = make_case(
            args.work_dir / "pytorch-column-major-default", converter, "pytorch-bin"
        )
        state, _, _ = column_major_state(default_layout, "exact")
        torch.save(state, default_layout["root"] / "pytorch_model.bin")
        refresh_receipt(default_layout)
        require_rejected(
            args.converter,
            default_layout,
            default_layout["root"] / "runtime.safetensors",
            "dense, contiguous",
        )

        allowed_layout = make_case(
            args.work_dir / "pytorch-column-major-allowed", converter, "pytorch-bin"
        )
        state, target, expected = column_major_state(allowed_layout, "exact")
        torch.save(state, allowed_layout["root"] / "pytorch_model.bin")
        refresh_receipt(allowed_layout)
        rewrite_profiles(
            allowed_layout,
            lambda profile: profile.update(
                {
                    "source_tensor_layout": "contiguous-or-exact-column-major"
                }
            ),
        )
        allowed_output = allowed_layout["root"] / "runtime.safetensors"
        require_success(
            run_converter(args.converter, allowed_layout, allowed_output),
            "audited exact column-major conversion",
        )
        _, allowed_tensors = read_artifact(allowed_output)
        if allowed_tensors[target.name] != expected:
            raise AssertionError(
                "column-major source was not serialized in logical row-major order"
            )

        for mode in ("offset", "stride"):
            corrupt_layout = make_case(
                args.work_dir / ("pytorch-column-major-" + mode),
                converter,
                "pytorch-bin",
            )
            state, _, _ = column_major_state(corrupt_layout, mode)
            torch.save(state, corrupt_layout["root"] / "pytorch_model.bin")
            refresh_receipt(corrupt_layout)
            rewrite_profiles(
                corrupt_layout,
                lambda profile: profile.update(
                    {
                        "source_tensor_layout": "contiguous-or-exact-column-major"
                    }
                ),
            )
            require_rejected(
                args.converter,
                corrupt_layout,
                corrupt_layout["root"] / "runtime.safetensors",
                "exact full-storage column-major",
            )

    for path in (
        args.converter,
        args.converter.parent / "evo" / "hf_checkpoint.py",
        args.converter.parent / "evo" / "geneb_artifact.py",
        Path(__file__),
    ):
        grammar = subprocess.run(
            [
                sys.executable,
                "-c",
                "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read(), "
                "filename=sys.argv[1], feature_version=(3,8))",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if grammar.returncode != 0:
            raise AssertionError("%s is not Python 3.8 grammar: %s" % (path, grammar.stderr))

    print("GENEB ESM converter contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
