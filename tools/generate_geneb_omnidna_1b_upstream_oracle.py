#!/usr/bin/env python3
"""Generate two independent, pinned Omni-DNA-1B GENEB CPU oracles."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


RUNTIME_ID = "geneb-omni-dna-1b"
REPO = "zehui127/Omni-DNA-1B"
REVISION = "0ea9d54e356b4e7354dc40e2d980e9ebcb2ccfd5"
CATALOG_CONTRACT_SHA256 = (
    "67646a96e22fc03c4df917ee8b03725cd581d4e1c36535c2843ac02391dacce5"
)
SOURCE_FILES = {
    "config.json": (
        1_311,
        "41560210c06cb75cec5f38adcd32d901829868a1fe5ade35d6dc94b3087a8ac8",
    ),
    "configuration_olmo.py": (
        1_213,
        "6e23d2c8ae0d9420670ee9a00de279d04e93c86d1a6b555dd62c1b49ecae12a5",
    ),
    "model.safetensors": (
        4_328_529_472,
        "12cba5f3b0c533801c7e05ee3ac93bf33f1557ac36cd11099e2e5ca1ca5ff5ad",
    ),
    "modeling_olmo.py": (
        25_885,
        "8e54a1c85cffe7eb9049549e2213d087cdfe56db4f7b0ae211171f98aeec1c21",
    ),
    "special_tokens_map.json": (
        695,
        "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    ),
    "tokenizer.json": (
        286_487,
        "c02de8344eee229f6355e3de63d18fc0d4110f276320774ff891d6a5f81d817b",
    ),
    "tokenizer_config.json": (
        1_167,
        "24f9d93d3a936bc05de25b7812a17c00d407d06df9c06227a067afaad7fcba0d",
    ),
}
PACKAGE_LOCK = {
    "ai2-olmo": "0.6.0",
    "datasets": "2.20.0",
    "numpy": "1.26.4",
    "safetensors": "0.4.5",
    "tokenizers": "0.20.3",
    "torch": "2.1.2",
    "transformers": "4.46.3",
}
SEQUENCES = (
    "ACGTNACGTNACGTN",
    "TTGCAANNTGCAACGT",
)
CONFIG_LOCK = {
    "activation_type": "swiglu",
    "alibi": False,
    "attention_dropout": 0.0,
    "attention_layer_norm": False,
    "attention_layer_norm_with_affine": False,
    "bias_for_layer_norm": False,
    "block_group_size": 1,
    "block_type": "sequential",
    "clip_qkv": None,
    "d_model": 2_048,
    "embedding_dropout": 0.0,
    "embedding_size": 4_096,
    "flash_attention": False,
    "include_bias": False,
    "layer_norm_type": "default",
    "layer_norm_with_affine": False,
    "max_sequence_length": 250,
    "mlp_hidden_size": None,
    "mlp_ratio": 8,
    "model_type": "olmo-gfm",
    "multi_query_attention": False,
    "n_heads": 16,
    "n_kv_heads": None,
    "n_layers": 16,
    "precision": "amp_bf16",
    "residual_dropout": 0.0,
    "rope": True,
    "rope_full_precision": True,
    "scale_logits": False,
    "vocab_size": 4_096,
    "weight_tying": True,
}


class OracleError(ValueError):
    """Raised when the pinned upstream contract differs."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise OracleError(f"JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=duplicate_checked_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} root must be an object")
    return value, payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, size: int, digest: str, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size != size:
        raise OracleError(f"{label} size differs")
    if sha256_file(resolved) != digest:
        raise OracleError(f"{label} SHA256 differs")
    return resolved


def validate_source_receipt(path: Path) -> tuple[dict[str, Path], str]:
    receipt, payload = read_json(path, "source receipt")
    expected_keys = {
        "catalog_contract_sha256",
        "catalog_path",
        "files",
        "kind",
        "load_path",
        "model_id",
        "repo",
        "requested_revision",
        "resolved_revision",
        "schema_version",
        "source_kind",
    }
    if set(receipt) != expected_keys:
        raise OracleError("source receipt fields differ")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["source_kind"] != "huggingface"
        or receipt["model_id"] != RUNTIME_ID
        or receipt["repo"] != REPO
        or receipt["requested_revision"] != "main"
        or receipt["resolved_revision"] != REVISION
        or receipt["load_path"] is not None
        or receipt["catalog_contract_sha256"] != CATALOG_CONTRACT_SHA256
    ):
        raise OracleError("source receipt identity differs")
    files = receipt["files"]
    if not isinstance(files, list) or len(files) != len(SOURCE_FILES):
        raise OracleError("source receipt file count differs")
    paths: dict[str, Path] = {}
    for raw in files:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "path",
            "sha256",
            "size",
        }:
            raise OracleError("source receipt file entry differs")
        name = raw["name"]
        if not isinstance(name, str) or name in paths or name not in SOURCE_FILES:
            raise OracleError("source receipt file name differs")
        size, digest = SOURCE_FILES[name]
        if raw["size"] != size or raw["sha256"] != digest:
            raise OracleError(f"source receipt identity differs for {name}")
        paths[name] = checked_file(Path(raw["path"]), size, digest, name)
    if set(paths) != set(SOURCE_FILES):
        raise OracleError("source receipt file set differs")
    return paths, hashlib.sha256(payload).hexdigest()


def validate_snapshot(snapshot: Path, receipt_paths: dict[str, Path]) -> dict[str, Any]:
    snapshot = snapshot.expanduser().resolve(strict=True)
    if snapshot.name != REVISION or not snapshot.is_dir():
        raise OracleError("snapshot revision differs")
    entries = sorted(path.name for path in snapshot.iterdir() if path.is_file())
    if entries != sorted(SOURCE_FILES):
        raise OracleError("snapshot file allowlist differs")
    manifest: dict[str, Any] = {}
    for name, (size, digest) in SOURCE_FILES.items():
        path = checked_file(snapshot / name, size, digest, f"snapshot {name}")
        if path != receipt_paths[name]:
            if path.stat().st_ino != receipt_paths[name].stat().st_ino:
                raise OracleError(f"snapshot/receipt locator differs for {name}")
        manifest[name] = {"size": size, "sha256": digest}
    return manifest


def package_environment() -> tuple[list[str], dict[str, str]]:
    locked: dict[str, str] = {}
    for name, expected in PACKAGE_LOCK.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise OracleError(f"package {name} differs: {actual} != {expected}")
        locked[name] = actual
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            packages.append(
                f"{name.lower().replace('_', '-')}=={distribution.version}"
            )
    return sorted(set(packages)), dict(sorted(locked.items()))


def expected_tensor_shapes() -> dict[str, tuple[int, ...]]:
    result = {"model.transformer.wte.weight": (4_096, 2_048)}
    for layer in range(16):
        prefix = f"model.transformer.blocks.{layer}."
        result.update(
            {
                prefix + "att_proj.weight": (6_144, 2_048),
                prefix + "attn_out.weight": (2_048, 2_048),
                prefix + "ff_out.weight": (2_048, 8_192),
                prefix + "ff_proj.weight": (16_384, 2_048),
            }
        )
    return result


def validate_checkpoint_header(checkpoint: Path) -> str:
    expected = expected_tensor_shapes()
    manifest = []
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        keys = sorted(source.keys())
        if keys != sorted(expected):
            raise OracleError("checkpoint tensor names differ")
        for name in keys:
            tensor = source.get_tensor(name)
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != expected[name]:
                raise OracleError(f"checkpoint tensor contract differs for {name}")
            manifest.append(
                {"name": name, "dtype": "F32", "shape": list(expected[name])}
            )
            del tensor
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def validate_config(config: Any) -> None:
    wrong = {
        key: (getattr(config, key, None), expected)
        for key, expected in CONFIG_LOCK.items()
        if getattr(config, key, None) != expected
    }
    if wrong:
        raise OracleError(f"official config differs: {wrong}")


def encode_case(tokenizer: Any, sequence: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [sequence],
        padding=True,
        truncation=True,
        max_length=250,
        return_tensors="pt",
    )
    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise OracleError("tokenizer output fields differ")
    if encoded["input_ids"].shape != encoded["attention_mask"].shape:
        raise OracleError("tokenizer input/mask shapes differ")
    return dict(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise OracleError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_MODULES_CACHE"] = str(output / "hf-modules")
    sys.dont_write_bytecode = True
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    receipt_paths, receipt_sha256 = validate_source_receipt(
        args.source_receipt.expanduser().resolve(strict=True)
    )
    snapshot = args.snapshot.expanduser().resolve(strict=True)
    source_manifest = validate_snapshot(snapshot, receipt_paths)
    packages, locked_packages = package_environment()
    checkpoint = receipt_paths["model.safetensors"]
    tensor_manifest_sha256 = validate_checkpoint_header(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=True,
    )
    validate_config(config)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    state = load_file(str(checkpoint), device="cpu")
    load = model.load_state_dict(state, strict=False)
    if list(load.missing_keys) != ["word_embeddings.weight"] or load.unexpected_keys:
        raise OracleError(f"checkpoint load contract differs: {load!r}")
    del state
    gc.collect()
    if model.word_embeddings is not model.model.transformer.wte:
        raise OracleError("word_embeddings/wte module alias is absent")
    model.to(device="cpu", dtype=torch.float32).eval()

    generator = Path(__file__).resolve(strict=True)
    generator_sha256 = sha256_file(generator)
    source_manifest_sha256 = hashlib.sha256(
        canonical_json(source_manifest)
    ).hexdigest()
    environment = {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "dont_write_bytecode": sys.dont_write_bytecode,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "package_lock": locked_packages,
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "transformers": {
            "version": importlib.metadata.version("transformers"),
            "revision": REVISION,
            "trust_remote_code": True,
            "local_files_only": True,
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "config_class": f"{config.__class__.__module__}.{config.__class__.__name__}",
            "load_patch": (
                "from_config + verified Safetensors state_dict; permit only tied "
                "word_embeddings.weight alias"
            ),
        },
        "source_files": source_manifest,
    }
    provenance = {
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_receipt_sha256": receipt_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "checkpoint_sha256": SOURCE_FILES["model.safetensors"][1],
        "checkpoint_tensor_manifest_sha256": tensor_manifest_sha256,
        "checkpoint_tensor_count": 65,
        "generator_sha256": generator_sha256,
        "independent_of_evo_native_runtime": True,
        "official_remote_modeling_code": True,
        "datasets_dependency_basis": (
            "Omni-DNA model card explicitly requires datasets; ai2-olmo 0.6.0 "
            "imports it at package import time but declares it only in the train extra"
        ),
        "trust_remote_code": True,
        "state_dict_missing_keys": ["word_embeddings.weight"],
        "state_dict_unexpected_keys": [],
        "weight_alias": "word_embeddings.weight=model.transformer.wte.weight",
        "pooling": "F32 masked sum followed by direct division",
    }

    cases = []
    for index, sequence in enumerate(SEQUENCES):
        if not sequence or any(base not in "ACGTN" for base in sequence):
            raise OracleError("canonical sequence alphabet differs")
        input_path = output / f"input-{index}.txt"
        input_path.write_bytes(sequence.encode("ascii"))
        input_sha256 = sha256_file(input_path)
        encoded = encode_case(tokenizer, sequence)
        with torch.inference_mode():
            result = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            if result.hidden_states is None or len(result.hidden_states) != 17:
                raise OracleError("hidden-state tap count differs")
            hidden = result.hidden_states[-1]
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != 2_048:
                raise OracleError("last hidden-state shape differs")
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        values = pooled.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        if values.shape != (1, 2_048) or not np.isfinite(values).all():
            raise OracleError("pooled oracle output shape/value differs")
        npy_path = output / f"input-{index}.attention-mask-mean.f32.npy"
        np.save(npy_path, values, allow_pickle=False)
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": RUNTIME_ID,
            "input_sha256": input_sha256,
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": [float(value) for value in values.reshape(-1)],
            "environment_lock": environment,
            "provenance": provenance,
        }
        vector_path = output / f"input-{index}.independent-oracle-vector.json"
        vector_path.write_bytes(canonical_json(vector))
        cases.append(
            {
                "index": index,
                "sequence": sequence,
                "input_size": input_path.stat().st_size,
                "input_sha256": input_sha256,
                "input_ids": encoded["input_ids"].tolist(),
                "attention_mask": encoded["attention_mask"].tolist(),
                "model_input_fields": sorted(encoded),
                "shape": list(values.shape),
                "npy_sha256": sha256_file(npy_path),
                "raw_f32_sha256": hashlib.sha256(
                    values.tobytes(order="C")
                ).hexdigest(),
                "vector_sha256": sha256_file(vector_path),
                "first_16": vector["values"][:16],
            }
        )
    literal_probes = {}
    for text in ("[MASK]A", "[PAD]A"):
        literal_probes[text] = tokenizer(
            text, add_special_tokens=True, truncation=False
        )["input_ids"]
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "receipt_sha256": receipt_sha256,
            "manifest_sha256": source_manifest_sha256,
        },
        "cases": cases,
        "tokenizer_literal_probes": literal_probes,
        "environment_lock": environment,
        "provenance": provenance,
    }
    report_path = output / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))
    summary = {
        "report_sha256": sha256_file(report_path),
        "cases": [
            {
                "index": case["index"],
                "input_sha256": case["input_sha256"],
                "input_ids": case["input_ids"],
                "npy_sha256": case["npy_sha256"],
                "vector_sha256": case["vector_sha256"],
                "first_16": case["first_16"],
            }
            for case in cases
        ],
        "tokenizer_literal_probes": literal_probes,
        "tensor_manifest_sha256": tensor_manifest_sha256,
    }
    print(canonical_json(summary).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OracleError, OSError, RuntimeError, ValueError) as error:
        print(
            f"generate_geneb_omnidna_1b_upstream_oracle: error: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2)
