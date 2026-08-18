#!/usr/bin/env python3
"""Generate an offline DNABERT-2 oracle from the pinned official remote code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer


REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
INPUTS = (
    ("input-0", b"ACGTNACGTNACGTN"),
    (
        "input-1",
        b"AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT",
    ),
)
SOURCE_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "bert_layers.py",
    "bert_padding.py",
    "config.json",
    "configuration_bert.py",
    "flash_attn_triton.py",
    "generation_config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def f32_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return array.astype("<f4", copy=False).tobytes(order="C")


def i64_bytes(values: list[int]) -> bytes:
    return b"".join(struct.pack("<q", value) for value in values)


def u8_bytes(values: list[int]) -> bytes:
    return bytes(values)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    return {
        "dtype": "float32",
        "shape": list(tensor.shape),
        "raw_little_endian_f32_sha256": sha256_bytes(f32_bytes(tensor)),
        "first_16_decimal": [float(value) for value in flat[:16]],
        "first_16_hex": [float(value).hex() for value in flat[:16]],
        "all_finite": bool(torch.isfinite(flat).all().item()),
    }


def numeric_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left64 = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right64 = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    difference = (left64 - right64).abs()
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    cosine = torch.dot(left64, right64) / denominator
    return {
        "bitwise_equal": bool(torch.equal(left, right)),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "cosine": float(cosine.item()),
    }


def package_lock() -> list[str]:
    records = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            records.append("%s==%s" % (name.lower().replace("_", "-"), distribution.version))
    return sorted(set(records))


def forward_inputs(encoded: Any) -> dict[str, torch.Tensor]:
    names = ("input_ids", "attention_mask", "token_type_ids")
    return {name: encoded[name].to(device="cpu") for name in names if name in encoded}


def token_record(tokenizer: Any, encoded: Any, row: int) -> dict[str, Any]:
    input_ids = [int(value) for value in encoded["input_ids"][row].tolist()]
    attention_mask = [int(value) for value in encoded["attention_mask"][row].tolist()]
    token_type_ids = (
        [int(value) for value in encoded["token_type_ids"][row].tolist()]
        if "token_type_ids" in encoded
        else []
    )
    special_tokens_mask = (
        [int(value) for value in encoded["special_tokens_mask"][row].tolist()]
        if "special_tokens_mask" in encoded
        else []
    )
    return {
        "input_ids": input_ids,
        "tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "special_tokens_mask": special_tokens_mask,
        "input_ids_little_endian_i64_sha256": sha256_bytes(i64_bytes(input_ids)),
        "attention_mask_u8_sha256": sha256_bytes(u8_bytes(attention_mask)),
        "effective_token_count": sum(attention_mask),
        "tensor_token_count": len(input_ids),
    }


def save_npy(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    return {
        "path": path.name,
        "file_sha256": sha256_file(path),
        **tensor_summary(tensor),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve(strict=True)
    if snapshot.name != REVISION:
        raise RuntimeError("snapshot directory does not name the pinned revision")
    output = args.output.resolve()
    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError("%s=1 is required" % variable)

    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    source_manifest = {}
    for name in SOURCE_FILES:
        path = snapshot / name
        source_manifest[name] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        trust_remote_code=True,
        local_files_only=True,
        revision=REVISION,
    )
    config = AutoConfig.from_pretrained(
        str(snapshot),
        trust_remote_code=True,
        local_files_only=True,
        revision=REVISION,
    )
    model, loading_info = AutoModel.from_pretrained(
        str(snapshot),
        trust_remote_code=True,
        local_files_only=True,
        revision=REVISION,
        output_loading_info=True,
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()

    module = sys.modules[model.__class__.__module__]
    module_path = Path(inspect.getsourcefile(model.__class__) or "")
    flash_function = getattr(module, "flash_attn_qkvpacked_func", "not-defined")
    if flash_function is not None:
        raise RuntimeError("oracle unexpectedly selected the Triton attention path")
    if REVISION not in model.__class__.__module__:
        raise RuntimeError("model class was not loaded from the pinned dynamic module")

    environment = {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_lock(),
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mps_available_but_unused": bool(torch.backends.mps.is_available()),
        },
        "transformers": {
            "version": transformers.__version__,
            "offline": os.environ["TRANSFORMERS_OFFLINE"],
            "hub_offline": os.environ["HF_HUB_OFFLINE"],
            "trust_remote_code": True,
            "local_files_only": True,
            "revision": REVISION,
            "config_class": "%s.%s" % (config.__class__.__module__, config.__class__.__name__),
            "model_class": "%s.%s" % (model.__class__.__module__, model.__class__.__name__),
            "remote_module_copy_sha256": sha256_file(module_path),
            "attention_implementation": "official MosaicBERT PyTorch fallback",
            "flash_attn_qkvpacked_func": None,
        },
        "source_files": source_manifest,
    }
    environment_path = output / "environment-lock.json"
    write_json(environment_path, environment)

    tokenizer_contract = {
        "class": "%s.%s" % (tokenizer.__class__.__module__, tokenizer.__class__.__name__),
        "is_fast": bool(tokenizer.is_fast),
        "vocab_size": int(tokenizer.vocab_size),
        "model_max_length_reported_by_tokenizer": int(tokenizer.model_max_length),
        "pinned_extractor_max_length": 512,
        "truncation": True,
        "token_truncation_side": tokenizer.truncation_side,
        "single_sequence_padding": False,
        "reference_batch_padding": "longest sequence in batch",
        "padding_side": tokenizer.padding_side,
        "add_special_tokens": "HF default true (argument omitted)",
        "special_tokens_map": dict(tokenizer.special_tokens_map),
        "special_token_ids": {
            "unk": int(tokenizer.unk_token_id),
            "cls": int(tokenizer.cls_token_id),
            "sep": int(tokenizer.sep_token_id),
            "pad": int(tokenizer.pad_token_id),
            "mask": int(tokenizer.mask_token_id),
        },
    }

    records = []
    single_hidden = []
    single_pooled = []
    for label, input_bytes in INPUTS:
        input_path = output / (label + ".txt")
        input_path.write_bytes(input_bytes)
        sequence = input_bytes.decode("ascii")
        encoded = tokenizer(
            sequence,
            max_length=512,
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        inputs = forward_inputs(encoded)
        with torch.inference_mode():
            hidden_first = model(**inputs)[0].to(dtype=torch.float32)
            hidden_second = model(**inputs)[0].to(dtype=torch.float32)
        mask = inputs["attention_mask"].unsqueeze(-1).to(dtype=torch.float32)
        pooled_first = (hidden_first * mask).sum(dim=1) / mask.sum(dim=1)
        pooled_second = (hidden_second * mask).sum(dim=1) / mask.sum(dim=1)
        if not torch.equal(hidden_first, hidden_second):
            raise RuntimeError("repeated official forward is not bitwise deterministic for " + label)
        if not torch.equal(pooled_first, pooled_second):
            raise RuntimeError("repeated official pooling is not bitwise deterministic for " + label)

        hidden_path = output / (label + ".last-hidden-state.f32.npy")
        pooled_path = output / (label + ".attention-mask-mean.f32.npy")
        token_path = output / (label + ".tokens.json")
        tokens = token_record(tokenizer, encoded, 0)
        write_json(token_path, tokens)
        hidden_evidence = save_npy(hidden_path, hidden_first)
        pooled_evidence = save_npy(pooled_path, pooled_first)
        records.append(
            {
                "label": label,
                "input": {
                    "path": input_path.name,
                    "byte_count": len(input_bytes),
                    "sha256": sha256_bytes(input_bytes),
                    "ascii": sequence,
                },
                "tokens": {"path": token_path.name, "file_sha256": sha256_file(token_path), **tokens},
                "hidden": hidden_evidence,
                "embedding": pooled_evidence,
                "repeat_forward": numeric_comparison(hidden_first, hidden_second),
                "repeat_embedding": numeric_comparison(pooled_first, pooled_second),
            }
        )
        single_hidden.append(hidden_first)
        single_pooled.append(pooled_first)

    sequences = [value.decode("ascii") for _, value in INPUTS]
    batched = tokenizer(
        sequences,
        max_length=512,
        truncation=True,
        padding=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    batch_inputs = forward_inputs(batched)
    with torch.inference_mode():
        batch_hidden = model(**batch_inputs)[0].to(dtype=torch.float32)
    batch_mask = batch_inputs["attention_mask"].unsqueeze(-1).to(dtype=torch.float32)
    batch_pooled = (batch_hidden * batch_mask).sum(dim=1) / batch_mask.sum(dim=1)
    batch_token_records = [token_record(tokenizer, batched, index) for index in range(len(INPUTS))]
    batch_token_path = output / "padding-diagnostic.tokens.json"
    write_json(batch_token_path, batch_token_records)
    batch_embedding_path = output / "padding-diagnostic.attention-mask-mean.f32.npy"
    batch_embedding_evidence = save_npy(batch_embedding_path, batch_pooled)

    batch_comparisons = []
    for index, record in enumerate(batch_token_records):
        effective = record["effective_token_count"]
        if record["input_ids"][:effective] != records[index]["tokens"]["input_ids"]:
            raise RuntimeError("batch token prefix differs from single-sequence tokenization")
        if record["attention_mask"][:effective] != records[index]["tokens"]["attention_mask"]:
            raise RuntimeError("batch attention-mask prefix differs from single sequence")
        batch_comparisons.append(
            {
                "label": INPUTS[index][0],
                "single_vs_padded_batch_embedding": numeric_comparison(
                    single_pooled[index][0], batch_pooled[index]
                ),
                "single_vs_padded_batch_effective_hidden": numeric_comparison(
                    single_hidden[index][0], batch_hidden[index, :effective]
                ),
                "padding_token_count": record["tensor_token_count"] - effective,
                "padded_hidden_max_abs": (
                    float(batch_hidden[index, effective:].abs().max().item())
                    if effective < record["tensor_token_count"]
                    else 0.0
                ),
            }
        )

    loading_info_record = {
        key: sorted(str(value) for value in loading_info.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    report = {
        "schema_version": 1,
        "oracle_identity": {
            "model": "zhihan1996/DNABERT-2-117M",
            "revision": REVISION,
            "source": "pinned local Hugging Face snapshot",
            "implementation": "official repository remote modeling code via trust_remote_code=True",
            "independent_of_evo_native_runtime": True,
            "checkpoint_sha256": source_manifest["pytorch_model.bin"]["sha256"],
            "checkpoint_size": source_manifest["pytorch_model.bin"]["size"],
            "environment_lock": environment_path.name,
            "environment_lock_sha256": sha256_file(environment_path),
            "generator": "tools/generate_geneb_dnabert2_upstream_oracle.py",
            "generator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "execution": {
            "device": "cpu",
            "dtype": "float32",
            "batching_for_canonical_embeddings": "one sequence per forward",
            "hidden_tap": "last-hidden-state",
            "pooling": "attention-mask-mean",
            "special_tokens": "included where attention_mask=1",
            "max_length": 512,
            "model_eval": True,
            "inference_mode": True,
            "official_attention_path": "PyTorch fallback because Triton import is unavailable",
        },
        "tokenizer": tokenizer_contract,
        "model": {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "config_hidden_size": int(config.hidden_size),
            "config_num_hidden_layers": int(config.num_hidden_layers),
            "config_num_attention_heads": int(config.num_attention_heads),
            "config_vocab_size": int(config.vocab_size),
            "config_torch_dtype": str(config.torch_dtype),
            "loading_info": loading_info_record,
        },
        "runner_ready_oracle_vectors": [
            label + ".independent-oracle-vector.json"
            for label, _ in INPUTS
        ],
        "canonical_single_sequence_records": records,
        "padding_diagnostic": {
            "purpose": "Expose right batch-max padding while proving masked pooling semantics; not used as the canonical oracle output.",
            "shape": list(batch_hidden.shape),
            "tokens": {
                "path": batch_token_path.name,
                "file_sha256": sha256_file(batch_token_path),
                "records": batch_token_records,
            },
            "embeddings": batch_embedding_evidence,
            "comparisons": batch_comparisons,
        },
    }
    report_path = output / "oracle-report.json"
    write_json(report_path, report)

    for index, (label, _) in enumerate(INPUTS):
        vector_path = output / (label + ".independent-oracle-vector.json")
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": "geneb-dnabert-2",
            "input_sha256": records[index]["input"]["sha256"],
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": [
                float(value)
                for value in single_pooled[index]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .view(-1)
            ],
            "environment_lock": environment,
            "provenance": {
                "source_repo": "zhihan1996/DNABERT-2-117M",
                "source_revision": REVISION,
                "checkpoint_sha256": source_manifest["pytorch_model.bin"]["sha256"],
                "official_remote_modeling_code": True,
                "trust_remote_code": True,
                "independent_of_evo_native_runtime": True,
                "oracle_report_sha256": sha256_file(report_path),
                "generator_sha256": sha256_file(Path(__file__).resolve()),
            },
        }
        write_json(vector_path, vector)

    artifact_paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "artifact-manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "files": [
            {"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_paths
        ],
    }
    manifest_path = output / "artifact-manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "report": report_path.name,
                "report_sha256": sha256_file(report_path),
                "manifest": manifest_path.name,
                "manifest_sha256": sha256_file(manifest_path),
                "record_embeddings": [record["embedding"] for record in records],
                "padding_comparisons": batch_comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
