#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate an independent pinned GROVER oracle with GENEB extractor semantics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer


REPO = "PoetschLab/GROVER"
REVISION = "6b223110f0d6963e849f55bc2a2f3cff0e38c7a4"
EXTRACTOR_REPO = "ultimativity/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = "6254c11a79d7d57e217458ebd1e3fab76e94cc514d3687d77076b87c3f7a17df"
INPUTS = (
    ("input-0", b"ACGTNACGTNACGTN"),
    ("input-1", b"AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT"),
)
SOURCE_FILES = {
    ".gitattributes": (
        1519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "README.md": (
        1404,
        "4d44860d1c0bd4caded11fb4fce1510d30260e3345d7e2677252085811fa4be8",
    ),
    "config.json": (
        1109,
        "de60db6f5cb476d9f38630517abec87803e32a87672ace7946b5cf82f68ac5de",
    ),
    "pytorch_model.bin": (
        348488361,
        "81fa4ee056e65ff1c06c32512589f3ba3bcb9d054960f35f634b9cf6da7aadda",
    ),
    "special_tokens_map.json": (
        77,
        "894589f317dfaaa10cbb6e467380b89bb848fe99f60ff6863ce120d81d71eff4",
    ),
    "tokenizer.json": (
        24637,
        "86b51bd8ecfc52cdc04f9f6787a2bd64fc5784709e7fd988b82d2e97348f9afd",
    ),
    "tokenizer_config.json": (
        314,
        "8c6c05d4a32b1235387c4d2a546bce640d7889f74d4b8cc6c8da549641e97517",
    ),
    "vocab.txt": (
        3848,
        "affab9041e04e027bbcd359648e0fbc8af1610e58a16dc7954a81fd4b28f4e76",
    ),
}


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            portable(key, label + " key")
            portable(item, "%s.%s" % (label, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            portable(item, "%s[%d]" % (label, index))
        return
    if not isinstance(value, str):
        return
    parsed = urlsplit(value)
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or parsed.scheme.lower() == "file"
    ):
        raise RuntimeError(label + " contains a local absolute filesystem path")


def f32_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return array.astype("<f4", copy=False).tobytes(order="C")


def i64_bytes(values: list[int]) -> bytes:
    return b"".join(struct.pack("<q", value) for value in values)


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
    return {
        "bitwise_equal": bool(torch.equal(left, right)),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "cosine": float((torch.dot(left64, right64) / denominator).item()),
    }


def package_lock() -> list[str]:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append("%s==%s" % (name.lower().replace("_", "-"), distribution.version))
    return sorted(set(packages))


def save_npy(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    return {"file": path.name, "file_sha256": sha256_file(path), **tensor_summary(tensor)}


def validate_sources(
    snapshot_argument: Path, extractor_argument: Path
) -> tuple[Path, Path, dict[str, Any]]:
    snapshot = snapshot_argument.resolve(strict=True)
    extractor = extractor_argument.resolve(strict=True)
    if snapshot.name != REVISION:
        raise RuntimeError("GROVER snapshot directory does not name the pinned revision")
    source_manifest = {}
    for name, (expected_size, expected_digest) in SOURCE_FILES.items():
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError("missing pinned GROVER source file: " + name)
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected_size or digest != expected_digest:
            raise RuntimeError("pinned GROVER source file differs: " + name)
        source_manifest[name] = {"size": size, "sha256": digest}
    if sha256_file(extractor) != EXTRACTOR_SHA256:
        raise RuntimeError("pinned GENEB GROVER extractor differs")
    return snapshot, extractor, source_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    snapshot, extractor, source_manifest = validate_sources(args.snapshot, args.extractor)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(variable + "=1 is required")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    model, loading_info = AutoModelForMaskedLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        output_hidden_states=True,
        output_loading_info=True,
    )
    model.to(device="cpu", dtype=torch.float32).eval()
    model_name = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    tokenizer_name = "%s.%s" % (
        tokenizer.__class__.__module__,
        tokenizer.__class__.__name__,
    )
    if model_name != "transformers.models.bert.modeling_bert.BertForMaskedLM":
        raise RuntimeError("GROVER did not select the official Transformers BERT")
    if tokenizer_name != "transformers.tokenization_utils_fast.PreTrainedTokenizerFast":
        raise RuntimeError("GROVER tokenizer class differs")
    if int(tokenizer.model_max_length) != 512 or tokenizer.padding_side != "right":
        raise RuntimeError("GROVER tokenizer length/padding contract differs")
    if int(tokenizer.vocab_size) != 610 or model.get_input_embeddings().weight.shape[0] != 609:
        raise RuntimeError("GROVER tokenizer/checkpoint vocabulary audit differs")
    loading = {
        key: sorted(str(value) for value in loading_info.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if (
        loading["missing_keys"]
        or loading["mismatched_keys"]
        or loading["error_msgs"]
        or loading["unexpected_keys"]
        != ["bert.pooler.dense.bias", "bert.pooler.dense.weight"]
    ):
        raise RuntimeError("GROVER official loading contract differs: %r" % loading)

    environment_lock = {
        "schema_version": 1,
        "oracle_contract": "geneb-independent-oracle-v1",
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
            "offline": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "model_class": model_name,
            "tokenizer_class": tokenizer_name,
        },
        "source_files": source_manifest,
    }
    portable(environment_lock, "environment_lock")
    environment_path = output / "environment-lock.json"
    environment_path.write_bytes(canonical_json(environment_lock))

    records = []
    pooled_values = []
    for label, input_bytes in INPUTS:
        sequence = input_bytes.decode("ascii")
        input_path = output / (label + ".txt")
        input_path.write_bytes(input_bytes)
        encoded = tokenizer(
            [sequence],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        evidence_encoding = tokenizer(
            [sequence],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_special_tokens_mask=True,
        )
        for name in ("input_ids", "token_type_ids", "attention_mask"):
            if name in encoded and not torch.equal(encoded[name], evidence_encoding[name]):
                raise RuntimeError("token evidence call changed " + name)
        cpu_inputs = {name: value.to(device="cpu") for name, value in encoded.items()}
        with torch.no_grad():
            output_first = model(**cpu_inputs)
            output_second = model(**cpu_inputs)
        hidden_first = output_first.hidden_states[-1].to(dtype=torch.float32)
        hidden_second = output_second.hidden_states[-1].to(dtype=torch.float32)
        mask = cpu_inputs["attention_mask"].unsqueeze(-1)
        pooled_first = (hidden_first * mask).sum(dim=1) / mask.sum(dim=1)
        pooled_second = (hidden_second * mask).sum(dim=1) / mask.sum(dim=1)
        if not torch.equal(hidden_first, hidden_second) or not torch.equal(
            pooled_first, pooled_second
        ):
            raise RuntimeError("GROVER upstream forward is not bitwise deterministic")

        ids = [int(value) for value in cpu_inputs["input_ids"][0].tolist()]
        attention_mask = [int(value) for value in cpu_inputs["attention_mask"][0].tolist()]
        token_type_ids = [int(value) for value in cpu_inputs["token_type_ids"][0].tolist()]
        special_mask = [
            int(value) for value in evidence_encoding["special_tokens_mask"][0].tolist()
        ]
        if 609 in ids:
            raise RuntimeError("official tokenizer emitted audited unreachable ID 609")
        tokens = {
            "input_ids": ids,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "special_tokens_mask": special_mask,
            "input_ids_little_endian_i64_sha256": sha256_bytes(i64_bytes(ids)),
            "attention_mask_u8_sha256": sha256_bytes(bytes(attention_mask)),
            "effective_token_count": sum(attention_mask),
            "tensor_token_count": len(ids),
        }
        token_path = output / (label + ".tokens.json")
        token_path.write_bytes(canonical_json(tokens))
        hidden_path = output / (label + ".last-hidden-state.f32.npy")
        embedding_path = output / (label + ".attention-mask-mean.f32.npy")
        record = {
            "label": label,
            "input": {
                "file": input_path.name,
                "size": len(input_bytes),
                "sha256": sha256_bytes(input_bytes),
                "ascii": sequence,
            },
            "tokens": {"file": token_path.name, "file_sha256": sha256_file(token_path), **tokens},
            "hidden": save_npy(hidden_path, hidden_first),
            "embedding": save_npy(embedding_path, pooled_first),
            "repeat_hidden": numeric_comparison(hidden_first, hidden_second),
            "repeat_embedding": numeric_comparison(pooled_first, pooled_second),
        }
        records.append(record)
        pooled_values.append(pooled_first)

    generator_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    report = {
        "schema_version": 1,
        "kind": "geneb-grover-official-oracle-report",
        "source": {"repo": REPO, "revision": REVISION, "files": source_manifest},
        "extractor": {
            "repo": EXTRACTOR_REPO,
            "revision": EXTRACTOR_REVISION,
            "file": "embedding_pipeline/extractors/grover.py",
            "sha256": sha256_file(extractor),
        },
        "execution": {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "hidden_tap": "outputs.hidden_states[-1]",
            "pooling": "attention-mask-mean",
            "model_eval": True,
            "torch_no_grad": True,
            "independent_of_evo_native_runtime": True,
        },
        "tokenizer": {
            "source_vocab_size": int(tokenizer.vocab_size),
            "model_embedding_rows": int(model.get_input_embeddings().weight.shape[0]),
            "audited_unreachable_id": 609,
            "official_emitted_unreachable_id": False,
            "model_max_length": int(tokenizer.model_max_length),
            "padding_side": tokenizer.padding_side,
        },
        "model": {
            "class": model_name,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "loading_info": loading,
        },
        "environment_lock_sha256": sha256_file(environment_path),
        "generator_sha256": generator_sha256,
        "records": records,
    }
    report_path = output / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))

    for index, (label, _) in enumerate(INPUTS):
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": "geneb-grover",
            "input_sha256": records[index]["input"]["sha256"],
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": [float(value) for value in pooled_values[index].contiguous().view(-1)],
            "environment_lock": environment_lock,
            "provenance": {
                "oracle_contract": "geneb-independent-oracle-v1",
                "source_repo": REPO,
                "source_revision": REVISION,
                "checkpoint_sha256": source_manifest["pytorch_model.bin"]["sha256"],
                "tokenizer_sha256": source_manifest["tokenizer.json"]["sha256"],
                "extractor_repo": EXTRACTOR_REPO,
                "extractor_revision": EXTRACTOR_REVISION,
                "extractor_file": "embedding_pipeline/extractors/grover.py",
                "extractor_sha256": sha256_file(extractor),
                "official_hf_modeling_code": True,
                "official_remote_modeling_code": False,
                "trust_remote_code": False,
                "model_class": model_name,
                "batch_size": 1,
                "hidden_tap": "outputs.hidden_states[-1]",
                "pooling": "attention-mask-mean",
                "independent_of_evo_native_runtime": True,
                "generator_sha256": generator_sha256,
                "oracle_report_sha256": sha256_file(report_path),
            },
        }
        portable(vector, "oracle vector")
        (output / (label + ".independent-oracle-vector.json")).write_bytes(
            canonical_json(vector)
        )

    files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact-manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "files": [
            {"file": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
    }
    manifest_path = output / "artifact-manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    print(
        json.dumps(
            {
                "environment_lock_sha256": sha256_file(environment_path),
                "generator_sha256": generator_sha256,
                "manifest_sha256": sha256_file(manifest_path),
                "records": [
                    {
                        "label": record["label"],
                        "input_sha256": record["input"]["sha256"],
                        "input_ids": record["tokens"]["input_ids"],
                        "embedding_sha256": record["embedding"][
                            "raw_little_endian_f32_sha256"
                        ],
                    }
                    for record in records
                ],
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
