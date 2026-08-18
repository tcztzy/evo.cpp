#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a pinned intended-Mosaic DNABERT-S normalized oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import struct
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer


REPO = "zhihan1996/DNABERT-S"
REVISION = "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"
EXTRACTOR_REPO = "ultimativity/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = "d6214d3a34b7ab7e4becc349dc123d91d4f6c36f96556870735ff6d8958d3691"
EXTRACTOR_PYPROJECT_SHA256 = (
    "616a15c0a751eb14687fefe691cd624fbf4cc104605f167a5782cb29d16060e0"
)
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
        54,
        "d58d735f6173814aab198c1e7e0f620809f85e37e0af8dab1330329692075e98",
    ),
    "bert_layers.py": (
        40807,
        "cd2fe0efbb90db624fe4e25aa96bd049513e81740ba3f980163c385e894df561",
    ),
    "bert_padding.py": (
        6099,
        "44d1c68afb1f585fdc66c150d4c60f1ed44a89c006abc57d50531d71940d7421",
    ),
    "config.json": (
        1025,
        "d54c142114563aa34ef7c2b54ccc921d08e50cd7c3cd69c439d5e0d2bbcfcdee",
    ),
    "configuration_bert.py": (
        1017,
        "d030166bd279ca81e9d2304eaff9419171f2f8c69ab419e7cd37873b72f47a22",
    ),
    "flash_attn_triton.py": (
        42737,
        "568d1ac3beca0b5e1df528a1f136aa19b6489a616fcf3784f33336a50bb1de81",
    ),
    "pytorch_model.bin": (
        468320053,
        "f3cfc3d0541859df64759e758cc4bc40fe1d5760799a7d098a9f9f09501c1e5f",
    ),
    "special_tokens_map.json": (
        125,
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    ),
    "tokenizer.json": (
        168203,
        "667f5a0e6e9630f7052b004784dda7c7aeba395cc4ca3c01b11b02e6aa29d7b1",
    ),
    "tokenizer_config.json": (
        268,
        "f13dbe9968de35aff25b06965f18e30449ec37ed94811bd0c7f97990f3b18ebb",
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
            packages.append(
                "%s==%s" % (name.lower().replace("_", "-"), distribution.version)
            )
    return sorted(set(packages))


def save_npy(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    return {"file": path.name, "file_sha256": sha256_file(path), **tensor_summary(tensor)}


def validate_sources(
    snapshot_argument: Path,
    extractor_argument: Path,
    pyproject_argument: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    snapshot = snapshot_argument.resolve(strict=True)
    extractor = extractor_argument.resolve(strict=True)
    pyproject = pyproject_argument.resolve(strict=True)
    if snapshot.name != REVISION:
        raise RuntimeError("DNABERT-S snapshot directory does not name the pinned revision")
    source_manifest = {}
    for name, (expected_size, expected_digest) in SOURCE_FILES.items():
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError("missing pinned DNABERT-S source file: " + name)
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected_size or digest != expected_digest:
            raise RuntimeError("pinned DNABERT-S source file differs: " + name)
        source_manifest[name] = {"size": size, "sha256": digest}
    if sha256_file(extractor) != EXTRACTOR_SHA256:
        raise RuntimeError("pinned GENEB DNABERT-S extractor differs")
    if sha256_file(pyproject) != EXTRACTOR_PYPROJECT_SHA256:
        raise RuntimeError("pinned GENEB embedding environment declaration differs")
    return snapshot, extractor, pyproject, source_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--extractor-pyproject", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    snapshot, extractor, pyproject, source_manifest = validate_sources(
        args.snapshot, args.extractor, args.extractor_pyproject
    )
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
        str(snapshot),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
        revision=REVISION,
    )
    config = AutoConfig.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        revision=REVISION,
    )
    model, loading_info = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        revision=REVISION,
        output_loading_info=True,
    )
    model.to(device="cpu", dtype=torch.float32).eval()
    model_name = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    config_name = "%s.%s" % (config.__class__.__module__, config.__class__.__name__)
    tokenizer_name = "%s.%s" % (
        tokenizer.__class__.__module__,
        tokenizer.__class__.__name__,
    )
    if REVISION not in model_name or not model_name.endswith(".bert_layers.BertModel"):
        raise RuntimeError("DNABERT-S did not select the pinned remote Mosaic model")
    if REVISION not in config_name or not config_name.endswith(
        ".configuration_bert.BertConfig"
    ):
        raise RuntimeError("DNABERT-S did not select the pinned remote config")
    if tokenizer_name != "transformers.tokenization_utils_fast.PreTrainedTokenizerFast":
        raise RuntimeError("DNABERT-S tokenizer class differs")
    loading = {
        key: sorted(str(value) for value in loading_info.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(loading.values()):
        raise RuntimeError("pinned remote DNABERT-S load was incomplete: %r" % loading)
    if sum(parameter.numel() for parameter in model.parameters()) != 117068544:
        raise RuntimeError("pinned remote DNABERT-S parameter count differs")
    module = sys.modules[model.__class__.__module__]
    module_path = Path(inspect.getsourcefile(model.__class__) or "")
    if sha256_file(module_path) != source_manifest["bert_layers.py"]["sha256"]:
        raise RuntimeError("loaded remote modeling code differs from pinned source")
    if getattr(module, "flash_attn_qkvpacked_func", "missing") is not None:
        raise RuntimeError("oracle unexpectedly selected the Triton attention path")
    expected_backend_truncation = {
        "max_length": 2000,
        "stride": 0,
        "strategy": "longest_first",
        "direction": "right",
    }
    expected_backend_padding = {
        "length": 2000,
        "pad_to_multiple_of": None,
        "pad_id": 3,
        "pad_token": "[PAD]",
        "pad_type_id": 0,
        "direction": "right",
    }
    if tokenizer.backend_tokenizer.truncation != expected_backend_truncation:
        raise RuntimeError("DNABERT-S backend truncation audit differs")
    if tokenizer.backend_tokenizer.padding != expected_backend_padding:
        raise RuntimeError("DNABERT-S backend padding audit differs")

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
            "trust_remote_code": True,
            "revision": REVISION,
            "config_class": config_name,
            "model_class": model_name,
            "tokenizer_class": tokenizer_name,
            "remote_modeling_sha256": sha256_file(module_path),
            "attention_implementation": "official MosaicBERT PyTorch fallback",
            "flash_attn_qkvpacked_func": None,
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
            max_length=512,
        )
        evidence_encoding = tokenizer(
            [sequence],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
        )
        for name in ("input_ids", "token_type_ids", "attention_mask"):
            if name in encoded and not torch.equal(encoded[name], evidence_encoding[name]):
                raise RuntimeError("token evidence call changed " + name)
        cpu_inputs = {name: value.to(device="cpu") for name, value in encoded.items()}
        with torch.no_grad():
            output_first = model(**cpu_inputs)
            output_second = model(**cpu_inputs)
        if not isinstance(output_first, tuple) or len(output_first) != 2:
            raise RuntimeError("pinned Mosaic BertModel output contract differs")
        hidden_first = output_first[0].to(dtype=torch.float32)
        hidden_second = output_second[0].to(dtype=torch.float32)
        pooled_first = hidden_first[:, 0, :]
        pooled_second = hidden_second[:, 0, :]
        if not torch.equal(hidden_first, hidden_second) or not torch.equal(
            pooled_first, pooled_second
        ):
            raise RuntimeError("DNABERT-S upstream forward is not bitwise deterministic")

        ids = [int(value) for value in cpu_inputs["input_ids"][0].tolist()]
        attention_mask = [int(value) for value in cpu_inputs["attention_mask"][0].tolist()]
        token_type_ids = [int(value) for value in cpu_inputs["token_type_ids"][0].tolist()]
        special_mask = [
            int(value) for value in evidence_encoding["special_tokens_mask"][0].tolist()
        ]
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
        embedding_path = output / (label + ".cls-token.f32.npy")
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
        "kind": "geneb-dnabert-s-normalized-official-oracle-report",
        "source": {"repo": REPO, "revision": REVISION, "files": source_manifest},
        "extractor": {
            "repo": EXTRACTOR_REPO,
            "revision": EXTRACTOR_REVISION,
            "file": "embedding_pipeline/extractors/dnabert-s.py",
            "sha256": sha256_file(extractor),
            "pyproject_file": "embedding_pipeline/pyproject.toml",
            "pyproject_sha256": sha256_file(pyproject),
        },
        "semantics": {
            "preset": "geneb-v4-normalized",
            "not_geneb_reference": True,
            "reference_blocked": True,
            "batch_size": 1,
            "padding": True,
            "truncation": True,
            "max_length": 512,
            "hidden_tap": "official Mosaic BertModel tuple[0]",
            "pooling": "cls-token",
            "trust_remote_code": True,
            "model_eval": True,
            "torch_no_grad": True,
            "independent_of_evo_native_runtime": True,
        },
        "tokenizer": {
            "vocab_size": int(tokenizer.vocab_size),
            "model_max_length": int(tokenizer.model_max_length),
            "padding_side": tokenizer.padding_side,
            "backend_truncation_audited_but_call_overridden": expected_backend_truncation,
            "backend_padding_audited_but_call_overridden": expected_backend_padding,
        },
        "model": {
            "class": model_name,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "loading_info": loading,
            "output_container": "tuple(sequence_output, pooled_output)",
            "remote_modeling_sha256": sha256_file(module_path),
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
            "runtime_id": "geneb-dnabert-s",
            "input_sha256": records[index]["input"]["sha256"],
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": [float(value) for value in pooled_values[index].contiguous().view(-1)],
            "environment_lock": environment_lock,
            "provenance": {
                "oracle_contract": "geneb-independent-oracle-v1",
                "benchmark_semantics": "geneb-v4-normalized",
                "not_geneb_reference": True,
                "source_repo": REPO,
                "source_revision": REVISION,
                "checkpoint_sha256": source_manifest["pytorch_model.bin"]["sha256"],
                "tokenizer_sha256": source_manifest["tokenizer.json"]["sha256"],
                "extractor_repo": EXTRACTOR_REPO,
                "extractor_revision": EXTRACTOR_REVISION,
                "extractor_file": "embedding_pipeline/extractors/dnabert-s.py",
                "extractor_sha256": sha256_file(extractor),
                "official_hf_modeling_code": False,
                "official_remote_modeling_code": True,
                "trust_remote_code": True,
                "model_class": model_name,
                "batch_size": 1,
                "hidden_tap": "official Mosaic BertModel tuple[0]",
                "pooling": "cls-token",
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
