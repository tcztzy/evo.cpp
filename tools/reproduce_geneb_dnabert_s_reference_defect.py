#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce GENEB DNABERT-S's omitted-trust builtin/random-init defect."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

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
SOURCE_FILES = {
    "config.json": (
        1025,
        "d54c142114563aa34ef7c2b54ccc921d08e50cd7c3cd69c439d5e0d2bbcfcdee",
    ),
    "pytorch_model.bin": (
        468320053,
        "f3cfc3d0541859df64759e758cc4bc40fe1d5760799a7d098a9f9f09501c1e5f",
    ),
    "tokenizer.json": (
        168203,
        "667f5a0e6e9630f7052b004784dda7c7aeba395cc4ca3c01b11b02e6aa29d7b1",
    ),
    "bert_layers.py": (
        40807,
        "cd2fe0efbb90db624fe4e25aa96bd049513e81740ba3f980163c385e894df561",
    ),
}
INPUT = b"ACGTNACGTNACGTN"


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


def package_lock() -> list[str]:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append(
                "%s==%s" % (name.lower().replace("_", "-"), distribution.version)
            )
    return sorted(set(packages))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--extractor-pyproject", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve(strict=True)
    extractor = args.extractor.resolve(strict=True)
    pyproject = args.extractor_pyproject.resolve(strict=True)
    if snapshot.name != REVISION:
        raise RuntimeError("DNABERT-S snapshot directory does not name the pinned revision")
    source_manifest = {}
    for name, (expected_size, expected_digest) in SOURCE_FILES.items():
        path = snapshot / name
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected_size or digest != expected_digest:
            raise RuntimeError("pinned DNABERT-S source file differs: " + name)
        source_manifest[name] = {"size": size, "sha256": digest}
    if sha256_file(extractor) != EXTRACTOR_SHA256:
        raise RuntimeError("pinned GENEB DNABERT-S extractor differs")
    if sha256_file(pyproject) != EXTRACTOR_PYPROJECT_SHA256:
        raise RuntimeError("pinned GENEB embedding pyproject differs")
    extractor_text = extractor.read_text(encoding="utf-8")
    if "AutoModel.from_pretrained(name_model)" not in extractor_text:
        raise RuntimeError("GENEB omitted-trust callsite differs")
    if "trust_remote_code" in extractor_text:
        raise RuntimeError("GENEB extractor unexpectedly enables remote code")
    pyproject_text = pyproject.read_text(encoding="utf-8")
    if '"transformers"' not in pyproject_text:
        raise RuntimeError("GENEB unpinned Transformers dependency declaration differs")
    if '"transformers==' in pyproject_text:
        raise RuntimeError("GENEB Transformers dependency is unexpectedly pinned")
    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(variable + "=1 is required")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    transformers.utils.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), use_fast=True, local_files_only=True
    )
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    config_name = "%s.%s" % (config.__class__.__module__, config.__class__.__name__)
    tokenizer_name = "%s.%s" % (
        tokenizer.__class__.__module__, tokenizer.__class__.__name__
    )
    if config_name != "transformers.models.bert.configuration_bert.BertConfig":
        raise RuntimeError("omitted trust did not select built-in BertConfig")

    runs = []
    canonical_loading = None
    for seed in (0, 1):
        torch.manual_seed(seed)
        model, loading_info = AutoModel.from_pretrained(
            str(snapshot), local_files_only=True, output_loading_info=True
        )
        model.to(device="cpu", dtype=torch.float32).eval()
        model_name = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
        if model_name != "transformers.models.bert.modeling_bert.BertModel":
            raise RuntimeError("omitted trust did not select built-in BertModel")
        loading = {
            key: sorted(str(value) for value in loading_info.get(key, []))
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        }
        if canonical_loading is None:
            canonical_loading = loading
        elif loading != canonical_loading:
            raise RuntimeError("loading mismatch set changed with random seed")
        if (
            len(loading["missing_keys"]) != 145
            or len(loading["unexpected_keys"]) != 84
            or loading["mismatched_keys"]
            or loading["error_msgs"]
        ):
            raise RuntimeError("built-in/random-init loading defect signature differs")
        encoded = tokenizer(
            [INPUT.decode("ascii")],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings,
        )
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state.to(dtype=torch.float32)
        cls = hidden[:, 0, :]
        parameter_map = dict(model.named_parameters())
        random_parameter_numel = sum(
            parameter_map[name].numel()
            for name in loading["missing_keys"]
            if name in parameter_map
        )
        runs.append(
            {
                "seed": seed,
                "model_class": model_name,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "random_initialized_parameter_count": random_parameter_numel,
                "input_ids": [int(value) for value in encoded["input_ids"][0].tolist()],
                "embedding": tensor_summary(cls),
            }
        )
        del hidden, cls, encoded, model
        gc.collect()

    first_values = torch.tensor(runs[0]["embedding"]["first_16_decimal"], dtype=torch.float64)
    second_values = torch.tensor(runs[1]["embedding"]["first_16_decimal"], dtype=torch.float64)
    if runs[0]["embedding"]["raw_little_endian_f32_sha256"] == runs[1]["embedding"][
        "raw_little_endian_f32_sha256"
    ]:
        raise RuntimeError("random-init defect did not change across seeds")
    loading_digests = {
        key + "_canonical_sha256": sha256_bytes(canonical_json(value))
        for key, value in canonical_loading.items()
    }
    generator_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    report = {
        "schema_version": 1,
        "kind": "geneb-dnabert-s-reference-defect-reproduction",
        "status": "reproduced",
        "conclusion": (
            "GENEB omits trust_remote_code; Transformers 4.38.1 selects built-in "
            "BertModel, discards Mosaic tensors, randomly initializes missing tensors, "
            "and produces seed-dependent CLS embeddings"
        ),
        "source": {"repo": REPO, "revision": REVISION, "files": source_manifest},
        "extractor": {
            "repo": EXTRACTOR_REPO,
            "revision": EXTRACTOR_REVISION,
            "file": "embedding_pipeline/extractors/dnabert-s.py",
            "sha256": sha256_file(extractor),
            "model_call": "AutoModel.from_pretrained(name_model)",
            "trust_remote_code_argument": "omitted",
            "pyproject_file": "embedding_pipeline/pyproject.toml",
            "pyproject_sha256": sha256_file(pyproject),
            "transformers_dependency": "unversioned",
        },
        "environment": {
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
            },
            "transformers": {
                "version": transformers.__version__,
                "offline": True,
                "local_files_only": True,
                "trust_remote_code_argument": "omitted",
                "config_class": config_name,
                "tokenizer_class": tokenizer_name,
            },
        },
        "loading_info": {
            **canonical_loading,
            **loading_digests,
            "missing_key_count": len(canonical_loading["missing_keys"]),
            "unexpected_key_count": len(canonical_loading["unexpected_keys"]),
        },
        "input": {
            "ascii": INPUT.decode("ascii"),
            "size": len(INPUT),
            "sha256": sha256_bytes(INPUT),
        },
        "runs": runs,
        "seed_first_16_max_abs": float((first_values - second_values).abs().max().item()),
        "generator_sha256": generator_sha256,
        "independent_of_evo_native_runtime": True,
        "not_an_oracle_vector": True,
        "benchmark_reference_status": "blocked",
    }
    portable(report, "defect report")
    payload = canonical_json(report)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_bytes(payload)
    print(
        json.dumps(
            {
                "generator_sha256": generator_sha256,
                "missing_key_count": len(canonical_loading["missing_keys"]),
                "output_sha256": sha256_bytes(payload),
                "seed_embedding_sha256": [
                    run["embedding"]["raw_little_endian_f32_sha256"] for run in runs
                ],
                "unexpected_key_count": len(canonical_loading["unexpected_keys"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
