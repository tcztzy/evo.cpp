#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a pinned, CPU-normalized Genomics-FM upstream oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit

import numpy as np
import torch
from transformers import AutoTokenizer


RUNTIME_ID = "geneb-genomics-fm"
SOURCE_REPO = "terry-r123/Genomics-FM"
SOURCE_REVISION = "edc9f16b2d146f07092c038b2187b72960b178ea"
EXTRACTOR_REPO = "darlednik/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = "372bdc3dc19d70ce3e369aa93a1a91e47d39cf0b7e3009cf6dbc0eb553fc274a"
PATCH_SHA256 = "8a1c255917a4942d47299f92e23745470e1b36064d68a7566433d9a0bbf917a3"
PATCH_OLD = "        gpu_number = int(input_ids.device.index)\n"
PATCH_NEW = (
    "        gpu_number = 0 if input_ids.device.index is None "
    "else int(input_ids.device.index)\n"
)
CODE_FILES = {
    "bert_layers.py": (
        42802,
        "b1c6b7e61a2342ad3fcb2a440197af8f0fac68acc130ed9e78f203562623ad27",
    ),
    "bert_padding.py": (
        6099,
        "44d1c68afb1f585fdc66c150d4c60f1ed44a89c006abc57d50531d71940d7421",
    ),
    "configuration_bert.py": (
        1017,
        "d030166bd279ca81e9d2304eaff9419171f2f8c69ab419e7cd37873b72f47a22",
    ),
}
CHECKPOINT_FILES = {
    "config.json": (
        1056,
        "83bd023650d1fd840494686f11a396c291af5a9782e997e31ae698805cf23e45",
    ),
    "pytorch_model.bin": (
        479674693,
        "c5368c3e745b9f6ae510dabfe7ce6fe66dc72b7b434dfb033e1ae8ef73505f16",
    ),
    "special_tokens_map.json": (
        217,
        "4d29eaf3ce7254329c323277d42822e6423f04317fae44a9017d292d07a7f926",
    ),
    "tokenizer.json": (
        168659,
        "84bd57fd379b1b1ade63764876fae800342c612b78bbd2d148dabd66b49d5a05",
    ),
    "tokenizer_config.json": (
        268,
        "f13dbe9968de35aff25b06965f18e30449ec37ed94811bd0c7f97990f3b18ebb",
    ),
}


class OracleError(RuntimeError):
    pass


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


def package_lock() -> List[str]:
    packages = []  # type: List[str]
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append(
                "%s==%s" % (name.lower().replace("_", "-"), distribution.version)
            )
    return sorted(set(packages))


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
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or urlsplit(value).scheme.lower() == "file"
    ):
        raise OracleError(label + " contains a machine-local path")


def validate_file(path: Path, expected: Tuple[int, str], label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise OracleError(label + " is missing")
    size = path.stat().st_size
    digest = sha256_file(path)
    if (size, digest) != expected:
        raise OracleError(label + " size/SHA256 differs")
    return {"size": size, "sha256": digest}


def read_fasta(path: Path) -> str:
    lines = path.read_text(encoding="ascii").splitlines()
    payload = "".join(line.strip() for line in lines if not line.startswith(">"))
    if not payload:
        raise OracleError("canonical FASTA contains no sequence")
    return payload


def import_patched_model(source_dir: Path) -> Tuple[Any, Any, str]:
    temporary = tempfile.TemporaryDirectory(prefix="geneb-genomics-fm-oracle-")
    package_root = Path(temporary.name) / "genomics_fm_oracle"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    for name in CODE_FILES:
        shutil.copy2(source_dir / name, package_root / name)
    model_path = package_root / "bert_layers.py"
    source = model_path.read_text(encoding="utf-8")
    if source.count(PATCH_OLD) != 1:
        temporary.cleanup()
        raise OracleError("pinned CPU defect line differs")
    model_path.write_text(source.replace(PATCH_OLD, PATCH_NEW), encoding="utf-8")
    sys.path.insert(0, temporary.name)
    try:
        configuration = importlib.import_module(
            "genomics_fm_oracle.configuration_bert"
        )
        modeling = importlib.import_module("genomics_fm_oracle.bert_layers")
    finally:
        sys.path.pop(0)
    modeling.flash_attn_qkvpacked_func = None
    # Retain the directory for the lifetime of imported module code.
    setattr(modeling, "_oracle_temporary_directory", temporary)
    return configuration.BertConfig, modeling.BertForMaskedLM, sha256_file(model_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise OracleError(variable + "=1 is required")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    source_repo = args.source_repo.resolve(strict=True)
    source_dir = args.source_dir.resolve(strict=True)
    checkpoint = args.checkpoint_dir.resolve(strict=True)
    extractor = args.extractor.resolve(strict=True)
    patch = args.patch.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    revision = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise OracleError("Genomics-FM source revision differs")
    code_manifest = {
        name: validate_file(source_dir / name, expected, "source " + name)
        for name, expected in CODE_FILES.items()
    }
    checkpoint_manifest = {
        name: validate_file(checkpoint / name, expected, "checkpoint " + name)
        for name, expected in CHECKPOINT_FILES.items()
    }
    if sha256_file(extractor) != EXTRACTOR_SHA256:
        raise OracleError("pinned GENEB Genomics-FM extractor differs")
    if sha256_file(patch) != PATCH_SHA256:
        raise OracleError("CPU normalization patch differs")

    sequence = read_fasta(input_path)
    transformed = re.sub(r"[^ACGT]", "N", sequence.strip().upper())
    if transformed != "ACGTNNNNACGT":
        raise OracleError("canonical input transform differs")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), use_fast=True, local_files_only=True
    )
    if tokenizer.vocab_size != 4096 or len(tokenizer) != 4100:
        raise OracleError("pinned tokenizer vocabulary differs")
    encoded = tokenizer(
        [transformed],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    expected_ids = [1, 5, 6, 1049, 0, 0, 0, 0, 5, 6, 1049, 2]
    if encoded["input_ids"][0].tolist() != expected_ids:
        raise OracleError("official tokenizer IDs differ")
    if encoded["attention_mask"][0].tolist() != [1] * len(expected_ids):
        raise OracleError("official tokenizer attention mask differs")
    if encoded["token_type_ids"][0].tolist() != [0] * len(expected_ids):
        raise OracleError("official tokenizer type IDs differ")

    BertConfig, BertForMaskedLM, patched_model_sha = import_patched_model(source_dir)
    config = BertConfig.from_pretrained(str(checkpoint), local_files_only=True)
    model = BertForMaskedLM(config)
    state = torch.load(
        str(checkpoint / "pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(state, dict) or len(state) != 143:
        raise OracleError("pinned checkpoint state dictionary differs")
    loading = model.load_state_dict(state, strict=True)
    if loading.missing_keys or loading.unexpected_keys:
        raise OracleError("pinned checkpoint load is incomplete")
    del state
    model.to(device="cpu", dtype=torch.float32).eval()
    if sum(parameter.numel() for parameter in model.parameters()) != 119906660:
        raise OracleError("pinned model parameter count differs")
    with torch.inference_mode():
        hidden_first, _ = model.bert(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded["token_type_ids"],
        )
        hidden_second, _ = model.bert(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded["token_type_ids"],
        )
    if not torch.equal(hidden_first, hidden_second):
        raise OracleError("pinned upstream forward is not bitwise deterministic")
    pooled = hidden_first[:, 0, :].to(device="cpu", dtype=torch.float32).contiguous()
    values = pooled.numpy().astype("<f4", copy=False)
    flat = [float(value) for value in values.reshape(-1)]
    if len(flat) != 768 or any(not math.isfinite(value) for value in flat):
        raise OracleError("upstream pooled embedding is invalid")

    source_files = dict(code_manifest)
    source_files.update(checkpoint_manifest)
    environment = {
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
        },
        "transformers": {
            "version": importlib.metadata.version("transformers"),
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "local_files_only": True,
            "model_class": "%s.%s"
            % (model.__class__.__module__, model.__class__.__name__),
            "tokenizer_class": "%s.%s"
            % (tokenizer.__class__.__module__, tokenizer.__class__.__name__),
            "attention_implementation": "official PyTorch fallback",
            "flash_attn_qkvpacked_func": None,
        },
        "source_files": source_files,
    }
    source_manifest = [
        {"name": name, "size": item["size"], "sha256": item["sha256"]}
        for name, item in sorted(source_files.items())
    ]
    generator_sha = sha256_file(Path(__file__).resolve(strict=True))
    clean_error = (
        "int() argument must be a string, a bytes-like object or a real number, "
        "not 'NoneType'"
    )
    provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-normalized-cpu",
        "benchmark_semantics": "geneb-v4-normalized",
        "not_clean_geneb_reference": True,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_file_manifest_sha256": sha256_bytes(canonical_json(source_manifest)),
        "checkpoint_sha256": CHECKPOINT_FILES["pytorch_model.bin"][1],
        "tokenizer_sha256": CHECKPOINT_FILES["tokenizer.json"][1],
        "extractor_repo": EXTRACTOR_REPO,
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_module": "embedding_pipeline/extractors/genomics-fm.py",
        "extractor_class": "GenomicsFMExtractor",
        "extractor_sha256": EXTRACTOR_SHA256,
        "normalization_patch_sha256": PATCH_SHA256,
        "patched_modeling_sha256": patched_model_sha,
        "clean_cpu_error_sha256": sha256_bytes(clean_error.encode("utf-8")),
        "clean_cpu_error": clean_error,
        "patch_scope": "replace the unused CPU-invalid device-index conversion",
        "position_encoding": "learned-absolute; extractor omits use_alibi",
        "hidden_tap": "BertModel sequence_output",
        "pooling": "cls-token",
        "independent_of_evo_native_runtime": True,
        "generator_sha256": generator_sha,
    }
    vector = {
        "schema_version": 1,
        "kind": "geneb-independent-oracle-vector",
        "runtime_id": RUNTIME_ID,
        "input_sha256": sha256_file(input_path),
        "backend": "cpu",
        "profile": "cpu-f32",
        "values": flat,
        "environment_lock": environment,
        "provenance": provenance,
    }
    portable(vector, "oracle vector")
    vector_path = output / "input-0.independent-oracle-vector.json"
    vector_path.write_bytes(canonical_json(vector))
    npy_path = output / "input-0.cls-token.f32.npy"
    np.save(npy_path, values, allow_pickle=False)
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "file_manifest_sha256": provenance["source_file_manifest_sha256"],
        },
        "input": {
            "sha256": vector["input_sha256"],
            "raw_sequence": sequence,
            "transformed_sequence": transformed,
            "input_ids": expected_ids,
        },
        "output": {
            "shape": list(values.shape),
            "npy_sha256": sha256_file(npy_path),
            "raw_f32_sha256": sha256_bytes(values.tobytes(order="C")),
            "vector_sha256": sha256_file(vector_path),
            "first_16": flat[:16],
        },
        "environment_lock": environment,
        "provenance": provenance,
    }
    portable(report, "oracle report")
    report_path = output / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))
    print(str(vector_path))
    print("vector_sha256=" + sha256_file(vector_path))
    print("report_sha256=" + sha256_file(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
