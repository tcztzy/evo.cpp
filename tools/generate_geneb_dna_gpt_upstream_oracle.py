#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a pinned GENEB DNA-GPT-0.1B-H CPU-F32 embedding oracle.

This program imports only the pinned GENEB-vendored DNAGPT implementation.
It does not import evo.cpp, converted artifacts, or native runtime math.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUNTIME_ID = "geneb-dna-gpt-0-1b-h"
GENEB_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
UPSTREAM_REVISION = "9b6c0931e3b2011ee5bbb4b988be3e19d62953ae"
CHECKPOINT_NAME = "dna_gpt0.1b_h.pth"
CHECKPOINT_SIZE = 455_736_335
CHECKPOINT_SHA256 = (
    "0a1574f3a496049b053229308c8839aba68702d4d4be264e70f7bde128342ee1"
)
SOURCE_FILES = {
    "dna_gpt/model/gpt.py": (
        11_214,
        "e07ae8f5b7bed3ef38009a68d3699a215e4bc6c0e860a77fa873779a2123a6f0",
    ),
    "dna_gpt/model/dna_gpt.py": (
        5_360,
        "be7e4b7bfd249a1c56f1d2b282d756f62482e69daa81054d57d72afa414caa8f",
    ),
    "dna_gpt/tokenizer.py": (
        3_081,
        "799bba1f99977fe9b998d5e0a765ded7b973d7929481497dc741291d575dec16",
    ),
    "dna_gpt/utils.py": (
        1_398,
        "aada73ee3d8f4900e077d4da4e4be0df9c34fcee0832539b1146a71c81341605",
    ),
}
EXTRACTOR_SIZE = 3_250
EXTRACTOR_SHA256 = (
    "3c9df4316dc049d447516527b378e6a4d944a63b83f4d656099d40e648656212"
)


class OracleError(RuntimeError):
    """Raised when pinned input, source, or execution semantics differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(
    path: Path, label: str, expected_size: int, expected_sha256: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{label} must be a nonsymlink regular file")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != expected_size or digest != expected_sha256:
        raise OracleError(f"{label} differs: size={size} sha256={digest}")
    return {"size": size, "sha256": digest}


def canonical_json(value: object) -> bytes:
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


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def read_single_fasta(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise OracleError("input must be a nonsymlink regular file")
    lines = path.read_text(encoding="ascii").splitlines()
    records: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if current:
                records.append("".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        records.append("".join(current))
    if len(records) != 1 or not records[0]:
        raise OracleError("input must contain exactly one nonempty FASTA record")
    try:
        records[0].encode("ascii")
    except UnicodeEncodeError as error:
        raise OracleError("input sequence must be ASCII") from error
    return records[0]


def git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise OracleError("cannot resolve pinned GENEB revision")
    return result.stdout.strip()


def package_versions() -> list[str]:
    return sorted(
        {
            "%s==%s"
            % (
                distribution.metadata["Name"].lower().replace("_", "-"),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    extractor = args.extractor.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geneb_root = extractor.parents[2]
    if git_revision(geneb_root) != GENEB_REVISION:
        raise OracleError("GENEB revision differs")
    source_lock = {
        name: checked_file(source_root / name, name, expected_size, expected_sha256)
        for name, (expected_size, expected_sha256) in SOURCE_FILES.items()
    }
    extractor_lock = checked_file(
        extractor, "GENEB DNAGPT extractor", EXTRACTOR_SIZE, EXTRACTOR_SHA256
    )
    checkpoint_lock = checked_file(
        checkpoint, CHECKPOINT_NAME, CHECKPOINT_SIZE, CHECKPOINT_SHA256
    )

    sequence = read_single_fasta(input_path)
    if len(sequence) > 4095 * 6:
        raise OracleError("canonical input exceeds the 4096-token model context")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    sys.path.insert(0, str(source_root))
    from dna_gpt.model import DNAGPT  # pylint: disable=import-error,import-outside-toplevel
    from dna_gpt.tokenizer import (  # pylint: disable=import-error,import-outside-toplevel
        KmerTokenizer,
    )

    special_tokens = (
        [str(value) for value in range(10)]
        + ["+", "-", "*", "/", "=", "&", "|", "!"]
        + ["M", "B", "P", "R", "I", "K", "L", "O", "Q", "S", "U", "V", "W", "Y", "X", "Z"]
    )
    tokenizer = KmerTokenizer(6, special_tokens, False)
    if len(tokenizer) != 15_659 or tokenizer.pad_id != 20:
        raise OracleError("pinned static KmerTokenizer topology differs")
    token_ids = tokenizer.encode(
        "<R>" + sequence, max_len=4096, device=torch.device("cpu")
    )
    if token_ids.ndim != 1 or token_ids.numel() < 2 or int(token_ids[0]) != 21:
        raise OracleError("pinned <R> prefix tokenization differs")

    model = DNAGPT.from_name("dna_gpt0.1b_h", vocab_size=len(tokenizer))
    raw = torch.load(
        str(checkpoint),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = raw.get("model", raw) if isinstance(raw, dict) else raw
    if not isinstance(state_dict, dict) or len(state_dict) != 84:
        raise OracleError("checkpoint state tensor set differs")
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise OracleError(
            "checkpoint load differs: missing=%r unexpected=%r"
            % (load_result.missing_keys, load_result.unexpected_keys)
        )
    model.to(device="cpu", dtype=torch.float32).eval()
    input_ids = token_ids.unsqueeze(0)
    attention_mask = input_ids.ne(tokenizer.pad_id).to(torch.float32)
    with torch.inference_mode():
        _, hidden = model(input_ids)
        if list(hidden.shape) != [1, input_ids.shape[1], 768]:
            raise OracleError("post-final-LN hidden shape differs")
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

    values = pooled.cpu().contiguous().numpy().astype("<f4", copy=False)
    npy_path = output_dir / "input-0.attention-mask-mean.f32.npy"
    np.save(npy_path, values, allow_pickle=False)
    generator = Path(__file__).resolve(strict=True)
    input_sha256 = sha256_file(input_path)

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
        "packages": package_versions(),
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "model_class": "%s.%s" % (model.__class__.__module__, model.__class__.__name__),
        "tokenizer_class": "%s.%s"
        % (tokenizer.__class__.__module__, tokenizer.__class__.__name__),
        "source_files": source_lock,
        "extractor": extractor_lock,
    }
    provenance = {
        "source_repo": "TencentAILabHealthcare/DNAGPT",
        "source_revision": UPSTREAM_REVISION,
        "geneb_revision": GENEB_REVISION,
        "checkpoint": checkpoint_lock,
        "generator_sha256": sha256_file(generator),
        "independent_of_evo_native_runtime": True,
        "oracle_execution_call": "pinned-geneb-reference",
        "clean_geneb_reference": True,
        "vendored_patch": "DNAGPT.forward returns (logits, post-final-LN hidden)",
        "tokenization": "literal <R> prefix plus static nonoverlapping 6-mers",
        "pooling": "attention-mask direct-f32-division mean",
    }
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
    vector_path = output_dir / "input-0.independent-oracle-vector.json"
    write_json(vector_path, vector)
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "input": {
            "sha256": input_sha256,
            "sequence_length": len(sequence),
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.to(torch.uint8).tolist(),
        },
        "output": {
            "shape": list(values.shape),
            "npy_sha256": sha256_file(npy_path),
            "raw_f32_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            "vector_sha256": sha256_file(vector_path),
            "first_16": vector["values"][:16],
        },
        "environment_lock": environment,
        "provenance": provenance,
    }
    report_path = output_dir / "oracle-report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "report_sha256": sha256_file(report_path),
                "vector_sha256": sha256_file(vector_path),
                "npy_sha256": sha256_file(npy_path),
                "shape": list(values.shape),
                "input_ids": input_ids.tolist(),
                "first_16": vector["values"][:16],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
