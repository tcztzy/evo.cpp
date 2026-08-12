#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the dependency-free side of the official ESMC acceptance gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def write_npy(path: Path, shape: tuple[int, ...], values: list[float]) -> None:
    shape_text = repr(shape)
    header = (
        f"{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}"
    ).encode("ascii")
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    header += b" " * padding + b"\n"
    path.write_bytes(
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header
        + struct.pack(f"<{len(values)}f", *values)
    )


def run(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(argument) for argument in arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparator", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    oracle = args.work_dir / "oracle"
    oracle.mkdir(parents=True)
    (oracle / "oracle.json").write_text(
        json.dumps({"schema_version": 1, "model_id": "esmc_300m"}),
        encoding="utf-8",
    )
    logits = [0.25, -0.5, 1.0, 2.0]
    hidden = [-1.0, 0.5, 0.75, -0.25, 1.25, 0.0]
    write_npy(oracle / "logits.npy", (2, 2), logits)
    write_npy(oracle / "final-hidden.npy", (2, 3), hidden)
    native_logits = args.work_dir / "native-logits.npy"
    native_hidden = args.work_dir / "native-hidden.npy"
    write_npy(native_logits, (2, 2), logits)
    write_npy(native_hidden, (2, 3), hidden)
    report = args.work_dir / "comparison.json"
    common = (
        sys.executable,
        args.comparator,
        "--oracle-dir",
        oracle,
        "--native-logits",
        native_logits,
        "--native-hidden",
        native_hidden,
        "--model-id",
        "esmc_300m",
        "--output",
        report,
    )
    accepted = run(*common)
    assert accepted.returncode == 0, accepted.stderr
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["passed"]
    assert result["results"]["logits"]["max_abs"] == 0.0
    assert result["results"]["final_hidden"]["cosine"] == 1.0

    write_npy(native_logits, (2, 2), [0.25, -0.5, 1.0, 2.006])
    rejected = run(*common)
    assert rejected.returncode == 1
    result = json.loads(report.read_text(encoding="utf-8"))
    assert not result["passed"]
    assert result["results"]["logits"]["max_abs"] > 5e-3

    help_result = run(sys.executable, args.generator, "--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "--reference-commit" in help_result.stdout
    specification = importlib.util.spec_from_file_location("esmc_oracle", args.generator)
    assert specification is not None and specification.loader is not None
    oracle_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(oracle_module)
    assert [oracle_module.swiglu_hidden_dim(width) for width in (960, 1152, 2560)] == [
        2560,
        3072,
        6912,
    ]
    print("official ESMC oracle tooling contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
