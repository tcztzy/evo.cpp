#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path


def write_npy(
    path: Path,
    rows: list[list[float]],
    *,
    alignment: int,
    always_pad_block: bool,
) -> None:
    columns = len(rows[0])
    header = (
        "{'descr': '<f4', 'fortran_order': False, "
        f"'shape': ({len(rows)}, {columns}), }}"
    )
    padding = (alignment - ((10 + len(header) + 1) % alignment)) % alignment
    if always_pad_block and padding == 0:
        padding = alignment
    header = header + " " * padding + "\n"
    flat = [value for row in rows for value in row]
    path.write_bytes(
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header.encode("ascii")
        + struct.pack(f"<{len(flat)}f", *flat)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    reference = args.work_dir / "reference.npy"
    libnpy_native = args.work_dir / "libnpy-native.npy"
    close = args.work_dir / "close.npy"
    bad = args.work_dir / "bad.npy"
    partial = args.work_dir / "partial.npy"
    reference_bytes = args.work_dir / "reference.bin"
    candidate_bytes = args.work_dir / "candidate.bin"
    report = args.work_dir / "report.json"
    reference_values = [[1.0, 3.0, -2.0], [4.0, 1.0, 0.0]]
    write_npy(
        reference,
        reference_values,
        alignment=64,
        always_pad_block=False,
    )
    write_npy(
        libnpy_native,
        reference_values,
        alignment=16,
        always_pad_block=True,
    )
    write_npy(
        close,
        [[1.001, 3.001, -2.001], [4.001, 1.001, 0.001]],
        alignment=16,
        always_pad_block=True,
    )
    write_npy(
        bad,
        [[4.0, 1.0, -2.0], [1.0, 5.0, 0.0]],
        alignment=16,
        always_pad_block=True,
    )
    write_npy(
        partial,
        [[1.001, 3.001, -2.001], [1.0, 5.0, 0.0]],
        alignment=16,
        always_pad_block=True,
    )
    reference_bytes.write_bytes(b"AC")
    candidate_bytes.write_bytes(b"AC")
    compatible = subprocess.run(
        [
            sys.executable,
            str(args.tool),
            "--reference",
            str(reference),
            "--candidate",
            str(libnpy_native),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if compatible.returncode != 0:
        raise AssertionError(compatible.stderr + compatible.stdout)
    compatibility_report = json.loads(compatible.stdout)
    if (
        not compatibility_report["passed"]
        or compatibility_report["minimum_cosine_observed"] != 1.0
        or compatibility_report["reference_sha256"]
        == compatibility_report["candidate_sha256"]
    ):
        raise AssertionError(
            "legacy evo.cpp and native libnpy containers are not interoperable"
        )
    good = subprocess.run(
        [
            sys.executable,
            str(args.tool),
            "--reference",
            str(reference),
            "--candidate",
            str(close),
            "--minimum-cosine",
            "0.999",
            "--reference-bytes",
            str(reference_bytes),
            "--candidate-bytes",
            str(candidate_bytes),
            "--output",
            str(report),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if good.returncode != 0:
        raise AssertionError(good.stderr + good.stdout)
    parsed = json.loads(report.read_text(encoding="utf-8"))
    if not parsed["passed"] or not parsed["top1_exact"]:
        raise AssertionError("close logit fixture did not pass")
    rejected = subprocess.run(
        [
            sys.executable,
            str(args.tool),
            "--reference",
            str(reference),
            "--candidate",
            str(bad),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if rejected.returncode == 0:
        raise AssertionError("top-1 mismatch was not rejected")
    relaxed = subprocess.run(
        [
            sys.executable,
            str(args.tool),
            "--reference",
            str(reference),
            "--candidate",
            str(partial),
            "--minimum-cosine",
            "0.1",
            "--minimum-top1-agreement",
            "0.5",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if relaxed.returncode != 0:
        raise AssertionError(relaxed.stderr + relaxed.stdout)
    relaxed_report = json.loads(relaxed.stdout)
    if (
        relaxed_report["top1_exact"]
        or relaxed_report["top1_agreement"] != 0.5
        or relaxed_report["minimum_top1_agreement_required"] != 0.5
    ):
        raise AssertionError("relaxed top-1 gate reported incorrect metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
