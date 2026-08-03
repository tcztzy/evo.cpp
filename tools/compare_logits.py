#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two row-major NPY v1 F32 logit matrices without NumPy."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path


def load_npy_f32(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    payload = path.read_bytes()
    if payload[:8] != b"\x93NUMPY\x01\x00" or len(payload) < 10:
        raise ValueError(f"{path} is not a NumPy v1 file")
    header_size = struct.unpack_from("<H", payload, 8)[0]
    payload_offset = 10 + header_size
    if payload_offset > len(payload):
        raise ValueError(f"{path} has a truncated NumPy header")
    header = ast.literal_eval(payload[10:payload_offset].decode("ascii"))
    shape = tuple(int(value) for value in header["shape"])
    if (
        header["descr"] != "<f4"
        or header["fortran_order"]
        or not shape
        or any(value <= 0 for value in shape)
    ):
        raise ValueError(f"{path} is not a nonempty row-major little-endian F32 array")
    count = 1
    for dimension in shape:
        count *= dimension
    if len(payload) - payload_offset != count * 4:
        raise ValueError(f"{path} payload size does not match shape {shape}")
    return shape, struct.unpack_from(f"<{count}f", payload, payload_offset)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(
    reference_path: Path,
    candidate_path: Path,
    minimum_cosine: float,
    minimum_top1_agreement: float,
    reference_bytes: Path | None,
    candidate_bytes: Path | None,
) -> dict[str, object]:
    reference_shape, reference = load_npy_f32(reference_path)
    candidate_shape, candidate = load_npy_f32(candidate_path)
    if reference_shape != candidate_shape:
        raise ValueError(
            f"logit shapes differ: {reference_shape} vs {candidate_shape}"
        )
    columns = reference_shape[-1]
    rows = len(reference) // columns
    row_metrics: list[dict[str, object]] = []
    for row in range(rows):
        left = reference[row * columns : (row + 1) * columns]
        right = candidate[row * columns : (row + 1) * columns]
        dot = math.fsum(
            first * second for first, second in zip(left, right, strict=True)
        )
        left_norm = math.sqrt(math.fsum(value * value for value in left))
        right_norm = math.sqrt(math.fsum(value * value for value in right))
        cosine = (
            1.0
            if left == right
            else dot / (left_norm * right_norm)
            if left_norm > 0.0 and right_norm > 0.0
            else 0.0
        )
        row_metrics.append(
            {
                "row": row,
                "cosine": cosine,
                "reference_top1": max(range(columns), key=left.__getitem__),
                "candidate_top1": max(range(columns), key=right.__getitem__),
                "max_abs_difference": max(
                    abs(first - second)
                    for first, second in zip(left, right, strict=True)
                ),
            }
        )
    finite = all(math.isfinite(value) for value in (*reference, *candidate))
    matching_top1 = sum(
        row["reference_top1"] == row["candidate_top1"] for row in row_metrics
    )
    top1_agreement = matching_top1 / rows
    top1_exact = matching_top1 == rows
    minimum_observed = min(float(row["cosine"]) for row in row_metrics)
    byte_exact = None
    if reference_bytes is not None and candidate_bytes is not None:
        byte_exact = reference_bytes.read_bytes() == candidate_bytes.read_bytes()
    passed = (
        finite
        and top1_agreement >= minimum_top1_agreement
        and minimum_observed >= minimum_cosine
        and byte_exact is not False
    )
    return {
        "reference": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "shape": list(reference_shape),
        "minimum_cosine_required": minimum_cosine,
        "minimum_cosine_observed": minimum_observed,
        "minimum_top1_agreement_required": minimum_top1_agreement,
        "top1_agreement": top1_agreement,
        "finite": finite,
        "top1_exact": top1_exact,
        "output_byte_exact": byte_exact,
        "rows": row_metrics,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument("--minimum-top1-agreement", type=float, default=1.0)
    parser.add_argument("--reference-bytes", type=Path)
    parser.add_argument("--candidate-bytes", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.minimum_cosine <= 1.0:
        parser.error("--minimum-cosine must be in (0, 1]")
    if not 0.0 <= args.minimum_top1_agreement <= 1.0:
        parser.error("--minimum-top1-agreement must be in [0, 1]")
    if (args.reference_bytes is None) != (args.candidate_bytes is None):
        parser.error("both byte-output paths must be supplied together")
    report = compare(
        args.reference,
        args.candidate,
        args.minimum_cosine,
        args.minimum_top1_agreement,
        args.reference_bytes,
        args.candidate_bytes,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
