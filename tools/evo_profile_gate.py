#!/usr/bin/env python3
"""Gate an approximate inference profile against exact numeric/scientific outputs."""

from __future__ import annotations

import argparse
import ast
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path


def load_npy_f32(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    payload = path.read_bytes()
    if len(payload) < 10 or payload[:6] != b"\x93NUMPY":
        raise ValueError(f"{path} is not an NPY file")
    major, minor = payload[6], payload[7]
    if (major, minor) == (1, 0):
        header_size = struct.unpack_from("<H", payload, 8)[0]
        header_start = 10
    elif major in (2, 3) and minor == 0:
        header_size = struct.unpack_from("<I", payload, 8)[0]
        header_start = 12
    else:
        raise ValueError(f"{path} uses unsupported NPY version {major}.{minor}")
    header_end = header_start + header_size
    try:
        header = ast.literal_eval(payload[header_start:header_end].decode("latin1"))
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"{path} has an invalid NPY header") from error
    if header.get("descr") != "<f4" or header.get("fortran_order") is not False:
        raise ValueError(f"{path} must be row-major little-endian F32")
    shape = tuple(header.get("shape", ()))
    if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError(f"{path} must contain a nonempty rank-2 matrix")
    count = math.prod(shape)
    if header_end + count * 4 != len(payload):
        raise ValueError(f"{path} payload length does not match shape {shape}")
    values = struct.unpack_from(f"<{count}f", payload, header_end)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path} contains non-finite logits")
    return shape, values


def numeric_metrics(
    exact_path: Path, candidate_path: Path
) -> dict[str, float | int | list[int]]:
    exact_shape, exact = load_npy_f32(exact_path)
    candidate_shape, candidate = load_npy_f32(candidate_path)
    if candidate_shape != exact_shape:
        raise ValueError(
            f"logit shape mismatch: exact={exact_shape}, candidate={candidate_shape}"
        )
    rows, columns = exact_shape
    absolute_error = [abs(left - right) for left, right in zip(exact, candidate)]
    cosine_values: list[float] = []
    top1_equal = 0
    for row in range(rows):
        start = row * columns
        exact_row = exact[start : start + columns]
        candidate_row = candidate[start : start + columns]
        dot = sum(left * right for left, right in zip(exact_row, candidate_row))
        exact_norm = math.sqrt(sum(value * value for value in exact_row))
        candidate_norm = math.sqrt(sum(value * value for value in candidate_row))
        if exact_norm == 0.0 or candidate_norm == 0.0:
            cosine = 1.0 if exact_row == candidate_row else 0.0
        else:
            cosine = dot / (exact_norm * candidate_norm)
        cosine_values.append(max(-1.0, min(1.0, cosine)))
        exact_top = max(range(columns), key=exact_row.__getitem__)
        candidate_top = max(range(columns), key=candidate_row.__getitem__)
        top1_equal += int(exact_top == candidate_top)
    return {
        "shape": list(exact_shape),
        "minimum_row_cosine": min(cosine_values),
        "mean_absolute_error": sum(absolute_error) / len(absolute_error),
        "maximum_absolute_error": max(absolute_error),
        "top1_agreement": top1_equal / rows,
    }


def load_bio(path: Path, expected_profile: str | None) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read biological benchmark {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError(f"{path} must use biological benchmark schema_version 1")
    profile = document.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError(f"{path} must declare a nonempty profile")
    if expected_profile is not None and profile != expected_profile:
        raise ValueError(
            f"{path} declares profile {profile!r}, expected {expected_profile!r}"
        )
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path} records must be an array")
    normalized: dict[str, tuple[str, float]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path} record {index} must be an object")
        identifier = record.get("id")
        task = record.get("task")
        score = record.get("score")
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(task, str)
            or not task
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError(f"{path} record {index} has invalid id/task/score")
        if identifier in normalized:
            raise ValueError(f"{path} contains duplicate record id {identifier!r}")
        normalized[identifier] = (task, float(score))
    return {"profile": profile, "records": normalized}


def biological_metrics(
    exact_path: Path,
    candidate_path: Path,
    minimum_records: int,
    minimum_variants: int,
) -> dict[str, float | int]:
    exact = load_bio(exact_path, "exact")
    candidate = load_bio(candidate_path, None)
    if candidate["profile"] == "exact":
        raise ValueError("candidate biological benchmark must declare a non-exact profile")
    exact_records = exact["records"]
    candidate_records = candidate["records"]
    assert isinstance(exact_records, dict) and isinstance(candidate_records, dict)
    if exact_records.keys() != candidate_records.keys():
        missing = sorted(exact_records.keys() - candidate_records.keys())
        extra = sorted(candidate_records.keys() - exact_records.keys())
        raise ValueError(f"biological record mismatch: missing={missing}, extra={extra}")
    if len(exact_records) < minimum_records:
        raise ValueError(
            f"biological benchmark has {len(exact_records)} records; "
            f"at least {minimum_records} are required"
        )
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    variants = 0
    variant_sign_equal = 0
    score_errors: list[float] = []
    for identifier, (exact_task, exact_score) in exact_records.items():
        candidate_task, candidate_score = candidate_records[identifier]
        if candidate_task != exact_task:
            raise ValueError(
                f"task mismatch for {identifier!r}: "
                f"exact={exact_task!r}, candidate={candidate_task!r}"
            )
        grouped[exact_task].append((exact_score, candidate_score))
        score_errors.append(abs(exact_score - candidate_score))
        if exact_task == "variant":
            variants += 1
            variant_sign_equal += int(
                (exact_score > 0.0) == (candidate_score > 0.0)
                and (exact_score < 0.0) == (candidate_score < 0.0)
            )
    if variants < minimum_variants:
        raise ValueError(
            f"biological benchmark has {variants} variants; "
            f"at least {minimum_variants} are required"
        )
    comparable_pairs = 0
    agreeing_pairs = 0
    for values in grouped.values():
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                exact_difference = values[left][0] - values[right][0]
                if exact_difference == 0.0:
                    continue
                candidate_difference = values[left][1] - values[right][1]
                comparable_pairs += 1
                agreeing_pairs += int(exact_difference * candidate_difference > 0.0)
    if comparable_pairs == 0:
        raise ValueError("biological benchmark has no comparable within-task ranks")
    return {
        "records": len(exact_records),
        "variant_records": variants,
        "mean_absolute_score_error": sum(score_errors) / len(score_errors),
        "maximum_absolute_score_error": max(score_errors),
        "within_task_rank_agreement": agreeing_pairs / comparable_pairs,
        "variant_sign_agreement": variant_sign_equal / variants,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-logits", required=True, type=Path)
    parser.add_argument("--candidate-logits", required=True, type=Path)
    parser.add_argument("--exact-bio", required=True, type=Path)
    parser.add_argument("--candidate-bio", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-logit-mae", type=float, default=0.25)
    parser.add_argument("--min-top1-agreement", type=float, default=0.95)
    parser.add_argument("--min-rank-agreement", type=float, default=0.95)
    parser.add_argument("--min-variant-sign-agreement", type=float, default=0.95)
    parser.add_argument("--minimum-records", type=int, default=4)
    parser.add_argument("--minimum-variants", type=int, default=2)
    args = parser.parse_args()
    if any(
        not 0.0 <= value <= 1.0
        for value in (
            args.min_top1_agreement,
            args.min_rank_agreement,
            args.min_variant_sign_agreement,
        )
    ) or not -1.0 <= args.min_cosine <= 1.0:
        parser.error("agreement/cosine thresholds are outside their ranges")
    if args.max_logit_mae < 0.0 or args.minimum_records < 1 or args.minimum_variants < 1:
        parser.error("MAE and minimum record counts must be positive")
    try:
        numeric = numeric_metrics(args.exact_logits, args.candidate_logits)
        biological = biological_metrics(
            args.exact_bio,
            args.candidate_bio,
            args.minimum_records,
            args.minimum_variants,
        )
        checks = {
            "minimum_row_cosine": numeric["minimum_row_cosine"]
            >= args.min_cosine,
            "mean_absolute_error": numeric["mean_absolute_error"]
            <= args.max_logit_mae,
            "top1_agreement": numeric["top1_agreement"]
            >= args.min_top1_agreement,
            "within_task_rank_agreement": biological["within_task_rank_agreement"]
            >= args.min_rank_agreement,
            "variant_sign_agreement": biological["variant_sign_agreement"]
            >= args.min_variant_sign_agreement,
        }
        report = {
            "schema_version": 1,
            "candidate_profile": load_bio(args.candidate_bio, None)["profile"],
            "thresholds": {
                "min_cosine": args.min_cosine,
                "max_logit_mae": args.max_logit_mae,
                "min_top1_agreement": args.min_top1_agreement,
                "min_rank_agreement": args.min_rank_agreement,
                "min_variant_sign_agreement": args.min_variant_sign_agreement,
            },
            "numeric": numeric,
            "biological": biological,
            "checks": checks,
            "passed": all(checks.values()),
        }
    except ValueError as error:
        print(f"profile gate: {error}", file=sys.stderr)
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        print(f"profile gate failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
