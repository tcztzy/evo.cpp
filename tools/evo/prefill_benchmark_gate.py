#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate deterministic evo prefill benchmark metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TypeVar

T = TypeVar("T", int, float)


def parse_mapping(
    values: list[str], name: str, value_type: type[T]
) -> dict[int, T]:
    parsed: dict[int, T] = {}
    for item in values:
        try:
            raw_length, raw_value = item.split("=", 1)
            length = int(raw_length)
            value = value_type(raw_value)
        except ValueError as error:
            raise ValueError(f"{name} must use LENGTH=VALUE: {item}") from error
        if length <= 0 or value <= 0 or length in parsed:
            raise ValueError(f"{name} contains an invalid entry: {item}")
        parsed[length] = value
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def read_metrics(path: Path) -> dict[str, object]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("evo_metrics "):
            records.append(json.loads(line.removeprefix("evo_metrics ")))
    if len(records) != 1:
        raise ValueError(f"{path} must contain exactly one evo_metrics record")
    return records[0]


def positive_number(record: dict[str, object], key: str, path: Path) -> float:
    value = record.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {key} is not numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{path}: {key} must be finite and positive")
    return converted


def positive_integer(record: dict[str, object], key: str, path: Path) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path}: {key} must be a positive integer")
    return value


def evaluate(
    artifact_dir: Path,
    official_metrics_path: Path,
    repeat_counts: dict[int, int],
    minimum_rates: dict[int, float],
    maximum_load_seconds: float,
) -> dict[str, object]:
    if repeat_counts.keys() != minimum_rates.keys():
        raise ValueError("repeat-count and minimum-rate lengths must match")
    official = json.loads(official_metrics_path.read_text(encoding="utf-8"))
    official_prefill = official.get("prefill")
    if not isinstance(official_prefill, dict):
        raise ValueError("official metrics do not contain a prefill object")

    cases: dict[str, object] = {}
    passed = True
    for length in sorted(repeat_counts):
        first_path = artifact_dir / f"metrics_{length}.log"
        repeat_path = artifact_dir / f"repeat_{length}.log"
        first = read_metrics(first_path)
        repeated = read_metrics(repeat_path)
        first_tokens = positive_integer(first, "prefill_tokens", first_path)
        repeated_tokens = positive_integer(
            repeated, "prefill_tokens", repeat_path
        )
        expected_repeated_tokens = length * repeat_counts[length]
        if first_tokens != length or repeated_tokens != expected_repeated_tokens:
            raise ValueError(
                f"length {length}: expected {length} first tokens and "
                f"{expected_repeated_tokens} repeated tokens"
            )
        first_seconds = positive_number(first, "prefill_seconds", first_path)
        repeated_seconds = positive_number(
            repeated, "prefill_seconds", repeat_path
        )
        first_load = positive_number(
            first, "model_load_seconds", first_path
        )
        repeated_load = positive_number(
            repeated, "model_load_seconds", repeat_path
        )
        first_rate = length / first_seconds
        repeated_rate = expected_repeated_tokens / repeated_seconds
        official_case = official_prefill.get(str(length))
        if not isinstance(official_case, dict):
            raise ValueError(f"official metrics omit prefill length {length}")
        official_rate_value = official_case.get("median_tokens_per_second")
        if not isinstance(official_rate_value, (int, float)):
            raise ValueError(f"official prefill length {length} has no rate")
        official_rate = float(official_rate_value)
        if not math.isfinite(official_rate) or official_rate <= 0.0:
            raise ValueError(f"official prefill length {length} rate is invalid")
        rate_passed = repeated_rate >= minimum_rates[length]
        load_passed = (
            first_load <= maximum_load_seconds
            and repeated_load <= maximum_load_seconds
        )
        passed = passed and rate_passed and load_passed
        cases[str(length)] = {
            "first_seconds": first_seconds,
            "first_tokens_per_second": first_rate,
            "repeat_count": repeat_counts[length],
            "repeated_seconds": repeated_seconds,
            "repeated_tokens_per_second": repeated_rate,
            "minimum_tokens_per_second": minimum_rates[length],
            "official_tokens_per_second": official_rate,
            "official_ratio": repeated_rate / official_rate,
            "first_model_load_seconds": first_load,
            "repeated_model_load_seconds": repeated_load,
            "rate_passed": rate_passed,
            "load_passed": load_passed,
        }
    return {
        "artifact_dir": str(artifact_dir),
        "official_metrics": str(official_metrics_path),
        "maximum_model_load_seconds": maximum_load_seconds,
        "cases": cases,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--official-metrics", required=True, type=Path)
    parser.add_argument(
        "--repeat-count",
        action="append",
        default=[],
        metavar="LENGTH=COUNT",
    )
    parser.add_argument(
        "--minimum-rate",
        action="append",
        default=[],
        metavar="LENGTH=TOKENS_PER_SECOND",
    )
    parser.add_argument("--maximum-load-seconds", type=float, default=5.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        not math.isfinite(args.maximum_load_seconds)
        or args.maximum_load_seconds <= 0.0
    ):
        parser.error("--maximum-load-seconds must be finite and positive")
    try:
        repeat_counts = parse_mapping(
            args.repeat_count, "--repeat-count", int
        )
        minimum_rates = parse_mapping(
            args.minimum_rate, "--minimum-rate", float
        )
        report = evaluate(
            args.artifact_dir,
            args.official_metrics,
            repeat_counts,
            minimum_rates,
            args.maximum_load_seconds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
