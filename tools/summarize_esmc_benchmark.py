#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Combine official ESMC timing JSON with native evo_metrics samples."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS_PREFIX = "evo_metrics "


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} root must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--native-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--length", action="append", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args()

    if args.warmups < 1 or args.repeats < 2:
        parser.error("warmups/repeats are outside the benchmark contract")
    lengths = list(dict.fromkeys(args.length))
    official = load_object(args.official)
    metrics = []
    for line in args.native_log.read_text(encoding="utf-8").splitlines():
        if line.startswith(METRICS_PREFIX):
            value = json.loads(line[len(METRICS_PREFIX) :])
            if value.get("architecture") == "ESMC":
                metrics.append(value)
    expected = len(lengths) * (args.warmups + args.repeats)
    if len(metrics) != expected:
        raise ValueError(f"native log has {len(metrics)} ESMC samples, expected {expected}")

    comparison: dict[str, Any] = {}
    offset = 0
    for length in lengths:
        group = metrics[offset : offset + args.warmups + args.repeats]
        offset += len(group)
        if any(int(item.get("prefill_tokens", -1)) != length for item in group):
            raise ValueError(f"native token count differs from requested length {length}")
        samples = [float(item["prefill_seconds"]) for item in group[args.warmups :]]
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("native timing sample is not finite and positive")
        native_median = statistics.median(samples)
        official_case = official.get("cases", {}).get(str(length))
        if not isinstance(official_case, dict):
            raise ValueError(f"official report omits length {length}")
        official_median = float(official_case["median_seconds"])
        comparison[str(length)] = {
            "native_samples_seconds": samples,
            "native_median_seconds": native_median,
            "native_median_tokens_per_second": length / native_median,
            "official_median_seconds": official_median,
            "official_median_tokens_per_second": length / official_median,
            "native_speedup": official_median / native_median,
        }

    model_loads = {float(item["model_load_seconds"]) for item in metrics}
    if len(model_loads) != 1:
        raise ValueError("native samples disagree on model-load time")
    report = {
        "schema_version": 1,
        "model_id": official.get("model_id"),
        "batch_size": 1,
        "dtype": "float32",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "official_load_seconds": official.get("load_seconds"),
        "native_load_seconds": model_loads.pop(),
        "cases": comparison,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
