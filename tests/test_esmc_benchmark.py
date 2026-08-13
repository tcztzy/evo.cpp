#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract-test the reproducible ESMC benchmark report combiner."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def run(tool: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(tool), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-tool", required=True, type=Path)
    parser.add_argument("--summarizer", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    help_result = run(args.official_tool, "--help")
    if help_result.returncode != 0 or "--reference-commit" not in help_result.stdout:
        raise AssertionError("official ESMC benchmark omitted its pinning contract")

    official = args.work_dir / "official.json"
    official.write_text(
        json.dumps(
            {
                "model_id": "esmc_test",
                "load_seconds": 3.0,
                "cases": {
                    "128": {"median_seconds": 0.2},
                    "512": {"median_seconds": 0.4},
                },
            }
        ),
        encoding="utf-8",
    )
    native = args.work_dir / "native.log"
    samples = []
    aggregate_seconds = 0.0
    aggregate_tokens = 0
    record_index = 0
    for length, values in ((128, (0.5, 0.1, 0.2)), (512, (0.8, 0.2, 0.4))):
        for seconds in values:
            aggregate_seconds += seconds
            aggregate_tokens += length
            samples.append(
                "evo_record_metrics "
                + json.dumps(
                    {
                        "architecture": "ESMC",
                        "record_index": record_index,
                        "prefill_tokens": length,
                        "prefill_seconds": seconds,
                    }
                )
            )
            record_index += 1
    aggregate = "evo_metrics " + json.dumps(
        {
            "architecture": "ESMC",
            "model_load_seconds": 2.0,
            "prefill_tokens": aggregate_tokens,
            "prefill_seconds": aggregate_seconds,
        }
    )
    native.write_text("\n".join([*samples, aggregate]) + "\n", encoding="utf-8")
    output = args.work_dir / "comparison.json"
    result = run(
        args.summarizer,
        "--official",
        str(official),
        "--native-log",
        str(native),
        "--output",
        str(output),
        "--length",
        "128",
        "--length",
        "512",
        "--warmups",
        "1",
        "--repeats",
        "2",
    )
    if result.returncode != 0:
        raise AssertionError(f"summarizer failed: {result.stderr}")
    report = json.loads(output.read_text(encoding="utf-8"))
    if report["native_load_seconds"] != 2.0:
        raise AssertionError("native model-load time was not preserved")
    if not math.isclose(
        report["cases"]["128"]["native_median_seconds"], 0.15, abs_tol=1e-12
    ):
        raise AssertionError("warmup exclusion or native median is wrong")
    if not math.isclose(
        report["cases"]["128"]["native_speedup"], 0.2 / 0.15, abs_tol=1e-12
    ):
        raise AssertionError("official/native speedup direction is wrong")

    output.unlink()
    incomplete = args.work_dir / "incomplete.log"
    incomplete.write_text(samples[0] + "\n", encoding="utf-8")
    failure = run(
        args.summarizer,
        "--official",
        str(official),
        "--native-log",
        str(incomplete),
        "--output",
        str(output),
        "--length",
        "128",
        "--length",
        "512",
        "--warmups",
        "1",
        "--repeats",
        "2",
    )
    if failure.returncode == 0 or "expected 6" not in failure.stderr:
        raise AssertionError("summarizer accepted an incomplete native timing log")

    print("ESMC benchmark contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
