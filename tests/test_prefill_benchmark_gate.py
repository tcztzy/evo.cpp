#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
import sys
from pathlib import Path


def write_metrics(
    path: Path,
    *,
    tokens: int,
    seconds: float,
    load_seconds: float = 2.0,
) -> None:
    record = {
        "model_load_seconds": load_seconds,
        "prefill_tokens": tokens,
        "prefill_seconds": seconds,
        "prefill_tokens_per_second": tokens / seconds,
    }
    path.write_text(
        "diagnostic\n"
        + "evo_metrics "
        + json.dumps(record, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def command(
    tool: Path, artifact_dir: Path, official: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        str(tool),
        "--artifact-dir",
        str(artifact_dir),
        "--official-metrics",
        str(official),
        "--repeat-count",
        "16=20",
        "--repeat-count",
        "128=20",
        "--repeat-count",
        "1024=10",
        "--minimum-rate",
        "16=1000",
        "--minimum-rate",
        "128=7000",
        "--minimum-rate",
        "1024=9000",
        "--maximum-load-seconds",
        "3",
        "--output",
        str(output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    official = args.work_dir / "official.json"
    output = args.work_dir / "report.json"
    official.write_text(
        json.dumps(
            {
                "prefill": {
                    "16": {"median_tokens_per_second": 600.0},
                    "128": {"median_tokens_per_second": 4800.0},
                    "1024": {"median_tokens_per_second": 9400.0},
                }
            }
        ),
        encoding="utf-8",
    )
    write_metrics(args.work_dir / "metrics_16.log", tokens=16, seconds=0.04)
    write_metrics(args.work_dir / "metrics_128.log", tokens=128, seconds=0.017)
    write_metrics(
        args.work_dir / "metrics_1024.log", tokens=1024, seconds=0.125
    )
    write_metrics(
        args.work_dir / "repeat_16.log", tokens=320, seconds=0.25
    )
    write_metrics(
        args.work_dir / "repeat_128.log", tokens=2560, seconds=0.34
    )
    write_metrics(
        args.work_dir / "repeat_1024.log", tokens=10240, seconds=1.05
    )

    accepted = subprocess.run(
        command(args.tool, args.work_dir, official, output),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if accepted.returncode != 0:
        raise AssertionError(accepted.stderr + accepted.stdout)
    report = json.loads(output.read_text(encoding="utf-8"))
    if (
        not report["passed"]
        or not report["cases"]["1024"]["rate_passed"]
        or report["cases"]["1024"]["official_ratio"] <= 1.0
    ):
        raise AssertionError("valid prefill benchmark did not pass")

    write_metrics(
        args.work_dir / "repeat_1024.log", tokens=10240, seconds=2.0
    )
    rejected = subprocess.run(
        command(args.tool, args.work_dir, official, output),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if rejected.returncode != 1:
        raise AssertionError("slow prefill benchmark was not rejected")
    report = json.loads(output.read_text(encoding="utf-8"))
    if report["passed"] or report["cases"]["1024"]["rate_passed"]:
        raise AssertionError("slow prefill report has incorrect gate status")

    write_metrics(
        args.work_dir / "repeat_1024.log", tokens=1024, seconds=0.1
    )
    malformed = subprocess.run(
        command(args.tool, args.work_dir, official, output),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if malformed.returncode == 0 or "expected 1024 first tokens" not in (
        malformed.stderr + malformed.stdout
    ):
        raise AssertionError("malformed repeated-token count was not rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
