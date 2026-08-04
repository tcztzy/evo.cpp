#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise resumable Evo 2 quality-gate artifact summarization."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-tool", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = args.work_dir / "prompts.csv"
    output_path = args.work_dir / "quality.json"
    binary_path = args.work_dir / "unused-evo"
    binary_path.write_bytes(b"fixture-binary")
    artifact_dir = args.work_dir / "quality-artifacts"
    artifact_dir.mkdir(exist_ok=True)

    sequences = ["AACCGGTT", "TTGGCCAA", "ACGTACGT", "TGCATGCA"]
    with prompts_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["sequence"])
        writer.writerows([sequence] for sequence in sequences)

    for index, sequence in enumerate(sequences):
        midpoint = 2 * (len(sequence) // 4)
        target = sequence[midpoint : midpoint + 2].encode()
        (artifact_dir / f"generated-{index}.bin").write_bytes(target)
        (artifact_dir / f"logits-{index}.npy").write_bytes(b"fixture")
        (artifact_dir / f"record-{index}.json").write_text(
            json.dumps({"metrics": {"peak_gpu_bytes": index + 1}}),
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            sys.executable,
            str(args.quality_tool),
            "--binary",
            str(binary_path),
            "--model",
            "unused-model.safetensors",
            "--model-sha256",
            "a" * 64,
            "--prompts",
            str(prompts_path),
            "--output",
            str(output_path),
            "--num-tokens",
            "2",
            "--expected-identity",
            "100",
            "--tolerance",
            "0",
            "--summarize-existing",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"quality-gate summarization failed:\n{result.stdout}{result.stderr}"
        )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if not report["passed"] or report["mean_identity_percent"] != 100.0:
        raise AssertionError("existing artifacts did not pass the exact gate")
    if len(report["records"]) != 4:
        raise AssertionError("quality report did not consolidate four prompts")
    if report["binary_sha256"] != (
        "74dd6752f5b83360c8fb780f06d74e5edb2712ea55a59b02a263584b091993a4"
    ):
        raise AssertionError("quality report did not identify its inference binary")
    if report["model_sha256"] != "a" * 64:
        raise AssertionError("quality report did not preserve its model digest")
    if report["records"][3]["metrics"]["peak_gpu_bytes"] != 4:
        raise AssertionError("quality report did not preserve partial-run metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
