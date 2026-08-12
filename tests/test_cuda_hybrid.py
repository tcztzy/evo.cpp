#!/usr/bin/env python3
"""Exercise explicit CUDA-prefix/CPU-suffix placement on a tiny model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        return 77
    args.work_dir.mkdir(parents=True, exist_ok=True)
    model = args.work_dir / "hybrid.safetensors"
    sequence = args.work_dir / "sequence.fa"
    subprocess.run(
        [sys.executable, str(args.generator), "--model", str(model), "--config-only"],
        check=True,
    )
    sequence.write_text(">hybrid\nAACCGGTT\n", encoding="ascii")
    command = [
        str(args.binary), "-m", str(model), "--score", str(sequence),
        "--ctx", "12", "--gpu", "0", "--gpu-layers", "17",
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    document = json.loads(first.stdout)
    if (
        document["backend"] != "cpu+cuda"
        or document["profile"] != "cpu-f32"
        or document["gpu_layers"] != 17
        or document["scored_tokens"] != 7
        or first.stdout != second.stdout
        or '"gpu_layers":17' not in first.stderr
    ):
        raise AssertionError("hybrid backend/profile/placement contract failed")
    generated = subprocess.run(
        [
            str(args.binary), "-m", str(model), "-p", "AC", "-n", "2",
            "--ctx", "12", "--gpu", "0", "--gpu-layers", "17",
            "--top-k", "1",
        ],
        check=True,
        capture_output=True,
    )
    if len(generated.stdout) != 2:
        raise AssertionError("hybrid generation did not emit requested bytes")
    rejected = subprocess.run(
        [
            str(args.binary), "-m", str(model), "--score", str(sequence),
            "--gpu", "0", "--gpu-layers", "50",
        ],
        check=False,
        capture_output=True,
    )
    if (
        rejected.returncode == 0
        or b"smaller than the model layer count" not in rejected.stderr
    ):
        raise AssertionError("all-GPU placement did not require all-CUDA mode")

    exact_logits = args.work_dir / "exact.npy"
    cpu_logits = args.work_dir / "cpu.npy"

    def score(extra: list[str], output: Path) -> dict[str, object]:
        result = subprocess.run(
            [
                str(args.binary), "-m", str(model), "--score", str(sequence),
                "--ctx", "12", "--dump-logits", str(output), *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    exact_score = score(["--gpu", "0", "--profile", "exact"], exact_logits)
    cpu_score = score(["--backend", "cpu"], cpu_logits)

    variants = [(3, "C", "T"), (5, "G", "A"), (2, "A", "G")]

    def variant(extra: list[str], position: int, reference: str, alternate: str) -> float:
        result = subprocess.run(
            [
                str(args.binary), "variant-score", "-m", str(model),
                "--sequence", "AACCGGTT", "--position", str(position),
                "--ref", reference, "--alt", alternate, "--window", "6",
                "--strand", "both", "--normalization", "mean", "--ctx", "12",
                *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(json.loads(result.stdout)["score"])

    exact_records: list[dict[str, object]] = [
        {"id": "sequence", "task": "sequence", "score": exact_score["log_likelihood"]}
    ]
    cpu_records: list[dict[str, object]] = [
        {"id": "sequence", "task": "sequence", "score": cpu_score["log_likelihood"]}
    ]
    for index, (position, reference, alternate) in enumerate(variants):
        exact_records.append(
            {
                "id": f"variant-{index}",
                "task": "variant",
                "score": variant(["--gpu", "0", "--profile", "exact"], position, reference, alternate),
            }
        )
        cpu_records.append(
            {
                "id": f"variant-{index}",
                "task": "variant",
                "score": variant(["--backend", "cpu"], position, reference, alternate),
            }
        )
    exact_bio = args.work_dir / "exact-bio.json"
    cpu_bio = args.work_dir / "cpu-bio.json"
    exact_bio.write_text(json.dumps({"schema_version": 1, "profile": "exact", "records": exact_records}))
    cpu_bio.write_text(json.dumps({"schema_version": 1, "profile": "cpu-f32", "records": cpu_records}))
    report = args.work_dir / "cpu-profile-report.json"
    gate = subprocess.run(
        [
            sys.executable, str(args.gate),
            "--exact-logits", str(exact_logits),
            "--candidate-logits", str(cpu_logits),
            "--exact-bio", str(exact_bio),
            "--candidate-bio", str(cpu_bio),
            "--report", str(report),
            "--min-cosine", "0.999",
            "--max-logit-mae", "0.1",
            "--min-top1-agreement", "0.95",
            "--min-rank-agreement", "1",
            "--min-variant-sign-agreement", "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if gate.returncode != 0:
        raise AssertionError(report.read_text() if report.exists() else gate.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
