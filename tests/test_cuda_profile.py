#!/usr/bin/env python3
"""Run exact-versus-Q8 numeric and biological gates on the CUDA fixture."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def visible_gpu_count() -> int:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return len(
            [
                item
                for item in configured.split(",")
                if item.strip() and item.strip() != "-1"
            ]
        )
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return 0
    result = subprocess.run(
        [executable, "--query-gpu=index", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    return len(result.stdout.splitlines()) if result.returncode == 0 else 0


def run_json(command: list[str]) -> tuple[dict[str, object], dict[str, object]]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    document = json.loads(result.stdout)
    prefix = "evo_metrics "
    metric_line = next(
        (line for line in result.stderr.splitlines() if line.startswith(prefix)), None
    )
    if metric_line is None:
        raise AssertionError("profile command omitted runtime metrics")
    return document, json.loads(metric_line[len(prefix) :])


def write_bio(path: Path, profile: str, records: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "profile": profile, "records": records}),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    if visible_gpu_count() < 1:
        print("SKIP: one visible CUDA device is required")
        return 77
    args.work_dir.mkdir(parents=True, exist_ok=True)
    model = args.work_dir / "tiny-profile-50l-512v.safetensors"
    subprocess.run(
        [
            sys.executable,
            str(args.generator),
            "--model",
            str(model),
            "--expected-logits",
            str(args.work_dir / "expected-logits.f32"),
            "--expected-decode",
            str(args.work_dir / "expected-decode.f32"),
            "--expected-layer",
            str(args.work_dir / "expected-layer.f32"),
        ],
        check=True,
    )
    sequence_path = args.work_dir / "sequence.txt"
    sequence_path.write_text("ACGTACGT")
    exact_logits = args.work_dir / "exact.npy"
    fast_logits = args.work_dir / "fast-q8.npy"

    def score(profile: str, logits: Path) -> tuple[dict[str, object], dict[str, object]]:
        return run_json(
            [
                str(args.binary),
                "-m",
                str(model),
                "--score",
                str(sequence_path),
                "--ctx",
                "12",
                "--gpu",
                "0",
                "--profile",
                profile,
                "--dump-logits",
                str(logits),
            ]
        )

    exact_score, exact_metrics = score("exact", exact_logits)
    fast_score, fast_metrics = score("fast-q8-kv", fast_logits)
    if (
        exact_score["profile"] != "exact"
        or exact_metrics["profile"] != "exact"
        or exact_metrics["kv_cache"] != "bf16_contiguous"
        or fast_score["profile"] != "fast-q8-kv"
        or fast_metrics["profile"] != "fast-q8-kv"
        or fast_metrics["kv_cache"] != "q8_paged"
    ):
        raise AssertionError("CLI output/metrics did not expose the selected profile")

    variants = [
        ("variant-c3t", 3, "C", "T"),
        ("variant-g5a", 5, "G", "A"),
        ("variant-a2g", 2, "A", "G"),
    ]

    def variant(profile: str, position: int, reference: str, alternate: str) -> float:
        document, metrics = run_json(
            [
                str(args.binary),
                "variant-score",
                "-m",
                str(model),
                "--sequence",
                "AACCGGTT",
                "--position",
                str(position),
                "--ref",
                reference,
                "--alt",
                alternate,
                "--window",
                "6",
                "--strand",
                "both",
                "--normalization",
                "mean",
                "--ctx",
                "12",
                "--gpu",
                "0",
                "--profile",
                profile,
            ]
        )
        if metrics["profile"] != profile:
            raise AssertionError("variant metrics omitted its execution profile")
        return float(document["score"])

    exact_records: list[dict[str, object]] = [
        {
            "id": "sequence-likelihood",
            "task": "sequence",
            "score": exact_score["log_likelihood"],
        }
    ]
    fast_records: list[dict[str, object]] = [
        {
            "id": "sequence-likelihood",
            "task": "sequence",
            "score": fast_score["log_likelihood"],
        }
    ]
    for identifier, position, reference, alternate in variants:
        exact_records.append(
            {
                "id": identifier,
                "task": "variant",
                "score": variant("exact", position, reference, alternate),
            }
        )
        fast_records.append(
            {
                "id": identifier,
                "task": "variant",
                "score": variant("fast-q8-kv", position, reference, alternate),
            }
        )
    exact_bio = args.work_dir / "exact-bio.json"
    fast_bio = args.work_dir / "fast-q8-bio.json"
    report = args.work_dir / "profile-report.json"
    write_bio(exact_bio, "exact", exact_records)
    write_bio(fast_bio, "fast-q8-kv", fast_records)
    gate = subprocess.run(
        [
            sys.executable,
            str(args.gate),
            "--exact-logits",
            str(exact_logits),
            "--candidate-logits",
            str(fast_logits),
            "--exact-bio",
            str(exact_bio),
            "--candidate-bio",
            str(fast_bio),
            "--report",
            str(report),
            "--min-cosine",
            "0.9999",
            "--max-logit-mae",
            "0.01",
            "--min-top1-agreement",
            "1",
            "--min-rank-agreement",
            "1",
            "--min-variant-sign-agreement",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if gate.returncode != 0:
        detail = report.read_text() if report.exists() else gate.stderr
        raise AssertionError(f"CUDA fast-Q8 acceptance gate failed:\n{detail}")
    if not json.loads(report.read_text())["passed"]:
        raise AssertionError("CUDA profile report did not record a pass")
    print("CUDA execution profile gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
