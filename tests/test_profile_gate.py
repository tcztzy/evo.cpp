#!/usr/bin/env python3
"""Contract tests for exact-versus-approximate profile acceptance gates."""

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path


def write_npy(path: Path, rows: list[list[float]]) -> None:
    shape = (len(rows), len(rows[0]))
    header = repr({"descr": "<f4", "fortran_order": False, "shape": shape})
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    encoded = (header + " " * padding + "\n").encode("ascii")
    values = [value for row in rows for value in row]
    path.write_bytes(
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(encoded))
        + encoded
        + struct.pack(f"<{len(values)}f", *values)
    )


def write_bio(path: Path, profile: str, scores: list[float]) -> None:
    tasks = ["sequence", "sequence", "variant", "variant"]
    document = {
        "schema_version": 1,
        "profile": profile,
        "records": [
            {"id": f"record-{index}", "task": task, "score": score}
            for index, (task, score) in enumerate(zip(tasks, scores))
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    exact_logits = args.work_dir / "exact.npy"
    candidate_logits = args.work_dir / "candidate.npy"
    exact_bio = args.work_dir / "exact.json"
    candidate_bio = args.work_dir / "candidate.json"
    report = args.work_dir / "report.json"
    exact = [[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]]
    candidate = [[3.01, 0.99, -1.0], [0.01, 1.99, 1.01]]
    write_npy(exact_logits, exact)
    write_npy(candidate_logits, candidate)
    write_bio(exact_bio, "exact", [3.0, 2.0, 0.8, -0.4])
    write_bio(candidate_bio, "fast-q8-kv", [2.99, 2.01, 0.79, -0.39])
    command = [
        sys.executable,
        str(args.tool),
        "--exact-logits",
        str(exact_logits),
        "--candidate-logits",
        str(candidate_logits),
        "--exact-bio",
        str(exact_bio),
        "--candidate-bio",
        str(candidate_bio),
        "--report",
        str(report),
        "--min-cosine",
        "0.999",
        "--max-logit-mae",
        "0.02",
        "--min-top1-agreement",
        "1",
        "--min-rank-agreement",
        "1",
        "--min-variant-sign-agreement",
        "1",
    ]
    passed = subprocess.run(command, check=False, capture_output=True, text=True)
    if passed.returncode != 0:
        raise AssertionError(f"passing profile gate failed: {passed.stderr}")
    document = json.loads(report.read_text())
    if not document["passed"] or document["candidate_profile"] != "fast-q8-kv":
        raise AssertionError("passing profile report omitted profile/result metadata")

    candidate[0] = [0.0, 4.0, -1.0]
    write_npy(candidate_logits, candidate)
    write_bio(candidate_bio, "fast-q8-kv", [2.0, 3.0, -0.8, -0.4])
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    if failed.returncode != 1:
        raise AssertionError(
            f"failing profile gate returned {failed.returncode}: {failed.stderr}"
        )
    failure_report = json.loads(report.read_text())
    if failure_report["passed"] or all(failure_report["checks"].values()):
        raise AssertionError("failing profile report did not name failed gates")
    print("profile gate contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
