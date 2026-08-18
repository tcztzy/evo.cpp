#!/usr/bin/env python3
"""Compare native DeepGene RoFormer with the independent NumPy oracle."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("oracle root must be an object")
    return value


def close_vectors(actual: List[Any], expected: List[Any], label: str) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise AssertionError(label + " length differs")
    for index, (left, right) in enumerate(zip(actual, expected)):
        if (
            isinstance(left, bool)
            or not isinstance(left, (int, float))
            or not math.isfinite(float(left))
            or abs(float(left) - float(right)) > 3.0e-6
        ):
            raise AssertionError(
                "{}[{}] differs: {} vs {}".format(label, index, left, right)
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    args = parser.parse_args()
    expected = load(args.fixture)
    if (
        expected.get("schema") != "geneb-roformer-tiny-oracle-v1"
        or expected.get("generator") != "independent-numpy-f32"
        or expected.get("payload_first_tokens") != [1, 2, 5, 6]
    ):
        raise AssertionError("RoFormer oracle provenance/manual token order differs")
    with tempfile.TemporaryDirectory(prefix="geneb-roformer-oracle-") as directory:
        regenerated_path = Path(directory) / "oracle.json"
        completed = subprocess.run(
            [sys.executable, str(args.generator), "--output", str(regenerated_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode == 77:
            return 77
        if completed.returncode != 0:
            raise AssertionError("RoFormer oracle generator failed: " + completed.stderr)
        if load(regenerated_path) != expected:
            raise AssertionError("checked-in RoFormer oracle is stale")
    completed = subprocess.run(
        [str(args.binary), "--dump-tiny"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise AssertionError("native RoFormer runner failed: " + completed.stderr)
    actual = json.loads(completed.stdout)
    for field in ("rows", "width", "payload_first_tokens"):
        if actual.get(field) != expected.get(field):
            raise AssertionError(field + " differs")
    actual_captures = actual.get("captures")
    expected_captures = expected.get("captures")
    if not isinstance(actual_captures, list) or len(actual_captures) != len(expected_captures):
        raise AssertionError("capture list differs")
    for left, right in zip(actual_captures, expected_captures):
        if left.get("layer") != right.get("layer"):
            raise AssertionError("capture layer differs")
        close_vectors(left.get("values"), right.get("values"), "capture")
    close_vectors(actual.get("final_hidden"), expected.get("final_hidden"), "final_hidden")
    close_vectors(actual.get("pooled"), expected.get("pooled"), "pooled")
    print("GENEB RoFormer independent vectors passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
