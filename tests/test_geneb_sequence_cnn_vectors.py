#!/usr/bin/env python3
"""Compare native Enformer/SPACE tiny outputs to an independent NumPy oracle."""

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
        raise AssertionError("sequence-CNN oracle root must be an object")
    return value


def close_vectors(actual: List[Any], expected: List[Any], label: str) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise AssertionError(label + " length differs")
    for index, (left, right) in enumerate(zip(actual, expected)):
        if (
            isinstance(left, bool)
            or not isinstance(left, (int, float))
            or not math.isfinite(float(left))
            or abs(float(left) - float(right)) > 5.0e-6
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
        expected.get("schema") != "geneb-sequence-cnn-tiny-oracle-v1"
        or expected.get("generator") != "independent-numpy-f32"
        or expected.get("sequence") != "acguNXTTAA"
        or expected.get("enformer_input_policy")
        != "uppercase-prefix-crop-right-zero-pad-u-invalid"
        or expected.get("space_input_policy")
        != "uppercase-center-crop-symmetric-n-pad-u-to-t"
    ):
        raise AssertionError("sequence-CNN oracle provenance/input policy differs")
    with tempfile.TemporaryDirectory(prefix="geneb-sequence-cnn-oracle-") as directory:
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
            raise AssertionError(
                "sequence-CNN oracle generator failed: " + completed.stderr
            )
        if load(regenerated_path) != expected:
            raise AssertionError("checked-in sequence-CNN oracle is stale")
    completed = subprocess.run(
        [str(args.binary), "--dump-tiny"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise AssertionError("native sequence-CNN runner failed: " + completed.stderr)
    actual = json.loads(completed.stdout)
    for variant in ("enformer", "space"):
        left = actual.get(variant)
        right = expected.get(variant)
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise AssertionError(variant + " result is missing")
        for field in ("rows", "width"):
            if left.get(field) != right.get(field):
                raise AssertionError(variant + " " + field + " differs")
        close_vectors(
            left.get("final_hidden"), right.get("final_hidden"), variant + " final"
        )
        close_vectors(left.get("pooled"), right.get("pooled"), variant + " pooled")
    print("GENEB sequence-CNN independent vectors passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
