#!/usr/bin/env python3
"""Compare native JanusDNA with the independent NumPy-F32 tiny oracle."""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise AssertionError("oracle root must be an object")
    return value


def close_vectors(actual: List[Any], expected: List[Any], label: str) -> None:
    if (
        not isinstance(actual, list)
        or not isinstance(expected, list)
        or len(actual) != len(expected)
    ):
        raise AssertionError("{} length differs".format(label))
    for index, (left, right) in enumerate(zip(actual, expected)):
        if (
            isinstance(left, bool)
            or not isinstance(left, (int, float))
            or isinstance(right, bool)
            or not isinstance(right, (int, float))
            or not math.isfinite(float(left))
            or not math.isfinite(float(right))
            or abs(float(left) - float(right)) > 7.0e-5
        ):
            raise AssertionError(
                "{}[{}] differs: {} vs {}".format(label, index, left, right)
            )


def compare_case(actual: Dict[str, Any], expected: Dict[str, Any]) -> None:
    for field in ("variant", "rows", "width"):
        if actual.get(field) != expected.get(field):
            raise AssertionError("{} differs".format(field))
    actual_captures = actual.get("captures")
    expected_captures = expected.get("captures")
    if (
        not isinstance(actual_captures, list)
        or not isinstance(expected_captures, list)
        or len(actual_captures) != len(expected_captures)
    ):
        raise AssertionError("capture list differs")
    for index, (left, right) in enumerate(zip(actual_captures, expected_captures)):
        if left.get("layer") != right.get("layer"):
            raise AssertionError("capture layer differs")
        close_vectors(
            left.get("values"), right.get("values"), "capture {}".format(index)
        )
    close_vectors(
        actual.get("final_hidden"), expected.get("final_hidden"), "final_hidden"
    )
    close_vectors(actual.get("pooled"), expected.get("pooled"), "pooled")
    post_norm = expected.get("post_final_norm")
    final_hidden = expected.get("final_hidden")
    if not isinstance(post_norm, list) or not isinstance(final_hidden, list):
        raise AssertionError("2x-final hidden evidence is missing")
    close_vectors(
        final_hidden, [2.0 * float(value) for value in post_norm], "2x final hidden"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    args = parser.parse_args()

    fixture = load(args.fixture)
    if (
        fixture.get("schema") != "geneb-janusdna-tiny-oracle-v1"
        or fixture.get("generator") != "independent-numpy-f32"
    ):
        raise AssertionError("fixture provenance differs")
    micro = fixture.get("micro")
    if (
        not isinstance(micro, dict)
        or micro.get("top2_equal_indices") != [0, 1]
        or micro.get("middle_mask_forward") != [1, 1, 1, 0]
        or micro.get("middle_mask_reverse_bug") != [1, 1, 1, 0]
        or micro.get("flex_scale") != 0.5
    ):
        raise AssertionError("non-intuitive Janus micro-contract differs")
    flex_mask = micro.get("flex_mask")
    if (
        not isinstance(flex_mask, list)
        or len(flex_mask) != 8
        or any(not isinstance(row, list) or len(row) != 8 for row in flex_mask)
    ):
        raise AssertionError("FlexAttention four-condition mask differs")

    with tempfile.TemporaryDirectory(prefix="geneb-janus-oracle-") as directory:
        regenerated_path = Path(directory) / "oracle.json"
        completed = subprocess.run(
            [sys.executable, str(args.generator), "--output", str(regenerated_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            if completed.returncode == 77 or "NumPy is required" in completed.stderr:
                return 77
            raise AssertionError("oracle generator failed: " + completed.stderr)
        regenerated = load(regenerated_path)
    if regenerated != fixture:
        raise AssertionError("checked-in Janus oracle is stale")

    cases = fixture.get("cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise AssertionError("fixture must contain two Janus variants")
    for expected in cases:
        completed = subprocess.run(
            [str(args.binary), "--dump-tiny", expected["variant"]],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise AssertionError("native tiny runner failed: " + completed.stderr)
        compare_case(json.loads(completed.stdout), expected)
    print("GENEB JanusDNA independent vectors passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
