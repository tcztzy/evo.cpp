#!/usr/bin/env python3
"""Compare the native GENEB BERT encoder with the independent NumPy oracle."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence


def run_generator(generator: Path, fixture: Path) -> int:
    process = subprocess.run(
        [sys.executable, str(generator), "--check", str(fixture)],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode == 77:
        print(process.stderr.strip() or "NumPy unavailable; skipping BERT oracle")
        return 77
    if process.returncode != 0:
        raise AssertionError(
            "GENEB BERT oracle generator check failed:\n"
            + process.stdout
            + process.stderr
        )
    return 0


def compare_array(
    label: str,
    wanted: Sequence[float],
    got: Sequence[float],
    tolerance: float,
) -> None:
    if len(wanted) != len(got):
        raise AssertionError(
            "%s: expected %d values, got %d" % (label, len(wanted), len(got))
        )
    for index, (wanted_value, got_value) in enumerate(zip(wanted, got)):
        if not math.isclose(
            wanted_value, got_value, abs_tol=tolerance, rel_tol=tolerance
        ):
            raise AssertionError(
                "%s[%d]: expected %.9g, got %.9g"
                % (label, index, wanted_value, got_value)
            )


def compare_case(
    expected: Dict[str, Any], actual: Dict[str, Any], tolerance: float
) -> None:
    for scalar in ("case", "rows", "width"):
        if actual.get(scalar) != expected[scalar]:
            raise AssertionError(
                "%s differs: expected %r, got %r"
                % (scalar, expected[scalar], actual.get(scalar))
            )
    wanted_captures = expected["captures"]
    got_captures = actual.get("captures")
    if (
        not isinstance(got_captures, dict)
        or got_captures.keys() != wanted_captures.keys()
    ):
        raise AssertionError(
            "capture layers differ: expected %r, got %r"
            % (sorted(wanted_captures), sorted(got_captures or {}))
        )
    for layer, wanted in wanted_captures.items():
        compare_array("capture %s" % layer, wanted, got_captures[layer], tolerance)
    compare_array(
        "final_hidden", expected["final_hidden"], actual["final_hidden"], tolerance
    )
    compare_array("pooled", expected["pooled"], actual["pooled"], tolerance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    args = parser.parse_args()

    if run_generator(args.generator, args.fixture) == 77:
        return 77
    expected = json.loads(args.fixture.read_text(encoding="utf-8"))
    if expected.get("format") != "geneb-bert-oracle-v1":
        raise AssertionError("unexpected GENEB BERT oracle format")
    tolerance = float(expected["reference"]["max_abs_tolerance"])
    for case_name, wanted in expected["cases"].items():
        process = subprocess.run(
            [str(args.binary), "--dump-case", case_name],
            check=True,
            text=True,
            capture_output=True,
        )
        compare_case(wanted, json.loads(process.stdout), tolerance)
    print("validated %d independent GENEB BERT cases" % len(expected["cases"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
