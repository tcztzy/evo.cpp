#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native GENEB Mamba fixtures with an independent NumPy oracle."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def run_json(command: Sequence[str]) -> Mapping[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode == 77:
        raise SystemExit(77)
    if completed.returncode != 0:
        raise AssertionError(
            "command failed: %s\n%s" % (" ".join(command), completed.stderr)
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise AssertionError("JSON result is not an object")
    return value


def compare(
    expected: Mapping[str, Any], actual: Mapping[str, Any], tolerance: float
) -> None:
    expected_vectors = expected.get("vectors")
    actual_vectors = actual.get("vectors")
    if not isinstance(expected_vectors, dict) or not isinstance(actual_vectors, dict):
        raise AssertionError("vector payload is missing")
    if set(expected_vectors) != set(actual_vectors):
        raise AssertionError("native/oracle vector keys differ")
    for name in sorted(expected_vectors):
        left = expected_vectors[name]
        right = actual_vectors[name]
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            raise AssertionError("vector shape differs for %s" % name)
        maximum = max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)
        if maximum > tolerance:
            raise AssertionError("%s max_abs %.9g exceeds %.9g" % (name, maximum, tolerance))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    args = parser.parse_args()
    expected = json.loads(args.fixture.read_text(encoding="utf-8"))
    generated = run_json([sys.executable, str(args.generator)])
    compare(expected, generated, 0.0)
    native = run_json([str(args.binary), "--dump-fixture"])
    compare(expected, native, 4.0e-5)
    print("GENEB Mamba NumPy oracle vectors passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
