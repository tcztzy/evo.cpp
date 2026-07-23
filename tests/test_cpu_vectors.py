#!/usr/bin/env python3
"""Compare native CPU math output with committed independent Python vectors."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    args = parser.parse_args()

    subprocess.run([sys.executable, str(args.generator), "--check", str(args.fixture)], check=True)
    expected = json.loads(args.fixture.read_text(encoding="utf-8"))
    actual = json.loads(
        subprocess.run(
            [str(args.binary), "--dump-json"], check=True, text=True, capture_output=True
        ).stdout
    )
    atol = expected["tolerance"]["atol"]
    rtol = expected["tolerance"]["rtol"]
    expected_vectors = expected["vectors"]
    actual_vectors = actual["vectors"]
    if expected_vectors.keys() != actual_vectors.keys():
        raise AssertionError(
            f"vector names differ: expected {sorted(expected_vectors)}, got {sorted(actual_vectors)}"
        )
    for name, expected_values in expected_vectors.items():
        actual_values = actual_vectors[name]
        if len(expected_values) != len(actual_values):
            raise AssertionError(
                f"{name}: expected {len(expected_values)} values, got {len(actual_values)}"
            )
        for index, (wanted, got) in enumerate(zip(expected_values, actual_values, strict=True)):
            if not math.isclose(wanted, got, abs_tol=atol, rel_tol=rtol):
                raise AssertionError(f"{name}[{index}]: expected {wanted:.9g}, got {got:.9g}")
    print(f"validated {len(expected_vectors)} CPU reference vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
