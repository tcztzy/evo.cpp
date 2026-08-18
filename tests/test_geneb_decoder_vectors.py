#!/usr/bin/env python3
"""Compare the native GENEB decoder with the committed independent oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


def vector_digest(order: list[str], vectors: dict[str, list[float]]) -> str:
    digest = hashlib.sha256()
    for name in order:
        for value in vectors[name]:
            digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def run_generator(generator: Path, fixture: Path) -> int:
    result = subprocess.run(
        [sys.executable, str(generator), "--check", str(fixture)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 77:
        print(result.stderr.strip() or "NumPy unavailable; skipping oracle regeneration")
        return 77
    if result.returncode != 0:
        raise AssertionError(
            "GENEB decoder oracle generator check failed:\n"
            + result.stdout
            + result.stderr
        )
    return 0


def compare_vectors(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    expected_vectors = expected["vectors"]
    actual_vectors = actual.get("vectors")
    if not isinstance(actual_vectors, dict):
        raise AssertionError("native output lacks a vectors object")
    if expected_vectors.keys() != actual_vectors.keys():
        raise AssertionError(
            "vector names differ: "
            f"expected {sorted(expected_vectors)}, got {sorted(actual_vectors)}"
        )
    for name in expected["vector_order"]:
        wanted = expected_vectors[name]
        got = actual_vectors[name]
        if len(wanted) != len(got):
            raise AssertionError(
                f"{name}: expected {len(wanted)} values, got {len(got)}"
            )
        tolerance = expected["tolerance"]["bf16" if name.startswith("bf16.") else "f32"]
        for index, (wanted_value, got_value) in enumerate(
            zip(wanted, got, strict=True)
        ):
            if name.startswith("bf16."):
                if struct.pack("<f", wanted_value) == struct.pack("<f", got_value):
                    continue
                raise AssertionError(
                    f"{name}[{index}]: expected exact BF16 {wanted_value:.9g}, "
                    f"got {got_value:.9g}"
                )
            if not math.isclose(
                wanted_value,
                got_value,
                abs_tol=tolerance["atol"],
                rel_tol=tolerance["rtol"],
            ):
                raise AssertionError(
                    f"{name}[{index}]: expected {wanted_value:.9g}, "
                    f"got {got_value:.9g}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    args = parser.parse_args()

    skipped = run_generator(args.generator, args.fixture)
    if skipped == 77:
        return 77
    expected = json.loads(args.fixture.read_text(encoding="utf-8"))
    if vector_digest(expected["vector_order"], expected["vectors"]) != expected[
        "vector_sha256"
    ]:
        raise AssertionError("committed GENEB decoder vector digest is invalid")

    process = subprocess.run(
        [str(args.binary), "--dump-json"],
        check=True,
        text=True,
        capture_output=True,
    )
    actual = json.loads(process.stdout)
    compare_vectors(expected, actual)
    print(
        f"validated {len(expected['vectors'])} GENEB decoder vectors; "
        f"sha256={expected['vector_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
