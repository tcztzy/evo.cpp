#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare native GENEB ESM output with the committed NumPy oracle."""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


EXACT_AFFINE_LAYER_NORM_SHA256 = (
    "933a1b8646bc1ccc4e499881e748058c9c2d1f64831b29991060024530fae1a5"
)


def vector_digest(order: List[str], vectors: Mapping[str, List[float]]) -> str:
    digest = hashlib.sha256()
    for name in order:
        for value in vectors[name]:
            digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def run_generator(generator: Path, fixture: Path) -> int:
    result = subprocess.run(
        [sys.executable, str(generator), "--check", str(fixture)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if result.returncode == 77:
        print(result.stderr.strip() or "NumPy unavailable; skipping ESM oracle")
        return 77
    if result.returncode != 0:
        raise AssertionError(
            "GENEB ESM oracle generator check failed:\n"
            + result.stdout
            + result.stderr
        )
    return 0


def verify_exact_affine_layer_norm(binary: Path, require_torch: bool) -> None:
    process = subprocess.run(
        [str(binary), "--dump-exact-layer-norm-bits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if process.returncode == 77:
        print("exact Agro-NT-1B LayerNorm correctly unavailable on this host")
        return
    if process.returncode != 0:
        raise AssertionError(
            "native exact affine LayerNorm vector failed:\n" + process.stderr
        )
    document = json.loads(process.stdout)
    bits = document.get("bits")
    if (
        not isinstance(bits, list)
        or len(bits) != 3000
        or any(
            not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF
            for value in bits
        )
    ):
        raise AssertionError("native exact affine LayerNorm bit vector is malformed")
    raw = b"".join(struct.pack("<I", value) for value in bits)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != EXACT_AFFINE_LAYER_NORM_SHA256:
        raise AssertionError(
            "pinned Torch 2.1.2 Apple-arm64 affine LayerNorm bits differ: got %s"
            % actual_sha256
        )

    if require_torch:
        try:
            import numpy as np
            import torch
            import torch.nn.functional as torch_functional
        except ModuleNotFoundError:
            print("PyTorch/NumPy unavailable; skipping pinned ESM Torch oracle")
            raise SystemExit(77)
        if torch.__version__.split("+")[0] == "2.1.2":
            values = np.asarray(
                [
                    np.float32(
                        np.float32(((index + 3) * 37) % 257 - 128)
                        / np.float32(31.0)
                    )
                    for index in range(3000)
                ],
                dtype=np.float32,
            ).reshape(2, 1500)
            scale = np.asarray(
                [
                    np.float32(
                        np.float32(1.0)
                        + np.float32(
                            np.float32(((index + 5) * 19) % 67 - 33)
                            / np.float32(211.0)
                        )
                    )
                    for index in range(1500)
                ],
                dtype=np.float32,
            )
            bias = np.asarray(
                [
                    np.float32(
                        np.float32(((index + 7) * 23) % 59 - 29)
                        / np.float32(307.0)
                    )
                    for index in range(1500)
                ],
                dtype=np.float32,
            )
            expected = (
                torch_functional.layer_norm(
                    torch.from_numpy(values),
                    (1500,),
                    weight=torch.from_numpy(scale),
                    bias=torch.from_numpy(bias),
                    eps=1.0e-12,
                )
                .contiguous()
                .numpy()
                .astype("<f4", copy=False)
                .tobytes(order="C")
            )
            if raw != expected:
                raise AssertionError(
                    "native exact affine LayerNorm is not bit-equal to pinned "
                    "Torch 2.1.2"
                )
    print("validated pinned exact Agro-NT-1B affine LayerNorm bit vector")


def compare_vectors(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    expected_vectors = expected["vectors"]
    actual_vectors = actual.get("vectors")
    if not isinstance(actual_vectors, dict):
        raise AssertionError("native output lacks vectors object")
    if set(expected_vectors) != set(actual_vectors):
        raise AssertionError(
            "vector names differ: expected %s got %s"
            % (sorted(expected_vectors), sorted(actual_vectors))
        )
    tolerance = expected["tolerance"]
    for name in expected["vector_order"]:
        wanted = expected_vectors[name]
        got = actual_vectors[name]
        if len(wanted) != len(got):
            raise AssertionError(
                "%s: expected %d values, got %d" % (name, len(wanted), len(got))
            )
        for index, pair in enumerate(zip(wanted, got)):
            wanted_value, got_value = pair
            if not math.isclose(
                wanted_value,
                got_value,
                abs_tol=tolerance["atol"],
                rel_tol=tolerance["rtol"],
            ):
                raise AssertionError(
                    "%s[%d]: expected %.9g, got %.9g"
                    % (name, index, wanted_value, got_value)
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    verify_exact_affine_layer_norm(args.binary, args.require_torch)
    skipped = run_generator(args.generator, args.fixture)
    if skipped == 77:
        return 77
    expected = json.loads(args.fixture.read_text(encoding="utf-8"))
    if vector_digest(expected["vector_order"], expected["vectors"]) != expected[
        "vector_sha256"
    ]:
        raise AssertionError("committed GENEB ESM vector digest is invalid")
    process = subprocess.run(
        [str(args.binary), "--dump-json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if process.returncode != 0:
        raise AssertionError("native vector dump failed: " + process.stderr)
    actual = json.loads(process.stdout)
    compare_vectors(expected, actual)
    print(
        "validated %d GENEB ESM vectors; sha256=%s"
        % (len(expected["vectors"]), expected["vector_sha256"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
