#!/usr/bin/env python3
"""Generate and execute the tiny native CUDA model integration fixture."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--check-precision-metadata", action="store_true")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    model = args.work_dir / "tiny-50l.evo2"
    expected_logits = args.work_dir / "expected-logits.f32"
    expected_decode = args.work_dir / "expected-decode.f32"
    expected_layer = args.work_dir / "expected-layer.f32"
    expected_chunked = args.work_dir / "expected-chunked.f32"
    dump = args.work_dir / "layer-17.npy"
    subprocess.run(
        [
            sys.executable,
            str(args.generator),
            "--model",
            str(model),
            "--expected-logits",
            str(expected_logits),
            "--expected-decode",
            str(expected_decode),
            "--expected-layer",
            str(expected_layer),
            "--expected-chunked",
            str(expected_chunked),
        ],
        check=True,
    )
    precision_models: list[Path] = []
    if args.check_precision_metadata:
        variants = (
            (
                args.work_dir / "invalid-projection-mismatch.evo2",
                ["--hyena-projection-dtype", "E4M3_SW"],
            ),
            (
                args.work_dir / "invalid-projection-dtype.evo2",
                ["--hyena-projection-dtype", "FP16"],
            ),
            (
                args.work_dir / "invalid-bf16-fp8-residue.evo2",
                ["--add-fp8-residue"],
            ),
        )
        for path, extra in variants:
            subprocess.run(
                [
                    sys.executable,
                    str(args.generator),
                    "--model",
                    str(path),
                    "--config-only",
                    *extra,
                ],
                check=True,
            )
            precision_models.append(path)
    result = subprocess.run(
        [
            str(args.binary),
            str(model),
            str(expected_logits),
            str(expected_decode),
            str(expected_layer),
            str(dump),
            str(expected_chunked),
            *(str(path) for path in precision_models),
        ],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
