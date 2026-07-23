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
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    model = args.work_dir / "tiny-50l.evo2"
    expected_logits = args.work_dir / "expected-logits.f32"
    expected_decode = args.work_dir / "expected-decode.f32"
    expected_layer = args.work_dir / "expected-layer.f32"
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
        ],
        check=True,
    )
    result = subprocess.run(
        [
            str(args.binary),
            str(model),
            str(expected_logits),
            str(expected_decode),
            str(expected_layer),
            str(dump),
        ],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
