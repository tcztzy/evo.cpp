#!/usr/bin/env python3
"""Generate and execute the tiny-model C ABI integration contract."""

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
    model = args.work_dir / "tiny-c-api.safetensors"
    subprocess.run(
        [
            sys.executable,
            str(args.generator),
            "--model",
            str(model),
            "--config-only",
        ],
        check=True,
    )
    return subprocess.run([str(args.binary), str(model)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
