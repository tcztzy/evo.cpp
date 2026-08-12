#!/usr/bin/env python3
"""Generate and execute the deterministic native ESMC CPU acceptance gate."""

from __future__ import annotations

import argparse
import os
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
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.generator.parent)
    generated = subprocess.run(
        [sys.executable, str(args.generator), "--output-dir", str(args.work_dir)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    artifact = Path(generated.stdout.splitlines()[0])
    subprocess.run(
        [str(args.binary), str(artifact), str(args.work_dir)],
        check=True,
    )
    print("ESMC backend contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
