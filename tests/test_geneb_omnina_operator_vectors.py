#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the native OmniNA block-0 audit with pinned upstream operators."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-binary", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    report = json.loads(
        (args.oracle_dir / "oracle-report.json").read_text(encoding="ascii")
    )
    block0 = report.get("block0_operator_vectors")
    if (
        not isinstance(block0, dict)
        or block0.get("input_index") != 1
        or block0.get("native_runtime_used") is not False
        or not isinstance(block0.get("operators"), list)
        or len(block0["operators"]) != 15
    ):
        raise AssertionError("upstream OmniNA block0 report contract differs")
    expected = {item["name"]: item for item in block0["operators"]}
    if len(expected) != 15:
        raise AssertionError("upstream OmniNA block0 operator names repeat")

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)
    result = subprocess.run(
        [str(args.audit_binary), str(args.artifact), str(args.work_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 77:
        print("SKIP: OmniNA exact operator audit requires Apple arm64")
        return 77
    if result.returncode != 0:
        raise AssertionError("native OmniNA operator audit failed: " + result.stderr)

    actual_names = {path.stem for path in args.work_dir.glob("*.f32")}
    if actual_names != set(expected):
        raise AssertionError("native OmniNA block0 operator set differs")
    for name, item in expected.items():
        native = (args.work_dir / (name + ".f32")).read_bytes()
        upstream = np.load(
            args.oracle_dir / ("input-1.block0-%s.f32.npy" % name),
            allow_pickle=False,
        )
        if upstream.dtype != np.dtype("<f4") or list(upstream.shape) != item["shape"]:
            raise AssertionError("upstream OmniNA operator array differs: " + name)
        upstream_raw = upstream.tobytes(order="C")
        if (
            sha256_bytes(upstream_raw) != item["raw_f32_sha256"]
            or native != upstream_raw
        ):
            raise AssertionError("native OmniNA operator differs: " + name)
    print("GENEB OmniNA 15 block0 operator vectors are bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
