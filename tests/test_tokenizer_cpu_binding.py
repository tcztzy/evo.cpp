#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove that CPU Model::encode resolves the tokenizer asset descriptor."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--c-api-binary", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    tokenizer = args.work_dir / "tiny-tokenizer.evo.json"
    asset = {
        "format": "evo-tokenizer-v1",
        "kind": "character",
        "normalization": [],
        "pre_tokenizer": {"kind": "none"},
        "model": {
            "unknown_policy": "unk",
            "match_special_literals": False,
        },
        "post_processor": {
            "prefix_ids": [],
            "suffix_ids": [],
            "padding": {"side": "none", "pad_id": None},
        },
        "special_tokens": {
            "unk": 402,
            "pad": None,
            "bos": None,
            "eos": None,
            "cls": None,
            "sep": None,
            "mask": None,
        },
        "vocab": [
            {"id": 400, "piece": "A"},
            {"id": 401, "piece": "C"},
            {"id": 402, "piece": "[UNK]"},
            {"id": 511, "piece": "[RESERVED]"},
        ],
    }
    tokenizer.write_text(
        json.dumps(asset, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    model = args.work_dir / "tiny-tokenizer-model.safetensors"
    subprocess.run(
        [
            sys.executable,
            str(args.generator),
            "--model",
            str(model),
            "--config-only",
            "--tokenizer-asset",
            str(tokenizer),
        ],
        check=True,
    )
    subprocess.run(
        [str(args.binary), "--tokenizer-vector", str(model)],
        check=True,
    )
    subprocess.run(
        [str(args.c_api_binary), "--tokenizer-vector", str(model)],
        check=True,
    )
    print("CPU tokenizer asset binding contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
