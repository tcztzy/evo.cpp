#!/usr/bin/env python3
"""Require libevo's defined dynamic exports to match the public C ABI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "evo_abi_version",
    "evo_backend_name",
    "evo_batch_add_sequence",
    "evo_batch_clear",
    "evo_batch_create",
    "evo_batch_free",
    "evo_batch_sequence_count",
    "evo_batch_token_count",
    "evo_context_capacity",
    "evo_context_create",
    "evo_context_decode",
    "evo_context_default_params",
    "evo_context_embed",
    "evo_context_free",
    "evo_context_position",
    "evo_context_profile",
    "evo_context_prefill",
    "evo_last_error",
    "evo_model_architecture",
    "evo_model_backend",
    "evo_model_default_params",
    "evo_model_decode_token",
    "evo_model_embedding_width",
    "evo_model_encode",
    "evo_model_free",
    "evo_model_id",
    "evo_model_load",
    "evo_model_layer_count",
    "evo_model_max_context",
    "evo_model_profile",
    "evo_model_vocab_size",
    "evo_sampler_create",
    "evo_sampler_default_params",
    "evo_sampler_free",
    "evo_sampler_sample",
    "evo_status_name",
    "evo_version_string",
}


def exported_symbols(library: Path) -> set[str]:
    command = (
        ["nm", "-gU", str(library)]
        if sys.platform == "darwin"
        else ["nm", "-D", "--defined-only", str(library)]
    )
    output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    symbols = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        symbol = line.split()[-1]
        symbols.add(symbol.removeprefix("_"))
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args()
    actual = exported_symbols(args.library)
    missing = sorted(EXPECTED - actual)
    unexpected = sorted(actual - EXPECTED)
    if missing or unexpected:
        if missing:
            print(f"missing C ABI exports: {missing}", file=sys.stderr)
        if unexpected:
            print(f"unexpected dynamic exports: {unexpected}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
