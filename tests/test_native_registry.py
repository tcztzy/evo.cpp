#!/usr/bin/env python3
"""Ensure the native runtime and converter consume identical official profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def indices(value: str) -> list[int]:
    return [] if not value else [int(item) for item in value.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    lines = subprocess.run(
        [str(args.binary)], check=True, text=True, capture_output=True
    ).stdout.splitlines()
    native: dict[str, list[str]] = {}
    architectures: dict[str, list[str]] = {}
    for line in lines:
        fields = line.split("|")
        if fields[0] == "@":
            assert len(fields) == 8, line
            architectures[fields[1]] = fields[2:]
            continue
        assert len(fields) == 20, line
        native[fields[0]] = fields[1:]
    assert set(native) == set(registry["models"])

    for model_id, entry in registry["models"].items():
        profile = registry["profiles"][entry["profile"]]
        fields = native[model_id]
        assert int(fields[0]) == 512
        assert int(fields[1]) == profile["hidden_size"]
        assert int(fields[2]) == profile["num_layers"]
        assert int(fields[3]) == profile["num_attention_heads"]
        assert int(fields[4]) == profile["inner_mlp_size"]
        assert int(fields[5]) == profile["hcs_filter_groups"]
        assert int(fields[6]) == profile["hcm_filter_groups"]
        assert int(fields[7]) == profile["hcl_filter_groups"]
        assert int(fields[8]) == profile["max_seqlen"]
        assert float(fields[9]) == profile["rotary_emb_base"]
        assert float(fields[10]) == profile["rotary_emb_scaling_factor"]
        assert bool(int(fields[11])) == profile["use_interpolated_rotary_pos_emb"]
        assert fields[12] == entry["projection_runtime_dtype"]
        assert fields[13] == entry["source_projection_dtype"]
        expected_hcm = "F32" if model_id == "evo2_40b_bionemo_bf16" else "BF16"
        assert fields[14] == expected_hcm
        assert indices(fields[15]) == profile["hcs_layer_idxs"]
        assert indices(fields[16]) == profile["hcm_layer_idxs"]
        assert indices(fields[17]) == profile["hcl_layer_idxs"]
        assert indices(fields[18]) == profile["attn_layer_idxs"]
    assert set(architectures) == {
        "StripedHyena2",
        "StripedHyena2Test",
        "HyenaDNA",
        "HyenaDNATest",
    }
    assert architectures["StripedHyena2"] == [
        "evo2-runtime-v1",
        "evo2-safetensors-v1",
        "byte",
        "3",
        "31",
        "0",
    ]
    assert architectures["HyenaDNA"] == [
        "hyenadna-runtime-v1",
        "hyenadna-safetensors-v1",
        "hyenadna-character",
        "1",
        "31",
        "0",
    ]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
