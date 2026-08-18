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
    native_esmc: dict[str, list[str]] = {}
    architectures: dict[str, list[str]] = {}
    factory_backends: dict[str, int] = {}
    for line in lines:
        fields = line.split("|")
        if fields[0] == "@":
            assert len(fields) == 9, line
            architectures[fields[1]] = fields[2:]
            continue
        if fields[0] == "&":
            assert len(fields) == 3, line
            factory_backends[fields[1]] = int(fields[2])
            continue
        if fields[0] == "$":
            assert len(fields) == 13, line
            native_esmc[fields[1]] = fields[2:]
            continue
        assert len(fields) == 22, line
        native[fields[0]] = fields[1:]
    evo_models = {
        model_id: entry
        for model_id, entry in registry["models"].items()
        if entry["family"] == "evo2"
    }
    assert set(native) == set(evo_models)

    for model_id, entry in evo_models.items():
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
        assert fields[19] == entry["exact_support"]
        assert fields[20] == (entry["exact_evidence"] or "")
    esmc_models = {
        model_id: entry
        for model_id, entry in registry["models"].items()
        if entry.get("family") == "esmc"
    }
    assert set(native_esmc) == set(esmc_models)
    for model_id, entry in esmc_models.items():
        profile = registry["esmc_profiles"][entry["profile"]]
        fields = native_esmc[model_id]
        assert fields[:3] == [
            entry["source_repo"],
            entry["source_revision"],
            entry["hosted_alias"],
        ]
        assert [int(value) for value in fields[3:9]] == [
            profile["vocab_size"],
            profile["hidden_size"],
            profile["num_layers"],
            profile["num_attention_heads"],
            profile["inner_mlp_size"],
            profile["max_seqlen"],
        ]
        assert fields[9:] == [
            entry["exact_support"],
            entry["exact_evidence"],
        ]
    registered_architecture_ids = {
        entry["id"] for entry in registry["runtime_architectures"]
    }
    fixture_architecture_ids = {
        "StripedHyena2Test",
        "HyenaDNATest",
        "ESMCTest",
    }
    assert set(architectures) == registered_architecture_ids | fixture_architecture_ids
    production_architectures = {
        entry["id"]: [entry["artifact_profile"], entry["runtime_abi"]]
        for entry in registry["runtime_architectures"]
    }
    production_descriptors = {
        entry["id"]: entry for entry in registry["runtime_architectures"]
    }
    artifact_profiles = {
        entry["id"]: [entry["runtime_abi"], entry["metadata_key"]]
        for entry in registry["artifact_profiles"]
    }
    assert len(artifact_profiles) == len(registry["artifact_profiles"])
    assert set(artifact_profiles) == {
        contract[0] for contract in production_architectures.values()
    }
    assert {contract[1] for contract in artifact_profiles.values()} == {
        "evo2.profile",
        "runtime.profile",
    }
    for profile, runtime_abi in production_architectures.values():
        assert artifact_profiles[profile][0] == runtime_abi
    assert set(production_architectures) == {
        name for name in architectures if not name.endswith("Test")
    }
    assert set(factory_backends) == set(architectures)
    backend_bits = {"cpu": 1, "cuda": 2, "mps": 4}
    capability_bits = {
        "generate": 1,
        "score": 2,
        "embed": 4,
        "variant": 8,
        "serve": 16,
        "logits": 32,
    }
    for name, contract in production_architectures.items():
        assert architectures[name][:2] == contract
        descriptor = production_descriptors[name]
        backends = descriptor["backends"]
        capabilities = descriptor["capabilities"]
        assert isinstance(backends, list) and len(backends) == len(set(backends))
        assert isinstance(capabilities, list) and len(capabilities) == len(
            set(capabilities)
        )
        declared_backends = int(architectures[name][4])
        declared_capabilities = int(architectures[name][5])
        assert sum(backend_bits[value] for value in backends) == declared_backends
        assert (
            sum(capability_bits[value] for value in capabilities)
            == declared_capabilities
        )
        assert factory_backends[name] == declared_backends
        if contract[0].startswith("geneb-"):
            assert backends == ["cpu"]
            assert capabilities == ["embed", "serve"]
    assert architectures["StripedHyena2"] == [
        "evo2-runtime-v1",
        "evo2-safetensors-v1",
        "striped-hyena-2",
        "byte",
        "7",
        "63",
        "0",
    ]
    assert architectures["HyenaDNA"] == [
        "hyenadna-runtime-v1",
        "hyenadna-safetensors-v1",
        "hyenadna",
        "hyenadna-character",
        "5",
        "63",
        "0",
    ]
    assert architectures["ESMC"] == [
        "esmc-runtime-v1",
        "esmc-safetensors-v1",
        "esmc",
        "esmc-protein",
        "7",
        "36",
        "0",
    ]
    assert architectures["GenebTransformerDecoder"] == [
        "geneb-decoder-runtime-v1",
        "geneb-decoder-safetensors-v1",
        "geneb-transformer-decoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebOlmoDecoder"] == [
        "geneb-olmo-runtime-v1",
        "geneb-olmo-safetensors-v1",
        "geneb-olmo-decoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebEsmEncoder"] == [
        "geneb-esm-runtime-v1",
        "geneb-esm-safetensors-v1",
        "geneb-esm-encoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebBertEncoder"] == [
        "geneb-bert-runtime-v1",
        "geneb-bert-safetensors-v1",
        "geneb-bert-encoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebGpt2Decoder"] == [
        "geneb-gpt2-runtime-v1",
        "geneb-gpt2-safetensors-v1",
        "geneb-gpt2-decoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebDnaGptDecoder"] == [
        "geneb-dna-gpt-runtime-v1",
        "geneb-dna-gpt-torch-pth-v1",
        "geneb-dna-gpt-decoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebCustomEncoder"] == [
        "geneb-custom-encoder-runtime-v1",
        "geneb-custom-encoder-safetensors-v1",
        "geneb-custom-encoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebMambaEncoder"] == [
        "geneb-mamba-runtime-v1",
        "geneb-mamba-safetensors-v1",
        "geneb-mamba-encoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebHyenaDnaDecoder"] == [
        "geneb-hyenadna-runtime-v1",
        "geneb-hyenadna-safetensors-v1",
        "geneb-hyenadna-decoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebStripedHyenaV1"] == [
        "geneb-evo1-runtime-v1",
        "geneb-evo1-safetensors-v1",
        "geneb-striped-hyena-v1",
        "artifact",
        "1",
        "20",
        "0",
    ]
    assert architectures["GenebJanusDnaEncoder"] == [
        "geneb-janusdna-runtime-v1",
        "geneb-janusdna-lightning-v1",
        "geneb-janusdna-encoder",
        "artifact",
        "1",
        "20",
        "0",
    ]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
