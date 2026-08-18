#!/usr/bin/env python3
"""Export pinned upstream hidden states for Omni-DNA-1B divergence audits."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from generate_geneb_omnidna_1b_upstream_oracle import (
    OracleError,
    SEQUENCES,
    canonical_json,
    encode_case,
    package_environment,
    sha256_file,
    validate_checkpoint_header,
    validate_config,
    validate_snapshot,
    validate_source_receipt,
)


EXPECTED_INPUT_IDS = [1, 5, 6, 1049, 0, 5, 6, 1049, 0, 5, 6, 1049, 0, 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise OracleError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_MODULES_CACHE"] = str(output / "hf-modules")
    sys.dont_write_bytecode = True
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    receipt_paths, receipt_sha256 = validate_source_receipt(
        args.source_receipt.expanduser().resolve(strict=True)
    )
    snapshot = args.snapshot.expanduser().resolve(strict=True)
    source_manifest = validate_snapshot(snapshot, receipt_paths)
    packages, package_lock = package_environment()
    tensor_manifest_sha256 = validate_checkpoint_header(
        receipt_paths["model.safetensors"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    validate_config(config)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    state = load_file(str(receipt_paths["model.safetensors"]), device="cpu")
    loaded = model.load_state_dict(state, strict=False)
    if list(loaded.missing_keys) != ["word_embeddings.weight"] or loaded.unexpected_keys:
        raise OracleError(f"checkpoint load contract differs: {loaded!r}")
    del state
    gc.collect()
    model.to(device="cpu", dtype=torch.float32).eval()

    encoded = encode_case(tokenizer, SEQUENCES[0])
    input_ids = encoded["input_ids"].tolist()
    if input_ids != [EXPECTED_INPUT_IDS]:
        raise OracleError(f"input IDs differ: {input_ids!r}")
    with torch.inference_mode():
        result = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    if result.hidden_states is None or len(result.hidden_states) != 17:
        raise OracleError("hidden-state tap count differs")

    layers = []
    for index, tensor in enumerate(result.hidden_states):
        values = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        if values.shape != (1, 14, 2_048) or not np.isfinite(values).all():
            raise OracleError(f"layer {index} hidden-state contract differs")
        path = output / f"layer-{index:02d}.f32.npy"
        np.save(path, values[0], allow_pickle=False)
        layers.append(
            {
                "layer": index,
                "shape": [14, 2_048],
                "npy_sha256": sha256_file(path),
                "raw_f32_sha256": hashlib.sha256(
                    values[0].tobytes(order="C")
                ).hexdigest(),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "geneb-omnidna-1b-upstream-layer-divergence-audit",
        "runtime_id": "geneb-omni-dna-1b",
        "source_receipt_sha256": receipt_sha256,
        "source_manifest": source_manifest,
        "checkpoint_tensor_manifest_sha256": tensor_manifest_sha256,
        "sequence": SEQUENCES[0],
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"].tolist(),
        "layers": layers,
        "environment_lock": {
            "packages": packages,
            "package_lock": package_lock,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "independent_of_evo_native_runtime": True,
    }
    report_path = output / "layer-audit-report.json"
    report_path.write_bytes(canonical_json(report))
    print(
        json.dumps(
            {
                "report_sha256": sha256_file(report_path),
                "layer_count": len(layers),
                "layers": layers,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OracleError, OSError, RuntimeError, ValueError) as error:
        print(
            f"audit_geneb_omnidna_1b_layer_divergence: error: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2)
