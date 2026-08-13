#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the pinned official Biohub ESMC implementation on one GPU."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from generate_esmc_official_oracle import (
    REFERENCE_COMMIT,
    REFERENCE_HASHES,
    OracleError,
    load_official_checkpoint,
    sha256,
    validate_source,
)


SEQUENCE_ALPHABET = "LAGVSERTIDPKQNFYMHWC"


def sequence_for_encoded_length(length: int) -> str:
    residues = length - 2
    return (SEQUENCE_ALPHABET * ((residues + len(SEQUENCE_ALPHABET) - 1) // len(SEQUENCE_ALPHABET)))[:residues]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--length", action="append", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference-commit", default=REFERENCE_COMMIT)
    args = parser.parse_args()

    if args.reference_commit != REFERENCE_COMMIT:
        parser.error(f"--reference-commit must be {REFERENCE_COMMIT}")
    if args.warmups < 1 or args.repeats < 2:
        parser.error("--warmups must be >= 1 and --repeats must be >= 2")
    lengths = list(dict.fromkeys(args.length))
    if any(length < 2 or length > 2048 for length in lengths):
        parser.error("--length must be in [2, 2048]")
    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")

    model_dir = args.model_dir.resolve()
    registry = args.registry.resolve()
    entry, source_files = validate_source(model_dir, registry, args.model_id)

    try:
        import torch
        import transformers
        from safetensors.torch import load_file
        from transformers.models.esmc.configuration_esmc import ESMCConfig
        from transformers.models.esmc import modeling_esmc
        from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
        from transformers.models.esmc.tokenization_esmc import ESMCTokenizer
    except ImportError as error:
        raise OracleError(
            "the isolated benchmark requires PyTorch and the pinned Biohub "
            "Transformers ESMC sources"
        ) from error

    reference_types = (ESMCConfig, ESMCForMaskedLM, ESMCTokenizer)
    reference_paths = [
        Path(inspect.getsourcefile(value) or "").resolve() for value in reference_types
    ]
    for path in reference_paths:
        if not path.is_file() or sha256(path) != REFERENCE_HASHES.get(path.name):
            raise OracleError(f"official reference source is not pinned: {path}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise OracleError("the production benchmark requires one CUDA device")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The pinned source explicitly supports this fallback. It avoids a Triton
    # JIT dependency on an unversioned libcuda symlink in the reference image.
    modeling_esmc._flash_attn_rotary_available = False
    modeling_esmc.apply_triton_rotary = None

    load_started = time.perf_counter()
    config = ESMCConfig.from_pretrained(str(model_dir), local_files_only=True)
    config._attn_implementation = "eager"
    tokenizer = ESMCTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = ESMCForMaskedLM(config)
    load_official_checkpoint(model, model_dir, load_file)
    model.eval().to(args.device)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    cases: dict[str, Any] = {}
    with torch.inference_mode():
        for length in lengths:
            encoded = tokenizer(
                sequence_for_encoded_length(length), return_tensors="pt"
            )
            actual_length = int(encoded["input_ids"].shape[1])
            if actual_length != length:
                raise OracleError(
                    f"official tokenizer produced length {actual_length}, expected {length}"
                )
            inputs = {name: tensor.to(args.device) for name, tensor in encoded.items()}
            samples: list[float] = []
            for iteration in range(args.warmups + args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = model(
                    **inputs,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                    compute_sae=False,
                )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if output.logits.shape != (1, length, int(config.vocab_size)):
                    raise OracleError("official benchmark returned unexpected logits shape")
                if iteration >= args.warmups:
                    samples.append(elapsed)
            median_seconds = statistics.median(samples)
            cases[str(length)] = {
                "samples_seconds": samples,
                "median_seconds": median_seconds,
                "median_tokens_per_second": length / median_seconds,
            }

    device = torch.cuda.get_device_properties(torch.device(args.device))
    report = {
        "schema_version": 1,
        "runtime": "official_biohub_transformers",
        "model_id": args.model_id,
        "source_repo": entry.get("source_repo"),
        "source_revision": entry.get("source_revision"),
        "source_files": source_files,
        "reference_commit": REFERENCE_COMMIT,
        "dtype": "float32",
        "batch_size": 1,
        "tf32": False,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "load_seconds": load_seconds,
        "cases": cases,
        "kernels": {
            "transformer_engine": bool(modeling_esmc._te_available),
            "xformers": bool(modeling_esmc._xformers_available),
            "flash_attention_2": bool(modeling_esmc._flash_attn_available),
            "rotary": "official_pytorch_fallback",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": device.name,
            "gpu_total_memory": device.total_memory,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
