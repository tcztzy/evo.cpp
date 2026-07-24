#!/usr/bin/env python3
"""Convert the official Evo 2 40B PyTorch checkpoint to EVO2C v1."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from evo2c.checkpoint import CheckpointError, load_checkpoint
from evo2c.format import FormatError, TensorSource, write_model
from evo2c.model_config import checkpoint_manifest, config_metadata, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream an official Evo 2 40B .pt checkpoint into an mmap-friendly EVO2C file."
    )
    parser.add_argument("--input", required=True, type=Path, help="merged evo2_40b.pt")
    parser.add_argument("--config", required=True, type=Path, help="official Evo 2 40B YAML config")
    parser.add_argument("--output", required=True, type=Path, help="output .evo2 file")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--source-sha256", help="precomputed 64-hex checkpoint SHA256")
    parser.add_argument("--chunk-mib", type=int, default=16, help="streaming chunk size (default: 16)")
    parser.add_argument("--force", action="store_true", help="atomically replace existing output")
    parser.add_argument("--dry-run", action="store_true", help="validate checkpoint without writing output")
    return parser.parse_args()


def progress(index: int, total: int, tensor: TensorSource) -> None:
    gib = tensor.nbytes / (1024**3)
    print(
        f"[{index:03d}/{total}] {tensor.name} {tensor.dtype}{tensor.shape} {gib:.3f} GiB",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    args = parse_args()
    try:
        if not args.input.is_file():
            raise CheckpointError(f"input checkpoint not found: {args.input}")
        if not args.config.is_file():
            raise CheckpointError(f"config not found: {args.config}")
        if args.chunk_mib <= 0:
            raise CheckpointError("--chunk-mib must be positive")
        if args.source_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", args.source_sha256):
            raise CheckpointError("--source-sha256 must contain exactly 64 hexadecimal characters")

        config = load_config(args.config)
        if not config.use_fp8_input_projections:
            raise CheckpointError(
                "official ARC conversion requires use_fp8_input_projections=true; "
                "use convert_bionemo_checkpoint.py for BF16 projections"
            )
        manifest = checkpoint_manifest(config)
        projection_layers = tuple(
            sorted(
                config.hcs_layer_idxs
                + config.hcm_layer_idxs
                + config.hcl_layer_idxs
            )
        )
        print(f"mmap loading {args.input} ...", file=sys.stderr, flush=True)
        sources, skipped, fp8_sources = load_checkpoint(
            args.input,
            manifest,
            fp8_projection_layers=projection_layers,
        )
        print(
            f"validated {len(sources)} checkpoint tensors and extracted "
            f"{len(fp8_sources)} software-FP8 tensors from "
            f"{len(projection_layers)} Hyena projections; "
            f"skipping {len(skipped) - len(projection_layers)} other documented "
            "TE extra-state entries",
            file=sys.stderr,
        )
        if args.dry_run:
            print("checkpoint validation passed", file=sys.stderr)
            return 0

        metadata = config_metadata(config, args.input.name, args.input.stat().st_size)
        if args.source_sha256 is not None:
            metadata["checkpoint.sha256"] = args.source_sha256.lower()
        metadata.update(
            {
                "fp8.software_projection_count": len(projection_layers),
                "fp8.forward_format": "E4M3FN",
                "fp8.backward_format": "E5M2",
                "fp8.amax_history_length": 16,
                "fp8.max_forward": 448.0,
                "fp8.inference_scale_update": False,
                "fp8.reference": "TransformerEngine-2.3-HYBRID",
            }
        )
        write_model(
            args.output,
            metadata,
            sources + fp8_sources,
            force=args.force,
            chunk_size=args.chunk_mib * 1024 * 1024,
            progress=progress,
        )
        print(f"wrote {args.output}", file=sys.stderr)
        return 0
    except (CheckpointError, FormatError, OSError, ValueError) as error:
        print(f"convert_checkpoint: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
