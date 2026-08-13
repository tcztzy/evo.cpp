#!/usr/bin/env python3
"""Convert an official Arc Evo 2 checkpoint to runtime-ready Safetensors."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from evo.checkpoint import (
    CheckpointError,
    load_checkpoint,
    prepare_runtime_image_sources,
    prepare_runtime_sources,
)
from evo.format import EVO2_PROFILE_VALUE, FormatError, TensorSource, write_model
from evo.model_config import (
    checkpoint_manifest,
    container_manifest,
    config_metadata,
    ignored_checkpoint_manifest,
    load_config,
    runtime_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream an official Arc Evo 2 .pt checkpoint into the "
            "runtime-ready Evo2 Safetensors profile."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="official (merged, if split) .pt")
    parser.add_argument("--config", required=True, type=Path, help="normalized strict Evo 2 config")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "output model.safetensors base name; large outputs use standard "
            "model-00001-of-000NN.safetensors shards and an index"
        ),
    )
    parser.add_argument("--source-sha256", help="precomputed 64-hex checkpoint SHA256")
    parser.add_argument("--chunk-mib", type=int, default=16, help="streaming chunk size (default: 16)")
    parser.add_argument(
        "--max-shard-mib",
        type=int,
        default=4096,
        help="maximum tensor payload per shard (default: 4096)",
    )
    parser.add_argument("--force", action="store_true", help="replace existing output artifacts")
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
        if args.max_shard_mib <= 0:
            raise CheckpointError("--max-shard-mib must be positive")
        if args.source_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", args.source_sha256):
            raise CheckpointError("--source-sha256 must contain exactly 64 hexadecimal characters")

        config = load_config(args.config)
        manifest = checkpoint_manifest(config)
        ignored_manifest = ignored_checkpoint_manifest(config)
        all_hyena_layers = tuple(
            sorted(
                config.hcs_layer_idxs
                + config.hcm_layer_idxs
                + config.hcl_layer_idxs
            )
        )
        projection_layers = all_hyena_layers if config.use_fp8_input_projections else ()
        print(f"mmap loading {args.input} ...", file=sys.stderr, flush=True)
        sources, skipped, fp8_sources = load_checkpoint(
            args.input,
            manifest,
            expected_extra_states=config.registry["extra_state_count"],
            fp8_projection_layers=projection_layers,
            ignored_manifest=ignored_manifest,
        )
        sources = prepare_runtime_sources(sources, runtime_manifest(config))
        image_sources = prepare_runtime_image_sources(
            sources,
            fp8_sources,
            projection_layers,
        )
        expected_container = container_manifest(config)
        actual_container = [
            (source.name, source.dtype, source.shape, source.nbytes)
            for source in image_sources
        ]
        expected_container_signature = [
            (spec.name, spec.dtype, spec.shape, spec.nbytes)
            for spec in expected_container
        ]
        if actual_container != expected_container_signature:
            raise CheckpointError(
                f"{config.model_id} output tensor table does not match its container manifest"
            )
        print(
            f"validated {len(sources)} checkpoint tensors and extracted "
            f"{len(fp8_sources)} software-FP8 tensors from "
            f"{len(projection_layers)} Hyena projections; "
            f"validated/recomputed {len(ignored_manifest)} time grids and skipped "
            f"{len(skipped) - len(projection_layers)} other documented TE extra-state entries",
            file=sys.stderr,
        )
        if args.dry_run:
            print("checkpoint validation passed", file=sys.stderr)
            return 0

        metadata = config_metadata(config, args.input.name, args.input.stat().st_size)
        if args.source_sha256 is not None:
            metadata["checkpoint.sha256"] = args.source_sha256.lower()
        metadata["hcm_filter_dtype"] = "BF16"
        if projection_layers:
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
        load_path = write_model(
            args.output,
            metadata,
            image_sources,
            artifact_profile=EVO2_PROFILE_VALUE,
            force=args.force,
            chunk_size=args.chunk_mib * 1024 * 1024,
            max_shard_size=args.max_shard_mib * 1024 * 1024,
            progress=progress,
        )
        print(f"wrote {load_path}", file=sys.stderr)
        return 0
    except (CheckpointError, FormatError, OSError, ValueError) as error:
        print(f"convert_checkpoint: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
