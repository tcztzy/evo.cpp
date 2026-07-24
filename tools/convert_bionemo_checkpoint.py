#!/usr/bin/env python3
"""Stream a BioNeMo Evo 2 40B BF16 DCP checkpoint into EVO2C v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evo2c.bionemo_checkpoint import (
    dcp_payload_size,
    dcp_storage_size,
    load_bionemo_checkpoint,
    normalized_dcp_kind,
    validate_source_sha256,
)
from evo2c.checkpoint import CheckpointError
from evo2c.format import FormatError, TensorSource, write_model
from evo2c.model_config import config_metadata, load_config


NGC_RESOURCE = "evo2/40b-1m-fp8-bf16:1.0"
NGC_MODEL = "nvidia/clara/evo2-40b-1m-fp8-bf16-nemo2:1.0"
MAPPING_REFERENCE = "NVIDIA-BioNeMo/bionemo-framework@b35c2556209282bd9389fba24f5931f6701e50c5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a BioNeMo NeMo2/MBridge Evo 2 40B BF16 distributed "
            "checkpoint into an mmap-friendly EVO2C file."
        )
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="NeMo2/MBridge DCP directory"
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Evo 2 40B BF16 YAML config"
    )
    parser.add_argument("--output", required=True, type=Path, help="output .evo2 file")
    parser.add_argument(
        "--source-sha256", help="64-hex NGC archive or checkpoint provenance SHA256"
    )
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate DCP metadata and mapping without reading tensor payloads",
    )
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
        if not args.config.is_file():
            raise CheckpointError(f"config not found: {args.config}")
        if args.chunk_mib <= 0:
            raise CheckpointError("--chunk-mib must be positive")
        source_sha256 = validate_source_sha256(args.source_sha256)
        config = load_config(args.config)
        if config.use_fp8_input_projections:
            raise CheckpointError(
                "BioNeMo BF16 conversion requires "
                "use_fp8_input_projections=false"
            )

        sources, reader = load_bionemo_checkpoint(args.input, config)
        checkpoint_kind = normalized_dcp_kind(reader.tensor_metadata)
        payload_size = dcp_payload_size(reader.tensor_metadata)
        print(
            f"validated {len(reader.tensor_metadata)} {checkpoint_kind} DCP tensors "
            f"({payload_size / (1024**3):.3f} GiB logical), mapped to "
            f"{len(sources)} EVO2C tensors; skipping "
            f"{len(reader.bytes_entries)} DCP byte entries",
            file=sys.stderr,
        )
        if args.dry_run:
            print("BioNeMo checkpoint validation passed", file=sys.stderr)
            return 0

        storage_size = dcp_storage_size(reader.directory)
        metadata = config_metadata(config, reader.directory.name, storage_size)
        metadata.update(
            {
                "format.producer": "evo2c-bionemo",
                "model.source_repo": f"ngc://{NGC_MODEL}",
                "checkpoint.format": "torch_dist",
                "checkpoint.kind": checkpoint_kind,
                "checkpoint.size": storage_size,
                "checkpoint.logical_tensor_bytes": payload_size,
                "checkpoint.tensor_count": len(reader.tensor_metadata),
                "checkpoint.extra_state_count": len(reader.bytes_entries),
                "config.use_fp8_input_projections": False,
                "hyena_projection_dtype": "BF16",
                "bionemo.resource": NGC_RESOURCE,
                "bionemo.mapping_reference": MAPPING_REFERENCE,
            }
        )
        if source_sha256 is not None:
            metadata["checkpoint.sha256"] = source_sha256
        write_model(
            args.output,
            metadata,
            sources,
            force=args.force,
            chunk_size=args.chunk_mib * 1024 * 1024,
            progress=progress,
        )
        print(f"wrote {args.output}", file=sys.stderr)
        return 0
    except (CheckpointError, FormatError, OSError, ValueError) as error:
        print(f"convert_bionemo_checkpoint: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
