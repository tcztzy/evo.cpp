#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract one BioNeMo predict_evo2 logit matrix as row-major F32 NPY."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="predict_evo2 .pt file or output directory containing exactly one",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--batch-index", type=int, default=0)
    return parser.parse_args()


def resolve_prediction(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("predictions__rank_*__dp_rank_*.pt"))
    if len(candidates) != 1:
        raise ValueError(
            f"{path} contains {len(candidates)} epoch prediction files; expected 1"
        )
    return candidates[0]


def main() -> int:
    args = parse_args()
    try:
        import numpy
        import torch

        prediction_path = resolve_prediction(args.input)
        payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
        required = {"token_logits", "pad_mask", "seq_idx", "tokens"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(
                "BioNeMo prediction is missing token_logits/pad_mask/seq_idx/tokens"
            )
        logits = payload["token_logits"]
        mask = payload["pad_mask"]
        tokens = payload["tokens"]
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[2] != 512
        ):
            raise ValueError(
                f"token_logits must have shape [batch, sequence, 512], got "
                f"{getattr(logits, 'shape', None)}"
            )
        if args.batch_index < 0 or args.batch_index >= logits.shape[0]:
            raise ValueError("--batch-index is outside the prediction batch")
        if (
            not isinstance(mask, torch.Tensor)
            or tuple(mask.shape) != tuple(logits.shape[:2])
            or not isinstance(tokens, torch.Tensor)
            or tuple(tokens.shape) != tuple(logits.shape[:2])
        ):
            raise ValueError("pad_mask/tokens shapes do not match token_logits")
        valid = int(mask[args.batch_index].sum().item())
        if valid <= 0 or valid > logits.shape[1]:
            raise ValueError(f"invalid unpadded sequence length {valid}")
        matrix = (
            logits[args.batch_index, :valid]
            .detach()
            .to(dtype=torch.float32)
            .contiguous()
            .numpy()
        )
        if not numpy.isfinite(matrix).all():
            raise ValueError("BioNeMo logits contain non-finite values")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        numpy.save(args.output, matrix, allow_pickle=False)
        top1 = matrix.argmax(axis=1).astype(int).tolist()
        report = {
            "source": str(prediction_path),
            "shape": list(matrix.shape),
            "batch_index": args.batch_index,
            "sequence_index": int(payload["seq_idx"][args.batch_index].item()),
            "tokens": tokens[args.batch_index, :valid].to(dtype=torch.int64).tolist(),
            "top1": top1,
            "finite": all(math.isfinite(float(value)) for value in matrix.flat),
        }
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report is not None:
            args.report.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"extract_bionemo_logits: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
