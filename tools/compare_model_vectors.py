#!/usr/bin/env python3
"""Compare native EVO2C layer/logit dumps with official Vortex BF16 vectors."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    return parser.parse_args()


def array_metrics(reference: np.ndarray, native: np.ndarray) -> dict[str, object]:
    if reference.shape != native.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, native={native.shape}")
    reference_f64 = reference.astype(np.float64, copy=False).reshape(-1)
    native_f64 = native.astype(np.float64, copy=False).reshape(-1)
    finite = bool(np.isfinite(reference_f64).all() and np.isfinite(native_f64).all())
    difference = np.abs(reference_f64 - native_f64)
    reference_norm = float(np.linalg.norm(reference_f64))
    native_norm = float(np.linalg.norm(native_f64))
    if reference_norm == 0.0 or native_norm == 0.0:
        cosine = 1.0 if np.array_equal(reference_f64, native_f64) else 0.0
    else:
        cosine = float(np.dot(reference_f64, native_f64) / (reference_norm * native_norm))
    return {
        "shape": list(reference.shape),
        "finite": finite,
        "cosine": cosine,
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }


def compare_directories(
    reference_dir: Path,
    native_dir: Path,
    minimum_cosine: float,
) -> dict[str, object]:
    manifest_path = reference_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = manifest.get("layers")
    if not isinstance(layers, int) or layers <= 0:
        raise ValueError("reference manifest has an invalid layer count")
    if not 0.0 < minimum_cosine <= 1.0:
        raise ValueError("minimum cosine must be in (0, 1]")

    layer_reports: list[dict[str, object]] = []
    first_failure: int | None = None
    for layer in range(layers):
        filename = f"layer_{layer:02d}.npy"
        metrics = array_metrics(
            np.load(reference_dir / filename, allow_pickle=False),
            np.load(native_dir / filename, allow_pickle=False),
        )
        passed = bool(metrics["finite"]) and float(metrics["cosine"]) >= minimum_cosine
        metrics.update({"layer": layer, "passed": passed})
        layer_reports.append(metrics)
        if not passed and first_failure is None:
            first_failure = layer

    reference_logits = np.load(reference_dir / "logits.npy", allow_pickle=False)
    native_logits = np.load(native_dir / "logits.npy", allow_pickle=False)
    logits_metrics = array_metrics(reference_logits, native_logits)
    if reference_logits.ndim != 2 or reference_logits.shape[0] == 0:
        raise ValueError("logits must have shape [tokens, vocabulary]")
    reference_top1 = int(np.argmax(reference_logits[-1]))
    native_top1 = int(np.argmax(native_logits[-1]))
    logits_passed = (
        bool(logits_metrics["finite"])
        and float(logits_metrics["cosine"]) >= minimum_cosine
        and reference_top1 == native_top1
    )
    logits_metrics.update(
        {
            "reference_last_top1": reference_top1,
            "native_last_top1": native_top1,
            "passed": logits_passed,
        }
    )
    return {
        "schema": 1,
        "minimum_cosine": minimum_cosine,
        "reference_manifest": manifest,
        "layers": layer_reports,
        "first_failing_layer": first_failure,
        "logits": logits_metrics,
        "passed": first_failure is None and logits_passed,
    }


def main() -> int:
    args = parse_args()
    try:
        report = compare_directories(
            args.reference_dir,
            args.native_dir,
            args.minimum_cosine,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report_partial = args.report.with_name(f".{args.report.name}.partial")
        report_partial.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_partial.replace(args.report)
        print(
            json.dumps(
                {
                    "passed": report["passed"],
                    "first_failing_layer": report["first_failing_layer"],
                    "logits": report["logits"],
                },
                sort_keys=True,
            )
        )
        return 0 if bool(report["passed"]) else 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"compare_model_vectors: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
