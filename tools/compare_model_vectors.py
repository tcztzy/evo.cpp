#!/usr/bin/env python3
"""Compare native EVO layer/logit dumps with official Vortex BF16 vectors."""

import argparse
import hashlib
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
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="fail on the first raw element mismatch, regardless of similarity",
    )
    return parser.parse_args()


def array_metrics(reference: np.ndarray, native: np.ndarray) -> dict[str, object]:
    if reference.shape != native.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, native={native.shape}")
    if reference.dtype != native.dtype:
        raise ValueError(f"dtype mismatch: reference={reference.dtype}, native={native.dtype}")
    reference_raw = np.ascontiguousarray(reference).view(np.uint8).reshape(
        reference.size, reference.dtype.itemsize
    )
    native_raw = np.ascontiguousarray(native).view(np.uint8).reshape(
        native.size, native.dtype.itemsize
    )
    unequal = np.any(reference_raw != native_raw, axis=1)
    unequal_indices = np.flatnonzero(unequal)
    raw_exact = unequal_indices.size == 0
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
        "dtype": str(reference.dtype),
        "raw_exact": raw_exact,
        "unequal_elements": int(unequal_indices.size),
        "first_unequal_flat_index": (
            None if raw_exact else int(unequal_indices[0])
        ),
        "reference_payload_sha256": hashlib.sha256(reference_raw).hexdigest(),
        "native_payload_sha256": hashlib.sha256(native_raw).hexdigest(),
        "finite": finite,
        "cosine": cosine,
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }


def compare_directories(
    reference_dir: Path,
    native_dir: Path,
    minimum_cosine: float,
    require_exact: bool = False,
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
        passed = (
            bool(metrics["finite"])
            and float(metrics["cosine"]) >= minimum_cosine
            and (not require_exact or bool(metrics["raw_exact"]))
        )
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
        and (not require_exact or bool(logits_metrics["raw_exact"]))
    )
    logits_metrics.update(
        {
            "reference_last_top1": reference_top1,
            "native_last_top1": native_top1,
            "passed": logits_passed,
        }
    )
    debug_reports: list[dict[str, object]] = []
    first_debug_failure: str | None = None
    debug_layer = manifest.get("debug_layer")
    if isinstance(debug_layer, int) and 0 <= debug_layer < layers:
        prefix = f"layer_{debug_layer}_"
        reference_paths = {
            path.name: path for path in reference_dir.glob(f"{prefix}*.npy")
        }
        native_names = {
            path.name for path in native_dir.glob(f"{prefix}*.npy")
        }
        for filename in sorted(reference_paths.keys() & native_names):
            reference_path = reference_paths[filename]
            native_path = native_dir / reference_path.name
            metrics = array_metrics(
                np.load(reference_path, allow_pickle=False),
                np.load(native_path, allow_pickle=False),
            )
            passed = (
                bool(metrics["finite"])
                and float(metrics["cosine"]) >= minimum_cosine
                and (not require_exact or bool(metrics["raw_exact"]))
            )
            metrics.update({"filename": reference_path.name, "passed": passed})
            debug_reports.append(metrics)
            if not passed and first_debug_failure is None:
                first_debug_failure = reference_path.name
    return {
        "schema": 1,
        "minimum_cosine": minimum_cosine,
        "require_exact": require_exact,
        "reference_manifest": manifest,
        "layers": layer_reports,
        "first_failing_layer": first_failure,
        "debug": debug_reports,
        "first_failing_debug": first_debug_failure,
        "logits": logits_metrics,
        "passed": (
            first_failure is None
            and first_debug_failure is None
            and logits_passed
        ),
    }


def main() -> int:
    args = parse_args()
    try:
        report = compare_directories(
            args.reference_dir,
            args.native_dir,
            args.minimum_cosine,
            args.require_exact,
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
                    "first_failing_debug": report["first_failing_debug"],
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
