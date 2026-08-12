#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate native ESMC logits and final hidden state against official NPY files."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path

REFERENCE_COMMIT = "3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_npy_f32(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    payload = path.read_bytes()
    if payload[:8] != b"\x93NUMPY\x01\x00" or len(payload) < 10:
        raise ValueError(f"{path} is not a NumPy v1 file")
    header_size = struct.unpack_from("<H", payload, 8)[0]
    payload_offset = 10 + header_size
    if payload_offset > len(payload):
        raise ValueError(f"{path} has a truncated NumPy header")
    header = ast.literal_eval(payload[10:payload_offset].decode("ascii"))
    shape = tuple(int(value) for value in header["shape"])
    if (
        header["descr"] != "<f4"
        or header["fortran_order"]
        or not shape
        or any(value <= 0 for value in shape)
    ):
        raise ValueError(f"{path} is not a nonempty row-major little-endian F32 array")
    count = math.prod(shape)
    if len(payload) - payload_offset != count * 4:
        raise ValueError(f"{path} payload size does not match shape {shape}")
    return shape, struct.unpack_from(f"<{count}f", payload, payload_offset)


def metrics(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    reference_shape, reference = load_npy_f32(reference_path)
    candidate_shape, candidate = load_npy_f32(candidate_path)
    if reference_shape != candidate_shape:
        raise ValueError(
            f"tensor shapes differ: {reference_shape} vs {candidate_shape}"
        )
    differences = tuple(
        abs(left - right)
        for left, right in zip(reference, candidate, strict=True)
    )
    dot = math.fsum(
        left * right for left, right in zip(reference, candidate, strict=True)
    )
    reference_norm = math.sqrt(math.fsum(value * value for value in reference))
    candidate_norm = math.sqrt(math.fsum(value * value for value in candidate))
    cosine = (
        dot / (reference_norm * candidate_norm)
        if reference_norm > 0.0 and candidate_norm > 0.0
        else 1.0 if reference == candidate else 0.0
    )
    return {
        "reference": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "shape": list(reference_shape),
        "finite": all(math.isfinite(value) for value in (*reference, *candidate)),
        "max_abs": max(differences),
        "mean_abs": math.fsum(differences) / len(differences),
        "cosine": cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--native-logits", required=True, type=Path)
    parser.add_argument("--native-hidden", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-abs", default=5e-3, type=float)
    parser.add_argument("--max-mean-abs", default=5e-4, type=float)
    parser.add_argument("--min-cosine", default=0.99999, type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_path = args.oracle_dir / "oracle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != args.model_id:
        parser.error("oracle model_id differs from --model-id")
    reference = manifest.get("official_reference")
    if not isinstance(reference, dict) or reference.get("commit") != REFERENCE_COMMIT:
        parser.error("oracle does not identify the pinned Biohub reference commit")
    if (
        manifest.get("attention_implementation") != "official_manual_f32"
        or manifest.get("rotary_implementation") != "official_pytorch_fallback"
        or manifest.get("layer_norm_linear_implementation")
        != "official_pytorch_fallback"
        or manifest.get("dtype") != "float32"
        or manifest.get("tf32") is not False
    ):
        parser.error("oracle implementation metadata differs from the portable gate")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        parser.error("oracle manifest has no output descriptors")
    for output_name, filename in (
        ("logits", "logits.npy"),
        ("final_hidden", "final-hidden.npy"),
    ):
        descriptor = outputs.get(output_name)
        output_path = args.oracle_dir / filename
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("path") != filename
            or descriptor.get("sha256") != sha256(output_path)
        ):
            parser.error(f"oracle {output_name} descriptor failed its SHA256 gate")
    results = {
        "logits": metrics(args.oracle_dir / "logits.npy", args.native_logits),
        "final_hidden": metrics(
            args.oracle_dir / "final-hidden.npy", args.native_hidden
        ),
    }
    passed = all(
        bool(result["finite"])
        and float(result["max_abs"]) <= args.max_abs
        and float(result["mean_abs"]) <= args.max_mean_abs
        and float(result["cosine"]) >= args.min_cosine
        for result in results.values()
    )
    report = {
        "schema_version": 1,
        "model_id": args.model_id,
        "oracle_manifest": str(manifest_path),
        "oracle_manifest_sha256": sha256(manifest_path),
        "thresholds": {
            "max_abs": args.max_abs,
            "max_mean_abs": args.max_mean_abs,
            "min_cosine": args.min_cosine,
        },
        "results": results,
        "passed": passed,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
