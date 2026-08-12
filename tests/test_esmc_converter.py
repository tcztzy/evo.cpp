#!/usr/bin/env python3
"""Exercise ESMC conversion, provenance gates, and atomic failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

from convert_esmc_checkpoint import expected_extra_states, expected_manifest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source(path: Path, topology: dict[str, int], corruption: str = "") -> None:
    tensors = expected_manifest(topology)
    if corruption == "missing":
        tensors.pop("lm_head.3.bias")
    root: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    payload = bytearray()
    for name, shape in tensors.items():
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * 4
        root[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        payload.extend(bytes((index % 251 for index in range(size))))
        offset += size
    for name in sorted(expected_extra_states(topology["num_layers"])):
        root[name] = {
            "dtype": "F32" if corruption == "extra-dtype" else "U8",
            "shape": [0],
            "data_offsets": [offset, offset],
        }
    raw = json.dumps(root, separators=(",", ":")).encode()
    header = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def create_case(root: Path, corruption: str = "") -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    topology = {
        "hidden_size": 2,
        "num_layers": 1,
        "num_attention_heads": 1,
        "inner_mlp_size": 2,
        "vocab_size": 64,
        "max_seqlen": 2048,
    }
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["ESMCForMaskedLM"],
                "d_model": 2,
                "dtype": "float32",
                "mask_token_id": 32,
                "model_type": "esmc",
                "n_heads": 1,
                "n_layers": 1,
                "pad_token_id": 1,
                "tie_word_embeddings": False,
                "vocab_size": 64,
            }
        ),
        encoding="utf-8",
    )
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text('{"fixture":"esmc-protein-v1"}\n', encoding="utf-8")
    weights = root / "model.safetensors"
    write_source(weights, topology, corruption)
    source_files = []
    for path in (config, tokenizer, weights):
        source_files.append(
            {"name": path.name, "size": path.stat().st_size, "sha256": digest(path)}
        )
    revision = "a" * 40
    registry = root / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "esmc_profiles": {"tiny": topology},
                "models": {
                    "esmc_tiny": {
                        "family": "esmc",
                        "architecture": "ESMC",
                        "profile": "tiny",
                        "source_repo": "biohub/ESMC-Tiny-Fixture",
                        "source_revision": revision,
                        "checkpoint_files": source_files,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = root / "source-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "source-checkpoint",
                "model_id": "esmc_tiny",
                "repo": "biohub/ESMC-Tiny-Fixture",
                "requested_revision": revision,
                "resolved_revision": revision,
                "files": [
                    {**entry, "path": str(root / str(entry["name"]))}
                    for entry in source_files
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return registry, receipt


def run_converter(
    converter: Path, registry: Path, receipt: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(converter.parent)
    return subprocess.run(
        [
            sys.executable,
            str(converter),
            "--registry",
            str(registry),
            "--receipt",
            str(receipt),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def runtime_metadata(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(size))
    return header["__metadata__"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--inspector", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    registry, receipt = create_case(args.work_dir / "valid")
    output = args.work_dir / "valid" / "runtime.safetensors"
    output.unlink(missing_ok=True)
    result = run_converter(args.converter, registry, receipt, output)
    if result.returncode != 0:
        raise AssertionError(f"valid conversion failed: {result.stderr}")
    metadata = runtime_metadata(output)
    if any(
        (
            metadata.get("runtime.profile") != "s:esmc-runtime-v1",
            metadata.get("runtime.abi") != "s:esmc-safetensors-v1",
            metadata.get("model.architecture") != "s:ESMC",
            metadata.get("runtime.embedding_layer_count") != "u:2",
        )
    ):
        raise AssertionError("runtime artifact omitted ESMC profile metadata")
    inspected = subprocess.run(
        [str(args.inspector), str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "profile=esmc-runtime-v1" not in inspected or "validation=ok" not in inspected:
        raise AssertionError("native loader rejected the converted ESMC artifact")

    bad_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    bad_receipt["resolved_revision"] = "b" * 40
    receipt.write_text(json.dumps(bad_receipt), encoding="utf-8")
    rejected = run_converter(
        args.converter, registry, receipt, args.work_dir / "bad-receipt.safetensors"
    )
    if rejected.returncode == 0 or "repository/revision" not in rejected.stderr:
        raise AssertionError("converter accepted a mismatched source receipt")

    for corruption in ("missing", "extra-dtype"):
        bad_registry, bad_source_receipt = create_case(
            args.work_dir / corruption, corruption
        )
        bad_output = args.work_dir / corruption / "runtime.safetensors"
        bad_output.unlink(missing_ok=True)
        rejected = run_converter(
            args.converter, bad_registry, bad_source_receipt, bad_output
        )
        if rejected.returncode == 0 or bad_output.exists():
            raise AssertionError(f"converter accepted/published {corruption}")

    print("ESMC converter contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
