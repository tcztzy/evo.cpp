#!/usr/bin/env python3
"""Closed-contract tests for the pinned DeepGene RoFormer converter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def receipt_file(name: str, path: Path) -> Dict[str, Any]:
    payload = path.read_bytes()
    return {
        "name": name,
        "size": len(payload),
        "sha256": digest(payload),
        "path": str(path.resolve()),
    }


def expect_failure(action: Any, label: str) -> None:
    from evo.geneb_t36_artifact import ConversionError

    try:
        action()
    except ConversionError:
        return
    raise AssertionError(label + " was accepted")


def test_receipt_extra_gate(directory: Path) -> None:
    from evo.geneb_t36_artifact import validate_receipt

    config = directory / "config.json"
    weights = directory / "pytorch_model.bin"
    extra = directory / "LICENSE"
    config.write_bytes(b"config")
    weights.write_bytes(b"weights")
    extra.write_bytes(b"same-revision license")
    source_url = "https://example.invalid/pinned-drive-folder"
    profile = {
        "runtime_id": "test-roformer",
        "source_url": source_url,
        "source_files": [
            {key: value for key, value in receipt_file("config.json", config).items() if key != "path"},
            {key: value for key, value in receipt_file("pytorch_model.bin", weights).items() if key != "path"},
        ],
    }
    receipt = {
        "schema_version": 1,
        "kind": "source-checkpoint",
        "model_id": profile["runtime_id"],
        "source_kind": "google-drive",
        "source_url": source_url,
        "files": [
            receipt_file("config.json", config),
            receipt_file("pytorch_model.bin", weights),
            receipt_file("LICENSE", extra),
        ],
    }
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    selected, _ = validate_receipt(receipt_path, profile, "roformer")
    if set(selected) != {"config.json", "pytorch_model.bin"}:
        raise AssertionError("RoFormer receipt extra influenced tensor selection")
    extra.write_bytes(b"corrupted extra")
    expect_failure(
        lambda: validate_receipt(receipt_path, profile, "roformer"),
        "corrupted unrelated RoFormer receipt file",
    )


def test_tokenizer_descriptor_gate(directory: Path) -> None:
    from evo.geneb_t36_artifact import validate_tokenizer_descriptor

    asset = directory / "tokenizer.evo.json"
    asset.write_bytes(b"compiled-tokenizer")
    asset_digest = digest(asset.read_bytes())
    compiler_digest = "a" * 64
    profile = {
        "tokenizer": {
            "compiler_manifest_sha256": compiler_digest,
            "compiled_asset_sha256": asset_digest,
            "compiled_asset_size": asset.stat().st_size,
        }
    }
    descriptor = {
        "converter.schema": "evo-tokenizer-conversion-receipt",
        "converter.version": 1,
        "compiler_manifest_sha256": compiler_digest,
        "source_receipt_contract_sha256": "b" * 64,
        "tokenizer.profile": "evo-tokenizer-v1",
        "tokenizer.path": "tokenizer.evo.json",
        "tokenizer.sha256": asset_digest,
        "tokenizer.size": asset.stat().st_size,
    }
    descriptor_path = directory / "tokenizer.descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    validate_tokenizer_descriptor(descriptor_path, directory, directory, profile)
    asset.write_bytes(b"corrupted-tokenizer")
    expect_failure(
        lambda: validate_tokenizer_descriptor(
            descriptor_path, directory, directory, profile
        ),
        "corrupted compiled tokenizer",
    )


def test_mmap_filename_boundary(directory: Path) -> None:
    import evo.geneb_t36_artifact as artifact

    observed = []

    class FakeTorch:
        @staticmethod
        def load(filename: object, **kwargs: object) -> Dict[str, object]:
            observed.append(filename)
            if kwargs != {
                "map_location": "cpu",
                "mmap": True,
                "weights_only": True,
            }:
                raise AssertionError("safe torch.load arguments differ")
            return {}

    original = artifact.importlib.import_module
    artifact.importlib.import_module = lambda name: FakeTorch
    try:
        expect_failure(
            lambda: artifact.load_torch_state(directory / "checkpoint.bin"),
            "empty checkpoint",
        )
    finally:
        artifact.importlib.import_module = original
    if len(observed) != 1 or not isinstance(observed[0], str):
        raise AssertionError("mmap checkpoint filename was not normalized to str")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--tokenizer-converter", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.converter.resolve().parent))
    from evo.geneb_t36_artifact import (
        ConversionError,
        load_profiles,
        roformer_runtime_specs,
        roformer_source_specs,
    )

    for converter in (args.converter, args.tokenizer_converter):
        completed = subprocess.run(
            [sys.executable, str(converter), "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(converter.name + " --help failed")
    profiles, root, _ = load_profiles(
        args.profiles.resolve(), "geneb-roformer-converter-v1"
    )
    if set(profiles) != {"geneb-deepgene"} or root[
        "implementation_contract"
    ]["source"] != "wds-seu/DeepGene@486343e5212361d6cd7ed03c624f430ed3d5f02e":
        raise AssertionError("DeepGene profile/code pin differs")
    source, aliases = roformer_source_specs()
    runtime = roformer_runtime_specs()
    if (
        len(source) != 203
        or any(item.dtype != "F32" for item in source)
        or len(runtime) != 196
        or sum(item.nbytes for item in runtime) != 352813056
        or aliases
        != {
            "cls.predictions.decoder.weight": "roformer.embeddings.word_embeddings.weight",
            "cls.predictions.decoder.bias": "cls.predictions.bias",
        }
        or max(len(item.name.encode("utf-8")) for item in runtime) >= 96
    ):
        raise AssertionError("DeepGene source/canonical tensor contract differs")
    manifest = json.loads(args.profiles.read_text(encoding="utf-8"))
    drift = copy.deepcopy(manifest)
    drift["models"][0]["source_files"][1]["sha256"] = "0" * 64
    with tempfile.TemporaryDirectory(prefix="geneb-roformer-profile-") as raw:
        directory = Path(raw)
        drift_path = directory / "drift.json"
        drift_path.write_text(json.dumps(drift), encoding="utf-8")
        try:
            load_profiles(drift_path, "geneb-roformer-converter-v1")
        except ConversionError:
            pass
        else:
            raise AssertionError("unpinned RoFormer profile was accepted")
        test_receipt_extra_gate(directory)
        test_tokenizer_descriptor_gate(directory)
        test_mmap_filename_boundary(directory)
    print("GENEB RoFormer converter contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
