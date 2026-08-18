#!/usr/bin/env python3
"""Closed-contract tests for the pinned Enformer/SPACE converter."""

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
    extra = directory / "README.md"
    config.write_bytes(b"config")
    weights.write_bytes(b"weights")
    extra.write_bytes(b"same-revision documentation")
    revision = "1" * 40
    profile = {
        "runtime_id": "test-sequence-cnn",
        "repo": "owner/repo",
        "requested_revision": "main",
        "revision": revision,
        "source_files": [
            {key: value for key, value in receipt_file("config.json", config).items() if key != "path"},
            {key: value for key, value in receipt_file("pytorch_model.bin", weights).items() if key != "path"},
        ],
    }
    receipt = {
        "schema_version": 1,
        "kind": "source-checkpoint",
        "model_id": profile["runtime_id"],
        "repo": profile["repo"],
        "requested_revision": "main",
        "resolved_revision": revision,
        "source_kind": "huggingface",
        "files": [
            receipt_file("config.json", config),
            receipt_file("pytorch_model.bin", weights),
            receipt_file("README.md", extra),
        ],
    }
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    selected, _ = validate_receipt(receipt_path, profile, "sequence-cnn")
    if set(selected) != {"config.json", "pytorch_model.bin"}:
        raise AssertionError("receipt extra influenced the critical subset")
    extra.write_bytes(b"corrupted extra")
    expect_failure(
        lambda: validate_receipt(receipt_path, profile, "sequence-cnn"),
        "corrupted unrelated receipt file",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.converter.resolve().parent))
    from evo.geneb_t36_artifact import (
        ConversionError,
        enformer_primary_specs,
        enformer_source_specs,
        load_profiles,
        roformer_runtime_specs,
        space_drop_specs,
        space_primary_specs,
    )

    completed = subprocess.run(
        [sys.executable, str(args.converter), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError("sequence-CNN converter --help failed")
    profiles, root, _ = load_profiles(
        args.profiles.resolve(), "geneb-sequence-cnn-converter-v1"
    )
    if set(profiles) != {"geneb-enformer", "geneb-space"}:
        raise AssertionError("sequence-CNN profile identities differ")
    if root["implementation_contracts"]["enformer"]["source"] != (
        "enformer-pytorch v0.8.8@9ffeb8b62927d752b4983ef308a28bf70b34b160"
    ):
        raise AssertionError("normalized Enformer code pin differs")
    enformer_source, aliases = enformer_source_specs()
    if (
        len(enformer_source) != 570
        or sum(item.dtype == "F32" for item in enformer_source) != 542
        or sum(item.dtype == "I64" for item in enformer_source) != 28
        or len(aliases) != 283
    ):
        raise AssertionError("Enformer source manifest differs")
    enformer_runtime = [item for item in enformer_primary_specs() if item.dtype == "F32"]
    space_source = space_primary_specs() + space_drop_specs()
    space_runtime = [item for item in space_primary_specs() if item.dtype == "F32"]
    if (
        len(enformer_runtime) != 269
        or sum(item.nbytes for item in enformer_runtime) != 919500800
        or len(space_source) != 363
        or sum(item.dtype == "F32" for item in space_source) != 349
        or sum(item.dtype == "I64" for item in space_source) != 14
        or len(space_runtime) != 315
        or sum(item.nbytes for item in space_runtime) != 2166370656
        or max(len(item.name.encode("utf-8")) for item in enformer_runtime + space_runtime) >= 96
    ):
        raise AssertionError("sequence-CNN canonical manifest differs")
    # Importing another family is harmless, but it must not contaminate this
    # converter's selected manifests.
    if len(roformer_runtime_specs()) != 196:
        raise AssertionError("shared T36 helper import is incomplete")
    manifest = json.loads(args.profiles.read_text(encoding="utf-8"))
    drift = copy.deepcopy(manifest)
    drift["models"][0]["revision"] = "0" * 40
    with tempfile.TemporaryDirectory(prefix="geneb-sequence-cnn-profile-") as raw:
        directory = Path(raw)
        drift_path = directory / "drift.json"
        drift_path.write_text(json.dumps(drift), encoding="utf-8")
        try:
            load_profiles(drift_path, "geneb-sequence-cnn-converter-v1")
        except ConversionError:
            pass
        else:
            raise AssertionError("unpinned sequence-CNN profile was accepted")
        test_receipt_extra_gate(directory)
    print("GENEB sequence-CNN converter contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
