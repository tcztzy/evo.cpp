#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile pinned Enformer/SPACE checkpoints for the native CPU runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if not (_SCRIPT_DIRECTORY / "evo").is_dir():
    _INSTALLED_PYTHON = _SCRIPT_DIRECTORY.parent / "share" / "evo" / "python"
    if _INSTALLED_PYTHON.is_dir():
        sys.path.insert(0, str(_INSTALLED_PYTHON))

from evo.geneb_t36_artifact import (  # noqa: E402
    ConversionError,
    GenebArtifactError,
    load_catalog_entry,
    load_profiles,
    select_profile,
    sequence_metadata,
    validate_config,
    validate_receipt,
    validated_sequence_tensors,
    write_artifact,
)


PROFILE_FORMAT = "geneb-sequence-cnn-converter-v1"
ARTIFACT_PROFILE = "geneb-sequence-cnn-runtime-v1"


def default_config_path(name: str) -> Path:
    local = _SCRIPT_DIRECTORY.parent / "configs" / name
    if local.is_file():
        return local
    return _SCRIPT_DIRECTORY.parent / "share" / "evo" / "configs" / name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=default_config_path("geneb-models.json"))
    parser.add_argument("--profiles", type=Path, default=default_config_path("geneb-sequence-cnn-models.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, profile_root, _profile_payload = load_profiles(args.profiles.resolve(), PROFILE_FORMAT)
        profile, _, _ = select_profile(args.receipt.resolve(), profiles)
        variant = profile.get("topology", {}).get("variant")
        implementations = profile_root.get("implementation_contracts")
        if variant not in ("enformer", "space") or not isinstance(implementations, dict) or variant not in implementations:
            raise ConversionError("sequence-CNN profile implementation contract differs")
        catalog_entry, catalog_root, catalog_payload = load_catalog_entry(args.catalog.resolve(), profile, "sequence-cnn")
        critical, receipt_payload = validate_receipt(
            args.receipt.resolve(), profile, "sequence-cnn",
            args.catalog.resolve(), catalog_payload, catalog_entry,
        )
        validate_config(critical["config.json"], profile)
        tensors = validated_sequence_tensors(
            critical["pytorch_model.bin"], variant, profile["source_contract"]
        )
        metadata = sequence_metadata(
            profile,
            implementations,
            receipt_payload,
            catalog_payload,
            catalog_root,
            catalog_entry,
        )
        write_artifact(args.output, ARTIFACT_PROFILE, metadata, tensors, args.force)
        print("wrote %s" % args.output)
        print("variant=%s tensors=%d" % (variant, len(tensors)))
        return 0
    except (ConversionError, FileExistsError, GenebArtifactError, ImportError, OSError, ValueError) as error:
        print("convert_geneb_sequence_cnn_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
