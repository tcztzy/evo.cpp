#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile the pinned DeepGene RoFormer-only checkpoint for native CPU."""

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
    roformer_metadata,
    select_profile,
    validate_config,
    validate_receipt,
    validate_tokenizer_descriptor,
    validated_roformer_tensors,
    write_artifact,
)


PROFILE_FORMAT = "geneb-roformer-converter-v1"
ARTIFACT_PROFILE = "geneb-roformer-runtime-v1"


def default_config_path(name: str) -> Path:
    local = _SCRIPT_DIRECTORY.parent / "configs" / name
    if local.is_file():
        return local
    return _SCRIPT_DIRECTORY.parent / "share" / "evo" / "configs" / name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=default_config_path("geneb-models.json"))
    parser.add_argument("--profiles", type=Path, default=default_config_path("geneb-roformer-models.json"))
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, profile_root, _profile_payload = load_profiles(args.profiles.resolve(), PROFILE_FORMAT)
        profile, _, _ = select_profile(args.receipt.resolve(), profiles)
        implementation = profile_root.get("implementation_contract")
        if profile["runtime_id"] != "geneb-deepgene" or not isinstance(implementation, dict):
            raise ConversionError("RoFormer profile implementation contract differs")
        catalog_entry, catalog_root, catalog_payload = load_catalog_entry(args.catalog.resolve(), profile, "roformer")
        critical, receipt_payload = validate_receipt(
            args.receipt.resolve(), profile, "roformer",
            args.catalog.resolve(), catalog_payload, catalog_entry,
        )
        validate_config(critical["config.json"], profile)
        tokenizer, descriptor_sha256 = validate_tokenizer_descriptor(
            args.tokenizer_descriptor.resolve(),
            args.tokenizer_root.resolve() if args.tokenizer_root is not None else None,
            args.output.resolve().parent,
            profile,
        )
        tensors = validated_roformer_tensors(
            critical["pytorch_model.bin"], profile["source_contract"]
        )
        metadata = roformer_metadata(
            profile,
            implementation,
            tokenizer,
            descriptor_sha256,
            receipt_payload,
            catalog_payload,
            catalog_root,
            catalog_entry,
        )
        write_artifact(args.output, ARTIFACT_PROFILE, metadata, tensors, args.force)
        print("wrote %s" % args.output)
        print("variant=deepgene-roformer tensors=%d" % len(tensors))
        return 0
    except (ConversionError, FileExistsError, GenebArtifactError, ImportError, OSError, ValueError) as error:
        print("convert_geneb_roformer_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
