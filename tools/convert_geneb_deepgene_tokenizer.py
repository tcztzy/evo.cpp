#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile DeepGene's pinned tokenizers-0.13 BPE JSON to evo-tokenizer-v1.

The old serializer predates the explicit ``byte_fallback`` field.  This
converter verifies that exact legacy shape, injects only ``false`` in memory,
and delegates all remaining validation and compilation to the shared compiler.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import convert_tokenizer_asset as compiler  # noqa: E402


LEGACY_BPE_FIELDS = {
    "type",
    "dropout",
    "unk_token",
    "continuing_subword_prefix",
    "end_of_word_suffix",
    "fuse_unk",
    "vocab",
    "merges",
}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--asset-path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    temporary = None
    try:
        (
            source_kind,
            output_kind,
            paths,
            manifest_bytes,
            receipt_contract,
            manifest,
        ) = compiler.verified_assets(args.manifest.resolve(), args.receipt.resolve())
        if source_kind != "huggingface-json" or output_kind != "bpe":
            raise compiler.ConversionError("DeepGene requires a Hugging Face BPE manifest")
        tokenizer, _ = compiler.read_json(paths["tokenizer"], "DeepGene tokenizer")
        model = compiler.object_value(tokenizer.get("model"), "DeepGene HF model")
        if set(model) != LEGACY_BPE_FIELDS:
            raise compiler.ConversionError(
                "DeepGene legacy BPE fields differ: missing=%s extra=%s"
                % (sorted(LEGACY_BPE_FIELDS - set(model)), sorted(set(model) - LEGACY_BPE_FIELDS))
            )
        if model.get("type") != "BPE" or model.get("fuse_unk") is not False:
            raise compiler.ConversionError("DeepGene legacy BPE behavior differs")
        normalized = copy.deepcopy(tokenizer)
        normalized["model"]["byte_fallback"] = False
        descriptor, temporary_name = tempfile.mkstemp(prefix="deepgene-tokenizer-", suffix=".json")
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as output:
            output.write(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        normalized_paths = dict(paths)
        normalized_paths["tokenizer"] = temporary
        asset = compiler.compile_huggingface(output_kind, normalized_paths, manifest["options"])
        payload = compiler.canonical_json(asset)
        if len(payload) > compiler.MAX_TOKENIZER_ASSET_BYTES:
            raise compiler.ConversionError("compiled tokenizer exceeds the runtime limit")
        asset_path = compiler.normalized_relative_path(
            args.asset_path if args.asset_path is not None else args.output.name,
            "artifact tokenizer path",
        )
        conversion = {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": sha256(manifest_bytes),
            "source_receipt_contract_sha256": sha256(receipt_contract),
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": asset_path,
            "tokenizer.sha256": sha256(payload),
            "tokenizer.size": len(payload),
        }
        compiler.publish_outputs(
            (
                (args.output, payload),
                (args.descriptor, compiler.canonical_json(conversion)),
            ),
            args.force,
        )
        print("wrote %s" % args.output)
        print("tokenizer.sha256=%s" % conversion["tokenizer.sha256"])
        return 0
    except (compiler.ConversionError, FileExistsError, OSError, ValueError) as error:
        print("convert_geneb_deepgene_tokenizer: error: %s" % error, file=sys.stderr)
        return 2
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
