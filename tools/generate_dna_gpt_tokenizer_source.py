#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned DNAGPT KmerTokenizer source kit without network access."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if not (_SCRIPT_DIRECTORY / "evo").is_dir():
    _INSTALLED_PYTHON = _SCRIPT_DIRECTORY.parent / "share" / "evo" / "python"
    if _INSTALLED_PYTHON.is_dir():
        sys.path.insert(0, str(_INSTALLED_PYTHON))

from evo.geneb_gpt_artifact import dna_gpt_vocab_pieces  # noqa: E402


MODEL_KINDS = {
    "geneb-dna-gpt-0-1b-h": "static-sixmer",
    "geneb-dna-gpt-3b-m": "dynamic-sixmer",
}


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def special_tokens() -> Dict[str, Any]:
    return {
        "unk": 0,
        "pad": 20,
        # GENEB's prefix-only handoff passes the literal <R> while disabling
        # the tokenizer post-processor.  Naming ID 21 here lets the runtime
        # recognize that literal before fixed six-mer chunking.
        "bos": 21,
        "eos": None,
        "cls": None,
        "sep": None,
        "mask": None,
    }


def post_processor() -> Dict[str, Any]:
    return {
        "prefix_ids": [21],
        "suffix_ids": [],
        "padding": {"side": "right", "pad_id": 20},
    }


def source_and_manifest(
    model_id: str, output_directory: Path
) -> Tuple[Path, bytes, Dict[str, Any]]:
    kind = MODEL_KINDS[model_id]
    pieces = dna_gpt_vocab_pieces(kind)
    if kind == "static-sixmer":
        source_name = "dna-gpt-static-sixmer.vocab.txt"
        source_payload = ("\n".join(pieces) + "\n").encode("ascii")
        source_kind = "vocab-text"
        runtime_kind = "kmer"
        options = {
            "normalization": [],
            "pre_tokenizer": {"kind": "none"},
            "model": {
                "k": 6,
                "stride": 6,
                "tail": "lookup",
                "unknown_policy": "unk",
                "match_special_literals": True,
            },
            "post_processor": post_processor(),
            "special_tokens": special_tokens(),
        }
    else:
        source_name = "dna-gpt-dynamic-sixmer.source.json"
        source_payload = canonical_json(
            {
                "format": "evo-tokenizer-source-v1",
                "kind": "kmer",
                "normalization": [],
                "pre_tokenizer": {"kind": "none"},
                "model": {
                    "k": 6,
                    "stride": 6,
                    "tail": "lookup",
                    "unknown_policy": "unk",
                    "match_special_literals": True,
                },
                "post_processor": post_processor(),
                "special_tokens": special_tokens(),
                "vocab": [
                    {"id": index, "piece": piece}
                    for index, piece in enumerate(pieces)
                ],
            }
        )
        source_kind = "custom"
        runtime_kind = "kmer"
        options = {}
    source_path = output_directory / source_name
    manifest = {
        "format": "evo-tokenizer-compiler-v1",
        "source": source_kind,
        "kind": runtime_kind,
        "files": [
            {
                "role": "vocab" if source_kind == "vocab-text" else "spec",
                "name": source_name,
                "size": len(source_payload),
                "sha256": sha256_bytes(source_payload),
            }
        ],
        "options": options,
    }
    return source_path, source_payload, manifest


def publish(outputs: Sequence[Tuple[Path, bytes]], force: bool) -> None:
    if not force:
        existing = [str(path) for path, _ in outputs if path.exists()]
        if existing:
            raise FileExistsError("output already exists: %s" % ", ".join(existing))
    staged = []  # type: List[Tuple[Path, Path]]
    try:
        for path, payload in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            staged.append((temporary, path))
        for temporary, path in staged:
            if force:
                os.replace(str(temporary), str(path))
            else:
                os.link(str(temporary), str(path))
                temporary.unlink()
    finally:
        for temporary, _ in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, choices=sorted(MODEL_KINDS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        output_directory = args.output_dir.resolve()
        source_path, source_payload, manifest = source_and_manifest(
            args.model_id, output_directory
        )
        role = manifest["files"][0]["role"]
        receipt = {
            "schema_version": 1,
            "kind": "tokenizer-source",
            "files": [
                {
                    "role": role,
                    "name": source_path.name,
                    "size": len(source_payload),
                    "sha256": sha256_bytes(source_payload),
                    "path": str(source_path),
                }
            ],
        }
        manifest_path = output_directory / (args.model_id + ".tokenizer-manifest.json")
        receipt_path = output_directory / (args.model_id + ".tokenizer-receipt.json")
        publish(
            [
                (source_path, source_payload),
                (manifest_path, canonical_json(manifest)),
                (receipt_path, canonical_json(receipt)),
            ],
            args.force,
        )
        print("source=%s" % source_path)
        print("manifest=%s" % manifest_path)
        print("receipt=%s" % receipt_path)
        return 0
    except (FileExistsError, OSError, ValueError) as error:
        print(
            "generate_dna_gpt_tokenizer_source: error: %s" % error,
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
