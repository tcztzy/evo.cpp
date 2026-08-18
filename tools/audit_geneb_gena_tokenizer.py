#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit pinned GENA-LM B90 tokenizer parity and B77 relocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from transformers import AutoTokenizer


SOURCE_REPO = "AIRI-Institute/gena-lm-bert-base"
REVISION = "416f055300346a5830ca49438daf5f4e136ed9a8"
TOKENIZER_SHA256 = "2fdbfdf74f08e8b98acc63e5a36c36d2ef44cf688f9485d95dde76acb6bd3034"
TOKENIZER_SIZE = 1538643
EXPECTED_ADDED_TOKENS = [
    {
        "id": token_id,
        "content": piece,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }
    for token_id, piece in enumerate(
        ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]")
    )
]
EXPECTED_NORMALIZER = None
EXPECTED_PRE_TOKENIZER = None
EXPECTED_POST_PROCESSOR = {
    "type": "TemplateProcessing",
    "single": [
        {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
    ],
    "pair": [
        {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        {"Sequence": {"id": "B", "type_id": 1}},
        {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
    ],
    "special_tokens": {
        "[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]},
        "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]},
    },
}
INPUTS = (
    ("canonical", "ACGTNACGTNACGTN", [1, 114, 9, 0, 114, 9, 0, 114, 9, 0, 2]),
    (
        "second-oracle",
        "AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT",
        [1, 2873, 211, 12176, 9, 0, 114, 9, 0, 0, 0, 0, 15426, 311, 418, 9, 2],
    ),
    ("b90-raw-mask", "[MASK]A", [1, 4, 6, 2]),
    ("b90-adjacent-raw-literals", "[MASK][MASK]", [1, 4, 4, 2]),
    ("whitespace-preserved", "  ACGT  ", [1, 0, 0, 114, 9, 0, 0, 2]),
    ("case-preserved", "acgt", [1, 0, 0, 0, 0, 2]),
    ("unknown-n", "N", [1, 0, 2]),
)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def i64_digest(values: list[int]) -> str:
    return sha256_bytes(b"".join(struct.pack("<q", value) for value in values))


def portable(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            portable(key, label + " key")
            portable(item, "%s.%s" % (label, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            portable(item, "%s[%d]" % (label, index))
        return
    if not isinstance(value, str):
        return
    parsed = urlsplit(value)
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or parsed.scheme.lower() == "file"
    ):
        raise RuntimeError(label + " contains a local absolute filesystem path")


def read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(label + " root must be an object")
    return value


def validate_compiled_root(
    root: Path, descriptor_path: Path
) -> tuple[dict[str, Any], Path]:
    descriptor = read_object(descriptor_path, "tokenizer descriptor")
    if descriptor.get("converter.schema") != "evo-tokenizer-conversion-receipt":
        raise RuntimeError("tokenizer descriptor schema differs")
    if descriptor.get("tokenizer.profile") != "evo-tokenizer-v1":
        raise RuntimeError("tokenizer descriptor profile differs")
    relative = descriptor.get("tokenizer.path")
    if (
        not isinstance(relative, str)
        or PurePosixPath(relative).is_absolute()
        or PureWindowsPath(relative).is_absolute()
    ):
        raise RuntimeError("tokenizer descriptor path differs")
    asset = (root / relative).resolve(strict=True)
    if (
        sha256_file(asset) != descriptor.get("tokenizer.sha256")
        or asset.stat().st_size != descriptor.get("tokenizer.size")
    ):
        raise RuntimeError("compiled tokenizer asset differs from descriptor")
    return descriptor, asset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root-a", required=True, type=Path)
    parser.add_argument("--descriptor-a", required=True, type=Path)
    parser.add_argument("--root-b", required=True, type=Path)
    parser.add_argument("--descriptor-b", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(variable + "=1 is required")

    snapshot = args.snapshot.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    root_a = args.root_a.resolve(strict=True)
    root_b = args.root_b.resolve(strict=True)
    descriptor_path_a = args.descriptor_a.resolve(strict=True)
    descriptor_path_b = args.descriptor_b.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    if snapshot.name != REVISION:
        raise RuntimeError("snapshot directory does not name the pinned revision")
    tokenizer_path = snapshot / "tokenizer.json"
    if (
        tokenizer_path.stat().st_size != TOKENIZER_SIZE
        or sha256_file(tokenizer_path) != TOKENIZER_SHA256
    ):
        raise RuntimeError("pinned GENA tokenizer source differs")
    if not os.access(str(runtime), os.X_OK):
        raise RuntimeError("tokenizer runtime verifier is not executable")

    manifest = read_object(manifest_path, "compiler manifest")
    if manifest.get("kind") != "bpe" or manifest.get("source") != "huggingface-json":
        raise RuntimeError("GENA compiler manifest kind/source differs")
    descriptor_a, asset_a = validate_compiled_root(root_a, descriptor_path_a)
    descriptor_b, asset_b = validate_compiled_root(root_b, descriptor_path_b)
    if asset_a.read_bytes() != asset_b.read_bytes():
        raise RuntimeError("B77 relocated tokenizer assets differ")
    if descriptor_path_a.read_bytes() != descriptor_path_b.read_bytes():
        raise RuntimeError("B77 relocated tokenizer descriptors differ")
    if descriptor_a != descriptor_b:
        raise RuntimeError("B77 relocated tokenizer descriptor objects differ")
    if descriptor_a.get("compiler_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("compiled tokenizer does not bind the audited manifest")

    source_tokenizer = read_object(tokenizer_path, "source tokenizer")
    if source_tokenizer.get("normalizer") != EXPECTED_NORMALIZER:
        raise RuntimeError("pinned GENA normalizer differs")
    if source_tokenizer.get("pre_tokenizer") != EXPECTED_PRE_TOKENIZER:
        raise RuntimeError("pinned GENA pre-tokenizer differs")
    if source_tokenizer.get("post_processor") != EXPECTED_POST_PROCESSOR:
        raise RuntimeError("pinned GENA post-processor differs")
    if source_tokenizer.get("added_tokens") != EXPECTED_ADDED_TOKENS:
        raise RuntimeError("pinned GENA AddedToken contract differs")

    compiled = read_object(asset_a, "compiled tokenizer asset")
    if compiled.get("normalization") != []:
        raise RuntimeError("compiled GENA normalization differs")
    if compiled.get("pre_tokenizer") != {"kind": "whole-input"}:
        raise RuntimeError("compiled GENA pre-tokenizer differs")
    compiled_model = compiled.get("model")
    if (
        not isinstance(compiled_model, dict)
        or compiled_model.get("literal_token_ids") != [0, 1, 2, 3, 4]
        or len(compiled_model.get("merges", [])) != 31990
    ):
        raise RuntimeError("compiled GENA B90 literal/merge contract differs")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), use_fast=True, local_files_only=True
    )
    if (
        tokenizer.__class__.__name__ != "PreTrainedTokenizerFast"
        or int(tokenizer.vocab_size) != 32000
        or len(tokenizer) != 32000
        or tokenizer.padding_side != "right"
    ):
        raise RuntimeError("official GENA tokenizer class/vocabulary differs")

    vectors = []
    for label, sequence, expected_ids in INPUTS:
        encoded = tokenizer(
            [sequence],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        ids = [int(value) for value in encoded["input_ids"][0]]
        attention_mask = [int(value) for value in encoded["attention_mask"][0]]
        special_mask = [int(value) for value in encoded["special_tokens_mask"][0]]
        if ids != expected_ids or attention_mask != [1] * len(ids):
            raise RuntimeError("official tokenizer vector differs for " + label)
        result = subprocess.run(
            [
                str(runtime),
                "--verify-asset",
                str(root_a),
                descriptor_a["tokenizer.path"],
                descriptor_a["tokenizer.sha256"],
                str(descriptor_a["tokenizer.size"]),
                sequence,
                ",".join(str(value) for value in ids),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "native tokenizer parity failed for %s: %s"
                % (label, result.stderr.decode("utf-8", errors="replace"))
            )
        vectors.append(
            {
                "label": label,
                "input_ascii": sequence,
                "input_size": len(sequence.encode("ascii")),
                "input_sha256": sha256_bytes(sequence.encode("ascii")),
                "input_ids": ids,
                "input_ids_little_endian_i64_sha256": i64_digest(ids),
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "attention_mask": attention_mask,
                "special_tokens_mask": special_mask,
                "native_exit_code": result.returncode,
                "native_stdout_sha256": sha256_bytes(result.stdout),
                "native_stderr_sha256": sha256_bytes(result.stderr),
            }
        )

    generator_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    report = {
        "schema_version": 1,
        "kind": "geneb-gena-lm-tokenizer-parity-audit",
        "status": "passed",
        "source": {
            "repo": SOURCE_REPO,
            "revision": REVISION,
            "tokenizer_size": TOKENIZER_SIZE,
            "tokenizer_sha256": TOKENIZER_SHA256,
        },
        "b77": {
            "relocated_asset_byte_identical": True,
            "relocated_descriptor_byte_identical": True,
            "compiler_manifest_sha256": descriptor_a["compiler_manifest_sha256"],
            "source_receipt_contract_sha256": descriptor_a[
                "source_receipt_contract_sha256"
            ],
            "descriptor_sha256": sha256_file(descriptor_path_a),
        },
        "b90": {
            "raw_input_longest_match_special_literals": True,
            "source_literal_token_ids": [0, 1, 2, 3, 4],
            "compiled_literal_token_ids": compiled_model["literal_token_ids"],
            "official_and_native_special_literal_ids": [1, 4, 6, 2],
            "status": "passed",
        },
        "compiled": {
            "profile": descriptor_a["tokenizer.profile"],
            "path": descriptor_a["tokenizer.path"],
            "size": descriptor_a["tokenizer.size"],
            "sha256": descriptor_a["tokenizer.sha256"],
        },
        "official_tokenizer": {
            "class": "%s.%s"
            % (tokenizer.__class__.__module__, tokenizer.__class__.__name__),
            "vocab_size": int(tokenizer.vocab_size),
            "length_including_added_tokens": len(tokenizer),
            "padding_side": tokenizer.padding_side,
        },
        "native_runtime": {
            "sha256": sha256_file(runtime),
            "used_only_for_compiled_tokenizer_parity": True,
            "not_used_as_model_oracle": True,
        },
        "vectors": vectors,
        "generator_sha256": generator_sha256,
    }
    portable(report, "tokenizer audit")
    payload = canonical_json(report)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "asset_sha256": descriptor_a["tokenizer.sha256"],
                "b90_ids": vectors[2]["input_ids"],
                "generator_sha256": generator_sha256,
                "output_sha256": sha256_bytes(payload),
                "vector_count": len(vectors),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit("tokenizer audit failed: %s" % error)
