#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit pinned DNABERT-2 HF tokenization and relocated native BPE parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from transformers import AutoTokenizer


REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
TOKENIZER_SHA256 = "5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3"
TOKENIZER_SIZE = 167908
INPUTS = (
    ("canonical", "ACGTNACGTNACGTN"),
    ("second-oracle", "AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT"),
    ("hf-whitespace", "A C\tG\nT"),
    ("case-preserved", "acgt"),
    ("unknown-n", "N"),
    ("empty", ""),
    ("b90-special-literal", "[MASK]A"),
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


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(label + " root must be an object")
    return value


def validate_compiled_root(
    root: Path, descriptor_path: Path
) -> tuple[dict[str, Any], Path]:
    descriptor = read_json(descriptor_path, "tokenizer descriptor")
    if descriptor.get("converter.schema") != "evo-tokenizer-conversion-receipt":
        raise RuntimeError("tokenizer descriptor schema differs")
    if descriptor.get("tokenizer.profile") != "evo-tokenizer-v1":
        raise RuntimeError("tokenizer descriptor profile differs")
    relative = descriptor.get("tokenizer.path")
    if not isinstance(relative, str) or PurePosixPath(relative).is_absolute():
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

    snapshot = args.snapshot.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    root_a = args.root_a.resolve(strict=True)
    root_b = args.root_b.resolve(strict=True)
    descriptor_path_a = args.descriptor_a.resolve(strict=True)
    descriptor_path_b = args.descriptor_b.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    if snapshot.name != REVISION:
        raise RuntimeError("snapshot directory does not name the pinned revision")
    source_tokenizer = snapshot / "tokenizer.json"
    if (
        source_tokenizer.stat().st_size != TOKENIZER_SIZE
        or sha256_file(source_tokenizer) != TOKENIZER_SHA256
    ):
        raise RuntimeError("pinned DNABERT-2 tokenizer source differs")
    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(variable + "=1 is required")
    if not os.access(str(runtime), os.X_OK):
        raise RuntimeError("tokenizer runtime verifier is not executable")

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
    compiled_asset = read_json(asset_a, "compiled tokenizer asset")
    compiled_model = compiled_asset.get("model")
    if (
        not isinstance(compiled_model, dict)
        or compiled_model.get("literal_token_ids") != [0, 1, 2, 3, 4]
    ):
        raise RuntimeError("B90 compiled literal token IDs differ")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
        revision=REVISION,
    )
    vectors = []
    for label, sequence in INPUTS:
        encoded = tokenizer(
            sequence,
            max_length=512,
            truncation=True,
            return_special_tokens_mask=True,
        )
        ids = [int(value) for value in encoded["input_ids"]]
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
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "attention_mask": [int(value) for value in encoded["attention_mask"]],
                "special_tokens_mask": [
                    int(value) for value in encoded["special_tokens_mask"]
                ],
                "native_exit_code": result.returncode,
                "native_stdout_sha256": sha256_bytes(result.stdout),
                "native_stderr_sha256": sha256_bytes(result.stderr),
            }
        )
    special_ids = next(
        item["input_ids"]
        for item in vectors
        if item["label"] == "b90-special-literal"
    )
    if special_ids != [1, 4, 5, 2]:
        raise RuntimeError("B90 official special-literal IDs differ")

    generator_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    report = {
        "schema_version": 1,
        "kind": "geneb-dnabert2-tokenizer-parity-audit",
        "status": "passed",
        "source": {
            "repo": "zhihan1996/DNABERT-2-117M",
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
            "source_exact_base_vocab_alias_ids": [0, 1, 2, 3, 4],
            "compiled_literal_token_ids": compiled_model["literal_token_ids"],
            "official_and_native_special_literal_input": "[MASK]A",
            "official_and_native_special_literal_ids": special_ids,
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
            "padding_side": tokenizer.padding_side,
            "trust_remote_code": True,
        },
        "native_runtime": {
            "sha256": sha256_file(runtime),
            "used_only_for_compiled_tokenizer_parity": True,
            "not_used_as_model_oracle": True,
        },
        "vectors": vectors,
        "generator_sha256": generator_sha256,
    }
    payload = canonical_json(report)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "asset_sha256": descriptor_a["tokenizer.sha256"],
                "generator_sha256": generator_sha256,
                "output_sha256": sha256_bytes(payload),
                "vector_count": len(vectors),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
