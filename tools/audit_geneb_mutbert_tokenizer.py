#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit MutBERT tokenizer truth, B77 relocation, B89, and B90 parity."""

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


REVISION = "b68d8d6c9ccd8167639b25fb979cbd39a5c5c60c"
TOKENIZER_SHA256 = "fd5232a7a128cfb88fcb2ea71568cffa92c24aa0b3ef129e73e9f40e5bf25b12"
TOKENIZER_SIZE = 2623
TOKENIZER_CONFIG_SHA256 = (
    "f9d18c81f4dd9dd7db02e9f27cc1203228147d890bfce9167c3af6465ff5b769"
)
TOKENIZER_CONFIG_SIZE = 158
VOCAB_TEXT_SHA256 = "57fb2e0c852a0408f870b96641df38becd8b28e3571db5fc9343399a62878624"
VOCAB_TEXT_SIZE = 38
LONG_DIFFERENTIAL_INPUT = "A" * 513
INPUTS = (
    ("canonical", "ACGTNACGTNACGTN"),
    ("second-oracle", "AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT"),
    ("hf-whitespace", "A C\tG\nT"),
    ("case-preserved", "acgt"),
    ("unknown-n", "N"),
    ("empty", ""),
    ("b90-special-literal", "[MASK]A"),
)
EXPECTED_VOCAB_MISMATCH = [
    {"piece": "A", "tokenizer_json_id": 5, "vocab_txt_id": 5},
    {"piece": "C", "tokenizer_json_id": 6, "vocab_txt_id": 7},
    {"piece": "G", "tokenizer_json_id": 7, "vocab_txt_id": 8},
    {"piece": "T", "tokenizer_json_id": 8, "vocab_txt_id": 6},
    {"piece": "[CLS]", "tokenizer_json_id": 1, "vocab_txt_id": 2},
    {"piece": "[MASK]", "tokenizer_json_id": 4, "vocab_txt_id": 4},
    {"piece": "[PAD]", "tokenizer_json_id": 3, "vocab_txt_id": 0},
    {"piece": "[SEP]", "tokenizer_json_id": 2, "vocab_txt_id": 3},
    {"piece": "[UNK]", "tokenizer_json_id": 0, "vocab_txt_id": 1},
]


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


def validate_compiled_root(root: Path, descriptor_path: Path) -> tuple[dict[str, Any], Path]:
    descriptor = read_object(descriptor_path, "tokenizer descriptor")
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


def summarized_ids(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "first_16": values[:16],
        "last_16": values[-16:],
        "little_endian_i64_sha256": i64_digest(values),
    }


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
    source_expectations = {
        "tokenizer.json": (TOKENIZER_SIZE, TOKENIZER_SHA256),
        "tokenizer_config.json": (TOKENIZER_CONFIG_SIZE, TOKENIZER_CONFIG_SHA256),
        "vocab.txt": (VOCAB_TEXT_SIZE, VOCAB_TEXT_SHA256),
    }
    for name, (size, digest) in source_expectations.items():
        path = snapshot / name
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError("pinned MutBERT tokenizer source differs: " + name)

    manifest = read_object(manifest_path, "compiler manifest")
    if manifest.get("kind") != "bpe" or manifest.get("source") != "huggingface-json":
        raise RuntimeError("MutBERT compiler manifest kind/source differs")
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
    asset = read_object(asset_a, "compiled tokenizer asset")
    if asset.get("model") != {"literal_token_ids": [0, 1, 2, 3, 4], "merges": []}:
        raise RuntimeError("B90 compiled literal special-token contract differs")
    if not os.access(str(runtime), os.X_OK):
        raise RuntimeError("tokenizer runtime verifier is not executable")
    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(variable + "=1 is required")

    source_tokenizer = read_object(snapshot / "tokenizer.json", "source tokenizer")
    json_vocab = source_tokenizer.get("model", {}).get("vocab")
    if not isinstance(json_vocab, dict):
        raise RuntimeError("source tokenizer vocabulary differs")
    text_pieces = (snapshot / "vocab.txt").read_text(encoding="utf-8").splitlines()
    text_vocab = {piece: index for index, piece in enumerate(text_pieces)}
    mismatch = [
        {
            "piece": piece,
            "tokenizer_json_id": json_vocab.get(piece),
            "vocab_txt_id": text_vocab.get(piece),
        }
        for piece in sorted(set(json_vocab) | set(text_vocab))
    ]
    if mismatch != EXPECTED_VOCAB_MISMATCH:
        raise RuntimeError("tokenizer.json/vocab.txt audited mapping differs")
    if json_vocab == text_vocab:
        raise RuntimeError("expected tokenizer.json/vocab.txt conflict disappeared")
    expected_added_aliases = [
        (0, "[UNK]"),
        (1, "[CLS]"),
        (2, "[SEP]"),
        (3, "[PAD]"),
        (4, "[MASK]"),
    ]
    actual_added_aliases = [
        (item.get("id"), item.get("content"))
        for item in source_tokenizer.get("added_tokens", [])
    ]
    if actual_added_aliases != expected_added_aliases:
        raise RuntimeError("source B90 special aliases differ")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), use_fast=True, local_files_only=True
    )
    if tokenizer.__class__.__name__ != "PreTrainedTokenizerFast":
        raise RuntimeError("official MutBERT tokenizer class differs")
    if int(tokenizer.vocab_size) != 9 or len(tokenizer) != 9:
        raise RuntimeError("official MutBERT tokenizer vocabulary size differs")
    if int(tokenizer.model_max_length) <= 10**20:
        raise RuntimeError("B89 tokenizer max-length sentinel differs")

    vectors = []
    for label, sequence in INPUTS:
        encoded = tokenizer(
            [sequence],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        ids = [int(value) for value in encoded["input_ids"][0].tolist()]
        command = [
            str(runtime),
            "--verify-asset",
            str(root_a),
            descriptor_a["tokenizer.path"],
            descriptor_a["tokenizer.sha256"],
            str(descriptor_a["tokenizer.size"]),
            sequence,
            ",".join(str(value) for value in ids),
        ]
        result = subprocess.run(
            command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
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
                "attention_mask": [
                    int(value) for value in encoded["attention_mask"][0].tolist()
                ],
                "special_tokens_mask": [
                    int(value)
                    for value in encoded["special_tokens_mask"][0].tolist()
                ],
                "native_exit_code": result.returncode,
                "native_stdout_sha256": sha256_bytes(result.stdout),
                "native_stderr_sha256": sha256_bytes(result.stderr),
            }
        )
    b90_vector = next(item for item in vectors if item["label"] == "b90-special-literal")
    if b90_vector["input_ids"] != [1, 4, 5, 2]:
        raise RuntimeError("B90 official [MASK]A vector differs")

    long_reference = tokenizer(
        [LONG_DIFFERENTIAL_INPUT],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    long_normalized = tokenizer(
        [LONG_DIFFERENTIAL_INPUT],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt",
    )
    reference_ids = [int(value) for value in long_reference["input_ids"][0].tolist()]
    normalized_ids = [int(value) for value in long_normalized["input_ids"][0].tolist()]
    if len(reference_ids) != 515 or len(normalized_ids) != 512:
        raise RuntimeError("B89 long-input differential counts differ")

    generator_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    report = {
        "schema_version": 1,
        "kind": "geneb-mutbert-tokenizer-parity-audit",
        "status": "passed",
        "source": {
            "repo": "JadenLong/MutBERT",
            "revision": REVISION,
            "files": {
                name: {"size": size, "sha256": digest}
                for name, (size, digest) in source_expectations.items()
            },
        },
        "execution_truth": {
            "selected": "tokenizer.json",
            "vocab_txt_used": False,
            "vocab_txt_audited_mismatch": True,
            "id_mappings": mismatch,
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
        "b89": {
            "status": "passed",
            "reference_claim": "fixed-short-inputs-only",
            "long_input_reference_parity_claimed": False,
            "tokenizer_model_max_length": int(tokenizer.model_max_length),
            "extractor_call_has_explicit_max_length": False,
            "input_raw_base_count": len(LONG_DIFFERENTIAL_INPUT),
            "input_sha256": sha256_bytes(LONG_DIFFERENTIAL_INPUT.encode("ascii")),
            "reference_no_explicit_max_length": summarized_ids(reference_ids),
            "normalized_explicit_max_length_512": summarized_ids(normalized_ids),
        },
        "b90": {
            "status": "passed",
            "source_exact_base_vocab_alias_ids": [0, 1, 2, 3, 4],
            "compiled_literal_token_ids": asset["model"]["literal_token_ids"],
            "official_and_native_special_literal_input": "[MASK]A",
            "official_and_native_special_literal_ids": [1, 4, 5, 2],
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
                "b89_reference_tokens": len(reference_ids),
                "b89_normalized_tokens": len(normalized_ids),
                "b90_ids": b90_vector["input_ids"],
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
