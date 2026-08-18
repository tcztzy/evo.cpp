#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit and stage the pinned GROVER tokenizer for strict native conversion."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import struct
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import tokenizers
from tokenizers import Tokenizer


REPO = "PoetschLab/GROVER"
REVISION = "6b223110f0d6963e849f55bc2a2f3cff0e38c7a4"
SOURCE_FILES = {
    "tokenizer.json": (
        24637,
        "86b51bd8ecfc52cdc04f9f6787a2bd64fc5784709e7fd988b82d2e97348f9afd",
    ),
    "tokenizer_config.json": (
        314,
        "8c6c05d4a32b1235387c4d2a546bce640d7889f74d4b8cc6c8da549641e97517",
    ),
    "special_tokens_map.json": (
        77,
        "894589f317dfaaa10cbb6e467380b89bb848fe99f60ff6863ce120d81d71eff4",
    ),
}
NORMALIZATION_DECISIONS = [
    "drop unreachable tokenizer vocab entry id 609 with empty piece",
    (
        "drop exactly four unreachable identity BPE merges: "
        "A<space>, C<space>, G<space>, T<space>"
    ),
]
NORMALIZATION_PATCH_SHA256 = (
    "75290af6fa6c6ddbb582706f70d53945ca2b1433aa5998828063f14ba4cd0c6c"
)
EMPTY_IDENTITY_MERGES = ["A ", "C ", "G ", "T "]
NATIVE_PARITY_INPUTS = (
    ("canonical", "ACGTNACGTNACGTN"),
    ("second-oracle", "AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT"),
    ("b90-special-literal", "[MASK]A"),
)


def canonical_json(value: Any, newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def validate_snapshot(path: Path) -> Path:
    snapshot = path.resolve(strict=True)
    if snapshot.name != REVISION:
        raise RuntimeError("GROVER snapshot directory does not name the pinned revision")
    for name, (expected_size, expected_digest) in SOURCE_FILES.items():
        source = snapshot / name
        if not source.is_file():
            raise RuntimeError("missing pinned GROVER tokenizer source: " + name)
        if source.stat().st_size != expected_size or sha256_file(source) != expected_digest:
            raise RuntimeError("pinned GROVER tokenizer source differs: " + name)
    return snapshot


def file_record(role: str, name: str, path: Path, include_path: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "name": name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if include_path:
        record["path"] = str(path.resolve(strict=True))
    return record


def audit_model(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = source.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise RuntimeError("GROVER tokenizer model is not BPE")
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise RuntimeError("GROVER BPE vocabulary/merges are invalid")
    by_id = {token_id: piece for piece, token_id in vocab.items()}
    if len(vocab) != 610 or sorted(by_id) != list(range(610)):
        raise RuntimeError("GROVER source vocabulary is not exact IDs 0..609")
    if by_id[609] != "" or any(
        piece == "" for token_id, piece in by_id.items() if token_id != 609
    ):
        raise RuntimeError("GROVER empty vocabulary tail differs")
    if merges[:4] != EMPTY_IDENTITY_MERGES:
        raise RuntimeError("GROVER empty-piece merge prefix differs")

    retained_pieces: set[str] = set()
    for index, merge in enumerate(merges[4:], start=4):
        parts = merge.split(" ") if isinstance(merge, str) else merge
        if (
            not isinstance(parts, list)
            or len(parts) != 2
            or any(not isinstance(part, str) or not part for part in parts)
        ):
            raise RuntimeError("GROVER merge %d has an empty/invalid operand" % index)
        retained_pieces.update(parts)
        retained_pieces.add(parts[0] + parts[1])
    if "" in retained_pieces:
        raise RuntimeError("empty piece participates in a retained merge")

    added_ids = []
    for token in source.get("added_tokens", []):
        if not isinstance(token, dict) or not isinstance(token.get("id"), int):
            raise RuntimeError("GROVER added token is invalid")
        added_ids.append(token["id"])
    if 609 in added_ids:
        raise RuntimeError("empty vocabulary tail participates in added tokens")
    if b"609" in canonical_json(source.get("post_processor"), newline=False):
        raise RuntimeError("empty vocabulary tail appears in the post processor")

    del vocab[""]
    model["merges"] = merges[4:]
    proof = {
        "source_merge_count": len(merges),
        "removed_empty_identity_merge_count": 4,
        "derived_merge_count": len(model["merges"]),
        "empty_piece_in_retained_merge_closure": False,
        "empty_id_in_added_tokens": False,
        "empty_id_in_post_processor": False,
        "checkpoint_embedding_rows": 609,
        "config_vocab_size": 609,
    }
    return source, proof


def parity_report(original_path: Path, audited_path: Path) -> dict[str, Any]:
    original = Tokenizer.from_file(str(original_path))
    audited = Tokenizer.from_file(str(audited_path))
    alphabet = "ACGTN"
    corpus = [""]
    for length in range(1, 8):
        corpus.extend("".join(value) for value in itertools.product(alphabet, repeat=length))
    corpus.extend(
        [
            "ACGT ACGT",
            " ACGT ",
            "acgt",
            "ACGT\nTGCA",
            "AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT",
            "[MASK]A",
        ]
    )

    digest = hashlib.sha256()
    maximum_tokens = 0
    for offset in range(0, len(corpus), 4096):
        batch = corpus[offset : offset + 4096]
        original_encodings = original.encode_batch(batch, add_special_tokens=True)
        audited_encodings = audited.encode_batch(batch, add_special_tokens=True)
        for sequence, expected, actual in zip(batch, original_encodings, audited_encodings):
            expected_fields = (
                expected.ids,
                expected.attention_mask,
                expected.type_ids,
                expected.special_tokens_mask,
            )
            actual_fields = (
                actual.ids,
                actual.attention_mask,
                actual.type_ids,
                actual.special_tokens_mask,
            )
            if expected_fields != actual_fields:
                raise RuntimeError("tokenizer parity differs for %r" % sequence)
            if 609 in expected.ids:
                raise RuntimeError("official tokenizer emitted unreachable ID 609")
            encoded = sequence.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
            for values in expected_fields:
                digest.update(struct.pack("<I", len(values)))
                for value in values:
                    digest.update(struct.pack("<I", value))
            maximum_tokens = max(maximum_tokens, len(expected.ids))

    return {
        "schema_version": 1,
        "kind": "geneb-grover-tokenizer-parity",
        "source_repo": REPO,
        "source_revision": REVISION,
        "original_tokenizer_sha256": sha256_file(original_path),
        "audited_tokenizer_sha256": sha256_file(audited_path),
        "tokenizers_version": tokenizers.__version__,
        "alphabet": alphabet,
        "exhaustive_lengths": [0, 7],
        "targeted_case_count": 6,
        "sequence_count": len(corpus),
        "maximum_tokens": maximum_tokens,
        "official_emitted_id_609": False,
        "ids_masks_type_ids_and_special_masks_equal": True,
        "corpus_and_encodings_sha256": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root-a", type=Path)
    parser.add_argument("--descriptor-a", type=Path)
    parser.add_argument("--root-b", type=Path)
    parser.add_argument("--descriptor-b", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()

    compiled_arguments = (
        args.root_a,
        args.descriptor_a,
        args.root_b,
        args.descriptor_b,
        args.runtime,
    )
    if any(value is not None for value in compiled_arguments) and not all(
        value is not None for value in compiled_arguments
    ):
        raise RuntimeError(
            "compiled GROVER audit requires both roots/descriptors and runtime"
        )

    snapshot = validate_snapshot(args.snapshot)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generator_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    source_path = snapshot / "tokenizer.json"
    source_payload = source_path.read_bytes()
    source = json.loads(source_payload.decode("utf-8"))
    audited, proof = audit_model(source)
    audited_path = output / "tokenizer.audited-609.json"
    audited_path.write_bytes(canonical_json(audited))

    if (
        sha256_bytes(canonical_json(NORMALIZATION_DECISIONS, newline=False))
        != NORMALIZATION_PATCH_SHA256
    ):
        raise RuntimeError("GROVER normalization-decision digest differs")

    audit = {
        "schema_version": 1,
        "kind": "geneb-grover-unreachable-tokenizer-tail-audit",
        "generator_sha256": generator_sha256,
        "normalization_decisions": NORMALIZATION_DECISIONS,
        "normalization_patch_sha256": NORMALIZATION_PATCH_SHA256,
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "file": "tokenizer.json",
            "size": len(source_payload),
            "sha256": sha256_bytes(source_payload),
            "vocab_size": 610,
        },
        "operation": {
            "removed_id": 609,
            "removed_piece": "",
            "removed_identity_merges": EMPTY_IDENTITY_MERGES,
            "reason": (
                "zero-length BPE piece is never emitted by pre-tokenization, so the "
                "four (base, empty) identity pairs are unreachable; removing the empty "
                "tail therefore requires removing exactly those four merges"
            ),
            "remaining_ids": {"first": 0, "last": 608, "count": 609},
        },
        "derived": {
            "file": audited_path.name,
            "size": audited_path.stat().st_size,
            "sha256": sha256_file(audited_path),
            "vocab_size": 609,
        },
        "proof": proof,
    }
    audit_path = output / "unreachable-tail-audit.json"
    audit_path.write_bytes(canonical_json(audit))

    sources = (
        ("tokenizer", audited_path.name, audited_path),
        ("tokenizer_config", "tokenizer_config.json", snapshot / "tokenizer_config.json"),
        (
            "special_tokens_map",
            "special_tokens_map.json",
            snapshot / "special_tokens_map.json",
        ),
    )
    manifest = {
        "format": "evo-tokenizer-compiler-v1",
        "source": "huggingface-json",
        "kind": "bpe",
        "files": [file_record(role, name, path, False) for role, name, path in sources],
        "options": {
            "special_tokens": {
                "unk": "[UNK]",
                "pad": "[PAD]",
                "bos": None,
                "eos": None,
                "cls": "[CLS]",
                "sep": "[SEP]",
                "mask": "[MASK]",
            },
            "padding_side": "right",
        },
    }
    receipt = {
        "schema_version": 1,
        "kind": "tokenizer-source",
        "files": [file_record(role, name, path, True) for role, name, path in sources],
    }
    manifest_path = output / "manifest.json"
    receipt_path = output / "source-receipt.json"
    manifest_path.write_bytes(canonical_json(manifest))
    receipt_path.write_bytes(canonical_json(receipt))

    parity = parity_report(source_path, audited_path)
    if all(value is not None for value in compiled_arguments):
        root_a = args.root_a.resolve(strict=True)
        root_b = args.root_b.resolve(strict=True)
        descriptor_path_a = args.descriptor_a.resolve(strict=True)
        descriptor_path_b = args.descriptor_b.resolve(strict=True)
        runtime = args.runtime.resolve(strict=True)
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

        audited_tokenizer = Tokenizer.from_file(str(audited_path))
        native_vectors = []
        for label, sequence in NATIVE_PARITY_INPUTS:
            expected = audited_tokenizer.encode(sequence, add_special_tokens=True)
            command = [
                str(runtime),
                "--verify-asset",
                str(root_a),
                descriptor_a["tokenizer.path"],
                descriptor_a["tokenizer.sha256"],
                str(descriptor_a["tokenizer.size"]),
                sequence,
                ",".join(str(value) for value in expected.ids),
            ]
            result = subprocess.run(
                command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "native tokenizer parity failed for %s: %s"
                    % (label, result.stderr.decode("utf-8", errors="replace"))
                )
            native_vectors.append(
                {
                    "label": label,
                    "input_ascii": sequence,
                    "input_sha256": sha256_bytes(sequence.encode("ascii")),
                    "input_ids": expected.ids,
                    "native_exit_code": result.returncode,
                    "native_stdout_sha256": sha256_bytes(result.stdout),
                    "native_stderr_sha256": sha256_bytes(result.stderr),
                }
            )
        special_literal_ids = next(
            item["input_ids"]
            for item in native_vectors
            if item["label"] == "b90-special-literal"
        )
        if special_literal_ids != [2, 4, 5, 3]:
            raise RuntimeError("B90 official special-literal IDs differ")
        parity.update(
            {
                "b77": {
                    "relocated_asset_byte_identical": True,
                    "relocated_descriptor_byte_identical": True,
                    "compiler_manifest_sha256": descriptor_a[
                        "compiler_manifest_sha256"
                    ],
                    "source_receipt_contract_sha256": descriptor_a[
                        "source_receipt_contract_sha256"
                    ],
                    "descriptor_sha256": sha256_file(descriptor_path_a),
                },
                "b90": {
                    "source_exact_base_vocab_alias_ids": [0, 1, 2, 3, 4],
                    "compiled_literal_token_ids": compiled_model[
                        "literal_token_ids"
                    ],
                    "official_and_native_special_literal_input": "[MASK]A",
                    "official_and_native_special_literal_ids": special_literal_ids,
                    "status": "passed",
                },
                "compiled": {
                    "profile": descriptor_a["tokenizer.profile"],
                    "path": descriptor_a["tokenizer.path"],
                    "size": descriptor_a["tokenizer.size"],
                    "sha256": descriptor_a["tokenizer.sha256"],
                },
                "native_runtime": {
                    "sha256": sha256_file(runtime),
                    "used_only_for_compiled_tokenizer_parity": True,
                    "not_used_as_model_oracle": True,
                },
                "native_vectors": native_vectors,
            }
        )
    parity["generator_sha256"] = generator_sha256
    parity_path = output / "tokenizer-parity.json"
    parity_path.write_bytes(canonical_json(parity))
    print(
        json.dumps(
            {
                "audit_sha256": sha256_file(audit_path),
                "audited_tokenizer_sha256": sha256_file(audited_path),
                "manifest_sha256": sha256_file(manifest_path),
                "parity_sha256": sha256_file(parity_path),
                "receipt_sha256": sha256_file(receipt_path),
                "sequence_count": parity["sequence_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
