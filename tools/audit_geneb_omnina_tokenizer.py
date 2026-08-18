#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit pinned OmniNA tokenizer parity and the narrow B97 contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from tokenizers import Tokenizer
from transformers import AutoTokenizer


RUNTIME_ID = "geneb-omnina-220m"
SOURCE_REPO = "XLS/OmniNA-220m"
REVISION = "64ea6ce7b250fc611773215ddcdd1ecca232de67"
TOKENIZER_SIZE = 2070715
TOKENIZER_SHA256 = "a067717a733809467fa5f47d9a6cdc2c69a019a81353f03d00cf4f2ee00d93d7"
ALPHABET = "ACGTN"
EXHAUSTIVE_MAX_LENGTH = 5
RANDOM_SEED = 0xB97
RANDOM_CASES = 1024
RANDOM_MAX_LENGTH = 2047
FIXED_IDS = {
    "ACGTNACGTNACGTN": [1, 2724, 3566, 30817, 12, 3566, 30817, 12, 30864],
    "ACGTNNNNNNNNNN": [
        1,
        2724,
        30864,
        30864,
        30864,
        30864,
        30864,
        30864,
        30864,
        30864,
        30864,
        30864,
    ],
    "[PAD]": [1, 32000],
    "[PAD]A": [1, 32000, 15],
    "A[PAD]": [1, 15, 32000],
    "A [PAD]": [1, 15, 30814, 32000],
}
EXPECTED_DECODER = {
    "type": "Sequence",
    "decoders": [
        {
            "type": "Replace",
            "pattern": {"String": "▁"},
            "content": " ",
        },
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0},
    ],
}
EXPECTED_NORMALIZER = {
    "type": "Sequence",
    "normalizers": [
        {"type": "Prepend", "prepend": "▁"},
        {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
    ],
}
EXPECTED_ADDED_TOKENS = [
    {
        "id": token_id,
        "content": piece,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": normalized,
        "special": True,
    }
    for token_id, piece, normalized in (
        (0, "<unk>", True),
        (1, "<s>", True),
        (2, "</s>", True),
        (32000, "[PAD]", False),
    )
]


class AuditError(RuntimeError):
    pass


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


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError("cannot read %s: %s" % (label, error)) from error
    if not isinstance(value, dict):
        raise AuditError(label + " root must be an object")
    return value


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
        raise AuditError(label + " contains a local absolute path")


def descriptor_asset(root: Path, descriptor_path: Path) -> tuple[dict[str, Any], Path]:
    descriptor = read_object(descriptor_path, "tokenizer descriptor")
    expected_keys = {
        "converter.schema",
        "converter.version",
        "compiler_manifest_sha256",
        "source_receipt_contract_sha256",
        "tokenizer.profile",
        "tokenizer.path",
        "tokenizer.sha256",
        "tokenizer.size",
    }
    if set(descriptor) != expected_keys:
        raise AuditError("tokenizer descriptor fields differ")
    relative = descriptor["tokenizer.path"]
    if (
        not isinstance(relative, str)
        or PurePosixPath(relative).is_absolute()
        or PureWindowsPath(relative).is_absolute()
    ):
        raise AuditError("tokenizer descriptor path is not portable")
    asset = (root / relative).resolve(strict=True)
    if (
        descriptor["converter.schema"] != "evo-tokenizer-conversion-receipt"
        or descriptor["converter.version"] != 1
        or descriptor["tokenizer.profile"] != "evo-tokenizer-v1"
        or descriptor["tokenizer.size"] != asset.stat().st_size
        or descriptor["tokenizer.sha256"] != sha256_file(asset)
    ):
        raise AuditError("tokenizer descriptor/asset integrity differs")
    return descriptor, asset


def official_ids(tokenizer: Any, sequence: str) -> list[int]:
    encoded = tokenizer(
        sequence,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [int(value) for value in encoded["input_ids"]]


def dna_cases() -> list[str]:
    cases = []
    for length in range(EXHAUSTIVE_MAX_LENGTH + 1):
        cases.extend("".join(value) for value in itertools.product(ALPHABET, repeat=length))
    generator = random.Random(RANDOM_SEED)
    for _ in range(RANDOM_CASES):
        length = generator.randint(EXHAUSTIVE_MAX_LENGTH + 1, RANDOM_MAX_LENGTH)
        cases.append("".join(generator.choice(ALPHABET) for _ in range(length)))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root-a", required=True, type=Path)
    parser.add_argument("--descriptor-a", required=True, type=Path)
    parser.add_argument("--root-b", required=True, type=Path)
    parser.add_argument("--descriptor-b", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
            if os.environ.get(variable) != "1":
                raise AuditError(variable + "=1 is required")
        snapshot = args.snapshot.resolve(strict=True)
        manifest_path = args.manifest.resolve(strict=True)
        root_a = args.root_a.resolve(strict=True)
        root_b = args.root_b.resolve(strict=True)
        descriptor_path_a = args.descriptor_a.resolve(strict=True)
        descriptor_path_b = args.descriptor_b.resolve(strict=True)
        runtime = args.runtime.resolve(strict=True)
        if snapshot.name != REVISION:
            raise AuditError("snapshot directory does not name the pinned revision")
        tokenizer_path = snapshot / "tokenizer.json"
        if (
            tokenizer_path.stat().st_size != TOKENIZER_SIZE
            or sha256_file(tokenizer_path) != TOKENIZER_SHA256
        ):
            raise AuditError("pinned OmniNA tokenizer source differs")
        manifest = read_object(manifest_path, "compiler manifest")
        options = manifest.get("options")
        if (
            manifest.get("source") != "huggingface-json"
            or manifest.get("kind") != "bpe"
            or not isinstance(options, dict)
            or options.get("ignore_hf_normalized_base_vocab_added_special_literals")
            is not True
            or options.get("ignore_hf_omnina_encode_inert_fields") is not True
        ):
            raise AuditError("OmniNA compiler manifest opt-in differs")

        descriptor_a, asset_a = descriptor_asset(root_a, descriptor_path_a)
        descriptor_b, asset_b = descriptor_asset(root_b, descriptor_path_b)
        if (
            asset_a.read_bytes() != asset_b.read_bytes()
            or descriptor_path_a.read_bytes() != descriptor_path_b.read_bytes()
            or descriptor_a != descriptor_b
        ):
            raise AuditError("B77 relocated OmniNA tokenizer outputs differ")
        if descriptor_a["compiler_manifest_sha256"] != sha256_file(manifest_path):
            raise AuditError("tokenizer descriptor does not bind compiler manifest")
        compiled = read_object(asset_a, "compiled tokenizer")
        if (
            compiled.get("normalization")
            != [
                {"op": "prepend-literal", "value": "▁"},
                {"op": "replace-literal", "from": " ", "to": "▁"},
            ]
            or compiled.get("pre_tokenizer") != {"kind": "whole-input"}
            or compiled.get("special_tokens")
            != {
                "unk": 0,
                "pad": 2,
                "bos": 1,
                "eos": 2,
                "cls": None,
                "sep": None,
                "mask": None,
            }
            or compiled.get("post_processor")
            != {
                "prefix_ids": [1],
                "suffix_ids": [],
                "padding": {"side": "left", "pad_id": 2},
            }
            or compiled.get("model", {}).get("literal_token_ids") != [32000]
            or len(compiled.get("model", {}).get("merges", [])) != 68819
            or len(compiled.get("vocab", [])) != 32001
        ):
            raise AuditError("compiled OmniNA tokenizer semantics differ")

        source = read_object(tokenizer_path, "source tokenizer")
        model = source.get("model")
        if (
            source.get("normalizer") != EXPECTED_NORMALIZER
            or source.get("pre_tokenizer") is not None
            or source.get("decoder") != EXPECTED_DECODER
            or source.get("added_tokens") != EXPECTED_ADDED_TOKENS
            or not isinstance(model, dict)
            or model.get("byte_fallback") is not True
            or model.get("fuse_unk") is not True
            or model.get("unk_token") != "<unk>"
        ):
            raise AuditError("pinned OmniNA source tokenizer shape differs")
        vocab = model.get("vocab")
        if not isinstance(vocab, dict):
            raise AuditError("pinned OmniNA vocabulary differs")
        byte_pieces = [
            piece
            for piece in vocab
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">")
        ]
        if byte_pieces:
            raise AuditError("pinned OmniNA unexpectedly contains byte pieces")

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, use_fast=True
        )
        if (
            tokenizer.__class__.__name__ != "LlamaTokenizerFast"
            or tokenizer.vocab_size != 32000
            or len(tokenizer) != 32001
            or tokenizer.padding_side != "left"
            or tokenizer.pad_token_id is not None
            or tokenizer.bos_token_id != 1
            or tokenizer.eos_token_id != 2
            or tokenizer.unk_token_id != 0
            or tokenizer.model_max_length < 10**20
        ):
            raise AuditError("official OmniNA tokenizer wrapper defaults differ")
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token != "</s>" or tokenizer.pad_token_id != 2:
            raise AuditError("GENEB EOS-as-pad override differs")

        all_cases = dna_cases()
        case_ids = [official_ids(tokenizer, sequence) for sequence in all_cases]
        for sequence, expected in FIXED_IDS.items():
            actual = official_ids(tokenizer, sequence)
            if actual != expected:
                raise AuditError("official fixed tokenizer vector differs for %r" % sequence)
            all_cases.append(sequence)
            case_ids.append(actual)
        vector_payload = "".join(
            "%s\t%s\n" % (sequence, ",".join(str(value) for value in values))
            for sequence, values in zip(all_cases, case_ids)
        ).encode("utf-8")

        source_without_fallback = json.loads(json.dumps(source))
        source_without_fallback["model"]["byte_fallback"] = False
        without_fallback = Tokenizer.from_str(
            json.dumps(source_without_fallback, ensure_ascii=False, separators=(",", ":"))
        )
        if any(
            without_fallback.encode(sequence).ids != values
            for sequence, values in zip(all_cases, case_ids)
        ):
            raise AuditError("byte_fallback=false changes audited DNA token IDs")
        source_without_fuse = json.loads(json.dumps(source))
        source_without_fuse["model"]["fuse_unk"] = False
        without_fuse = Tokenizer.from_str(
            json.dumps(source_without_fuse, ensure_ascii=False, separators=(",", ":"))
        )
        if any(
            without_fuse.encode(sequence).ids != values
            for sequence, values in zip(all_cases, case_ids)
        ):
            raise AuditError("fuse_unk=false changes audited DNA token IDs")
        repeated_unknown = "😀😀"
        fused_unknown_ids = tokenizer(repeated_unknown, add_special_tokens=True)[
            "input_ids"
        ]
        unfused_unknown_ids = without_fuse.encode(repeated_unknown).ids
        if fused_unknown_ids != [1, 30814, 0] or unfused_unknown_ids != [1, 30814, 0, 0]:
            raise AuditError("repeated-unknown fuse_unk differential differs")

        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="omnina-token-vectors-", suffix=".tsv", delete=False
        ) as temporary:
            temporary.write(vector_payload)
            vector_path = Path(temporary.name)
        try:
            result = subprocess.run(
                [
                    str(runtime),
                    "--verify-asset-vectors",
                    str(root_a),
                    descriptor_a["tokenizer.path"],
                    descriptor_a["tokenizer.sha256"],
                    str(descriptor_a["tokenizer.size"]),
                    str(vector_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            vector_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise AuditError("native OmniNA token parity failed: %s" % result.stderr)

        batch = tokenizer(["A", "ACGTNACGTNACGTN"], padding=True)
        if (
            batch["input_ids"][0] != [2, 2, 2, 2, 2, 2, 2, 1, 15]
            or batch["attention_mask"][0] != [0, 0, 0, 0, 0, 0, 0, 1, 1]
        ):
            raise AuditError("official left/EOS batch padding differs")

        normalized_alias_examples = {
            sequence: official_ids(tokenizer, sequence)
            for sequence in ("<s>A", "A<s>", "A <s>")
        }
        report = {
            "schema_version": 1,
            "kind": "geneb-tokenizer-audit",
            "runtime_id": RUNTIME_ID,
            "source": {
                "repo": SOURCE_REPO,
                "revision": REVISION,
                "tokenizer_size": TOKENIZER_SIZE,
                "tokenizer_sha256": TOKENIZER_SHA256,
            },
            "compiler": {
                "manifest_sha256": sha256_file(manifest_path),
                "asset_size": descriptor_a["tokenizer.size"],
                "asset_sha256": descriptor_a["tokenizer.sha256"],
                "descriptor_sha256": sha256_file(descriptor_path_a),
                "source_receipt_contract_sha256": descriptor_a[
                    "source_receipt_contract_sha256"
                ],
                "normalized_base_aliases_ignored": [0, 1, 2],
                "raw_literal_ids": [32000],
                "byte_fallback_ignored": True,
                "fuse_unk_ignored": True,
                "decode_only_sequence_ignored": True,
            },
            "official_environment": {
                "transformers": importlib.metadata.version("transformers"),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "tokenizer_class": "%s.%s"
                % (tokenizer.__class__.__module__, tokenizer.__class__.__name__),
                "local_files_only": True,
            },
            "parity": {
                "status": "passed",
                "domain": "raw strings over A/C/G/T/N plus fixed appended-[PAD] diagnostics",
                "exhaustive_alphabet": ALPHABET,
                "exhaustive_max_length": EXHAUSTIVE_MAX_LENGTH,
                "random_seed": RANDOM_SEED,
                "random_cases": RANDOM_CASES,
                "random_max_length": RANDOM_MAX_LENGTH,
                "total_cases": len(all_cases),
                "vector_payload_sha256": sha256_bytes(vector_payload),
                "fixed_ids": FIXED_IDS,
                "b77_relocated_asset_byte_identical": True,
                "b77_relocated_descriptor_byte_identical": True,
            },
            "differentials": {
                "byte_piece_count": 0,
                "byte_fallback_false_dna_ids_equal": True,
                "fuse_unk_false_dna_ids_equal": True,
                "repeated_unknown_input_utf8_sha256": sha256_bytes(
                    repeated_unknown.encode("utf-8")
                ),
                "repeated_unknown_official_fused_ids": fused_unknown_ids,
                "repeated_unknown_unfused_ids": unfused_unknown_ids,
                "general_unicode_parity_claimed": False,
                "normalized_alias_examples_official_only": normalized_alias_examples,
                "normalized_alias_parity_claimed": False,
                "wrapper_model_max_length_is_unknown_sentinel": True,
                "long_input_reference_parity_claimed": False,
                "official_padding_side": "left",
                "geneb_pad_token": "</s>",
                "geneb_pad_token_id": 2,
                "appended_pad_literal_id": 32000,
                "batch_reference_parity_claimed": False,
            },
        }
        portable(report, "audit report")
        payload = canonical_json(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and not args.force:
            raise AuditError("output already exists")
        args.output.write_bytes(payload)
        print(
            json.dumps(
                {
                    "output_sha256": sha256_bytes(payload),
                    "asset_sha256": descriptor_a["tokenizer.sha256"],
                    "descriptor_sha256": sha256_file(descriptor_path_a),
                    "total_cases": len(all_cases),
                },
                sort_keys=True,
            )
        )
    except (AuditError, OSError, ValueError) as error:
        print("error: %s" % error, file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
