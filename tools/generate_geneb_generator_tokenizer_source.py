#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned GENERator-Eukaryote indexed-vocab tokenizer source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REQUESTED_REVISION = "main"
COMMON_TOKENIZER_SOURCE_FILES = {
    "modeling_generator.py": (
        9_927,
        "c605e5d3b201a93a4f8db998442a4a12ab018b214c50005a53f213e6cbbabe35",
    ),
    "special_tokens_map.json": (
        122,
        "4844a00974957de4445830bfc5892cb342f8a8dfaa50fe305a22875523c5a0f7",
    ),
    "tokenizer.py": (
        5_898,
        "211b26c254b872671c5387806a8c71820a26d2139c9a8221f6310486032f88ae",
    ),
    "tokenizer_config.json": (
        1_311,
        "e3545bbe1896de15730cde4c7a355af89fb61b713cd4444cf11a07100588cc35",
    ),
    "vocab.txt": (
        48_399,
        "068e0d76ca42ac672ade448afb2462ce4bd800a1fc1ad863d0b65992d90cf881",
    ),
}
EXPECTED_SOURCE_FILES_1_2B = {
    "config.json": (
        752,
        "2b47f1cbbadd52ad0f64e1802e3e7932e7f39cf8d42174e82bad4f6436504145",
    ),
    "model.safetensors": (
        4_648_274_384,
        "2bd949f005bcb1547f4f01d15a68491d51adb46c01e9790e4d9331500f6884d9",
    ),
    **COMMON_TOKENIZER_SOURCE_FILES,
}
EXPECTED_SOURCE_FILES_3B = {
    "config.json": (
        752,
        "a864d63840a26f0f9d7e5e2d014f361c188873148f39913a712aa14cd1b38b25",
    ),
    "model-00001-of-00003.safetensors": (
        4_996_117_216,
        "1d8a34d79bcbe6b6a19151d21889abc9dc653a4b436696e4eeb3f558fa1c72ca",
    ),
    "model-00002-of-00003.safetensors": (
        4_964_291_160,
        "227857e93be2d5fa24adf795648a1364e9e97049361e48b055717744b7d875a7",
    ),
    "model-00003-of-00003.safetensors": (
        2_032_674_024,
        "4905894c5aa4dcbcd29b682907bae7afee50c9472a52d8c715776d63ce8fbb2b",
    ),
    "model.safetensors.index.json": (
        22_464,
        "53d262ea3f8b70c296863b86df1c4df5e5558d284f3656e363c5adeedf1500b3",
    ),
    **COMMON_TOKENIZER_SOURCE_FILES,
}
SPECIAL_TOKENS = (
    "<oov>",
    "<s>",
    "</s>",
    "<pad>",
    "<mask>",
    "<bog>",
    "<eog>",
    "<bok>",
    "<eok>",
    "<+>",
    "<->",
    "<cds>",
    "<pseudo>",
    "<tRNA>",
    "<rRNA>",
    "<ncRNA>",
    "<miscRNA>",
    "<mam>",
    "<vrt>",
    "<inv>",
    "<pln>",
    "<fng>",
    "<prt>",
    "<arc>",
    "<bct>",
    "<mit>",
    "<plt>",
    "<plm>",
    "<vir>",
    "<sp0>",
    "<sp1>",
    "<sp2>",
)
EXPECTED_SPECIAL_MAP = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "mask_token": "<mask>",
    "pad_token": "<pad>",
    "unk_token": "<oov>",
}
EXPECTED_CONFIG_1_2B = {
    "architectures": ["GENERatorForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoModelForCausalLM": "modeling_generator.GENERatorForCausalLM"
    },
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 5632,
    "max_position_embeddings": 16384,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 26,
    "num_key_value_heads": 4,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.44.0",
    "use_cache": True,
    "vocab_size": 4128,
}
EXPECTED_CONFIG_3B = {
    **EXPECTED_CONFIG_1_2B,
    "hidden_size": 3072,
    "intermediate_size": 8448,
    "num_hidden_layers": 30,
}
EXPECTED_ADDED_TOKENS = {
    str(token_id): {
        "content": piece,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    }
    for token_id, piece in enumerate(SPECIAL_TOKENS[:5])
}
EXPECTED_TOKENIZER_CONFIG = {
    "add_bos_token": True,
    "add_eos_token": False,
    "added_tokens_decoder": EXPECTED_ADDED_TOKENS,
    "auto_map": {"AutoTokenizer": ["tokenizer.DNAKmerTokenizer", None]},
    "bos_token": "<s>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "</s>",
    "extra_special_tokens": {},
    "k": 6,
    "mask_token": "<mask>",
    "model_max_length": 1000000000000000019884624838656,
    "pad_token": "<pad>",
    "tokenizer_class": "DNAKmerTokenizer",
    "unk_token": "<oov>",
}


@dataclass(frozen=True)
class ModelSpec:
    runtime_id: str
    source_repo: str
    source_revision: str
    catalog_contract_sha256: str
    source_name: str
    manifest_name: str
    receipt_name: str
    audit_name: str
    source_files: dict[str, tuple[int, str]]
    expected_config: dict[str, Any]


MODEL_SPECS = {
    "geneb-generator-eukaryote-1-2b": ModelSpec(
        runtime_id="geneb-generator-eukaryote-1-2b",
        source_repo="GenerTeam/GENERator-eukaryote-1.2b-base",
        source_revision="5e872c94264891f9adf59d8ea64e426bb68badb5",
        catalog_contract_sha256=(
            "96faa077f463e99b02199b3b8ae6dd41cb42d7e350d07c983a82bb008072bcef"
        ),
        source_name="generator-eukaryote-1-2b-kmer.source.json",
        manifest_name="geneb-generator-eukaryote-1-2b-kmer-v1.json",
        receipt_name="geneb-generator-eukaryote-1-2b.tokenizer-receipt.json",
        audit_name="geneb-generator-eukaryote-1-2b.tokenizer-audit.json",
        source_files=EXPECTED_SOURCE_FILES_1_2B,
        expected_config=EXPECTED_CONFIG_1_2B,
    ),
    "geneb-generator-eukaryote-3b": ModelSpec(
        runtime_id="geneb-generator-eukaryote-3b",
        source_repo="GenerTeam/GENERator-eukaryote-3b-base",
        source_revision="7515b17659f092997335226c3fb6aafd06c0add9",
        catalog_contract_sha256=(
            "26b8cb7cb4e36aac1e16d52c14fc622b27ab5019b26a8c02bb067e60578d51d1"
        ),
        source_name="generator-eukaryote-3b-kmer.source.json",
        manifest_name="geneb-generator-eukaryote-3b-kmer-v1.json",
        receipt_name="geneb-generator-eukaryote-3b.tokenizer-receipt.json",
        audit_name="geneb-generator-eukaryote-3b.tokenizer-audit.json",
        source_files=EXPECTED_SOURCE_FILES_3B,
        expected_config=EXPECTED_CONFIG_3B,
    ),
}


class GenerationError(RuntimeError):
    """Raised when the pinned source or generated contract differs."""


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GenerationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise GenerationError(f"{label} root must be an object")
    return value


def validate_source_receipt(
    receipt_path: Path, snapshot: Path, model: ModelSpec
) -> str:
    receipt = load_object(receipt_path, "source receipt")
    if set(receipt) != {
        "schema_version",
        "kind",
        "model_id",
        "repo",
        "requested_revision",
        "resolved_revision",
        "source_kind",
        "catalog_path",
        "catalog_contract_sha256",
        "load_path",
        "files",
    }:
        raise GenerationError("source receipt fields differ")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != model.runtime_id
        or receipt["repo"] != model.source_repo
        or receipt["requested_revision"] != REQUESTED_REVISION
        or receipt["resolved_revision"] != model.source_revision
        or receipt["source_kind"] != "huggingface"
        or receipt["load_path"] is not None
    ):
        raise GenerationError("source receipt pinned identity differs")
    contract = receipt["catalog_contract_sha256"]
    if (
        not isinstance(contract, str)
        or len(contract) != 64
        or any(character not in "0123456789abcdef" for character in contract)
    ):
        raise GenerationError("source receipt catalog contract digest is invalid")
    if contract != model.catalog_contract_sha256:
        raise GenerationError("source receipt catalog contract differs")
    raw_files = receipt["files"]
    if not isinstance(raw_files, list) or len(raw_files) != len(model.source_files):
        raise GenerationError(
            f"source receipt must contain exactly {len(model.source_files)} files"
        )
    seen: set[str] = set()
    for index, raw in enumerate(raw_files):
        label = f"source receipt files[{index}]"
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "path",
            "size",
            "sha256",
        }:
            raise GenerationError(label + " fields differ")
        name = raw["name"]
        if (
            not isinstance(name, str)
            or PurePosixPath(name).is_absolute()
            or len(PurePosixPath(name).parts) != 1
            or name in seen
            or name not in model.source_files
        ):
            raise GenerationError(label + " name is duplicated or unexpected")
        seen.add(name)
        size, digest = model.source_files[name]
        if raw["size"] != size or raw["sha256"] != digest:
            raise GenerationError("source receipt metadata differs for " + name)
        locator = Path(raw["path"])
        logical = snapshot / name
        resolved_candidates: set[Path] = set()
        for candidate in (locator, logical):
            if not candidate.is_file() or candidate.stat().st_size != size:
                raise GenerationError("pinned source file differs for " + name)
            resolved_candidates.add(candidate.resolve(strict=True))
        if len(resolved_candidates) != 1:
            raise GenerationError("source receipt locator differs for " + name)
        if sha256_file(next(iter(resolved_candidates))) != digest:
            raise GenerationError("pinned source file differs for " + name)
    if seen != set(model.source_files):
        raise GenerationError("source receipt file set differs")
    return contract


def declared_special_tokens(tokenizer_path: Path) -> tuple[str, ...]:
    try:
        syntax = ast.parse(tokenizer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise GenerationError(f"cannot parse pinned tokenizer.py: {error}") from error
    values: list[tuple[str, ...]] = []
    for node in ast.walk(syntax):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "special_tokens"
        ):
            try:
                literal = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as error:
                raise GenerationError("tokenizer.py special token list is not literal") from error
            if not isinstance(literal, list) or not all(
                isinstance(piece, str) for piece in literal
            ):
                raise GenerationError("tokenizer.py special token list differs")
            values.append(tuple(literal))
    if values != [SPECIAL_TOKENS]:
        raise GenerationError("tokenizer.py 32-special-literal contract differs")
    return values[0]


def validate_indexed_vocab(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise GenerationError(f"cannot read pinned indexed vocab: {error}") from error
    expected = list(SPECIAL_TOKENS) + [
        "".join(kmer) for kmer in itertools.product("ATCG", repeat=6)
    ]
    if len(lines) != len(expected):
        raise GenerationError("indexed vocab line count differs")
    pieces: list[str] = []
    for expected_id, (line, expected_piece) in enumerate(zip(lines, expected, strict=True)):
        fields = line.rsplit(" ", 1)
        if len(fields) != 2 or not fields[1].isdigit():
            raise GenerationError(f"indexed vocab line {expected_id} syntax differs")
        piece, raw_id = fields
        if int(raw_id) != expected_id or piece != expected_piece:
            raise GenerationError(f"indexed vocab line {expected_id} differs")
        pieces.append(piece)
    return pieces


def source_contract(pieces: list[str]) -> dict[str, Any]:
    return {
        "format": "evo-tokenizer-source-v1",
        "kind": "kmer",
        "model": {
            "k": 6,
            "match_special_literals": True,
            "stride": 6,
            "tail": "lookup",
            "unknown_policy": "unk",
        },
        "normalization": [],
        "post_processor": {
            "padding": {"pad_id": 3, "side": "right"},
            "prefix_ids": [1],
            "suffix_ids": [],
        },
        "pre_tokenizer": {"kind": "none"},
        "special_tokens": {
            "bos": 1,
            "cls": None,
            "eos": 2,
            "mask": 4,
            "pad": 3,
            "sep": None,
            "unk": 0,
        },
        "vocab": [
            {"id": token_id, "piece": piece}
            for token_id, piece in enumerate(pieces)
        ],
    }


def compiler_manifest(
    source_payload: bytes, model: ModelSpec
) -> dict[str, Any]:
    return {
        "files": [
            {
                "name": model.source_name,
                "role": "spec",
                "sha256": sha256_bytes(source_payload),
                "size": len(source_payload),
            }
        ],
        "format": "evo-tokenizer-compiler-v1",
        "kind": "kmer",
        "options": {},
        "source": "custom",
    }


def write_atomic(path: Path, payload: bytes, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise GenerationError("output already exists: " + str(path))
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix="." + path.name + ".", dir=path.parent, delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        model = MODEL_SPECS[args.model]
        snapshot = args.snapshot.resolve(strict=True)
        source_receipt = args.source_receipt.resolve(strict=True)
        output = args.output_dir.resolve()
        if snapshot.name != model.source_revision:
            raise GenerationError("snapshot directory does not name pinned revision")
        catalog_contract = validate_source_receipt(source_receipt, snapshot, model)
        if (
            load_object(snapshot / "config.json", "model config")
            != model.expected_config
        ):
            raise GenerationError("pinned model config differs")
        tokenizer_config = load_object(
            snapshot / "tokenizer_config.json", "tokenizer config"
        )
        if tokenizer_config != EXPECTED_TOKENIZER_CONFIG:
            raise GenerationError("pinned tokenizer config/AddedToken flags differ")
        if (
            load_object(snapshot / "special_tokens_map.json", "special token map")
            != EXPECTED_SPECIAL_MAP
        ):
            raise GenerationError("pinned special token map differs")
        declared_special_tokens(snapshot / "tokenizer.py")
        pieces = validate_indexed_vocab(snapshot / "vocab.txt")

        source_payload = canonical_json(source_contract(pieces))
        manifest_payload = canonical_json(compiler_manifest(source_payload, model))
        receipt_payload = canonical_json(
            {
                "files": [
                    {
                        "name": model.source_name,
                        "path": str(output / model.source_name),
                        "role": "spec",
                        "sha256": sha256_bytes(source_payload),
                        "size": len(source_payload),
                    }
                ],
                "kind": "tokenizer-source",
                "schema_version": 1,
            }
        )
        source_manifest = [
            {"name": name, "sha256": digest, "size": size}
            for name, (size, digest) in sorted(model.source_files.items())
        ]
        audit_payload = canonical_json(
            {
                "catalog_contract_sha256": catalog_contract,
                "compiler_manifest": {
                    "name": model.manifest_name,
                    "sha256": sha256_bytes(manifest_payload),
                    "size": len(manifest_payload),
                },
                "format": "geneb-generator-tokenizer-audit-v1",
                "indexed_vocab": {
                    "count": len(pieces),
                    "first_kmer": {"id": 32, "piece": "AAAAAA"},
                    "last_kmer": {"id": 4127, "piece": "GGGGGG"},
                    "order": "32 special literals then itertools.product('ATCG', repeat=6)",
                    "special_literals": list(SPECIAL_TOKENS),
                },
                "official_vectors": [
                    {
                        "input": "<s>ACGTNACGTNAC",
                        "input_ids": [1, 1, 0, 0],
                    },
                    {
                        "input": "<s>AAAAAATTTTTT",
                        "input_ids": [1, 1, 32, 1397],
                    },
                ],
                "runtime_id": model.runtime_id,
                "source": {
                    "file_manifest_sha256": sha256_bytes(
                        canonical_json(source_manifest)
                    ),
                    "files": source_manifest,
                    "repo": model.source_repo,
                    "revision": model.source_revision,
                },
                "tokenizer_source": {
                    "name": model.source_name,
                    "sha256": sha256_bytes(source_payload),
                    "size": len(source_payload),
                },
            }
        )
        outputs = (
            (output / model.source_name, source_payload),
            (output / model.manifest_name, manifest_payload),
            (output / model.receipt_name, receipt_payload),
            (output / model.audit_name, audit_payload),
        )
        for path, payload in outputs:
            write_atomic(path, payload, args.force)
        for path, payload in outputs:
            print(f"{path.name} size={len(payload)} sha256={sha256_bytes(payload)}")
        return 0
    except (GenerationError, OSError, ValueError) as error:
        print("generate_geneb_generator_tokenizer_source: error: " + str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
