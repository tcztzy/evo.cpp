#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract-test the two pinned GENERator indexed-vocab source profiles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_generator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("geneb_generator_tokenizer", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load GENERator tokenizer source generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_generation_error(generator: Any, operation: Any, fragment: str) -> None:
    try:
        operation()
    except generator.GenerationError as error:
        if fragment not in str(error):
            raise AssertionError(
                f"wrong GENERator tokenizer rejection: {error}"
            ) from error
    else:
        raise AssertionError("corrupt GENERator tokenizer source was accepted")


def require_oracle_error(oracle: Any, operation: Any, fragment: str) -> None:
    try:
        operation()
    except oracle.OracleError as error:
        if fragment not in str(error):
            raise AssertionError(f"wrong 3B oracle rejection: {error}") from error
    else:
        raise AssertionError("corrupt 3B oracle source was accepted")


def receipt_value(model: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "source-checkpoint",
        "model_id": model.runtime_id,
        "repo": model.source_repo,
        "requested_revision": "main",
        "resolved_revision": model.source_revision,
        "source_kind": "huggingface",
        "catalog_path": "configs/geneb-models.json",
        "catalog_contract_sha256": model.catalog_contract_sha256,
        "load_path": None,
        "files": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--oracle-3b", required=True, type=Path)
    parser.add_argument("--manifest-1-2b", required=True, type=Path)
    parser.add_argument("--manifest-3b", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    work = args.work_dir.resolve()
    if work.name != "geneb-generator-tokenizer":
        raise AssertionError("refusing to clean unexpected test directory")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    generator = load_generator(args.generator.resolve(strict=True))
    oracle_3b = load_generator(args.oracle_3b.resolve(strict=True))
    expected_ids = {
        "geneb-generator-eukaryote-1-2b",
        "geneb-generator-eukaryote-3b",
    }
    if set(generator.MODEL_SPECS) != expected_ids:
        raise AssertionError("GENERATOR explicit model selection differs")

    pieces = list(generator.SPECIAL_TOKENS) + [
        "".join(kmer) for kmer in itertools.product("ATCG", repeat=6)
    ]
    vocab = work / "vocab.txt"
    vocab.write_text(
        "".join(f"{piece} {token_id}\n" for token_id, piece in enumerate(pieces)),
        encoding="ascii",
    )
    if generator.validate_indexed_vocab(vocab) != pieces:
        raise AssertionError("valid GENERator indexed vocab changed")

    source = generator.canonical_json(generator.source_contract(pieces))
    if (
        len(source) != 118_955
        or digest(source)
        != "f15a3d60f6f9d2eeceafc27719956e162285a7cd5133205295423ae9b3175bba"
    ):
        raise AssertionError("GENERATOR tokenizer source bytes changed")
    expected_manifests = {
        "geneb-generator-eukaryote-1-2b": (
            args.manifest_1_2b,
            251,
            "6b4b7a42172797d332196aa599cadad002a8df7df969ec58fb74751f1b61ec53",
        ),
        "geneb-generator-eukaryote-3b": (
            args.manifest_3b,
            249,
            "d157aa64db892021262905ffa1baf32719c7e60706cfd133c7379822e2471d3e",
        ),
    }
    for runtime_id, (path, size, sha256) in expected_manifests.items():
        model = generator.MODEL_SPECS[runtime_id]
        manifest = generator.canonical_json(
            generator.compiler_manifest(source, model)
        )
        if len(manifest) != size or digest(manifest) != sha256:
            raise AssertionError(runtime_id + " compiler manifest bytes changed")
        if path.resolve(strict=True).read_bytes() != manifest:
            raise AssertionError(runtime_id + " repo compiler manifest differs")

    corrupted = vocab.read_text(encoding="ascii").splitlines()
    corrupted[32] = "AAAAAA 33"
    bad_id = work / "bad-id.txt"
    bad_id.write_text("\n".join(corrupted) + "\n", encoding="ascii")
    require_generation_error(
        generator,
        lambda: generator.validate_indexed_vocab(bad_id),
        "line 32 differs",
    )
    corrupted = vocab.read_text(encoding="ascii").splitlines()
    corrupted[0] = "<wrong> 0"
    bad_literal = work / "bad-literal.txt"
    bad_literal.write_text("\n".join(corrupted) + "\n", encoding="ascii")
    require_generation_error(
        generator,
        lambda: generator.validate_indexed_vocab(bad_literal),
        "line 0 differs",
    )

    one = generator.MODEL_SPECS["geneb-generator-eukaryote-1-2b"]
    three = generator.MODEL_SPECS["geneb-generator-eukaryote-3b"]
    oracle_snapshot = work / three.source_revision
    oracle_snapshot.mkdir()
    if (
        oracle_3b.RUNTIME_ID != three.runtime_id
        or oracle_3b.SOURCE_REPO != three.source_repo
        or oracle_3b.SOURCE_REVISION != three.source_revision
        or oracle_3b.CATALOG_CONTRACT_SHA256 != three.catalog_contract_sha256
        or oracle_3b.EXPECTED_SOURCE_FILES != three.source_files
        or oracle_3b.EXPECTED_CONFIG != three.expected_config
    ):
        raise AssertionError("3B tokenizer/oracle pinned source contracts differ")
    if tuple(item["input_ids"] for item in oracle_3b.INPUTS) != (
        [1, 1, 0, 0],
        [1, 1, 32, 1397],
    ):
        raise AssertionError("3B frozen official tokenizer vectors differ")
    if (
        len(oracle_3b.LOCKED_PACKAGES) != 26
        or len(set(oracle_3b.LOCKED_PACKAGES)) != 26
        or any(item.startswith("accelerate==") for item in oracle_3b.LOCKED_PACKAGES)
        or oracle_3b.CHECKPOINT_TENSOR_COUNT != 273
        or oracle_3b.CHECKPOINT_LOGICAL_BYTES != 11_993_051_136
        or sum(oracle_3b.CHECKPOINT_SHARD_TENSOR_COUNTS.values()) != 273
        or set(oracle_3b.CHECKPOINT_SHARD_TENSOR_COUNTS)
        != set(oracle_3b.CHECKPOINT_SHARDS)
    ):
        raise AssertionError("3B B132 serial CPU-F32 loader lock differs")
    receipt = work / "receipt.json"
    receipt.write_text(json.dumps(receipt_value(one)), encoding="utf-8")
    require_generation_error(
        generator,
        lambda: generator.validate_source_receipt(receipt, work, three),
        "pinned identity differs",
    )
    receipt.write_text(json.dumps(receipt_value(three)), encoding="utf-8")
    require_generation_error(
        generator,
        lambda: generator.validate_source_receipt(receipt, work, three),
        "exactly 10 files",
    )
    stale = receipt_value(three)
    stale["catalog_contract_sha256"] = "a" * 64
    receipt.write_text(json.dumps(stale), encoding="utf-8")
    require_generation_error(
        generator,
        lambda: generator.validate_source_receipt(receipt, work, three),
        "catalog contract differs",
    )
    receipt.write_text(json.dumps(receipt_value(one)), encoding="utf-8")
    require_oracle_error(
        oracle_3b,
        lambda: oracle_3b.validate_sources(receipt, oracle_snapshot),
        "pinned identity differs",
    )
    receipt.write_text(json.dumps(receipt_value(three)), encoding="utf-8")
    require_oracle_error(
        oracle_3b,
        lambda: oracle_3b.validate_sources(receipt, oracle_snapshot),
        "file count differs",
    )
    require_oracle_error(
        oracle_3b,
        lambda: oracle_3b.portable({"path": "/private/tmp/leak"}, "fixture"),
        "absolute filesystem path",
    )
    print("GENERATOR tokenizer source contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
