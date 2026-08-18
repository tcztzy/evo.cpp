#!/usr/bin/env python3
"""Compile and audit the pinned Omni-DNA-1B tokenizer from two source roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REVISION = "0ea9d54e356b4e7354dc40e2d980e9ebcb2ccfd5"
MANIFEST_SHA256 = "a43c15ea23faa74779065971045d916c3596bce011b29a6d801cfb7e83c231b7"
SOURCE_SIZE = 286_487
SOURCE_SHA256 = "c02de8344eee229f6355e3de63d18fc0d4110f276320774ff891d6a5f81d817b"
ASSET_SIZE = 182_189
ASSET_SHA256 = "371c4385d340f46222e845abde5343768fcac26d432a9d97e369ad96bd2aa781"
DESCRIPTOR_SHA256 = "caeb95afc7991adc06fd48c0981e6fb6c18efc25c0ee670dd7b949281885c9de"
SOURCE_RECEIPT_CONTRACT_SHA256 = (
    "a537d3f1a40725b11e2dbe350cb7febb980304ac48c92cacb8fec0bfda5f1250"
)
SPECIALS = {
    "unk": 0,
    "cls": 1,
    "sep": 2,
    "pad": 3,
    "mask": 4,
    "bos": None,
    "eos": None,
}


class AuditError(ValueError):
    """Raised when tokenizer bytes differ from the frozen contract."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, size: int, digest: str, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size != size:
        raise AuditError(f"{label} size differs")
    if sha256_file(resolved) != digest:
        raise AuditError(f"{label} SHA256 differs")
    return resolved


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"{label} root must be an object")
    return value, payload


def prepare_root(
    root: Path,
    source: Path,
    manifest: Path,
    compiler: Path,
) -> dict[str, Path]:
    root = root.expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise AuditError(f"audit root must be absent or empty: {root}")
    source_root = root / "source"
    artifact_root = root / "artifact"
    source_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    copied = source_root / "tokenizer.json"
    shutil.copyfile(source, copied)
    checked_file(copied, SOURCE_SIZE, SOURCE_SHA256, "copied tokenizer")
    receipt = source_root / "source-receipt.json"
    receipt.write_bytes(
        canonical_json(
            {
                "schema_version": 1,
                "kind": "tokenizer-source",
                "files": [
                    {
                        "role": "tokenizer",
                        "name": "tokenizer.json",
                        "size": SOURCE_SIZE,
                        "sha256": SOURCE_SHA256,
                        "path": str(copied),
                    }
                ],
            }
        )
    )
    asset = artifact_root / "tokenizer.evo.json"
    descriptor = artifact_root / "tokenizer-descriptor.json"
    result = subprocess.run(
        [
            sys.executable,
            str(compiler),
            "--manifest",
            str(manifest),
            "--receipt",
            str(receipt),
            "--output",
            str(asset),
            "--descriptor",
            str(descriptor),
            "--asset-path",
            asset.name,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise AuditError(
            f"tokenizer compiler failed for {root}: {result.stderr.strip()}"
        )
    checked_file(asset, ASSET_SIZE, ASSET_SHA256, "compiled tokenizer")
    checked_file(
        descriptor,
        descriptor.stat().st_size,
        DESCRIPTOR_SHA256,
        "tokenizer descriptor",
    )
    return {
        "root": root,
        "source": copied,
        "receipt": receipt,
        "asset": asset,
        "descriptor": descriptor,
    }


def validate_asset(path: Path) -> dict[str, Any]:
    asset, _ = load_json(path, "compiled tokenizer")
    if set(asset) != {
        "format",
        "kind",
        "model",
        "normalization",
        "post_processor",
        "pre_tokenizer",
        "special_tokens",
        "vocab",
    }:
        raise AuditError("compiled tokenizer fields differ")
    model = asset.get("model")
    vocab = asset.get("vocab")
    if not isinstance(model, dict) or not isinstance(vocab, list):
        raise AuditError("compiled tokenizer model/vocab shape differs")
    if (
        asset["format"] != "evo-tokenizer-v1"
        or asset["kind"] != "bpe"
        or asset["normalization"] != []
        or asset["pre_tokenizer"] != {"kind": "hf-whitespace-ascii"}
        or asset["special_tokens"] != SPECIALS
        or model.get("literal_token_ids") != [0, 1, 2, 3, 4]
        or len(model.get("merges", [])) != 4_087
        or len(vocab) != 4_096
        or vocab[:5]
        != [
            {"id": 0, "piece": "[UNK]"},
            {"id": 1, "piece": "[CLS]"},
            {"id": 2, "piece": "[SEP]"},
            {"id": 3, "piece": "[PAD]"},
            {"id": 4, "piece": "[MASK]"},
        ]
        or asset["post_processor"]
        != {
            "padding": {"pad_id": 3, "side": "right"},
            "prefix_ids": [1],
            "suffix_ids": [2],
        }
    ):
        raise AuditError("compiled tokenizer semantic contract differs")
    return {
        "format": asset["format"],
        "kind": asset["kind"],
        "vocab_size": len(vocab),
        "merge_count": len(model["merges"]),
        "literal_token_ids": model["literal_token_ids"],
        "special_tokens": asset["special_tokens"],
        "post_processor": asset["post_processor"],
    }


def validate_descriptor(path: Path) -> dict[str, Any]:
    descriptor, _ = load_json(path, "tokenizer descriptor")
    expected = {
        "compiler_manifest_sha256": MANIFEST_SHA256,
        "converter.schema": "evo-tokenizer-conversion-receipt",
        "converter.version": 1,
        "source_receipt_contract_sha256": SOURCE_RECEIPT_CONTRACT_SHA256,
        "tokenizer.path": "tokenizer.evo.json",
        "tokenizer.profile": "evo-tokenizer-v1",
        "tokenizer.sha256": ASSET_SHA256,
        "tokenizer.size": ASSET_SIZE,
    }
    if descriptor != expected:
        raise AuditError("tokenizer descriptor contract differs")
    return descriptor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--root-a", required=True, type=Path)
    parser.add_argument("--root-b", required=True, type=Path)
    args = parser.parse_args()

    snapshot = args.snapshot.expanduser().resolve(strict=True)
    if snapshot.name != REVISION:
        raise AuditError("snapshot revision differs")
    source = checked_file(
        snapshot / "tokenizer.json", SOURCE_SIZE, SOURCE_SHA256, "source tokenizer"
    )
    manifest = checked_file(
        args.manifest, 515, MANIFEST_SHA256, "tokenizer compiler manifest"
    )
    compiler = args.compiler.expanduser().resolve(strict=True)
    first = prepare_root(args.root_a, source, manifest, compiler)
    second = prepare_root(args.root_b, source, manifest, compiler)
    if first["root"] == second["root"]:
        raise AuditError("audit roots must differ")
    for key in ("source", "receipt", "asset", "descriptor"):
        if first[key] == second[key]:
            raise AuditError(f"audit {key} paths must differ")
    if first["asset"].read_bytes() != second["asset"].read_bytes():
        raise AuditError("compiled tokenizer bytes differ across roots")
    if first["descriptor"].read_bytes() != second["descriptor"].read_bytes():
        raise AuditError("descriptor bytes differ across roots")
    if first["receipt"].read_bytes() == second["receipt"].read_bytes():
        raise AuditError("source receipt locator bytes unexpectedly match")
    semantics = validate_asset(first["asset"])
    descriptor = validate_descriptor(first["descriptor"])
    report = {
        "schema_version": 1,
        "kind": "geneb-omni-dna-1b-tokenizer-audit",
        "source": {"size": SOURCE_SIZE, "sha256": SOURCE_SHA256},
        "manifest": {"size": 515, "sha256": MANIFEST_SHA256},
        "asset": {
            "size": ASSET_SIZE,
            "sha256": ASSET_SHA256,
            "semantics": semantics,
        },
        "descriptor": {
            "size": first["descriptor"].stat().st_size,
            "sha256": DESCRIPTOR_SHA256,
            "source_receipt_contract_sha256": descriptor[
                "source_receipt_contract_sha256"
            ],
        },
        "cross_root": {
            "source_receipt_bytes_differ": True,
            "asset_bytes_identical": True,
            "descriptor_bytes_identical": True,
        },
    }
    print(canonical_json(report).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError, ValueError) as error:
        print(f"audit_geneb_omni_dna_1b_tokenizer: error: {error}", file=sys.stderr)
        raise SystemExit(2)
