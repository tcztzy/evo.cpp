#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit NT-2.5B-MS's bounded vocab and native tokenizer parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from transformers import AutoTokenizer


RUNTIME_ID = "geneb-nt-2-5b-ms"
REPO = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
REQUESTED_REVISION = "main"
REVISION = "b746b125aacd1b0970c05a32fd71ba726754542e"
CATALOG_CONTRACT_SHA256 = (
    "17519bb61cf0e6efd33d456b4afa49f07c1651bb9bb60580af9ab02490b8995f"
)
EXPECTED_FILES = {
    "config.json": (
        707,
        "7d003a6f278562634e87e63a51a1c2c685dc495c35e16cc309e6a5d912954988",
    ),
    "pytorch_model-00001-of-00002.bin": (
        9913283486,
        "9333a3ceaf85e027547e73068476ecb79b1ac4e5bae91abcdae7f7f7619d7092",
    ),
    "pytorch_model-00002-of-00002.bin": (
        278112047,
        "d599e301221170496d93158cf46f8e5872621f31fc16e570bc947b27f9917979",
    ),
    "pytorch_model.bin.index.json": (
        46039,
        "71088b187400fee5eec7f6e866d77d22fcd1dd38b930b1359e22d3fd0e284203",
    ),
    "special_tokens_map.json": (
        101,
        "d6dc30bf018166daab248b0abf7efda6fd1b1e0a2d1bee5b31b23db2ebdaee77",
    ),
    "tokenizer_config.json": (
        129,
        "e516a38aaf16ce0e84ce178e5f23b98f8b9993bea1a0e2166326efb4e8440214",
    ),
    "vocab.txt": (
        28718,
        "f1b544e27897936b50bbd925850fa8a08b421c33bb9c26e3711c140c061d0d4c",
    ),
}
MANIFEST_SHA256 = (
    "a19b30c4284c792be67a79dba336156d0cdebd2b2f4d57ea4d1a7cc62975cb26"
)
ASSET_SHA256 = (
    "f574aad91e382548e0bc6f87fefd0b2022fcfeaf612c7c90be5c133f74a75794"
)
ASSET_SIZE = 118259
SOURCE_RECEIPT_CONTRACT_SHA256 = (
    "3c9bc95e63930dccd733d81e51e75522475127b08ea080468575742882c22662"
)
COMPILED_VOCAB_SIZE = 4105
SOURCE_VOCAB_SIZE = 4107
EXCLUDED_SUFFIX = ((4105, "<eos>"), (4106, "<bos>"))
SAFE_INPUTS = (
    ("input-0", "ATTCCGATTCCGATTCCG"),
    (
        "input-1",
        "AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT",
    ),
    ("single-n", "N"),
    ("single-a", "A"),
)
EXPECTED_EXCLUDED_IDS = {
    "<eos>A": [3, 4105, 4100],
    "<bos>A": [3, 4106, 4100],
}
LARGE_CHECKPOINT_FILES = {
    "pytorch_model-00001-of-00002.bin",
    "pytorch_model-00002-of-00002.bin",
}


class AuditError(RuntimeError):
    """Raised when NT-2.5B tokenizer provenance or behavior drifts."""


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
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError("cannot read %s: %s" % (label, error)) from error
    if not isinstance(value, dict):
        raise AuditError(label + " root must be an object")
    return value


def require_portable(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_portable(key, label + " key")
            require_portable(item, "%s.%s" % (label, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            require_portable(item, "%s[%d]" % (label, index))
        return
    if not isinstance(value, str):
        return
    parsed = urlsplit(value)
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or parsed.scheme.lower() == "file"
    ):
        raise AuditError(label + " contains a local absolute filesystem path")


def reject_excluded_literal(sequence: str) -> None:
    for _, literal in EXCLUDED_SUFFIX:
        if literal in sequence:
            raise AuditError("input contains source-only excluded literal " + literal)


def validate_source_receipt(receipt_path: Path, snapshot: Path) -> None:
    receipt = read_json(receipt_path, "source receipt")
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
        raise AuditError("source receipt fields differ")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != RUNTIME_ID
        or receipt["repo"] != REPO
        or receipt["requested_revision"] != REQUESTED_REVISION
        or receipt["resolved_revision"] != REVISION
        or receipt["source_kind"] != "huggingface"
        or receipt["catalog_contract_sha256"] != CATALOG_CONTRACT_SHA256
        or receipt["load_path"] is not None
    ):
        raise AuditError("source receipt pinned provenance differs")
    raw_files = receipt["files"]
    if not isinstance(raw_files, list) or len(raw_files) != len(EXPECTED_FILES):
        raise AuditError("source receipt file count differs")
    seen: set[str] = set()
    for index, item in enumerate(raw_files):
        label = "source receipt files[%d]" % index
        if not isinstance(item, dict) or set(item) != {
            "name",
            "path",
            "size",
            "sha256",
        }:
            raise AuditError(label + " fields differ")
        name = item["name"]
        if not isinstance(name, str) or name in seen or name not in EXPECTED_FILES:
            raise AuditError("source receipt file set differs")
        seen.add(name)
        size, digest = EXPECTED_FILES[name]
        if item["size"] != size or item["sha256"] != digest:
            raise AuditError("source receipt metadata differs for " + name)
        locator = Path(item["path"])
        source = snapshot / name
        if (
            locator.resolve(strict=True) != source.resolve(strict=True)
            or locator.stat().st_size != size
            or source.stat().st_size != size
        ):
            raise AuditError("source receipt locator differs for " + name)
    if seen != set(EXPECTED_FILES):
        raise AuditError("source receipt file set differs")


def validate_compiled_root(
    root: Path, descriptor_path: Path
) -> tuple[dict[str, Any], Path]:
    descriptor = read_json(descriptor_path, "tokenizer descriptor")
    if descriptor != {
        "compiler_manifest_sha256": MANIFEST_SHA256,
        "converter.schema": "evo-tokenizer-conversion-receipt",
        "converter.version": 1,
        "source_receipt_contract_sha256": SOURCE_RECEIPT_CONTRACT_SHA256,
        "tokenizer.path": "assets/nt-2-5b-ms-tokenizer.evo.json",
        "tokenizer.profile": "evo-tokenizer-v1",
        "tokenizer.sha256": ASSET_SHA256,
        "tokenizer.size": ASSET_SIZE,
    }:
        raise AuditError("compiled tokenizer descriptor differs")
    asset = (root / descriptor["tokenizer.path"]).resolve(strict=True)
    if asset.stat().st_size != ASSET_SIZE or sha256_file(asset) != ASSET_SHA256:
        raise AuditError("compiled tokenizer bytes differ")
    return descriptor, asset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root-a", required=True, type=Path)
    parser.add_argument("--descriptor-a", required=True, type=Path)
    parser.add_argument("--root-b", required=True, type=Path)
    parser.add_argument("--descriptor-b", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve(strict=True)
    receipt = args.receipt.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    root_a = args.root_a.resolve(strict=True)
    root_b = args.root_b.resolve(strict=True)
    descriptor_path_a = args.descriptor_a.resolve(strict=True)
    descriptor_path_b = args.descriptor_b.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    if snapshot.name != REVISION:
        raise AuditError("snapshot directory does not name the pinned revision")
    validate_source_receipt(receipt, snapshot)
    snapshot_files = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if snapshot_files != set(EXPECTED_FILES):
        raise AuditError("bounded snapshot contains missing or extra files")
    for name, (size, digest) in EXPECTED_FILES.items():
        path = snapshot / name
        if path.stat().st_size != size or (
            name not in LARGE_CHECKPOINT_FILES and sha256_file(path) != digest
        ):
            raise AuditError("pinned source file differs: " + name)
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise AuditError("pinned NT-2.5B tokenizer compiler manifest differs")
    if importlib.metadata.version("transformers") != "4.32.0":
        raise AuditError("Transformers 4.32.0 is required")
    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise AuditError(variable + "=1 is required")
    if not os.access(str(runtime), os.X_OK):
        raise AuditError("tokenizer runtime verifier is not executable")

    vocab_payload = (snapshot / "vocab.txt").read_bytes()
    vocab = vocab_payload.decode("ascii").splitlines()
    if (
        len(vocab) != SOURCE_VOCAB_SIZE
        or vocab[:4] != ["<unk>", "<pad>", "<mask>", "<cls>"]
        or vocab[4100:4105] != ["A", "T", "C", "G", "N"]
        or tuple(enumerate(vocab[4105:], start=4105)) != EXCLUDED_SUFFIX
        or vocab_payload.endswith(b"\n")
    ):
        raise AuditError("full pinned NT-2.5B vocabulary boundary differs")

    descriptor_a, asset_a = validate_compiled_root(root_a, descriptor_path_a)
    descriptor_b, asset_b = validate_compiled_root(root_b, descriptor_path_b)
    if asset_a.read_bytes() != asset_b.read_bytes():
        raise AuditError("relocated compiled tokenizer assets differ")
    if descriptor_path_a.read_bytes() != descriptor_path_b.read_bytes():
        raise AuditError("relocated tokenizer descriptors differ")
    compiled = read_json(asset_a, "compiled tokenizer asset")
    compiled_vocab = compiled.get("vocab")
    if (
        not isinstance(compiled_vocab, list)
        or len(compiled_vocab) != COMPILED_VOCAB_SIZE
        or compiled_vocab[-5:]
        != [
            {"id": 4100, "piece": "A"},
            {"id": 4101, "piece": "T"},
            {"id": 4102, "piece": "C"},
            {"id": 4103, "piece": "G"},
            {"id": 4104, "piece": "N"},
        ]
        or any(
            item.get("id", COMPILED_VOCAB_SIZE) >= COMPILED_VOCAB_SIZE
            for item in compiled_vocab
        )
        or any(
            item.get("piece") in {"<eos>", "<bos>"} for item in compiled_vocab
        )
    ):
        raise AuditError("compiled runtime vocabulary boundary differs")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        revision=REVISION,
    )
    tokenizer_class = "%s.%s" % (
        tokenizer.__class__.__module__,
        tokenizer.__class__.__name__,
    )
    if (
        tokenizer_class
        != "transformers.models.esm.tokenization_esm.EsmTokenizer"
        or tokenizer.model_max_length != 1000
        or tokenizer.padding_side != "right"
        or tokenizer.vocab_size != SOURCE_VOCAB_SIZE
        or len(tokenizer) != SOURCE_VOCAB_SIZE
        or tokenizer.all_special_ids != [0, 1, 3, 2]
    ):
        raise AuditError("official NT-2.5B tokenizer contract differs")

    vectors = []
    for label, sequence in SAFE_INPUTS:
        reject_excluded_literal(sequence)
        encoded = tokenizer(sequence, truncation=True)
        ids = [int(value) for value in encoded["input_ids"]]
        if not ids or max(ids) >= COMPILED_VOCAB_SIZE:
            raise AuditError("safe official input emitted an OOB ID")
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
            raise AuditError(
                "native safe-input tokenizer parity failed for %s: %s"
                % (label, result.stderr.decode("utf-8", errors="replace"))
            )
        vectors.append(
            {
                "label": label,
                "input_ascii": sequence,
                "input_sha256": sha256_bytes(sequence.encode("ascii")),
                "input_ids": ids,
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "maximum_id": max(ids),
                "native_exit_code": result.returncode,
                "native_stdout_sha256": sha256_bytes(result.stdout),
                "native_stderr_sha256": sha256_bytes(result.stderr),
            }
        )

    rejected = []
    for sequence, expected_ids in EXPECTED_EXCLUDED_IDS.items():
        ids = [int(value) for value in tokenizer(sequence)["input_ids"]]
        if ids != expected_ids or max(ids) < COMPILED_VOCAB_SIZE:
            raise AuditError("official excluded literal behavior differs")
        try:
            reject_excluded_literal(sequence)
        except AuditError:
            pass
        else:
            raise AuditError("excluded source literal entered the accepted input domain")
        rejected.append(
            {
                "input_ascii": sequence,
                "official_input_ids": ids,
                "offending_ids": [
                    value for value in ids if value >= COMPILED_VOCAB_SIZE
                ],
                "policy": "reject-before-model",
            }
        )

    report = {
        "schema_version": 1,
        "kind": "geneb-nt-2-5b-ms-tokenizer-boundary-audit",
        "runtime_id": RUNTIME_ID,
        "status": "passed",
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "catalog_contract_sha256": CATALOG_CONTRACT_SHA256,
            "files": {
                name: {"size": size, "sha256": digest}
                for name, (size, digest) in sorted(EXPECTED_FILES.items())
            },
            "vocab_size": SOURCE_VOCAB_SIZE,
            "vocab_sha256": EXPECTED_FILES["vocab.txt"][1],
            "final_newline": False,
        },
        "compiler": {
            "manifest_sha256": MANIFEST_SHA256,
            "source_receipt_contract_sha256": SOURCE_RECEIPT_CONTRACT_SHA256,
            "compiled_asset_sha256": ASSET_SHA256,
            "compiled_asset_size": ASSET_SIZE,
            "compiled_vocab_size": COMPILED_VOCAB_SIZE,
            "excluded_suffix": [
                {"id": token_id, "piece": piece, "input_policy": "reject"}
                for token_id, piece in EXCLUDED_SUFFIX
            ],
            "relocated_asset_byte_identical": True,
            "relocated_descriptor_byte_identical": True,
            "descriptor_sha256": sha256_file(descriptor_path_a),
        },
        "official": {
            "transformers_version": importlib.metadata.version("transformers"),
            "tokenizer_class": tokenizer_class,
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side,
            "reported_vocab_size": tokenizer.vocab_size,
        },
        "accepted_vectors": vectors,
        "rejected_source_literals": rejected,
        "generator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
    }
    require_portable(report, "tokenizer audit")
    output = args.output.resolve()
    output.write_bytes(canonical_json(report))
    print(
        json.dumps(
            {
                "output_sha256": sha256_file(output),
                "safe_vectors": len(vectors),
                "rejected_literals": len(rejected),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError, ValueError) as error:
        raise SystemExit("audit_geneb_nt_2_5b_tokenizer: error: %s" % error)
