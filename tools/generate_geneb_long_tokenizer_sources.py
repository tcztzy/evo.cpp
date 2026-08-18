#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic offline tokenizer compiler inputs for T33."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def payload(value: Dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def write(path: Path, value: Dict[str, Any]) -> bytes:
    data = payload(value)
    path.write_bytes(data)
    return data


def manifest(source_name: str, source_payload: bytes, kind: str) -> Dict[str, Any]:
    return {
        "format": "evo-tokenizer-compiler-v1",
        "source": "custom",
        "kind": kind,
        "files": [
            {
                "role": "spec",
                "name": source_name,
                "size": len(source_payload),
                "sha256": hashlib.sha256(source_payload).hexdigest(),
            }
        ],
        "options": {},
    }


def hyenadna_source() -> Dict[str, Any]:
    pieces = [
        "[CLS]",
        "[SEP]",
        "[BOS]",
        "[MASK]",
        "[PAD]",
        "[RESERVED]",
        "[UNK]",
        "A",
        "C",
        "G",
        "T",
        "N",
    ]
    return {
        "format": "evo-tokenizer-source-v1",
        "kind": "single-nucleotide",
        "normalization": [],
        "pre_tokenizer": {"kind": "none"},
        "model": {"unknown_policy": "unk", "match_special_literals": False},
        "post_processor": {
            "prefix_ids": [],
            "suffix_ids": [],
            "padding": {"side": "left", "pad_id": 4},
        },
        "special_tokens": {
            "unk": 6,
            "pad": 4,
            "bos": 2,
            "eos": 1,
            "cls": 0,
            "sep": 1,
            "mask": 3,
        },
        "vocab": [{"id": index, "piece": piece} for index, piece in enumerate(pieces)],
    }


def evo1_source() -> Dict[str, Any]:
    encoder = [chr(0x100 + index) for index in range(256)]
    vocab = [
        {"id": index, "piece": encoder[index]} for index in range(256)
    ]  # type: List[Dict[str, Any]]
    vocab.extend(
        {"id": index, "piece": "<reserved-%03d>" % index}
        for index in range(256, 512)
    )
    return {
        "format": "evo-tokenizer-source-v1",
        "kind": "byte-bpe",
        "normalization": [],
        "pre_tokenizer": {"kind": "whole-input"},
        "model": {
            "add_prefix_space": False,
            "byte_encoder": encoder,
            "merges": [],
        },
        "post_processor": {
            "prefix_ids": [],
            "suffix_ids": [],
            "padding": {"side": "none", "pad_id": None},
        },
        "special_tokens": {
            "unk": None,
            "pad": None,
            "bos": None,
            "eos": None,
            "cls": None,
            "sep": None,
            "mask": None,
        },
        "vocab": vocab,
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "configs" / "tokenizers"
    root.mkdir(parents=True, exist_ok=True)
    for stem, kind, source in (
        ("geneb-hyenadna-character-v1", "single-nucleotide", hyenadna_source()),
        ("geneb-evo1-byte-v1", "byte-bpe", evo1_source()),
    ):
        source_name = stem.replace("-v1", "-source-v1") + ".json"
        source_payload = write(root / source_name, source)
        manifest_payload = write(
            root / (stem + ".json"), manifest(source_name, source_payload, kind)
        )
        print(
            "%s source_sha256=%s manifest_sha256=%s"
            % (
                stem,
                hashlib.sha256(source_payload).hexdigest(),
                hashlib.sha256(manifest_payload).hexdigest(),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
