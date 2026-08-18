#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile verified tokenizer sources into canonical evo-tokenizer-v1 JSON."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


UINT32_MAX = (1 << 32) - 1
MAX_TOKENIZER_ASSET_BYTES = 64 << 20
MAX_LITERAL_BYTES = 4096
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SPECIAL_NAMES = ("unk", "pad", "bos", "eos", "cls", "sep", "mask")
RUNTIME_KINDS = {
    "character",
    "single-nucleotide",
    "kmer",
    "wordpiece",
    "bpe",
    "byte-bpe",
    "longest-match",
    "kmer-bpe",
}


class ConversionError(ValueError):
    """Raised when source assets are outside the closed compiler contract."""


def duplicate_checked_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        if key in result:
            raise ConversionError("JSON object contains duplicate key %r" % key)
        result[key] = value
    return result


def read_json(path: Path, label: str) -> Tuple[Dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=duplicate_checked_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversionError("cannot read %s: %s" % (label, error))
    if not isinstance(value, dict):
        raise ConversionError("%s root must be an object" % label)
    return value, payload


def exact_keys(
    value: Dict[str, Any],
    required: Iterable[str],
    optional: Iterable[str],
    label: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    actual = set(value)
    if actual != required_set | (actual & set(optional)) or not required_set <= actual:
        raise ConversionError(
            "%s fields differ: missing=%s extra=%s"
            % (label, sorted(required_set - actual), sorted(actual - allowed))
        )


def object_value(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversionError("%s must be an object" % label)
    return value


def list_value(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        raise ConversionError("%s must be an array" % label)
    return value


def string_value(value: Any, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ConversionError("%s must be a %sstring" % (label, "nonempty " if not allow_empty else ""))
    if "\x00" in value:
        raise ConversionError("%s must not contain NUL" % label)
    return value


def bool_value(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConversionError("%s must be boolean" % label)
    return value


def uint32_value(value: Any, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConversionError("%s must be a uint32" % label)
    minimum = 1 if positive else 0
    if value < minimum or value > UINT32_MAX:
        raise ConversionError("%s is outside uint32 range" % label)
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ConversionError("cannot hash source asset %s: %s" % (path, error))
    return digest.hexdigest()


def normalized_relative_path(value: Any, label: str) -> str:
    text = string_value(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or str(path) != text or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise ConversionError("%s must be a normalized relative path" % label)
    return text


def validate_file_entry(
    value: Any, label: str, include_path: bool
) -> Dict[str, Any]:
    entry = object_value(value, label)
    required = ["role", "name", "size", "sha256"]
    if include_path:
        required.append("path")
    exact_keys(entry, required, [], label)
    result = {
        "role": string_value(entry["role"], label + ".role"),
        "name": normalized_relative_path(entry["name"], label + ".name"),
        "size": uint32_value(entry["size"], label + ".size"),
        "sha256": string_value(entry["sha256"], label + ".sha256").lower(),
    }
    if not SHA256_RE.fullmatch(result["sha256"]):
        raise ConversionError("%s.sha256 must be 64 lowercase hex digits" % label)
    if include_path:
        result["path"] = string_value(entry["path"], label + ".path")
    return result


def verified_assets(
    manifest_path: Path, receipt_path: Path
) -> Tuple[str, str, Dict[str, Path], bytes, bytes, Dict[str, Any]]:
    manifest, manifest_bytes = read_json(manifest_path, "tokenizer manifest")
    exact_keys(
        manifest,
        ["format", "source", "kind", "files", "options"],
        [],
        "tokenizer manifest",
    )
    if manifest["format"] != "evo-tokenizer-compiler-v1":
        raise ConversionError("tokenizer manifest format is unsupported")
    source_kind = string_value(manifest["source"], "manifest.source")
    output_kind = string_value(manifest["kind"], "manifest.kind")
    allowed = {
        "huggingface-json": {"bpe", "byte-bpe", "wordpiece"},
        "vocab-text": {"kmer", "longest-match", "wordpiece"},
        "custom": {
            "character",
            "single-nucleotide",
            "mixed",
            "biotoken",
            "longest-match",
            "kmer",
            "kmer-bpe",
            "byte-bpe",
        },
    }
    if source_kind not in allowed or output_kind not in allowed[source_kind]:
        raise ConversionError("manifest source/kind combination is unsupported")
    manifest_files = list_value(manifest["files"], "manifest.files")
    expected = {}  # type: Dict[str, Dict[str, Any]]
    names = set()  # type: Set[str]
    for index, raw in enumerate(manifest_files):
        entry = validate_file_entry(raw, "manifest.files[%d]" % index, False)
        role = entry["role"]
        if role in expected or entry["name"] in names:
            raise ConversionError("manifest assets contain duplicate role or name")
        expected[role] = entry
        names.add(entry["name"])
    role_sets = {
        "huggingface-json": (
            {"tokenizer"},
            {"tokenizer_config", "special_tokens_map"},
        ),
        "vocab-text": ({"vocab"}, set()),
        "custom": ({"spec"}, set()),
    }
    required_roles, optional_roles = role_sets[source_kind]
    if not required_roles <= set(expected) or not set(expected) <= (
        required_roles | optional_roles
    ):
        raise ConversionError(
            "manifest asset roles differ: missing=%s extra=%s"
            % (
                sorted(required_roles - set(expected)),
                sorted(set(expected) - required_roles - optional_roles),
            )
        )

    receipt, receipt_bytes = read_json(receipt_path, "tokenizer source receipt")
    exact_keys(receipt, ["schema_version", "kind", "files"], [], "source receipt")
    if receipt["schema_version"] != 1 or receipt["kind"] != "tokenizer-source":
        raise ConversionError("source receipt schema/kind is unsupported")
    actual = {}  # type: Dict[str, Dict[str, Any]]
    receipt_names = set()  # type: Set[str]
    for index, raw in enumerate(list_value(receipt["files"], "receipt.files")):
        entry = validate_file_entry(raw, "receipt.files[%d]" % index, True)
        role = entry["role"]
        if role in actual or entry["name"] in receipt_names:
            raise ConversionError("receipt assets contain duplicate role or name")
        actual[role] = entry
        receipt_names.add(entry["name"])
    if set(actual) != set(expected):
        raise ConversionError("receipt has missing or extra tokenizer assets")
    paths = {}  # type: Dict[str, Path]
    for role in sorted(expected):
        registered = expected[role]
        received = actual[role]
        for key in ("name", "size", "sha256"):
            if received[key] != registered[key]:
                raise ConversionError("receipt %s differs for role %s" % (key, role))
        path = Path(received["path"])
        if not path.is_absolute():
            path = receipt_path.parent / path
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ConversionError("source asset %s is missing: %s" % (role, error))
        if size != registered["size"] or sha256_file(path) != registered["sha256"]:
            raise ConversionError("source asset hash/size mismatch for role %s" % role)
        paths[role] = path
    portable_receipt = {
        "schema_version": receipt["schema_version"],
        "kind": receipt["kind"],
        "files": [
            {
                "role": entry["role"],
                "name": entry["name"],
                "size": entry["size"],
                "sha256": entry["sha256"],
            }
            for entry in sorted(
                actual.values(), key=lambda value: (value["role"], value["name"])
            )
        ],
    }
    return (
        source_kind,
        output_kind,
        paths,
        manifest_bytes,
        canonical_json(portable_receipt),
        manifest,
    )


def validate_normalization(value: Any) -> List[Dict[str, Any]]:
    result = []  # type: List[Dict[str, Any]]
    raw_operations = list_value(value, "normalization")
    if len(raw_operations) > 32:
        raise ConversionError("normalization has more than 32 operations")
    for index, raw in enumerate(raw_operations):
        item = object_value(raw, "normalization[%d]" % index)
        op = string_value(item.get("op"), "normalization[%d].op" % index)
        if op in (
            "ascii-uppercase",
            "ascii-lowercase",
            "u-to-t",
            "strip-ascii-whitespace",
        ):
            exact_keys(item, ["op"], [], "normalization[%d]" % index)
            result.append({"op": op})
        elif op == "prepend-literal":
            exact_keys(item, ["op", "value"], [], "normalization[%d]" % index)
            result.append(
                {"op": op, "value": string_value(item["value"], "prepend value")}
            )
        elif op == "replace-literal":
            exact_keys(
                item, ["op", "from", "to"], [], "normalization[%d]" % index
            )
            source = string_value(item["from"], "replace from")
            replacement = string_value(item["to"], "replace to", True)
            result.append({"op": op, "from": source, "to": replacement})
        elif op == "replace-byte-run":
            exact_keys(
                item,
                ["op", "byte", "min_count", "replacement"],
                [],
                "normalization[%d]" % index,
            )
            byte = string_value(item["byte"], "replace byte")
            try:
                encoded_byte = byte.encode("ascii", "strict")
            except UnicodeEncodeError:
                raise ConversionError("replace-byte-run byte must be ASCII")
            if len(encoded_byte) != 1:
                raise ConversionError("replace-byte-run byte must be one ASCII byte")
            result.append(
                {
                    "op": op,
                    "byte": byte,
                    "min_count": uint32_value(
                        item["min_count"], "replace min_count", True
                    ),
                    "replacement": string_value(
                        item["replacement"], "replace replacement", True
                    ),
                }
            )
        else:
            raise ConversionError("normalization operation %r is unsupported" % op)
        for field in ("value", "from", "to", "replacement"):
            if field in result[-1] and len(result[-1][field].encode("utf-8")) > MAX_LITERAL_BYTES:
                raise ConversionError("normalization literal exceeds 4096 bytes")
    return result


def validate_pre_tokenizer(value: Any) -> Dict[str, Any]:
    item = object_value(value, "pre_tokenizer")
    kind = string_value(item.get("kind"), "pre_tokenizer.kind")
    if kind in (
        "none",
        "whole-input",
        "ascii-whitespace",
        "hf-whitespace-ascii",
    ):
        exact_keys(item, ["kind"], [], "pre_tokenizer")
        return {"kind": kind}
    if kind == "split-isolated":
        exact_keys(item, ["kind", "literal"], [], "pre_tokenizer")
        result = {
            "kind": kind,
            "literal": string_value(item["literal"], "pre_tokenizer.literal"),
        }
        if len(result["literal"].encode("utf-8")) > MAX_LITERAL_BYTES:
            raise ConversionError("pre_tokenizer literal exceeds 4096 bytes")
        return result
    raise ConversionError("pre_tokenizer kind %r is unsupported" % kind)


def validate_vocab(value: Any) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    entries = []  # type: List[Dict[str, Any]]
    ids = set()  # type: Set[int]
    pieces = set()  # type: Set[str]
    by_piece = {}  # type: Dict[str, int]
    for index, raw in enumerate(list_value(value, "vocab")):
        item = object_value(raw, "vocab[%d]" % index)
        exact_keys(item, ["id", "piece"], [], "vocab[%d]" % index)
        token_id = uint32_value(item["id"], "vocab[%d].id" % index)
        piece = string_value(item["piece"], "vocab[%d].piece" % index)
        if token_id in ids or piece in pieces:
            raise ConversionError("vocab contains duplicate IDs or pieces")
        ids.add(token_id)
        pieces.add(piece)
        by_piece[piece] = token_id
        entries.append({"id": token_id, "piece": piece})
    entries.sort(key=lambda entry: entry["id"])
    if not entries or [entry["id"] for entry in entries] != list(range(len(entries))):
        raise ConversionError("vocab IDs must be contiguous from zero")
    return entries, by_piece


def validate_special_tokens(
    value: Any, vocab_ids: Set[int]
) -> Dict[str, Optional[int]]:
    item = object_value(value, "special_tokens")
    exact_keys(item, SPECIAL_NAMES, [], "special_tokens")
    result = {}  # type: Dict[str, Optional[int]]
    for name in SPECIAL_NAMES:
        raw = item[name]
        if raw is None:
            result[name] = None
            continue
        token_id = uint32_value(raw, "special_tokens.%s" % name)
        if token_id not in vocab_ids:
            raise ConversionError("special token %s is outside vocab" % name)
        result[name] = token_id
    return result


def validate_post_processor(
    value: Any, vocab_ids: Set[int]
) -> Dict[str, Any]:
    item = object_value(value, "post_processor")
    exact_keys(item, ["prefix_ids", "suffix_ids", "padding"], [], "post_processor")
    result = {}  # type: Dict[str, Any]
    for field in ("prefix_ids", "suffix_ids"):
        values = []  # type: List[int]
        for index, raw in enumerate(list_value(item[field], "post_processor." + field)):
            token_id = uint32_value(raw, "%s[%d]" % (field, index))
            if token_id not in vocab_ids:
                raise ConversionError("post-processor token is outside vocab")
            values.append(token_id)
        if len(values) > 32:
            raise ConversionError("post-processor ID array exceeds 32 entries")
        result[field] = values
    padding = object_value(item["padding"], "post_processor.padding")
    exact_keys(padding, ["side", "pad_id"], [], "post_processor.padding")
    side = string_value(padding["side"], "padding.side")
    if side not in ("none", "left", "right"):
        raise ConversionError("padding.side is unsupported")
    pad_id = padding["pad_id"]
    if pad_id is not None:
        pad_id = uint32_value(pad_id, "padding.pad_id")
        if pad_id not in vocab_ids:
            raise ConversionError("padding pad_id is outside vocab")
    if side != "none" and pad_id is None:
        raise ConversionError("enabled padding requires pad_id")
    if side == "none" and pad_id is not None:
        raise ConversionError("padding side none requires pad_id=null")
    result["padding"] = {"side": side, "pad_id": pad_id}
    return result


def validate_merges(value: Any, pieces: Set[str], label: str) -> List[List[str]]:
    result = []  # type: List[List[str]]
    seen = set()  # type: Set[Tuple[str, str]]
    for index, raw in enumerate(list_value(value, label)):
        pair = list_value(raw, "%s[%d]" % (label, index))
        if len(pair) != 2:
            raise ConversionError("%s[%d] must contain two pieces" % (label, index))
        left = string_value(pair[0], "%s[%d][0]" % (label, index))
        right = string_value(pair[1], "%s[%d][1]" % (label, index))
        key = (left, right)
        if key in seen:
            raise ConversionError("%s contains a duplicate merge" % label)
        if left not in pieces or right not in pieces or left + right not in pieces:
            raise ConversionError("%s merge references an unknown piece" % label)
        seen.add(key)
        result.append([left, right])
    return result


def validate_model(
    kind: str,
    value: Any,
    pieces: Set[str],
    special_tokens: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    item = object_value(value, "model")
    if kind in ("character", "single-nucleotide", "longest-match"):
        exact_keys(item, ["unknown_policy", "match_special_literals"], [], "model")
        policy = string_value(item["unknown_policy"], "model.unknown_policy")
        if policy not in ("error", "unk"):
            raise ConversionError("model unknown_policy is unsupported")
        result = {
            "unknown_policy": policy,
            "match_special_literals": bool_value(
                item["match_special_literals"], "model.match_special_literals"
            ),
        }
    elif kind == "kmer":
        exact_keys(
            item,
            ["k", "stride", "tail", "unknown_policy"],
            ["match_special_literals"],
            "model",
        )
        tail = string_value(item["tail"], "model.tail")
        policy = string_value(item["unknown_policy"], "model.unknown_policy")
        if tail not in ("drop", "error", "unk", "lookup") or policy not in ("error", "unk"):
            raise ConversionError("kmer tail/unknown policy is unsupported")
        result = {
            "k": uint32_value(item["k"], "model.k", True),
            "stride": uint32_value(item["stride"], "model.stride", True),
            "tail": tail,
            "unknown_policy": policy,
        }
        if "match_special_literals" in item:
            result["match_special_literals"] = bool_value(
                item["match_special_literals"], "model.match_special_literals"
            )
        if result["k"] > 4096 or result["stride"] > result["k"]:
            raise ConversionError("kmer k/stride exceeds runtime bounds")
    elif kind == "wordpiece":
        exact_keys(
            item,
            ["continuation_prefix", "max_input_chars_per_word"],
            [],
            "model",
        )
        result = {
            "continuation_prefix": string_value(
                item["continuation_prefix"], "model.continuation_prefix"
            ),
            "max_input_chars_per_word": uint32_value(
                item["max_input_chars_per_word"],
                "model.max_input_chars_per_word",
                True,
            ),
        }
        if result["max_input_chars_per_word"] > (1 << 20):
            raise ConversionError("wordpiece maximum word bytes exceeds runtime bound")
    elif kind == "bpe":
        exact_keys(item, ["merges"], ["literal_token_ids"], "model")
        result = {"merges": validate_merges(item["merges"], pieces, "model.merges")}
        if "literal_token_ids" in item:
            raw_ids = list_value(item["literal_token_ids"], "model.literal_token_ids")
            if not raw_ids or len(raw_ids) > 32:
                raise ConversionError(
                    "model.literal_token_ids must contain 1..32 IDs"
                )
            literal_ids = [
                uint32_value(raw, "model.literal_token_ids") for raw in raw_ids
            ]
            if len(set(literal_ids)) != len(literal_ids) or any(
                token_id >= len(pieces) for token_id in literal_ids
            ):
                raise ConversionError(
                    "model.literal_token_ids contains duplicate/out-of-vocab IDs"
                )
            result["literal_token_ids"] = literal_ids
    elif kind == "byte-bpe":
        exact_keys(item, ["add_prefix_space", "byte_encoder", "merges"], [], "model")
        encoder = list_value(item["byte_encoder"], "model.byte_encoder")
        if len(encoder) != 256:
            raise ConversionError("byte_encoder must contain 256 entries")
        encoded = []  # type: List[str]
        for index, raw in enumerate(encoder):
            character = string_value(raw, "byte_encoder[%d]" % index)
            if len(character) != 1:
                raise ConversionError("byte_encoder entries must be one Unicode scalar")
            encoded.append(character)
        if len(set(encoded)) != 256:
            raise ConversionError("byte_encoder entries must be unique")
        if any(character not in pieces for character in encoded):
            raise ConversionError("byte_encoder entries must all exist in vocab")
        result = {
            "add_prefix_space": bool_value(
                item["add_prefix_space"], "model.add_prefix_space"
            ),
            "byte_encoder": encoded,
            "merges": validate_merges(item["merges"], pieces, "model.merges"),
        }
    elif kind == "kmer-bpe":
        exact_keys(
            item,
            ["k", "stride", "tail", "unknown_policy", "merges"],
            [],
            "model",
        )
        tail = string_value(item["tail"], "model.tail")
        policy = string_value(item["unknown_policy"], "model.unknown_policy")
        if tail not in ("drop", "error", "unk") or policy not in ("error", "unk"):
            raise ConversionError("kmer-bpe tail/unknown policy is unsupported")
        result = {
            "k": uint32_value(item["k"], "model.k", True),
            "stride": uint32_value(item["stride"], "model.stride", True),
            "tail": tail,
            "unknown_policy": policy,
            "merges": validate_merges(item["merges"], pieces, "model.merges"),
        }
        if result["k"] > 4096 or result["stride"] > result["k"]:
            raise ConversionError("kmer-bpe k/stride exceeds runtime bounds")
    else:
        raise ConversionError("runtime tokenizer kind is unsupported")
    if result.get("unknown_policy") == "unk" and special_tokens["unk"] is None:
        raise ConversionError("unknown_policy=unk requires an unk special token")
    if result.get("tail") == "unk" and special_tokens["unk"] is None:
        raise ConversionError("tail=unk requires an unk special token")
    return result


def validate_runtime_asset(value: Any) -> Dict[str, Any]:
    root = object_value(value, "runtime tokenizer")
    exact_keys(
        root,
        [
            "format",
            "kind",
            "normalization",
            "pre_tokenizer",
            "model",
            "post_processor",
            "special_tokens",
            "vocab",
        ],
        [],
        "runtime tokenizer",
    )
    if root["format"] != "evo-tokenizer-v1":
        raise ConversionError("runtime tokenizer format is unsupported")
    kind = string_value(root["kind"], "runtime tokenizer kind")
    if kind not in RUNTIME_KINDS:
        raise ConversionError("runtime tokenizer kind is unsupported")
    vocab, by_piece = validate_vocab(root["vocab"])
    vocab_ids = set(entry["id"] for entry in vocab)
    special_tokens = validate_special_tokens(root["special_tokens"], vocab_ids)
    pre_tokenizer = validate_pre_tokenizer(root["pre_tokenizer"])
    allowed_pre = {
        "character": {"none"},
        "single-nucleotide": {"none"},
        "kmer": {"none"},
        "wordpiece": {"ascii-whitespace", "hf-whitespace-ascii"},
        "bpe": {
            "whole-input",
            "ascii-whitespace",
            "hf-whitespace-ascii",
            "split-isolated",
        },
        "byte-bpe": {"whole-input"},
        "longest-match": {"none", "whole-input"},
        "kmer-bpe": {"none"},
    }
    if pre_tokenizer["kind"] not in allowed_pre[kind]:
        raise ConversionError("pre_tokenizer is incompatible with tokenizer kind")
    model = validate_model(kind, root["model"], set(by_piece), special_tokens)
    post_processor = validate_post_processor(root["post_processor"], vocab_ids)
    if (
        post_processor["padding"]["pad_id"] is not None
        and special_tokens["pad"] != post_processor["padding"]["pad_id"]
    ):
        raise ConversionError("padding pad_id differs from special_tokens.pad")
    return {
        "format": "evo-tokenizer-v1",
        "kind": kind,
        "normalization": validate_normalization(root["normalization"]),
        "pre_tokenizer": pre_tokenizer,
        "model": model,
        "post_processor": post_processor,
        "special_tokens": special_tokens,
        "vocab": vocab,
    }


def hf_vocab(
    model: Dict[str, Any],
    added_tokens: Any,
    ignore_normalized_base_aliases: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[int], List[int], List[int]]:
    raw_vocab = object_value(model.get("vocab"), "HF model.vocab")
    by_piece = {}  # type: Dict[str, int]
    by_id = {}  # type: Dict[int, str]
    for piece, raw_id in raw_vocab.items():
        token_id = uint32_value(raw_id, "HF vocab ID")
        string_value(piece, "HF vocab piece")
        if token_id in by_id:
            raise ConversionError("HF vocab contains duplicate IDs")
        by_piece[piece] = token_id
        by_id[token_id] = piece
    if not by_id or sorted(by_id) != list(range(len(by_id))):
        raise ConversionError("HF model vocabulary IDs must be contiguous from zero")
    seen_added_ids = set()  # type: Set[int]
    seen_added_pieces = set()  # type: Set[str]
    literal_ids = []  # type: List[int]
    appended_ids = []  # type: List[int]
    ignored_normalized_alias_ids = []  # type: List[int]
    for index, raw in enumerate(list_value(added_tokens, "HF added_tokens")):
        item = object_value(raw, "HF added_tokens[%d]" % index)
        exact_keys(
            item,
            ["id", "content", "single_word", "lstrip", "rstrip", "normalized", "special"],
            [],
            "HF added token",
        )
        token_id = uint32_value(item["id"], "HF added token ID")
        piece = string_value(item["content"], "HF added token content")
        for field in ("single_word", "lstrip", "rstrip", "normalized", "special"):
            bool_value(item[field], "HF added token " + field)
        if item["single_word"] or item["lstrip"] or item["rstrip"] or not item["special"]:
            raise ConversionError(
                "HF added token behavior is not representable by evo-tokenizer-v1"
            )
        if token_id in seen_added_ids or piece in seen_added_pieces:
            raise ConversionError("HF added_tokens contains duplicates")
        seen_added_ids.add(token_id)
        seen_added_pieces.add(piece)
        existing_piece = by_id.get(token_id)
        existing_id = by_piece.get(piece)
        if item["normalized"]:
            if (
                not ignore_normalized_base_aliases
                or existing_piece != piece
                or existing_id != token_id
            ):
                raise ConversionError(
                    "ignored normalized HF added tokens must be exact base-vocab aliases"
                )
            ignored_normalized_alias_ids.append(token_id)
            continue
        if existing_piece == piece and existing_id == token_id:
            literal_ids.append(token_id)
            continue
        if existing_piece is not None or existing_id is not None:
            raise ConversionError(
                "HF added token collides with model vocabulary"
            )
        if token_id != len(by_id):
            raise ConversionError(
                "HF appended added-token IDs must be contiguous after model vocab"
            )
        by_id[token_id] = piece
        by_piece[piece] = token_id
        literal_ids.append(token_id)
        appended_ids.append(token_id)
    entries = [{"id": token_id, "piece": piece} for token_id, piece in by_id.items()]
    entries.sort(key=lambda entry: entry["id"])
    if not entries or [entry["id"] for entry in entries] != list(range(len(entries))):
        raise ConversionError("HF vocabulary IDs must be contiguous from zero")
    return entries, by_piece, literal_ids, appended_ids, ignored_normalized_alias_ids


def hf_normalization(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    item = object_value(value, "HF normalizer")
    kind = string_value(item.get("type"), "HF normalizer.type")
    if kind == "Sequence":
        exact_keys(item, ["type", "normalizers"], [], "HF Sequence normalizer")
        result = []  # type: List[Dict[str, Any]]
        for child in list_value(item["normalizers"], "HF Sequence normalizers"):
            result.extend(hf_normalization(child))
        return result
    if kind in ("Lowercase", "Uppercase"):
        exact_keys(item, ["type"], [], "HF case normalizer")
        return [{"op": "ascii-lowercase" if kind == "Lowercase" else "ascii-uppercase"}]
    if kind == "Prepend":
        exact_keys(item, ["type", "prepend"], [], "HF Prepend normalizer")
        return [{"op": "prepend-literal", "value": string_value(item["prepend"], "HF prepend")}]
    if kind == "Strip":
        exact_keys(item, ["type", "strip_left", "strip_right"], [], "HF Strip normalizer")
        if not bool_value(item["strip_left"], "HF strip_left") or not bool_value(
            item["strip_right"], "HF strip_right"
        ):
            raise ConversionError("one-sided HF Strip normalization is unsupported")
        return [{"op": "strip-ascii-whitespace"}]
    if kind == "Replace":
        exact_keys(item, ["type", "pattern", "content"], [], "HF Replace normalizer")
        pattern = object_value(item["pattern"], "HF Replace pattern")
        replacement = string_value(item["content"], "HF Replace content", True)
        if set(pattern) == {"String"}:
            return [
                {
                    "op": "replace-literal",
                    "from": string_value(pattern["String"], "HF Replace literal"),
                    "to": replacement,
                }
            ]
        if set(pattern) == {"Regex"}:
            expression = string_value(pattern["Regex"], "HF Replace regex")
            if expression == "^":
                return [{"op": "prepend-literal", "value": replacement}]
            match = re.fullmatch(r"(.)\+", expression)
            minimum = 1
            if match is None:
                bounded = re.fullmatch(r"(.)\{([1-9][0-9]*),\}", expression)
                if bounded is not None:
                    match = bounded
                    minimum = int(bounded.group(2))
            if match is None:
                raise ConversionError("HF Replace regex is outside byte-run subset")
            byte = match.group(1)
            try:
                if len(byte.encode("ascii")) != 1:
                    raise UnicodeEncodeError("ascii", byte, 0, len(byte), "not ASCII")
            except UnicodeEncodeError:
                raise ConversionError("HF Replace regex byte must be ASCII")
            return [
                {
                    "op": "replace-byte-run",
                    "byte": byte,
                    "min_count": minimum,
                    "replacement": replacement,
                }
            ]
        raise ConversionError("HF Replace pattern is unsupported")
    if kind == "BertNormalizer":
        exact_keys(
            item,
            ["type", "clean_text", "handle_chinese_chars", "strip_accents", "lowercase"],
            [],
            "HF BertNormalizer",
        )
        if (
            bool_value(item["clean_text"], "HF clean_text")
            or bool_value(item["handle_chinese_chars"], "HF handle_chinese_chars")
            or item["strip_accents"] not in (None, False)
        ):
            raise ConversionError("HF BertNormalizer uses unsupported Unicode transforms")
        return [{"op": "ascii-lowercase"}] if bool_value(
            item["lowercase"], "HF lowercase"
        ) else []
    raise ConversionError("HF normalizer construct %r is unsupported" % kind)


def hf_pre_tokenizer(value: Any, kind: str) -> Tuple[Dict[str, Any], bool]:
    if value is None:
        return {"kind": "whole-input"}, False
    item = object_value(value, "HF pre_tokenizer")
    construct = string_value(item.get("type"), "HF pre_tokenizer.type")
    if construct == "Whitespace":
        exact_keys(item, ["type"], [], "HF Whitespace pre_tokenizer")
        # tokenizers::Whitespace uses Unicode regex word/punctuation classes.
        # The portable runtime preserves its complete ASCII subset and rejects
        # non-ASCII input instead of silently applying different boundaries.
        return {"kind": "hf-whitespace-ascii"}, False
    if construct == "Split":
        exact_keys(item, ["type", "pattern", "behavior", "invert"], [], "HF Split pre_tokenizer")
        pattern = object_value(item["pattern"], "HF Split pattern")
        if (
            set(pattern) != {"String"}
            or item["behavior"] != "Isolated"
            or bool_value(item["invert"], "HF Split invert")
        ):
            raise ConversionError("only HF Split(String, Isolated, invert=false) is supported")
        return {
            "kind": "split-isolated",
            "literal": string_value(pattern["String"], "HF Split literal"),
        }, False
    if construct == "ByteLevel":
        exact_keys(
            item,
            ["type", "add_prefix_space", "trim_offsets", "use_regex"],
            [],
            "HF ByteLevel pre_tokenizer",
        )
        if kind != "byte-bpe" or bool_value(
            item["use_regex"], "HF ByteLevel use_regex"
        ):
            raise ConversionError(
                "HF ByteLevel is supported only for byte-bpe with use_regex=false"
            )
        bool_value(item["trim_offsets"], "HF ByteLevel trim_offsets")
        return {"kind": "whole-input"}, bool_value(
            item["add_prefix_space"], "HF ByteLevel add_prefix_space"
        )
    raise ConversionError("HF pre_tokenizer construct %r is unsupported" % construct)


def hf_merges(value: Any) -> List[List[str]]:
    result = []  # type: List[List[str]]
    for index, raw in enumerate(list_value(value, "HF BPE merges")):
        if isinstance(raw, str):
            pair = raw.split(" ")
        else:
            pair = list_value(raw, "HF merge[%d]" % index)
        if len(pair) != 2:
            raise ConversionError("HF BPE merge must contain exactly two pieces")
        result.append(
            [
                string_value(pair[0], "HF merge left"),
                string_value(pair[1], "HF merge right"),
            ]
        )
    return result


def special_piece(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return string_value(value, label)
    item = object_value(value, label)
    exact_keys(
        item,
        ["content", "lstrip", "normalized", "rstrip", "single_word"],
        ["special"],
        label,
    )
    for field in ("lstrip", "normalized", "rstrip", "single_word"):
        bool_value(item[field], label + "." + field)
    if "special" in item:
        bool_value(item["special"], label + ".special")
    return string_value(item["content"], label + ".content")


def config_specials(path: Path, role: str) -> Dict[str, Optional[str]]:
    config, _ = read_json(path, role)
    result = {}  # type: Dict[str, Optional[str]]
    allowed_keys = set(name + "_token" for name in SPECIAL_NAMES)
    if role == "special_tokens_map" and not set(config) <= allowed_keys:
        raise ConversionError("special_tokens_map contains unknown fields")
    for name in SPECIAL_NAMES:
        key = name + "_token"
        if key in config:
            result[name] = special_piece(config[key], role + "." + key)
    return result


def hf_template_post_processor(
    item: Dict[str, Any], by_piece: Dict[str, int]
) -> Tuple[List[int], List[int]]:
    exact_keys(item, ["type", "single", "pair", "special_tokens"], [], "HF TemplateProcessing")
    definitions = object_value(item["special_tokens"], "HF template special_tokens")
    special_ids = {}  # type: Dict[str, int]
    for name, raw in definitions.items():
        descriptor = object_value(raw, "HF template special token")
        exact_keys(descriptor, ["id", "ids", "tokens"], [], "HF template special token")
        if descriptor["id"] != name:
            raise ConversionError("HF template special token key/id differ")
        ids = list_value(descriptor["ids"], "HF template special IDs")
        tokens = list_value(descriptor["tokens"], "HF template special tokens")
        if len(ids) != 1 or len(tokens) != 1 or tokens[0] != name:
            raise ConversionError("multi-token HF template specials are unsupported")
        token_id = uint32_value(ids[0], "HF template special ID")
        if by_piece.get(name) != token_id:
            raise ConversionError("HF template special differs from vocab")
        special_ids[name] = token_id
    single = list_value(item["single"], "HF template single")
    prefix = []  # type: List[int]
    suffix = []  # type: List[int]
    sequence_seen = False
    for raw in single:
        part = object_value(raw, "HF template single part")
        if set(part) == {"Sequence"}:
            sequence = object_value(part["Sequence"], "HF template Sequence")
            exact_keys(sequence, ["id", "type_id"], [], "HF template Sequence")
            if sequence_seen or sequence["id"] != "A" or uint32_value(
                sequence["type_id"], "HF template type_id"
            ) != 0:
                raise ConversionError("HF single template must contain one type-0 sequence A")
            sequence_seen = True
        elif set(part) == {"SpecialToken"}:
            special = object_value(part["SpecialToken"], "HF template SpecialToken")
            exact_keys(special, ["id", "type_id"], [], "HF template SpecialToken")
            name = string_value(special["id"], "HF template special id")
            if uint32_value(special["type_id"], "HF template type_id") != 0 or name not in special_ids:
                raise ConversionError("HF template references an invalid special token")
            (suffix if sequence_seen else prefix).append(special_ids[name])
        else:
            raise ConversionError("HF template single part is unsupported")
    if not sequence_seen:
        raise ConversionError("HF single template omits sequence A")
    pair = list_value(item["pair"], "HF template pair")
    if pair:
        pair_sequences = []  # type: List[str]
        for raw in pair:
            part = object_value(raw, "HF template pair part")
            if set(part) == {"Sequence"}:
                sequence = object_value(part["Sequence"], "HF pair Sequence")
                exact_keys(sequence, ["id", "type_id"], [], "HF pair Sequence")
                pair_sequences.append(string_value(sequence["id"], "HF pair sequence id"))
                uint32_value(sequence["type_id"], "HF pair type_id")
            elif set(part) == {"SpecialToken"}:
                special = object_value(part["SpecialToken"], "HF pair SpecialToken")
                exact_keys(special, ["id", "type_id"], [], "HF pair SpecialToken")
                if special["id"] not in special_ids:
                    raise ConversionError("HF pair template references unknown special")
                uint32_value(special["type_id"], "HF pair type_id")
            else:
                raise ConversionError("HF template pair part is unsupported")
        if sorted(pair_sequences) != ["A", "B"]:
            raise ConversionError("HF pair template must contain sequences A and B")
    return prefix, suffix


def hf_post_processor(
    value: Any, by_piece: Dict[str, int]
) -> Tuple[List[int], List[int]]:
    if value is None:
        return [], []
    item = object_value(value, "HF post_processor")
    kind = string_value(item.get("type"), "HF post_processor.type")
    if kind == "TemplateProcessing":
        return hf_template_post_processor(item, by_piece)
    if kind == "BertProcessing":
        exact_keys(item, ["type", "sep", "cls"], [], "HF BertProcessing")
        values = {}  # type: Dict[str, int]
        for name in ("sep", "cls"):
            pair = list_value(item[name], "HF BertProcessing " + name)
            if len(pair) != 2:
                raise ConversionError("HF BertProcessing token must be [piece,id]")
            piece = string_value(pair[0], "HF BertProcessing piece")
            token_id = uint32_value(pair[1], "HF BertProcessing ID")
            if by_piece.get(piece) != token_id:
                raise ConversionError("HF BertProcessing token differs from vocab")
            values[name] = token_id
        return [values["cls"]], [values["sep"]]
    raise ConversionError("HF post_processor construct %r is unsupported" % kind)


def validate_hf_decoder(
    value: Any, kind: str, ignore_omnina_encode_inert_fields: bool
) -> None:
    if ignore_omnina_encode_inert_fields:
        expected = {
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
        if kind != "bpe" or value != expected:
            raise ConversionError(
                "ignored OmniNA decoder must match the exact encode-inert Sequence"
            )
        return
    if value is None:
        return
    item = object_value(value, "HF decoder")
    construct = string_value(item.get("type"), "HF decoder.type")
    if construct in ("BPE", "BPEDecoder") and kind == "bpe":
        exact_keys(item, ["type", "suffix"], [], "HF BPEDecoder")
        string_value(item["suffix"], "HF BPEDecoder suffix")
        return
    if construct == "WordPiece" and kind == "wordpiece":
        exact_keys(item, ["type", "prefix", "cleanup"], [], "HF WordPiece decoder")
        string_value(item["prefix"], "HF WordPiece decoder prefix")
        bool_value(item["cleanup"], "HF WordPiece decoder cleanup")
        return
    if construct == "ByteLevel" and kind == "byte-bpe":
        exact_keys(
            item,
            ["type", "add_prefix_space", "trim_offsets", "use_regex"],
            [],
            "HF ByteLevel decoder",
        )
        for field in ("add_prefix_space", "trim_offsets", "use_regex"):
            bool_value(item[field], "HF ByteLevel decoder " + field)
        return
    raise ConversionError("HF decoder construct %r is unsupported" % construct)


def bytes_to_unicode() -> List[str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    mapping = dict(zip(byte_values, codepoints))
    return [chr(mapping[byte]) for byte in range(256)]


def compile_huggingface(
    kind: str, paths: Dict[str, Path], options_value: Any
) -> Dict[str, Any]:
    options = object_value(options_value, "manifest.options")
    exact_keys(
        options,
        ["special_tokens", "padding_side"],
        [
            "ignore_hf_backend_truncation_padding",
            "ignore_hf_base_vocab_added_special_literals",
            "ignore_hf_normalized_base_vocab_added_special_literals",
            "ignore_hf_omnina_encode_inert_fields",
        ],
        "manifest.options",
    )
    ignore_backend_length_state = (
        bool_value(
            options["ignore_hf_backend_truncation_padding"],
            "manifest ignore_hf_backend_truncation_padding",
        )
        if "ignore_hf_backend_truncation_padding" in options
        else False
    )
    if (
        "ignore_hf_backend_truncation_padding" in options
        and not ignore_backend_length_state
    ):
        raise ConversionError(
            "manifest ignore_hf_backend_truncation_padding must be true when present"
        )
    ignore_base_literal_aliases = (
        bool_value(
            options["ignore_hf_base_vocab_added_special_literals"],
            "manifest ignore_hf_base_vocab_added_special_literals",
        )
        if "ignore_hf_base_vocab_added_special_literals" in options
        else False
    )
    if (
        "ignore_hf_base_vocab_added_special_literals" in options
        and not ignore_base_literal_aliases
    ):
        raise ConversionError(
            "manifest ignore_hf_base_vocab_added_special_literals must be true when present"
        )
    ignore_normalized_base_literal_aliases = (
        bool_value(
            options["ignore_hf_normalized_base_vocab_added_special_literals"],
            "manifest ignore_hf_normalized_base_vocab_added_special_literals",
        )
        if "ignore_hf_normalized_base_vocab_added_special_literals" in options
        else False
    )
    if (
        "ignore_hf_normalized_base_vocab_added_special_literals" in options
        and not ignore_normalized_base_literal_aliases
    ):
        raise ConversionError(
            "manifest ignore_hf_normalized_base_vocab_added_special_literals "
            "must be true when present"
        )
    ignore_omnina_encode_inert_fields = (
        bool_value(
            options["ignore_hf_omnina_encode_inert_fields"],
            "manifest ignore_hf_omnina_encode_inert_fields",
        )
        if "ignore_hf_omnina_encode_inert_fields" in options
        else False
    )
    if (
        "ignore_hf_omnina_encode_inert_fields" in options
        and not ignore_omnina_encode_inert_fields
    ):
        raise ConversionError(
            "manifest ignore_hf_omnina_encode_inert_fields must be true when present"
        )
    if ignore_normalized_base_literal_aliases != ignore_omnina_encode_inert_fields:
        raise ConversionError(
            "OmniNA normalized-alias and encode-inert opt-ins must be enabled together"
        )
    if ignore_base_literal_aliases and ignore_normalized_base_literal_aliases:
        raise ConversionError("HF base-vocab added-token ignore modes are mutually exclusive")
    requested_specials = object_value(options["special_tokens"], "manifest special_tokens")
    exact_keys(requested_specials, SPECIAL_NAMES, [], "manifest special_tokens")
    special_pieces = {}  # type: Dict[str, Optional[str]]
    for name in SPECIAL_NAMES:
        raw = requested_specials[name]
        special_pieces[name] = None if raw is None else string_value(
            raw, "manifest special token " + name
        )
    padding_side = string_value(options["padding_side"], "manifest padding_side")
    if padding_side not in ("none", "left", "right"):
        raise ConversionError("manifest padding_side is unsupported")

    tokenizer, _ = read_json(paths["tokenizer"], "HF tokenizer.json")
    exact_keys(
        tokenizer,
        [
            "version",
            "truncation",
            "padding",
            "added_tokens",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "model",
        ],
        [],
        "HF tokenizer",
    )
    if tokenizer["version"] != "1.0":
        raise ConversionError("HF tokenizer version is unsupported")
    ignored_truncation_max_length = None  # type: Optional[int]
    if tokenizer["truncation"] is not None:
        if not ignore_backend_length_state:
            raise ConversionError("HF tokenizer truncation is unsupported")
        truncation = object_value(tokenizer["truncation"], "HF truncation")
        exact_keys(
            truncation,
            ["direction", "max_length", "strategy", "stride"],
            [],
            "HF truncation",
        )
        if (
            truncation["direction"] != "Right"
            or truncation["strategy"] != "LongestFirst"
            or uint32_value(truncation["stride"], "HF truncation stride") != 0
        ):
            raise ConversionError("ignored HF truncation state is unsupported")
        ignored_truncation_max_length = uint32_value(
            truncation["max_length"], "HF truncation max_length", True
        )
    elif ignore_backend_length_state:
        raise ConversionError(
            "ignored HF backend length state requires source truncation"
        )
    model = object_value(tokenizer["model"], "HF model")
    model_type = model.get("type")
    if kind in ("bpe", "byte-bpe"):
        exact_keys(
            model,
            [
                "type",
                "dropout",
                "unk_token",
                "continuing_subword_prefix",
                "end_of_word_suffix",
                "fuse_unk",
                "vocab",
                "merges",
            ],
            ["ignore_merges", "byte_fallback"],
            "HF BPE model",
        )
        fuse_unk = bool_value(model["fuse_unk"], "HF BPE fuse_unk")
        byte_fallback = (
            bool_value(model["byte_fallback"], "HF BPE byte_fallback")
            if "byte_fallback" in model
            else False
        )
        if (
            model_type != "BPE"
            or model["dropout"] is not None
            or model["continuing_subword_prefix"] is not None
            or model["end_of_word_suffix"] is not None
            or ("ignore_merges" in model and bool_value(model["ignore_merges"], "HF BPE ignore_merges"))
        ):
            raise ConversionError("HF BPE model uses unsupported behavior")
        if ignore_omnina_encode_inert_fields:
            if kind != "bpe" or not fuse_unk or not byte_fallback:
                raise ConversionError(
                    "OmniNA encode-inert opt-in requires fuse_unk=true and byte_fallback=true"
                )
        elif fuse_unk or byte_fallback:
            raise ConversionError("HF BPE model uses unsupported behavior")
    else:
        exact_keys(
            model,
            ["type", "unk_token", "continuing_subword_prefix", "max_input_chars_per_word", "vocab"],
            [],
            "HF WordPiece model",
        )
        if model_type != "WordPiece":
            raise ConversionError("HF model type differs from manifest kind")
    (
        vocab,
        by_piece,
        literal_ids,
        appended_ids,
        ignored_normalized_alias_ids,
    ) = hf_vocab(
        model,
        tokenizer["added_tokens"],
        ignore_normalized_base_literal_aliases,
    )
    normalization = hf_normalization(tokenizer["normalizer"])
    if ignore_base_literal_aliases:
        if kind != "bpe":
            raise ConversionError(
                "ignored HF base-vocab added-token literals require BPE"
            )
        if appended_ids:
            raise ConversionError(
                "ignored HF base-vocab added-token literals forbid appended tokens"
            )
        if not literal_ids:
            raise ConversionError(
                "ignored HF base-vocab added-token literals require exact aliases"
            )
        if not normalization:
            raise ConversionError(
                "ignored HF base-vocab added-token literals require normalization"
            )
        literal_ids = []
    if ignore_normalized_base_literal_aliases:
        if kind != "bpe" or not ignored_normalized_alias_ids:
            raise ConversionError(
                "ignored normalized HF base-vocab aliases require exact aliases"
            )
        if not appended_ids:
            raise ConversionError(
                "OmniNA normalized-alias opt-in requires an appended raw literal"
            )
    vocab_ids = set(entry["id"] for entry in vocab)
    for role in ("tokenizer_config", "special_tokens_map"):
        if role in paths:
            supplied = config_specials(paths[role], role)
            for name, piece in supplied.items():
                # Some pinned PreTrainedTokenizerFast configs expose a wrapper
                # unk token even though tokenizers::BPE has unk_token=null.
                # In that closed shape the runtime must preserve the model's
                # error policy rather than inventing an unknown fallback.
                wrapper_only_bpe_unk = (
                    kind == "bpe"
                    and name == "unk"
                    and model.get("unk_token") is None
                    and special_pieces[name] is None
                )
                if piece != special_pieces[name] and not wrapper_only_bpe_unk:
                    raise ConversionError("%s %s differs from manifest" % (role, name))
    special_ids = {}  # type: Dict[str, Optional[int]]
    for name in SPECIAL_NAMES:
        piece = special_pieces[name]
        if piece is None:
            special_ids[name] = None
        elif piece not in by_piece:
            raise ConversionError("special token %s is missing from HF vocab" % name)
        else:
            special_ids[name] = by_piece[piece]
    model_unk = model["unk_token"]
    if model_unk != special_pieces["unk"]:
        raise ConversionError("HF model unk_token differs from manifest")
    if ignore_omnina_encode_inert_fields:
        if not isinstance(model_unk, str) or not model_unk or model_unk not in by_piece:
            raise ConversionError("OmniNA encode-inert opt-in requires a nonempty unknown token")
        byte_pieces = [
            piece
            for piece in by_piece
            if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece) is not None
        ]
        if byte_pieces:
            raise ConversionError(
                "OmniNA encode-inert byte fallback requires zero byte pieces"
            )

    pre_tokenizer, add_prefix_space = hf_pre_tokenizer(tokenizer["pre_tokenizer"], kind)
    prefix_ids, suffix_ids = hf_post_processor(tokenizer["post_processor"], by_piece)
    padding = tokenizer["padding"]
    if padding is not None:
        padding_object = object_value(padding, "HF padding")
        exact_keys(
            padding_object,
            ["strategy", "direction", "pad_to_multiple_of", "pad_id", "pad_type_id", "pad_token"],
            [],
            "HF padding",
        )
        if ignore_backend_length_state:
            fixed_strategy = object_value(
                padding_object["strategy"], "ignored HF padding strategy"
            )
            exact_keys(
                fixed_strategy, ["Fixed"], [], "ignored HF padding strategy"
            )
            fixed_length = uint32_value(
                fixed_strategy["Fixed"], "ignored HF fixed padding length", True
            )
            if fixed_length != ignored_truncation_max_length:
                raise ConversionError(
                    "ignored HF truncation and fixed padding lengths differ"
                )
        elif (
            padding_object["strategy"] != "BatchLongest"
            or padding_object["pad_to_multiple_of"] is not None
        ):
            raise ConversionError("only HF BatchLongest padding is supported")
        if padding_object["pad_to_multiple_of"] is not None:
            raise ConversionError("HF padding multiple is unsupported")
        direction = string_value(padding_object["direction"], "HF padding direction").lower()
        if direction != padding_side or padding_side == "none":
            raise ConversionError("HF padding direction differs from manifest")
        if uint32_value(padding_object["pad_type_id"], "HF pad_type_id") != 0:
            raise ConversionError("HF padding pad_type_id must be zero")
        pad_id = uint32_value(padding_object["pad_id"], "HF padding pad_id")
        if (
            pad_id != special_ids["pad"]
            or padding_object["pad_token"] != special_pieces["pad"]
        ):
            raise ConversionError("HF padding token differs from manifest")
    elif ignore_backend_length_state:
        raise ConversionError("ignored HF backend length state requires source padding")
    pad_id = special_ids["pad"] if padding_side != "none" else None
    if padding_side != "none" and pad_id is None:
        raise ConversionError("enabled padding requires pad special token")
    validate_hf_decoder(
        tokenizer["decoder"], kind, ignore_omnina_encode_inert_fields
    )

    if kind in ("bpe", "byte-bpe"):
        merges = hf_merges(model["merges"])
        runtime_model = {"merges": merges}  # type: Dict[str, Any]
        if kind == "byte-bpe":
            runtime_model = {
                "add_prefix_space": add_prefix_space,
                "byte_encoder": bytes_to_unicode(),
                "merges": merges,
            }
        elif literal_ids:
            runtime_model["literal_token_ids"] = literal_ids
    else:
        runtime_model = {
            "continuation_prefix": string_value(
                model["continuing_subword_prefix"], "HF WordPiece continuation prefix"
            ),
            "max_input_chars_per_word": uint32_value(
                model["max_input_chars_per_word"], "HF WordPiece max chars", True
            ),
        }
    if appended_ids and kind != "bpe":
        raise ConversionError(
            "appended HF added tokens require BPE"
        )
    return validate_runtime_asset(
        {
            "format": "evo-tokenizer-v1",
            "kind": kind,
            "normalization": normalization,
            "pre_tokenizer": pre_tokenizer,
            "model": runtime_model,
            "post_processor": {
                "prefix_ids": prefix_ids,
                "suffix_ids": suffix_ids,
                "padding": {"side": padding_side, "pad_id": pad_id},
            },
            "special_tokens": special_ids,
            "vocab": vocab,
        }
    )


def compile_vocab_text(kind: str, path: Path, options_value: Any) -> Dict[str, Any]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConversionError("cannot read vocab text: %s" % error)
    if text.startswith("\ufeff"):
        raise ConversionError("vocab text must not contain a BOM")
    pieces = text.splitlines()
    if not pieces or any(not piece or "\x00" in piece for piece in pieces):
        raise ConversionError("vocab text contains an empty or invalid piece")
    if len(set(pieces)) != len(pieces):
        raise ConversionError("vocab text contains duplicate pieces")
    options = object_value(options_value, "manifest.options")
    exact_keys(
        options,
        ["normalization", "pre_tokenizer", "model", "post_processor", "special_tokens"],
        ["audited_vocab_boundary"],
        "manifest.options",
    )
    compiled_pieces = pieces
    if "audited_vocab_boundary" in options:
        boundary = object_value(
            options["audited_vocab_boundary"],
            "manifest audited_vocab_boundary",
        )
        exact_keys(
            boundary,
            ["source_size", "compiled_size", "excluded_suffix"],
            [],
            "manifest audited_vocab_boundary",
        )
        source_size = uint32_value(
            boundary["source_size"],
            "manifest audited_vocab_boundary.source_size",
            True,
        )
        compiled_size = uint32_value(
            boundary["compiled_size"],
            "manifest audited_vocab_boundary.compiled_size",
            True,
        )
        excluded = list_value(
            boundary["excluded_suffix"],
            "manifest audited_vocab_boundary.excluded_suffix",
        )
        if (
            source_size != len(pieces)
            or compiled_size >= source_size
            or len(excluded) != source_size - compiled_size
            or not excluded
            or len(excluded) > 32
        ):
            raise ConversionError(
                "audited vocab boundary does not exactly cover the source suffix"
            )
        for offset, raw in enumerate(excluded):
            label = "manifest audited_vocab_boundary.excluded_suffix[%d]" % offset
            item = object_value(raw, label)
            exact_keys(item, ["id", "piece", "input_policy"], [], label)
            token_id = uint32_value(item["id"], label + ".id")
            piece = string_value(item["piece"], label + ".piece")
            policy = string_value(item["input_policy"], label + ".input_policy")
            expected_id = compiled_size + offset
            if (
                token_id != expected_id
                or pieces[expected_id] != piece
                or policy != "reject"
            ):
                raise ConversionError(
                    "audited vocab excluded suffix differs at ID %d" % expected_id
                )
        compiled_pieces = pieces[:compiled_size]
    return validate_runtime_asset(
        {
            "format": "evo-tokenizer-v1",
            "kind": kind,
            "normalization": options["normalization"],
            "pre_tokenizer": options["pre_tokenizer"],
            "model": options["model"],
            "post_processor": options["post_processor"],
            "special_tokens": options["special_tokens"],
            "vocab": [
                {"id": index, "piece": piece}
                for index, piece in enumerate(compiled_pieces)
            ],
        }
    )


def compile_custom(kind: str, path: Path, options_value: Any) -> Dict[str, Any]:
    options = object_value(options_value, "manifest.options")
    exact_keys(options, [], [], "manifest.options")
    source, _ = read_json(path, "custom tokenizer spec")
    exact_keys(
        source,
        [
            "format",
            "kind",
            "normalization",
            "pre_tokenizer",
            "model",
            "post_processor",
            "special_tokens",
            "vocab",
        ],
        [],
        "custom tokenizer spec",
    )
    if source["format"] != "evo-tokenizer-source-v1" or source["kind"] != kind:
        raise ConversionError("custom tokenizer format/kind differs from manifest")
    runtime_kind = "longest-match" if kind in ("mixed", "biotoken") else kind
    source["format"] = "evo-tokenizer-v1"
    source["kind"] = runtime_kind
    return validate_runtime_asset(source)


def canonical_json(value: Dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConversionError("cannot serialize tokenizer asset: %s" % error)


def stage_file(path: Path, payload: bytes) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        return temporary
    except OSError as error:
        raise ConversionError("cannot stage output %s: %s" % (path, error))


def publish_outputs(outputs: Sequence[Tuple[Path, bytes]], force: bool) -> None:
    destinations = [path for path, _ in outputs]
    if len(set(destinations)) != len(destinations):
        raise ConversionError("output and descriptor paths must be distinct")
    if not force:
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise FileExistsError("output already exists: %s" % ", ".join(existing))
    staged = []  # type: List[Tuple[Path, Path]]
    try:
        for path, payload in outputs:
            staged.append((path, stage_file(path, payload)))
        for path, temporary in staged:
            os.replace(str(temporary), str(path))
    finally:
        for _, temporary in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--asset-path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        (
            source_kind,
            output_kind,
            paths,
            manifest_bytes,
            receipt_contract,
            manifest,
        ) = verified_assets(args.manifest.resolve(), args.receipt.resolve())
        if source_kind == "huggingface-json":
            asset = compile_huggingface(output_kind, paths, manifest["options"])
        elif source_kind == "vocab-text":
            asset = compile_vocab_text(output_kind, paths["vocab"], manifest["options"])
        else:
            asset = compile_custom(output_kind, paths["spec"], manifest["options"])
        payload = canonical_json(asset)
        if len(payload) > MAX_TOKENIZER_ASSET_BYTES:
            raise ConversionError("compiled tokenizer exceeds the 64 MiB runtime limit")
        asset_path = normalized_relative_path(
            args.asset_path if args.asset_path is not None else args.output.name,
            "artifact tokenizer path",
        )
        descriptor = {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": sha256_bytes(manifest_bytes),
            "source_receipt_contract_sha256": sha256_bytes(receipt_contract),
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": asset_path,
            "tokenizer.sha256": sha256_bytes(payload),
            "tokenizer.size": len(payload),
        }
        descriptor_payload = canonical_json(descriptor)
        publish_outputs(
            ((args.output, payload), (args.descriptor, descriptor_payload)), args.force
        )
        print("wrote %s" % args.output)
        print("tokenizer.sha256=%s" % descriptor["tokenizer.sha256"])
        return 0
    except (ConversionError, FileExistsError, OSError) as error:
        print("convert_tokenizer_asset: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
