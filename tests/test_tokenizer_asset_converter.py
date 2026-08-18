#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise deterministic tokenizer compilation and strict corruption gates."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SPECIAL_NAMES = ("unk", "pad", "bos", "eos", "cls", "sep", "mask")


def canonical(value: Dict[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_receipt_contract_digest(path: Path) -> str:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    projection = {
        "schema_version": receipt["schema_version"],
        "kind": receipt["kind"],
        "files": sorted(
            [
                {
                    key: item[key]
                    for key in ("role", "name", "size", "sha256")
                }
                for item in receipt["files"]
            ],
            key=lambda item: (item["role"], item["name"]),
        ),
    }
    return hashlib.sha256(canonical(projection)).hexdigest()


def added_token(token_id: int, content: str) -> Dict[str, Any]:
    return {
        "id": token_id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


def special_pieces(**values: Optional[str]) -> Dict[str, Optional[str]]:
    return {name: values.get(name) for name in SPECIAL_NAMES}


def special_ids(**values: Optional[int]) -> Dict[str, Optional[int]]:
    return {name: values.get(name) for name in SPECIAL_NAMES}


def output_post(
    prefix: List[int], suffix: List[int], side: str, pad_id: Optional[int]
) -> Dict[str, Any]:
    return {
        "prefix_ids": prefix,
        "suffix_ids": suffix,
        "padding": {"side": side, "pad_id": pad_id},
    }


def template_special(piece: str, token_id: int) -> Dict[str, Any]:
    return {"id": piece, "ids": [token_id], "tokens": [piece]}


def bpe_tokenizer() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "Strip", "strip_left": True, "strip_right": True},
                {
                    "type": "Replace",
                    "pattern": {"Regex": "^"},
                    "content": "<dna>",
                },
                {"type": "Uppercase"},
                {
                    "type": "Replace",
                    "pattern": {"String": "U"},
                    "content": "T",
                },
                {
                    "type": "Replace",
                    "pattern": {"Regex": "N{10,}"},
                    "content": "N",
                },
            ],
        },
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": "N"},
            "behavior": "Isolated",
            "invert": False,
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            "pair": [],
            "special_tokens": {
                "[CLS]": template_special("[CLS]", 1),
                "[SEP]": template_special("[SEP]", 2),
            },
        },
        "decoder": {"type": "BPEDecoder", "suffix": "</w>"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {
                "[UNK]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[PAD]": 3,
                "A": 4,
                "C": 5,
                "AC": 6,
                "N": 7,
            },
            "merges": [["A", "C"]],
        },
    }


def wordpiece_tokenizer() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            added_token(0, "[UNK]"),
            added_token(1, "[CLS]"),
            added_token(2, "[SEP]"),
            added_token(3, "[PAD]"),
        ],
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": False,
            "handle_chinese_chars": False,
            "strip_accents": None,
            "lowercase": True,
        },
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "BertProcessing",
            "sep": ["[SEP]", 2],
            "cls": ["[CLS]", 1],
        },
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 64,
            "vocab": {
                "[UNK]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "[PAD]": 3,
                "AC": 4,
                "##GT": 5,
            },
        },
    }


def byte_bpe_tokenizer() -> Dict[str, Any]:
    byte_pieces = bytes_to_unicode()
    vocab = {"<unk>": 0}
    for index, piece in enumerate(byte_pieces, start=1):
        vocab[piece] = index
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [added_token(0, "<unk>")],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": False,
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": [],
        },
    }


def plant_caduceus_tokenizer() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            added_token(0, "[PAD]"),
            added_token(1, "[MASK]"),
            added_token(2, "[UNK]"),
        ],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {
                "[PAD]": 0,
                "[MASK]": 1,
                "[UNK]": 2,
                "a": 3,
                "c": 4,
                "g": 5,
                "t": 6,
            },
            "merges": [],
        },
    }


def omnina_tokenizer() -> Dict[str, Any]:
    normalized_aliases = [
        added_token(0, "<unk>"),
        added_token(1, "<s>"),
        added_token(2, "</s>"),
    ]
    for item in normalized_aliases:
        item["normalized"] = True
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": normalized_aliases + [added_token(6, "[PAD]")],
        "normalizer": {
            "type": "Sequence",
            "normalizers": [
                {"type": "Prepend", "prepend": "▁"},
                {
                    "type": "Replace",
                    "pattern": {"String": " "},
                    "content": "▁",
                },
            ],
        },
        "pre_tokenizer": None,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [],
            "special_tokens": {"<s>": template_special("<s>", 1)},
        },
        "decoder": {
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
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "vocab": {
                "<unk>": 0,
                "<s>": 1,
                "</s>": 2,
                "▁": 3,
                "A": 4,
                "▁A": 5,
            },
            "merges": [["▁", "A"]],
        },
    }


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


def create_case(
    root: Path,
    source: str,
    kind: str,
    files: Dict[str, Tuple[str, bytes]],
    options: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_files = []  # type: List[Dict[str, Any]]
    receipt_files = []  # type: List[Dict[str, Any]]
    for role, (name, payload) in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entry = {
            "role": role,
            "name": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        manifest_files.append(entry)
        receipt_files.append(dict(entry, path=str(path.resolve())))
    manifest = root / "manifest.json"
    receipt = root / "source-receipt.json"
    write_json(
        manifest,
        {
            "format": "evo-tokenizer-compiler-v1",
            "source": source,
            "kind": kind,
            "files": manifest_files,
            "options": options,
        },
    )
    write_json(
        receipt,
        {"schema_version": 1, "kind": "tokenizer-source", "files": receipt_files},
    )
    return manifest, receipt, root / "tokenizer.evo.json", root / "descriptor.json"


def run_converter(
    converter: Path,
    manifest: Path,
    receipt: Path,
    output: Path,
    descriptor: Path,
    force: bool = False,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(converter),
        "--manifest",
        str(manifest),
        "--receipt",
        str(receipt),
        "--output",
        str(output),
        "--descriptor",
        str(descriptor),
        "--asset-path",
        "tokenizers/tiny.json",
    ]
    if force:
        command.append("--force")
    return subprocess.run(command, check=False, capture_output=True, text=True)


def assert_success(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode != 0:
        raise AssertionError("%s failed: %s" % (label, result.stderr))


def check_descriptor(output: Path, descriptor: Path) -> None:
    payload = output.read_bytes()
    descriptor_payload = descriptor.read_bytes()
    if not payload.endswith(b"\n") or not descriptor_payload.endswith(b"\n"):
        raise AssertionError("tokenizer output/descriptor omitted final newline")
    value = json.loads(descriptor_payload)
    expected = {
        "converter.schema": "evo-tokenizer-conversion-receipt",
        "converter.version": 1,
        "compiler_manifest_sha256": digest(output.parent / "manifest.json"),
        "source_receipt_contract_sha256": source_receipt_contract_digest(
            output.parent / "source-receipt.json"
        ),
        "tokenizer.profile": "evo-tokenizer-v1",
        "tokenizer.path": "tokenizers/tiny.json",
        "tokenizer.sha256": hashlib.sha256(payload).hexdigest(),
        "tokenizer.size": len(payload),
    }
    if value != expected or descriptor_payload != canonical(expected):
        raise AssertionError("tokenizer descriptor is not canonical or exact")


def custom_spec(kind: str) -> Dict[str, Any]:
    if kind in ("mixed", "biotoken"):
        vocab = [
            {"id": 0, "piece": "<unk>"},
            {"id": 1, "piece": "A"},
            {"id": 2, "piece": "CG"},
        ]
        model = {"unknown_policy": "unk", "match_special_literals": True}
        return {
            "format": "evo-tokenizer-source-v1",
            "kind": kind,
            "normalization": [{"op": "ascii-uppercase"}],
            "pre_tokenizer": {"kind": "whole-input"},
            "model": model,
            "post_processor": output_post([], [], "none", None),
            "special_tokens": special_ids(unk=0),
            "vocab": vocab,
        }
    if kind == "kmer-bpe":
        return {
            "format": "evo-tokenizer-source-v1",
            "kind": kind,
            "normalization": [{"op": "ascii-uppercase"}],
            "pre_tokenizer": {"kind": "none"},
            "model": {
                "k": 2,
                "stride": 2,
                "tail": "error",
                "unknown_policy": "error",
                "merges": [["AC", "GT"]],
            },
            "post_processor": output_post([], [], "none", None),
            "special_tokens": special_ids(),
            "vocab": [
                {"id": 0, "piece": "AC"},
                {"id": 1, "piece": "GT"},
                {"id": 2, "piece": "ACGT"},
            ],
        }
    return {
        "format": "evo-tokenizer-source-v1",
        "kind": kind,
        "normalization": [{"op": "ascii-uppercase"}, {"op": "u-to-t"}],
        "pre_tokenizer": {"kind": "none"},
        "model": {"unknown_policy": "error", "match_special_literals": False},
        "post_processor": output_post([], [], "none", None),
        "special_tokens": special_ids(),
        "vocab": [
            {"id": 0, "piece": "A"},
            {"id": 1, "piece": "C"},
            {"id": 2, "piece": "G"},
            {"id": 3, "piece": "T"},
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)

    bpe_options = {
        "special_tokens": special_pieces(
            unk="[UNK]", pad="[PAD]", cls="[CLS]", sep="[SEP]"
        ),
        "padding_side": "right",
    }
    manifest, receipt, output, descriptor = create_case(
        args.work_dir / "hf-bpe",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(bpe_tokenizer()))},
        bpe_options,
    )
    first = run_converter(args.converter, manifest, receipt, output, descriptor)
    assert_success(first, "HF BPE conversion")
    first_payload = output.read_bytes()
    first_descriptor = descriptor.read_bytes()
    check_descriptor(output, descriptor)
    relocated_manifest, relocated_receipt, relocated_output, relocated_descriptor = (
        create_case(
            args.work_dir / "hf-bpe-relocated-source",
            "huggingface-json",
            "bpe",
            {"tokenizer": ("tokenizer.json", canonical(bpe_tokenizer()))},
            bpe_options,
        )
    )
    if receipt.read_bytes() == relocated_receipt.read_bytes():
        raise AssertionError("relocated source receipts unexpectedly have equal bytes")
    assert_success(
        run_converter(
            args.converter,
            relocated_manifest,
            relocated_receipt,
            relocated_output,
            relocated_descriptor,
        ),
        "relocated HF BPE conversion",
    )
    if (
        relocated_output.read_bytes() != first_payload
        or relocated_descriptor.read_bytes() != first_descriptor
    ):
        raise AssertionError(
            "tokenizer asset/descriptor changed across absolute source roots"
        )
    parsed = json.loads(first_payload)
    if first_payload != canonical(parsed):
        raise AssertionError("compiled tokenizer JSON is not compact canonical bytes")
    if (
        parsed["kind"] != "bpe"
        or parsed["pre_tokenizer"] != {"kind": "split-isolated", "literal": "N"}
        or parsed["model"]["merges"] != [["A", "C"]]
        or parsed["post_processor"]["prefix_ids"] != [1]
        or parsed["post_processor"]["suffix_ids"] != [2]
    ):
        raise AssertionError("HF BPE constructs were not preserved exactly")
    if parsed["normalization"] != [
        {"op": "strip-ascii-whitespace"},
        {"op": "prepend-literal", "value": "<dna>"},
        {"op": "ascii-uppercase"},
        {"op": "replace-literal", "from": "U", "to": "T"},
        {
            "op": "replace-byte-run",
            "byte": "N",
            "min_count": 10,
            "replacement": "N",
        },
    ]:
        raise AssertionError("ordered HF normalization subset was not preserved")

    # B80: some pinned fast-tokenizer sources retain training-time backend
    # truncation/fixed-padding state that the extractor explicitly overrides
    # on every call.  A manifest must opt in to ignoring the exact closed
    # shape; the wrapper/runtime padding contract remains explicit above.
    backend_length_state = copy.deepcopy(bpe_tokenizer())
    backend_length_state["truncation"] = {
        "direction": "Right",
        "max_length": 2000,
        "strategy": "LongestFirst",
        "stride": 0,
    }
    backend_length_state["padding"] = {
        "strategy": {"Fixed": 2000},
        "direction": "Right",
        "pad_to_multiple_of": None,
        "pad_id": 3,
        "pad_type_id": 0,
        "pad_token": "[PAD]",
    }
    backend_length_options = dict(
        bpe_options, ignore_hf_backend_truncation_padding=True
    )
    (
        backend_length_manifest,
        backend_length_receipt,
        backend_length_output,
        backend_length_descriptor,
    ) = create_case(
        args.work_dir / "hf-bpe-ignored-backend-length-state",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(backend_length_state))},
        backend_length_options,
    )
    assert_success(
        run_converter(
            args.converter,
            backend_length_manifest,
            backend_length_receipt,
            backend_length_output,
            backend_length_descriptor,
        ),
        "ignored HF backend truncation/fixed-padding conversion",
    )
    if backend_length_output.read_bytes() != first_payload:
        raise AssertionError("ignored HF backend length state changed compiled bytes")
    (
        relocated_length_manifest,
        relocated_length_receipt,
        relocated_length_output,
        relocated_length_descriptor,
    ) = create_case(
        args.work_dir / "hf-bpe-ignored-backend-length-state-relocated",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(backend_length_state))},
        backend_length_options,
    )
    assert_success(
        run_converter(
            args.converter,
            relocated_length_manifest,
            relocated_length_receipt,
            relocated_length_output,
            relocated_length_descriptor,
        ),
        "relocated ignored HF backend length state conversion",
    )
    if (
        relocated_length_output.read_bytes() != backend_length_output.read_bytes()
        or relocated_length_descriptor.read_bytes()
        != backend_length_descriptor.read_bytes()
    ):
        raise AssertionError(
            "ignored backend length asset/descriptor changed across source roots"
        )

    # B78: older tokenizers releases spell the same BPE decoder as `BPE`
    # instead of `BPEDecoder`. Decoder metadata does not participate in
    # encoding, so both exact spellings must compile to identical assets.
    legacy_decoder = copy.deepcopy(bpe_tokenizer())
    legacy_decoder["decoder"]["type"] = "BPE"
    (
        legacy_decoder_manifest,
        legacy_decoder_receipt,
        legacy_decoder_output,
        legacy_decoder_descriptor,
    ) = create_case(
        args.work_dir / "hf-bpe-legacy-decoder",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(legacy_decoder))},
        bpe_options,
    )
    assert_success(
        run_converter(
            args.converter,
            legacy_decoder_manifest,
            legacy_decoder_receipt,
            legacy_decoder_output,
            legacy_decoder_descriptor,
        ),
        "legacy HF BPE decoder conversion",
    )
    if legacy_decoder_output.read_bytes() != first_payload:
        raise AssertionError("legacy BPE decoder alias changed compiled bytes")

    for label, source_pre, expected_pre in (
        ("whole", None, {"kind": "whole-input"}),
        (
            "whitespace",
            {"type": "Whitespace"},
            {"kind": "hf-whitespace-ascii"},
        ),
    ):
        variant = copy.deepcopy(bpe_tokenizer())
        variant["pre_tokenizer"] = source_pre
        variant_manifest, variant_receipt, variant_output, variant_descriptor = create_case(
            args.work_dir / ("hf-bpe-" + label),
            "huggingface-json",
            "bpe",
            {"tokenizer": ("tokenizer.json", canonical(variant))},
            bpe_options,
        )
        assert_success(
            run_converter(
                args.converter,
                variant_manifest,
                variant_receipt,
                variant_output,
                variant_descriptor,
            ),
            "HF BPE %s-input conversion" % label,
        )
        if json.loads(variant_output.read_bytes())["pre_tokenizer"] != expected_pre:
            raise AssertionError("HF BPE %s pre-tokenizer was not preserved" % label)

    # Older tokenizers JSON (including the pinned DNABERT-2 asset) predates
    # the explicit byte_fallback=false field.  Absence is the legacy spelling
    # of the same closed behavior, while true remains unsupported below.
    legacy_bpe = copy.deepcopy(bpe_tokenizer())
    del legacy_bpe["model"]["byte_fallback"]
    legacy_manifest, legacy_receipt, legacy_output, legacy_descriptor = create_case(
        args.work_dir / "hf-bpe-legacy-no-byte-fallback",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(legacy_bpe))},
        bpe_options,
    )
    assert_success(
        run_converter(
            args.converter,
            legacy_manifest,
            legacy_receipt,
            legacy_output,
            legacy_descriptor,
        ),
        "legacy HF BPE conversion without byte_fallback",
    )
    if legacy_output.read_bytes() != first_payload:
        raise AssertionError("omitted byte_fallback did not normalize to false")

    # GPT2-Gene's pinned tokenizer exposes <unk> only through the wrapper
    # configs while tokenizers::BPE itself has unk_token=null.  Its older
    # AddedToken maps also omit the now-optional `special` field.
    wrapper_only = copy.deepcopy(bpe_tokenizer())
    wrapper_only["normalizer"] = None
    wrapper_only["pre_tokenizer"] = {"type": "Whitespace"}
    wrapper_only["post_processor"] = None
    wrapper_only["model"]["unk_token"] = None
    wrapper_config = canonical(
        {"unk_token": "[UNK]", "pad_token": "[PAD]", "padding_side": "right"}
    )
    wrapper_special_map = canonical(
        {
            "unk_token": {
                "content": "[UNK]",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
            },
            "pad_token": {
                "content": "[PAD]",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
            },
        }
    )
    wrapper_manifest, wrapper_receipt, wrapper_output, wrapper_descriptor = create_case(
        args.work_dir / "hf-bpe-wrapper-only-unk",
        "huggingface-json",
        "bpe",
        {
            "tokenizer": ("tokenizer.json", canonical(wrapper_only)),
            "tokenizer_config": ("tokenizer_config.json", wrapper_config),
            "special_tokens_map": ("special_tokens_map.json", wrapper_special_map),
        },
        {
            "special_tokens": special_pieces(pad="[PAD]"),
            "padding_side": "right",
        },
    )
    assert_success(
        run_converter(
            args.converter,
            wrapper_manifest,
            wrapper_receipt,
            wrapper_output,
            wrapper_descriptor,
        ),
        "HF BPE wrapper-only unknown conversion",
    )
    wrapper_asset = json.loads(wrapper_output.read_bytes())
    if (
        wrapper_asset["special_tokens"]["unk"] is not None
        or wrapper_asset["special_tokens"]["pad"] != 3
        or wrapper_asset["pre_tokenizer"] != {"kind": "hf-whitespace-ascii"}
    ):
        raise AssertionError("wrapper-only BPE unknown policy was not preserved")

    appended = copy.deepcopy(wrapper_only)
    appended["added_tokens"].append(added_token(8, "[BPE]"))
    appended_options = {
        "special_tokens": special_pieces(pad="[PAD]"),
        "padding_side": "right",
    }
    appended_manifest, appended_receipt, appended_output, appended_descriptor = create_case(
        args.work_dir / "hf-bpe-appended-special",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(appended))},
        appended_options,
    )
    assert_success(
        run_converter(
            args.converter,
            appended_manifest,
            appended_receipt,
            appended_output,
            appended_descriptor,
        ),
        "HF BPE appended special conversion",
    )
    appended_asset = json.loads(appended_output.read_bytes())
    if (
        appended_asset["vocab"][-1] != {"id": 8, "piece": "[BPE]"}
        or appended_asset["model"].get("literal_token_ids") != [8]
    ):
        raise AssertionError("contiguous appended HF special was not preserved")
    appended_runtime = subprocess.run(
        [
            str(args.runtime),
            "--verify-asset",
            str(appended_output.parent),
            appended_output.name,
            digest(appended_output),
            str(appended_output.stat().st_size),
            "[BPE]",
            "8",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if appended_runtime.returncode != 0:
        raise AssertionError(
            "appended HF special did not load/encode in C++: %s"
            % appended_runtime.stderr
        )

    mutbert_alias = copy.deepcopy(wrapper_only)
    mutbert_alias["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        ],
        "pair": [],
        "special_tokens": {
            "[CLS]": template_special("[CLS]", 1),
            "[SEP]": template_special("[SEP]", 2),
        },
    }
    mutbert_alias["model"]["unk_token"] = "[UNK]"
    mutbert_alias["model"]["vocab"] = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[PAD]": 3,
        "[MASK]": 4,
        "A": 5,
        "C": 6,
        "G": 7,
        "T": 8,
    }
    mutbert_alias["model"]["merges"] = []
    mutbert_alias["added_tokens"] = [
        added_token(0, "[UNK]"),
        added_token(1, "[CLS]"),
        added_token(2, "[SEP]"),
        added_token(3, "[PAD]"),
        added_token(4, "[MASK]"),
    ]
    alias_options = {
        "special_tokens": special_pieces(
            unk="[UNK]", cls="[CLS]", sep="[SEP]", pad="[PAD]", mask="[MASK]"
        ),
        "padding_side": "right",
    }
    alias_manifest, alias_receipt, alias_output, alias_descriptor = create_case(
        args.work_dir / "hf-bpe-base-vocab-special-alias",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(mutbert_alias))},
        alias_options,
    )
    assert_success(
        run_converter(
            args.converter,
            alias_manifest,
            alias_receipt,
            alias_output,
            alias_descriptor,
        ),
        "HF BPE base-vocab special alias conversion",
    )
    alias_asset = json.loads(alias_output.read_bytes())
    if alias_asset["model"].get("literal_token_ids") != [0, 1, 2, 3, 4]:
        raise AssertionError("base-vocab HF special aliases were not preserved")
    alias_runtime = subprocess.run(
        [
            str(args.runtime),
            "--verify-asset",
            str(alias_output.parent),
            alias_output.name,
            digest(alias_output),
            str(alias_output.stat().st_size),
            "[MASK]A",
            "1,4,5,2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if alias_runtime.returncode != 0:
        raise AssertionError(
            "base-vocab HF special alias did not match official token IDs in C++: %s"
            % alias_runtime.stderr
        )

    gena_alias = copy.deepcopy(mutbert_alias)
    gena_alias["normalizer"] = {
        "type": "Sequence",
        "normalizers": [
            {"type": "Strip", "strip_left": True, "strip_right": True},
            {
                "type": "Replace",
                "pattern": {"Regex": "N{10,}"},
                "content": "-",
            },
        ],
    }
    gena_alias["pre_tokenizer"] = {
        "type": "Split",
        "pattern": {"String": "-"},
        "behavior": "Isolated",
        "invert": False,
    }
    gena_alias["decoder"] = {"type": "BPEDecoder", "suffix": " "}
    gena_alias["model"]["vocab"] = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[PAD]": 3,
        "[MASK]": 4,
        "-": 5,
        "A": 6,
        "C": 7,
        "G": 8,
        "T": 9,
        "AC": 10,
        "ACG": 11,
    }
    gena_alias["model"]["merges"] = [["A", "C"], ["AC", "G"]]
    gena_alias["added_tokens"] = [
        added_token(0, "[UNK]"),
        added_token(1, "[CLS]"),
        added_token(2, "[SEP]"),
        added_token(3, "[PAD]"),
        added_token(4, "[MASK]"),
        added_token(5, "-"),
    ]
    gena_manifest, gena_receipt, gena_output, gena_descriptor = create_case(
        args.work_dir / "hf-bpe-gena-normalized-special-aliases",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(gena_alias))},
        alias_options,
    )
    assert_success(
        run_converter(
            args.converter,
            gena_manifest,
            gena_receipt,
            gena_output,
            gena_descriptor,
        ),
        "GENA normalized base-vocab special alias conversion",
    )
    gena_asset = json.loads(gena_output.read_bytes())
    if gena_asset["model"].get("literal_token_ids") != [0, 1, 2, 3, 4, 5]:
        raise AssertionError("GENA normalized special aliases were not preserved")
    for source_text, expected_ids in (
        ("  [MASK]ACGTNNNNNNNNNN  ", "1,4,11,9,5,2"),
        ("NNNNNNNNNN", "1,5,2"),
        ("[MASK][MASK]", "1,4,4,2"),
    ):
        gena_runtime = subprocess.run(
            [
                str(args.runtime),
                "--verify-asset",
                str(gena_output.parent),
                gena_output.name,
                digest(gena_output),
                str(gena_output.stat().st_size),
                source_text,
                expected_ids,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if gena_runtime.returncode != 0:
            raise AssertionError(
                "GENA pre-normalization special literal parity failed in C++: %s"
                % gena_runtime.stderr
            )
    (
        relocated_gena_manifest,
        relocated_gena_receipt,
        relocated_gena_output,
        relocated_gena_descriptor,
    ) = create_case(
        args.work_dir / "hf-bpe-gena-normalized-special-aliases-relocated",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(gena_alias))},
        alias_options,
    )
    assert_success(
        run_converter(
            args.converter,
            relocated_gena_manifest,
            relocated_gena_receipt,
            relocated_gena_output,
            relocated_gena_descriptor,
        ),
        "relocated GENA normalized special alias conversion",
    )
    if (
        relocated_gena_output.read_bytes() != gena_output.read_bytes()
        or relocated_gena_descriptor.read_bytes() != gena_descriptor.read_bytes()
    ):
        raise AssertionError(
            "GENA normalized tokenizer changed across absolute source roots"
        )

    omnina_source = omnina_tokenizer()
    omnina_options = {
        "ignore_hf_normalized_base_vocab_added_special_literals": True,
        "ignore_hf_omnina_encode_inert_fields": True,
        "special_tokens": special_pieces(
            unk="<unk>", pad="</s>", bos="<s>", eos="</s>"
        ),
        "padding_side": "left",
    }
    (
        omnina_manifest,
        omnina_receipt,
        omnina_output,
        omnina_descriptor,
    ) = create_case(
        args.work_dir / "hf-bpe-omnina-inert-fields",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(omnina_source))},
        omnina_options,
    )
    assert_success(
        run_converter(
            args.converter,
            omnina_manifest,
            omnina_receipt,
            omnina_output,
            omnina_descriptor,
        ),
        "OmniNA audited encode-inert field conversion",
    )
    omnina_asset = json.loads(omnina_output.read_bytes())
    if (
        omnina_asset["model"].get("literal_token_ids") != [6]
        or omnina_asset["special_tokens"]
        != special_ids(unk=0, pad=2, bos=1, eos=2)
        or omnina_asset["post_processor"]
        != output_post([1], [], "left", 2)
    ):
        raise AssertionError("OmniNA audited tokenizer contract was not preserved")
    for source_text, expected_ids in (
        ("A", "1,5"),
        ("[PAD]A", "1,6,5"),
        ("A[PAD]", "1,5,6"),
    ):
        omnina_runtime = subprocess.run(
            [
                str(args.runtime),
                "--verify-asset",
                str(omnina_output.parent),
                omnina_output.name,
                digest(omnina_output),
                str(omnina_output.stat().st_size),
                source_text,
                expected_ids,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if omnina_runtime.returncode != 0:
            raise AssertionError(
                "OmniNA appended raw literal/native DNA parity failed: %s"
                % omnina_runtime.stderr
            )
    (
        relocated_omnina_manifest,
        relocated_omnina_receipt,
        relocated_omnina_output,
        relocated_omnina_descriptor,
    ) = create_case(
        args.work_dir / "hf-bpe-omnina-inert-fields-relocated",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(omnina_source))},
        omnina_options,
    )
    assert_success(
        run_converter(
            args.converter,
            relocated_omnina_manifest,
            relocated_omnina_receipt,
            relocated_omnina_output,
            relocated_omnina_descriptor,
        ),
        "relocated OmniNA audited tokenizer conversion",
    )
    if (
        relocated_omnina_output.read_bytes() != omnina_output.read_bytes()
        or relocated_omnina_descriptor.read_bytes()
        != omnina_descriptor.read_bytes()
    ):
        raise AssertionError("OmniNA tokenizer changed across absolute source roots")

    plant_source = plant_caduceus_tokenizer()
    plant_options = {
        "ignore_hf_base_vocab_added_special_literals": True,
        "special_tokens": special_pieces(
            unk="[UNK]", pad="[PAD]", mask="[MASK]"
        ),
        "padding_side": "right",
    }
    plant_manifest, plant_receipt, plant_output, plant_descriptor = create_case(
        args.work_dir / "hf-bpe-audited-normalized-base-vocab-aliases",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(plant_source))},
        plant_options,
    )
    assert_success(
        run_converter(
            args.converter,
            plant_manifest,
            plant_receipt,
            plant_output,
            plant_descriptor,
        ),
        "audited normalized base-vocab alias conversion",
    )
    if (
        digest(plant_output)
        != "87ffd8fbda108a491a652200aecebecb4a5bcaa39875927b3733708cb765f7e2"
        or plant_output.stat().st_size != 502
    ):
        raise AssertionError("PlantCaduceus pinned tokenizer asset bytes drifted")
    plant_asset = json.loads(plant_output.read_bytes())
    if "literal_token_ids" in plant_asset["model"]:
        raise AssertionError("ignored PlantCaduceus aliases leaked into runtime model")
    for source_text, expected_ids in (
        ("ACGTN", "3,4,5,6,2"),
        ("acgt", "3,4,5,6"),
        (" A C G T ", "3,4,5,6"),
    ):
        plant_runtime = subprocess.run(
            [
                str(args.runtime),
                "--verify-asset-no-special",
                str(plant_output.parent),
                plant_output.name,
                digest(plant_output),
                str(plant_output.stat().st_size),
                source_text,
                expected_ids,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if plant_runtime.returncode != 0:
            raise AssertionError(
                "PlantCaduceus audited DNA parity failed in C++: %s"
                % plant_runtime.stderr
            )

    gapped = copy.deepcopy(wrapper_only)
    gapped["added_tokens"].append(added_token(9, "[BPE]"))
    gap_manifest, gap_receipt, gap_output, gap_descriptor = create_case(
        args.work_dir / "hf-bpe-gapped-special",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(gapped))},
        appended_options,
    )
    gap_result = run_converter(
        args.converter, gap_manifest, gap_receipt, gap_output, gap_descriptor
    )
    if (
        gap_result.returncode == 0
        or "must be contiguous" not in gap_result.stderr
        or gap_output.exists()
        or gap_descriptor.exists()
    ):
        raise AssertionError("gapped appended HF special was accepted")
    repeated = run_converter(
        args.converter, manifest, receipt, output, descriptor, force=True
    )
    assert_success(repeated, "deterministic HF BPE conversion")
    if output.read_bytes() != first_payload or descriptor.read_bytes() != first_descriptor:
        raise AssertionError("repeated conversion changed canonical bytes")
    refused = run_converter(args.converter, manifest, receipt, output, descriptor)
    if refused.returncode == 0 or output.read_bytes() != first_payload:
        raise AssertionError("converter overwrote output without --force")

    tokenizer_config = canonical(
        {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "padding_side": "right",
            "model_max_length": 64,
        }
    )
    special_map = canonical(
        {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
        }
    )
    word_manifest, word_receipt, word_output, word_descriptor = create_case(
        args.work_dir / "hf-wordpiece",
        "huggingface-json",
        "wordpiece",
        {
            "tokenizer": ("tokenizer.json", canonical(wordpiece_tokenizer())),
            "tokenizer_config": ("tokenizer_config.json", tokenizer_config),
            "special_tokens_map": ("special_tokens_map.json", special_map),
        },
        bpe_options,
    )
    assert_success(
        run_converter(
            args.converter,
            word_manifest,
            word_receipt,
            word_output,
            word_descriptor,
        ),
        "HF WordPiece conversion",
    )
    word = json.loads(word_output.read_bytes())
    if (
        word["kind"] != "wordpiece"
        or word["normalization"] != [{"op": "ascii-lowercase"}]
        or word["model"]["continuation_prefix"] != "##"
        or word["post_processor"]["prefix_ids"] != [1]
    ):
        raise AssertionError("HF WordPiece/Bert constructs were not preserved")
    check_descriptor(word_output, word_descriptor)

    byte_manifest, byte_receipt, byte_output, byte_descriptor = create_case(
        args.work_dir / "hf-byte-bpe",
        "huggingface-json",
        "byte-bpe",
        {"tokenizer": ("tokenizer.json", canonical(byte_bpe_tokenizer()))},
        {
            "special_tokens": special_pieces(unk="<unk>"),
            "padding_side": "none",
        },
    )
    assert_success(
        run_converter(
            args.converter,
            byte_manifest,
            byte_receipt,
            byte_output,
            byte_descriptor,
        ),
        "HF byte-BPE conversion",
    )
    byte_asset = json.loads(byte_output.read_bytes())
    if len(byte_asset["model"]["byte_encoder"]) != 256 or not byte_asset["model"][
        "add_prefix_space"
    ]:
        raise AssertionError("byte-BPE encoder table was not frozen")

    kmer_options = {
        "normalization": [{"op": "ascii-uppercase"}],
        "pre_tokenizer": {"kind": "none"},
        "model": {"k": 2, "stride": 2, "tail": "error", "unknown_policy": "error"},
        "post_processor": output_post([], [], "none", None),
        "special_tokens": special_ids(),
    }
    kmer_manifest, kmer_receipt, kmer_output, kmer_descriptor = create_case(
        args.work_dir / "vocab-kmer",
        "vocab-text",
        "kmer",
        {"vocab": ("vocab.txt", b"AA\nAC\nCG\nGT\n")},
        kmer_options,
    )
    assert_success(
        run_converter(
            args.converter,
            kmer_manifest,
            kmer_receipt,
            kmer_output,
            kmer_descriptor,
        ),
        "vocab k-mer conversion",
    )
    if json.loads(kmer_output.read_bytes())["vocab"][-1] != {"id": 3, "piece": "GT"}:
        raise AssertionError("vocab.txt order did not define canonical IDs")
    kmer_descriptor_value = json.loads(kmer_descriptor.read_text(encoding="utf-8"))
    runtime_root = args.work_dir / "runtime-artifact"
    runtime_asset = runtime_root / kmer_descriptor_value["tokenizer.path"]
    runtime_asset.parent.mkdir(parents=True)
    shutil.copyfile(kmer_output, runtime_asset)
    runtime_result = subprocess.run(
        [
            str(args.runtime),
            "--verify-asset",
            str(runtime_root),
            kmer_descriptor_value["tokenizer.path"],
            kmer_descriptor_value["tokenizer.sha256"],
            str(kmer_descriptor_value["tokenizer.size"]),
            "AACG",
            "0,2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if runtime_result.returncode != 0:
        raise AssertionError(
            "converter output did not load/encode in C++ runtime: %s"
            % runtime_result.stderr
        )

    longest_options = {
        "normalization": [{"op": "ascii-uppercase"}],
        "pre_tokenizer": {"kind": "none"},
        "model": {"unknown_policy": "unk", "match_special_literals": True},
        "post_processor": output_post([3], [], "right", 1),
        "special_tokens": special_ids(unk=0, pad=1, cls=3, mask=2),
    }
    longest_manifest, longest_receipt, longest_output, longest_descriptor = create_case(
        args.work_dir / "vocab-longest-match",
        "vocab-text",
        "longest-match",
        {
            "vocab": (
                "vocab.txt",
                b"[UNK]\n[PAD]\n[MASK]\n[CLS]\nAAAAAA\nA\nX\n",
            )
        },
        longest_options,
    )
    assert_success(
        run_converter(
            args.converter,
            longest_manifest,
            longest_receipt,
            longest_output,
            longest_descriptor,
        ),
        "vocab longest-match conversion",
    )
    longest_runtime = subprocess.run(
        [
            str(args.runtime),
            "--verify-asset",
            str(longest_output.parent),
            longest_output.name,
            digest(longest_output),
            str(longest_output.stat().st_size),
            "AAAAAAXA",
            "3,4,6,5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if longest_runtime.returncode != 0:
        raise AssertionError(
            "vocab longest-match output did not load/encode in C++ runtime: %s"
            % longest_runtime.stderr
        )

    boundary_vocab = b"[UNK]\n[PAD]\n[MASK]\n[CLS]\nAAAAAA\nA\n<eos>\n<bos>"
    boundary_options = {
        "normalization": [],
        "pre_tokenizer": {"kind": "none"},
        "model": {"unknown_policy": "unk", "match_special_literals": True},
        "post_processor": output_post([3], [], "right", 1),
        "special_tokens": special_ids(unk=0, pad=1, cls=3, mask=2),
        "audited_vocab_boundary": {
            "source_size": 8,
            "compiled_size": 6,
            "excluded_suffix": [
                {"id": 6, "piece": "<eos>", "input_policy": "reject"},
                {"id": 7, "piece": "<bos>", "input_policy": "reject"},
            ],
        },
    }
    (
        boundary_manifest,
        boundary_receipt,
        boundary_output,
        boundary_descriptor,
    ) = create_case(
        args.work_dir / "vocab-audited-boundary",
        "vocab-text",
        "longest-match",
        {"vocab": ("vocab.txt", boundary_vocab)},
        boundary_options,
    )
    assert_success(
        run_converter(
            args.converter,
            boundary_manifest,
            boundary_receipt,
            boundary_output,
            boundary_descriptor,
        ),
        "audited vocab boundary conversion",
    )
    boundary_asset = json.loads(boundary_output.read_bytes())
    if (
        len(boundary_asset["vocab"]) != 6
        or boundary_asset["vocab"][-1] != {"id": 5, "piece": "A"}
        or any(
            item["piece"] in ("<eos>", "<bos>")
            for item in boundary_asset["vocab"]
        )
    ):
        raise AssertionError("audited source-only vocab suffix reached the runtime asset")
    boundary_runtime = subprocess.run(
        [
            str(args.runtime),
            "--verify-asset",
            str(boundary_output.parent),
            boundary_output.name,
            digest(boundary_output),
            str(boundary_output.stat().st_size),
            "AAAAAAA",
            "3,4,5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if boundary_runtime.returncode != 0:
        raise AssertionError(
            "audited vocab boundary output failed in C++: %s"
            % boundary_runtime.stderr
        )

    vocab_wordpiece_options = {
        "normalization": [{"op": "ascii-uppercase"}],
        "pre_tokenizer": {"kind": "ascii-whitespace"},
        "model": {"continuation_prefix": "##", "max_input_chars_per_word": 64},
        "post_processor": output_post([], [], "none", None),
        "special_tokens": special_ids(unk=0, pad=1),
    }
    vocab_word_manifest, vocab_word_receipt, vocab_word_output, vocab_word_descriptor = create_case(
        args.work_dir / "vocab-wordpiece",
        "vocab-text",
        "wordpiece",
        {"vocab": ("vocab.txt", b"[UNK]\n[PAD]\nAC\n##GT\n")},
        vocab_wordpiece_options,
    )
    assert_success(
        run_converter(
            args.converter,
            vocab_word_manifest,
            vocab_word_receipt,
            vocab_word_output,
            vocab_word_descriptor,
        ),
        "vocab WordPiece conversion",
    )
    if json.loads(vocab_word_output.read_bytes())["kind"] != "wordpiece":
        raise AssertionError("vocab.txt WordPiece kind was not preserved")

    for kind in ("single-nucleotide", "mixed", "biotoken", "kmer-bpe"):
        source_spec = custom_spec(kind)
        custom_manifest, custom_receipt, custom_output, custom_descriptor = create_case(
            args.work_dir / ("custom-" + kind),
            "custom",
            kind,
            {"spec": ("tokenizer-spec.json", canonical(source_spec))},
            {},
        )
        assert_success(
            run_converter(
                args.converter,
                custom_manifest,
                custom_receipt,
                custom_output,
                custom_descriptor,
            ),
            "custom %s conversion" % kind,
        )
        compiled = json.loads(custom_output.read_bytes())
        expected_kind = "longest-match" if kind in ("mixed", "biotoken") else kind
        if compiled["kind"] != expected_kind:
            raise AssertionError("custom %s mapped to wrong runtime kind" % kind)

    corrupt_root = args.work_dir / "corruptions"
    corrupt_root.mkdir()
    cases = []  # type: List[Tuple[str, Path, Path]]

    wrong_boundary_tail_manifest, wrong_boundary_tail_receipt, _, _ = create_case(
        corrupt_root / "audited-boundary-tail",
        "vocab-text",
        "longest-match",
        {"vocab": ("vocab.txt", boundary_vocab[:-5] + b"<bad>")},
        boundary_options,
    )
    cases.append(
        (
            "audited vocab excluded suffix content drift",
            wrong_boundary_tail_manifest,
            wrong_boundary_tail_receipt,
        )
    )
    wrong_boundary_id_options = copy.deepcopy(boundary_options)
    wrong_boundary_id_options["audited_vocab_boundary"]["excluded_suffix"][0][
        "id"
    ] = 7
    wrong_boundary_id_manifest, wrong_boundary_id_receipt, _, _ = create_case(
        corrupt_root / "audited-boundary-id",
        "vocab-text",
        "longest-match",
        {"vocab": ("vocab.txt", boundary_vocab)},
        wrong_boundary_id_options,
    )
    cases.append(
        (
            "audited vocab excluded suffix ID drift",
            wrong_boundary_id_manifest,
            wrong_boundary_id_receipt,
        )
    )
    wrong_boundary_policy_options = copy.deepcopy(boundary_options)
    wrong_boundary_policy_options["audited_vocab_boundary"]["excluded_suffix"][
        0
    ]["input_policy"] = "accept"
    wrong_boundary_policy_manifest, wrong_boundary_policy_receipt, _, _ = create_case(
        corrupt_root / "audited-boundary-policy",
        "vocab-text",
        "longest-match",
        {"vocab": ("vocab.txt", boundary_vocab)},
        wrong_boundary_policy_options,
    )
    cases.append(
        (
            "audited vocab excluded suffix input policy drift",
            wrong_boundary_policy_manifest,
            wrong_boundary_policy_receipt,
        )
    )
    wrong_boundary_size_options = copy.deepcopy(boundary_options)
    wrong_boundary_size_options["audited_vocab_boundary"]["source_size"] = 9
    wrong_boundary_size_manifest, wrong_boundary_size_receipt, _, _ = create_case(
        corrupt_root / "audited-boundary-size",
        "vocab-text",
        "longest-match",
        {"vocab": ("vocab.txt", boundary_vocab)},
        wrong_boundary_size_options,
    )
    cases.append(
        (
            "audited vocab source size drift",
            wrong_boundary_size_manifest,
            wrong_boundary_size_receipt,
        )
    )

    bad_hash_manifest = args.work_dir / "hf-bpe" / "manifest.json"
    bad_hash_receipt = corrupt_root / "bad-hash-receipt.json"
    bad_hash = json.loads(receipt.read_text(encoding="utf-8"))
    bad_hash["files"][0]["sha256"] = "0" * 64
    write_json(bad_hash_receipt, bad_hash)
    cases.append(("hash mismatch", bad_hash_manifest, bad_hash_receipt))

    extra_receipt = corrupt_root / "extra-receipt.json"
    extra = json.loads(receipt.read_text(encoding="utf-8"))
    extra["files"].append(
        {
            "role": "unexpected",
            "name": "extra.json",
            "path": str((corrupt_root / "extra.json").resolve()),
            "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
    (corrupt_root / "extra.json").write_bytes(b"")
    write_json(extra_receipt, extra)
    cases.append(("extra asset", bad_hash_manifest, extra_receipt))

    unknown_hf = bpe_tokenizer()
    unknown_hf["normalizer"] = {"type": "NFC"}
    unknown_manifest, unknown_receipt, _, _ = create_case(
        corrupt_root / "unknown-hf",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(unknown_hf))},
        bpe_options,
    )
    cases.append(("unknown HF construct", unknown_manifest, unknown_receipt))

    unapproved_backend_length = copy.deepcopy(backend_length_state)
    unapproved_length_manifest, unapproved_length_receipt, _, _ = create_case(
        corrupt_root / "unapproved-backend-length-state",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(unapproved_backend_length))},
        bpe_options,
    )
    cases.append(
        (
            "unapproved HF backend length state",
            unapproved_length_manifest,
            unapproved_length_receipt,
        )
    )

    false_backend_length_options = dict(
        bpe_options, ignore_hf_backend_truncation_padding=False
    )
    false_length_manifest, false_length_receipt, _, _ = create_case(
        corrupt_root / "false-backend-length-opt-in",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(bpe_tokenizer()))},
        false_backend_length_options,
    )
    cases.append(
        (
            "false HF backend length opt-in",
            false_length_manifest,
            false_length_receipt,
        )
    )

    mismatched_backend_length = copy.deepcopy(backend_length_state)
    mismatched_backend_length["padding"]["strategy"] = {"Fixed": 1999}
    mismatched_length_manifest, mismatched_length_receipt, _, _ = create_case(
        corrupt_root / "mismatched-backend-length-state",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(mismatched_backend_length))},
        backend_length_options,
    )
    cases.append(
        (
            "mismatched HF backend length state",
            mismatched_length_manifest,
            mismatched_length_receipt,
        )
    )

    decoder_extra = bpe_tokenizer()
    decoder_extra["decoder"] = {
        "type": "BPE",
        "suffix": "</w>",
        "unexpected": False,
    }
    decoder_manifest, decoder_receipt, _, _ = create_case(
        corrupt_root / "legacy-bpe-decoder-extra-field",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(decoder_extra))},
        bpe_options,
    )
    cases.append(("legacy BPE decoder unknown field", decoder_manifest, decoder_receipt))

    duplicate_merge = bpe_tokenizer()
    duplicate_merge["model"]["merges"] = [["A", "C"], ["A", "C"]]
    merge_manifest, merge_receipt, _, _ = create_case(
        corrupt_root / "duplicate-merge",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(duplicate_merge))},
        bpe_options,
    )
    cases.append(("duplicate merge", merge_manifest, merge_receipt))

    duplicate_id = bpe_tokenizer()
    duplicate_id["model"]["vocab"]["C"] = 4
    id_manifest, id_receipt, _, _ = create_case(
        corrupt_root / "duplicate-id",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(duplicate_id))},
        bpe_options,
    )
    cases.append(("duplicate ID", id_manifest, id_receipt))

    byte_fallback_bpe = bpe_tokenizer()
    byte_fallback_bpe["model"]["byte_fallback"] = True
    byte_fallback_manifest, byte_fallback_receipt, _, _ = create_case(
        corrupt_root / "bpe-byte-fallback",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(byte_fallback_bpe))},
        bpe_options,
    )
    cases.append(("unsupported BPE byte fallback", byte_fallback_manifest, byte_fallback_receipt))

    unapproved_omnina = omnina_tokenizer()
    unapproved_omnina_manifest, unapproved_omnina_receipt, _, _ = create_case(
        corrupt_root / "unapproved-omnina-inert-fields",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(unapproved_omnina))},
        {
            "special_tokens": special_pieces(
                unk="<unk>", pad="</s>", bos="<s>", eos="</s>"
            ),
            "padding_side": "left",
        },
    )
    cases.append(
        (
            "unapproved OmniNA encode-inert fields",
            unapproved_omnina_manifest,
            unapproved_omnina_receipt,
        )
    )

    false_omnina_options = copy.deepcopy(omnina_options)
    false_omnina_options["ignore_hf_omnina_encode_inert_fields"] = False
    false_omnina_manifest, false_omnina_receipt, _, _ = create_case(
        corrupt_root / "false-omnina-inert-opt-in",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(omnina_tokenizer()))},
        false_omnina_options,
    )
    cases.append(
        ("false OmniNA inert opt-in", false_omnina_manifest, false_omnina_receipt)
    )

    byte_piece_omnina = omnina_tokenizer()
    byte_piece_omnina["model"]["vocab"]["<0x41>"] = 6
    byte_piece_omnina["added_tokens"][-1]["id"] = 7
    byte_piece_manifest, byte_piece_receipt, _, _ = create_case(
        corrupt_root / "omnina-byte-piece",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(byte_piece_omnina))},
        omnina_options,
    )
    cases.append(("OmniNA byte piece", byte_piece_manifest, byte_piece_receipt))

    no_unk_omnina = omnina_tokenizer()
    no_unk_omnina["model"]["unk_token"] = None
    no_unk_manifest, no_unk_receipt, _, _ = create_case(
        corrupt_root / "omnina-missing-unk",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(no_unk_omnina))},
        omnina_options,
    )
    cases.append(("OmniNA missing unknown", no_unk_manifest, no_unk_receipt))

    unfused_omnina = omnina_tokenizer()
    unfused_omnina["model"]["fuse_unk"] = False
    unfused_manifest, unfused_receipt, _, _ = create_case(
        corrupt_root / "omnina-unfused-unknown",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(unfused_omnina))},
        omnina_options,
    )
    cases.append(("OmniNA fuse_unk drift", unfused_manifest, unfused_receipt))

    decoder_omnina = omnina_tokenizer()
    decoder_omnina["decoder"]["decoders"][0]["content"] = "_"
    decoder_omnina_manifest, decoder_omnina_receipt, _, _ = create_case(
        corrupt_root / "omnina-decoder-drift",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(decoder_omnina))},
        omnina_options,
    )
    cases.append(
        (
            "OmniNA decoder drift",
            decoder_omnina_manifest,
            decoder_omnina_receipt,
        )
    )

    normalized_appended_omnina = omnina_tokenizer()
    normalized_appended_omnina["added_tokens"][-1]["normalized"] = True
    normalized_appended_manifest, normalized_appended_receipt, _, _ = create_case(
        corrupt_root / "omnina-normalized-appended",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(normalized_appended_omnina))},
        omnina_options,
    )
    cases.append(
        (
            "OmniNA normalized appended literal",
            normalized_appended_manifest,
            normalized_appended_receipt,
        )
    )

    regex_byte_bpe = byte_bpe_tokenizer()
    regex_byte_bpe["pre_tokenizer"]["use_regex"] = True
    regex_manifest, regex_receipt, _, _ = create_case(
        corrupt_root / "bytelevel-regex",
        "huggingface-json",
        "byte-bpe",
        {"tokenizer": ("tokenizer.json", canonical(regex_byte_bpe))},
        {
            "special_tokens": special_pieces(unk="<unk>"),
            "padding_side": "none",
        },
    )
    cases.append(("unrepresented ByteLevel regex", regex_manifest, regex_receipt))

    mismatched_alias = bpe_tokenizer()
    mismatched_alias["normalizer"] = None
    mismatched_alias["added_tokens"] = [added_token(1, "[UNK]")]
    mismatch_manifest, mismatch_receipt, _, _ = create_case(
        corrupt_root / "base-vocab-added-token-id-mismatch",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(mismatched_alias))},
        bpe_options,
    )
    cases.append(("base-vocab added-token ID mismatch", mismatch_manifest, mismatch_receipt))

    normalized_flag_alias = copy.deepcopy(gena_alias)
    normalized_flag_alias["added_tokens"][4]["normalized"] = True
    normalized_alias_manifest, normalized_alias_receipt, _, _ = create_case(
        corrupt_root / "normalized-added-token-flag",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(normalized_flag_alias))},
        alias_options,
    )
    cases.append(
        (
            "normalized added-token flag",
            normalized_alias_manifest,
            normalized_alias_receipt,
        )
    )

    too_many_aliases = bpe_tokenizer()
    too_many_aliases["normalizer"] = None
    too_many_aliases["added_tokens"] = []
    for index in range(33):
        token_id = len(too_many_aliases["model"]["vocab"])
        piece = "[SPECIAL-%d]" % index
        too_many_aliases["model"]["vocab"][piece] = token_id
        too_many_aliases["added_tokens"].append(added_token(token_id, piece))
    too_many_manifest, too_many_receipt, _, _ = create_case(
        corrupt_root / "too-many-base-vocab-added-token-aliases",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(too_many_aliases))},
        bpe_options,
    )
    cases.append(("too many base-vocab added-token aliases", too_many_manifest, too_many_receipt))

    stripped_added = bpe_tokenizer()
    stripped_added["added_tokens"] = [added_token(0, "[UNK]")]
    stripped_added["added_tokens"][0]["lstrip"] = True
    added_manifest, added_receipt, _, _ = create_case(
        corrupt_root / "added-token-flags",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(stripped_added))},
        bpe_options,
    )
    cases.append(("unrepresented added-token flags", added_manifest, added_receipt))

    explicit_false_options = copy.deepcopy(plant_options)
    explicit_false_options["ignore_hf_base_vocab_added_special_literals"] = False
    explicit_false_manifest, explicit_false_receipt, _, _ = create_case(
        corrupt_root / "false-ignore-base-vocab-added-special-literals",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(plant_caduceus_tokenizer()))},
        explicit_false_options,
    )
    cases.append(
        (
            "false base-vocab added-special literal opt-in",
            explicit_false_manifest,
            explicit_false_receipt,
        )
    )

    ignored_appended = plant_caduceus_tokenizer()
    ignored_appended["added_tokens"].append(added_token(7, "[EXTRA]"))
    ignored_appended_manifest, ignored_appended_receipt, _, _ = create_case(
        corrupt_root / "ignored-base-vocab-alias-with-appended-token",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(ignored_appended))},
        plant_options,
    )
    cases.append(
        (
            "ignored base-vocab aliases with appended token",
            ignored_appended_manifest,
            ignored_appended_receipt,
        )
    )

    ignored_without_aliases = plant_caduceus_tokenizer()
    ignored_without_aliases["added_tokens"] = []
    ignored_without_aliases_manifest, ignored_without_aliases_receipt, _, _ = create_case(
        corrupt_root / "ignored-base-vocab-aliases-without-aliases",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(ignored_without_aliases))},
        plant_options,
    )
    cases.append(
        (
            "ignored base-vocab aliases without aliases",
            ignored_without_aliases_manifest,
            ignored_without_aliases_receipt,
        )
    )

    ignored_without_normalizer = plant_caduceus_tokenizer()
    ignored_without_normalizer["normalizer"] = None
    ignored_without_normalizer_manifest, ignored_without_normalizer_receipt, _, _ = create_case(
        corrupt_root / "ignored-base-vocab-aliases-without-normalizer",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(ignored_without_normalizer))},
        plant_options,
    )
    cases.append(
        (
            "ignored base-vocab aliases without normalizer",
            ignored_without_normalizer_manifest,
            ignored_without_normalizer_receipt,
        )
    )

    ignored_mismatched_alias = plant_caduceus_tokenizer()
    ignored_mismatched_alias["added_tokens"][0]["id"] = 1
    ignored_mismatched_manifest, ignored_mismatched_receipt, _, _ = create_case(
        corrupt_root / "ignored-base-vocab-alias-id-mismatch",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(ignored_mismatched_alias))},
        plant_options,
    )
    cases.append(
        (
            "ignored base-vocab alias ID mismatch",
            ignored_mismatched_manifest,
            ignored_mismatched_receipt,
        )
    )

    ignored_flagged_alias = plant_caduceus_tokenizer()
    ignored_flagged_alias["added_tokens"][0]["lstrip"] = True
    ignored_flagged_manifest, ignored_flagged_receipt, _, _ = create_case(
        corrupt_root / "ignored-base-vocab-alias-flags",
        "huggingface-json",
        "bpe",
        {"tokenizer": ("tokenizer.json", canonical(ignored_flagged_alias))},
        plant_options,
    )
    cases.append(
        (
            "ignored base-vocab alias flags",
            ignored_flagged_manifest,
            ignored_flagged_receipt,
        )
    )

    invalid_padding = custom_spec("single-nucleotide")
    invalid_padding["post_processor"]["padding"] = {"side": "none", "pad_id": 0}
    invalid_padding["special_tokens"]["pad"] = 0
    padding_manifest, padding_receipt, _, _ = create_case(
        corrupt_root / "none-padding-id",
        "custom",
        "single-nucleotide",
        {"spec": ("tokenizer-spec.json", canonical(invalid_padding))},
        {},
    )
    cases.append(("pad ID with disabled padding", padding_manifest, padding_receipt))

    invalid_custom = custom_spec("single-nucleotide")
    invalid_custom["vocab"][0]["id"] = 1 << 32
    invalid_manifest, invalid_receipt, _, _ = create_case(
        corrupt_root / "invalid-id",
        "custom",
        "single-nucleotide",
        {"spec": ("tokenizer-spec.json", canonical(invalid_custom))},
        {},
    )
    cases.append(("invalid uint32 ID", invalid_manifest, invalid_receipt))

    source_tamper = corrupt_root / "source-tamper"
    tamper_manifest, tamper_receipt, _, _ = create_case(
        source_tamper,
        "custom",
        "single-nucleotide",
        {"spec": ("tokenizer-spec.json", canonical(custom_spec("single-nucleotide")))},
        {},
    )
    with (source_tamper / "tokenizer-spec.json").open("ab") as output_stream:
        output_stream.write(b" ")
    cases.append(("modified source bytes", tamper_manifest, tamper_receipt))

    missing_receipt = corrupt_root / "missing-receipt.json"
    missing = json.loads(receipt.read_text(encoding="utf-8"))
    missing["files"] = []
    write_json(missing_receipt, missing)
    cases.append(("missing receipt asset", bad_hash_manifest, missing_receipt))

    sentinel_output = corrupt_root / "sentinel-output.json"
    sentinel_descriptor = corrupt_root / "sentinel-descriptor.json"
    sentinel_output.write_bytes(b"sentinel-output\n")
    sentinel_descriptor.write_bytes(b"sentinel-descriptor\n")
    for label, bad_manifest, bad_receipt_path in cases:
        before_output = sentinel_output.read_bytes()
        before_descriptor = sentinel_descriptor.read_bytes()
        rejected = run_converter(
            args.converter,
            bad_manifest,
            bad_receipt_path,
            sentinel_output,
            sentinel_descriptor,
            force=True,
        )
        if rejected.returncode == 0:
            raise AssertionError("converter accepted %s" % label)
        if (
            sentinel_output.read_bytes() != before_output
            or sentinel_descriptor.read_bytes() != before_descriptor
        ):
            raise AssertionError("failed %s conversion was not atomic" % label)

    repository_tokenizers = (
        Path(__file__).resolve().parents[1] / "configs" / "tokenizers"
    )
    opted_in_manifests = []
    for tokenizer_manifest_path in sorted(repository_tokenizers.glob("*.json")):
        tokenizer_manifest = json.loads(
            tokenizer_manifest_path.read_text(encoding="utf-8")
        )
        if (
            tokenizer_manifest.get("options", {}).get(
                "ignore_hf_base_vocab_added_special_literals"
            )
            is True
        ):
            opted_in_manifests.append(tokenizer_manifest_path.name)
    if opted_in_manifests != ["geneb-plant-caduceus-bpe-v1.json"]:
        raise AssertionError(
            "base-vocab added-special literal ignore opt-in escaped PlantCaduceus: %r"
            % opted_in_manifests
        )
    omnina_opted_in_manifests = []
    for tokenizer_manifest_path in sorted(repository_tokenizers.glob("*.json")):
        tokenizer_manifest = json.loads(
            tokenizer_manifest_path.read_text(encoding="utf-8")
        )
        options = tokenizer_manifest.get("options", {})
        if (
            options.get("ignore_hf_normalized_base_vocab_added_special_literals")
            is True
            or options.get("ignore_hf_omnina_encode_inert_fields") is True
        ):
            if not (
                options.get(
                    "ignore_hf_normalized_base_vocab_added_special_literals"
                )
                is True
                and options.get("ignore_hf_omnina_encode_inert_fields") is True
            ):
                raise AssertionError("OmniNA compiler opt-ins were separated")
            omnina_opted_in_manifests.append(tokenizer_manifest_path.name)
    if omnina_opted_in_manifests != ["geneb-omnina-220m-bpe-v1.json"]:
        raise AssertionError(
            "OmniNA tokenizer opt-in escaped its exact manifest: %r"
            % omnina_opted_in_manifests
        )
    boundary_opted_in_manifests = []
    for tokenizer_manifest_path in sorted(repository_tokenizers.glob("*.json")):
        tokenizer_manifest = json.loads(
            tokenizer_manifest_path.read_text(encoding="utf-8")
        )
        if "audited_vocab_boundary" in tokenizer_manifest.get("options", {}):
            boundary_opted_in_manifests.append(tokenizer_manifest_path.name)
    if boundary_opted_in_manifests != [
        "geneb-agro-nt-1b-6mer-v1.json",
        "geneb-nt-2-5b-ms-6mer-v1.json",
    ]:
        raise AssertionError(
            "audited vocab boundary escaped its exact NT profiles: %r"
            % boundary_opted_in_manifests
        )
    agro_repository_manifest = (
        repository_tokenizers / "geneb-agro-nt-1b-6mer-v1.json"
    )
    if (
        digest(agro_repository_manifest)
        != "a19b30c4284c792be67a79dba336156d0cdebd2b2f4d57ea4d1a7cc62975cb26"
    ):
        raise AssertionError("Agro-NT-1B production compiler manifest drifted")
    nt25_repository_manifest = (
        repository_tokenizers / "geneb-nt-2-5b-ms-6mer-v1.json"
    )
    if (
        digest(nt25_repository_manifest)
        != "a19b30c4284c792be67a79dba336156d0cdebd2b2f4d57ea4d1a7cc62975cb26"
        or nt25_repository_manifest == agro_repository_manifest
    ):
        raise AssertionError("NT-2.5B dedicated compiler manifest drifted")
    plant_repository_manifest = (
        repository_tokenizers / "geneb-plant-caduceus-bpe-v1.json"
    )
    if (
        digest(plant_repository_manifest)
        != "348b2a93e9c035dba994cf3944a29d28ab193c1ae2539ab1e1048d2a907a7b69"
    ):
        raise AssertionError("PlantCaduceus production compiler manifest drifted")
    gena_repository_manifest = (
        repository_tokenizers / "geneb-gena-lm-t2t-multi-bpe-v1.json"
    )
    if (
        digest(gena_repository_manifest)
        != "eefe067af743b49e6bc5e219579ecc9a22b9d6523793bf28caf705ee050ffe99"
    ):
        raise AssertionError("GENA-LM-T2T-Multi compiler manifest drifted")

    grammar = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ast,pathlib; ast.parse(pathlib.Path(%r).read_text(), feature_version=8)"
            % str(args.converter),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if grammar.returncode != 0:
        raise AssertionError("converter is not Python 3.8 grammar: %s" % grammar.stderr)

    print("tokenizer asset converter contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
