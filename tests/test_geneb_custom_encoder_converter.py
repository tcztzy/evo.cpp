#!/usr/bin/env python3
"""Offline contract tests for the strict custom-encoder converter."""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical(value))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elements(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def tensor_payload(spec: Any, topology: Mapping[str, Any]) -> bytes:
    count = elements(spec.shape)
    if "rot_emb.inv_freq" in spec.name:
        return b"".join(
            struct.pack(
                "<f",
                struct.unpack(
                    "<e",
                    struct.pack(
                        "<e",
                        1.0
                        / (
                            topology["rope_base"]
                            ** (float(2 * index) / topology["head_dim"])
                        ),
                    ),
                )[0],
            )
            for index in range(count)
        )
    seed = sum(spec.name.encode("ascii")) % 83
    return b"".join(
        struct.pack("<f", ((seed + index * 7) % 97 - 48) * 0.001)
        for index in range(count)
    )


def write_safetensors(path: Path, specs: Sequence[Any], topology: Mapping[str, Any]) -> None:
    payload_by_name = {
        spec.name: tensor_payload(spec, topology) for spec in specs
    }  # type: Dict[str, bytes]
    if topology["variant"] == "genomics-fm":
        payload_by_name["cls.predictions.decoder.weight"] = payload_by_name[
            "bert.embeddings.word_embeddings.weight"
        ]
    header = {}  # type: Dict[str, Any]
    payloads = []  # type: List[bytes]
    offset = 0
    for spec in sorted(specs, key=lambda item: item.name):
        payload = payload_by_name[spec.name]
        header[spec.name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    padded = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + b"".join(payloads))


def read_safetensors(path: Path) -> Tuple[Dict[str, Any], bytes]:
    payload = path.read_bytes()
    size = struct.unpack("<Q", payload[:8])[0]
    return json.loads(payload[8 : 8 + size].decode("utf-8")), payload[8 + size :]


def rewrite_safetensors(path: Path, header: Mapping[str, Any], data: bytes) -> None:
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    padded = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + data)


def tiny_topology(variant: str) -> Dict[str, Any]:
    luca = variant == "lucaone"
    return {
        "variant": variant,
        "vocab_size": 39 if luca else 7268,
        "tokenizer_vocab_size": 39 if luca else 4100,
        "hidden_size": 8 if luca else 4,
        "num_layers": 1,
        "num_attention_heads": 2,
        "head_dim": 4 if luca else 2,
        "inner_size": 6,
        "max_seqlen": 8,
        "type_vocab_size": 2,
        "pad_token_id": 0 if luca else 3,
        "cls_token_id": 2 if luca else 1,
        "sep_token_id": 3 if luca else 2,
        "layer_norm_epsilon": 1.0e-5 if luca else 1.0e-12,
        "rope_base": 10000.0 if luca else 0.0,
        "position_encoding": "rope-split-half" if luca else "absolute",
        "norm_placement": "pre" if luca else "post",
        "qkv_layout": "separate" if luca else "fused-qkv",
        "mlp_kind": "gelu" if luca else "gated-gelu",
        "pooling": "attention-mask-mean" if luca else "cls-token",
        "attention_bias": True,
        "mlp_input_bias": luca,
        "mlp_output_bias": True,
        "embedding_layer_norm": not luca,
        "final_layer_norm": luca,
        "unpad_masked_tokens": not luca,
        "token_type_embeddings": True,
        "weight_dtype": "F32",
    }


def source_config(topology: Mapping[str, Any]) -> Dict[str, Any]:
    common = {
        "vocab_size": topology["vocab_size"],
        "hidden_size": topology["hidden_size"],
        "num_hidden_layers": topology["num_layers"],
        "num_attention_heads": topology["num_attention_heads"],
        "max_position_embeddings": topology["max_seqlen"],
        "type_vocab_size": 2,
        "torch_dtype": "float32",
        "hidden_act": "gelu",
    }
    if topology["variant"] == "lucaone":
        common.update(
            {
                "ffn_dim": topology["inner_size"],
                "pad_token_id": 0,
                "bos_token_id": 2,
                "sep_token_id": 3,
                "no_position_embeddings": True,
                "no_token_type_embeddings": False,
                "use_embed_layer_norm": False,
                "use_last_layer_norm": True,
                "embed_scale": 1.0,
                "token_dropout": False,
                "layer_norm_eps": 1.0e-12,
            }
        )
    else:
        common.update(
            {
                "intermediate_size": topology["inner_size"],
                "pad_token_id": 0,
                "position_embedding_type": "absolute",
                "layer_norm_eps": 1.0e-12,
                "attention_probs_dropout_prob": 0.0,
            }
        )
    return common


def tokenizer_asset(topology: Mapping[str, Any]) -> Dict[str, Any]:
    luca = topology["variant"] == "lucaone"
    if luca:
        pieces = ["[PAD]", "[UNUSED-1]", "[CLS]", "[SEP]", "[MASK]", "A", "T", "C", "G", "[GENE-UNK]"]
        pieces.extend("[UNUSED-%d]" % index for index in range(10, 39))
        model = {"unknown_policy": "unk", "match_special_literals": False}
        normalization = [{"op": "ascii-uppercase"}, {"op": "u-to-t"}]
        pre = {"kind": "none"}
        special = {"unk": 9, "pad": 0, "bos": None, "eos": None, "cls": 2, "sep": 3, "mask": 4}
    else:
        pieces = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "A", "C", "G", "T", "AC"]
        pieces.extend("[UNUSED-%d]" % index for index in range(10, 4096))
        pieces.extend(["[BPE]", "[KMER]", "[ALIBI]", "[APE]"])
        model = {
            "merges": [["A", "C"]],
            "literal_token_ids": [0, 1, 2, 3, 4, 4096, 4097, 4098, 4099],
        }
        normalization = []
        pre = {"kind": "hf-whitespace-ascii"}
        special = {"unk": 0, "pad": 3, "bos": None, "eos": None, "cls": 1, "sep": 2, "mask": 4}
    return {
        "format": "evo-tokenizer-v1",
        "kind": "character" if luca else "bpe",
        "normalization": normalization,
        "pre_tokenizer": pre,
        "model": model,
        "post_processor": {
            "prefix_ids": [topology["cls_token_id"]],
            "suffix_ids": [topology["sep_token_id"]],
            "padding": {"side": "right", "pad_id": topology["pad_token_id"]},
        },
        "special_tokens": special,
        "vocab": [{"id": index, "piece": piece} for index, piece in enumerate(pieces)],
    }


def receipt_entry(path: Path, name: str) -> Dict[str, Any]:
    return {"name": name, "size": path.stat().st_size, "sha256": digest(path), "path": str(path.resolve())}


def update_catalog(catalog: Dict[str, Any], profile: Mapping[str, Any]) -> None:
    topology = profile["topology"]
    entry = next(item for item in catalog["models"] if item["runtime_id"] == profile["runtime_id"])
    entry["tokenizer"]["max_tokens"] = topology["max_seqlen"]
    entry["context"]["declared_max_tokens"] = topology["max_seqlen"]
    entry["context"]["reference_max_tokens"] = topology["max_seqlen"]
    for preset in entry["embedding_presets"].values():
        preset["output_width"] = topology["hidden_size"]


def make_cases(work: Path, converter: Any, production_profiles: Path, production_catalog: Path) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    manifest = json.loads(production_profiles.read_text(encoding="utf-8"))
    catalog = json.loads(production_catalog.read_text(encoding="utf-8"))
    cases = []  # type: List[Dict[str, Any]]
    for profile in manifest["models"]:
        variant = "lucaone" if profile["runtime_id"] == "geneb-lucaone" else "genomics-fm"
        topology = tiny_topology(variant)
        profile["topology"] = topology
        profile["source_format"] = "safetensors"
        root = work / variant
        root.mkdir(parents=True)
        config_path = root / "config.json"
        write_json(config_path, source_config(topology))
        profile["config_sha256"] = digest(config_path)
        source_specs, runtime_specs = converter.source_specs(profile)
        weight_path = root / "model.safetensors"
        write_safetensors(weight_path, source_specs, topology)
        profile["source_files"] = [
            {key: value for key, value in receipt_entry(config_path, "config.json").items() if key != "path"},
            {key: value for key, value in receipt_entry(weight_path, "model.safetensors").items() if key != "path"},
        ]
        tokenizer_manifest = root / "tokenizer-manifest.json"
        write_json(tokenizer_manifest, {"fixture": variant})
        profile["tokenizer_manifest"] = str(tokenizer_manifest.relative_to(work))
        asset = root / "tokenizer.json"
        write_json(asset, tokenizer_asset(topology))
        descriptor = root / "tokenizer-descriptor.json"
        write_json(
            descriptor,
            {
                "converter.schema": "evo-tokenizer-conversion-receipt",
                "converter.version": 1,
                "compiler_manifest_sha256": digest(tokenizer_manifest),
                "source_receipt_contract_sha256": "5" * 64,
                "tokenizer.profile": "evo-tokenizer-v1",
                "tokenizer.path": "tokenizer.json",
                "tokenizer.sha256": digest(asset),
                "tokenizer.size": asset.stat().st_size,
            },
        )
        receipt = root / "source-receipt.json"
        receipt_value = {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": profile["runtime_id"],
            "source_kind": profile["source_kind"],
            "files": [
                receipt_entry(config_path, "config.json"),
                receipt_entry(weight_path, "model.safetensors"),
            ],
        }
        if profile["source_kind"] == "huggingface":
            receipt_value.update(
                {
                    "repo": profile["repo"],
                    "requested_revision": profile["requested_revision"],
                    "resolved_revision": profile["revision"],
                }
            )
        else:
            receipt_value["source_url"] = profile["source_url"]
        write_json(receipt, receipt_value)
        update_catalog(catalog, profile)
        cases.append(
            {
                "variant": variant,
                "profile": profile,
                "root": root,
                "receipt": receipt,
                "descriptor": descriptor,
                "source_specs": source_specs,
                "runtime_specs": runtime_specs,
            }
        )
    profiles_path = work / "profiles.json"
    catalog_path = work / "catalog.json"
    write_json(profiles_path, manifest)
    write_json(catalog_path, catalog)
    return profiles_path, catalog_path, cases


def command(converter: Path, profiles: Path, catalog: Path, case: Mapping[str, Any], output: Path) -> List[str]:
    return [
        sys.executable,
        str(converter),
        "--receipt",
        str(case["receipt"]),
        "--catalog",
        str(catalog),
        "--profiles",
        str(profiles),
        "--tokenizer-descriptor",
        str(case["descriptor"]),
        "--output",
        str(output),
    ]


def require_success(arguments: Sequence[str], label: str) -> subprocess.CompletedProcess:
    result = subprocess.run(arguments, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise AssertionError("{} failed:\n{}{}".format(label, result.stdout, result.stderr))
    return result


def require_failure(arguments: Sequence[str], output: Path, phrase: str, label: str) -> None:
    if output.exists():
        output.unlink()
    result = subprocess.run(arguments, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode == 0 or output.exists() or phrase not in result.stdout + result.stderr:
        raise AssertionError("{} did not fail atomically with {!r}:\n{}{}".format(label, phrase, result.stdout, result.stderr))


def require_runtime_failure(arguments: Sequence[str], phrase: str, label: str) -> None:
    result = subprocess.run(arguments, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode == 0 or phrase not in result.stdout + result.stderr:
        raise AssertionError("{} did not fail with {!r}:\n{}{}".format(label, phrase, result.stdout, result.stderr))


def verify_production(converter: Any, path: Path) -> None:
    profiles, _ = converter.load_profiles(path)
    expected = {
        "geneb-lucaone": ("lucaone", 344, 350, 39, 39),
        "geneb-genomics-fm": ("genomics-fm", 137, 143, 7268, 4100),
    }
    for runtime_id, values in expected.items():
        profile = profiles[runtime_id]
        source_names = {spec.name for spec in converter.source_specs(profile)[0]}
        actual = (
            profile["topology"]["variant"],
            len(converter.canonical_specs(profile["topology"])),
            len(converter.source_specs(profile)[0]),
            profile["topology"]["vocab_size"],
            profile["topology"]["tokenizer_vocab_size"],
        )
        if actual != values:
            raise AssertionError("production profile differs: {} {!r}".format(runtime_id, actual))
        if runtime_id == "geneb-genomics-fm" and (
            "cls.predictions.decoder.bias" not in source_names
            or "cls.predictions.bias" in source_names
        ):
            raise AssertionError("Genomics-FM pinned MLM decoder bias key differs")
        if runtime_id == "geneb-lucaone":
            expected_inverse = {
                "lucaone.encoder.layers.%d.self_attn.rot_emb.inv_freq" % layer
                for layer in range(20)
            }
            actual_inverse = {
                name for name in source_names if name.endswith("self_attn.rot_emb.inv_freq")
            }
            if actual_inverse != expected_inverse:
                raise AssertionError("LucaOne rotary layer ordinals differ")


def source_corruption_case(
    work: Path,
    profiles: Path,
    case: Mapping[str, Any],
    weights: Path,
    label: str,
) -> Tuple[Path, Dict[str, Any]]:
    manifest = json.loads(profiles.read_text(encoding="utf-8"))
    profile = next(
        item for item in manifest["models"] if item["runtime_id"] == "geneb-lucaone"
    )
    profile["source_files"][1] = {
        "name": "model.safetensors",
        "size": weights.stat().st_size,
        "sha256": digest(weights),
    }
    corrupt_profiles = work / (label + "-profiles.json")
    write_json(corrupt_profiles, manifest)
    receipt = json.loads(case["receipt"].read_text(encoding="utf-8"))
    receipt["files"][1] = receipt_entry(weights, "model.safetensors")
    corrupt_receipt = case["root"] / (label + "-receipt.json")
    write_json(corrupt_receipt, receipt)
    corrupt_case = dict(case)
    corrupt_case["receipt"] = corrupt_receipt
    return corrupt_profiles, corrupt_case


def verify_artifact(path: Path, case: Mapping[str, Any]) -> None:
    header, _ = read_safetensors(path)
    metadata = header.pop("__metadata__")
    expected_names = {spec.name for spec in case["runtime_specs"]}
    if set(header) != expected_names:
        raise AssertionError("runtime tensor set differs")
    topology = case["profile"]["topology"]
    expected = {
        "runtime.profile": "s:geneb-custom-encoder-runtime-v1",
        "runtime.abi": "s:geneb-custom-encoder-safetensors-v1",
        "model.architecture": "s:GenebCustomEncoder",
        "custom.variant": "s:" + case["variant"],
        "runtime.tokenizer_vocabulary_size": "u:%d" % topology["tokenizer_vocab_size"],
        "custom.tokenizer_vocab_size": "u:%d" % topology["tokenizer_vocab_size"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise AssertionError("artifact metadata differs for " + key)
    omitted = {spec.name for spec in case["source_specs"]} - expected_names
    if omitted & set(header):
        raise AssertionError("MLM-only tensors leaked into runtime artifact")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--native-loader", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    sys.path.insert(0, str(args.converter.resolve().parent))
    spec = importlib.util.spec_from_file_location("custom_converter", args.converter)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import converter")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    verify_production(converter, args.profiles)
    out_of_range_special = tiny_topology("lucaone")
    out_of_range_special["vocab_size"] = 3
    out_of_range_special["tokenizer_vocab_size"] = 3
    try:
        converter.validate_topology(out_of_range_special, "corrupt topology")
    except converter.ConversionError as error:
        if "special token ID exceeds tokenizer vocabulary" not in str(error):
            raise
    else:
        raise AssertionError("out-of-range special token topology was accepted")
    profiles, catalog, cases = make_cases(args.work_dir, converter, args.profiles, args.catalog)

    for case in cases:
        output = case["root"] / "model.runtime.safetensors"
        require_success(command(args.converter, profiles, catalog, case, output), case["variant"] + " conversion")
        verify_artifact(output, case)
        duplicate = case["root"] / "duplicate.safetensors"
        require_success(command(args.converter, profiles, catalog, case, duplicate), case["variant"] + " deterministic conversion")
        if digest(output) != digest(duplicate):
            raise AssertionError("conversion is not deterministic")
        loaded = require_success([str(args.native_loader), str(output)], case["variant"] + " ModelFile/native load")
        if json.loads(loaded.stdout)["variant"] != case["variant"]:
            raise AssertionError("native loader variant differs")
        adapted = require_success(
            [str(args.native_loader), "--verify-cpu-adapter", str(output)],
            case["variant"] + " public CPU adapter",
        )
        adapter_result = json.loads(adapted.stdout)
        topology = case["profile"]["topology"]
        if (
            adapter_result["variant"] != case["variant"]
            or adapter_result["model_vocab"] != topology["vocab_size"]
            or adapter_result["tokenizer_vocab"] != topology["tokenizer_vocab_size"]
            or adapter_result["pooling"] != topology["pooling"]
        ):
            raise AssertionError("public CPU adapter contract differs")
        corrupted = case["root"] / "bad-common-tokenizer-vocab.safetensors"
        header, data = read_safetensors(output)
        metadata = header["__metadata__"]
        metadata["runtime.tokenizer_vocabulary_size"] = "u:%d" % (
            topology["tokenizer_vocab_size"] - 1
        )
        rewrite_safetensors(corrupted, header, data)
        require_runtime_failure(
            [str(args.native_loader), str(corrupted)],
            "runtime.tokenizer_vocabulary_size",
            case["variant"] + " common tokenizer-vocab corruption",
        )

    luca = cases[0]
    inverse_name = "lucaone.encoder.layers.0.self_attn.rot_emb.inv_freq"
    ideal_inverse = luca["root"] / "ideal-double-inv-freq.safetensors"
    header, data = read_safetensors(luca["root"] / "model.safetensors")
    inverse = header[inverse_name]
    start, end = inverse["data_offsets"]
    mutable = bytearray(data)
    mutable[start:end] = b"".join(
        struct.pack(
            "<f",
            1.0
            / (
                luca["profile"]["topology"]["rope_base"]
                ** (
                    float(2 * index)
                    / luca["profile"]["topology"]["head_dim"]
                )
            ),
        )
        for index in range(luca["profile"]["topology"]["head_dim"] // 2)
    )
    rewrite_safetensors(ideal_inverse, header, bytes(mutable))
    ideal_profiles, ideal_case = source_corruption_case(
        args.work_dir, profiles, luca, ideal_inverse, "ideal-double-inv-freq"
    )
    bad_output = luca["root"] / "ideal-double-inv-freq-output.safetensors"
    require_failure(
        command(args.converter, ideal_profiles, catalog, ideal_case, bad_output),
        bad_output,
        "rotary inv_freq buffer differs",
        "ideal-double rotary corruption",
    )

    byte_corruption = luca["root"] / "single-byte-inv-freq.safetensors"
    header, data = read_safetensors(luca["root"] / "model.safetensors")
    mutable = bytearray(data)
    mutable[header[inverse_name]["data_offsets"][0]] ^= 1
    rewrite_safetensors(byte_corruption, header, bytes(mutable))
    byte_profiles, byte_case = source_corruption_case(
        args.work_dir, profiles, luca, byte_corruption, "single-byte-inv-freq"
    )
    bad_output = luca["root"] / "single-byte-inv-freq-output.safetensors"
    require_failure(
        command(args.converter, byte_profiles, catalog, byte_case, bad_output),
        bad_output,
        "rotary inv_freq buffer differs",
        "single-byte rotary corruption",
    )

    missing_inverse = luca["root"] / "missing-layer-inv-freq.safetensors"
    write_safetensors(
        missing_inverse,
        [spec for spec in luca["source_specs"] if spec.name != inverse_name],
        luca["profile"]["topology"],
    )
    missing_profiles, missing_case = source_corruption_case(
        args.work_dir, profiles, luca, missing_inverse, "missing-layer-inv-freq"
    )
    bad_output = luca["root"] / "missing-layer-inv-freq-output.safetensors"
    require_failure(
        command(args.converter, missing_profiles, catalog, missing_case, bad_output),
        bad_output,
        "tensor manifest mismatch",
        "missing rotary layer corruption",
    )

    extra_inverse = luca["root"] / "extra-layer-inv-freq.safetensors"
    write_safetensors(
        extra_inverse,
        list(luca["source_specs"])
        + [
            converter.TensorSpec(
                "lucaone.encoder.layers.1.self_attn.rot_emb.inv_freq",
                "F32",
                (luca["profile"]["topology"]["head_dim"] // 2,),
            )
        ],
        luca["profile"]["topology"],
    )
    extra_profiles, extra_case = source_corruption_case(
        args.work_dir, profiles, luca, extra_inverse, "extra-layer-inv-freq"
    )
    bad_output = luca["root"] / "extra-layer-inv-freq-output.safetensors"
    require_failure(
        command(args.converter, extra_profiles, catalog, extra_case, bad_output),
        bad_output,
        "tensor manifest mismatch",
        "extra rotary layer corruption",
    )

    bad_receipt = luca["root"] / "bad-receipt.json"
    receipt = json.loads(luca["receipt"].read_text(encoding="utf-8"))
    receipt["files"][1]["sha256"] = "0" * 64
    write_json(bad_receipt, receipt)
    bad_case = dict(luca)
    bad_case["receipt"] = bad_receipt
    bad_output = luca["root"] / "bad-receipt-output.safetensors"
    require_failure(command(args.converter, profiles, catalog, bad_case, bad_output), bad_output, "pinned size/SHA256", "receipt corruption")

    wrong_asset = luca["root"] / "wrong-tokenizer.json"
    asset = tokenizer_asset(luca["profile"]["topology"])
    asset["kind"] = "bpe"
    write_json(wrong_asset, asset)
    descriptor_value = json.loads(luca["descriptor"].read_text(encoding="utf-8"))
    descriptor_value["tokenizer.path"] = wrong_asset.name
    descriptor_value["tokenizer.sha256"] = digest(wrong_asset)
    descriptor_value["tokenizer.size"] = wrong_asset.stat().st_size
    wrong_descriptor = luca["root"] / "wrong-tokenizer-descriptor.json"
    write_json(wrong_descriptor, descriptor_value)
    bad_case = dict(luca)
    bad_case["descriptor"] = wrong_descriptor
    bad_output = luca["root"] / "bad-tokenizer-output.safetensors"
    require_failure(command(args.converter, profiles, catalog, bad_case, bad_output), bad_output, "kind/vocab/special", "tokenizer corruption")

    malformed_asset = luca["root"] / "malformed-vocab-tokenizer.json"
    asset = tokenizer_asset(luca["profile"]["topology"])
    asset["vocab"].append("not-a-vocabulary-entry")
    write_json(malformed_asset, asset)
    descriptor_value = json.loads(luca["descriptor"].read_text(encoding="utf-8"))
    descriptor_value["tokenizer.path"] = malformed_asset.name
    descriptor_value["tokenizer.sha256"] = digest(malformed_asset)
    descriptor_value["tokenizer.size"] = malformed_asset.stat().st_size
    malformed_descriptor = luca["root"] / "malformed-vocab-tokenizer-descriptor.json"
    write_json(malformed_descriptor, descriptor_value)
    bad_case = dict(luca)
    bad_case["descriptor"] = malformed_descriptor
    bad_output = luca["root"] / "malformed-vocab-output.safetensors"
    require_failure(
        command(args.converter, profiles, catalog, bad_case, bad_output),
        bad_output,
        "kind/vocab/special",
        "tokenizer vocabulary entry corruption",
    )

    drift_manifest = json.loads(profiles.read_text(encoding="utf-8"))
    drift_manifest["models"][0]["topology"]["layer_norm_epsilon"] = 1.0e-12
    drift_profiles = args.work_dir / "drift-profiles.json"
    write_json(drift_profiles, drift_manifest)
    bad_output = luca["root"] / "drift-output.safetensors"
    require_failure(command(args.converter, drift_profiles, catalog, luca, bad_output), bad_output, "unsupported custom-encoder semantics", "topology corruption")

    genomics = cases[1]
    missing_base_literal_asset = genomics["root"] / "missing-base-literal-tokenizer.json"
    asset = tokenizer_asset(genomics["profile"]["topology"])
    asset["model"]["literal_token_ids"] = [4096, 4097, 4098, 4099]
    write_json(missing_base_literal_asset, asset)
    descriptor_value = json.loads(genomics["descriptor"].read_text(encoding="utf-8"))
    descriptor_value["tokenizer.path"] = missing_base_literal_asset.name
    descriptor_value["tokenizer.sha256"] = digest(missing_base_literal_asset)
    descriptor_value["tokenizer.size"] = missing_base_literal_asset.stat().st_size
    missing_base_literal_descriptor = (
        genomics["root"] / "missing-base-literal-tokenizer-descriptor.json"
    )
    write_json(missing_base_literal_descriptor, descriptor_value)
    bad_case = dict(genomics)
    bad_case["descriptor"] = missing_base_literal_descriptor
    bad_output = genomics["root"] / "missing-base-literal-output.safetensors"
    require_failure(
        command(args.converter, profiles, catalog, bad_case, bad_output),
        bad_output,
        "root BPE/literal-token contract differs",
        "base-vocabulary AddedToken literal corruption",
    )

    untied_weights = genomics["root"] / "untied.safetensors"
    header, data = read_safetensors(genomics["root"] / "model.safetensors")
    decoder = header["cls.predictions.decoder.weight"]
    absolute = decoder["data_offsets"][0]
    mutable = bytearray(data)
    mutable[absolute] ^= 1
    rewrite_safetensors(untied_weights, header, bytes(mutable))
    untied_manifest = json.loads(profiles.read_text(encoding="utf-8"))
    profile = untied_manifest["models"][1]
    profile["source_files"][1] = {"name": "model.safetensors", "size": untied_weights.stat().st_size, "sha256": digest(untied_weights)}
    untied_profiles = args.work_dir / "untied-profiles.json"
    write_json(untied_profiles, untied_manifest)
    untied_receipt_value = json.loads(genomics["receipt"].read_text(encoding="utf-8"))
    untied_receipt_value["files"][1] = receipt_entry(untied_weights, "model.safetensors")
    untied_receipt = genomics["root"] / "untied-receipt.json"
    write_json(untied_receipt, untied_receipt_value)
    bad_case = dict(genomics)
    bad_case["receipt"] = untied_receipt
    bad_output = genomics["root"] / "untied-output.safetensors"
    require_failure(command(args.converter, untied_profiles, catalog, bad_case, bad_output), bad_output, "not tied", "tied embedding corruption")

    wrong_bias_weights = genomics["root"] / "wrong-mlm-bias-key.safetensors"
    header, data = read_safetensors(genomics["root"] / "model.safetensors")
    header["cls.predictions.bias"] = header.pop("cls.predictions.decoder.bias")
    rewrite_safetensors(wrong_bias_weights, header, data)
    wrong_bias_manifest = json.loads(profiles.read_text(encoding="utf-8"))
    profile = wrong_bias_manifest["models"][1]
    profile["source_files"][1] = {
        "name": "model.safetensors",
        "size": wrong_bias_weights.stat().st_size,
        "sha256": digest(wrong_bias_weights),
    }
    wrong_bias_profiles = args.work_dir / "wrong-mlm-bias-profiles.json"
    write_json(wrong_bias_profiles, wrong_bias_manifest)
    wrong_bias_receipt_value = json.loads(
        genomics["receipt"].read_text(encoding="utf-8")
    )
    wrong_bias_receipt_value["files"][1] = receipt_entry(
        wrong_bias_weights, "model.safetensors"
    )
    wrong_bias_receipt = genomics["root"] / "wrong-mlm-bias-receipt.json"
    write_json(wrong_bias_receipt, wrong_bias_receipt_value)
    bad_case = dict(genomics)
    bad_case["receipt"] = wrong_bias_receipt
    bad_output = genomics["root"] / "wrong-mlm-bias-output.safetensors"
    require_failure(
        command(
            args.converter,
            wrong_bias_profiles,
            catalog,
            bad_case,
            bad_output,
        ),
        bad_output,
        "tensor manifest mismatch",
        "MLM decoder bias key corruption",
    )

    print("GENEB custom encoder converter contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
