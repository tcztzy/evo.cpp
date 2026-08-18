#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tiny deterministic and corruption contracts for the GENEB decoder converter."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


PRODUCTION_IDS = {
    "geneb-metagene-1",
    "geneb-genome-ocean-4b",
    "geneb-generator-eukaryote-3b",
    "geneb-generator-eukaryote-1-2b",
    "geneb-genome-ocean-500m",
    "geneb-biofm-265m",
    "geneb-omnina-220m",
}
METAGENE_TOKENIZER_PINS = {
    "tokenizer_manifest_sha256": (
        "11d6a12137231c641e860454396713f971e692b1eb07538b7d4ba6977e50066f"
    ),
    "tokenizer_source_receipt_contract_sha256": (
        "f73347a1ae68023529482f27ff2a9f452261f61174b847e9f249de55bf565b54"
    ),
    "tokenizer_asset_sha256": (
        "c5e193c3b1124c7076ada0fe6d8688d5d19c42cec1e40176d3248db07768ae60"
    ),
    "tokenizer_asset_size": 43998,
}
GENERATOR_3B_TOKENIZER_PINS = {
    "tokenizer_manifest_sha256": (
        "d157aa64db892021262905ffa1baf32719c7e60706cfd133c7379822e2471d3e"
    ),
    "tokenizer_source_receipt_contract_sha256": (
        "0200690abcd8f9b1fe9a395342115b06994ef03fe783f85101821050ba7829f8"
    ),
    "tokenizer_asset_sha256": (
        "b2797c844e85287ff24bfff796e7b83a61783796045af696355718f04a28bb47"
    ),
    "tokenizer_asset_size": 118948,
}


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_value(dtype: str) -> Dict[str, Any]:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 5,
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "intermediate_size": 6,
        "max_position_embeddings": 16,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "torch_dtype": "float32" if dtype == "F32" else "bfloat16",
    }


def topology_value(dtype: str) -> Dict[str, Any]:
    return {
        "vocab_size": 5,
        "hidden_size": 4,
        "num_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "rotary_dim": 2,
        "inner_mlp_size": 6,
        "max_seqlen": 16,
        "sliding_window": 0,
        "rms_norm_epsilon": 0.00001,
        "rms_epsilon_placement": "inside-sqrt",
        "rope_base": 10000.0,
        "rope_position_scale": 1.0,
        "rope_layout": "split-half",
        "mlp_activation": "swiglu",
        "attention_bias": False,
        "mlp_bias": False,
        "embedding_dtype": dtype,
        "projection_dtype": dtype,
        "norm_dtype": dtype,
        "activation_dtype": dtype,
    }


def tensor_specs(converter: Any, topology: Mapping[str, Any]) -> List[Any]:
    return converter.canonical_tensor_specs(topology) + [
        converter.TensorSpec(
            "lm_head.weight",
            topology["projection_dtype"],
            (topology["vocab_size"], topology["hidden_size"]),
        )
    ]


def payload_for(name: str, size: int, dtype: str) -> bytes:
    seed = sum(name.encode("ascii")) % 251
    if dtype == "F32":
        values = []
        for index in range(size // 4):
            if "norm.weight" in name or "layernorm.weight" in name:
                value = 1.0 + ((seed + index) % 5) / 128.0
            else:
                value = (((seed + index * 7) % 19) - 9) / 64.0
            values.append(struct.pack("<f", value))
        return b"".join(values)
    if dtype == "BF16":
        bits = 0x3F80 if "norm.weight" in name or "layernorm.weight" in name else 0x3D00
        return struct.pack("<H", bits) * (size // 2)
    return bytes(((seed + index * 17) % 251 for index in range(size)))


def spec_bytes(spec: Any, dtype_override: Optional[str] = None) -> int:
    dtype = dtype_override if dtype_override is not None else spec.dtype
    width = {"F32": 4, "BF16": 2, "F16": 2}[dtype]
    count = 1
    for dimension in spec.shape:
        count *= dimension
    return count * width


def write_safetensors(
    path: Path,
    specs: Sequence[Any],
    dtype_override: Optional[Tuple[str, str]] = None,
    trailing: bytes = b"",
) -> Dict[str, bytes]:
    root = {"__metadata__": {"format": "pt"}}  # type: Dict[str, Any]
    payloads = {}  # type: Dict[str, bytes]
    offset = 0
    for spec in specs:
        dtype = (
            dtype_override[1]
            if dtype_override is not None and spec.name == dtype_override[0]
            else spec.dtype
        )
        size = spec_bytes(spec, dtype)
        payload = payload_for(spec.name, size, dtype)
        root[spec.name] = {
            "dtype": dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + size],
        }
        payloads[spec.name] = payload
        offset += size
    raw = json.dumps(root, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(header))
        + header
        + b"".join(payloads[spec.name] for spec in specs)
        + trailing
    )
    return payloads


def receipt_entry(path: Path) -> Dict[str, Any]:
    return {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": digest(path),
        "path": str(path.resolve()),
    }


def refresh_receipt(case: Mapping[str, Any]) -> None:
    receipt_path = case["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"] = [receipt_entry(path) for path in case["source_files"]]
    write_json(receipt_path, receipt)


def read_artifact(path: Path) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        data = source.read()
    metadata = header.pop("__metadata__")
    tensors = {
        name: data[value["data_offsets"][0] : value["data_offsets"][1]]
        for name, value in header.items()
    }
    return metadata, tensors


def rewrite_artifact_header(
    source_path: Path,
    output_path: Path,
    mutate: Callable[[Dict[str, Any]], None],
) -> None:
    with source_path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        payload = source.read()
    mutate(header)
    raw = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    encoded = raw + b" " * ((-len(raw)) % 8)
    output_path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def require_native_rejected(native_loader: Path, artifact: Path, label: str) -> None:
    result = subprocess.run(
        [str(native_loader), "--verify-artifact", str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        raise AssertionError("native loader accepted corrupt runtime %s" % label)


def make_case(
    root: Path,
    converter: Any,
    dtype: str = "F32",
    sharded: bool = False,
    source_format: str = "safetensors",
    legacy_rotary_inv_freq: bool = False,
    tokenizer_pins: bool = False,
) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    runtime_id = (
        "geneb-omnina-220m"
        if legacy_rotary_inv_freq
        else "geneb-decoder-tiny"
    )
    revision = "a" * 40
    repo = "fixture/DecoderTiny"
    config = root / "config.json"
    write_json(config, config_value(dtype))
    topology = topology_value(dtype)
    profile = {
        "runtime_id": runtime_id,
        "geneb_model_id": "DecoderTiny",
        "paper_name": "DecoderTiny",
        "catalog_architecture": "llama-causal-lm",
        "repo": repo,
        "revision": revision,
        "source_format": source_format,
        "config_sha256": digest(config),
        "config_required": config_value(dtype),
        "config_defaults": {"sliding_window": None},
        "lm_head_policy": "validate-and-omit",
        "topology": topology,
    }
    if legacy_rotary_inv_freq:
        profile["f32_math_kernel"] = converter.OMNINA_F32_MATH_KERNEL
        profile["legacy_rotary_inv_freq"] = dict(
            converter.OMNINA_LEGACY_ROTARY_INV_FREQ
        )
    profiles = root / "profiles.json"
    write_json(
        profiles,
        {
            "schema_version": 1,
            "format": "geneb-decoder-converter-v1",
            "models": [profile],
        },
    )
    catalog = root / "catalog.json"
    write_json(
        catalog,
        {
            "schema_version": 1,
            "suite": {"id": "geneb-v4", "raw_safety_cap_bytes": 16777216},
            "models": [
                {
                    "runtime_id": runtime_id,
                    "geneb_model_id": "DecoderTiny",
                    "paper_name": "DecoderTiny",
                    "family": "transformer-decoder",
                    "architecture": "llama-causal-lm",
                    "source": {
                        "kind": "huggingface",
                        "repo": repo,
                        "requested_revision": revision,
                        "revision": revision,
                        "immutable": True,
                    },
                    "context": {
                        "unit": "tokens",
                        "declared_max_tokens": 16,
                        "reference_max_tokens": 16,
                        "length_policy": "tokenizer-truncate",
                    },
                    "input_transform": {
                        "case": "preserve",
                        "strip_ascii_whitespace": False,
                        "u_to_t": False,
                        "invalid": "tokenizer-defined",
                        "frame_trim": None,
                        "raw_crop": None,
                        "fixed_pad": None,
                        "prefix": None,
                        "special_tokens": "tokenizer-default",
                        "token_truncation": "right",
                    },
                    "embedding_presets": {
                        "reference": {
                            "hidden_tap": "post-final-norm",
                            "pooling": "masked-mean",
                            "special_tokens": "include",
                            "mask_domain": "attention-mask",
                            "output_width": 4,
                        },
                        "normalized": {
                            "hidden_tap": "post-final-norm",
                            "pooling": "masked-mean",
                            "special_tokens": "include",
                            "mask_domain": "attention-mask",
                            "output_width": 4,
                        },
                    },
                    "provenance": {
                        "extractor": {"commit": "b" * 40},
                        "reference_patch": {"sha256": "c" * 64},
                        "normalization_patch_sha256": "d" * 64,
                    },
                }
            ],
        },
    )

    specs = tensor_specs(converter, topology)
    payloads = {}  # type: Dict[str, bytes]
    source_files = [config]
    if source_format == "safetensors" and sharded:
        split = len(specs) // 2
        groups = [specs[:split], specs[split:]]
        owners = {}  # type: Dict[str, str]
        for index, group in enumerate(groups, start=1):
            shard = root / ("model-%05d-of-00002.safetensors" % index)
            payloads.update(write_safetensors(shard, group))
            source_files.append(shard)
            owners.update({spec.name: shard.name for spec in group})
        index_path = root / "model.safetensors.index.json"
        write_json(
            index_path,
            {
                "metadata": {
                    "total_size": sum(spec.nbytes for spec in specs)
                },
                "weight_map": owners,
            },
        )
        source_files.insert(1, index_path)
    elif source_format == "safetensors":
        weights = root / "model.safetensors"
        payloads.update(write_safetensors(weights, specs))
        source_files.append(weights)
    else:
        weights = root / "pytorch_model.bin"
        weights.write_bytes(b"verified OmniNA fixture requires optional torch")
        source_files.append(weights)

    receipt = root / "source-receipt.json"
    write_json(
        receipt,
        {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": runtime_id,
            "repo": repo,
            "requested_revision": revision,
            "resolved_revision": revision,
            "files": [receipt_entry(path) for path in source_files],
            "load_path": None,
            "source_kind": "huggingface",
            "catalog_path": str(catalog.resolve()),
            "catalog_sha256": digest(catalog),
        },
    )
    tokenizer_root = root
    tokenizer_asset = tokenizer_root / "assets" / "tokenizer.json"
    tokenizer_asset.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_asset.write_text(
        '{"format":"evo-tokenizer-v1","kind":"character",'
        '"normalization":[{"op":"ascii-uppercase"}],'
        '"pre_tokenizer":{"kind":"none"},'
        '"model":{"unknown_policy":"unk","match_special_literals":false},'
        '"post_processor":{"prefix_ids":[],"suffix_ids":[],'
        '"padding":{"side":"none","pad_id":null}},'
        '"special_tokens":{"unk":4,"pad":null,"bos":null,"eos":null,'
        '"cls":null,"sep":null,"mask":null},'
        '"vocab":[{"id":0,"piece":"A"},{"id":1,"piece":"C"},'
        '{"id":2,"piece":"G"},{"id":3,"piece":"T"},'
        '{"id":4,"piece":"[UNK]"}]}\n',
        encoding="utf-8",
    )
    tokenizer_descriptor = root / "tokenizer-descriptor.json"
    write_json(
        tokenizer_descriptor,
        {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": "1" * 64,
            "source_receipt_contract_sha256": "2" * 64,
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": "assets/tokenizer.json",
            "tokenizer.sha256": digest(tokenizer_asset),
            "tokenizer.size": tokenizer_asset.stat().st_size,
        },
    )
    if tokenizer_pins:
        descriptor = json.loads(tokenizer_descriptor.read_text(encoding="utf-8"))
        profile.update(
            {
                "tokenizer_manifest_sha256": descriptor[
                    "compiler_manifest_sha256"
                ],
                "tokenizer_source_receipt_contract_sha256": descriptor[
                    "source_receipt_contract_sha256"
                ],
                "tokenizer_asset_sha256": descriptor["tokenizer.sha256"],
                "tokenizer_asset_size": descriptor["tokenizer.size"],
            }
        )
        write_json(
            profiles,
            {
                "schema_version": 1,
                "format": "geneb-decoder-converter-v1",
                "models": [profile],
            },
        )
    return {
        "root": root,
        "profile": profile,
        "profiles": profiles,
        "catalog": catalog,
        "config": config,
        "receipt": receipt,
        "source_files": source_files,
        "specs": specs,
        "payloads": payloads,
        "tokenizer_root": tokenizer_root,
        "tokenizer_asset": tokenizer_asset,
        "tokenizer_descriptor": tokenizer_descriptor,
    }


def run_converter(
    converter_path: Path,
    case: Mapping[str, Any],
    output: Path,
    force: bool = False,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(converter_path),
        "--receipt",
        str(case["receipt"]),
        "--catalog",
        str(case["catalog"]),
        "--profiles",
        str(case["profiles"]),
        "--tokenizer-descriptor",
        str(case["tokenizer_descriptor"]),
        "--tokenizer-root",
        str(case["tokenizer_root"]),
        "--output",
        str(output),
    ]
    if force:
        command.append("--force")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(converter_path.parent)
    return subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment
    )


def require_success(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode != 0:
        raise AssertionError("%s failed: %s" % (label, result.stderr))


def require_rejected(
    converter_path: Path,
    case: Mapping[str, Any],
    output: Path,
    fragment: Optional[str] = None,
) -> None:
    result = run_converter(converter_path, case, output)
    if result.returncode == 0:
        raise AssertionError("corrupt case unexpectedly converted: %s" % output.parent.name)
    if fragment is not None and fragment not in result.stderr:
        raise AssertionError(
            "%s rejection omitted %r: %s" % (output.parent.name, fragment, result.stderr)
        )
    if output.exists() or list(output.parent.glob(".%s.*.tmp" % output.name)):
        raise AssertionError("corrupt conversion left partial output")


def rewrite_profiles(case: Mapping[str, Any], mutate: Callable[[Dict[str, Any]], None]) -> None:
    root = json.loads(case["profiles"].read_text(encoding="utf-8"))
    mutate(root["models"][0])
    write_json(case["profiles"], root)


def validate_production_profiles(converter: Any, path: Path) -> None:
    profiles, _ = converter.load_profiles(path)
    if set(profiles) != PRODUCTION_IDS:
        raise AssertionError("production decoder profile set is not the T28 seven")
    expected_tokenizer_pins = {
        "geneb-metagene-1": METAGENE_TOKENIZER_PINS,
        "geneb-generator-eukaryote-3b": GENERATOR_3B_TOKENIZER_PINS,
    }
    for runtime_id, expected_pins in expected_tokenizer_pins.items():
        actual_tokenizer_pins = {
            key: profiles[runtime_id].get(key) for key in expected_pins
        }
        if actual_tokenizer_pins != expected_pins:
            raise AssertionError(runtime_id + " tokenizer profile pins drifted")
    tokenizer_pin_profiles = {
        runtime_id
        for runtime_id, profile in profiles.items()
        if any(key in profile for key in METAGENE_TOKENIZER_PINS)
    }
    if tokenizer_pin_profiles != set(expected_tokenizer_pins):
        raise AssertionError("tokenizer profile pin ownership drifted")
    biofm = profiles["geneb-biofm-265m"]
    if (
        biofm["config_required"].get("head_dim") != 128
        or biofm["topology"]["head_dim"] != 128
        or biofm["topology"]["num_attention_heads"] * 128 != 4096
        or biofm["topology"]["hidden_size"] != 640
        or biofm.get("attention_kernel")
        != converter.BIOFM_ATTENTION_KERNEL
    ):
        raise AssertionError("BioFM attention width/kernel contract drifted")
    attention_kernel_profiles = {
        runtime_id
        for runtime_id, profile in profiles.items()
        if "attention_kernel" in profile
    }
    if attention_kernel_profiles != {"geneb-biofm-265m"}:
        raise AssertionError("portable CPU Flash kernel escaped the BioFM profile")
    bf16 = {
        runtime_id
        for runtime_id, profile in profiles.items()
        if profile["topology"]["activation_dtype"] == "BF16"
    }
    if bf16 != {"geneb-biofm-265m"}:
        raise AssertionError("production activation dtype matrix drifted")
    for runtime_id in (
        "geneb-genome-ocean-4b",
        "geneb-genome-ocean-500m",
    ):
        topology = profiles[runtime_id]["topology"]
        if (
            [
                topology["embedding_dtype"],
                topology["projection_dtype"],
                topology["norm_dtype"],
            ]
            != ["BF16", "BF16", "BF16"]
            or topology["activation_dtype"] != "F32"
        ):
            raise AssertionError(
                "%s CPU-F32 activation/source dtype gate drifted" % runtime_id
            )
    omnina = profiles["geneb-omnina-220m"]
    if omnina["source_format"] != "pytorch-bin":
        raise AssertionError("OmniNA must retain its verified pytorch_model.bin path")
    legacy_rotary_profiles = {
        runtime_id
        for runtime_id, profile in profiles.items()
        if "legacy_rotary_inv_freq" in profile
    }
    if (
        legacy_rotary_profiles != {"geneb-omnina-220m"}
        or omnina["legacy_rotary_inv_freq"]
        != converter.OMNINA_LEGACY_ROTARY_INV_FREQ
    ):
        raise AssertionError("OmniNA legacy rotary validate-and-omit gate drifted")
    f32_math_kernel_profiles = {
        runtime_id
        for runtime_id, profile in profiles.items()
        if "f32_math_kernel" in profile
    }
    if (
        f32_math_kernel_profiles != {"geneb-omnina-220m"}
        or omnina["f32_math_kernel"] != converter.OMNINA_F32_MATH_KERNEL
    ):
        raise AssertionError("OmniNA Apple-arm64 exact F32 kernel gate drifted")
    if any(
        profile["lm_head_policy"] != "validate-and-omit"
        or profile["topology"]["rope_layout"] != "split-half"
        for profile in profiles.values()
    ):
        raise AssertionError("production lm_head/RoPE policy drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--native-loader", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    sys.path.insert(0, str(args.converter.parent))
    import convert_geneb_decoder_checkpoint as converter

    validate_production_profiles(converter, args.profiles)
    production_root = json.loads(args.profiles.read_text(encoding="utf-8"))
    production_profiles, _ = converter.load_profiles(args.profiles)
    metagene_profile = production_profiles["geneb-metagene-1"]
    wrong_descriptor = args.work_dir / "metagene-wrong-tokenizer-descriptor.json"
    write_json(
        wrong_descriptor,
        {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": METAGENE_TOKENIZER_PINS[
                "tokenizer_manifest_sha256"
            ],
            "source_receipt_contract_sha256": "0" * 64,
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": "tokenizers/geneb-metagene-1-bpe-v1.json",
            "tokenizer.sha256": METAGENE_TOKENIZER_PINS[
                "tokenizer_asset_sha256"
            ],
            "tokenizer.size": METAGENE_TOKENIZER_PINS[
                "tokenizer_asset_size"
            ],
        },
    )
    try:
        converter.validate_tokenizer_descriptor(
            wrong_descriptor,
            args.work_dir,
            args.work_dir,
            metagene_profile,
        )
    except converter.ConversionError as error:
        if "tokenizer descriptor differs from pinned profile" not in str(error):
            raise AssertionError(
                "METAGENE-1 descriptor mismatch returned wrong error: %s" % error
            )
    else:
        raise AssertionError("METAGENE-1 cross-source descriptor was accepted")

    generator_3b_profile = production_profiles["geneb-generator-eukaryote-3b"]
    wrong_generator_descriptor = (
        args.work_dir / "generator-3b-wrong-tokenizer-descriptor.json"
    )
    write_json(
        wrong_generator_descriptor,
        {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": GENERATOR_3B_TOKENIZER_PINS[
                "tokenizer_manifest_sha256"
            ],
            "source_receipt_contract_sha256": METAGENE_TOKENIZER_PINS[
                "tokenizer_source_receipt_contract_sha256"
            ],
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": "tokenizer.json",
            "tokenizer.sha256": GENERATOR_3B_TOKENIZER_PINS[
                "tokenizer_asset_sha256"
            ],
            "tokenizer.size": GENERATOR_3B_TOKENIZER_PINS[
                "tokenizer_asset_size"
            ],
        },
    )
    try:
        converter.validate_tokenizer_descriptor(
            wrong_generator_descriptor,
            args.work_dir,
            args.work_dir,
            generator_3b_profile,
        )
    except converter.ConversionError as error:
        if "tokenizer descriptor differs from pinned profile" not in str(error):
            raise AssertionError(
                "GENERATOR-3B descriptor mismatch returned wrong error: %s" % error
            )
    else:
        raise AssertionError("GENERATOR-3B cross-source descriptor was accepted")

    wrong_owner_root = json.loads(json.dumps(production_root))
    wrong_owner_sibling = next(
        item
        for item in wrong_owner_root["models"]
        if item["runtime_id"] == "geneb-genome-ocean-4b"
    )
    for key, value in METAGENE_TOKENIZER_PINS.items():
        wrong_owner_sibling[key] = value
    wrong_owner_path = args.work_dir / "metagene-tokenizer-pins-on-sibling.json"
    write_json(wrong_owner_path, wrong_owner_root)
    try:
        validate_production_profiles(converter, wrong_owner_path)
    except AssertionError as error:
        if "tokenizer profile pin ownership drifted" not in str(error):
            raise AssertionError(
                "METAGENE-1 tokenizer ownership returned wrong error: %s" % error
            )
    else:
        raise AssertionError("METAGENE-1 tokenizer pins escaped to a sibling")

    for label, mutate in (
        (
            "missing-biofm-attention-kernel",
            lambda profile: profile.pop("attention_kernel"),
        ),
        (
            "attention-kernel-on-sibling",
            lambda profile: profile.update(
                {"attention_kernel": converter.BIOFM_ATTENTION_KERNEL}
            ),
        ),
        (
            "missing-omnina-f32-math-kernel",
            lambda profile: profile.pop("f32_math_kernel"),
        ),
        (
            "f32-math-kernel-on-sibling",
            lambda profile: profile.update(
                {"f32_math_kernel": converter.OMNINA_F32_MATH_KERNEL}
            ),
        ),
    ):
        corrupt_root = json.loads(json.dumps(production_root))
        runtime_id = {
            "missing-biofm-attention-kernel": "geneb-biofm-265m",
            "attention-kernel-on-sibling": "geneb-omnina-220m",
            "missing-omnina-f32-math-kernel": "geneb-omnina-220m",
            "f32-math-kernel-on-sibling": "geneb-metagene-1",
        }[label]
        profile = next(
            item for item in corrupt_root["models"] if item["runtime_id"] == runtime_id
        )
        mutate(profile)
        corrupt_path = args.work_dir / (label + ".json")
        write_json(corrupt_path, corrupt_root)
        try:
            converter.load_profiles(corrupt_path)
        except converter.ConversionError as error:
            expected = (
                "attention_kernel is not the exact BioFM contract"
                if "attention-kernel" in label
                else "f32_math_kernel is not the exact OmniNA contract"
            )
            if expected not in str(error):
                raise AssertionError(
                    "%s returned wrong profile error: %s" % (label, error)
                )
        else:
            raise AssertionError("%s profile corruption was accepted" % label)

    valid = make_case(args.work_dir / "valid-f32", converter)
    output_a = valid["root"] / "runtime-a.safetensors"
    output_b = valid["root"] / "runtime-b.safetensors"
    require_success(run_converter(args.converter, valid, output_a), "F32 conversion")
    require_success(run_converter(args.converter, valid, output_b), "repeat conversion")
    if output_a.read_bytes() != output_b.read_bytes():
        raise AssertionError("identical conversion inputs did not produce identical bytes")
    base_profiles = json.loads(valid["profiles"].read_text(encoding="utf-8"))
    sibling_profiles = json.loads(json.dumps(base_profiles))
    sibling = json.loads(json.dumps(sibling_profiles["models"][0]))
    sibling.update(
        {
            "runtime_id": "geneb-decoder-sibling",
            "geneb_model_id": "DecoderSibling",
            "paper_name": "DecoderSibling",
            "repo": "fixture/DecoderSibling",
            "revision": "e" * 40,
        }
    )
    sibling_profiles["models"].append(sibling)
    write_json(valid["profiles"], sibling_profiles)
    sibling_output = valid["root"] / "runtime-sibling-change.safetensors"
    require_success(
        run_converter(args.converter, valid, sibling_output),
        "sibling-profile-independent decoder conversion",
    )
    if output_a.read_bytes() != sibling_output.read_bytes():
        raise AssertionError("sibling decoder profile changed selected artifact bytes")
    write_json(valid["profiles"], base_profiles)

    pinned = make_case(
        args.work_dir / "tokenizer-profile-pins", converter, tokenizer_pins=True
    )
    pinned_output = pinned["root"] / "runtime.safetensors"
    require_success(
        run_converter(args.converter, pinned, pinned_output),
        "all-or-none tokenizer profile pin conversion",
    )

    cross_model = make_case(
        args.work_dir / "cross-model-tokenizer-descriptor",
        converter,
        tokenizer_pins=True,
    )
    descriptor = json.loads(
        cross_model["tokenizer_descriptor"].read_text(encoding="utf-8")
    )
    descriptor["source_receipt_contract_sha256"] = "3" * 64
    write_json(cross_model["tokenizer_descriptor"], descriptor)
    require_rejected(
        args.converter,
        cross_model,
        cross_model["root"] / "runtime.safetensors",
        "tokenizer descriptor differs from pinned profile",
    )

    partial_pins = make_case(
        args.work_dir / "partial-tokenizer-profile-pins",
        converter,
        tokenizer_pins=True,
    )
    rewrite_profiles(
        partial_pins, lambda profile: profile.pop("tokenizer_asset_size")
    )
    require_rejected(
        args.converter,
        partial_pins,
        partial_pins["root"] / "runtime.safetensors",
        "tokenizer profile pins must appear together",
    )

    metadata, output_tensors = read_artifact(output_a)
    expected_runtime_names = {spec.name for spec in valid["specs"][:-1]}
    if set(output_tensors) != expected_runtime_names or "lm_head.weight" in output_tensors:
        raise AssertionError("runtime artifact did not validate-and-omit lm_head")
    if any(
        output_tensors[name] != valid["payloads"][name]
        for name in expected_runtime_names
    ):
        raise AssertionError("F32 tensor payload changed during conversion")
    required_metadata = {
        "runtime.profile": "s:geneb-decoder-runtime-v1",
        "runtime.abi": "s:geneb-decoder-safetensors-v1",
        "model.architecture": "s:GenebTransformerDecoder",
        "config.vocab_size": "u:5",
        "config.hidden_size": "u:4",
        "config.num_layers": "u:1",
        "config.max_seqlen": "u:16",
        "runtime.embedding_layer_count": "u:2",
        "decoder.head_dim": "u:2",
        "decoder.activation_dtype": "s:F32",
        "decoder.rope_layout": "s:split-half",
        "source.lm_head_policy": "s:validate-and-omit",
        "tokenizer.profile": "s:evo-tokenizer-v1",
        "tokenizer.path": "s:assets/tokenizer.json",
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise AssertionError("runtime metadata differs: %s" % metadata)
    if "compiler_manifest_sha256" in metadata or "converter.schema" in metadata:
        raise AssertionError("tokenizer conversion receipt leaked into artifact metadata")
    if args.native_loader is not None:
        native = subprocess.run(
            [str(args.native_loader), "--verify-artifact", str(output_a)],
            check=False,
            capture_output=True,
            text=True,
        )
        if native.returncode != 0:
            raise AssertionError("native ModelFile/load_artifact rejected output: %s" % native.stderr)
        adapter = subprocess.run(
            [str(args.native_loader), "--verify-cpu-adapter", str(output_a)],
            check=False,
            capture_output=True,
            text=True,
        )
        if adapter.returncode != 0:
            raise AssertionError("native CPU Model adapter rejected output: %s" % adapter.stderr)
        metadata_corruptions = {
            "abi": lambda root: root["__metadata__"].update(
                {"runtime.abi": "s:wrong-abi"}
            ),
            "architecture": lambda root: root["__metadata__"].update(
                {"model.architecture": "s:WrongDecoder"}
            ),
            "activation-dtype": lambda root: root["__metadata__"].update(
                {"decoder.activation_dtype": "s:F16"}
            ),
            "attention-kernel": lambda root: root["__metadata__"].update(
                {
                    "decoder.attention_kernel":
                    "s:torch-cpu-flash-bf16-portable"
                }
            ),
            "f32-math-kernel-unknown": lambda root: root["__metadata__"].update(
                {"decoder.f32_math_kernel": "s:not-a-kernel"}
            ),
            "f32-math-kernel-wrong-topology": lambda root: root[
                "__metadata__"
            ].update(
                {
                    "decoder.f32_math_kernel":
                    "s:torch-2.7.1-apple-arm64-exact-v1"
                }
            ),
            "common-config": lambda root: root["__metadata__"].update(
                {"config.hidden_size": "u:5"}
            ),
        }
        for label, mutate in metadata_corruptions.items():
            corrupt = valid["root"] / ("runtime-corrupt-%s.safetensors" % label)
            rewrite_artifact_header(output_a, corrupt, mutate)
            require_native_rejected(args.native_loader, corrupt, label)

        def corrupt_tensor_shape(root: Dict[str, Any]) -> None:
            root["model.embed_tokens.weight"]["shape"] = [4, 5]

        tensor_corrupt = valid["root"] / "runtime-corrupt-tensor-shape.safetensors"
        rewrite_artifact_header(output_a, tensor_corrupt, corrupt_tensor_shape)
        require_native_rejected(args.native_loader, tensor_corrupt, "tensor shape")

    bf16 = make_case(args.work_dir / "valid-bf16-sharded", converter, "BF16", True)
    bf16_output = bf16["root"] / "runtime.safetensors"
    require_success(run_converter(args.converter, bf16, bf16_output), "BF16 sharded conversion")
    bf16_metadata, bf16_tensors = read_artifact(bf16_output)
    if bf16_metadata.get("decoder.activation_dtype") != "s:BF16" or any(
        bf16_tensors[name] != bf16["payloads"][name] for name in bf16_tensors
    ):
        raise AssertionError("BF16 payload/activation metadata was not preserved")

    for runtime_id in (
        "geneb-genome-ocean-4b",
        "geneb-genome-ocean-500m",
    ):
        gate = make_case(
            args.work_dir / ("%s-activation-gate" % runtime_id),
            converter,
            "BF16",
        )
        profile = json.loads(gate["profiles"].read_text(encoding="utf-8"))[
            "models"
        ][0]
        profile["runtime_id"] = runtime_id
        profile["topology"]["activation_dtype"] = "F32"
        converter.validate_config(gate["config"], profile)
        for label, mutate in (
            (
                "bf16-activation",
                lambda value: value["topology"].update(
                    {"activation_dtype": "BF16"}
                ),
            ),
            (
                "f32-source",
                lambda value: value["topology"].update(
                    {"projection_dtype": "F32"}
                ),
            ),
        ):
            corrupt = json.loads(json.dumps(profile))
            mutate(corrupt)
            try:
                converter.validate_config(gate["config"], corrupt)
            except converter.ConversionError as error:
                if "semantic topology mismatch" not in str(error):
                    raise AssertionError(
                        "%s %s gate returned wrong error: %s"
                        % (runtime_id, label, error)
                    )
            else:
                raise AssertionError(
                    "%s %s corruption was accepted" % (runtime_id, label)
                )

    bad_hash = make_case(args.work_dir / "bad-hash", converter)
    bad_hash["source_files"][-1].write_bytes(bad_hash["source_files"][-1].read_bytes() + b"x")
    require_rejected(
        args.converter,
        bad_hash,
        bad_hash["root"] / "runtime.safetensors",
        "integrity mismatch",
    )

    bad_catalog_receipt = make_case(
        args.work_dir / "bad-receipt-catalog", converter
    )
    receipt_value = json.loads(
        bad_catalog_receipt["receipt"].read_text(encoding="utf-8")
    )
    receipt_value["catalog_sha256"] = "0" * 64
    write_json(bad_catalog_receipt["receipt"], receipt_value)
    require_rejected(
        args.converter,
        bad_catalog_receipt,
        bad_catalog_receipt["root"] / "runtime.safetensors",
        "catalog_sha256 differs",
    )

    extra_receipt = make_case(args.work_dir / "extra-receipt-asset", converter)
    extra_asset = extra_receipt["root"] / "generation_config.json"
    extra_asset.write_bytes(b"{}\n")
    extra_receipt["source_files"].append(extra_asset)
    refresh_receipt(extra_receipt)
    require_success(
        run_converter(
            args.converter,
            extra_receipt,
            extra_receipt["root"] / "runtime-with-extra.safetensors",
        ),
        "verified non-critical receipt asset",
    )
    extra_asset.write_bytes(b"corrupt after receipt\n")
    require_rejected(
        args.converter,
        extra_receipt,
        extra_receipt["root"] / "runtime-corrupt-extra.safetensors",
        "integrity mismatch",
    )

    bad_missing_asset = make_case(args.work_dir / "bad-missing-asset", converter)
    bad_missing_asset["source_files"].remove(bad_missing_asset["config"])
    refresh_receipt(bad_missing_asset)
    require_rejected(
        args.converter,
        bad_missing_asset,
        bad_missing_asset["root"] / "runtime.safetensors",
        "missing config.json",
    )

    bad_config = make_case(args.work_dir / "bad-config", converter)
    changed = config_value("F32")
    changed["hidden_size"] = 6
    write_json(bad_config["config"], changed)
    refresh_receipt(bad_config)
    rewrite_profiles(
        bad_config,
        lambda profile: profile.update({"config_sha256": digest(bad_config["config"])}),
    )
    require_rejected(
        args.converter,
        bad_config,
        bad_config["root"] / "runtime.safetensors",
        "topology gate",
    )

    for corruption in ("missing", "extra", "shape", "dtype"):
        case = make_case(args.work_dir / ("bad-tensor-" + corruption), converter)
        specs = list(case["specs"])
        dtype_override = None
        if corruption == "missing":
            specs.pop(3)
        elif corruption == "extra":
            specs.append(converter.TensorSpec("model.extra.weight", "F32", (1,)))
        elif corruption == "shape":
            original = specs[3]
            specs[3] = converter.TensorSpec(
                original.name, original.dtype, (original.shape[0] + 1, original.shape[1])
            )
        else:
            dtype_override = (specs[3].name, "F16")
        write_safetensors(case["source_files"][-1], specs, dtype_override)
        refresh_receipt(case)
        require_rejected(
            args.converter,
            case,
            case["root"] / "runtime.safetensors",
            "outside allowed set"
            if corruption == "dtype"
            else "tensor manifest mismatch",
        )

    bad_index = make_case(args.work_dir / "bad-index", converter, "BF16", True)
    index_path = bad_index["source_files"][1]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["metadata"]["total_size"] += 2
    write_json(index_path, index)
    refresh_receipt(bad_index)
    require_rejected(
        args.converter,
        bad_index,
        bad_index["root"] / "runtime.safetensors",
        "total_size",
    )

    bad_owner = make_case(args.work_dir / "bad-index-owner", converter, "BF16", True)
    owner_index_path = bad_owner["source_files"][1]
    owner_index = json.loads(owner_index_path.read_text(encoding="utf-8"))
    first_name = sorted(owner_index["weight_map"])[0]
    original_owner = owner_index["weight_map"][first_name]
    ghost_path = bad_owner["root"] / "ghost-00001-of-00002.safetensors"
    ghost_path.write_bytes((bad_owner["root"] / original_owner).read_bytes())
    bad_owner["source_files"].append(ghost_path)
    owner_index["weight_map"][first_name] = "ghost-00001-of-00002.safetensors"
    write_json(owner_index_path, owner_index)
    refresh_receipt(bad_owner)
    require_rejected(
        args.converter,
        bad_owner,
        bad_owner["root"] / "runtime.safetensors",
        "occurs in multiple shards",
    )

    bad_trailing = make_case(args.work_dir / "bad-trailing", converter)
    bad_trailing["source_files"][-1].write_bytes(
        bad_trailing["source_files"][-1].read_bytes() + b"trailing"
    )
    refresh_receipt(bad_trailing)
    require_rejected(
        args.converter,
        bad_trailing,
        bad_trailing["root"] / "runtime.safetensors",
        "complete file payload",
    )

    bad_tokenizer = make_case(args.work_dir / "bad-tokenizer", converter)
    descriptor = json.loads(
        bad_tokenizer["tokenizer_descriptor"].read_text(encoding="utf-8")
    )
    descriptor["tokenizer.sha256"] = "0" * 64
    write_json(bad_tokenizer["tokenizer_descriptor"], descriptor)
    require_rejected(
        args.converter,
        bad_tokenizer,
        bad_tokenizer["root"] / "runtime.safetensors",
        "tokenizer asset integrity mismatch",
    )

    existing = valid["root"] / "existing.safetensors"
    existing.write_bytes(b"keep-me")
    rejected_existing = run_converter(args.converter, valid, existing)
    if rejected_existing.returncode == 0 or existing.read_bytes() != b"keep-me":
        raise AssertionError("converter overwrote an existing output without --force")

    torch_case = make_case(
        args.work_dir / "omnina-pytorch", converter, "F32", False, "pytorch-bin"
    )
    torch_bad_hash = make_case(
        args.work_dir / "omnina-bad-hash", converter, "F32", False, "pytorch-bin"
    )
    torch_bad_hash["source_files"][-1].write_bytes(b"tampered before torch import")
    bad_torch_result = run_converter(
        args.converter,
        torch_bad_hash,
        torch_bad_hash["root"] / "runtime.safetensors",
    )
    if (
        bad_torch_result.returncode == 0
        or "integrity mismatch" not in bad_torch_result.stderr
        or "requires offline PyTorch" in bad_torch_result.stderr
    ):
        raise AssertionError("OmniNA imported torch before receipt verification")
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        require_rejected(
            args.converter,
            torch_case,
            torch_case["root"] / "runtime.safetensors",
            "requires offline PyTorch",
        )
    else:
        def prepare_content_addressed_torch_case(
            case: Mapping[str, Any], rotary_mutation: Optional[str] = None
        ) -> None:
            state = {
                spec.name: torch.zeros(spec.shape, dtype=torch.float32)
                for spec in case["specs"]
            }
            legacy_rotary = case["profile"].get("legacy_rotary_inv_freq")
            if legacy_rotary is not None:
                raw = bytes.fromhex(
                    "803f403f103fd83ea23e733e363e083ecd3d9a3d663d2d3d023dc23c"
                    "923c5a3c243cf63bb83b8a3b4f3b1b3be93aaf3a833a453a133add39"
                    "a63979393a390c39"
                )
                rotary = torch.tensor(list(raw), dtype=torch.uint8).view(
                    torch.bfloat16
                )
                for layer in range(legacy_rotary["layer_count"]):
                    name = (
                        "model.layers.%d.self_attn.rotary_emb.inv_freq" % layer
                    )
                    state[name] = rotary.clone()
                last_name = (
                    "model.layers.%d.self_attn.rotary_emb.inv_freq"
                    % (legacy_rotary["layer_count"] - 1)
                )
                if rotary_mutation == "missing":
                    del state[last_name]
                elif rotary_mutation == "name":
                    state[
                        "model.layers.16.self_attn.rotary_emb.inv_freq"
                    ] = state.pop(last_name)
                elif rotary_mutation == "extra":
                    state[
                        "model.layers.16.self_attn.rotary_emb.inv_freq"
                    ] = rotary.clone()
                elif rotary_mutation == "value":
                    state[last_name][7] = 0.25
                elif rotary_mutation == "dtype":
                    state[last_name] = state[last_name].float()
                elif rotary_mutation == "shape":
                    state[last_name] = state[last_name][:-1].clone()
                elif rotary_mutation is not None:
                    raise AssertionError("unknown rotary mutation")
            weights = case["source_files"][-1]
            torch.save(state, weights)
            blob = weights.parent / digest(weights)
            weights.replace(blob)
            case["source_files"][-1] = blob
            refresh_receipt(case)
            receipt = json.loads(case["receipt"].read_text(encoding="utf-8"))
            receipt["files"][-1]["name"] = "pytorch_model.bin"
            write_json(case["receipt"], receipt)

        prepare_content_addressed_torch_case(torch_case)
        torch_output = torch_case["root"] / "runtime.safetensors"
        require_success(
            run_converter(args.converter, torch_case, torch_output),
            "content-addressed OmniNA torch conversion",
        )

        rotary_case = make_case(
            args.work_dir / "omnina-legacy-rotary",
            converter,
            source_format="pytorch-bin",
            legacy_rotary_inv_freq=True,
        )
        prepare_content_addressed_torch_case(rotary_case)
        require_success(
            run_converter(
                args.converter,
                rotary_case,
                rotary_case["root"] / "runtime.safetensors",
            ),
            "OmniNA legacy rotary validate-and-omit conversion",
        )
        rotary_metadata, _ = read_artifact(
            rotary_case["root"] / "runtime.safetensors"
        )
        if rotary_metadata.get("decoder.f32_math_kernel") != (
            "s:" + converter.OMNINA_F32_MATH_KERNEL
        ):
            raise AssertionError("OmniNA artifact omitted exact F32 math kernel")
        for mutation, fragment in (
            ("missing", "rotary inv_freq tensors are missing"),
            ("name", "rotary inv_freq tensors are missing"),
            ("extra", "tensor manifest mismatch"),
            ("value", "rotary inv_freq tensor"),
            ("dtype", "dtype differs"),
            ("shape", "shape differs"),
        ):
            corrupt_rotary = make_case(
                args.work_dir / ("omnina-legacy-rotary-" + mutation),
                converter,
                source_format="pytorch-bin",
                legacy_rotary_inv_freq=True,
            )
            prepare_content_addressed_torch_case(corrupt_rotary, mutation)
            require_rejected(
                args.converter,
                corrupt_rotary,
                corrupt_rotary["root"] / "runtime.safetensors",
                fragment,
            )

        bad_logical_name = make_case(
            args.work_dir / "omnina-bad-logical-name",
            converter,
            "F32",
            False,
            "pytorch-bin",
        )
        prepare_content_addressed_torch_case(bad_logical_name)
        bad_name_receipt = json.loads(
            bad_logical_name["receipt"].read_text(encoding="utf-8")
        )
        bad_name_receipt["files"][-1]["name"] = "weights.bin"
        write_json(bad_logical_name["receipt"], bad_name_receipt)
        require_rejected(
            args.converter,
            bad_logical_name,
            bad_logical_name["root"] / "runtime.safetensors",
            "PyTorch checkpoint assets differ",
        )

        extra_torch_asset = make_case(
            args.work_dir / "omnina-extra-torch-asset",
            converter,
            "F32",
            False,
            "pytorch-bin",
        )
        prepare_content_addressed_torch_case(extra_torch_asset)
        optimizer = extra_torch_asset["root"] / "optimizer.bin"
        optimizer.write_bytes(b"not part of the pinned checkpoint set")
        extra_receipt = json.loads(
            extra_torch_asset["receipt"].read_text(encoding="utf-8")
        )
        extra_receipt["files"].append(receipt_entry(optimizer))
        write_json(extra_torch_asset["receipt"], extra_receipt)
        require_rejected(
            args.converter,
            extra_torch_asset,
            extra_torch_asset["root"] / "runtime.safetensors",
            "PyTorch checkpoint assets differ",
        )

    grammar = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read(), "
            "filename=sys.argv[1], feature_version=(3,8))",
            str(args.converter),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if grammar.returncode != 0:
        raise AssertionError("converter is not Python 3.8 grammar: %s" % grammar.stderr)
    module_path = args.converter.parent / "evo" / "hf_checkpoint.py"
    module_grammar = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read(), "
            "filename=sys.argv[1], feature_version=(3,8))",
            str(module_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if module_grammar.returncode != 0:
        raise AssertionError("shared reader is not Python 3.8 grammar: %s" % module_grammar.stderr)

    print("GENEB decoder converter contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
