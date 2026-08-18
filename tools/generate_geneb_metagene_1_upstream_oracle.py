#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned METAGENE-1 GENEB CPU-F32 embedding oracle.

The generator uses only the pinned Hugging Face/GENEB upstream stack.  It never
imports evo.cpp, converted artifacts, or native runtime math.  ``--audit-only``
validates the small source, tokenizer, extractor, dependency, and operator
contracts without opening any checkpoint shard or executing the real model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


RUNTIME_ID = "geneb-metagene-1"
MODEL_ID = "METAGENE-1"
REPO = "metagene-ai/METAGENE-1"
REQUESTED_REVISION = "main"
REVISION = "ad8a1e0ee62b85058bfc05d823d8e8d4759edc48"
GENEB_REPO = "darlednik/GENEB"
GENEB_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
GENEB_EXTRACTOR_RELATIVE = "embedding_pipeline/extractors/metagene.py"
GENEB_EXTRACTOR_SIZE = 1_723
GENEB_EXTRACTOR_SHA256 = (
    "9d3100dc4dee5f1bd21f3bac3f7105bb13c88232d68d92fb52310c23c2e56faf"
)
GENEB_PYPROJECT_SIZE = 2_759
GENEB_PYPROJECT_SHA256 = (
    "616a15c0a751eb14687fefe691cd624fbf4cc104605f167a5782cb29d16060e0"
)
GENEB_UV_LOCK_SIZE = 255_360
GENEB_UV_LOCK_SHA256 = (
    "e1fce57ba7eaf9fddebf300161c2840b6e6722bb1c82ac3a5b4e8e093a699953"
)

CATALOG_CONTRACT_SHA256 = (
    "8102f69393a48f55e082740ca8484879ffe5c32fc2ca25e5b5d4a2d54a51ce88"
)
PROFILE_CONTRACT_SHA256 = (
    "d79570191192f57707c4e5e77ecfffdd02a1c5f2c2779d9be0852301aa1479bc"
)
TOKENIZER_MANIFEST_RELATIVE = "configs/tokenizers/geneb-metagene-1-bpe-v1.json"
TOKENIZER_MANIFEST_SIZE = 517
TOKENIZER_MANIFEST_SHA256 = (
    "11d6a12137231c641e860454396713f971e692b1eb07538b7d4ba6977e50066f"
)
TOKENIZER_SOURCE_RECEIPT_CONTRACT_SHA256 = (
    "f73347a1ae68023529482f27ff2a9f452261f61174b847e9f249de55bf565b54"
)
TOKENIZER_ASSET_SIZE = 43_998
TOKENIZER_ASSET_SHA256 = (
    "c5e193c3b1124c7076ada0fe6d8688d5d19c42cec1e40176d3248db07768ae60"
)
TOKENIZER_DESCRIPTOR_SIZE = 479
TOKENIZER_DESCRIPTOR_SHA256 = (
    "7d297b3de9f2460df05f5b7b0093851f4f03bb54fe8b132f5473c02db6c9d575"
)

SOURCE_FILES = {
    "config.json": (
        903,
        "dc3751f8648b4ab24a3c0c42026267e1a2561b93249f54b460c0374e62d33f98",
    ),
    "model-00001-of-00006.safetensors": (
        4_941_093_144,
        "7f8e8789e005c63157a5c61e1a5974109136d44259020ac9e779f33a0b487480",
    ),
    "model-00002-of-00006.safetensors": (
        4_991_424_824,
        "4fa6abadcb468608f19df31644f74c9151c28c8df3007a9b4f7946ea96bb6784",
    ),
    "model-00003-of-00006.safetensors": (
        4_924_315_880,
        "8ecf5b3c2a7311ca4cb39a8832d276817ff0cafb328f70b1dab12a3a382774b7",
    ),
    "model-00004-of-00006.safetensors": (
        4_857_206_904,
        "d481bc4e96ae0b2302931395315a3dc373e556d4269c564d4b306f6a1e5c451e",
    ),
    "model-00005-of-00006.safetensors": (
        4_857_206_904,
        "25778d64535c016d32cedf1fceaf13a22bf52b3c557e129c2cdbb723eab017a6",
    ),
    "model-00006-of-00006.safetensors": (
        1_367_426_832,
        "8abf5e8a7aef8884cf729f8efb97f02f799876e31ab326cb99565e7c52444a39",
    ),
    "model.safetensors.index.json": (
        23_950,
        "a5957f6e6126ead8147acc48ee650770368c0ae2c86bbdd516fb041300149077",
    ),
    "special_tokens_map.json": (
        160,
        "f81b8d149b5b7baf0d3c634ffc7a12a83cfbdd60fea6eef9bbfcb2e9856f3b74",
    ),
    "tokenizer.json": (
        42_154,
        "3fd754434de0e3aa76ad233e1bcbe1562a21a91a4e13dda947fd7dce401f0b88",
    ),
    "tokenizer_config.json": (
        1_376,
        "520cd0838fd728266a9791264e418a93cf10b019224faf8b5aed85d60bcea332",
    ),
}
WEIGHT_FILES = {
    name for name in SOURCE_FILES if name.endswith(".safetensors")
}
SMALL_SOURCE_FILES = {
    name: value for name, value in SOURCE_FILES.items() if name not in WEIGHT_FILES
}
EXCLUDED_FALLBACK = {
    "tokenizer.model": (
        6_128,
        "0bcccad15ab3d220976cbfaf12718e4236f2159e5d0915d95c91f8d866b6569c",
    )
}
CHECKPOINT_LOGICAL_BYTES = 25_938_640_896
CHECKPOINT_PHYSICAL_BYTES = 25_938_674_488
SOURCE_TENSOR_COUNT = 291
PARAMETER_COUNT = CHECKPOINT_LOGICAL_BYTES // 4
ARTIFACT_TENSOR_BYTES = CHECKPOINT_LOGICAL_BYTES - 1_024 * 4_096 * 4
MINIMUM_ORACLE_HOST_MEMORY_BYTES = 48 * 1024**3
RECOMMENDED_ORACLE_HOST_MEMORY_BYTES = 64 * 1024**3
SUPPORTED_ORACLE_HOSTS = {
    ("Darwin", "arm64"): "darwin-arm64",
    ("Linux", "x86_64"): "linux-x86_64",
}

EXPECTED_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 1024,
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "max_position_embeddings": 512,
    "max_sequence_length": 2048,
    "rms_norm_eps": 0.00001,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "hidden_act": "silu",
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.46.3",
    "bos_token_id": 3,
    "eos_token_id": 4,
    "pad_token_id": 0,
}
EXPECTED_TOPOLOGY = {
    "vocab_size": 1024,
    "hidden_size": 4096,
    "num_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "head_dim": 128,
    "rotary_dim": 128,
    "inner_mlp_size": 11008,
    "max_seqlen": 512,
    "sliding_window": 0,
    "rms_norm_epsilon": 0.00001,
    "rms_epsilon_placement": "inside-sqrt",
    "rope_base": 10000.0,
    "rope_position_scale": 1.0,
    "rope_layout": "split-half",
    "mlp_activation": "swiglu",
    "attention_bias": False,
    "mlp_bias": False,
    "embedding_dtype": "F32",
    "projection_dtype": "F32",
    "norm_dtype": "F32",
    "activation_dtype": "F32",
}
EXPECTED_SHARD_COUNTS = {
    "model-00001-of-00006.safetensors": 56,
    "model-00002-of-00006.safetensors": 56,
    "model-00003-of-00006.safetensors": 55,
    "model-00004-of-00006.safetensors": 54,
    "model-00005-of-00006.safetensors": 54,
    "model-00006-of-00006.safetensors": 16,
}
INPUTS = (
    {
        "label": "input-0",
        "header": "geneb-metagene-1-input-0",
        "sequence": "ACGTNACGTNACGTA",
        "size": 42,
        "sha256": "51a958bd49ca046fab1f353e90f08c30fb6db0e78079120a7019a758a0be9f28",
        "input_ids": [252, 290, 11, 7, 8, 290, 11, 7, 8, 38, 4],
        "tokens": ["_AC", "GT", "N", "A", "C", "GT", "N", "A", "C", "GTA", "[EOS]"],
    },
    {
        "label": "input-1",
        "header": "geneb-metagene-1-input-1",
        "sequence": "NNNNNNACGTACNNNTATTGCA",
        "size": 49,
        "sha256": "222952d0f2484b21eb645c3f94c3e16016e002795d98144a174a6a4eff9ffd3c",
        "input_ids": [6, 11, 11, 11, 11, 11, 11, 7, 8, 77, 11, 11, 11, 111, 34, 4],
        "tokens": ["_", "N", "N", "N", "N", "N", "N", "A", "C", "GTAC", "N", "N", "N", "TATT", "GCA", "[EOS]"],
    },
)
TOLERANCES = {"max_abs": 0.0001, "mean_abs": 0.00001, "cosine": 0.999999}

REQUIRED_PACKAGE_VERSIONS = {
    "filelock": "3.20.3",
    "huggingface-hub": "0.36.0",
    "numpy": "1.26.4",
    "packaging": "25.0",
    "pyyaml": "6.0.3",
    "regex": "2025.11.3",
    "requests": "2.32.5",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "torch": "2.7.1",
    "tqdm": "4.67.1",
    "transformers": "4.57.3",
}
FULL_ONLY_PACKAGE_VERSIONS = {"accelerate": "1.12.0"}
GENEB_LOCK_VERSIONS = {
    "accelerate": "1.12.0",
    "numpy": "1.26.4",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "torch": "2.7.1+cu126",
    "transformers": "4.57.3",
}
THREAD_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_ENABLE_PARALLEL_LOADING": "false",
    "LANG": "C",
    "LC_ALL": "C",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
    "VECLIB_MAXIMUM_THREADS": "1",
}
TRANSFORMERS_SOURCE_SHA256 = {
    "LlamaRMSNorm": "6cf5b1db2ae8fd5ab4f0a49fcc176117f6a80f357560cf244c18e3ff640c6a81",
    "LlamaRotaryEmbedding": "031c8cecfccf85d7466e66fd576fb49da8526fa5e3be62c70c8e8c9c97ee2162",
    "apply_rotary_pos_emb": "d1e7d21b55bc5ca988047b17ea15ac7d674428b30073c3e101eb782180a58a25",
    "eager_attention_forward": "8dd801c3a8d26a45507f388d43f47ff8640dd51c4ee3f9c71b3efc87a668788c",
    "LlamaMLP": "c35b6e992834901a86bdaa4f6df75407051095fe23ea0ccdbc30bf302005f90c",
    "LlamaDecoderLayer": "9a0b1ec85b8c3b8f9acc4859ec707696f660057cdc7d744a84a194a1fd4c94a1",
    "LlamaModel": "d118ea71078b13561a8ea79ab47febf312d7acf67063a5fb425ece15a8e8a720",
    "LlamaForCausalLM": "ba8b77aeb084b0d338d22818a15012c6e2fe50ef379e0f770e2adc3e7780ebb6",
}
DARWIN_ARM64_OPERATOR_SHA256 = {
    "rmsnorm_raw_f32_sha256": (
        "5508a7b5b5b7bfb6d1a7d9d651ef94951bb6c6be0fdc75d6aed75098d2930165"
    ),
    "split_half_rope_raw_f32_sha256": (
        "d9ae2aa864cdb7901dba7cfbcd72a8366bc50e7e05b615ee5a330fb6e58f7f12"
    ),
    "sdpa_raw_f32_sha256": (
        "974c359338753008a899d2d1ad32de410ebcf361303e4dbabfaa016e4a113065"
    ),
    "swiglu_raw_f32_sha256": (
        "30d6130e58fe750174bcff2fbac70ae7bbc6a9ec4939b3950e123d1f2b6be6b1"
    ),
}
# Filled only after a reviewed Linux/x86_64 ``--audit-only`` execution in the
# pinned environment.  Until then Linux may emit candidates but may not emit a
# real model oracle.
LINUX_X86_64_OPERATOR_SHA256: Mapping[str, str] | None = {
    "rmsnorm_raw_f32_sha256": "5508a7b5b5b7bfb6d1a7d9d651ef94951bb6c6be0fdc75d6aed75098d2930165",
    "split_half_rope_raw_f32_sha256": "d9ae2aa864cdb7901dba7cfbcd72a8366bc50e7e05b615ee5a330fb6e58f7f12",
    "sdpa_raw_f32_sha256": "974c359338753008a899d2d1ad32de410ebcf361303e4dbabfaa016e4a113065",
    "swiglu_raw_f32_sha256": "6f361bf92696f003c143f8fb6004f89af3e5a127831e0c21f96fbe91b8f7f6a9",
}


class OracleError(RuntimeError):
    """Raised when a pinned source, environment, or oracle contract differs."""


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


def physical_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def validate_oracle_host(*, full: bool) -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    host_profile = SUPPORTED_ORACLE_HOSTS.get((system, machine))
    if host_profile is None:
        raise OracleError(
            "oracle host must be Darwin/arm64 or Linux/x86_64"
        )
    memory_bytes = physical_memory_bytes()
    memory_eligible = (
        memory_bytes is not None
        and memory_bytes >= MINIMUM_ORACLE_HOST_MEMORY_BYTES
    )
    if full and not memory_eligible:
        raise OracleError(
            "real oracle requires at least 48 GiB physical memory; "
            "use --audit-only on a smaller supported host"
        )
    return {
        "contract": "darwin-arm64-or-linux-x86_64-cpu-f32",
        "host_profile": host_profile,
        "system": system,
        "machine": machine,
        "physical_memory_bytes": memory_bytes,
        "minimum_real_oracle_memory_bytes": MINIMUM_ORACLE_HOST_MEMORY_BYTES,
        "recommended_real_oracle_memory_bytes": (
            RECOMMENDED_ORACLE_HOST_MEMORY_BYTES
        ),
        "real_oracle_memory_eligible": memory_eligible,
        "oracle_device": "cpu",
        "oracle_dtype": "float32",
        "cross_isa_bit_exact_claimed": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OracleError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise OracleError(label + " root must be an object")
    return value


def checked_file(
    path: Path, label: str, expected_size: int, expected_sha256: str
) -> dict[str, Any]:
    if not path.is_file():
        raise OracleError(label + " must be a regular file or resolved cache link")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != expected_size or digest != expected_sha256:
        raise OracleError(
            f"{label} differs: size={size} sha256={digest}"
        )
    return {"size": size, "sha256": digest}


def normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OracleError(label + " must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise OracleError(label + " must be a normalized relative path")
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
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or urlsplit(value).scheme.lower() == "file"
    ):
        raise OracleError(label + " contains a machine-local path")


def catalog_contract(catalog: Mapping[str, Any], entry: Mapping[str, Any]) -> str:
    model = copy.deepcopy(dict(entry))
    for key in ("oracle", "runtime_support", "backends", "promotion_state"):
        model.pop(key, None)
    return sha256_bytes(
        canonical_json(
            {
                "schema_version": catalog.get("schema_version"),
                "suite": catalog.get("suite"),
                "model": model,
            }
        )
    )


def profile_contract(root: Mapping[str, Any], profile: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "schema_version": root.get("schema_version"),
                "format": root.get("format"),
                "model": copy.deepcopy(dict(profile)),
            }
        )
    )


def validate_catalog(path: Path) -> dict[str, Any]:
    catalog = load_json(path, "GENEB catalog")
    suite = catalog.get("suite")
    if (
        catalog.get("schema_version") != 1
        or not isinstance(suite, dict)
        or suite.get("id") != "geneb-v4"
        or suite.get("extractor_commit") != GENEB_REVISION
    ):
        raise OracleError("GENEB catalog identity differs")
    matches = [
        value
        for value in catalog.get("models", [])
        if isinstance(value, dict) and value.get("runtime_id") == RUNTIME_ID
    ]
    if len(matches) != 1:
        raise OracleError("catalog must contain exactly one METAGENE-1 row")
    entry = matches[0]
    if any(
        entry.get(key) != value
        for key, value in {
            "runtime_id": RUNTIME_ID,
            "geneb_model_id": MODEL_ID,
            "paper_name": MODEL_ID,
            "family": "transformer-decoder",
            "architecture": "llama-causal-lm",
        }.items()
    ):
        raise OracleError("METAGENE-1 catalog identity differs")

    source = entry.get("source")
    expected_files = [
        {"path": name, "size": size, "sha256": digest}
        for name, (size, digest) in SOURCE_FILES.items()
    ]
    if (
        not isinstance(source, dict)
        or source.get("kind") != "huggingface"
        or source.get("repo") != REPO
        or source.get("requested_revision") != REQUESTED_REVISION
        or source.get("revision") != REVISION
        or source.get("immutable") is not True
        or source.get("required_files") != expected_files
        or source.get("receipt")
        != {
            "required": True,
            "per_file_size": True,
            "per_file_sha256": True,
            "manifest_status": "resolved-at-fetch",
        }
    ):
        raise OracleError("METAGENE-1 exact eleven-file source closure differs")

    tokenizer = entry.get("tokenizer")
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("kind") != "bpe"
        or tokenizer.get("asset_source") != "model-source"
        or tokenizer.get("assets")
        != [
            {
                "role": "compiler-manifest",
                "path": TOKENIZER_MANIFEST_RELATIVE,
                "size": TOKENIZER_MANIFEST_SIZE,
                "sha256": TOKENIZER_MANIFEST_SHA256,
            }
        ]
        or tokenizer.get("add_special_tokens") is not True
        or tokenizer.get("padding_side") != "right"
        or tokenizer.get("pad_to") != "batch-max"
        or tokenizer.get("max_tokens") != 512
        or tokenizer.get("unknown_fields") != []
    ):
        raise OracleError("METAGENE-1 catalog tokenizer contract differs")
    context = entry.get("context")
    transform = entry.get("input_transform")
    if (
        not isinstance(context, dict)
        or context.get("declared_max_tokens") != 512
        or context.get("reference_max_tokens") != 512
        or context.get("length_policy") != "tokenizer-truncate"
        or not isinstance(transform, dict)
        or transform.get("case") != "preserve"
        or transform.get("strip_ascii_whitespace") is not False
        or transform.get("invalid") != "tokenizer-defined"
        or transform.get("special_tokens") != "tokenizer-default"
        or transform.get("token_truncation") != "right"
    ):
        raise OracleError("METAGENE-1 context/input transform differs")
    presets = entry.get("embedding_presets")
    if (
        not isinstance(presets, dict)
        or presets.get("reference")
        != {
            "hidden_tap": "last-hidden-state",
            "pooling": "unmasked-mean-all-token-rows",
            "special_tokens": "include",
            "mask_domain": "all-token-rows",
            "output_width": 4096,
        }
        or presets.get("normalized")
        != {
            "hidden_tap": "last-hidden-state",
            "pooling": "attention-mask-mean",
            "special_tokens": "include",
            "mask_domain": "attention-mask",
            "output_width": 4096,
        }
    ):
        raise OracleError("METAGENE-1 embedding presets differ")
    digest = catalog_contract(catalog, entry)
    if digest != CATALOG_CONTRACT_SHA256:
        raise OracleError("METAGENE-1 catalog contract digest differs")
    return {
        "catalog_contract_sha256": digest,
        "required_file_count": len(expected_files),
        "required_file_bytes": sum(item[0] for item in SOURCE_FILES.values()),
    }


def validate_profile(path: Path) -> dict[str, Any]:
    root = load_json(path, "GENEB decoder profiles")
    if root.get("schema_version") != 1 or root.get("format") != "geneb-decoder-converter-v1":
        raise OracleError("decoder profile manifest identity differs")
    matches = [
        value
        for value in root.get("models", [])
        if isinstance(value, dict) and value.get("runtime_id") == RUNTIME_ID
    ]
    if len(matches) != 1:
        raise OracleError("decoder profiles must contain exactly one METAGENE-1 row")
    profile = matches[0]
    expected_required = dict(EXPECTED_CONFIG)
    for key in ("max_sequence_length", "transformers_version", "bos_token_id", "eos_token_id", "pad_token_id"):
        expected_required.pop(key)
    expected_required["head_dim"] = 128
    if (
        profile.get("geneb_model_id") != MODEL_ID
        or profile.get("paper_name") != MODEL_ID
        or profile.get("catalog_architecture") != "llama-causal-lm"
        or profile.get("repo") != REPO
        or profile.get("revision") != REVISION
        or profile.get("source_format") != "safetensors"
        or profile.get("config_sha256") != SOURCE_FILES["config.json"][1]
        or profile.get("config_required") != expected_required
        or profile.get("config_defaults") != {"sliding_window": None}
        or profile.get("lm_head_policy") != "validate-and-omit"
        or profile.get("topology") != EXPECTED_TOPOLOGY
        or profile.get("tokenizer_manifest_sha256") != TOKENIZER_MANIFEST_SHA256
        or profile.get("tokenizer_source_receipt_contract_sha256")
        != TOKENIZER_SOURCE_RECEIPT_CONTRACT_SHA256
        or profile.get("tokenizer_asset_sha256") != TOKENIZER_ASSET_SHA256
        or profile.get("tokenizer_asset_size") != TOKENIZER_ASSET_SIZE
    ):
        raise OracleError("METAGENE-1 selected decoder profile differs")
    digest = profile_contract(root, profile)
    if digest != PROFILE_CONTRACT_SHA256:
        raise OracleError("METAGENE-1 selected profile contract digest differs")
    return {"selected_contract_sha256": digest, "topology": EXPECTED_TOPOLOGY}


def validate_manifest(path: Path) -> dict[str, Any]:
    checked_file(
        path,
        "METAGENE-1 tokenizer compiler manifest",
        TOKENIZER_MANIFEST_SIZE,
        TOKENIZER_MANIFEST_SHA256,
    )
    manifest = load_json(path, "METAGENE-1 tokenizer compiler manifest")
    expected = {
        "format": "evo-tokenizer-compiler-v1",
        "source": "huggingface-json",
        "kind": "bpe",
        "files": [
            {
                "role": "tokenizer",
                "name": "tokenizer.json",
                "size": SOURCE_FILES["tokenizer.json"][0],
                "sha256": SOURCE_FILES["tokenizer.json"][1],
            }
        ],
        "options": {
            "special_tokens": {
                "unk": "[UNK]",
                "pad": "[PAD]",
                "bos": "[BOS]",
                "eos": "[EOS]",
                "cls": None,
                "sep": "[SEP]",
                "mask": "[MASK]",
            },
            "padding_side": "right",
        },
    }
    if manifest != expected:
        raise OracleError("METAGENE-1 tokenizer compiler manifest semantics differ")
    return {"size": TOKENIZER_MANIFEST_SIZE, "sha256": TOKENIZER_MANIFEST_SHA256}


def expected_tensor_names() -> set[str]:
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(32):
        prefix = f"model.layers.{layer}."
        names.update(
            {
                prefix + "input_layernorm.weight",
                prefix + "self_attn.q_proj.weight",
                prefix + "self_attn.k_proj.weight",
                prefix + "self_attn.v_proj.weight",
                prefix + "self_attn.o_proj.weight",
                prefix + "post_attention_layernorm.weight",
                prefix + "mlp.gate_proj.weight",
                prefix + "mlp.up_proj.weight",
                prefix + "mlp.down_proj.weight",
            }
        )
    return names


def validate_config_and_index(source_root: Path) -> dict[str, Any]:
    config = load_json(source_root / "config.json", "METAGENE-1 config")
    differences = {
        key: (config.get(key), expected)
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if differences or config.get("auto_map") is not None:
        raise OracleError(f"METAGENE-1 config semantics differ: {differences}")
    index = load_json(
        source_root / "model.safetensors.index.json", "METAGENE-1 index"
    )
    if set(index) != {"metadata", "weight_map"} or index.get("metadata") != {
        "total_size": CHECKPOINT_LOGICAL_BYTES
    }:
        raise OracleError("METAGENE-1 checkpoint index metadata differs")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or set(weight_map) != expected_tensor_names():
        raise OracleError("METAGENE-1 checkpoint index tensor names differ")
    counts = {
        shard: list(weight_map.values()).count(shard)
        for shard in sorted(set(weight_map.values()))
    }
    if counts != EXPECTED_SHARD_COUNTS:
        raise OracleError("METAGENE-1 checkpoint index shard ownership differs")
    return {
        "logical_tensor_bytes": CHECKPOINT_LOGICAL_BYTES,
        "tensor_count": len(weight_map),
        "shard_tensor_counts": counts,
        "lm_head_shard": weight_map["lm_head.weight"],
    }


def validate_tokenizer_json_contract(source_root: Path) -> dict[str, Any]:
    source = load_json(source_root / "tokenizer.json", "official tokenizer.json")
    if (
        source.get("normalizer")
        != {"type": "Replace", "pattern": {"Regex": "^"}, "content": "_"}
        or source.get("pre_tokenizer") != {"type": "Whitespace"}
        or source.get("decoder") is not None
    ):
        raise OracleError("official tokenizer normalization/pretokenization differs")
    model = source.get("model")
    if (
        not isinstance(model, dict)
        or model.get("type") != "BPE"
        or model.get("dropout") is not None
        or model.get("unk_token") != "[UNK]"
        or model.get("fuse_unk") is not False
        or model.get("byte_fallback") is not False
        or model.get("ignore_merges") is not False
        or not isinstance(model.get("vocab"), dict)
        or len(model["vocab"]) != 1024
        or not isinstance(model.get("merges"), list)
        or len(model["merges"]) != 1012
    ):
        raise OracleError("official tokenizer BPE model differs")
    post = source.get("post_processor")
    if (
        not isinstance(post, dict)
        or post.get("type") != "TemplateProcessing"
        or post.get("single")
        != [
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[EOS]", "type_id": 0}},
        ]
    ):
        raise OracleError("official tokenizer EOS-only postprocessor differs")
    config = load_json(source_root / "tokenizer_config.json", "tokenizer config")
    expected_specials = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "mask_token": "[MASK]",
    }
    if (
        config.get("tokenizer_class") != "PreTrainedTokenizerFast"
        or config.get("model_max_length") != 512
        or config.get("clean_up_tokenization_spaces") is not False
        or any(config.get(key) != value for key, value in expected_specials.items())
    ):
        raise OracleError("official tokenizer wrapper config differs")
    special_map = load_json(
        source_root / "special_tokens_map.json", "special token map"
    )
    if special_map != expected_specials:
        raise OracleError("official raw special-token aliases differ")
    return {
        "normalizer": "replace-regex-start-with-underscore",
        "pre_tokenizer": "Whitespace",
        "post_processor": "EOS-only",
        "vocab_size": 1024,
        "merge_count": 1012,
        "fallback_tokenizer_model_used": False,
    }


def validate_small_sources(source_root: Path) -> dict[str, Any]:
    actual = {path.name for path in source_root.iterdir() if path.is_file()}
    expected = set(SMALL_SOURCE_FILES) | set(EXCLUDED_FALLBACK)
    if actual != expected:
        raise OracleError(
            f"small preflight source set differs: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    locks = {
        name: checked_file(source_root / name, name, size, digest)
        for name, (size, digest) in {**SMALL_SOURCE_FILES, **EXCLUDED_FALLBACK}.items()
    }
    index = validate_config_and_index(source_root)
    tokenizer = validate_tokenizer_json_contract(source_root)
    return {
        "files": locks,
        "index": index,
        "tokenizer": tokenizer,
        "checkpoint_shard_opened": False,
        "excluded_fallback": {
            "name": "tokenizer.model",
            **locks["tokenizer.model"],
            "cataloged": False,
            "executed": False,
        },
    }


def git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise OracleError(
            "git %s failed: %s" % (" ".join(arguments), result.stderr.strip())
        )
    return result.stdout.strip()


def validate_extractor(geneb_root: Path) -> dict[str, Any]:
    if git_output(geneb_root, "rev-parse", "HEAD") != GENEB_REVISION:
        raise OracleError("pinned GENEB revision differs")
    extractor = geneb_root / GENEB_EXTRACTOR_RELATIVE
    checked_file(
        extractor,
        "GENEB METAGENE extractor",
        GENEB_EXTRACTOR_SIZE,
        GENEB_EXTRACTOR_SHA256,
    )
    committed = subprocess.run(
        ["git", "-C", str(geneb_root), "show", "HEAD:" + GENEB_EXTRACTOR_RELATIVE],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if committed.returncode != 0 or committed.stdout != extractor.read_bytes():
        raise OracleError("GENEB METAGENE extractor is not the committed payload")
    source = extractor.read_text(encoding="utf-8")
    required_fragments = (
        "AutoTokenizer.from_pretrained(",
        "name_model, trust_remote_code=True",
        "AutoModelForCausalLM.from_pretrained(",
        'device_map="auto"',
        "output_hidden_states=True",
        "return_dict=True",
        "padding=True",
        "truncation=True",
        "out = self.model(**inputs)",
        "last_hidden = out.hidden_states[-1]",
        "embs = last_hidden.mean(dim=1)",
        "embs = embs.to(torch.float32)",
    )
    if any(fragment not in source for fragment in required_fragments):
        raise OracleError("GENEB METAGENE extractor call/pooling semantics differ")
    pyproject = geneb_root / "embedding_pipeline" / "pyproject.toml"
    lock_path = geneb_root / "embedding_pipeline" / "uv.lock"
    checked_file(
        pyproject,
        "GENEB embedding pyproject",
        GENEB_PYPROJECT_SIZE,
        GENEB_PYPROJECT_SHA256,
    )
    checked_file(
        lock_path,
        "GENEB embedding uv.lock",
        GENEB_UV_LOCK_SIZE,
        GENEB_UV_LOCK_SHA256,
    )
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    versions = {
        item.get("name"): item.get("version")
        for item in lock.get("package", [])
        if isinstance(item, dict) and item.get("name") in GENEB_LOCK_VERSIONS
    }
    if versions != GENEB_LOCK_VERSIONS:
        raise OracleError(f"GENEB executable dependency lock differs: {versions}")
    return {
        "repo": GENEB_REPO,
        "revision": GENEB_REVISION,
        "path": GENEB_EXTRACTOR_RELATIVE,
        "size": GENEB_EXTRACTOR_SIZE,
        "sha256": GENEB_EXTRACTOR_SHA256,
        "pyproject_sha256": GENEB_PYPROJECT_SHA256,
        "uv_lock_sha256": GENEB_UV_LOCK_SHA256,
        "locked_packages": versions,
        "reference_pooling": "unmasked-mean-all-token-rows",
        "normalized_pooling": "attention-mask-mean",
    }


def validate_descriptor(asset: Path, descriptor_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    asset_lock = checked_file(
        asset, "compiled METAGENE tokenizer", TOKENIZER_ASSET_SIZE, TOKENIZER_ASSET_SHA256
    )
    checked_file(
        descriptor_path,
        "compiled METAGENE tokenizer descriptor",
        TOKENIZER_DESCRIPTOR_SIZE,
        TOKENIZER_DESCRIPTOR_SHA256,
    )
    descriptor = load_json(descriptor_path, "tokenizer descriptor")
    expected = {
        "compiler_manifest_sha256": TOKENIZER_MANIFEST_SHA256,
        "converter.schema": "evo-tokenizer-conversion-receipt",
        "converter.version": 1,
        "source_receipt_contract_sha256": TOKENIZER_SOURCE_RECEIPT_CONTRACT_SHA256,
        "tokenizer.path": "tokenizers/geneb-metagene-1-bpe-v1.json",
        "tokenizer.profile": "evo-tokenizer-v1",
        "tokenizer.sha256": TOKENIZER_ASSET_SHA256,
        "tokenizer.size": TOKENIZER_ASSET_SIZE,
    }
    if descriptor != expected:
        raise OracleError("compiled tokenizer descriptor differs")
    return asset_lock, {
        "size": TOKENIZER_DESCRIPTOR_SIZE,
        "sha256": TOKENIZER_DESCRIPTOR_SHA256,
        "contract": descriptor,
    }


def read_inputs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    if len(paths) != len(INPUTS):
        raise OracleError("exactly two frozen input FASTA files are required")
    records = []
    for path, expected in zip(paths, INPUTS, strict=True):
        checked_file(path, expected["label"], expected["size"], expected["sha256"])
        try:
            lines = path.read_text(encoding="ascii").splitlines()
        except (OSError, UnicodeError) as error:
            raise OracleError("cannot read frozen FASTA input") from error
        if lines != [">" + expected["header"], expected["sequence"]]:
            raise OracleError(expected["label"] + " FASTA bytes differ")
        records.append(
            {
                "label": expected["label"],
                "path": f"configs/evidence/oracles/{RUNTIME_ID}/{expected['label']}.fa",
                "size": expected["size"],
                "sha256": expected["sha256"],
                "sequence": expected["sequence"],
                "input_ids": list(expected["input_ids"]),
                "attention_mask": [1] * len(expected["input_ids"]),
            }
        )
    return records


def package_lock() -> list[str]:
    values = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            values.append(
                f"{name.lower().replace('_', '-')}=={distribution.version}"
            )
    return sorted(set(values))


def validate_environment(*, full: bool) -> dict[str, Any]:
    host = validate_oracle_host(full=full)
    required = dict(REQUIRED_PACKAGE_VERSIONS)
    if host["host_profile"] == "linux-x86_64":
        required["torch"] = "2.7.1+cpu"
    if full:
        required.update(FULL_ONLY_PACKAGE_VERSIONS)
    observed = {}
    for name, expected in required.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise OracleError("required oracle package is missing: " + name) from error
        if actual != expected:
            raise OracleError(
                f"oracle package {name} differs: expected {expected}, got {actual}"
            )
        observed[name] = actual
    if (
        platform.python_implementation() != "CPython"
        or platform.python_version() != "3.11.15"
    ):
        raise OracleError("oracle requires exact CPython 3.11.15")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise OracleError(f"{name}={expected} is required")
    for name, expected in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        if os.environ.get(name) != expected:
            raise OracleError(f"{name}={expected} is required")
    import torch

    torch_cuda_version = getattr(torch.version, "cuda", None)
    torch_hip_version = getattr(torch.version, "hip", None)
    if torch_cuda_version is not None or torch_hip_version is not None:
        raise OracleError("oracle requires an official CPU-only Torch build")
    values = package_lock()
    build_vector = {
        "python_build": list(platform.python_build()),
        "python_compiler": platform.python_compiler(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch_cuda_version,
        "torch_hip_version": torch_hip_version,
        "torch_config_sha256": sha256_bytes(
            torch.__config__.show().encode("utf-8")
        ),
    }
    return {
        "python": platform.python_version(),
        "packages": observed,
        "package_vector_sha256": sha256_bytes(canonical_json(observed)),
        "package_lock": values,
        "package_lock_sha256": sha256_bytes(canonical_json(values)),
        "platform": host,
        "build_vector": build_vector,
        "build_vector_sha256": sha256_bytes(canonical_json(build_vector)),
        "accelerate_required_for_real_model": True,
        "accelerate_validated": full,
        "thread_environment": dict(THREAD_ENVIRONMENT),
    }


def validate_official_tokenizer(
    source_root: Path, records: Sequence[Mapping[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(
        str(source_root), local_files_only=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(source_root), local_files_only=True, trust_remote_code=True, use_fast=True
    )
    tokenizer_class = f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
    config_class = f"{config.__class__.__module__}.{config.__class__.__name__}"
    backend_class = tokenizer.backend_tokenizer.model.__class__.__name__
    if (
        tokenizer_class
        != "transformers.tokenization_utils_fast.PreTrainedTokenizerFast"
        or config_class != "transformers.models.llama.configuration_llama.LlamaConfig"
        or backend_class != "BPE"
        or tokenizer.vocab_size != 1024
        or len(tokenizer) != 1024
        or tokenizer.model_max_length != 512
        or tokenizer.padding_side != "right"
        or tokenizer.truncation_side != "right"
        or tokenizer.pad_token_id != 0
        or tokenizer.unk_token_id != 1
        or tokenizer.sep_token_id != 2
        or tokenizer.bos_token_id != 3
        or tokenizer.eos_token_id != 4
        or tokenizer.mask_token_id != 5
    ):
        raise OracleError("official fast BPE tokenizer/config class contract differs")
    encoded_records = []
    for record, expected in zip(records, INPUTS, strict=True):
        encoded = tokenizer(
            record["sequence"],
            add_special_tokens=True,
            truncation=True,
            return_attention_mask=True,
        )
        input_ids = [int(value) for value in encoded["input_ids"]]
        mask = [int(value) for value in encoded["attention_mask"]]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        if (
            input_ids != expected["input_ids"]
            or mask != [1] * len(input_ids)
            or tokens != expected["tokens"]
        ):
            raise OracleError(expected["label"] + " official token IDs differ")
        encoded_records.append({"input_ids": input_ids, "attention_mask": mask})
    batch = tokenizer(
        [record["sequence"] for record in records],
        add_special_tokens=True,
        truncation=True,
        padding=True,
        return_attention_mask=True,
    )
    expected_batch_ids = [
        INPUTS[0]["input_ids"] + [0] * 5,
        INPUTS[1]["input_ids"],
    ]
    expected_batch_mask = [[1] * 11 + [0] * 5, [1] * 16]
    if (
        batch["input_ids"] != expected_batch_ids
        or batch["attention_mask"] != expected_batch_mask
    ):
        raise OracleError("official right-padded two-record tokenizer batch differs")
    long_record = tokenizer(
        "N" * 2048,
        add_special_tokens=True,
        truncation=True,
        return_attention_mask=True,
    )
    if (
        long_record["input_ids"] != [6] + [11] * 510 + [4]
        or long_record["attention_mask"] != [1] * 512
    ):
        raise OracleError("official 512-token truncation/EOS retention differs")
    return tokenizer, {
        "class": tokenizer_class,
        "config_class": config_class,
        "backend_model": backend_class,
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "special_token_ids": {
            "pad": 0,
            "unk": 1,
            "sep": 2,
            "bos": 3,
            "eos": 4,
            "mask": 5,
        },
        "long_2048_nt_result_tokens": 512,
        "long_2048_nt_retains_eos": True,
        "config_max_sequence_length_2048_ignored_by_wrapper": True,
        "records": encoded_records,
        "batch_input_ids": expected_batch_ids,
        "batch_attention_mask": expected_batch_mask,
    }


def compiled_bpe_ids(asset: Mapping[str, Any], sequence: str) -> list[int]:
    vocab_rows = asset.get("vocab")
    model = asset.get("model")
    if not isinstance(vocab_rows, list) or not isinstance(model, dict):
        raise OracleError("compiled tokenizer BPE structure differs")
    vocab = {
        item.get("piece"): item.get("id")
        for item in vocab_rows
        if isinstance(item, dict)
    }
    merges = model.get("merges")
    if len(vocab) != 1024 or not isinstance(merges, list) or len(merges) != 1012:
        raise OracleError("compiled tokenizer vocab/merge count differs")
    ranks = {
        (pair[0], pair[1]): rank
        for rank, pair in enumerate(merges)
        if isinstance(pair, list) and len(pair) == 2
    }
    if len(ranks) != len(merges):
        raise OracleError("compiled tokenizer merge list is malformed or duplicated")
    symbols = list("_" + sequence)
    while len(symbols) > 1:
        candidates = [
            (ranks[(symbols[index], symbols[index + 1])], index)
            for index in range(len(symbols) - 1)
            if (symbols[index], symbols[index + 1]) in ranks
        ]
        if not candidates:
            break
        _, index = min(candidates)
        symbols[index : index + 2] = [symbols[index] + symbols[index + 1]]
    ids = [int(vocab.get(piece, 1)) for piece in symbols]
    ids.extend(asset["post_processor"]["suffix_ids"])
    return ids


def validate_compiled_tokenizer(
    asset_path: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    asset = load_json(asset_path, "compiled METAGENE tokenizer")
    if (
        asset.get("format") != "evo-tokenizer-v1"
        or asset.get("kind") != "bpe"
        or asset.get("normalization") != [{"op": "prepend-literal", "value": "_"}]
        or asset.get("pre_tokenizer") != {"kind": "hf-whitespace-ascii"}
        or asset.get("post_processor")
        != {
            "padding": {"pad_id": 0, "side": "right"},
            "prefix_ids": [],
            "suffix_ids": [4],
        }
        or asset.get("special_tokens")
        != {"bos": 3, "cls": None, "eos": 4, "mask": 5, "pad": 0, "sep": 2, "unk": 1}
        or asset.get("model", {}).get("literal_token_ids") != [0, 1, 2, 3, 4, 5]
    ):
        raise OracleError("compiled tokenizer semantic contract differs")
    observed = [compiled_bpe_ids(asset, record["sequence"]) for record in records]
    expected = [item["input_ids"] for item in INPUTS]
    if observed != expected:
        raise OracleError("compiled tokenizer does not match official input IDs")
    return {
        "official_input_parity": True,
        "input_ids": observed,
        "literal_token_ids": [0, 1, 2, 3, 4, 5],
        "merge_count": 1012,
    }


def f32_bytes(tensor: Any) -> bytes:
    return (
        tensor.detach()
        .to(device="cpu", dtype=__import__("torch").float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes(order="C")
    )


def validate_operator_contract(host_profile: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    import transformers.models.llama.modeling_llama as llama
    from transformers import LlamaConfig

    torch.manual_seed(0)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise OracleError("Torch interop thread count differs")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    source_hashes = {
        name: sha256_bytes(inspect.getsource(getattr(llama, name)).encode("utf-8"))
        for name in TRANSFORMERS_SOURCE_SHA256
    }
    if source_hashes != TRANSFORMERS_SOURCE_SHA256:
        raise OracleError("pinned Transformers Llama implementation source differs")

    x = torch.tensor(
        [[[0.25, -0.5, 0.75, -1.0], [1.25, -1.5, 1.75, -2.0]]],
        dtype=torch.float32,
    )
    rms = llama.LlamaRMSNorm(4, eps=1e-5)
    with torch.no_grad():
        rms.weight.copy_(torch.tensor([0.75, 1.0, 1.25, 1.5]))
        actual_rms = rms(x)
        repeated_rms = rms(x)
        manual_hidden = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        manual_rms = rms.weight * manual_hidden
    if not torch.equal(actual_rms, repeated_rms) or not torch.equal(actual_rms, manual_rms):
        raise OracleError("pinned Llama RMSNorm F32 operator differs")

    query = torch.tensor(
        [[[[0.2, -0.1, 0.4, 0.3], [-0.3, 0.5, 0.1, -0.2]]]],
        dtype=torch.float32,
    )
    key = query * torch.tensor(0.75, dtype=torch.float32)
    cos = torch.tensor([[[1.0, 0.75, 1.0, 0.75], [0.5, 0.25, 0.5, 0.25]]])
    sin = torch.tensor([[[0.0, 0.25, 0.0, 0.25], [0.5, 0.75, 0.5, 0.75]]])
    actual_q, actual_k = llama.apply_rotary_pos_emb(query, key, cos, sin)
    rotate = lambda value: torch.cat((-value[..., 2:], value[..., :2]), dim=-1)
    manual_q = query * cos.unsqueeze(1) + rotate(query) * sin.unsqueeze(1)
    manual_k = key * cos.unsqueeze(1) + rotate(key) * sin.unsqueeze(1)
    if not torch.equal(actual_q, manual_q) or not torch.equal(actual_k, manual_k):
        raise OracleError("pinned split-half RoPE operator differs")

    q = torch.tensor(
        [[[[0.2, -0.1], [0.4, 0.3], [-0.3, 0.5]], [[0.1, 0.6], [-0.2, 0.4], [0.7, -0.5]]]],
        dtype=torch.float32,
    )
    k = q * torch.tensor(0.8, dtype=torch.float32)
    v = torch.flip(q, dims=[-1]) * torch.tensor(1.25, dtype=torch.float32)
    sdpa = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    repeated_sdpa = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(2.0)
    causal = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    manual_attention = torch.matmul(torch.softmax(scores, dim=-1, dtype=torch.float32), v)
    if (
        not torch.equal(sdpa, repeated_sdpa)
        or not torch.allclose(sdpa, manual_attention, atol=1e-6, rtol=0.0)
    ):
        raise OracleError("Torch CPU-F32 SDPA/eager causal attention contract differs")

    mlp_config = LlamaConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        mlp_bias=False,
    )
    mlp = llama.LlamaMLP(mlp_config)
    with torch.no_grad():
        for index, parameter in enumerate(mlp.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_((values - values.mean()) / float(32 + index))
        actual_mlp = mlp(x)
        manual_mlp = mlp.down_proj(functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    if not torch.equal(actual_mlp, manual_mlp):
        raise OracleError("pinned Llama SwiGLU operator differs")
    observed_hashes = {
        "rmsnorm_raw_f32_sha256": sha256_bytes(f32_bytes(actual_rms)),
        "split_half_rope_raw_f32_sha256": sha256_bytes(
            f32_bytes(torch.cat((actual_q, actual_k), dim=0))
        ),
        "sdpa_raw_f32_sha256": sha256_bytes(f32_bytes(sdpa)),
        "swiglu_raw_f32_sha256": sha256_bytes(f32_bytes(actual_mlp)),
    }
    if host_profile == "darwin-arm64":
        expected_hashes: Mapping[str, str] | None = (
            DARWIN_ARM64_OPERATOR_SHA256
        )
    elif host_profile == "linux-x86_64":
        expected_hashes = LINUX_X86_64_OPERATOR_SHA256
    else:
        raise OracleError("unsupported operator host profile")
    if expected_hashes is not None and observed_hashes != expected_hashes:
        raise OracleError(
            f"{host_profile} pinned Llama operator vector differs"
        )
    return {
        "kind": "transformers-4.57.3-standard-llama-cpu-f32",
        "host_profile": host_profile,
        "platform_vector_status": (
            "frozen"
            if expected_hashes is not None
            else "candidate-pending-remote-audit-only-freeze"
        ),
        "expected_sha256": (
            dict(expected_hashes) if expected_hashes is not None else None
        ),
        "source_sha256": source_hashes,
        **observed_hashes,
        "repeated_bit_equal": True,
        "sdpa_matches_explicit_causal_attention_atol": 1e-6,
    }


def validate_full_sources(
    source_root: Path, receipt_path: Path, catalog_path: Path
) -> tuple[dict[str, Any], str]:
    if source_root.name != REVISION or not source_root.is_dir():
        raise OracleError("full source root must be the pinned revision snapshot")
    actual_names = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual_names != set(SOURCE_FILES):
        raise OracleError("full source snapshot is not the exact eleven-file closure")
    receipt = load_json(receipt_path, "source receipt")
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
        raise OracleError("source receipt fields differ")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "source-checkpoint"
        or receipt.get("model_id") != RUNTIME_ID
        or receipt.get("repo") != REPO
        or receipt.get("requested_revision") != REQUESTED_REVISION
        or receipt.get("resolved_revision") != REVISION
        or receipt.get("source_kind") != "huggingface"
        or receipt.get("load_path") is not None
        or receipt.get("catalog_contract_sha256") != CATALOG_CONTRACT_SHA256
        or Path(receipt.get("catalog_path", "")).resolve() != catalog_path.resolve()
    ):
        raise OracleError("source receipt provenance/catalog binding differs")
    rows = receipt.get("files")
    if not isinstance(rows, list) or len(rows) != len(SOURCE_FILES):
        raise OracleError("source receipt must contain exactly eleven files")
    verified = {}
    portable_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "path", "size", "sha256"}:
            raise OracleError(f"source receipt files[{index}] differs")
        name = normalized_relative(row.get("name"), f"source receipt files[{index}].name")
        if name in verified or name not in SOURCE_FILES:
            raise OracleError("source receipt file set is duplicated or unexpected")
        size, digest = SOURCE_FILES[name]
        if row.get("size") != size or row.get("sha256") != digest:
            raise OracleError("source receipt size/SHA differs for " + name)
        logical = source_root / name
        locator = Path(row["path"])
        if not locator.is_file() or not logical.is_file():
            raise OracleError("source receipt locator is missing for " + name)
        try:
            same = os.path.samefile(locator, logical)
        except OSError as error:
            raise OracleError("cannot resolve source receipt locator") from error
        if not same:
            raise OracleError("source receipt locator differs from snapshot for " + name)
        checked_file(logical, "full source " + name, size, digest)
        verified[name] = {"size": size, "sha256": digest}
        portable_rows.append({"name": name, "size": size, "sha256": digest})
    if set(verified) != set(SOURCE_FILES):
        raise OracleError("source receipt exact file set differs")
    validate_config_and_index(source_root)
    validate_tokenizer_json_contract(source_root)
    portable = {
        "schema_version": 1,
        "kind": "source-checkpoint",
        "model_id": RUNTIME_ID,
        "repo": REPO,
        "requested_revision": REQUESTED_REVISION,
        "resolved_revision": REVISION,
        "source_kind": "huggingface",
        "catalog_contract_sha256": CATALOG_CONTRACT_SHA256,
        "files": sorted(portable_rows, key=lambda item: item["name"]),
    }
    return dict(sorted(verified.items())), sha256_bytes(canonical_json(portable))


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json(value))


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
    values = [float(value) for value in flat[:16]]
    return {
        "dtype": "float32",
        "shape": list(tensor.shape),
        "raw_little_endian_f32_sha256": sha256_bytes(f32_bytes(tensor)),
        "all_finite": bool(torch.isfinite(flat).all().item()),
        "first_16_decimal": values,
        "first_16_hex": [value.hex() for value in values],
    }


def comparison(left: Any, right: Any) -> dict[str, Any]:
    import torch

    left64 = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right64 = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    delta = (left64 - right64).abs()
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    return {
        "bitwise_equal": bool(torch.equal(left, right)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "cosine": float((torch.dot(left64, right64) / denominator).item()),
    }


def execute_oracle(
    source_root: Path,
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    output_root: Path,
    source_files: Mapping[str, Any],
    receipt_contract_sha256: str,
    environment: Mapping[str, Any],
    extractor: Mapping[str, Any],
    operator: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise OracleError("oracle output directory must be absent or empty")
    output_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(source_root),
        revision=REVISION,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": "cpu"},
        use_safetensors=True,
        output_hidden_states=True,
        return_dict=True,
        output_loading_info=True,
    )
    expected_loading_fields = {
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    }
    if set(loading) != expected_loading_fields or any(loading.get(key) for key in expected_loading_fields):
        raise OracleError(f"official checkpoint loading info differs: {loading}")
    model_class = f"{model.__class__.__module__}.{model.__class__.__name__}"
    state = model.state_dict()
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    state_logical_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in state.values()
    )
    device_map = getattr(model, "hf_device_map", None)
    if (
        model_class != "transformers.models.llama.modeling_llama.LlamaForCausalLM"
        or model.config.model_type != "llama"
        or model.config.hidden_size != 4096
        or model.config.num_hidden_layers != 32
        or model.config.num_attention_heads != 32
        or model.config.num_key_value_heads != 32
        or model.config.intermediate_size != 11008
        or model.config.vocab_size != 1024
        or model.config.max_position_embeddings != 512
        or getattr(model.config, "_attn_implementation", None) != "sdpa"
        or len(state) != SOURCE_TENSOR_COUNT
        or parameter_count != PARAMETER_COUNT
        or state_logical_bytes != CHECKPOINT_LOGICAL_BYTES
        or any(tensor.dtype != torch.float32 for tensor in state.values())
        or any(tensor.device.type != "cpu" for tensor in state.values())
        or any(tensor.is_meta for tensor in state.values())
        or any(parameter.dtype != torch.float32 for parameter in parameters)
        or any(parameter.device.type != "cpu" for parameter in parameters)
        or any(parameter.is_meta for parameter in parameters)
        or device_map != {"": "cpu"}
    ):
        raise OracleError("official standard Llama model/state/CPU-F32 contract differs")
    del state
    model.eval()

    source_manifest = [
        {"name": name, "size": item["size"], "sha256": item["sha256"]}
        for name, item in sorted(source_files.items())
    ]
    environment_lock = {
        "schema_version": 1,
        "oracle_contract": "geneb-independent-oracle-v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": list(environment["package_lock"]),
        "source_files": dict(sorted(source_files.items())),
        "thread_environment": dict(environment["thread_environment"]),
        "build_vector": dict(environment["build_vector"]),
        "operator": dict(operator),
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "num_threads": 1,
            "num_interop_threads": 1,
            "float32_matmul_precision": "highest",
            "cuda_version": None,
            "hip_version": None,
        },
        "transformers": {
            "version": transformers.__version__,
            "model_class": model_class,
            "attention_implementation": model.config._attn_implementation,
            "tokenizer_class": (
                f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
            ),
            "local_files_only": True,
            "trust_remote_code": True,
        },
        "extractor": {
            "repo": extractor["repo"],
            "revision": extractor["revision"],
            "file": extractor["path"],
            "size": extractor["size"],
            "sha256": extractor["sha256"],
        },
    }
    base_provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-normalized",
        "benchmark_semantics": "geneb-v4-normalized",
        "catalog_contract_sha256": CATALOG_CONTRACT_SHA256,
        "converter_profile_contract_sha256": PROFILE_CONTRACT_SHA256,
        "source_repo": REPO,
        "source_requested_revision": REQUESTED_REVISION,
        "source_revision": REVISION,
        "source_receipt_contract_sha256": receipt_contract_sha256,
        "source_file_manifest_sha256": sha256_bytes(canonical_json(source_manifest)),
        "source_weight_dtype": "bfloat16",
        "extractor_repo": extractor["repo"],
        "extractor_revision": extractor["revision"],
        "extractor_module": extractor["path"],
        "extractor_sha256": extractor["sha256"],
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "hidden_tap": "outputs.hidden_states[-1]",
        "pooling": "attention-mask-f32-sum-then-direct-f32-division",
        "reference_pooling": "last_hidden.mean(dim=1)",
        "special_tokens": "include-bos-and-eos",
        "padding_side": "right",
        "output_width": 4096,
        "parameter_count": parameter_count,
        "model_class": model_class,
        "model_attention_implementation": model.config._attn_implementation,
        "model_state_tensor_count": SOURCE_TENSOR_COUNT,
        "model_state_logical_bytes": state_logical_bytes,
        "model_placement": {
            "requested_device_map": {"": "cpu"},
            "observed_hf_device_map": device_map,
            "cpu_f32_only": True,
            "cuda_mps_disk_meta_rejected": True,
        },
        "loader": "official-from_pretrained-device_map-cpu",
        "official_standard_transformers_code": True,
        "execution_dtype": "float32",
        "repeated_forward_bit_equal": True,
        "tolerances": dict(TOLERANCES),
        "independent_of_evo_native_runtime": True,
    }

    outputs = []
    report_inputs = []
    for record, expected in zip(records, INPUTS, strict=True):
        encoded = tokenizer(
            [record["sequence"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if (
            encoded["input_ids"].tolist() != [expected["input_ids"]]
            or encoded["attention_mask"].tolist() != [[1] * len(expected["input_ids"])]
        ):
            raise OracleError(expected["label"] + " execution tokenization differs")
        encoded = {name: value.to(device="cpu") for name, value in encoded.items()}
        repeated = []
        reference = None
        hidden_summary = None
        for _ in range(2):
            with torch.inference_mode():
                result = model(**encoded)
            if result.hidden_states is None or len(result.hidden_states) != 33:
                raise OracleError(expected["label"] + " hidden-state tap count differs")
            hidden = result.hidden_states[-1]
            if (
                hidden.shape != (1, len(expected["input_ids"]), 4096)
                or hidden.dtype != torch.float32
                or hidden.device.type != "cpu"
                or hidden.is_meta
            ):
                raise OracleError(expected["label"] + " final hidden shape/dtype differs")
            mask = encoded["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
            normalized = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            repeated.append(normalized.to(device="cpu", dtype=torch.float32).contiguous())
            reference = hidden.mean(dim=1).to(device="cpu", dtype=torch.float32).contiguous()
            hidden_summary = tensor_summary(hidden)
        if not torch.equal(repeated[0], repeated[1]):
            raise OracleError(expected["label"] + " repeated upstream output differs")
        array = repeated[0].numpy().astype("<f4", copy=False)
        npy_path = output_root / (expected["label"] + ".attention-mask-mean.f32.npy")
        np.save(npy_path, array, allow_pickle=False)
        flat = [float(value) for value in array.reshape(-1)]
        provenance = dict(base_provenance)
        provenance.update(
            {
                "input_label": expected["label"],
                "expected_input_ids": list(expected["input_ids"]),
                "token_count": len(expected["input_ids"]),
                "reference_unmasked_mean_vs_normalized": comparison(
                    reference, repeated[0]
                ),
            }
        )
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": RUNTIME_ID,
            "input_sha256": expected["sha256"],
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": flat,
            "environment_lock": environment_lock,
            "provenance": provenance,
        }
        require_portable(vector, "oracle vector")
        vector_path = output_root / (expected["label"] + ".independent-oracle-vector.json")
        vector_path.write_bytes(canonical_json(vector))
        outputs.append(
            {
                "label": expected["label"],
                "vector": {"path": vector_path.name, "size": vector_path.stat().st_size, "sha256": sha256_file(vector_path)},
                "npy": {"path": npy_path.name, "size": npy_path.stat().st_size, "sha256": sha256_file(npy_path)},
            }
        )
        report_inputs.append(
            {
                "label": expected["label"],
                "sha256": expected["sha256"],
                "sequence": expected["sequence"],
                "input_ids": list(expected["input_ids"]),
                "attention_mask": [1] * len(expected["input_ids"]),
                "hidden": hidden_summary,
                "output": {
                    "shape": [1, 4096],
                    "npy_path": npy_path.name,
                    "npy_sha256": sha256_file(npy_path),
                    "raw_f32_sha256": sha256_bytes(array.tobytes(order="C")),
                    "vector_sha256": sha256_file(vector_path),
                    "first_16": flat[:16],
                    "reference_unmasked_mean_vs_normalized": comparison(
                        reference, repeated[0]
                    ),
                },
            }
        )
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "oracle_profile": "official-geneb-cpu-f32-normalized",
        "source": {
            "kind": "huggingface",
            "repo": REPO,
            "requested_revision": REQUESTED_REVISION,
            "revision": REVISION,
            "files": source_manifest,
            "receipt_contract_sha256": receipt_contract_sha256,
        },
        "inputs": report_inputs,
        "environment_lock": environment_lock,
        "provenance": base_provenance,
        "outputs": outputs,
    }
    require_portable(report, "oracle report")
    report_path = output_root / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))
    return {"report": str(report_path), "outputs": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--geneb-repo", required=True, type=Path)
    parser.add_argument("--tokenizer-asset", required=True, type=Path)
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--input", action="append", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve(strict=True)
    catalog = (
        args.catalog.resolve(strict=True)
        if args.catalog is not None
        else repo_root / "configs" / "geneb-models.json"
    )
    profiles = (
        args.profiles.resolve(strict=True)
        if args.profiles is not None
        else repo_root / "configs" / "geneb-decoder-models.json"
    )
    manifest = (
        args.manifest.resolve(strict=True)
        if args.manifest is not None
        else repo_root / TOKENIZER_MANIFEST_RELATIVE
    )
    source_root = args.source_root.resolve(strict=True)
    geneb_root = args.geneb_repo.resolve(strict=True)
    asset = args.tokenizer_asset.resolve(strict=True)
    descriptor = args.tokenizer_descriptor.resolve(strict=True)
    input_paths = (
        [path.resolve(strict=True) for path in args.input]
        if args.input
        else [
            repo_root / "configs" / "evidence" / "oracles" / RUNTIME_ID / f"input-{index}.fa"
            for index in range(2)
        ]
    )

    catalog_lock = validate_catalog(catalog)
    profile_lock = validate_profile(profiles)
    manifest_lock = validate_manifest(manifest)
    extractor_lock = validate_extractor(geneb_root)
    asset_lock, descriptor_lock = validate_descriptor(asset, descriptor)
    records = read_inputs(input_paths)
    environment = validate_environment(full=not args.audit_only)
    operator_lock = validate_operator_contract(
        environment["platform"]["host_profile"]
    )
    if (
        not args.audit_only
        and operator_lock["platform_vector_status"] != "frozen"
    ):
        raise OracleError(
            "Linux-x86_64 operator vectors must be frozen from a remote "
            "--audit-only result before real oracle output"
        )
    if args.audit_only:
        if args.receipt is not None or args.output_dir is not None:
            raise OracleError("audit-only must not receive receipt/output arguments")
        small = validate_small_sources(source_root)
        source_files = None
        receipt_contract_sha256 = None
    else:
        if args.receipt is None or args.output_dir is None:
            raise OracleError("real oracle requires --receipt and --output-dir")
        source_files, receipt_contract_sha256 = validate_full_sources(
            source_root, args.receipt.resolve(strict=True), catalog
        )
    tokenizer, tokenizer_lock = validate_official_tokenizer(source_root, records)
    compiled_lock = validate_compiled_tokenizer(asset, records)

    if args.audit_only:
        result = {
            "schema_version": 1,
            "runtime_id": RUNTIME_ID,
            "status": "preflight-pass-no-checkpoint-shard-opened",
            "catalog": catalog_lock,
            "profile": profile_lock,
            "manifest": manifest_lock,
            "source": small,
            "extractor": extractor_lock,
            "environment": environment,
            "tokenizer": {
                "official": tokenizer_lock,
                "compiled_asset": asset_lock,
                "compiled_descriptor": descriptor_lock,
                "compiled_parity": compiled_lock,
            },
            "operator": operator_lock,
            "inputs": records,
            "tolerances": TOLERANCES,
            "official_model_executed": False,
            "native_runtime_imported": False,
            "conversion_executed": False,
        }
        print(canonical_json(result).decode("ascii"), end="")
        return 0

    if source_files is None or receipt_contract_sha256 is None:
        raise OracleError("validated real-oracle source contract is missing")
    result = execute_oracle(
        source_root,
        records,
        tokenizer,
        args.output_dir.resolve(),
        source_files,
        receipt_contract_sha256,
        environment,
        extractor_lock,
        operator_lock,
    )
    print(canonical_json(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, OracleError, ValueError) as error:
        print(
            "generate_geneb_metagene_1_upstream_oracle: error: %s" % error,
            file=sys.stderr,
        )
        raise SystemExit(2) from error
