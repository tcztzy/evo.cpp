#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned, normalized, GPU-free Evo-1-131k CPU-F32 oracle.

The clean upstream reference is intentionally not represented by this program:
the pinned model requires CUDA-only ``flash_attn`` during construction.  This
generator validates that clean failure, applies the B131 local attention patch,
and labels every output ``pinned-upstream-normalized-gpu-free``.  ``--audit-only``
opens only the thirteen small source/config files; it never opens a checkpoint
shard, imports native evo.cpp code, or executes the full model.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import torch


RUNTIME_ID = "geneb-evo-1-131k"
REPO_ID = "togethercomputer/evo-1-131k-base"
REQUESTED_REVISION = "1.1_fix"
REVISION = "c206aab77ae5967a069c4200ecb1858588528c9d"
PROFILE_PATH = "configs/geneb-evo1-models.json"
PROFILE_SIZE = 3_429
PROFILE_SHA256 = "ca65c6fb6b3d7cfc0df44711809fe1ca707057a0a75b954d094ff61936915422"
TOKENIZER_MANIFEST_PATH = "configs/tokenizers/geneb-evo1-byte-v1.json"
TOKENIZER_MANIFEST_SIZE = 309
TOKENIZER_MANIFEST_SHA256 = (
    "c6793d5a5bc25945f9b8b892cde3ff366569d8f697d1258c6ec5b0f0738a8f3d"
)
TOKENIZER_SOURCE_PATH = "configs/tokenizers/geneb-evo1-byte-source-v1.json"
TOKENIZER_SOURCE_SIZE = 34_215
TOKENIZER_SOURCE_SHA256 = (
    "209ad95ffb1c31c6ecbe2d1a7c8ab4170089f0a72db54f3a9f2e0e8399248b25"
)
PATCH_PATH = "configs/evidence/oracles/geneb-evo-1-131k/gpu-free-patch.json"
PATCH_SIZE = 3_260
PATCH_SHA256 = "2d08ae84da2ecc3456eb638a4240941b776e4b9500aa1bb688790391e0a73ca2"
PORTABLE_PATH = (
    "configs/evidence/oracles/geneb-evo-1-131k/evo1_portable_mha.py"
)
PORTABLE_SIZE = 9_752
PORTABLE_SHA256 = (
    "7538743b25b7381330723108e53353912ff06ab3c6c0a643ab4caa473bd9c51e"
)
CHECKPOINT_MANIFEST_SHA256 = (
    "8021ede2367eeb16b6a37a27f41024efd7cede2e5b76d0cc50746ddde81cf18a"
)
SOURCE_TENSOR_COUNT = 438
SOURCE_TENSOR_BYTES = 12_913_164_672
F32_PARAMETER_BYTES = 25_811_124_992
MINIMUM_ORACLE_HOST_MEMORY_BYTES = 48 * 1024**3
RECOMMENDED_ORACLE_HOST_MEMORY_BYTES = 64 * 1024**3
SUPPORTED_ORACLE_HOSTS = {
    ("Darwin", "arm64"): "darwin-arm64",
    ("Linux", "x86_64"): "linux-x86_64",
}
ATTENTION_LAYERS = (8, 16, 24)
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIMENSION = 128
MAX_TOKENS = 131_072
TOLERANCES = {"max_abs": 0.01, "mean_abs": 0.0005, "cosine": 0.999999}

SOURCE_FILES = {
    "cache.py": (
        1_378,
        "4aeaaf482d1da04ee9c31d47fe85bdce8e9c9e03a5cd3e183843e902c9114b73",
    ),
    "config.json": (
        1_896,
        "8d5c77f3a67a9e4a0c6abc3771a81aa8300fb0efb60e4ec534b34d3b71c949d2",
    ),
    "configuration_hyena.py": (
        3_134,
        "5be99310371546c6a28d0287f6b9d6149a1c827341adf818ec8d0fc8bb45b277",
    ),
    "engine.py": (
        13_450,
        "8fe4cd97ec6b6be43807fe2d33ec16b53d3ff831e5d33c25802e701041221639",
    ),
    "layers.py": (
        5_387,
        "52c7909406acccd07d872f839cd22f3f494c8555ac96d51ce7eb2629cc8a98da",
    ),
    "model-00001-of-00003.safetensors": (
        4_980_059_464,
        "c4241bfbca221240a21ecdc2e927326ed5a603f93b74826ab13b82a3e9a0b087",
    ),
    "model-00002-of-00003.safetensors": (
        4_929_849_248,
        "554e7dff40a84e3ba81e98b856d0df5a8014ec4e19c2c9f0790ced34efddda03",
    ),
    "model-00003-of-00003.safetensors": (
        3_003_304_856,
        "2b573d0b24dc978ddf96b9c8bbe6f4f4ba2ddd718aeda8a7f9160cc2a1cae950",
    ),
    "model.py": (
        19_474,
        "6a03a9c28bd7e282cb253d051ae7ec39159d14d3a1b2272913c8ff1747f3d87a",
    ),
    "model.safetensors.index.json": (
        34_860,
        "f23c7372cf792218f7a4bd7433d00561555aedcba58410cc79ffae52b824c045",
    ),
    "modeling_hyena.py": (
        5_549,
        "adcbcdc7690057b8d109b9bd520bee05a534d88af4cb8ba59426e78900a8cee0",
    ),
    "positional_embeddings.py": (
        4_944,
        "fc22bdb7447ae3c0adea17c00442c3ec040f0dbb61e54c4486d14bbc5217c3ae",
    ),
    "special_tokens_map.json": (
        3,
        "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
    ),
    "tokenizer.py": (
        4_397,
        "15e37ca8a1994a1bb4a9ac526d746808ef77a1b68e26ea9fe8c03f823d25c7be",
    ),
    "tokenizer_config.json": (
        299,
        "52fdc041aadead5cc38e72ba7db3e64965526cd974f5f9aa1185a3d9b7aca32c",
    ),
    "utils.py": (
        2_870,
        "c757a53aeab1e0c0292f9b117cd982121bd980f7d72823930f5ed4938c9eb64b",
    ),
}
WEIGHT_FILES = {
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
}
CONVERSION_ASSETS = (
    "config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)
INPUTS = (
    {
        "name": "input-0",
        "path": "configs/evidence/oracles/geneb-evo-1-131k/input-0.fa",
        "size": 36,
        "sha256": "f5a9f822a2bf61d50ed0456a6d36b16f912cc3b781166a8f25679ce81e11a5ad",
        "sequence": "ACGTNACGT",
        "input_ids": [65, 67, 71, 84, 78, 65, 67, 71, 84],
    },
    {
        "name": "input-1",
        "path": "configs/evidence/oracles/geneb-evo-1-131k/input-1.fa",
        "size": 35,
        "sha256": "c98d5083f58738c71bd316bcf96d3e07335093914cffd9bd110ffa7cb9612e59",
        "sequence": "ttgca?NX",
        "input_ids": [116, 116, 103, 99, 97, 63, 78, 88],
    },
)
EXPECTED_PACKAGES = {
    "accelerate": "0.27.2",
    "einops": "0.7.0",
    "numpy": "1.26.4",
    "safetensors": "0.4.2",
    "torch": "2.1.2",
    "transformers": "4.38.2",
}
LINUX_X86_64_EXPECTED_PACKAGES = {
    **EXPECTED_PACKAGES,
    "torch": "2.1.2+cpu",
}

MODEL_IMPORT_OLD = b'''try:\n    from flash_attn.modules.mha import MHA\nexcept ImportError:\n    "flash_attn not installed"\n    \ntry:\n    from .positional_embeddings import swap_mha_rope\nexcept ImportError:\n    "could not import swap_mha_rope from positional_embeddings.py"\n'''
MODEL_IMPORT_NEW = b'''from .evo1_portable_mha import MHA\nfrom .positional_embeddings import swap_mha_rope\n'''
POSITION_IMPORT_OLD = b'''from flash_attn.layers.rotary import RotaryEmbedding\nfrom flash_attn.modules.mha import MHA\n'''
POSITION_IMPORT_NEW = b'''from .evo1_portable_mha import MHA, RotaryEmbedding\n'''

OPERATOR_OUTPUT = (
    0.13960000872612,
    -0.08720000088214874,
    -0.027800001204013824,
    -0.03580000251531601,
    0.058254726231098175,
    -0.1141967922449112,
    0.08648505806922913,
    0.0028106942772865295,
    0.10221720486879349,
    -0.13967476785182953,
    0.06042758375406265,
    -0.05038441717624664,
)
OPERATOR_OUTPUT_SHA256 = (
    "dd087a9938611dd258597594301a0d16518579d1b2db8fdfcb9edaf9a18467f8"
)
# Filled only after the pinned environment has produced a Linux/x86_64
# ``--audit-only`` result and that result has been reviewed.  A missing value
# deliberately blocks real model output on Linux while still allowing the
# small audit to emit the candidate hash.
LINUX_X86_64_OPERATOR_OUTPUT_SHA256: str | None = (
    "7793f22e0775624de4616355f92055ccc2a8f62e0b9662bfeb2ac718ba23f3f1"
)


class OracleError(RuntimeError):
    """Raised when a frozen source, patch, input, or environment differs."""


def canonical_json(value: object) -> bytes:
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
    torch_cuda_version = getattr(torch.version, "cuda", None)
    torch_hip_version = getattr(torch.version, "hip", None)
    if torch_cuda_version is not None or torch_hip_version is not None:
        raise OracleError("oracle requires an official CPU-only Torch build")
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
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "build_vector": build_vector,
        "build_vector_sha256": sha256_bytes(canonical_json(build_vector)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def package_lock(packages: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {"name": name, "version": packages[name]}
        for name in sorted(packages)
    ]


def catalog_contract_sha256(
    catalog: Mapping[str, Any], row: Mapping[str, Any]
) -> str:
    model = json.loads(json.dumps(row))
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


def checked_file(
    path: Path, label: str, expected_size: int, expected_sha256: str
) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{label} must be a nonsymlink regular file")
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    if len(payload) != expected_size or digest != expected_sha256:
        raise OracleError(
            f"{label} differs: size={len(payload)} sha256={digest}"
        )
    return payload


def load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OracleError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} root must be an object")
    return value


def source_descriptors(raw: object, label: str) -> dict[str, tuple[int, str]]:
    if not isinstance(raw, list):
        raise OracleError(f"{label} must be an array")
    result: dict[str, tuple[int, str]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise OracleError(f"{label}[{index}] differs")
        name, size, digest = item["path"], item["size"], item["sha256"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in result
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            raise OracleError(f"{label}[{index}] differs")
        result[name] = (size, digest)
    return result


def validate_profile_and_catalog(repo_root: Path, catalog_path: Path) -> dict[str, Any]:
    profile_payload = checked_file(
        repo_root / PROFILE_PATH,
        "Evo-1 profile",
        PROFILE_SIZE,
        PROFILE_SHA256,
    )
    profile_root = load_json_bytes(profile_payload, "Evo-1 profile")
    models = profile_root.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise OracleError("Evo-1 profile selection differs")
    profile = models[0]
    expected_files = {
        name: {"size": size, "sha256": digest}
        for name, (size, digest) in SOURCE_FILES.items()
    }
    expected_topology = {
        "vocab_size": 512,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "inner_width": 10928,
        "state_width": 8,
        "short_filter_width": 3,
        "max_seqlen": MAX_TOKENS,
        "norm_epsilon": 0.000001,
        "rope_theta": 10000.0,
        "rope_scaling_factor": 16.0,
        "attention_layers": list(ATTENTION_LAYERS),
    }
    if (
        profile.get("runtime_id") != RUNTIME_ID
        or profile.get("repo") != REPO_ID
        or profile.get("requested_revision") != REQUESTED_REVISION
        or profile.get("revision") != REVISION
        or profile.get("checkpoint_manifest_sha256")
        != CHECKPOINT_MANIFEST_SHA256
        or profile.get("source_files") != expected_files
        or profile.get("conversion_assets") != list(CONVERSION_ASSETS)
        or profile.get("topology") != expected_topology
        or profile.get("tokenizer")
        != {
            "compiler_manifest": "tokenizers/geneb-evo1-byte-v1.json",
            "compiler_manifest_sha256": TOKENIZER_MANIFEST_SHA256,
        }
    ):
        raise OracleError("Evo-1 selected profile contract differs")

    catalog = load_json_bytes(catalog_path.read_bytes(), "GENEB catalog")
    rows = catalog.get("models")
    matches = (
        [row for row in rows if isinstance(row, dict) and row.get("runtime_id") == RUNTIME_ID]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1:
        raise OracleError("Evo-1 catalog row is missing or duplicated")
    row = matches[0]
    source = row.get("source")
    tokenizer = row.get("tokenizer")
    benchmark = row.get("benchmark_provenance")
    provenance = row.get("provenance")
    if not all(isinstance(value, dict) for value in (source, tokenizer, benchmark, provenance)):
        raise OracleError("Evo-1 catalog contract is incomplete")
    if (
        source.get("repo") != REPO_ID
        or source.get("requested_revision") != REQUESTED_REVISION
        or source.get("revision") != REVISION
        or source_descriptors(source.get("required_files"), "catalog required_files")
        != SOURCE_FILES
        or tokenizer.get("assets")
        != [
            {
                "role": "compiler-manifest",
                "path": TOKENIZER_MANIFEST_PATH,
                "size": TOKENIZER_MANIFEST_SIZE,
                "sha256": TOKENIZER_MANIFEST_SHA256,
            }
        ]
        or benchmark.get("upstream_status") != "broken"
        or benchmark.get("reference_status") != "blocked"
        or benchmark.get("normalized_status") != "protocol-compatible"
        or "CUDA-only flash_attn" not in str(benchmark.get("reason"))
        or not any(
            "CUDA-only flash_attn" in str(item)
            for item in provenance.get("known_defects", [])
        )
    ):
        raise OracleError("Evo-1 catalog source/provenance contract differs")
    normalized = row.get("embedding_presets", {}).get("normalized")
    if normalized != {
        "hidden_tap": "model-final-hidden",
        "pooling": "per-record-mean",
        "special_tokens": "none",
        "mask_domain": "record-token-rows",
        "output_width": HIDDEN_SIZE,
    }:
        raise OracleError("Evo-1 normalized embedding contract differs")
    return {
        "profile": profile,
        "catalog": catalog,
        "catalog_row": row,
        "catalog_contract_sha256": catalog_contract_sha256(catalog, row),
    }


def read_fasta(path: Path, expected: Mapping[str, Any]) -> str:
    payload = checked_file(
        path, expected["name"], expected["size"], expected["sha256"]
    )
    lines = payload.decode("ascii").splitlines()
    if len(lines) != 2 or not lines[0].startswith(">") or not lines[1]:
        raise OracleError(f"{expected['name']} must contain one FASTA record")
    return lines[1]


def validate_inputs(repo_root: Path) -> list[dict[str, Any]]:
    result = []
    for expected in INPUTS:
        sequence = read_fasta(repo_root / expected["path"], expected)
        input_ids = list(sequence.encode("utf-8"))
        if sequence != expected["sequence"] or input_ids != expected["input_ids"]:
            raise OracleError(f"{expected['name']} official byte IDs differ")
        if len(input_ids) > MAX_TOKENS:
            raise OracleError(f"{expected['name']} exceeds the frozen context")
        result.append(
            {
                "name": expected["name"],
                "sequence": sequence,
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
        )
    return result


def validate_tokenizer(repo_root: Path, source_root: Path) -> None:
    manifest = load_json_bytes(
        checked_file(
            repo_root / TOKENIZER_MANIFEST_PATH,
            "Evo-1 tokenizer manifest",
            TOKENIZER_MANIFEST_SIZE,
            TOKENIZER_MANIFEST_SHA256,
        ),
        "Evo-1 tokenizer manifest",
    )
    if manifest != {
        "format": "evo-tokenizer-compiler-v1",
        "source": "custom",
        "kind": "byte-bpe",
        "files": [
            {
                "role": "spec",
                "name": "geneb-evo1-byte-source-v1.json",
                "size": TOKENIZER_SOURCE_SIZE,
                "sha256": TOKENIZER_SOURCE_SHA256,
            }
        ],
        "options": {},
    }:
        raise OracleError("Evo-1 tokenizer compiler manifest differs")
    source = load_json_bytes(
        checked_file(
            repo_root / TOKENIZER_SOURCE_PATH,
            "Evo-1 tokenizer source",
            TOKENIZER_SOURCE_SIZE,
            TOKENIZER_SOURCE_SHA256,
        ),
        "Evo-1 tokenizer source",
    )
    vocabulary = source.get("vocab")
    byte_encoder = source.get("model", {}).get("byte_encoder")
    if (
        source.get("kind") != "byte-bpe"
        or source.get("normalization") != []
        or source.get("pre_tokenizer") != {"kind": "whole-input"}
        or source.get("model", {}).get("add_prefix_space") is not False
        or source.get("model", {}).get("merges") != []
        or byte_encoder != [chr(0x100 + index) for index in range(256)]
        or not isinstance(vocabulary, list)
        or len(vocabulary) != 512
        or vocabulary[:256]
        != [{"id": index, "piece": chr(0x100 + index)} for index in range(256)]
        or vocabulary[256:]
        != [
            {"id": index, "piece": f"<reserved-{index}>"}
            for index in range(256, 512)
        ]
        or source.get("post_processor", {}).get("prefix_ids") != []
        or source.get("post_processor", {}).get("suffix_ids") != []
    ):
        raise OracleError("Evo-1 compiled byte-tokenizer semantics differ")
    tokenizer_source = checked_file(
        source_root / "tokenizer.py",
        "pinned upstream tokenizer.py",
        *SOURCE_FILES["tokenizer.py"],
    )
    required_fragments = (
        b"return 512",
        b"return np.frombuffer(text.encode('utf-8'), dtype=np.uint8)",
        b"first_ids = self.byte_tokenize(text).tolist()",
        b"add_special_tokens=kwargs.get('add_special_tokens', False)",
    )
    if any(fragment not in tokenizer_source for fragment in required_fragments):
        raise OracleError("pinned upstream tokenizer byte path differs")


def validate_small_source(source_root: Path) -> dict[str, bytes]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise OracleError("source root must be a nonsymlink directory")
    result: dict[str, bytes] = {}
    for name, (size, digest) in SOURCE_FILES.items():
        if name in WEIGHT_FILES:
            continue
        result[name] = checked_file(
            source_root / name, f"pinned source {name}", size, digest
        )
    config = load_json_bytes(result["config.json"], "pinned config.json")
    if (
        config.get("use_flash_attn") is not True
        or config.get("attn_layer_idxs") != list(ATTENTION_LAYERS)
        or config.get("use_flash_rmsnorm") is not False
        or config.get("use_flash_depthwise") is not False
        or config.get("use_flashfft") is not False
        or config.get("hidden_size") != HIDDEN_SIZE
        or config.get("num_attention_heads") != NUM_HEADS
        or config.get("rotary_emb_base") != 10000
        or config.get("rotary_emb_scaling_factor") != 16
        or config.get("use_interpolated_rotary_pos_emb") is not True
    ):
        raise OracleError("pinned clean attention/dependency config differs")
    model = result["model.py"]
    position = result["positional_embeddings.py"]
    if (
        model.count(MODEL_IMPORT_OLD) != 1
        or b"self.inner_mha_cls = MHA(" not in model
        or b"use_flash_attn=self.config.use_flash_attn" not in model
        or position.count(POSITION_IMPORT_OLD) != 1
    ):
        raise OracleError("clean upstream failure signature differs")
    return result


def validate_patch_assets(repo_root: Path) -> dict[str, Any]:
    descriptor = load_json_bytes(
        checked_file(
            repo_root / PATCH_PATH,
            "Evo-1 GPU-free patch descriptor",
            PATCH_SIZE,
            PATCH_SHA256,
        ),
        "Evo-1 GPU-free patch descriptor",
    )
    portable = checked_file(
        repo_root / PORTABLE_PATH,
        "Evo-1 portable MHA",
        PORTABLE_SIZE,
        PORTABLE_SHA256,
    )
    if (
        descriptor.get("runtime_id") != RUNTIME_ID
        or descriptor.get("revision") != REVISION
        or descriptor.get("attention_layers") != list(ATTENTION_LAYERS)
        or descriptor.get("clean_upstream_status") != "broken"
        or descriptor.get("reference_status") != "blocked"
        or descriptor.get("normalized_oracle_profile")
        != "pinned-upstream-normalized-gpu-free"
        or descriptor.get("added_file", {}).get("sha256") != sha256_bytes(portable)
        or descriptor.get("semantic_basis")
        != {
            "flash_attn_version": "2.5.5",
            "geneb_environment_path": "embedding_pipeline/utility_modules/EVO_project/evo/semantic_mining/semantic_mining.yml",
            "geneb_environment_size": 3121,
            "geneb_environment_sha256": "b5f3e3bec6ca98bacd32824c43157b365edc8909cf1dc59745f77933b2fd347e",
            "mha_source_path": "flash_attn/modules/mha.py",
            "mha_source_size": 43075,
            "mha_source_sha256": "ff1dd03f9cc058175e670a300c9f90ab873a6b51cd8e2a3a076084e47ec79d80",
            "rotary_source_path": "flash_attn/layers/rotary.py",
            "rotary_source_size": 18874,
            "rotary_source_sha256": "4660edb88a5b158f9da8b0132b068f4e3b95b307062602f6d7f2ef1f09f6bb0f",
        }
    ):
        raise OracleError("Evo-1 GPU-free patch contract differs")
    return descriptor


def prepare_patched_source(
    repo_root: Path,
    source_root: Path,
    work_root: Path,
    small_source: Mapping[str, bytes],
    descriptor: Mapping[str, Any],
) -> Path:
    if work_root.is_symlink():
        raise OracleError("patch work root must not be a symlink")
    work_root.mkdir(parents=True, exist_ok=True)
    patched = work_root / "evo1_normalized"
    if patched.exists():
        if patched.is_symlink() or not patched.is_dir() or any(patched.iterdir()):
            raise OracleError("patched source destination must be absent or empty")
    else:
        patched.mkdir()
    operations = descriptor.get("operations")
    if not isinstance(operations, list) or len(operations) != 2:
        raise OracleError("patch operation set differs")
    operation_by_path = {
        item.get("path"): item for item in operations if isinstance(item, dict)
    }
    replacements = {
        "model.py": (MODEL_IMPORT_OLD, MODEL_IMPORT_NEW),
        "positional_embeddings.py": (POSITION_IMPORT_OLD, POSITION_IMPORT_NEW),
    }
    for name, payload in small_source.items():
        output = patched / name
        if name in replacements:
            old, new = replacements[name]
            if payload.count(old) != 1:
                raise OracleError(f"patch target {name} no longer has one exact hunk")
            operation = operation_by_path.get(name)
            if not isinstance(operation, dict):
                raise OracleError(f"patch descriptor omits {name}")
            if (
                operation.get("old_bytes_size") != len(old)
                or operation.get("old_bytes_sha256") != sha256_bytes(old)
                or operation.get("new_bytes_size") != len(new)
                or operation.get("new_bytes_sha256") != sha256_bytes(new)
            ):
                raise OracleError(f"patch hunk descriptor for {name} differs")
            payload = payload.replace(old, new)
            if (
                len(payload) != operation.get("patched_size")
                or sha256_bytes(payload) != operation.get("patched_sha256")
            ):
                raise OracleError(f"patched {name} size/SHA differs")
        output.write_bytes(payload)
    portable = checked_file(
        repo_root / PORTABLE_PATH,
        "Evo-1 portable MHA",
        PORTABLE_SIZE,
        PORTABLE_SHA256,
    )
    (patched / "evo1_portable_mha.py").write_bytes(portable)
    (patched / "__init__.py").write_bytes(b"")
    dependency_scan(patched)
    return patched


def dependency_scan(patched: Path) -> None:
    protected = (
        patched / "model.py",
        patched / "positional_embeddings.py",
        patched / "evo1_portable_mha.py",
    )
    for path in protected:
        payload = path.read_bytes()
        try:
            tree = ast.parse(payload, filename=path.name)
        except SyntaxError as error:
            raise OracleError(f"patched {path.name} is not valid Python") from error
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.append(node.module)
            if any(
                name == "flash_attn"
                or name.startswith("flash_attn.")
                or name == "evo"
                or name.startswith("evo.")
                for name in names
            ):
                raise OracleError(
                    f"GPU/native dependency import remains in active patch file {path.name}"
                )
    layers = (patched / "layers.py").read_text(encoding="utf-8")
    if layers.count("from flash_attn.ops.rms_norm import rms_norm") != 1:
        raise OracleError("guarded upstream RMSNorm dependency signature differs")
    config = json.loads((patched / "config.json").read_text(encoding="utf-8"))
    if any(
        config.get(name) is not False
        for name in ("use_flash_rmsnorm", "use_flash_depthwise", "use_flashfft")
    ):
        raise OracleError("a non-attention GPU dependency is enabled")


def import_file(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OracleError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operator_self_test(
    patched: Path, host_profile: str
) -> dict[str, Any]:
    module = import_file(patched / "evo1_portable_mha.py", "evo1_operator_gate")

    class ScaledRotary(module.RotaryEmbedding):  # type: ignore[misc, name-defined]
        def _update_cos_sin_cache(
            self,
            seqlen: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            if dtype != torch.float32:
                raise OracleError("operator rotary dtype differs")
            positions = torch.arange(
                seqlen, device=device, dtype=torch.float32
            ) / 16.0
            inv_freq = (
                self._compute_inv_freq(device=device)
                if self.inv_freq.dtype != torch.float32
                else self.inv_freq
            )
            frequencies = torch.outer(positions, inv_freq)
            self._seq_len_cached = seqlen
            self._cos_cached = torch.cos(frequencies).to(dtype)
            self._sin_cached = torch.sin(frequencies).to(dtype)

    attention = module.MHA(
        embed_dim=4,
        num_heads=2,
        num_heads_kv=2,
        qkv_proj_bias=True,
        out_proj_bias=True,
        causal=True,
        layer_idx=8,
        rotary_emb_dim=2,
        rotary_emb_base=10000.0,
        use_flash_attn=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    attention.rotary_emb = ScaledRotary(2, base=10000.0)
    attention.rotary_emb.register_buffer(
        "inv_freq", attention.rotary_emb.inv_freq
    )
    if set(attention.state_dict()) != {
        "Wqkv.weight",
        "Wqkv.bias",
        "out_proj.weight",
        "out_proj.bias",
        "rotary_emb.inv_freq",
    }:
        raise OracleError("portable attention parameter names differ")
    qkv_weight = torch.tensor(
        [
            [0.10, -0.20, 0.30, -0.40],
            [0.05, 0.15, -0.25, 0.35],
            [-0.12, 0.22, 0.32, -0.42],
            [0.18, -0.28, 0.38, 0.08],
            [0.07, -0.17, 0.27, -0.37],
            [-0.09, 0.19, -0.29, 0.39],
            [0.11, 0.21, -0.31, -0.41],
            [-0.13, -0.23, 0.33, 0.43],
            [0.14, -0.24, 0.34, -0.44],
            [-0.16, 0.26, -0.36, 0.46],
            [0.08, 0.18, 0.28, -0.38],
            [-0.06, -0.16, 0.36, 0.26],
        ],
        dtype=torch.float32,
    )
    qkv_bias = torch.tensor(
        [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.10, 0.11, -0.12],
        dtype=torch.float32,
    )
    output_weight = torch.tensor(
        [
            [0.3, -0.2, 0.1, 0.4],
            [-0.1, 0.5, -0.3, 0.2],
            [0.2, 0.1, 0.4, -0.5],
            [-0.4, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    output_bias = torch.tensor([0.01, 0.02, -0.03, 0.04], dtype=torch.float32)
    inputs = torch.tensor(
        [[[0.2, -0.1, 0.4, 0.3], [-0.3, 0.5, 0.1, -0.2], [0.6, -0.4, 0.2, 0.1]]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        attention.Wqkv.weight.copy_(qkv_weight)
        attention.Wqkv.bias.copy_(qkv_bias)
        attention.out_proj.weight.copy_(output_weight)
        attention.out_proj.bias.copy_(output_bias)
        first = attention(inputs)
        second = attention(inputs)
    payload = first.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
    observed_sha256 = sha256_bytes(payload)
    if not torch.equal(first, second):
        raise OracleError("portable attention F32 operator is not repeatable")
    if host_profile == "darwin-arm64":
        expected = torch.tensor(OPERATOR_OUTPUT, dtype=torch.float32).reshape(
            1, 3, 4
        )
        expected_sha256 = OPERATOR_OUTPUT_SHA256
        if (
            not torch.equal(first, expected)
            or observed_sha256 != expected_sha256
        ):
            raise OracleError("Apple-arm64 attention operator vector differs")
    elif host_profile == "linux-x86_64":
        expected_sha256 = LINUX_X86_64_OPERATOR_OUTPUT_SHA256
        if (
            expected_sha256 is not None
            and observed_sha256 != expected_sha256
        ):
            raise OracleError("Linux-x86_64 attention operator vector differs")
    else:
        raise OracleError("unsupported operator host profile")
    vector_status = (
        "frozen"
        if expected_sha256 is not None
        else "candidate-pending-remote-audit-only-freeze"
    )
    return {
        "shape": [1, 3, 4],
        "host_profile": host_profile,
        "sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "platform_vector_status": vector_status,
        "repeated_bit_equal": True,
    }


def validate_receipt(
    receipt_path: Path,
    source_root: Path,
    catalog_path: Path,
    expected_catalog_contract: str,
) -> dict[str, Any]:
    receipt = load_json_bytes(receipt_path.read_bytes(), "source receipt")
    if set(receipt) != {
        "schema_version",
        "kind",
        "model_id",
        "repo",
        "requested_revision",
        "resolved_revision",
        "files",
        "load_path",
        "source_kind",
        "catalog_path",
        "catalog_contract_sha256",
    }:
        raise OracleError("source receipt fields differ")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "source-checkpoint"
        or receipt.get("model_id") != RUNTIME_ID
        or receipt.get("repo") != REPO_ID
        or receipt.get("requested_revision") != REQUESTED_REVISION
        or receipt.get("resolved_revision") != REVISION
        or receipt.get("load_path") is not None
        or receipt.get("source_kind") != "huggingface"
        or Path(receipt.get("catalog_path", "")).resolve() != catalog_path.resolve()
        or receipt.get("catalog_contract_sha256") != expected_catalog_contract
    ):
        raise OracleError("source receipt identity/catalog contract differs")
    raw_files = receipt.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != len(SOURCE_FILES):
        raise OracleError("source receipt must contain the exact 16-file closure")
    seen: dict[str, tuple[int, str]] = {}
    root = source_root.resolve()
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict) or set(item) != {"name", "size", "sha256", "path"}:
            raise OracleError(f"receipt.files[{index}] differs")
        name = item["name"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in seen
            or name not in SOURCE_FILES
        ):
            raise OracleError(f"receipt.files[{index}].name differs")
        path = Path(item["path"])
        expected_path = root / name
        if path.is_symlink() or path.resolve() != expected_path:
            raise OracleError(f"receipt path for {name} leaves the source root")
        payload = checked_file(path, f"receipt source {name}", *SOURCE_FILES[name])
        if item["size"] != len(payload) or item["sha256"] != sha256_bytes(payload):
            raise OracleError(f"receipt descriptor for {name} differs")
        seen[name] = (item["size"], item["sha256"])
    if seen != SOURCE_FILES:
        raise OracleError("source receipt exact file set differs")
    return receipt


class BlockedDependencyFinder(importlib.abc.MetaPathFinder):
    PREFIXES = ("flash_attn", "flashfftconv")

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object = None,
    ) -> None:
        if fullname.startswith(self.PREFIXES):
            raise ImportError(f"GPU dependency {fullname} is forbidden")
        return None


def package_versions(host_profile: str) -> dict[str, str]:
    if host_profile == "darwin-arm64":
        expected = EXPECTED_PACKAGES
    elif host_profile == "linux-x86_64":
        expected = LINUX_X86_64_EXPECTED_PACKAGES
    else:
        raise OracleError("unsupported package host profile")
    result: dict[str, str] = {}
    for name in expected:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise OracleError(f"required package {name} is missing") from error
    if result != expected:
        raise OracleError(f"oracle package versions differ: {result}")
    return result


def round_bf16(value: Any) -> Any:
    """Round float32 to bfloat16 using the native RNE bias (BF16 rounding)."""
    tensor = value.to(device="cpu", dtype=torch.float32).contiguous()
    bits = tensor.view(torch.int32)
    exponent = bits & 0x7F800000
    finite = exponent != 0x7F800000
    bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = bits + torch.where(finite, bias, torch.zeros_like(bits))
    return (rounded & -65536).view(torch.float32)


def evo1_linear(
    values: Any,
    weight: Any,
    bias: Any,
    width: int,
) -> Any:
    """Double-accumulate matmul with BF16 output rounding (native linear_bf16)."""
    del width  # retained for symmetry with the C++ signature
    output = torch.matmul(values.to(torch.float64), weight.to(torch.float64).t())
    if bias is not None:
        output = output + bias.to(torch.float64)
    return round_bf16(output.to(torch.float32))


def evo1_rms(values: Any, scale: Any, epsilon: float, width: int) -> Any:
    """RMSNorm with double reduction, eps outside sqrt, and BF16 output."""
    squares = (values.to(torch.float64) * values.to(torch.float64)).sum(
        dim=-1, keepdim=True
    )
    denominator = torch.sqrt(squares / float(width)) + epsilon
    return round_bf16(
        (values.to(torch.float64) / denominator * scale.to(torch.float64)).to(
            torch.float32
        )
    )


def evo1_add(left: Any, right: Any) -> Any:
    return round_bf16(
        (left.to(torch.float64) + right.to(torch.float64)).to(torch.float32)
    )


def evo1_gelu(value: Any) -> Any:
    doubled = value.to(torch.float64)
    return round_bf16(
        (
            0.5
            * doubled
            * (1.0 + torch.erf(doubled * 0.7071067811865475))
        ).to(torch.float32)
    )


def evo1_mlp(values: Any, l1: Any, l2: Any, l3: Any) -> Any:
    width = values.shape[-1]
    first = evo1_linear(values, l1, None, width)
    second = evo1_linear(values, l2, None, width)
    gated = round_bf16(
        (evo1_gelu(first).to(torch.float64) * second.to(torch.float64)).to(
            torch.float32
        )
    )
    return evo1_linear(gated, l3, None, first.shape[-1])


def evo1_apply_rope(
    query: Any, key: Any, heads: int, theta: float, scale: float
) -> None:
    rows = query.shape[0]
    width = query.shape[1]
    head_width = width // heads
    half = head_width // 2
    inverse = torch.pow(
        torch.tensor(theta, dtype=torch.float64),
        (-2.0 * torch.arange(half, dtype=torch.float64) / float(head_width)),
    )
    positions = (
        torch.arange(rows, dtype=torch.float64) / float(scale)
    )
    angles = torch.outer(positions, inverse)  # (rows, half)
    cosine = round_bf16(torch.cos(angles).to(torch.float32)).unsqueeze(1)
    sine = round_bf16(torch.sin(angles).to(torch.float32)).unsqueeze(1)
    q = query.view(rows, heads, head_width)
    k = key.view(rows, heads, head_width)
    first = q[..., :half]
    second = q[..., half:]
    rot_q_first = round_bf16(
        (first.to(torch.float64) * cosine.to(torch.float64)
         - second.to(torch.float64) * sine.to(torch.float64)).to(torch.float32)
    )
    rot_q_second = round_bf16(
        (second.to(torch.float64) * cosine.to(torch.float64)
         + first.to(torch.float64) * sine.to(torch.float64)).to(torch.float32)
    )
    query.copy_(
        torch.cat((rot_q_first, rot_q_second), dim=-1).reshape(rows, width)
    )
    first = k[..., :half]
    second = k[..., half:]
    rot_k_first = round_bf16(
        (first.to(torch.float64) * cosine.to(torch.float64)
         - second.to(torch.float64) * sine.to(torch.float64)).to(torch.float32)
    )
    rot_k_second = round_bf16(
        (second.to(torch.float64) * cosine.to(torch.float64)
         + first.to(torch.float64) * sine.to(torch.float64)).to(torch.float32)
    )
    key.copy_(
        torch.cat((rot_k_first, rot_k_second), dim=-1).reshape(rows, width)
    )


def evo1_attention(query: Any, key: Any, value: Any, heads: int) -> Any:
    rows = query.shape[0]
    width = query.shape[1]
    head_width = width // heads
    scale = 1.0 / (float(head_width) ** 0.5)
    q = query.view(rows, heads, head_width).to(torch.float64)
    k = key.view(rows, heads, head_width).to(torch.float64)
    v = value.view(rows, heads, head_width).to(torch.float64)
    scores = torch.einsum("thd,shd->hts", q, k) * scale
    causal = torch.tril(
        torch.ones(rows, rows, dtype=torch.bool), diagonal=0
    )
    scores = scores.masked_fill(~causal, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    attended = torch.einsum("hts,shd->thd", probabilities, v)
    return round_bf16(attended.reshape(rows, width).to(torch.float32))


def evo1_hyena_conv(
    projected: Any, short_weight: Any, short_bias: Any,
    poles: Any, residues: Any, diagonal: Any, heads: int, width: int,
) -> Any:
    rows = projected.shape[0]
    channels = projected.shape[-1]
    head_width = width // heads
    # Short filter: CausalConv1d, left-padded so output[t] only sees t and
    # earlier taps.  Double accumulation then BF16 rounding matches native.
    weight = short_weight.to(torch.float64).view(channels, 1, -1)
    short_values = projected.to(torch.float64).reshape(rows, channels).t().unsqueeze(0)
    short_out = torch.nn.functional.conv1d(
        short_values,
        weight,
        bias=short_bias.to(torch.float64),
        padding=short_weight.shape[-1] - 1,
        groups=channels,
    )[0].t()[:rows]  # (rows, channels)
    short_out = round_bf16(short_out.to(torch.float32))
    # Long IIR filter impulse response in double precision.
    kernel = torch.zeros(width, rows, dtype=torch.float64)
    for state in range(poles.shape[1]):
        pole = torch.complex(
            poles[:, state, 0, 0].to(torch.float64),
            poles[:, state, 0, 1].to(torch.float64),
        )
        residue = torch.complex(
            residues[:, state, 0, 0].to(torch.float64),
            residues[:, state, 0, 1].to(torch.float64),
        )
        powers = torch.exp(
            torch.log(pole).unsqueeze(1)
            * torch.arange(rows, dtype=torch.float64).unsqueeze(0)
        )  # (width, rows)
        kernel = kernel + (residue.unsqueeze(1) * powers).real
    kernel = kernel.t().contiguous()  # (rows, width), kernel[lag, c]
    # Gated signal and per-head x2/x1/value projection columns.
    head_index = torch.arange(width) // head_width
    within = torch.arange(width) % head_width
    x2 = head_index * (head_width * 3) + within
    x1 = x2 + head_width
    value_column = x1 + head_width
    signal = round_bf16(short_out[:, x1] * short_out[:, value_column])
    # Causal convolution in double precision, then cast once before BF16.
    conv = torch.zeros(rows, width, dtype=torch.float64)
    for row in range(rows):
        for source in range(row + 1):
            conv[row] += (
                signal[source].to(torch.float64) * kernel[row - source]
            )
    conv = conv.to(torch.float32)
    diagonal_only = round_bf16(signal * diagonal)
    filtered = round_bf16(conv + diagonal_only)
    return round_bf16(filtered * short_out[:, x2])


def evo1_hyena_block(hidden: Any, weights: Mapping[str, Any], prefix: str,
                     heads: int, epsilon: float) -> Any:
    width = hidden.shape[-1]
    normalized = evo1_rms(hidden, weights[prefix + "pre_norm.scale"], epsilon, width)
    projected = evo1_linear(
        normalized,
        weights[prefix + "projections.weight"],
        weights[prefix + "projections.bias"],
        width,
    )
    mixed = evo1_hyena_conv(
        projected,
        weights[prefix + "filter.short_filter_weight"],
        weights[prefix + "filter.short_filter_bias"],
        weights[prefix + "filter.poles"],
        weights[prefix + "filter.residues"],
        weights[prefix + "filter.D"],
        heads,
        width,
    )
    out = evo1_linear(
        mixed,
        weights[prefix + "out_filter_dense.weight"],
        weights[prefix + "out_filter_dense.bias"],
        width,
    )
    hidden = evo1_add(hidden, out)
    normalized = evo1_rms(hidden, weights[prefix + "post_norm.scale"], epsilon, width)
    return evo1_add(hidden, evo1_mlp(
        normalized,
        weights[prefix + "mlp.l1.weight"],
        weights[prefix + "mlp.l2.weight"],
        weights[prefix + "mlp.l3.weight"],
    ))


def evo1_attention_block(hidden: Any, weights: Mapping[str, Any], prefix: str,
                         heads: int, theta: float, scale: float,
                         epsilon: float) -> Any:
    width = hidden.shape[-1]
    normalized = evo1_rms(hidden, weights[prefix + "pre_norm.scale"], epsilon, width)
    qkv = evo1_linear(
        normalized,
        weights[prefix + "inner_mha_cls.Wqkv.weight"],
        weights[prefix + "inner_mha_cls.Wqkv.bias"],
        width,
    )
    query = qkv[:, :width].clone()
    key = qkv[:, width : width * 2].clone()
    value = qkv[:, width * 2 :].clone()
    evo1_apply_rope(query, key, heads, theta, scale)
    attended = evo1_attention(query, key, value, heads)
    out = evo1_linear(
        attended,
        weights[prefix + "inner_mha_cls.out_proj.weight"],
        weights[prefix + "inner_mha_cls.out_proj.bias"],
        width,
    )
    hidden = evo1_add(hidden, out)
    normalized = evo1_rms(hidden, weights[prefix + "post_norm.scale"], epsilon, width)
    return evo1_add(hidden, evo1_mlp(
        normalized,
        weights[prefix + "mlp.l1.weight"],
        weights[prefix + "mlp.l2.weight"],
        weights[prefix + "mlp.l3.weight"],
    ))


def evo1_forward(
    weights: Mapping[str, Any], input_ids: Any, heads: int, layers: int,
    attention: Any, theta: float, rope_scale: float, epsilon: float,
) -> Any:
    embedding = weights["backbone.embedding_layer.weight"]
    hidden = embedding[input_ids.reshape(-1)].clone()  # (rows, width)
    for layer in range(layers):
        prefix = f"backbone.blocks.{layer}."
        if layer in attention:
            hidden = evo1_attention_block(
                hidden, weights, prefix, heads, theta, rope_scale, epsilon
            )
        else:
            hidden = evo1_hyena_block(
                hidden, weights, prefix, heads, epsilon
            )
    width = hidden.shape[-1]
    return evo1_rms(hidden, weights["backbone.norm.scale"], epsilon, width)


def run_model(
    patched: Path,
    source_root: Path,
    inputs: Sequence[Mapping[str, Any]],
    output_root: Path,
    packages: Mapping[str, str],
) -> dict[str, Any]:
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise OracleError("torch interop threads differ")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from accelerate import init_empty_weights, load_checkpoint_in_model

    parent = str(patched.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    finder = BlockedDependencyFinder()
    sys.meta_path.insert(0, finder)
    try:
        configuration = importlib.import_module(
            "evo1_normalized.configuration_hyena"
        )
        modeling = importlib.import_module("evo1_normalized.modeling_hyena")
        config_value = json.loads((patched / "config.json").read_text(encoding="utf-8"))
        config = configuration.StripedHyenaConfig(**config_value)
        config.use_cache = False
        with init_empty_weights():
            model = modeling.StripedHyenaModelForCausalLM(config)
        load_checkpoint_in_model(
            model,
            checkpoint=str(source_root),
            device_map={"": "cpu"},
            dtype=torch.float32,
        )
    finally:
        sys.meta_path.remove(finder)
    if any(name.startswith(("flash_attn", "flashfftconv")) for name in sys.modules):
        raise OracleError("a forbidden GPU dependency was imported")
    model.to(device="cpu", dtype=torch.float32).eval()
    parameters = list(model.parameters())
    if (
        not parameters
        or any(parameter.device.type != "cpu" for parameter in parameters)
        or any(parameter.dtype != torch.float32 for parameter in parameters)
        or any(parameter.is_meta for parameter in parameters)
    ):
        raise OracleError("loaded model is not fully materialized CPU F32")
    parameter_count = sum(parameter.numel() for parameter in parameters)

    output_root.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []

    state = model.state_dict()
    weights = {
        name: value.to(device="cpu", dtype=torch.float32).contiguous()
        for name, value in state.items()
        if value.numel() > 0
    }

    def forward_once(input_ids: Any) -> Any:
        return evo1_forward(
            weights,
            input_ids,
            NUM_HEADS,
            NUM_LAYERS,
            set(ATTENTION_LAYERS),
            10000.0,
            16.0,
            0.000001,
        )

    for item in inputs:
        input_ids = torch.tensor(
            [item["input_ids"]], device="cpu", dtype=torch.long
        )
        repeated: list[torch.Tensor] = []
        for _ in range(2):
            with torch.inference_mode():
                hidden = forward_once(input_ids)
            if hidden.shape != (len(item["input_ids"]), HIDDEN_SIZE):
                raise OracleError("captured final hidden shape differs")
            pooled = hidden.to(torch.float64).sum(dim=0).to(torch.float32) / float(hidden.shape[0])
            repeated.append(
                pooled.to(device="cpu", dtype=torch.float32).contiguous()
            )
        if not torch.equal(repeated[0], repeated[1]):
            raise OracleError(f"{item['name']} repeated upstream output differs")
        array = repeated[0].numpy().astype("<f4", copy=False)
        output_path = output_root / f"{item['name']}.npy"
        np.save(output_path, array, allow_pickle=False)
        outputs.append(
            {
                "name": item["name"],
                "path": output_path.name,
                "size": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "shape": list(array.shape),
                "raw_f32_sha256": sha256_bytes(array.tobytes(order="C")),
                "values": [float(value) for value in array.reshape(-1)],
            }
        )
    return {
        "packages": dict(packages),
        "outputs": outputs,
        "parameter_count": parameter_count,
        "model_class": (
            f"{model.__class__.__module__}.{model.__class__.__name__}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    catalog = (
        args.catalog.resolve()
        if args.catalog is not None
        else repo_root / "configs" / "geneb-models.json"
    )
    contracts = validate_profile_and_catalog(repo_root, catalog)
    platform_provenance = validate_oracle_host(full=not args.audit_only)
    packages = package_versions(platform_provenance["host_profile"])
    inputs = validate_inputs(repo_root)
    small_source = validate_small_source(args.source_root.resolve())
    validate_tokenizer(repo_root, args.source_root.resolve())
    descriptor = validate_patch_assets(repo_root)
    patched = prepare_patched_source(
        repo_root,
        args.source_root.resolve(),
        args.work_root.resolve(),
        small_source,
        descriptor,
    )
    operator = operator_self_test(
        patched, platform_provenance["host_profile"]
    )
    if args.audit_only:
        if args.receipt is not None or args.output_root is not None:
            raise OracleError("audit-only must not receive receipt/output paths")
        print(
            canonical_json(
                {
                    "runtime_id": RUNTIME_ID,
                    "status": "preflight-pass-no-checkpoint-read",
                    "oracle_profile": "pinned-upstream-normalized-gpu-free",
                    "clean_upstream_status": "broken",
                    "reference_status": "blocked",
                    "source_file_count": len(SOURCE_FILES),
                    "small_source_file_count": len(small_source),
                    "catalog_contract_sha256": contracts[
                        "catalog_contract_sha256"
                    ],
                    "attention_layers": list(ATTENTION_LAYERS),
                    "environment": {
                        "platform": platform_provenance,
                        "packages": packages,
                        "package_vector_sha256": sha256_bytes(
                            canonical_json(packages)
                        ),
                    },
                    "inputs": inputs,
                    "tolerances": TOLERANCES,
                    "operator": operator,
                }
            ).decode("ascii"),
            end="",
        )
        return 0
    if operator["platform_vector_status"] != "frozen":
        raise OracleError(
            "Linux-x86_64 operator vector must be frozen from a remote "
            "--audit-only result before real oracle output"
        )
    if args.receipt is None or args.output_root is None:
        raise OracleError("real oracle requires --receipt and --output-root")
    receipt = validate_receipt(
        args.receipt.resolve(),
        args.source_root.resolve(),
        catalog,
        contracts["catalog_contract_sha256"],
    )
    result = run_model(
        patched,
        args.source_root.resolve(),
        inputs,
        args.output_root.resolve(),
        packages,
    )
    portable_receipt = {
        "schema_version": 1,
        "kind": "source-checkpoint",
        "model_id": RUNTIME_ID,
        "repo": REPO_ID,
        "requested_revision": REQUESTED_REVISION,
        "resolved_revision": REVISION,
        "source_kind": "huggingface",
        "catalog_contract_sha256": contracts["catalog_contract_sha256"],
        "files": sorted(
            (
                {
                    "name": item["name"],
                    "size": item["size"],
                    "sha256": item["sha256"],
                }
                for item in receipt["files"]
            ),
            key=lambda item: item["name"],
        ),
    }
    receipt_contract_sha256 = sha256_bytes(canonical_json(portable_receipt))
    source_manifest = [
        {"name": name, "size": size, "sha256": digest}
        for name, (size, digest) in sorted(SOURCE_FILES.items())
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
        "packages": package_lock(result["packages"]),
        "source_files": {
            name: {"size": size, "sha256": digest}
            for name, (size, digest) in sorted(SOURCE_FILES.items())
        },
        "thread_environment": {
            "TOKENIZERS_PARALLELISM": "false",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
        },
        "build_vector": dict(platform_provenance["build_vector"]),
        "operator": dict(operator),
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "num_threads": 1,
            "num_interop_threads": 1,
            "cuda_version": None,
            "hip_version": None,
        },
        "transformers": {
            "version": result["packages"]["transformers"],
            "model_class": result["model_class"],
            "loader": "accelerate-load_checkpoint_in_model-device_map-cpu",
        },
        "patch": {
            "descriptor_path": PATCH_PATH,
            "descriptor_sha256": PATCH_SHA256,
            "portable_mha_path": PORTABLE_PATH,
            "portable_mha_sha256": PORTABLE_SHA256,
            "attention_layers": list(ATTENTION_LAYERS),
        },
    }
    base_provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-normalized-gpu-free",
        "benchmark_semantics": "geneb-v4-normalized",
        "clean_upstream_status": "broken",
        "reference_status": "blocked",
        "catalog_contract_sha256": contracts["catalog_contract_sha256"],
        "profile_contract_sha256": PROFILE_SHA256,
        "source_repo": REPO_ID,
        "source_requested_revision": REQUESTED_REVISION,
        "source_revision": REVISION,
        "source_receipt_contract_sha256": receipt_contract_sha256,
        "source_file_manifest_sha256": sha256_bytes(
            canonical_json(source_manifest)
        ),
        "source_weight_dtype": "bfloat16",
        "patch_sha256": PATCH_SHA256,
        "portable_mha_sha256": PORTABLE_SHA256,
        "attention_layers": list(ATTENTION_LAYERS),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "hidden_tap": "post-final-rmsnorm",
        "pooling": "per-record-f32-sum-then-direct-division",
        "special_tokens": "none",
        "tokenizer": "official ByteTokenizer UTF-8 byte_tokenize without special tokens",
        "output_width": HIDDEN_SIZE,
        "parameter_count": result["parameter_count"],
        "model_class": result["model_class"],
        "loader": "accelerate-load_checkpoint_in_model-device_map-cpu",
        "execution_dtype": "float32",
        "repeated_forward_bit_equal": True,
        "tolerances": dict(TOLERANCES),
        "independent_of_evo_native_runtime": True,
    }
    vectors = []
    report_inputs = []
    for expected, record, output in zip(INPUTS, inputs, result["outputs"], strict=True):
        if output["name"] != expected["name"] or record["name"] != expected["name"]:
            raise OracleError("oracle output ordering differs")
        flat = output["values"]
        provenance = dict(base_provenance)
        provenance.update(
            {
                "input_label": expected["name"],
                "expected_input_ids": list(expected["input_ids"]),
                "token_count": len(expected["input_ids"]),
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
        vector_path = args.output_root.resolve() / (
            expected["name"] + ".independent-oracle-vector.json"
        )
        vector_path.write_bytes(canonical_json(vector))
        vectors.append(
            {
                "label": expected["name"],
                "vector": {
                    "path": vector_path.name,
                    "size": vector_path.stat().st_size,
                    "sha256": sha256_file(vector_path),
                },
                "npy": {
                    "path": output["path"],
                    "size": output["size"],
                    "sha256": output["sha256"],
                },
            }
        )
        report_inputs.append(
            {
                "label": expected["name"],
                "sha256": expected["sha256"],
                "sequence": expected["sequence"],
                "input_ids": list(expected["input_ids"]),
                "attention_mask": list(record["attention_mask"]),
                "output": {
                    "shape": output["shape"],
                    "npy_path": output["path"],
                    "npy_sha256": output["sha256"],
                    "raw_f32_sha256": output["raw_f32_sha256"],
                    "vector_sha256": sha256_file(vector_path),
                    "first_16": flat[:16],
                },
            }
        )
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "oracle_profile": "pinned-upstream-normalized-gpu-free",
        "clean_upstream_status": "broken",
        "reference_status": "blocked",
        "source": {
            "kind": "huggingface",
            "repo": REPO_ID,
            "requested_revision": REQUESTED_REVISION,
            "revision": REVISION,
            "files": source_manifest,
            "receipt_contract_sha256": receipt_contract_sha256,
        },
        "inputs": report_inputs,
        "environment_lock": environment_lock,
        "provenance": base_provenance,
        "outputs": vectors,
    }
    require_portable(report, "oracle report")
    report_path = args.output_root.resolve() / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))
    del contracts, result
    gc.collect()
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
