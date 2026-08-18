#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned GENEB DNA-GPT-3B-M CPU-F32 embedding oracle.

This program imports only the pinned GENEB-vendored DNAGPT implementation.
It does not import evo.cpp, converted artifacts, or native runtime math.  The
``--audit-only`` path validates only small source/input/tokenizer contracts and
does not open the manually acquired 6.23 GB checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


RUNTIME_ID = "geneb-dna-gpt-3b-m"
MODEL_NAME = "dna_gpt3b_m"
PAPER_NAME = "DNA-GPT-3B-M"
GENEB_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
UPSTREAM_REPO = "TencentAILabHealthcare/DNAGPT"
UPSTREAM_REVISION = "9b6c0931e3b2011ee5bbb4b988be3e19d62953ae"
SOURCE_URL = "https://drive.google.com/drive/folders/10UPPx6V13oQW6knuLV7d8SRIA3D6hYor"
GOOGLE_DRIVE_FILE_ID = "1pQ3Ai7C-ObzKkKTRwuf6eshVneKHzYEg"
CHECKPOINT_NAME = "dna_gpt3b_m.pth"
CHECKPOINT_SIZE = 6_227_969_411
CHECKPOINT_PAYLOAD_OVERHEAD = 124_291
SOURCE_TENSOR_COUNT = 372
SOURCE_TENSOR_BYTES = 6_227_845_120
SOURCE_MANIFEST_SHA256 = (
    "0394af26ac57ba360bb5009a3892e2bc9a1b87d34cfc38371c5386804d65309f"
)
TOKENIZER_MANIFEST_PATH = "configs/tokenizers/geneb-dna-gpt-3b-m-kmer-v1.json"
TOKENIZER_MANIFEST_SIZE = 244
TOKENIZER_MANIFEST_SHA256 = (
    "a68ef6e9b9343388fb762b7cb71ef38ba5f5407fb92c4217fd7da0bffc9c3649"
)
TOKENIZER_ASSET_SIZE = 571_216
TOKENIZER_ASSET_SHA256 = (
    "8dc2953cdf6bcc676093a5812f26bceacdf2cf92256128356d0bb6ac86f0683f"
)

VOCAB_SIZE = 19_564
HIDDEN_SIZE = 2_048
NUM_LAYERS = 60
NUM_HEADS = 64
INNER_WIDTH = 8_192
MAX_TOKENS = 512
NORM_EPSILON = 0.00001
PAD_ID = 20
PREFIX_ID = 21

SOURCE_FILES = {
    "dna_gpt/model/gpt.py": (
        11_214,
        "e07ae8f5b7bed3ef38009a68d3699a215e4bc6c0e860a77fa873779a2123a6f0",
    ),
    "dna_gpt/model/dna_gpt.py": (
        5_360,
        "be7e4b7bfd249a1c56f1d2b282d756f62482e69daa81054d57d72afa414caa8f",
    ),
    "dna_gpt/tokenizer.py": (
        3_081,
        "799bba1f99977fe9b998d5e0a765ded7b973d7929481497dc741291d575dec16",
    ),
    "dna_gpt/utils.py": (
        1_398,
        "aada73ee3d8f4900e077d4da4e4be0df9c34fcee0832539b1146a71c81341605",
    ),
}
EXTRACTOR_SIZE = 3_250
EXTRACTOR_SHA256 = (
    "3c9df4316dc049d447516527b378e6a4d944a63b83f4d656099d40e648656212"
)

INPUTS = (
    {
        "name": "input-0",
        "sha256": "ed7e6526a13e80f320a984ecf4614a8e77b5405906adbe0207a4b359a024b910",
        "sequence": "ACGTNACGTNACGTA",
        "input_ids": [21, 9290, 15072, 135],
    },
    {
        "name": "input-1",
        "sha256": "14aece18ae2895ed4bb0f1adc796c95f0e6e30293c72ebabcdd309522ed94f14",
        "sequence": "NNNNNNACGTACNNNTATTGCA",
        "input_ids": [21, 3939, 9297, 4048, 755],
    },
)

EXPECTED_PACKAGES = (
    "certifi==2026.7.22",
    "charset-normalizer==3.5.1",
    "einops==0.7.0",
    "filelock==3.19.1",
    "fsspec==2025.10.0",
    "hf-xet==1.6.0",
    "huggingface-hub==0.36.2",
    "idna==3.18",
    "importlib-metadata==8.7.1",
    "jinja2==3.1.6",
    "markupsafe==3.0.3",
    "mpmath==1.3.0",
    "networkx==3.2.1",
    "numpy==1.26.4",
    "packaging==26.3",
    "pip==21.2.4",
    "polars==0.20.31",
    "pyfaidx==0.8.1.1",
    "pyyaml==6.0.3",
    "regex==2026.1.15",
    "requests==2.32.5",
    "safetensors==0.4.2",
    "setuptools==58.0.4",
    "sympy==1.14.0",
    "tokenizers==0.13.3",
    "torch==2.1.2",
    "tqdm==4.70.0",
    "transformers==4.29.2",
    "typing-extensions==4.16.0",
    "urllib3==2.6.3",
    "zipp==3.23.1",
)


class OracleError(RuntimeError):
    """Raised when a pinned source, input, environment, or execution differs."""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(
    path: Path, label: str, expected_size: int, expected_sha256: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{label} must be a nonsymlink regular file")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != expected_size or digest != expected_sha256:
        raise OracleError(f"{label} differs: size={size} sha256={digest}")
    return {"size": size, "sha256": digest}


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{label} must be a nonsymlink regular file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OracleError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} root must be an object")
    return value, payload


def git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise OracleError("cannot resolve pinned GENEB revision")
    return result.stdout.strip()


def package_versions() -> list[str]:
    return sorted(
        {
            "%s==%s"
            % (
                distribution.metadata["Name"].lower().replace("_", "-"),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )


def expected_tensor_specs() -> list[tuple[str, tuple[int, ...]]]:
    specs = [
        ("transformer.wte.weight", (VOCAB_SIZE, HIDDEN_SIZE)),
        ("transformer.wpe.weight", (MAX_TOKENS, HIDDEN_SIZE)),
    ]
    for layer in range(NUM_LAYERS):
        prefix = f"transformer.h.{layer}."
        specs.extend(
            [
                (prefix + "attn.c_attn.weight", (HIDDEN_SIZE * 3, HIDDEN_SIZE)),
                (prefix + "attn.c_proj.weight", (HIDDEN_SIZE, HIDDEN_SIZE)),
                (prefix + "mlp.c_fc.weight", (INNER_WIDTH, HIDDEN_SIZE)),
                (prefix + "mlp.c_proj.weight", (HIDDEN_SIZE, INNER_WIDTH)),
                (prefix + "ln_1.weight", (HIDDEN_SIZE,)),
                (prefix + "ln_2.weight", (HIDDEN_SIZE,)),
            ]
        )
    specs.extend(
        [
            ("transformer.ln_f.weight", (HIDDEN_SIZE,)),
            ("number_embedding.0.weight", (HIDDEN_SIZE, 1)),
            ("number_embedding.2.weight", (HIDDEN_SIZE,)),
            ("number_embedding.3.weight", (HIDDEN_SIZE, HIDDEN_SIZE)),
            ("num_regression.0.weight", (HIDDEN_SIZE, HIDDEN_SIZE)),
            ("num_regression.2.weight", (HIDDEN_SIZE,)),
            ("num_regression.3.weight", (1, HIDDEN_SIZE)),
            ("mlm_head.0.weight", (HIDDEN_SIZE, HIDDEN_SIZE)),
            ("mlm_head.2.weight", (HIDDEN_SIZE,)),
            ("mlm_head.3.weight", (VOCAB_SIZE, HIDDEN_SIZE)),
        ]
    )
    return specs


def validate_expected_manifest() -> dict[str, Any]:
    specs = expected_tensor_specs()
    manifest = sorted(
        [
            {"name": name, "dtype": "BF16", "shape": list(shape)}
            for name, shape in specs
        ],
        key=lambda item: item["name"],
    )
    payload = canonical_json(manifest)
    tensor_bytes = sum(
        2 * int(np.prod(shape, dtype=np.int64)) for _, shape in specs
    )
    digest = sha256_bytes(payload)
    if (
        len(specs) != SOURCE_TENSOR_COUNT
        or tensor_bytes != SOURCE_TENSOR_BYTES
        or digest != SOURCE_MANIFEST_SHA256
        or CHECKPOINT_SIZE - tensor_bytes != CHECKPOINT_PAYLOAD_OVERHEAD
    ):
        raise OracleError("pinned DNA-GPT-3B-M source manifest is internally inconsistent")
    return {
        "tensor_count": len(specs),
        "logical_tensor_bytes": tensor_bytes,
        "manifest_sha256": digest,
        "checkpoint_container_overhead": CHECKPOINT_PAYLOAD_OVERHEAD,
    }


def validate_small_sources(
    source_root: Path, extractor: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    geneb_root = extractor.parents[2]
    expected_source_root = (
        geneb_root
        / "embedding_pipeline"
        / "utility_modules"
        / "DNAGPT_project"
    ).resolve()
    if source_root != expected_source_root:
        raise OracleError("DNAGPT source root is not the pinned GENEB vendored tree")
    if git_revision(geneb_root) != GENEB_REVISION:
        raise OracleError("GENEB revision differs")
    source_lock = {
        name: checked_file(source_root / name, name, expected_size, expected_sha256)
        for name, (expected_size, expected_sha256) in SOURCE_FILES.items()
    }
    extractor_lock = checked_file(
        extractor, "GENEB DNAGPT extractor", EXTRACTOR_SIZE, EXTRACTOR_SHA256
    )
    return source_lock, extractor_lock


def read_frozen_input(path: Path, spec: dict[str, Any]) -> str:
    checked_file(path, spec["name"], path.stat().st_size, spec["sha256"])
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) != 2 or not lines[0].startswith(">") or lines[1] != spec["sequence"]:
        raise OracleError(f"{spec['name']} FASTA bytes/sequence differ")
    return lines[1]


def validate_tokenizer(
    source_root: Path, input_paths: tuple[Path, Path]
) -> tuple[Any, list[dict[str, Any]]]:
    sys.path.insert(0, str(source_root))
    from dna_gpt.tokenizer import (  # pylint: disable=import-error,import-outside-toplevel
        KmerTokenizer,
    )

    special_tokens = (
        [str(value) for value in range(10)]
        + ["+", "-", "*", "/", "=", "&", "|", "!"]
        + ["M", "B", "P", "R", "I", "K", "L", "O", "Q", "S", "U", "V", "W", "Y", "X", "Z"]
    )
    tokenizer = KmerTokenizer(6, special_tokens, True)
    if (
        len(tokenizer) != VOCAB_SIZE
        or tokenizer.pad_id != PAD_ID
        or tokenizer.piece_to_id("<R>") != PREFIX_ID
    ):
        raise OracleError("pinned dynamic KmerTokenizer topology differs")
    inputs = []
    for path, spec in zip(input_paths, INPUTS):
        sequence = read_frozen_input(path, spec)
        token_ids = tokenizer.encode(
            "<R>" + sequence,
            max_len=MAX_TOKENS,
            device=torch.device("cpu"),
        )
        observed = [int(value) for value in token_ids.tolist()]
        if observed != spec["input_ids"]:
            raise OracleError(
                f"{spec['name']} official dynamic-sixmer IDs differ: {observed}"
            )
        inputs.append(
            {
                "name": spec["name"],
                "sha256": spec["sha256"],
                "sequence": sequence,
                "input_ids": observed,
                "attention_mask": [1] * len(observed),
            }
        )
    return tokenizer, inputs


def catalog_contract_sha256(catalog: dict[str, Any], entry: dict[str, Any]) -> str:
    model = json.loads(json.dumps(entry))
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


def validate_catalog(path: Path) -> tuple[dict[str, Any], str, str]:
    catalog, _ = load_json(path, "GENEB catalog")
    suite = catalog.get("suite")
    paper = suite.get("paper") if isinstance(suite, dict) else None
    if (
        catalog.get("schema_version") != 1
        or not isinstance(suite, dict)
        or suite.get("id") != "geneb-v4"
        or suite.get("extractor_commit") != GENEB_REVISION
        or not isinstance(paper, dict)
        or paper.get("arxiv_id") != "2606.04525"
        or paper.get("version") != "v4"
        or paper.get("table") != "Table 4"
        or paper.get("evaluated_models") != 40
    ):
        raise OracleError("GENEB catalog identity differs")
    matches = [
        item
        for item in catalog.get("models", [])
        if isinstance(item, dict) and item.get("runtime_id") == RUNTIME_ID
    ]
    if len(matches) != 1:
        raise OracleError("GENEB catalog must contain exactly one DNA-GPT-3B-M row")
    entry = matches[0]
    identity = {
        "runtime_id": RUNTIME_ID,
        "geneb_model_id": MODEL_NAME,
        "paper_name": PAPER_NAME,
        "family": "transformer-decoder",
        "architecture": "dna-gpt-custom-decoder",
    }
    if any(entry.get(key) != value for key, value in identity.items()):
        raise OracleError("GENEB catalog DNA-GPT-3B-M identity differs")
    tokenizer = entry.get("tokenizer")
    context = entry.get("context")
    transform = entry.get("input_transform")
    presets = entry.get("embedding_presets")
    expected_asset = {
        "role": "compiler-manifest",
        "path": TOKENIZER_MANIFEST_PATH,
        "size": TOKENIZER_MANIFEST_SIZE,
        "sha256": TOKENIZER_MANIFEST_SHA256,
    }
    if not isinstance(tokenizer, dict) or tokenizer.get("assets") != [expected_asset]:
        raise OracleError(
            "DNA-GPT-3B-M catalog source/tokenizer SHA pins are not frozen"
        )
    if (
        tokenizer.get("kind") != "k-mer"
        or tokenizer.get("asset_source") != "embedded-code-contract"
        or tokenizer.get("add_special_tokens") is not False
        or tokenizer.get("padding_side") != "right"
        or tokenizer.get("pad_to") != "batch-max"
        or tokenizer.get("max_tokens") != MAX_TOKENS
        or tokenizer.get("k") != 6
        or tokenizer.get("dynamic") is not True
        or tokenizer.get("unknown_fields") != []
        or not isinstance(context, dict)
        or context.get("declared_max_tokens") != MAX_TOKENS
        or context.get("reference_max_tokens") != MAX_TOKENS
        or context.get("unit") != "tokens"
        or context.get("length_policy") != "tokenizer-truncate"
        or context.get("unknown_fields") != []
        or not isinstance(transform, dict)
        or transform.get("case") != "preserve"
        or transform.get("strip_ascii_whitespace") is not False
        or transform.get("u_to_t") is not False
        or transform.get("invalid") != "tokenizer-defined"
        or transform.get("prefix") != "<R>"
        or transform.get("special_tokens") != "prefix-only"
        or transform.get("token_truncation") != "right"
        or not isinstance(presets, dict)
    ):
        raise OracleError("GENEB catalog DNA-GPT-3B-M tokenizer/context differs")
    expected_preset = {
        "hidden_tap": "post-final-layernorm",
        "pooling": "attention-mask-mean",
        "special_tokens": "include-prefix",
        "mask_domain": "non-pad-token-rows",
        "output_width": HIDDEN_SIZE,
    }
    if any(presets.get(name) != expected_preset for name in ("reference", "normalized")):
        raise OracleError("GENEB catalog DNA-GPT-3B-M embedding preset differs")
    source = entry.get("source")
    if not isinstance(source, dict):
        raise OracleError("GENEB catalog DNA-GPT-3B-M source is missing")
    required = source.get("required_files")
    if (
        source.get("kind") != "google-drive"
        or source.get("repo") is not None
        or source.get("requested_revision") is not None
        or source.get("revision") is not None
        or source.get("immutable") is not False
        or source.get("url") != SOURCE_URL
        or not isinstance(required, list)
        or len(required) != 1
        or not isinstance(required[0], dict)
        or set(required[0]) != {"path", "size", "sha256"}
        or required[0].get("path") != CHECKPOINT_NAME
        or required[0].get("size") != CHECKPOINT_SIZE
        or not isinstance(required[0].get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", required[0]["sha256"]) is None
    ):
        raise OracleError(
            "DNA-GPT-3B-M catalog source/tokenizer SHA pins are not frozen"
        )
    return entry, required[0]["sha256"], catalog_contract_sha256(catalog, entry)


def validate_receipt(
    path: Path, catalog_sha256: str
) -> tuple[Path, dict[str, Any], str]:
    receipt, _ = load_json(path, "manual source receipt")
    if set(receipt) != {
        "schema_version",
        "kind",
        "model_id",
        "source_kind",
        "source_url",
        "files",
    }:
        raise OracleError("manual source receipt fields differ")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != RUNTIME_ID
        or receipt["source_kind"] != "google-drive"
        or receipt["source_url"] != SOURCE_URL
        or not isinstance(receipt["files"], list)
        or len(receipt["files"]) != 1
    ):
        raise OracleError("manual source receipt identity differs")
    item = receipt["files"][0]
    if not isinstance(item, dict) or set(item) != {"name", "size", "sha256", "path"}:
        raise OracleError("manual source receipt file entry differs")
    if (
        item.get("name") != CHECKPOINT_NAME
        or item.get("size") != CHECKPOINT_SIZE
        or item.get("sha256") != catalog_sha256
        or not isinstance(item.get("path"), str)
        or not item["path"]
    ):
        raise OracleError("manual source receipt differs from frozen catalog pin")
    checkpoint = Path(item["path"]).resolve()
    checkpoint_lock = checked_file(
        checkpoint, CHECKPOINT_NAME, CHECKPOINT_SIZE, catalog_sha256
    )
    portable = {
        "schema_version": receipt["schema_version"],
        "kind": receipt["kind"],
        "model_id": receipt["model_id"],
        "source_kind": receipt["source_kind"],
        "source_url": receipt["source_url"],
        "files": [
            {
                "name": item["name"],
                "size": item["size"],
                "sha256": item["sha256"],
            }
        ],
    }
    return checkpoint, checkpoint_lock, sha256_bytes(canonical_json(portable))


def validate_environment() -> list[str]:
    packages = package_versions()
    if (
        platform.python_implementation() != "CPython"
        or platform.python_version() != "3.9.6"
        or platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or torch.__version__ != "2.1.2"
        or np.__version__ != "1.26.4"
        or packages != list(EXPECTED_PACKAGES)
    ):
        raise OracleError(
            "oracle environment differs from the promoted DNA-GPT-0.1B-H lock"
        )
    return packages


def validate_checkpoint_layout(checkpoint: Path) -> dict[str, Any]:
    try:
        state = torch.load(
            str(checkpoint), map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError as error:
        raise OracleError("pinned torch.load weights_only+mmap API is unavailable") from error
    if not isinstance(state, dict) or "model" in state:
        raise OracleError("checkpoint root must be the exact raw state dictionary")
    expected = dict(expected_tensor_specs())
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise OracleError(
            f"checkpoint tensor set differs: missing={missing} extra={extra}"
        )
    tensor_bytes = 0
    for name in sorted(expected):
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.bfloat16
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != expected[name]
        ):
            raise OracleError(f"checkpoint tensor layout differs: {name}")
        tensor_bytes += tensor.numel() * tensor.element_size()
    if len(state) != SOURCE_TENSOR_COUNT or tensor_bytes != SOURCE_TENSOR_BYTES:
        raise OracleError("checkpoint tensor count/bytes differ")
    return state


def execute_oracle(
    source_root: Path,
    tokenizer: Any,
    inputs: list[dict[str, Any]],
    checkpoint: Path,
    checkpoint_lock: dict[str, Any],
    receipt_contract_sha256: str,
    catalog_contract: str,
    source_lock: dict[str, Any],
    extractor_lock: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    packages = validate_environment()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    state_dict = validate_checkpoint_layout(checkpoint)
    from dna_gpt.model import DNAGPT  # pylint: disable=import-error,import-outside-toplevel

    model = DNAGPT.from_name(MODEL_NAME, vocab_size=len(tokenizer))
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise OracleError(
            "checkpoint load differs: missing=%r unexpected=%r"
            % (load_result.missing_keys, load_result.unexpected_keys)
        )
    del state_dict
    gc.collect()
    model.to(device="cpu", dtype=torch.float32).eval()
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if (
        parameter_count != SOURCE_TENSOR_BYTES // 2
        or any(parameter.device.type != "cpu" for parameter in parameters)
        or any(parameter.dtype != torch.float32 for parameter in parameters)
    ):
        raise OracleError("loaded model parameter count/device/dtype differs")

    generator = Path(__file__).resolve(strict=True)
    environment = {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "model_class": "%s.%s" % (
            model.__class__.__module__,
            model.__class__.__name__,
        ),
        "tokenizer_class": "%s.%s" % (
            tokenizer.__class__.__module__,
            tokenizer.__class__.__name__,
        ),
        "source_files": source_lock,
        "extractor": extractor_lock,
    }
    provenance = {
        "source_repo": UPSTREAM_REPO,
        "source_revision": UPSTREAM_REVISION,
        "geneb_revision": GENEB_REVISION,
        "google_drive_file_id": GOOGLE_DRIVE_FILE_ID,
        "checkpoint": checkpoint_lock,
        "portable_source_receipt_contract_sha256": receipt_contract_sha256,
        "catalog_contract_sha256": catalog_contract,
        "source_tensor_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "generator_sha256": sha256_file(generator),
        "independent_of_evo_native_runtime": True,
        "oracle_execution_call": "pinned-geneb-reference",
        "clean_geneb_reference": True,
        "vendored_patch": "DNAGPT.forward returns (logits, post-final-LN hidden)",
        "tokenization": "literal <R> prefix plus dynamic nonoverlapping 1..6-mers",
        "pooling": "attention-mask direct-f32-division mean",
        "checkpoint_loading": "torch.load(weights_only=True,mmap=True), raw BF16 state copied into official F32 model",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for input_spec in inputs:
        input_ids = torch.tensor(
            [input_spec["input_ids"]], dtype=torch.long, device="cpu"
        )
        attention_mask = input_ids.ne(PAD_ID).to(torch.float32)
        pooled_runs = []
        with torch.inference_mode():
            for _ in range(2):
                _, hidden = model(input_ids)
                if list(hidden.shape) != [1, input_ids.shape[1], HIDDEN_SIZE]:
                    raise OracleError("post-final-LN hidden shape differs")
                mask = attention_mask.unsqueeze(-1)
                pooled_runs.append((hidden * mask).sum(dim=1) / mask.sum(dim=1))
        if not torch.equal(pooled_runs[0], pooled_runs[1]):
            raise OracleError(f"{input_spec['name']} repeated official forwards differ")
        values = pooled_runs[0].cpu().contiguous().numpy().astype("<f4", copy=False)
        npy_path = output_dir / (
            input_spec["name"] + ".attention-mask-mean.f32.npy"
        )
        np.save(npy_path, values, allow_pickle=False)
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": RUNTIME_ID,
            "input_sha256": input_spec["sha256"],
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": [float(value) for value in values.reshape(-1)],
            "environment_lock": environment,
            "provenance": provenance,
        }
        vector_path = output_dir / (
            input_spec["name"] + ".independent-oracle-vector.json"
        )
        vector_path.write_bytes(canonical_json(vector))
        summaries.append(
            {
                "name": input_spec["name"],
                "input_sha256": input_spec["sha256"],
                "sequence_length": len(input_spec["sequence"]),
                "input_ids": [input_spec["input_ids"]],
                "attention_mask": [[1] * len(input_spec["input_ids"])],
                "output": {
                    "shape": list(values.shape),
                    "npy_sha256": sha256_file(npy_path),
                    "raw_f32_sha256": sha256_bytes(values.tobytes(order="C")),
                    "vector_sha256": sha256_file(vector_path),
                    "first_16": vector["values"][:16],
                },
            }
        )
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "inputs": summaries,
        "environment_lock": environment,
        "provenance": provenance,
    }
    report_path = output_dir / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))
    return {
        "report_sha256": sha256_file(report_path),
        "inputs": [item["output"] for item in summaries],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--input-0", required=True, type=Path)
    parser.add_argument("--input-1", required=True, type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    extractor = args.extractor.resolve(strict=True)
    input_paths = (
        args.input_0.resolve(strict=True),
        args.input_1.resolve(strict=True),
    )
    source_lock, extractor_lock = validate_small_sources(source_root, extractor)
    manifest = validate_expected_manifest()
    tokenizer, inputs = validate_tokenizer(source_root, input_paths)
    static_result = {
        "runtime_id": RUNTIME_ID,
        "checkpoint": {
            "google_drive_file_id": GOOGLE_DRIVE_FILE_ID,
            "name": CHECKPOINT_NAME,
            "size": CHECKPOINT_SIZE,
            "sha256": None,
            "payload_opened": False,
        },
        "source_manifest": manifest,
        "source_files": source_lock,
        "extractor": extractor_lock,
        "inputs": inputs,
        "thresholds": {
            "max_abs": 0.0001,
            "mean_abs": 0.00001,
            "cosine": 0.999999,
        },
    }
    if args.audit_only:
        print(json.dumps(static_result, sort_keys=True))
        return 0
    if args.catalog is None or args.receipt is None or args.output_dir is None:
        raise OracleError(
            "oracle execution requires --catalog, --receipt, and --output-dir"
        )
    _, checkpoint_sha256, catalog_contract = validate_catalog(
        args.catalog.resolve(strict=True)
    )
    checkpoint, checkpoint_lock, receipt_contract = validate_receipt(
        args.receipt.resolve(strict=True), checkpoint_sha256
    )
    result = execute_oracle(
        source_root,
        tokenizer,
        inputs,
        checkpoint,
        checkpoint_lock,
        receipt_contract,
        catalog_contract,
        source_lock,
        extractor_lock,
        args.output_dir.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, OracleError, ValueError) as error:
        print(
            "generate_geneb_dna_gpt_3b_m_upstream_oracle: error: %s" % error,
            file=sys.stderr,
        )
        raise SystemExit(2) from error
