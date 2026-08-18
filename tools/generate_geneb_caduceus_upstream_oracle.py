#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate normalized Caduceus-family oracles from pinned upstream code.

This tool deliberately does not import or reproduce evo.cpp's native Mamba
implementation.  It loads the immutable Hugging Face remote model code and
checkpoint, then runs the official mamba-ssm v2.2.5 Python modules with the
official pure-PyTorch selective-scan and normalization reference functions.

The pinned GENEB extractor passes ``attention_mask`` to a model forward that
does not accept it.  The sole normalization patch removes that keyword from
the model call.  The unchanged attention mask is used only for the final mean
pool, and this oracle requires one unpadded sequence so model internals cannot
be contaminated by padding.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.machinery
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "ec1e80e285fbc5ff13ab61066432cb62e5090d780bc05828d6bbba1d04564a34"
)
EXTRACTOR_PYPROJECT_SHA256 = (
    "616a15c0a751eb14687fefe691cd624fbf4cc104605f167a5782cb29d16060e0"
)
DEFECT_PATCH_SHA256 = (
    "a1870ce745d0d6c07f6af132e87c42a9928deffc1223bdd386ba3faa3a9a1b3e"
)
MAMBA_VERSION = "2.2.5"
MAMBA_REVISION = "e0761ece1db07e0949dd88b4f4cd440420a19fd9"


@dataclass(frozen=True)
class ModelSpec:
    runtime_id: str
    source_repo: str
    source_revision: str
    source_receipt_sha256: str
    d_model: int
    n_layer: int
    vocab_size: int
    rcps: bool
    residual_in_fp32: bool
    output_width: int
    checkpoint_name: str
    architectures: tuple[str, ...]
    tokenizer_model_max_length: int
    runtime_hard_max_tokens: int | None
    advertised_training_length: int | None
    padding_side: str
    tokenizer_kind: str
    emitted_vocab_size: int
    expected_source_files: dict[str, tuple[int, str]]


COMMON_SOURCE_FILES = {
    ".gitattributes": (
        1519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "configuration_caduceus.py": (
        1964,
        "f6d9785a13b3f62eef8d0911954d69cf2e60ac26e83ddc3975f8710c7d6864f9",
    ),
    "modeling_caduceus.py": (
        28938,
        "9c79343c590721395af3d4323eb3992a2d01684e683f9f24db4acb9d9e56f655",
    ),
    "modeling_rcps.py": (
        9981,
        "38e8e3dc8f7583a9fee9c2b69369db8bf4b340c8db92208dd02500bf53ac7da5",
    ),
    "special_tokens_map.json": (
        173,
        "5a7f632604e110003e8003e3b2eada8740c0069f98e13728be4b883c520cc71d",
    ),
    "tokenization_caduceus.py": (
        4966,
        "a85b0ee68a4764a3e27c11910972dc5ffa737204aa298b1e9cc798c18228da7e",
    ),
}
MODEL_SPECS = {
    "geneb-caduceus-ph-1k": ModelSpec(
        runtime_id="geneb-caduceus-ph-1k",
        source_repo=(
            "kuleshov-group/"
            "caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3"
        ),
        source_revision="9865108985311772704c2ece0bac082153a6167b",
        source_receipt_sha256=(
            "d047b883b19bb607b1ff9751fcaa82a8286657ae2571f62cd22c0a9f6d1a09d4"
        ),
        d_model=256,
        n_layer=4,
        vocab_size=16,
        rcps=False,
        residual_in_fp32=False,
        output_width=256,
        checkpoint_name="model.safetensors",
        architectures=("CaduceusForMaskedLM",),
        tokenizer_model_max_length=1024,
        runtime_hard_max_tokens=1024,
        advertised_training_length=None,
        padding_side="left",
        tokenizer_kind="single-nucleotide",
        emitted_vocab_size=12,
        expected_source_files={
            **COMMON_SOURCE_FILES,
            "README.md": (
                5178,
                "6936b52be9212b61193c08317559a1999a91b5f10f3382f54b0639aaf7a26c8d",
            ),
            "config.json": (
                1375,
                "4bcbb5df87ce19bf281acf5081d1ca7071bbe0c71f91429c8e7cd646105a5d09",
            ),
            "model.safetensors": (
                7746792,
                "7180a75f6cb08029309eda31d85b45623c7f095cc83f0f797c5a34b64f1f70de",
            ),
            "tokenizer_config.json": (
                1483,
                "185289a2ea93a4d19ff86c6488d5857e9c5c2c28ac9776bb95bcfff9827a6077",
            ),
        },
    ),
    "geneb-caduceus-ps-131k": ModelSpec(
        runtime_id="geneb-caduceus-ps-131k",
        source_repo=(
            "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
        ),
        source_revision="d89eeb853136ea64da7feb3d0c8e909771b17ae6",
        source_receipt_sha256=(
            "c9f299ac6bd4eb02f653e26e3cbbd45833f55d7b130c5476ff0f44f9e6d481b7"
        ),
        d_model=256,
        n_layer=16,
        vocab_size=16,
        rcps=True,
        residual_in_fp32=False,
        output_width=512,
        checkpoint_name="model.safetensors",
        architectures=("CaduceusForMaskedLM",),
        tokenizer_model_max_length=131072,
        runtime_hard_max_tokens=131072,
        advertised_training_length=None,
        padding_side="left",
        tokenizer_kind="single-nucleotide",
        emitted_vocab_size=12,
        expected_source_files={
            **COMMON_SOURCE_FILES,
            "README.md": (
                2534,
                "6f4a109b2eb2e516b0ce73a3124448883b71c3ea350d013aee98838d3927cb25",
            ),
            "config.json": (
                1375,
                "d0f8eaf9d62370ffae2d41c92d00b9424fae31c823baf21111602dabc35eef45",
            ),
            "model.safetensors": (
                30937760,
                "a3e6976fe90460ff5d90d371b457d0263dc65e156ea5651b4a452e98228daef2",
            ),
            "tokenizer_config.json": (
                1485,
                "54b8a214423d01e4cf90fe5751d03d2a7117b4f7286b5727d9ee60fdd0bb7c8a",
            ),
        },
    ),
    "geneb-plant-caduceus": ModelSpec(
        runtime_id="geneb-plant-caduceus",
        source_repo="kuleshov-group/PlantCaduceus_l32",
        source_revision="e624c13c3d35415348b854c87a218893b23564f7",
        source_receipt_sha256=(
            "8d6ebd0eab40e5fddf8f8b5c6ea63d8dc04b5951cd2995a5d1ad4c21898f354b"
        ),
        d_model=1024,
        n_layer=32,
        vocab_size=8,
        rcps=True,
        residual_in_fp32=True,
        output_width=2048,
        checkpoint_name="pytorch_model.bin",
        architectures=("Caduceus",),
        tokenizer_model_max_length=1000000000000000019884624838656,
        runtime_hard_max_tokens=None,
        advertised_training_length=512,
        padding_side="right",
        tokenizer_kind="bpe",
        emitted_vocab_size=7,
        expected_source_files={
            ".gitattributes": (
                1519,
                "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
            ),
            "README.md": (
                2716,
                "c0778a49f845d600ed93b6e6756c70e39ad7fb0208892ca2763964ed5060af47",
            ),
            "config.json": (
                1326,
                "c3cb3d4ef77160a053835823a11dcf8a4b2a7913ce103369731587693255eee1",
            ),
            "configuration_caduceus.py": (
                1964,
                "f6d9785a13b3f62eef8d0911954d69cf2e60ac26e83ddc3975f8710c7d6864f9",
            ),
            "modeling_caduceus.py": (
                28617,
                "7fd8b55a1c7b1ac30fe2e33534d568d7f76c5e228deb2875cfbdeb9eec194f8e",
            ),
            "modeling_rcps.py": (
                9981,
                "38e8e3dc8f7583a9fee9c2b69369db8bf4b340c8db92208dd02500bf53ac7da5",
            ),
            "pytorch_model.bin": (
                901671450,
                "308d9125a0e2a350252c902c27c308bbcf56a1d8198f399f9e7c26d9b552ec10",
            ),
            "special_tokens_map.json": (
                419,
                "861c50dcd75107ba291017c96a1ef74ba76c9478396e521bae87e75421a6be5e",
            ),
            "tokenization_caduceus.py": (
                4956,
                "628f64872b5f466416d15716297b905c0cc6162c49b37b50b93a456c4c112e73",
            ),
            "tokenizer.json": (
                1117,
                "b22a14e4ea2dc98b0e80db6a38494bd1f6d386959935e38a3a6bb1d03215099f",
            ),
            "tokenizer_config.json": (
                754,
                "315463dde4293b24b3a45d1e5c54f3003aca791ae975db22b54cb2f4b8fc5b47",
            ),
        },
    ),
}
EXPECTED_MAMBA_FILES = {
    "mamba_ssm/modules/mamba_simple.py": (
        11710,
        "a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3",
    ),
    "mamba_ssm/modules/block.py": (
        3729,
        "b62e755195c277a027c5d9cc8d576a8ae4a1d1317143b91370b2f8ce683b4cc1",
    ),
    "mamba_ssm/ops/selective_scan_interface.py": (
        20025,
        "543edd077f11c80cccd5c5e2f926420d9105fd238d9e8960e53660cfa234dce8",
    ),
    "mamba_ssm/ops/triton/layer_norm.py": (
        35966,
        "006fb18f7098fc244a318c899841ad4c1a6ea0f614dfe7a1feb4e2e38185235f",
    ),
}
EXTRACTOR_CALL = (
    b"                outputs = self.model(\n"
    b"                    input_ids=input_ids,\n"
    b"                    attention_mask=attention_mask,\n"
    b"                )\n"
)
NORMALIZED_EXTRACTOR_CALL = (
    b"                outputs = self.model(\n"
    b"                    input_ids=input_ids,\n"
    b"                )\n"
)


class OracleError(RuntimeError):
    """Raised when the pinned upstream oracle contract is not satisfied."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def require_file(path: Path, label: str, *, allow_symlink: bool = False) -> Path:
    if path.is_symlink() and not allow_symlink:
        raise OracleError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise OracleError(f"{label} is not a regular file: {path}")
    return path


def load_json(
    path: Path, label: str, *, allow_symlink: bool = False
) -> dict[str, Any]:
    require_file(path, label, allow_symlink=allow_symlink)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} root must be an object")
    return value


def checked_file(
    path: Path,
    label: str,
    expected_size: int,
    expected_sha256: str,
    *,
    allow_symlink: bool = False,
) -> dict[str, Any]:
    require_file(path, label, allow_symlink=allow_symlink)
    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise OracleError(
            f"{label} differs: size={actual_size} sha256={actual_sha256}"
        )
    return {"size": actual_size, "sha256": actual_sha256}


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


def validate_source(
    snapshot: Path, receipt_path: Path, model_spec: ModelSpec
) -> dict[str, Any]:
    checked_file(
        receipt_path,
        "source receipt",
        receipt_path.stat().st_size,
        model_spec.source_receipt_sha256,
    )
    receipt = load_json(receipt_path, "source receipt")
    identity = (
        receipt.get("schema_version"),
        receipt.get("kind"),
        receipt.get("model_id"),
        receipt.get("repo"),
        receipt.get("resolved_revision"),
    )
    expected_identity = (
        1,
        "source-checkpoint",
        model_spec.runtime_id,
        model_spec.source_repo,
        model_spec.source_revision,
    )
    if identity != expected_identity:
        raise OracleError(f"source receipt identity differs: {identity!r}")
    raw_files = receipt.get("files")
    if not isinstance(raw_files, list):
        raise OracleError("source receipt files must be an array")
    files: dict[str, dict[str, Any]] = {}
    for entry in raw_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise OracleError("source receipt file entry is malformed")
        name = entry["name"]
        if name in files:
            raise OracleError(f"source receipt duplicates {name}")
        files[name] = entry
    if set(files) != set(model_spec.expected_source_files):
        raise OracleError("source receipt logical file set differs")

    manifest: dict[str, Any] = {}
    for name, (expected_size, expected_sha256) in (
        model_spec.expected_source_files.items()
    ):
        entry = files[name]
        if entry.get("size") != expected_size or entry.get("sha256") != expected_sha256:
            raise OracleError(f"source receipt metadata differs for {name}")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise OracleError(f"source receipt path is invalid for {name}")
        blob = Path(raw_path)
        checked_file(blob, f"source blob {name}", expected_size, expected_sha256)
        logical = snapshot / name
        checked_file(
            logical,
            f"snapshot file {name}",
            expected_size,
            expected_sha256,
            allow_symlink=True,
        )
        if logical.resolve(strict=True) != blob.resolve(strict=True):
            raise OracleError(f"snapshot file {name} does not resolve to receipt blob")
        manifest[name] = {"size": expected_size, "sha256": expected_sha256}
    return manifest


def validate_model_semantics(snapshot: Path, model_spec: ModelSpec) -> dict[str, Any]:
    config = load_json(
        snapshot / "config.json", "model config", allow_symlink=True
    )
    expected_config = {
        "architectures": list(model_spec.architectures),
        "bidirectional": True,
        "bidirectional_strategy": "add",
        "bidirectional_weight_tie": True,
        "d_model": model_spec.d_model,
        "fused_add_norm": True,
        "n_layer": model_spec.n_layer,
        "norm_epsilon": 0.00001,
        "rcps": model_spec.rcps,
        "residual_in_fp32": model_spec.residual_in_fp32,
        "rms_norm": True,
        "torch_dtype": "float32",
        "vocab_size": model_spec.vocab_size,
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            raise OracleError(
                f"pinned model config {name} differs: {config.get(name)!r}"
            )
    ssm_config = config.get("ssm_cfg")
    if not isinstance(ssm_config, dict):
        raise OracleError("pinned model config ssm_cfg must be an object")
    expected_ssm = {
        "bias": False,
        "conv_bias": True,
        "d_conv": 4,
        "d_state": 16,
        "dt_rank": "auto",
        "expand": 2,
    }
    for name, expected in expected_ssm.items():
        if ssm_config.get(name) != expected:
            raise OracleError(
                f"pinned model config ssm_cfg.{name} differs: "
                f"{ssm_config.get(name)!r}"
            )
    calculated_width = model_spec.d_model * (2 if model_spec.rcps else 1)
    if calculated_width != model_spec.output_width:
        raise OracleError("frozen RCPS/output-width contract is inconsistent")

    tokenizer_config = load_json(
        snapshot / "tokenizer_config.json",
        "tokenizer config",
        allow_symlink=True,
    )
    configured_padding_side = tokenizer_config.get("padding_side", "right")
    if configured_padding_side != model_spec.padding_side:
        raise OracleError("pinned tokenizer padding_side differs")
    if (
        tokenizer_config.get("model_max_length")
        != model_spec.tokenizer_model_max_length
    ):
        raise OracleError("pinned tokenizer model_max_length differs")
    if model_spec.tokenizer_kind == "bpe":
        tokenizer_json = load_json(
            snapshot / "tokenizer.json", "BPE tokenizer", allow_symlink=True
        )
        tokenizer_model = tokenizer_json.get("model")
        if not isinstance(tokenizer_model, dict):
            raise OracleError("pinned BPE tokenizer model must be an object")
        expected_vocab = {
            "[PAD]": 0,
            "[MASK]": 1,
            "[UNK]": 2,
            "a": 3,
            "c": 4,
            "g": 5,
            "t": 6,
        }
        if tokenizer_model.get("type") != "BPE":
            raise OracleError("pinned tokenizer model is not BPE")
        if tokenizer_model.get("vocab") != expected_vocab:
            raise OracleError("pinned BPE tokenizer vocabulary differs")
        if tokenizer_model.get("merges") != []:
            raise OracleError("pinned BPE tokenizer merges differ")
        if tokenizer_json.get("normalizer") != {"type": "Lowercase"}:
            raise OracleError("pinned BPE tokenizer normalizer differs")
        if tokenizer_json.get("pre_tokenizer") != {"type": "Whitespace"}:
            raise OracleError("pinned BPE tokenizer pre-tokenizer differs")
    return {
        "advertised_training_length": model_spec.advertised_training_length,
        "bidirectional": True,
        "bidirectional_strategy": "add",
        "bidirectional_weight_tie": True,
        "d_model": model_spec.d_model,
        "layers": model_spec.n_layer,
        "runtime_hard_max_tokens": model_spec.runtime_hard_max_tokens,
        "tokenizer_model_max_length": model_spec.tokenizer_model_max_length,
        "output_width": model_spec.output_width,
        "padding_side": model_spec.padding_side,
        "rcps": model_spec.rcps,
        "residual_in_fp32": model_spec.residual_in_fp32,
        "tokenizer_kind": model_spec.tokenizer_kind,
        "tokenizer_emitted_vocab_size": model_spec.emitted_vocab_size,
        "vocab_size": model_spec.vocab_size,
    }


def validate_extractor(
    extractor_repo: Path, extractor_path: Path, defect_patch: Path
) -> dict[str, Any]:
    if git_output(extractor_repo, "rev-parse", "HEAD") != EXTRACTOR_REVISION:
        raise OracleError("GENEB extractor checkout revision differs")
    relative = extractor_path.resolve(strict=True).relative_to(
        extractor_repo.resolve(strict=True)
    )
    committed = subprocess.run(
        ["git", "-C", str(extractor_repo), "show", f"HEAD:{relative.as_posix()}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if committed.returncode != 0:
        raise OracleError("cannot read the committed GENEB extractor")
    payload = require_file(extractor_path, "GENEB extractor").read_bytes()
    if payload != committed.stdout or sha256_bytes(payload) != EXTRACTOR_SHA256:
        raise OracleError("GENEB extractor differs from its pinned commit")
    pyproject = extractor_repo / "embedding_pipeline" / "pyproject.toml"
    checked_file(
        pyproject,
        "GENEB extractor pyproject",
        pyproject.stat().st_size,
        EXTRACTOR_PYPROJECT_SHA256,
    )
    if b"releases/download/v2.2.5/mamba_ssm-2.2.5" not in pyproject.read_bytes():
        raise OracleError("GENEB extractor no longer pins mamba-ssm v2.2.5")
    checked_file(
        defect_patch,
        "GENEB normalization patch",
        defect_patch.stat().st_size,
        DEFECT_PATCH_SHA256,
    )
    if payload.count(EXTRACTOR_CALL) != 1:
        raise OracleError("GENEB attention-mask defect call site differs")
    normalized = payload.replace(EXTRACTOR_CALL, NORMALIZED_EXTRACTOR_CALL)
    return {
        "repo_revision": EXTRACTOR_REVISION,
        "source_sha256": EXTRACTOR_SHA256,
        "pyproject_sha256": EXTRACTOR_PYPROJECT_SHA256,
        "patch_sha256": DEFECT_PATCH_SHA256,
        "normalized_source_sha256": sha256_bytes(normalized),
    }


def validate_mamba_source(source: Path) -> dict[str, Any]:
    if git_output(source, "rev-parse", "HEAD") != MAMBA_REVISION:
        raise OracleError("mamba-ssm checkout revision differs")
    if git_output(source, "describe", "--exact-match", "--tags") != f"v{MAMBA_VERSION}":
        raise OracleError("mamba-ssm checkout tag differs")
    if git_output(source, "status", "--short"):
        raise OracleError("mamba-ssm checkout must be clean")
    manifest: dict[str, Any] = {}
    for name, (expected_size, expected_sha256) in EXPECTED_MAMBA_FILES.items():
        path = source / name
        manifest[name] = checked_file(
            path, f"mamba-ssm source {name}", expected_size, expected_sha256
        )
    return manifest


def extract_ast_definitions(
    path: Path, names: set[str], globals_dict: dict[str, Any]
) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    found = {node.name for node in selected}
    if found != names:
        raise OracleError(
            f"official reference definitions differ: missing={sorted(names - found)}"
        )
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(globals_dict)
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def package_module(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(path)]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    sys.modules[name] = module
    return module


def install_official_cpu_reference(mamba_source: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from einops import rearrange, repeat

    package = mamba_source / "mamba_ssm"
    package_module("mamba_ssm", package).__version__ = MAMBA_VERSION
    package_module("mamba_ssm.modules", package / "modules")
    package_module("mamba_ssm.ops", package / "ops")
    package_module("mamba_ssm.ops.triton", package / "ops" / "triton")

    scan_path = package / "ops" / "selective_scan_interface.py"
    scan_defs = extract_ast_definitions(
        scan_path,
        {"selective_scan_ref"},
        {"torch": torch, "F": functional, "rearrange": rearrange, "repeat": repeat},
    )
    scan_module = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    scan_module.__file__ = str(scan_path)
    scan_module.__package__ = "mamba_ssm.ops"
    scan_module.__spec__ = importlib.machinery.ModuleSpec(
        scan_module.__name__, loader=None
    )
    scan_module.selective_scan_ref = scan_defs["selective_scan_ref"]
    scan_module.selective_scan_fn = scan_defs["selective_scan_ref"]
    scan_module.mamba_inner_fn = None
    sys.modules[scan_module.__name__] = scan_module

    norm_path = package / "ops" / "triton" / "layer_norm.py"
    norm_defs = extract_ast_definitions(
        norm_path,
        {"layer_norm_ref", "rms_norm_ref"},
        {"torch": torch, "F": functional},
    )
    layer_norm_ref = norm_defs["layer_norm_ref"]
    rms_norm_ref = norm_defs["rms_norm_ref"]

    def layer_norm_fn(
        x: Any,
        weight: Any,
        bias: Any,
        residual: Any = None,
        x1: Any = None,
        weight1: Any = None,
        bias1: Any = None,
        eps: float = 1e-6,
        dropout_p: float = 0.0,
        rowscale: Any = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
        return_dropout_mask: bool = False,
    ) -> Any:
        if return_dropout_mask:
            raise OracleError("dropout-mask return is not part of this CPU oracle")
        function = rms_norm_ref if is_rms_norm else layer_norm_ref
        return function(
            x,
            weight,
            bias,
            residual=residual,
            x1=x1,
            weight1=weight1,
            bias1=bias1,
            eps=eps,
            dropout_p=dropout_p,
            rowscale=rowscale,
            prenorm=prenorm,
            upcast=residual_in_fp32,
        )

    def rms_norm_fn(
        x: Any,
        weight: Any,
        bias: Any,
        residual: Any = None,
        x1: Any = None,
        weight1: Any = None,
        bias1: Any = None,
        eps: float = 1e-6,
        dropout_p: float = 0.0,
        rowscale: Any = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        return_dropout_mask: bool = False,
    ) -> Any:
        return layer_norm_fn(
            x,
            weight,
            bias,
            residual=residual,
            x1=x1,
            weight1=weight1,
            bias1=bias1,
            eps=eps,
            dropout_p=dropout_p,
            rowscale=rowscale,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        )

    rms_norm_class = extract_ast_definitions(
        norm_path,
        {"RMSNorm"},
        {"torch": torch, "rms_norm_fn": rms_norm_fn},
    )["RMSNorm"]
    norm_module = types.ModuleType("mamba_ssm.ops.triton.layer_norm")
    norm_module.__file__ = str(norm_path)
    norm_module.__package__ = "mamba_ssm.ops.triton"
    norm_module.__spec__ = importlib.machinery.ModuleSpec(
        norm_module.__name__, loader=None
    )
    norm_module.RMSNorm = rms_norm_class
    norm_module.layer_norm_fn = layer_norm_fn
    norm_module.rms_norm_fn = rms_norm_fn
    sys.modules[norm_module.__name__] = norm_module
    # The pinned remote model tries the legacy name before the v2 name.
    sys.modules["mamba_ssm.ops.triton.layernorm"] = norm_module

    mamba_module = importlib.import_module("mamba_ssm.modules.mamba_simple")
    block_module = importlib.import_module("mamba_ssm.modules.block")
    expected_mamba = (package / "modules" / "mamba_simple.py").resolve()
    expected_block = (package / "modules" / "block.py").resolve()
    if Path(mamba_module.__file__).resolve() != expected_mamba:
        raise OracleError("loaded Mamba class does not come from the pinned source")
    if Path(block_module.__file__).resolve() != expected_block:
        raise OracleError("loaded Block class does not come from the pinned source")
    return {
        "mamba_class_file": "mamba_ssm/modules/mamba_simple.py",
        "block_class_file": "mamba_ssm/modules/block.py",
        "selective_scan": "official selective_scan_ref AST from v2.2.5",
        "normalization": "official layer_norm_ref/rms_norm_ref AST from v2.2.5",
        "causal_conv1d": "torch.nn.Conv1d official mamba_simple fallback",
        "native_runtime_imported": False,
    }


def read_single_fasta(path: Path) -> tuple[str, str]:
    require_file(path, "canonical FASTA")
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise OracleError(f"cannot read canonical FASTA: {error}") from error
    lines = text.splitlines()
    headers = [index for index, line in enumerate(lines) if line.startswith(">")]
    if headers != [0] or len(lines) < 2 or not lines[0][1:]:
        raise OracleError("canonical FASTA must contain exactly one named record")
    sequence = "".join(lines[1:])
    if not sequence or any(character.isspace() for character in sequence):
        raise OracleError("canonical FASTA sequence must be nonempty and unbroken")
    return lines[0][1:], sequence


def package_lock() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", "freeze"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise OracleError(f"cannot freeze Python environment: {result.stderr.strip()}")
    return sorted(line for line in result.stdout.splitlines() if line)


def write_atomic(path: Path, payload: bytes, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise OracleError(f"oracle output already exists: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    model_spec = MODEL_SPECS[args.model]
    for name, value in {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
        "LANG": "C",
        "LC_ALL": "C",
    }.items():
        os.environ[name] = value

    source_manifest = validate_source(
        args.snapshot, args.source_receipt, model_spec
    )
    model_semantics = validate_model_semantics(args.snapshot, model_spec)
    extractor = validate_extractor(
        args.extractor_repo, args.extractor, args.defect_patch
    )
    mamba_manifest = validate_mamba_source(args.mamba_source)
    _, sequence = read_single_fasta(args.input)
    input_sha256 = sha256_file(args.input)
    cpu_reference = install_official_cpu_reference(args.mamba_source)

    import torch
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.snapshot,
        output_hidden_states=True,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.to("cpu")
    model.eval()
    if tokenizer.padding_side != model_spec.padding_side:
        raise OracleError("loaded tokenizer does not preserve padding side")
    if tokenizer.model_max_length != model_spec.tokenizer_model_max_length:
        raise OracleError("loaded tokenizer model_max_length differs")
    if int(tokenizer.vocab_size) != model_spec.emitted_vocab_size:
        raise OracleError("loaded tokenizer emitted vocabulary size differs")
    if bool(model.config.rcps) != model_spec.rcps:
        raise OracleError("loaded model RCPS mode differs")
    if int(model.config.d_model) != model_spec.d_model:
        raise OracleError("loaded model d_model differs")
    encoded = tokenizer(
        [sequence],
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded["attention_mask"].to("cpu")
    if tuple(input_ids.shape) != (1, len(sequence)):
        raise OracleError(f"tokenizer shape differs: {tuple(input_ids.shape)}")
    if tuple(attention_mask.shape) != tuple(input_ids.shape):
        raise OracleError("attention mask shape differs from token IDs")
    if not bool(torch.all(attention_mask == 1)):
        raise OracleError("single-sequence oracle unexpectedly contains padding")

    clean_error: TypeError | None = None
    with torch.inference_mode():
        try:
            model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError as error:
            clean_error = error
    if clean_error is None or "unexpected keyword argument 'attention_mask'" not in str(
        clean_error
    ):
        raise OracleError("pinned GENEB attention_mask TypeError was not reproduced")

    with torch.inference_mode():
        # GENEB normalization patch: do not pass the mask into model.forward.
        outputs = model(input_ids=input_ids)
        if outputs.hidden_states is None or not outputs.hidden_states:
            raise OracleError("upstream model did not return hidden states")
        last_hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = pooled.to(dtype=torch.float32, device="cpu")
    if tuple(pooled.shape) != (1, model_spec.output_width):
        raise OracleError(f"upstream pooled shape differs: {tuple(pooled.shape)}")
    values = [float(value) for value in pooled[0].tolist()]
    if not all(math.isfinite(value) for value in values):
        raise OracleError("upstream oracle contains non-finite values")

    generator_sha256 = sha256_file(Path(__file__).resolve())
    exception_message = str(clean_error)
    environment_lock = {
        "schema_version": 1,
        "oracle_contract": "geneb-independent-oracle-v1",
        "packages": package_lock(),
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "mps_available_but_unused": bool(torch.backends.mps.is_available()),
        },
        "transformers": {
            "version": transformers.__version__,
            "hub_offline": os.environ["HF_HUB_OFFLINE"],
            "offline": os.environ["TRANSFORMERS_OFFLINE"],
            "local_files_only": True,
            "trust_remote_code": True,
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "tokenizer_class": (
                f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
            ),
        },
        "mamba_ssm": {
            "version": MAMBA_VERSION,
            "revision": MAMBA_REVISION,
            "source_files": mamba_manifest,
            "cpu_reference": cpu_reference,
        },
        "source_files": source_manifest,
        "model_semantics": model_semantics,
    }
    provenance = {
        "kind": "pinned-upstream-normalized",
        "oracle_contract": "geneb-independent-oracle-v1",
        "independent_of_evo_native_runtime": True,
        "official_remote_modeling_code": True,
        "source_repo": model_spec.source_repo,
        "source_revision": model_spec.source_revision,
        "checkpoint_name": model_spec.checkpoint_name,
        "checkpoint_sha256": model_spec.expected_source_files[
            model_spec.checkpoint_name
        ][1],
        "source_receipt_sha256": model_spec.source_receipt_sha256,
        "extractor_repo": "ultimativity/GENEB",
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_sha256": extractor["source_sha256"],
        "extractor_pyproject_sha256": extractor["pyproject_sha256"],
        "normalization_patch": {
            "path": args.defect_patch.name,
            "sha256": extractor["patch_sha256"],
            "normalized_extractor_sha256": extractor["normalized_source_sha256"],
            "operation": "omit attention_mask from model.forward only",
        },
        "clean_forward_defect": {
            "type": type(clean_error).__name__,
            "message": exception_message,
            "message_sha256": sha256_bytes(exception_message.encode("utf-8")),
        },
        "attention_mask_usage": "post-forward-mean-only",
        "batch_size": 1,
        "advertised_training_length": model_spec.advertised_training_length,
        "padding_semantics": (
            f"{model_spec.padding_side} verified; no padding in singleton probe"
        ),
        "padding_tokens": 0,
        "input_tokens": int(input_ids.shape[1]),
        "hidden_tap": "outputs.hidden_states[-1]",
        "runtime_hard_max_tokens": model_spec.runtime_hard_max_tokens,
        "tokenizer_model_max_length": model_spec.tokenizer_model_max_length,
        "tokenizer_kind": model_spec.tokenizer_kind,
        "output_width": model_spec.output_width,
        "pooling": "attention-mask-mean",
        "rcps": model_spec.rcps,
        "generator_sha256": generator_sha256,
    }
    return {
        "schema_version": 1,
        "kind": "geneb-independent-oracle-vector",
        "runtime_id": model_spec.runtime_id,
        "input_sha256": input_sha256,
        "backend": "cpu",
        "profile": "cpu-f32",
        "values": values,
        "environment_lock": environment_lock,
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--extractor-repo", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--mamba-source", required=True, type=Path)
    parser.add_argument("--defect-patch", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        oracle = generate(args)
        payload = canonical_json(oracle)
        write_atomic(args.output, payload, args.force)
        values = oracle["values"]
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": sha256_bytes(payload),
                    "values": len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "l2_norm": math.sqrt(sum(value * value for value in values)),
                    "model": oracle["runtime_id"],
                    "normalization_patch_sha256": DEFECT_PATCH_SHA256,
                    "output_width": len(values),
                    "rcps": oracle["provenance"]["rcps"],
                },
                sort_keys=True,
            )
        )
    except (OracleError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
