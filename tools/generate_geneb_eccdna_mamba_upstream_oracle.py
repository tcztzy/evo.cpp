#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned GENEB eccDNAMamba CPU F32 embedding oracle offline.

The model wrapper, checkpoint, tokenizer, and Mamba2/MLP/Block classes come
from immutable upstream sources.  CUDA/Triton kernels are replaced only by
the official pure-PyTorch reference definitions from the same mamba-ssm tag.
This program never imports or invokes evo.cpp's native runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import types
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit


RUNTIME_ID = "geneb-eccdna-mamba"
SOURCE_REPO = "eccDNAMamba/eccDNAMamba-1M"
SOURCE_REVISION = "36d95a3f783fa93640ce9e070ebde3f0ebed175d"
GENEB_REPO = "darlednik/GENEB"
GENEB_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SIZE = 2137
EXTRACTOR_SHA256 = (
    "928e7e4b46eca6e58a279ff895cfc39c0d498347671b5b542e694b0c97906d52"
)
GENEB_MODEL_SIZE = 6298
GENEB_MODEL_SHA256 = (
    "6f2c8941d4087493d4685ba35954c1209945c8fba771acba2d7f71546e949aa6"
)
GENEB_PYPROJECT_SIZE = 2759
GENEB_PYPROJECT_SHA256 = (
    "616a15c0a751eb14687fefe691cd624fbf4cc104605f167a5782cb29d16060e0"
)
MAMBA_REPO = "state-spaces/mamba"
MAMBA_VERSION = "2.2.4"
MAMBA_REVISION = "95d8aba8a8c75aedcaa6143713b11e745e7cd0d9"
MAMBA_SDIST_SHA256 = (
    "e4114c69302796c91b71e90032c2d974f611608fab331582a80de6eaf075efb9"
)


SOURCE_FILES = {
    ".gitattributes": (1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
    "README.md": (1513, "8ee08aeaed0f397d9259bb07bb13b8f151fcda39399b82b942f6ffbb3f619336"),
    "config.json": (1060, "19a61e4b6033ce0b8b5e990ea74476ca02e2cb9fd7f8c777372db0a32669710d"),
    "model.safetensors": (2149753112, "cbb1c8cd8038c8984637000bea3a135b68b9e9ce97cb7f3bd6fedc24d4888cb9"),
    "modeling_BiMambaForMaskedLM.py": (6298, GENEB_MODEL_SHA256),
    "special_tokens_map.json": (557, "89f4a8d64900c5a7b7701f9fdee93df4f604638cba5cdb4e647550c245f9394d"),
    "tokenizer.json": (288744, "71d33b47f9635fbfa843b537362db6ec6bffd1db2bfeeba629aa31a4f093abdf"),
    "tokenizer_config.json": (972, "acd1390b79ebfb49205b38aaa5a22a85afdee027db4c961052ce43c006a2ce8e"),
    "training_args.bin": (5304, "15a959515df98572b9f6666e1193becf1b94247b43f287a5c95c0ed7e67769da"),
}


MAMBA_FILES = {
    "pyproject.toml": (964, "ba10b7ecd4e595818ee2e5dd53f1b54c279176833e448056975eb962b09d7071"),
    "setup.py": (13660, "86373633789fde9e7d896654c10d2abd92dbab86ee8d85069c216c57c56571a5"),
    "mamba_ssm/models/config_mamba.py": (489, "2a72c1686f775b56547e39ca4406ba10148d12fd7a791c57ce2ba85126010fcd"),
    "mamba_ssm/models/mixer_seq_simple.py": (11667, "13409d7044e930ea3271e4b8ddceaf8155ec49b8e5ac299fba7bb0df6d80cb21"),
    "mamba_ssm/modules/mamba2.py": (17479, "605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707"),
    "mamba_ssm/modules/mamba_simple.py": (11710, "a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3"),
    "mamba_ssm/modules/block.py": (3729, "b62e755195c277a027c5d9cc8d576a8ae4a1d1317143b91370b2f8ce683b4cc1"),
    "mamba_ssm/modules/mlp.py": (1130, "d3aa360ae67608d2582975f25ec9adc57613e442f2edefcd88e91363e6de5021"),
    "mamba_ssm/modules/mha.py": (13392, "4244dabb6871742c53b925fce0a5a061ad7072e46430264b70bc225977014bb8"),
    "mamba_ssm/ops/selective_scan_interface.py": (19852, "c79f4432536402fe4d6c1026ccf8d5fc0aaa62b790e9a622295aee14ac45dbcc"),
    "mamba_ssm/ops/triton/ssd_combined.py": (53538, "3717ba118605d9a5c9271d9195e2168c55b82242adcd46b143d3d9918f7c72ba"),
    "mamba_ssm/ops/triton/layer_norm.py": (35966, "006fb18f7098fc244a318c899841ad4c1a6ea0f614dfe7a1feb4e2e38185235f"),
    "mamba_ssm/ops/triton/layernorm_gated.py": (17763, "eb6252e247b90f1c8a75946efbc1a221e0c4da701b6757ddae49f3495cf7a42f"),
}


def deterministic_sequence(length: int) -> str:
    state = 0x6D2B79F5
    alphabet = "ACGT"
    result = []
    for _ in range(length):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        result.append(alphabet[(state >> 30) & 3])
    return "".join(result)


INPUTS = (
    ("input-0", "ACGTTGCAACGTGGTTAACCGTACGATCGTTA"),
    ("input-1", deterministic_sequence(192)),
)


class OracleError(RuntimeError):
    """Raised when pinned source or upstream execution differs."""


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            portable(key, label + " key")
            portable(item, "%s.%s" % (label, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            portable(item, "%s[%d]" % (label, index))
        return
    if not isinstance(value, str):
        return
    parsed = urlsplit(value)
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or parsed.scheme.lower() == "file"
    ):
        raise OracleError(label + " contains a local absolute filesystem path")


def checked_file(path: Path, size: int, digest: str, label: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise OracleError(label + " is not a regular file")
    actual = {"size": resolved.stat().st_size, "sha256": sha256_file(resolved)}
    if actual != {"size": size, "sha256": digest}:
        raise OracleError(label + " differs from the pinned source")
    return actual


def git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise OracleError("git %s failed: %s" % (" ".join(arguments), result.stderr.strip()))
    return result.stdout.strip()


def validate_snapshot(snapshot: Path, receipt_path: Path) -> tuple[dict[str, Any], str]:
    snapshot = snapshot.resolve(strict=True)
    if snapshot.name != SOURCE_REVISION:
        raise OracleError("snapshot directory does not name the pinned revision")
    receipt_bytes = receipt_path.resolve(strict=True).read_bytes()
    receipt = json.loads(receipt_bytes)
    if not isinstance(receipt, dict):
        raise OracleError("source receipt root differs")
    expected_identity = {
        "schema_version": 1,
        "kind": "source-checkpoint",
        "model_id": RUNTIME_ID,
        "repo": SOURCE_REPO,
        "requested_revision": "main",
        "resolved_revision": SOURCE_REVISION,
        "source_kind": "huggingface",
        "load_path": None,
    }
    wrong = {key: (receipt.get(key), value) for key, value in expected_identity.items() if receipt.get(key) != value}
    if wrong:
        raise OracleError("source receipt identity differs: %s" % wrong)
    raw_files = receipt.get("files")
    if not isinstance(raw_files, list) or len(raw_files) != len(SOURCE_FILES):
        raise OracleError("source receipt file set differs")
    received: dict[str, tuple[int, str]] = {}
    for item in raw_files:
        if not isinstance(item, dict) or set(item) != {"name", "size", "sha256", "path"}:
            raise OracleError("source receipt file entry differs")
        name = item["name"]
        if name in received or name not in SOURCE_FILES:
            raise OracleError("source receipt file name differs")
        received[name] = (item["size"], item["sha256"])
        expected_size, expected_digest = SOURCE_FILES[name]
        checked_file(Path(item["path"]), expected_size, expected_digest, "receipt " + name)
        checked_file(snapshot / name, expected_size, expected_digest, "snapshot " + name)
    if received != SOURCE_FILES:
        raise OracleError("source receipt size/SHA256 manifest differs")
    portable_receipt = {
        **expected_identity,
        "files": [
            {"name": name, "size": size, "sha256": digest}
            for name, (size, digest) in sorted(SOURCE_FILES.items())
        ],
    }
    manifest = {
        name: {"size": size, "sha256": digest}
        for name, (size, digest) in sorted(SOURCE_FILES.items())
    }
    return manifest, sha256_bytes(canonical_json(portable_receipt))


def validate_geneb(
    repository: Path, extractor: Path, model_source: Path
) -> dict[str, Any]:
    repository = repository.resolve(strict=True)
    if git_output(repository, "rev-parse", "HEAD") != GENEB_REVISION:
        raise OracleError("GENEB checkout revision differs")
    if git_output(repository, "status", "--short"):
        raise OracleError("GENEB checkout must be clean")
    checked_file(extractor, EXTRACTOR_SIZE, EXTRACTOR_SHA256, "GENEB extractor")
    checked_file(model_source, GENEB_MODEL_SIZE, GENEB_MODEL_SHA256, "GENEB model wrapper")
    checked_file(
        repository / "embedding_pipeline" / "pyproject.toml",
        GENEB_PYPROJECT_SIZE,
        GENEB_PYPROJECT_SHA256,
        "GENEB pyproject",
    )
    extractor_text = extractor.read_text(encoding="utf-8")
    required_fragments = (
        "BiMambaForMaskedLM.from_pretrained",
        "padding=True",
        "truncation=True",
        "hs = out.hidden_states",
        "pooled = summed / lengths",
        "self.model.mamba_forward.use_mem_eff_path = False",
    )
    if any(fragment not in extractor_text for fragment in required_fragments):
        raise OracleError("GENEB extractor semantics differ")
    return {
        "repo": GENEB_REPO,
        "revision": GENEB_REVISION,
        "extractor": {"size": EXTRACTOR_SIZE, "sha256": EXTRACTOR_SHA256},
        "model_wrapper": {"size": GENEB_MODEL_SIZE, "sha256": GENEB_MODEL_SHA256},
        "model_wrapper_matches_snapshot_remote_code": True,
        "pyproject": {
            "size": GENEB_PYPROJECT_SIZE,
            "sha256": GENEB_PYPROJECT_SHA256,
        },
    }


def validate_mamba(source: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    if git_output(source, "rev-parse", "HEAD") != MAMBA_REVISION:
        raise OracleError("mamba-ssm checkout revision differs")
    if git_output(source, "describe", "--exact-match", "--tags") != "v" + MAMBA_VERSION:
        raise OracleError("mamba-ssm checkout tag differs")
    if git_output(source, "status", "--short"):
        raise OracleError("mamba-ssm checkout must be clean")
    return {
        name: checked_file(source / name, size, digest, "mamba-ssm " + name)
        for name, (size, digest) in sorted(MAMBA_FILES.items())
    }


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
        raise OracleError("official reference definitions differ: %s" % sorted(names - found))
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


def plain_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


def import_source(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OracleError("cannot load pinned source module " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_official_cpu_reference(mamba_source: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from einops import rearrange, repeat

    package = mamba_source / "mamba_ssm"
    package_module("mamba_ssm", package).__version__ = MAMBA_VERSION
    for name, relative in (
        ("mamba_ssm.models", "models"),
        ("mamba_ssm.modules", "modules"),
        ("mamba_ssm.ops", "ops"),
        ("mamba_ssm.ops.triton", "ops/triton"),
        ("mamba_ssm.distributed", "distributed"),
        ("mamba_ssm.utils", "utils"),
    ):
        package_module(name, package / relative)

    scan_path = package / "ops" / "selective_scan_interface.py"
    selective_scan_ref = extract_ast_definitions(
        scan_path,
        {"selective_scan_ref"},
        {"torch": torch, "F": functional, "rearrange": rearrange, "repeat": repeat},
    )["selective_scan_ref"]
    scan_module = plain_module("mamba_ssm.ops.selective_scan_interface")
    scan_module.selective_scan_ref = selective_scan_ref
    scan_module.selective_scan_fn = selective_scan_ref
    scan_module.mamba_inner_fn = None

    norm_path = package / "ops" / "triton" / "layer_norm.py"
    norm_defs = extract_ast_definitions(
        norm_path,
        {"layer_norm_ref", "rms_norm_ref"},
        {"torch": torch, "F": functional},
    )

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
            raise OracleError("dropout-mask return is outside this oracle")
        function = norm_defs["rms_norm_ref"] if is_rms_norm else norm_defs["layer_norm_ref"]
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

    def rms_norm_fn(x: Any, weight: Any, bias: Any, **kwargs: Any) -> Any:
        return layer_norm_fn(x, weight, bias, is_rms_norm=True, **kwargs)

    rms_norm_class = extract_ast_definitions(
        norm_path,
        {"RMSNorm"},
        {"torch": torch, "rms_norm_fn": rms_norm_fn},
    )["RMSNorm"]
    norm_module = plain_module("mamba_ssm.ops.triton.layer_norm")
    norm_module.RMSNorm = rms_norm_class
    norm_module.layer_norm_fn = layer_norm_fn
    norm_module.rms_norm_fn = rms_norm_fn
    sys.modules["mamba_ssm.ops.triton.layernorm"] = norm_module

    gated_path = package / "ops" / "triton" / "layernorm_gated.py"
    gated_ref = extract_ast_definitions(
        gated_path,
        {"rms_norm_ref"},
        {"torch": torch, "F": functional, "rearrange": rearrange},
    )["rms_norm_ref"]

    class RMSNormGated(torch.nn.Module):
        def __init__(
            self,
            hidden_size: int,
            eps: float = 1e-5,
            group_size: int | None = None,
            norm_before_gate: bool = True,
            device: Any = None,
            dtype: Any = None,
        ) -> None:
            super().__init__()
            self.eps = eps
            self.group_size = group_size
            self.norm_before_gate = norm_before_gate
            self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
            self.register_parameter("bias", None)
            torch.nn.init.ones_(self.weight)

        def forward(self, x: Any, z: Any = None) -> Any:
            return gated_ref(
                x,
                self.weight,
                self.bias,
                z=z,
                eps=self.eps,
                group_size=self.group_size,
                norm_before_gate=self.norm_before_gate,
                upcast=True,
            )

    gated_module = plain_module("mamba_ssm.ops.triton.layernorm_gated")
    gated_module.RMSNorm = RMSNormGated
    gated_module.rms_norm_ref = gated_ref

    ssd_path = package / "ops" / "triton" / "ssd_combined.py"
    ssd_selective_scan = extract_ast_definitions(
        ssd_path,
        {"ssd_selective_scan"},
        {"torch": torch, "F": functional, "rearrange": rearrange, "repeat": repeat},
    )["ssd_selective_scan"]

    def mamba_chunk_scan_combined(
        x: Any,
        dt: Any,
        A: Any,
        B: Any,
        C: Any,
        chunk_size: int,
        D: Any = None,
        z: Any = None,
        dt_bias: Any = None,
        initial_states: Any = None,
        seq_idx: Any = None,
        cu_seqlens: Any = None,
        dt_softplus: bool = False,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        return_final_states: bool = False,
        return_varlen_states: bool = False,
    ) -> Any:
        del chunk_size
        if (
            initial_states is not None
            or seq_idx is not None
            or cu_seqlens is not None
            or return_final_states
            or return_varlen_states
        ):
            raise OracleError("stateful/varlen Mamba2 is outside this oracle")
        return ssd_selective_scan(
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
        )

    def unavailable_fast_path(*_args: Any, **_kwargs: Any) -> Any:
        raise OracleError("Mamba2 memory-efficient CUDA path was unexpectedly selected")

    ssd_module = plain_module("mamba_ssm.ops.triton.ssd_combined")
    ssd_module.mamba_chunk_scan_combined = mamba_chunk_scan_combined
    ssd_module.mamba_split_conv1d_scan_combined = unavailable_fast_path
    state_module = plain_module("mamba_ssm.ops.triton.selective_state_update")
    state_module.selective_state_update = None

    class UnusedModule(torch.nn.Module):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__()
            raise OracleError("an unused upstream auxiliary module was selected")

    parallel_module = plain_module("mamba_ssm.distributed.tensor_parallel")
    parallel_module.ColumnParallelLinear = UnusedModule
    parallel_module.RowParallelLinear = UnusedModule
    distributed_module = plain_module("mamba_ssm.distributed.distributed_utils")
    distributed_module.all_reduce = unavailable_fast_path
    distributed_module.reduce_scatter = unavailable_fast_path
    simple_module = plain_module("mamba_ssm.modules.mamba_simple")
    simple_module.Mamba = UnusedModule
    mha_module = plain_module("mamba_ssm.modules.mha")
    mha_module.MHA = UnusedModule
    generation_module = plain_module("mamba_ssm.utils.generation")
    generation_module.GenerationMixin = object
    hf_module = plain_module("mamba_ssm.utils.hf")
    hf_module.load_config_hf = unavailable_fast_path
    hf_module.load_state_dict_hf = unavailable_fast_path

    imported = {}
    for name, relative in (
        ("mamba_ssm.models.config_mamba", "models/config_mamba.py"),
        ("mamba_ssm.modules.mlp", "modules/mlp.py"),
        ("mamba_ssm.modules.mamba2", "modules/mamba2.py"),
        ("mamba_ssm.modules.block", "modules/block.py"),
        ("mamba_ssm.models.mixer_seq_simple", "models/mixer_seq_simple.py"),
    ):
        imported[name] = import_source(name, package / relative)
    return {
        "official_class_files": {
            name: str(Path(module.__file__).resolve().relative_to(mamba_source.resolve()))
            for name, module in sorted(imported.items())
        },
        "selective_scan": "official selective_scan_ref AST from mamba-ssm v2.2.4",
        "ssd_adapter": "official ssd_selective_scan AST mapped to the non-fused Mamba2 call",
        "normalization": "official RMSNorm reference ASTs from mamba-ssm v2.2.4",
        "causal_conv1d": "official torch.nn.Conv1d fallback in Mamba2.forward",
        "memory_efficient_path": False,
        "native_runtime_imported": False,
    }


def package_lock() -> list[str]:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append("%s==%s" % (name.lower().replace("_", "-"), distribution.version))
    return sorted(set(packages))


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    return {
        "dtype": "float32",
        "shape": list(tensor.shape),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "raw_little_endian_sha256": sha256_bytes(flat.numpy().astype("<f4", copy=False).tobytes()),
    }


def save_npy(path: Path, tensor: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype("<f4", copy=False)
    np.save(path, array, allow_pickle=False)
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor": tensor_summary(tensor),
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
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

    snapshot = args.snapshot.resolve(strict=True)
    source_manifest, receipt_contract_sha256 = validate_snapshot(snapshot, args.source_receipt)
    geneb_audit = validate_geneb(args.geneb_repo, args.extractor, args.model_source)
    if sha256_file(snapshot / "modeling_BiMambaForMaskedLM.py") != sha256_file(args.model_source):
        raise OracleError("GENEB and snapshot model wrappers differ")
    mamba_manifest = validate_mamba(args.mamba_source)
    cpu_reference = install_official_cpu_reference(args.mamba_source.resolve(strict=True))

    import numpy as np
    import safetensors
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    model_module = import_source("geneb_pinned_eccdna_model", args.model_source.resolve(strict=True))
    model_class = model_module.BiMambaForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device="cpu", dtype=torch.float32).eval()

    checkpoint_path = snapshot / "model.safetensors"
    with safe_open(checkpoint_path, framework="pt", device="cpu") as source:
        checkpoint_keys = sorted(source.keys())
        if len(checkpoint_keys) != 584:
            raise OracleError("checkpoint tensor count differs")
        if any(source.get_tensor(name).dtype != torch.float32 for name in checkpoint_keys):
            raise OracleError("checkpoint contains non-F32 tensors")
    model_keys = sorted(model.state_dict())
    if model_keys != checkpoint_keys:
        raise OracleError("loaded model state manifest differs from checkpoint")
    if sum(parameter.numel() for parameter in model.parameters()) != 537420672:
        raise OracleError("loaded model parameter count differs")

    mixers = []
    for direction in (model.mamba_forward, model.mamba_backward):
        for layer in direction.backbone.layers:
            mixer = layer.mixer
            if mixer.__class__.__module__ != "mamba_ssm.modules.mamba2":
                raise OracleError("loaded mixer class differs from pinned Mamba2")
            mixer.use_mem_eff_path = False
            mixers.append(mixer)
    if len(mixers) != 48 or any(mixer.use_mem_eff_path for mixer in mixers):
        raise OracleError("Mamba2 slow-path normalization differs")

    expected_tokenizer = {
        "vocab_size": 4096,
        "padding_side": "right",
        "model_max_length": 1000000000000000019884624838656,
        "pad_token_id": 0,
        "cls_token_id": 1,
        "unk_token_id": 2,
        "mask_token_id": 3,
    }
    actual_tokenizer = {
        key: getattr(tokenizer, key) for key in expected_tokenizer
    }
    if actual_tokenizer != expected_tokenizer:
        raise OracleError("loaded tokenizer contract differs: %s" % actual_tokenizer)
    config_contract = {
        "architectures": ["BiMambaForMaskedLM"],
        "d_model": 768,
        "n_layer": 24,
        "vocab_size": 4096,
        "d_intermediate": 0,
        "ssm_cfg": {"layer": "Mamba2"},
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "torch_dtype": "float32",
    }
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    wrong_config = {
        key: (config.get(key), value)
        for key, value in config_contract.items()
        if config.get(key) != value
    }
    if wrong_config:
        raise OracleError("checkpoint config differs: %s" % wrong_config)

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
        "packages": package_lock(),
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mps_available_but_unused": bool(torch.backends.mps.is_available()),
        },
        "transformers": {
            "version": transformers.__version__,
            "offline": True,
            "local_files_only": True,
            "model_class": "%s.%s" % (model.__class__.__module__, model.__class__.__name__),
            "tokenizer_class": "%s.%s" % (tokenizer.__class__.__module__, tokenizer.__class__.__name__),
        },
        "safetensors": {"version": safetensors.__version__, "tensor_count": len(checkpoint_keys)},
        "numpy": {"version": np.__version__},
        "source_files": source_manifest,
        "geneb": geneb_audit,
        "mamba_ssm": {
            "repo": MAMBA_REPO,
            "version": MAMBA_VERSION,
            "revision": MAMBA_REVISION,
            "sdist_sha256": MAMBA_SDIST_SHA256,
            "source_files": mamba_manifest,
            "cpu_reference": cpu_reference,
        },
        "model_semantics": {
            "width": 768,
            "layers_per_direction": 24,
            "directions": 2,
            "mamba2_inner_width": 1536,
            "mamba2_state_width": 128,
            "mamba2_heads": 24,
            "mamba2_head_width": 64,
            "mamba2_groups": 1,
            "mamba2_convolution_width": 4,
            "gated_mlp_width": 3072,
            "hidden_tap": "MaskedLMOutput.hidden_states",
            "projection": "lm_head_proj([forward,reverse])",
            "pooling": "attention-mask-mean",
            "special_tokens": "include",
        },
    }
    portable(environment_lock, "environment lock")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise OracleError("oracle output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for label, sequence in INPUTS:
        input_bytes = (">probe\n" + sequence + "\n").encode("ascii")
        input_path = output / (label + ".fasta")
        input_path.write_bytes(input_bytes)
        encoded = tokenizer(
            [sequence],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to("cpu")
        attention_mask = encoded["attention_mask"].to("cpu")
        if input_ids.shape != attention_mask.shape or input_ids.shape[0] != 1:
            raise OracleError("tokenizer tensor shapes differ")
        if input_ids[0, 0].item() != 1 or not bool(torch.all(attention_mask == 1)):
            raise OracleError("CLS/no-padding singleton tokenizer semantics differ")
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = result.hidden_states
            if hidden is None or tuple(hidden.shape) != (1, input_ids.shape[1], 768):
                raise OracleError("upstream projected hidden shape differs")
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).to(
                device="cpu", dtype=torch.float32
            )
        if tuple(pooled.shape) != (1, 768) or not bool(torch.isfinite(pooled).all()):
            raise OracleError("upstream pooled output differs")
        npy = save_npy(output / (label + ".attention-mask-mean.f32.npy"), pooled)
        provenance = {
            "kind": "pinned-upstream-with-official-cpu-reference-kernels",
            "oracle_contract": "geneb-independent-oracle-v1",
            "independent_of_evo_native_runtime": True,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "checkpoint_sha256": SOURCE_FILES["model.safetensors"][1],
            "portable_source_receipt_contract_sha256": receipt_contract_sha256,
            "extractor_repo": GENEB_REPO,
            "extractor_revision": GENEB_REVISION,
            "extractor_sha256": EXTRACTOR_SHA256,
            "model_wrapper_sha256": GENEB_MODEL_SHA256,
            "mamba_repo": MAMBA_REPO,
            "mamba_revision": MAMBA_REVISION,
            "normalization_decision": "disable every Mamba2 memory-efficient CUDA path",
            "normalization_math": "official non-fused Mamba2 path plus official PyTorch references",
            "batch_size": 1,
            "padding_tokens": 0,
            "input_tokens": int(input_ids.shape[1]),
            "token_ids": [int(value) for value in input_ids[0].tolist()],
            "hidden_tap": "MaskedLMOutput.hidden_states",
            "pooling": "attention-mask-mean",
            "special_tokens": "include",
            "output_width": 768,
            "generator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        }
        portable(provenance, "provenance")
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": RUNTIME_ID,
            "input_sha256": sha256_bytes(input_bytes),
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": [float(value) for value in pooled.reshape(-1).tolist()],
            "environment_lock": environment_lock,
            "provenance": provenance,
        }
        portable(vector, "oracle vector")
        vector_path = output / (label + ".independent-oracle-vector.json")
        vector_path.write_bytes(canonical_json(vector))
        records.append(
            {
                "label": label,
                "sequence_length": len(sequence),
                "input_size": len(input_bytes),
                "input_sha256": sha256_bytes(input_bytes),
                "input_tokens": int(input_ids.shape[1]),
                "token_ids_sha256": sha256_bytes(
                    bytes().join(int(value).to_bytes(4, "little") for value in input_ids[0].tolist())
                ),
                "pooled": npy,
                "oracle_vector_file": vector_path.name,
                "oracle_vector_sha256": sha256_file(vector_path),
            }
        )

    (output / "environment-lock.json").write_bytes(canonical_json(environment_lock))
    report = {
        "schema_version": 1,
        "kind": "geneb-eccdna-mamba-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "checkpoint_sha256": SOURCE_FILES["model.safetensors"][1],
        "portable_source_receipt_contract_sha256": receipt_contract_sha256,
        "generator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "environment_lock_sha256": sha256_file(output / "environment-lock.json"),
        "record_count": len(records),
        "records": records,
    }
    portable(report, "oracle report")
    (output / "oracle-report.json").write_bytes(canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--geneb-repo", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--mamba-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = generate(args)
        print(
            json.dumps(
                {
                    "generator_sha256": report["generator_sha256"],
                    "output": str(args.output_dir),
                    "record_count": report["record_count"],
                    "report_sha256": sha256_file(args.output_dir / "oracle-report.json"),
                },
                sort_keys=True,
            )
        )
        return 0
    except (ImportError, OSError, OracleError, ValueError) as error:
        print("generate_geneb_eccdna_mamba_upstream_oracle: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
