#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate GPU-free normalized JanusDNA oracles from pinned upstream code.

The clean GENEB extractor is intentionally *not* presented as executable on
CPU: the pinned model defaults to CUDA-only Mamba kernels.  This generator
first reproduces that failure in the upstream mixer, then applies the hashed
one-line extractor normalization which selects the upstream slow Mamba path.
The two attention kernels which remain CUDA-only are replaced by independent
dense PyTorch definitions of their audited upstream semantics.  No evo.cpp
runtime implementation or converted artifact is imported by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


UPSTREAM_REVISION = "3f449542529a2948d73062514cf9844705a4277a"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "2e5bb3f8f5c95bdaebdbe2730ed5590d64a10d298a40837760ec4d05dbf36cda"
)
NORMALIZED_EXTRACTOR_SHA256 = (
    "803fc18784110554da7df593954edd79dcfd65357d1bda89af4b2e6fed9a9bcd"
)
DEFECT_PATCH_SHA256 = (
    "cb45b59041f949b7928b01ed82a47034c0c4cde0ee0546c7b9d0164a3cf68547"
)
CLEAN_ERROR = (
    "Fast Mamba kernels are not available. Make sure to they are installed "
    "and that the mamba module is on a CUDA device"
)
CLEAN_ERROR_SHA256 = (
    "71c7e07c4c1cf8d84b5bf4cb81c62251d7217d06c1c33084afc4df4d9ce054de"
)
EXTRACTOR_FRAGMENT = (
    b'        cfg["bidirectional"] = str(cfg["bidirectional"]).strip().lower()'
    b'.strip(",") == "true"\n'
)
NORMALIZED_EXTRACTOR_FRAGMENT = (
    EXTRACTOR_FRAGMENT + b'        cfg["use_mamba_kernels"] = False\n'
)
UPSTREAM_FILES = {
    "LICENSE": (
        11357,
        "53c3bce42b068bd4c9a7831e18d4d7e7eab1b9cd00b8a3faac0aa96793c99bc5",
    ),
    "caduceus/tokenization_caduceus.py": (
        4966,
        "a85b0ee68a4764a3e27c11910972dc5ffa737204aa298b1e9cc798c18228da7e",
    ),
    "janusdna/configuration_janusdna.py": (
        13071,
        "431b37095d6d30df6ac4d5a73e6b1c9d00405c667fca132c7258e7ac378bdc7c",
    ),
    "janusdna/modeling_janusdna.py": (
        119617,
        "5e23060319e5ab8ef5ead566ceca6d49c7c89f9df0aae808f9399cacbbe00a68",
    ),
    "janusdna.yml": (
        6055,
        "78c7734b6bd3687d5afa4e421e24afa8536a0ac1fbe890ef0d6f1ffd62aed36b",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    runtime_id: str
    checkpoint_name: str
    checkpoint_size: int
    checkpoint_sha256: str
    config_name: str
    config_size: int
    config_sha256: str
    state_entries: int
    clean_entries: int
    variant: str
    has_middle_attention: bool


MODEL_SPECS = {
    "geneb-janusdna-72-w": ModelSpec(
        runtime_id="geneb-janusdna-72-w",
        checkpoint_name="72_with_midattn.ckpt",
        checkpoint_size=108242317,
        checkpoint_sha256=(
            "94790a3f6d6e719cb63e2c26190bb4968b50a06bdee5aa01ca48b57932c9bc3e"
        ),
        config_name="72_with_midattn_model_config.json",
        config_size=1608,
        config_sha256=(
            "ba89cced27cc2a868c3e79e11b2e3479f31c56e520d3d55c893d5b4274e6346f"
        ),
        state_entries=639,
        clean_entries=636,
        variant="with-middle-attention",
        has_middle_attention=True,
    ),
    "geneb-janusdna-72-wo": ModelSpec(
        runtime_id="geneb-janusdna-72-wo",
        checkpoint_name="72_without_midattn.ckpt",
        checkpoint_size=108361237,
        checkpoint_sha256=(
            "015cd69eb0157eada2ed706677e64274a3cc53c2d76cfad6e1004e065ac033df"
        ),
        config_name="72_without_midattn_model_config.json",
        config_size=1610,
        config_sha256=(
            "0fe6f1e8e54f2a95cf045c511510ffb930d76ac2b93fdc657f7277864a58df9f"
        ),
        state_entries=655,
        clean_entries=652,
        variant="without-middle-attention",
        has_middle_attention=False,
    ),
}


class OracleError(RuntimeError):
    """Raised when pinned source or normalized execution differs."""


def canonical_json(value: Any) -> bytes:
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


def regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{label} must be a nonsymlink regular file: {path}")
    return path


def checked_file(
    path: Path, label: str, expected_size: int, expected_sha256: str
) -> dict[str, Any]:
    regular_file(path, label)
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != expected_size or digest != expected_sha256:
        raise OracleError(f"{label} differs: size={size} sha256={digest}")
    return {"size": size, "sha256": digest}


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = regular_file(path, label).read_bytes()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OracleError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OracleError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} root must be an object")
    return value, payload


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
            "git %s failed: %s"
            % (" ".join(arguments), result.stderr.strip())
        )
    return result.stdout.strip()


def validate_repositories(
    upstream_repo: Path, extractor_repo: Path, patch_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    upstream_repo = upstream_repo.resolve(strict=True)
    extractor_repo = extractor_repo.resolve(strict=True)
    if git_output(upstream_repo, "rev-parse", "HEAD") != UPSTREAM_REVISION:
        raise OracleError("JanusDNA upstream revision differs")
    if git_output(extractor_repo, "rev-parse", "HEAD") != EXTRACTOR_REVISION:
        raise OracleError("GENEB extractor revision differs")
    if git_output(upstream_repo, "status", "--porcelain", "--untracked-files=no"):
        raise OracleError("JanusDNA tracked working tree is dirty")
    if git_output(extractor_repo, "status", "--porcelain", "--untracked-files=no"):
        raise OracleError("GENEB tracked working tree is dirty")
    upstream_manifest = {
        name: checked_file(upstream_repo / name, name, size, digest)
        for name, (size, digest) in sorted(UPSTREAM_FILES.items())
    }
    extractor = (
        extractor_repo / "embedding_pipeline" / "extractors" / "janusdna.py"
    )
    extractor_info = checked_file(
        extractor, "GENEB JanusDNA extractor", 3467, EXTRACTOR_SHA256
    )
    source = extractor.read_bytes()
    if source.count(EXTRACTOR_FRAGMENT) != 1:
        raise OracleError("GENEB extractor normalization site differs")
    normalized = source.replace(
        EXTRACTOR_FRAGMENT, NORMALIZED_EXTRACTOR_FRAGMENT
    )
    if sha256_bytes(normalized) != NORMALIZED_EXTRACTOR_SHA256:
        raise OracleError("normalized GENEB extractor bytes differ")
    patch_info = checked_file(
        patch_path, "JanusDNA GPU-free patch", 540, DEFECT_PATCH_SHA256
    )
    return upstream_manifest, {
        "source": extractor_info,
        "normalized_sha256": NORMALIZED_EXTRACTOR_SHA256,
        "patch": patch_info,
    }


def normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OracleError(f"{label} must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise OracleError(f"{label} must be a normalized relative path")
    return value


def validate_receipt(
    path: Path, spec: ModelSpec
) -> tuple[dict[str, Path], str, dict[str, Any]]:
    receipt, _payload = load_json(path, "JanusDNA source receipt")
    if set(receipt) != {
        "schema_version",
        "kind",
        "model_id",
        "source_kind",
        "source_url",
        "files",
    }:
        raise OracleError("JanusDNA source receipt fields differ")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "geneb-janusdna-manual-source"
        or receipt["model_id"] != spec.runtime_id
        or receipt["source_kind"] != "harvard-dataverse"
        or receipt["source_url"]
        != "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FHDT0RN"
    ):
        raise OracleError("JanusDNA source receipt identity differs")
    expected = {
        spec.checkpoint_name: (spec.checkpoint_size, spec.checkpoint_sha256),
        spec.config_name: (spec.config_size, spec.config_sha256),
    }
    raw_files = receipt["files"]
    if not isinstance(raw_files, list) or len(raw_files) != 2:
        raise OracleError("JanusDNA source receipt file count differs")
    paths: dict[str, Path] = {}
    contract_files: list[dict[str, Any]] = []
    for index, item in enumerate(raw_files):
        label = f"JanusDNA source receipt files[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "name",
            "path",
            "size",
            "sha256",
        }:
            raise OracleError(f"{label} fields differ")
        name = normalized_relative(item["name"], f"{label}.name")
        if name in paths or name not in expected:
            raise OracleError("JanusDNA receipt file set differs")
        size, digest = expected[name]
        if item["size"] != size or item["sha256"] != digest:
            raise OracleError(f"JanusDNA receipt metadata differs for {name}")
        locator = Path(item["path"])
        checked_file(locator, f"JanusDNA receipt locator {name}", size, digest)
        paths[name] = locator.resolve(strict=True)
        contract_files.append({"name": name, "size": size, "sha256": digest})
    if set(paths) != set(expected):
        raise OracleError("JanusDNA receipt file set is incomplete")
    contract = {
        "schema_version": 1,
        "kind": "geneb-janusdna-manual-source",
        "model_id": spec.runtime_id,
        "source_kind": "harvard-dataverse",
        "files": sorted(contract_files, key=lambda item: item["name"]),
    }
    contract_payload = canonical_json(contract)
    return paths, sha256_bytes(contract_payload), contract


def validate_config(path: Path, spec: ModelSpec) -> dict[str, Any]:
    root, _payload = load_json(path, "JanusDNA Dataverse config")
    if set(root) != {"config"} or not isinstance(root["config"], dict):
        raise OracleError("JanusDNA Dataverse config schema differs")
    config = dict(root["config"])
    expected = {
        "_target_": "janusdna.configuration_janusdna.JanusDNAConfig",
        "hidden_size": 72,
        "num_hidden_layers": 8,
        "vocab_size": 12,
        "num_attention_heads": 4,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "intermediate_size": 288,
        "rms_norm_eps": 1.0e-6,
        "attn_layer_offset": 4 if spec.has_middle_attention else 100,
        "final_attention": True,
        "final_attention_class": "flex_attention",
        "flex_attn_n_embd": 128,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise OracleError(f"JanusDNA config field differs: {key}")
    ssm = config.get("ssm_cfg")
    if not isinstance(ssm, dict) or (
        ssm.get("d_state"),
        ssm.get("d_conv"),
        ssm.get("expand"),
        ssm.get("dt_rank"),
        ssm.get("conv_bias"),
        ssm.get("bias"),
    ) != (16, 4, 2, "auto", True, False):
        raise OracleError("JanusDNA Mamba config differs")
    return config


def read_single_fasta(path: Path) -> tuple[str, str, str]:
    payload = regular_file(path, "canonical JanusDNA FASTA").read_bytes()
    try:
        text = payload.decode("ascii")
    except UnicodeError as error:
        raise OracleError("canonical JanusDNA FASTA must be ASCII") from error
    lines = text.splitlines()
    if len(lines) != 2 or not lines[0].startswith(">") or not lines[0][1:]:
        raise OracleError("canonical JanusDNA FASTA must contain one record")
    sequence = lines[1]
    if not sequence or any(base not in "ACGTNacgtn" for base in sequence):
        raise OracleError("canonical JanusDNA sequence must use A/C/G/T/N")
    return lines[0][1:], sequence, sha256_bytes(payload)


def package_lock() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", "freeze"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise OracleError(f"cannot freeze oracle environment: {result.stderr.strip()}")
    packages = sorted(line for line in result.stdout.splitlines() if line)
    expected = {
        "numpy": "1.26.0",
        "torch": "2.5.0",
        "transformers": "4.49.0",
        "tokenizers": "0.21.0",
        "safetensors": "0.5.2",
        "omegaconf": "2.3.0",
    }
    actual: dict[str, str] = {}
    for item in packages:
        if "==" in item:
            name, version = item.split("==", 1)
            actual[name.lower()] = version
    if any(actual.get(name) != version for name, version in expected.items()):
        raise OracleError(f"JanusDNA oracle package lock differs: {actual}")
    return packages


def install_cpu_fallbacks(module: Any, torch: Any) -> dict[str, Any]:
    functional = torch.nn.functional

    class CpuCausalAttention(module.JanusDNAAttention):
        def forward(
            self,
            hidden_states: Any,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_value: Any = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Any = None,
        ) -> tuple[Any, Any, Any]:
            del position_ids, cache_position
            if past_key_value is not None or use_cache or output_attentions:
                raise OracleError("GPU-free middle attention only supports embedding")
            if attention_mask is None or attention_mask.ndim != 2:
                raise OracleError("GPU-free middle attention mask differs")
            batch, rows, _width = hidden_states.shape
            if tuple(attention_mask.shape) != (batch, rows):
                raise OracleError("GPU-free middle attention dimensions differ")
            query = self.q_proj(hidden_states).view(
                batch, rows, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key = self.k_proj(hidden_states).view(
                batch, rows, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value = self.v_proj(hidden_states).view(
                batch, rows, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            key = module.repeat_kv(key, self.num_key_value_groups)
            value = module.repeat_kv(value, self.num_key_value_groups)
            score = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            valid = attention_mask.to(dtype=torch.bool)
            causal = torch.ones(
                (rows, rows), dtype=torch.bool, device=hidden_states.device
            ).tril()
            allowed = (
                causal[None, None, :, :]
                & valid[:, None, :, None]
                & valid[:, None, None, :]
            )
            score = score.masked_fill(~allowed, torch.finfo(score.dtype).min)
            weights = functional.softmax(score, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            weights = weights * valid[:, None, :, None].to(weights.dtype)
            attended = torch.matmul(weights, value)
            attended = attended.transpose(1, 2).contiguous().reshape(
                batch, rows, self.hidden_size
            )
            return self.o_proj(attended), None, past_key_value

    def passthrough_mask(
        self: Any, attention_mask: Any, input_tensor: Any, cache_position: Any
    ) -> Any:
        del self, input_tensor, cache_position
        return attention_mask

    def dense_flex_forward(
        self: Any,
        hidden_states: Any,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        del attention_mask, position_ids, cache_position, kwargs
        if past_key_value is not None or use_cache or output_attentions:
            raise OracleError("GPU-free final attention only supports embedding")
        batch, rows, _width = hidden_states.shape
        if rows % 2 != 0:
            raise OracleError("GPU-free final attention row count differs")
        query = self.q_proj(hidden_states).view(
            batch, rows, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, rows, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, rows, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        key = module.repeat_kv(key, self.num_key_value_groups)
        value = module.repeat_kv(value, self.num_key_value_groups)
        padded_head = self.config.flex_attn_n_embd // self.num_heads
        if padded_head < self.head_dim or padded_head != 32:
            raise OracleError("GPU-free FlexAttention padded head differs")
        if padded_head != self.head_dim:
            padding = (0, padded_head - self.head_dim)
            query = functional.pad(query, padding)
            key = functional.pad(key, padding)
            value = functional.pad(value, padding)
        original_rows = rows // 2
        query_index = torch.arange(rows, device=hidden_states.device)[:, None]
        key_index = torch.arange(rows, device=hidden_states.device)[None, :]
        first = (
            (key_index < original_rows)
            & (query_index < original_rows)
            & (query_index >= key_index)
        )
        second = (
            (key_index >= original_rows)
            & (query_index < original_rows)
            & (key_index >= original_rows + query_index + 2)
        )
        third = (
            (key_index < original_rows)
            & (query_index >= original_rows)
            & (query_index >= key_index + original_rows + 2)
        )
        fourth = (
            (key_index >= original_rows)
            & (query_index >= original_rows)
            & (query_index <= key_index)
        )
        allowed = first | second | third | fourth
        score = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(padded_head)
        score = score.masked_fill(
            ~allowed[None, None, :, :], torch.finfo(score.dtype).min
        )
        weights = functional.softmax(score, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attended = torch.matmul(weights, value)[..., : self.head_dim]
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch, rows, self.hidden_size
        )
        return self.o_projs[0](attended), None, past_key_value

    module.JANUSDNA_ATTENTION_CLASSES["eager"] = CpuCausalAttention
    module.JanusDNAModel._update_causal_mask = passthrough_mask
    module.JanusDNAFlexAttention.forward = dense_flex_forward
    contract = {
        "middle_attention": {
            "implementation": "pure-pytorch-dense-causal",
            "query_mask": "right-padding-original-mask",
            "key_mask": "right-padding-original-mask",
            "scale": "one-over-sqrt-18",
            "softmax_dtype": "float32",
        },
        "final_attention": {
            "implementation": "pure-pytorch-dense-four-condition-mask",
            "padding_mask": "ignored-by-pinned-upstream",
            "qkv_head_padding": "18-to-32-zero-pad",
            "scale": "one-over-sqrt-32",
            "softmax_dtype": "float32",
        },
        "native_runtime_imported": False,
    }
    return {**contract, "contract_sha256": sha256_bytes(canonical_json(contract))}


def clean_state_dict(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise OracleError("JanusDNA Lightning state_dict is not an object")
    cleaned: dict[str, Any] = {}
    for raw_name, value in state.items():
        if raw_name.startswith(
            ("train_torchmetrics.", "val_torchmetrics.", "test_torchmetrics.")
        ):
            continue
        name = raw_name
        if name.startswith("model.model."):
            name = name[len("model.model.") :]
        elif name.startswith("model."):
            name = name[len("model.") :]
        if name.startswith("module."):
            name = name[len("module.") :]
        if name in cleaned:
            raise OracleError(f"JanusDNA cleaned state name collides: {name}")
        cleaned[name] = value
    return cleaned


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
    spec = MODEL_SPECS[args.model]
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
    upstream_manifest, extractor_manifest = validate_repositories(
        args.upstream_repo, args.extractor_repo, args.defect_patch
    )
    source_paths, receipt_contract_sha256, receipt_contract = validate_receipt(
        args.source_receipt, spec
    )
    config = validate_config(source_paths[spec.config_name], spec)
    record_name, sequence, input_sha256 = read_single_fasta(args.input)
    packages = package_lock()

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    if torch.__version__ != "2.5.0" or transformers.__version__ != "4.49.0":
        raise OracleError("JanusDNA torch/Transformers versions differ")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)

    upstream_repo = args.upstream_repo.resolve(strict=True)
    sys.path.insert(0, str(upstream_repo))
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    configuration = importlib.import_module("janusdna.configuration_janusdna")
    modeling = importlib.import_module("janusdna.modeling_janusdna")
    tokenizer_path = upstream_repo / "caduceus/tokenization_caduceus.py"
    tokenizer_spec = importlib.util.spec_from_file_location(
        "janusdna_pinned_tokenization_caduceus", tokenizer_path
    )
    if tokenizer_spec is None or tokenizer_spec.loader is None:
        raise OracleError("cannot load pinned JanusDNA tokenizer module")
    tokenization = importlib.util.module_from_spec(tokenizer_spec)
    tokenizer_spec.loader.exec_module(tokenization)
    expected_modules = {
        configuration: upstream_repo / "janusdna/configuration_janusdna.py",
        modeling: upstream_repo / "janusdna/modeling_janusdna.py",
        tokenization: tokenizer_path,
    }
    for module, expected_path in expected_modules.items():
        if Path(module.__file__).resolve() != expected_path.resolve():
            raise OracleError("JanusDNA module did not load from pinned source")

    raw_config = dict(config)
    raw_config.pop("_target_")
    raw_config["bidirectional"] = (
        str(raw_config["bidirectional"]).strip().lower().strip(",") == "true"
    )
    clean_config = dict(raw_config)
    clean_config["attn_implementation"] = "eager"
    clean_mixer = modeling.JanusDNAMambaMixer(
        configuration.JanusDNAConfig(**clean_config), layer_idx=0
    )
    clean_error: ValueError | None = None
    try:
        clean_mixer(torch.zeros((1, 1, 72), dtype=torch.float32))
    except ValueError as error:
        clean_error = error
    if clean_error is None or str(clean_error) != CLEAN_ERROR:
        raise OracleError("pinned JanusDNA clean CPU fast-kernel defect differs")
    if sha256_bytes(str(clean_error).encode("utf-8")) != CLEAN_ERROR_SHA256:
        raise OracleError("pinned JanusDNA clean CPU error digest differs")
    del clean_mixer

    fallback_contract = install_cpu_fallbacks(modeling, torch)
    normalized_config = dict(raw_config)
    normalized_config["use_mamba_kernels"] = False
    normalized_config["attn_implementation"] = "eager"
    model_config = configuration.JanusDNAConfig(**normalized_config)
    torch.manual_seed(0)
    model = modeling.JanusDNAModel(model_config).to("cpu").eval()
    checkpoint = torch.load(
        str(source_paths[spec.checkpoint_name]),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise OracleError("JanusDNA Lightning checkpoint root differs")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or len(state) != spec.state_entries:
        raise OracleError("JanusDNA Lightning state entry count differs")
    cleaned = clean_state_dict(state)
    if len(cleaned) != spec.clean_entries:
        raise OracleError("JanusDNA cleaned state entry count differs")
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    expected_missing = [
        "final_mlp.0.weight",
        "final_mlp.0.bias",
        "final_mlp.2.weight",
        "final_mlp.2.bias",
    ]
    if list(missing) != expected_missing or list(unexpected) != ["lm_head.weight"]:
        raise OracleError(
            f"JanusDNA load manifest differs: missing={missing} unexpected={unexpected}"
        )
    model.final_mlp = torch.nn.Identity()
    if (
        model.config.vocab_size != 16
        or model.config.hidden_size != 72
        or model.config.num_hidden_layers != 8
        or model.config.use_mamba_kernels is not False
    ):
        raise OracleError("loaded JanusDNA normalized topology differs")

    tokenizer = tokenization.CaduceusTokenizer(
        model_max_length=1024, padding_side="right"
    )
    encoded = tokenizer(
        [sequence],
        return_tensors="pt",
        padding="max_length",
        max_length=1024,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded["attention_mask"].to("cpu")
    expected_token_ids = [
        {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}[base]
        for base in sequence.upper()[:1024]
    ]
    effective_tokens = len(expected_token_ids)
    if (
        tuple(input_ids.shape) != (1, 1024)
        or tuple(attention_mask.shape) != (1, 1024)
        or input_ids[0, :effective_tokens].tolist() != expected_token_ids
        or input_ids[0, effective_tokens:].tolist()
        != [4] * (1024 - effective_tokens)
        or attention_mask[0, :effective_tokens].tolist()
        != [1] * effective_tokens
        or attention_mask[0, effective_tokens:].tolist()
        != [0] * (1024 - effective_tokens)
    ):
        raise OracleError("pinned JanusDNA tokenizer output differs")

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = pooled.to(dtype=torch.float32, device="cpu")
    if tuple(hidden.shape) != (1, 1024, 72) or tuple(pooled.shape) != (1, 72):
        raise OracleError("JanusDNA upstream output shape differs")
    values = [float(value) for value in pooled[0].tolist()]
    if not all(math.isfinite(value) for value in values):
        raise OracleError("JanusDNA upstream oracle contains non-finite values")

    generator_sha256 = sha256_file(Path(__file__).resolve())
    environment_lock = {
        "schema_version": 1,
        "oracle_contract": "geneb-independent-oracle-v1",
        "packages": packages,
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
        },
        "upstream_source_files": upstream_manifest,
        "fallback_contract": fallback_contract,
    }
    provenance = {
        "kind": "pinned-upstream-normalized-gpu-free",
        "oracle_contract": "geneb-independent-oracle-v1",
        "independent_of_evo_native_runtime": True,
        "clean_geneb_reference": False,
        "clean_upstream_status": "broken-cpu-fast-kernel",
        "upstream_repo": "Qihao-Duan/JanusDNA",
        "upstream_revision": UPSTREAM_REVISION,
        "extractor_repo": "ultimativity/GENEB",
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_sha256": EXTRACTOR_SHA256,
        "normalization_patch": {
            "path": args.defect_patch.name,
            "sha256": DEFECT_PATCH_SHA256,
            "normalized_extractor_sha256": NORMALIZED_EXTRACTOR_SHA256,
            "operation": "set cfg.use_mamba_kernels=false",
        },
        "clean_forward_defect": {
            "type": type(clean_error).__name__,
            "message": str(clean_error),
            "message_sha256": CLEAN_ERROR_SHA256,
        },
        "gpu_free_attention_fallback": fallback_contract,
        "checkpoint_name": spec.checkpoint_name,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "config_name": spec.config_name,
        "config_sha256": spec.config_sha256,
        "source_receipt_contract_sha256": receipt_contract_sha256,
        "source_receipt_contract": receipt_contract,
        "record_name": record_name,
        "variant": spec.variant,
        "batch_size": 1,
        "input_tokens": 1024,
        "effective_tokens": effective_tokens,
        "padding_side": "right",
        "padding_tokens": 1024 - effective_tokens,
        "special_tokens": "none",
        "hidden_tap": "twice-post-final-rmsnorm-after-identity-final-mlp-residual",
        "pooling": "attention-mask-mean-direct-f32-division",
        "output_width": 72,
        "generator_sha256": generator_sha256,
    }
    return {
        "schema_version": 1,
        "kind": "geneb-independent-oracle-vector",
        "runtime_id": spec.runtime_id,
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
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--upstream-repo", required=True, type=Path)
    parser.add_argument("--extractor-repo", required=True, type=Path)
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
                    "model": oracle["runtime_id"],
                    "output": str(args.output),
                    "sha256": sha256_bytes(payload),
                    "values": len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "l2_norm": math.sqrt(sum(value * value for value in values)),
                    "normalization_patch_sha256": DEFECT_PATCH_SHA256,
                    "clean_reference": False,
                },
                sort_keys=True,
            )
        )
    except (ImportError, OSError, OracleError, RuntimeError, ValueError) as error:
        print(f"generate_geneb_janusdna_upstream_oracle: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
