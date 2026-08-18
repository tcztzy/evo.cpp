#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate two pinned OmniNA-220M GENEB oracles with official HF code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit


RUNTIME_ID = "geneb-omnina-220m"
SOURCE_REPO = "XLS/OmniNA-220m"
REQUESTED_REVISION = "main"
SOURCE_REVISION = "64ea6ce7b250fc611773215ddcdd1ecca232de67"
CATALOG_CONTRACT_SHA256 = (
    "f31580575bbfeca92702c21763c2fbd7cbd3d6146a35a958a53d74083b1342bc"
)
EXTRACTOR_REPO = "ultimativity/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "201d5dd397959599260dda9cf71c9343fbaf330db59521ccb2714c3422782462"
)
EXTRACTOR_PYPROJECT_SHA256 = (
    "616a15c0a751eb14687fefe691cd624fbf4cc104605f167a5782cb29d16060e0"
)
EXTRACTOR_UV_LOCK_SHA256 = (
    "e1fce57ba7eaf9fddebf300161c2840b6e6722bb1c82ac3a5b4e8e093a699953"
)
EXPECTED_SOURCE_FILES = {
    ".gitattributes": (
        1519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "README.md": (
        31,
        "4bcf87ecfbbb8e07a01b21415a970c8b53a5283bf6872b657040d3f45c9241f7",
    ),
    "added_tokens.json": (
        21,
        "75e3ca5df2973756aa612cb17246ef6020a68ff8d94671508987d373642f7a36",
    ),
    "config.json": (
        535,
        "201dfb2a49b3fecd72d9659faa67133e61fe212bc110f743ae5b5b61340ef9fe",
    ),
    "pytorch_model.bin": (
        1336083259,
        "18aab85bffe62f35d55a64a4794716b4977afeefdec5330799223da33342be19",
    ),
    "tokenizer.json": (
        2070715,
        "a067717a733809467fa5f47d9a6cdc2c69a019a81353f03d00cf4f2ee00d93d7",
    ),
}
EXPECTED_CONFIG = {
    "_name_or_path": "llama-220m/",
    "architectures": ["LlamaForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 16,
    "num_hidden_layers": 16,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 32001,
}
INPUTS = (
    {
        "label": "omnina_input_0",
        "sequence": "ACGTNACGTNACGTN",
        "input_ids": [1, 2724, 3566, 30817, 12, 3566, 30817, 12, 30864],
    },
    {
        "label": "omnina_input_1",
        "sequence": "ACGTNNNNNNNNNN",
        "input_ids": [
            1,
            2724,
            30864,
            30864,
            30864,
            30864,
            30864,
            30864,
            30864,
            30864,
            30864,
            30864,
        ],
    },
)
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
MODEL_MAX_LENGTH_SENTINEL = 1000000000000000019884624838656
BLOCK0_INPUT_INDEX = 1
BLOCK0_OPERATOR_RAW_F32_SHA256 = {
    "attention_norm": "752e326082f040a11c074cc6fd1394d029f1bad1298e0624e2902dbd922428b0",
    "query_linear": "e486f5bcb796ee3c8070855c19ced46da40130090a1a7b8b6fffaac7c196c859",
    "key_linear": "581567ccefab31a580b2168bfdc7ef74a3533a20a6ac8c7c3a01e756ab037af1",
    "value_linear": "a62839371e0ee6b2fc59097688bde133c827ab42bd650f03e1824121d8ede331",
    "query_rope": "66b586fa6154687dee90ac6239338ae2f5484ca2fed0ecc758ad223137fa0cbf",
    "key_rope": "c149e442cab0a2e5267f79ed7a5f92ac3685a0c19099bdc400b40b4d1249f598",
    "attended": "ed5076fdbd22979c1deaaefb4168345f445d6efd2e97d14bd6d1fd2173f2b03f",
    "attention_projected": "9354f7b9d9bbd339df980985188ee78ba9847c65bc1a8c14b5d2c4e301fd6620",
    "attention_residual": "4867ad6c1a49636f5b83f11313f21c033092bb205816af33202b2d88b5d8af47",
    "ffn_norm": "dabddfe94146228a8453792d7206092626eee108f3b32767a1ee594a04dfec95",
    "up_linear": "16d9447a9f8beb8b6457c7ee7f65bcc13403a78f554c8cdd65c64e638016e7e8",
    "gate_linear": "b357a7ab5430c125c6b1c709e772a395a20a0626310a070fe1b1aee8db0f9124",
    "swiglu": "e13c90d38214fc39b4a37b50f7bd106f5514460727f122c6a1ef9b14c7a7a61c",
    "mlp_projected": "a696cb7270d703217b00221c8d69497bc9ba610002c50761fd519ae67a143524",
    "block_output": "4430591ab067fc57a2871fd88124f8cedc4e5fc09de666f22b00d648ba741925",
}
BLOCK0_INNER_OPERATORS = {"up_linear", "gate_linear", "swiglu"}


class OracleError(RuntimeError):
    """Raised when pinned upstream oracle semantics differ."""


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
        raise OracleError(f"{label} root must be an object")
    return value


def normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OracleError(f"{label} must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise OracleError(f"{label} must be a normalized relative path")
    return value


def portable(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            portable(key, label + " key")
            portable(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            portable(item, f"{label}[{index}]")
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


def validate_sources(
    receipt_path: Path, snapshot: Path
) -> dict[str, dict[str, Any]]:
    receipt = load_json(receipt_path, "source receipt")
    expected_fields = {
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
    }
    if set(receipt) != expected_fields:
        raise OracleError("source receipt fields differ")
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != RUNTIME_ID
        or receipt["repo"] != SOURCE_REPO
        or receipt["requested_revision"] != REQUESTED_REVISION
        or receipt["resolved_revision"] != SOURCE_REVISION
        or receipt["source_kind"] != "huggingface"
        or receipt["catalog_contract_sha256"] != CATALOG_CONTRACT_SHA256
        or receipt["load_path"] is not None
    ):
        raise OracleError("source receipt pinned provenance differs")
    if snapshot.name != SOURCE_REVISION or not snapshot.is_dir():
        raise OracleError("snapshot path is not the pinned resolved revision")
    raw_files = receipt["files"]
    if not isinstance(raw_files, list) or len(raw_files) != len(EXPECTED_SOURCE_FILES):
        raise OracleError("source receipt file count differs")
    verified: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_files):
        label = f"source receipt files[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "name",
            "path",
            "size",
            "sha256",
        }:
            raise OracleError(f"{label} fields differ")
        name = normalized_relative(item["name"], label + ".name")
        if name in verified or name not in EXPECTED_SOURCE_FILES:
            raise OracleError("source receipt file set is duplicated or unexpected")
        expected_size, expected_sha256 = EXPECTED_SOURCE_FILES[name]
        if item["size"] != expected_size or item["sha256"] != expected_sha256:
            raise OracleError(f"source receipt metadata differs for {name}")
        locator = Path(item["path"])
        logical = snapshot / name
        for candidate, candidate_label in (
            (locator, "receipt locator"),
            (logical, "snapshot file"),
        ):
            if not candidate.is_file():
                raise OracleError(f"{candidate_label} is missing for {name}")
            if (
                candidate.stat().st_size != expected_size
                or sha256_file(candidate) != expected_sha256
            ):
                raise OracleError(f"{candidate_label} differs for {name}")
        if logical.resolve(strict=True) != locator.resolve(strict=True):
            raise OracleError(f"snapshot file does not resolve to receipt blob: {name}")
        verified[name] = {"size": expected_size, "sha256": expected_sha256}
    snapshot_names = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if snapshot_names != set(EXPECTED_SOURCE_FILES):
        raise OracleError("snapshot contains missing or extra regular files")
    return dict(sorted(verified.items()))


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


def validate_extractor(repository: Path, extractor: Path) -> dict[str, Any]:
    if git_output(repository, "rev-parse", "HEAD") != EXTRACTOR_REVISION:
        raise OracleError("GENEB extractor checkout revision differs")
    root = repository.resolve(strict=True)
    relative = extractor.resolve(strict=True).relative_to(root)
    committed = subprocess.run(
        ["git", "-C", str(repository), "show", f"HEAD:{relative.as_posix()}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = extractor.read_bytes()
    if (
        committed.returncode != 0
        or payload != committed.stdout
        or sha256_bytes(payload) != EXTRACTOR_SHA256
    ):
        raise OracleError("GENEB OmniNA extractor differs from its pinned commit")
    pyproject = root / "embedding_pipeline" / "pyproject.toml"
    uv_lock = root / "embedding_pipeline" / "uv.lock"
    if sha256_file(pyproject) != EXTRACTOR_PYPROJECT_SHA256:
        raise OracleError("GENEB extractor pyproject differs")
    if sha256_file(uv_lock) != EXTRACTOR_UV_LOCK_SHA256:
        raise OracleError("GENEB extractor uv.lock differs")
    required_fragments = (
        b"self.tokenizer = AutoTokenizer.from_pretrained(name_model)",
        b"self.tokenizer.pad_token = self.tokenizer.eos_token",
        b"self.model = AutoModel.from_pretrained(",
        b"output_hidden_states=True",
        b"padding=True,",
        b"truncation=True",
        b"out = self.model(**enc)",
        b"hidden = out.hidden_states[-1]",
        b"embs = (summed / lengths).cpu().numpy()",
    )
    if any(payload.count(fragment) != 1 for fragment in required_fragments):
        raise OracleError("GENEB OmniNA extractor call/pooling contract differs")
    return {
        "revision": EXTRACTOR_REVISION,
        "source_sha256": EXTRACTOR_SHA256,
        "pyproject_sha256": EXTRACTOR_PYPROJECT_SHA256,
        "uv_lock_sha256": EXTRACTOR_UV_LOCK_SHA256,
    }


def read_input(path: Path, expected: dict[str, Any]) -> tuple[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise OracleError(f"cannot read oracle FASTA: {error}") from error
    if len(lines) < 2 or not lines[0].startswith(">") or not lines[0][1:]:
        raise OracleError("oracle input must be one named FASTA record")
    if any(line.startswith(">") for line in lines[1:]):
        raise OracleError("oracle input must contain exactly one FASTA record")
    label = lines[0][1:]
    sequence = "".join(lines[1:])
    if label != expected["label"] or sequence != expected["sequence"]:
        raise OracleError("OmniNA oracle FASTA label/sequence differs")
    return label, sequence


def validate_packages() -> list[str]:
    effective: list[str] = []
    for name, expected in sorted(REQUIRED_PACKAGE_VERSIONS.items()):
        actual = importlib.metadata.version(name)
        if name == "torch":
            actual = actual.split("+")[0]
        if actual != expected:
            raise OracleError(
                f"locked package {name} differs: expected {expected}, got {actual}"
            )
        effective.append(f"{name}=={expected}")
    return effective


def prepare_output(output: Path, force: bool) -> None:
    expected_names = {
        "input-0.independent-oracle-vector.json",
        "input-0.attention-mask-mean.f32.npy",
        "input-1.independent-oracle-vector.json",
        "input-1.attention-mask-mean.f32.npy",
        "oracle-report.json",
    }
    expected_names.update(
        "input-1.block0-%s.f32.npy" % name
        for name in BLOCK0_OPERATOR_RAW_F32_SHA256
    )
    if output.exists():
        if not output.is_dir():
            raise OracleError("oracle output exists and is not a directory")
        names = {path.name for path in output.iterdir()}
        if names and (not force or not names.issubset(expected_names)):
            raise OracleError("oracle output directory must be absent or empty")
        if force:
            for name in names:
                (output / name).unlink()
    else:
        output.mkdir(parents=True)


def capture_block0_operators(
    model: Any, encoded: dict[str, Any], torch: Any, modeling_llama: Any
) -> dict[str, Any]:
    """Capture official layer-0 operator boundaries without native code."""

    captured: dict[str, Any] = {}

    def record(name: str, tensor: Any) -> None:
        if name in captured:
            raise OracleError(f"block0 operator {name!r} was captured twice")
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise OracleError(f"block0 operator {name!r} is not an F32 tensor")
        captured[name] = tensor.detach().cpu().contiguous().clone()

    def output_hook(name: str) -> Any:
        def hook(_module: Any, _inputs: Any, output_value: Any) -> None:
            record(name, output_value)

        return hook

    def input_hook(name: str) -> Any:
        def hook(_module: Any, inputs: Any) -> None:
            if not isinstance(inputs, tuple) or len(inputs) != 1:
                raise OracleError(f"block0 operator {name!r} input differs")
            record(name, inputs[0])

        return hook

    layer = model.layers[0]
    hooks = [
        layer.input_layernorm.register_forward_hook(output_hook("attention_norm")),
        layer.self_attn.q_proj.register_forward_hook(output_hook("query_linear")),
        layer.self_attn.k_proj.register_forward_hook(output_hook("key_linear")),
        layer.self_attn.v_proj.register_forward_hook(output_hook("value_linear")),
        layer.self_attn.o_proj.register_forward_pre_hook(input_hook("attended")),
        layer.self_attn.o_proj.register_forward_hook(
            output_hook("attention_projected")
        ),
        layer.post_attention_layernorm.register_forward_pre_hook(
            input_hook("attention_residual")
        ),
        layer.post_attention_layernorm.register_forward_hook(output_hook("ffn_norm")),
        layer.mlp.up_proj.register_forward_hook(output_hook("up_linear")),
        layer.mlp.gate_proj.register_forward_hook(output_hook("gate_linear")),
        layer.mlp.down_proj.register_forward_pre_hook(input_hook("swiglu")),
        layer.mlp.down_proj.register_forward_hook(output_hook("mlp_projected")),
        layer.register_forward_hook(output_hook("block_output")),
    ]
    original_apply_rope = modeling_llama.apply_rotary_pos_emb
    rope_calls = 0

    def capture_apply_rope(*args: Any, **kwargs: Any) -> Any:
        nonlocal rope_calls
        result = original_apply_rope(*args, **kwargs)
        if rope_calls == 0:
            if not isinstance(result, tuple) or len(result) != 2:
                raise OracleError("official apply_rotary_pos_emb output differs")
            record("query_rope", result[0].transpose(1, 2).contiguous())
            record("key_rope", result[1].transpose(1, 2).contiguous())
        rope_calls += 1
        return result

    modeling_llama.apply_rotary_pos_emb = capture_apply_rope
    try:
        with torch.inference_mode():
            model(**encoded)
    finally:
        modeling_llama.apply_rotary_pos_emb = original_apply_rope
        for handle in hooks:
            handle.remove()
    if rope_calls != 16 or set(captured) != set(BLOCK0_OPERATOR_RAW_F32_SHA256):
        raise OracleError(
            "official block0 operator capture set differs: rope_calls=%d names=%s"
            % (rope_calls, sorted(captured))
        )
    rows = int(encoded["input_ids"].shape[1])
    for name, tensor in captured.items():
        width = 4096 if name in BLOCK0_INNER_OPERATORS else 1024
        expected_shape = (
            (1, rows, 16, 64)
            if name in {"query_rope", "key_rope"}
            else (1, rows, width)
        )
        if tuple(tensor.shape) != expected_shape:
            raise OracleError(f"block0 operator {name!r} shape differs")
        raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        if sha256_bytes(raw) != BLOCK0_OPERATOR_RAW_F32_SHA256[name]:
            raise OracleError(f"block0 operator {name!r} raw F32 digest differs")
    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--extractor-repo", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--input-0", required=True, type=Path)
    parser.add_argument("--input-1", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

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
    receipt = args.source_receipt.resolve(strict=True)
    extractor_repo = args.extractor_repo.resolve(strict=True)
    extractor_path = args.extractor.resolve(strict=True)
    input_paths = [args.input_0.resolve(strict=True), args.input_1.resolve(strict=True)]
    output = args.output_dir.resolve()
    prepare_output(output, args.force)
    source_files = validate_sources(receipt, snapshot)
    extractor = validate_extractor(extractor_repo, extractor_path)
    input_records = [
        read_input(path, expected)
        for path, expected in zip(input_paths, INPUTS, strict=True)
    ]
    if load_json(snapshot / "config.json", "model config") != EXPECTED_CONFIG:
        raise OracleError("pinned OmniNA model config differs")

    import numpy as np
    import torch
    import tokenizers
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from transformers.models.llama import modeling_llama

    packages = validate_packages()
    if (
        transformers.__version__ != "4.57.3"
        or tokenizers.__version__ != "0.22.2"
        or torch.__version__.split("+")[0] != "2.7.1"
        or np.__version__ != "1.26.4"
    ):
        raise OracleError("loaded locked package versions differ")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if (
        tokenizer.__class__.__module__
        != "transformers.models.llama.tokenization_llama_fast"
        or tokenizer.__class__.__name__ != "LlamaTokenizerFast"
        or tokenizer.model_input_names != ["input_ids", "attention_mask"]
        or tokenizer.padding_side != "left"
        or tokenizer.truncation_side != "right"
        or tokenizer.model_max_length != MODEL_MAX_LENGTH_SENTINEL
        or tokenizer.pad_token is not None
        or tokenizer.pad_token_id is not None
        or tokenizer.unk_token_id != 0
        or tokenizer.bos_token_id != 1
        or tokenizer.eos_token_id != 2
        or tokenizer.vocab_size != 32000
        or len(tokenizer) != 32001
    ):
        raise OracleError("official locked tokenizer wrapper contract differs")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token != "</s>" or tokenizer.pad_token_id != 2:
        raise OracleError("GENEB EOS-as-pad mutation differs")

    model, loading = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        output_hidden_states=True,
        output_loading_info=True,
    )
    if (
        loading.get("missing_keys")
        or sorted(loading.get("unexpected_keys", [])) != ["lm_head.weight"]
        or loading.get("mismatched_keys")
        or loading.get("error_msgs")
    ):
        raise OracleError(f"official checkpoint loading info differs: {loading!r}")
    if (
        model.__class__.__module__
        != "transformers.models.llama.modeling_llama"
        or model.__class__.__name__ != "LlamaModel"
        or not model.config.output_hidden_states
        or len(model.layers) != 16
        or model.embed_tokens.num_embeddings != 32001
        or model.embed_tokens.embedding_dim != 1024
        or model.config._attn_implementation != "sdpa"
    ):
        raise OracleError("official standard LlamaModel topology differs")
    model.to(device="cpu", dtype=torch.float32).eval()

    source_manifest = [
        {"name": name, "size": item["size"], "sha256": item["sha256"]}
        for name, item in source_files.items()
    ]
    environment = {
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
        "packages": packages,
        "geneb_uv_lock_sha256": EXTRACTOR_UV_LOCK_SHA256,
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
            "tokenizers_version": tokenizers.__version__,
            "local_files_only": True,
            "trust_remote_code": False,
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "tokenizer_class": (
                f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
            ),
            "model_input_names": tokenizer.model_input_names,
            "attention_implementation": model.config._attn_implementation,
        },
        "source_files": source_files,
        "model_semantics": {
            "hidden_size": 1024,
            "layers": 16,
            "attention_heads": 16,
            "intermediate_size": 4096,
            "declared_max_tokens": 2048,
            "reference_max_tokens": None,
            "norm": "rmsnorm",
            "rope_layout": "split-half",
            "hidden_tap": "outputs.hidden_states[-1]",
        },
    }
    generator = Path(__file__).resolve(strict=True)
    base_provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-reference",
        "oracle_execution_call": "exact-reference",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_file_manifest_sha256": sha256_bytes(canonical_json(source_manifest)),
        "checkpoint_sha256": EXPECTED_SOURCE_FILES["pytorch_model.bin"][1],
        "official_standard_transformers_code": True,
        "trust_remote_code": False,
        "independent_of_evo_native_runtime": True,
        "generator_sha256": sha256_file(generator),
        "extractor_repo": EXTRACTOR_REPO,
        "extractor_revision": extractor["revision"],
        "extractor_module": "embedding_pipeline/extractors/omnina.py",
        "extractor_class": "OmniNAExtractor",
        "extractor_sha256": extractor["source_sha256"],
        "extractor_pyproject_sha256": extractor["pyproject_sha256"],
        "extractor_uv_lock_sha256": extractor["uv_lock_sha256"],
        "model_entrypoint": "AutoModel.from_pretrained(..., output_hidden_states=True)",
        "attention_implementation": "sdpa (PyTorch 2.7.1 CPU Flash)",
        "hidden_tap": "outputs.hidden_states[-1]",
        "pooling": "attention-mask-mean-direct-f32-division",
        "special_tokens": "include",
        "padding_side": "left",
        "pad_token_id": 2,
        "pad_token_source": "eos_token",
        "reference_truncation": "truncation=True with unbounded tokenizer sentinel; no-op",
        "reference_max_tokens": None,
        "normalized_max_tokens": 2048,
        "parity_scope": "fixed short raw A/C/G/T/N inputs; CPU F32; batch size 1",
        "non_claims": [
            "general Unicode tokenizer parity",
            "inputs exceeding 2048 normalized tokens",
            "batched reference parity",
        ],
        "historical_version_differential": {
            "checkpoint_era": {
                "transformers": "4.29.2",
                "tokenizers": "0.13.3",
                "model_input_names": [
                    "input_ids",
                    "token_type_ids",
                    "attention_mask",
                ],
                "clean_call": "TypeError: LlamaModel.forward() got an unexpected keyword argument 'token_type_ids'",
            },
            "geneb_uv_lock": {
                "transformers": "4.57.3",
                "tokenizers": "0.22.2",
                "model_input_names": ["input_ids", "attention_mask"],
                "clean_call": "passed",
            },
            "patch_operations": [],
        },
    }

    report_records: list[dict[str, Any]] = []
    block0_operator_records: list[dict[str, Any]] = []
    for index, ((label, sequence), expected, input_path) in enumerate(
        zip(input_records, INPUTS, input_paths, strict=True)
    ):
        encoded = tokenizer(
            [sequence], return_tensors="pt", padding=True, truncation=True
        )
        if list(encoded.keys()) != ["input_ids", "attention_mask"]:
            raise OracleError("clean locked encoding keys differ")
        expected_ids = expected["input_ids"]
        if (
            encoded["input_ids"].tolist() != [expected_ids]
            or encoded["attention_mask"].tolist() != [[1] * len(expected_ids)]
        ):
            raise OracleError(f"official locked token IDs differ for input {index}")
        with torch.inference_mode():
            outputs = model(**encoded)
            if outputs.hidden_states is None or len(outputs.hidden_states) != 17:
                raise OracleError("official hidden-state tap count differs")
            hidden = outputs.hidden_states[-1]
            if (
                tuple(hidden.shape) != (1, len(expected_ids), 1024)
                or hidden.dtype != torch.float32
            ):
                raise OracleError("official last hidden-state shape/dtype differs")
            mask = encoded["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1)
            pooled = (summed / lengths).cpu().contiguous()
        values_array = pooled.numpy().astype("<f4", copy=False)
        values = [float(value) for value in values_array.reshape(-1)]
        if len(values) != 1024 or any(not math.isfinite(value) for value in values):
            raise OracleError("official pooled embedding is invalid")

        provenance = dict(base_provenance)
        provenance.update(
            {
                "input_label": label,
                "input_token_ids": expected_ids,
                "attention_mask": [1] * len(expected_ids),
                "input_tokens": len(expected_ids),
                "padding_tokens": 0,
            }
        )
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": RUNTIME_ID,
            "input_sha256": sha256_file(input_path),
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": values,
            "environment_lock": environment,
            "provenance": provenance,
        }
        portable(vector, f"oracle vector {index}")
        vector_path = output / f"input-{index}.independent-oracle-vector.json"
        vector_path.write_bytes(canonical_json(vector))
        npy_path = output / f"input-{index}.attention-mask-mean.f32.npy"
        np.save(npy_path, values_array, allow_pickle=False)
        report_records.append(
            {
                "index": index,
                "label": label,
                "input": {
                    "sha256": vector["input_sha256"],
                    "byte_size": input_path.stat().st_size,
                    "sequence_length": len(sequence),
                    "input_ids": expected_ids,
                    "attention_mask": [1] * len(expected_ids),
                },
                "output": {
                    "shape": list(values_array.shape),
                    "npy_sha256": sha256_file(npy_path),
                    "raw_f32_sha256": sha256_bytes(values_array.tobytes(order="C")),
                    "vector_sha256": sha256_file(vector_path),
                    "first_16": values[:16],
                },
            }
        )
        if index == BLOCK0_INPUT_INDEX:
            captures = capture_block0_operators(
                model, encoded, torch, modeling_llama
            )
            for name in BLOCK0_OPERATOR_RAW_F32_SHA256:
                operator_array = captures[name].numpy().astype("<f4", copy=False)
                operator_path = output / (
                    "input-1.block0-%s.f32.npy" % name
                )
                np.save(operator_path, operator_array, allow_pickle=False)
                flat = operator_array.reshape(-1)
                block0_operator_records.append(
                    {
                        "name": name,
                        "shape": list(operator_array.shape),
                        "npy_sha256": sha256_file(operator_path),
                        "raw_f32_sha256": sha256_bytes(
                            operator_array.tobytes(order="C")
                        ),
                        "first_8": [float(value) for value in flat[:8]],
                    }
                )

    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source": {
            "repo": SOURCE_REPO,
            "requested_revision": REQUESTED_REVISION,
            "revision": SOURCE_REVISION,
            "catalog_contract_sha256": CATALOG_CONTRACT_SHA256,
            "file_manifest_sha256": base_provenance["source_file_manifest_sha256"],
        },
        "records": report_records,
        "block0_operator_vectors": {
            "input_index": BLOCK0_INPUT_INDEX,
            "input_ids": INPUTS[BLOCK0_INPUT_INDEX]["input_ids"],
            "source": "official Transformers 4.57.3 hooks and apply_rotary_pos_emb wrapper",
            "native_runtime_used": False,
            "operators": block0_operator_records,
        },
        "environment_lock": environment,
        "provenance": base_provenance,
    }
    portable(report, "oracle report")
    report_path = output / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))
    print(
        json.dumps(
            {
                "report_sha256": sha256_file(report_path),
                "records": [
                    {
                        "vector_sha256": record["output"]["vector_sha256"],
                        "npy_sha256": record["output"]["npy_sha256"],
                        "raw_f32_sha256": record["output"]["raw_f32_sha256"],
                        "shape": record["output"]["shape"],
                        "first_16": record["output"]["first_16"],
                    }
                    for record in report_records
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OracleError, OSError, ValueError) as error:
        print(
            "generate_geneb_omnina_upstream_oracle: error: %s" % error,
            file=sys.stderr,
        )
        raise SystemExit(1)
