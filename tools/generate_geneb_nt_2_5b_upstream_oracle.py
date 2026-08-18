#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned NT-2.5B-MS GENEB CPU-F32 oracle offline."""

from __future__ import annotations

import argparse
import gc
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

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


RUNTIME_ID = "geneb-nt-2-5b-ms"
REPO = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
REQUESTED_REVISION = "main"
REVISION = "b746b125aacd1b0970c05a32fd71ba726754542e"
CATALOG_CONTRACT_SHA256 = (
    "17519bb61cf0e6efd33d456b4afa49f07c1651bb9bb60580af9ab02490b8995f"
)
GENEB_REPO = "darlednik/GENEB"
GENEB_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SIZE = 2084
EXTRACTOR_SHA256 = (
    "50332b6070b944dd4d26dc2641ebadfea7270513d34294add5f7066295567c43"
)
EXPECTED_FILES = {
    "config.json": (
        707,
        "7d003a6f278562634e87e63a51a1c2c685dc495c35e16cc309e6a5d912954988",
    ),
    "pytorch_model-00001-of-00002.bin": (
        9913283486,
        "9333a3ceaf85e027547e73068476ecb79b1ac4e5bae91abcdae7f7f7619d7092",
    ),
    "pytorch_model-00002-of-00002.bin": (
        278112047,
        "d599e301221170496d93158cf46f8e5872621f31fc16e570bc947b27f9917979",
    ),
    "pytorch_model.bin.index.json": (
        46039,
        "71088b187400fee5eec7f6e866d77d22fcd1dd38b930b1359e22d3fd0e284203",
    ),
    "special_tokens_map.json": (
        101,
        "d6dc30bf018166daab248b0abf7efda6fd1b1e0a2d1bee5b31b23db2ebdaee77",
    ),
    "tokenizer_config.json": (
        129,
        "e516a38aaf16ce0e84ce178e5f23b98f8b9993bea1a0e2166326efb4e8440214",
    ),
    "vocab.txt": (
        28718,
        "f1b544e27897936b50bbd925850fa8a08b421c33bb9c26e3711c140c061d0d4c",
    ),
}
EXPECTED_INPUTS = (
    {
        "label": "input-0",
        "sequence": "ATTCCGATTCCGATTCCG",
        "sha256": "1a042ddc3370c899410a3011701608d674f9afc9eaf67d12ae16cdbc0067beb9",
        "input_ids": [3, 367, 367, 367],
    },
    {
        "label": "input-1",
        "sequence": "AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT",
        "sha256": "51359a6dfdb95751d69690fc8063bfbb9956cfa364cead1f5e6dfdeab91d3432",
        "input_ids": [
            3,
            14,
            2819,
            1366,
            4103,
            4101,
            4104,
            4100,
            4102,
            4103,
            4101,
            4104,
            4104,
            4104,
            4104,
            731,
            2097,
            4100,
            4102,
            4103,
            4101,
        ],
    },
)
EXPECTED_PACKAGE_LOCK = (
    "accelerate==0.21.0",
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
    "psutil==5.9.8",
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
    "transformers==4.32.0",
    "typing-extensions==4.16.0",
    "urllib3==2.6.3",
    "zipp==3.23.1",
)
REQUIRED_THREAD_ENVIRONMENT = {
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
EXPECTED_PARAMETER_COUNT = 2547801226
# The source index has 525 tensors. Transformers registers position_ids as a
# non-persistent buffer, so the loaded official model state has the other 524.
EXPECTED_STATE_TENSOR_COUNT = 524
EXPECTED_COLUMN_MAJOR_PARAMETER_COUNT = 194
SOURCE_VOCAB_SIZE = 4107
MODEL_VOCAB_SIZE = 4105
MODEL_MAX_LENGTH = 1000
HIDDEN_SIZE = 2560
NUM_HIDDEN_LAYERS = 32


class OracleError(RuntimeError):
    """Raised when the pinned NT-2.5B upstream contract differs."""


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
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise OracleError("%s contains duplicate key %r" % (label, key))
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError("cannot load %s: %s" % (label, error)) from error
    if not isinstance(value, dict):
        raise OracleError("%s root must be an object" % label)
    return value


def normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OracleError("%s must be a nonempty string" % label)
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise OracleError("%s must be a normalized relative path" % label)
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


def validate_sources(receipt_path: Path, snapshot: Path) -> dict[str, dict[str, Any]]:
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
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != RUNTIME_ID
        or receipt["repo"] != REPO
        or receipt["requested_revision"] != REQUESTED_REVISION
        or receipt["resolved_revision"] != REVISION
        or receipt["source_kind"] != "huggingface"
        or receipt["catalog_contract_sha256"] != CATALOG_CONTRACT_SHA256
        or receipt["load_path"] is not None
    ):
        raise OracleError("source receipt pinned provenance differs")
    if snapshot.name != REVISION or not snapshot.is_dir():
        raise OracleError("snapshot path is not the pinned resolved revision")
    raw_files = receipt["files"]
    if not isinstance(raw_files, list) or len(raw_files) != len(EXPECTED_FILES):
        raise OracleError("source receipt file count differs")
    verified: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_files):
        label = "source receipt files[%d]" % index
        if not isinstance(item, dict) or set(item) != {
            "name",
            "path",
            "size",
            "sha256",
        }:
            raise OracleError(label + " fields differ")
        name = normalized_relative(item["name"], label + ".name")
        if name in verified or name not in EXPECTED_FILES:
            raise OracleError("source receipt file set is duplicated or unexpected")
        expected_size, expected_sha256 = EXPECTED_FILES[name]
        if item["size"] != expected_size or item["sha256"] != expected_sha256:
            raise OracleError("source receipt size/SHA256 differs for " + name)
        locator = Path(item["path"])
        snapshot_file = snapshot / name
        try:
            resolved_locator = locator.resolve(strict=True)
            resolved_snapshot = snapshot_file.resolve(strict=True)
        except OSError as error:
            raise OracleError("source locator is unavailable for " + name) from error
        if (
            resolved_locator != resolved_snapshot
            or not resolved_locator.is_file()
            or resolved_locator.stat().st_size != expected_size
            or sha256_file(resolved_locator) != expected_sha256
        ):
            raise OracleError("receipt/snapshot source differs for " + name)
        verified[name] = {"size": expected_size, "sha256": expected_sha256}
    snapshot_names = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if snapshot_names != set(EXPECTED_FILES) or set(verified) != set(EXPECTED_FILES):
        raise OracleError("bounded snapshot file closure differs")
    return dict(sorted(verified.items()))


def validate_extractor(repository: Path, extractor: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OracleError("cannot resolve pinned GENEB revision") from error
    expected = repository / "embedding_pipeline" / "extractors" / "nt.py"
    if (
        revision != GENEB_REVISION
        or extractor != expected.resolve(strict=True)
        or extractor.stat().st_size != EXTRACTOR_SIZE
        or sha256_file(extractor) != EXTRACTOR_SHA256
    ):
        raise OracleError("pinned GENEB NT extractor differs")
    source = extractor.read_text(encoding="utf-8")
    required_fragments = (
        'padding="max_length"',
        "max_length=self.max_length",
        "output_hidden_states=True",
        "hidden = outputs.hidden_states[-1]",
        "mask = attention_mask.unsqueeze(-1)",
        "seq_emb = (sum_hidden / lengths).cpu().numpy()",
    )
    if any(fragment not in source for fragment in required_fragments):
        raise OracleError("pinned GENEB NT extraction semantics differ")
    return {
        "repo": GENEB_REPO,
        "revision": GENEB_REVISION,
        "file": "embedding_pipeline/extractors/nt.py",
        "size": EXTRACTOR_SIZE,
        "sha256": EXTRACTOR_SHA256,
    }


def read_input(path: Path, expected: dict[str, Any]) -> str:
    if sha256_file(path) != expected["sha256"]:
        raise OracleError(expected["label"] + " input SHA256 differs")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise OracleError("cannot read canonical input: %s" % error) from error
    if len(lines) != 2 or not lines[0].startswith(">") or not lines[0][1:]:
        raise OracleError(expected["label"] + " must be one named two-line FASTA")
    if lines[1] != expected["sequence"]:
        raise OracleError(expected["label"] + " sequence differs")
    return lines[1]


def package_lock() -> tuple[str, ...]:
    values = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            values.append(
                "%s==%s" % (name.lower().replace("_", "-"), distribution.version)
            )
    return tuple(sorted(set(values)))


def validate_environment() -> tuple[str, ...]:
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise OracleError(variable + "=1 is required")
    if not os.environ.get("HF_MODULES_CACHE"):
        raise OracleError("HF_MODULES_CACHE must be explicitly isolated")
    for name, expected in REQUIRED_THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise OracleError("%s=%s is required" % (name, expected))
    actual = package_lock()
    if actual != EXPECTED_PACKAGE_LOCK:
        raise OracleError("complete oracle package lock differs")
    if platform.python_implementation() != "CPython" or platform.python_version() != "3.9.6":
        raise OracleError("CPython 3.9.6 is required")
    return actual


def tokenizer_contract(tokenizer: Any) -> tuple[str, dict[str, Any]]:
    class_name = "%s.%s" % (
        tokenizer.__class__.__module__, tokenizer.__class__.__name__
    )
    if (
        class_name != "transformers.models.esm.tokenization_esm.EsmTokenizer"
        or tokenizer.model_max_length != MODEL_MAX_LENGTH
        or tokenizer.padding_side != "right"
        or tokenizer.vocab_size != SOURCE_VOCAB_SIZE
        or len(tokenizer) != SOURCE_VOCAB_SIZE
        or tokenizer.all_special_ids != [0, 1, 3, 2]
    ):
        raise OracleError("official NT-2.5B tokenizer contract differs")
    return class_name, {
        "vocab_size": tokenizer.vocab_size,
        "length": len(tokenizer),
        "model_max_length": tokenizer.model_max_length,
        "padding_side": tokenizer.padding_side,
        "special_token_ids": list(tokenizer.all_special_ids),
        "compiled_model_vocab_boundary": MODEL_VOCAB_SIZE,
    }


def normalize_official_parameter_layout(model: Any) -> list[str]:
    """Match the row-major parameters produced by ordinary from_pretrained."""

    column_major = []
    for name, parameter in model.named_parameters():
        if parameter.is_contiguous():
            continue
        shape = tuple(parameter.shape)
        if (
            len(shape) != 2
            or parameter.storage_offset() != 0
            or parameter.untyped_storage().nbytes()
            != parameter.numel() * parameter.element_size()
            or tuple(parameter.stride()) != (1, shape[0])
        ):
            raise OracleError("official low-memory parameter layout differs for " + name)
        column_major.append(name)
    if len(column_major) != EXPECTED_COLUMN_MAJOR_PARAMETER_COUNT:
        raise OracleError("official low-memory column-major parameter count differs")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in column_major:
                contiguous = parameter.detach().contiguous()
                if not torch.equal(parameter.detach(), contiguous):
                    raise OracleError("parameter value changed while normalizing " + name)
                parameter.data = contiguous
    if any(
        not parameter.is_contiguous()
        or parameter.storage_offset() != 0
        or parameter.untyped_storage().nbytes()
        != parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ):
        raise OracleError("official parameter layout normalization failed")
    return column_major


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--geneb-repo", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.input) != len(EXPECTED_INPUTS):
        raise OracleError("exactly two canonical --input arguments are required")
    snapshot = args.snapshot.resolve(strict=True)
    receipt = args.receipt.resolve(strict=True)
    geneb_repo = args.geneb_repo.resolve(strict=True)
    extractor = args.extractor.resolve(strict=True)
    input_paths = [path.resolve(strict=True) for path in args.input]
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise OracleError("oracle output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    packages = validate_environment()
    print("validating pinned 7-file source closure", file=sys.stderr, flush=True)
    source_files = validate_sources(receipt, snapshot)
    print("validated pinned source closure", file=sys.stderr, flush=True)
    extractor_lock = validate_extractor(geneb_repo, extractor)
    sequences = [
        read_input(path, expected)
        for path, expected in zip(input_paths, EXPECTED_INPUTS)
    ]
    config = load_json(snapshot / "config.json", "model config")
    if (
        config.get("architectures") != ["EsmForMaskedLM"]
        or config.get("model_type") != "esm"
        or config.get("hidden_size") != HIDDEN_SIZE
        or config.get("intermediate_size") != 10240
        or config.get("num_attention_heads") != 20
        or config.get("num_hidden_layers") != NUM_HIDDEN_LAYERS
        or config.get("max_position_embeddings") != 1002
        or config.get("vocab_size") != MODEL_VOCAB_SIZE
        or config.get("position_embedding_type") != "absolute"
        or config.get("layer_norm_eps") != 1e-12
        or config.get("token_dropout") is not True
        or config.get("torch_dtype") != "float32"
        or "auto_map" in config
    ):
        raise OracleError("pinned NT-2.5B model config differs")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        revision=REVISION,
    )
    tokenizer_class, tokenizer_metadata = tokenizer_contract(tokenizer)
    print("loading official ESM with low_cpu_mem_usage", file=sys.stderr, flush=True)
    model, loading = AutoModelForMaskedLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        revision=REVISION,
        use_safetensors=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise OracleError("official checkpoint loading info differs: %r" % loading)
    model_class = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    state = model.state_dict()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    embeddings = model.get_input_embeddings().weight
    if (
        model_class != "transformers.models.esm.modeling_esm.EsmForMaskedLM"
        or model.config.model_type != "esm"
        or getattr(model.config, "auto_map", None) is not None
        or model.config.vocab_size != MODEL_VOCAB_SIZE
        or model.config.hidden_size != HIDDEN_SIZE
        or model.config.num_hidden_layers != NUM_HIDDEN_LAYERS
        or len(state) != EXPECTED_STATE_TENSOR_COUNT
        or parameter_count != EXPECTED_PARAMETER_COUNT
        or list(embeddings.shape) != [MODEL_VOCAB_SIZE, HIDDEN_SIZE]
        or embeddings.dtype != torch.float32
        or any(tensor.is_meta for tensor in state.values())
        or any(tensor.dtype not in (torch.float32, torch.int64) for tensor in state.values())
    ):
        raise OracleError("official NT-2.5B model topology/state differs")
    del state
    normalized_parameters = normalize_official_parameter_layout(model)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    largest_normalized_parameter_bytes = max(
        parameter.numel() * parameter.element_size()
        for name, parameter in model.named_parameters()
        if name in normalized_parameters
    )
    model.to(device="cpu", dtype=torch.float32).eval()
    print("loaded and normalized official ESM parameters", file=sys.stderr, flush=True)

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
        "packages": list(packages),
        "thread_environment": dict(sorted(REQUIRED_THREAD_ENVIRONMENT.items())),
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "build_config_sha256": sha256_bytes(torch.__config__.show().encode("utf-8")),
        },
        "transformers": {
            "version": importlib.metadata.version("transformers"),
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "safetensors_version": importlib.metadata.version("safetensors"),
            "accelerate_version": importlib.metadata.version("accelerate"),
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "model_class": model_class,
            "tokenizer_class": tokenizer_class,
        },
        "source_files": source_files,
        "extractor": extractor_lock,
    }
    base_provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-reference",
        "benchmark_semantics": "geneb-v4-reference",
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_file_manifest_sha256": sha256_bytes(canonical_json(source_manifest)),
        "checkpoint_index_sha256": EXPECTED_FILES["pytorch_model.bin.index.json"][1],
        "checkpoint_shard_sha256": [
            EXPECTED_FILES["pytorch_model-00001-of-00002.bin"][1],
            EXPECTED_FILES["pytorch_model-00002-of-00002.bin"][1],
        ],
        "extractor_repo": GENEB_REPO,
        "extractor_revision": GENEB_REVISION,
        "extractor_module": "embedding_pipeline/extractors/nt.py",
        "extractor_class": "NucleotideTransformerExtractor",
        "extractor_sha256": EXTRACTOR_SHA256,
        "official_remote_modeling_code": False,
        "official_builtin_transformers_esm": True,
        "official_loading_entrypoint": "AutoModelForMaskedLM.from_pretrained",
        "loading_optimization": "low_cpu_mem_usage=True via accelerate==0.21.0",
        "parameter_layout": "row-major-contiguous-equivalent-to-default-from_pretrained",
        "normalized_column_major_parameter_count": len(normalized_parameters),
        "normalized_column_major_parameter_names_sha256": sha256_bytes(
            canonical_json(normalized_parameters)
        ),
        "parameter_bytes": parameter_bytes,
        "largest_layout_normalization_tensor_bytes": largest_normalized_parameter_bytes,
        "layout_normalization_peak_bound_bytes": (
            parameter_bytes + largest_normalized_parameter_bytes
        ),
        "independent_of_evo_native_runtime": True,
        "generator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "hidden_tap": "outputs.hidden_states[-1]",
        "pooling": "attention-mask-mean-direct-f32-division",
        "special_tokens": "include-cls",
        "model_max_length": MODEL_MAX_LENGTH,
        "source_vocab_size": SOURCE_VOCAB_SIZE,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "parameter_count": parameter_count,
        "state_tensor_count": EXPECTED_STATE_TENSOR_COUNT,
        "tokenizer": tokenizer_metadata,
    }

    report_inputs = []
    for input_path, sequence, expected in zip(input_paths, sequences, EXPECTED_INPUTS):
        print(
            "running official full-length forward " + expected["label"],
            file=sys.stderr,
            flush=True,
        )
        encoded = tokenizer(
            [sequence],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MODEL_MAX_LENGTH,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_count = len(expected["input_ids"])
        if (
            list(input_ids.shape) != [1, MODEL_MAX_LENGTH]
            or list(attention_mask.shape) != [1, MODEL_MAX_LENGTH]
            or input_ids[0, :token_count].tolist() != expected["input_ids"]
            or int(attention_mask.sum().item()) != token_count
            or any(value != 1 for value in input_ids[0, token_count:].tolist())
            or int(input_ids.max().item()) >= MODEL_VOCAB_SIZE
        ):
            raise OracleError(expected["label"] + " official tokenization differs")
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = result.hidden_states
            if hidden_states is None or len(hidden_states) != NUM_HIDDEN_LAYERS + 1:
                raise OracleError(expected["label"] + " hidden-state tap count differs")
            hidden = hidden_states[-1]
            if (
                list(hidden.shape) != [1, MODEL_MAX_LENGTH, HIDDEN_SIZE]
                or hidden.dtype != torch.float32
            ):
                raise OracleError(expected["label"] + " hidden-state shape/dtype differs")
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        values = pooled.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        flat = [float(value) for value in values.reshape(-1)]
        if len(flat) != HIDDEN_SIZE or any(not math.isfinite(value) for value in flat):
            raise OracleError(expected["label"] + " pooled embedding is invalid")
        provenance = dict(base_provenance)
        provenance.update(
            {
                "input_label": expected["label"],
                "token_count": token_count,
                "expected_input_ids": list(expected["input_ids"]),
                "accepted_input_max_token_id": int(input_ids.max().item()),
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
            "environment_lock": environment,
            "provenance": provenance,
        }
        require_portable(vector, "oracle vector")
        vector_path = output / (expected["label"] + ".independent-oracle-vector.json")
        vector_path.write_bytes(canonical_json(vector))
        npy_path = output / (expected["label"] + ".attention-mask-mean.f32.npy")
        np.save(npy_path, values, allow_pickle=False)
        report_inputs.append(
            {
                "label": expected["label"],
                "sha256": expected["sha256"],
                "sequence_length": len(sequence),
                "input_ids": list(expected["input_ids"]),
                "nonpadding_tokens": token_count,
                "maximum_token_id": int(input_ids.max().item()),
                "output": {
                    "shape": list(values.shape),
                    "npy_sha256": sha256_file(npy_path),
                    "raw_f32_sha256": sha256_bytes(values.tobytes(order="C")),
                    "vector_sha256": sha256_file(vector_path),
                    "first_16": flat[:16],
                },
            }
        )
        del result, hidden_states, hidden, mask, pooled, values
        gc.collect()
        print(
            "completed official full-length forward " + expected["label"],
            file=sys.stderr,
            flush=True,
        )

    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "file_manifest_sha256": base_provenance["source_file_manifest_sha256"],
        },
        "inputs": report_inputs,
        "environment_lock": environment,
        "provenance": base_provenance,
    }
    require_portable(report, "oracle report")
    report_path = output / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))
    print(
        json.dumps(
            {
                "report_sha256": sha256_file(report_path),
                "vectors": {
                    item["label"]: item["output"]["vector_sha256"]
                    for item in report_inputs
                },
                "npy": {
                    item["label"]: item["output"]["npy_sha256"]
                    for item in report_inputs
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OracleError, OSError, ValueError) as error:
        raise SystemExit(
            "generate_geneb_nt_2_5b_upstream_oracle: error: %s" % error
        )
