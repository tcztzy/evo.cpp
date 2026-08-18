#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned GenomeOcean-4B GENEB CPU-F32 oracle offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


RUNTIME_ID = "geneb-genome-ocean-4b"
REPO = "DOEJGI/GenomeOcean-4B"
REQUESTED_REVISION = "main"
REVISION = "2bed2fc3ed47c5f6955ba3e64563512c9b338dfb"
CATALOG_CONTRACT_SHA256 = (
    "77137e2132fc4af6937035d995ea6564865738ec1bf8fe25d4c30a9e859cd201"
)
CONVERTER_PROFILE_CONTRACT_SHA256 = (
    "ecc34b024b8a3c2fcff8f65a9d46d96a1104bdab14051b144ccf9a65cfb0988e"
)
GENEB_REPO = "darlednik/GENEB"
GENEB_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SIZE = 2740
EXTRACTOR_SHA256 = (
    "937fc3e45fe38c4288fd05fa4f7407f4252f9c802b8609f3fc3b1c19b33d5eac"
)
EXPECTED_FILES = {
    "config.json": (
        830,
        "a0b5b7abe4c05bc68b7b4d2e58cfb8e4905707bcbf13fab8a3e8368fa410a064",
    ),
    "model-00001-of-00002.safetensors": (
        4_989_311_712,
        "570ae8543a5b23850965d1812830c589d1eb16ae145482723333ea46104d89f3",
    ),
    "model-00002-of-00002.safetensors": (
        3_517_063_264,
        "eeba5b84b9d8fb74aab67c1936a331b8417f8e748f9893f7ee063293219f40d0",
    ),
    "model.safetensors.index.json": (
        18_005,
        "c78559ac21c752220cfa1226d1049c327cf11597203caab8bf38b1d19000d112",
    ),
    "special_tokens_map.json": (
        695,
        "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    ),
    "tokenizer.json": (
        166_434,
        "ea61df7c1f41bd98f336d663389f587509e7359a278eb8b126e43f21dc4ba289",
    ),
    "tokenizer_config.json": (
        1_136,
        "50757b50caea808b47681dfcb31566b5efaac5ea06c9cd7272903522814df57f",
    ),
}
WEIGHT_FILES = {
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
}
EXPECTED_INPUTS = (
    {
        "label": "input-0",
        "sequence": "ATTCCGATTCCGATTCCG",
        "sha256": "9274955736bf8dbd80717a9019e01696c8bf4464292f0d783f704d62e20b30c4",
        "input_ids": [1, 30, 1104, 1104, 22, 2],
    },
    {
        "label": "input-1",
        "sequence": "ACGTNNNNNNNNNNACGT",
        "sha256": "d14962bb71f5b0484b5e02027034d4faa230d953a07f72a860f7c011dceac3ca",
        "input_ids": [1, 29, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 29, 9, 2],
    },
)
EXPECTED_VERSIONS = {
    "accelerate": "0.27.2",
    "numpy": "1.26.4",
    "safetensors": "0.4.2",
    "tokenizers": "0.15.2",
    "torch": "2.1.2",
    "transformers": "4.38.2",
}
EXPECTED_PACKAGE_LOCK = (
    "accelerate==0.27.2",
    "certifi==2026.7.22",
    "charset-normalizer==3.5.1",
    "filelock==3.32.3",
    "fsspec==2026.7.0",
    "hf-xet==1.6.0",
    "huggingface-hub==0.36.2",
    "idna==3.18",
    "jinja2==3.1.6",
    "markupsafe==3.0.3",
    "mpmath==1.3.0",
    "networkx==3.6.1",
    "numpy==1.26.4",
    "packaging==26.3",
    "psutil==7.2.2",
    "pyyaml==6.0.3",
    "regex==2026.7.19",
    "requests==2.34.2",
    "safetensors==0.4.2",
    "sympy==1.14.0",
    "tokenizers==0.15.2",
    "torch==2.1.2",
    "tqdm==4.70.0",
    "transformers==4.38.2",
    "typing-extensions==4.16.0",
    "urllib3==2.7.0",
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
EXPECTED_CONFIG = {
    "architectures": ["MistralForCausalLM"],
    "model_type": "mistral",
    "vocab_size": 4096,
    "hidden_size": 3072,
    "num_hidden_layers": 24,
    "num_attention_heads": 12,
    "num_key_value_heads": 4,
    "intermediate_size": 16384,
    "max_position_embeddings": 32768,
    "sliding_window": None,
    "rms_norm_eps": 0.00001,
    "rope_theta": 1000000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.38.2",
}
PARAMETER_COUNT = 4_253_174_784
STATE_TENSOR_COUNT = 219
OUTPUT_WIDTH = 3072
REFERENCE_MAX_TOKENS = 10240


class OracleError(RuntimeError):
    """Raised when the pinned GenomeOcean-4B oracle contract differs."""


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
        for key, item in pairs:
            if key in result:
                raise OracleError("%s contains duplicate key %r" % (label, key))
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError("cannot load %s: %s" % (label, error)) from error
    if not isinstance(value, dict):
        raise OracleError(label + " root must be an object")
    return value


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


def validate_environment() -> None:
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise OracleError(variable + "=1 is required")
    if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
        raise OracleError("TOKENIZERS_PARALLELISM=false is required")
    if not os.environ.get("HF_MODULES_CACHE"):
        raise OracleError("HF_MODULES_CACHE must be explicitly isolated")
    for name, expected in REQUIRED_THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise OracleError("%s=%s is required" % (name, expected))
    for package, expected in EXPECTED_VERSIONS.items():
        actual = (
            torch.__version__
            if package == "torch"
            else importlib.metadata.version(package)
        )
        if actual != expected:
            raise OracleError(
                "%s==%s is required, found %s" % (package, expected, actual)
            )
    if torch.get_default_dtype() != torch.float32:
        raise OracleError("Torch default dtype must remain float32")
    actual_package_lock = package_lock()
    if actual_package_lock != list(EXPECTED_PACKAGE_LOCK):
        raise OracleError(
            "oracle package lock differs: %r" % actual_package_lock
        )


def validate_config(snapshot: Path) -> dict[str, Any]:
    config = load_json(snapshot / "config.json", "official config")
    differences = {
        key: (config.get(key), expected)
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if differences:
        raise OracleError("official GenomeOcean-4B config differs: %r" % differences)
    return config


def validate_index(snapshot: Path) -> dict[str, Any]:
    index = load_json(
        snapshot / "model.safetensors.index.json", "checkpoint index"
    )
    if set(index) != {"metadata", "weight_map"}:
        raise OracleError("checkpoint index fields differ")
    if index["metadata"] != {"total_size": 8_506_349_568}:
        raise OracleError("checkpoint logical size differs")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or len(weight_map) != STATE_TENSOR_COUNT:
        raise OracleError("checkpoint index tensor count differs")
    counts = {
        name: list(weight_map.values()).count(name)
        for name in sorted(set(weight_map.values()))
    }
    expected = {
        "model-00001-of-00002.safetensors": 130,
        "model-00002-of-00002.safetensors": 89,
    }
    if counts != expected:
        raise OracleError("checkpoint shard selection/counts differ")
    return {"total_size": 8_506_349_568, "tensor_count": 219, "shards": counts}


def checked_file(
    path: Path,
    expected_size: int,
    expected_sha256: str,
    label: str,
    verify_payload: bool,
) -> None:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise OracleError("cannot stat %s: %s" % (label, error)) from error
    if not path.is_file() or size != expected_size:
        raise OracleError("%s size/type differs" % label)
    if verify_payload and sha256_file(path) != expected_sha256:
        raise OracleError("%s SHA256 differs" % label)


def validate_sources(
    receipt_path: Path,
    snapshot: Path,
    *,
    verify_weight_payloads: bool = True,
) -> dict[str, dict[str, Any]]:
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
        verify_payload = verify_weight_payloads or name not in WEIGHT_FILES
        locator = Path(item["path"])
        snapshot_file = snapshot / name
        checked_file(
            locator,
            expected_size,
            expected_sha256,
            "receipt locator for " + name,
            verify_payload,
        )
        same_file = False
        try:
            same_file = os.path.samefile(locator, snapshot_file)
        except OSError:
            pass
        checked_file(
            snapshot_file,
            expected_size,
            expected_sha256,
            "snapshot file for " + name,
            verify_payload and not same_file,
        )
        verified[name] = {"size": expected_size, "sha256": expected_sha256}
    if set(verified) != set(EXPECTED_FILES):
        raise OracleError("source receipt exact seven-file closure differs")
    validate_config(snapshot)
    validate_index(snapshot)
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
    expected_path = repository / "embedding_pipeline" / "extractors" / "genomeocean.py"
    if (
        revision != GENEB_REVISION
        or extractor != expected_path.resolve(strict=True)
        or extractor.stat().st_size != EXTRACTOR_SIZE
        or sha256_file(extractor) != EXTRACTOR_SHA256
    ):
        raise OracleError("pinned GENEB GenomeOcean extractor differs")
    source = extractor.read_text(encoding="utf-8")
    required_fragments = (
        "torch.bfloat16 if self.device.startswith(\"cuda\") else torch.float32",
        '"DOEJGI/GenomeOcean-4B": 10240',
        "padding_side=\"left\"",
        "torch_dtype=self.dtype",
        "output_hidden_states=True",
        "use_cache=False",
        "hidden = outputs.hidden_states[-1]",
        "mask = attention_mask.unsqueeze(-1).type_as(hidden)",
        "summed = (hidden * mask).sum(dim=1)",
        "counts = mask.sum(dim=1).clamp(min=1.0)",
        "emb = summed / counts",
    )
    if any(fragment not in source for fragment in required_fragments):
        raise OracleError("pinned GENEB GenomeOcean extraction semantics differ")
    return {
        "repo": GENEB_REPO,
        "revision": GENEB_REVISION,
        "file": "embedding_pipeline/extractors/genomeocean.py",
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
        raise OracleError(expected["label"] + " must be one two-line FASTA record")
    if lines[1] != expected["sequence"]:
        raise OracleError(expected["label"] + " sequence differs")
    return lines[1]


def tokenizer_contract(tokenizer: Any) -> tuple[str, dict[str, Any]]:
    class_name = "%s.%s" % (
        tokenizer.__class__.__module__, tokenizer.__class__.__name__
    )
    if (
        class_name
        != "transformers.tokenization_utils_fast.PreTrainedTokenizerFast"
        or tokenizer.vocab_size != 4096
        or len(tokenizer) != 4096
        or tokenizer.padding_side != "left"
        or tokenizer.pad_token_id != 3
        or tokenizer.unk_token_id != 0
        or tokenizer.cls_token_id != 1
        or tokenizer.sep_token_id != 2
        or tokenizer.mask_token_id != 4
        or tokenizer.bos_token_id is not None
        or tokenizer.eos_token_id is not None
        or tokenizer.model_max_length != 1000000000000000019884624838656
    ):
        raise OracleError("official GenomeOcean tokenizer contract differs")
    return class_name, {
        "vocab_size": tokenizer.vocab_size,
        "length": len(tokenizer),
        "padding_side": tokenizer.padding_side,
        "model_max_length": tokenizer.model_max_length,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def encode_inputs(
    tokenizer: Any, sequences: list[str]
) -> list[dict[str, list[int]]]:
    encoded_inputs = []
    for sequence, expected in zip(sequences, EXPECTED_INPUTS):
        encoded = tokenizer(
            [sequence],
            padding=True,
            truncation=True,
            max_length=REFERENCE_MAX_TOKENS,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].tolist()
        attention_mask = encoded["attention_mask"].tolist()
        if (
            input_ids != [expected["input_ids"]]
            or attention_mask != [[1] * len(expected["input_ids"])]
        ):
            raise OracleError(expected["label"] + " official tokenization differs")
        encoded_inputs.append(
            {"input_ids": input_ids[0], "attention_mask": attention_mask[0]}
        )
    return encoded_inputs


def package_lock() -> list[str]:
    values = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            values.append(
                "%s==%s" % (name.lower().replace("_", "-"), distribution.version)
            )
    return sorted(set(values))


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

    validate_environment()
    source_files = validate_sources(receipt, snapshot)
    extractor_lock = validate_extractor(geneb_repo, extractor)
    sequences = [
        read_input(path, expected)
        for path, expected in zip(input_paths, EXPECTED_INPUTS)
    ]

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer_class, tokenizer_metadata = tokenizer_contract(tokenizer)
    encoded_inputs = encode_inputs(tokenizer, sequences)

    model, loading = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    expected_loading_fields = {
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    }
    if set(loading) != expected_loading_fields or any(
        loading.get(key) for key in expected_loading_fields
    ):
        raise OracleError("official checkpoint loading info differs: %r" % loading)
    model_class = "%s.%s" % (
        model.__class__.__module__, model.__class__.__name__
    )
    state = model.state_dict()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        model_class
        != "transformers.models.mistral.modeling_mistral.MistralForCausalLM"
        or model.config.model_type != "mistral"
        or model.config.hidden_size != 3072
        or model.config.num_hidden_layers != 24
        or model.config.num_attention_heads != 12
        or model.config.num_key_value_heads != 4
        or model.config.intermediate_size != 16384
        or model.config.vocab_size != 4096
        or getattr(model.config, "_attn_implementation", None) != "sdpa"
        or len(state) != STATE_TENSOR_COUNT
        or parameter_count != PARAMETER_COUNT
        or any(tensor.dtype != torch.float32 for tensor in state.values())
        or any(parameter.device.type != "cpu" for parameter in model.parameters())
    ):
        raise OracleError("official GenomeOcean-4B model/state contract differs")
    del state
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
        "packages": package_lock(),
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
            "accelerate_version": importlib.metadata.version("accelerate"),
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "safetensors_version": importlib.metadata.version("safetensors"),
            "local_files_only": True,
            "trust_remote_code": True,
            "model_class": model_class,
            "tokenizer_class": tokenizer_class,
            "attention_implementation": model.config._attn_implementation,
            "low_cpu_mem_usage": True,
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
        "checkpoint_index_sha256": EXPECTED_FILES[
            "model.safetensors.index.json"
        ][1],
        "checkpoint_shard_sha256": [
            EXPECTED_FILES["model-00001-of-00002.safetensors"][1],
            EXPECTED_FILES["model-00002-of-00002.safetensors"][1],
        ],
        "catalog_contract_sha256": CATALOG_CONTRACT_SHA256,
        "converter_profile_contract_sha256": CONVERTER_PROFILE_CONTRACT_SHA256,
        "extractor_repo": GENEB_REPO,
        "extractor_revision": GENEB_REVISION,
        "extractor_module": "embedding_pipeline/extractors/genomeocean.py",
        "extractor_class": "GenomeOceanExtractor",
        "extractor_sha256": EXTRACTOR_SHA256,
        "official_standard_transformers_code": True,
        "independent_of_evo_native_runtime": True,
        "generator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "hidden_tap": "outputs.hidden_states[-1]",
        "pooling": "attention-mask-mean-direct-f32-division",
        "special_tokens": "include-cls-and-sep",
        "padding_side": "left",
        "extractor_max_length": REFERENCE_MAX_TOKENS,
        "use_cache": False,
        "source_weight_dtype": "bfloat16",
        "execution_dtype": "float32",
        "loader": "official-from_pretrained-low_cpu_mem_usage",
        "parameter_count": parameter_count,
        "state_tensor_count": STATE_TENSOR_COUNT,
        "tokenizer": tokenizer_metadata,
    }

    output.mkdir(parents=True, exist_ok=True)
    report_inputs = []
    for expected, sequence, encoded in zip(
        EXPECTED_INPUTS, sequences, encoded_inputs
    ):
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor(
            [encoded["attention_mask"]], dtype=torch.long
        )
        with torch.inference_mode():
            first_result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            first_hidden_states = first_result.hidden_states
            if first_hidden_states is None or len(first_hidden_states) != 25:
                raise OracleError(expected["label"] + " hidden-state tap count differs")
            first_hidden = first_hidden_states[-1]
            first_mask = attention_mask.unsqueeze(-1).to(dtype=first_hidden.dtype)
            first_pooled = (first_hidden * first_mask).sum(dim=1) / first_mask.sum(
                dim=1
            ).clamp(min=1.0)
            second_result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            second_hidden_states = second_result.hidden_states
            if second_hidden_states is None or len(second_hidden_states) != 25:
                raise OracleError(expected["label"] + " repeat tap count differs")
            second_hidden = second_hidden_states[-1]
            second_mask = attention_mask.unsqueeze(-1).to(dtype=second_hidden.dtype)
            second_pooled = (second_hidden * second_mask).sum(
                dim=1
            ) / second_mask.sum(dim=1).clamp(min=1.0)
        if (
            list(first_hidden.shape)
            != [1, len(encoded["input_ids"]), OUTPUT_WIDTH]
            or first_hidden.dtype != torch.float32
            or list(second_hidden.shape) != list(first_hidden.shape)
            or second_hidden.dtype != torch.float32
            or not torch.equal(first_pooled, second_pooled)
        ):
            raise OracleError(
                expected["label"] + " hidden shape/dtype/determinism differs"
            )
        values = (
            first_pooled.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        )
        flat = [float(value) for value in values.reshape(-1)]
        if len(flat) != OUTPUT_WIDTH or any(not math.isfinite(value) for value in flat):
            raise OracleError(expected["label"] + " pooled embedding is invalid")
        provenance = dict(base_provenance)
        provenance.update(
            {
                "input_label": expected["label"],
                "token_count": len(encoded["input_ids"]),
                "expected_input_ids": list(expected["input_ids"]),
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
        vector_path = output / (
            expected["label"] + ".independent-oracle-vector.json"
        )
        vector_path.write_bytes(canonical_json(vector))
        npy_path = output / (
            expected["label"] + ".attention-mask-mean.f32.npy"
        )
        np.save(npy_path, values, allow_pickle=False)
        report_inputs.append(
            {
                "label": expected["label"],
                "sha256": expected["sha256"],
                "sequence": sequence,
                "input_ids": list(encoded["input_ids"]),
                "attention_mask": list(encoded["attention_mask"]),
                "output": {
                    "shape": list(values.shape),
                    "npy_sha256": sha256_file(npy_path),
                    "raw_f32_sha256": sha256_bytes(values.tobytes(order="C")),
                    "vector_sha256": sha256_file(vector_path),
                    "first_16": flat[:16],
                },
            }
        )
        del (
            first_result,
            first_hidden_states,
            first_hidden,
            first_mask,
            first_pooled,
            second_result,
            second_hidden_states,
            second_hidden,
            second_mask,
            second_pooled,
            input_ids,
            attention_mask,
        )

    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "file_manifest_sha256": base_provenance[
                "source_file_manifest_sha256"
            ],
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
            "generate_geneb_genome_ocean_4b_upstream_oracle: error: %s" % error
        )
