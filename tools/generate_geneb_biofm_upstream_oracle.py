#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned BioFM-265M single-record CPU-BF16 GENEB oracle.

The numerical model and tokenizer classes are loaded from the pinned upstream
Transformers/BioFM sources.  evo.cpp and converted artifacts are never imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch


RUNTIME_ID = "geneb-biofm-265m"
SOURCE_REPO = "m42-health/BioFM-265M"
SOURCE_REVISION = "52218bbdd383c123f6ced585c9cf2f62ae1fbb17"
GENEB_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXPECTED_SNAPSHOT_FILES = {
    "config.json": (5429, "68656f74065d92cd077bc1e1b54ecdb3061dfc74a67c67b64cf103ec07b82620"),
    "model.safetensors": (528959288, "ff6ea18ca716b5686ad3e0f025e0d3c86257a88766b6aa0063db5cb6493ae108"),
    "tokenizer_config.json": (129, "8429c2b5485bd51a010c0980d5871c11c6386e770886e9f87ab0d03dcd7700bb"),
}
EXPECTED_GENEB_FILES = {
    "embedding_pipeline/extractors/biofm.py": (
        1594,
        "9fd02f49398b5b3b491391c4ec0b51e1caef793e5e857e57d496df3d6b0d52e8",
    ),
    "embedding_pipeline/utility_modules/biofm/biofm-eval/biofm_eval/models/base.py": (
        233,
        "595970f22227d907618c2203cf4e8139e999ec927c05b07e0f803af5240153b0",
    ),
    "embedding_pipeline/utility_modules/biofm/biofm-eval/biofm_eval/tokenizers/base.py": (
        7078,
        "b723eeeae5c25e6e84102d4109c5d2b487984f52af685faea49f35a8c35944df",
    ),
    "embedding_pipeline/utility_modules/biofm/biofm-eval/biofm_eval/embedder.py": (
        8405,
        "7daa98c41b62598e06f3198364d76151f99480eda6e4c0a42ca216a52a73cab3",
    ),
    "embedding_pipeline/utility_modules/biofm/biofm-eval/pyproject.toml": (
        1278,
        "34064539ecf12e24d47029af8c07dee2c16610d5b0808f77e7938bc626e6033f",
    ),
}
EXPECTED_CONFIG = {
    "architectures": ["MistralForCausalLM"],
    "model_type": "mistral",
    "vocab_size": 512,
    "hidden_size": 640,
    "num_hidden_layers": 23,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 2560,
    "max_position_embeddings": 32770,
    "sliding_window": None,
    "rms_norm_eps": 1e-5,
    "rope_theta": 100000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": False,
}


class OracleError(RuntimeError):
    """Raised when pinned source or execution semantics differ."""


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


def write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, label: str, size: int, sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{label} must be a nonsymlink regular file")
    actual = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    if actual != {"size": size, "sha256": sha256}:
        raise OracleError(f"{label} differs: {actual!r}")
    return actual


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


def normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OracleError(f"{label} must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise OracleError(f"{label} must be a normalized relative path")
    return value


def validate_receipt(receipt_path: Path, snapshot: Path) -> tuple[str, dict[str, Any]]:
    receipt = load_json(receipt_path, "source receipt")
    expected_keys = {
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
    if set(receipt) != expected_keys:
        raise OracleError("source receipt fields differ")
    identity = (
        receipt["schema_version"],
        receipt["kind"],
        receipt["model_id"],
        receipt["repo"],
        receipt["requested_revision"],
        receipt["resolved_revision"],
        receipt["source_kind"],
        receipt["load_path"],
    )
    if identity != (
        1,
        "source-checkpoint",
        RUNTIME_ID,
        SOURCE_REPO,
        "main",
        SOURCE_REVISION,
        "huggingface",
        None,
    ):
        raise OracleError(f"source receipt identity differs: {identity!r}")
    catalog_contract = receipt["catalog_contract_sha256"]
    if not isinstance(catalog_contract, str) or len(catalog_contract) != 64:
        raise OracleError("source receipt catalog contract differs")
    files = receipt["files"]
    if not isinstance(files, list) or len(files) != len(EXPECTED_SNAPSHOT_FILES):
        raise OracleError("source receipt file count differs")
    verified: dict[str, Any] = {}
    for index, item in enumerate(files):
        label = f"source receipt files[{index}]"
        if not isinstance(item, dict) or set(item) != {"name", "path", "size", "sha256"}:
            raise OracleError(f"{label} fields differ")
        name = normalized_relative(item["name"], f"{label}.name")
        if name in verified or name not in EXPECTED_SNAPSHOT_FILES:
            raise OracleError(f"{label}.name differs")
        size, digest = EXPECTED_SNAPSHOT_FILES[name]
        if item["size"] != size or item["sha256"] != digest:
            raise OracleError(f"{label} descriptor differs")
        locator = Path(item["path"])
        verified[name] = checked_file(locator, label, size, digest)
        checked_file(
            (snapshot / name).resolve(strict=True), f"snapshot {name}", size, digest
        )
    if set(verified) != set(EXPECTED_SNAPSHOT_FILES):
        raise OracleError("source receipt file set differs")
    return catalog_contract, verified


def read_single_fasta(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise OracleError("input must be a nonsymlink regular file")
    records: list[str] = []
    current: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if current:
                records.append("".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        records.append("".join(current))
    if len(records) != 1 or not records[0]:
        raise OracleError("input must contain exactly one nonempty FASTA record")
    return records[0]


def load_source_class(path: Path, module_name: str, class_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise OracleError(f"cannot load pinned source module {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--geneb-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve(strict=True)
    receipt = args.receipt.resolve(strict=True)
    geneb_root = args.geneb_root.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if snapshot.name != SOURCE_REVISION or git_revision(geneb_root) != GENEB_REVISION:
        raise OracleError("pinned source revision differs")

    catalog_contract, snapshot_lock = validate_receipt(receipt, snapshot)
    geneb_lock = {
        name: checked_file(geneb_root / name, name, size, digest)
        for name, (size, digest) in EXPECTED_GENEB_FILES.items()
    }
    config = load_json(snapshot / "config.json", "BioFM config")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise OracleError(f"BioFM config key {key!r} differs")
    tokenizer_config = load_json(snapshot / "tokenizer_config.json", "tokenizer config")
    if tokenizer_config != {"char_ords": [68, 78, 65, 84, 71, 67], "model_max_length": 6144}:
        raise OracleError("BioFM tokenizer config differs")

    sequence = read_single_fasta(input_path)
    if len(sequence) + 2 > 6144:
        raise OracleError("canonical input exceeds the pinned tokenizer cap")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    source_base = geneb_root / "embedding_pipeline/utility_modules/biofm/biofm-eval/biofm_eval"
    AnnotatedModel = load_source_class(
        source_base / "models/base.py", "pinned_biofm_model", "AnnotatedModel"
    )
    AnnotationTokenizer = load_source_class(
        source_base / "tokenizers/base.py",
        "pinned_biofm_tokenizer",
        "AnnotationTokenizer",
    )
    tokenizer = AnnotationTokenizer.from_pretrained(snapshot)
    if tokenizer.vocab_size != 24 or tokenizer.padding_side != "left":
        raise OracleError("pinned AnnotationTokenizer topology differs")
    model = AnnotatedModel.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.bfloat16
    ).eval()
    if model.__class__.__name__ != "AnnotatedModel" or model.config._attn_implementation != "sdpa":
        raise OracleError("pinned BioFM model dispatch differs")

    def encode_batch(sequences: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            [" ".join(item) for item in sequences],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    encoded = encode_batch([sequence])
    expected_ids = [6] + [tokenizer.convert_tokens_to_ids(ch) for ch in sequence] + [7]
    if encoded["input_ids"].tolist() != [expected_ids]:
        raise OracleError("pinned BioFM token IDs differ")

    def forward_without_mask(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        if result.hidden_states is None or len(result.hidden_states) != 24:
            raise OracleError("pinned post-final RMS hidden tap differs")
        hidden = result.hidden_states[-1]
        indices = input_ids.ne(tokenizer.pad_token_id).sum(dim=1) - 2
        selected = hidden[torch.arange(hidden.shape[0]), indices]
        return selected.detach().float().cpu(), indices.cpu()

    pooled, selected_indices = forward_without_mask(encoded["input_ids"])
    if selected_indices.tolist() != [len(expected_ids) - 2]:
        raise OracleError("single-record BioFM selected row differs")

    longer = sequence + "ACGTNACGTN"
    defect_encoded = encode_batch([sequence, longer])
    defect_pooled, defect_indices = forward_without_mask(defect_encoded["input_ids"])
    defect_max_abs = float((pooled[0] - defect_pooled[0]).abs().max().item())
    if defect_max_abs <= 1e-4 or int(defect_indices[0]) != len(expected_ids) - 2:
        raise OracleError("pinned left-padding batch defect was not reproduced")

    values = pooled.contiguous().numpy().astype("<f4", copy=False)
    npy_path = output_dir / "input-0.penultimate-valid-token.f32.npy"
    np.save(npy_path, values, allow_pickle=False)
    generator = Path(__file__).resolve(strict=True)
    input_sha256 = sha256_file(input_path)
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
        "packages": package_versions(),
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "bfloat16",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "transformers": {
            "version": importlib.metadata.version("transformers"),
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "attention_implementation": model.config._attn_implementation,
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "source_files": {
            **{name: value for name, value in snapshot_lock.items()},
            **{f"GENEB/{name}": value for name, value in geneb_lock.items()},
        },
    }
    provenance = {
        "kind": "pinned-geneb-single-record-normalized-oracle",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "extractor_repo": "darlednik/GENEB",
        "extractor_revision": GENEB_REVISION,
        "catalog_contract_sha256": catalog_contract,
        "generator_sha256": sha256_file(generator),
        "independent_of_evo_native_runtime": True,
        "official_transformers_implementation": True,
        "pinned_biofm_model_source": True,
        "single_record_reference_equivalent": True,
        "clean_batched_reference_status": "blocked-left-padding-mask-drop-and-absolute-index",
        "normalization": "run each record without padding and select the penultimate visible row",
        "hidden_tap": "post-final-rmsnorm",
        "pooling": "penultimate-valid-token",
    }
    vector = {
        "schema_version": 1,
        "kind": "geneb-independent-oracle-vector",
        "runtime_id": RUNTIME_ID,
        "input_sha256": input_sha256,
        "backend": "cpu",
        "profile": "cpu-f32",
        "values": [float(value) for value in values.reshape(-1)],
        "environment_lock": environment,
        "provenance": provenance,
    }
    vector_path = output_dir / "input-0.independent-oracle-vector.json"
    write_json(vector_path, vector)
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source": {"repo": SOURCE_REPO, "revision": SOURCE_REVISION},
        "input": {
            "sha256": input_sha256,
            "sequence_length": len(sequence),
            "input_ids": encoded["input_ids"].tolist(),
            "attention_mask": encoded["attention_mask"].tolist(),
            "selected_row": selected_indices.tolist(),
        },
        "output": {
            "shape": list(values.shape),
            "npy_sha256": sha256_file(npy_path),
            "raw_f32_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            "vector_sha256": sha256_file(vector_path),
            "first_16": vector["values"][:16],
        },
        "reference_defect": {
            "status": "reproduced",
            "padding_side": tokenizer.padding_side,
            "model_receives_attention_mask": False,
            "short_single_selected_row": int(selected_indices[0]),
            "short_batched_selected_row": int(defect_indices[0]),
            "short_record_max_abs": defect_max_abs,
            "batch_input_ids": defect_encoded["input_ids"].tolist(),
            "batch_attention_mask": defect_encoded["attention_mask"].tolist(),
        },
        "environment_lock": environment,
        "provenance": provenance,
    }
    report_path = output_dir / "oracle-report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "report_sha256": sha256_file(report_path),
                "vector_sha256": sha256_file(vector_path),
                "npy_sha256": sha256_file(npy_path),
                "shape": list(values.shape),
                "input_ids": encoded["input_ids"].tolist(),
                "reference_defect_max_abs": defect_max_abs,
                "first_16": vector["values"][:16],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OracleError as error:
        raise SystemExit(f"generate_geneb_biofm_upstream_oracle: error: {error}") from error
