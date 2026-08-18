#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate two pinned GENERator-Eukaryote-1.2B GENEB CPU-F32 oracles."""

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


RUNTIME_ID = "geneb-generator-eukaryote-1-2b"
SOURCE_REPO = "GenerTeam/GENERator-eukaryote-1.2b-base"
REQUESTED_REVISION = "main"
SOURCE_REVISION = "5e872c94264891f9adf59d8ea64e426bb68badb5"
CATALOG_CONTRACT_SHA256 = (
    "96faa077f463e99b02199b3b8ae6dd41cb42d7e350d07c983a82bb008072bcef"
)
EXTRACTOR_REPO = "ultimativity/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "450dadf4372b7e667c7d963e8aa06802f98408ccc9de93727b9b193d76cf89f5"
)
EXTRACTOR_PYPROJECT_SHA256 = (
    "616a15c0a751eb14687fefe691cd624fbf4cc104605f167a5782cb29d16060e0"
)
EXTRACTOR_UV_LOCK_SHA256 = (
    "e1fce57ba7eaf9fddebf300161c2840b6e6722bb1c82ac3a5b4e8e093a699953"
)
EXPECTED_SOURCE_FILES = {
    "config.json": (
        752,
        "2b47f1cbbadd52ad0f64e1802e3e7932e7f39cf8d42174e82bad4f6436504145",
    ),
    "model.safetensors": (
        4_648_274_384,
        "2bd949f005bcb1547f4f01d15a68491d51adb46c01e9790e4d9331500f6884d9",
    ),
    "modeling_generator.py": (
        9_927,
        "c605e5d3b201a93a4f8db998442a4a12ab018b214c50005a53f213e6cbbabe35",
    ),
    "special_tokens_map.json": (
        122,
        "4844a00974957de4445830bfc5892cb342f8a8dfaa50fe305a22875523c5a0f7",
    ),
    "tokenizer.py": (
        5_898,
        "211b26c254b872671c5387806a8c71820a26d2139c9a8221f6310486032f88ae",
    ),
    "tokenizer_config.json": (
        1_311,
        "e3545bbe1896de15730cde4c7a355af89fb61b713cd4444cf11a07100588cc35",
    ),
    "vocab.txt": (
        48_399,
        "068e0d76ca42ac672ade448afb2462ce4bd800a1fc1ad863d0b65992d90cf881",
    ),
}
EXPECTED_CONFIG = {
    "architectures": ["GENERatorForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoModelForCausalLM": "modeling_generator.GENERatorForCausalLM"
    },
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 5632,
    "max_position_embeddings": 16384,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 26,
    "num_key_value_heads": 4,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.44.0",
    "use_cache": True,
    "vocab_size": 4128,
}
INPUTS = (
    {
        "label": "generator_input_0",
        "sequence": "ACGTNACGTNACGTN",
        "trimmed": "ACGTNACGTNAC",
        "processed": "<s>ACGTNACGTNAC",
        "input_ids": [1, 1, 0, 0],
    },
    {
        "label": "generator_input_1",
        "sequence": "AAAAAATTTTTTGG",
        "trimmed": "AAAAAATTTTTT",
        "processed": "<s>AAAAAATTTTTT",
        "input_ids": [1, 1, 32, 1397],
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


class OracleError(RuntimeError):
    """Raised when the pinned upstream oracle contract differs."""


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
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
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
        raise OracleError(label + " must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise OracleError(label + " must be a normalized relative path")
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
    receipt = load_object(receipt_path, "source receipt")
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
        or receipt["repo"] != SOURCE_REPO
        or receipt["requested_revision"] != REQUESTED_REVISION
        or receipt["resolved_revision"] != SOURCE_REVISION
        or receipt["source_kind"] != "huggingface"
        or receipt["catalog_contract_sha256"] != CATALOG_CONTRACT_SHA256
        or receipt["load_path"] is not None
    ):
        raise OracleError("source receipt pinned identity differs")
    if snapshot.name != SOURCE_REVISION or not snapshot.is_dir():
        raise OracleError("snapshot path does not name pinned revision")
    raw_files = receipt["files"]
    if not isinstance(raw_files, list) or len(raw_files) != len(EXPECTED_SOURCE_FILES):
        raise OracleError("source receipt file count differs")
    verified: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_files):
        label = f"source receipt files[{index}]"
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "path",
            "size",
            "sha256",
        }:
            raise OracleError(label + " fields differ")
        name = normalized_relative(raw["name"], label + ".name")
        if name in verified or name not in EXPECTED_SOURCE_FILES:
            raise OracleError("source receipt file set is duplicated or unexpected")
        expected_size, expected_sha256 = EXPECTED_SOURCE_FILES[name]
        if raw["size"] != expected_size or raw["sha256"] != expected_sha256:
            raise OracleError("source receipt metadata differs for " + name)
        locator = Path(raw["path"])
        logical = snapshot / name
        resolved: set[Path] = set()
        for candidate in (locator, logical):
            if not candidate.is_file() or candidate.stat().st_size != expected_size:
                raise OracleError("pinned source file differs for " + name)
            resolved.add(candidate.resolve(strict=True))
        if len(resolved) != 1:
            raise OracleError("source receipt locator differs for " + name)
        if sha256_file(next(iter(resolved))) != expected_sha256:
            raise OracleError("pinned source file differs for " + name)
        verified[name] = {"sha256": expected_sha256, "size": expected_size}
    if set(verified) != set(EXPECTED_SOURCE_FILES):
        raise OracleError("source receipt file set differs")
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


def validate_extractor(repository: Path, extractor: Path) -> dict[str, str]:
    if git_output(repository, "rev-parse", "HEAD") != EXTRACTOR_REVISION:
        raise OracleError("GENEB extractor checkout revision differs")
    relative = extractor.resolve(strict=True).relative_to(repository.resolve(strict=True))
    committed = subprocess.run(
        ["git", "-C", str(repository), "show", f"HEAD:{relative.as_posix()}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = extractor.read_bytes()
    if (
        committed.returncode != 0
        or committed.stdout != payload
        or sha256_bytes(payload) != EXTRACTOR_SHA256
    ):
        raise OracleError("GENEB extractor differs from pinned commit")
    pyproject = repository / "embedding_pipeline" / "pyproject.toml"
    uv_lock = repository / "embedding_pipeline" / "uv.lock"
    if sha256_file(pyproject) != EXTRACTOR_PYPROJECT_SHA256:
        raise OracleError("GENEB pyproject differs")
    if sha256_file(uv_lock) != EXTRACTOR_UV_LOCK_SHA256:
        raise OracleError("GENEB uv.lock differs")
    fragments = (
        b"trust_remote_code=True",
        b"self.tokenizer.padding_side = \"right\"",
        b"self.max_length = self.model.config.max_position_embeddings",
        b"s2 = self._trim_to_multiple_of_6(s)",
        b"processed.append(self.bos_token + s2 if s2 else self.bos_token)",
        b"last_hidden = out.hidden_states[-1]",
        b"summed = (last_hidden * mask).sum(dim=1)",
        b"emb = summed / counts",
    )
    if any(fragment not in payload for fragment in fragments):
        raise OracleError("GENEB extractor execution contract differs")
    return {
        "pyproject_sha256": EXTRACTOR_PYPROJECT_SHA256,
        "revision": EXTRACTOR_REVISION,
        "source_sha256": EXTRACTOR_SHA256,
        "uv_lock_sha256": EXTRACTOR_UV_LOCK_SHA256,
    }


def read_input(path: Path, expected: dict[str, Any]) -> tuple[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise OracleError(f"cannot read oracle input: {error}") from error
    if (
        len(lines) != 2
        or lines[0] != ">" + expected["label"]
        or lines[1] != expected["sequence"]
    ):
        raise OracleError("oracle input differs for " + expected["label"])
    return expected["label"], expected["sequence"]


def validate_packages() -> list[str]:
    for name, version in REQUIRED_PACKAGE_VERSIONS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise OracleError("locked package is missing: " + name) from error
        if actual != version:
            raise OracleError(
                f"locked package {name} differs: expected {version}, got {actual}"
            )
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            packages.append(
                f"{name.lower().replace('_', '-')}=={distribution.version}"
            )
    return sorted(set(packages))


def prepare_output(path: Path, force: bool) -> None:
    expected = {
        "input-0.independent-oracle-vector.json",
        "input-0.attention-mask-mean.f32.npy",
        "input-1.independent-oracle-vector.json",
        "input-1.attention-mask-mean.f32.npy",
        "oracle-report.json",
    }
    if path.exists():
        if not path.is_dir():
            raise OracleError("oracle output is not a directory")
        existing = {item.name for item in path.iterdir()}
        if existing and not force:
            raise OracleError("oracle output directory is not empty")
        if not existing <= expected:
            raise OracleError("oracle output contains unexpected files")
        if force:
            for name in existing:
                (path / name).unlink()
    else:
        path.mkdir(parents=True)


def write_atomic(path: Path, payload: bytes) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="wb", prefix="." + path.name + ".", dir=path.parent, delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


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

    try:
        snapshot = args.snapshot.resolve(strict=True)
        receipt = args.source_receipt.resolve(strict=True)
        extractor_repo = args.extractor_repo.resolve(strict=True)
        extractor_path = args.extractor.resolve(strict=True)
        inputs = [args.input_0.resolve(strict=True), args.input_1.resolve(strict=True)]
        output = args.output_dir.resolve()
        prepare_output(output, args.force)
        source_files = validate_sources(receipt, snapshot)
        extractor = validate_extractor(extractor_repo, extractor_path)
        input_records = [
            read_input(path, expected)
            for path, expected in zip(inputs, INPUTS, strict=True)
        ]
        if load_object(snapshot / "config.json", "model config") != EXPECTED_CONFIG:
            raise OracleError("pinned model config differs")

        import numpy as np
        import torch
        import tokenizers
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        packages = validate_packages()
        torch.manual_seed(0)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            revision=SOURCE_REVISION,
            trust_remote_code=True,
        )
        if (
            tokenizer.__class__.__name__ != "DNAKmerTokenizer"
            or int(tokenizer.vocab_size) != 4128
            or len(tokenizer) != 4128
            or tokenizer.k != 6
            or not tokenizer.add_bos_token
            or tokenizer.add_eos_token
            or tokenizer.padding_side != "right"
            or tokenizer.truncation_side != "right"
            or tokenizer.model_max_length != MODEL_MAX_LENGTH_SENTINEL
            or tokenizer.unk_token_id != 0
            or tokenizer.bos_token_id != 1
            or tokenizer.eos_token_id != 2
            or tokenizer.pad_token_id != 3
            or tokenizer.mask_token_id != 4
        ):
            raise OracleError("official remote tokenizer contract differs")

        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            revision=SOURCE_REVISION,
            trust_remote_code=True,
        )
        if (
            model.__class__.__name__ != "GENERatorForCausalLM"
            or model.config.max_position_embeddings != 16384
            or model.config.hidden_size != 2048
            or len(model.model.layers) != 26
            or model.config._attn_implementation != "sdpa"
            or not hasattr(model, "tokenizer")
            or not hasattr(model, "_kmer_ids")
        ):
            raise OracleError("official remote model class/topology differs")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise OracleError("official checkpoint did not load entirely as F32")
        model.to(device="cpu", dtype=torch.float32).eval()

        source_manifest = [
            {"name": name, "sha256": item["sha256"], "size": item["size"]}
            for name, item in source_files.items()
        ]
        environment = {
            "geneb_uv_lock_sha256": EXTRACTOR_UV_LOCK_SHA256,
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
            "schema_version": 1,
            "source_files": source_files,
            "torch": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "device": "cpu",
                "dtype": "float32",
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "mps_available_but_unused": bool(torch.backends.mps.is_available()),
                "num_interop_threads": torch.get_num_interop_threads(),
                "num_threads": torch.get_num_threads(),
                "version": torch.__version__,
            },
            "transformers": {
                "attention_implementation": model.config._attn_implementation,
                "local_files_only": True,
                "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "tokenizer_class": (
                    f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
                ),
                "tokenizers_version": tokenizers.__version__,
                "trust_remote_code": True,
                "version": transformers.__version__,
            },
        }
        generator = Path(__file__).resolve(strict=True)
        base_provenance = {
            "attention_implementation": "sdpa",
            "checkpoint_sha256": EXPECTED_SOURCE_FILES["model.safetensors"][1],
            "extractor_class": "GeneratorExtractor",
            "extractor_module": "embedding_pipeline/extractors/generator.py",
            "extractor_pyproject_sha256": extractor["pyproject_sha256"],
            "extractor_repo": EXTRACTOR_REPO,
            "extractor_revision": extractor["revision"],
            "extractor_sha256": extractor["source_sha256"],
            "extractor_uv_lock_sha256": extractor["uv_lock_sha256"],
            "generator_sha256": sha256_file(generator),
            "hidden_tap": "outputs.hidden_states[-1]",
            "independent_of_evo_native_runtime": True,
            "kind": "pinned-upstream-reference",
            "model_entrypoint": "AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)",
            "official_remote_model_code": True,
            "oracle_contract": "geneb-independent-oracle-v1",
            "oracle_execution_call": "exact-reference",
            "padding_side": "right",
            "parity_scope": "fixed short raw A/C/G/T/N inputs; CPU F32; batch size 1",
            "pooling": "attention-mask-mean-direct-f32-division",
            "preprocess": "right trim raw bases to a multiple of 6, then prepend tokenizer BOS text",
            "source_file_manifest_sha256": sha256_bytes(
                canonical_json(source_manifest)
            ),
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "special_tokens": "include; manual BOS text plus tokenizer-added BOS",
            "trust_remote_code": True,
        }

        report_records: list[dict[str, Any]] = []
        for index, ((label, sequence), expected, input_path) in enumerate(
            zip(input_records, INPUTS, inputs, strict=True)
        ):
            trimmed = sequence[: (len(sequence) // 6) * 6]
            processed = tokenizer.bos_token + trimmed if trimmed else tokenizer.bos_token
            if trimmed != expected["trimmed"] or processed != expected["processed"]:
                raise OracleError(f"GENEB preprocessing differs for input {index}")
            encoded = tokenizer(
                [processed],
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                max_length=model.config.max_position_embeddings,
                truncation=True,
            )
            if list(encoded.keys()) != ["input_ids", "attention_mask"]:
                raise OracleError("official tokenizer clean encoding keys differ")
            ids = encoded["input_ids"].tolist()
            attention_mask = encoded["attention_mask"].tolist()
            if ids != [expected["input_ids"]] or attention_mask != [[1, 1, 1, 1]]:
                raise OracleError(f"official tokenizer IDs differ for input {index}")
            with torch.inference_mode():
                result = model(
                    **encoded,
                    return_dict=True,
                    output_hidden_states=True,
                )
                if result.hidden_states is None or len(result.hidden_states) != 27:
                    raise OracleError("official hidden-state tap count differs")
                hidden = result.hidden_states[-1]
                if tuple(hidden.shape) != (1, 4, 2048) or hidden.dtype != torch.float32:
                    raise OracleError("official last hidden state shape/dtype differs")
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts
            values = pooled.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
            flat = [float(value) for value in values.reshape(-1)]
            if len(flat) != 2048 or any(not math.isfinite(value) for value in flat):
                raise OracleError("official pooled embedding is invalid")

            npy_path = output / f"input-{index}.attention-mask-mean.f32.npy"
            with npy_path.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            provenance = {
                **base_provenance,
                "input_index": index,
                "nonpadding_tokens": 4,
            }
            vector = {
                "backend": "cpu",
                "environment_lock": environment,
                "input_sha256": sha256_file(input_path),
                "kind": "geneb-independent-oracle-vector",
                "profile": "cpu-f32",
                "provenance": provenance,
                "runtime_id": RUNTIME_ID,
                "schema_version": 1,
                "values": flat,
            }
            portable(vector, f"oracle vector {index}")
            vector_path = output / f"input-{index}.independent-oracle-vector.json"
            write_atomic(vector_path, canonical_json(vector))
            raw = values.tobytes(order="C")
            report_records.append(
                {
                    "input": {
                        "attention_mask": attention_mask,
                        "index": index,
                        "input_ids": ids,
                        "label": label,
                        "processed_sequence": processed,
                        "raw_sequence": sequence,
                        "raw_sequence_length": len(sequence),
                        "sha256": vector["input_sha256"],
                        "trimmed_sequence": trimmed,
                    },
                    "output": {
                        "first_values": flat[:8],
                        "npy_sha256": sha256_file(npy_path),
                        "raw_f32_sha256": sha256_bytes(raw),
                        "shape": list(values.shape),
                        "vector_sha256": sha256_file(vector_path),
                    },
                }
            )

        report = {
            "environment_lock": environment,
            "generator_sha256": base_provenance["generator_sha256"],
            "kind": "geneb-independent-upstream-oracle-report",
            "oracle_execution_call": "exact-reference",
            "records": report_records,
            "runtime_id": RUNTIME_ID,
            "schema_version": 1,
            "source": {
                "catalog_contract_sha256": CATALOG_CONTRACT_SHA256,
                "file_manifest_sha256": base_provenance[
                    "source_file_manifest_sha256"
                ],
                "repo": SOURCE_REPO,
                "revision": SOURCE_REVISION,
            },
            "status": "passed",
        }
        portable(report, "oracle report")
        report_path = output / "oracle-report.json"
        write_atomic(report_path, canonical_json(report))
        print(
            json.dumps(
                {
                    "generator_sha256": base_provenance["generator_sha256"],
                    "oracle_report_sha256": sha256_file(report_path),
                    "record_count": len(report_records),
                    "vectors": [
                        record["output"]["vector_sha256"] for record in report_records
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    except (OracleError, OSError, ValueError) as error:
        print("generate_geneb_generator_upstream_oracle: error: " + str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
