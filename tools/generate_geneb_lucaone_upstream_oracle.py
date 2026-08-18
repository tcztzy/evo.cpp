#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned LucaOne GENEB CPU-F32 oracle fully offline."""

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
from transformers import AutoModel, AutoTokenizer


RUNTIME_ID = "geneb-lucaone"
REPO = "LucaGroup/LucaOne-default-step36M"
REQUESTED_REVISION = "main"
REVISION = "f0d2807eb097a1507b911022d979b502cd62b74c"
CATALOG_CONTRACT_SHA256 = (
    "0dd2d9b95f1fb5cd53f5dbb48cea9a1a05da8214b3818ae710de03744f34b7b4"
)
GENEB_REPO = "darlednik/GENEB"
GENEB_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SIZE = 3138
EXTRACTOR_SHA256 = (
    "4c471626e28f5ed98b820df46a1205262aae07da52a6e27c4242ec2697ae5787"
)
EXPECTED_FILES = {
    ".gitattributes": (
        1519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "README.md": (
        6704,
        "7136d86c3793dada5c43bf37cc57fb55d2430201e15825d5c9b9f88b8c094dd3",
    ),
    "__init__.py": (
        1618,
        "beaa2f9a99fe356c9b489e1dddfd90d3207e198b434173a557ced673257005e8",
    ),
    "config.json": (
        1625,
        "6fa2e45926a9ac2c939979fdcc69f62b292bb1d69188f1965a95480938d5e16d",
    ),
    "configuration_lucaone.py": (
        3987,
        "e0e09c8dced937daf4782eaf6d4fd3fe05a20fae116ca3d28d1fe8b4799e6038",
    ),
    "model-00001-of-00002.safetensors": (
        4930881120,
        "f4f62f17030bbf8353cfabd8d8291688b9b7a6632d55710178651b4ea8a529c6",
    ),
    "model-00002-of-00002.safetensors": (
        1390366196,
        "1b2b9ec5fa2159a09c671e5d0dde6c1ea9537219f5ceab51a53e189cb79c5fb5",
    ),
    "model.safetensors.index.json": (
        30893,
        "f4c1163ad1afc62437362116c1691a8d68f49b9d7b3722f162fb510abd4b2819",
    ),
    "modeling_lucaone.py": (
        56477,
        "e38c7723bc560f09be7ea4b64a337d0f889b1ab0360a0dc44539d6fbd133a83c",
    ),
    "special_tokens_map.json": (
        125,
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    ),
    "tokenization_lucaone.py": (
        16482,
        "2928bf429c12110818920d262b413f3314d2227ef21ffe7ac3345ad5d30d60e4",
    ),
    "tokenizer_config.json": (
        1238,
        "fbd1ad75ff2879ea218659d5e2fbcff98217b49872246e6cbb020f2fce7720f3",
    ),
    "vocab.json": (
        442,
        "4661d174fe446ce6ceb458a0d5d0adab310bf53d01dceb99b3952b020c1ee1ee",
    ),
}
EXPECTED_INPUTS = (
    {
        "label": "input-0",
        "sequence": "ACGTNNNNACGT",
        "sha256": "c0dbf01cee442008fadfd38ed0e89c80dc63a8aaa8f5d8d803893e1624d89ba0",
        "input_ids": [2, 5, 7, 8, 6, 9, 9, 9, 9, 5, 7, 8, 6, 3],
    },
    {
        "label": "input-1",
        "sequence": "ttgcau?NXacgt",
        "sha256": "91e4c782b4e779975b18eeb7ed9d075ca559e9b8670590152374495ef074ddb5",
        "input_ids": [2, 6, 6, 8, 7, 5, 6, 9, 9, 9, 5, 7, 8, 6, 3],
    },
)
EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "safetensors": "0.7.0",
    "tokenizers": "0.22.2",
    "torch": "2.7.1",
    "transformers": "4.57.3",
}
EXPECTED_LM_HEAD_KEYS = {
    "lm_head.bias",
    "lm_head.decoder.weight",
    "lm_head.dense.bias",
    "lm_head.dense.weight",
    "lm_head.layer_norm.bias",
    "lm_head.layer_norm.weight",
}
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


class OracleError(RuntimeError):
    """Raised when the pinned LucaOne oracle contract differs."""


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
        for candidate, candidate_label in (
            (Path(item["path"]), "receipt locator"),
            (snapshot / name, "snapshot file"),
        ):
            if (
                not candidate.is_file()
                or candidate.stat().st_size != expected_size
                or sha256_file(candidate) != expected_sha256
            ):
                raise OracleError("%s differs for %s" % (candidate_label, name))
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
    expected = repository / "embedding_pipeline" / "extractors" / "lucaone.py"
    if (
        revision != GENEB_REVISION
        or extractor != expected.resolve(strict=True)
        or extractor.stat().st_size != EXTRACTOR_SIZE
        or sha256_file(extractor) != EXTRACTOR_SHA256
    ):
        raise OracleError("pinned GENEB LucaOne extractor differs")
    source = extractor.read_text(encoding="utf-8")
    required_fragments = (
        'self.seq_type = "gene"',
        "self.add_special_tokens = True",
        "task_level=\"token_level\"",
        "task_type=\"embedding\"",
        "max_length=self.max_length",
        "hidden = outputs.last_hidden_state",
        "mask = attention_mask.unsqueeze(-1).float()",
        "emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)",
    )
    if any(fragment not in source for fragment in required_fragments):
        raise OracleError("pinned GENEB LucaOne extraction semantics differ")
    return {
        "repo": GENEB_REPO,
        "revision": GENEB_REVISION,
        "file": "embedding_pipeline/extractors/lucaone.py",
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
    if len(lines) != 2 or not lines[0].startswith(">"):
        raise OracleError(expected["label"] + " must be one two-line FASTA record")
    if lines[1] != expected["sequence"]:
        raise OracleError(expected["label"] + " sequence differs")
    return lines[1]


def package_lock() -> list[str]:
    values = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            values.append(
                "%s==%s" % (name.lower().replace("_", "-"), distribution.version)
            )
    return sorted(set(values))


def validate_environment() -> None:
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise OracleError(variable + "=1 is required")
    if not os.environ.get("HF_MODULES_CACHE"):
        raise OracleError("HF_MODULES_CACHE must be explicitly isolated")
    for name, expected in REQUIRED_THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise OracleError("%s=%s is required" % (name, expected))
    for package, expected in EXPECTED_VERSIONS.items():
        actual = torch.__version__ if package == "torch" else importlib.metadata.version(package)
        if actual != expected:
            raise OracleError("%s==%s is required, found %s" % (package, expected, actual))


def tokenizer_contract(tokenizer: Any) -> tuple[str, dict[str, Any]]:
    class_name = "%s.%s" % (
        tokenizer.__class__.__module__, tokenizer.__class__.__name__
    )
    if (
        tokenizer.__class__.__name__ != "LucaGPLMTokenizer"
        or not class_name.endswith(".tokenization_lucaone.LucaGPLMTokenizer")
        or tokenizer.vocab_size != 39
        or len(tokenizer) != 39
        or tokenizer.padding_side != "right"
        or tokenizer.pad_token_id != 0
        or tokenizer.unk_token_id != 1
        or tokenizer.cls_token_id != 2
        or tokenizer.sep_token_id != 3
        or tokenizer.mask_token_id != 4
        or tokenizer.model_max_length < 10**20
    ):
        raise OracleError("official LucaOne tokenizer contract differs")
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
    }


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
    validate_environment()
    source_files = validate_sources(receipt, snapshot)
    extractor_lock = validate_extractor(geneb_repo, extractor)
    sequences = [
        read_input(path, expected)
        for path, expected in zip(input_paths, EXPECTED_INPUTS)
    ]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True
    )
    tokenizer_class, tokenizer_metadata = tokenizer_contract(tokenizer)
    model, loading = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        task_level="token_level",
        task_type="embedding",
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise OracleError("official checkpoint loading info differs: %r" % loading)
    unexpected = set(loading.get("unexpected_keys", []))
    if unexpected != EXPECTED_LM_HEAD_KEYS:
        raise OracleError("official omitted MLM head set differs: %r" % sorted(unexpected))
    model_class = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    state = model.state_dict()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        model.__class__.__name__ != "LucaGPLMModel"
        or not model_class.endswith(".modeling_lucaone.LucaGPLMModel")
        or model.config.model_type != "lucaone"
        or model.config.max_position_embeddings != 4096
        or model.config.hidden_size != 2560
        or model.config.num_hidden_layers != 20
        or model.config.num_attention_heads != 40
        or model.config.vocab_size != 39
        or len(state) != 344
        or parameter_count != 1573639680
        or any(tensor.dtype != torch.float32 for tensor in state.values())
    ):
        raise OracleError("official LucaOne model topology/state contract differs")
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
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "safetensors_version": importlib.metadata.version("safetensors"),
            "local_files_only": True,
            "trust_remote_code": True,
            "model_class": model_class,
            "tokenizer_class": tokenizer_class,
            "attention_implementation": "pinned LucaOne SDPA path",
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
        "extractor_repo": GENEB_REPO,
        "extractor_revision": GENEB_REVISION,
        "extractor_module": "embedding_pipeline/extractors/lucaone.py",
        "extractor_class": "LucaOneExtractor",
        "extractor_sha256": EXTRACTOR_SHA256,
        "official_remote_modeling_code": True,
        "official_gene_tokenizer": True,
        "independent_of_evo_native_runtime": True,
        "generator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "hidden_tap": "outputs.last_hidden_state",
        "pooling": "attention-mask-mean-direct-f32-division",
        "special_tokens": "include-cls-and-sep",
        "model_max_length": 4096,
        "parameter_count": parameter_count,
        "state_tensor_count": 344,
        "omitted_mlm_head": sorted(EXPECTED_LM_HEAD_KEYS),
        "tokenizer": tokenizer_metadata,
    }

    report_inputs = []
    for input_path, sequence, expected in zip(input_paths, sequences, EXPECTED_INPUTS):
        encoded = tokenizer.encode_plus(
            text=sequence,
            seq_type="gene",
            add_special_tokens=True,
            truncation=True,
            max_length=4096,
            return_attention_mask=True,
        )
        input_ids_list = encoded.get("input_ids")
        attention_mask_list = encoded.get("attention_mask")
        token_type_ids_list = encoded.get("token_type_ids")
        if (
            input_ids_list != expected["input_ids"]
            or attention_mask_list != [1] * len(expected["input_ids"])
            or token_type_ids_list != [0] * len(expected["input_ids"])
        ):
            raise OracleError(expected["label"] + " official tokenization differs")
        input_ids = torch.tensor([input_ids_list], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask_list], dtype=torch.long)
        with torch.inference_mode():
            first_hidden = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            first_mask = attention_mask.unsqueeze(-1).to(dtype=torch.float32)
            first_pooled = (first_hidden * first_mask).sum(dim=1) / first_mask.sum(
                dim=1
            ).clamp(min=1.0)
            second_hidden = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            second_mask = attention_mask.unsqueeze(-1).to(dtype=torch.float32)
            second_pooled = (second_hidden * second_mask).sum(
                dim=1
            ) / second_mask.sum(dim=1).clamp(min=1.0)
        if (
            list(first_hidden.shape) != [1, len(input_ids_list), 2560]
            or first_hidden.dtype != torch.float32
            or not torch.equal(first_pooled, second_pooled)
        ):
            raise OracleError(expected["label"] + " forward shape/determinism differs")
        values = (
            first_pooled.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        )
        flat = [float(value) for value in values.reshape(-1)]
        if len(flat) != 2560 or any(not math.isfinite(value) for value in flat):
            raise OracleError(expected["label"] + " pooled embedding is invalid")
        provenance = dict(base_provenance)
        provenance.update(
            {
                "input_label": expected["label"],
                "token_count": len(input_ids_list),
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
        vector_path = output / (expected["label"] + ".independent-oracle-vector.json")
        vector_path.write_bytes(canonical_json(vector))
        npy_path = output / (expected["label"] + ".attention-mask-mean.f32.npy")
        np.save(npy_path, values, allow_pickle=False)
        report_inputs.append(
            {
                "label": expected["label"],
                "sha256": expected["sha256"],
                "sequence": sequence,
                "input_ids": list(input_ids_list),
                "attention_mask": list(attention_mask_list),
                "token_type_ids": list(token_type_ids_list),
                "output": {
                    "shape": list(values.shape),
                    "npy_sha256": sha256_file(npy_path),
                    "raw_f32_sha256": sha256_bytes(values.tobytes(order="C")),
                    "vector_sha256": sha256_file(vector_path),
                    "first_16": flat[:16],
                },
            }
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
            "generate_geneb_lucaone_upstream_oracle: error: %s" % error
        )
