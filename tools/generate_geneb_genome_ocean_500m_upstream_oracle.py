#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned GenomeOcean-500M GENEB vector with HF Mistral."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


RUNTIME_ID = "geneb-genome-ocean-500m"
REPO = "DOEJGI/GenomeOcean-500M"
REQUESTED_REVISION = "main"
REVISION = "0d9f453925aca9c278505cf0103b6b4311092052"
CATALOG_CONTRACT_SHA256 = (
    "6c3a2380a4c09451304684d1fcc4eaa9233f1c3f0f8e874ad6aa78c526f19645"
)
EXTRACTOR_REPO = "darlednik/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "937fc3e45fe38c4288fd05fa4f7407f4252f9c802b8609f3fc3b1c19b33d5eac"
)
CANONICAL_SEQUENCE = "ATTCCGATTCCGATTCCG"
EXPECTED_FILES = {
    ".gitattributes": (
        1519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "LICENSE": (
        2548,
        "7b3295ce7c8bd6899b64ea5287375a26c6c9888664dce6fb4331a2b3532531a4",
    ),
    "README.md": (
        1766,
        "739c44e36031eda54bf48280eda0a343401e77cca8cf1246cfe76b0d1ec7897b",
    ),
    "config.json": (
        778,
        "658f9bec0142adf8994a7bc11581069e854639ae4643db272b937317ce4cbbf2",
    ),
    "generation_config.json": (
        132,
        "cac7ca78f1c75af762fdea735defe1d569fe6beefde013b8e3610977dac70f90",
    ),
    "model.safetensors": (
        1082234168,
        "b6315d77656839e51328d2ba04924fe7585782b7e1edd2fa5649fbbcd31b68cb",
    ),
    "special_tokens_map.json": (
        695,
        "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    ),
    "tokenizer.json": (
        166434,
        "ea61df7c1f41bd98f336d663389f587509e7359a278eb8b126e43f21dc4ba289",
    ),
    "tokenizer_config.json": (
        1136,
        "50757b50caea808b47681dfcb31566b5efaac5ea06c9cd7272903522814df57f",
    ),
}


class OracleError(RuntimeError):
    """Raised when the pinned upstream contract differs."""


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


def load_json(path: Path, label: str) -> Dict[str, Any]:
    def reject_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        value = {}  # type: Dict[str, Any]
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
        raise OracleError("cannot load %s: %s" % (label, error))
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


def validate_sources(receipt_path: Path, snapshot: Path) -> Dict[str, Dict[str, Any]]:
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
    verified = {}  # type: Dict[str, Dict[str, Any]]
    for index, item in enumerate(raw_files):
        label = "source receipt files[%d]" % index
        if not isinstance(item, dict) or set(item) != {"name", "path", "size", "sha256"}:
            raise OracleError("%s fields differ" % label)
        name = normalized_relative(item["name"], label + ".name")
        if name in verified or name not in EXPECTED_FILES:
            raise OracleError("source receipt file set is duplicated or unexpected")
        expected_size, expected_sha256 = EXPECTED_FILES[name]
        if item["size"] != expected_size or item["sha256"] != expected_sha256:
            raise OracleError("source receipt pinned size/SHA256 differs for %s" % name)
        locator = Path(item["path"])
        snapshot_path = snapshot / name
        for candidate, candidate_label in (
            (locator, "receipt locator"),
            (snapshot_path, "snapshot file"),
        ):
            if not candidate.is_file():
                raise OracleError("%s is missing for %s" % (candidate_label, name))
            if (
                candidate.stat().st_size != expected_size
                or sha256_file(candidate) != expected_sha256
            ):
                raise OracleError("%s size/SHA256 differs for %s" % (candidate_label, name))
        verified[name] = {"size": expected_size, "sha256": expected_sha256}
    snapshot_names = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if snapshot_names != set(EXPECTED_FILES):
        raise OracleError("snapshot contains missing or extra regular files")
    return dict(sorted(verified.items()))


def read_canonical_sequence(path: Path) -> str:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise OracleError("cannot read canonical input: %s" % error)
    if len(lines) < 2 or not lines[0].startswith(">") or not lines[0][1:]:
        raise OracleError("canonical input must be one named FASTA record")
    if any(line.startswith(">") for line in lines[1:]):
        raise OracleError("canonical input must contain exactly one FASTA record")
    sequence = "".join(lines[1:])
    if sequence != CANONICAL_SEQUENCE:
        raise OracleError("canonical GenomeOcean-500M oracle sequence differs")
    return sequence


def package_lock() -> List[str]:
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
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve(strict=True)
    receipt = args.receipt.expanduser().resolve(strict=True)
    input_path = args.input.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise OracleError("oracle output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    source_files = validate_sources(receipt, snapshot)
    sequence = read_canonical_sequence(input_path)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=True,
        padding_side="left",
    )
    if (
        tokenizer.padding_side != "left"
        or tokenizer.model_max_length != 1000000000000000019884624838656
    ):
        raise OracleError("official tokenizer context/padding contract differs")
    encoded = tokenizer(
        [sequence],
        padding=True,
        truncation=True,
        max_length=1024,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if (
        input_ids.tolist() != [[1, 30, 1104, 1104, 22, 2]]
        or attention_mask.tolist() != [[1, 1, 1, 1, 1, 1]]
    ):
        raise OracleError("official tokenizer IDs/padding differ")

    model, loading = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise OracleError("official checkpoint loading info differs: %r" % loading)
    if (
        model.__class__.__module__ != "transformers.models.mistral.modeling_mistral"
        or model.__class__.__name__ != "MistralForCausalLM"
        or getattr(model.config, "_attn_implementation", None) != "sdpa"
    ):
        raise OracleError("official Mistral implementation/attention path differs")
    model.to(device="cpu", dtype=torch.float32).eval()
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = result.hidden_states
        if hidden_states is None or len(hidden_states) != 15:
            raise OracleError("official hidden-state tap count differs")
        hidden = hidden_states[-1]
        if list(hidden.shape) != [1, 6, 1536] or hidden.dtype != torch.float32:
            raise OracleError("official last hidden-state shape/dtype differs")
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    values = pooled.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    flat = [float(value) for value in values.reshape(-1)]
    if len(flat) != 1536 or any(not math.isfinite(value) for value in flat):
        raise OracleError("official pooled embedding is invalid")

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
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "transformers": {
            "version": importlib.metadata.version("transformers"),
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "safetensors_version": importlib.metadata.version("safetensors"),
            "local_files_only": True,
            "trust_remote_code": True,
            "model_class": "%s.%s"
            % (model.__class__.__module__, model.__class__.__name__),
            "tokenizer_class": "%s.%s"
            % (tokenizer.__class__.__module__, tokenizer.__class__.__name__),
            "attention_implementation": model.config._attn_implementation,
        },
        "source_files": source_files,
    }
    generator = Path(__file__).resolve(strict=True)
    provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-reference",
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_file_manifest_sha256": sha256_bytes(canonical_json(source_manifest)),
        "checkpoint_sha256": EXPECTED_FILES["model.safetensors"][1],
        "official_standard_transformers_code": True,
        "independent_of_evo_native_runtime": True,
        "generator_sha256": sha256_file(generator),
        "extractor_repo": EXTRACTOR_REPO,
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_module": "embedding_pipeline/extractors/genomeocean.py",
        "extractor_class": "GenomeOceanExtractor",
        "extractor_sha256": EXTRACTOR_SHA256,
        "hidden_tap": "outputs.hidden_states[-1]",
        "pooling": "attention-mask-mean",
        "special_tokens": "include",
        "padding_side": "left",
        "extractor_max_length": 1024,
        "use_cache": False,
        "nonpadding_tokens": 6,
    }
    vector = {
        "schema_version": 1,
        "kind": "geneb-independent-oracle-vector",
        "runtime_id": RUNTIME_ID,
        "input_sha256": sha256_file(input_path),
        "backend": "cpu",
        "profile": "cpu-f32",
        "values": flat,
        "environment_lock": environment,
        "provenance": provenance,
    }
    vector_path = output / "input-0.independent-oracle-vector.json"
    vector_path.write_bytes(canonical_json(vector))
    npy_path = output / "input-0.attention-mask-mean.f32.npy"
    np.save(npy_path, values, allow_pickle=False)
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "file_manifest_sha256": provenance["source_file_manifest_sha256"],
        },
        "input": {
            "sha256": vector["input_sha256"],
            "sequence_length": len(sequence),
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
        },
        "output": {
            "shape": list(values.shape),
            "npy_sha256": sha256_file(npy_path),
            "raw_f32_sha256": sha256_bytes(values.tobytes(order="C")),
            "vector_sha256": sha256_file(vector_path),
            "first_16": flat[:16],
        },
        "environment_lock": environment,
        "provenance": provenance,
    }
    report_path = output / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))
    print(
        json.dumps(
            {
                "report_sha256": sha256_file(report_path),
                "vector_sha256": sha256_file(vector_path),
                "npy_sha256": sha256_file(npy_path),
                "shape": list(values.shape),
                "first_16": flat[:16],
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
            "generate_geneb_genome_ocean_500m_upstream_oracle: error: %s" % error
        )
