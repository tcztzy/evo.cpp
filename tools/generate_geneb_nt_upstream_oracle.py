#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned NT-v2-50M GENEB vector with official HF code."""

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
from transformers import AutoModelForMaskedLM, AutoTokenizer


RUNTIME_ID = "geneb-nt-v2-50m-ms"
REPO = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
REQUESTED_REVISION = "main"
REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"
CATALOG_CONTRACT_SHA256 = (
    "738be05adcc6b3dfc676c1bcc79ee8de9e108fcf8e8ff06d47e24a07424e01ff"
)
EXTRACTOR_REPO = "ultimativity/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "50332b6070b944dd4d26dc2641ebadfea7270513d34294add5f7066295567c43"
)
CANONICAL_SEQUENCE = "ATTCCGATTCCGATTCCG"
EXPECTED_FILES = {
    ".gitattributes": (1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
    "README.md": (6339, "e526d7b98f106bc2ca9ba73fa166ff5fd62853812e9757a6692eeedf25e42923"),
    "config.json": (1064, "e20f497248c7cb264c7cd4582dbcfd52dc4cbf74a97fc711559b8c8f71c635db"),
    "esm_config.py": (14876, "a44e859baa08465ecdcd76b0a73f6e4fe011245a212931c403de1149ae9613ec"),
    "jax_model/hyperparams.json": (659, "d7b8c2682aaefbd1d693fa81b63b21e1a080e4b19fd263060d0c2990c23afad3"),
    "jax_model/pytree_ckpt.joblib": (219437507, "3015d19197e0d8e4dcddc03bd819e71491cdeca604ec69633cdb9e662566791f"),
    "model.safetensors": (223642688, "17e75af297556ea56828716d8aa539e8f12b8b625547204b74171ab91aa33569"),
    "modeling_esm.py": (58205, "f2b003f45d2fa4f94e92d8bbef927b719dfdd7ab7c7a95743daa2b10ab140cb3"),
    "pytorch_model.bin": (223682957, "3b177d140664f3157ecd262825b5caca5855c10ae9c1ea43445c349f5e09c0b8"),
    "special_tokens_map.json": (101, "d6dc30bf018166daab248b0abf7efda6fd1b1e0a2d1bee5b31b23db2ebdaee77"),
    "tokenizer_config.json": (129, "253d338919eba938e50b776f3243cc739462c207fe64e3d2e81cc5e681bee45b"),
    "vocab.txt": (28718, "c00e0ad166d6ab3f7540ebc92270392e581bb3106763412f29034b49323e1052"),
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
    if set(verified) != set(EXPECTED_FILES):
        raise OracleError("source receipt file set differs")
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
        raise OracleError("canonical NT-v2-50M oracle sequence differs")
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
        snapshot, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.model_max_length != 2048 or tokenizer.padding_side != "right":
        raise OracleError("official tokenizer context/padding contract differs")
    encoded = tokenizer(
        [sequence],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if (
        list(input_ids.shape) != [1, 2048]
        or list(attention_mask.shape) != [1, 2048]
        or input_ids[0, :4].tolist() != [3, 369, 369, 369]
        or int(attention_mask.sum().item()) != 4
        or any(value != 1 for value in input_ids[0, 4:].tolist())
    ):
        raise OracleError("official tokenizer IDs/padding differ")

    model, loading = AutoModelForMaskedLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise OracleError("official checkpoint loading info differs: %r" % loading)
    model.to(device="cpu", dtype=torch.float32).eval()
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = result.hidden_states
        if hidden_states is None or len(hidden_states) != 13:
            raise OracleError("official hidden-state tap count differs")
        hidden = hidden_states[-1]
        if list(hidden.shape) != [1, 2048, 512] or hidden.dtype != torch.float32:
            raise OracleError("official last hidden-state shape/dtype differs")
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    values = pooled.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    flat = [float(value) for value in values.reshape(-1)]
    if len(flat) != 512 or any(not math.isfinite(value) for value in flat):
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
        "official_remote_modeling_code": True,
        "independent_of_evo_native_runtime": True,
        "generator_sha256": sha256_file(generator),
        "extractor_repo": EXTRACTOR_REPO,
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_module": "embedding_pipeline/extractors/nt.py",
        "extractor_class": "NucleotideTransformerExtractor",
        "extractor_sha256": EXTRACTOR_SHA256,
        "hidden_tap": "outputs.hidden_states[-1]",
        "pooling": "attention-mask-mean",
        "special_tokens": "include",
        "model_max_length": 2048,
        "nonpadding_tokens": 4,
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
    vector_path = output / "nt-v2-50m-ms.independent-oracle-vector.json"
    vector_path.write_bytes(canonical_json(vector))
    npy_path = output / "nt-v2-50m-ms.attention-mask-mean.f32.npy"
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
            "input_ids_prefix": input_ids[0, :8].tolist(),
            "nonpadding_tokens": int(attention_mask.sum().item()),
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
        raise SystemExit("generate_geneb_nt_upstream_oracle: error: %s" % error)
