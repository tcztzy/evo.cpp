#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned Agro-NT-1B GENEB CPU-F32 oracle offline."""

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
from transformers import AutoModelForMaskedLM, AutoTokenizer


RUNTIME_ID = "geneb-agro-nt-1b"
REPO = "InstaDeepAI/agro-nucleotide-transformer-1b"
REQUESTED_REVISION = "main"
REVISION = "b0e1ea1f53a2bf5bb29f8eab7a7e553bf06c1ab1"
CATALOG_CONTRACT_SHA256 = (
    "4e9ce24d33fdb5981d0f9bdb2ebd1ac45128e199a6030ab1560a05400ffffecb"
)
GENEB_REPO = "darlednik/GENEB"
GENEB_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SIZE = 2084
EXTRACTOR_SHA256 = (
    "50332b6070b944dd4d26dc2641ebadfea7270513d34294add5f7066295567c43"
)
CANONICAL_SEQUENCE = "ATTCCGATTCCGATTCCG"
EXPECTED_FILES = {
    "config.json": (
        707,
        "4063f8250f32d922611d8b36f0def1bb53b7ae129d6439c50c8b6c340e8eb0bd",
    ),
    "pytorch_model.bin": (
        3965239677,
        "eff67add0325047da01c1e7022d05c0b6cbe1b5cc5580d6e9ccca19ace8eb6e8",
    ),
    "special_tokens_map.json": (
        101,
        "d6dc30bf018166daab248b0abf7efda6fd1b1e0a2d1bee5b31b23db2ebdaee77",
    ),
    "tokenizer_config.json": (
        129,
        "882fac59d6209cfb4e11f5e933195a4b117f8efb6c9aa30fea8ded53ea3f9bdd",
    ),
    "vocab.txt": (
        28718,
        "f1b544e27897936b50bbd925850fa8a08b421c33bb9c26e3711c140c061d0d4c",
    ),
}


class OracleError(RuntimeError):
    """Raised when the pinned Agro upstream contract differs."""


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
    parsed = urlsplit(value)
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or parsed.scheme.lower() == "file"
    ):
        raise OracleError(label + " contains a local absolute filesystem path")


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
            raise OracleError("%s fields differ" % label)
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


def read_canonical_sequence(path: Path) -> str:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise OracleError("cannot read canonical input: %s" % error) from error
    if len(lines) < 2 or not lines[0].startswith(">") or not lines[0][1:]:
        raise OracleError("canonical input must be one named FASTA record")
    if any(line.startswith(">") for line in lines[1:]):
        raise OracleError("canonical input must contain exactly one FASTA record")
    sequence = "".join(lines[1:])
    if sequence != CANONICAL_SEQUENCE:
        raise OracleError("canonical Agro oracle sequence differs")
    return sequence


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
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve(strict=True)
    receipt = args.receipt.resolve(strict=True)
    geneb_repo = args.geneb_repo.resolve(strict=True)
    extractor = args.extractor.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise OracleError("oracle output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    source_files = validate_sources(receipt, snapshot)
    extractor_lock = validate_extractor(geneb_repo, extractor)
    sequence = read_canonical_sequence(input_path)

    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise OracleError(variable + "=1 is required")
    if importlib.metadata.version("transformers") != "4.32.0":
        raise OracleError("Transformers 4.32.0 is required")
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
    tokenizer_class = "%s.%s" % (
        tokenizer.__class__.__module__,
        tokenizer.__class__.__name__,
    )
    if (
        tokenizer_class
        != "transformers.models.esm.tokenization_esm.EsmTokenizer"
        or tokenizer.model_max_length != 1024
        or tokenizer.padding_side != "right"
        or tokenizer.vocab_size != 4107
        or len(tokenizer) != 4107
    ):
        raise OracleError("official tokenizer contract differs")
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
        list(input_ids.shape) != [1, 1024]
        or list(attention_mask.shape) != [1, 1024]
        or input_ids[0, :4].tolist() != [3, 367, 367, 367]
        or int(attention_mask.sum().item()) != 4
        or any(value != 1 for value in input_ids[0, 4:].tolist())
        or int(input_ids.max().item()) >= 4105
    ):
        raise OracleError("official tokenizer IDs/padding differ")

    model, loading = AutoModelForMaskedLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=False,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise OracleError("official checkpoint loading info differs: %r" % loading)
    model_class = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    embeddings = model.get_input_embeddings().weight
    if (
        model_class != "transformers.models.esm.modeling_esm.EsmForMaskedLM"
        or model.config.model_type != "esm"
        or getattr(model.config, "auto_map", None) is not None
        or model.config.vocab_size != 4105
        or list(embeddings.shape) != [4105, 1500]
        or embeddings.dtype != torch.float32
    ):
        raise OracleError("Agro config unexpectedly loaded remote model code")
    model.to(device="cpu", dtype=torch.float32).eval()
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = result.hidden_states
        if hidden_states is None or len(hidden_states) != 41:
            raise OracleError("official hidden-state tap count differs")
        hidden = hidden_states[-1]
        if list(hidden.shape) != [1, 1024, 1500] or hidden.dtype != torch.float32:
            raise OracleError("official last hidden-state shape/dtype differs")
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    values = pooled.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    flat = [float(value) for value in values.reshape(-1)]
    if len(flat) != 1500 or any(not math.isfinite(value) for value in flat):
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
            "model_class": model_class,
            "tokenizer_class": tokenizer_class,
        },
        "source_files": source_files,
        "extractor": extractor_lock,
    }
    provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-reference",
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_file_manifest_sha256": sha256_bytes(canonical_json(source_manifest)),
        "checkpoint_sha256": EXPECTED_FILES["pytorch_model.bin"][1],
        "official_remote_modeling_code": False,
        "official_builtin_transformers_esm": True,
        "independent_of_evo_native_runtime": True,
        "generator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "extractor_repo": GENEB_REPO,
        "extractor_revision": GENEB_REVISION,
        "extractor_module": "embedding_pipeline/extractors/nt.py",
        "extractor_class": "NucleotideTransformerExtractor",
        "extractor_sha256": EXTRACTOR_SHA256,
        "hidden_tap": "outputs.hidden_states[-1]",
        "pooling": "attention-mask-mean",
        "special_tokens": "include",
        "model_max_length": 1024,
        "nonpadding_tokens": 4,
        "source_vocab_size": 4107,
        "model_vocab_size": 4105,
        "accepted_input_max_token_id": int(input_ids.max().item()),
        "input_embedding_shape": list(embeddings.shape),
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
    require_portable(vector, "oracle vector")
    vector_path = output / "agro-nt-1b.independent-oracle-vector.json"
    vector_path.write_bytes(canonical_json(vector))
    npy_path = output / "agro-nt-1b.attention-mask-mean.f32.npy"
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
            "maximum_token_id": int(input_ids.max().item()),
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
    require_portable(report, "oracle report")
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
        raise SystemExit("generate_geneb_agro_nt_upstream_oracle: error: %s" % error)
