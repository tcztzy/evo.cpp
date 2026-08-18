#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate normalized HyenaDNA CPU oracles from pinned upstream code.

The pinned GENEB extractor left-pads a batch but deliberately forwards only
``input_ids`` to HyenaDNA.  The causal model does not accept an attention mask,
so pad tokens change valid hidden rows.  This generator records that reference
defect and emits an independent normalized oracle by running each record alone,
where no padding enters the model.  It never imports the evo.cpp runtime or a
converted runtime artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Tuple
from urllib.parse import urlsplit

import numpy as np
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer


EXTRACTOR_REPO = "darlednik/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "b449660cc7c2f0efb06e771ca0206f890b7cd7a523bbf24b877e7494449f8941"
)
NORMALIZATION_PATCH_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
SHARED_SOURCE_FILES = {
    ".gitattributes": (
        1519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "README.md": (
        6911,
        "18ee2eeab45dd561dfd266e8756d267389cef724839e4250f093acfb7c9c57b8",
    ),
    "configuration_hyena.py": (
        3093,
        "29cb6cae67f5010e279d67349e688663043f0bd6791a09c36d96da8cac3668a6",
    ),
    "modeling_hyena.py": (
        22584,
        "d78029cacea2c9259f1293fdb50b60f90872089469185e27d5c769086c475607",
    ),
    "special_tokens_map.json": (
        971,
        "026831cec9f2ac5e501aa9d3a8b3a93c021aebe55813574c493e40b49ad28582",
    ),
    "tokenization_hyena.py": (
        4057,
        "d424da2f794958b0360e033bbf5edc9ee4e3e3126e883093621348f07761e408",
    ),
}
MODEL_SPECS = {
    "geneb-hyenadna-medium-160k": {
        "repo": "LongSafari/hyenadna-medium-160k-seqlen-hf",
        "revision": "7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce",
        "max_length": 160002,
        "input": b">t39-hyenadna-medium\nACGTNACGT\n",
        "source_files": {
            **SHARED_SOURCE_FILES,
            "config.json": (
                984,
                "257651ce75a3239e9e3c76c43ce7dd0d5b78c1023aef4dbd4536564aa428f379",
            ),
            "model.safetensors": (
                56971992,
                "16685b36caf5d144da391f1b540ed47e89d79b3efd38899325492f174b8e852a",
            ),
            "tokenizer_config.json": (
                1482,
                "dd519d81a1891aa4cb1933e4787b60fa449ff818ab4c531953899dc538633429",
            ),
        },
    },
    "geneb-hyenadna-large-1m": {
        "repo": "LongSafari/hyenadna-large-1m-seqlen-hf",
        "revision": "0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446",
        "max_length": 1000002,
        "input": b">t39-hyenadna-large\nACGTNACGT\n",
        "source_files": {
            **SHARED_SOURCE_FILES,
            "config.json": (
                982,
                "e63e0dcb3bc0ff6f5207b3e0dbad48f6f0933dde42a47872cf1ab86e4b43f038",
            ),
            "model.safetensors": (
                218252304,
                "deafb53209bfafb314d14bd81108546a34d473c07ed7ab7a355674376b56ad12",
            ),
            "tokenizer_config.json": (
                1483,
                "7ceea744a19e155f056818b05e80195093ade809614964d8016ce2505c1485d4",
            ),
        },
    },
}


class OracleError(RuntimeError):
    """Raised when pinned source or normalized execution differs."""


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
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            portable(key, label + " key")
            portable(item, "%s.%s" % (label, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            portable(item, "%s[%d]" % (label, index))
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


def package_lock() -> List[str]:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append(
                "%s==%s"
                % (name.lower().replace("_", "-"), distribution.version)
            )
    return sorted(set(packages))


def validate_sources(
    snapshot: Path, extractor: Path, spec: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    if not snapshot.is_dir() or snapshot.name != spec["revision"]:
        raise OracleError("pinned HyenaDNA snapshot revision differs")
    result = {}  # type: Dict[str, Dict[str, Any]]
    for name, (size, digest) in spec["source_files"].items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise OracleError("pinned HyenaDNA source differs: " + name)
        result[name] = {"size": size, "sha256": digest}
    if extractor.is_symlink() or not extractor.is_file():
        raise OracleError("pinned GENEB extractor must be a nonsymlink file")
    extractor_payload = extractor.read_bytes()
    if sha256_bytes(extractor_payload) != EXTRACTOR_SHA256:
        raise OracleError("pinned GENEB HyenaDNA extractor differs")
    extractor_text = extractor_payload.decode("utf-8")
    required_fragments = (
        "padding=True,",
        "add_special_tokens=False,",
        'inputs = {"input_ids": enc["input_ids"].to(self.model.device)}',
        'mask = enc["attention_mask"]',
    )
    if any(extractor_text.count(fragment) != 1 for fragment in required_fragments):
        raise OracleError("pinned GENEB HyenaDNA batching semantics differ")
    result["GENEB/embedding_pipeline/extractors/hyenadna.py"] = {
        "size": len(extractor_payload),
        "sha256": EXTRACTOR_SHA256,
    }
    return result


def f32_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return array.astype("<f4", copy=False).tobytes(order="C")


def i64_bytes(values: List[int]) -> bytes:
    return b"".join(struct.pack("<q", value) for value in values)


def tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    return {
        "dtype": "float32",
        "shape": list(tensor.shape),
        "raw_little_endian_f32_sha256": sha256_bytes(f32_bytes(tensor)),
        "first_16_decimal": [float(value) for value in flat[:16]],
        "first_16_hex": [float(value).hex() for value in flat[:16]],
        "all_finite": bool(torch.isfinite(flat).all().item()),
    }


def save_npy(path: Path, tensor: torch.Tensor) -> Dict[str, Any]:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    return {"file": path.name, "file_sha256": sha256_file(path), **tensor_summary(tensor)}


def hidden_and_pool(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output = model(input_ids=input_ids, output_hidden_states=True)
    hidden = output.hidden_states[-1].to(dtype=torch.float32)
    mask = attention_mask.to(dtype=torch.float32).unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return hidden, pooled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=sorted(MODEL_SPECS), default="geneb-hyenadna-medium-160k"
    )
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise OracleError(variable + "=1 is required")
    spec = MODEL_SPECS[args.model]
    runtime_id = args.model
    source_repo = spec["repo"]
    source_revision = spec["revision"]
    inputs = (
        ("input-0", spec["input"], "ACGTNACGT"),
        ("defect-short", b">hyena-padding-defect\nTGCA\n", "TGCA"),
    )
    snapshot = args.snapshot.absolute()
    extractor = args.extractor.resolve(strict=True)
    source_files = validate_sources(snapshot, extractor, spec)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
        output_hidden_states=True,
    ).to(device="cpu", dtype=torch.float32).eval()
    model_name = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    tokenizer_name = "%s.%s" % (
        tokenizer.__class__.__module__,
        tokenizer.__class__.__name__,
    )
    if not model_name.endswith("modeling_hyena.HyenaDNAForSequenceClassification"):
        raise OracleError("pinned upstream HyenaDNA model class differs")
    if not tokenizer_name.endswith("tokenization_hyena.HyenaDNATokenizer"):
        raise OracleError("pinned upstream HyenaDNA tokenizer class differs")
    if (
        tokenizer.padding_side != "left"
        or tokenizer.model_max_length != spec["max_length"]
    ):
        raise OracleError("pinned HyenaDNA tokenizer padding/context differs")

    sequences = [item[2] for item in inputs]
    batch = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    if batch["input_ids"].tolist()[1][:-4] != [4] * 5:
        raise OracleError("pinned tokenizer did not left-pad the shorter record")
    _, reference_batch_pool = hidden_and_pool(
        model, batch["input_ids"], batch["attention_mask"]
    )

    normalized = []  # type: List[Tuple[torch.Tensor, torch.Tensor, List[int]]]
    for sequence in sequences:
        encoded = tokenizer(
            sequence,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        if not bool(encoded["attention_mask"].all().item()):
            raise OracleError("normalized single-record path unexpectedly padded")
        hidden, pooled = hidden_and_pool(
            model, encoded["input_ids"], encoded["attention_mask"]
        )
        normalized.append((hidden, pooled, encoded["input_ids"][0].tolist()))

    first_difference = float(
        (normalized[0][1] - reference_batch_pool[0:1]).abs().max().item()
    )
    short_difference = float(
        (normalized[1][1] - reference_batch_pool[1:2]).abs().max().item()
    )
    if first_difference != 0.0 or short_difference <= 0.1:
        raise OracleError("pinned HyenaDNA padding defect was not reproduced")

    environment_lock = {
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
            "mps_available_but_unused": bool(torch.backends.mps.is_available()),
        },
        "transformers": {
            "version": transformers.__version__,
            "offline": True,
            "local_files_only": True,
            "trust_remote_code": True,
            "model_class": model_name,
            "tokenizer_class": tokenizer_name,
        },
        "source_files": source_files,
    }
    portable(environment_lock, "environment_lock")
    (output / "environment-lock.json").write_bytes(canonical_json(environment_lock))

    records = []
    for (label, input_bytes, sequence), (hidden, pooled, token_ids) in zip(
        inputs, normalized
    ):
        input_path = output / (label + ".fa")
        input_path.write_bytes(input_bytes)
        npy = save_npy(output / (label + ".normalized-mean.f32.npy"), pooled)
        provenance = {
            "kind": "pinned-upstream-normalized-no-padding-oracle",
            "source_repo": source_repo,
            "source_revision": source_revision,
            "extractor_repo": EXTRACTOR_REPO,
            "extractor_revision": EXTRACTOR_REVISION,
            "extractor_sha256": EXTRACTOR_SHA256,
            "normalization_patch_sha256": NORMALIZATION_PATCH_SHA256,
            "clean_reference_status": "blocked-batch-padding-contamination",
            "reference_padding_side": "left",
            "model_receives_attention_mask": False,
            "normalization": "run each record alone so no padding enters forward",
            "hidden_tap": "post-final-layernorm",
            "pooling": "attention-mask-mean",
        }
        portable(provenance, "provenance")
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": runtime_id,
            "input_sha256": sha256_bytes(input_bytes),
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": [float(value) for value in pooled.reshape(-1)],
            "environment_lock": environment_lock,
            "provenance": provenance,
        }
        portable(vector, "oracle vector")
        vector_path = output / (label + ".independent-oracle-vector.json")
        vector_path.write_bytes(canonical_json(vector))
        records.append(
            {
                "label": label,
                "input_size": len(input_bytes),
                "input_sha256": sha256_bytes(input_bytes),
                "sequence_sha256": sha256_bytes(sequence.encode("ascii")),
                "token_count": len(token_ids),
                "token_ids_little_endian_i64_sha256": sha256_bytes(
                    i64_bytes(token_ids)
                ),
                "hidden_shape": list(hidden.shape),
                "pooled": npy,
                "oracle_vector_file": vector_path.name,
                "oracle_vector_sha256": sha256_file(vector_path),
            }
        )

    report = {
        "schema_version": 1,
        "kind": "geneb-hyenadna-upstream-oracle-report",
        "runtime_id": runtime_id,
        "source_repo": source_repo,
        "source_revision": source_revision,
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_sha256": EXTRACTOR_SHA256,
        "normalization_patch_sha256": NORMALIZATION_PATCH_SHA256,
        "environment_lock_sha256": sha256_file(output / "environment-lock.json"),
        "reference_defect": {
            "status": "reproduced",
            "padding_side": "left",
            "model_receives_attention_mask": False,
            "equal_length_record_max_abs": first_difference,
            "short_record_max_abs": short_difference,
        },
        "records": records,
    }
    portable(report, "report")
    (output / "oracle-report.json").write_bytes(canonical_json(report))
    print(str(output / "oracle-report.json"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OracleError as error:
        print("generate_geneb_hyenadna_upstream_oracle: error: %s" % error, file=sys.stderr)
        raise SystemExit(2)
