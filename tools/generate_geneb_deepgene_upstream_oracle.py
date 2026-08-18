#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned DeepGene RoFormer-only GENEB oracle offline."""

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
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit

import numpy as np
import tokenizers
import torch
import transformers


RUNTIME_ID = "geneb-deepgene"
SOURCE_REPO = "wds-seu/DeepGene"
SOURCE_REVISION = "486343e5212361d6cd7ed03c624f430ed3d5f02e"
EXTRACTOR_REPO = "darlednik/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "46d41164cf36abcd10dc6e14c5c0ba64acd8fa4d70776c4f34bb08328f683ff6"
)
MODEL_FILES = {
    "config.json": (
        718,
        "09a9d102fa421711e4da2cf1018e042d98cce189ba3ee27c49cf0851f1f626f2",
    ),
    "pytorch_model.bin": (
        355268434,
        "4717e43cfe1036c5abc9aedd9ed158264aa1e8183e8cbe1375d4503c373e99d6",
    ),
}
TOKENIZER_SIZE = 167908
TOKENIZER_SHA256 = (
    "5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3"
)
MODEL_CODE = {
    "configuration_roformer.py": (
        "28672584950f4bf08975024cdf529ab9cc3aa4497ef4f9a1b3863b2e971cc68e"
    ),
    "modeling_roformer.py": (
        "1849ee8eb6931d5479083eaf01ec7c2bff124555423c30c337e0bfe889fab3e4"
    ),
}
INPUTS = (
    ("input-0", b"ACGTNACGTNACGTN"),
    ("input-1", b"AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT"),
)


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, size: int, digest: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError("missing pinned " + label)
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise RuntimeError("pinned " + label + " differs")


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
        raise RuntimeError(label + " contains a local absolute filesystem path")


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


def numeric_comparison(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    left64 = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right64 = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    difference = (left64 - right64).abs()
    denominator = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    return {
        "bitwise_equal": bool(torch.equal(left, right)),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "cosine": float((torch.dot(left64, right64) / denominator).item()),
    }


def package_lock() -> List[str]:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append(
                "%s==%s" % (name.lower().replace("_", "-"), distribution.version)
            )
    return sorted(set(packages))


def save_npy(path: Path, tensor: torch.Tensor) -> Dict[str, Any]:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    return {"file": path.name, "file_sha256": sha256_file(path), **tensor_summary(tensor)}


def validate_sources(
    model_argument: Path,
    tokenizer_argument: Path,
    code_argument: Path,
    extractor_argument: Path,
) -> Tuple[Path, Path, Path, Path, Dict[str, Any]]:
    model = model_argument.resolve(strict=True)
    tokenizer_path = tokenizer_argument.resolve(strict=True)
    code = code_argument.resolve(strict=True)
    extractor = extractor_argument.resolve(strict=True)
    manifest = {}  # type: Dict[str, Any]
    for name, (size, digest) in MODEL_FILES.items():
        require_file(model / name, size, digest, "DeepGene " + name)
        manifest[name] = {"size": size, "sha256": digest}
    require_file(tokenizer_path, TOKENIZER_SIZE, TOKENIZER_SHA256, "DeepGene tokenizer")
    manifest["tokenizer.json"] = {
        "size": TOKENIZER_SIZE,
        "sha256": TOKENIZER_SHA256,
    }
    for name, digest in MODEL_CODE.items():
        path = code / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError("pinned DeepGene code differs: " + name)
        manifest[name] = {"size": path.stat().st_size, "sha256": digest}
    if sha256_file(extractor) != EXTRACTOR_SHA256:
        raise RuntimeError("pinned GENEB DeepGene extractor differs")
    return model, tokenizer_path, code, extractor, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--model-code", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    model_dir, tokenizer_path, code, extractor, source_manifest = validate_sources(
        args.model_dir, args.tokenizer, args.model_code, args.extractor
    )
    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(variable + "=1 is required")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    sys.path.insert(0, str(code))
    from modeling_roformer import RoFormerForMaskedLM  # type: ignore

    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    model, loading_info = RoFormerForMaskedLM.from_pretrained(
        str(model_dir), local_files_only=True, output_loading_info=True
    )
    model.to(device="cpu", dtype=torch.float32).eval()
    model_name = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    if model_name != "modeling_roformer.RoFormerForMaskedLM":
        raise RuntimeError("DeepGene official model class differs")
    loading = {
        key: sorted(str(value) for value in loading_info.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if (
        loading["missing_keys"] != ["roformer.encoder.embed_positions.weight"]
        or loading["unexpected_keys"]
        or loading["mismatched_keys"]
        or loading["error_msgs"]
    ):
        raise RuntimeError("DeepGene official loading contract differs: %r" % loading)
    if (
        model.config.vocab_size != 4096
        or model.config.hidden_size != 768
        or model.config.num_hidden_layers != 12
        or model.config.max_position_embeddings != 5120
    ):
        raise RuntimeError("DeepGene official topology differs")

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
            "model_class": model_name,
        },
        "tokenizers": {"version": tokenizers.__version__},
        "source_files": source_manifest,
    }
    portable(environment_lock, "environment_lock")
    environment_path = output / "environment-lock.json"
    environment_path.write_bytes(canonical_json(environment_lock))

    records = []
    pooled_values = []
    for label, input_bytes in INPUTS:
        sequence = input_bytes.decode("ascii")
        input_path = output / (label + ".txt")
        input_path.write_bytes(input_bytes)
        payload = tokenizer.encode(sequence, add_special_tokens=False).ids
        if len(payload) + 2 >= model.config.max_position_embeddings:
            payload = payload[: model.config.max_position_embeddings - 2]
        ids = [1, 2] + [int(value) for value in payload]
        attention_mask = [1] * len(ids)
        position_ids = list(range(len(ids)))
        input_ids = torch.tensor([ids], dtype=torch.long)
        mask = torch.tensor([attention_mask], dtype=torch.bool)
        positions = torch.tensor([position_ids], dtype=torch.long)
        with torch.no_grad():
            first = model.roformer(
                input_ids=input_ids,
                pos_ids=positions,
                attention_mask=mask,
                return_dict=True,
            ).last_hidden_state.to(dtype=torch.float32)
            second = model.roformer(
                input_ids=input_ids,
                pos_ids=positions,
                attention_mask=mask,
                return_dict=True,
            ).last_hidden_state.to(dtype=torch.float32)
        pooled_first = (first * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1)
        pooled_second = (second * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1)
        if not torch.equal(first, second) or not torch.equal(pooled_first, pooled_second):
            raise RuntimeError("DeepGene upstream forward is not bitwise deterministic")
        tokens = {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "input_ids_little_endian_i64_sha256": sha256_bytes(i64_bytes(ids)),
            "attention_mask_u8_sha256": sha256_bytes(bytes(attention_mask)),
            "effective_token_count": len(ids),
            "tensor_token_count": len(ids),
        }
        token_path = output / (label + ".tokens.json")
        token_path.write_bytes(canonical_json(tokens))
        hidden_path = output / (label + ".last-hidden-state.f32.npy")
        embedding_path = output / (label + ".attention-mask-mean.f32.npy")
        record = {
            "label": label,
            "input": {
                "file": input_path.name,
                "size": len(input_bytes),
                "sha256": sha256_bytes(input_bytes),
                "ascii": sequence,
            },
            "tokens": {
                "file": token_path.name,
                "file_sha256": sha256_file(token_path),
                **tokens,
            },
            "hidden": save_npy(hidden_path, first),
            "embedding": save_npy(embedding_path, pooled_first),
            "repeat_hidden": numeric_comparison(first, second),
            "repeat_embedding": numeric_comparison(pooled_first, pooled_second),
        }
        records.append(record)
        pooled_values.append(pooled_first)

    generator_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    report = {
        "schema_version": 1,
        "kind": "geneb-deepgene-official-oracle-report",
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "files": source_manifest,
        },
        "extractor": {
            "repo": EXTRACTOR_REPO,
            "revision": EXTRACTOR_REVISION,
            "file": "embedding_pipeline/extractors/deepgene.py",
            "sha256": sha256_file(extractor),
        },
        "execution": {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
            "manual_special_order": "cls-sep-payload",
            "hidden_tap": "model.roformer(...).last_hidden_state",
            "pooling": "attention-mask-mean",
            "graph_stage_executed": False,
            "model_eval": True,
            "torch_no_grad": True,
            "independent_of_evo_native_runtime": True,
        },
        "model": {
            "class": model_name,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "loading_info": loading,
        },
        "environment_lock_sha256": sha256_file(environment_path),
        "generator_sha256": generator_sha256,
        "records": records,
    }
    portable(report, "oracle report")
    report_path = output / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))

    for index, (label, _) in enumerate(INPUTS):
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": RUNTIME_ID,
            "input_sha256": records[index]["input"]["sha256"],
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": [
                float(value) for value in pooled_values[index].contiguous().view(-1)
            ],
            "environment_lock": environment_lock,
            "provenance": {
                "oracle_contract": "geneb-independent-oracle-v1",
                "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION,
                "checkpoint_sha256": MODEL_FILES["pytorch_model.bin"][1],
                "tokenizer_sha256": TOKENIZER_SHA256,
                "extractor_repo": EXTRACTOR_REPO,
                "extractor_revision": EXTRACTOR_REVISION,
                "extractor_file": "embedding_pipeline/extractors/deepgene.py",
                "extractor_sha256": sha256_file(extractor),
                "official_modeling_code_revision": SOURCE_REVISION,
                "modeling_sha256": MODEL_CODE["modeling_roformer.py"],
                "model_class": model_name,
                "batch_size": 1,
                "hidden_tap": "model.roformer(...).last_hidden_state",
                "pooling": "attention-mask-mean",
                "graph_stage_executed": False,
                "independent_of_evo_native_runtime": True,
                "generator_sha256": generator_sha256,
                "oracle_report_sha256": sha256_file(report_path),
            },
        }
        portable(vector, "oracle vector")
        (output / (label + ".independent-oracle-vector.json")).write_bytes(
            canonical_json(vector)
        )

    files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact-manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "files": [
            {"file": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
    }
    manifest_path = output / "artifact-manifest.json"
    manifest_path.write_bytes(canonical_json(manifest))
    print(
        json.dumps(
            {
                "environment_lock_sha256": sha256_file(environment_path),
                "generator_sha256": generator_sha256,
                "manifest_sha256": sha256_file(manifest_path),
                "report_sha256": sha256_file(report_path),
                "records": [
                    {
                        "label": record["label"],
                        "input_sha256": record["input"]["sha256"],
                        "input_ids": record["tokens"]["input_ids"],
                        "embedding_sha256": record["embedding"][
                            "raw_little_endian_f32_sha256"
                        ],
                    }
                    for record in records
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
