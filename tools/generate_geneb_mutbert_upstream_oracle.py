#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a pinned, independent GENEB MutBERT upstream oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import struct
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer


REPO = "JadenLong/MutBERT"
REVISION = "b68d8d6c9ccd8167639b25fb979cbd39a5c5c60c"
EXTRACTOR_REPO = "ultimativity/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = "1b00957bbe5a2010a27405d610724136d189b4ab553f802588aec1bdf0feeb31"
EXTRACTOR_PYPROJECT_SHA256 = (
    "616a15c0a751eb14687fefe691cd624fbf4cc104605f167a5782cb29d16060e0"
)
INPUTS = (
    ("input-0", b"ACGTNACGTNACGTN"),
    ("input-1", b"AAAACCCCGGGGTTTTACGTNACGTNNNNACGTTGCAACGTACGT"),
)
LONG_DIFFERENTIAL_INPUT = b"A" * 513
PREDECLARED_TOLERANCE = {
    "max_abs": 5e-5,
    "mean_abs": 5e-6,
    "min_cosine": 0.99999999,
}
EXPECTED_LOADING = {
    "missing_keys": [],
    "unexpected_keys": [
        "cls.predictions.bias",
        "cls.predictions.decoder.bias",
        "cls.predictions.decoder.weight",
        "cls.predictions.transform.LayerNorm.bias",
        "cls.predictions.transform.LayerNorm.weight",
        "cls.predictions.transform.dense.bias",
        "cls.predictions.transform.dense.weight",
    ],
    "mismatched_keys": [],
    "error_msgs": [],
}
SOURCE_FILES = {
    ".gitattributes": (
        1519,
        "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    ),
    "README.md": (
        3295,
        "917cc255ff7601fc48c2f83b99d6c0210e2d0342fb6397816459e04cac9be3a9",
    ),
    "config.json": (
        855,
        "543cf23dba83fb917c9bec4d92db330e9609fa76c3e629675024fe5e6ccd3d33",
    ),
    "modeling_mutbert.py": (
        46585,
        "aaeabe7f94be71980e6db9e98f1f3019eea2a526ab64b0ed3ccccb16d3a415db",
    ),
    "pytorch_model.bin": (
        342710382,
        "73a10975b93bed0f259698985f8420b86c7c5dedcf8b992053643085dd9d1140",
    ),
    "tokenizer.json": (
        2623,
        "fd5232a7a128cfb88fcb2ea71568cffa92c24aa0b3ef129e73e9f40e5bf25b12",
    ),
    "tokenizer_config.json": (
        158,
        "f9d18c81f4dd9dd7db02e9f27cc1203228147d890bfce9167c3af6465ff5b769",
    ),
    "vocab.txt": (
        38,
        "57fb2e0c852a0408f870b96641df38becd8b28e3571db5fc9343399a62878624",
    ),
}


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


def i64_bytes(values: list[int]) -> bytes:
    return b"".join(struct.pack("<q", value) for value in values)


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    return {
        "dtype": "float32",
        "shape": list(tensor.shape),
        "raw_little_endian_f32_sha256": sha256_bytes(f32_bytes(tensor)),
        "first_16_decimal": [float(value) for value in flat[:16]],
        "first_16_hex": [float(value).hex() for value in flat[:16]],
        "all_finite": bool(torch.isfinite(flat).all().item()),
    }


def numeric_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
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


def package_lock() -> list[str]:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append(
                "%s==%s" % (name.lower().replace("_", "-"), distribution.version)
            )
    return sorted(set(packages))


def save_npy(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    return {"file": path.name, "file_sha256": sha256_file(path), **tensor_summary(tensor)}


def validate_sources(
    snapshot_argument: Path,
    extractor_argument: Path,
    pyproject_argument: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    snapshot = snapshot_argument.resolve(strict=True)
    extractor = extractor_argument.resolve(strict=True)
    pyproject = pyproject_argument.resolve(strict=True)
    if snapshot.name != REVISION:
        raise RuntimeError("MutBERT snapshot directory does not name the pinned revision")
    source_manifest = {}
    for name, (expected_size, expected_digest) in SOURCE_FILES.items():
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError("missing pinned MutBERT source file: " + name)
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != expected_size or digest != expected_digest:
            raise RuntimeError("pinned MutBERT source file differs: " + name)
        source_manifest[name] = {"size": size, "sha256": digest}
    if sha256_file(extractor) != EXTRACTOR_SHA256:
        raise RuntimeError("pinned GENEB MutBERT extractor differs")
    if sha256_file(pyproject) != EXTRACTOR_PYPROJECT_SHA256:
        raise RuntimeError("pinned GENEB embedding environment declaration differs")
    return snapshot, extractor, pyproject, source_manifest


def summarize_ids(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "first_16": values[:16],
        "last_16": values[-16:],
        "little_endian_i64_sha256": sha256_bytes(i64_bytes(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--extractor-pyproject", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    snapshot, extractor, pyproject, source_manifest = validate_sources(
        args.snapshot, args.extractor, args.extractor_pyproject
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(variable + "=1 is required")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        use_fast=True,
        local_files_only=True,
        revision=REVISION,
    )
    config = AutoConfig.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        revision=REVISION,
    )
    model, loading_info = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=True,
        revision=REVISION,
        output_loading_info=True,
    )
    model.to(device="cpu", dtype=torch.float32).eval()
    model_name = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    config_name = "%s.%s" % (config.__class__.__module__, config.__class__.__name__)
    tokenizer_name = "%s.%s" % (
        tokenizer.__class__.__module__, tokenizer.__class__.__name__
    )
    if REVISION not in model_name or not model_name.endswith(
        ".modeling_mutbert.RoPEBertModel"
    ):
        raise RuntimeError("MutBERT did not select the pinned remote model")
    if REVISION not in config_name or not config_name.endswith(
        ".modeling_mutbert.RoPEBertConfig"
    ):
        raise RuntimeError("MutBERT did not select the pinned remote config")
    if tokenizer_name != "transformers.tokenization_utils_fast.PreTrainedTokenizerFast":
        raise RuntimeError("MutBERT tokenizer class differs")
    loading = {
        key: sorted(str(value) for value in loading_info.get(key, []))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if loading != EXPECTED_LOADING:
        raise RuntimeError("pinned remote MutBERT loading contract differs: %r" % loading)
    if sum(parameter.numel() for parameter in model.parameters()) != 85064448:
        raise RuntimeError("pinned remote MutBERT parameter count differs")
    module_path = Path(inspect.getsourcefile(model.__class__) or "")
    if sha256_file(module_path) != source_manifest["modeling_mutbert.py"]["sha256"]:
        raise RuntimeError("loaded remote modeling code differs from pinned source")
    if int(tokenizer.vocab_size) != 9 or len(tokenizer) != 9:
        raise RuntimeError("pinned MutBERT tokenizer vocabulary size differs")
    if tokenizer.backend_tokenizer.truncation is not None:
        raise RuntimeError("MutBERT source tokenizer unexpectedly embeds truncation")
    if tokenizer.backend_tokenizer.padding is not None:
        raise RuntimeError("MutBERT source tokenizer unexpectedly embeds padding")
    if int(tokenizer.model_max_length) <= 10**20:
        raise RuntimeError("B89 infinite tokenizer max-length sentinel differs")

    # B89: this is the exact extractor call, and it deliberately has no max_length.
    long_sequence = LONG_DIFFERENTIAL_INPUT.decode("ascii")
    long_reference = tokenizer(
        [long_sequence],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    long_normalized = tokenizer(
        [long_sequence],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt",
    )
    long_reference_ids = [int(value) for value in long_reference["input_ids"][0]]
    long_normalized_ids = [int(value) for value in long_normalized["input_ids"][0]]
    if len(long_reference_ids) != 515 or len(long_normalized_ids) != 512:
        raise RuntimeError("B89 >512 differential token counts differ")
    if long_reference_ids[-1] != 2 or long_normalized_ids[-1] != 2:
        raise RuntimeError("B89 differential lost the SEP special token")

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
            "model_trust_remote_code": True,
            "tokenizer_trust_remote_code": False,
            "revision": REVISION,
            "config_class": config_name,
            "model_class": model_name,
            "tokenizer_class": tokenizer_name,
            "remote_modeling_sha256": sha256_file(module_path),
            "attention_implementation": "official remote RoPEBertSdpaAttention",
        },
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
        # This call is byte-for-byte equivalent in options to the pinned extractor.
        encoded = tokenizer(
            [sequence],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        evidence_encoding = tokenizer(
            [sequence],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        for name in ("input_ids", "token_type_ids", "attention_mask"):
            if name in encoded and not torch.equal(encoded[name], evidence_encoding[name]):
                raise RuntimeError("token evidence call changed " + name)
        input_ids = encoded["input_ids"].to(device="cpu")
        attention_mask = encoded["attention_mask"].to(device="cpu")
        one_hot = F.one_hot(input_ids, num_classes=len(tokenizer)).to(torch.float32)
        masked_one_hot = one_hot * attention_mask.unsqueeze(-1).to(one_hot.dtype)
        with torch.no_grad():
            output_first = model(
                masked_one_hot, attention_mask=attention_mask, return_dict=True
            )
            output_second = model(
                masked_one_hot, attention_mask=attention_mask, return_dict=True
            )
        hidden_first = output_first.last_hidden_state.to(dtype=torch.float32)
        hidden_second = output_second.last_hidden_state.to(dtype=torch.float32)
        mask = attention_mask.unsqueeze(-1).to(hidden_first.dtype)
        pooled_first = (hidden_first * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled_second = (hidden_second * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if not torch.equal(hidden_first, hidden_second) or not torch.equal(
            pooled_first, pooled_second
        ):
            raise RuntimeError("MutBERT upstream forward is not bitwise deterministic")
        upstream_pooler_comparison = numeric_comparison(
            pooled_first, output_first.pooler_output.to(dtype=torch.float32)
        )
        if not upstream_pooler_comparison["bitwise_equal"]:
            raise RuntimeError("GENEB manual pooling differs from upstream mean pooler")

        ids = [int(value) for value in input_ids[0].tolist()]
        attention = [int(value) for value in attention_mask[0].tolist()]
        token_type_ids = [int(value) for value in encoded["token_type_ids"][0].tolist()]
        special_mask = [
            int(value) for value in evidence_encoding["special_tokens_mask"][0].tolist()
        ]
        tokens = {
            "input_ids": ids,
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "attention_mask": attention,
            "token_type_ids": token_type_ids,
            "special_tokens_mask": special_mask,
            "input_ids_little_endian_i64_sha256": sha256_bytes(i64_bytes(ids)),
            "attention_mask_u8_sha256": sha256_bytes(bytes(attention)),
            "effective_token_count": sum(attention),
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
            "tokens": {"file": token_path.name, "file_sha256": sha256_file(token_path), **tokens},
            "one_hot": tensor_summary(one_hot),
            "masked_one_hot": tensor_summary(masked_one_hot),
            "hidden": save_npy(hidden_path, hidden_first),
            "embedding": save_npy(embedding_path, pooled_first),
            "upstream_pooler_comparison": upstream_pooler_comparison,
            "repeat_hidden": numeric_comparison(hidden_first, hidden_second),
            "repeat_embedding": numeric_comparison(pooled_first, pooled_second),
        }
        records.append(record)
        pooled_values.append(pooled_first)

    generator_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    b89 = {
        "status": "passed",
        "reference_claim": "fixed-short-inputs-only",
        "long_input_reference_parity_claimed": False,
        "tokenizer_model_max_length": int(tokenizer.model_max_length),
        "extractor_call_has_explicit_max_length": False,
        "input": {
            "raw_base_count": len(LONG_DIFFERENTIAL_INPUT),
            "sha256": sha256_bytes(LONG_DIFFERENTIAL_INPUT),
        },
        "reference_no_explicit_max_length": summarize_ids(long_reference_ids),
        "normalized_explicit_max_length_512": summarize_ids(long_normalized_ids),
    }
    report = {
        "schema_version": 1,
        "kind": "geneb-mutbert-official-upstream-oracle-report",
        "source": {"repo": REPO, "revision": REVISION, "files": source_manifest},
        "extractor": {
            "repo": EXTRACTOR_REPO,
            "revision": EXTRACTOR_REVISION,
            "file": "embedding_pipeline/extractors/mutbert.py",
            "sha256": sha256_file(extractor),
            "pyproject_file": "embedding_pipeline/pyproject.toml",
            "pyproject_sha256": sha256_file(pyproject),
        },
        "semantics": {
            "preset": "geneb-v4-normalized",
            "oracle_execution_call": "exact-reference",
            "reference_exact_for_fixed_short_inputs": True,
            "normalized_short_inputs_equivalent_to_reference": True,
            "reference_status": "blocked-long-input-semantics",
            "reference_long_input_parity_claimed": False,
            "normalized_runtime_max_tokens": 512,
            "batch_size": 1,
            "add_special_tokens": True,
            "padding": True,
            "truncation": True,
            "explicit_max_length": None,
            "one_hot_num_classes": 9,
            "zero_padded_one_hot_rows": True,
            "hidden_tap": "last_hidden_state",
            "pooling": "attention-mask-mean-including-special-tokens",
            "model_trust_remote_code": True,
            "tokenizer_trust_remote_code": False,
            "model_eval": True,
            "torch_no_grad": True,
            "independent_of_evo_native_runtime": True,
        },
        "predeclared_native_tolerance": PREDECLARED_TOLERANCE,
        "b89_long_input_differential": b89,
        "tokenizer": {
            "vocab_size": int(tokenizer.vocab_size),
            "length_including_added_tokens": len(tokenizer),
            "model_max_length": int(tokenizer.model_max_length),
            "padding_side": tokenizer.padding_side,
            "backend_truncation": tokenizer.backend_tokenizer.truncation,
            "backend_padding": tokenizer.backend_tokenizer.padding,
            "execution_truth": "tokenizer.json",
            "vocab_txt_used": False,
        },
        "model": {
            "class": model_name,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "loading_info": loading,
            "output_container": "BaseModelOutputWithPooling",
            "remote_modeling_sha256": sha256_file(module_path),
        },
        "environment_lock_sha256": sha256_file(environment_path),
        "generator_sha256": generator_sha256,
        "records": records,
    }
    report_path = output / "oracle-report.json"
    portable(report, "oracle report")
    report_path.write_bytes(canonical_json(report))

    for index, (label, _) in enumerate(INPUTS):
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": "geneb-mutbert",
            "input_sha256": records[index]["input"]["sha256"],
            "backend": "cpu",
            "profile": "cpu-f32",
            "values": [float(value) for value in pooled_values[index].contiguous().view(-1)],
            "environment_lock": environment_lock,
            "provenance": {
                "oracle_contract": "geneb-independent-oracle-v1",
                "benchmark_semantics": "geneb-v4-normalized",
                "oracle_execution_call": "exact-reference",
                "reference_exact_for_fixed_short_input": True,
                "normalized_short_input_equivalent_to_reference": True,
                "reference_status": "blocked-long-input-semantics",
                "reference_long_input_parity_claimed": False,
                "normalized_runtime_max_tokens": 512,
                "source_repo": REPO,
                "source_revision": REVISION,
                "checkpoint_sha256": source_manifest["pytorch_model.bin"]["sha256"],
                "tokenizer_sha256": source_manifest["tokenizer.json"]["sha256"],
                "extractor_repo": EXTRACTOR_REPO,
                "extractor_revision": EXTRACTOR_REVISION,
                "extractor_file": "embedding_pipeline/extractors/mutbert.py",
                "extractor_sha256": sha256_file(extractor),
                "official_hf_modeling_code": False,
                "official_remote_modeling_code": True,
                "model_trust_remote_code": True,
                "tokenizer_trust_remote_code": False,
                "model_class": model_name,
                "batch_size": 1,
                "one_hot_num_classes": 9,
                "hidden_tap": "last_hidden_state",
                "pooling": "attention-mask-mean-including-special-tokens",
                "predeclared_native_tolerance": PREDECLARED_TOLERANCE,
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
                "b89": b89,
                "environment_lock_sha256": sha256_file(environment_path),
                "generator_sha256": generator_sha256,
                "manifest_sha256": sha256_file(manifest_path),
                "records": [
                    {
                        "label": record["label"],
                        "input_sha256": record["input"]["sha256"],
                        "input_ids": record["tokens"]["input_ids"],
                        "one_hot_sha256": record["masked_one_hot"][
                            "raw_little_endian_f32_sha256"
                        ],
                        "embedding_sha256": record["embedding"][
                            "raw_little_endian_f32_sha256"
                        ],
                    }
                    for record in records
                ],
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
