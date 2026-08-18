#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate pinned GENA-LM-Large-T2T GENEB oracles with official HF code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import generate_geneb_gena_upstream_oracle as common


RUNTIME_ID = "geneb-gena-lm-large-t2t"
SOURCE_REPO = "AIRI-Institute/gena-lm-bert-large-t2t"
REQUESTED_REVISION = "main"
SOURCE_REVISION = "f997a8a7c1ba5feb6d11de46354a41c51ffe9660"
CATALOG_CONTRACT_SHA256 = (
    "9e63758ee979472bf73f18fd4ebf5aa9a0ca1ecb3203032d7723deec5ba58903"
)
EXTRACTOR_REPO = "ultimativity/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "6a9e5ba5720e93e7078881aabf575944f48959337444e37f19d9a6aa6962516d"
)
EXTRACTOR_PYPROJECT_SHA256 = (
    "616a15c0a751eb14687fefe691cd624fbf4cc104605f167a5782cb29d16060e0"
)
EXPECTED_SOURCE_FILES = {
    ".gitattributes": (
        1477,
        "db7c0371f46f0840b8f25794c1e3321c9b5820a8cb6ba9694a46fc64b8fae5a6",
    ),
    "README.md": (
        4754,
        "70078db7c64e5d28a8d1f4ca198cb51a8435c4f1c93e950d42534b435a39b27f",
    ),
    "config.json": (
        718,
        "28aeb28e3326fe377d98ac5b624e3e81005026671e3bc278b6a5dc37df34098a",
    ),
    "modeling_bert.py": (
        97481,
        "300f66dedc55c0a17d1975ef2bbc27aabcb09569599768c6ae8f1b766e2093b9",
    ),
    "model.safetensors": (
        1478017192,
        "d91cc5aab8e72f01009f9884a98142a74c189733ddfad8d0f91544a784730fa4",
    ),
    "pytorch_model.bin": (
        1478135113,
        "6334b8cc62182a1db370a388e115d346d9fdb6eb6083cbbd969d82bdbaa2eb1e",
    ),
    "special_tokens_map.json": (
        112,
        "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    ),
    "tokenizer.json": (
        1476210,
        "440cec76fcec5288d10f70b64f3f59289fe313dd1b6fc42bcf0fac319297dc3d",
    ),
    "tokenizer_config.json": (
        46,
        "98fece87a533726e5d34601535095b6caae9515770e720d06bb2a62e8017641f",
    ),
}
EXPECTED_CONFIG = {
    "architectures": ["BertForMaskedLM"],
    "attention_probs_dropout_prob": 0.1,
    "auto_map": {"AutoModel": "modeling_bert.BertForMaskedLM"},
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-5,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "pad_token_id": 3,
    "position_embedding_type": "absolute",
    "pre_layer_norm": True,
    "transformers_version": "4.6.0.dev0",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 32000,
}
INPUT_TOKEN_IDS = {
    "ACGTNACGTNACGTN": [1, 120, 5428, 120, 5428, 120, 5428, 2],
    "ACGTNNNNNNNNNN": [1, 120, 15, 5, 2],
}


OracleError = common.OracleError


def configure_common_contract() -> None:
    common.RUNTIME_ID = RUNTIME_ID
    common.SOURCE_REPO = SOURCE_REPO
    common.REQUESTED_REVISION = REQUESTED_REVISION
    common.SOURCE_REVISION = SOURCE_REVISION
    common.CATALOG_CONTRACT_SHA256 = CATALOG_CONTRACT_SHA256
    common.EXPECTED_SOURCE_FILES = EXPECTED_SOURCE_FILES
    common.EXTRACTOR_REPO = EXTRACTOR_REPO
    common.EXTRACTOR_REVISION = EXTRACTOR_REVISION
    common.EXTRACTOR_SHA256 = EXTRACTOR_SHA256
    common.EXTRACTOR_PYPROJECT_SHA256 = EXTRACTOR_PYPROJECT_SHA256


def read_input(path: Path) -> tuple[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise OracleError("cannot read oracle FASTA: %s" % error) from error
    if len(lines) < 2 or not lines[0].startswith(">") or not lines[0][1:]:
        raise OracleError("oracle input must be one named FASTA record")
    if any(line.startswith(">") for line in lines[1:]):
        raise OracleError("oracle input must contain exactly one FASTA record")
    sequence = "".join(lines[1:])
    if sequence not in INPUT_TOKEN_IDS:
        raise OracleError("oracle sequence is not one of the pinned cases")
    return lines[0][1:], sequence


def package_lock() -> list[str]:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            packages.append(
                "%s==%s" % (name.lower().replace("_", "-"), distribution.version)
            )
    return sorted(set(packages))


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


def generate(args: argparse.Namespace) -> dict[str, Any]:
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

    configure_common_contract()
    snapshot = args.snapshot.resolve(strict=True)
    receipt = args.source_receipt.resolve(strict=True)
    extractor_repo = args.extractor_repo.resolve(strict=True)
    extractor_path = args.extractor.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    source_files = common.validate_sources(receipt, snapshot)
    extractor = common.validate_extractor(extractor_repo, extractor_path)
    _, sequence = read_input(input_path)
    config = common.load_json(snapshot / "config.json", "model config")
    if config != EXPECTED_CONFIG:
        raise OracleError("pinned GENA-LM-Large-T2T config differs")

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if (
        tokenizer.vocab_size != 32000
        or tokenizer.padding_side != "right"
        or tokenizer.unk_token_id != 0
        or tokenizer.cls_token_id != 1
        or tokenizer.sep_token_id != 2
        or tokenizer.pad_token_id != 3
        or tokenizer.mask_token_id != 4
        or tokenizer.convert_tokens_to_ids("-") != 5
    ):
        raise OracleError("official tokenizer vocabulary/special contract differs")
    encoded = tokenizer(
        [sequence],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids")
    expected_ids = INPUT_TOKEN_IDS[sequence]
    if (
        input_ids.tolist() != [expected_ids]
        or attention_mask.tolist() != [[1] * len(expected_ids)]
        or token_type_ids is None
        or token_type_ids.tolist() != [[0] * len(expected_ids)]
    ):
        raise OracleError("official tokenizer IDs/mask/type IDs differ")

    model, loading = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise OracleError("official checkpoint loading info differs: %r" % loading)
    if model.__class__.__name__ != "BertForMaskedLM" or not hasattr(model, "bert"):
        raise OracleError("AutoModel did not load pinned masked-LM wrapper/.bert")
    if (
        not bool(model.config.pre_layer_norm)
        or hasattr(model.config, "last_layer_norm")
        or not bool(model.bert.encoder.last_layer_norm)
        or not hasattr(model.bert.encoder, "last_layer_ln")
        or len(model.bert.encoder.layer) != 24
    ):
        raise OracleError("official pre-LN/default-final-norm topology differs")
    for index, layer in enumerate(model.bert.encoder.layer):
        if (
            not hasattr(layer, "pre_attention_ln")
            or not hasattr(layer, "post_attention_ln")
            or hasattr(layer.attention.output, "LayerNorm")
            or hasattr(layer.output, "LayerNorm")
        ):
            raise OracleError("official pre-LN topology differs at layer %d" % index)

    model.to(device="cpu", dtype=torch.float32).eval()
    with torch.inference_mode():
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        pooled = pooled.to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(last_hidden.shape) != (1, len(expected_ids), 1024):
        raise OracleError("official last hidden shape differs: %r" % (last_hidden.shape,))
    if tuple(pooled.shape) != (1, 1024):
        raise OracleError("official pooled shape differs: %r" % (pooled.shape,))
    values = [float(value) for value in pooled[0].tolist()]
    if any(not math.isfinite(value) for value in values):
        raise OracleError("official oracle contains non-finite values")

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
            "mps_available_but_unused": bool(torch.backends.mps.is_available()),
        },
        "transformers": {
            "version": transformers.__version__,
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "local_files_only": True,
            "trust_remote_code": True,
            "model_class": "%s.%s" % (
                model.__class__.__module__,
                model.__class__.__name__,
            ),
            "backbone_class": "%s.%s" % (
                model.bert.__class__.__module__,
                model.bert.__class__.__name__,
            ),
            "tokenizer_class": "%s.%s" % (
                tokenizer.__class__.__module__,
                tokenizer.__class__.__name__,
            ),
        },
        "source_files": source_files,
        "model_semantics": {
            "hidden_size": 1024,
            "layers": 24,
            "attention_heads": 16,
            "intermediate_size": 4096,
            "max_tokens": 512,
            "norm_placement": "pre",
            "final_layer_norm": True,
            "hidden_tap": "model.bert(...).last_hidden_state",
        },
    }
    provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-reference",
        "oracle_execution_call": "exact-reference",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_file_manifest_sha256": common.sha256_bytes(
            common.canonical_json(source_manifest)
        ),
        "checkpoint_sha256": EXPECTED_SOURCE_FILES["model.safetensors"][1],
        "official_remote_modeling_code": True,
        "independent_of_evo_native_runtime": True,
        "generator_sha256": common.sha256_file(Path(__file__).resolve(strict=True)),
        "shared_generator_sha256": common.sha256_file(
            Path(common.__file__).resolve(strict=True)
        ),
        "extractor_repo": EXTRACTOR_REPO,
        "extractor_revision": extractor["revision"],
        "extractor_module": "embedding_pipeline/extractors/genalm.py",
        "extractor_class": "GENALMExtractor",
        "extractor_sha256": extractor["source_sha256"],
        "extractor_pyproject_sha256": extractor["pyproject_sha256"],
        "model_entrypoint": "AutoModel(...).bert",
        "hidden_tap": "outputs.last_hidden_state",
        "pooling": "attention-mask-mean-direct-f32-division",
        "special_tokens": "include",
        "cls_token_id": 1,
        "sep_token_id": 2,
        "extractor_max_tokens": 512,
        "input_token_ids": expected_ids,
        "attention_mask": [1] * len(expected_ids),
        "input_tokens": len(expected_ids),
        "padding_tokens": 0,
        "norm_placement": "pre",
        "final_layer_norm": True,
    }
    oracle = {
        "schema_version": 1,
        "kind": "geneb-independent-oracle-vector",
        "runtime_id": RUNTIME_ID,
        "input_sha256": common.sha256_file(input_path),
        "backend": "cpu",
        "profile": "cpu-f32",
        "values": values,
        "environment_lock": environment,
        "provenance": provenance,
    }
    portable(oracle, "oracle vector")
    return oracle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--extractor-repo", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        oracle = generate(args)
        payload = common.canonical_json(oracle)
        common.write_atomic(args.output, payload, args.force)
        values = oracle["values"]
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": common.sha256_bytes(payload),
                    "values": len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "l2_norm": math.sqrt(sum(value * value for value in values)),
                    "input_tokens": oracle["provenance"]["input_tokens"],
                },
                sort_keys=True,
            )
        )
    except (OracleError, OSError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
