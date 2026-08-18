#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned NT-v2-50M-3mer GENEB vector with official HF code."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from generate_geneb_nt_upstream_oracle import (
    OracleError,
    canonical_json,
    load_json,
    normalized_relative,
    package_lock,
    read_canonical_sequence,
    sha256_bytes,
    sha256_file,
)


RUNTIME_ID = "geneb-nt-v2-50m-3mer-ms"
REPO = "InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species"
REQUESTED_REVISION = "main"
REVISION = "ff82eaf931e483feeb6bf7ecf03f1febe6b2fe76"
CATALOG_CONTRACT_SHA256 = (
    "c8a2241c2ccde1c03ee352cefcf1d50a372642ffd53d992a19028c05f780748a"
)
EXTRACTOR_REPO = "ultimativity/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "50332b6070b944dd4d26dc2641ebadfea7270513d34294add5f7066295567c43"
)
HELPER_SHA256 = (
    "f5d5f51d64c21619d9782675a7584a1bb383c1e07c3e16a8640ba87a77bb8d1a"
)
EXPECTED_IDS = [3, 11, 49, 11, 49, 11, 49]
EXPECTED_FILES = {
    ".gitattributes": (1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
    "README.md": (6746, "ae0cdd012731213dcb8e538037f938624fe56f7770f1ace6fa12569a8b1d31b0"),
    "config.json": (1119, "2ea52c61d1b53e4bbade5b2cc1fa401e8709e7439f9366bdc6e9add1d063efb5"),
    "esm_config.py": (14865, "96d22305fb7988839455e351b7b5f600b75cfc06ebe059c615a470e2281a7d3c"),
    "jax_model/hyperparams.json": (739, "f7db5650deb52aa77febc86c85856b2e3ec60387f376d20cca094f0b127fc0c0"),
    "jax_model/pytree_ckpt.joblib": (202906307, "889877863cf47d0a2cfcf41eaafd97f91def86bfc745c41c30b3a88f68bec340"),
    "modeling_esm.py": (58166, "a827ad0132def2f1c5c122559b87a96fb77deb06f3f2ae0b451dc0a7d2858fc8"),
    "pytorch_model.bin": (207152202, "df558322bba2e3eb9d4ef0ad0b763c2d6578f7fb05e5d19358fe7a30ce1bef61"),
    "special_tokens_map.json": (101, "d6dc30bf018166daab248b0abf7efda6fd1b1e0a2d1bee5b31b23db2ebdaee77"),
    "tokenizer_config.json": (129, "253d338919eba938e50b776f3243cc739462c207fe64e3d2e81cc5e681bee45b"),
    "vocab.txt": (302, "0976e2b4722307d16c669e2daa57f5912fc7f62e08389540751b5474c93dcfd0"),
}


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
    files = receipt["files"]
    if not isinstance(files, list) or len(files) != len(EXPECTED_FILES):
        raise OracleError("source receipt file count differs")
    verified = {}  # type: Dict[str, Dict[str, Any]]
    for index, item in enumerate(files):
        label = "source receipt files[%d]" % index
        if not isinstance(item, dict) or set(item) != {"name", "path", "size", "sha256"}:
            raise OracleError("%s fields differ" % label)
        name = normalized_relative(item["name"], label + ".name")
        if name in verified or name not in EXPECTED_FILES:
            raise OracleError("source receipt file set is duplicated or unexpected")
        size, digest = EXPECTED_FILES[name]
        if item["size"] != size or item["sha256"] != digest:
            raise OracleError("source receipt pinned size/SHA256 differs for %s" % name)
        for candidate, candidate_label in (
            (Path(item["path"]), "receipt locator"),
            (snapshot / name, "snapshot file"),
        ):
            if (
                not candidate.is_file()
                or candidate.stat().st_size != size
                or sha256_file(candidate) != digest
            ):
                raise OracleError("%s integrity differs for %s" % (candidate_label, name))
        verified[name] = {"size": size, "sha256": digest}
    snapshot_names = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if set(verified) != set(EXPECTED_FILES) or snapshot_names != set(EXPECTED_FILES):
        raise OracleError("pinned source snapshot file set differs")
    return dict(sorted(verified.items()))


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
    generator = Path(__file__).resolve(strict=True)
    helper = generator.with_name("generate_geneb_nt_upstream_oracle.py")
    if sha256_file(helper) != HELPER_SHA256:
        raise OracleError("pinned NT oracle helper SHA256 differs")
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
    if (
        len(tokenizer) != 75
        or tokenizer.model_max_length != 2048
        or tokenizer.padding_side != "right"
    ):
        raise OracleError("official 3-mer tokenizer contract differs")
    encoded = tokenizer(
        [sequence],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=2048,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if (
        list(input_ids.shape) != [1, 2048]
        or list(attention_mask.shape) != [1, 2048]
        or input_ids[0, : len(EXPECTED_IDS)].tolist() != EXPECTED_IDS
        or int(attention_mask.sum().item()) != len(EXPECTED_IDS)
        or any(value != 1 for value in input_ids[0, len(EXPECTED_IDS) :].tolist())
    ):
        raise OracleError("official 3-mer tokenizer IDs/padding differ")

    model, loading = AutoModelForMaskedLM.from_pretrained(
        snapshot,
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
    if model.config.vocab_size != 75 or float(model.config.layer_norm_eps) != 1e-5:
        raise OracleError("official 3-mer model vocab/LayerNorm epsilon differs")
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
    provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-reference",
        "source_repo": REPO,
        "source_revision": REVISION,
        "source_file_manifest_sha256": sha256_bytes(canonical_json(source_manifest)),
        "checkpoint_sha256": EXPECTED_FILES["pytorch_model.bin"][1],
        "official_remote_modeling_code": True,
        "independent_of_evo_native_runtime": True,
        "generator_sha256": sha256_file(generator),
        "helper_sha256": HELPER_SHA256,
        "extractor_repo": EXTRACTOR_REPO,
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_module": "embedding_pipeline/extractors/nt.py",
        "extractor_class": "NucleotideTransformerExtractor",
        "extractor_sha256": EXTRACTOR_SHA256,
        "source_config_duplicate_policy": "attention_probs_dropout_prob=0.0x2",
        "source_tensor_layout": "contiguous-or-exact-column-major",
        "hidden_tap": "outputs.hidden_states[-1]",
        "pooling": "attention-mask-mean",
        "special_tokens": "include",
        "model_max_length": 2048,
        "nonpadding_tokens": len(EXPECTED_IDS),
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
    vector_path = output / "nt-v2-50m-3mer-ms.independent-oracle-vector.json"
    vector_path.write_bytes(canonical_json(vector))
    npy_path = output / "nt-v2-50m-3mer-ms.attention-mask-mean.f32.npy"
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
            "input_ids_prefix": input_ids[0, :10].tolist(),
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
        raise SystemExit("generate_geneb_nt_3mer_upstream_oracle: error: %s" % error)
