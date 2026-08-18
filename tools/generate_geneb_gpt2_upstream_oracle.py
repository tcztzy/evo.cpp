#!/usr/bin/env python3
"""Generate pinned upstream GPT2-Gene GENEB embedding oracles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_SPECS = {
    "geneb-gpt2-gene-v1": {
        "repo": "dnagpt/gpt2_gene_v1",
        "revision": "ffcc7a4135ec0439800ee95fdc413706685136e8",
    },
    "geneb-gpt2-gene-multi-v2": {
        "repo": "dnagpt/gpt2_gene_multi_v2_ft",
        "revision": "b74ac7de9a489e936f612ded2036e01a33eec743",
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(
        (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[args.model]
    repo = spec["repo"]
    revision = spec["revision"]
    if snapshot.name != revision:
        raise RuntimeError("snapshot revision differs")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    sequence = input_path.read_text(encoding="ascii").strip()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded = tokenizer(
        [sequence],
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=False,
        return_tensors="pt",
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        result = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if result.hidden_states is None or len(result.hidden_states) != 13:
            raise RuntimeError("hidden-state tap count differs")
        hidden = result.hidden_states[-1]
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    values = pooled.cpu().contiguous().numpy().astype("<f4", copy=False)
    npy = output / "input-0.attention-mask-mean.f32.npy"
    np.save(npy, values, allow_pickle=False)
    source_files = {}
    for path in sorted(snapshot.iterdir(), key=lambda item: item.name):
        if path.is_file():
            source_files[path.name] = {"size": path.stat().st_size, "sha256": digest(path)}
    packages = sorted(
        {
            "%s==%s" % (
                distribution.metadata["Name"].lower().replace("_", "-"),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    environment = {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
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
            "model_class": "%s.%s" % (model.__class__.__module__, model.__class__.__name__),
            "tokenizer_class": "%s.%s" % (
                tokenizer.__class__.__module__, tokenizer.__class__.__name__
            ),
            "attention_implementation": getattr(model.config, "_attn_implementation", "eager"),
            "revision": revision,
            "trust_remote_code": False,
            "local_files_only": True,
        },
        "source_files": source_files,
    }
    generator = Path(__file__).resolve(strict=True)
    provenance = {
        "source_repo": repo,
        "source_revision": revision,
        "checkpoint_sha256": digest(snapshot / "model.safetensors"),
        "generator_sha256": digest(generator),
        "independent_of_evo_native_runtime": True,
        "official_transformers_implementation": True,
        "trust_remote_code": False,
    }
    input_sha = digest(input_path)
    vector = {
        "schema_version": 1,
        "kind": "geneb-independent-oracle-vector",
        "runtime_id": args.model,
        "input_sha256": input_sha,
        "backend": "cpu",
        "profile": "cpu-f32",
        "values": [float(value) for value in values.reshape(-1)],
        "environment_lock": environment,
        "provenance": provenance,
    }
    vector_path = output / "input-0.independent-oracle-vector.json"
    write_json(vector_path, vector)
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": vector["runtime_id"],
        "source": {"repo": repo, "revision": revision},
        "input": {
            "sha256": input_sha,
            "sequence_length": len(sequence),
            "input_ids": encoded["input_ids"].tolist(),
            "attention_mask": encoded["attention_mask"].tolist(),
        },
        "output": {
            "shape": list(values.shape),
            "npy_sha256": digest(npy),
            "raw_f32_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            "vector_sha256": digest(vector_path),
            "first_16": vector["values"][:16],
        },
        "environment_lock": environment,
        "provenance": provenance,
    }
    report_path = output / "oracle-report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "report_sha256": digest(report_path),
                "vector_sha256": digest(vector_path),
                "npy_sha256": digest(npy),
                "shape": list(values.shape),
                "input_ids": encoded["input_ids"].tolist(),
                "first_16": vector["values"][:16],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
