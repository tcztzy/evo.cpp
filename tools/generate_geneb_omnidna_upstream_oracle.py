#!/usr/bin/env python3
"""Generate an independent, pinned Omni-DNA-300M GENEB embedding oracle."""

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
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REVISION = "23587a0177a1f00d18b7079a2e81150b80e63f1c"
REPO = "zehui127/Omni-DNA-300M"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if snapshot.name != REVISION:
        raise RuntimeError("snapshot revision differs")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    sequence = input_path.read_text(encoding="ascii").strip()
    if not sequence or any(base not in "ACGTN" for base in sequence):
        raise RuntimeError("canonical input must be non-empty A/C/G/T/N")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    encoded = tokenizer(
        [sequence],
        padding=True,
        truncation=True,
        max_length=250,
        return_tensors="pt",
    )
    config = AutoConfig.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    checkpoint = snapshot / "model.safetensors"
    state = load_file(str(checkpoint), device="cpu")
    load = model.load_state_dict(state, strict=False)
    if list(load.missing_keys) != ["word_embeddings.weight"] or load.unexpected_keys:
        raise RuntimeError("checkpoint load contract differs: %r" % (load,))
    if model.word_embeddings is not model.model.transformer.wte:
        raise RuntimeError("expected word_embeddings/wte module alias is absent")
    model.to(device="cpu", dtype=torch.float32).eval()
    with torch.inference_mode():
        result = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if result.hidden_states is None or len(result.hidden_states) != 17:
            raise RuntimeError("hidden-state tap count differs")
        hidden = result.hidden_states[-1]
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    values = pooled.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    npy_path = output / "input-0.attention-mask-mean.f32.npy"
    np.save(npy_path, values, allow_pickle=False)

    source_files = {}
    for path in sorted(snapshot.iterdir(), key=lambda item: item.name):
        if path.is_file():
            source_files[path.name] = {"size": path.stat().st_size, "sha256": sha256(path)}
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name and version:
            packages.append("%s==%s" % (name.lower().replace("_", "-"), version))
    packages = sorted(set(packages))
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
            "revision": REVISION,
            "trust_remote_code": True,
            "local_files_only": True,
            "model_class": "%s.%s" % (model.__class__.__module__, model.__class__.__name__),
            "config_class": "%s.%s" % (config.__class__.__module__, config.__class__.__name__),
            "load_patch": "from_config + verified Safetensors state_dict; permit only tied word_embeddings.weight alias",
        },
        "source_files": source_files,
    }
    generator = Path(__file__).resolve(strict=True)
    provenance = {
        "source_repo": REPO,
        "source_revision": REVISION,
        "checkpoint_sha256": sha256(checkpoint),
        "generator_sha256": sha256(generator),
        "independent_of_evo_native_runtime": True,
        "official_remote_modeling_code": True,
        "trust_remote_code": True,
        "state_dict_missing_keys": ["word_embeddings.weight"],
        "state_dict_unexpected_keys": [],
        "weight_alias": "word_embeddings.weight=model.transformer.wte.weight",
    }
    input_sha = sha256(input_path)
    vector = {
        "schema_version": 1,
        "kind": "geneb-independent-oracle-vector",
        "runtime_id": "geneb-omni-dna-300m",
        "input_sha256": input_sha,
        "backend": "cpu",
        "profile": "cpu-f32",
        "values": [float(value) for value in values.reshape(-1)],
        "environment_lock": environment,
        "provenance": provenance,
    }
    vector_path = output / "input-0.independent-oracle-vector.json"
    vector_path.write_bytes(canonical_json(vector))
    report = {
        "schema_version": 1,
        "kind": "geneb-independent-upstream-oracle-report",
        "runtime_id": vector["runtime_id"],
        "source": {"repo": REPO, "revision": REVISION},
        "input": {
            "sha256": input_sha,
            "sequence_length": len(sequence),
            "input_ids": encoded["input_ids"].tolist(),
            "attention_mask": encoded["attention_mask"].tolist(),
        },
        "output": {
            "shape": list(values.shape),
            "npy_sha256": sha256(npy_path),
            "raw_f32_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            "vector_sha256": sha256(vector_path),
            "first_16": vector["values"][:16],
        },
        "environment_lock": environment,
        "provenance": provenance,
    }
    report_path = output / "oracle-report.json"
    report_path.write_bytes(canonical_json(report))
    summary = {
        "report_sha256": sha256(report_path),
        "vector_sha256": sha256(vector_path),
        "npy_sha256": sha256(npy_path),
        "shape": list(values.shape),
        "input_ids": encoded["input_ids"].tolist(),
        "first_16": vector["values"][:16],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
