#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a pinned Biohub Transformers ESMC oracle outside the runtime."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
from pathlib import Path
from typing import Any


REFERENCE_COMMIT = "3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf"
REFERENCE_HASHES = {
    "configuration_esmc.py": "e3382ce2874e114398ac3a541a9eee77671afdfc3762cc1cbba76e4ab4cbeb21",
    "modeling_esmc.py": "721df21d122c8223361d4458a15596d7e1d6f0b8afbcbe6d1c1d7b64975f4e2d",
    "tokenization_esmc.py": "e06f3d4c7313fc75db72211aa49fc7143180fb57c7e6ffb65e3e19034a47322d",
}
ORACLE_SEQUENCE = "LAGV<mask>ERT"
ORACLE_TOKEN_IDS = [0, 4, 5, 6, 7, 32, 9, 10, 11, 2]


class OracleError(RuntimeError):
    """Raised when the reference environment or source is not pinned."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} root must be an object")
    return value


def validate_source(
    model_dir: Path, registry_path: Path, model_id: str
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    registry = load_json(registry_path, "model registry")
    models = registry.get("models")
    entry = models.get(model_id) if isinstance(models, dict) else None
    if not isinstance(entry, dict) or entry.get("family") != "esmc":
        raise OracleError(f"{model_id!r} is not a registered ESMC model")
    checkpoint_files = entry.get("checkpoint_files")
    if not isinstance(checkpoint_files, list) or not checkpoint_files:
        raise OracleError(f"{model_id!r} has no registered checkpoint files")
    validated: list[dict[str, object]] = []
    for item in checkpoint_files:
        if not isinstance(item, dict):
            raise OracleError("checkpoint file descriptor must be an object")
        name = item.get("name")
        size = item.get("size")
        expected_hash = item.get("sha256")
        if not isinstance(name, str) or not isinstance(size, int) or not isinstance(
            expected_hash, str
        ):
            raise OracleError("checkpoint file descriptor is incomplete")
        path = model_dir / name
        try:
            actual_size = path.stat().st_size
        except OSError as error:
            raise OracleError(f"registered source file is unavailable: {path}") from error
        actual_hash = sha256(path)
        if actual_size != size or actual_hash != expected_hash:
            raise OracleError(
                f"source integrity mismatch for {name}: size={actual_size}, "
                f"sha256={actual_hash}"
            )
        validated.append(
            {"name": name, "size": actual_size, "sha256": actual_hash}
        )
    return entry, validated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence", default=ORACLE_SEQUENCE)
    parser.add_argument("--reference-commit", default=REFERENCE_COMMIT)
    args = parser.parse_args()

    if args.reference_commit != REFERENCE_COMMIT:
        parser.error(f"--reference-commit must be {REFERENCE_COMMIT}")
    if args.sequence != ORACLE_SEQUENCE:
        parser.error(f"--sequence must be the pinned acceptance input {ORACLE_SEQUENCE!r}")
    if args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")

    model_dir = args.model_dir.resolve()
    registry_path = args.registry.resolve()
    entry, source_files = validate_source(model_dir, registry_path, args.model_id)

    try:
        import numpy as np
        import torch
        import transformers
        from transformers.models.esmc.configuration_esmc import ESMCConfig
        from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM
        from transformers.models.esmc.tokenization_esmc import ESMCTokenizer
    except ImportError as error:
        raise OracleError(
            "the isolated oracle environment requires NumPy, PyTorch, and the "
            "pinned Biohub Transformers ESMC sources"
        ) from error

    modeling_path = Path(inspect.getsourcefile(ESMCForMaskedLM) or "").resolve()
    tokenizer_path = Path(inspect.getsourcefile(ESMCTokenizer) or "").resolve()
    configuration_path = Path(inspect.getsourcefile(ESMCConfig) or "").resolve()
    reference_paths = (configuration_path, modeling_path, tokenizer_path)
    if any(not path.is_file() for path in reference_paths):
        raise OracleError("cannot resolve the imported Biohub ESMC source files")
    for path in reference_paths:
        actual_hash = sha256(path)
        expected_hash = REFERENCE_HASHES.get(path.name)
        if actual_hash != expected_hash:
            raise OracleError(
                f"official reference hash mismatch for {path}: {actual_hash}"
            )

    config = ESMCConfig.from_pretrained(str(model_dir), local_files_only=True)
    config._attn_implementation = "eager"
    profile = load_json(registry_path, "model registry").get("esmc_profiles", {})
    topology = profile.get(entry.get("profile")) if isinstance(profile, dict) else None
    expected_topology = (
        int(topology["hidden_size"]),
        int(topology["num_layers"]),
        int(topology["num_attention_heads"]),
        int(topology["inner_mlp_size"]),
        int(topology["vocab_size"]),
    ) if isinstance(topology, dict) else None
    actual_topology = (
        int(config.d_model),
        int(config.n_layers),
        int(config.n_heads),
        int(config.expansion_ratio * config.d_model),
        int(config.vocab_size),
    )
    if expected_topology is None or actual_topology != expected_topology:
        raise OracleError(
            f"official config topology {actual_topology} differs from registry "
            f"{expected_topology}"
        )

    tokenizer = ESMCTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    encoded = tokenizer(args.sequence, return_tensors="pt")
    token_ids = encoded["input_ids"][0].tolist()
    if token_ids != ORACLE_TOKEN_IDS:
        raise OracleError(
            f"official tokenizer produced {token_ids}, expected {ORACLE_TOKEN_IDS}"
        )
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise OracleError("the production-size official oracle requires a CUDA device")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ESMCForMaskedLM.from_pretrained(
        str(model_dir), config=config, local_files_only=True, dtype=torch.float32
    )
    model.eval().to(args.device)
    inputs = {name: tensor.to(args.device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        output = model(
            **inputs,
            output_hidden_states=False,
            return_dict=True,
            compute_sae=False,
        )
    logits = output.logits[0].detach().cpu().to(torch.float32).numpy()
    final_hidden = (
        output.last_hidden_state[0].detach().cpu().to(torch.float32).numpy()
    )

    args.output_dir.mkdir(parents=True)
    logits_path = args.output_dir / "logits.npy"
    hidden_path = args.output_dir / "final-hidden.npy"
    np.save(logits_path, np.asarray(logits, dtype="<f4"), allow_pickle=False)
    np.save(hidden_path, np.asarray(final_hidden, dtype="<f4"), allow_pickle=False)
    device = torch.cuda.get_device_properties(torch.device(args.device))
    manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "source_repo": entry.get("source_repo"),
        "source_revision": entry.get("source_revision"),
        "source_files": source_files,
        "official_reference": {
            "repository": "Biohub/transformers",
            "commit": REFERENCE_COMMIT,
            "configuration_esmc_path": str(configuration_path),
            "configuration_esmc_sha256": sha256(configuration_path),
            "modeling_esmc_path": str(modeling_path),
            "modeling_esmc_sha256": sha256(modeling_path),
            "tokenization_esmc_path": str(tokenizer_path),
            "tokenization_esmc_sha256": sha256(tokenizer_path),
        },
        "sequence": args.sequence,
        "token_ids": token_ids,
        "attention_implementation": "eager",
        "dtype": "float32",
        "tf32": False,
        "outputs": {
            "logits": {
                "path": logits_path.name,
                "shape": list(logits.shape),
                "sha256": sha256(logits_path),
            },
            "final_hidden": {
                "path": hidden_path.name,
                "shape": list(final_hidden.shape),
                "sha256": sha256(hidden_path),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": device.name,
            "gpu_total_memory": device.total_memory,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    manifest_path = args.output_dir / "oracle.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
