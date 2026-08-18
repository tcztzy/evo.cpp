#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the patched, pinned GENEB Enformer CPU oracle offline."""

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


RUNTIME_ID = "geneb-enformer"
SOURCE_REPO = "EleutherAI/enformer-official-rough"
SOURCE_REVISION = "affe5713ae9017460706a44108289b13c5fee16c"
EXTRACTOR_REPO = "darlednik/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "acc0ad761b1b1586abc4ea4796afc0fa8a053a4bfb1a1993065e6bde3f56f12d"
)
PATCH_SHA256 = (
    "77dfce0361962f4a5bbe5c2ad0c80bd9eaf5899fffb64bbc988fba01b477afda"
)
CONFIG_SIZE = 439
CONFIG_SHA256 = (
    "ad7a0e07b2fe40fd4c93a8528416406b4429f303946f5a93209ef26f160a034f"
)
CHECKPOINT_SIZE = 1005149571
CHECKPOINT_SHA256 = (
    "99b09d602e195d89c7d4debe144bb2f43907ba0d74006e97098e99d9171c439c"
)
MODEL_CODE = {
    "__init__.py": "a155bba4a3c350b9886466fc711f4550a18026fa506c1611bdea5e3a5e359ae5",
    "config_enformer.py": "715f799e2a55f0b7f991b434197a17718694a4e85e2bc7c47c5486114db2193a",
    "data.py": "ec84cabc8476c7409726f5a0d0fc3ae4b88716f9f1d724cc8b87c7a2c3ca8a9a",
    "modeling_enformer.py": "7d3d8860560e353983e85c325ead0ba7aab4281957251828392c49dc7426c008",
}
SEQUENCE_LENGTH = 196608
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, size: int, digest: str, label: str) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != size
        or sha256_file(path) != digest
    ):
        raise RuntimeError("pinned %s differs" % label)


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


def validate_sources(
    config_argument: Path,
    checkpoint_argument: Path,
    code_argument: Path,
    extractor_argument: Path,
    patch_argument: Path,
) -> Tuple[Path, Path, Path, Dict[str, Any]]:
    config = config_argument.resolve(strict=True)
    checkpoint = checkpoint_argument.resolve(strict=True)
    code = code_argument.resolve(strict=True)
    extractor = extractor_argument.resolve(strict=True)
    patch = patch_argument.resolve(strict=True)
    require_file(config, CONFIG_SIZE, CONFIG_SHA256, "Enformer config")
    require_file(
        checkpoint, CHECKPOINT_SIZE, CHECKPOINT_SHA256, "Enformer checkpoint"
    )
    if extractor.is_symlink() or sha256_file(extractor) != EXTRACTOR_SHA256:
        raise RuntimeError("pinned GENEB Enformer extractor differs")
    if patch.is_symlink() or sha256_file(patch) != PATCH_SHA256:
        raise RuntimeError("pinned GENEB Enformer patch differs")
    extractor_text = extractor.read_text(encoding="utf-8")
    if extractor_text.count("self.seq_length = int(seq_length)") != 1:
        raise RuntimeError("Enformer extractor defect site differs")
    if (
        patch.read_text(encoding="utf-8").count(
            "+        self.seq_length = 196608"
        )
        != 1
    ):
        raise RuntimeError("Enformer reference patch operation differs")
    source_files = {
        "config.json": {"size": CONFIG_SIZE, "sha256": CONFIG_SHA256},
        "pytorch_model.bin": {
            "size": CHECKPOINT_SIZE,
            "sha256": CHECKPOINT_SHA256,
        },
    }
    package = code / "enformer_pytorch"
    for name, digest in MODEL_CODE.items():
        path = package / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise RuntimeError("pinned Enformer implementation differs: " + name)
        source_files["enformer_pytorch/" + name] = {
            "size": path.stat().st_size,
            "sha256": digest,
        }
    source_files["GENEB/enformer.py"] = {
        "size": extractor.stat().st_size,
        "sha256": EXTRACTOR_SHA256,
    }
    source_files["enformer-seq-length.patch"] = {
        "size": patch.stat().st_size,
        "sha256": PATCH_SHA256,
    }
    return config, checkpoint, code, source_files


def encoded_indices(sequence: str) -> List[int]:
    normalized = sequence.upper()
    if len(normalized) >= SEQUENCE_LENGTH:
        normalized = normalized[:SEQUENCE_LENGTH]
    else:
        normalized += "N" * (SEQUENCE_LENGTH - len(normalized))
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    return [mapping.get(character, 4) for character in normalized]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-code", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config_path, checkpoint_path, code_path, source_files = validate_sources(
        args.config,
        args.checkpoint,
        args.model_code,
        args.extractor,
        args.patch,
    )
    for variable in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(variable + "=1 is required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    sys.path.insert(0, str(code_path))
    from enformer_pytorch import Enformer, EnformerConfig  # type: ignore

    config = EnformerConfig.from_json_file(str(config_path))
    if (
        config.dim != 1536
        or config.depth != 11
        or config.heads != 8
        or config.attn_dim_key != 64
        or config.target_length != 896
        or config.num_downsamples != 7
        or config.use_tf_gamma is not False
    ):
        raise RuntimeError("pinned Enformer topology differs")
    model = Enformer(config)
    state = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(state, Mapping) or len(state) != 570:
        raise RuntimeError("pinned Enformer state manifest differs")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("pinned Enformer strict load differs")
    del state
    model.to(device="cpu", dtype=torch.float32).eval()
    model_name = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    if model_name != "enformer_pytorch.modeling_enformer.Enformer":
        raise RuntimeError("pinned Enformer model class differs")

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
        "source_files": source_files,
    }
    portable(environment_lock, "environment_lock")
    (output / "environment-lock.json").write_bytes(canonical_json(environment_lock))

    records = []
    for label, input_bytes in INPUTS:
        input_path = output / (label + ".txt")
        input_path.write_bytes(input_bytes)
        indices = encoded_indices(input_bytes.decode("ascii"))
        input_ids = torch.tensor([indices], dtype=torch.long)
        with torch.no_grad():
            _, embeddings = model(input_ids, return_embeddings=True)
            pooled = embeddings.mean(dim=1).to(dtype=torch.float32)
        if list(embeddings.shape) != [1, 896, 3072] or list(pooled.shape) != [1, 3072]:
            raise RuntimeError("pinned Enformer returned shape differs")
        npy = save_npy(output / (label + ".spatial-mean.f32.npy"), pooled)
        provenance = {
            "kind": "pinned-upstream-with-audited-reference-patch",
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "extractor_repo": EXTRACTOR_REPO,
            "extractor_revision": EXTRACTOR_REVISION,
            "extractor_sha256": EXTRACTOR_SHA256,
            "reference_patch_sha256": PATCH_SHA256,
            "reference_patch_operation": "set undefined seq_length to 196608",
            "use_tf_gamma": False,
            "hidden_tap": "returned-embeddings",
            "pooling": "spatial-mean",
        }
        portable(provenance, "provenance")
        vector = {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": RUNTIME_ID,
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
                "encoded_length": len(indices),
                "encoded_indices_little_endian_i64_sha256": sha256_bytes(
                    i64_bytes(indices)
                ),
                "embeddings_shape": list(embeddings.shape),
                "pooled": npy,
                "oracle_vector_file": vector_path.name,
                "oracle_vector_sha256": sha256_file(vector_path),
            }
        )

    report = {
        "schema_version": 1,
        "kind": "geneb-enformer-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_sha256": EXTRACTOR_SHA256,
        "reference_patch_sha256": PATCH_SHA256,
        "environment_lock_sha256": sha256_file(output / "environment-lock.json"),
        "records": records,
    }
    portable(report, "report")
    (output / "oracle-report.json").write_bytes(canonical_json(report))
    print(str(output / "oracle-report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
