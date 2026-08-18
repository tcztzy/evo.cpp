#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned GENEB SPACE CPU embedding oracle offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Tuple
from urllib.parse import urlsplit

import numpy as np
import torch
import transformers


RUNTIME_ID = "geneb-space"
SOURCE_REPO = "yangyz1230/space"
SOURCE_REVISION = "63c9f5011877cc20b3e9d2d63dcdb1d547e62c18"
IMPLEMENTATION_REPO = "ZhuJiwei111/SPACE"
IMPLEMENTATION_REVISION = "4cdba18b80f948410623acee4b27a988ae7ddace"
EXTRACTOR_REPO = "darlednik/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SIZE = 2375
EXTRACTOR_SHA256 = (
    "c77f338d3e65a3c57df85e7ce931c39649d5ebdba2185ec5a38985a483f6a0af"
)
CONFIG_SIZE = 700
CONFIG_SHA256 = (
    "aa0c989abae4593aa22faa7b9f544237015d431dd33ebea8a6a80a4f60fc739c"
)
CHECKPOINT_SIZE = 2355245554
CHECKPOINT_SHA256 = (
    "f22ce262adc3347a8182bcd3bb5cd630ded4f1586b6c2a9841de6d14ea61cc81"
)
OFFICIAL_MODEL_FILES = {
    "model/__init__.py": (
        271,
        "438edc8b870b8c2c6b39d120967f748ae0520a8ad0192bfeb5300e706872201f",
    ),
    "model/config_space.py": (
        1730,
        "493d517e70e172345a97643482d505b13194dd8068c0ecfb48f515d02cb538c9",
    ),
    "model/data.py": (
        6659,
        "ec84cabc8476c7409726f5a0d0fc3ae4b88716f9f1d724cc8b87c7a2c3ca8a9a",
    ),
    "model/modeling_space.py": (
        10834,
        "b2f8111f34fd85c800446d5148762bfe9fb32fde40d13098bdb89dd333a2408a",
    ),
    "model/modules.py": (
        25410,
        "56e5c00dcf4b17671cbe3184c797571f42c93e311c7e630b0028c1d2f017b0b7",
    ),
    "model/precomputed/tf_gammas.pt": (
        393841,
        "595eb2f3b4aca209b98af6eb22e12ba41ca950c997547abcc7e2c93274034308",
    ),
}
VENDORED_MODEL_FILES = {
    **OFFICIAL_MODEL_FILES,
    "model/modules.py": (
        25422,
        "9a7ffe23c97c241f71838ddc8c5114e34c734b9344ae588322ae450197053a2e",
    ),
}
TARGET_FILES = {
    "datasets/pretrain/targets_human_sorted.txt": (
        800919,
        "d90233175a9fec389e5ac04cc4aac8cfc08c4aa86217e2e5c10221de78023580",
    ),
    "datasets/pretrain/targets_mouse_sorted.txt": (
        250458,
        "f76c869a38bfc79690d7a43e86f155db9b8ac8085ff5a717f99358b3253cd230",
    ),
}
SEQUENCE_LENGTH = 131072
TARGET_LENGTH = 896


def long_input() -> bytes:
    prefix = b"TTTTinvalidUacgt"
    suffix = b"nrxGGGG"
    body_length = 131101 - len(prefix) - len(suffix)
    pattern = b"acgtUNRx"
    return prefix + (pattern * ((body_length + len(pattern) - 1) // len(pattern)))[:body_length] + suffix


INPUTS = (
    ("input-0", b"uACGTnrxT"),
    ("input-1", long_input()),
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


def one_hot_sequence(sequence: bytes) -> Tuple[torch.Tensor, Dict[str, int]]:
    normalized = sequence.decode("ascii").upper().replace("U", "T")
    original_length = len(normalized)
    if original_length >= SEQUENCE_LENGTH:
        crop_left = (original_length - SEQUENCE_LENGTH) // 2
        crop_right = original_length - SEQUENCE_LENGTH - crop_left
        transformed = normalized[crop_left : crop_left + SEQUENCE_LENGTH]
        pad_left = 0
        pad_right = 0
    else:
        crop_left = 0
        crop_right = 0
        padding = SEQUENCE_LENGTH - original_length
        pad_left = padding // 2
        pad_right = padding - pad_left
        transformed = "N" * pad_left + normalized + "N" * pad_right
    array = np.zeros((SEQUENCE_LENGTH, 4), dtype=np.float32)
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    for index, character in enumerate(transformed):
        column = mapping.get(character)
        if column is not None:
            array[index, column] = 1.0
    return torch.from_numpy(array), {
        "original_length": original_length,
        "effective_length": SEQUENCE_LENGTH,
        "crop_left": crop_left,
        "crop_right": crop_right,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "valid_bases": sum(character in mapping for character in transformed),
        "zero_vector_bases": sum(character not in mapping for character in transformed),
    }


def validate_early_return_contract(modeling_path: Path) -> Dict[str, Any]:
    text = modeling_path.read_text(encoding="utf-8")
    forward_start = text.find("    def forward(\n")
    forward_end = text.find("\n    @classmethod\n", forward_start)
    if forward_start < 0 or forward_end < 0:
        raise RuntimeError("pinned SPACE forward method boundary differs")
    forward = text[forward_start:forward_end]
    ordered_fragments = (
        "        x = self.crop_final(x)\n        x = self.final_pointwise(x)\n",
        "        if return_only_embeddings:\n",
        "            return x\n            # return x, species_statistics\n",
        "        out = self.heads[species](x)\n",
        "        if return_embeddings:\n            return out, x\n",
    )
    positions = []
    start = 0
    for fragment in ordered_fragments:
        position = forward.find(fragment, start)
        if position < 0 or forward.count(fragment) != 1:
            raise RuntimeError("pinned SPACE embedding return control flow differs")
        positions.append(position)
        start = position + len(fragment)
    region = forward[positions[0] : positions[-1] + len(ordered_fragments[-1])]
    forbidden = (
        "x +=",
        "x -=",
        "x *=",
        "x /=",
        "x.copy_",
        "x.add_",
        "x.mul_",
    )
    between = forward[
        positions[0] + len(ordered_fragments[0]) : positions[-1]
    ]
    if any(operation in between for operation in forbidden):
        raise RuntimeError("SPACE post-embedding branch mutates returned x")
    return {
        "kind": "pinned-source-control-flow-plus-forward-hook",
        "source_region_sha256": sha256_bytes(region.encode("utf-8")),
        "return_only_expression": "return x",
        "return_embeddings_expression": "return out, x",
        "input_shape_contract": "batched (extractor torch.stack output)",
        "no_batch_branch_taken": False,
        "post_embedding_x_mutation": False,
    }


def validate_sources(
    config_argument: Path,
    checkpoint_argument: Path,
    official_argument: Path,
    vendored_argument: Path,
    extractor_argument: Path,
) -> Tuple[Path, Path, Path, Dict[str, Any], Dict[str, Any]]:
    config = config_argument.resolve(strict=True)
    checkpoint = checkpoint_argument.resolve(strict=True)
    official = official_argument.resolve(strict=True)
    vendored = vendored_argument.resolve(strict=True)
    extractor = extractor_argument.resolve(strict=True)
    require_file(config, CONFIG_SIZE, CONFIG_SHA256, "SPACE config")
    require_file(checkpoint, CHECKPOINT_SIZE, CHECKPOINT_SHA256, "SPACE checkpoint")
    require_file(extractor, EXTRACTOR_SIZE, EXTRACTOR_SHA256, "GENEB SPACE extractor")
    source_files = {
        "config.json": {"size": CONFIG_SIZE, "sha256": CONFIG_SHA256},
        "pytorch_model.bin": {
            "size": CHECKPOINT_SIZE,
            "sha256": CHECKPOINT_SHA256,
        },
        "GENEB/space.py": {
            "size": EXTRACTOR_SIZE,
            "sha256": EXTRACTOR_SHA256,
        },
    }  # type: Dict[str, Any]
    for name, (size, digest) in OFFICIAL_MODEL_FILES.items():
        require_file(official / name, size, digest, "official SPACE " + name)
        source_files["SPACE/" + name] = {"size": size, "sha256": digest}
    for name, (size, digest) in VENDORED_MODEL_FILES.items():
        require_file(vendored / name, size, digest, "GENEB vendored SPACE " + name)
        source_files["GENEB/SPACE/" + name] = {"size": size, "sha256": digest}
    for name, (size, digest) in TARGET_FILES.items():
        require_file(vendored / name, size, digest, "GENEB vendored SPACE " + name)
        source_files["GENEB/SPACE/" + name] = {"size": size, "sha256": digest}

    official_modules = (official / "model/modules.py").read_text(encoding="utf-8")
    vendored_modules = (vendored / "model/modules.py").read_text(encoding="utf-8")
    expected_vendored = official_modules.replace(
        'TF_GAMMAS = torch.load(str(DIR / "precomputed" / "tf_gammas.pt"))\n\n# helpers',
        'TF_GAMMAS = torch.load(str(DIR / "precomputed" / "tf_gammas.pt"))\nROOT = Path(DIR).parents[0]\n# helpers',
    ).replace(
        'pd.read_csv(f"/home/jiwei_zhu/SPACE/datasets/pretrain/targets_{sp}_sorted.txt", sep="\\t")',
        'pd.read_csv(f"{ROOT}/datasets/pretrain/targets_{sp}_sorted.txt", sep="\\t")',
    )
    if expected_vendored != vendored_modules:
        raise RuntimeError("GENEB SPACE vendoring differs beyond the audited dataset-path patch")
    extractor_text = extractor.read_text(encoding="utf-8")
    extractor_fragments = (
        "self.model = Space.from_pretrained(name_model).to(self.device).eval()",
        "self.seq_length = 131_072",
        "seq = seq.upper().replace('U', 'T')",
        "output = self.model(tensor, return_embeddings=True)",
        "emb = output[-1].mean(dim=1).detach().cpu().numpy()",
    )
    if any(extractor_text.count(fragment) != 1 for fragment in extractor_fragments):
        raise RuntimeError("pinned GENEB SPACE extractor semantics differ")
    early_return = validate_early_return_contract(official / "model/modeling_space.py")
    patch = {
        "status": "audited-nonnumerical-vendoring-patch",
        "official_modules_sha256": OFFICIAL_MODEL_FILES["model/modules.py"][1],
        "vendored_modules_sha256": VENDORED_MODEL_FILES["model/modules.py"][1],
        "operations": [
            "define repository-relative SPACE root",
            "replace absolute targets table locator with repository-relative locator",
        ],
        "numerical_operations_changed": False,
    }
    return config, checkpoint, vendored, source_files, {
        "early_return": early_return,
        "vendoring_patch": patch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--official-source", required=True, type=Path)
    parser.add_argument("--vendored-source", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config_path, checkpoint_path, vendored_path, source_files, audits = validate_sources(
        args.config,
        args.checkpoint,
        args.official_source,
        args.vendored_source,
        args.extractor,
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
    sys.path.insert(0, str(vendored_path))
    from model.config_space import SpaceConfig  # type: ignore
    from model.modeling_space import TrainingSpace  # type: ignore

    config = SpaceConfig.from_json_file(str(config_path))
    required_config = {
        "dim": 1536,
        "depth": 11,
        "heads": 8,
        "attn_dim_key": 64,
        "target_length": TARGET_LENGTH,
        "num_downsamples": 7,
        "use_tf_gamma": False,
        "seq_length": SEQUENCE_LENGTH,
        "topk": 3,
        "species_num_experts": 4,
        "tracks_num_experts": 8,
        "moe": ["species", "tracks"],
    }
    wrong = {
        key: (getattr(config, key, None), value)
        for key, value in required_config.items()
        if getattr(config, key, None) != value
    }
    if wrong or config.output_heads != {"human": 5313, "mouse": 1643}:
        raise RuntimeError("pinned SPACE topology differs: %s" % wrong)

    training_model = TrainingSpace(config)
    state = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(state, Mapping) or len(state) != 363:
        raise RuntimeError("pinned SPACE state manifest differs")
    if (
        sum(value.dtype == torch.float32 for value in state.values()) != 349
        or sum(value.dtype == torch.int64 for value in state.values()) != 14
    ):
        raise RuntimeError("pinned SPACE state dtype manifest differs")
    load_result = training_model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError("pinned SPACE strict load differs")
    del state
    model = training_model.model.to(device="cpu", dtype=torch.float32).eval()
    model_name = "%s.%s" % (model.__class__.__module__, model.__class__.__name__)
    if model_name != "model.modeling_space.Space":
        raise RuntimeError("pinned SPACE model class differs")

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
        "model_semantics": {
            "species": "human",
            "input_length": SEQUENCE_LENGTH,
            "target_length": TARGET_LENGTH,
            "sequence_embedding_width": 3072,
            "hidden_tap": "returned-sequence-embedding",
            "pooling": "spatial-mean",
            "return_only_embeddings": True,
        },
        "source_files": source_files,
        "source_audits": audits,
    }
    portable(environment_lock, "environment_lock")
    (output / "environment-lock.json").write_bytes(canonical_json(environment_lock))

    records = []
    for record_index, (label, input_bytes) in enumerate(INPUTS):
        input_path = output / (label + ".txt")
        input_path.write_bytes(input_bytes)
        one_hot, transform = one_hot_sequence(input_bytes)
        batch = one_hot.unsqueeze(0)
        captured = []  # type: List[torch.Tensor]
        hook = None
        if record_index == 0:
            hook = model.final_pointwise.register_forward_hook(
                lambda _module, _inputs, result: captured.append(result.detach().clone())
            )
        with torch.no_grad():
            embeddings = model(
                batch,
                return_only_embeddings=True,
                species="human",
                target_length=TARGET_LENGTH,
            )
            pooled = embeddings.mean(dim=1).to(dtype=torch.float32)
        if hook is not None:
            hook.remove()
            if len(captured) != 1 or not torch.equal(captured[0], embeddings):
                raise RuntimeError("SPACE early return differs from final pointwise x")
            audits["early_return"]["forward_hook_bitwise_equal"] = True
            audits["early_return"]["unused_heads_and_tracks_executed"] = False
        if list(embeddings.shape) != [1, TARGET_LENGTH, 3072] or list(pooled.shape) != [1, 3072]:
            raise RuntimeError("pinned SPACE returned shape differs")
        npy = save_npy(output / (label + ".spatial-mean.f32.npy"), pooled)
        provenance = {
            "kind": "pinned-upstream-with-audited-nonnumerical-vendoring-patch",
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "implementation_repo": IMPLEMENTATION_REPO,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "extractor_repo": EXTRACTOR_REPO,
            "extractor_revision": EXTRACTOR_REVISION,
            "extractor_sha256": EXTRACTOR_SHA256,
            "vendoring_patch": audits["vendoring_patch"],
            "early_return_equivalence": audits["early_return"],
            "species": "human",
            "hidden_tap": "returned-sequence-embedding",
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
                "transform": transform,
                "one_hot_raw_little_endian_f32_sha256": sha256_bytes(
                    one_hot.numpy().astype("<f4", copy=False).tobytes(order="C")
                ),
                "embeddings_shape": list(embeddings.shape),
                "pooled": npy,
                "oracle_vector_file": vector_path.name,
                "oracle_vector_sha256": sha256_file(vector_path),
            }
        )

    # The first record dynamically completed the source-level early-return proof;
    # rewrite the shared lock before freezing its digest into the report/vectors.
    portable(environment_lock, "environment_lock")
    (output / "environment-lock.json").write_bytes(canonical_json(environment_lock))
    for record in records:
        vector_path = output / record["oracle_vector_file"]
        vector = json.loads(vector_path.read_text(encoding="ascii"))
        vector["environment_lock"] = environment_lock
        vector["provenance"]["early_return_equivalence"] = audits["early_return"]
        vector_path.write_bytes(canonical_json(vector))
        record["oracle_vector_sha256"] = sha256_file(vector_path)

    report = {
        "schema_version": 1,
        "kind": "geneb-space-upstream-oracle-report",
        "runtime_id": RUNTIME_ID,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "implementation_repo": IMPLEMENTATION_REPO,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "extractor_revision": EXTRACTOR_REVISION,
        "extractor_sha256": EXTRACTOR_SHA256,
        "environment_lock_sha256": sha256_file(output / "environment-lock.json"),
        "records": records,
    }
    portable(report, "report")
    (output / "oracle-report.json").write_bytes(canonical_json(report))
    print(str(output / "oracle-report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
