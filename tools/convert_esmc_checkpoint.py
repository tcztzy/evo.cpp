#!/usr/bin/env python3
"""Convert pinned Biohub ESMC F32 Safetensors without a Torch dependency."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import struct
import sys
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from evo.format import ESMC_PROFILE_VALUE, FormatError, write_model


class ConversionError(ValueError):
    """Raised when a source artifact is outside the registered contract."""


@dataclasses.dataclass(frozen=True, slots=True)
class FileTensorSource:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    offset: int
    nbytes: int

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        with self.path.open("rb") as source:
            source.seek(self.offset)
            remaining = self.nbytes
            while remaining:
                chunk = source.read(min(chunk_size, remaining))
                if not chunk:
                    raise FormatError(f"source tensor {self.name!r} ended early")
                remaining -= len(chunk)
                yield memoryview(chunk)


EXTRA_STATE_RE = re.compile(
    r"esmc\.transformer\.blocks\.(\d+)\."
    r"(?:attn\.(?:layernorm_qkv|out_proj)|ffn)\._extra_state"
)


def default_registry() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "model-registry.json"


def load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversionError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConversionError(f"{label} root must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversionError(f"{label} must be a positive integer")
    return value


def validate_receipt(
    receipt_path: Path, registry_path: Path
) -> tuple[str, dict[str, object], dict[str, int], dict[str, Path], str]:
    receipt_bytes = receipt_path.read_bytes()
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ConversionError(f"cannot read source receipt: {error}") from error
    registry = load_json(registry_path, "model registry")
    if not isinstance(receipt, dict):
        raise ConversionError("source receipt root must be an object")
    model_id = receipt.get("model_id")
    models = registry.get("models")
    entry = models.get(model_id) if isinstance(models, dict) else None
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "source-checkpoint"
        or not isinstance(model_id, str)
        or not isinstance(entry, dict)
        or entry.get("family") != "esmc"
    ):
        raise ConversionError("receipt does not identify a registered ESMC source")
    repo = entry.get("source_repo")
    revision = entry.get("source_revision")
    if receipt.get("repo") != repo or receipt.get("resolved_revision") != revision:
        raise ConversionError("receipt repository/revision differs from the registry")
    profiles = registry.get("esmc_profiles")
    profile = profiles.get(entry.get("profile")) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        raise ConversionError("registered ESMC topology is missing")
    topology = {
        key: positive_integer(profile.get(key), f"esmc profile {key}")
        for key in (
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "inner_mlp_size",
            "vocab_size",
            "max_seqlen",
        )
    }
    registered_files = entry.get("checkpoint_files")
    receipt_files = receipt.get("files")
    if not isinstance(registered_files, list) or not isinstance(receipt_files, list):
        raise ConversionError("registry/receipt file manifest is missing")

    def entries(values: list[object], label: str) -> dict[str, dict[str, object]]:
        output: dict[str, dict[str, object]] = {}
        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise ConversionError(f"{label}[{index}] must be an object")
            name = value.get("name")
            size = value.get("size")
            digest = value.get("sha256")
            if (
                not isinstance(name, str)
                or PurePosixPath(name).is_absolute()
                or any(part in {"", ".", ".."} for part in PurePosixPath(name).parts)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or name in output
            ):
                raise ConversionError(f"{label}[{index}] is invalid or duplicated")
            output[name] = value
        return output

    expected = entries(registered_files, "registry checkpoint_files")
    actual = entries(receipt_files, "receipt files")
    if set(actual) != set(expected):
        raise ConversionError("receipt file set differs from the registry")
    paths: dict[str, Path] = {}
    for name, expected_file in expected.items():
        actual_file = actual[name]
        if actual_file.get("size") != expected_file.get("size") or actual_file.get(
            "sha256"
        ) != expected_file.get("sha256"):
            raise ConversionError(f"receipt integrity fields differ for {name}")
        path_value = actual_file.get("path")
        if not isinstance(path_value, str):
            raise ConversionError(f"receipt path is missing for {name}")
        path = Path(path_value).resolve()
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ConversionError(f"cannot stat source file {name}: {error}") from error
        digest = sha256(path)
        if size != expected_file["size"] or digest != expected_file["sha256"]:
            raise ConversionError(
                f"source integrity mismatch for {name}: size={size}, sha256={digest}"
            )
        paths[name] = path
    return (
        model_id,
        entry,
        topology,
        paths,
        hashlib.sha256(receipt_bytes).hexdigest(),
    )


def validate_config(path: Path, topology: dict[str, int]) -> None:
    config = load_json(path, "ESMC config")
    expected = {
        "architectures": ["ESMCForMaskedLM"],
        "d_model": topology["hidden_size"],
        "dtype": "float32",
        "mask_token_id": 32,
        "model_type": "esmc",
        "n_heads": topology["num_attention_heads"],
        "n_layers": topology["num_layers"],
        "pad_token_id": 1,
        "tie_word_embeddings": False,
        "vocab_size": topology["vocab_size"],
    }
    wrong = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if wrong:
        raise ConversionError(f"ESMC config differs from registered topology: {wrong}")
    if topology["hidden_size"] % topology["num_attention_heads"] != 0:
        raise ConversionError("ESMC hidden size is not divisible by attention heads")


def expected_manifest(topology: dict[str, int]) -> dict[str, tuple[int, ...]]:
    width = topology["hidden_size"]
    inner = topology["inner_mlp_size"]
    vocab = topology["vocab_size"]
    expected: dict[str, tuple[int, ...]] = {
        "esmc.embed.weight": (vocab, width),
        "esmc.transformer.norm.weight": (width,),
        "lm_head.0.weight": (width, width),
        "lm_head.0.bias": (width,),
        "lm_head.2.weight": (width,),
        "lm_head.2.bias": (width,),
        "lm_head.3.weight": (vocab, width),
        "lm_head.3.bias": (vocab,),
    }
    for layer in range(topology["num_layers"]):
        block = f"esmc.transformer.blocks.{layer}"
        expected.update(
            {
                f"{block}.attn.layernorm_qkv.layer_norm_weight": (width,),
                f"{block}.attn.layernorm_qkv.layer_norm_bias": (width,),
                f"{block}.attn.layernorm_qkv.weight": (3 * width, width),
                f"{block}.attn.q_ln.weight": (width,),
                f"{block}.attn.k_ln.weight": (width,),
                f"{block}.attn.out_proj.weight": (width, width),
                f"{block}.ffn.layer_norm_weight": (width,),
                f"{block}.ffn.layer_norm_bias": (width,),
                f"{block}.ffn.fc1_weight": (2 * inner, width),
                f"{block}.ffn.fc2_weight": (width, inner),
            }
        )
    return expected


def expected_extra_states(layers: int) -> set[str]:
    return {
        f"esmc.transformer.blocks.{layer}.{suffix}._extra_state"
        for layer in range(layers)
        for suffix in ("attn.layernorm_qkv", "attn.out_proj", "ffn")
    }


def read_safetensors(path: Path) -> tuple[list[FileTensorSource], set[str]]:
    try:
        size = path.stat().st_size
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ConversionError(f"{path.name}: Safetensors prefix is truncated")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size == 0 or header_size > 16 * 1024 * 1024:
                raise ConversionError(
                    f"{path.name}: Safetensors header size is invalid"
                )
            header = json.loads(source.read(header_size))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversionError(
            f"cannot read source Safetensors {path}: {error}"
        ) from error
    if not isinstance(header, dict):
        raise ConversionError(f"{path.name}: Safetensors header must be an object")
    data_offset = 8 + header_size
    tensors: list[FileTensorSource] = []
    extras: set[str] = set()
    ranges: list[tuple[int, int, str]] = []
    zero_offsets: list[tuple[int, str]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            if descriptor != {"format": "pt"}:
                raise ConversionError(f"{path.name}: source metadata must be format=pt")
            continue
        if not isinstance(descriptor, dict) or not isinstance(name, str):
            raise ConversionError(f"{path.name}: tensor descriptor is invalid")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in offsets
            )
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or data_offset + offsets[1] > size
        ):
            raise ConversionError(f"{path.name}: tensor {name!r} descriptor is invalid")
        if EXTRA_STATE_RE.fullmatch(name):
            if (
                descriptor.get("dtype") != "U8"
                or shape != [0]
                or offsets[0] != offsets[1]
            ):
                raise ConversionError(
                    f"{path.name}: extra-state {name!r} is not zero-byte U8[0]"
                )
            if name in extras:
                raise ConversionError(f"{path.name}: duplicate extra-state {name!r}")
            extras.add(name)
            zero_offsets.append((offsets[0], name))
            continue
        if (
            descriptor.get("dtype") != "F32"
            or not shape
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in shape
            )
            or offsets[1] <= offsets[0]
        ):
            raise ConversionError(
                f"{path.name}: source tensor {name!r} must use nonempty F32"
            )
        count = 1
        for dimension in shape:
            count *= dimension
        if offsets[1] - offsets[0] != count * 4:
            raise ConversionError(
                f"{path.name}: source tensor {name!r} payload is invalid"
            )
        tensors.append(
            FileTensorSource(
                name, "F32", tuple(shape), path, data_offset + offsets[0], count * 4
            )
        )
        ranges.append((offsets[0], offsets[1], name))
    cursor = 0
    for begin, end, name in sorted(ranges):
        if begin != cursor:
            raise ConversionError(
                f"{path.name}: payload has a gap/overlap near {name!r}"
            )
        cursor = end
    if data_offset + cursor != size:
        raise ConversionError(f"{path.name}: tensors do not cover the complete payload")
    if any(offset != cursor for offset, _ in zero_offsets):
        raise ConversionError(
            f"{path.name}: zero-byte extra states must follow real payload"
        )
    return tensors, extras


def read_source_tensors(
    paths: dict[str, Path], topology: dict[str, int]
) -> list[FileTensorSource]:
    index_path = paths.get("model.safetensors.index.json")
    shard_names = sorted(name for name in paths if name.endswith(".safetensors"))
    if not shard_names:
        raise ConversionError("source receipt contains no Safetensors weights")
    tensors: list[FileTensorSource] = []
    extras: set[str] = set()
    owner: dict[str, str] = {}
    for name in shard_names:
        shard_tensors, shard_extras = read_safetensors(paths[name])
        for tensor in shard_tensors:
            if tensor.name in owner:
                raise ConversionError(f"source tensor {tensor.name!r} is duplicated")
            owner[tensor.name] = name
            tensors.append(tensor)
        for extra in shard_extras:
            if extra in owner:
                raise ConversionError(f"source extra-state {extra!r} is duplicated")
            owner[extra] = name
        extras.update(shard_extras)
    if index_path is not None:
        index = load_json(index_path, "ESMC Safetensors index")
        weight_map = index.get("weight_map")
        metadata = index.get("metadata")
        if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
            raise ConversionError(
                "ESMC Safetensors index is missing metadata/weight_map"
            )
        if weight_map != owner:
            raise ConversionError(
                "ESMC Safetensors index differs from shard descriptors"
            )
        total_size = metadata.get("total_size")
        if total_size != sum(tensor.nbytes for tensor in tensors):
            raise ConversionError("ESMC Safetensors index total_size is incorrect")
    elif len(shard_names) != 1 or shard_names[0] != "model.safetensors":
        raise ConversionError(
            "sharded ESMC source requires model.safetensors.index.json"
        )
    expected = expected_manifest(topology)
    actual = {tensor.name: tensor.shape for tensor in tensors}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong = sorted(
            name
            for name in set(actual) & set(expected)
            if actual[name] != expected[name]
        )
        raise ConversionError(
            f"ESMC tensor manifest mismatch: missing={missing}, extra={extra}, wrong_shape={wrong}"
        )
    expected_extras = expected_extra_states(topology["num_layers"])
    if extras != expected_extras:
        raise ConversionError(
            "ESMC extra-state manifest mismatch: "
            f"missing={sorted(expected_extras - extras)}, extra={sorted(extras - expected_extras)}"
        )
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=default_registry())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        model_id, entry, topology, paths, receipt_hash = validate_receipt(
            args.receipt.resolve(), args.registry.resolve()
        )
        validate_config(paths["config.json"], topology)
        tensors = read_source_tensors(paths, topology)
        metadata: dict[str, object] = {
            "runtime.abi": "esmc-safetensors-v1",
            "model.id": model_id,
            "model.architecture": "ESMC",
            "config.vocab_size": topology["vocab_size"],
            "config.hidden_size": topology["hidden_size"],
            "config.num_layers": topology["num_layers"],
            "config.num_attention_heads": topology["num_attention_heads"],
            "config.inner_mlp_size": topology["inner_mlp_size"],
            "config.max_seqlen": topology["max_seqlen"],
            "config.layer_norm_epsilon": 1e-5,
            "config.rope_base": 10000.0,
            "config.residue_scaling_factor": (topology["num_layers"] / 36.0) ** 0.5,
            "runtime.embedding_layer_count": topology["num_layers"] + 1,
            "tokenizer.kind": "esmc-protein-v1",
            "tokenizer.vocab_size": 64,
            "tokenizer.cls_token_id": 0,
            "tokenizer.pad_token_id": 1,
            "tokenizer.eos_token_id": 2,
            "tokenizer.unk_token_id": 3,
            "tokenizer.mask_token_id": 32,
            "tokenizer.sha256": str(
                next(
                    item["sha256"]
                    for item in entry["checkpoint_files"]
                    if item["name"] == "tokenizer.json"
                )
            ),
            "source.repo": str(entry["source_repo"]),
            "source.revision": str(entry["source_revision"]),
            "source.receipt_sha256": receipt_hash,
        }

        def progress(index: int, count: int, tensor: FileTensorSource) -> None:
            print(f"[{index}/{count}] {tensor.name}", file=sys.stderr)

        load_path = write_model(
            args.output,
            metadata,
            tensors,
            artifact_profile=ESMC_PROFILE_VALUE,
            force=args.force,
            progress=progress,
        )
        print(f"wrote {load_path}")
        print(f"source_receipt_sha256={receipt_hash}")
        return 0
    except (ConversionError, FormatError, FileExistsError, KeyError, OSError) as error:
        print(f"convert_esmc_checkpoint: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
