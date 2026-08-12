#!/usr/bin/env python3
"""Convert an official Hugging Face HyenaDNA F32 CausalLM artifact."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import struct
import sys
from collections.abc import Iterator
from pathlib import Path

from evo.format import HYENADNA_PROFILE_VALUE, FormatError, write_model


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


def integer(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversionError(f"config {key} must be a positive integer")
    return value


def number(config: dict[str, object], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionError(f"config {key} must be numeric")
    return float(value)


def load_config(path: Path) -> tuple[dict[str, object], dict[str, int | float | str]]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"cannot read HyenaDNA config: {error}") from error
    if not isinstance(config, dict):
        raise ConversionError("HyenaDNA config root must be an object")
    values: dict[str, int | float | str] = {
        "model_id": str(config.get("_name_or_path", "hyenadna")),
        "vocab_size": integer(config, "vocab_size"),
        "pad_vocab_size_multiple": integer(config, "pad_vocab_size_multiple"),
        "hidden_size": integer(config, "d_model"),
        "num_layers": integer(config, "n_layer"),
        "max_seqlen": integer(config, "max_seq_len"),
        "inner_mlp_size": integer(config, "d_inner"),
        "embedding_dim": integer(config, "emb_dim"),
        "filter_order": integer(config, "filter_order"),
        "hyena_order": integer(config, "hyena_order"),
        "num_inner_mlps": integer(config, "num_inner_mlps"),
        "short_filter_length": integer(config, "short_filter_order"),
        "layer_norm_epsilon": number(config, "layer_norm_epsilon"),
        "pad_token_id": integer(config, "pad_token_id"),
    }
    padded_vocab = int(values["vocab_size"])
    multiple = int(values["pad_vocab_size_multiple"])
    padded_vocab += (-padded_vocab) % multiple
    if (
        config.get("model_type") != "hyenadna"
        or config.get("torch_dtype") != "float32"
        or config.get("use_bias") is not True
        or config.get("train_freq") is not True
        or config.get("tie_word_embeddings") is not False
        or values["vocab_size"] != 12
        or padded_vocab != 16
        or values["hyena_order"] != 2
        or values["short_filter_length"] != 3
        or int(values["embedding_dim"]) < 3
        or int(values["embedding_dim"]) % 2 == 0
        or int(values["max_seqlen"]) > 4096
        or not (0.0 < float(values["layer_norm_epsilon"]) < 1.0)
    ):
        raise ConversionError(
            "config is outside the registered F32 HyenaDNA direct-convolution profile"
        )
    values["padded_vocab_size"] = padded_vocab
    return config, values


def expected_manifest(values: dict[str, int | float | str]) -> dict[str, tuple[int, ...]]:
    width = int(values["hidden_size"])
    inner = int(values["inner_mlp_size"])
    order = int(values["filter_order"])
    emb = int(values["embedding_dim"])
    length = int(values["max_seqlen"])
    projected = width * 3
    expected: dict[str, tuple[int, ...]] = {
        "hyena.backbone.embeddings.word_embeddings.weight": (16, width),
        "hyena.backbone.ln_f.weight": (width,),
        "hyena.backbone.ln_f.bias": (width,),
        "lm_head.weight": (16, width),
    }
    for layer in range(int(values["num_layers"])):
        prefix = f"hyena.backbone.layers.{layer}"
        mixer = f"{prefix}.mixer"
        filter_prefix = f"{mixer}.filter_fn"
        expected.update(
            {
                f"{prefix}.norm1.weight": (width,),
                f"{prefix}.norm1.bias": (width,),
                f"{prefix}.norm2.weight": (width,),
                f"{prefix}.norm2.bias": (width,),
                f"{mixer}.in_proj.weight": (projected, width),
                f"{mixer}.in_proj.bias": (projected,),
                f"{mixer}.short_filter.weight": (projected, 1, 3),
                f"{mixer}.short_filter.bias": (projected,),
                f"{mixer}.out_proj.weight": (width, width),
                f"{mixer}.out_proj.bias": (width,),
                f"{filter_prefix}.bias": (width,),
                f"{filter_prefix}.implicit_filter.0.weight": (order, emb),
                f"{filter_prefix}.implicit_filter.0.bias": (order,),
                f"{filter_prefix}.implicit_filter.1.freq": (1, order),
                f"{filter_prefix}.modulation.deltas": (1, 1, width),
                f"{filter_prefix}.pos_emb.t": (1, length, 1),
                f"{filter_prefix}.pos_emb.z": (1, length, emb),
                f"{prefix}.mlp.fc1.weight": (inner, width),
                f"{prefix}.mlp.fc1.bias": (inner,),
                f"{prefix}.mlp.fc2.weight": (width, inner),
                f"{prefix}.mlp.fc2.bias": (width,),
            }
        )
        for index in range(int(values["num_inner_mlps"])):
            module = 2 + index * 2
            expected[f"{filter_prefix}.implicit_filter.{module}.weight"] = (
                order,
                order,
            )
            expected[f"{filter_prefix}.implicit_filter.{module}.bias"] = (order,)
        output_module = 2 + int(values["num_inner_mlps"]) * 2
        expected[f"{filter_prefix}.implicit_filter.{output_module}.weight"] = (
            width,
            order,
        )
    return expected


def read_safetensors(path: Path) -> list[FileTensorSource]:
    try:
        size = path.stat().st_size
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ConversionError("source Safetensors prefix is truncated")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size == 0 or header_size > 16 * 1024 * 1024:
                raise ConversionError("source Safetensors header size is invalid")
            header = json.loads(source.read(header_size))
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"cannot read source Safetensors: {error}") from error
    if not isinstance(header, dict):
        raise ConversionError("source Safetensors header must be an object")
    data_offset = 8 + header_size
    tensors: list[FileTensorSource] = []
    ranges: list[tuple[int, int, str]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(descriptor, dict) or descriptor.get("dtype") != "F32":
            raise ConversionError(f"source tensor {name!r} must use F32")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(shape, list)
            or not shape
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
        ):
            raise ConversionError(f"source tensor {name!r} descriptor is invalid")
        count = 1
        for dimension in shape:
            count *= dimension
        if offsets[1] - offsets[0] != count * 4 or data_offset + offsets[1] > size:
            raise ConversionError(f"source tensor {name!r} payload is invalid")
        tensors.append(
            FileTensorSource(
                name,
                "F32",
                tuple(shape),
                path,
                data_offset + offsets[0],
                count * 4,
            )
        )
        ranges.append((offsets[0], offsets[1], name))
    cursor = 0
    for begin, end, name in sorted(ranges):
        if begin != cursor:
            raise ConversionError(f"source payload has a gap/overlap near {name!r}")
        cursor = end
    if data_offset + cursor != size:
        raise ConversionError("source tensors do not cover the complete payload")
    return tensors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--source-sha256")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        _, values = load_config(args.config)
        tensors = read_safetensors(args.input)
        expected = expected_manifest(values)
        actual = {tensor.name: tensor.shape for tensor in tensors}
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            wrong = sorted(
                name for name in set(actual) & set(expected) if actual[name] != expected[name]
            )
            raise ConversionError(
                f"HyenaDNA tensor manifest mismatch: missing={missing}, extra={extra}, wrong_shape={wrong}"
            )
        source_hash = sha256(args.input)
        if args.source_sha256 is not None and source_hash != args.source_sha256.lower():
            raise ConversionError("source SHA256 does not match --source-sha256")
        metadata: dict[str, object] = {
            "runtime.abi": "hyenadna-safetensors-v1",
            "model.id": args.model_id or values["model_id"],
            "model.architecture": "HyenaDNA",
            "config.vocab_size": values["padded_vocab_size"],
            "config.source_vocab_size": values["vocab_size"],
            "config.hidden_size": values["hidden_size"],
            "config.num_layers": values["num_layers"],
            "config.max_seqlen": values["max_seqlen"],
            "config.inner_mlp_size": values["inner_mlp_size"],
            "config.embedding_dim": values["embedding_dim"],
            "config.filter_order": values["filter_order"],
            "config.hyena_order": values["hyena_order"],
            "config.num_inner_mlps": values["num_inner_mlps"],
            "config.short_filter_length": values["short_filter_length"],
            "config.layer_norm_epsilon": values["layer_norm_epsilon"],
            "config.pad_token_id": values["pad_token_id"],
            "tokenizer.kind": "hyenadna-character-v1",
            "checkpoint.sha256": source_hash,
            "checkpoint.source": args.input.name,
        }
        if args.revision is not None:
            metadata["checkpoint.revision"] = args.revision
        load_path = write_model(
            args.output,
            metadata,
            tensors,
            artifact_profile=HYENADNA_PROFILE_VALUE,
            force=args.force,
        )
        print(f"wrote {load_path}")
        print(f"source_sha256={source_hash}")
        return 0
    except (ConversionError, FormatError, FileExistsError, OSError) as error:
        print(f"convert_hyenadna_checkpoint: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
