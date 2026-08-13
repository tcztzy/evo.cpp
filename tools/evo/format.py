"""Deterministic streaming writer for runtime-ready Safetensors shards."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import struct
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol


EVO2_PROFILE_KEY = "evo2.profile"
EVO2_PROFILE_VALUE = "evo2-runtime-v1"
RUNTIME_PROFILE_KEY = "runtime.profile"
HYENADNA_PROFILE_VALUE = "hyenadna-runtime-v1"
ESMC_PROFILE_VALUE = "esmc-runtime-v1"
HEADER_ALIGNMENT = 8
MAX_HEADER_SIZE = 16 * 1024 * 1024
MAX_RANK = 8
NAME_CAPACITY = 96
DEFAULT_CHUNK_SIZE = 16 * 1024 * 1024
DEFAULT_MAX_SHARD_SIZE = 4 * 1024 * 1024 * 1024

SAFETENSORS_DTYPES = {
    "F32": "F32",
    "BF16": "BF16",
    "E4M3_SW": "F8_E4M3",
}
KEY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class FormatError(ValueError):
    """Raised when an input cannot be represented by the runtime profile."""


class TensorSource(Protocol):
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class BytesTensorSource:
    name: str
    dtype: str
    shape: tuple[int, ...]
    payload: bytes

    @property
    def nbytes(self) -> int:
        return len(self.payload)

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        view = memoryview(self.payload)
        for offset in range(0, len(view), chunk_size):
            yield view[offset : offset + chunk_size]


def _expected_nbytes(dtype: str, shape: tuple[int, ...]) -> int:
    elements = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise FormatError(f"invalid tensor dimension {dimension!r}")
        elements *= dimension
        if elements >= 1 << 64:
            raise FormatError("tensor element count exceeds uint64")
    if dtype == "F32":
        return elements * 4
    if dtype == "BF16":
        return elements * 2
    if dtype == "E4M3_SW":
        return elements
    raise FormatError(f"unsupported tensor dtype {dtype!r}")


def _validate_tensor(source: TensorSource) -> None:
    try:
        encoded_name = source.name.encode("ascii")
    except UnicodeEncodeError as error:
        raise FormatError(f"tensor name must be ASCII: {source.name!r}") from error
    if not KEY_PATTERN.fullmatch(source.name):
        raise FormatError(f"tensor name has invalid characters: {source.name!r}")
    if len(encoded_name) >= NAME_CAPACITY:
        raise FormatError(f"tensor name exceeds {NAME_CAPACITY - 1} bytes: {source.name!r}")
    if len(source.shape) > MAX_RANK:
        raise FormatError(f"tensor {source.name!r} rank exceeds {MAX_RANK}")
    expected = _expected_nbytes(source.dtype, source.shape)
    if source.nbytes != expected:
        raise FormatError(
            f"tensor {source.name!r} has {source.nbytes} bytes; "
            f"{source.dtype}{source.shape} requires {expected}"
        )


def _encode_metadata_value(value: object) -> str:
    if isinstance(value, bool):
        return f"b:{int(value)}"
    if isinstance(value, int):
        if value < 0 or value >= 1 << 64:
            raise FormatError(f"metadata integer outside uint64: {value}")
        return f"u:{value}"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FormatError("metadata float must be finite")
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        return f"f:{bits:016x}"
    if isinstance(value, str):
        return f"s:{value}"
    if isinstance(value, bytes):
        return f"x:{value.hex()}"
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        if any(item < 0 or item >= 1 << 64 for item in value):
            raise FormatError("metadata u64 list contains an out-of-range value")
        return "l:" + ",".join(str(item) for item in value)
    raise FormatError(f"unsupported metadata value type: {type(value).__name__}")


def encode_metadata(
    metadata: Mapping[str, object], artifact_profile: str
) -> dict[str, str]:
    if len(metadata) >= 4096:
        raise FormatError("metadata entry count exceeds 4095")
    encoded: dict[str, str] = {}
    for key in sorted(metadata):
        if not KEY_PATTERN.fullmatch(key) or len(key.encode("ascii")) > 255:
            raise FormatError(f"invalid metadata key {key!r}")
        if key in {EVO2_PROFILE_KEY, RUNTIME_PROFILE_KEY}:
            raise FormatError(f"{key} is reserved by the writer")
        encoded[key] = _encode_metadata_value(metadata[key])
    if artifact_profile == EVO2_PROFILE_VALUE:
        encoded[EVO2_PROFILE_KEY] = f"s:{EVO2_PROFILE_VALUE}"
    elif artifact_profile in {HYENADNA_PROFILE_VALUE, ESMC_PROFILE_VALUE}:
        encoded[RUNTIME_PROFILE_KEY] = f"s:{artifact_profile}"
    else:
        raise FormatError(f"unsupported artifact profile {artifact_profile!r}")
    return dict(sorted(encoded.items()))


def _encode_header(
    metadata: Mapping[str, object],
    tensors: Sequence[TensorSource],
    artifact_profile: str,
) -> bytes:
    root: dict[str, object] = {
        "__metadata__": encode_metadata(metadata, artifact_profile)
    }
    offset = 0
    for source in tensors:
        end = offset + source.nbytes
        if end >= 1 << 64:
            raise FormatError("Safetensors data buffer exceeds uint64")
        root[source.name] = {
            "dtype": SAFETENSORS_DTYPES[source.dtype],
            "shape": list(source.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    raw = json.dumps(
        root,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(raw)) % HEADER_ALIGNMENT
    header = raw + b" " * padding
    if len(header) == 0 or len(header) > MAX_HEADER_SIZE:
        raise FormatError("Safetensors header exceeds 16 MiB")
    return header


def _ordered_tensors(tensors: Sequence[TensorSource]) -> list[TensorSource]:
    by_name = {source.name: source for source in tensors}
    globals_first = [
        by_name[name]
        for name in ("embedding_layer.weight", "unembed.weight", "norm.scale")
        if name in by_name
    ]
    global_names = {source.name for source in globals_first}
    return [
        *globals_first,
        *(source for source in tensors if source.name not in global_names),
    ]


def _split_tensors(
    tensors: Sequence[TensorSource], max_shard_size: int
) -> list[list[TensorSource]]:
    if max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive")
    shards: list[list[TensorSource]] = []
    shard: list[TensorSource] = []
    shard_size = 0
    for source in tensors:
        if shard and shard_size + source.nbytes > max_shard_size:
            shards.append(shard)
            shard = []
            shard_size = 0
        shard.append(source)
        shard_size += source.nbytes
    if shard:
        shards.append(shard)
    return shards


def plan_shards(
    tensors: Sequence[TensorSource],
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
) -> list[list[TensorSource]]:
    """Plan deterministic tensor-boundary shards in runtime load order."""
    return _split_tensors(_ordered_tensors(tensors), max_shard_size)


def _artifact_paths(output_path: Path, shard_count: int) -> tuple[list[Path], Path]:
    if output_path.suffix != ".safetensors":
        raise FormatError("output path must end in .safetensors")
    if shard_count == 1:
        return [output_path], output_path
    stem = output_path.name.removesuffix(".safetensors")
    shards = [
        output_path.with_name(
            f"{stem}-{index:05d}-of-{shard_count:05d}.safetensors"
        )
        for index in range(1, shard_count + 1)
    ]
    return shards, output_path.with_name(output_path.name + ".index.json")


def _temporary_path(output_path: Path) -> tuple[object, Path]:
    temporary = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    return temporary, Path(temporary.name)


def _write_safetensors(
    output_path: Path,
    metadata: Mapping[str, object],
    tensors: Sequence[TensorSource],
    *,
    artifact_profile: str,
    chunk_size: int,
    progress: Callable[[TensorSource], None] | None,
) -> Path:
    encoded_header = _encode_header(metadata, tensors, artifact_profile)
    temporary, temp_path = _temporary_path(output_path)
    try:
        with temporary as output:
            total_size = (
                8 + len(encoded_header) + sum(source.nbytes for source in tensors)
            )
            output.truncate(total_size)
            output.seek(0)
            output.write(struct.pack("<Q", len(encoded_header)))
            output.write(encoded_header)
            for source in tensors:
                if progress is not None:
                    progress(source)
                written = 0
                for raw_chunk in source.iter_chunks(chunk_size):
                    chunk = memoryview(raw_chunk)
                    if not chunk.contiguous:
                        raise FormatError(
                            f"tensor {source.name!r} yielded a non-contiguous chunk"
                        )
                    chunk = chunk.cast("B")
                    if written + len(chunk) > source.nbytes:
                        raise FormatError(
                            f"tensor {source.name!r} yielded more bytes than declared"
                        )
                    output.write(chunk)
                    written += len(chunk)
                if written != source.nbytes:
                    raise FormatError(
                        f"tensor {source.name!r} yielded {written} bytes, "
                        f"expected {source.nbytes}"
                    )
            output.flush()
            os.fsync(output.fileno())
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _write_index(
    output_path: Path,
    tensors: Sequence[TensorSource],
    shard_paths: Sequence[Path],
    shards: Sequence[Sequence[TensorSource]],
) -> Path:
    weight_map = {
        source.name: shard_path.name
        for shard_path, shard in zip(shard_paths, shards, strict=True)
        for source in shard
    }
    if len(weight_map) != len(tensors):
        raise FormatError("internal error while building shard weight map")
    payload = {
        "metadata": {"total_size": sum(source.nbytes for source in tensors)},
        "weight_map": weight_map,
    }
    temporary, temp_path = _temporary_path(output_path)
    try:
        with temporary as output:
            output.write(
                json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode(
                    "ascii"
                )
            )
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _publish_artifacts(
    temporary_paths: Sequence[Path],
    output_paths: Sequence[Path],
    *,
    force: bool,
) -> None:
    if len(temporary_paths) != len(output_paths):
        raise FormatError("internal error while publishing model artifacts")
    if not force:
        for path in output_paths:
            if path.exists():
                raise FileExistsError(f"output already exists: {path}")

    published: list[Path] = []
    try:
        for temporary, output in zip(temporary_paths, output_paths, strict=True):
            if force:
                os.replace(temporary, output)
            else:
                os.link(temporary, output)
                temporary.unlink()
                published.append(output)
    except BaseException:
        if not force:
            for path in published:
                path.unlink(missing_ok=True)
        raise


def write_model(
    output_path: Path,
    metadata: Mapping[str, object],
    tensors: Sequence[TensorSource],
    *,
    artifact_profile: str,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    progress: Callable[[int, int, TensorSource], None] | None = None,
) -> Path:
    """Write one Safetensors file or standard size-based shards.

    The returned path is the path the runtime should open: the Safetensors file
    for a single shard, or ``model.safetensors.index.json`` for multiple shards.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_shard_size <= 0:
        raise ValueError("max_shard_size must be positive")
    if not tensors:
        raise FormatError("model must contain at least one tensor")
    if len(tensors) > 1_000_000:
        raise FormatError("tensor count exceeds 1000000")

    names: set[str] = set()
    for source in tensors:
        _validate_tensor(source)
        if source.name == "__metadata__" or source.name in names:
            raise FormatError(f"duplicate or reserved tensor name {source.name!r}")
        names.add(source.name)
    shards = plan_shards(tensors, max_shard_size)
    tensors = [source for shard in shards for source in shard]

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_paths, load_path = _artifact_paths(output_path, len(shards))
    output_paths = [*shard_paths]
    if len(shards) > 1:
        output_paths.append(load_path)
    if not force:
        preflight_paths = {*output_paths, output_path}
        for path in preflight_paths:
            if path.exists():
                raise FileExistsError(f"output already exists: {path}")

    written = 0

    def report(source: TensorSource) -> None:
        nonlocal written
        written += 1
        if progress is not None:
            progress(written, len(tensors), source)

    temporary_paths: list[Path] = []
    try:
        for shard_path, shard in zip(shard_paths, shards, strict=True):
            temporary_paths.append(
                _write_safetensors(
                    shard_path,
                    metadata,
                    shard,
                    artifact_profile=artifact_profile,
                    chunk_size=chunk_size,
                    progress=report,
                )
            )
        if len(shards) > 1:
            temporary_paths.append(
                _write_index(load_path, tensors, shard_paths, shards)
            )
        _publish_artifacts(temporary_paths, output_paths, force=force)
    except BaseException:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    return load_path
