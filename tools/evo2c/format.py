"""Streaming writer for the EVO2C v1 model container."""

from __future__ import annotations

import dataclasses
import os
import re
import struct
import tempfile
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol


MAGIC = b"EVO2C\0\0\0"
HEADER_SIZE = 128
DESCRIPTOR_SIZE = 256
NAME_CAPACITY = 96
MAX_RANK = 8
ALIGNMENT = 64
HEADER_CRC_OFFSET = 80
DESCRIPTOR_CRC_OFFSET = 196
DEFAULT_CHUNK_SIZE = 16 * 1024 * 1024

DTYPE_IDS = {"F32": 1, "BF16": 2, "Q8_0": 3, "E4M3_SW": 4}
METADATA_TYPE_IDS = {"string": 1, "u64": 2, "f64": 3, "bool": 4, "u64[]": 5, "bytes": 6}
KEY_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class FormatError(ValueError):
    """Raised when an input cannot be represented by EVO2C v1."""


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


@dataclasses.dataclass(frozen=True, slots=True)
class _TensorRecord:
    source: TensorSource
    offset: int
    crc32: int


def _align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


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
    if dtype == "Q8_0":
        if elements % 32:
            raise FormatError("Q8_0 element count must be divisible by 32")
        return elements // 32 * 34
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
            f"tensor {source.name!r} has {source.nbytes} bytes; {source.dtype}{source.shape} requires {expected}"
        )


def _encode_metadata_value(value: object) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return METADATA_TYPE_IDS["bool"], bytes([int(value)])
    if isinstance(value, int):
        if value < 0 or value >= 1 << 64:
            raise FormatError(f"metadata integer outside uint64: {value}")
        return METADATA_TYPE_IDS["u64"], struct.pack("<Q", value)
    if isinstance(value, float):
        return METADATA_TYPE_IDS["f64"], struct.pack("<d", value)
    if isinstance(value, str):
        return METADATA_TYPE_IDS["string"], value.encode("utf-8")
    if isinstance(value, bytes):
        return METADATA_TYPE_IDS["bytes"], value
    if isinstance(value, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        if any(item < 0 or item >= 1 << 64 for item in value):
            raise FormatError("metadata u64 list contains an out-of-range value")
        return METADATA_TYPE_IDS["u64[]"], b"".join(struct.pack("<Q", item) for item in value)
    raise FormatError(f"unsupported metadata value type: {type(value).__name__}")


def encode_metadata(metadata: Mapping[str, object]) -> bytes:
    if len(metadata) > 4096:
        raise FormatError("metadata entry count exceeds 4096")
    section = bytearray(16)
    section[:4] = b"META"
    struct.pack_into("<H", section, 4, 1)
    struct.pack_into("<I", section, 8, len(metadata))
    for key in sorted(metadata):
        if not KEY_PATTERN.fullmatch(key) or len(key.encode("ascii")) > 255:
            raise FormatError(f"invalid metadata key {key!r}")
        type_id, encoded_value = _encode_metadata_value(metadata[key])
        encoded_key = key.encode("ascii")
        section.extend(struct.pack("<HBBI", len(encoded_key), type_id, 0, len(encoded_value)))
        section.extend(encoded_key)
        section.extend(encoded_value)
        section.extend(b"\0" * (_align_up(len(section), 8) - len(section)))
    struct.pack_into("<I", section, 12, zlib.crc32(section[16:]))
    if len(section) > 16 * 1024 * 1024:
        raise FormatError("metadata section exceeds 16 MiB")
    return bytes(section)


def _descriptor(record: _TensorRecord) -> bytes:
    source = record.source
    descriptor = bytearray(DESCRIPTOR_SIZE)
    encoded_name = source.name.encode("ascii")
    descriptor[: len(encoded_name)] = encoded_name
    descriptor[96] = DTYPE_IDS[source.dtype]
    descriptor[97] = len(source.shape)
    elements = 1
    for index, dimension in enumerate(source.shape):
        struct.pack_into("<Q", descriptor, 104 + index * 8, dimension)
        elements *= dimension
    struct.pack_into("<Q", descriptor, 168, record.offset)
    struct.pack_into("<Q", descriptor, 176, source.nbytes)
    struct.pack_into("<Q", descriptor, 184, elements)
    struct.pack_into("<I", descriptor, 192, record.crc32)
    struct.pack_into("<I", descriptor, DESCRIPTOR_CRC_OFFSET, 0)
    struct.pack_into("<I", descriptor, DESCRIPTOR_CRC_OFFSET, zlib.crc32(descriptor))
    return bytes(descriptor)


def _header(
    *,
    file_size: int,
    metadata_size: int,
    table_offset: int,
    tensor_count: int,
    data_offset: int,
) -> bytes:
    header = bytearray(HEADER_SIZE)
    header[:8] = MAGIC
    struct.pack_into("<I", header, 8, 1)
    struct.pack_into("<I", header, 12, 0x01020304)
    struct.pack_into("<I", header, 16, HEADER_SIZE)
    struct.pack_into("<Q", header, 24, file_size)
    struct.pack_into("<Q", header, 32, HEADER_SIZE)
    struct.pack_into("<Q", header, 40, metadata_size)
    struct.pack_into("<Q", header, 48, table_offset)
    struct.pack_into("<Q", header, 56, tensor_count)
    struct.pack_into("<I", header, 64, DESCRIPTOR_SIZE)
    struct.pack_into("<I", header, 68, ALIGNMENT)
    struct.pack_into("<Q", header, 72, data_offset)
    struct.pack_into("<I", header, HEADER_CRC_OFFSET, 0)
    struct.pack_into("<I", header, HEADER_CRC_OFFSET, zlib.crc32(header))
    return bytes(header)


def _publish(temp_path: Path, output_path: Path, *, force: bool) -> None:
    if force:
        os.replace(temp_path, output_path)
        return
    try:
        os.link(temp_path, output_path)
    except FileExistsError:
        raise FileExistsError(f"output already exists: {output_path}") from None
    else:
        temp_path.unlink()


def write_model(
    output_path: Path,
    metadata: Mapping[str, object],
    tensors: Sequence[TensorSource],
    *,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: Callable[[int, int, TensorSource], None] | None = None,
) -> None:
    """Write a model atomically while holding at most one chunk of tensor data."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not tensors:
        raise FormatError("model must contain at least one tensor")
    if len(tensors) > 1_000_000:
        raise FormatError("tensor count exceeds 1000000")
    names: set[str] = set()
    for source in tensors:
        _validate_tensor(source)
        if source.name in names:
            raise FormatError(f"duplicate tensor name {source.name!r}")
        names.add(source.name)

    encoded_metadata = encode_metadata(metadata)
    table_offset = _align_up(HEADER_SIZE + len(encoded_metadata))
    data_offset = _align_up(table_offset + len(tensors) * DESCRIPTOR_SIZE)
    planned_offsets: list[int] = []
    cursor = data_offset
    for source in tensors:
        cursor = _align_up(cursor)
        planned_offsets.append(cursor)
        cursor += source.nbytes
        if cursor >= 1 << 64:
            raise FormatError("model file size exceeds uint64")
    file_size = cursor

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(f"output already exists: {output_path}")

    temporary = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temp_path = Path(temporary.name)
    records: list[_TensorRecord] = []
    try:
        with temporary as output:
            output.write(b"\0" * data_offset)
            output.seek(HEADER_SIZE)
            output.write(encoded_metadata)

            for index, (source, offset) in enumerate(zip(tensors, planned_offsets, strict=True), start=1):
                if progress is not None:
                    progress(index, len(tensors), source)
                output.seek(offset)
                crc = 0
                written = 0
                for raw_chunk in source.iter_chunks(chunk_size):
                    chunk = memoryview(raw_chunk)
                    if not chunk.contiguous:
                        raise FormatError(f"tensor {source.name!r} yielded a non-contiguous chunk")
                    chunk = chunk.cast("B")
                    if written + len(chunk) > source.nbytes:
                        raise FormatError(f"tensor {source.name!r} yielded more bytes than declared")
                    output.write(chunk)
                    crc = zlib.crc32(chunk, crc)
                    written += len(chunk)
                if written != source.nbytes:
                    raise FormatError(
                        f"tensor {source.name!r} yielded {written} bytes, expected {source.nbytes}"
                    )
                records.append(_TensorRecord(source=source, offset=offset, crc32=crc))

            output.truncate(file_size)
            output.seek(table_offset)
            for record in records:
                output.write(_descriptor(record))
            output.seek(0)
            output.write(
                _header(
                    file_size=file_size,
                    metadata_size=len(encoded_metadata),
                    table_offset=table_offset,
                    tensor_count=len(tensors),
                    data_offset=data_offset,
                )
            )
            output.flush()
            os.fsync(output.fileno())
        _publish(temp_path, output_path, force=force)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
