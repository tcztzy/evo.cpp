#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict Python 3.8-compatible readers for verified HF checkpoints.

The native runtime never imports this module.  It is shared by offline
converters which need to validate standard Hugging Face Safetensors shards or
an explicitly permitted ``pytorch_model.bin`` before publishing an artifact.
"""

import dataclasses
import importlib
import json
import re
import struct
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple


MAX_HEADER_SIZE = 16 * 1024 * 1024
MAX_TENSOR_COUNT = 1000000
TENSOR_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
# I64 is a validation-only source dtype. Readers reject it by default; an
# architecture converter must opt in for a closed manifest of buffers which it
# validates and drops. Runtime artifacts remain limited to F32/BF16 tensors.
DTYPE_BYTES = {"F32": 4, "BF16": 2, "I64": 8}
DEFAULT_SOURCE_DTYPES = ("F32", "BF16")


class CheckpointError(ValueError):
    """Raised when a checkpoint is outside the closed conversion contract."""


def _duplicate_checked_object(
    pairs: Sequence[Tuple[str, Any]]
) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        if key in result:
            raise CheckpointError("JSON object contains duplicate key %r" % key)
        result[key] = value
    return result


def load_json_bytes(payload: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_duplicate_checked_object
        )
    except CheckpointError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError("cannot parse %s: %s" % (label, error))
    if not isinstance(value, dict):
        raise CheckpointError("%s root must be an object" % label)
    return value


def load_json(path: Path, label: str) -> Tuple[Dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CheckpointError("cannot read %s %s: %s" % (label, path, error))
    return load_json_bytes(payload, label), payload


def normalized_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CheckpointError("%s must be a nonempty relative path" % label)
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise CheckpointError("%s must be a normalized relative path" % label)
    return value


def checked_shape(value: Any, label: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or not value or len(value) > 8:
        raise CheckpointError("%s must be a nonempty rank-1..8 array" % label)
    shape = []  # type: List[int]
    elements = 1
    for index, dimension in enumerate(value):
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            or dimension >= (1 << 64)
        ):
            raise CheckpointError("%s[%d] is not a positive uint64" % (label, index))
        if elements > ((1 << 64) - 1) // dimension:
            raise CheckpointError("%s element count exceeds uint64" % label)
        elements *= dimension
        shape.append(dimension)
    return tuple(shape)


def tensor_nbytes(dtype: str, shape: Sequence[int]) -> int:
    element_bytes = DTYPE_BYTES.get(dtype)
    if element_bytes is None:
        raise CheckpointError(
            "unsupported source dtype %r; only F32, BF16, and validation-only I64 are accepted"
            % dtype
        )
    elements = 1
    for dimension in shape:
        if dimension <= 0 or elements > ((1 << 64) - 1) // dimension:
            raise CheckpointError("tensor shape is zero or exceeds uint64")
        elements *= dimension
    if elements > ((1 << 64) - 1) // element_bytes:
        raise CheckpointError("tensor payload exceeds uint64")
    return elements * element_bytes


def _checked_allowed_dtypes(allowed_dtypes: Sequence[str]) -> Tuple[str, ...]:
    result = tuple(allowed_dtypes)
    if (
        not result
        or len(set(result)) != len(result)
        or any(dtype not in DTYPE_BYTES for dtype in result)
    ):
        raise CheckpointError(
            "allowed source dtypes must be a nonempty unique subset of F32/BF16/I64"
        )
    return result


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: Tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return tensor_nbytes(self.dtype, self.shape)


@dataclasses.dataclass(frozen=True)
class FileTensorSource:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    path: Path
    offset: int
    nbytes: int

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        try:
            with self.path.open("rb") as source:
                source.seek(self.offset)
                remaining = self.nbytes
                while remaining:
                    chunk = source.read(min(chunk_size, remaining))
                    if not chunk:
                        raise CheckpointError(
                            "source tensor %r ended before its verified extent"
                            % self.name
                        )
                    remaining -= len(chunk)
                    yield memoryview(chunk)
        except OSError as error:
            raise CheckpointError(
                "cannot stream source tensor %r: %s" % (self.name, error)
            )


@dataclasses.dataclass(frozen=True)
class TorchTensorSource:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    nbytes: int
    tensor: Any

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        torch = importlib.import_module("torch")
        raw = self.tensor.detach().view(torch.uint8).reshape(-1).numpy()
        view = memoryview(raw).cast("B")
        if len(view) != self.nbytes:
            raise CheckpointError(
                "tensor %r exposed %d bytes; expected %d"
                % (self.name, len(view), self.nbytes)
            )
        for offset in range(0, self.nbytes, chunk_size):
            yield view[offset : offset + min(chunk_size, self.nbytes - offset)]


TensorSource = Any


def _validate_tensor_name(name: Any, label: str) -> str:
    if (
        not isinstance(name, str)
        or not TENSOR_NAME_RE.fullmatch(name)
        or len(name.encode("ascii")) >= 256
        or name == "__metadata__"
    ):
        raise CheckpointError("%s has an invalid tensor name %r" % (label, name))
    return name


def read_safetensors(
    path: Path,
    allowed_dtypes: Sequence[str] = DEFAULT_SOURCE_DTYPES,
) -> List[FileTensorSource]:
    """Parse one complete Safetensors file without loading tensor payloads."""
    checked_dtypes = _checked_allowed_dtypes(allowed_dtypes)
    try:
        file_size = path.stat().st_size
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise CheckpointError("%s: Safetensors prefix is truncated" % path.name)
            header_size = struct.unpack("<Q", prefix)[0]
            if (
                header_size == 0
                or header_size > MAX_HEADER_SIZE
                or header_size % 8 != 0
                or 8 + header_size > file_size
            ):
                raise CheckpointError(
                    "%s: Safetensors header extent is invalid" % path.name
                )
            header_payload = source.read(header_size)
            if len(header_payload) != header_size:
                raise CheckpointError(
                    "%s: Safetensors header is truncated" % path.name
                )
    except CheckpointError:
        raise
    except OSError as error:
        raise CheckpointError("cannot open source Safetensors %s: %s" % (path, error))

    header = load_json_bytes(header_payload, "%s Safetensors header" % path.name)
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in metadata.items())
    ):
        raise CheckpointError(
            "%s: __metadata__ must contain only string pairs" % path.name
        )
    if not header or len(header) > MAX_TENSOR_COUNT:
        raise CheckpointError(
            "%s: tensor count must be in [1,%d]" % (path.name, MAX_TENSOR_COUNT)
        )

    data_start = 8 + header_size
    ranges = []  # type: List[Tuple[int, int, str]]
    tensors = []  # type: List[FileTensorSource]
    for raw_name, raw_descriptor in header.items():
        name = _validate_tensor_name(raw_name, path.name)
        if not isinstance(raw_descriptor, dict) or set(raw_descriptor) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise CheckpointError(
                "%s: tensor %r descriptor fields are invalid" % (path.name, name)
            )
        dtype = raw_descriptor["dtype"]
        if dtype not in checked_dtypes:
            raise CheckpointError(
                "%s: tensor %r uses source dtype %r outside allowed set %s"
                % (path.name, name, dtype, checked_dtypes)
            )
        shape = checked_shape(raw_descriptor["shape"], "%s:%s.shape" % (path.name, name))
        offsets = raw_descriptor["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] <= offsets[0]
            or offsets[1] >= (1 << 64)
        ):
            raise CheckpointError(
                "%s: tensor %r data_offsets are invalid" % (path.name, name)
            )
        nbytes = tensor_nbytes(dtype, shape)
        if offsets[1] - offsets[0] != nbytes or data_start + offsets[1] > file_size:
            raise CheckpointError(
                "%s: tensor %r payload extent does not match %s%s"
                % (path.name, name, dtype, shape)
            )
        tensors.append(
            FileTensorSource(
                name=name,
                dtype=dtype,
                shape=shape,
                path=path,
                offset=data_start + offsets[0],
                nbytes=nbytes,
            )
        )
        ranges.append((offsets[0], offsets[1], name))

    cursor = 0
    for begin, end, name in sorted(ranges):
        if begin != cursor:
            raise CheckpointError(
                "%s: tensor payload has a gap/overlap before %r" % (path.name, name)
            )
        cursor = end
    if data_start + cursor != file_size:
        raise CheckpointError(
            "%s: tensors do not cover the complete file payload" % path.name
        )
    return tensors


def _validate_index(
    index_path: Path,
    owners: Mapping[str, str],
    total_size: int,
) -> None:
    index, _ = load_json(index_path, "HF Safetensors shard index")
    if set(index) != {"metadata", "weight_map"}:
        raise CheckpointError(
            "HF Safetensors index fields must be exactly metadata/weight_map"
        )
    metadata = index["metadata"]
    weight_map = index["weight_map"]
    if not isinstance(metadata, dict) or set(metadata) != {"total_size"}:
        raise CheckpointError(
            "HF Safetensors index metadata must contain exactly total_size"
        )
    if (
        isinstance(metadata["total_size"], bool)
        or not isinstance(metadata["total_size"], int)
        or metadata["total_size"] != total_size
    ):
        raise CheckpointError("HF Safetensors index total_size is incorrect")
    if not isinstance(weight_map, dict):
        raise CheckpointError("HF Safetensors index weight_map must be an object")
    normalized = {}  # type: Dict[str, str]
    for raw_name, raw_owner in weight_map.items():
        name = _validate_tensor_name(raw_name, "HF Safetensors index")
        owner = normalized_relative_path(raw_owner, "weight_map[%s]" % name)
        if PurePosixPath(owner).name != owner or not owner.endswith(".safetensors"):
            raise CheckpointError(
                "weight_map[%s] must name a sibling Safetensors shard" % name
            )
        normalized[name] = owner
    if normalized != dict(owners):
        missing = sorted(set(owners) - set(normalized))
        extra = sorted(set(normalized) - set(owners))
        wrong = sorted(
            name
            for name in set(normalized) & set(owners)
            if normalized[name] != owners[name]
        )
        raise CheckpointError(
            "HF Safetensors index differs from shards: missing=%s extra=%s wrong_owner=%s"
            % (missing, extra, wrong)
        )


def read_hf_safetensors(
    paths: Mapping[str, Path],
    allowed_dtypes: Sequence[str] = DEFAULT_SOURCE_DTYPES,
) -> List[FileTensorSource]:
    """Read HF weight files (one model file or index+shards) and verify ownership.

    Callers keep policy for non-weight assets such as ``config.json``.  This
    makes the reader reusable across architecture-specific converters while
    keeping the standard HF weight container closed.
    """
    checked_dtypes = _checked_allowed_dtypes(allowed_dtypes)
    names = set(paths)
    index_name = "model.safetensors.index.json"
    shard_names = sorted(name for name in names if name.endswith(".safetensors"))
    if index_name not in names:
        expected = {"model.safetensors"}
        if names != expected:
            raise CheckpointError(
                "single-file HF Safetensors weight set differs: missing=%s extra=%s"
                % (sorted(expected - names), sorted(names - expected))
            )
    else:
        non_weights = names - set(shard_names) - {index_name}
        if (
            index_name not in names
            or non_weights
            or not shard_names
            or "model.safetensors" in shard_names
        ):
            raise CheckpointError(
                "sharded HF Safetensors weight set has missing/extra assets: %s"
                % sorted(non_weights)
            )

    tensors = []  # type: List[FileTensorSource]
    owners = {}  # type: Dict[str, str]
    for name in shard_names:
        for tensor in read_safetensors(paths[name], checked_dtypes):
            if tensor.name in owners:
                raise CheckpointError(
                    "source tensor %r occurs in multiple shards" % tensor.name
                )
            owners[tensor.name] = name
            tensors.append(tensor)
    if index_name in names:
        _validate_index(
            paths[index_name], owners, sum(tensor.nbytes for tensor in tensors)
        )
        indexed_shards = set(owners.values())
        if indexed_shards != set(shard_names):
            raise CheckpointError(
                "HF shard file set differs from weight_map owners: missing=%s extra=%s"
                % (
                    sorted(indexed_shards - set(shard_names)),
                    sorted(set(shard_names) - indexed_shards),
                )
            )
    return tensors


def select_hf_safetensors_paths(
    paths: Mapping[str, Path],
) -> Dict[str, Path]:
    """Select one canonical HF Safetensors checkpoint from a verified receipt.

    A source receipt may cover the complete immutable snapshot (tokenizer,
    README, license, and remote-code assets).  Only the weight index and the
    shards named by that index are passed to the strict tensor reader.
    """
    index_name = "model.safetensors.index.json"
    if index_name not in paths:
        if "model.safetensors" not in paths:
            raise CheckpointError("HF receipt is missing model.safetensors")
        return {"model.safetensors": paths["model.safetensors"]}
    index, _ = load_json(paths[index_name], "HF Safetensors shard index")
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise CheckpointError(
            "HF Safetensors index fields must be exactly metadata/weight_map"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("HF Safetensors index weight_map must be nonempty")
    owners = set()  # type: set
    for raw_name, raw_owner in weight_map.items():
        name = _validate_tensor_name(raw_name, "HF Safetensors index")
        owner = normalized_relative_path(raw_owner, "weight_map[%s]" % name)
        if PurePosixPath(owner).name != owner or not owner.endswith(".safetensors"):
            raise CheckpointError(
                "weight_map[%s] must name a sibling Safetensors shard" % name
            )
        owners.add(owner)
    required = {index_name} | owners
    missing = sorted(required - set(paths))
    if missing:
        raise CheckpointError("HF receipt is missing Safetensors assets: %s" % missing)
    return {name: paths[name] for name in sorted(required)}


def select_hf_torch_paths(paths: Mapping[str, Path]) -> Dict[str, Path]:
    """Select one canonical HF PyTorch checkpoint from a verified receipt."""
    index_name = "pytorch_model.bin.index.json"
    if index_name not in paths:
        if "pytorch_model.bin" not in paths:
            raise CheckpointError("HF receipt is missing pytorch_model.bin")
        return {"pytorch_model.bin": paths["pytorch_model.bin"]}
    index, _ = load_json(paths[index_name], "HF PyTorch shard index")
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise CheckpointError(
            "HF PyTorch index fields must be exactly metadata/weight_map"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("HF PyTorch index weight_map must be nonempty")
    owners = set()  # type: set
    for raw_name, raw_owner in weight_map.items():
        name = _validate_tensor_name(raw_name, "HF PyTorch index")
        owner = normalized_relative_path(raw_owner, "weight_map[%s]" % name)
        if (
            PurePosixPath(owner).name != owner
            or re.fullmatch(r"pytorch_model-[0-9]{5}-of-[0-9]{5}[.]bin", owner)
            is None
        ):
            raise CheckpointError(
                "weight_map[%s] must name a sibling PyTorch shard" % name
            )
        owners.add(owner)
    required = {index_name} | owners
    missing = sorted(required - set(paths))
    if missing:
        raise CheckpointError("HF receipt is missing PyTorch assets: %s" % missing)
    return {name: paths[name] for name in sorted(required)}


def _load_hf_torch_file(
    path: Path,
    allowed_dtypes: Sequence[str] = DEFAULT_SOURCE_DTYPES,
    allow_exact_column_major: bool = False,
) -> List[TorchTensorSource]:
    """Safely load one verified HF PyTorch state-dict file."""
    checked_dtypes = _checked_allowed_dtypes(allowed_dtypes)
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise CheckpointError(
            "HF .bin conversion requires offline PyTorch with "
            "torch.load(weights_only=True); install requirements-convert.txt"
        ) from error
    try:
        # PyTorch 2.1 requires a concrete string filename when mmap=True;
        # pathlib.Path is accepted by newer releases but must not make the
        # verified converter contract version-dependent.
        state = torch.load(
            str(path), map_location="cpu", mmap=True, weights_only=True
        )
    except TypeError as error:
        raise CheckpointError(
            "installed PyTorch lacks the required torch.load(weights_only=True, mmap=True) API"
        ) from error
    except Exception as error:
        raise CheckpointError(
            "failed to safely load verified HF checkpoint %s: %s"
            % (path.name, error)
        ) from error
    if not isinstance(state, dict):
        raise CheckpointError("HF PyTorch checkpoint root must be a state dictionary")
    if not state or len(state) > MAX_TENSOR_COUNT:
        raise CheckpointError("HF PyTorch checkpoint tensor count is invalid")

    sources = []  # type: List[TorchTensorSource]
    for raw_name, tensor in state.items():
        name = _validate_tensor_name(raw_name, "HF PyTorch checkpoint")
        if not isinstance(tensor, torch.Tensor):
            raise CheckpointError(
                "HF PyTorch checkpoint entry %r is not a tensor" % name
            )
        if tensor.dtype == torch.float32:
            dtype = "F32"
        elif tensor.dtype == torch.bfloat16:
            dtype = "BF16"
        elif tensor.dtype == torch.int64:
            dtype = "I64"
        else:
            raise CheckpointError(
                "tensor %r uses unsupported PyTorch dtype %s; only F32/BF16/I64 are accepted"
                % (name, tensor.dtype)
            )
        if dtype not in checked_dtypes:
            raise CheckpointError(
                "tensor %r uses source dtype %r outside allowed set %s"
                % (name, dtype, checked_dtypes)
            )
        shape = checked_shape(list(tensor.shape), "tensor %s.shape" % name)
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise CheckpointError(
                "tensor %r must be dense, contiguous, and CPU-resident" % name
            )
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes != tensor_nbytes(dtype, shape):
            raise CheckpointError("tensor %r byte extent is inconsistent" % name)
        if not tensor.is_contiguous():
            if not allow_exact_column_major:
                raise CheckpointError(
                    "tensor %r must be dense, contiguous, and CPU-resident" % name
                )
            expected_stride = (1, shape[0]) if len(shape) == 2 else None
            try:
                storage_nbytes = tensor.untyped_storage().nbytes()
            except (AttributeError, RuntimeError) as error:
                raise CheckpointError(
                    "tensor %r cannot expose audited column-major storage" % name
                ) from error
            if (
                len(shape) != 2
                or tensor.storage_offset() != 0
                or tuple(tensor.stride()) != expected_stride
                or storage_nbytes != nbytes
                or tensor.is_conj()
                or tensor.is_neg()
            ):
                raise CheckpointError(
                    "tensor %r is not an exact full-storage column-major matrix"
                    % name
                )
            tensor = tensor.detach().contiguous()
            if not tensor.is_contiguous() or tensor.storage_offset() != 0:
                raise CheckpointError(
                    "tensor %r could not be materialized as logical contiguous values"
                    % name
                )
        sources.append(TorchTensorSource(name, dtype, shape, nbytes, tensor))
    return sources


def load_hf_torch_checkpoint(
    path: Path,
    allowed_dtypes: Sequence[str] = DEFAULT_SOURCE_DTYPES,
    allow_exact_column_major: bool = False,
) -> List[TorchTensorSource]:
    """Safely load one verified ``pytorch_model.bin`` checkpoint."""
    if path.name != "pytorch_model.bin":
        raise CheckpointError(
            "PyTorch checkpoint path must be named pytorch_model.bin"
        )
    return _load_hf_torch_file(
        path, allowed_dtypes, allow_exact_column_major
    )


def load_torch_checkpoint(
    path: Path,
    allowed_dtypes: Sequence[str] = DEFAULT_SOURCE_DTYPES,
    allow_exact_column_major: bool = False,
) -> List[TorchTensorSource]:
    """Safely load a verified direct PyTorch state dict with any basename.

    The caller must verify the file receipt before invoking this function.
    HF converters should retain :func:`load_hf_torch_checkpoint`, which also
    enforces the standard ``pytorch_model.bin`` basename.
    """
    return _load_hf_torch_file(
        path, allowed_dtypes, allow_exact_column_major
    )


def load_hf_torch_checkpoints(
    paths: Mapping[str, Path],
    allowed_dtypes: Sequence[str] = DEFAULT_SOURCE_DTYPES,
    allow_exact_column_major: bool = False,
) -> List[TorchTensorSource]:
    """Load a strict single-file or indexed sharded HF PyTorch checkpoint."""

    names = set(paths)
    if names == {"pytorch_model.bin"}:
        # The verified receipt key carries the canonical HF filename.  Cache
        # implementations commonly resolve that key to a content-addressed
        # blob whose physical basename is its digest, so rechecking the
        # physical basename here would reject a valid immutable snapshot.
        return _load_hf_torch_file(
            paths["pytorch_model.bin"],
            allowed_dtypes,
            allow_exact_column_major,
        )

    checked_dtypes = _checked_allowed_dtypes(allowed_dtypes)

    index_name = "pytorch_model.bin.index.json"
    shard_pattern = re.compile(r"pytorch_model-([0-9]{5})-of-([0-9]{5})[.]bin")
    shard_names = sorted(name for name in names if shard_pattern.fullmatch(name))
    expected = {index_name} | set(shard_names)
    if names != expected or not shard_names:
        raise CheckpointError(
            "sharded HF PyTorch weight set differs: missing=%s extra=%s"
            % (sorted(expected - names), sorted(names - expected))
        )
    totals = set()  # type: set
    indices = set()  # type: set
    for name in shard_names:
        match = shard_pattern.fullmatch(name)
        if match is None:
            raise CheckpointError("internal PyTorch shard-name validation failed")
        index = int(match.group(1))
        total = int(match.group(2))
        if index <= 0 or total <= 0 or index > total:
            raise CheckpointError("HF PyTorch shard ordinal is invalid")
        indices.add(index)
        totals.add(total)
    if len(totals) != 1:
        raise CheckpointError("HF PyTorch shard totals disagree")
    total_shards = next(iter(totals))
    if total_shards != len(shard_names) or indices != set(range(1, total_shards + 1)):
        raise CheckpointError("HF PyTorch shard sequence is incomplete")

    index, _ = load_json(paths[index_name], "HF PyTorch shard index")
    if set(index) != {"metadata", "weight_map"}:
        raise CheckpointError(
            "HF PyTorch index fields must be exactly metadata/weight_map"
        )
    metadata = index["metadata"]
    weight_map = index["weight_map"]
    if not isinstance(metadata, dict) or set(metadata) != {"total_size"}:
        raise CheckpointError(
            "HF PyTorch index metadata must contain exactly total_size"
        )
    total_size = metadata["total_size"]
    if (
        isinstance(total_size, bool)
        or not isinstance(total_size, int)
        or total_size <= 0
        or total_size >= (1 << 64)
    ):
        raise CheckpointError("HF PyTorch index total_size is invalid")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("HF PyTorch index weight_map must be nonempty")
    declared = {}  # type: Dict[str, str]
    for raw_name, raw_owner in weight_map.items():
        name = _validate_tensor_name(raw_name, "HF PyTorch index")
        owner = normalized_relative_path(raw_owner, "weight_map[%s]" % name)
        if PurePosixPath(owner).name != owner or owner not in shard_names:
            raise CheckpointError(
                "weight_map[%s] names an unknown PyTorch shard" % name
            )
        declared[name] = owner

    tensors = []  # type: List[TorchTensorSource]
    actual_owners = {}  # type: Dict[str, str]
    for shard_name in shard_names:
        for tensor in _load_hf_torch_file(
            paths[shard_name], checked_dtypes, allow_exact_column_major
        ):
            if tensor.name in actual_owners:
                raise CheckpointError(
                    "source tensor %r occurs in multiple PyTorch shards"
                    % tensor.name
                )
            actual_owners[tensor.name] = shard_name
            tensors.append(tensor)
    missing = sorted(set(declared) - set(actual_owners))
    extra = sorted(set(actual_owners) - set(declared))
    wrong_owner = sorted(
        name
        for name in set(declared) & set(actual_owners)
        if declared[name] != actual_owners[name]
    )
    if missing or extra or wrong_owner:
        raise CheckpointError(
            "HF PyTorch index differs from shards: missing=%s extra=%s wrong_owner=%s"
            % (missing, extra, wrong_owner)
        )
    actual_size = sum(tensor.nbytes for tensor in tensors)
    if actual_size != total_size:
        raise CheckpointError(
            "HF PyTorch index total_size is %d; tensors contain %d"
            % (total_size, actual_size)
        )
    return tensors


def validate_tensor_manifest(
    sources: Sequence[TensorSource], expected: Sequence[TensorSpec]
) -> List[TensorSource]:
    """Require an exact tensor name/shape/dtype set and return manifest order."""
    if not expected:
        raise CheckpointError("expected tensor manifest must not be empty")
    expected_by_name = {}  # type: Dict[str, TensorSpec]
    for spec in expected:
        _validate_tensor_name(spec.name, "expected manifest")
        if spec.name in expected_by_name:
            raise CheckpointError("expected tensor manifest has duplicate names")
        if spec.nbytes <= 0:
            raise CheckpointError("expected tensor %r is empty" % spec.name)
        expected_by_name[spec.name] = spec

    actual_by_name = {}  # type: Dict[str, TensorSource]
    for source in sources:
        if source.name in actual_by_name:
            raise CheckpointError("source tensor %r is duplicated" % source.name)
        actual_by_name[source.name] = source
    missing = sorted(set(expected_by_name) - set(actual_by_name))
    extra = sorted(set(actual_by_name) - set(expected_by_name))
    wrong_shape = []  # type: List[str]
    wrong_dtype = []  # type: List[str]
    wrong_bytes = []  # type: List[str]
    for name in sorted(set(expected_by_name) & set(actual_by_name)):
        spec = expected_by_name[name]
        source = actual_by_name[name]
        if tuple(source.shape) != spec.shape:
            wrong_shape.append(name)
        if source.dtype != spec.dtype:
            wrong_dtype.append(name)
        if source.nbytes != spec.nbytes:
            wrong_bytes.append(name)
    if missing or extra or wrong_shape or wrong_dtype or wrong_bytes:
        raise CheckpointError(
            "tensor manifest mismatch: missing=%s extra=%s wrong_shape=%s "
            "wrong_dtype=%s wrong_bytes=%s"
            % (missing, extra, wrong_shape, wrong_dtype, wrong_bytes)
        )
    return [actual_by_name[spec.name] for spec in expected]


__all__ = [
    "CheckpointError",
    "FileTensorSource",
    "TensorSpec",
    "TorchTensorSource",
    "load_hf_torch_checkpoint",
    "load_hf_torch_checkpoints",
    "load_torch_checkpoint",
    "load_json",
    "load_json_bytes",
    "normalized_relative_path",
    "read_hf_safetensors",
    "read_safetensors",
    "tensor_nbytes",
    "validate_tensor_manifest",
]
