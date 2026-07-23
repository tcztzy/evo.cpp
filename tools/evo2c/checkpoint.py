"""PyTorch checkpoint loading isolated from the native runtime and format writer."""

from __future__ import annotations

import dataclasses
import importlib
import io
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from evo2c.model_config import EXPECTED_EXTRA_STATE_COUNT, TensorSpec


class CheckpointError(ValueError):
    """Raised when a checkpoint does not match the strict manifest."""


@dataclasses.dataclass(frozen=True, slots=True)
class TorchTensorSource:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    tensor: Any

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        torch = importlib.import_module("torch")
        raw = self.tensor.detach().view(torch.uint8).reshape(-1).numpy()
        view = memoryview(raw).cast("B")
        if len(view) != self.nbytes:
            raise CheckpointError(
                f"tensor {self.name!r} exposed {len(view)} bytes, expected {self.nbytes}"
            )
        for offset in range(0, len(view), chunk_size):
            yield view[offset : offset + chunk_size]


def _torch_dtype_name(torch: Any, tensor: Any) -> str | None:
    if tensor.dtype == torch.bfloat16:
        return "BF16"
    if tensor.dtype == torch.float32:
        return "F32"
    return None


def load_checkpoint(
    path: Path,
    manifest: Sequence[TensorSpec],
    *,
    expected_extra_states: int = EXPECTED_EXTRA_STATE_COUNT,
) -> tuple[list[TorchTensorSource], list[str]]:
    """mmap a checkpoint, validate all entries, and return zero-copy tensor sources."""
    if path.name.rpartition(".part")[2].isdigit():
        raise CheckpointError("input is a checkpoint part; merge all .partN files in order first")
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise CheckpointError(
            "PyTorch is required only for conversion; install requirements-convert.txt"
        ) from error
    torch.serialization.add_safe_globals([io.BytesIO])
    try:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as error:
        raise CheckpointError(f"failed to load checkpoint {path}: {error}") from error
    if not isinstance(state, dict):
        raise CheckpointError(f"checkpoint root must be dict, got {type(state).__name__}")

    tensors = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
    non_tensors = {key: value for key, value in state.items() if not isinstance(value, torch.Tensor)}
    invalid_non_tensors = [
        key
        for key, value in non_tensors.items()
        if not key.endswith("._extra_state") or not isinstance(value, io.BytesIO)
    ]
    if invalid_non_tensors:
        raise CheckpointError(
            "unknown non-tensor checkpoint entries: " + ", ".join(sorted(invalid_non_tensors)[:10])
        )
    if len(non_tensors) != expected_extra_states:
        raise CheckpointError(
            f"checkpoint has {len(non_tensors)} TE extra-state entries; expected {expected_extra_states}"
        )

    expected = {spec.name: spec for spec in manifest}
    missing = sorted(expected.keys() - tensors.keys())
    extra = sorted(tensors.keys() - expected.keys())
    if missing or extra:
        details = []
        if missing:
            details.append("missing tensors: " + ", ".join(missing[:10]))
        if extra:
            details.append("unknown tensors: " + ", ".join(extra[:10]))
        raise CheckpointError("; ".join(details))

    sources: list[TorchTensorSource] = []
    for spec in manifest:
        tensor = tensors[spec.name]
        actual_dtype = _torch_dtype_name(torch, tensor)
        actual_shape = tuple(tensor.shape)
        if actual_dtype != spec.dtype:
            raise CheckpointError(
                f"tensor {spec.name!r} dtype={tensor.dtype}; expected {spec.dtype}"
            )
        if actual_shape != spec.shape:
            raise CheckpointError(
                f"tensor {spec.name!r} shape={actual_shape}; expected {spec.shape}"
            )
        if tensor.device.type != "cpu":
            raise CheckpointError(f"tensor {spec.name!r} was not mapped to CPU")
        if tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.storage_offset() != 0:
            raise CheckpointError(f"tensor {spec.name!r} must be dense contiguous with storage_offset=0")
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes != spec.nbytes:
            raise CheckpointError(
                f"tensor {spec.name!r} has {nbytes} bytes; manifest requires {spec.nbytes}"
            )
        sources.append(
            TorchTensorSource(
                name=spec.name,
                dtype=spec.dtype,
                shape=spec.shape,
                nbytes=nbytes,
                tensor=tensor,
            )
        )
    return sources, sorted(non_tensors)
