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


def _projection_fp8_sources(
    torch: Any,
    extra_states: dict[str, Any],
    layers: Sequence[int],
) -> list[TorchTensorSource]:
    sources: list[TorchTensorSource] = []
    for layer in layers:
        key = f"blocks.{layer}.projections._extra_state"
        stream = extra_states.get(key)
        if stream is None:
            raise CheckpointError(f"missing FP8 projection extra state: {key}")
        stream.seek(0)
        try:
            state = torch.load(stream, map_location="cpu", weights_only=True)
        except Exception as error:
            raise CheckpointError(f"failed to load {key}: {error}") from error
        if not isinstance(state, dict):
            raise CheckpointError(f"{key} payload must be a dictionary")

        scale = state.get("scale_fwd")
        scale_inv = state.get("scale_inv_fwd")
        history = state.get("amax_history_fwd")
        tensors = (
            ("scale_fwd", scale, (3,)),
            ("scale_inv_fwd", scale_inv, (3,)),
            ("amax_history_fwd", history, (16, 3)),
        )
        for name, tensor, shape in tensors:
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != shape
                or tensor.device.type != "cpu"
                or not tensor.is_contiguous()
            ):
                raise CheckpointError(
                    f"{key} {name} must be dense CPU F32{shape}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise CheckpointError(f"{key} {name} contains non-finite values")
        if not bool((scale > 0.0).all()) or not bool((scale_inv > 0.0).all()):
            raise CheckpointError(f"{key} forward scales must be positive")
        if not torch.equal(scale_inv, scale.reciprocal()):
            raise CheckpointError(
                f"{key} scale_inv_fwd is not the float32 reciprocal of scale_fwd"
            )
        if not bool((history >= 0.0).all()):
            raise CheckpointError(f"{key} amax_history_fwd must be nonnegative")
        history_max = history.max(dim=0).values
        expected_scale = torch.where(
            history_max > 0.0,
            torch.full_like(history_max, 448.0) / history_max,
            scale,
        )
        if not bool(
            torch.allclose(scale[:2], expected_scale[:2], rtol=2e-7, atol=0.0)
        ):
            raise CheckpointError(
                f"{key} input/weight scales do not match E4M3 delayed-scale history"
            )

        variables = state.get("extra_fp8_variables")
        required_variables = {
            "fp8_checkpoint": True,
            "num_gemms": 1,
            "fp8_max_fwd": 448.0,
            "fp8_max_bwd": 57344.0,
        }
        if not isinstance(variables, dict) or any(
            variables.get(name) != value
            for name, value in required_variables.items()
        ):
            raise CheckpointError(
                f"{key} does not describe one HYBRID E4M3/E5M2 GEMM"
            )

        prefix = f"blocks.{layer}.projections"
        exported = (
            (f"{prefix}.fp8_scale_fwd", scale[:2].clone()),
            (f"{prefix}.fp8_scale_inv_fwd", scale_inv[:2].clone()),
            (
                f"{prefix}.fp8_amax_history_fwd",
                history[:, :2].contiguous(),
            ),
        )
        for name, tensor in exported:
            sources.append(
                TorchTensorSource(
                    name=name,
                    dtype="F32",
                    shape=tuple(tensor.shape),
                    nbytes=tensor.numel() * tensor.element_size(),
                    tensor=tensor,
                )
            )
    return sources


def load_checkpoint(
    path: Path,
    manifest: Sequence[TensorSpec],
    *,
    expected_extra_states: int = EXPECTED_EXTRA_STATE_COUNT,
    fp8_projection_layers: Sequence[int] = (),
) -> tuple[list[TorchTensorSource], list[str], list[TorchTensorSource]]:
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
    fp8_sources = _projection_fp8_sources(
        torch, non_tensors, fp8_projection_layers
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
    return sources, sorted(non_tensors), fp8_sources
