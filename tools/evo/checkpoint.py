"""PyTorch checkpoint loading isolated from the native runtime and format writer."""

from __future__ import annotations

import dataclasses
import importlib
import io
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from evo.model_config import EXPECTED_EXTRA_STATE_COUNT, TensorSpec


class CheckpointError(ValueError):
    """Raised when a checkpoint does not match the strict manifest."""


@dataclasses.dataclass(frozen=True, slots=True)
class TorchTensorSource:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    tensor: Any

    def read_range(self, offset: int, length: int) -> memoryview:
        if offset < 0 or length < 0 or offset > self.nbytes - length:
            raise CheckpointError(f"tensor {self.name!r} read is out of range")
        torch = importlib.import_module("torch")
        raw = self.tensor.detach().view(torch.uint8).reshape(-1).numpy()
        view = memoryview(raw).cast("B")
        if len(view) != self.nbytes:
            raise CheckpointError(
                f"tensor {self.name!r} exposed {len(view)} bytes, expected {self.nbytes}"
            )
        return view[offset : offset + length]

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        for offset in range(0, self.nbytes, chunk_size):
            yield self.read_range(offset, min(chunk_size, self.nbytes - offset))


@dataclasses.dataclass(frozen=True, slots=True)
class TorchWidenF32Source:
    """Losslessly widen a small BF16 tensor while streaming the output."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    tensor: Any

    def read_range(self, offset: int, length: int) -> memoryview:
        if offset < 0 or length < 0 or offset > self.nbytes - length:
            raise CheckpointError(f"tensor {self.name!r} read is out of range")
        torch = importlib.import_module("torch")
        widened = self.tensor.detach().float().contiguous()
        raw = widened.view(torch.uint8).reshape(-1).numpy()
        view = memoryview(raw).cast("B")
        if len(view) != self.nbytes:
            raise CheckpointError(
                f"tensor {self.name!r} widened to {len(view)} bytes, expected {self.nbytes}"
            )
        return view[offset : offset + length]

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        for offset in range(0, self.nbytes, chunk_size):
            yield self.read_range(offset, min(chunk_size, self.nbytes - offset))


@dataclasses.dataclass(frozen=True, slots=True)
class TorchE4M3Source:
    """Stream final scaled E4M3FN codes from a BF16 or F32 weight tensor."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    tensor: Any
    scale: float

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        torch = importlib.import_module("torch")
        float8 = getattr(torch, "float8_e4m3fn", None)
        if float8 is None:
            raise CheckpointError(
                "this PyTorch build cannot encode float8_e4m3fn; "
                "upgrade the conversion environment"
            )
        flat = self.tensor.detach().reshape(-1)
        if flat.numel() != self.nbytes:
            raise CheckpointError(
                f"tensor {self.name!r} element count does not match E4M3 output size"
            )
        for offset in range(0, self.nbytes, chunk_size):
            count = min(chunk_size, self.nbytes - offset)
            scaled = flat[offset : offset + count].to(
                dtype=torch.float32, copy=True
            )
            scaled.mul_(self.scale).clamp_(min=-448.0, max=448.0)
            encoded = scaled.to(float8).view(torch.uint8).contiguous().numpy()
            yield memoryview(encoded).cast("B")


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
    ignored_manifest: Sequence[TensorSpec] = (),
) -> tuple[list[TorchTensorSource], list[str], list[TorchTensorSource]]:
    """mmap a checkpoint, validate all entries, and return zero-copy tensor sources."""
    if path.name.rpartition(".part")[2].isdigit():
        raise CheckpointError("input is a checkpoint part; merge all .partN files in order first")
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise CheckpointError(
            "PyTorch is required for offline .pt conversion; "
            "install requirements-convert.txt"
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

    all_manifest = tuple(manifest) + tuple(ignored_manifest)
    expected = {spec.name: spec for spec in all_manifest}
    if len(expected) != len(all_manifest):
        raise CheckpointError("checkpoint and ignored manifests contain duplicate names")
    missing = sorted(expected.keys() - tensors.keys())
    extra = sorted(tensors.keys() - expected.keys())
    if missing or extra:
        details = []
        if missing:
            details.append("missing tensors: " + ", ".join(missing[:10]))
        if extra:
            details.append("unknown tensors: " + ", ".join(extra[:10]))
        raise CheckpointError("; ".join(details))
    validated: dict[str, TorchTensorSource] = {}
    for spec in all_manifest:
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
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise CheckpointError(f"tensor {spec.name!r} must be dense contiguous")
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes != spec.nbytes:
            raise CheckpointError(
                f"tensor {spec.name!r} has {nbytes} bytes; manifest requires {spec.nbytes}"
            )
        validated[spec.name] = TorchTensorSource(
                name=spec.name,
                dtype=spec.dtype,
                shape=spec.shape,
                nbytes=nbytes,
                tensor=tensor,
            )
    sources = [validated[spec.name] for spec in manifest]
    return sources, sorted(non_tensors), fp8_sources


def prepare_runtime_sources(
    sources: Sequence[TorchTensorSource],
    runtime_manifest: Sequence[TensorSpec],
) -> list[TorchTensorSource | TorchWidenF32Source]:
    """Match the runtime manifest, allowing only exact BF16-to-F32 widening."""
    source_by_name = {source.name: source for source in sources}
    if len(source_by_name) != len(sources):
        raise CheckpointError("source list contains duplicate tensor names")
    expected_names = {spec.name for spec in runtime_manifest}
    if set(source_by_name) != expected_names:
        raise CheckpointError("source and runtime manifests contain different tensor names")
    converted: list[TorchTensorSource | TorchWidenF32Source] = []
    for spec in runtime_manifest:
        source = source_by_name[spec.name]
        if source.shape != spec.shape:
            raise CheckpointError(f"runtime shape mismatch for {spec.name}")
        if source.dtype == spec.dtype and source.nbytes == spec.nbytes:
            converted.append(source)
        elif source.dtype == "BF16" and spec.dtype == "F32":
            converted.append(
                TorchWidenF32Source(
                    name=source.name,
                    dtype="F32",
                    shape=source.shape,
                    nbytes=spec.nbytes,
                    tensor=source.tensor,
                )
            )
        else:
            raise CheckpointError(
                f"unsafe runtime conversion for {spec.name}: {source.dtype} to {spec.dtype}"
            )
    return converted


def prepare_runtime_image_sources(
    sources: Sequence[TorchTensorSource | TorchWidenF32Source],
    fp8_sources: Sequence[TorchTensorSource],
    fp8_projection_layers: Sequence[int],
) -> list[TorchTensorSource | TorchWidenF32Source | TorchE4M3Source]:
    """Produce final Safetensors payloads in global-then-layer load order."""
    torch = importlib.import_module("torch")
    by_name = {source.name: source for source in sources}
    if len(by_name) != len(sources):
        raise CheckpointError("runtime source list contains duplicate tensor names")
    fp8_by_name = {source.name: source for source in fp8_sources}
    if len(fp8_by_name) != len(fp8_sources):
        raise CheckpointError("FP8 source list contains duplicate tensor names")

    projection_layers = set(fp8_projection_layers)
    expected_fp8_names: set[str] = set()
    converted: list[
        TorchTensorSource | TorchWidenF32Source | TorchE4M3Source
    ] = []
    for source in sources:
        layer = next(
            (
                index
                for index in projection_layers
                if source.name == f"blocks.{index}.projections.weight"
            ),
            None,
        )
        if layer is None:
            converted.append(source)
            continue

        prefix = f"blocks.{layer}.projections"
        scale_name = f"{prefix}.fp8_scale_fwd"
        inverse_name = f"{prefix}.fp8_scale_inv_fwd"
        history_name = f"{prefix}.fp8_amax_history_fwd"
        expected_fp8_names.update({scale_name, inverse_name, history_name})
        scale_source = fp8_by_name.get(scale_name)
        inverse_source = fp8_by_name.get(inverse_name)
        history_source = fp8_by_name.get(history_name)
        if scale_source is None or inverse_source is None or history_source is None:
            raise CheckpointError(f"incomplete FP8 state for {prefix}")

        scale = scale_source.tensor
        inverse = inverse_source.tensor
        output_scale = inverse[0] * inverse[1]
        runtime_scales = torch.stack((scale[0], output_scale)).contiguous()
        converted.append(
            TorchTensorSource(
                name=f"{prefix}.fp8_runtime_scales",
                dtype="F32",
                shape=(2,),
                nbytes=2 * runtime_scales.element_size(),
                tensor=runtime_scales,
            )
        )
        converted.append(
            TorchE4M3Source(
                name=source.name,
                dtype="E4M3_SW",
                shape=source.shape,
                nbytes=source.nbytes // (4 if source.dtype == "F32" else 2),
                tensor=source.tensor,
                scale=float(scale[1].item()),
            )
        )

    if set(fp8_by_name) != expected_fp8_names:
        unknown = sorted(set(fp8_by_name) - expected_fp8_names)
        missing = sorted(expected_fp8_names - set(fp8_by_name))
        details: list[str] = []
        if unknown:
            details.append("unexpected FP8 state: " + ", ".join(unknown[:10]))
        if missing:
            details.append("missing FP8 state: " + ", ".join(missing[:10]))
        raise CheckpointError("; ".join(details))

    globals_by_name = {source.name: source for source in converted}
    ordered = [
        globals_by_name[name]
        for name in ("embedding_layer.weight", "unembed.weight", "norm.scale")
        if name in globals_by_name
    ]
    global_names = {source.name for source in ordered}
    ordered.extend(source for source in converted if source.name not in global_names)
    return ordered
