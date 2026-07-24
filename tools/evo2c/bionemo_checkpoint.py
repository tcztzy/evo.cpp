"""Streaming BioNeMo NeMo2/MBridge DCP to EVO2C tensor mapping."""

from __future__ import annotations

import dataclasses
import importlib
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from evo2c.checkpoint import CheckpointError
from evo2c.model_config import (
    ModelConfig,
    TensorSpec,
    bionemo_checkpoint_manifest,
)


@dataclasses.dataclass(frozen=True, slots=True)
class MappingGroup:
    outputs: tuple[TensorSpec, ...]
    source_options: tuple[tuple[str, ...], ...]
    transform: str


@dataclasses.dataclass(frozen=True, slots=True)
class BoundMappingGroup:
    mapping: MappingGroup
    sources: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class DcpTensorMetadata:
    physical_name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.elements * (2 if self.dtype == "BF16" else 4)


def _fixed_sources(*names: str) -> tuple[tuple[str, ...], ...]:
    return (tuple(names),)


def build_mapping_groups(config: ModelConfig) -> list[MappingGroup]:
    """Describe NVIDIA's MBridge-to-Vortex mapping in output manifest order."""
    manifest = bionemo_checkpoint_manifest(config)
    by_name = {spec.name: spec for spec in manifest}
    groups: list[MappingGroup] = []

    def add(
        output_names: Sequence[str],
        source_options: tuple[tuple[str, ...], ...],
        transform: str,
    ) -> None:
        groups.append(
            MappingGroup(
                outputs=tuple(by_name[name] for name in output_names),
                source_options=source_options,
                transform=transform,
            )
        )

    add(
        ("embedding_layer.weight", "unembed.weight"),
        _fixed_sources("embedding.word_embeddings.weight"),
        "tied",
    )

    for layer in range(config.num_layers):
        source_prefix = f"decoder.layers.{layer}"
        output_prefix = f"blocks.{layer}"
        if layer in config.attn_layer_idxs:
            add(
                (f"{output_prefix}.pre_norm.scale",),
                (
                    (f"{source_prefix}.self_attention.linear_qkv.layer_norm_weight",),
                    (f"{source_prefix}.input_layernorm.weight",),
                ),
                "cast_f32",
            )
            add(
                (f"{output_prefix}.inner_mha_cls.out_proj.bias",),
                _fixed_sources(f"{source_prefix}.self_attention.linear_proj.bias"),
                "identity",
            )
            add(
                (f"{output_prefix}.inner_mha_cls.out_proj.weight",),
                _fixed_sources(f"{source_prefix}.self_attention.linear_proj.weight"),
                "identity",
            )
            add(
                (f"{output_prefix}.inner_mha_cls.Wqkv.weight",),
                _fixed_sources(f"{source_prefix}.self_attention.linear_qkv.weight"),
                "identity",
            )
            add(
                (f"{output_prefix}.inner_mha_cls.rotary_emb.inv_freq",),
                _fixed_sources(),
                "inv_freq",
            )
        else:
            add(
                (f"{output_prefix}.pre_norm.scale",),
                (
                    (f"{source_prefix}.mixer.dense_projection.layer_norm_weight",),
                    (f"{source_prefix}.norm.weight",),
                ),
                "cast_f32",
            )
            add(
                (f"{output_prefix}.out_filter_dense.bias",),
                _fixed_sources(f"{source_prefix}.mixer.dense.bias"),
                "identity",
            )
            add(
                (f"{output_prefix}.out_filter_dense.weight",),
                _fixed_sources(f"{source_prefix}.mixer.dense.weight"),
                "identity",
            )
            add(
                (f"{output_prefix}.projections.weight",),
                _fixed_sources(f"{source_prefix}.mixer.dense_projection.weight"),
                "identity",
            )
            add(
                (f"{output_prefix}.filter.short_filter_weight",),
                _fixed_sources(
                    f"{source_prefix}.mixer.hyena_proj_conv.short_conv_weight"
                ),
                "unsqueeze_middle",
            )
            if layer in config.hcs_layer_idxs:
                add(
                    (f"{output_prefix}.filter.h",),
                    _fixed_sources(
                        f"{source_prefix}.mixer.mixer.short_conv.short_conv_weight"
                    ),
                    "unsqueeze_middle",
                )
            elif layer in config.hcm_layer_idxs:
                add(
                    (f"{output_prefix}.filter.D",),
                    _fixed_sources(f"{source_prefix}.mixer.mixer.conv_bias"),
                    "identity",
                )
                add(
                    (f"{output_prefix}.filter.h",),
                    _fixed_sources(
                        f"{source_prefix}.mixer.mixer.filter.h",
                        f"{source_prefix}.mixer.mixer.filter.decay",
                    ),
                    "hcm_filter",
                )
            elif layer in config.hcl_layer_idxs:
                add(
                    (f"{output_prefix}.filter.D",),
                    _fixed_sources(f"{source_prefix}.mixer.mixer.conv_bias"),
                    "identity",
                )
                add(
                    (
                        f"{output_prefix}.filter.log_poles",
                        f"{output_prefix}.filter.residues",
                    ),
                    _fixed_sources(
                        f"{source_prefix}.mixer.mixer.filter.p",
                        f"{source_prefix}.mixer.mixer.filter.gamma",
                        f"{source_prefix}.mixer.mixer.filter.R",
                    ),
                    "hcl_filter",
                )
            else:
                raise AssertionError(f"unclassified Hyena layer {layer}")

        add(
            (
                f"{output_prefix}.mlp.l1.weight",
                f"{output_prefix}.mlp.l2.weight",
            ),
            _fixed_sources(f"{source_prefix}.mlp.linear_fc1.weight"),
            "split_fc1",
        )
        add(
            (f"{output_prefix}.mlp.l3.weight",),
            _fixed_sources(f"{source_prefix}.mlp.linear_fc2.weight"),
            "identity",
        )
        add(
            (f"{output_prefix}.post_norm.scale",),
            (
                (f"{source_prefix}.mlp.linear_fc1.layer_norm_weight",),
                (f"{source_prefix}.pre_mlp_layernorm.weight",),
            ),
            "cast_f32",
        )

    add(
        ("norm.scale",),
        _fixed_sources("decoder.final_norm.weight"),
        "cast_f32",
    )
    output_names = [spec.name for group in groups for spec in group.outputs]
    manifest_names = [spec.name for spec in manifest]
    if output_names != manifest_names:
        raise AssertionError("BioNeMo mapping order does not match EVO2C manifest")
    return groups


def resolve_mapping(
    groups: Sequence[MappingGroup],
    metadata: Mapping[str, DcpTensorMetadata],
) -> list[BoundMappingGroup]:
    """Bind TE/non-TE alternatives and reject missing, duplicate, or extra data."""
    available = set(metadata)
    used: set[str] = set()
    bound: list[BoundMappingGroup] = []
    missing: list[str] = []
    for group in groups:
        matches = [
            option
            for option in group.source_options
            if all(name in available for name in option)
        ]
        if not matches:
            alternatives = ["+".join(option) for option in group.source_options]
            missing.append("|".join(alternatives))
            continue
        if len(matches) != 1:
            outputs = ",".join(spec.name for spec in group.outputs)
            raise CheckpointError(
                f"ambiguous BioNeMo source mapping for {outputs}: "
                + " or ".join("+".join(option) for option in matches)
            )
        sources = matches[0]
        duplicates = sorted(set(sources) & used)
        if duplicates:
            raise CheckpointError(
                "BioNeMo tensors mapped more than once: " + ", ".join(duplicates)
            )
        used.update(sources)
        _validate_source_metadata(group, sources, metadata)
        bound.append(BoundMappingGroup(mapping=group, sources=sources))
    if missing:
        raise CheckpointError(
            "missing BioNeMo tensors: " + ", ".join(missing[:10])
        )
    extra = sorted(available - used)
    if extra:
        raise CheckpointError(
            "unknown BioNeMo data tensors: " + ", ".join(extra[:10])
        )
    return bound


def _validate_source_metadata(
    group: MappingGroup,
    sources: Sequence[str],
    metadata: Mapping[str, DcpTensorMetadata],
) -> None:
    if not sources:
        return
    inputs = [metadata[name] for name in sources]
    output_shapes = [spec.shape for spec in group.outputs]
    transform = group.transform

    if transform in {"identity", "cast_f32"}:
        if len(inputs) != 1 or inputs[0].shape != output_shapes[0]:
            raise CheckpointError(
                f"{sources[0]} shape={inputs[0].shape}; expected {output_shapes[0]}"
            )
    elif transform == "tied":
        if len(inputs) != 1 or any(inputs[0].shape != shape for shape in output_shapes):
            raise CheckpointError(
                f"{sources[0]} shape={inputs[0].shape}; expected tied {output_shapes[0]}"
            )
    elif transform == "unsqueeze_middle":
        expected = output_shapes[0]
        accepted = {expected, (expected[0], expected[2])}
        if len(inputs) != 1 or inputs[0].shape not in accepted:
            raise CheckpointError(
                f"{sources[0]} shape={inputs[0].shape}; expected one of {sorted(accepted)}"
            )
    elif transform == "split_fc1":
        first, second = output_shapes
        expected = (first[0] + second[0], first[1])
        if first != second or len(inputs) != 1 or inputs[0].shape != expected:
            raise CheckpointError(
                f"{sources[0]} shape={inputs[0].shape}; expected merged FC1 {expected}"
            )
    elif transform == "hcm_filter":
        expected = output_shapes[0]
        if (
            len(inputs) != 2
            or len(inputs[0].shape) != 2
            or inputs[0].shape != inputs[1].shape
            or inputs[0].shape[0] != expected[0]
            or inputs[0].shape[1] < expected[2]
        ):
            shapes = [item.shape for item in inputs]
            raise CheckpointError(
                f"medium-Hyena h/decay shapes={shapes}; expected ({expected[0]},L>={expected[2]})"
            )
    elif transform == "hcl_filter":
        groups_count, state_size = output_shapes[1]
        expected_elements = groups_count * state_size
        if (
            len(inputs) != 3
            or any(item.elements != expected_elements for item in inputs)
        ):
            shapes = [item.shape for item in inputs]
            raise CheckpointError(
                f"long-Hyena p/gamma/R shapes={shapes}; "
                f"each requires {expected_elements} elements"
            )
    elif transform != "inv_freq":
        raise AssertionError(f"unknown BioNeMo transform {transform}")

    if transform in {"identity", "tied", "unsqueeze_middle", "split_fc1"}:
        if any(item.dtype != "BF16" for item in inputs):
            raise CheckpointError(
                f"{'/'.join(sources)} must be BF16 for bit-exact export"
            )
    elif transform == "cast_f32":
        if any(item.dtype not in {"BF16", "F32"} for item in inputs):
            raise CheckpointError(f"{'/'.join(sources)} must be BF16 or F32")
    elif transform in {"hcm_filter", "hcl_filter"}:
        if any(item.dtype != "F32" for item in inputs):
            raise CheckpointError(f"{'/'.join(sources)} must be F32")


def resolve_dcp_directory(path: Path) -> Path:
    """Resolve a NeMo2/MBridge checkpoint root to one DCP directory."""
    path = path.resolve()
    if path.is_file() and path.name == ".metadata":
        return path.parent
    if not path.exists():
        raise CheckpointError(f"BioNeMo checkpoint not found: {path}")
    if not path.is_dir():
        raise CheckpointError(f"BioNeMo checkpoint must be a directory: {path}")
    if (path / ".metadata").is_file():
        return path
    latest = path / "latest_checkpointed_iteration.txt"
    if latest.is_file():
        try:
            iteration = int(latest.read_text(encoding="utf-8").strip())
        except ValueError as error:
            raise CheckpointError(f"invalid {latest}") from error
        candidate = path / f"iter_{iteration:07d}"
        if (candidate / ".metadata").is_file():
            return candidate
    candidates = sorted(metadata.parent for metadata in path.rglob(".metadata"))
    if not candidates:
        raise CheckpointError(f"no DCP .metadata found under {path}")
    if len(candidates) != 1:
        rendered = ", ".join(str(candidate) for candidate in candidates[:10])
        raise CheckpointError(
            f"multiple DCP checkpoints found under {path}; pass one directly: {rendered}"
        )
    return candidates[0]


def _torch_dtype_name(torch: Any, dtype: Any) -> str | None:
    if dtype == torch.bfloat16:
        return "BF16"
    if dtype == torch.float32:
        return "F32"
    return None


class DcpReader:
    """Read selected DCP tensors one at a time on CPU."""

    def __init__(self, directory: Path):
        try:
            torch = importlib.import_module("torch")
            dcp = importlib.import_module("torch.distributed.checkpoint")
            metadata_module = importlib.import_module(
                "torch.distributed.checkpoint.metadata"
            )
        except ModuleNotFoundError as error:
            raise CheckpointError(
                "PyTorch >=2.6 is required only for BioNeMo conversion; "
                "install requirements-convert.txt"
            ) from error
        self._torch = torch
        self._dcp = dcp
        self.directory = resolve_dcp_directory(directory)
        self._reader = dcp.FileSystemReader(str(self.directory))
        try:
            checkpoint_metadata = self._reader.read_metadata()
        except Exception as error:
            raise CheckpointError(
                f"failed to read DCP metadata from {self.directory}: {error}"
            ) from error

        bytes_metadata = metadata_module.BytesStorageMetadata
        tensors: dict[str, DcpTensorMetadata] = {}
        bytes_entries: list[str] = []
        for physical_name, item in checkpoint_metadata.state_dict_metadata.items():
            if isinstance(item, bytes_metadata):
                bytes_entries.append(physical_name)
                continue
            logical_name = (
                physical_name[len("module.") :]
                if physical_name.startswith("module.")
                else physical_name
            )
            if logical_name in tensors:
                raise CheckpointError(
                    f"duplicate DCP tensor after module-prefix normalization: {logical_name}"
                )
            dtype = _torch_dtype_name(torch, item.properties.dtype)
            if dtype is None:
                raise CheckpointError(
                    f"unsupported DCP dtype for {physical_name}: {item.properties.dtype}"
                )
            tensors[logical_name] = DcpTensorMetadata(
                physical_name=physical_name,
                dtype=dtype,
                shape=tuple(int(dimension) for dimension in item.size),
            )
        self.tensor_metadata = tensors
        self.bytes_entries = tuple(sorted(bytes_entries))

    def load(self, logical_name: str) -> Any:
        item = self.tensor_metadata[logical_name]
        dtype = (
            self._torch.bfloat16 if item.dtype == "BF16" else self._torch.float32
        )
        tensor = self._torch.empty(item.shape, dtype=dtype, device="cpu")
        try:
            self._dcp.load(
                state_dict={item.physical_name: tensor},
                storage_reader=self._reader,
                no_dist=True,
            )
        except Exception as error:
            raise CheckpointError(
                f"failed to load DCP tensor {item.physical_name}: {error}"
            ) from error
        if (
            tensor.device.type != "cpu"
            or tensor.layout != self._torch.strided
            or not tensor.is_contiguous()
        ):
            raise CheckpointError(
                f"DCP tensor {item.physical_name} is not dense contiguous CPU storage"
            )
        return tensor


def _transform_tensors(
    torch: Any,
    group: MappingGroup,
    inputs: Sequence[Any],
    config: ModelConfig,
) -> dict[str, Any]:
    names = [spec.name for spec in group.outputs]
    transform = group.transform
    if transform == "identity":
        outputs = {names[0]: inputs[0]}
    elif transform == "tied":
        outputs = {name: inputs[0] for name in names}
    elif transform == "cast_f32":
        outputs = {names[0]: inputs[0].to(torch.float32)}
    elif transform == "unsqueeze_middle":
        tensor = inputs[0]
        outputs = {names[0]: tensor[:, None, :] if tensor.dim() == 2 else tensor}
    elif transform == "split_fc1":
        half = inputs[0].shape[0] // 2
        outputs = {names[0]: inputs[0][:half], names[1]: inputs[0][half:]}
    elif transform == "hcm_filter":
        length = group.outputs[0].shape[2]
        folded = (
            inputs[0].to(torch.float32)[:, :length]
            * inputs[1].to(torch.float32)[:, :length]
        )
        outputs = {names[0]: folded[:, None, :]}
    elif transform == "hcl_filter":
        groups_count, state_size = group.outputs[1].shape
        poles = inputs[0].to(torch.float32).reshape(groups_count, state_size)
        gamma = inputs[1].to(torch.float32).reshape(groups_count, state_size)
        residues = inputs[2].to(torch.float32).reshape(groups_count, state_size)
        outputs = {
            names[0]: (-torch.exp(poles) * torch.exp(gamma))[:, :, None],
            names[1]: residues,
        }
    elif transform == "inv_freq":
        rotary_dim = config.hidden_size // config.num_attention_heads
        positions = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
        outputs = {
            names[0]: 1.0
            / (
                float(config.rotary_emb_base)
                ** (positions / float(rotary_dim))
            )
        }
    else:
        raise AssertionError(f"unknown BioNeMo transform {transform}")

    if set(outputs) != set(names):
        raise AssertionError(f"transform {transform} produced wrong outputs")
    expected_by_name = {spec.name: spec for spec in group.outputs}
    for name, tensor in outputs.items():
        spec = expected_by_name[name]
        actual_dtype = _torch_dtype_name(torch, tensor.dtype)
        if actual_dtype != spec.dtype:
            raise CheckpointError(
                f"converted tensor {name} dtype={tensor.dtype}; expected {spec.dtype}"
            )
        if tuple(tensor.shape) != spec.shape:
            raise CheckpointError(
                f"converted tensor {name} shape={tuple(tensor.shape)}; expected {spec.shape}"
            )
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise CheckpointError(f"converted tensor {name} must be dense CPU storage")
    return outputs


class _LazyTransform:
    def __init__(
        self,
        reader: DcpReader,
        bound: BoundMappingGroup,
        config: ModelConfig,
    ):
        self.reader = reader
        self.bound = bound
        self.config = config
        self._cache: dict[str, Any] | None = None

    def take(self, output_name: str) -> Any:
        if self._cache is None:
            inputs = [self.reader.load(name) for name in self.bound.sources]
            self._cache = _transform_tensors(
                self.reader._torch, self.bound.mapping, inputs, self.config
            )
        try:
            tensor = self._cache.pop(output_name)
        except KeyError as error:
            raise CheckpointError(
                f"BioNeMo transform output requested twice: {output_name}"
            ) from error
        if not self._cache:
            self._cache = None
        return tensor


@dataclasses.dataclass(frozen=True, slots=True)
class DcpTensorSource:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    transform: _LazyTransform

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        tensor = self.transform.take(self.name)
        raw = tensor.detach().view(self.transform.reader._torch.uint8).reshape(-1).numpy()
        view = memoryview(raw).cast("B")
        if len(view) != self.nbytes:
            raise CheckpointError(
                f"converted tensor {self.name} exposed {len(view)} bytes; expected {self.nbytes}"
            )
        for offset in range(0, len(view), chunk_size):
            yield view[offset : offset + chunk_size]


def load_bionemo_checkpoint(
    path: Path,
    config: ModelConfig,
) -> tuple[list[DcpTensorSource], DcpReader]:
    """Validate DCP metadata and return lazy, streaming EVO2C tensor sources."""
    reader = DcpReader(path)
    groups = build_mapping_groups(config)
    bound_groups = resolve_mapping(groups, reader.tensor_metadata)
    sources: list[DcpTensorSource] = []
    for bound in bound_groups:
        transform = _LazyTransform(reader, bound, config)
        for spec in bound.mapping.outputs:
            sources.append(
                DcpTensorSource(
                    name=spec.name,
                    dtype=spec.dtype,
                    shape=spec.shape,
                    nbytes=spec.nbytes,
                    transform=transform,
                )
            )
    return sources, reader


def dcp_payload_size(metadata: Mapping[str, DcpTensorMetadata]) -> int:
    return sum(item.nbytes for item in metadata.values())


def dcp_storage_size(directory: Path) -> int:
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def normalized_dcp_kind(metadata: Mapping[str, DcpTensorMetadata]) -> str:
    physical_names = [item.physical_name for item in metadata.values()]
    if physical_names and all(name.startswith("module.") for name in physical_names):
        return "NeMo2"
    if any(name.startswith("module.") for name in physical_names):
        raise CheckpointError("DCP mixes module-prefixed and MBridge tensor names")
    return "MBridge"


def validate_source_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise CheckpointError(
            "--source-sha256 must contain exactly 64 hexadecimal characters"
        )
    return value.lower()
