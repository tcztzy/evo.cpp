"""Strict official Evo 2 model registry and checkpoint tensor manifests."""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any


# Backward-compatible names for existing 40B converter clients.
HCS_LAYERS = (0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46)
HCM_LAYERS = (1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47)
HCL_LAYERS = (2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48)
ATTN_LAYERS = (3, 10, 17, 24, 31, 35, 42, 49)
EXPECTED_TENSOR_COUNT = 537
EXPECTED_BF16_TENSOR_COUNT = 400
EXPECTED_F32_TENSOR_COUNT = 137
EXPECTED_TENSOR_BYTES = 82_252_533_760
EXPECTED_EXTRA_STATE_COUNT = 258
EXPECTED_BIONEMO_BF16_TENSOR_COUNT = 386
EXPECTED_BIONEMO_F32_TENSOR_COUNT = 151
EXPECTED_BIONEMO_TENSOR_BYTES = 82_254_368_768

_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "model-registry.json"


class ConfigError(ValueError):
    """Raised when a config does not exactly match an official supported model."""


@dataclasses.dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        if self.dtype not in {"BF16", "F32"}:
            raise AssertionError(f"unsupported manifest dtype {self.dtype}")
        return self.elements * (2 if self.dtype == "BF16" else 4)


def load_model_registry(path: Path = _REGISTRY_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read model registry {path}: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or not isinstance(value.get("profiles"), dict)
        or not isinstance(value.get("models"), dict)
    ):
        raise ConfigError(f"{path}: unsupported or incomplete registry schema")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    model_name: str
    vocab_size: int
    hidden_size: int
    num_filters: int
    num_layers: int
    num_attention_heads: int
    inner_mlp_size: int
    state_size: int
    short_filter_length: int
    hcs_filter_length: int
    hcm_filter_length: int
    hcs_filter_groups: int
    hcm_filter_groups: int
    hcl_filter_groups: int
    hcs_layer_idxs: tuple[int, ...]
    hcm_layer_idxs: tuple[int, ...]
    hcl_layer_idxs: tuple[int, ...]
    attn_layer_idxs: tuple[int, ...]
    eps: float
    rotary_emb_base: float
    rotary_emb_scaling_factor: float
    max_seqlen: int
    max_batch_size: int
    inner_size_multiple_of: int
    proj_groups: int
    tie_embeddings: bool
    short_filter_bias: bool
    use_fp8_input_projections: bool
    use_interpolated_rotary_pos_emb: bool
    evo2_style_activations: bool
    interleave: bool
    column_split: bool
    column_split_hyena: bool
    hyena_flip_x1x2: bool
    final_norm: bool
    qkv_proj_bias: bool
    mha_out_proj_bias: bool
    hyena_out_proj_bias: bool
    tokenizer_type: str
    prefill_style: str
    mlp_activation: str

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = {field.name for field in dataclasses.fields(cls)} | {
            "model_parallel_size",
            "pipe_parallel_size",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ConfigError("unknown config keys: " + ", ".join(unknown))

        def integer(key: str) -> int:
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{key} must be an integer")
            return value

        def number(key: str) -> float:
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{key} must be numeric")
            return float(value)

        def boolean(key: str) -> bool:
            value = values.get(key)
            if not isinstance(value, bool):
                raise ConfigError(f"{key} must be boolean")
            return value

        def string(key: str) -> str:
            value = values.get(key)
            if not isinstance(value, str) or not value:
                raise ConfigError(f"{key} must be a nonempty string")
            return value

        def indices(key: str) -> tuple[int, ...]:
            value = values.get(key)
            if not isinstance(value, list) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in value
            ):
                raise ConfigError(f"{key} must be an integer list")
            return tuple(value)

        for parallel_key in ("model_parallel_size", "pipe_parallel_size"):
            if integer(parallel_key) != 1:
                raise ConfigError(f"{parallel_key} must be 1 for native conversion")

        config = cls(
            model_id=string("model_id"),
            model_name=string("model_name"),
            vocab_size=integer("vocab_size"),
            hidden_size=integer("hidden_size"),
            num_filters=integer("num_filters"),
            num_layers=integer("num_layers"),
            num_attention_heads=integer("num_attention_heads"),
            inner_mlp_size=integer("inner_mlp_size"),
            state_size=integer("state_size"),
            short_filter_length=integer("short_filter_length"),
            hcs_filter_length=integer("hcs_filter_length"),
            hcm_filter_length=integer("hcm_filter_length"),
            hcs_filter_groups=integer("hcs_filter_groups"),
            hcm_filter_groups=integer("hcm_filter_groups"),
            hcl_filter_groups=integer("hcl_filter_groups"),
            hcs_layer_idxs=indices("hcs_layer_idxs"),
            hcm_layer_idxs=indices("hcm_layer_idxs"),
            hcl_layer_idxs=indices("hcl_layer_idxs"),
            attn_layer_idxs=indices("attn_layer_idxs"),
            eps=number("eps"),
            rotary_emb_base=number("rotary_emb_base"),
            rotary_emb_scaling_factor=number("rotary_emb_scaling_factor"),
            max_seqlen=integer("max_seqlen"),
            max_batch_size=integer("max_batch_size"),
            inner_size_multiple_of=integer("inner_size_multiple_of"),
            proj_groups=integer("proj_groups"),
            tie_embeddings=boolean("tie_embeddings"),
            short_filter_bias=boolean("short_filter_bias"),
            use_fp8_input_projections=boolean("use_fp8_input_projections"),
            use_interpolated_rotary_pos_emb=boolean("use_interpolated_rotary_pos_emb"),
            evo2_style_activations=boolean("evo2_style_activations"),
            interleave=boolean("interleave"),
            column_split=boolean("column_split"),
            column_split_hyena=boolean("column_split_hyena"),
            hyena_flip_x1x2=boolean("hyena_flip_x1x2"),
            final_norm=boolean("final_norm"),
            qkv_proj_bias=boolean("qkv_proj_bias"),
            mha_out_proj_bias=boolean("mha_out_proj_bias"),
            hyena_out_proj_bias=boolean("hyena_out_proj_bias"),
            tokenizer_type=string("tokenizer_type"),
            prefill_style=string("prefill_style"),
            mlp_activation=string("mlp_activation"),
        )
        config.validate_supported()
        return config

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def registry(self) -> dict[str, Any]:
        registry = load_model_registry()
        entry = registry["models"].get(self.model_id)
        if not isinstance(entry, dict):
            raise ConfigError(
                f"unsupported model_id={self.model_id!r}; expected one of "
                + ", ".join(sorted(registry["models"]))
            )
        return entry

    @property
    def profile(self) -> dict[str, Any]:
        registry = load_model_registry()
        entry = self.registry
        profile = registry["profiles"].get(entry.get("profile"))
        if not isinstance(profile, dict):
            raise ConfigError(f"registry profile missing for {self.model_id}")
        return profile

    def validate_supported(self) -> None:
        expected_model_name = self.registry["expected_model_name"]
        if self.model_name != expected_model_name:
            raise ConfigError(
                f"{self.model_id}: model_name must be {expected_model_name!r}, "
                f"got {self.model_name!r}"
            )
        profile = self.profile
        for key, expected in profile.items():
            if (
                key == "use_fp8_input_projections"
                and self.model_id == "evo2_40b_bionemo_bf16"
            ):
                expected = False
            actual = getattr(self, key)
            if isinstance(actual, tuple):
                expected = tuple(expected)
            if actual != expected:
                raise ConfigError(
                    f"{self.model_id}: unsupported {key}={actual!r}; expected {expected!r}"
                )
        exact_values = {
            "vocab_size": (self.vocab_size, 512),
            "num_filters": (self.num_filters, self.hidden_size),
            "state_size": (self.state_size, 16),
            "short_filter_length": (self.short_filter_length, 3),
            "hcs_filter_length": (self.hcs_filter_length, 7),
            "hcm_filter_length": (self.hcm_filter_length, 128),
            "max_batch_size": (self.max_batch_size, 1),
            "inner_size_multiple_of": (self.inner_size_multiple_of, 128),
            "proj_groups": (self.proj_groups, 1),
        }
        for key, (actual, expected) in exact_values.items():
            if actual != expected:
                raise ConfigError(f"{self.model_id}: {key} must be {expected}, got {actual}")
        layers = (
            self.hcs_layer_idxs
            + self.hcm_layer_idxs
            + self.hcl_layer_idxs
            + self.attn_layer_idxs
        )
        if sorted(layers) != list(range(self.num_layers)) or len(set(layers)) != self.num_layers:
            raise ConfigError(
                f"{self.model_id}: layer sets must be disjoint and cover 0..{self.num_layers - 1}"
            )
        if self.hidden_size % self.num_attention_heads or self.head_size != 128:
            raise ConfigError(f"{self.model_id}: attention head size must be 128")
        required = {
            "tie_embeddings": self.tie_embeddings,
            "evo2_style_activations": self.evo2_style_activations,
            "interleave": self.interleave,
            "column_split": self.column_split,
            "final_norm": self.final_norm,
            "mha_out_proj_bias": self.mha_out_proj_bias,
            "hyena_out_proj_bias": self.hyena_out_proj_bias,
        }
        forbidden = {
            "short_filter_bias": self.short_filter_bias,
            "column_split_hyena": self.column_split_hyena,
            "hyena_flip_x1x2": self.hyena_flip_x1x2,
            "qkv_proj_bias": self.qkv_proj_bias,
        }
        for key, value in required.items():
            if not value:
                raise ConfigError(f"{self.model_id}: {key} must be true")
        for key, value in forbidden.items():
            if value:
                raise ConfigError(f"{self.model_id}: {key} must be false")
        if self.eps != 1e-6:
            raise ConfigError(f"{self.model_id}: eps must be 1e-6")
        exact_strings = {
            "tokenizer_type": (self.tokenizer_type, "CharLevelTokenizer"),
            "prefill_style": (self.prefill_style, "fft"),
            "mlp_activation": (self.mlp_activation, "gelu"),
        }
        for key, (actual, expected) in exact_strings.items():
            if actual != expected:
                raise ConfigError(f"{self.model_id}: {key} must be {expected!r}")

    def validate_supported_40b(self) -> None:
        """Compatibility shim for callers which previously selected only 40B."""
        self.validate_supported()
        if self.hidden_size != 8192 or self.num_layers != 50:
            raise ConfigError(f"{self.model_id} is supported, but is not a 40B topology")


def _parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith("["):
        value = ast.literal_eval(text)
        if not isinstance(value, list):
            raise ConfigError(f"expected list, got {text!r}")
        return value
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", text):
        return float(text)
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return ast.literal_eval(text)
    return text


def load_config(path: Path) -> ModelConfig:
    """Parse the flat scalar/list subset used by official Evo 2 YAML configs."""
    values: dict[str, Any] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ConfigError(f"{path}:{line_number}: expected key: value")
        key, raw_value = (part.strip() for part in line.split(":", 1))
        if not key or not raw_value:
            raise ConfigError(f"{path}:{line_number}: empty key or value")
        if key in values:
            raise ConfigError(f"{path}:{line_number}: duplicate key {key!r}")
        values[key] = _parse_scalar(raw_value)
    return ModelConfig.from_mapping(values)


def _base_manifest(config: ModelConfig, *, runtime: bool) -> list[TensorSpec]:
    config.validate_supported()
    entry = config.registry
    norm_dtype = "F32" if runtime else entry["source_norm_dtype"]
    rope_dtype = "F32" if runtime else entry["source_rope_dtype"]
    projection_dtype = entry["source_projection_dtype"]
    width = config.hidden_size
    inner = config.inner_mlp_size
    specs = [
        TensorSpec("embedding_layer.weight", "BF16", (config.vocab_size, width)),
        TensorSpec("unembed.weight", "BF16", (config.vocab_size, width)),
    ]
    for layer in range(config.num_layers):
        prefix = f"blocks.{layer}"
        specs.append(TensorSpec(f"{prefix}.pre_norm.scale", norm_dtype, (width,)))
        if layer in config.attn_layer_idxs:
            specs.extend(
                [
                    TensorSpec(f"{prefix}.inner_mha_cls.out_proj.bias", "BF16", (width,)),
                    TensorSpec(f"{prefix}.inner_mha_cls.out_proj.weight", "BF16", (width, width)),
                    TensorSpec(f"{prefix}.inner_mha_cls.Wqkv.weight", "BF16", (3 * width, width)),
                    TensorSpec(
                        f"{prefix}.inner_mha_cls.rotary_emb.inv_freq",
                        rope_dtype,
                        (config.head_size // 2,),
                    ),
                ]
            )
        else:
            specs.extend(
                [
                    TensorSpec(f"{prefix}.out_filter_dense.bias", "BF16", (width,)),
                    TensorSpec(f"{prefix}.out_filter_dense.weight", "BF16", (width, width)),
                    TensorSpec(f"{prefix}.projections.weight", projection_dtype, (3 * width, width)),
                    TensorSpec(
                        f"{prefix}.filter.short_filter_weight",
                        "BF16",
                        (3 * width, 1, config.short_filter_length),
                    ),
                ]
            )
            if layer in config.hcs_layer_idxs:
                specs.append(
                    TensorSpec(
                        f"{prefix}.filter.h",
                        "BF16",
                        (config.hcs_filter_groups, 1, config.hcs_filter_length),
                    )
                )
            elif layer in config.hcm_layer_idxs:
                specs.extend(
                    [
                        TensorSpec(f"{prefix}.filter.D", "BF16", (width,)),
                        TensorSpec(
                            f"{prefix}.filter.h",
                            "BF16",
                            (config.hcm_filter_groups, 1, config.hcm_filter_length),
                        ),
                    ]
                )
            elif layer in config.hcl_layer_idxs:
                specs.extend(
                    [
                        TensorSpec(f"{prefix}.filter.D", "BF16", (width,)),
                        TensorSpec(
                            f"{prefix}.filter.log_poles",
                            "F32",
                            (width, config.state_size, 1),
                        ),
                        TensorSpec(
                            f"{prefix}.filter.residues",
                            "F32",
                            (width, config.state_size),
                        ),
                    ]
                )
            else:
                raise AssertionError(f"unclassified Hyena layer {layer}")
        specs.extend(
            [
                TensorSpec(f"{prefix}.mlp.l1.weight", "BF16", (inner, width)),
                TensorSpec(f"{prefix}.mlp.l2.weight", "BF16", (inner, width)),
                TensorSpec(f"{prefix}.mlp.l3.weight", "BF16", (width, inner)),
                TensorSpec(f"{prefix}.post_norm.scale", norm_dtype, (width,)),
            ]
        )
    specs.append(TensorSpec("norm.scale", norm_dtype, (width,)))
    return specs


def _validate_manifest(
    config: ModelConfig, specs: list[TensorSpec], registry_key: str
) -> list[TensorSpec]:
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise AssertionError("manifest generated duplicate tensor names")
    actual = {
        "tensors": len(specs),
        "bf16": sum(spec.dtype == "BF16" for spec in specs),
        "f32": sum(spec.dtype == "F32" for spec in specs),
        "bytes": sum(spec.nbytes for spec in specs),
    }
    expected = config.registry[registry_key]
    if actual != expected:
        raise AssertionError(
            f"{config.model_id} {registry_key} drift: {actual}; expected {expected}"
        )
    return specs


def checkpoint_manifest(config: ModelConfig) -> list[TensorSpec]:
    """Required tensor entries in the official Arc checkpoint."""
    if config.model_id == "evo2_40b_bionemo_bf16":
        raise ConfigError("BioNeMo DCP uses bionemo_checkpoint_manifest()")
    return _validate_manifest(config, _base_manifest(config, runtime=False), "source_manifest")


def ignored_checkpoint_manifest(config: ModelConfig) -> list[TensorSpec]:
    """Official deterministic time grids validated but recomputed by the runtime."""
    length = config.registry["ignored_time_grid_length"]
    if not length:
        return []
    return [
        TensorSpec(f"blocks.{layer}.mixer.mixer.filter.t", "F32", (1, 1, length))
        for layer in config.hcl_layer_idxs
    ]


def runtime_manifest(config: ModelConfig) -> list[TensorSpec]:
    """Tensor entries written to EVO2C after exact small-vector widening."""
    if config.model_id == "evo2_40b_bionemo_bf16":
        return bionemo_checkpoint_manifest(config)
    return _validate_manifest(config, _base_manifest(config, runtime=True), "runtime_manifest")


def container_manifest(config: ModelConfig) -> list[TensorSpec]:
    """Exact tensor table written to a production EVO2C container."""
    specs = runtime_manifest(config)
    if config.use_fp8_input_projections:
        for layer in sorted(
            config.hcs_layer_idxs + config.hcm_layer_idxs + config.hcl_layer_idxs
        ):
            prefix = f"blocks.{layer}.projections"
            specs.extend(
                [
                    TensorSpec(f"{prefix}.fp8_scale_fwd", "F32", (2,)),
                    TensorSpec(f"{prefix}.fp8_scale_inv_fwd", "F32", (2,)),
                    TensorSpec(f"{prefix}.fp8_amax_history_fwd", "F32", (16, 2)),
                ]
            )
    return _validate_manifest(config, specs, "container_manifest")


def bionemo_checkpoint_manifest(config: ModelConfig) -> list[TensorSpec]:
    """Vortex-compatible manifest produced from the official BioNeMo 40B DCP."""
    if config.model_id != "evo2_40b_bionemo_bf16":
        raise ConfigError("BioNeMo mapping is supported only for evo2_40b_bionemo_bf16")
    specs = _base_manifest(config, runtime=True)
    medium = {f"blocks.{layer}.filter.h" for layer in config.hcm_layer_idxs}
    specs = [
        dataclasses.replace(spec, dtype="F32") if spec.name in medium else spec
        for spec in specs
    ]
    return _validate_manifest(config, specs, "runtime_manifest")


def config_metadata(
    config: ModelConfig, checkpoint_name: str, checkpoint_size: int
) -> dict[str, Any]:
    registry = load_model_registry()
    entry = config.registry
    metadata = {
        "format.producer": "evo2c",
        "model.id": config.model_id,
        "model.name": config.model_name,
        "model.architecture": "StripedHyena2",
        "model.source_repo": entry["source_repo"],
        "model.source_revision": entry["source_revision"],
        "model.registry_schema": registry["schema_version"],
        "model.architecture_revision": registry["architecture_revision"],
        "model.runtime_layout_revision": registry["runtime_layout_revision"],
        "checkpoint.filename": checkpoint_name,
        "checkpoint.size": checkpoint_size,
        "checkpoint.extra_state_count": entry["extra_state_count"],
        "checkpoint.source_projection_dtype": entry["source_projection_dtype"],
        "checkpoint.source_norm_dtype": entry["source_norm_dtype"],
        "checkpoint.source_rope_dtype": entry["source_rope_dtype"],
        "conversion.exact_widen_norm_to_f32": entry["source_norm_dtype"] == "BF16",
        "conversion.exact_widen_rope_to_f32": entry["source_rope_dtype"] == "BF16",
        "hyena_projection_dtype": entry["projection_runtime_dtype"],
        "hyena_projection_weight_dtype": entry["source_projection_dtype"],
        **{
            f"config.{field.name}": (
                list(value) if isinstance(value := getattr(config, field.name), tuple) else value
            )
            for field in dataclasses.fields(config)
            if field.name not in {"model_id", "model_name"}
        },
    }
    if config.model_id in {"evo2_40b", "evo2_40b_bionemo_bf16"}:
        # Preserve the published v1 40B artifact bytes and SHA256. The native
        # loader recognizes these two legacy profiles by their complete,
        # unambiguous metadata signature.
        for key in (
            "model.id",
            "model.source_revision",
            "model.registry_schema",
            "model.architecture_revision",
            "model.runtime_layout_revision",
            "checkpoint.source_projection_dtype",
            "checkpoint.source_norm_dtype",
            "checkpoint.source_rope_dtype",
            "conversion.exact_widen_norm_to_f32",
            "conversion.exact_widen_rope_to_f32",
            "hyena_projection_weight_dtype",
        ):
            del metadata[key]
    return metadata
