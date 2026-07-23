"""Strict Evo 2 40B config parsing and checkpoint tensor manifest."""

from __future__ import annotations

import ast
import dataclasses
import math
import re
from pathlib import Path
from typing import Any


HCS_LAYERS = (0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46)
HCM_LAYERS = (1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47)
HCL_LAYERS = (2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48)
ATTN_LAYERS = (3, 10, 17, 24, 31, 35, 42, 49)

EXPECTED_TENSOR_COUNT = 537
EXPECTED_BF16_TENSOR_COUNT = 400
EXPECTED_F32_TENSOR_COUNT = 137
EXPECTED_TENSOR_BYTES = 82_252_533_760
EXPECTED_EXTRA_STATE_COUNT = 258


class ConfigError(ValueError):
    """Raised when config does not describe the supported 40B topology."""


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
        return self.elements * (2 if self.dtype == "BF16" else 4)


@dataclasses.dataclass(frozen=True, slots=True)
class ModelConfig:
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
            if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
                raise ConfigError(f"{key} must be an integer list")
            return tuple(value)

        model_name = values.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ConfigError("model_name must be a nonempty string")

        config = cls(
            model_name=model_name,
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
        config.validate_supported_40b()
        return config

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate_supported_40b(self) -> None:
        exact_values = {
            "vocab_size": (self.vocab_size, 512),
            "hidden_size": (self.hidden_size, 8192),
            "num_filters": (self.num_filters, 8192),
            "num_layers": (self.num_layers, 50),
            "num_attention_heads": (self.num_attention_heads, 64),
            "inner_mlp_size": (self.inner_mlp_size, 22528),
            "state_size": (self.state_size, 16),
            "short_filter_length": (self.short_filter_length, 3),
            "hcs_filter_length": (self.hcs_filter_length, 7),
            "hcm_filter_length": (self.hcm_filter_length, 128),
            "hcs_filter_groups": (self.hcs_filter_groups, 512),
            "hcm_filter_groups": (self.hcm_filter_groups, 512),
            "hcl_filter_groups": (self.hcl_filter_groups, 8192),
            "max_seqlen": (self.max_seqlen, 1_048_576),
            "max_batch_size": (self.max_batch_size, 1),
            "inner_size_multiple_of": (self.inner_size_multiple_of, 128),
            "proj_groups": (self.proj_groups, 1),
        }
        for key, (actual, expected) in exact_values.items():
            if actual != expected:
                raise ConfigError(f"unsupported {key}={actual}; Evo 2 40B requires {expected}")

        exact_layers = {
            "hcs_layer_idxs": (self.hcs_layer_idxs, HCS_LAYERS),
            "hcm_layer_idxs": (self.hcm_layer_idxs, HCM_LAYERS),
            "hcl_layer_idxs": (self.hcl_layer_idxs, HCL_LAYERS),
            "attn_layer_idxs": (self.attn_layer_idxs, ATTN_LAYERS),
        }
        for key, (actual, expected) in exact_layers.items():
            if actual != expected:
                raise ConfigError(f"unsupported {key}={list(actual)}; expected {list(expected)}")

        all_layers = self.hcs_layer_idxs + self.hcm_layer_idxs + self.hcl_layer_idxs + self.attn_layer_idxs
        if sorted(all_layers) != list(range(self.num_layers)) or len(set(all_layers)) != self.num_layers:
            raise ConfigError("layer sets must be disjoint and cover layers 0..49")
        if self.hidden_size % self.num_attention_heads != 0 or self.head_size != 128:
            raise ConfigError("Evo 2 40B attention requires 64 heads of size 128")

        required_true = {
            "tie_embeddings": self.tie_embeddings,
            "use_fp8_input_projections": self.use_fp8_input_projections,
            "use_interpolated_rotary_pos_emb": self.use_interpolated_rotary_pos_emb,
            "evo2_style_activations": self.evo2_style_activations,
            "interleave": self.interleave,
            "column_split": self.column_split,
            "final_norm": self.final_norm,
            "mha_out_proj_bias": self.mha_out_proj_bias,
            "hyena_out_proj_bias": self.hyena_out_proj_bias,
        }
        for key, value in required_true.items():
            if not value:
                raise ConfigError(f"{key} must be true for Evo 2 40B checkpoint semantics")
        required_false = {
            "short_filter_bias": self.short_filter_bias,
            "column_split_hyena": self.column_split_hyena,
            "hyena_flip_x1x2": self.hyena_flip_x1x2,
            "qkv_proj_bias": self.qkv_proj_bias,
        }
        for key, value in required_false.items():
            if value:
                raise ConfigError(f"{key} must be false for Evo 2 40B checkpoint semantics")
        if self.eps != 1e-6 or self.rotary_emb_base != 1e6 or self.rotary_emb_scaling_factor != 128:
            raise ConfigError("RMSNorm/RoPE constants do not match Evo 2 40B")
        exact_strings = {
            "tokenizer_type": (self.tokenizer_type, "CharLevelTokenizer"),
            "prefill_style": (self.prefill_style, "fft"),
            "mlp_activation": (self.mlp_activation, "gelu"),
        }
        for key, (actual, expected) in exact_strings.items():
            if actual != expected:
                raise ConfigError(f"{key} must be {expected!r}, got {actual!r}")


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
    if (text.startswith("\"") and text.endswith("\"")) or (
        text.startswith("'") and text.endswith("'")
    ):
        return ast.literal_eval(text)
    return text


def load_config(path: Path) -> ModelConfig:
    """Parse the flat scalar/list subset used by official Evo 2 YAML configs."""
    values: dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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


def checkpoint_manifest(config: ModelConfig) -> list[TensorSpec]:
    """Return exact tensor data entries present in official evo2_40b.pt."""
    config.validate_supported_40b()
    width = config.hidden_size
    inner = config.inner_mlp_size
    specs = [
        TensorSpec("embedding_layer.weight", "BF16", (config.vocab_size, width)),
        # Present in checkpoint even though current Vortex ties final projection to embedding.
        TensorSpec("unembed.weight", "BF16", (config.vocab_size, width)),
    ]

    for layer in range(config.num_layers):
        prefix = f"blocks.{layer}"
        specs.append(TensorSpec(f"{prefix}.pre_norm.scale", "F32", (width,)))
        if layer in config.attn_layer_idxs:
            specs.extend(
                [
                    TensorSpec(f"{prefix}.inner_mha_cls.out_proj.bias", "BF16", (width,)),
                    TensorSpec(f"{prefix}.inner_mha_cls.out_proj.weight", "BF16", (width, width)),
                    TensorSpec(f"{prefix}.inner_mha_cls.Wqkv.weight", "BF16", (3 * width, width)),
                    TensorSpec(
                        f"{prefix}.inner_mha_cls.rotary_emb.inv_freq",
                        "F32",
                        (config.head_size // 2,),
                    ),
                ]
            )
        else:
            specs.extend(
                [
                    TensorSpec(f"{prefix}.out_filter_dense.bias", "BF16", (width,)),
                    TensorSpec(f"{prefix}.out_filter_dense.weight", "BF16", (width, width)),
                    TensorSpec(f"{prefix}.projections.weight", "BF16", (3 * width, width)),
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
                TensorSpec(f"{prefix}.post_norm.scale", "F32", (width,)),
            ]
        )
    specs.append(TensorSpec("norm.scale", "F32", (width,)))

    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise AssertionError("manifest generated duplicate tensor names")
    dtype_counts = {dtype: sum(spec.dtype == dtype for spec in specs) for dtype in ("BF16", "F32")}
    total_bytes = sum(spec.nbytes for spec in specs)
    expected = (
        len(specs),
        dtype_counts["BF16"],
        dtype_counts["F32"],
        total_bytes,
    )
    if expected != (
        EXPECTED_TENSOR_COUNT,
        EXPECTED_BF16_TENSOR_COUNT,
        EXPECTED_F32_TENSOR_COUNT,
        EXPECTED_TENSOR_BYTES,
    ):
        raise AssertionError(f"40B manifest drift: {expected}")
    return specs


def config_metadata(config: ModelConfig, checkpoint_name: str, checkpoint_size: int) -> dict[str, Any]:
    return {
        "format.producer": "evo2c",
        "model.name": config.model_name,
        "model.architecture": "StripedHyena2",
        "model.source_repo": "arcinstitute/evo2_40b",
        "checkpoint.filename": checkpoint_name,
        "checkpoint.size": checkpoint_size,
        "checkpoint.extra_state_count": EXPECTED_EXTRA_STATE_COUNT,
        "config.vocab_size": config.vocab_size,
        "config.hidden_size": config.hidden_size,
        "config.num_filters": config.num_filters,
        "config.num_layers": config.num_layers,
        "config.num_attention_heads": config.num_attention_heads,
        "config.inner_mlp_size": config.inner_mlp_size,
        "config.state_size": config.state_size,
        "config.short_filter_length": config.short_filter_length,
        "config.hcs_filter_length": config.hcs_filter_length,
        "config.hcm_filter_length": config.hcm_filter_length,
        "config.hcs_filter_groups": config.hcs_filter_groups,
        "config.hcm_filter_groups": config.hcm_filter_groups,
        "config.hcl_filter_groups": config.hcl_filter_groups,
        "config.hcs_layer_idxs": list(config.hcs_layer_idxs),
        "config.hcm_layer_idxs": list(config.hcm_layer_idxs),
        "config.hcl_layer_idxs": list(config.hcl_layer_idxs),
        "config.attn_layer_idxs": list(config.attn_layer_idxs),
        "config.eps": config.eps,
        "config.rotary_emb_base": config.rotary_emb_base,
        "config.rotary_emb_scaling_factor": config.rotary_emb_scaling_factor,
        "config.max_seqlen": config.max_seqlen,
        "config.max_batch_size": config.max_batch_size,
        "config.inner_size_multiple_of": config.inner_size_multiple_of,
        "config.proj_groups": config.proj_groups,
        "config.tie_embeddings": config.tie_embeddings,
        "config.short_filter_bias": config.short_filter_bias,
        "config.use_fp8_input_projections": config.use_fp8_input_projections,
        "config.use_interpolated_rotary_pos_emb": config.use_interpolated_rotary_pos_emb,
        "config.evo2_style_activations": config.evo2_style_activations,
        "config.interleave": config.interleave,
        "config.column_split": config.column_split,
        "config.column_split_hyena": config.column_split_hyena,
        "config.hyena_flip_x1x2": config.hyena_flip_x1x2,
        "config.final_norm": config.final_norm,
        "config.qkv_proj_bias": config.qkv_proj_bias,
        "config.mha_out_proj_bias": config.mha_out_proj_bias,
        "config.hyena_out_proj_bias": config.hyena_out_proj_bias,
        "config.tokenizer_type": config.tokenizer_type,
        "config.prefill_style": config.prefill_style,
        "config.mlp_activation": config.mlp_activation,
    }
