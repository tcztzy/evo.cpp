#!/usr/bin/env python3
"""Generate the independent tiny GENEB Llama/Mistral decoder oracle.

The portable runtime stores activations as F32.  This oracle models BF16 eager
execution by applying round-to-nearest-even casts at the same tensor-operation
boundaries.  ``--torch-check`` additionally compares the two easy-to-misstate
boundaries (LlamaRMSNorm and split-half RoPE) with real PyTorch eager BF16 ops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:
    print("NumPy is required to generate GENEB decoder oracle vectors", file=sys.stderr)
    raise SystemExit(77)


TOKENS = (1, 4, 2, 7)
POSITION_OFFSET = 3
VOCABULARY = 11
WIDTH = 8
LAYERS = 2
QUERY_HEADS = 2
KV_HEADS = 1
HEAD_DIMENSION = 4
ROTARY_DIMENSION = 4
INNER_WIDTH = 12
SLIDING_WINDOW = 3
RMS_EPSILON = np.float32(1.0e-5)
ROPE_BASE = np.float32(500000.0)

RMS_PROBE_INPUT = (0.1001, -0.3003, 0.7007, -1.1009)
RMS_PROBE_WEIGHT = (1.003, 0.997, 1.011, 0.989)
ROPE_PROBE_INPUT = (0.1001, -0.3003, 0.7007, -1.1009)
ROPE_PROBE_POSITION = 7


def bf16_round(values: Any) -> np.ndarray:
    """Return F32 storage containing IEEE RNE BF16 values."""

    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32).copy()
    finite = (bits & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bits = np.where(finite, bits + bias, bits).astype(np.uint32, copy=False)
    bits &= np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def cast(value: Any, activation_dtype: str) -> np.float32:
    item = np.float32(value)
    if activation_dtype == "BF16":
        return np.float32(bf16_round(np.asarray([item], dtype=np.float32))[0])
    return item


def tensor_requirements() -> list[tuple[str, tuple[int, ...]]]:
    result: list[tuple[str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", (VOCABULARY, WIDTH))
    ]
    query_width = QUERY_HEADS * HEAD_DIMENSION
    kv_width = KV_HEADS * HEAD_DIMENSION
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}."
        result.extend(
            [
                (prefix + "input_layernorm.weight", (WIDTH,)),
                (prefix + "self_attn.q_proj.weight", (query_width, WIDTH)),
                (prefix + "self_attn.k_proj.weight", (kv_width, WIDTH)),
                (prefix + "self_attn.v_proj.weight", (kv_width, WIDTH)),
                (prefix + "self_attn.o_proj.weight", (WIDTH, query_width)),
                (prefix + "self_attn.q_proj.bias", (query_width,)),
                (prefix + "self_attn.k_proj.bias", (kv_width,)),
                (prefix + "self_attn.v_proj.bias", (kv_width,)),
                (prefix + "self_attn.o_proj.bias", (WIDTH,)),
                (prefix + "post_attention_layernorm.weight", (WIDTH,)),
                (prefix + "mlp.gate_proj.weight", (INNER_WIDTH, WIDTH)),
                (prefix + "mlp.up_proj.weight", (INNER_WIDTH, WIDTH)),
                (prefix + "mlp.down_proj.weight", (WIDTH, INNER_WIDTH)),
                (prefix + "mlp.gate_proj.bias", (INNER_WIDTH,)),
                (prefix + "mlp.up_proj.bias", (INNER_WIDTH,)),
                (prefix + "mlp.down_proj.bias", (WIDTH,)),
            ]
        )
    result.append(("model.norm.weight", (WIDTH,)))
    return result


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> np.float32:
    integer = ((tensor_index + 1) * 17 + (element_index + 3) * 13) % 37 - 18
    value = np.float32(np.float32(integer) / np.float32(29.0))
    if "layernorm.weight" in name or name == "model.norm.weight":
        value = np.float32(np.float32(1.0) + np.float32(value * np.float32(0.05)))
    elif name.endswith(".bias"):
        value = np.float32(value * np.float32(0.1))
    else:
        value = np.float32(value * np.float32(0.2))
    return value


def fixture_weights(weight_dtype: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for tensor_index, (name, shape) in enumerate(tensor_requirements()):
        count = math.prod(shape)
        values = np.asarray(
            [fixture_scalar(name, tensor_index, index) for index in range(count)],
            dtype=np.float32,
        ).reshape(shape)
        result[name] = bf16_round(values) if weight_dtype == "BF16" else values
    return result


def linear(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    activation_dtype: str,
) -> np.ndarray:
    rows, input_width = values.shape
    output_width = weight.shape[0]
    result = np.zeros((rows, output_width), dtype=np.float32)
    for row in range(rows):
        for target in range(output_width):
            total = np.float32(0.0 if bias is None else bias[target])
            for source in range(input_width):
                product = np.float32(values[row, source] * weight[target, source])
                total = np.float32(total + product)
            result[row, target] = cast(total, activation_dtype)
    return result


def rms_norm(
    values: np.ndarray,
    weight: np.ndarray,
    epsilon: np.float32,
    activation_dtype: str,
) -> np.ndarray:
    rows, width = values.shape
    result = np.empty_like(values)
    for row in range(rows):
        sum_squares = np.float32(0.0)
        for column in range(width):
            square = np.float32(values[row, column] * values[row, column])
            sum_squares = np.float32(sum_squares + square)
        mean_square = np.float32(sum_squares / np.float32(width))
        denominator = np.float32(np.sqrt(np.float32(mean_square + epsilon)))
        for column in range(width):
            # HF: F32 normalize -> cast(input dtype) -> multiply norm weight.
            normalized = cast(
                np.float32(values[row, column] / denominator), activation_dtype
            )
            result[row, column] = cast(
                np.float32(normalized * weight[column]), activation_dtype
            )
    return result


def apply_split_half_rope(
    values: np.ndarray,
    heads: int,
    position_offset: int,
    rope_base: np.float32,
    activation_dtype: str,
) -> np.ndarray:
    rows = values.shape[0]
    result = values.reshape(rows, heads, HEAD_DIMENSION).copy()
    pairs = ROTARY_DIMENSION // 2
    for row in range(rows):
        position = np.float32(position_offset + row)
        for head in range(heads):
            for pair in range(pairs):
                exponent = np.float32(
                    np.float32(2.0) * np.float32(pair) / np.float32(ROTARY_DIMENSION)
                )
                denominator = np.float32(np.power(rope_base, exponent))
                angle = np.float32(position / denominator)
                cosine = cast(np.float32(np.cos(angle)), activation_dtype)
                sine = cast(np.float32(np.sin(angle)), activation_dtype)
                first = np.float32(result[row, head, pair])
                second = np.float32(result[row, head, pair + pairs])
                first_cosine = cast(np.float32(first * cosine), activation_dtype)
                second_sine = cast(np.float32(second * sine), activation_dtype)
                second_cosine = cast(np.float32(second * cosine), activation_dtype)
                first_sine = cast(np.float32(first * sine), activation_dtype)
                result[row, head, pair] = cast(
                    np.float32(first_cosine - second_sine), activation_dtype
                )
                result[row, head, pair + pairs] = cast(
                    np.float32(second_cosine + first_sine), activation_dtype
                )
    return result.reshape(rows, heads * HEAD_DIMENSION)


def causal_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    activation_dtype: str,
) -> np.ndarray:
    rows = query.shape[0]
    query = query.reshape(rows, QUERY_HEADS, HEAD_DIMENSION)
    key = key.reshape(rows, KV_HEADS, HEAD_DIMENSION)
    value = value.reshape(rows, KV_HEADS, HEAD_DIMENSION)
    result = np.zeros_like(query)
    heads_per_kv = QUERY_HEADS // KV_HEADS
    scale = np.float32(np.float32(1.0) / np.sqrt(np.float32(HEAD_DIMENSION)))
    for row in range(rows):
        first = 0 if row + 1 <= SLIDING_WINDOW else row + 1 - SLIDING_WINDOW
        for query_head in range(QUERY_HEADS):
            kv_head = query_head // heads_per_kv
            scores: list[np.float32] = []
            for source in range(first, row + 1):
                score = np.float32(0.0)
                for dimension in range(HEAD_DIMENSION):
                    product = np.float32(
                        query[row, query_head, dimension]
                        * key[source, kv_head, dimension]
                    )
                    score = np.float32(score + product)
                score = cast(score, activation_dtype)
                scores.append(cast(np.float32(score * scale), activation_dtype))
            maximum = max(scores)
            exponentials: list[np.float32] = []
            denominator = np.float32(0.0)
            for score in scores:
                exponential = np.float32(np.exp(np.float32(score - maximum)))
                exponentials.append(exponential)
                denominator = np.float32(denominator + exponential)
            probabilities = [
                cast(np.float32(item / denominator), activation_dtype)
                for item in exponentials
            ]
            for dimension in range(HEAD_DIMENSION):
                total = np.float32(0.0)
                for index, source in enumerate(range(first, row + 1)):
                    product = np.float32(
                        probabilities[index] * value[source, kv_head, dimension]
                    )
                    total = np.float32(total + product)
                result[row, query_head, dimension] = cast(total, activation_dtype)
    return result.reshape(rows, QUERY_HEADS * HEAD_DIMENSION)


def silu(value: np.float32) -> np.float32:
    if value >= np.float32(0.0):
        return np.float32(
            value / np.float32(np.float32(1.0) + np.exp(np.float32(-value)))
        )
    exponential = np.float32(np.exp(value))
    return np.float32(
        value * exponential / np.float32(np.float32(1.0) + exponential)
    )


def forward(weight_dtype: str, activation_dtype: str) -> dict[str, list[float]]:
    weights = fixture_weights(weight_dtype)
    hidden = weights["model.embed_tokens.weight"][list(TOKENS), :].copy()
    if activation_dtype == "BF16":
        hidden = bf16_round(hidden)
    vectors: dict[str, list[float]] = {"capture_0": hidden.reshape(-1).tolist()}
    query_width = QUERY_HEADS * HEAD_DIMENSION
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}."
        normalized = rms_norm(
            hidden,
            weights[prefix + "input_layernorm.weight"],
            RMS_EPSILON,
            activation_dtype,
        )
        query = linear(
            normalized,
            weights[prefix + "self_attn.q_proj.weight"],
            weights[prefix + "self_attn.q_proj.bias"],
            activation_dtype,
        )
        key = linear(
            normalized,
            weights[prefix + "self_attn.k_proj.weight"],
            weights[prefix + "self_attn.k_proj.bias"],
            activation_dtype,
        )
        value = linear(
            normalized,
            weights[prefix + "self_attn.v_proj.weight"],
            weights[prefix + "self_attn.v_proj.bias"],
            activation_dtype,
        )
        query = apply_split_half_rope(
            query, QUERY_HEADS, POSITION_OFFSET, ROPE_BASE, activation_dtype
        )
        key = apply_split_half_rope(
            key, KV_HEADS, POSITION_OFFSET, ROPE_BASE, activation_dtype
        )
        attended = causal_attention(query, key, value, activation_dtype)
        projected = linear(
            attended.reshape(len(TOKENS), query_width),
            weights[prefix + "self_attn.o_proj.weight"],
            weights[prefix + "self_attn.o_proj.bias"],
            activation_dtype,
        )
        hidden = np.asarray(
            [
                cast(np.float32(left + right), activation_dtype)
                for left, right in zip(hidden.reshape(-1), projected.reshape(-1), strict=True)
            ],
            dtype=np.float32,
        ).reshape(hidden.shape)
        normalized = rms_norm(
            hidden,
            weights[prefix + "post_attention_layernorm.weight"],
            RMS_EPSILON,
            activation_dtype,
        )
        up = linear(
            normalized,
            weights[prefix + "mlp.up_proj.weight"],
            weights[prefix + "mlp.up_proj.bias"],
            activation_dtype,
        )
        gate = linear(
            normalized,
            weights[prefix + "mlp.gate_proj.weight"],
            weights[prefix + "mlp.gate_proj.bias"],
            activation_dtype,
        )
        for index in range(up.size):
            activated = cast(silu(gate.reshape(-1)[index]), activation_dtype)
            up.reshape(-1)[index] = cast(
                np.float32(up.reshape(-1)[index] * activated), activation_dtype
            )
        mlp_output = linear(
            up,
            weights[prefix + "mlp.down_proj.weight"],
            weights[prefix + "mlp.down_proj.bias"],
            activation_dtype,
        )
        hidden = np.asarray(
            [
                cast(np.float32(left + right), activation_dtype)
                for left, right in zip(hidden.reshape(-1), mlp_output.reshape(-1), strict=True)
            ],
            dtype=np.float32,
        ).reshape(hidden.shape)
        if layer + 1 < LAYERS:
            vectors[f"capture_{layer + 1}"] = hidden.reshape(-1).tolist()
    final_hidden = rms_norm(
        hidden, weights["model.norm.weight"], RMS_EPSILON, activation_dtype
    )
    vectors[f"capture_{LAYERS}"] = final_hidden.reshape(-1).tolist()
    vectors["final_hidden"] = final_hidden.reshape(-1).tolist()
    return vectors


def boundary_probes() -> dict[str, list[float]]:
    rms_input = bf16_round(np.asarray(RMS_PROBE_INPUT, dtype=np.float32)).reshape(1, 4)
    rms_weight = bf16_round(np.asarray(RMS_PROBE_WEIGHT, dtype=np.float32))
    rms_output = rms_norm(rms_input, rms_weight, RMS_EPSILON, "BF16")
    rope_input = bf16_round(np.asarray(ROPE_PROBE_INPUT, dtype=np.float32)).reshape(1, 4)
    rope_output = apply_split_half_rope(
        rope_input, 1, ROPE_PROBE_POSITION, np.float32(10000.0), "BF16"
    )
    return {
        "bf16.rmsnorm_probe": rms_output.reshape(-1).tolist(),
        "bf16.rope_split_probe": rope_output.reshape(-1).tolist(),
    }


def torch_check(
    probes: dict[str, list[float]], expected_forward: dict[str, list[float]]
) -> str:
    try:
        import torch
        import torch.nn.functional as torch_functional
    except ModuleNotFoundError:
        print("PyTorch is required for --torch-check", file=sys.stderr)
        raise SystemExit(77)

    rms_input = torch.tensor(RMS_PROBE_INPUT, dtype=torch.float32).to(torch.bfloat16)
    rms_weight = torch.tensor(RMS_PROBE_WEIGHT, dtype=torch.float32).to(torch.bfloat16)
    normalized = rms_input.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + float(RMS_EPSILON))
    rms_output = (rms_weight * normalized.to(torch.bfloat16)).float().tolist()

    rope_input = torch.tensor(ROPE_PROBE_INPUT, dtype=torch.float32).to(torch.bfloat16)
    inv_frequency = 1.0 / (
        10000.0
        ** (torch.arange(0, ROTARY_DIMENSION, 2, dtype=torch.float32) / ROTARY_DIMENSION)
    )
    frequencies = inv_frequency * float(ROPE_PROBE_POSITION)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos().to(torch.bfloat16)
    sine = embedding.sin().to(torch.bfloat16)
    first, second = rope_input[:2], rope_input[2:]
    rotated = torch.cat((-second, first), dim=-1)
    rope_output = ((rope_input * cosine) + (rotated * sine)).float().tolist()

    actual = {
        "bf16.rmsnorm_probe": rms_output,
        "bf16.rope_split_probe": rope_output,
    }
    for name, expected in probes.items():
        if np.asarray(actual[name], dtype=np.float32).tobytes() != np.asarray(
            expected, dtype=np.float32
        ).tobytes():
            raise AssertionError(
                f"{name} differs from PyTorch eager BF16: "
                f"expected {expected}, got {actual[name]}"
            )

    weights = {
        name: torch.tensor(value, dtype=torch.float32).to(torch.bfloat16)
        for name, value in fixture_weights("BF16").items()
    }

    def torch_rms(values: Any, weight: Any) -> Any:
        normalized_values = values.float()
        variance_values = normalized_values.pow(2).mean(-1, keepdim=True)
        normalized_values = normalized_values * torch.rsqrt(
            variance_values + float(RMS_EPSILON)
        )
        return weight * normalized_values.to(values.dtype)

    def torch_rope(values: Any, position_offset: int) -> Any:
        positions = torch.arange(
            position_offset,
            position_offset + values.shape[0],
            dtype=torch.float32,
        )
        inverse_frequency = 1.0 / (
            float(ROPE_BASE)
            ** (
                torch.arange(0, ROTARY_DIMENSION, 2, dtype=torch.float32)
                / ROTARY_DIMENSION
            )
        )
        frequencies_values = torch.outer(positions, inverse_frequency)
        embedding_values = torch.cat(
            (frequencies_values, frequencies_values), dim=-1
        )
        cosine_values = embedding_values.cos().to(values.dtype)[:, None, :]
        sine_values = embedding_values.sin().to(values.dtype)[:, None, :]
        first_values, second_values = values[..., :2], values[..., 2:]
        rotated_values = torch.cat((-second_values, first_values), dim=-1)
        return (values * cosine_values) + (rotated_values * sine_values)

    hidden = weights["model.embed_tokens.weight"][list(TOKENS), :]
    torch_vectors: dict[str, list[float]] = {
        "capture_0": hidden.float().reshape(-1).tolist()
    }
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}."
        normalized = torch_rms(
            hidden, weights[prefix + "input_layernorm.weight"]
        )
        query = torch_functional.linear(
            normalized,
            weights[prefix + "self_attn.q_proj.weight"],
            weights[prefix + "self_attn.q_proj.bias"],
        ).reshape(len(TOKENS), QUERY_HEADS, HEAD_DIMENSION)
        key = torch_functional.linear(
            normalized,
            weights[prefix + "self_attn.k_proj.weight"],
            weights[prefix + "self_attn.k_proj.bias"],
        ).reshape(len(TOKENS), KV_HEADS, HEAD_DIMENSION)
        value = torch_functional.linear(
            normalized,
            weights[prefix + "self_attn.v_proj.weight"],
            weights[prefix + "self_attn.v_proj.bias"],
        ).reshape(len(TOKENS), KV_HEADS, HEAD_DIMENSION)
        query = torch_rope(query, POSITION_OFFSET)
        key = torch_rope(key, POSITION_OFFSET)
        attended = torch.empty_like(query)
        heads_per_kv = QUERY_HEADS // KV_HEADS
        for row in range(len(TOKENS)):
            first_source = (
                0
                if row + 1 <= SLIDING_WINDOW
                else row + 1 - SLIDING_WINDOW
            )
            for query_head in range(QUERY_HEADS):
                kv_head = query_head // heads_per_kv
                scores = torch.matmul(
                    query[row, query_head],
                    key[first_source : row + 1, kv_head].transpose(0, 1),
                ) * (1.0 / math.sqrt(HEAD_DIMENSION))
                probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                    query.dtype
                )
                attended[row, query_head] = torch.matmul(
                    probabilities, value[first_source : row + 1, kv_head]
                )
        projected = torch_functional.linear(
            attended.reshape(len(TOKENS), QUERY_HEADS * HEAD_DIMENSION),
            weights[prefix + "self_attn.o_proj.weight"],
            weights[prefix + "self_attn.o_proj.bias"],
        )
        hidden = hidden + projected
        normalized = torch_rms(
            hidden, weights[prefix + "post_attention_layernorm.weight"]
        )
        gate = torch_functional.linear(
            normalized,
            weights[prefix + "mlp.gate_proj.weight"],
            weights[prefix + "mlp.gate_proj.bias"],
        )
        up = torch_functional.linear(
            normalized,
            weights[prefix + "mlp.up_proj.weight"],
            weights[prefix + "mlp.up_proj.bias"],
        )
        mlp_output = torch_functional.linear(
            torch_functional.silu(gate) * up,
            weights[prefix + "mlp.down_proj.weight"],
            weights[prefix + "mlp.down_proj.bias"],
        )
        hidden = hidden + mlp_output
        if layer + 1 < LAYERS:
            torch_vectors[f"capture_{layer + 1}"] = (
                hidden.float().reshape(-1).tolist()
            )
    final_hidden = torch_rms(hidden, weights["model.norm.weight"])
    torch_vectors[f"capture_{LAYERS}"] = final_hidden.float().reshape(-1).tolist()
    torch_vectors["final_hidden"] = final_hidden.float().reshape(-1).tolist()

    for name, expected in expected_forward.items():
        actual_values = torch_vectors[name]
        if np.asarray(actual_values, dtype=np.float32).tobytes() != np.asarray(
            expected, dtype=np.float32
        ).tobytes():
            differences = np.abs(
                np.asarray(actual_values, dtype=np.float32)
                - np.asarray(expected, dtype=np.float32)
            )
            raise AssertionError(
                f"bf16.{name} differs from full PyTorch eager oracle: "
                f"max_abs={float(differences.max())}"
            )
    return str(torch.__version__)


def vector_digest(order: list[str], vectors: dict[str, list[float]]) -> str:
    digest = hashlib.sha256()
    for name in order:
        digest.update(np.asarray(vectors[name], dtype="<f4").tobytes())
    return digest.hexdigest()


def build_fixture(*, validate_torch: bool) -> dict[str, Any]:
    vectors: dict[str, list[float]] = {}
    bf16_forward: dict[str, list[float]] = {}
    for prefix, dtype in (("f32", "F32"), ("bf16", "BF16")):
        generated_forward = forward(dtype, dtype)
        if dtype == "BF16":
            bf16_forward = generated_forward
        for name, values in generated_forward.items():
            vectors[f"{prefix}.{name}"] = values
    probes = boundary_probes()
    vectors.update(probes)
    torch_version = (
        torch_check(probes, bf16_forward) if validate_torch else "run --torch-check"
    )
    order = list(vectors)
    return {
        "schema": 1,
        "reference": {
            "implementation": "independent NumPy scalar reference",
            "torch_bf16_full_forward_check": torch_version,
            "rope_layout": "split-half",
            "rms_epsilon_placement": "inside-sqrt",
            "bf16_rounding": "IEEE round-to-nearest-even at HF eager boundaries",
        },
        "tolerance": {
            "f32": {"atol": 2.0e-5, "rtol": 2.0e-4},
            "bf16": {"atol": 0.0, "rtol": 0.0},
        },
        "vector_order": order,
        "vector_sha256": vector_digest(order, vectors),
        "vectors": vectors,
    }


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path)
    parser.add_argument("--torch-check", action="store_true")
    args = parser.parse_args()

    generated = build_fixture(validate_torch=args.torch_check)
    if args.check is not None:
        committed = json.loads(args.check.read_text(encoding="utf-8"))
        # The check command need not have PyTorch installed. Ignore the
        # environment-reporting field while checking all numerical content.
        generated["reference"]["torch_bf16_full_forward_check"] = committed[
            "reference"
        ]["torch_bf16_full_forward_check"]
        if canonical(generated) != canonical(committed):
            raise AssertionError(f"GENEB decoder oracle fixture is stale: {args.check}")
        print(f"GENEB decoder oracle fixture is current: {args.check}")
        return 0
    print(json.dumps(generated, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
