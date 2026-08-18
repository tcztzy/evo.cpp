#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy oracle for tiny HyenaDNA and Evo-1 fixtures.

The oracle intentionally uses direct causal convolution instead of the native
FFT implementation.  It spells out float32 accumulation, Evo-1 BF16 rounding,
the per-head x2/x1/v projection layout, interpolated split-half RoPE, streaming
causal attention, and the pinned sqrt(mean(x^2))+epsilon RMS denominator.
It does not import runtime or converter code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    print("NumPy is required for the GENEB long-Hyena oracle", file=sys.stderr)
    raise SystemExit(77)


Shape = Tuple[int, ...]
Requirement = Tuple[str, str, Shape]
Weights = Dict[str, np.ndarray]


HYENA = {
    "vocab": 12,
    "embedding_rows": 16,
    "width": 4,
    "layers": 1,
    "inner": 6,
    "filter": 3,
    "positional": 3,
    "short": 2,
    "maximum": 4,
    "epsilon": np.float32(1.0e-5),
}
EVO = {
    "vocab": 8,
    "width": 4,
    "layers": 2,
    "heads": 2,
    "inner": 6,
    "state": 2,
    "short": 2,
    "maximum": 8,
    "epsilon": np.float32(1.0e-4),
    "rope_theta": np.float32(10000.0),
    "rope_scale": np.float32(2.0),
    "attention": (1,),
}


def product(shape: Sequence[int]) -> int:
    value = 1
    for dimension in shape:
        value *= dimension
    return value


def hyena_requirements() -> List[Requirement]:
    width = int(HYENA["width"])
    inner = int(HYENA["inner"])
    filt = int(HYENA["filter"])
    positional = int(HYENA["positional"])
    maximum = int(HYENA["maximum"])
    short = int(HYENA["short"])
    result = [
        (
            "hyena.backbone.embeddings.word_embeddings.weight",
            "F32",
            (int(HYENA["embedding_rows"]), width),
        )
    ]  # type: List[Requirement]
    for layer in range(int(HYENA["layers"])):
        prefix = "hyena.backbone.layers.%d." % layer
        result.extend(
            [
                (prefix + "mixer.filter_fn.bias", "F32", (width,)),
                (prefix + "mixer.filter_fn.implicit_filter.0.bias", "F32", (filt,)),
                (prefix + "mixer.filter_fn.implicit_filter.0.weight", "F32", (filt, positional)),
                (prefix + "mixer.filter_fn.implicit_filter.1.freq", "F32", (1, filt)),
                (prefix + "mixer.filter_fn.implicit_filter.2.bias", "F32", (filt,)),
                (prefix + "mixer.filter_fn.implicit_filter.2.weight", "F32", (filt, filt)),
                (prefix + "mixer.filter_fn.implicit_filter.4.bias", "F32", (filt,)),
                (prefix + "mixer.filter_fn.implicit_filter.4.weight", "F32", (filt, filt)),
                (prefix + "mixer.filter_fn.implicit_filter.6.weight", "F32", (width, filt)),
                (prefix + "mixer.filter_fn.modulation.deltas", "F32", (1, 1, width)),
                (prefix + "mixer.filter_fn.pos_emb.t", "F32", (1, maximum, 1)),
                (prefix + "mixer.filter_fn.pos_emb.z", "F32", (1, maximum, positional)),
                (prefix + "mixer.in_proj.bias", "F32", (width * 3,)),
                (prefix + "mixer.in_proj.weight", "F32", (width * 3, width)),
                (prefix + "mixer.out_proj.bias", "F32", (width,)),
                (prefix + "mixer.out_proj.weight", "F32", (width, width)),
                (prefix + "mixer.short_filter.bias", "F32", (width * 3,)),
                (prefix + "mixer.short_filter.weight", "F32", (width * 3, 1, short)),
                (prefix + "mlp.fc1.bias", "F32", (inner,)),
                (prefix + "mlp.fc1.weight", "F32", (inner, width)),
                (prefix + "mlp.fc2.bias", "F32", (width,)),
                (prefix + "mlp.fc2.weight", "F32", (width, inner)),
                (prefix + "norm1.bias", "F32", (width,)),
                (prefix + "norm1.weight", "F32", (width,)),
                (prefix + "norm2.bias", "F32", (width,)),
                (prefix + "norm2.weight", "F32", (width,)),
            ]
        )
    result.extend(
        [
            ("hyena.backbone.ln_f.bias", "F32", (width,)),
            ("hyena.backbone.ln_f.weight", "F32", (width,)),
            ("lm_head.weight", "F32", (int(HYENA["embedding_rows"]), width)),
        ]
    )
    return result


def evo_requirements() -> List[Requirement]:
    width = int(EVO["width"])
    inner = int(EVO["inner"])
    state = int(EVO["state"])
    short = int(EVO["short"])
    attention = set(EVO["attention"])
    result = []  # type: List[Requirement]
    for layer in range(int(EVO["layers"])):
        prefix = "backbone.blocks.%d." % layer
        if layer in attention:
            result.extend(
                [
                    (prefix + "inner_mha_cls.Wqkv.bias", "BF16", (width * 3,)),
                    (prefix + "inner_mha_cls.Wqkv.weight", "BF16", (width * 3, width)),
                    (prefix + "inner_mha_cls.out_proj.bias", "BF16", (width,)),
                    (prefix + "inner_mha_cls.out_proj.weight", "BF16", (width, width)),
                    (prefix + "inner_mha_cls.rotary_emb.inv_freq", "BF16", (width // int(EVO["heads"]) // 2,)),
                ]
            )
        else:
            result.extend(
                [
                    (prefix + "filter.D", "BF16", (width,)),
                    (prefix + "filter.poles", "F32", (width, state, 1, 2)),
                    (prefix + "filter.residues", "F32", (width, state, 1, 2)),
                    (prefix + "filter.short_filter_bias", "BF16", (width * 3,)),
                    (prefix + "filter.short_filter_weight", "BF16", (width * 3, 1, short)),
                    (prefix + "out_filter_dense.bias", "BF16", (width,)),
                    (prefix + "out_filter_dense.weight", "BF16", (width, width)),
                    (prefix + "projections.bias", "BF16", (width * 3,)),
                    (prefix + "projections.weight", "BF16", (width * 3, width)),
                ]
            )
        result.extend(
            [
                (prefix + "mlp.l1.weight", "BF16", (inner, width)),
                (prefix + "mlp.l2.weight", "BF16", (inner, width)),
                (prefix + "mlp.l3.weight", "BF16", (width, inner)),
                (prefix + "post_norm.scale", "BF16", (width,)),
                (prefix + "pre_norm.scale", "BF16", (width,)),
            ]
        )
    result.extend(
        [
            ("backbone.embedding_layer.weight", "BF16", (int(EVO["vocab"]), width)),
            ("backbone.norm.scale", "BF16", (width,)),
        ]
    )
    return result


def bf16(value: np.float32) -> np.float32:
    scalar = np.asarray([np.float32(value)], dtype=np.float32)
    bits = scalar.view(np.uint32)[0]
    if bits & np.uint32(0x7F800000) != np.uint32(0x7F800000):
        bits = np.uint32(bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
    upper = np.asarray([np.uint32(bits & np.uint32(0xFFFF0000))], dtype=np.uint32)
    return upper.view(np.float32)[0]


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> np.float32:
    integer = ((tensor_index + 5) * 17 + (element_index + 3) * 11) % 37 - 18
    value = np.float32(np.float32(integer) / np.float32(41.0))
    if name == "hyena.backbone.embeddings.word_embeddings.weight":
        row = element_index // 4
        column = element_index % 4
        if row == 4:
            return np.float32(np.float32(0.35) - np.float32(column) * np.float32(0.11))
        return np.float32(value * np.float32(0.24))
    if "hyena.backbone.layers" in name:
        if "norm" in name:
            return np.float32(1.0) if "bias" not in name else np.float32(0.0)
        if "mixer.in_proj.weight" in name:
            return np.float32(0.55) if element_index % 4 == (element_index // 4) % 4 else np.float32(0.0)
        if any(
            piece in name
            for piece in (
                "mixer.in_proj.bias",
                "mixer.short_filter.bias",
                "mixer.out_proj.bias",
                "mlp.fc1.bias",
                "mlp.fc2.bias",
            )
        ):
            return np.float32(0.0)
        if "mixer.short_filter.weight" in name:
            return np.float32(0.35) if element_index % 2 == 0 else np.float32(0.65)
        if "mixer.out_proj.weight" in name:
            return np.float32(0.45) if element_index % 4 == element_index // 4 else np.float32(0.0)
        if "mixer.filter_fn.bias" in name:
            return np.float32(0.4)
    if "norm" in name and "bias" not in name:
        return np.float32(np.float32(1.0) + value * np.float32(0.03))
    if name == "backbone.norm.scale" or ".scale" in name:
        return np.float32(np.float32(1.0) + value * np.float32(0.03))
    if "filter.poles" in name:
        if element_index % 2 == 0:
            return np.float32(np.float32(0.72) + value * np.float32(0.02))
        return np.float32(np.float32(0.04) + value * np.float32(0.01))
    if "filter.residues" in name:
        return np.float32(value * np.float32(0.08))
    if "modulation.deltas" in name:
        return np.float32(np.float32(-1.1) + value * np.float32(0.1))
    if "implicit_filter.1.freq" in name:
        return np.float32(np.float32(1.7) + value * np.float32(0.05))
    if "pos_emb.t" in name:
        return np.float32(np.float32(element_index) / np.float32(3.0))
    if "pos_emb.z" in name:
        row = element_index // 3
        column = element_index % 3
        if column == 0:
            return np.float32(np.float32(row) / np.float32(3.0))
        return np.float32(math.cos(float(row))) if column == 1 else np.float32(math.sin(float(row)))
    if "embedding" in name:
        return np.float32(value * np.float32(0.24))
    return np.float32(value * np.float32(0.09))


def fixture_weights(requirements: Sequence[Requirement]) -> Weights:
    result = {}  # type: Weights
    for tensor_index, (name, dtype, shape) in enumerate(requirements):
        values = []  # type: List[np.float32]
        for element_index in range(product(shape)):
            value = fixture_scalar(name, tensor_index, element_index)
            values.append(bf16(value) if dtype == "BF16" else value)
        result[name] = np.asarray(values, dtype=np.float32).reshape(shape)
    return result


def linear(values: np.ndarray, weight: np.ndarray, bias: np.ndarray = None, round_output: bool = False) -> np.ndarray:
    rows, input_width = values.shape
    output_width = weight.shape[0]
    result = np.zeros((rows, output_width), dtype=np.float32)
    for row in range(rows):
        for target in range(output_width):
            total = np.float32(0.0) if bias is None else np.float32(bias[target])
            for source in range(input_width):
                total = np.float32(total + np.float32(values[row, source] * weight[target, source]))
            result[row, target] = bf16(total) if round_output else total
    return result


def layer_norm(values: np.ndarray, scale: np.ndarray, bias: np.ndarray, epsilon: np.float32) -> np.ndarray:
    rows, width = values.shape
    result = np.zeros_like(values)
    for row in range(rows):
        total = np.float32(0.0)
        for column in range(width):
            total = np.float32(total + values[row, column])
        mean = np.float32(total / np.float32(width))
        squares = np.float32(0.0)
        for column in range(width):
            centered = np.float32(values[row, column] - mean)
            squares = np.float32(squares + np.float32(centered * centered))
        inverse = np.float32(np.float32(1.0) / np.sqrt(np.float32(squares / np.float32(width) + epsilon)))
        for column in range(width):
            result[row, column] = np.float32(
                np.float32(np.float32(values[row, column] - mean) * inverse) * scale[column] + bias[column]
            )
    return result


def direct_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    rows = signal.shape[0]
    output = np.zeros(rows, dtype=np.float32)
    for row in range(rows):
        total = np.float32(0.0)
        for source in range(row + 1):
            total = np.float32(total + np.float32(signal[source] * kernel[row - source]))
        output[row] = total
    return output


def hyena_mixer(values: np.ndarray, weights: Mapping[str, np.ndarray]) -> np.ndarray:
    width = int(HYENA["width"])
    rows = values.shape[0]
    prefix = "hyena.backbone.layers.0."
    projected = linear(values, weights[prefix + "mixer.in_proj.weight"], weights[prefix + "mixer.in_proj.bias"])
    channels = width * 3
    filtered = np.zeros_like(projected)
    for row in range(rows):
        for channel in range(channels):
            total = np.float32(weights[prefix + "mixer.short_filter.bias"][channel])
            for tap in range(int(HYENA["short"])):
                source = row + tap - (int(HYENA["short"]) - 1)
                if source >= 0:
                    total = np.float32(
                        total
                        + np.float32(
                            projected[source, channel]
                            * weights[prefix + "mixer.short_filter.weight"][channel, 0, tap]
                        )
                    )
            filtered[row, channel] = total

    position = weights[prefix + "mixer.filter_fn.pos_emb.z"][0, :rows, :]
    hidden = linear(
        position,
        weights[prefix + "mixer.filter_fn.implicit_filter.0.weight"],
        weights[prefix + "mixer.filter_fn.implicit_filter.0.bias"],
    )
    frequency = weights[prefix + "mixer.filter_fn.implicit_filter.1.freq"].reshape(-1)
    for row in range(rows):
        for column in range(int(HYENA["filter"])):
            hidden[row, column] = np.float32(np.sin(np.float32(frequency[column] * hidden[row, column])))
    hidden = linear(
        hidden,
        weights[prefix + "mixer.filter_fn.implicit_filter.2.weight"],
        weights[prefix + "mixer.filter_fn.implicit_filter.2.bias"],
    )
    for row in range(rows):
        for column in range(int(HYENA["filter"])):
            hidden[row, column] = np.float32(np.sin(np.float32(frequency[column] * hidden[row, column])))
    hidden = linear(
        hidden,
        weights[prefix + "mixer.filter_fn.implicit_filter.4.weight"],
        weights[prefix + "mixer.filter_fn.implicit_filter.4.bias"],
    )
    for row in range(rows):
        for column in range(int(HYENA["filter"])):
            hidden[row, column] = np.float32(np.sin(np.float32(frequency[column] * hidden[row, column])))
    kernel = linear(hidden, weights[prefix + "mixer.filter_fn.implicit_filter.6.weight"])
    times = weights[prefix + "mixer.filter_fn.pos_emb.t"].reshape(-1)
    deltas = weights[prefix + "mixer.filter_fn.modulation.deltas"].reshape(-1)
    for row in range(rows):
        for channel in range(width):
            decay = np.float32(np.exp(np.float32(-times[row] * np.abs(deltas[channel]))) + np.float32(0.05))
            kernel[row, channel] = np.float32(kernel[row, channel] * decay)

    mixed = np.zeros((rows, width), dtype=np.float32)
    diagonal = weights[prefix + "mixer.filter_fn.bias"]
    for channel in range(width):
        signal = np.asarray(
            [np.float32(filtered[row, width + channel] * filtered[row, width * 2 + channel]) for row in range(rows)],
            dtype=np.float32,
        )
        convolved = direct_convolution(signal, kernel[:, channel])
        for row in range(rows):
            mixed[row, channel] = np.float32(
                filtered[row, channel]
                * np.float32(convolved[row] + np.float32(signal[row] * diagonal[channel]))
            )
    return linear(mixed, weights[prefix + "mixer.out_proj.weight"], weights[prefix + "mixer.out_proj.bias"])


def hyena_forward(tokens: Sequence[int], mask: Sequence[int], weights: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    hidden = weights["hyena.backbone.embeddings.word_embeddings.weight"][list(tokens), :].copy()
    prefix = "hyena.backbone.layers.0."
    normalized = layer_norm(hidden, weights[prefix + "norm1.weight"], weights[prefix + "norm1.bias"], HYENA["epsilon"])
    hidden = np.asarray(hidden + hyena_mixer(normalized, weights), dtype=np.float32)
    normalized = layer_norm(hidden, weights[prefix + "norm2.weight"], weights[prefix + "norm2.bias"], HYENA["epsilon"])
    feed_forward = linear(normalized, weights[prefix + "mlp.fc1.weight"], weights[prefix + "mlp.fc1.bias"])
    flat = feed_forward.reshape(-1)
    scale = np.float32(0.7978845608028654)
    for index in range(flat.shape[0]):
        value = flat[index]
        cubic = np.float32(np.float32(value * value) * value)
        inner = np.float32(scale * np.float32(value + np.float32(0.044715) * cubic))
        flat[index] = np.float32(np.float32(0.5) * value * np.float32(np.float32(1.0) + np.tanh(inner)))
    hidden = np.asarray(
        hidden + linear(feed_forward, weights[prefix + "mlp.fc2.weight"], weights[prefix + "mlp.fc2.bias"]),
        dtype=np.float32,
    )
    final = layer_norm(
        hidden,
        weights["hyena.backbone.ln_f.weight"],
        weights["hyena.backbone.ln_f.bias"],
        HYENA["epsilon"],
    )
    pooled = np.zeros(int(HYENA["width"]), dtype=np.float32)
    count = 0
    for row, keep in enumerate(mask):
        if keep:
            count += 1
            for column in range(int(HYENA["width"])):
                pooled[column] = np.float32(pooled[column] + final[row, column])
    inverse = np.float32(np.float32(1.0) / np.float32(count))
    for column in range(pooled.shape[0]):
        pooled[column] = np.float32(pooled[column] * inverse)
    return final, pooled


def rms_bf16(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    rows, width = values.shape
    result = np.zeros_like(values)
    for row in range(rows):
        squares = np.float32(0.0)
        for column in range(width):
            squares = np.float32(squares + np.float32(values[row, column] * values[row, column]))
        denominator = np.float32(np.sqrt(np.float32(squares / np.float32(width))) + EVO["epsilon"])
        for column in range(width):
            result[row, column] = bf16(np.float32(np.float32(values[row, column] / denominator) * scale[column]))
    return result


def add_bf16(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty_like(left)
    for index in range(left.size):
        result.reshape(-1)[index] = bf16(np.float32(left.reshape(-1)[index] + right.reshape(-1)[index]))
    return result


def evo_mlp(values: np.ndarray, weights: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    first = linear(values, weights[prefix + "mlp.l1.weight"], round_output=True)
    second = linear(values, weights[prefix + "mlp.l2.weight"], round_output=True)
    for index in range(first.size):
        value = first.reshape(-1)[index]
        activated = bf16(
            np.float32(
                np.float32(0.5)
                * value
                * np.float32(np.float32(1.0) + np.float32(math.erf(float(np.float32(value * np.float32(0.7071067811865475))))))
            )
        )
        first.reshape(-1)[index] = bf16(np.float32(activated * second.reshape(-1)[index]))
    return linear(first, weights[prefix + "mlp.l3.weight"], round_output=True)


def evo_hyena_block(hidden: np.ndarray, weights: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    rows = hidden.shape[0]
    width = int(EVO["width"])
    heads = int(EVO["heads"])
    normalized = rms_bf16(hidden, weights[prefix + "pre_norm.scale"])
    projected = linear(
        normalized,
        weights[prefix + "projections.weight"],
        weights[prefix + "projections.bias"],
        True,
    )
    channels = width * 3
    short_output = np.zeros_like(projected)
    for row in range(rows):
        for channel in range(channels):
            total = np.float32(weights[prefix + "filter.short_filter_bias"][channel])
            for tap in range(int(EVO["short"])):
                source = row + tap - (int(EVO["short"]) - 1)
                if source >= 0:
                    total = np.float32(
                        total
                        + np.float32(
                            projected[source, channel]
                            * weights[prefix + "filter.short_filter_weight"][channel, 0, tap]
                        )
                    )
            short_output[row, channel] = bf16(total)
    mixed = np.zeros((rows, width), dtype=np.float32)
    head_width = width // heads
    poles = weights[prefix + "filter.poles"]
    residues = weights[prefix + "filter.residues"]
    diagonal = weights[prefix + "filter.D"]
    for channel in range(width):
        head = channel // head_width
        within = channel % head_width
        x2_channel = head * head_width * 3 + within
        x1_channel = x2_channel + head_width
        value_channel = x1_channel + head_width
        signal = np.asarray(
            [bf16(np.float32(short_output[row, x1_channel] * short_output[row, value_channel])) for row in range(rows)],
            dtype=np.float32,
        )
        kernel = np.zeros(rows, dtype=np.float32)
        for row in range(rows):
            total = np.complex64(0.0 + 0.0j)
            for state in range(int(EVO["state"])):
                pole = np.complex64(complex(float(poles[channel, state, 0, 0]), float(poles[channel, state, 0, 1])))
                residue = np.complex64(complex(float(residues[channel, state, 0, 0]), float(residues[channel, state, 0, 1])))
                power = np.complex64(np.exp(np.complex64(np.log(pole) * np.float32(row))))
                total = np.complex64(total + np.complex64(residue * power))
            kernel[row] = np.float32(total.real)
        convolved = direct_convolution(signal, kernel)
        for row in range(rows):
            filtered = bf16(np.float32(convolved[row] + bf16(np.float32(signal[row] * diagonal[channel]))))
            mixed[row, channel] = bf16(np.float32(filtered * short_output[row, x2_channel]))
    projected_output = linear(
        mixed,
        weights[prefix + "out_filter_dense.weight"],
        weights[prefix + "out_filter_dense.bias"],
        True,
    )
    hidden = add_bf16(hidden, projected_output)
    normalized = rms_bf16(hidden, weights[prefix + "post_norm.scale"])
    return add_bf16(hidden, evo_mlp(normalized, weights, prefix))


def apply_rope(query: np.ndarray, key: np.ndarray) -> None:
    rows = query.shape[0]
    width = int(EVO["width"])
    heads = int(EVO["heads"])
    head_width = width // heads
    half = head_width // 2
    for row in range(rows):
        for pair in range(half):
            exponent = np.float32(np.float32(-2.0) * np.float32(pair) / np.float32(head_width))
            inverse = np.float32(np.power(EVO["rope_theta"], exponent))
            angle = np.float32(np.float32(np.float32(row) / EVO["rope_scale"]) * inverse)
            cosine = bf16(np.float32(np.cos(angle)))
            sine = bf16(np.float32(np.sin(angle)))
            for head in range(heads):
                first = head * head_width + pair
                second = first + half
                q0, q1 = query[row, first], query[row, second]
                k0, k1 = key[row, first], key[row, second]
                query[row, first] = bf16(np.float32(np.float32(q0 * cosine) - np.float32(q1 * sine)))
                query[row, second] = bf16(np.float32(np.float32(q1 * cosine) + np.float32(q0 * sine)))
                key[row, first] = bf16(np.float32(np.float32(k0 * cosine) - np.float32(k1 * sine)))
                key[row, second] = bf16(np.float32(np.float32(k1 * cosine) + np.float32(k0 * sine)))


def causal_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    rows = query.shape[0]
    width = int(EVO["width"])
    heads = int(EVO["heads"])
    head_width = width // heads
    scale = np.float32(np.float32(1.0) / np.sqrt(np.float32(head_width)))
    output = np.zeros_like(query)
    for row in range(rows):
        for head in range(heads):
            logits = []  # type: List[np.float32]
            maximum = np.float32(-np.inf)
            for source in range(row + 1):
                total = np.float32(0.0)
                for column in range(head_width):
                    index = head * head_width + column
                    total = np.float32(total + np.float32(query[row, index] * key[source, index]))
                score = np.float32(total * scale)
                logits.append(score)
                maximum = np.maximum(maximum, score)
            denominator = np.float32(0.0)
            for source in range(len(logits)):
                logits[source] = np.float32(np.exp(np.float32(logits[source] - maximum)))
                denominator = np.float32(denominator + logits[source])
            for column in range(head_width):
                total = np.float32(0.0)
                index = head * head_width + column
                for source in range(row + 1):
                    total = np.float32(total + np.float32(np.float32(logits[source] / denominator) * value[source, index]))
                output[row, index] = bf16(total)
    return output


def evo_attention_block(hidden: np.ndarray, weights: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    width = int(EVO["width"])
    normalized = rms_bf16(hidden, weights[prefix + "pre_norm.scale"])
    qkv = linear(
        normalized,
        weights[prefix + "inner_mha_cls.Wqkv.weight"],
        weights[prefix + "inner_mha_cls.Wqkv.bias"],
        True,
    )
    query = qkv[:, :width].copy()
    key = qkv[:, width : width * 2].copy()
    value = qkv[:, width * 2 :].copy()
    apply_rope(query, key)
    attended = causal_attention(query, key, value)
    projected = linear(
        attended,
        weights[prefix + "inner_mha_cls.out_proj.weight"],
        weights[prefix + "inner_mha_cls.out_proj.bias"],
        True,
    )
    hidden = add_bf16(hidden, projected)
    normalized = rms_bf16(hidden, weights[prefix + "post_norm.scale"])
    return add_bf16(hidden, evo_mlp(normalized, weights, prefix))


def evo_forward(tokens: Sequence[int], weights: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    hidden = weights["backbone.embedding_layer.weight"][list(tokens), :].copy()
    for layer in range(int(EVO["layers"])):
        prefix = "backbone.blocks.%d." % layer
        if layer in set(EVO["attention"]):
            hidden = evo_attention_block(hidden, weights, prefix)
        else:
            hidden = evo_hyena_block(hidden, weights, prefix)
    final = rms_bf16(hidden, weights["backbone.norm.scale"])
    pooled = np.zeros(int(EVO["width"]), dtype=np.float32)
    for row in range(final.shape[0]):
        for column in range(final.shape[1]):
            pooled[column] = np.float32(pooled[column] + final[row, column])
    inverse = np.float32(np.float32(1.0) / np.float32(final.shape[0]))
    for column in range(pooled.shape[0]):
        pooled[column] = np.float32(pooled[column] * inverse)
    return final, pooled


def values(array: np.ndarray) -> List[float]:
    return [float(value) for value in array.reshape(-1)]


def generate() -> Dict[str, object]:
    hyena_weights = fixture_weights(hyena_requirements())
    evo_weights = fixture_weights(evo_requirements())
    reference_final, reference_pooled = hyena_forward((4, 7, 8), (0, 1, 1), hyena_weights)
    normalized_final, normalized_pooled = hyena_forward((7, 8), (1, 1), hyena_weights)
    evo_final, evo_pooled = evo_forward((1, 2, 3), evo_weights)
    return {
        "vectors": {
            "hyena.reference.final": values(reference_final),
            "hyena.reference.pooled": values(reference_pooled),
            "hyena.normalized.final": values(normalized_final),
            "hyena.normalized.pooled": values(normalized_pooled),
            "evo.final": values(evo_final),
            "evo.pooled": values(evo_pooled),
        },
        "work": {"evo_attention_pairs": 12, "evo_peak_logits": 3},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(generate(), sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
