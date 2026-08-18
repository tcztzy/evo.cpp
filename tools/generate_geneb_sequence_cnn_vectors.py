#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate independent NumPy-F32 Enformer/SPACE tiny oracle vectors."""

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
    print("NumPy is required for the GENEB sequence-CNN oracle", file=sys.stderr)
    raise SystemExit(77)


Shape = Tuple[int, ...]
Requirement = Tuple[str, Shape]


def topology(variant: str) -> Dict[str, object]:
    result: Dict[str, object] = {
        "variant": variant,
        "input_length": 8,
        "stem_width": 4,
        "tower_widths": [4],
        "width": 4,
        "output_width": 6,
        "layers": 1,
        "heads": 2,
        "key_dimension": 2,
        "value_dimension": 2,
        "relative_feature_width": 6,
        "target_length": 2,
        "epsilon": np.float32(1.0e-5),
        "gelu_scale": np.float32(1.702),
        "experts": 0,
        "top_k": 0,
        "gate_slope": np.float32(0.0),
        "species": "",
    }
    if variant == "space":
        result.update(
            {
                "experts": 4,
                "top_k": 3,
                "gate_slope": np.float32(0.01),
                "species": "human",
            }
        )
    return result


def add_norm(result: List[Requirement], prefix: str, width: int) -> None:
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        result.append((prefix + suffix, (width,)))


def requirements(topo: Mapping[str, object]) -> List[Requirement]:
    stem = int(topo["stem_width"])
    width = int(topo["width"])
    result: List[Requirement] = [
        ("stem.0.weight", (stem, 4, 15)),
        ("stem.0.bias", (stem,)),
    ]
    add_norm(result, "stem.1.fn.0.", stem)
    result.extend(
        [
            ("stem.1.fn.2.weight", (stem, stem, 1)),
            ("stem.1.fn.2.bias", (stem,)),
            ("stem.2.to_attn_logits.weight", (stem, stem, 1, 1)),
        ]
    )
    previous = stem
    for index, raw_width in enumerate(topo["tower_widths"]):
        tower_width = int(raw_width)
        prefix = "conv_tower.{}.".format(index)
        add_norm(result, prefix + "0.0.", previous)
        result.extend(
            [
                (prefix + "0.2.weight", (tower_width, previous, 5)),
                (prefix + "0.2.bias", (tower_width,)),
            ]
        )
        add_norm(result, prefix + "1.fn.0.", tower_width)
        result.extend(
            [
                (prefix + "1.fn.2.weight", (tower_width, tower_width, 1)),
                (prefix + "1.fn.2.bias", (tower_width,)),
                (
                    prefix + "2.to_attn_logits.weight",
                    (tower_width, tower_width, 1, 1),
                ),
            ]
        )
        previous = tower_width
    variant = str(topo["variant"])
    if variant == "space":
        result.extend(
            [
                ("transformer.species_embedding.human", (1, 1, width)),
                ("transformer.species_embedding.mouse", (1, 1, width)),
            ]
        )
    for layer in range(int(topo["layers"])):
        prefix = (
            "transformer.{}.".format(layer)
            if variant == "enformer"
            else "transformer.transformer.{}.".format(layer)
        )
        attention = "0.fn." if variant == "enformer" else "attention.fn."
        result.extend(
            [
                (prefix + attention + "0.weight", (width,)),
                (prefix + attention + "0.bias", (width,)),
                (
                    prefix + attention + "1.rel_content_bias",
                    (1, int(topo["heads"]), 1, int(topo["key_dimension"])),
                ),
                (
                    prefix + attention + "1.rel_pos_bias",
                    (1, int(topo["heads"]), 1, int(topo["key_dimension"])),
                ),
                (
                    prefix + attention + "1.to_q.weight",
                    (int(topo["heads"]) * int(topo["key_dimension"]), width),
                ),
                (
                    prefix + attention + "1.to_k.weight",
                    (int(topo["heads"]) * int(topo["key_dimension"]), width),
                ),
                (prefix + attention + "1.to_v.weight", (width, width)),
                (prefix + attention + "1.to_out.weight", (width, width)),
                (prefix + attention + "1.to_out.bias", (width,)),
                (
                    prefix + attention + "1.to_rel_k.weight",
                    (
                        int(topo["heads"]) * int(topo["key_dimension"]),
                        int(topo["relative_feature_width"]),
                    ),
                ),
            ]
        )
        if variant == "enformer":
            result.extend(
                [
                    (prefix + "1.fn.0.weight", (width,)),
                    (prefix + "1.fn.0.bias", (width,)),
                    (prefix + "1.fn.1.weight", (width * 2, width)),
                    (prefix + "1.fn.1.bias", (width * 2,)),
                    (prefix + "1.fn.4.weight", (width, width * 2)),
                    (prefix + "1.fn.4.bias", (width,)),
                ]
            )
        else:
            feed = prefix + "feed_forward."
            for species in ("human", "mouse"):
                result.extend(
                    [
                        (
                            feed + "gates." + species + ".0.weight",
                            (int(topo["experts"]), width),
                        ),
                        (
                            feed + "gates." + species + ".0.bias",
                            (int(topo["experts"]),),
                        ),
                    ]
                )
            result.extend(
                [
                    (feed + "layer_norm.weight", (width,)),
                    (feed + "layer_norm.bias", (width,)),
                    (feed + "input.weight", (int(topo["experts"]), width, width * 2)),
                    (feed + "input.bias", (int(topo["experts"]), width * 2)),
                    (feed + "output.weight", (int(topo["experts"]), width * 2, width)),
                    (feed + "output.bias", (int(topo["experts"]), width)),
                ]
            )
    add_norm(result, "final_pointwise.1.0.", width)
    result.extend(
        [
            ("final_pointwise.1.2.weight", (int(topo["output_width"]), width, 1)),
            ("final_pointwise.1.2.bias", (int(topo["output_width"]),)),
        ]
    )
    return result


def fixture_scalar(name: str, shape: Shape, tensor_index: int, element_index: int) -> np.float32:
    integer = ((tensor_index + 7) * 11 + (element_index + 3) * 5) % 29 - 14
    base = np.float32(integer / 100.0)
    if name.endswith("running_var"):
        return np.float32(1.0) + base * np.float32(0.1)
    if name.endswith("running_mean"):
        return base * np.float32(0.05)
    if len(shape) == 1 and name.endswith(".weight"):
        return np.float32(1.0) + base * np.float32(0.1)
    if name.endswith(".bias"):
        return base * np.float32(0.08)
    if "species_embedding" in name:
        return base * np.float32(0.3)
    if "gates." in name:
        return base * np.float32(0.4)
    return base * np.float32(0.2)


def weights(topo: Mapping[str, object]) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for tensor_index, (name, shape) in enumerate(requirements(topo)):
        result[name] = np.asarray(
            [
                fixture_scalar(name, shape, tensor_index, index)
                for index in range(math.prod(shape))
            ],
            dtype=np.float32,
        ).reshape(shape)
    return result


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    rows = x.shape[0]
    result = np.empty((rows, weight.shape[0]), dtype=np.float32)
    for row in range(rows):
        for out in range(weight.shape[0]):
            total = np.float32(0.0 if bias is None else bias[out])
            for column in range(weight.shape[1]):
                total = np.float32(total + np.float32(x[row, column] * weight[out, column]))
            result[row, out] = total
    return result


def layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, epsilon: np.float32) -> np.ndarray:
    result = np.empty_like(x)
    for row in range(x.shape[0]):
        mean = sum(float(value) for value in x[row]) / x.shape[1]
        variance = sum((float(value) - mean) ** 2 for value in x[row]) / x.shape[1]
        inverse = np.float32(1.0) / np.sqrt(np.float32(variance) + epsilon, dtype=np.float32)
        float_mean = np.float32(mean)
        for column in range(x.shape[1]):
            normalized = np.float32(np.float32(x[row, column] - float_mean) * inverse)
            result[row, column] = np.float32(normalized * scale[column] + bias[column])
    return result


def batch_norm(
    x: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    epsilon: np.float32,
) -> np.ndarray:
    result = np.empty_like(x)
    for column in range(x.shape[1]):
        multiplier = np.float32(scale[column] / np.sqrt(np.float32(running_var[column] + epsilon), dtype=np.float32))
        for row in range(x.shape[0]):
            result[row, column] = np.float32(
                np.float32(x[row, column] - running_mean[column]) * multiplier + bias[column]
            )
    return result


def convolution(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    rows, input_width = x.shape
    output_width, _, kernel = weight.shape
    padding = kernel // 2
    result = np.empty((rows, output_width), dtype=np.float32)
    for row in range(rows):
        for out in range(output_width):
            total = np.float32(bias[out])
            for column in range(input_width):
                for tap in range(kernel):
                    source = row + tap - padding
                    if 0 <= source < rows:
                        total = np.float32(
                            total + np.float32(x[source, column] * weight[out, column, tap])
                        )
            result[row, out] = total
    return result


def sigmoid_gelu(x: np.ndarray, scale: np.float32) -> np.ndarray:
    flat = x.reshape(-1)
    result = np.empty_like(flat)
    for index, value in enumerate(flat):
        result[index] = np.float32(value / np.float32(1.0 + math.exp(float(-scale * value))))
    return result.reshape(x.shape)


def attention_pool(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rows, width = x.shape
    result = np.empty(((rows + 1) // 2, width), dtype=np.float32)
    for pooled in range(result.shape[0]):
        first = pooled * 2
        if first + 1 == rows:
            result[pooled] = x[first]
            continue
        for out in range(width):
            first_logit = np.float32(0.0)
            second_logit = np.float32(0.0)
            for column in range(width):
                coefficient = weight[out, column, 0, 0]
                first_logit = np.float32(first_logit + np.float32(x[first, column] * coefficient))
                second_logit = np.float32(second_logit + np.float32(x[first + 1, column] * coefficient))
            maximum = max(first_logit, second_logit)
            first_exp = np.float32(math.exp(float(first_logit - maximum)))
            second_exp = np.float32(math.exp(float(second_logit - maximum)))
            probability = np.float32(first_exp / np.float32(first_exp + second_exp))
            result[pooled, out] = np.float32(
                x[first, out] * probability
                + x[first + 1, out] * np.float32(1.0 - probability)
            )
    return result


def relative_features(rows: int, feature_width: int) -> np.ndarray:
    basis = feature_width // 6
    result = np.zeros((rows * 2 - 1, feature_width), dtype=np.float32)
    maximum_exponent = math.log2(rows)
    standard_deviation = rows / (2.0 * basis)
    for index in range(rows * 2 - 1):
        signed_distance = index - (rows - 1)
        distance = abs(float(signed_distance))
        gamma_values: List[float] = []
        for component in range(basis):
            fraction = 0.0 if basis == 1 else component / (basis - 1)
            half_life = 2.0 ** (3.0 + (maximum_exponent - 3.0) * fraction)
            result[index, component] = np.float32(
                math.exp(-math.log(2.0) * distance / half_life)
            )
            center_width = 2.0 ** (component + 1.0) - 1.0
            result[index, basis + component] = np.float32(center_width > distance)
            start_mean = rows / basis
            mean = start_mean if basis == 1 else start_mean + (rows - start_mean) * fraction
            concentration = (mean / standard_deviation) ** 2
            rate = mean / (standard_deviation**2)
            probability = 0.0
            if distance > 0.0:
                log_probability = (
                    (concentration - 1.0) * math.log(distance)
                    - rate * distance
                    - (math.lgamma(concentration) - concentration * math.log(rate))
                )
                probability = math.exp(log_probability)
            gamma_values.append(probability + 1.0e-8)
        gamma_maximum = max(gamma_values)
        for component, value in enumerate(gamma_values):
            result[index, basis * 2 + component] = np.float32(value / gamma_maximum)
        sign = -1.0 if signed_distance < 0 else 1.0 if signed_distance > 0 else 0.0
        for component in range(feature_width // 2):
            result[index, feature_width // 2 + component] = np.float32(
                sign * result[index, component]
            )
    return result


def relative_attention(
    x: np.ndarray, topo: Mapping[str, object], state: Mapping[str, np.ndarray], prefix: str
) -> np.ndarray:
    rows = x.shape[0]
    heads = int(topo["heads"])
    key_dimension = int(topo["key_dimension"])
    value_dimension = int(topo["value_dimension"])
    query = linear(x, state[prefix + "to_q.weight"])
    key = linear(x, state[prefix + "to_k.weight"])
    value = linear(x, state[prefix + "to_v.weight"])
    scale = np.float32(1.0 / math.sqrt(key_dimension))
    query = np.asarray(query * scale, dtype=np.float32)
    relative_key = linear(
        relative_features(rows, int(topo["relative_feature_width"])),
        state[prefix + "to_rel_k.weight"],
    )
    content_bias = state[prefix + "rel_content_bias"].reshape(heads, key_dimension)
    position_bias = state[prefix + "rel_pos_bias"].reshape(heads, key_dimension)
    context = np.zeros((rows, heads, value_dimension), dtype=np.float32)
    for query_row in range(rows):
        for head in range(heads):
            scores: List[np.float32] = []
            for key_row in range(rows):
                relative_row = rows - 1 - query_row + key_row
                total = np.float32(0.0)
                for column in range(key_dimension):
                    q = query[query_row, head * key_dimension + column]
                    total = np.float32(
                        total
                        + np.float32(
                            np.float32(q + content_bias[head, column])
                            * key[key_row, head * key_dimension + column]
                        )
                        + np.float32(
                            np.float32(q + position_bias[head, column])
                            * relative_key[relative_row, head * key_dimension + column]
                        )
                    )
                scores.append(total)
            maximum = max(scores)
            denominator = 0.0
            probabilities: List[np.float32] = []
            for score in scores:
                probability = np.float32(math.exp(float(score - maximum)))
                probabilities.append(probability)
                denominator += float(probability)
            for key_row, probability in enumerate(probabilities):
                for column in range(value_dimension):
                    context[query_row, head, column] = np.float32(
                        context[query_row, head, column]
                        + np.float32(
                            probability * value[key_row, head * value_dimension + column]
                        )
                    )
            context[query_row, head] = np.asarray(
                context[query_row, head] * np.float32(1.0 / denominator), dtype=np.float32
            )
    return linear(
        context.reshape(rows, heads * value_dimension),
        state[prefix + "to_out.weight"],
        state[prefix + "to_out.bias"],
    )


def transform(sequence: str, topo: Mapping[str, object]) -> np.ndarray:
    length = int(topo["input_length"])
    variant = str(topo["variant"])
    result = np.zeros((length, 4), dtype=np.float32)
    source_start = 0
    destination_start = 0
    retained = min(len(sequence), length)
    if variant == "space":
        if len(sequence) > length:
            source_start = (len(sequence) - length) // 2
        else:
            destination_start = (length - len(sequence)) // 2
    for index in range(retained):
        base = sequence[source_start + index].upper()
        if variant == "space" and base == "U":
            base = "T"
        if base in "ACGT":
            result[destination_start + index, "ACGT".index(base)] = np.float32(1.0)
    return result


def norm_from(state: Mapping[str, np.ndarray], prefix: str) -> Tuple[np.ndarray, ...]:
    return tuple(
        state[prefix + suffix]
        for suffix in ("weight", "bias", "running_mean", "running_var")
    )


def oracle_variant(variant: str) -> Dict[str, object]:
    topo = topology(variant)
    state = weights(topo)
    hidden = transform("acguNXTTAA", topo)
    stage = convolution(hidden, state["stem.0.weight"], state["stem.0.bias"])
    normalized = batch_norm(stage, *norm_from(state, "stem.1.fn.0."), np.float32(topo["epsilon"]))
    normalized = sigmoid_gelu(normalized, np.float32(topo["gelu_scale"]))
    residual = convolution(normalized, state["stem.1.fn.2.weight"], state["stem.1.fn.2.bias"])
    hidden = attention_pool(
        np.asarray(stage + residual, dtype=np.float32), state["stem.2.to_attn_logits.weight"]
    )
    width = int(topo["stem_width"])
    for index, raw_width in enumerate(topo["tower_widths"]):
        output_width = int(raw_width)
        prefix = "conv_tower.{}.".format(index)
        normalized = batch_norm(hidden, *norm_from(state, prefix + "0.0."), np.float32(topo["epsilon"]))
        normalized = sigmoid_gelu(normalized, np.float32(topo["gelu_scale"]))
        stage = convolution(normalized, state[prefix + "0.2.weight"], state[prefix + "0.2.bias"])
        normalized = batch_norm(stage, *norm_from(state, prefix + "1.fn.0."), np.float32(topo["epsilon"]))
        normalized = sigmoid_gelu(normalized, np.float32(topo["gelu_scale"]))
        residual = convolution(normalized, state[prefix + "1.fn.2.weight"], state[prefix + "1.fn.2.bias"])
        hidden = attention_pool(
            np.asarray(stage + residual, dtype=np.float32),
            state[prefix + "2.to_attn_logits.weight"],
        )
        width = output_width
    if variant == "space":
        species = state["transformer.species_embedding." + str(topo["species"])].reshape(1, width)
        hidden = np.concatenate((hidden, species), axis=0).astype(np.float32)
    for layer in range(int(topo["layers"])):
        prefix = (
            "transformer.{}.".format(layer)
            if variant == "enformer"
            else "transformer.transformer.{}.".format(layer)
        )
        attention = "0.fn." if variant == "enformer" else "attention.fn."
        attention_prefix = prefix + attention + "1."
        normalized = layer_norm(
            hidden,
            state[prefix + attention + "0.weight"],
            state[prefix + attention + "0.bias"],
            np.float32(topo["epsilon"]),
        )
        hidden = np.asarray(
            hidden + relative_attention(normalized, topo, state, attention_prefix), dtype=np.float32
        )
        if variant == "enformer":
            normalized = layer_norm(
                hidden,
                state[prefix + "1.fn.0.weight"],
                state[prefix + "1.fn.0.bias"],
                np.float32(topo["epsilon"]),
            )
            intermediate = linear(
                normalized,
                state[prefix + "1.fn.1.weight"],
                state[prefix + "1.fn.1.bias"],
            )
            intermediate = np.maximum(intermediate, np.float32(0.0)).astype(np.float32)
            hidden = np.asarray(
                hidden
                + linear(
                    intermediate,
                    state[prefix + "1.fn.4.weight"],
                    state[prefix + "1.fn.4.bias"],
                ),
                dtype=np.float32,
            )
        else:
            feed = prefix + "feed_forward."
            gate = linear(
                hidden,
                state[feed + "gates." + str(topo["species"]) + ".0.weight"],
                state[feed + "gates." + str(topo["species"]) + ".0.bias"],
            )
            gate = np.where(gate < 0, gate * np.float32(topo["gate_slope"]), gate).astype(np.float32)
            normalized = layer_norm(
                hidden,
                state[feed + "layer_norm.weight"],
                state[feed + "layer_norm.bias"],
                np.float32(topo["epsilon"]),
            )
            residual = np.zeros_like(hidden)
            for row in range(hidden.shape[0]):
                selected = sorted(range(int(topo["experts"])), key=lambda expert: float(gate[row, expert]), reverse=True)[: int(topo["top_k"])]
                maximum = max(gate[row, expert] for expert in selected)
                probabilities = [
                    np.float32(math.exp(float(gate[row, expert] - maximum)))
                    for expert in selected
                ]
                denominator = np.float32(0.0)
                for probability in probabilities:
                    denominator = np.float32(denominator + probability)
                probabilities = [np.float32(value / denominator) for value in probabilities]
                for probability, expert in zip(probabilities, selected):
                    intermediate = np.empty(width * 2, dtype=np.float32)
                    for out in range(width * 2):
                        total = np.float32(state[feed + "input.bias"][expert, out])
                        for column in range(width):
                            total = np.float32(
                                total
                                + np.float32(
                                    normalized[row, column]
                                    * state[feed + "input.weight"][expert, column, out]
                                )
                            )
                        intermediate[out] = max(total, np.float32(0.0))
                    for out in range(width):
                        total = np.float32(state[feed + "output.bias"][expert, out])
                        for column in range(width * 2):
                            total = np.float32(
                                total
                                + np.float32(
                                    intermediate[column]
                                    * state[feed + "output.weight"][expert, column, out]
                                )
                            )
                        residual[row, out] = np.float32(
                            residual[row, out] + np.float32(probability * total)
                        )
            hidden = np.asarray(hidden + residual, dtype=np.float32)
    if variant == "space":
        hidden = hidden[:-1]
    target = int(topo["target_length"])
    trim = (hidden.shape[0] - target) // 2
    hidden = hidden[trim : trim + target].copy()
    normalized = batch_norm(
        hidden,
        *norm_from(state, "final_pointwise.1.0."),
        np.float32(topo["epsilon"]),
    )
    normalized = sigmoid_gelu(normalized, np.float32(topo["gelu_scale"]))
    final_hidden = convolution(
        normalized,
        state["final_pointwise.1.2.weight"],
        state["final_pointwise.1.2.bias"],
    )
    final_hidden = sigmoid_gelu(final_hidden, np.float32(topo["gelu_scale"]))
    pooled = np.mean(final_hidden, axis=0, dtype=np.float32)
    return {
        "rows": int(final_hidden.shape[0]),
        "width": int(final_hidden.shape[1]),
        "final_hidden": final_hidden.reshape(-1).tolist(),
        "pooled": pooled.tolist(),
    }


def oracle() -> Dict[str, object]:
    return {
        "schema": "geneb-sequence-cnn-tiny-oracle-v1",
        "generator": "independent-numpy-f32",
        "sequence": "acguNXTTAA",
        "enformer_input_policy": "uppercase-prefix-crop-right-zero-pad-u-invalid",
        "space_input_policy": "uppercase-center-crop-symmetric-n-pad-u-to-t",
        "enformer": oracle_variant("enformer"),
        "space": oracle_variant("space"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(oracle(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
