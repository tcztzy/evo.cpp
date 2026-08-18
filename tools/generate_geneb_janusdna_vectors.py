#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the independent NumPy-F32 tiny JanusDNA oracle.

This file deliberately spells out the two directional Mamba1 paths, three
projected-parameter RMS norms, top-2 MoE, the pinned middle-attention reverse
mask behavior, padded-head FlexAttention, custom fusion, and the Identity
final-MLP residual. It does not import runtime or JanusDNA source code.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    print("NumPy is required for the GENEB JanusDNA oracle", file=sys.stderr)
    raise SystemExit(77)


Topology = Dict[str, Any]
Weights = Dict[str, np.ndarray]


def topology(variant: str) -> Topology:
    return {
        "variant": variant,
        "vocab": 16,
        "tokenizer_vocab": 12,
        "width": 4,
        "layers": 2,
        "heads": 2,
        "head_dim": 2,
        "flex_head_dim": 4,
        "inner": 8,
        "state": 2,
        "conv": 2,
        "dt_rank": 2,
        "mlp": 8,
        "experts": 4,
        "top_k": 2,
        "max_seqlen": 8,
        "mid_layer": 0,
        "epsilon": np.float32(1.0e-6),
    }


def add_mlp(
    result: List[Tuple[str, Tuple[int, ...]]], prefix: str, topo: Mapping[str, Any]
) -> None:
    width, mlp = int(topo["width"]), int(topo["mlp"])
    result.extend(
        [
            (prefix + "gate_proj.weight", (mlp, width)),
            (prefix + "up_proj.weight", (mlp, width)),
            (prefix + "down_proj.weight", (width, mlp)),
        ]
    )


def add_feed_forward(
    result: List[Tuple[str, Tuple[int, ...]]],
    prefix: str,
    expert_layer: bool,
    topo: Mapping[str, Any],
) -> None:
    if not expert_layer:
        add_mlp(result, prefix, topo)
        return
    result.append(
        (prefix + "router.weight", (int(topo["experts"]), int(topo["width"])))
    )
    for expert in range(int(topo["experts"])):
        add_mlp(result, prefix + "experts.{}.".format(expert), topo)


def add_mamba_direction(
    result: List[Tuple[str, Tuple[int, ...]]],
    prefix: str,
    tied: bool,
    topo: Mapping[str, Any],
) -> None:
    width = int(topo["width"])
    inner = int(topo["inner"])
    state = int(topo["state"])
    rank = int(topo["dt_rank"])
    if not tied:
        result.append((prefix + "in_proj.weight", (inner * 2, width)))
    result.extend(
        [
            (prefix + "A_log", (inner, state)),
            (prefix + "D", (inner,)),
            (prefix + "conv1d.weight", (inner, 1, int(topo["conv"]))),
            (prefix + "conv1d.bias", (inner,)),
            (prefix + "x_proj.weight", (rank + state * 2, inner)),
            (prefix + "dt_proj.weight", (inner, rank)),
            (prefix + "dt_proj.bias", (inner,)),
        ]
    )
    if not tied:
        result.append((prefix + "out_proj.weight", (width, inner)))
    result.extend(
        [
            (prefix + "dt_layernorm.weight", (rank,)),
            (prefix + "b_layernorm.weight", (state,)),
            (prefix + "c_layernorm.weight", (state,)),
        ]
    )


def add_mamba_layer(
    result: List[Tuple[str, Tuple[int, ...]]], layer: int, topo: Mapping[str, Any]
) -> None:
    prefix = "layers.{}.mamba_module.".format(layer)
    add_mamba_direction(result, prefix + "mamba_fwd.", False, topo)
    add_mamba_direction(result, prefix + "mamba_rev.", True, topo)
    expert = layer % 2 == 1
    for direction in ("fwd", "bwd"):
        result.extend(
            [
                (
                    prefix + "input_layernorm_{}.weight".format(direction),
                    (int(topo["width"]),),
                ),
                (
                    prefix + "pre_ff_layernorm_{}.weight".format(direction),
                    (int(topo["width"]),),
                ),
            ]
        )
        add_feed_forward(
            result, prefix + "feed_forward_{}.".format(direction), expert, topo
        )


def add_attention_layer(
    result: List[Tuple[str, Tuple[int, ...]]], layer: int, topo: Mapping[str, Any]
) -> None:
    prefix = "layers.{}.attn.".format(layer)
    expert = layer % 2 == 1
    width = int(topo["width"])
    for direction in ("fwd", "bwd"):
        result.extend(
            [
                (prefix + "input_layernorm_{}.weight".format(direction), (width,)),
                (prefix + "pre_ff_layernorm_{}.weight".format(direction), (width,)),
            ]
        )
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            result.append(
                (
                    prefix + "self_attn_{}.{}.weight".format(direction, projection),
                    (width, width),
                )
            )
        add_feed_forward(
            result, prefix + "feed_forward_{}.".format(direction), expert, topo
        )


def requirements(topo: Mapping[str, Any]) -> List[Tuple[str, Tuple[int, ...]]]:
    width = int(topo["width"])
    result = [("embed_tokens.weight", (int(topo["vocab"]), width))]
    for layer in range(int(topo["layers"])):
        if topo["variant"] == "with-middle-attention" and layer == int(
            topo["mid_layer"]
        ):
            add_attention_layer(result, layer, topo)
        else:
            add_mamba_layer(result, layer, topo)
    result.append(("final_layernorm.weight", (width,)))
    for projection in ("q_proj", "k_proj", "v_proj"):
        result.append(
            ("final_attention.self_attn.{}.weight".format(projection), (width, width))
        )
    result.append(("final_attention.self_attn.o_projs.0.weight", (width, width)))
    add_mlp(result, "final_attention.feed_forward.", topo)
    result.extend(
        [
            ("final_attention.input_layernorm.weight", (width,)),
            ("final_attention.pre_ff_layernorm.weight", (width,)),
        ]
    )
    return result


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> np.float32:
    integer = ((tensor_index + 5) * 19 + (element_index + 3) * 11) % 47 - 23
    value = np.float32(np.float32(integer) / np.float32(41.0))
    if "layernorm.weight" in name:
        return np.float32(np.float32(1.0) + value * np.float32(0.025))
    if name.endswith("A_log"):
        return np.float32(np.float32(-0.55) + value * np.float32(0.04))
    if name.endswith("dt_proj.bias"):
        return np.float32(np.float32(-1.0) + value * np.float32(0.07))
    if name.endswith(".D"):
        return np.float32(np.float32(0.45) + value * np.float32(0.05))
    if name == "embed_tokens.weight":
        return np.float32(value * np.float32(0.18))
    if name.endswith("router.weight"):
        return np.float32(value * np.float32(0.09))
    return np.float32(value * np.float32(0.075))


def fixture_weights(topo: Mapping[str, Any]) -> Weights:
    output = {}  # type: Weights
    for tensor_index, (name, shape) in enumerate(requirements(topo)):
        count = 1
        for dimension in shape:
            count *= dimension
        output[name] = np.asarray(
            [fixture_scalar(name, tensor_index, index) for index in range(count)],
            dtype=np.float32,
        ).reshape(shape)
    return output


def linear(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rows, input_width = values.shape
    result = np.zeros((rows, weight.shape[0]), dtype=np.float32)
    for row in range(rows):
        for target in range(weight.shape[0]):
            total = np.float32(0.0)
            for source in range(input_width):
                total = np.float32(
                    total + np.float32(values[row, source] * weight[target, source])
                )
            result[row, target] = total
    return result


def silu(value: np.float32) -> np.float32:
    return np.float32(value / np.float32(np.float32(1.0) + np.exp(np.float32(-value))))


def rms_norm(values: np.ndarray, scale: np.ndarray, epsilon: np.float32) -> np.ndarray:
    rows, width = values.shape
    result = np.zeros_like(values)
    for row in range(rows):
        square_sum = np.float32(0.0)
        for column in range(width):
            square_sum = np.float32(
                square_sum + np.float32(values[row, column] * values[row, column])
            )
        inverse = np.float32(
            np.float32(1.0)
            / np.sqrt(np.float32(square_sum / np.float32(width) + epsilon))
        )
        for column in range(width):
            result[row, column] = np.float32(
                np.float32(values[row, column] * inverse) * scale[column]
            )
    return result


def causal_conv_silu(
    values: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    rows, channels = values.shape
    kernel = weight.shape[-1]
    output = np.zeros_like(values)
    for row in range(rows):
        for channel in range(channels):
            total = np.float32(bias[channel])
            for tap in range(kernel):
                lag = kernel - 1 - tap
                if row >= lag:
                    total = np.float32(
                        total
                        + np.float32(
                            values[row - lag, channel] * weight[channel, 0, tap]
                        )
                    )
            output[row, channel] = silu(total)
    return output


def softplus(value: np.float32) -> np.float32:
    if value > np.float32(20.0):
        return value
    if value < np.float32(-20.0):
        return np.float32(np.exp(value))
    return np.float32(np.log1p(np.exp(value)))


def mamba_mixer(
    values: np.ndarray,
    prefix: str,
    topo: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
    tied_prefix: Optional[str] = None,
) -> np.ndarray:
    projection_prefix = prefix if tied_prefix is None else tied_prefix
    projected = linear(values, weights[projection_prefix + "in_proj.weight"])
    inner = int(topo["inner"])
    x = projected[:, :inner].copy()
    gate = projected[:, inner:].copy()
    x = causal_conv_silu(
        x, weights[prefix + "conv1d.weight"], weights[prefix + "conv1d.bias"]
    )
    parameters = linear(x, weights[prefix + "x_proj.weight"])
    rank, state_width = int(topo["dt_rank"]), int(topo["state"])
    rank_values = rms_norm(
        parameters[:, :rank].copy(),
        weights[prefix + "dt_layernorm.weight"],
        np.float32(1.0e-6),
    )
    b_values = rms_norm(
        parameters[:, rank : rank + state_width].copy(),
        weights[prefix + "b_layernorm.weight"],
        np.float32(1.0e-6),
    )
    c_values = rms_norm(
        parameters[:, rank + state_width :].copy(),
        weights[prefix + "c_layernorm.weight"],
        np.float32(1.0e-6),
    )
    raw_dt = linear(rank_values, weights[prefix + "dt_proj.weight"])
    state = np.zeros((inner, state_width), dtype=np.float32)
    scanned = np.zeros_like(x)
    for row in range(values.shape[0]):
        for channel in range(inner):
            delta = softplus(
                np.float32(
                    raw_dt[row, channel] + weights[prefix + "dt_proj.bias"][channel]
                )
            )
            total = np.float32(0.0)
            for state_index in range(state_width):
                a_value = np.float32(
                    -np.exp(np.float32(weights[prefix + "A_log"][channel, state_index]))
                )
                decay = np.float32(np.exp(np.float32(delta * a_value)))
                state[channel, state_index] = np.float32(
                    np.float32(decay * state[channel, state_index])
                    + np.float32(
                        np.float32(delta * b_values[row, state_index]) * x[row, channel]
                    )
                )
                total = np.float32(
                    total
                    + np.float32(
                        state[channel, state_index] * c_values[row, state_index]
                    )
                )
            total = np.float32(
                total + np.float32(weights[prefix + "D"][channel] * x[row, channel])
            )
            scanned[row, channel] = np.float32(total * silu(gate[row, channel]))
    return linear(scanned, weights[projection_prefix + "out_proj.weight"])


def mlp(
    values: np.ndarray, prefix: str, weights: Mapping[str, np.ndarray]
) -> np.ndarray:
    gate = linear(values, weights[prefix + "gate_proj.weight"])
    up = linear(values, weights[prefix + "up_proj.weight"])
    activated = np.zeros_like(gate)
    for row in range(gate.shape[0]):
        for column in range(gate.shape[1]):
            activated[row, column] = np.float32(
                silu(gate[row, column]) * up[row, column]
            )
    return linear(activated, weights[prefix + "down_proj.weight"])


def top_two(probabilities: Sequence[np.float32]) -> Tuple[int, int]:
    first = 0
    for expert in range(1, len(probabilities)):
        if probabilities[expert] > probabilities[first]:
            first = expert
    second = 1 if first == 0 else 0
    for expert in range(len(probabilities)):
        if expert == first:
            continue
        if probabilities[expert] > probabilities[second] or (
            probabilities[expert] == probabilities[second] and expert < second
        ):
            second = expert
    return first, second


def moe(
    values: np.ndarray,
    prefix: str,
    topo: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    logits = linear(values, weights[prefix + "router.weight"])
    output = np.zeros_like(values)
    for row in range(values.shape[0]):
        maximum = np.max(logits[row])
        probabilities = np.exp(
            np.asarray(logits[row] - maximum, dtype=np.float32)
        ).astype(np.float32)
        denominator = np.float32(0.0)
        for probability in probabilities:
            denominator = np.float32(denominator + probability)
        probabilities = np.asarray(probabilities / denominator, dtype=np.float32)
        selected = top_two(probabilities)
        for expert in selected:
            current = mlp(
                values[row : row + 1], prefix + "experts.{}.".format(expert), weights
            )
            for column in range(values.shape[1]):
                output[row, column] = np.float32(
                    output[row, column]
                    + np.float32(current[0, column] * probabilities[expert])
                )
    return output


def feed_forward(
    values: np.ndarray,
    prefix: str,
    expert: bool,
    topo: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    return (
        moe(values, prefix, topo, weights) if expert else mlp(values, prefix, weights)
    )


def softmax_attention(
    values: np.ndarray,
    prefix: str,
    topo: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    query = linear(values, weights[prefix + "q_proj.weight"])
    key = linear(values, weights[prefix + "k_proj.weight"])
    value = linear(values, weights[prefix + "v_proj.weight"])
    rows, width = values.shape
    heads, head_dim = int(topo["heads"]), int(topo["head_dim"])
    attended = np.zeros_like(values)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    for head in range(heads):
        offset = head * head_dim
        for query_row in range(rows):
            if mask[query_row] == 0:
                continue
            scores = []  # type: List[np.float32]
            keys = []  # type: List[int]
            for key_row in range(query_row + 1):
                if mask[key_row] == 0:
                    continue
                total = np.float32(0.0)
                for column in range(head_dim):
                    total = np.float32(
                        total
                        + np.float32(
                            query[query_row, offset + column]
                            * key[key_row, offset + column]
                        )
                    )
                scores.append(np.float32(total * scale))
                keys.append(key_row)
            maximum = max(scores)
            probabilities = [
                np.float32(np.exp(np.float32(score - maximum))) for score in scores
            ]
            denominator = np.float32(0.0)
            for probability in probabilities:
                denominator = np.float32(denominator + probability)
            for column in range(head_dim):
                total = np.float32(0.0)
                for index, key_row in enumerate(keys):
                    total = np.float32(
                        total
                        + np.float32(
                            np.float32(probabilities[index] / denominator)
                            * value[key_row, offset + column]
                        )
                    )
                attended[query_row, offset + column] = total
    return linear(attended, weights[prefix + "o_proj.weight"])


def flex_visible(query: int, key: int, rows: int) -> bool:
    return (
        (key < rows and query < rows and query >= key)
        or (key >= rows and query < rows and key >= rows + query + 2)
        or (key < rows and query >= rows and query >= key + rows + 2)
        or (key >= rows and query >= rows and query <= key)
    )


def flex_attention(
    values: np.ndarray, topo: Mapping[str, Any], weights: Mapping[str, np.ndarray]
) -> np.ndarray:
    prefix = "final_attention.self_attn."
    query = linear(values, weights[prefix + "q_proj.weight"])
    key = linear(values, weights[prefix + "k_proj.weight"])
    value = linear(values, weights[prefix + "v_proj.weight"])
    total_rows, width = values.shape
    rows = total_rows // 2
    heads, head_dim = int(topo["heads"]), int(topo["head_dim"])
    scale = np.float32(1.0 / math.sqrt(int(topo["flex_head_dim"])))
    attended = np.zeros_like(values)
    for head in range(heads):
        offset = head * head_dim
        for query_row in range(total_rows):
            keys = [
                key_row
                for key_row in range(total_rows)
                if flex_visible(query_row, key_row, rows)
            ]
            scores = []  # type: List[np.float32]
            for key_row in keys:
                total = np.float32(0.0)
                for column in range(head_dim):
                    total = np.float32(
                        total
                        + np.float32(
                            query[query_row, offset + column]
                            * key[key_row, offset + column]
                        )
                    )
                scores.append(np.float32(total * scale))
            maximum = max(scores)
            probabilities = [
                np.float32(np.exp(np.float32(score - maximum))) for score in scores
            ]
            denominator = np.float32(0.0)
            for probability in probabilities:
                denominator = np.float32(denominator + probability)
            for column in range(head_dim):
                total = np.float32(0.0)
                for index, key_row in enumerate(keys):
                    total = np.float32(
                        total
                        + np.float32(
                            np.float32(probabilities[index] / denominator)
                            * value[key_row, offset + column]
                        )
                    )
                attended[query_row, offset + column] = total
    return linear(attended, weights[prefix + "o_projs.0.weight"])


def run_direction_feed_forward(
    hidden: np.ndarray,
    prefix: str,
    direction: str,
    expert: bool,
    topo: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    normalized = rms_norm(
        hidden,
        weights[prefix + "pre_ff_layernorm_{}.weight".format(direction)],
        topo["epsilon"],
    )
    increment = feed_forward(
        normalized, prefix + "feed_forward_{}.".format(direction), expert, topo, weights
    )
    return np.asarray(hidden + increment, dtype=np.float32)


def final_fusion(hidden: np.ndarray) -> np.ndarray:
    total_rows, width = hidden.shape
    rows = total_rows // 2
    output = np.zeros((rows, width), dtype=np.float32)
    output[0] = hidden[rows + 1]
    for row in range(rows - 2):
        output[row + 1] = np.asarray(
            hidden[row] + hidden[rows + 2 + row], dtype=np.float32
        )
    output[rows - 1] = hidden[rows - 2]
    return output


def run_case(variant: str) -> Dict[str, Any]:
    topo = topology(variant)
    weights = fixture_weights(topo)
    tokens = np.asarray([7, 8, 11, 4], dtype=np.int64)
    mask = np.asarray([1, 1, 1, 0], dtype=np.uint8)
    original = weights["embed_tokens.weight"][tokens].copy()
    forward = original.copy()
    reverse = original[::-1].copy()
    inner_probe = None  # type: Optional[np.ndarray]
    for layer in range(int(topo["layers"])):
        expert = layer % 2 == 1
        if variant == "with-middle-attention" and layer == int(topo["mid_layer"]):
            prefix = "layers.{}.attn.".format(layer)
            normalized = rms_norm(
                forward, weights[prefix + "input_layernorm_fwd.weight"], topo["epsilon"]
            )
            forward = np.asarray(
                forward
                + softmax_attention(
                    normalized, prefix + "self_attn_fwd.", topo, weights, mask
                ),
                dtype=np.float32,
            )
            forward = run_direction_feed_forward(
                forward, prefix, "fwd", expert, topo, weights
            )
            normalized = rms_norm(
                reverse, weights[prefix + "input_layernorm_bwd.weight"], topo["epsilon"]
            )
            # Intentional source behavior: do not reverse the right-padding mask.
            reverse = np.asarray(
                reverse
                + softmax_attention(
                    normalized, prefix + "self_attn_bwd.", topo, weights, mask
                ),
                dtype=np.float32,
            )
            reverse = run_direction_feed_forward(
                reverse, prefix, "bwd", expert, topo, weights
            )
        else:
            prefix = "layers.{}.mamba_module.".format(layer)
            fwd_norm = rms_norm(
                forward, weights[prefix + "input_layernorm_fwd.weight"], topo["epsilon"]
            )
            fwd_increment = mamba_mixer(fwd_norm, prefix + "mamba_fwd.", topo, weights)
            if inner_probe is None:
                inner_probe = fwd_increment.copy()
            forward = np.asarray(forward + fwd_increment, dtype=np.float32)
            forward = run_direction_feed_forward(
                forward, prefix, "fwd", expert, topo, weights
            )
            rev_norm = rms_norm(
                reverse, weights[prefix + "input_layernorm_bwd.weight"], topo["epsilon"]
            )
            rev_increment = mamba_mixer(
                rev_norm,
                prefix + "mamba_rev.",
                topo,
                weights,
                prefix + "mamba_fwd.",
            )
            reverse = np.asarray(reverse + rev_increment, dtype=np.float32)
            reverse = run_direction_feed_forward(
                reverse, prefix, "bwd", expert, topo, weights
            )
    doubled = np.concatenate([forward, reverse[::-1]], axis=0).astype(np.float32)
    normalized = rms_norm(
        doubled, weights["final_attention.input_layernorm.weight"], topo["epsilon"]
    )
    doubled = np.asarray(
        doubled + flex_attention(normalized, topo, weights), dtype=np.float32
    )
    normalized = rms_norm(
        doubled, weights["final_attention.pre_ff_layernorm.weight"], topo["epsilon"]
    )
    doubled = np.asarray(
        doubled + mlp(normalized, "final_attention.feed_forward.", weights),
        dtype=np.float32,
    )
    fused = final_fusion(doubled)
    post_norm = rms_norm(fused, weights["final_layernorm.weight"], topo["epsilon"])
    final_hidden = np.asarray(post_norm * np.float32(2.0), dtype=np.float32)
    pooled = np.zeros(int(topo["width"]), dtype=np.float32)
    denominator = np.float32(0.0)
    for row in range(len(tokens)):
        if mask[row] == 0:
            continue
        denominator = np.float32(denominator + np.float32(1.0))
        pooled = np.asarray(pooled + final_hidden[row], dtype=np.float32)
    pooled = np.asarray(pooled / denominator, dtype=np.float32)
    return {
        "variant": variant,
        "rows": 4,
        "width": 4,
        "tokens": tokens.tolist(),
        "mask": mask.tolist(),
        "captures": [
            {"layer": 0, "values": floats(original)},
            {"layer": 2, "values": floats(final_hidden)},
        ],
        "inner_norm_probe": [] if inner_probe is None else floats(inner_probe),
        "post_final_norm": floats(post_norm),
        "final_hidden": floats(final_hidden),
        "pooled": floats(pooled),
    }


def floats(values: np.ndarray) -> List[float]:
    return [float(value) for value in values.reshape(-1)]


def micro_vectors() -> Dict[str, Any]:
    equal = np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    probe = np.arange(32, dtype=np.float32).reshape(8, 4)
    fused = final_fusion(probe)
    return {
        "top2_equal_indices": list(top_two(equal)),
        "top2_equal_weights": [float(equal[index]) for index in top_two(equal)],
        "flex_scale": float(np.float32(1.0 / math.sqrt(4.0))),
        "flex_mask": [
            [1 if flex_visible(query, key, 4) else 0 for key in range(8)]
            for query in range(8)
        ],
        "fusion_input": floats(probe),
        "fusion_output": floats(fused),
        "middle_mask_forward": [1, 1, 1, 0],
        "middle_mask_reverse_bug": [1, 1, 1, 0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "schema": "geneb-janusdna-tiny-oracle-v1",
        "generator": "independent-numpy-f32",
        "cases": [
            run_case("with-middle-attention"),
            run_case("without-middle-attention"),
        ],
        "micro": micro_vectors(),
    }
    args.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
