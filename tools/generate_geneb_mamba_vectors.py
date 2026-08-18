#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate independent NumPy vectors for tiny Caduceus/eccDNAMamba.

The oracle spells out causal depthwise convolution, Mamba1 selective scan,
Mamba2 SSD, fused residual RMSNorm, bidirectional reversal, Caduceus RCPS, and
eccDNAMamba's two independent backbones. It does not import runtime code.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    print("NumPy is required for the GENEB Mamba oracle", file=sys.stderr)
    raise SystemExit(77)


Topology = Dict[str, Any]
Weights = Dict[str, np.ndarray]


def caduceus_topology(rcps: bool) -> Topology:
    return {
        "variant": "caduceus",
        "vocab": 6,
        "width": 2,
        "output_width": 4 if rcps else 2,
        "layers": 1,
        "inner": 4,
        "state": 2,
        "conv": 2,
        "dt_rank": 1,
        "epsilon": np.float32(1.0e-5),
        "rcps": rcps,
        "complement": (0, 1, 2, 4, 3, 5) if rcps else (),
    }


def ecc_topology() -> Topology:
    return {
        "variant": "ecc",
        "vocab": 6,
        "width": 2,
        "output_width": 2,
        "layers": 1,
        "inner": 4,
        "state": 2,
        "conv": 2,
        "mlp": 3,
        "head_width": 2,
        "heads": 2,
        "groups": 1,
        "epsilon": np.float32(1.0e-5),
    }


def add_mamba1_direction(
    result: List[Tuple[str, Tuple[int, ...]]],
    prefix: str,
    topology: Mapping[str, Any],
    tied: bool,
) -> None:
    inner = int(topology["inner"])
    width = int(topology["width"])
    state = int(topology["state"])
    rank = int(topology["dt_rank"])
    if not tied:
        result.append((prefix + "in_proj.weight", (inner * 2, width)))
    result.extend(
        [
            (prefix + "conv1d.weight", (inner, 1, int(topology["conv"]))),
            (prefix + "conv1d.bias", (inner,)),
            (prefix + "x_proj.weight", (rank + state * 2, inner)),
            (prefix + "dt_proj.weight", (inner, rank)),
            (prefix + "dt_proj.bias", (inner,)),
            (prefix + "A_log", (inner, state)),
            (prefix + "D", (inner,)),
        ]
    )
    if not tied:
        result.append((prefix + "out_proj.weight", (width, inner)))


def requirements(topology: Mapping[str, Any]) -> List[Tuple[str, Tuple[int, ...]]]:
    width = int(topology["width"])
    result = []  # type: List[Tuple[str, Tuple[int, ...]]]
    if topology["variant"] == "caduceus":
        result.append(("embedding.weight", (int(topology["vocab"]), width)))
        for layer in range(int(topology["layers"])):
            prefix = "layers.%d." % layer
            result.append((prefix + "norm.weight", (width,)))
            add_mamba1_direction(result, prefix + "forward.", topology, False)
            add_mamba1_direction(result, prefix + "reverse.", topology, True)
        result.append(("final_norm.weight", (width,)))
        return result
    inner = int(topology["inner"])
    grouped_state = int(topology["groups"]) * int(topology["state"])
    result.append(("token_embedding.weight", (int(topology["vocab"]), width)))
    for direction in ("forward", "reverse"):
        for layer in range(int(topology["layers"])):
            prefix = "%s.layers.%d." % (direction, layer)
            result.extend(
                [
                    (prefix + "norm.weight", (width,)),
                    (
                        prefix + "mixer.in_proj.weight",
                        (inner * 2 + grouped_state * 2 + int(topology["heads"]), width),
                    ),
                    (
                        prefix + "mixer.conv1d.weight",
                        (inner + grouped_state * 2, 1, int(topology["conv"])),
                    ),
                    (prefix + "mixer.conv1d.bias", (inner + grouped_state * 2,)),
                    (prefix + "mixer.dt_bias", (int(topology["heads"]),)),
                    (prefix + "mixer.A_log", (int(topology["heads"]),)),
                    (prefix + "mixer.D", (int(topology["heads"]),)),
                    (prefix + "mixer.norm.weight", (inner,)),
                    (prefix + "mixer.out_proj.weight", (width, inner)),
                    (prefix + "norm2.weight", (width,)),
                    (prefix + "mlp.fc1.weight", (int(topology["mlp"]) * 2, width)),
                    (prefix + "mlp.fc2.weight", (width, int(topology["mlp"]))),
                ]
            )
        result.append((direction + ".final_norm.weight", (width,)))
    result.append(("projection.weight", (width, width * 2)))
    return result


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> np.float32:
    integer = ((tensor_index + 3) * 17 + (element_index + 7) * 13) % 41 - 20
    value = np.float32(np.float32(integer) / np.float32(37.0))
    if "norm.weight" in name or name == "final_norm.weight":
        return np.float32(np.float32(1.0) + value * np.float32(0.035))
    if "A_log" in name:
        return np.float32(np.float32(-0.7) + value * np.float32(0.05))
    if "dt_proj.bias" in name or "dt_bias" in name:
        return np.float32(np.float32(-1.2) + value * np.float32(0.08))
    if name.endswith(".D"):
        return np.float32(np.float32(0.6) + value * np.float32(0.04))
    if "embedding.weight" in name:
        return np.float32(value * np.float32(0.22))
    return np.float32(value * np.float32(0.11))


def fixture_weights(topology: Mapping[str, Any]) -> Weights:
    result = {}  # type: Weights
    for tensor_index, (name, shape) in enumerate(requirements(topology)):
        count = 1
        for dimension in shape:
            count *= dimension
        result[name] = np.asarray(
            [fixture_scalar(name, tensor_index, index) for index in range(count)],
            dtype=np.float32,
        ).reshape(shape)
    return result


def linear(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rows, input_width = values.shape
    output_width = weight.shape[0]
    result = np.zeros((rows, output_width), dtype=np.float32)
    for row in range(rows):
        for target in range(output_width):
            total = np.float32(0.0)
            for source in range(input_width):
                total = np.float32(
                    total
                    + np.float32(values[row, source] * weight[target, source])
                )
            result[row, target] = total
    return result


def silu(value: np.float32) -> np.float32:
    if value >= np.float32(0.0):
        exponential = np.float32(np.exp(np.float32(-value)))
        return np.float32(value / np.float32(np.float32(1.0) + exponential))
    exponential = np.float32(np.exp(value))
    return np.float32(
        np.float32(value * exponential) / np.float32(np.float32(1.0) + exponential)
    )


def softplus(value: np.float32) -> np.float32:
    if value > np.float32(20.0):
        return value
    if value < np.float32(-20.0):
        return np.float32(np.exp(value))
    return np.float32(np.log1p(np.exp(value)))


def causal_conv_silu(
    values: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    rows, channels = values.shape
    kernel = weight.shape[-1]
    result = np.zeros_like(values)
    for row in range(rows):
        for channel in range(channels):
            total = np.float32(bias[channel])
            for tap in range(kernel):
                lag = kernel - 1 - tap
                if row >= lag:
                    total = np.float32(
                        total
                        + np.float32(values[row - lag, channel] * weight[channel, 0, tap])
                    )
            result[row, channel] = silu(total)
    return result


def rms_norm(values: np.ndarray, scale: np.ndarray, epsilon: np.float32) -> np.ndarray:
    rows, width = values.shape
    result = np.zeros_like(values)
    for row in range(rows):
        total = np.float32(0.0)
        for column in range(width):
            total = np.float32(total + np.float32(values[row, column] ** 2))
        inverse = np.float32(
            np.float32(1.0)
            / np.sqrt(np.float32(total / np.float32(width) + epsilon))
        )
        for column in range(width):
            result[row, column] = np.float32(
                np.float32(values[row, column] * inverse) * scale[column]
            )
    return result


def add_norm(
    hidden: np.ndarray,
    residual: Optional[np.ndarray],
    scale: np.ndarray,
    epsilon: np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    summed = hidden.copy() if residual is None else np.asarray(hidden + residual, dtype=np.float32)
    return rms_norm(summed, scale, epsilon), summed


def reverse_rows(values: np.ndarray, channels: bool = False) -> np.ndarray:
    result = values[::-1, :]
    if channels:
        result = result[:, ::-1]
    return result.copy()


def mamba1_mixer(
    values: np.ndarray,
    prefix: str,
    topology: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
    tied_prefix: Optional[str] = None,
    projected_parameter_norm: bool = False,
) -> np.ndarray:
    in_prefix = prefix if tied_prefix is None else tied_prefix
    projected = linear(values, weights[in_prefix + "in_proj.weight"])
    inner = int(topology["inner"])
    x = projected[:, :inner].copy()
    gate = projected[:, inner:].copy()
    x = causal_conv_silu(
        x, weights[prefix + "conv1d.weight"], weights[prefix + "conv1d.bias"]
    )
    parameters = linear(x, weights[prefix + "x_proj.weight"])
    rank = int(topology["dt_rank"])
    state_width = int(topology["state"])
    rank_values = parameters[:, :rank].copy()
    b_values = parameters[:, rank : rank + state_width].copy()
    c_values = parameters[:, rank + state_width :].copy()
    if projected_parameter_norm:
        epsilon = np.float32(1.0e-6)
        rank_values = rms_norm(
            rank_values, weights[prefix + "dt_layernorm.weight"], epsilon
        )
        b_values = rms_norm(
            b_values, weights[prefix + "B_layernorm.weight"], epsilon
        )
        c_values = rms_norm(
            c_values, weights[prefix + "C_layernorm.weight"], epsilon
        )
    raw_dt = linear(rank_values, weights[prefix + "dt_proj.weight"])
    state = np.zeros((inner, state_width), dtype=np.float32)
    scanned = np.zeros_like(x)
    for row in range(values.shape[0]):
        for channel in range(inner):
            delta = softplus(
                np.float32(raw_dt[row, channel] + weights[prefix + "dt_proj.bias"][channel])
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
                        np.float32(delta * b_values[row, state_index])
                        * x[row, channel]
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
    return linear(scanned, weights[in_prefix + "out_proj.weight"])


def inner_norm_fixture() -> Tuple[np.ndarray, Topology, Weights, str]:
    def values(count: int, scale: float, shift: int) -> np.ndarray:
        return np.asarray(
            [np.float32((int((index * 11 + 5) % 17) - shift) * scale)
             for index in range(count)],
            dtype=np.float32,
        )

    prefix = "primitive."
    weights = {
        prefix + "in_proj.weight": values(16, 0.035, 8).reshape(8, 2),
        prefix + "conv1d.weight": values(8, 0.021, 7).reshape(4, 1, 2),
        prefix + "conv1d.bias": values(4, 0.013, 6),
        prefix + "x_proj.weight": values(24, 0.027, 8).reshape(6, 4),
        prefix + "dt_proj.weight": values(8, 0.019, 7).reshape(4, 2),
        prefix + "dt_proj.bias": values(4, 0.041, 10),
        prefix + "A_log": values(8, 0.017, 5).reshape(4, 2),
        prefix + "D": values(4, 0.023, 4),
        prefix + "out_proj.weight": values(8, 0.031, 8).reshape(2, 4),
        prefix + "dt_layernorm.weight": np.asarray([0.8, 1.1], dtype=np.float32),
        prefix + "B_layernorm.weight": np.asarray([1.2, 0.9], dtype=np.float32),
        prefix + "C_layernorm.weight": np.asarray([0.7, 1.3], dtype=np.float32),
    }
    topology = {"inner": 4, "state": 2, "dt_rank": 2}
    inputs = np.asarray(
        [[0.2, -0.3], [0.7, 0.1], [-0.4, 0.5]], dtype=np.float32
    )
    return inputs, topology, weights, prefix


def inner_norm_primitive() -> List[float]:
    inputs, topology, weights, prefix = inner_norm_fixture()
    return mamba1_mixer(
        inputs, prefix, topology, weights, projected_parameter_norm=True
    ).reshape(-1).tolist()


def bidirectional_mamba1(
    values: np.ndarray,
    layer_prefix: str,
    topology: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    forward_prefix = layer_prefix + "forward."
    reverse_prefix = layer_prefix + "reverse."
    forward = mamba1_mixer(values, forward_prefix, topology, weights)
    reverse = mamba1_mixer(
        reverse_rows(values), reverse_prefix, topology, weights, forward_prefix
    )
    return np.asarray(forward + reverse_rows(reverse), dtype=np.float32)


def caduceus_forward(topology: Mapping[str, Any]) -> Dict[str, List[float]]:
    weights = fixture_weights(topology)
    tokens = (3, 4, 5)
    width = int(topology["width"])
    base = weights["embedding.weight"][list(tokens), :].copy()
    if topology["rcps"]:
        complement = [int(topology["complement"][token]) for token in tokens]
        second = weights["embedding.weight"][complement, ::-1].copy()
        hidden = np.concatenate([base, second], axis=1)
    else:
        hidden = base
    vectors = {"capture0": hidden.reshape(-1).tolist()}
    residual = None  # type: Optional[np.ndarray]
    for layer in range(int(topology["layers"])):
        prefix = "layers.%d." % layer
        scale = weights[prefix + "norm.weight"]
        if not topology["rcps"]:
            normalized, residual = add_norm(
                hidden, residual, scale, np.float32(topology["epsilon"])
            )
            hidden = bidirectional_mamba1(normalized, prefix, topology, weights)
        else:
            first = hidden[:, :width]
            second = hidden[:, width:]
            residual_first = None if residual is None else residual[:, :width]
            residual_second = None if residual is None else residual[:, width:]
            norm_first, residual_first_new = add_norm(
                second, residual_second, scale, np.float32(topology["epsilon"])
            )
            norm_second_reversed, residual_second_reversed = add_norm(
                reverse_rows(first, True),
                None if residual_first is None else reverse_rows(residual_first, True),
                scale,
                np.float32(topology["epsilon"]),
            )
            norm_second = reverse_rows(norm_second_reversed, True)
            residual_second_new = reverse_rows(residual_second_reversed, True)
            mixed_first = bidirectional_mamba1(norm_first, prefix, topology, weights)
            mixed_second = reverse_rows(
                bidirectional_mamba1(
                    reverse_rows(norm_second, True), prefix, topology, weights
                ),
                True,
            )
            hidden = np.concatenate([mixed_first, mixed_second], axis=1)
            residual = np.concatenate(
                [residual_first_new, residual_second_new], axis=1
            )
    scale = weights["final_norm.weight"]
    if not topology["rcps"]:
        final, _ = add_norm(
            hidden, residual, scale, np.float32(topology["epsilon"])
        )
    else:
        parts = []  # type: List[np.ndarray]
        for half in range(2):
            hidden_half = hidden[:, half * width : (half + 1) * width]
            residual_half = residual[:, half * width : (half + 1) * width]
            if half == 1:
                hidden_half = reverse_rows(hidden_half, True)
                residual_half = reverse_rows(residual_half, True)
            normalized, _ = add_norm(
                hidden_half,
                residual_half,
                scale,
                np.float32(topology["epsilon"]),
            )
            if half == 1:
                normalized = reverse_rows(normalized, True)
            parts.append(normalized)
        final = np.concatenate(parts, axis=1)
    vectors["capture1"] = final.reshape(-1).tolist()
    vectors["final"] = final.reshape(-1).tolist()
    vectors["pooled"] = np.asarray(
        (final[0, :] + final[1, :]) / np.float32(2.0), dtype=np.float32
    ).tolist()
    return vectors


def group_rms_gated(
    values: np.ndarray,
    gate: np.ndarray,
    scale: np.ndarray,
    groups: int,
    epsilon: np.float32,
) -> np.ndarray:
    rows, width = values.shape
    group_width = width // groups
    result = np.zeros_like(values)
    gated = np.zeros_like(values)
    for row in range(rows):
        for column in range(width):
            gated[row, column] = np.float32(values[row, column] * silu(gate[row, column]))
        for group in range(groups):
            begin = group * group_width
            total = np.float32(0.0)
            for offset in range(group_width):
                value = gated[row, begin + offset]
                total = np.float32(total + np.float32(value * value))
            inverse = np.float32(
                np.float32(1.0)
                / np.sqrt(np.float32(total / np.float32(group_width) + epsilon))
            )
            for offset in range(group_width):
                column = begin + offset
                result[row, column] = np.float32(
                    np.float32(gated[row, column] * inverse) * scale[column]
                )
    return result


def mamba2_mixer(
    values: np.ndarray,
    prefix: str,
    topology: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    projected = linear(values, weights[prefix + "in_proj.weight"])
    inner = int(topology["inner"])
    state_width = int(topology["state"])
    groups = int(topology["groups"])
    grouped_state = groups * state_width
    heads = int(topology["heads"])
    head_width = int(topology["head_width"])
    gate = projected[:, :inner]
    conv_input = projected[:, inner : inner + inner + grouped_state * 2]
    raw_dt = projected[:, -heads:]
    convolved = causal_conv_silu(
        conv_input, weights[prefix + "conv1d.weight"], weights[prefix + "conv1d.bias"]
    )
    x = convolved[:, :inner]
    b_values = convolved[:, inner : inner + grouped_state]
    c_values = convolved[:, inner + grouped_state :]
    state = np.zeros((inner, state_width), dtype=np.float32)
    scanned = np.zeros_like(x)
    heads_per_group = heads // groups
    for row in range(values.shape[0]):
        for head in range(heads):
            delta = softplus(
                np.float32(raw_dt[row, head] + weights[prefix + "dt_bias"][head])
            )
            a_value = np.float32(-np.exp(np.float32(weights[prefix + "A_log"][head])))
            decay = np.float32(np.exp(np.float32(delta * a_value)))
            group = head // heads_per_group
            for lane in range(head_width):
                channel = head * head_width + lane
                total = np.float32(0.0)
                for state_index in range(state_width):
                    state[channel, state_index] = np.float32(
                        np.float32(decay * state[channel, state_index])
                        + np.float32(
                            np.float32(
                                delta
                                * b_values[
                                    row, group * state_width + state_index
                                ]
                            )
                            * x[row, channel]
                        )
                    )
                    total = np.float32(
                        total
                        + np.float32(
                            state[channel, state_index]
                            * c_values[row, group * state_width + state_index]
                        )
                    )
                scanned[row, channel] = np.float32(
                    total + np.float32(weights[prefix + "D"][head] * x[row, channel])
                )
    normalized = group_rms_gated(
        scanned,
        gate,
        weights[prefix + "norm.weight"],
        groups,
        np.float32(1.0e-5),
    )
    return linear(normalized, weights[prefix + "out_proj.weight"])


def gated_mlp(
    values: np.ndarray, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    projected = linear(values, first)
    half = projected.shape[1] // 2
    activated = np.zeros((values.shape[0], half), dtype=np.float32)
    for row in range(values.shape[0]):
        for column in range(half):
            activated[row, column] = np.float32(
                projected[row, column] * silu(projected[row, half + column])
            )
    return linear(activated, second)


def ecc_direction(
    embeddings: np.ndarray,
    mask: Sequence[int],
    direction: str,
    topology: Mapping[str, Any],
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    hidden = embeddings.copy()
    residual = None  # type: Optional[np.ndarray]
    epsilon = np.float32(topology["epsilon"])
    for layer in range(int(topology["layers"])):
        prefix = "%s.layers.%d." % (direction, layer)
        normalized, residual = add_norm(
            hidden, residual, weights[prefix + "norm.weight"], epsilon
        )
        hidden = mamba2_mixer(normalized, prefix + "mixer.", topology, weights)
        normalized, residual = add_norm(
            hidden, residual, weights[prefix + "norm2.weight"], epsilon
        )
        hidden = gated_mlp(
            normalized,
            weights[prefix + "mlp.fc1.weight"],
            weights[prefix + "mlp.fc2.weight"],
        )
        for row in range(hidden.shape[0]):
            hidden[row, :] = np.asarray(
                hidden[row, :] * np.float32(mask[row]), dtype=np.float32
            )
            residual[row, :] = np.asarray(
                residual[row, :] * np.float32(mask[row]), dtype=np.float32
            )
    final, _ = add_norm(
        hidden, residual, weights[direction + ".final_norm.weight"], epsilon
    )
    return final


def ecc_forward(topology: Mapping[str, Any]) -> Dict[str, List[float]]:
    weights = fixture_weights(topology)
    tokens = (3, 4, 5)
    mask = (1, 1, 0)
    embeddings = weights["token_embedding.weight"][list(tokens), :].copy()
    forward = ecc_direction(embeddings, mask, "forward", topology, weights)
    reverse = ecc_direction(
        reverse_rows(embeddings), tuple(reversed(mask)), "reverse", topology, weights
    )
    combined = np.concatenate([forward, reverse_rows(reverse)], axis=1)
    final = linear(combined, weights["projection.weight"])
    return {
        "capture0": embeddings.reshape(-1).tolist(),
        "capture1": final.reshape(-1).tolist(),
        "final": final.reshape(-1).tolist(),
        "pooled": np.asarray(
            (final[0, :] + final[1, :]) / np.float32(2.0), dtype=np.float32
        ).tolist(),
    }


def generate() -> Dict[str, Any]:
    vectors = {}  # type: Dict[str, List[float]]
    for prefix, values in (
        ("caduceus", caduceus_forward(caduceus_topology(False))),
        ("rcps", caduceus_forward(caduceus_topology(True))),
        ("ecc", ecc_forward(ecc_topology())),
    ):
        for name, vector in values.items():
            vectors[prefix + "." + name] = vector
    vectors["primitive.inner_norm"] = inner_norm_primitive()
    return {"schema_version": 1, "oracle": "numpy-f32", "vectors": vectors}


def torch_check() -> None:
    """Cross-check the recurrence primitives with independent PyTorch ops."""
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError:
        print("PyTorch unavailable; skipping GENEB Mamba PyTorch oracle")
        raise SystemExit(77)

    def tensor(value: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(value))

    def torch_rms(values: Any, scale: Any, epsilon: float) -> Any:
        inverse = torch.rsqrt(
            torch.mean(values * values, dim=-1, keepdim=True) + epsilon
        )
        return values * inverse * scale

    def torch_conv(values: Any, weight: Any, bias: Any) -> Any:
        channels = int(values.shape[1])
        kernel = int(weight.shape[-1])
        sequence = values.transpose(0, 1).unsqueeze(0)
        padded = functional.pad(sequence, (kernel - 1, 0))
        convolved = functional.conv1d(
            padded, weight, bias, groups=channels
        )
        return functional.silu(convolved).squeeze(0).transpose(0, 1).contiguous()

    def torch_mamba1(
        values: Any,
        prefix: str,
        topology: Mapping[str, Any],
        weights: Mapping[str, np.ndarray],
        projected_parameter_norm: bool = False,
    ) -> Any:
        projected = functional.linear(values, tensor(weights[prefix + "in_proj.weight"]))
        inner = int(topology["inner"])
        x = projected[:, :inner]
        gate = projected[:, inner:]
        x = torch_conv(
            x,
            tensor(weights[prefix + "conv1d.weight"]),
            tensor(weights[prefix + "conv1d.bias"]),
        )
        parameters = functional.linear(x, tensor(weights[prefix + "x_proj.weight"]))
        rank = int(topology["dt_rank"])
        state_width = int(topology["state"])
        rank_values = parameters[:, :rank]
        b_values = parameters[:, rank : rank + state_width]
        c_values = parameters[:, rank + state_width :]
        if projected_parameter_norm:
            rank_values = torch_rms(
                rank_values,
                tensor(weights[prefix + "dt_layernorm.weight"]),
                1.0e-6,
            )
            b_values = torch_rms(
                b_values,
                tensor(weights[prefix + "B_layernorm.weight"]),
                1.0e-6,
            )
            c_values = torch_rms(
                c_values,
                tensor(weights[prefix + "C_layernorm.weight"]),
                1.0e-6,
            )
        raw_dt = functional.linear(
            rank_values, tensor(weights[prefix + "dt_proj.weight"])
        )
        a_values = -torch.exp(tensor(weights[prefix + "A_log"]))
        time_bias = tensor(weights[prefix + "dt_proj.bias"])
        skip = tensor(weights[prefix + "D"])
        state = torch.zeros((inner, state_width), dtype=torch.float32)
        rows = []
        for row in range(int(values.shape[0])):
            delta = functional.softplus(raw_dt[row] + time_bias)
            state = (
                torch.exp(delta[:, None] * a_values) * state
                + delta[:, None] * b_values[row][None, :] * x[row][:, None]
            )
            scanned = torch.sum(state * c_values[row][None, :], dim=-1)
            scanned = (scanned + skip * x[row]) * functional.silu(gate[row])
            rows.append(scanned)
        return functional.linear(
            torch.stack(rows), tensor(weights[prefix + "out_proj.weight"])
        )

    def torch_mamba2(
        values: Any,
        prefix: str,
        topology: Mapping[str, Any],
        weights: Mapping[str, np.ndarray],
    ) -> Any:
        projected = functional.linear(values, tensor(weights[prefix + "in_proj.weight"]))
        inner = int(topology["inner"])
        state_width = int(topology["state"])
        groups = int(topology["groups"])
        grouped_state = groups * state_width
        heads = int(topology["heads"])
        head_width = int(topology["head_width"])
        gate = projected[:, :inner]
        convolution_input = projected[:, inner : inner * 2 + grouped_state * 2]
        raw_dt = projected[:, -heads:]
        convolved = torch_conv(
            convolution_input,
            tensor(weights[prefix + "conv1d.weight"]),
            tensor(weights[prefix + "conv1d.bias"]),
        )
        x = convolved[:, :inner]
        b_values = convolved[:, inner : inner + grouped_state]
        c_values = convolved[:, inner + grouped_state :]
        a_values = -torch.exp(tensor(weights[prefix + "A_log"]))
        time_bias = tensor(weights[prefix + "dt_bias"])
        skip = tensor(weights[prefix + "D"])
        state = torch.zeros((inner, state_width), dtype=torch.float32)
        heads_per_group = heads // groups
        scan_rows = []
        for row in range(int(values.shape[0])):
            channels = []
            for head in range(heads):
                delta = functional.softplus(raw_dt[row, head] + time_bias[head])
                decay = torch.exp(delta * a_values[head])
                group = head // heads_per_group
                begin = group * state_width
                for lane in range(head_width):
                    channel = head * head_width + lane
                    next_state = (
                        decay * state[channel]
                        + delta
                        * b_values[row, begin : begin + state_width]
                        * x[row, channel]
                    )
                    state = torch.cat(
                        (state[:channel], next_state[None, :], state[channel + 1 :]),
                        dim=0,
                    )
                    scanned = torch.sum(
                        next_state * c_values[row, begin : begin + state_width]
                    )
                    channels.append(scanned + skip[head] * x[row, channel])
            scan_rows.append(torch.stack(channels))
        scanned = torch.stack(scan_rows)
        gated = scanned * functional.silu(gate)
        group_width = inner // groups
        grouped = gated.reshape(int(values.shape[0]), groups, group_width)
        normalized = grouped * torch.rsqrt(
            torch.mean(grouped * grouped, dim=-1, keepdim=True) + 1.0e-5
        )
        normalized = normalized.reshape(int(values.shape[0]), inner)
        normalized = normalized * tensor(weights[prefix + "norm.weight"])
        return functional.linear(
            normalized, tensor(weights[prefix + "out_proj.weight"])
        )

    def assert_close(label: str, actual: Any, expected: np.ndarray) -> None:
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(), expected, atol=3.0e-6, rtol=3.0e-6,
            err_msg=label,
        )

    caduceus = caduceus_topology(False)
    caduceus_weights = fixture_weights(caduceus)
    prefix = "layers.0.forward."
    caduceus_input = caduceus_weights["embedding.weight"][[3, 4, 5], :].copy()
    assert_close(
        "Mamba1 selective scan",
        torch_mamba1(tensor(caduceus_input), prefix, caduceus, caduceus_weights),
        mamba1_mixer(caduceus_input, prefix, caduceus, caduceus_weights),
    )

    inner_input, inner_topology, inner_weights, inner_prefix = inner_norm_fixture()
    assert_close(
        "Mamba1 projected dt/B/C RMSNorm",
        torch_mamba1(
            tensor(inner_input), inner_prefix, inner_topology, inner_weights, True
        ),
        mamba1_mixer(
            inner_input, inner_prefix, inner_topology, inner_weights,
            projected_parameter_norm=True,
        ),
    )

    ecc = ecc_topology()
    ecc_weights = fixture_weights(ecc)
    ecc_prefix = "forward.layers.0.mixer."
    ecc_input = ecc_weights["token_embedding.weight"][[3, 4, 5], :].copy()
    assert_close(
        "Mamba2 SSD and RMSNorm-gate",
        torch_mamba2(tensor(ecc_input), ecc_prefix, ecc, ecc_weights),
        mamba2_mixer(ecc_input, ecc_prefix, ecc, ecc_weights),
    )
    print(
        "GENEB Mamba PyTorch causal-conv/Mamba1/Mamba2/inner-RMS oracle passed"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--torch-check", action="store_true")
    args = parser.parse_args()
    if args.torch_check:
        torch_check()
        return 0
    payload = json.dumps(generate(), sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
