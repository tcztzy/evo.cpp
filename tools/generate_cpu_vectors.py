#!/usr/bin/env python3
"""Generate independent, dependency-free CPU F32 reference vectors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


VORTEX_COMMIT = "8b00afebeac745d1f31e7e2788f0e0e39fa47637"


def equivalent(left: Any, right: Any) -> bool:
    """Compare fixtures while tolerating libm last-bit drift across Python releases."""
    if isinstance(left, float) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-14, abs_tol=1e-14)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            equivalent(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def values(count: int, offset: int, divisor: float) -> list[float]:
    return [((((index + offset) * 37 + 11) % 29) - 14) / divisor for index in range(count)]


def rms_norm(x: list[float], rows: int, width: int, scale: list[float], eps: float) -> list[float]:
    output = []
    for row in range(rows):
        current = x[row * width : (row + 1) * width]
        denominator = math.sqrt(sum(item * item for item in current) / width) + eps
        output.extend(item / denominator * scale[column] for column, item in enumerate(current))
    return output


def linear(
    x: list[float], rows: int, width: int, weight: list[float], out: int, bias: list[float] | None = None
) -> list[float]:
    output = []
    for row in range(rows):
        for target in range(out):
            total = 0.0 if bias is None else bias[target]
            for source in range(width):
                total += x[row * width + source] * weight[target * width + source]
            output.append(total)
    return output


def gated_mlp(
    x: list[float],
    rows: int,
    width: int,
    inner: int,
    l1: list[float],
    l2: list[float],
    l3: list[float],
    use_gelu: bool,
) -> list[float]:
    first = linear(x, rows, width, l1, inner)
    second = linear(x, rows, width, l2, inner)
    for index in range(len(first)):
        activated = (
            0.5 * first[index] * (1.0 + math.erf(first[index] / math.sqrt(2.0)))
            if use_gelu
            else first[index]
        )
        first[index] = activated * second[index]
    return linear(first, rows, inner, l3, width)


def causal_fir(
    x: list[float],
    length: int,
    channels: int,
    weight: list[float],
    kernel: int,
    bias: list[float],
    *,
    causal_convolution: bool = False,
    gated_bias: bool = False,
) -> list[float]:
    output = [0.0] * len(x)
    for time in range(length):
        for channel in range(channels):
            total = 0.0 if gated_bias else bias[channel]
            for tap in range(kernel):
                delay = tap if causal_convolution else kernel - 1 - tap
                if time >= delay:
                    total += x[(time - delay) * channels + channel] * weight[channel * kernel + tap]
            if gated_bias:
                total += bias[channel] * x[time * channels + channel]
            output[time * channels + channel] = total
    return output


def split_hyena(projection: list[float], length: int, width: int) -> tuple[list[float], ...]:
    x2: list[float] = []
    x1: list[float] = []
    value: list[float] = []
    for time in range(length):
        for channel in range(width):
            source = time * width * 3 + channel * 3
            x2.append(projection[source])
            x1.append(projection[source + 1])
            value.append(projection[source + 2])
    return x2, x1, value


def hcl(
    x2: list[float],
    x1: list[float],
    value: list[float],
    length: int,
    width: int,
    direct: list[float],
    log_poles: list[float],
    residues: list[float],
    state_size: int,
    state: list[float],
) -> tuple[list[float], list[float]]:
    output = [0.0] * len(x2)
    state = state.copy()
    for time in range(length):
        for channel in range(width):
            data_index = time * width + channel
            x1v = x1[data_index] * value[data_index]
            modal = 0.0
            for state_index in range(state_size):
                index = channel * state_size + state_index
                state[index] = math.exp(log_poles[index]) * state[index] + x1v
                modal += residues[index] * state[index]
            output[data_index] = x2[data_index] * (modal + direct[channel] * x1v)
    return output, state


def rope(
    tensor: list[float],
    length: int,
    heads: int,
    head_dim: int,
    inverse_frequency: list[float],
    offset: int,
    position_scale: float,
) -> list[float]:
    output = tensor.copy()
    half = head_dim // 2
    for time in range(length):
        position = (offset + time) / position_scale
        for head in range(heads):
            base = (time * heads + head) * head_dim
            for dimension in range(half):
                angle = position * inverse_frequency[dimension]
                cosine, sine = math.cos(angle), math.sin(angle)
                first = output[base + dimension]
                second = output[base + half + dimension]
                output[base + dimension] = first * cosine - second * sine
                output[base + half + dimension] = second * cosine + first * sine
    return output


def attention(
    query: list[float],
    key: list[float],
    value: list[float],
    length: int,
    heads: int,
    head_dim: int,
) -> list[float]:
    output = [0.0] * len(query)
    scale = 1.0 / math.sqrt(head_dim)
    for time in range(length):
        for head in range(heads):
            query_base = (time * heads + head) * head_dim
            scores = []
            for source in range(time + 1):
                key_base = (source * heads + head) * head_dim
                scores.append(
                    sum(query[query_base + dim] * key[key_base + dim] for dim in range(head_dim)) * scale
                )
            maximum = max(scores)
            probabilities = [math.exp(score - maximum) for score in scores]
            denominator = sum(probabilities)
            for source, probability in enumerate(probabilities):
                value_base = (source * heads + head) * head_dim
                for dimension in range(head_dim):
                    output[query_base + dimension] += (
                        probability / denominator * value[value_base + dimension]
                    )
    return output


def build_vectors() -> dict[str, Any]:
    rms_input = values(8, 0, 7.0)
    rms_scale = [1.0 + item * 0.1 for item in values(4, 31, 19.0)]

    linear_input = values(8, 7, 9.0)
    linear_weight = values(12, 19, 13.0)
    linear_bias = values(3, 43, 17.0)

    mlp_input = values(6, 3, 11.0)
    l1 = values(15, 17, 23.0)
    l2 = values(15, 41, 29.0)
    l3 = values(15, 67, 31.0)

    fir_input = values(15, 5, 13.0)
    fir_weight = values(9, 29, 17.0)
    fir_bias = values(3, 53, 19.0)

    projection = values(24, 11, 15.0)
    split_x2, split_x1, split_value = split_hyena(projection, 2, 4)

    hcl_x2 = values(12, 2, 9.0)
    hcl_x1 = values(12, 37, 11.0)
    hcl_value = values(12, 71, 13.0)
    direct = values(3, 13, 17.0)
    log_poles = [-0.03 * (index + 1) for index in range(6)]
    residues = values(6, 23, 19.0)
    initial_state = values(6, 47, 31.0)
    hcl_output, hcl_state = hcl(
        hcl_x2, hcl_x1, hcl_value, 4, 3, direct, log_poles, residues, 2, initial_state
    )

    query = values(24, 59, 13.0)
    key = values(24, 83, 17.0)
    attention_value = values(24, 107, 19.0)
    inverse_frequency = [1.0, 0.01]
    rope_query = rope(query, 3, 2, 4, inverse_frequency, 5, 128.0)
    rope_key = rope(key, 3, 2, 4, inverse_frequency, 5, 128.0)

    return {
        "schema": 1,
        "reference": {
            "implementation": "Zymrael/vortex",
            "commit": VORTEX_COMMIT,
            "rmsnorm_epsilon": "after_sqrt",
            "hyena_projection_order": "x2,x1,v",
            "rope": "gpt-neox-linear-scale",
        },
        "tolerance": {"atol": 1e-5, "rtol": 1e-4},
        "shapes": {
            "rmsnorm": [2, 4],
            "linear": [2, 3],
            "mlp": [2, 3],
            "fir": [5, 3],
            "fir_causal_gated": [5, 3],
            "hyena_split": [2, 4],
            "hcl": [4, 3],
            "hcl_state": [3, 2],
            "rope": [3, 2, 4],
            "attention": [3, 2, 4],
        },
        "vectors": {
            "rmsnorm": rms_norm(rms_input, 2, 4, rms_scale, 1e-6),
            "linear": linear(linear_input, 2, 4, linear_weight, 3, linear_bias),
            "mlp_gelu": gated_mlp(mlp_input, 2, 3, 5, l1, l2, l3, True),
            "mlp_identity": gated_mlp(mlp_input, 2, 3, 5, l1, l2, l3, False),
            "fir": causal_fir(fir_input, 5, 3, fir_weight, 3, fir_bias),
            "fir_causal_gated": causal_fir(
                fir_input,
                5,
                3,
                fir_weight,
                3,
                fir_bias,
                causal_convolution=True,
                gated_bias=True,
            ),
            "hyena_x2": split_x2,
            "hyena_x1": split_x1,
            "hyena_value": split_value,
            "hcl": hcl_output,
            "hcl_state": hcl_state,
            "rope_query": rope_query,
            "rope_key": rope_key,
            "attention": attention(rope_query, rope_key, attention_value, 3, 2, 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    generated = build_vectors()
    if args.check is not None:
        existing = json.loads(args.check.read_text(encoding="utf-8"))
        if not equivalent(existing, generated):
            print(f"reference fixture is stale: {args.check}", file=sys.stderr)
            return 1
    encoded = json.dumps(generated, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    elif args.check is None:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
