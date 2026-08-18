#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy/PyTorch oracle for GPT2-Gene and DNA-GPT runtimes."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    print("NumPy is required for the GENEB GPT oracle", file=sys.stderr)
    raise SystemExit(77)


TOKENS = (1, 4, 2, 6)
MASK = (1, 1, 1, 0)
VOCABULARY = 7
WIDTH = 4
LAYERS = 2
HEADS = 2
HEAD_WIDTH = 2
INNER_WIDTH = 8
MAX_LENGTH = 8
EPSILON = np.float32(1.0e-5)


def requirements(family: str) -> List[Tuple[str, Tuple[int, ...]]]:
    in_out = family == "gpt2"

    def matrix(input_width: int, output_width: int) -> Tuple[int, ...]:
        return (input_width, output_width) if in_out else (output_width, input_width)

    result = [
        ("transformer.wte.weight", (VOCABULARY, WIDTH)),
        ("transformer.wpe.weight", (MAX_LENGTH, WIDTH)),
    ]
    for layer in range(LAYERS):
        prefix = "transformer.h.%d." % layer
        result.append((prefix + "attn.c_attn.weight", matrix(WIDTH, WIDTH * 3)))
        if family == "gpt2":
            result.append((prefix + "attn.c_attn.bias", (WIDTH * 3,)))
        result.append((prefix + "attn.c_proj.weight", matrix(WIDTH, WIDTH)))
        if family == "gpt2":
            result.append((prefix + "attn.c_proj.bias", (WIDTH,)))
        result.append((prefix + "ln_1.weight", (WIDTH,)))
        if family == "gpt2":
            result.append((prefix + "ln_1.bias", (WIDTH,)))
        result.append((prefix + "ln_2.weight", (WIDTH,)))
        if family == "gpt2":
            result.append((prefix + "ln_2.bias", (WIDTH,)))
        result.append((prefix + "mlp.c_fc.weight", matrix(WIDTH, INNER_WIDTH)))
        if family == "gpt2":
            result.append((prefix + "mlp.c_fc.bias", (INNER_WIDTH,)))
        result.append((prefix + "mlp.c_proj.weight", matrix(INNER_WIDTH, WIDTH)))
        if family == "gpt2":
            result.append((prefix + "mlp.c_proj.bias", (WIDTH,)))
    result.append(("transformer.ln_f.weight", (WIDTH,)))
    if family == "gpt2":
        result.append(("transformer.ln_f.bias", (WIDTH,)))
    return result


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> np.float32:
    integer = ((tensor_index + 3) * 17 + (element_index + 7) * 13) % 47 - 23
    value = np.float32(np.float32(integer) / np.float32(37.0))
    if "ln_" in name or name == "transformer.ln_f.weight":
        if name.endswith(".weight"):
            value = np.float32(np.float32(1.0) + value * np.float32(0.04))
        else:
            value = np.float32(value * np.float32(0.025))
    elif name.endswith(".bias"):
        value = np.float32(value * np.float32(0.035))
    elif name in ("transformer.wte.weight", "transformer.wpe.weight"):
        value = np.float32(value * np.float32(0.18))
    else:
        value = np.float32(value * np.float32(0.11))
    return value


def weights(family: str) -> Dict[str, np.ndarray]:
    result = {}  # type: Dict[str, np.ndarray]
    for tensor_index, (name, shape) in enumerate(requirements(family)):
        count = math.prod(shape)
        result[name] = np.asarray(
            [fixture_scalar(name, tensor_index, index) for index in range(count)],
            dtype=np.float32,
        ).reshape(shape)
    return result


def linear(
    values: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    layout: str,
) -> np.ndarray:
    rows, input_width = values.shape
    output_width = weight.shape[0] if layout == "out-in" else weight.shape[1]
    result = np.zeros((rows, output_width), dtype=np.float32)
    for row in range(rows):
        for target in range(output_width):
            total = np.float32(0.0 if bias is None else bias[target])
            for source in range(input_width):
                item = (
                    weight[target, source]
                    if layout == "out-in"
                    else weight[source, target]
                )
                total = np.float32(
                    total + np.float32(values[row, source] * item)
                )
            result[row, target] = total
    return result


def layer_norm(
    values: np.ndarray, scale: np.ndarray, bias: Optional[np.ndarray]
) -> np.ndarray:
    rows, width = values.shape
    result = np.empty_like(values)
    for row in range(rows):
        total = np.float32(0.0)
        for column in range(width):
            total = np.float32(total + values[row, column])
        mean = np.float32(total / np.float32(width))
        square_sum = np.float32(0.0)
        for column in range(width):
            centered = np.float32(values[row, column] - mean)
            square_sum = np.float32(
                square_sum + np.float32(centered * centered)
            )
        inverse = np.float32(
            np.float32(1.0)
            / np.sqrt(
                np.float32(square_sum / np.float32(width) + EPSILON)
            )
        )
        for column in range(width):
            item = np.float32(
                np.float32(values[row, column] - mean)
                * inverse
                * scale[column]
            )
            if bias is not None:
                item = np.float32(item + bias[column])
            result[row, column] = item
    return result


def attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    uses_mask: bool,
) -> np.ndarray:
    rows = query.shape[0]
    query = query.reshape(rows, HEADS, HEAD_WIDTH)
    key = key.reshape(rows, HEADS, HEAD_WIDTH)
    value = value.reshape(rows, HEADS, HEAD_WIDTH)
    result = np.zeros_like(query)
    scale = np.float32(np.float32(1.0) / np.sqrt(np.float32(HEAD_WIDTH)))
    for row in range(rows):
        for head in range(HEADS):
            visible = [
                source
                for source in range(row + 1)
                if not uses_mask or MASK[source] != 0
            ]
            scores = []  # type: List[np.float32]
            for source in visible:
                total = np.float32(0.0)
                for dimension in range(HEAD_WIDTH):
                    total = np.float32(
                        total
                        + np.float32(
                            query[row, head, dimension]
                            * key[source, head, dimension]
                        )
                    )
                scores.append(np.float32(total * scale))
            maximum = max(scores)
            probabilities = []  # type: List[np.float32]
            denominator = np.float32(0.0)
            for score in scores:
                probability = np.float32(np.exp(np.float32(score - maximum)))
                probabilities.append(probability)
                denominator = np.float32(denominator + probability)
            for dimension in range(HEAD_WIDTH):
                total = np.float32(0.0)
                for index, source in enumerate(visible):
                    probability = np.float32(probabilities[index] / denominator)
                    total = np.float32(
                        total
                        + np.float32(
                            probability * value[source, head, dimension]
                        )
                    )
                result[row, head, dimension] = total
    return result.reshape(rows, WIDTH)


def gelu_tanh(value: np.float32) -> np.float32:
    coefficient = np.float32(0.7978845608028654)
    cubic = np.float32(value * value * value)
    inner = np.float32(
        coefficient * np.float32(value + np.float32(np.float32(0.044715) * cubic))
    )
    return np.float32(
        np.float32(0.5)
        * value
        * np.float32(np.float32(1.0) + np.tanh(inner))
    )


def forward(family: str) -> Dict[str, List[float]]:
    values = weights(family)
    layout = "in-out" if family == "gpt2" else "out-in"
    has_bias = family == "gpt2"
    hidden = np.asarray(
        values["transformer.wte.weight"][list(TOKENS), :]
        + values["transformer.wpe.weight"][: len(TOKENS), :],
        dtype=np.float32,
    )
    vectors = {"capture0": hidden.reshape(-1).tolist()}
    for layer in range(LAYERS):
        prefix = "transformer.h.%d." % layer
        normalized = layer_norm(
            hidden,
            values[prefix + "ln_1.weight"],
            values[prefix + "ln_1.bias"] if has_bias else None,
        )
        fused = linear(
            normalized,
            values[prefix + "attn.c_attn.weight"],
            values[prefix + "attn.c_attn.bias"] if has_bias else None,
            layout,
        )
        query, key, value = np.split(fused, 3, axis=1)
        attended = attention(query, key, value, family == "gpt2")
        hidden = np.asarray(
            hidden
            + linear(
                attended,
                values[prefix + "attn.c_proj.weight"],
                values[prefix + "attn.c_proj.bias"] if has_bias else None,
                layout,
            ),
            dtype=np.float32,
        )
        normalized = layer_norm(
            hidden,
            values[prefix + "ln_2.weight"],
            values[prefix + "ln_2.bias"] if has_bias else None,
        )
        feed_forward = linear(
            normalized,
            values[prefix + "mlp.c_fc.weight"],
            values[prefix + "mlp.c_fc.bias"] if has_bias else None,
            layout,
        )
        for index in range(feed_forward.size):
            feed_forward.reshape(-1)[index] = gelu_tanh(
                feed_forward.reshape(-1)[index]
            )
        hidden = np.asarray(
            hidden
            + linear(
                feed_forward,
                values[prefix + "mlp.c_proj.weight"],
                values[prefix + "mlp.c_proj.bias"] if has_bias else None,
                layout,
            ),
            dtype=np.float32,
        )
        if layer + 1 < LAYERS:
            vectors["capture%d" % (layer + 1)] = hidden.reshape(-1).tolist()
    final_hidden = layer_norm(
        hidden,
        values["transformer.ln_f.weight"],
        values["transformer.ln_f.bias"] if has_bias else None,
    )
    vectors["capture%d" % LAYERS] = final_hidden.reshape(-1).tolist()
    vectors["final"] = final_hidden.reshape(-1).tolist()
    pooled = np.zeros((WIDTH,), dtype=np.float32)
    for row in range(3):
        pooled = np.asarray(pooled + final_hidden[row, :], dtype=np.float32)
    pooled = np.asarray(pooled / np.float32(3.0), dtype=np.float32)
    vectors["pooled"] = pooled.tolist()
    return vectors


def compare(expected: Mapping[str, Sequence[float]], actual: object) -> None:
    if not isinstance(actual, dict) or set(expected) != set(actual):
        raise AssertionError("native GENEB GPT vector names differ")
    for name in sorted(expected):
        wanted = expected[name]
        got = actual[name]
        if not isinstance(got, list) or len(got) != len(wanted):
            raise AssertionError("%s vector length differs" % name)
        for index, (left, right) in enumerate(zip(wanted, got)):
            if not math.isclose(left, right, abs_tol=6.0e-5, rel_tol=3.0e-5):
                raise AssertionError(
                    "%s[%d]: expected %.9g, got %.9g"
                    % (name, index, left, right)
                )


def check_torch_primitives() -> None:
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError:
        print("PyTorch unavailable; skipping GENEB GPT primitive oracle")
        raise SystemExit(77)
    for family in ("gpt2", "dna"):
        values = weights(family)
        hidden = np.asarray(
            values["transformer.wte.weight"][list(TOKENS), :]
            + values["transformer.wpe.weight"][: len(TOKENS), :],
            dtype=np.float32,
        )
        prefix = "transformer.h.0."
        torch_hidden = torch.from_numpy(hidden)
        torch_normalized = functional.layer_norm(
            torch_hidden,
            (WIDTH,),
            torch.from_numpy(values[prefix + "ln_1.weight"]),
            torch.from_numpy(values[prefix + "ln_1.bias"])
            if family == "gpt2"
            else None,
            float(EPSILON),
        )
        numpy_normalized = layer_norm(
            hidden,
            values[prefix + "ln_1.weight"],
            values[prefix + "ln_1.bias"] if family == "gpt2" else None,
        )
        if not np.allclose(
            torch_normalized.numpy(), numpy_normalized, atol=2.0e-6, rtol=2.0e-6
        ):
            raise AssertionError("PyTorch %s LayerNorm primitive differs" % family)
        weight = torch.from_numpy(values[prefix + "attn.c_attn.weight"])
        bias = (
            torch.from_numpy(values[prefix + "attn.c_attn.bias"])
            if family == "gpt2"
            else None
        )
        if family == "gpt2":
            torch_qkv = torch.addmm(bias, torch_normalized, weight)
            layout = "in-out"
        else:
            torch_qkv = functional.linear(torch_normalized, weight, None)
            layout = "out-in"
        numpy_qkv = linear(numpy_normalized, weight.numpy(), None if bias is None else bias.numpy(), layout)
        if not np.allclose(torch_qkv.numpy(), numpy_qkv, atol=3.0e-6, rtol=3.0e-6):
            raise AssertionError("PyTorch %s fused QKV/layout differs" % family)
        if tuple(chunk.shape[-1] for chunk in torch_qkv.chunk(3, dim=-1)) != (
            WIDTH,
            WIDTH,
            WIDTH,
        ):
            raise AssertionError("PyTorch fused QKV chunk order/width differs")
    probe = torch.tensor([-1.25, -0.2, 0.4, 1.1], dtype=torch.float32)
    torch_gelu = functional.gelu(probe, approximate="tanh").numpy()
    numpy_gelu = np.asarray([gelu_tanh(np.float32(item)) for item in probe.numpy()])
    if not np.allclose(torch_gelu, numpy_gelu, atol=2.0e-6, rtol=2.0e-6):
        raise AssertionError("PyTorch tanh GELU primitive differs")
    print("PyTorch Conv1D/Linear, fused-QKV, LayerNorm, and tanh-GELU oracle passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    if args.require_torch:
        check_torch_primitives()
    process = subprocess.run(
        [str(args.binary), "--dump-json"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    document = json.loads(process.stdout)
    expected = {}  # type: Dict[str, List[float]]
    for family in ("gpt2", "dna"):
        for name, vector in forward(family).items():
            expected[family + "." + name] = vector
    compare(expected, document.get("vectors"))
    if np.max(
        np.abs(
            np.asarray(expected["gpt2.final"], dtype=np.float32)
            - np.asarray(expected["dna.final"], dtype=np.float32)
        )
    ) < np.float32(1.0e-3):
        raise AssertionError("family-specific topology probes are not discriminating")
    print("GENEB GPT independent NumPy oracle passed (%d vectors)" % len(expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
