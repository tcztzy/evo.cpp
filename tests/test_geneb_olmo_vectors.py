#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy/PyTorch oracle for the tiny OLMo-GFM runtime."""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    print("NumPy is required for the GENEB OLMo oracle", file=sys.stderr)
    raise SystemExit(77)


TOKENS = (1, 4, 2, 6)
MASK = (1, 1, 1, 0)
VOCABULARY = 7
WIDTH = 4
LAYERS = 2
HEADS = 2
HEAD_DIMENSION = 2
FUSED_MLP_WIDTH = 12
EPSILON = np.float32(1.0e-5)
ROPE_THETA = np.float32(10000.0)
EXACT_LAYER_NORM_SHA256 = (
    "b3487d70ee4a5dc435b4d4689fc15462c651edaaa43748dbaab4acd7701a6c9b"
)


def requirements(norm_type: str) -> List[Tuple[str, Tuple[int, ...]]]:
    result = [("model.transformer.wte.weight", (VOCABULARY, WIDTH))]
    for layer in range(LAYERS):
        prefix = "model.transformer.blocks.%d." % layer
        result.append((prefix + "att_proj.weight", (WIDTH * 3, WIDTH)))
        if norm_type == "rmsnorm":
            result.append((prefix + "attn_norm.weight", (WIDTH,)))
        result.append((prefix + "attn_out.weight", (WIDTH, WIDTH)))
        if norm_type == "rmsnorm":
            result.append((prefix + "ff_norm.weight", (WIDTH,)))
        result.extend(
            [
                (prefix + "ff_out.weight", (WIDTH, FUSED_MLP_WIDTH // 2)),
                (prefix + "ff_proj.weight", (FUSED_MLP_WIDTH, WIDTH)),
            ]
        )
    if norm_type == "rmsnorm":
        result.append(("model.transformer.ln_f.weight", (WIDTH,)))
    return result


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> np.float32:
    integer = ((tensor_index + 2) * 19 + (element_index + 5) * 11) % 43 - 21
    value = np.float32(np.float32(integer) / np.float32(31.0))
    if "norm.weight" in name or name == "model.transformer.ln_f.weight":
        value = np.float32(np.float32(1.0) + value * np.float32(0.04))
    elif name == "model.transformer.wte.weight":
        value = np.float32(value * np.float32(0.3))
    else:
        value = np.float32(value * np.float32(0.12))
    return value


def fixture_weights(norm_type: str) -> Dict[str, np.ndarray]:
    result = {}  # type: Dict[str, np.ndarray]
    for tensor_index, (name, shape) in enumerate(requirements(norm_type)):
        count = 1
        for dimension in shape:
            count *= dimension
        values = np.asarray(
            [fixture_scalar(name, tensor_index, index) for index in range(count)],
            dtype=np.float32,
        )
        result[name] = values.reshape(shape)
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


def normalize(
    values: np.ndarray, norm_type: str, scale: np.ndarray
) -> np.ndarray:
    rows, width = values.shape
    result = np.zeros_like(values)
    for row in range(rows):
        if norm_type == "layernorm":
            total = np.float32(0.0)
            for column in range(width):
                total = np.float32(total + values[row, column])
            mean = np.float32(total / np.float32(width))
            sum_square = np.float32(0.0)
            for column in range(width):
                centered = np.float32(values[row, column] - mean)
                sum_square = np.float32(
                    sum_square + np.float32(centered * centered)
                )
            variance = np.float32(sum_square / np.float32(width))
            inverse = np.float32(
                np.float32(1.0) / np.sqrt(np.float32(variance + EPSILON))
            )
            for column in range(width):
                result[row, column] = np.float32(
                    np.float32(values[row, column] - mean) * inverse
                )
        else:
            sum_square = np.float32(0.0)
            for column in range(width):
                value = values[row, column]
                sum_square = np.float32(sum_square + np.float32(value * value))
            mean_square = np.float32(sum_square / np.float32(width))
            inverse = np.float32(
                np.float32(1.0) / np.sqrt(np.float32(mean_square + EPSILON))
            )
            for column in range(width):
                result[row, column] = np.float32(
                    np.float32(values[row, column] * inverse) * scale[column]
                )
    return result


def rope(values: np.ndarray) -> np.ndarray:
    rows = values.shape[0]
    result = values.reshape(rows, HEADS, HEAD_DIMENSION).copy()
    half = HEAD_DIMENSION // 2
    for row in range(rows):
        for head in range(HEADS):
            for pair in range(half):
                exponent = np.float32(
                    np.float32(pair * 2) / np.float32(HEAD_DIMENSION)
                )
                angle = np.float32(
                    np.float32(row) / np.power(ROPE_THETA, exponent)
                )
                cosine = np.float32(np.cos(angle))
                sine = np.float32(np.sin(angle))
                first = result[row, head, pair]
                second = result[row, head, pair + half]
                result[row, head, pair] = np.float32(
                    np.float32(first * cosine) - np.float32(second * sine)
                )
                result[row, head, pair + half] = np.float32(
                    np.float32(second * cosine) + np.float32(first * sine)
                )
    return result.reshape(rows, WIDTH)


def causal_attention(
    query: np.ndarray, key: np.ndarray, value: np.ndarray
) -> np.ndarray:
    rows = query.shape[0]
    query_heads = query.reshape(rows, HEADS, HEAD_DIMENSION)
    key_heads = key.reshape(rows, HEADS, HEAD_DIMENSION)
    value_heads = value.reshape(rows, HEADS, HEAD_DIMENSION)
    result = np.zeros_like(query_heads)
    scale = np.float32(
        np.float32(1.0) / np.sqrt(np.float32(HEAD_DIMENSION))
    )
    for row in range(rows):
        for head in range(HEADS):
            scores = []  # type: List[np.float32]
            visible = []  # type: List[int]
            for source in range(row + 1):
                if MASK[source] == 0:
                    continue
                score = np.float32(0.0)
                for dimension in range(HEAD_DIMENSION):
                    score = np.float32(
                        score
                        + np.float32(
                            query_heads[row, head, dimension]
                            * key_heads[source, head, dimension]
                        )
                    )
                scores.append(np.float32(score * scale))
                visible.append(source)
            maximum = max(scores)
            probabilities = []  # type: List[np.float32]
            denominator = np.float32(0.0)
            for score in scores:
                probability = np.float32(np.exp(np.float32(score - maximum)))
                probabilities.append(probability)
                denominator = np.float32(denominator + probability)
            probabilities = [
                np.float32(probability / denominator)
                for probability in probabilities
            ]
            for dimension in range(HEAD_DIMENSION):
                total = np.float32(0.0)
                for index, source in enumerate(visible):
                    total = np.float32(
                        total
                        + np.float32(
                            probabilities[index]
                            * value_heads[source, head, dimension]
                        )
                    )
                result[row, head, dimension] = total
    return result.reshape(rows, WIDTH)


def silu(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    for index, value in enumerate(values.reshape(-1)):
        result.reshape(-1)[index] = np.float32(
            value / np.float32(np.float32(1.0) + np.exp(np.float32(-value)))
        )
    return result


def forward(
    norm_type: str, qkv_layout: str = "q-k-v", swiglu_layout: str = "x-gate"
) -> Dict[str, List[float]]:
    weights = fixture_weights(norm_type)
    hidden = weights["model.transformer.wte.weight"][list(TOKENS), :].copy()
    vectors = {"capture0": hidden.reshape(-1).tolist()}
    for layer in range(LAYERS):
        prefix = "model.transformer.blocks.%d." % layer
        attn_scale = (
            weights[prefix + "attn_norm.weight"]
            if norm_type == "rmsnorm"
            else None
        )
        normalized = normalize(hidden, norm_type, attn_scale)
        fused_qkv = linear(normalized, weights[prefix + "att_proj.weight"])
        chunks = np.split(fused_qkv, 3, axis=1)
        if qkv_layout == "q-k-v":
            query, key, value = chunks
        else:
            key, query, value = chunks
        attended = causal_attention(rope(query), rope(key), value)
        hidden = np.asarray(
            hidden + linear(attended, weights[prefix + "attn_out.weight"]),
            dtype=np.float32,
        )
        ff_scale = (
            weights[prefix + "ff_norm.weight"]
            if norm_type == "rmsnorm"
            else None
        )
        normalized = normalize(hidden, norm_type, ff_scale)
        fused_ff = linear(normalized, weights[prefix + "ff_proj.weight"])
        first, second = np.split(fused_ff, 2, axis=1)
        if swiglu_layout == "x-gate":
            x_values, gate = first, second
        else:
            gate, x_values = first, second
        activated = np.asarray(silu(gate) * x_values, dtype=np.float32)
        hidden = np.asarray(
            hidden + linear(activated, weights[prefix + "ff_out.weight"]),
            dtype=np.float32,
        )
        if layer + 1 < LAYERS:
            vectors["capture%d" % (layer + 1)] = hidden.reshape(-1).tolist()
    final_scale = (
        weights["model.transformer.ln_f.weight"]
        if norm_type == "rmsnorm"
        else None
    )
    final_hidden = normalize(hidden, norm_type, final_scale)
    vectors["capture%d" % LAYERS] = final_hidden.reshape(-1).tolist()
    vectors["final"] = final_hidden.reshape(-1).tolist()
    vectors["pooled"] = np.asarray(
        final_hidden[:3, :].sum(axis=0, dtype=np.float32) / np.float32(3.0),
        dtype=np.float32,
    ).tolist()
    return vectors


def require_torch_primitives() -> None:
    try:
        import torch
        import torch.nn.functional as torch_functional
    except ModuleNotFoundError:
        print("PyTorch unavailable; skipping PyTorch OLMo primitive oracle")
        raise SystemExit(77)
    weights = fixture_weights("rmsnorm")
    hidden = weights["model.transformer.wte.weight"][list(TOKENS), :].copy()
    prefix = "model.transformer.blocks.0."
    torch_hidden = torch.from_numpy(hidden)
    torch_qkv = torch_functional.linear(
        torch_hidden, torch.from_numpy(weights[prefix + "att_proj.weight"])
    )
    numpy_qkv = linear(hidden, weights[prefix + "att_proj.weight"])
    if not np.allclose(torch_qkv.numpy(), numpy_qkv, atol=2.0e-6, rtol=2.0e-6):
        raise AssertionError("PyTorch fused QKV primitive differs from NumPy oracle")
    first, gate = torch_qkv.chunk(2, dim=-1)
    torch_swiglu = torch_functional.silu(gate) * first
    numpy_swiglu = silu(numpy_qkv[:, WIDTH * 3 // 2 :]) * numpy_qkv[:, : WIDTH * 3 // 2]
    if not np.allclose(
        torch_swiglu.numpy(), numpy_swiglu, atol=2.0e-6, rtol=2.0e-6
    ):
        raise AssertionError("PyTorch OLMo x/gate SwiGLU primitive differs")
    print("PyTorch fused-QKV and x/gate SwiGLU primitive oracle passed")


def verify_exact_layer_norm(binary: Path, require_torch: bool) -> None:
    process = subprocess.run(
        [str(binary), "--dump-exact-layer-norm-bits"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode == 77:
        print("exact Omni-DNA-1B LayerNorm correctly unavailable on this host")
        return
    if process.returncode != 0:
        raise AssertionError(
            "native exact LayerNorm vector failed:\n" + process.stderr
        )
    document = json.loads(process.stdout)
    bits = document.get("bits")
    if (
        not isinstance(bits, list)
        or len(bits) != 4096
        or any(not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF for value in bits)
    ):
        raise AssertionError("native exact LayerNorm bit vector is malformed")
    raw = b"".join(struct.pack("<I", value) for value in bits)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != EXACT_LAYER_NORM_SHA256:
        raise AssertionError(
            "pinned Torch 2.1.2 Apple-arm64 LayerNorm bits differ: got %s"
            % actual_sha256
        )

    if require_torch:
        import torch
        import torch.nn.functional as torch_functional

        if torch.__version__.split("+")[0] == "2.1.2":
            values = np.asarray(
                [
                    np.float32(
                        np.float32(((index + 3) * 37) % 257 - 128)
                        / np.float32(31.0)
                    )
                    for index in range(4096)
                ],
                dtype=np.float32,
            ).reshape(2, 2048)
            expected = torch_functional.layer_norm(
                torch.from_numpy(values), (2048,), weight=None, bias=None,
                eps=1.0e-5
            ).contiguous().numpy().astype("<f4", copy=False).tobytes(order="C")
            if raw != expected:
                raise AssertionError(
                    "native exact LayerNorm is not bit-equal to pinned Torch 2.1.2"
                )
    print("validated pinned exact Omni-DNA-1B LayerNorm bit vector")


def compare(expected: Mapping[str, Sequence[float]], actual: object) -> None:
    if not isinstance(actual, dict):
        raise AssertionError("native output vectors must be an object")
    if set(expected) != set(actual):
        raise AssertionError(
            "vector names differ: expected=%s actual=%s"
            % (sorted(expected), sorted(actual))
        )
    for name in sorted(expected):
        wanted = expected[name]
        got = actual[name]
        if not isinstance(got, list) or len(got) != len(wanted):
            raise AssertionError("%s vector length differs" % name)
        for index, pair in enumerate(zip(wanted, got)):
            wanted_value, got_value = pair
            if not math.isclose(
                wanted_value, got_value, abs_tol=5.0e-5, rel_tol=2.0e-5
            ):
                raise AssertionError(
                    "%s[%d]: expected %.9g, got %.9g"
                    % (name, index, wanted_value, got_value)
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    if args.require_torch:
        require_torch_primitives()
    verify_exact_layer_norm(args.binary, args.require_torch)
    process = subprocess.run(
        [str(args.binary), "--dump-json"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    document = json.loads(process.stdout)
    expected = {}  # type: Dict[str, List[float]]
    for norm_type in ("layernorm", "rmsnorm"):
        for name, values in forward(norm_type).items():
            expected[norm_type + "." + name] = values
    compare(expected, document.get("vectors"))

    canonical = forward("rmsnorm")
    swapped_qkv = forward("rmsnorm", qkv_layout="k-q-v")
    swapped_swiglu = forward("rmsnorm", swiglu_layout="gate-x")
    if np.max(
        np.abs(
            np.asarray(canonical["final"], dtype=np.float32)
            - np.asarray(swapped_qkv["final"], dtype=np.float32)
        )
    ) <= np.float32(1.0e-6):
        raise AssertionError("tiny fixture does not distinguish fused QKV order")
    if np.max(
        np.abs(
            np.asarray(canonical["final"], dtype=np.float32)
            - np.asarray(swapped_swiglu["final"], dtype=np.float32)
        )
    ) <= np.float32(1.0e-6):
        raise AssertionError("tiny fixture does not distinguish x/gate SwiGLU order")
    print("validated %d independent OLMo-GFM vectors" % len(expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
