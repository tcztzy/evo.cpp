#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the independent NumPy-F32 DeepGene RoFormer tiny oracle."""

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
    print("NumPy is required for the GENEB RoFormer oracle", file=sys.stderr)
    raise SystemExit(77)


Shape = Tuple[int, ...]


def topology() -> Dict[str, object]:
    return {
        "vocab": 8,
        "width": 4,
        "layers": 2,
        "heads": 2,
        "head_dim": 2,
        "inner": 6,
        "epsilon": np.float32(1.0e-6),
        "rope_base": np.float32(10000.0),
    }


def requirements(topo: Mapping[str, object]) -> List[Tuple[str, Shape]]:
    vocab = int(topo["vocab"])
    width = int(topo["width"])
    inner = int(topo["inner"])
    result = [
        ("roformer.embeddings.word_embeddings.weight", (vocab, width)),
        ("roformer.embeddings.token_type_embeddings.weight", (2, width)),
        ("roformer.embeddings.LayerNorm.weight", (width,)),
        ("roformer.embeddings.LayerNorm.bias", (width,)),
    ]
    for layer in range(int(topo["layers"])):
        prefix = "roformer.encoder.layer.{}.".format(layer)
        for projection in ("query", "key", "value"):
            result.extend(
                [
                    (prefix + "attention.self." + projection + ".weight", (width, width)),
                    (prefix + "attention.self." + projection + ".bias", (width,)),
                ]
            )
        result.extend(
            [
                (prefix + "attention.output.dense.weight", (width, width)),
                (prefix + "attention.output.dense.bias", (width,)),
                (prefix + "attention.output.LayerNorm.weight", (width,)),
                (prefix + "attention.output.LayerNorm.bias", (width,)),
                (prefix + "intermediate.dense.weight", (inner, width)),
                (prefix + "intermediate.dense.bias", (inner,)),
                (prefix + "output.dense.weight", (width, inner)),
                (prefix + "output.dense.bias", (width,)),
                (prefix + "output.LayerNorm.weight", (width,)),
                (prefix + "output.LayerNorm.bias", (width,)),
            ]
        )
    return result


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> np.float32:
    integer = ((tensor_index + 3) * 13 + (element_index + 5) * 7) % 31 - 15
    base = np.float32(integer / 100.0)
    if "LayerNorm.weight" in name:
        return np.float32(1.0) + base * np.float32(0.1)
    if name == "roformer.embeddings.word_embeddings.weight":
        return base * np.float32(0.6)
    if name == "roformer.embeddings.token_type_embeddings.weight":
        return base * np.float32(0.2)
    if name.endswith(".bias"):
        return base * np.float32(0.15)
    return base * np.float32(0.35)


def weights(topo: Mapping[str, object]) -> Dict[str, np.ndarray]:
    result = {}  # type: Dict[str, np.ndarray]
    for tensor_index, (name, shape) in enumerate(requirements(topo)):
        count = math.prod(shape)
        result[name] = np.asarray(
            [fixture_scalar(name, tensor_index, index) for index in range(count)],
            dtype=np.float32,
        ).reshape(shape)
    return result


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(x @ weight.T + bias, dtype=np.float32)


def layer_norm(
    x: np.ndarray, scale: np.ndarray, bias: np.ndarray, epsilon: np.float32
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = np.asarray(x - mean, dtype=np.float32)
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    normalized = centered / np.sqrt(variance + epsilon, dtype=np.float32)
    return np.asarray(normalized * scale + bias, dtype=np.float32)


def rope(x: np.ndarray, positions: Sequence[int], topo: Mapping[str, object]) -> np.ndarray:
    result = x.reshape(len(positions), int(topo["heads"]), int(topo["head_dim"])).copy()
    pairs = int(topo["head_dim"]) // 2
    for row, position in enumerate(positions):
        for head in range(int(topo["heads"])):
            for pair in range(pairs):
                exponent = np.float32(2.0 * pair / int(topo["head_dim"]))
                angle = np.float32(position) / np.float32(
                    np.power(np.float32(topo["rope_base"]), exponent)
                )
                cosine = np.float32(np.cos(angle))
                sine = np.float32(np.sin(angle))
                first = result[row, head, pair * 2]
                second = result[row, head, pair * 2 + 1]
                result[row, head, pair * 2] = np.float32(first * cosine - second * sine)
                result[row, head, pair * 2 + 1] = np.float32(second * cosine + first * sine)
    return result


def attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray,
    topo: Mapping[str, object],
) -> np.ndarray:
    rows = query.shape[0]
    heads = int(topo["heads"])
    head_dim = int(topo["head_dim"])
    output = np.zeros((rows, heads, head_dim), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    for row in range(rows):
        for head in range(heads):
            scores = np.asarray(
                [
                    np.sum(query[row, head] * key[key_row, head], dtype=np.float32)
                    * scale
                    for key_row in range(rows)
                    if mask[key_row] != 0
                ],
                dtype=np.float32,
            )
            probabilities = np.exp(scores - np.max(scores), dtype=np.float32)
            probabilities = np.asarray(
                probabilities / np.sum(probabilities, dtype=np.float32),
                dtype=np.float32,
            )
            active = [index for index in range(rows) if mask[index] != 0]
            for probability, key_row in zip(probabilities, active):
                output[row, head] += probability * value[key_row, head]
    return output.reshape(rows, heads * head_dim)


def gelu(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    result = np.empty_like(flat)
    for index, value in enumerate(flat):
        result[index] = np.float32(
            np.float32(0.5)
            * value
            * np.float32(1.0 + math.erf(float(value) / math.sqrt(2.0)))
        )
    return result.reshape(x.shape)


def oracle() -> Dict[str, object]:
    topo = topology()
    state = weights(topo)
    tokens = np.asarray([1, 2, 5, 6, 3], dtype=np.int64)
    mask = np.asarray([1, 1, 1, 1, 0], dtype=np.uint8)
    width = int(topo["width"])
    hidden = np.asarray(
        state["roformer.embeddings.word_embeddings.weight"][tokens]
        + state["roformer.embeddings.token_type_embeddings.weight"][0],
        dtype=np.float32,
    )
    hidden = layer_norm(
        hidden,
        state["roformer.embeddings.LayerNorm.weight"],
        state["roformer.embeddings.LayerNorm.bias"],
        np.float32(topo["epsilon"]),
    )
    captures = [{"layer": 0, "values": hidden.reshape(-1).tolist()}]
    positions = [index if mask[index] else len(tokens) - 1 for index in range(len(tokens))]
    for layer in range(int(topo["layers"])):
        prefix = "roformer.encoder.layer.{}.".format(layer)
        query = linear(
            hidden,
            state[prefix + "attention.self.query.weight"],
            state[prefix + "attention.self.query.bias"],
        )
        key = linear(
            hidden,
            state[prefix + "attention.self.key.weight"],
            state[prefix + "attention.self.key.bias"],
        )
        value = linear(
            hidden,
            state[prefix + "attention.self.value.weight"],
            state[prefix + "attention.self.value.bias"],
        ).reshape(len(tokens), int(topo["heads"]), int(topo["head_dim"]))
        context = attention(rope(query, positions, topo), rope(key, positions, topo), value, mask, topo)
        update = linear(
            context,
            state[prefix + "attention.output.dense.weight"],
            state[prefix + "attention.output.dense.bias"],
        )
        hidden = layer_norm(
            np.asarray(hidden + update, dtype=np.float32),
            state[prefix + "attention.output.LayerNorm.weight"],
            state[prefix + "attention.output.LayerNorm.bias"],
            np.float32(topo["epsilon"]),
        )
        intermediate = gelu(
            linear(
                hidden,
                state[prefix + "intermediate.dense.weight"],
                state[prefix + "intermediate.dense.bias"],
            )
        )
        update = linear(
            intermediate,
            state[prefix + "output.dense.weight"],
            state[prefix + "output.dense.bias"],
        )
        hidden = layer_norm(
            np.asarray(hidden + update, dtype=np.float32),
            state[prefix + "output.LayerNorm.weight"],
            state[prefix + "output.LayerNorm.bias"],
            np.float32(topo["epsilon"]),
        )
        captures.append({"layer": layer + 1, "values": hidden.reshape(-1).tolist()})
    pooled = np.mean(hidden[mask != 0], axis=0, dtype=np.float32)
    return {
        "schema": "geneb-roformer-tiny-oracle-v1",
        "generator": "independent-numpy-f32",
        "rows": len(tokens),
        "width": width,
        "captures": captures,
        "final_hidden": hidden.reshape(-1).tolist(),
        "pooled": pooled.tolist(),
        "payload_first_tokens": [1, 2, 5, 6],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(oracle(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
