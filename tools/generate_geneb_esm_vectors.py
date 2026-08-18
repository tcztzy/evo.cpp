#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the independent tiny NT/Agro-NT ESM encoder oracle.

The implementation is deliberately NumPy-only and spells out the pinned HF
operation order: padding-cumsum absolute positions, ESM token dropout,
pre-LayerNorm residual blocks, query scaling before split-half RoPE, full
bidirectional key masking, exact-erf GELU or NT-v2 SwiGLU, final LayerNorm, and
GENEB attention-mask mean pooling. It is Python 3.8 grammar compatible.
"""

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    print("NumPy is required to generate GENEB ESM oracle vectors", file=sys.stderr)
    raise SystemExit(77)


Topology = Dict[str, Any]
TensorMap = Dict[str, np.ndarray]


def absolute_topology() -> Topology:
    return {
        "vocab": 8,
        "width": 6,
        "layers": 2,
        "heads": 2,
        "head_dim": 3,
        "inner": 8,
        "max_tokens": 6,
        "max_positions": 8,
        "epsilon": np.float32(1.0e-5),
        "rope_base": np.float32(10000.0),
        "position": "absolute",
        "activation": "gelu",
        "attention_bias": True,
        "ffn_bias": True,
        "token_dropout": True,
        "pad_id": 1,
        "mask_id": 2,
        "cls_id": 3,
    }


def rotary_topology() -> Topology:
    result = dict(absolute_topology())
    result.update(
        {
            "width": 8,
            "head_dim": 4,
            "inner": 7,
            "position": "rotary",
            "activation": "swiglu",
            "ffn_bias": False,
            "token_dropout": False,
        }
    )
    return result


def tensor_requirements(topology: Mapping[str, Any]) -> List[Tuple[str, Tuple[int, ...]]]:
    width = int(topology["width"])
    result = [
        ("esm.embeddings.word_embeddings.weight", (int(topology["vocab"]), width))
    ]  # type: List[Tuple[str, Tuple[int, ...]]]
    if topology["position"] == "absolute":
        result.append(
            (
                "esm.embeddings.position_embeddings.weight",
                (int(topology["max_positions"]), width),
            )
        )
    for layer in range(int(topology["layers"])):
        prefix = "esm.encoder.layer.%d." % layer
        result.extend(
            [
                (prefix + "attention.LayerNorm.weight", (width,)),
                (prefix + "attention.LayerNorm.bias", (width,)),
            ]
        )
        for projection in ("query", "key", "value"):
            result.append(
                (prefix + "attention.self." + projection + ".weight", (width, width))
            )
            if topology["attention_bias"]:
                result.append(
                    (prefix + "attention.self." + projection + ".bias", (width,))
                )
        result.append((prefix + "attention.output.dense.weight", (width, width)))
        if topology["attention_bias"]:
            result.append((prefix + "attention.output.dense.bias", (width,)))
        result.extend(
            [
                (prefix + "LayerNorm.weight", (width,)),
                (prefix + "LayerNorm.bias", (width,)),
            ]
        )
        fused_inner = int(topology["inner"])
        if topology["activation"] == "swiglu":
            fused_inner *= 2
        result.append((prefix + "intermediate.dense.weight", (fused_inner, width)))
        if topology["ffn_bias"]:
            result.append((prefix + "intermediate.dense.bias", (fused_inner,)))
        result.append(
            (prefix + "output.dense.weight", (width, int(topology["inner"])))
        )
        if topology["ffn_bias"]:
            result.append((prefix + "output.dense.bias", (width,)))
    result.extend(
        [
            ("esm.encoder.emb_layer_norm_after.weight", (width,)),
            ("esm.encoder.emb_layer_norm_after.bias", (width,)),
        ]
    )
    return result


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> np.float32:
    integer = ((tensor_index + 3) * 19 + (element_index + 5) * 11) % 43 - 21
    value = np.float32(np.float32(integer) / np.float32(31.0))
    if "LayerNorm.weight" in name or "layer_norm_after.weight" in name:
        value = np.float32(np.float32(1.0) + np.float32(value * np.float32(0.04)))
    elif name.endswith(".bias"):
        value = np.float32(value * np.float32(0.04))
    elif "embeddings." in name:
        value = np.float32(value * np.float32(0.12))
    else:
        value = np.float32(value * np.float32(0.16))
    return value


def fixture_weights(topology: Mapping[str, Any]) -> TensorMap:
    result = {}  # type: TensorMap
    for tensor_index, (name, shape) in enumerate(tensor_requirements(topology)):
        count = 1
        for dimension in shape:
            count *= dimension
        values = [
            fixture_scalar(name, tensor_index, element_index)
            for element_index in range(count)
        ]
        result[name] = np.asarray(values, dtype=np.float32).reshape(shape)
    return result


def linear(
    values: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]
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
            result[row, target] = total
    return result


def layer_norm(
    values: np.ndarray, scale: np.ndarray, bias: np.ndarray, epsilon: np.float32
) -> np.ndarray:
    rows, width = values.shape
    result = np.zeros_like(values)
    for row in range(rows):
        total = np.float32(0.0)
        for column in range(width):
            total = np.float32(total + values[row, column])
        mean = np.float32(total / np.float32(width))
        sum_squares = np.float32(0.0)
        for column in range(width):
            centered = np.float32(values[row, column] - mean)
            sum_squares = np.float32(
                sum_squares + np.float32(centered * centered)
            )
        variance = np.float32(sum_squares / np.float32(width))
        inverse = np.float32(
            np.float32(1.0) / np.float32(np.sqrt(np.float32(variance + epsilon)))
        )
        for column in range(width):
            centered = np.float32(values[row, column] - mean)
            normalized = np.float32(centered * inverse)
            weighted = np.float32(normalized * scale[column])
            result[row, column] = np.float32(weighted + bias[column])
    return result


def apply_rotary(values: np.ndarray, topology: Mapping[str, Any]) -> np.ndarray:
    rows = values.shape[0]
    heads = int(topology["heads"])
    head_dim = int(topology["head_dim"])
    half = head_dim // 2
    result = values.reshape(rows, heads, head_dim).copy()
    for row in range(rows):
        for pair in range(half):
            exponent = np.float32(np.float32(pair * 2) / np.float32(head_dim))
            denominator = np.float32(
                np.power(np.float32(topology["rope_base"]), exponent)
            )
            angle = np.float32(np.float32(row) / denominator)
            cosine = np.float32(np.cos(angle))
            sine = np.float32(np.sin(angle))
            for head in range(heads):
                first = np.float32(result[row, head, pair])
                second = np.float32(result[row, head, pair + half])
                result[row, head, pair] = np.float32(
                    np.float32(first * cosine) - np.float32(second * sine)
                )
                result[row, head, pair + half] = np.float32(
                    np.float32(second * cosine) + np.float32(first * sine)
                )
    return result.reshape(rows, heads * head_dim)


def attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Sequence[int],
    topology: Mapping[str, Any],
) -> np.ndarray:
    rows = query.shape[0]
    heads = int(topology["heads"])
    head_dim = int(topology["head_dim"])
    query_heads = query.reshape(rows, heads, head_dim)
    key_heads = key.reshape(rows, heads, head_dim)
    value_heads = value.reshape(rows, heads, head_dim)
    result = np.zeros_like(query_heads)
    valid_sources = [index for index, item in enumerate(mask) if item]
    for row in range(rows):
        for head in range(heads):
            scores = []  # type: List[np.float32]
            for source in valid_sources:
                score = np.float32(0.0)
                for dimension in range(head_dim):
                    product = np.float32(
                        query_heads[row, head, dimension]
                        * key_heads[source, head, dimension]
                    )
                    score = np.float32(score + product)
                scores.append(score)
            maximum = max(scores)
            exponentials = []  # type: List[np.float32]
            denominator = np.float32(0.0)
            for score in scores:
                item = np.float32(np.exp(np.float32(score - maximum)))
                exponentials.append(item)
                denominator = np.float32(denominator + item)
            probabilities = [
                np.float32(item / denominator) for item in exponentials
            ]
            for dimension in range(head_dim):
                total = np.float32(0.0)
                for source_index, source in enumerate(valid_sources):
                    product = np.float32(
                        probabilities[source_index]
                        * value_heads[source, head, dimension]
                    )
                    total = np.float32(total + product)
                result[row, head, dimension] = total
    return result.reshape(rows, heads * head_dim)


def gelu(value: np.float32) -> np.float32:
    argument = np.float32(value * np.float32(0.70710678118654752440))
    error_function = np.float32(math.erf(float(argument)))
    first = np.float32(value * np.float32(0.5))
    return np.float32(first * np.float32(np.float32(1.0) + error_function))


def silu(value: np.float32) -> np.float32:
    if value >= np.float32(0.0):
        denominator = np.float32(
            np.float32(1.0) + np.float32(np.exp(np.float32(-value)))
        )
        return np.float32(value / denominator)
    exponential = np.float32(np.exp(value))
    denominator = np.float32(np.float32(1.0) + exponential)
    return np.float32(np.float32(value * exponential) / denominator)


def run_fixture(topology: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    weights = fixture_weights(topology)
    if topology["token_dropout"]:
        tokens = [3, 4, 2, 5, 1, 1]
    else:
        tokens = [3, 4, 6, 5, 1, 1]
    mask = [1, 1, 1, 1, 0, 0]
    width = int(topology["width"])
    hidden = weights["esm.embeddings.word_embeddings.weight"][tokens].copy()
    if topology["token_dropout"]:
        for row, token in enumerate(tokens):
            if token == topology["mask_id"]:
                hidden[row, :] = np.float32(0.0)
        mask_count = sum(1 for token in tokens if token == topology["mask_id"])
        valid_count = sum(mask)
        observed = np.float32(np.float32(mask_count) / np.float32(valid_count))
        token_scale = np.float32(
            np.float32(0.88) / np.float32(np.float32(1.0) - observed)
        )
        hidden = np.asarray(hidden * token_scale, dtype=np.float32)
    if topology["position"] == "absolute":
        position_table = weights["esm.embeddings.position_embeddings.weight"]
        non_padding = 0
        for row, valid in enumerate(mask):
            position = int(topology["pad_id"])
            if valid:
                non_padding += 1
                position += non_padding
            hidden[row, :] = np.asarray(
                hidden[row, :] + position_table[position, :], dtype=np.float32
            )
    for row, valid in enumerate(mask):
        if not valid:
            hidden[row, :] = np.float32(0.0)

    vectors = {"capture_0": hidden.copy()}  # type: Dict[str, np.ndarray]
    query_scale = np.float32(
        np.float32(1.0) / np.float32(np.sqrt(np.float32(topology["head_dim"])))
    )
    for layer in range(int(topology["layers"])):
        prefix = "esm.encoder.layer.%d." % layer
        normalized = layer_norm(
            hidden,
            weights[prefix + "attention.LayerNorm.weight"],
            weights[prefix + "attention.LayerNorm.bias"],
            np.float32(topology["epsilon"]),
        )
        q_bias = (
            weights[prefix + "attention.self.query.bias"]
            if topology["attention_bias"]
            else None
        )
        k_bias = (
            weights[prefix + "attention.self.key.bias"]
            if topology["attention_bias"]
            else None
        )
        v_bias = (
            weights[prefix + "attention.self.value.bias"]
            if topology["attention_bias"]
            else None
        )
        query = linear(
            normalized, weights[prefix + "attention.self.query.weight"], q_bias
        )
        key = linear(
            normalized, weights[prefix + "attention.self.key.weight"], k_bias
        )
        value = linear(
            normalized, weights[prefix + "attention.self.value.weight"], v_bias
        )
        query = np.asarray(query * query_scale, dtype=np.float32)
        if topology["position"] == "rotary":
            query = apply_rotary(query, topology)
            key = apply_rotary(key, topology)
        attended = attention(query, key, value, mask, topology)
        output_bias = (
            weights[prefix + "attention.output.dense.bias"]
            if topology["attention_bias"]
            else None
        )
        projected = linear(
            attended,
            weights[prefix + "attention.output.dense.weight"],
            output_bias,
        )
        hidden = np.asarray(hidden + projected, dtype=np.float32)

        normalized = layer_norm(
            hidden,
            weights[prefix + "LayerNorm.weight"],
            weights[prefix + "LayerNorm.bias"],
            np.float32(topology["epsilon"]),
        )
        intermediate_bias = (
            weights[prefix + "intermediate.dense.bias"]
            if topology["ffn_bias"]
            else None
        )
        intermediate = linear(
            normalized,
            weights[prefix + "intermediate.dense.weight"],
            intermediate_bias,
        )
        inner = int(topology["inner"])
        activated = np.zeros((len(tokens), inner), dtype=np.float32)
        if topology["activation"] == "gelu":
            for row in range(len(tokens)):
                for column in range(inner):
                    activated[row, column] = gelu(intermediate[row, column])
        else:
            for row in range(len(tokens)):
                for column in range(inner):
                    activated[row, column] = np.float32(
                        silu(intermediate[row, column])
                        * intermediate[row, inner + column]
                    )
        down_bias = (
            weights[prefix + "output.dense.bias"]
            if topology["ffn_bias"]
            else None
        )
        feed_forward = linear(
            activated, weights[prefix + "output.dense.weight"], down_bias
        )
        hidden = np.asarray(hidden + feed_forward, dtype=np.float32)
        if layer + 1 < int(topology["layers"]):
            vectors["capture_%d" % (layer + 1)] = hidden.copy()

    final_hidden = layer_norm(
        hidden,
        weights["esm.encoder.emb_layer_norm_after.weight"],
        weights["esm.encoder.emb_layer_norm_after.bias"],
        np.float32(topology["epsilon"]),
    )
    vectors["capture_%d" % int(topology["layers"])] = final_hidden.copy()
    vectors["final_hidden"] = final_hidden.copy()
    pooled = np.zeros((width,), dtype=np.float32)
    valid_count = 0
    for row, valid in enumerate(mask):
        if not valid:
            continue
        valid_count += 1
        for column in range(width):
            pooled[column] = np.float32(
                pooled[column] + final_hidden[row, column]
            )
    inverse = np.float32(np.float32(1.0) / np.float32(valid_count))
    pooled = np.asarray(pooled * inverse, dtype=np.float32)
    vectors["pooled"] = pooled
    return vectors


def make_document() -> Dict[str, Any]:
    vectors = {}  # type: Dict[str, List[float]]
    order = []  # type: List[str]
    for prefix, topology in (
        ("absolute", absolute_topology()),
        ("rotary", rotary_topology()),
    ):
        result = run_fixture(topology)
        for suffix in (
            "capture_0",
            "capture_1",
            "capture_2",
            "final_hidden",
            "pooled",
        ):
            name = prefix + "." + suffix
            order.append(name)
            vectors[name] = [float(value) for value in result[suffix].reshape(-1)]
    digest = hashlib.sha256()
    for name in order:
        for value in vectors[name]:
            digest.update(struct.pack("<f", value))
    return {
        "schema_version": 1,
        "oracle": "independent-numpy-hf-esm-reference",
        "semantics": {
            "absolute_position_ids": "pad-id-or-pad-id-plus-valid-cumsum",
            "token_dropout": "zero-mask-then-0.88/(1-observed)",
            "normalization": "pre-layernorm-plus-final-layernorm",
            "attention": "query-scale-before-rope-bidirectional-key-mask",
            "rope": "split-half-base-10000",
            "pooling": "attention-mask-mean-including-cls",
        },
        "tolerance": {"atol": 3.0e-5, "rtol": 3.0e-5},
        "vector_order": order,
        "vectors": vectors,
        "vector_sha256": digest.hexdigest(),
    }


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    document = make_document()
    payload = canonical_bytes(document)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        print("wrote %s" % args.output)
        return 0
    try:
        current = args.check.read_bytes()
    except OSError as error:
        print("cannot read committed oracle: %s" % error, file=sys.stderr)
        return 1
    if current != payload:
        print(
            "committed GENEB ESM oracle differs; regenerate with --output",
            file=sys.stderr,
        )
        return 1
    print("GENEB ESM oracle is current: %s" % document["vector_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
