#!/usr/bin/env python3
"""Generate independent NumPy tiny oracles for LucaOne and Genomics-FM."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def requirements(variant: str) -> List[Tuple[str, Tuple[int, ...]]]:
    vocab = 10 if variant == "lucaone" else 12
    width, layers, inner, positions = 4, 2, 6, 8
    result = []  # type: List[Tuple[str, Tuple[int, ...]]]
    if variant == "lucaone":
        result.extend(
            [
                ("lucaone.embeddings.embed_tokens.weight", (vocab, width)),
                ("lucaone.embeddings.embed_type.weight", (2, width)),
            ]
        )
        for layer in range(layers):
            prefix = "lucaone.encoder.layers.{}.".format(layer)
            result.extend(
                [
                    (prefix + "pre_layer_norm.weight", (width,)),
                    (prefix + "pre_layer_norm.bias", (width,)),
                ]
            )
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                result.extend(
                    [
                        (prefix + "self_attn." + projection + ".weight", (width, width)),
                        (prefix + "self_attn." + projection + ".bias", (width,)),
                    ]
                )
            result.extend(
                [
                    (prefix + "self_attn.rot_emb.inv_freq", (1,)),
                    (prefix + "post_layer_norm.weight", (width,)),
                    (prefix + "post_layer_norm.bias", (width,)),
                    (prefix + "fc1.weight", (inner, width)),
                    (prefix + "fc1.bias", (inner,)),
                    (prefix + "fc2.weight", (width, inner)),
                    (prefix + "fc2.bias", (width,)),
                ]
            )
        result.extend(
            [
                ("lucaone.encoder.last_layer_norm.weight", (width,)),
                ("lucaone.encoder.last_layer_norm.bias", (width,)),
            ]
        )
        return result

    result.extend(
        [
            ("bert.embeddings.word_embeddings.weight", (vocab, width)),
            ("bert.embeddings.position_embeddings.weight", (positions, width)),
            ("bert.embeddings.token_type_embeddings.weight", (2, width)),
            ("bert.embeddings.LayerNorm.weight", (width,)),
            ("bert.embeddings.LayerNorm.bias", (width,)),
        ]
    )
    for layer in range(layers):
        prefix = "bert.encoder.layer.{}.".format(layer)
        result.extend(
            [
                (prefix + "attention.self.Wqkv.weight", (3 * width, width)),
                (prefix + "attention.self.Wqkv.bias", (3 * width,)),
                (prefix + "attention.output.dense.weight", (width, width)),
                (prefix + "attention.output.dense.bias", (width,)),
                (prefix + "attention.output.LayerNorm.weight", (width,)),
                (prefix + "attention.output.LayerNorm.bias", (width,)),
                (prefix + "mlp.gated_layers.weight", (2 * inner, width)),
                (prefix + "mlp.wo.weight", (width, inner)),
                (prefix + "mlp.wo.bias", (width,)),
                (prefix + "mlp.layernorm.weight", (width,)),
                (prefix + "mlp.layernorm.bias", (width,)),
            ]
        )
    return result


def tiny_weights(variant: str) -> Dict[str, np.ndarray]:
    output = {}  # type: Dict[str, np.ndarray]
    for tensor_index, (name, shape) in enumerate(requirements(variant)):
        elements = int(np.prod(shape))
        if "LayerNorm.weight" in name or "layer_norm.weight" in name or "layernorm.weight" in name:
            values = [0.93 + 0.02 * (index % 5) for index in range(elements)]
        elif "rot_emb.inv_freq" in name:
            values = [1.0]
        elif ".bias" in name:
            values = [(((tensor_index + index) % 7) - 3) * 0.004 for index in range(elements)]
        else:
            values = [(((tensor_index * 11 + index * 7) % 19) - 9) * 0.013 for index in range(elements)]
        output[name] = np.asarray(values, dtype=np.float32).reshape(shape)
    return output


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, epsilon: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = np.asarray(x - mean, dtype=np.float32)
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    return np.asarray(centered / np.sqrt(variance + np.float32(epsilon)) * weight + bias, dtype=np.float32)


def linear(x: np.ndarray, weight: np.ndarray, bias: Any = None) -> np.ndarray:
    result = np.asarray(x @ weight.T, dtype=np.float32)
    if bias is not None:
        result = np.asarray(result + bias, dtype=np.float32)
    return result


def gelu(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    result = np.empty_like(flat)
    inverse_sqrt_two = np.float32(0.70710678118654752440)
    for index, value in enumerate(flat):
        result[index] = np.float32(value * np.float32(0.5) * np.float32(1.0 + math.erf(float(value * inverse_sqrt_two))))
    return result.reshape(x.shape)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray, pre_scaled: bool) -> np.ndarray:
    rows, heads, head_dim = q.shape
    result = np.zeros_like(q)
    scale = np.float32(1.0 / math.sqrt(head_dim))
    visible = np.nonzero(mask)[0]
    for row in range(rows):
        if not pre_scaled and mask[row] == 0:
            continue
        for head in range(heads):
            scores = []
            for source in visible:
                score = np.float32(np.dot(q[row, head], k[source, head]))
                scores.append(score if pre_scaled else np.float32(score * scale))
            scores_array = np.asarray(scores, dtype=np.float32)
            probabilities = np.exp(scores_array - np.max(scores_array)).astype(np.float32)
            probabilities = np.asarray(probabilities / np.sum(probabilities, dtype=np.float32), dtype=np.float32)
            for dimension in range(head_dim):
                total = np.float32(0.0)
                for index, source in enumerate(visible):
                    total = np.float32(total + np.float32(probabilities[index] * v[source, head, dimension]))
                result[row, head, dimension] = total
    return result.reshape(rows, heads * head_dim)


def luca_oracle(weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    tokens = np.asarray([2, 5, 6, 3, 0], dtype=np.int64)
    mask = np.asarray([1, 1, 1, 1, 0], dtype=np.uint8)
    hidden = weights["lucaone.embeddings.embed_tokens.weight"][tokens] + weights["lucaone.embeddings.embed_type.weight"][0]
    hidden = np.asarray(hidden, dtype=np.float32)
    hidden[tokens == 0] = 0.0
    captures = [hidden.copy()]
    for layer in range(2):
        prefix = "lucaone.encoder.layers.{}.".format(layer)
        normalized = layer_norm(hidden, weights[prefix + "pre_layer_norm.weight"], weights[prefix + "pre_layer_norm.bias"], 1.0e-5)
        q = linear(normalized, weights[prefix + "self_attn.q_proj.weight"], weights[prefix + "self_attn.q_proj.bias"])
        k = linear(normalized, weights[prefix + "self_attn.k_proj.weight"], weights[prefix + "self_attn.k_proj.bias"])
        v = linear(normalized, weights[prefix + "self_attn.v_proj.weight"], weights[prefix + "self_attn.v_proj.bias"])
        q = np.asarray(q * np.float32(1.0 / math.sqrt(2.0)), dtype=np.float32).reshape(5, 2, 2)
        k = k.reshape(5, 2, 2)
        v = v.reshape(5, 2, 2)
        inverse = weights[prefix + "self_attn.rot_emb.inv_freq"]
        for row in range(5):
            cosine = np.float32(math.cos(float(np.float32(row) * inverse[0])))
            sine = np.float32(math.sin(float(np.float32(row) * inverse[0])))
            for head in range(2):
                q_first, q_second = q[row, head, 0], q[row, head, 1]
                k_first, k_second = k[row, head, 0], k[row, head, 1]
                q[row, head, 0] = np.float32(q_first * cosine - q_second * sine)
                q[row, head, 1] = np.float32(q_second * cosine + q_first * sine)
                k[row, head, 0] = np.float32(k_first * cosine - k_second * sine)
                k[row, head, 1] = np.float32(k_second * cosine + k_first * sine)
        attended = attention(q, k, v, mask, True)
        hidden = np.asarray(hidden + linear(attended, weights[prefix + "self_attn.out_proj.weight"], weights[prefix + "self_attn.out_proj.bias"]), dtype=np.float32)
        normalized = layer_norm(hidden, weights[prefix + "post_layer_norm.weight"], weights[prefix + "post_layer_norm.bias"], 1.0e-5)
        intermediate = gelu(linear(normalized, weights[prefix + "fc1.weight"], weights[prefix + "fc1.bias"]))
        hidden = np.asarray(hidden + linear(intermediate, weights[prefix + "fc2.weight"], weights[prefix + "fc2.bias"]), dtype=np.float32)
        if layer == 0:
            captures.append(hidden.copy())
    hidden = layer_norm(hidden, weights["lucaone.encoder.last_layer_norm.weight"], weights["lucaone.encoder.last_layer_norm.bias"], 1.0e-5)
    captures.append(hidden.copy())
    pooled = np.mean(hidden[mask != 0], axis=0, dtype=np.float32)
    return result_object("lucaone", captures, hidden, pooled)


def genomics_oracle(weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    tokens = np.asarray([1, 4, 5, 2, 3], dtype=np.int64)
    mask = np.asarray([1, 1, 1, 1, 0], dtype=np.uint8)
    hidden = (
        weights["bert.embeddings.word_embeddings.weight"][tokens]
        + weights["bert.embeddings.position_embeddings.weight"][:5]
        + weights["bert.embeddings.token_type_embeddings.weight"][0]
    )
    hidden = layer_norm(hidden, weights["bert.embeddings.LayerNorm.weight"], weights["bert.embeddings.LayerNorm.bias"], 1.0e-12)
    captures = [hidden.copy()]
    for layer in range(2):
        prefix = "bert.encoder.layer.{}.".format(layer)
        fused = linear(hidden, weights[prefix + "attention.self.Wqkv.weight"], weights[prefix + "attention.self.Wqkv.bias"])
        q, k, v = np.split(fused, 3, axis=1)
        attended = attention(q.reshape(5, 2, 2), k.reshape(5, 2, 2), v.reshape(5, 2, 2), mask, False)
        attention_output = np.asarray(hidden + linear(attended, weights[prefix + "attention.output.dense.weight"], weights[prefix + "attention.output.dense.bias"]), dtype=np.float32)
        attention_output[mask == 0] = 0.0
        attention_output = layer_norm(attention_output, weights[prefix + "attention.output.LayerNorm.weight"], weights[prefix + "attention.output.LayerNorm.bias"], 1.0e-12)
        attention_output[mask == 0] = 0.0
        expanded = linear(attention_output, weights[prefix + "mlp.gated_layers.weight"])
        gated, non_gated = np.split(expanded, 2, axis=1)
        activated = np.asarray(gelu(gated) * non_gated, dtype=np.float32)
        hidden = np.asarray(attention_output + linear(activated, weights[prefix + "mlp.wo.weight"], weights[prefix + "mlp.wo.bias"]), dtype=np.float32)
        hidden[mask == 0] = 0.0
        hidden = layer_norm(hidden, weights[prefix + "mlp.layernorm.weight"], weights[prefix + "mlp.layernorm.bias"], 1.0e-12)
        hidden[mask == 0] = 0.0
        captures.append(hidden.copy())
    return result_object("genomics-fm", captures, hidden, hidden[0])


def floats(values: np.ndarray) -> List[float]:
    return [float(item) for item in values.reshape(-1)]


def result_object(variant: str, captures: Sequence[np.ndarray], final_hidden: np.ndarray, pooled: np.ndarray) -> Dict[str, Any]:
    return {
        "variant": variant,
        "rows": 5,
        "width": 4,
        "captures": [
            {"layer": layer, "values": floats(values)}
            for layer, values in enumerate(captures)
        ],
        "final_hidden": floats(final_hidden),
        "pooled": floats(pooled),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "schema": "geneb-custom-encoder-tiny-oracle-v1",
        "generator": "independent-numpy-f32",
        "cases": [luca_oracle(tiny_weights("lucaone")), genomics_oracle(tiny_weights("genomics-fm"))],
    }
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
