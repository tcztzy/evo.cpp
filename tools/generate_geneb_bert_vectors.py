#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate an independent tiny oracle for the GENEB BERT encoder profile."""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    print("NumPy is required to generate GENEB BERT oracle vectors", file=sys.stderr)
    sys.exit(77)


def fnv1a(text: str) -> int:
    value = 2166136261
    for byte in text.encode("ascii"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def topology(kind: str) -> Dict[str, Any]:
    result = {
        "vocab": 9,
        "width": 4,
        "layers": 2,
        "heads": 2,
        "head_dim": 2,
        "inner": 6,
        "max_seqlen": 8,
        "type_vocab": 2,
        "epsilon": 1.0e-5,
        "rope_base": 0.0,
        "position": "absolute",
        "norm": "post",
        "mlp": "gelu",
        "qkv": "separate",
        "input": "token-ids",
        "pooling": "attention-mask-mean",
        "final_norm": False,
        "unpad": False,
        "attention_bias": True,
        "mlp_input_bias": True,
        "mlp_output_bias": True,
    }
    if kind in ("gena-final", "gena-no-final"):
        result["norm"] = "pre"
        result["final_norm"] = kind == "gena-final"
    elif kind == "mosaic":
        result.update(
            {
                "position": "alibi",
                "mlp": "gated-gelu",
                "qkv": "fused",
                "pooling": "cls-token",
                "unpad": True,
                "mlp_input_bias": False,
            }
        )
    elif kind == "mutbert":
        result.update(
            {
                "position": "rope",
                "input": "soft-vocabulary",
                "rope_base": 10000.0,
            }
        )
    elif kind != "standard":
        raise ValueError("unknown oracle case %r" % kind)
    return result


def layer_prefix(layer: int) -> str:
    return "bert.encoder.layer.%d." % layer


def requirements(config: Dict[str, Any]) -> List[Tuple[str, Tuple[int, ...]]]:
    width = config["width"]
    inner = config["inner"]
    result = [
        ("bert.embeddings.word_embeddings.weight", (config["vocab"], width)),
    ]
    if config["position"] == "absolute":
        result.append(
            (
                "bert.embeddings.position_embeddings.weight",
                (config["max_seqlen"], width),
            )
        )
    result.extend(
        [
            (
                "bert.embeddings.token_type_embeddings.weight",
                (config["type_vocab"], width),
            ),
            ("bert.embeddings.LayerNorm.weight", (width,)),
            ("bert.embeddings.LayerNorm.bias", (width,)),
        ]
    )
    for layer in range(config["layers"]):
        prefix = layer_prefix(layer)
        if config["norm"] == "pre":
            result.extend(
                [
                    (prefix + "pre_attention_ln.weight", (width,)),
                    (prefix + "pre_attention_ln.bias", (width,)),
                ]
            )
        if config["qkv"] == "separate":
            for projection in ("query", "key", "value"):
                result.append(
                    (
                        prefix + "attention.self.%s.weight" % projection,
                        (width, width),
                    )
                )
                if config["attention_bias"]:
                    result.append(
                        (
                            prefix + "attention.self.%s.bias" % projection,
                            (width,),
                        )
                    )
        else:
            result.append((prefix + "attention.self.Wqkv.weight", (3 * width, width)))
            if config["attention_bias"]:
                result.append((prefix + "attention.self.Wqkv.bias", (3 * width,)))
        result.append((prefix + "attention.output.dense.weight", (width, width)))
        if config["attention_bias"]:
            result.append((prefix + "attention.output.dense.bias", (width,)))
        if config["norm"] == "post":
            result.extend(
                [
                    (prefix + "attention.output.LayerNorm.weight", (width,)),
                    (prefix + "attention.output.LayerNorm.bias", (width,)),
                ]
            )
        if config["mlp"] == "gelu":
            result.append((prefix + "intermediate.dense.weight", (inner, width)))
            if config["mlp_input_bias"]:
                result.append((prefix + "intermediate.dense.bias", (inner,)))
            result.append((prefix + "output.dense.weight", (width, inner)))
            if config["mlp_output_bias"]:
                result.append((prefix + "output.dense.bias", (width,)))
        else:
            result.append((prefix + "mlp.gated_layers.weight", (2 * inner, width)))
            result.append((prefix + "mlp.wo.weight", (width, inner)))
            if config["mlp_output_bias"]:
                result.append((prefix + "mlp.wo.bias", (width,)))
        if config["norm"] == "pre":
            result.extend(
                [
                    (prefix + "post_attention_ln.weight", (width,)),
                    (prefix + "post_attention_ln.bias", (width,)),
                ]
            )
        elif config["mlp"] == "gated-gelu":
            result.extend(
                [
                    (prefix + "mlp.layernorm.weight", (width,)),
                    (prefix + "mlp.layernorm.bias", (width,)),
                ]
            )
        else:
            result.extend(
                [
                    (prefix + "output.LayerNorm.weight", (width,)),
                    (prefix + "output.LayerNorm.bias", (width,)),
                ]
            )
    if config["final_norm"]:
        result.extend(
            [
                ("bert.encoder.last_layer_ln.weight", (width,)),
                ("bert.encoder.last_layer_ln.bias", (width,)),
            ]
        )
    return result


def is_norm_weight(name: str) -> bool:
    return (
        "LayerNorm.weight" in name
        or "_ln.weight" in name
        or "layernorm.weight" in name
    )


def is_norm_bias(name: str) -> bool:
    return (
        "LayerNorm.bias" in name
        or "_ln.bias" in name
        or "layernorm.bias" in name
    )


def weights(config: Dict[str, Any]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for name, shape in requirements(config):
        count = int(np.prod(shape))
        seed = fnv1a(name)
        values = np.empty(count, dtype=np.float32)
        for index in range(count):
            angle = np.float32((seed % 997 + index * 17) * np.float32(0.013))
            oscillation = np.float32(np.sin(angle))
            value = np.float32(oscillation * np.float32(0.075))
            if is_norm_weight(name):
                value = np.float32(1.0) + np.float32(
                    oscillation * np.float32(0.08)
                )
            elif is_norm_bias(name):
                value = np.float32(oscillation * np.float32(0.015))
            values[index] = value
        result[name] = values.reshape(shape)
    return result


def layer_norm(values: Any, scale: Any, bias: Any, epsilon: float) -> Any:
    mean = values.mean(axis=-1, keepdims=True, dtype=np.float32)
    centered = np.asarray(values - mean, dtype=np.float32)
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    return np.asarray(
        centered / np.sqrt(variance + np.float32(epsilon)) * scale + bias,
        dtype=np.float32,
    )


def linear(values: Any, weight: Any, bias: Optional[Any]) -> Any:
    output = np.asarray(values @ weight.T, dtype=np.float32)
    if bias is not None:
        output = np.asarray(output + bias, dtype=np.float32)
    return output


def gelu(values: Any) -> Any:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    output = np.empty_like(flat)
    inverse_sqrt_two = np.float32(0.70710678118654752440)
    for index, value in enumerate(flat):
        output[index] = np.float32(
            np.float32(0.5)
            * value
            * np.float32(1.0 + math.erf(float(value * inverse_sqrt_two)))
        )
    return output.reshape(values.shape)


def slopes(heads: int) -> Any:
    def power_of_two(count: int) -> List[float]:
        start = 2.0 ** (-(2.0 ** (-(math.log2(count) - 3.0))))
        return [float(np.float32(start * (start ** index))) for index in range(count)]

    if math.log2(heads).is_integer():
        return np.asarray(power_of_two(heads), dtype=np.float32)
    closest = 2 ** math.floor(math.log2(heads))
    result = power_of_two(closest)
    expanded = power_of_two(2 * closest)
    result.extend(expanded[0::2][: heads - closest])
    return np.asarray(result, dtype=np.float32)


def apply_rope(values: Any, rope_base: float) -> Any:
    output = np.asarray(values, dtype=np.float32).copy()
    head_dim = output.shape[-1]
    pairs = head_dim // 2
    for row in range(output.shape[0]):
        for pair in range(pairs):
            exponent = np.float32(2.0 * pair / head_dim)
            angle = np.float32(row) / np.float32(
                np.power(np.float32(rope_base), exponent)
            )
            cosine = np.float32(np.cos(angle))
            sine = np.float32(np.sin(angle))
            first = output[row, :, pair].copy()
            second = output[row, :, pair + pairs].copy()
            output[row, :, pair] = np.asarray(
                first * cosine - second * sine, dtype=np.float32
            )
            output[row, :, pair + pairs] = np.asarray(
                second * cosine + first * sine, dtype=np.float32
            )
    return output


def attention(
    query: Any,
    key: Any,
    value: Any,
    mask: Any,
    use_alibi: bool,
    unpad: bool,
) -> Any:
    rows, heads, head_dim = query.shape
    output = np.zeros_like(query)
    head_slopes = slopes(heads) if use_alibi else None
    scale = np.float32(1.0 / math.sqrt(head_dim))
    for row in range(rows):
        if unpad and not bool(mask[row]):
            continue
        for head in range(heads):
            scores = np.asarray(query[row, head] @ key[:, head].T, dtype=np.float32)
            scores = np.asarray(scores * scale, dtype=np.float32)
            if head_slopes is not None:
                distance = np.abs(np.arange(rows, dtype=np.int64) - row).astype(np.float32)
                scores = np.asarray(scores - head_slopes[head] * distance, dtype=np.float32)
            scores = scores.copy()
            scores[~mask] = -np.inf
            maximum = np.max(scores)
            probabilities = np.exp(scores - maximum, dtype=np.float32)
            probabilities[~mask] = np.float32(0.0)
            probabilities = np.asarray(
                probabilities / probabilities.sum(dtype=np.float32), dtype=np.float32
            )
            output[row, head] = np.asarray(
                probabilities @ value[:, head], dtype=np.float32
            )
    return output


def run_case(kind: str) -> Dict[str, Any]:
    config = topology(kind)
    tensors = weights(config)
    rows = 3
    width = config["width"]
    mask = np.asarray([True, True, True])
    token_ids = np.asarray([1, 5, 2], dtype=np.int64)
    if config["input"] == "soft-vocabulary":
        soft = np.zeros((rows, config["vocab"]), dtype=np.float32)
        soft[np.arange(rows), token_ids] = np.float32(1.0)
        embedding = np.asarray(
            soft @ tensors["bert.embeddings.word_embeddings.weight"],
            dtype=np.float32,
        )
    else:
        embedding = tensors["bert.embeddings.word_embeddings.weight"][token_ids].copy()
    embedding = np.asarray(
        embedding + tensors["bert.embeddings.token_type_embeddings.weight"][0],
        dtype=np.float32,
    )
    if config["position"] == "absolute":
        embedding = np.asarray(
            embedding
            + tensors["bert.embeddings.position_embeddings.weight"][:rows],
            dtype=np.float32,
        )
    hidden = layer_norm(
        embedding,
        tensors["bert.embeddings.LayerNorm.weight"],
        tensors["bert.embeddings.LayerNorm.bias"],
        config["epsilon"],
    )
    captures = {"0": hidden.reshape(-1).astype(float).tolist()}
    for layer_index in range(config["layers"]):
        prefix = layer_prefix(layer_index)
        if config["norm"] == "pre":
            attention_input = layer_norm(
                hidden,
                tensors[prefix + "pre_attention_ln.weight"],
                tensors[prefix + "pre_attention_ln.bias"],
                config["epsilon"],
            )
        else:
            attention_input = hidden
        if config["qkv"] == "separate":
            projected = []
            for name in ("query", "key", "value"):
                projected.append(
                    linear(
                        attention_input,
                        tensors[prefix + "attention.self.%s.weight" % name],
                        tensors[prefix + "attention.self.%s.bias" % name],
                    ).reshape(rows, config["heads"], config["head_dim"])
                )
            query, key, value = projected
        else:
            fused = linear(
                attention_input,
                tensors[prefix + "attention.self.Wqkv.weight"],
                tensors[prefix + "attention.self.Wqkv.bias"],
            )
            query, key, value = [
                item.reshape(rows, config["heads"], config["head_dim"])
                for item in np.split(fused, 3, axis=-1)
            ]
        if config["position"] == "rope":
            query = apply_rope(query, config["rope_base"])
            key = apply_rope(key, config["rope_base"])
        attended = attention(
            query,
            key,
            value,
            mask,
            config["position"] == "alibi",
            config["unpad"],
        ).reshape(rows, width)
        attention_output = linear(
            attended,
            tensors[prefix + "attention.output.dense.weight"],
            tensors[prefix + "attention.output.dense.bias"],
        )
        attention_output = np.asarray(attention_output + hidden, dtype=np.float32)
        if config["norm"] == "post":
            attention_output = layer_norm(
                attention_output,
                tensors[prefix + "attention.output.LayerNorm.weight"],
                tensors[prefix + "attention.output.LayerNorm.bias"],
                config["epsilon"],
            )
        if config["norm"] == "pre":
            mlp_input = layer_norm(
                attention_output,
                tensors[prefix + "post_attention_ln.weight"],
                tensors[prefix + "post_attention_ln.bias"],
                config["epsilon"],
            )
        else:
            mlp_input = attention_output
        if config["mlp"] == "gelu":
            expanded = linear(
                mlp_input,
                tensors[prefix + "intermediate.dense.weight"],
                tensors[prefix + "intermediate.dense.bias"],
            )
            activated = gelu(expanded)
            mlp_output = linear(
                activated,
                tensors[prefix + "output.dense.weight"],
                tensors[prefix + "output.dense.bias"],
            )
        else:
            expanded = linear(
                mlp_input, tensors[prefix + "mlp.gated_layers.weight"], None
            )
            gated, non_gated = np.split(expanded, 2, axis=-1)
            activated = np.asarray(gelu(gated) * non_gated, dtype=np.float32)
            mlp_output = linear(
                activated,
                tensors[prefix + "mlp.wo.weight"],
                tensors[prefix + "mlp.wo.bias"],
            )
        hidden = np.asarray(mlp_output + attention_output, dtype=np.float32)
        if config["norm"] == "post":
            if config["mlp"] == "gated-gelu":
                norm_prefix = prefix + "mlp.layernorm"
            else:
                norm_prefix = prefix + "output.LayerNorm"
            hidden = layer_norm(
                hidden,
                tensors[norm_prefix + ".weight"],
                tensors[norm_prefix + ".bias"],
                config["epsilon"],
            )
        public_layer = layer_index + 1
        if not (config["final_norm"] and public_layer == config["layers"]):
            captures[str(public_layer)] = hidden.reshape(-1).astype(float).tolist()
    if config["final_norm"]:
        hidden = layer_norm(
            hidden,
            tensors["bert.encoder.last_layer_ln.weight"],
            tensors["bert.encoder.last_layer_ln.bias"],
            config["epsilon"],
        )
        captures[str(config["layers"])] = hidden.reshape(-1).astype(float).tolist()
    if config["pooling"] == "cls-token":
        pooled = hidden[0]
    else:
        pooled = hidden[mask].mean(axis=0, dtype=np.float32)
    return {
        "case": kind,
        "rows": rows,
        "width": width,
        "captures": captures,
        "final_hidden": hidden.reshape(-1).astype(float).tolist(),
        "pooled": pooled.astype(float).tolist(),
    }


def fixture() -> Dict[str, Any]:
    cases = {}
    for kind in ("standard", "gena-final", "gena-no-final", "mosaic", "mutbert"):
        cases[kind] = run_case(kind)
    return {
        "format": "geneb-bert-oracle-v1",
        "reference": {
            "implementation": "independent NumPy eager encoder",
            "dropout": "eval-disabled",
            "dtype": "F32",
            "max_abs_tolerance": 5.0e-5,
        },
        "cases": cases,
    }


def canonical_bytes(value: Dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % path.name, dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    generated = canonical_bytes(fixture())
    if args.check is not None:
        if args.check.read_bytes() != generated:
            raise AssertionError("GENEB BERT oracle fixture is stale: %s" % args.check)
        print("GENEB BERT oracle fixture is current: %s" % args.check)
    elif args.output is not None:
        atomic_write(args.output, generated)
    else:
        sys.stdout.buffer.write(generated)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, OSError, ValueError) as error:
        print("GENEB BERT oracle error: %s" % error, file=sys.stderr)
        sys.exit(1)
