#!/usr/bin/env python3
"""Generate a deterministic tiny HyenaDNA HF-shaped fixture and oracle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

from evo.format import HYENADNA_PROFILE_VALUE, BytesTensorSource, write_model


VOCAB = 16
SOURCE_VOCAB = 12
WIDTH = 4
INNER = 8
LAYERS = 2
MAX_LENGTH = 16
EMBEDDING_DIMENSION = 5
FILTER_ORDER = 6
INNER_FILTER_LAYERS = 2
TOKENS = (7, 8, 9, 10, 11, 7, 8, 9)
EPSILON = 1e-5


def f32(value: float) -> float:
    """Round one operation result to IEEE-754 binary32."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    data: tuple[float, ...]

    def bytes(self) -> bytes:
        return struct.pack(f"<{len(self.data)}f", *self.data)


def values(
    name: str, shape: tuple[int, ...], scale: float, center: float = 0.0
) -> Tensor:
    count = math.prod(shape)
    seed = sum(
        (index + 1) * byte for index, byte in enumerate(name.encode("ascii"))
    )
    data = tuple(
        f32(center + ((((index + 3) * 29 + seed * 11) % 31) - 15) * scale)
        for index in range(count)
    )
    return Tensor(shape, data)


def rows(tensor: Tensor) -> list[list[float]]:
    columns = tensor.shape[-1]
    return [
        list(tensor.data[offset : offset + columns])
        for offset in range(0, len(tensor.data), columns)
    ]


def vector(tensor: Tensor) -> list[float]:
    return list(tensor.data)


def positional() -> tuple[Tensor, Tensor, Tensor]:
    times = [f32(index / (MAX_LENGTH - 1)) for index in range(MAX_LENGTH)]
    frequency = (f32(1e-4), f32(1.0))
    encoded: list[float] = []
    two_pi = f32(2.0 * math.pi)
    for index, time in enumerate(times):
        angle = f32(f32(two_pi * f32(float(index))) / f32(float(MAX_LENGTH)))
        encoded.append(time)
        encoded.extend(f32(math.cos(f32(-item * angle))) for item in frequency)
        encoded.extend(f32(math.sin(f32(-item * angle))) for item in frequency)
    minimum = math.log(1e-2) / 1.5
    maximum = math.log(1e-2) / 0.3
    deltas = tuple(
        f32(minimum + (maximum - minimum) * index / (WIDTH - 1))
        for index in range(WIDTH)
    )
    return (
        Tensor((1, MAX_LENGTH, 1), tuple(times)),
        Tensor((1, MAX_LENGTH, EMBEDDING_DIMENSION), tuple(encoded)),
        Tensor((1, 1, WIDTH), deltas),
    )


def tensors() -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    result["hyena.backbone.embeddings.word_embeddings.weight"] = values(
        "embedding", (VOCAB, WIDTH), 0.015
    )
    result["hyena.backbone.ln_f.weight"] = values(
        "final.weight", (WIDTH,), 0.01, 1.0
    )
    result["hyena.backbone.ln_f.bias"] = values("final.bias", (WIDTH,), 0.002)
    result["lm_head.weight"] = values("head", (VOCAB, WIDTH), 0.02)
    time, encoded, deltas = positional()
    for index in range(LAYERS):
        prefix = f"hyena.backbone.layers.{index}"
        mixer = f"{prefix}.mixer"
        filter_prefix = f"{mixer}.filter_fn"
        for norm in ("norm1", "norm2"):
            result[f"{prefix}.{norm}.weight"] = values(
                f"{index}.{norm}.weight", (WIDTH,), 0.01, 1.0
            )
            result[f"{prefix}.{norm}.bias"] = values(
                f"{index}.{norm}.bias", (WIDTH,), 0.002
            )
        result[f"{mixer}.in_proj.weight"] = values(
            f"{index}.in.weight", (WIDTH * 3, WIDTH), 0.018
        )
        result[f"{mixer}.in_proj.bias"] = values(
            f"{index}.in.bias", (WIDTH * 3,), 0.002
        )
        result[f"{mixer}.short_filter.weight"] = values(
            f"{index}.short.weight", (WIDTH * 3, 1, 3), 0.04
        )
        result[f"{mixer}.short_filter.bias"] = values(
            f"{index}.short.bias", (WIDTH * 3,), 0.002
        )
        result[f"{mixer}.out_proj.weight"] = values(
            f"{index}.out.weight", (WIDTH, WIDTH), 0.025
        )
        result[f"{mixer}.out_proj.bias"] = values(
            f"{index}.out.bias", (WIDTH,), 0.002
        )
        result[f"{filter_prefix}.bias"] = values(
            f"{index}.filter.bias", (WIDTH,), 0.02
        )
        result[f"{filter_prefix}.implicit_filter.0.weight"] = values(
            f"{index}.filter.0.weight",
            (FILTER_ORDER, EMBEDDING_DIMENSION),
            0.03,
        )
        result[f"{filter_prefix}.implicit_filter.0.bias"] = values(
            f"{index}.filter.0.bias", (FILTER_ORDER,), 0.004
        )
        result[f"{filter_prefix}.implicit_filter.1.freq"] = values(
            f"{index}.filter.freq", (1, FILTER_ORDER), 0.03, 2.0
        )
        for inner_index in range(INNER_FILTER_LAYERS):
            module = 2 + inner_index * 2
            result[f"{filter_prefix}.implicit_filter.{module}.weight"] = values(
                f"{index}.filter.{module}.weight",
                (FILTER_ORDER, FILTER_ORDER),
                0.025,
            )
            result[f"{filter_prefix}.implicit_filter.{module}.bias"] = values(
                f"{index}.filter.{module}.bias", (FILTER_ORDER,), 0.004
            )
        output_module = 2 + INNER_FILTER_LAYERS * 2
        result[f"{filter_prefix}.implicit_filter.{output_module}.weight"] = values(
            f"{index}.filter.output", (WIDTH, FILTER_ORDER), 0.03
        )
        result[f"{filter_prefix}.modulation.deltas"] = deltas
        result[f"{filter_prefix}.pos_emb.t"] = time
        result[f"{filter_prefix}.pos_emb.z"] = encoded
        result[f"{prefix}.mlp.fc1.weight"] = values(
            f"{index}.mlp.up.weight", (INNER, WIDTH), 0.02
        )
        result[f"{prefix}.mlp.fc1.bias"] = values(
            f"{index}.mlp.up.bias", (INNER,), 0.002
        )
        result[f"{prefix}.mlp.fc2.weight"] = values(
            f"{index}.mlp.down.weight", (WIDTH, INNER), 0.02
        )
        result[f"{prefix}.mlp.fc2.bias"] = values(
            f"{index}.mlp.down.bias", (WIDTH,), 0.002
        )
    return result


def layer_norm(
    inputs: list[list[float]], weight: list[float], bias: list[float]
) -> list[list[float]]:
    result: list[list[float]] = []
    for row in inputs:
        mean = f32(sum(row) / len(row))
        variance = f32(sum(f32(f32(item - mean) ** 2) for item in row) / len(row))
        inverse = f32(1.0 / math.sqrt(f32(variance + f32(EPSILON))))
        result.append(
            [
                f32(f32(f32(item - mean) * inverse) * weight[index] + bias[index])
                for index, item in enumerate(row)
            ]
        )
    return result


def linear(
    inputs: list[list[float]], weight: Tensor, bias: Tensor | None = None
) -> list[list[float]]:
    weight_rows = rows(weight)
    bias_values = vector(bias) if bias is not None else None
    result: list[list[float]] = []
    for input_row in inputs:
        output_row: list[float] = []
        for output, weight_row in enumerate(weight_rows):
            total = bias_values[output] if bias_values is not None else 0.0
            for item, coefficient in zip(input_row, weight_row):
                total = f32(total + f32(item * coefficient))
            output_row.append(total)
        result.append(output_row)
    return result


def add(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [f32(a + b) for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def multiply(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    return [
        [f32(a * b) for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def oracle(weights: dict[str, Tensor]) -> list[float]:
    embedding = rows(weights["hyena.backbone.embeddings.word_embeddings.weight"])
    hidden = [embedding[token].copy() for token in TOKENS]
    length = len(hidden)
    for index in range(LAYERS):
        prefix = f"hyena.backbone.layers.{index}"
        mixer = f"{prefix}.mixer"
        filter_prefix = f"{mixer}.filter_fn"
        residual = [row.copy() for row in hidden]
        normalized = layer_norm(
            hidden,
            vector(weights[f"{prefix}.norm1.weight"]),
            vector(weights[f"{prefix}.norm1.bias"]),
        )
        projected = linear(
            normalized,
            weights[f"{mixer}.in_proj.weight"],
            weights[f"{mixer}.in_proj.bias"],
        )
        short_weight = rows(weights[f"{mixer}.short_filter.weight"])
        short_bias = vector(weights[f"{mixer}.short_filter.bias"])
        filtered = [[0.0] * (WIDTH * 3) for _ in range(length)]
        for position in range(length):
            for channel in range(WIDTH * 3):
                total = short_bias[channel]
                for tap in range(3):
                    source = position - (2 - tap)
                    if source >= 0:
                        total = f32(
                            total
                            + f32(projected[source][channel] * short_weight[channel][tap])
                        )
                filtered[position][channel] = total
        x0 = [row[:WIDTH] for row in filtered]
        x1 = [row[WIDTH : WIDTH * 2] for row in filtered]
        value = [row[WIDTH * 2 :] for row in filtered]
        gated = multiply(value, x1)
        activation = linear(
            rows(weights[f"{filter_prefix}.pos_emb.z"])[:length],
            weights[f"{filter_prefix}.implicit_filter.0.weight"],
            weights[f"{filter_prefix}.implicit_filter.0.bias"],
        )
        frequency = vector(weights[f"{filter_prefix}.implicit_filter.1.freq"])
        activation = [
            [f32(math.sin(f32(item * frequency[column]))) for column, item in enumerate(row)]
            for row in activation
        ]
        for inner_index in range(INNER_FILTER_LAYERS):
            module = 2 + inner_index * 2
            activation = linear(
                activation,
                weights[f"{filter_prefix}.implicit_filter.{module}.weight"],
                weights[f"{filter_prefix}.implicit_filter.{module}.bias"],
            )
            activation = [
                [
                    f32(math.sin(f32(item * frequency[column])))
                    for column, item in enumerate(row)
                ]
                for row in activation
            ]
        output_module = 2 + INNER_FILTER_LAYERS * 2
        kernel = linear(
            activation,
            weights[f"{filter_prefix}.implicit_filter.{output_module}.weight"],
        )
        times = vector(weights[f"{filter_prefix}.pos_emb.t"])
        deltas = vector(weights[f"{filter_prefix}.modulation.deltas"])
        kernel = [
            [
                f32(
                    item
                    * f32(math.exp(f32(-times[position] * abs(deltas[channel]))) + 0.05)
                )
                for channel, item in enumerate(row)
            ]
            for position, row in enumerate(kernel)
        ]
        direct = vector(weights[f"{filter_prefix}.bias"])
        convolved = [[0.0] * WIDTH for _ in range(length)]
        for position in range(length):
            for channel in range(WIDTH):
                total = f32(gated[position][channel] * direct[channel])
                for lag in range(position + 1):
                    total = f32(
                        total
                        + f32(kernel[lag][channel] * gated[position - lag][channel])
                    )
                convolved[position][channel] = total
        hidden = add(
            linear(
                multiply(convolved, x0),
                weights[f"{mixer}.out_proj.weight"],
                weights[f"{mixer}.out_proj.bias"],
            ),
            residual,
        )
        residual = [row.copy() for row in hidden]
        normalized = layer_norm(
            hidden,
            vector(weights[f"{prefix}.norm2.weight"]),
            vector(weights[f"{prefix}.norm2.bias"]),
        )
        mlp = linear(
            normalized,
            weights[f"{prefix}.mlp.fc1.weight"],
            weights[f"{prefix}.mlp.fc1.bias"],
        )
        for row in mlp:
            for column, item in enumerate(row):
                cubic = f32(f32(item * item) * item)
                inner = f32(0.7978845608028654 * f32(item + f32(0.044715 * cubic)))
                row[column] = f32(f32(0.5 * item) * f32(1.0 + math.tanh(inner)))
        hidden = add(
            linear(
                mlp,
                weights[f"{prefix}.mlp.fc2.weight"],
                weights[f"{prefix}.mlp.fc2.bias"],
            ),
            residual,
        )
    hidden = layer_norm(
        hidden,
        vector(weights["hyena.backbone.ln_f.weight"]),
        vector(weights["hyena.backbone.ln_f.bias"]),
    )
    logits = linear(hidden, weights["lm_head.weight"])
    return [item for row in logits for item in row]


def sources(weights: dict[str, Tensor]) -> list[BytesTensorSource]:
    return [
        BytesTensorSource(name, "F32", value.shape, value.bytes())
        for name, value in weights.items()
    ]


def metadata(architecture: str) -> dict[str, object]:
    result: dict[str, object] = {
        "runtime.abi": "hyenadna-safetensors-v1",
        "model.id": "tiny-hyenadna",
        "model.architecture": architecture,
        "config.vocab_size": VOCAB,
        "config.source_vocab_size": SOURCE_VOCAB,
        "config.hidden_size": WIDTH,
        "config.num_layers": LAYERS,
        "config.max_seqlen": MAX_LENGTH,
        "config.inner_mlp_size": INNER,
        "config.embedding_dim": EMBEDDING_DIMENSION,
        "config.filter_order": FILTER_ORDER,
        "config.hyena_order": 2,
        "config.num_inner_mlps": INNER_FILTER_LAYERS,
        "config.short_filter_length": 3,
        "config.layer_norm_epsilon": EPSILON,
        "config.pad_token_id": 4,
        "tokenizer.kind": "hyenadna-character-v1",
    }
    if architecture == "HyenaDNATest":
        result["fixture.synthetic"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--oracle", type=Path)
    args = parser.parse_args()
    if all(path is None for path in (args.runtime, args.source, args.config, args.oracle)):
        parser.error("at least one output is required")
    weights = tensors()
    if args.runtime is not None:
        write_model(
            args.runtime,
            metadata("HyenaDNATest"),
            sources(weights),
            artifact_profile=HYENADNA_PROFILE_VALUE,
            force=True,
        )
    if args.source is not None:
        write_model(
            args.source,
            metadata("HyenaDNA"),
            sources(weights),
            artifact_profile=HYENADNA_PROFILE_VALUE,
            force=True,
        )
    if args.config is not None:
        config = {
            "_name_or_path": "tiny-hyenadna",
            "model_type": "hyenadna",
            "torch_dtype": "float32",
            "use_bias": True,
            "train_freq": True,
            "tie_word_embeddings": False,
            "vocab_size": SOURCE_VOCAB,
            "pad_vocab_size_multiple": 8,
            "d_model": WIDTH,
            "n_layer": LAYERS,
            "max_seq_len": MAX_LENGTH,
            "d_inner": INNER,
            "emb_dim": EMBEDDING_DIMENSION,
            "filter_order": FILTER_ORDER,
            "hyena_order": 2,
            "num_inner_mlps": INNER_FILTER_LAYERS,
            "short_filter_order": 3,
            "layer_norm_epsilon": EPSILON,
            "pad_token_id": 4,
        }
        args.config.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    if args.oracle is not None:
        values_out = oracle(weights)
        args.oracle.write_bytes(struct.pack(f"<{len(values_out)}f", *values_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
