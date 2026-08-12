#!/usr/bin/env python3
"""Generate a deterministic two-block ESMC fixture and scalar F32 oracle."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

from evo.format import ESMC_PROFILE_VALUE, BytesTensorSource, write_model


WIDTH = 4
HEADS = 1
LAYERS = 2
INNER = 4
VOCAB = 64
MAX_LENGTH = 16
EPSILON = 1e-5
ROPE_BASE = 10000.0
RESIDUE_SCALE = math.sqrt(LAYERS / 36.0)
SEQUENCE = "LAG<mask>|Z?"
TOKENS = [0, 4, 5, 6, 32, 31, 27, 3, 2]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def vector(count: int, seed: int, scale: float = 0.025) -> list[float]:
    return [f32((((index * 7 + seed) % 19) - 9) * scale) for index in range(count)]


def norm_weight(count: int, seed: int) -> list[float]:
    return [f32(1.0 + (((index + seed) % 5) - 2) * 0.03) for index in range(count)]


def tensor(name: str, shape: tuple[int, ...], values: list[float]) -> BytesTensorSource:
    count = math.prod(shape)
    assert count == len(values)
    return BytesTensorSource(name, "F32", shape, struct.pack(f"<{count}f", *values))


def build_weights() -> tuple[list[BytesTensorSource], dict[str, list[float]]]:
    values: dict[str, list[float]] = {}

    def add(name: str, shape: tuple[int, ...], payload: list[float]) -> None:
        values[name] = payload
        tensors.append(tensor(name, shape, payload))

    tensors: list[BytesTensorSource] = []
    add("esmc.embed.weight", (VOCAB, WIDTH), vector(VOCAB * WIDTH, 1, 0.035))
    for layer in range(LAYERS):
        prefix = f"esmc.transformer.blocks.{layer}"
        add(
            f"{prefix}.attn.layernorm_qkv.layer_norm_weight",
            (WIDTH,),
            norm_weight(WIDTH, layer),
        )
        add(
            f"{prefix}.attn.layernorm_qkv.layer_norm_bias",
            (WIDTH,),
            vector(WIDTH, 20 + layer, 0.005),
        )
        add(
            f"{prefix}.attn.layernorm_qkv.weight",
            (3 * WIDTH, WIDTH),
            vector(3 * WIDTH * WIDTH, 30 + layer, 0.02),
        )
        add(f"{prefix}.attn.q_ln.weight", (WIDTH,), norm_weight(WIDTH, 2 + layer))
        add(f"{prefix}.attn.k_ln.weight", (WIDTH,), norm_weight(WIDTH, 3 + layer))
        add(
            f"{prefix}.attn.out_proj.weight",
            (WIDTH, WIDTH),
            vector(WIDTH * WIDTH, 40 + layer, 0.018),
        )
        add(
            f"{prefix}.ffn.layer_norm_weight",
            (WIDTH,),
            norm_weight(WIDTH, 4 + layer),
        )
        add(
            f"{prefix}.ffn.layer_norm_bias",
            (WIDTH,),
            vector(WIDTH, 50 + layer, 0.004),
        )
        add(
            f"{prefix}.ffn.fc1_weight",
            (2 * INNER, WIDTH),
            vector(2 * INNER * WIDTH, 60 + layer, 0.021),
        )
        add(
            f"{prefix}.ffn.fc2_weight",
            (WIDTH, INNER),
            vector(WIDTH * INNER, 70 + layer, 0.019),
        )
    add("esmc.transformer.norm.weight", (WIDTH,), norm_weight(WIDTH, 8))
    add("lm_head.0.weight", (WIDTH, WIDTH), vector(WIDTH * WIDTH, 90, 0.023))
    add("lm_head.0.bias", (WIDTH,), vector(WIDTH, 91, 0.006))
    add("lm_head.2.weight", (WIDTH,), norm_weight(WIDTH, 9))
    add("lm_head.2.bias", (WIDTH,), vector(WIDTH, 92, 0.004))
    add("lm_head.3.weight", (VOCAB, WIDTH), vector(VOCAB * WIDTH, 93, 0.017))
    add("lm_head.3.bias", (VOCAB,), vector(VOCAB, 94, 0.003))
    return tensors, values


def linear(
    inputs: list[float],
    rows: int,
    source_width: int,
    weight: list[float],
    target_width: int,
    bias: list[float] | None = None,
) -> list[float]:
    output: list[float] = []
    for row in range(rows):
        for target in range(target_width):
            total = 0.0 if bias is None else bias[target]
            for source in range(source_width):
                product = f32(
                    inputs[row * source_width + source]
                    * weight[target * source_width + source]
                )
                total = f32(total + product)
            output.append(total)
    return output


def layer_norm(
    inputs: list[float],
    rows: int,
    width: int,
    weight: list[float],
    bias: list[float] | None,
) -> list[float]:
    output: list[float] = []
    for row in range(rows):
        current = inputs[row * width : (row + 1) * width]
        mean = 0.0
        for value in current:
            mean = f32(mean + value)
        mean = f32(mean / width)
        variance = 0.0
        for value in current:
            centered = f32(value - mean)
            variance = f32(variance + f32(centered * centered))
        variance = f32(variance / width)
        inverse = f32(1.0 / math.sqrt(f32(variance + EPSILON)))
        for column, value in enumerate(current):
            result = f32(f32(f32(value - mean) * inverse) * weight[column])
            if bias is not None:
                result = f32(result + bias[column])
            output.append(result)
    return output


def add_scaled(left: list[float], right: list[float]) -> list[float]:
    return [f32(a + f32(b / RESIDUE_SCALE)) for a, b in zip(left, right, strict=True)]


def attention(
    hidden: list[float], rows: int, layer: int, weights: dict[str, list[float]]
) -> list[float]:
    prefix = f"esmc.transformer.blocks.{layer}"
    normalized = layer_norm(
        hidden,
        rows,
        WIDTH,
        weights[f"{prefix}.attn.layernorm_qkv.layer_norm_weight"],
        weights[f"{prefix}.attn.layernorm_qkv.layer_norm_bias"],
    )
    qkv = linear(
        normalized,
        rows,
        WIDTH,
        weights[f"{prefix}.attn.layernorm_qkv.weight"],
        WIDTH * 3,
    )
    query: list[float] = []
    key: list[float] = []
    value: list[float] = []
    for row in range(rows):
        base = row * WIDTH * 3
        query.extend(qkv[base : base + WIDTH])
        key.extend(qkv[base + WIDTH : base + WIDTH * 2])
        value.extend(qkv[base + WIDTH * 2 : base + WIDTH * 3])
    query = layer_norm(query, rows, WIDTH, weights[f"{prefix}.attn.q_ln.weight"], None)
    key = layer_norm(key, rows, WIDTH, weights[f"{prefix}.attn.k_ln.weight"], None)
    half = WIDTH // 2
    for row in range(rows):
        for pair in range(half):
            frequency = f32(1.0 / math.pow(ROPE_BASE, f32(pair * 2 / WIDTH)))
            angle = f32(row * frequency)
            cosine = f32(math.cos(angle))
            sine = f32(math.sin(angle))
            for array in (query, key):
                first = array[row * WIDTH + pair]
                second = array[row * WIDTH + half + pair]
                array[row * WIDTH + pair] = f32(
                    f32(first * cosine) - f32(second * sine)
                )
                array[row * WIDTH + half + pair] = f32(
                    f32(second * cosine) + f32(first * sine)
                )
    context = [0.0] * (rows * WIDTH)
    scale = f32(1.0 / math.sqrt(WIDTH))
    for target in range(rows):
        scores: list[float] = []
        for source in range(rows):
            score = 0.0
            for dimension in range(WIDTH):
                score = f32(
                    score
                    + f32(
                        query[target * WIDTH + dimension]
                        * key[source * WIDTH + dimension]
                    )
                )
            scores.append(f32(score * scale))
        maximum = max(scores)
        probabilities = [f32(math.exp(f32(score - maximum))) for score in scores]
        denominator = 0.0
        for probability in probabilities:
            denominator = f32(denominator + probability)
        for dimension in range(WIDTH):
            total = 0.0
            for source in range(rows):
                contribution = f32(
                    f32(probabilities[source] / denominator)
                    * value[source * WIDTH + dimension]
                )
                total = f32(total + contribution)
            context[target * WIDTH + dimension] = total
    return linear(
        context,
        rows,
        WIDTH,
        weights[f"{prefix}.attn.out_proj.weight"],
        WIDTH,
    )


def oracle(weights: dict[str, list[float]]) -> tuple[list[list[float]], list[float]]:
    rows = len(TOKENS)
    hidden = [
        weights["esmc.embed.weight"][token * WIDTH + column]
        for token in TOKENS
        for column in range(WIDTH)
    ]
    captures = [hidden.copy()]
    for layer in range(LAYERS):
        prefix = f"esmc.transformer.blocks.{layer}"
        hidden = add_scaled(hidden, attention(hidden, rows, layer, weights))
        normalized = layer_norm(
            hidden,
            rows,
            WIDTH,
            weights[f"{prefix}.ffn.layer_norm_weight"],
            weights[f"{prefix}.ffn.layer_norm_bias"],
        )
        projected = linear(
            normalized,
            rows,
            WIDTH,
            weights[f"{prefix}.ffn.fc1_weight"],
            INNER * 2,
        )
        gated: list[float] = []
        for row in range(rows):
            for column in range(INNER):
                first = projected[row * INNER * 2 + column]
                second = projected[row * INNER * 2 + INNER + column]
                gated.append(
                    f32(f32(first / f32(1.0 + f32(math.exp(-first)))) * second)
                )
        update = linear(
            gated,
            rows,
            INNER,
            weights[f"{prefix}.ffn.fc2_weight"],
            WIDTH,
        )
        hidden = add_scaled(hidden, update)
        if layer + 1 < LAYERS:
            captures.append(hidden.copy())
    final_hidden = layer_norm(
        hidden, rows, WIDTH, weights["esmc.transformer.norm.weight"], None
    )
    captures.append(final_hidden)
    projected = linear(
        final_hidden,
        rows,
        WIDTH,
        weights["lm_head.0.weight"],
        WIDTH,
        weights["lm_head.0.bias"],
    )
    for index, value in enumerate(projected):
        projected[index] = f32(
            f32(0.5 * value)
            * f32(1.0 + f32(math.erf(f32(value * 0.7071067811865475244))))
        )
    normalized = layer_norm(
        projected,
        rows,
        WIDTH,
        weights["lm_head.2.weight"],
        weights["lm_head.2.bias"],
    )
    logits = linear(
        normalized,
        rows,
        WIDTH,
        weights["lm_head.3.weight"],
        VOCAB,
        weights["lm_head.3.bias"],
    )
    return captures, logits


def write_floats(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensors, weights = build_weights()
    artifact = args.output_dir / "tiny-esmc.safetensors"
    metadata: dict[str, object] = {
        "runtime.abi": "esmc-safetensors-v1",
        "model.id": "esmc_tiny_test",
        "model.architecture": "ESMCTest",
        "fixture.synthetic": True,
        "config.vocab_size": VOCAB,
        "config.hidden_size": WIDTH,
        "config.num_layers": LAYERS,
        "config.num_attention_heads": HEADS,
        "config.inner_mlp_size": INNER,
        "config.max_seqlen": MAX_LENGTH,
        "config.layer_norm_epsilon": EPSILON,
        "config.rope_base": ROPE_BASE,
        "config.residue_scaling_factor": RESIDUE_SCALE,
        "runtime.embedding_layer_count": LAYERS + 1,
        "tokenizer.kind": "esmc-protein-v1",
    }
    write_model(
        artifact,
        metadata,
        tensors,
        artifact_profile=ESMC_PROFILE_VALUE,
        force=True,
    )
    captures, logits = oracle(weights)
    for layer, values in enumerate(captures):
        write_floats(args.output_dir / f"layer-{layer}.f32", values)
    write_floats(args.output_dir / "logits.f32", logits)
    print(artifact)
    print(SEQUENCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
