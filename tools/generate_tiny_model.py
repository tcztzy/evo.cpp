#!/usr/bin/env python3
"""Generate a deterministic 50-layer tiny StripedHyena2 model and oracle."""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from evo2c.format import BytesTensorSource, write_model


VOCAB = 512
WIDTH = 8
LAYERS = 50
HEADS = 2
HEAD_DIM = WIDTH // HEADS
STATE = 2
INNER = 12
SHORT = 3
HCS_KERNEL = 3
HCM_KERNEL = 4
HCS_GROUPS = 2
HCM_GROUPS = 2
EPSILON = 1e-6
ROPE_SCALE = 128.0
PROMPT = [2, 5, 7, 3]
CHUNKED_PROMPT = [2, 5, 7, 3, 9, 11, 13, 17, 19]
DECODE_TOKEN = 9
DUMP_LAYER = 17
HCS = [0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46]
HCM = [1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47]
HCL = [2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48]
ATTENTION = [3, 10, 17, 24, 31, 35, 42, 49]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def bf16_payload(values: list[float]) -> bytes:
    payload = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
        payload.extend(struct.pack("<H", bits >> 16))
    return bytes(payload)


def deterministic(name: str, count: int, scale: float, *, center: float = 0.0) -> list[float]:
    seed = sum((index + 1) * byte for index, byte in enumerate(name.encode("ascii")))
    return [bf16(center + ((((index + 3) * 37 + seed * 13) % 29) - 14) * scale) for index in range(count)]


def f32_values(name: str, count: int, scale: float, *, center: float = 0.0) -> list[float]:
    seed = sum((index + 1) * byte for index, byte in enumerate(name.encode("ascii")))
    return [f32(center + ((((index + 5) * 31 + seed * 7) % 23) - 11) * scale) for index in range(count)]


def linear(rows: list[list[float]], weight: list[float], out: int, bias: list[float] | None = None) -> list[list[float]]:
    width = len(rows[0])
    result: list[list[float]] = []
    for row in rows:
        output = []
        for target in range(out):
            total = f32(0.0 if bias is None else bias[target])
            for source in range(width):
                total = f32(total + f32(row[source] * weight[target * width + source]))
            output.append(bf16(total))
        result.append(output)
    return result


def rms_norm(rows: list[list[float]], scale: list[float]) -> list[list[float]]:
    output = []
    for row in rows:
        total = f32(0.0)
        for value in row:
            total = f32(total + f32(value * value))
        denominator = f32(math.sqrt(f32(total / len(row))) + EPSILON)
        output.append([bf16(f32(f32(value / denominator) * scale[index])) for index, value in enumerate(row)])
    return output


def add(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[bf16(f32(a + b)) for a, b in zip(x, y, strict=True)] for x, y in zip(left, right, strict=True)]


def gate(first: list[list[float]], second: list[list[float]]) -> list[list[float]]:
    return [[bf16(f32(a * b)) for a, b in zip(x, y, strict=True)] for x, y in zip(first, second, strict=True)]


def grouped_fir(
    rows: list[list[float]],
    weight: list[float],
    groups: int,
    kernel: int,
    *,
    causal: bool,
    direct: list[float] | None = None,
) -> list[list[float]]:
    channels = len(rows[0])
    per_group = channels // groups
    output = [[0.0] * channels for _ in rows]
    for time in range(len(rows)):
        for channel in range(channels):
            group = channel // per_group
            total = f32(0.0)
            for tap in range(kernel):
                delay = tap if causal else kernel - 1 - tap
                if time >= delay:
                    total = f32(total + f32(rows[time - delay][channel] * weight[group * kernel + tap]))
            if direct is not None:
                total = f32(total + f32(direct[channel] * rows[time][channel]))
            output[time][channel] = bf16(total)
    return output


def fir_decode(
    row: list[float],
    state: list[list[float]],
    weight: list[float],
    groups: int,
    kernel: int,
    *,
    causal: bool,
    direct: list[float] | None = None,
) -> list[float]:
    channels = len(row)
    per_group = channels // groups
    output = [0.0] * channels
    for channel in range(channels):
        group = channel // per_group
        total = f32(0.0)
        for index, old in enumerate(state[channel]):
            tap = index if not causal else kernel - 1 - index
            total = f32(total + f32(old * weight[group * kernel + tap]))
        current_tap = kernel - 1 if not causal else 0
        total = f32(total + f32(row[channel] * weight[group * kernel + current_tap]))
        if direct is not None:
            total = f32(total + f32(direct[channel] * row[channel]))
        output[channel] = bf16(total)
        state[channel] = state[channel][1:] + [row[channel]]
    return output


def cache_fir(rows: list[list[float]], kernel: int) -> list[list[float]]:
    channels = len(rows[0])
    valid = min(len(rows), kernel - 1)
    return [[0.0] * (kernel - 1 - valid) + [rows[time][channel] for time in range(len(rows) - valid, len(rows))] for channel in range(channels)]


def split_triples(rows: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    x2, x1, value = [], [], []
    for row in rows:
        x2.append(row[0::3])
        x1.append(row[1::3])
        value.append(row[2::3])
    return x2, x1, value


def hcl_prefill(
    x2: list[list[float]],
    x1: list[list[float]],
    value: list[list[float]],
    direct: list[float],
    log_poles: list[float],
    residues: list[float],
) -> tuple[list[list[float]], list[list[float]]]:
    state = [[0.0] * STATE for _ in range(WIDTH)]
    output: list[list[float]] = []
    for time in range(len(x2)):
        row = []
        for channel in range(WIDTH):
            gated = bf16(f32(x1[time][channel] * value[time][channel]))
            modal = f32(0.0)
            for mode in range(STATE):
                index = channel * STATE + mode
                state[channel][mode] = f32(f32(math.exp(log_poles[index])) * state[channel][mode] + gated)
                modal = f32(modal + f32(residues[index] * state[channel][mode]))
            row.append(bf16(f32(x2[time][channel] * f32(modal + f32(direct[channel] * gated)))))
        output.append(row)
    return output, state


def hcl_decode(
    x2: list[float],
    x1: list[float],
    value: list[float],
    direct: list[float],
    log_poles: list[float],
    residues: list[float],
    state: list[list[float]],
) -> list[float]:
    output = []
    for channel in range(WIDTH):
        gated = bf16(f32(x1[channel] * value[channel]))
        modal = f32(0.0)
        for mode in range(STATE):
            index = channel * STATE + mode
            state[channel][mode] = f32(f32(math.exp(log_poles[index])) * state[channel][mode] + gated)
            modal = f32(modal + f32(residues[index] * state[channel][mode]))
        output.append(bf16(f32(x2[channel] * f32(modal + f32(direct[channel] * gated)))))
    return output


def apply_rope(rows: list[list[float]], inverse: list[float], offset: int) -> list[list[float]]:
    output = [row.copy() for row in rows]
    for time, row in enumerate(output):
        position = f32((offset + time) / ROPE_SCALE)
        for head in range(HEADS):
            base = head * HEAD_DIM
            half = HEAD_DIM // 2
            for pair in range(half):
                angle = f32(position * inverse[pair])
                cosine, sine = f32(math.cos(angle)), f32(math.sin(angle))
                first, second = row[base + pair], row[base + half + pair]
                row[base + pair] = bf16(f32(f32(first * cosine) - f32(second * sine)))
                row[base + half + pair] = bf16(f32(f32(second * cosine) + f32(first * sine)))
    return output


def attention_chunk(
    query: list[list[float]],
    key_cache: list[list[float]],
    value_cache: list[list[float]],
    prefix: int,
) -> list[list[float]]:
    output: list[list[float]] = []
    scale = f32(1.0 / math.sqrt(HEAD_DIM))
    for time, query_row in enumerate(query):
        row = [0.0] * WIDTH
        sources = prefix + time + 1
        for head in range(HEADS):
            scores = []
            for source in range(sources):
                total = f32(0.0)
                for dimension in range(HEAD_DIM):
                    index = head * HEAD_DIM + dimension
                    total = f32(total + f32(query_row[index] * key_cache[source][index]))
                scores.append(f32(total * scale))
            maximum = max(scores)
            probabilities = [f32(math.exp(f32(score - maximum))) for score in scores]
            denominator = f32(0.0)
            for probability in probabilities:
                denominator = f32(denominator + probability)
            for dimension in range(HEAD_DIM):
                total = f32(0.0)
                index = head * HEAD_DIM + dimension
                for source, probability in enumerate(probabilities):
                    total = f32(total + f32(f32(probability / denominator) * value_cache[source][index]))
                row[index] = bf16(total)
        output.append(row)
    return output


@dataclass
class LayerState:
    short: list[list[float]] | None = None
    inner: list[list[float]] | None = None
    iir: list[list[float]] | None = None
    key: list[list[float]] = field(default_factory=list)
    value: list[list[float]] = field(default_factory=list)


class TinyModel:
    def __init__(self, *, add_fp8_residue: bool = False) -> None:
        self.tensors: dict[str, tuple[str, tuple[int, ...], list[float]]] = {}
        self.states = [LayerState() for _ in range(LAYERS)]
        self._build_weights()
        if add_fp8_residue:
            self.tensors["blocks.0.projections.fp8_scale_fwd"] = (
                "F32",
                (2,),
                [1.0, 1.0],
            )

    def add_bf16(self, name: str, shape: tuple[int, ...], scale: float, *, center: float = 0.0) -> None:
        count = math.prod(shape)
        self.tensors[name] = ("BF16", shape, deterministic(name, count, scale, center=center))

    def add_f32(self, name: str, shape: tuple[int, ...], scale: float, *, center: float = 0.0) -> None:
        count = math.prod(shape)
        self.tensors[name] = ("F32", shape, f32_values(name, count, scale, center=center))

    def _build_weights(self) -> None:
        self.add_bf16("embedding_layer.weight", (VOCAB, WIDTH), 0.025)
        self.tensors["unembed.weight"] = ("BF16", (VOCAB, WIDTH), [bf16(0.75)] * (VOCAB * WIDTH))
        self.add_f32("norm.scale", (WIDTH,), 0.002, center=1.0)
        for layer in range(LAYERS):
            prefix = f"blocks.{layer}"
            self.add_f32(f"{prefix}.pre_norm.scale", (WIDTH,), 0.002, center=1.0)
            self.add_f32(f"{prefix}.post_norm.scale", (WIDTH,), 0.002, center=1.0)
            self.add_bf16(f"{prefix}.mlp.l1.weight", (INNER, WIDTH), 0.004)
            self.add_bf16(f"{prefix}.mlp.l2.weight", (INNER, WIDTH), 0.004)
            self.add_bf16(f"{prefix}.mlp.l3.weight", (WIDTH, INNER), 0.004)
            if layer in ATTENTION:
                self.add_bf16(f"{prefix}.inner_mha_cls.Wqkv.weight", (3 * WIDTH, WIDTH), 0.006)
                self.add_bf16(f"{prefix}.inner_mha_cls.out_proj.weight", (WIDTH, WIDTH), 0.006)
                self.add_bf16(f"{prefix}.inner_mha_cls.out_proj.bias", (WIDTH,), 0.001)
                inverse = [f32(1.0), f32(0.01)]
                self.tensors[f"{prefix}.inner_mha_cls.rotary_emb.inv_freq"] = (
                    "F32",
                    (HEAD_DIM // 2,),
                    inverse,
                )
            else:
                self.add_bf16(f"{prefix}.projections.weight", (3 * WIDTH, WIDTH), 0.006)
                self.add_bf16(f"{prefix}.out_filter_dense.weight", (WIDTH, WIDTH), 0.006)
                self.add_bf16(f"{prefix}.out_filter_dense.bias", (WIDTH,), 0.001)
                self.add_bf16(f"{prefix}.filter.short_filter_weight", (3 * WIDTH, 1, SHORT), 0.02)
                if layer in HCS:
                    self.add_bf16(f"{prefix}.filter.h", (HCS_GROUPS, 1, HCS_KERNEL), 0.02)
                elif layer in HCM:
                    self.add_f32(f"{prefix}.filter.h", (HCM_GROUPS, 1, HCM_KERNEL), 0.02)
                    self.add_bf16(f"{prefix}.filter.D", (WIDTH,), 0.003)
                else:
                    self.add_bf16(f"{prefix}.filter.D", (WIDTH,), 0.003)
                    poles = [-0.02 * (index % STATE + 1) for index in range(WIDTH * STATE)]
                    self.tensors[f"{prefix}.filter.log_poles"] = ("F32", (WIDTH, STATE, 1), [f32(x) for x in poles])
                    self.add_f32(f"{prefix}.filter.residues", (WIDTH, STATE), 0.004)

    def tensor(self, name: str) -> list[float]:
        return self.tensors[name][2]

    def mixer(self, hidden: list[list[float]], layer: int, *, prefill: bool) -> list[list[float]]:
        prefix = f"blocks.{layer}"
        state = self.states[layer]
        if layer in ATTENTION:
            projected = linear(hidden, self.tensor(f"{prefix}.inner_mha_cls.Wqkv.weight"), 3 * WIDTH)
            query = [row[:WIDTH] for row in projected]
            key = [row[WIDTH : 2 * WIDTH] for row in projected]
            value = [row[2 * WIDTH :] for row in projected]
            offset = len(state.key)
            inverse = self.tensor(f"{prefix}.inner_mha_cls.rotary_emb.inv_freq")
            query = apply_rope(query, inverse, offset)
            key = apply_rope(key, inverse, offset)
            state.key.extend(key)
            state.value.extend(value)
            return attention_chunk(query, state.key, state.value, offset)

        projected = linear(hidden, self.tensor(f"{prefix}.projections.weight"), 3 * WIDTH)
        short_weight = self.tensor(f"{prefix}.filter.short_filter_weight")
        if prefill:
            filtered = grouped_fir(projected, short_weight, 3 * WIDTH, SHORT, causal=False)
            state.short = cache_fir(projected, SHORT)
        else:
            assert state.short is not None
            filtered = [fir_decode(projected[0], state.short, short_weight, 3 * WIDTH, SHORT, causal=False)]
        x2, x1, value = split_triples(filtered)
        gated = gate(x1, value)
        if layer in HCS:
            weight = self.tensor(f"{prefix}.filter.h")
            if prefill:
                mixed = grouped_fir(gated, weight, HCS_GROUPS, HCS_KERNEL, causal=False)
                state.inner = cache_fir(gated, HCS_KERNEL)
            else:
                assert state.inner is not None
                mixed = [fir_decode(gated[0], state.inner, weight, HCS_GROUPS, HCS_KERNEL, causal=False)]
            return gate(mixed, x2)
        if layer in HCM:
            weight = self.tensor(f"{prefix}.filter.h")
            direct = self.tensor(f"{prefix}.filter.D")
            if prefill:
                mixed = grouped_fir(gated, weight, HCM_GROUPS, HCM_KERNEL, causal=True, direct=direct)
                state.inner = cache_fir(gated, HCM_KERNEL)
            else:
                assert state.inner is not None
                mixed = [fir_decode(gated[0], state.inner, weight, HCM_GROUPS, HCM_KERNEL, causal=True, direct=direct)]
            return gate(mixed, x2)
        direct = self.tensor(f"{prefix}.filter.D")
        poles = self.tensor(f"{prefix}.filter.log_poles")
        residues = self.tensor(f"{prefix}.filter.residues")
        if prefill:
            mixed, state.iir = hcl_prefill(x2, x1, value, direct, poles, residues)
            return mixed
        assert state.iir is not None
        return [hcl_decode(x2[0], x1[0], value[0], direct, poles, residues, state.iir)]

    def forward(self, tokens: list[int], *, prefill: bool) -> tuple[list[list[float]], list[list[float]] | None]:
        embedding = self.tensor("embedding_layer.weight")
        hidden = [embedding[token * WIDTH : (token + 1) * WIDTH] for token in tokens]
        dumped = None
        for layer in range(LAYERS):
            prefix = f"blocks.{layer}"
            normalized = rms_norm(hidden, self.tensor(f"{prefix}.pre_norm.scale"))
            mixed = self.mixer(normalized, layer, prefill=prefill)
            if layer in ATTENTION:
                output_weight = self.tensor(f"{prefix}.inner_mha_cls.out_proj.weight")
                output_bias = self.tensor(f"{prefix}.inner_mha_cls.out_proj.bias")
            else:
                output_weight = self.tensor(f"{prefix}.out_filter_dense.weight")
                output_bias = self.tensor(f"{prefix}.out_filter_dense.bias")
            residual = add(linear(mixed, output_weight, WIDTH, output_bias), hidden)
            post = rms_norm(residual, self.tensor(f"{prefix}.post_norm.scale"))
            first = linear(post, self.tensor(f"{prefix}.mlp.l1.weight"), INNER)
            second = linear(post, self.tensor(f"{prefix}.mlp.l2.weight"), INNER)
            gated = []
            for arow, brow in zip(first, second, strict=True):
                row = []
                for a, b in zip(arow, brow, strict=True):
                    activated = f32(0.5 * a * (1.0 + math.erf(a / math.sqrt(2.0)))) if layer == 0 else a
                    row.append(bf16(f32(activated * b)))
                gated.append(row)
            hidden = add(linear(gated, self.tensor(f"{prefix}.mlp.l3.weight"), WIDTH), residual)
            if layer == DUMP_LAYER and prefill:
                dumped = [row.copy() for row in hidden]
        normalized = rms_norm(hidden, self.tensor("norm.scale"))
        logits = linear(normalized, embedding, VOCAB)
        return logits, dumped

    def sources(self) -> list[BytesTensorSource]:
        sources = []
        for name, (dtype, shape, values) in self.tensors.items():
            payload = bf16_payload(values) if dtype == "BF16" else struct.pack(f"<{len(values)}f", *values)
            sources.append(BytesTensorSource(name, dtype, shape, payload))
        return sources


def write_f32(path: Path, rows: list[list[float]]) -> None:
    flat = [value for row in rows for value in row]
    path.write_bytes(struct.pack(f"<{len(flat)}f", *flat))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--expected-logits", type=Path)
    parser.add_argument("--expected-decode", type=Path)
    parser.add_argument("--expected-layer", type=Path)
    parser.add_argument("--expected-chunked", type=Path)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--hyena-projection-dtype", default="BF16")
    parser.add_argument("--use-fp8-input-projections", action="store_true")
    parser.add_argument("--add-fp8-residue", action="store_true")
    args = parser.parse_args()
    if not args.config_only and any(
        path is None
        for path in (args.expected_logits, args.expected_decode, args.expected_layer)
    ):
        parser.error(
            "--expected-logits, --expected-decode, and --expected-layer "
            "are required unless --config-only is used"
        )
    model = TinyModel(add_fp8_residue=args.add_fp8_residue)
    metadata = {
        "runtime.abi": "evo2-safetensors-v1",
        "model.name": "tiny-evo2c-50l",
        "model.architecture": "StripedHyena2Test",
        "fixture.synthetic": True,
        "config.vocab_size": VOCAB,
        "config.hidden_size": WIDTH,
        "config.num_layers": LAYERS,
        "config.num_attention_heads": HEADS,
        "config.state_size": STATE,
        "config.inner_mlp_size": INNER,
        "config.short_filter_length": SHORT,
        "config.hcs_filter_length": HCS_KERNEL,
        "config.hcm_filter_length": HCM_KERNEL,
        "config.hcs_filter_groups": HCS_GROUPS,
        "config.hcm_filter_groups": HCM_GROUPS,
        "config.hcs_layer_idxs": HCS,
        "config.hcm_layer_idxs": HCM,
        "config.hcl_layer_idxs": HCL,
        "config.attn_layer_idxs": ATTENTION,
        "config.eps": EPSILON,
        "config.rotary_emb_scaling_factor": ROPE_SCALE,
        "config.tie_embeddings": True,
        "config.column_split_hyena": False,
        "config.interleave": True,
        "config.hyena_flip_x1x2": False,
        "config.use_fp8_input_projections": args.use_fp8_input_projections,
        "hyena_projection_dtype": args.hyena_projection_dtype,
        "hcm_filter_dtype": "F32",
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    write_model(args.model, metadata, model.sources(), force=True)
    if args.config_only:
        return 0

    logits, layer = model.forward(PROMPT, prefill=True)
    decode, _ = model.forward([DECODE_TOKEN], prefill=False)
    assert layer is not None
    assert args.expected_logits is not None
    assert args.expected_decode is not None
    assert args.expected_layer is not None
    write_f32(args.expected_logits, logits)
    write_f32(args.expected_decode, decode)
    write_f32(args.expected_layer, layer)
    if args.expected_chunked is not None:
        chunked_oracle = TinyModel()
        chunked_logits, _ = chunked_oracle.forward(CHUNKED_PROMPT, prefill=True)
        write_f32(args.expected_chunked, chunked_logits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
