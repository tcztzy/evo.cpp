#!/usr/bin/env python3
"""Exercise the CUDA score and generation CLI with a synthetic 50-layer model."""

import argparse
import ast
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def visible_gpu_count() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        return len(
            [
                device
                for device in cuda_visible_devices.split(",")
                if device.strip() and device.strip() != "-1"
            ]
        )
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return 0
    result = subprocess.run(
        [executable, "--query-gpu=index", "--format=csv,noheader"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def npy_f32(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    payload = path.read_bytes()
    if payload[:8] != b"\x93NUMPY\x01\x00":
        raise AssertionError(f"{path} is not a NumPy v1 file")
    header_size = struct.unpack_from("<H", payload, 8)[0]
    header = ast.literal_eval(payload[10 : 10 + header_size].decode("ascii"))
    if header["descr"] != "<f4" or header["fortran_order"]:
        raise AssertionError(f"{path} is not a row-major little-endian F32 array")
    shape = tuple(header["shape"])
    count = math.prod(shape)
    values = struct.unpack_from(f"<{count}f", payload, 10 + header_size)
    if 10 + header_size + count * 4 != len(payload):
        raise AssertionError(f"{path} payload size does not match its shape")
    return shape, values


def npy_shape(path: Path) -> tuple[int, ...]:
    return npy_f32(path)[0]


def run_checked(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr.decode(errors='replace')}"
        )
    if b"evo_metrics {" not in result.stderr:
        raise AssertionError("successful CUDA CLI command omitted metrics")
    return result


def metrics(result: subprocess.CompletedProcess[bytes]) -> dict[str, object]:
    prefix = b"evo_metrics "
    for line in result.stderr.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise AssertionError("successful CUDA CLI command omitted parseable metrics")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    gpu_count = min(visible_gpu_count(), 4)
    if gpu_count < 1:
        print("SKIP: one visible CUDA device is required")
        return 77
    gpu_list = ",".join(str(index) for index in range(gpu_count))

    args.work_dir.mkdir(parents=True, exist_ok=True)
    model = args.work_dir / "tiny-50l-512v.safetensors"
    subprocess.run(
        [
            sys.executable,
            str(args.generator),
            "--model",
            str(model),
            "--expected-logits",
            str(args.work_dir / "expected-logits.f32"),
            "--expected-decode",
            str(args.work_dir / "expected-decode.f32"),
            "--expected-layer",
            str(args.work_dir / "expected-layer.f32"),
        ],
        check=True,
    )

    generation_logits = args.work_dir / "generation-logits.npy"
    generation_layer = args.work_dir / "generation-layer.npy"
    generation_command = [
        str(args.binary),
        "-m",
        str(model),
        "-p",
        "AC",
        "-n",
        "3",
        "--ctx",
        "8",
        "--gpu",
        gpu_list,
        "--top-k",
        "1",
        "--dump-logits",
        str(generation_logits),
        "--dump-layer",
        f"17:{generation_layer}",
    ]
    first = run_checked(generation_command)
    second = run_checked(generation_command)
    if len(first.stdout) != 3 or first.stdout != second.stdout:
        raise AssertionError("greedy generation is not byte-exact across repeated runs")
    if npy_shape(generation_logits) != (3, 512):
        raise AssertionError("generation logit dump has the wrong shape")
    if npy_shape(generation_layer) != (2, 8):
        raise AssertionError("generation layer dump has the wrong shape")

    full_prompt_logits = args.work_dir / "full-prompt-logits.npy"
    forced_prompt_logits = args.work_dir / "forced-prompt-logits.npy"
    prompt_command = [
        str(args.binary),
        "-m",
        str(model),
        "-p",
        "ACGT",
        "-n",
        "1",
        "--ctx",
        "8",
        "--gpu",
        gpu_list,
        "--top-k",
        "1",
    ]
    full_prompt = run_checked(
        [*prompt_command, "--dump-logits", str(full_prompt_logits)]
    )
    forced_prompt = run_checked(
        [
            *prompt_command,
            "--force-prompt-threshold",
            "2",
            "--dump-logits",
            str(forced_prompt_logits),
        ]
    )
    if full_prompt.stdout != forced_prompt.stdout:
        raise AssertionError(
            "teacher-forced prompt cache changed the greedy continuation"
        )
    if full_prompt_logits.read_bytes() != forced_prompt_logits.read_bytes():
        raise AssertionError(
            "teacher-forced prompt cache changed a raw generation-logit bit"
        )

    long_prompt_logits = args.work_dir / "long-prompt-chunked-logits.npy"
    long_forced_logits = args.work_dir / "long-prompt-forced-logits.npy"
    long_prompt_command = [
        str(args.binary),
        "-m",
        str(model),
        "-p",
        "ACGTACGTA",
        "-n",
        "1",
        "--ctx",
        "12",
        "--gpu",
        gpu_list,
        "--top-k",
        "1",
    ]
    long_chunked = run_checked(
        [*long_prompt_command, "--dump-logits", str(long_prompt_logits)]
    )
    long_forced = run_checked(
        [
            *long_prompt_command,
            "--force-prompt-threshold",
            "8",
            "--dump-logits",
            str(long_forced_logits),
        ]
    )
    if long_chunked.stdout != long_forced.stdout:
        raise AssertionError(
            "multi-token continuation changed the greedy token relative to "
            "one-token cached decode"
        )
    if long_prompt_logits.read_bytes() != long_forced_logits.read_bytes():
        raise AssertionError(
            "default and explicit exact prompt forcing changed a raw logit bit"
        )
    chunked_metrics = metrics(long_chunked)
    forced_metrics = metrics(long_forced)
    if (
        chunked_metrics["prefill_tokens"] != 8
        or chunked_metrics["teacher_force_tokens"] != 1
        or chunked_metrics["kv_cache"] != "bf16_contiguous"
        or forced_metrics["prefill_tokens"] != 8
        or forced_metrics["teacher_force_tokens"] != 1
    ):
        raise AssertionError(
            "default or explicit exact prompt-forcing metrics are incorrect"
        )

    rejected_q8 = subprocess.run(
        [
            str(args.binary),
            "-m",
            str(model),
            "-p",
            "AC",
            "-n",
            "1",
            "--ctx",
            "131072",
            "--gpu",
            gpu_list,
            "--top-k",
            "1",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if (
        rejected_q8.returncode == 0
        or b"exact Vortex cached attention requires a BF16 KV cache"
        not in rejected_q8.stderr
    ):
        raise AssertionError(
            "exact generation silently accepted the approximate Q8 KV path"
        )

    rejected_threshold = subprocess.run(
        [
            *long_prompt_command,
            "--force-prompt-threshold",
            "9",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if (
        rejected_threshold.returncode == 0
        or b"exact cached prefill must fit one activation chunk"
        not in rejected_threshold.stderr
    ):
        raise AssertionError(
            "oversized exact cached prefill threshold was not rejected"
        )

    score_input = args.work_dir / "score.txt"
    score_input.write_bytes(b"ACGT")
    score_logits = args.work_dir / "score-logits.npy"
    score_result = run_checked(
        [
            str(args.binary),
            "-m",
            str(model),
            "--score",
            str(score_input),
            "--ctx",
            "8",
            "--gpu",
            gpu_list,
            "--dump-logits",
            str(score_logits),
        ]
    )
    score = json.loads(score_result.stdout)
    if score["tokens"] != 4 or score["scored_tokens"] != 3:
        raise AssertionError("score JSONL reported the wrong token counts")
    if len(score["token_log_likelihoods"]) != 3:
        raise AssertionError("score JSONL omitted per-token log likelihoods")
    if not all(
        math.isfinite(value)
        for value in [
            score["log_likelihood"],
            score["mean_log_likelihood"],
            score["perplexity"],
            *score["token_log_likelihoods"],
        ]
    ):
        raise AssertionError("score JSONL contains non-finite values")
    if npy_shape(score_logits) != (4, 512):
        raise AssertionError("score logit dump has the wrong shape")
    _, score_values = npy_f32(score_logits)
    expected_scores = []
    for row, target in enumerate(b"CGT"):
        logits = score_values[row * 512 : (row + 1) * 512]
        maximum = max(logits)
        normalizer = math.fsum(math.exp(value - maximum) for value in logits)
        expected_scores.append(logits[target] - maximum - math.log(normalizer))
    if any(
        not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected in zip(
            score["token_log_likelihoods"], expected_scores, strict=True
        )
    ):
        raise AssertionError(
            "score JSONL did not apply exactly one log-softmax normalization"
        )

    long_score_input = args.work_dir / "long-score.txt"
    long_score_input.write_bytes(b"ACGTACGTA")
    long_score_logits = args.work_dir / "long-score-logits.npy"
    long_score_result = run_checked(
        [
            str(args.binary),
            "-m",
            str(model),
            "--score",
            str(long_score_input),
            "--ctx",
            "12",
            "--gpu",
            gpu_list,
            "--dump-logits",
            str(long_score_logits),
        ]
    )
    long_score = json.loads(long_score_result.stdout)
    if (
        long_score["tokens"] != 9
        or long_score["scored_tokens"] != 8
        or npy_shape(long_score_logits) != (9, 512)
        or metrics(long_score_result)["prefill_tokens"] != 9
    ):
        raise AssertionError(
            "score mode did not concatenate logits across activation chunks"
        )

    print("CUDA CLI tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
