#!/usr/bin/env python3
"""Generate the full tiny Evo fixture and run the portable CPU backend."""

from __future__ import annotations

import argparse
import http.client
import json
import selectors
import shutil
import subprocess
import sys
import time
from pathlib import Path


def wait_for_server(process: subprocess.Popen[bytes]) -> int:
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stderr, selectors.EVENT_READ)
    deadline = time.monotonic() + 30
    prefix = b"evo_server listening 127.0.0.1:"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(process.stderr.read().decode(errors="replace"))
        for key, _ in selector.select(timeout=0.25):
            line = key.fileobj.readline()
            if line.startswith(prefix):
                return int(line[len(prefix) :])
    raise AssertionError("CPU server did not announce its port")


def server_json(port: int, method: str, path: str, payload: object | None = None) -> dict[str, object]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = None if payload is None else json.dumps(payload)
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    document = json.loads(response.read())
    connection.close()
    if response.status != 200:
        raise AssertionError(f"CPU server {path} failed: {response.status} {document}")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--c-api-binary", required=True, type=Path)
    parser.add_argument("--cli-binary", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    model = args.work_dir / "tiny-cpu.safetensors"
    logits = args.work_dir / "logits.f32"
    decode = args.work_dir / "decode.f32"
    layer = args.work_dir / "layer.f32"
    subprocess.run(
        [
            sys.executable,
            str(args.generator),
            "--model",
            str(model),
            "--expected-logits",
            str(logits),
            "--expected-decode",
            str(decode),
            "--expected-layer",
            str(layer),
        ],
        check=True,
    )
    subprocess.run(
        [str(args.binary), str(model), str(logits), str(decode), str(layer)],
        check=True,
    )
    subprocess.run([str(args.c_api_binary), str(model)], check=True)
    sequence = args.work_dir / "sequence.fa"
    sequence.write_text(">cpu\nAACCGGTT\n", encoding="ascii")
    score = subprocess.run(
        [
            str(args.cli_binary),
            "-m",
            str(model),
            "--score",
            str(sequence),
            "--ctx",
            "12",
            "--backend",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    score_document = json.loads(score.stdout)
    if (
        score_document["backend"] != "cpu"
        or score_document["profile"] != "cpu-f32"
        or score_document["scored_tokens"] != 7
        or '"kernel":"' not in score.stderr
    ):
        raise AssertionError("CPU CLI did not expose backend/profile/kernel")

    score_alias = subprocess.run(
        [
            str(args.cli_binary),
            "score",
            "-m",
            str(model),
            "--input",
            str(sequence),
            "--ctx",
            "12",
            "--backend",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if score_alias.stdout != score.stdout:
        raise AssertionError("score subcommand changed legacy scoring JSONL")

    generated = subprocess.run(
        [
            str(args.cli_binary),
            "-m",
            str(model),
            "-p",
            "AC",
            "-n",
            "2",
            "--ctx",
            "12",
            "--backend",
            "cpu",
            "--top-k",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    if len(generated.stdout) != 2:
        raise AssertionError("CPU CLI generation did not emit exact byte count")
    run_alias = subprocess.run(
        [
            str(args.cli_binary),
            "run",
            "-m",
            str(model),
            "-p",
            "AC",
            "-n",
            "2",
            "--ctx",
            "12",
            "--backend",
            "cpu",
            "--top-k",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    if run_alias.stdout != generated.stdout:
        raise AssertionError("run subcommand changed legacy generated bytes")

    benchmark = subprocess.run(
        [
            str(args.cli_binary),
            "bench",
            "-m",
            str(model),
            "--input",
            str(sequence),
            "--ctx",
            "12",
            "--backend",
            "cpu",
            "--warmup",
            "0",
            "--repetitions",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    benchmark_document = json.loads(benchmark.stdout)
    if any(
        (
            benchmark_document["command"] != "bench",
            benchmark_document["architecture"] != "StripedHyena2Test",
            benchmark_document["artifact_profile"] != "evo2-runtime-v1",
            benchmark_document["execution_profile"] != "cpu-f32",
            benchmark_document["backend"] != "cpu",
            benchmark_document["warmup"] != 0,
            benchmark_document["repetitions"] != 2,
            benchmark_document["tokens"] != 8,
            len(benchmark_document["samples_seconds"]) != 2,
            not benchmark_document["input"]["identity"].startswith("fnv1a64:"),
            benchmark_document["median_seconds"] <= 0,
            benchmark_document["median_tokens_per_second"] <= 0,
        )
    ):
        raise AssertionError("CPU benchmark omitted reproducibility identity")

    output = args.work_dir / "embeddings"
    shutil.rmtree(output, ignore_errors=True)
    subprocess.run(
        [
            str(args.cli_binary),
            "embed",
            "-m",
            str(model),
            "--input",
            str(sequence),
            "--output",
            str(output),
            "--layer",
            "17",
            "--pooling",
            "mean",
            "--ctx",
            "12",
            "--backend",
            "cpu",
        ],
        check=True,
    )
    embedding = json.loads((output / "embeddings.jsonl").read_text())
    if embedding["backend"] != "cpu" or embedding["profile"] != "cpu-f32":
        raise AssertionError("CPU embedding manifest omitted execution mode")

    variant = subprocess.run(
        [
            str(args.cli_binary),
            "variant-score",
            "-m",
            str(model),
            "--sequence",
            "AACCGGTT",
            "--position",
            "3",
            "--ref",
            "C",
            "--alt",
            "T",
            "--window",
            "6",
            "--strand",
            "both",
            "--normalization",
            "mean",
            "--ctx",
            "12",
            "--backend",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    variant_document = json.loads(variant.stdout)
    if (
        variant_document["backend"] != "cpu"
        or variant_document["profile"] != "cpu-f32"
        or len(variant_document["strands"]) != 2
    ):
        raise AssertionError("CPU variant output omitted backend contract")

    server = subprocess.Popen(
        [
            str(args.cli_binary),
            "serve",
            "-m",
            str(model),
            "--ctx",
            "12",
            "--backend",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--max-queue",
            "4",
            "--max-batch",
            "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        port = wait_for_server(server)
        health = server_json(port, "GET", "/health")
        scored = server_json(port, "POST", "/v1/score", {"sequence": "ACGT"})
        if (
            health["backend"] != "cpu"
            or health["execution_profile"] != "cpu-f32"
            or scored["profile"] != "cpu-f32"
        ):
            raise AssertionError("CPU server did not expose backend/profile")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
