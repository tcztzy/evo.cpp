#!/usr/bin/env python3
"""Exercise isolated CUDA contexts through every native HTTP route."""

import argparse
import concurrent.futures
import http.client
import json
import os
import selectors
import shutil
import subprocess
import sys
import time
from pathlib import Path


def visible_gpu_count() -> int:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return len(
            [
                item
                for item in configured.split(",")
                if item.strip() and item.strip() != "-1"
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
    return len(result.stdout.splitlines()) if result.returncode == 0 else 0


def request(
    port: int,
    method: str,
    path: str,
    payload: object | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str, bytes]:
    body = b"" if payload is None else json.dumps(payload).encode()
    request_headers = dict(headers or {})
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    response_body = response.read()
    content_type = response.getheader("Content-Type", "")
    status = response.status
    connection.close()
    return status, content_type, response_body


def json_request(
    port: int,
    path: str,
    payload: object,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    status, content_type, body = request(
        port, "POST", path, payload=payload, headers=headers
    )
    if "application/json" not in content_type:
        raise AssertionError(f"{path} returned content type {content_type!r}")
    return status, json.loads(body)


def wait_for_server(process: subprocess.Popen[bytes]) -> tuple[int, bytes]:
    selector = selectors.DefaultSelector()
    assert process.stderr is not None
    selector.register(process.stderr, selectors.EVENT_READ)
    captured = bytearray()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            captured.extend(process.stderr.read())
            raise AssertionError(
                f"server exited before listening ({process.returncode}): "
                f"{captured.decode(errors='replace')}"
            )
        for key, _ in selector.select(timeout=0.25):
            line = key.fileobj.readline()
            captured.extend(line)
            prefix = b"evo_server listening 127.0.0.1:"
            if line.startswith(prefix):
                return int(line[len(prefix) :]), bytes(captured)
    raise AssertionError(
        "server did not announce a port: " + captured.decode(errors="replace")
    )


def metric(text: str, name: str) -> int:
    for line in text.splitlines():
        key, value = line.split()
        if key == name:
            return int(value)
    raise AssertionError(f"missing metric {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    if visible_gpu_count() < 1:
        print("SKIP: one visible CUDA device is required")
        return 77

    args.work_dir.mkdir(parents=True, exist_ok=True)
    model = args.work_dir / "tiny-server-50l-512v.safetensors"
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
    command = [
        str(args.binary),
        "serve",
        "-m",
        str(model),
        "--ctx",
        "12",
        "--gpu",
        "0",
        "--port",
        "0",
        "--max-queue",
        "16",
        "--max-batch",
        "4",
        "--batch-window-ms",
        "50",
        "--max-request-bytes",
        "4096",
        "--max-sequence-bytes",
        "12",
        "--max-embedding-values",
        "128",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    startup = b""
    try:
        port, startup = wait_for_server(process)
        status, content_type, body = request(port, "GET", "/health")
        health = json.loads(body)
        if status != 200 or "application/json" not in content_type or any(
            (
                health["status"] != "ok",
                health["backend"] != "cuda",
                health["profile"] != "evo2-runtime-v1",
                health["context_size"] != 12,
                health["batching"] != "isolated_context_microbatch",
            )
        ):
            raise AssertionError(f"health contract failed: {status} {health}")

        score_status, score = json_request(
            port, "/v1/score", {"sequence": "ACGTAC"}
        )
        if (
            score_status != 200
            or score["scored_tokens"] != 5
            or len(score["token_log_likelihoods"]) != 5
        ):
            raise AssertionError(f"score contract failed: {score_status} {score}")

        generation_status, generation = json_request(
            port,
            "/v1/generate",
            {"prompt": "AC", "max_tokens": 3, "top_k": 1, "seed": 7},
        )
        if (
            generation_status != 200
            or generation["generated_tokens"] != 3
            or len(generation["tokens"]) != 3
        ):
            raise AssertionError(
                f"generation contract failed: {generation_status} {generation}"
            )

        embedding_status, embedding = json_request(
            port,
            "/v1/embeddings",
            {"sequence": "ACGT", "layer": 17, "pooling": "mean"},
        )
        if (
            embedding_status != 200
            or embedding["shape"] != [8]
            or len(embedding["embedding"]) != 8
        ):
            raise AssertionError(
                f"embedding contract failed: {embedding_status} {embedding}"
            )

        variant_status, variant = json_request(
            port,
            "/v1/variants",
            {
                "sequence": "AACCGGTT",
                "position": 3,
                "ref": "C",
                "alt": "T",
                "window": 6,
                "strand": "both",
                "normalization": "mean",
            },
        )
        if (
            variant_status != 200
            or len(variant["strands"]) != 2
            or variant["window_start"] != 0
            or variant["window_end"] != 6
        ):
            raise AssertionError(
                f"variant contract failed: {variant_status} {variant}"
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    json_request, port, "/v1/score", {"sequence": "ACGTAC"}
                )
                for _ in range(2)
            ]
            concurrent_scores = [future.result() for future in futures]
        if any(item[0] != 200 for item in concurrent_scores) or any(
            item[1]["token_log_likelihoods"]
            != score["token_log_likelihoods"]
            for item in concurrent_scores
        ):
            raise AssertionError(
                "concurrent request contexts changed deterministic score output"
            )

        limit_status, limit = json_request(
            port, "/v1/score", {"sequence": "A" * 13}
        )
        if limit_status != 413 or limit["error"]["code"] != "request_too_large":
            raise AssertionError(f"sequence limit failed closed: {limit_status} {limit}")

        oversized_status, _, _ = request(
            port, "POST", "/v1/score", {"sequence": "A" * 5000}
        )
        if oversized_status != 413:
            raise AssertionError(f"HTTP body limit returned {oversized_status}")

        deadline_status, deadline = json_request(
            port,
            "/v1/generate",
            {"prompt": "AC", "max_tokens": 8, "top_k": 1},
            headers={"X-Evo-Timeout-Ms": "0"},
        )
        if (
            deadline_status != 408
            or deadline["error"]["code"] != "deadline_exceeded"
        ):
            raise AssertionError(
                f"cancellation deadline failed: {deadline_status} {deadline}"
            )

        metrics_status, metrics_type, metrics_body = request(
            port, "GET", "/metrics"
        )
        metrics = metrics_body.decode()
        if metrics_status != 200 or "text/plain" not in metrics_type:
            raise AssertionError("metrics endpoint has the wrong HTTP contract")
        if any(
            (
                metric(metrics, "evo_scheduler_active_peak") < 2,
                metric(metrics, "evo_scheduler_batch_items_total")
                <= metric(metrics, "evo_scheduler_batches_total"),
                metric(metrics, "evo_scheduler_cancelled_total") < 1,
            )
        ):
            raise AssertionError(f"batching/cancellation metrics failed:\n{metrics}")
    finally:
        process.terminate()
        try:
            stdout, remaining_stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, remaining_stderr = process.communicate()
        if process.returncode not in (0, -15):
            raise AssertionError(
                f"server shutdown failed ({process.returncode}): "
                f"stdout={stdout!r}\n"
                f"stderr={(startup + remaining_stderr).decode(errors='replace')}"
            )
    print("CUDA server integration passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
