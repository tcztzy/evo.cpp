#!/usr/bin/env python3
"""Exercise the macOS MPS backend across all registered model families."""

from __future__ import annotations

import argparse
import ast
import http.client
import json
import math
import os
import selectors
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path


class MpsUnavailable(RuntimeError):
    """The host satisfies the build contract but exposes no Metal device."""


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode == 7 and "no Metal device is available" in result.stderr:
        raise MpsUnavailable(result.stderr.strip())
    result.check_returncode()
    return result


def load_npy(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    payload = path.read_bytes()
    if not payload.startswith(b"\x93NUMPY") or len(payload) < 10:
        raise AssertionError(f"not an NPY file: {path}")
    if payload[6] == 1:
        header_size = struct.unpack_from("<H", payload, 8)[0]
        header_offset = 10
    elif payload[6] in (2, 3):
        header_size = struct.unpack_from("<I", payload, 8)[0]
        header_offset = 12
    else:
        raise AssertionError(f"unsupported NPY version: {payload[6]}")
    header = ast.literal_eval(
        payload[header_offset : header_offset + header_size].decode("latin1").strip()
    )
    shape = tuple(int(value) for value in header["shape"])
    count = math.prod(shape)
    data_offset = header_offset + header_size
    if header["descr"] not in ("<f4", "=f4") or header["fortran_order"]:
        raise AssertionError("expected row-major F32 NPY")
    if len(payload) - data_offset != count * 4:
        raise AssertionError("NPY payload size does not match its shape")
    return shape, struct.unpack_from(f"<{count}f", payload, data_offset)


def compare_npy(left: Path, right: Path, tolerance: float, label: str) -> None:
    left_shape, left_values = load_npy(left)
    right_shape, right_values = load_npy(right)
    if left_shape != right_shape:
        raise AssertionError(f"{label} shape mismatch: {left_shape} != {right_shape}")
    maximum = max(abs(a - b) for a, b in zip(left_values, right_values))
    if maximum > tolerance:
        raise AssertionError(f"{label} maximum error {maximum} > {tolerance}")


def assert_mps_identity(
    result: subprocess.CompletedProcess[str], label: str, *, stdout: bool = True
) -> None:
    if (
        (stdout and '"backend":"mps"' not in result.stdout)
        or (stdout and '"profile":"mps-f32"' not in result.stdout)
        or '"backend":"mps"' not in result.stderr
        or '"profile":"mps-f32"' not in result.stderr
        or '"kernel":"mps-f32-gemm+host-ops"' not in result.stderr
    ):
        raise AssertionError(f"{label} omitted transparent MPS runtime identity")


def wait_for_server(process: subprocess.Popen[bytes]) -> int:
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stderr, selectors.EVENT_READ)
    prefix = b"evo_server listening 127.0.0.1:"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(process.stderr.read().decode(errors="replace"))
        for key, _ in selector.select(timeout=0.25):
            line = key.fileobj.readline()
            if line.startswith(prefix):
                return int(line[len(prefix) :])
    raise AssertionError("MPS server did not announce its port")


def server_json(port: int, method: str, path: str, payload: object | None = None) -> dict[str, object]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = None if payload is None else json.dumps(payload)
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    document = json.loads(response.read())
    connection.close()
    if response.status != 200:
        raise AssertionError(f"MPS server {path}: {response.status} {document}")
    return document


def striped_gate(args: argparse.Namespace) -> None:
    directory = args.work_dir / "striped"
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "model.safetensors"
    run(
        [
            sys.executable,
            str(args.tiny_generator),
            "--model",
            str(model),
            "--expected-logits",
            str(directory / "expected-logits.f32"),
            "--expected-decode",
            str(directory / "expected-decode.f32"),
            "--expected-layer",
            str(directory / "expected-layer.f32"),
        ]
    )
    sequence = directory / "input.fa"
    sequence.write_text(">striped\nAACCGGTT\n", encoding="ascii")
    common = ["--input", str(sequence), "--ctx", "12"]
    cpu = run(
        [
            str(args.cli_binary),
            "score",
            "-m",
            str(model),
            *common,
            "--backend",
            "cpu",
            "--dump-logits",
            str(directory / "cpu.npy"),
        ]
    )
    mps = run(
        [
            str(args.cli_binary),
            "score",
            "-m",
            str(model),
            *common,
            "--backend",
            "mps",
            "--dump-logits",
            str(directory / "mps.npy"),
        ]
    )
    assert_mps_identity(mps, "StripedHyena2 score")
    compare_npy(directory / "cpu.npy", directory / "mps.npy", 1.0e-4, "StripedHyena2")
    if json.loads(cpu.stdout)["backend"] != "cpu":
        raise AssertionError("CPU control did not use the CPU backend")

    automatic = subprocess.run(
        [
            str(args.cli_binary),
            "score",
            "-m",
            str(model),
            *common,
            "--backend",
            "auto",
            "--gpu",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    if automatic.returncode == 0 or "built without CUDA" not in automatic.stderr:
        raise AssertionError("automatic backend selection silently chose MPS")

    benchmark = run(
        [
            str(args.cli_binary),
            "bench",
            "-m",
            str(model),
            *common,
            "--backend",
            "mps",
            "--warmup",
            "0",
            "--repetitions",
            "1",
        ]
    )
    benchmark_document = json.loads(benchmark.stdout)
    if benchmark_document["backend"] != "mps" or benchmark_document["execution_profile"] != "mps-f32":
        raise AssertionError("MPS benchmark omitted backend/profile")

    embedding_dir = directory / "embedding"
    shutil.rmtree(embedding_dir, ignore_errors=True)
    run(
        [
            str(args.cli_binary),
            "embed",
            "-m",
            str(model),
            *common,
            "--backend",
            "mps",
            "--output",
            str(embedding_dir),
            "--layer",
            "17",
            "--pooling",
            "mean",
        ]
    )
    manifest = json.loads((embedding_dir / "embeddings.jsonl").read_text())
    if manifest["backend"] != "mps" or manifest["profile"] != "mps-f32":
        raise AssertionError("MPS embedding manifest omitted backend/profile")

    variant = run(
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
            "mps",
        ]
    )
    if json.loads(variant.stdout)["backend"] != "mps":
        raise AssertionError("MPS variant output omitted backend")

    run([str(args.c_api_binary), str(model)])
    server = subprocess.Popen(
        [
            str(args.cli_binary),
            "serve",
            "-m",
            str(model),
            "--ctx",
            "12",
            "--backend",
            "mps",
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
            health["backend"] != "mps"
            or health["execution_profile"] != "mps-f32"
            or scored["profile"] != "mps-f32"
        ):
            raise AssertionError("MPS server omitted backend/profile")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def hyenadna_gate(args: argparse.Namespace) -> None:
    directory = args.work_dir / "hyenadna"
    directory.mkdir(parents=True, exist_ok=True)
    runtime = directory / "runtime.safetensors"
    run(
        [
            sys.executable,
            str(args.hyenadna_generator),
            "--runtime",
            str(runtime),
            "--source",
            str(directory / "source.safetensors"),
            "--config",
            str(directory / "config.json"),
            "--oracle",
            str(directory / "oracle.bin"),
        ],
        env={**os.environ, "PYTHONPATH": str(args.hyenadna_generator.parent)},
    )
    sequence = directory / "input.fa"
    sequence.write_text(">hyena\nACGTNACG\n", encoding="ascii")
    for backend in ("cpu", "mps"):
        result = run(
            [
                str(args.cli_binary),
                "score",
                "-m",
                str(runtime),
                "--input",
                str(sequence),
                "--ctx",
                "16",
                "--backend",
                backend,
                "--dump-logits",
                str(directory / f"{backend}.npy"),
            ]
        )
        if backend == "mps":
            assert_mps_identity(result, "HyenaDNA score")
    compare_npy(directory / "cpu.npy", directory / "mps.npy", 1.0e-5, "HyenaDNA")
    generated = run(
        [
            str(args.cli_binary),
            "run",
            "-m",
            str(runtime),
            "-p",
            "ACGTNACG",
            "-n",
            "1",
            "--ctx",
            "16",
            "--top-k",
            "1",
            "--backend",
            "mps",
        ]
    )
    if len(generated.stdout) != 1:
        raise AssertionError("MPS HyenaDNA generation did not emit one base")
    assert_mps_identity(generated, "HyenaDNA generation", stdout=False)


def esmc_gate(args: argparse.Namespace) -> None:
    directory = args.work_dir / "esmc"
    directory.mkdir(parents=True, exist_ok=True)
    generated = run(
        [sys.executable, str(args.esmc_generator), "--output-dir", str(directory)],
        env={**os.environ, "PYTHONPATH": str(args.esmc_generator.parent)},
    )
    artifact = Path(generated.stdout.splitlines()[0])
    sequence = directory / "input.fa"
    sequence.write_text(">protein\nLAGV\n", encoding="ascii")
    for backend in ("cpu", "mps"):
        output = directory / f"{backend}-logits"
        shutil.rmtree(output, ignore_errors=True)
        result = run(
            [
                str(args.cli_binary),
                "logits",
                "-m",
                str(artifact),
                "--input",
                str(sequence),
                "--output",
                str(output),
                "--ctx",
                "16",
                "--backend",
                backend,
            ]
        )
        if backend == "mps":
            assert_mps_identity(result, "ESMC logits", stdout=False)
    compare_npy(
        directory / "cpu-logits" / "000000.npy",
        directory / "mps-logits" / "000000.npy",
        1.0e-4,
        "ESMC",
    )
    manifest = json.loads((directory / "mps-logits" / "logits.jsonl").read_text())
    if manifest["backend"] != "mps" or manifest["profile"] != "mps-f32":
        raise AssertionError("ESMC MPS logits manifest omitted backend/profile")

    embedding = directory / "mps-embedding"
    shutil.rmtree(embedding, ignore_errors=True)
    run(
        [
            str(args.cli_binary),
            "embed",
            "-m",
            str(artifact),
            "--input",
            str(sequence),
            "--output",
            str(embedding),
            "--layer",
            "2",
            "--pooling",
            "none",
            "--ctx",
            "16",
            "--backend",
            "mps",
        ]
    )
    embedding_manifest = json.loads((embedding / "embeddings.jsonl").read_text())
    if embedding_manifest["backend"] != "mps" or embedding_manifest["shape"] != [6, 4]:
        raise AssertionError("ESMC MPS embedding contract is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-binary", required=True, type=Path)
    parser.add_argument("--c-api-binary", required=True, type=Path)
    parser.add_argument("--tiny-generator", required=True, type=Path)
    parser.add_argument("--hyenadna-generator", required=True, type=Path)
    parser.add_argument("--esmc-generator", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    try:
        striped_gate(args)
        hyenadna_gate(args)
        esmc_gate(args)
    except MpsUnavailable as error:
        print(f"SKIP: {error}")
        return 77
    print("MPS backend contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
