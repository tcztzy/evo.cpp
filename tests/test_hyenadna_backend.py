#!/usr/bin/env python3
"""Architecture-registry, converter, CPU, C ABI, CLI, and server contract."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs)


def post(port: int, path: str, document: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(document).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def load_raw_f32(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    payload = path.read_bytes()
    if len(payload) % 4 != 0:
        raise AssertionError("raw F32 oracle has a partial element")
    count = len(payload) // 4
    return (8, 16), struct.unpack(f"<{count}f", payload)


def load_npy_f32(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    payload = path.read_bytes()
    if not payload.startswith(b"\x93NUMPY") or len(payload) < 10:
        raise AssertionError("logit output is not an NPY file")
    major = payload[6]
    if major == 1:
        header_size = struct.unpack_from("<H", payload, 8)[0]
        header_offset = 10
    elif major in (2, 3):
        header_size = struct.unpack_from("<I", payload, 8)[0]
        header_offset = 12
    else:
        raise AssertionError(f"unsupported NPY version {major}")
    header = ast.literal_eval(
        payload[header_offset : header_offset + header_size].decode("latin1").strip()
    )
    if header.get("descr") not in ("<f4", "=f4") or header.get("fortran_order"):
        raise AssertionError("logit NPY must be little-endian row-major F32")
    shape = tuple(int(item) for item in header.get("shape", ()))
    count = math.prod(shape)
    data_offset = header_offset + header_size
    if len(payload) - data_offset != count * 4:
        raise AssertionError("logit NPY payload size does not match shape")
    return shape, struct.unpack_from(f"<{count}f", payload, data_offset)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--inspector", required=True, type=Path)
    parser.add_argument("--c-api-binary", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    runtime = args.work_dir / "fixture.safetensors"
    source = args.work_dir / "source.safetensors"
    config = args.work_dir / "config.json"
    oracle = args.work_dir / "oracle.bin"
    converted = args.work_dir / "converted.safetensors"
    for path in (runtime, source, config, oracle, converted):
        path.unlink(missing_ok=True)
    run(
        [
            sys.executable,
            str(args.generator),
            "--runtime",
            str(runtime),
            "--source",
            str(source),
            "--config",
            str(config),
            "--oracle",
            str(oracle),
        ],
        env={**os.environ, "PYTHONPATH": str(args.generator.parent)},
    )
    converted_result = run(
        [
            sys.executable,
            str(args.converter),
            "--input",
            str(source),
            "--config",
            str(config),
            "--output",
            str(converted),
            "--model-id",
            "tiny-hyenadna",
            "--revision",
            "fixture-revision",
            "--force",
        ],
        env={**os.environ, "PYTHONPATH": str(args.generator.parent)},
    )
    if "source_sha256=" not in converted_result.stdout:
        raise AssertionError("converter did not publish source identity")
    inspected = run([str(args.inspector), str(converted)]).stdout
    if (
        "profile=hyenadna-runtime-v1" not in inspected
        or "model.architecture type=string value=HyenaDNA" not in inspected
        or "validation=ok" not in inspected
    ):
        raise AssertionError("native artifact reader did not accept HyenaDNA profile")

    bad_config = args.work_dir / "bad-config.json"
    bad = json.loads(config.read_text(encoding="utf-8"))
    bad["hyena_order"] = 3
    bad_config.write_text(json.dumps(bad), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(args.converter),
            "--input",
            str(source),
            "--config",
            str(bad_config),
            "--output",
            str(args.work_dir / "bad.safetensors"),
            "--force",
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(args.generator.parent)},
    )
    if rejected.returncode == 0 or "outside the registered" not in rejected.stderr:
        raise AssertionError("converter accepted unsupported HyenaDNA semantics")

    run([str(args.c_api_binary), str(converted)])
    fasta = args.work_dir / "input.fa"
    fasta.write_text(">tiny\nACGTNACG\n", encoding="ascii")
    logits = args.work_dir / "logits.npy"
    score = run(
        [
            str(args.binary),
            "-m",
            str(converted),
            "--backend",
            "cpu",
            "--ctx",
            "16",
            "--score",
            str(fasta),
            "--dump-logits",
            str(logits),
        ]
    )
    document = json.loads(score.stdout)
    if (
        document["model_id"] != "tiny-hyenadna"
        or document["profile"] != "cpu-f32"
        or document["tokens"] != 8
        or document["scored_tokens"] != 7
        or '"kernel":"direct-convolution-f32"' not in score.stderr
    ):
        raise AssertionError("HyenaDNA score surface omitted runtime identity")
    expected_shape, expected = load_raw_f32(oracle)
    actual_shape, actual = load_npy_f32(logits)
    if actual_shape != expected_shape:
        raise AssertionError(
            f"HyenaDNA logit shape mismatch: {actual_shape} != {expected_shape}"
        )
    maximum_error = max(abs(left - right) for left, right in zip(actual, expected))
    if maximum_error > 1e-5:
        raise AssertionError(f"HyenaDNA logits exceeded oracle envelope: {maximum_error}")

    generated = run(
        [
            str(args.binary),
            "-m",
            str(converted),
            "--backend",
            "cpu",
            "--ctx",
            "16",
            "-p",
            "ACGTNACG",
            "-n",
            "1",
            "--top-k",
            "1",
        ]
    )
    if generated.stdout != "T":
        raise AssertionError("HyenaDNA generation did not detokenize DNA")

    embeddings = args.work_dir / "embeddings"
    if embeddings.exists():
        for path in embeddings.iterdir():
            path.unlink()
        embeddings.rmdir()
    run(
        [
            str(args.binary),
            "embed",
            "-m",
            str(converted),
            "--backend",
            "cpu",
            "--ctx",
            "16",
            "--input",
            str(fasta),
            "--output",
            str(embeddings),
            "--layer",
            "1",
            "--pooling",
            "mean",
        ]
    )
    embedding = json.loads((embeddings / "embeddings.jsonl").read_text())
    if embedding["shape"] != [1, 4] or embedding["model_id"] != "tiny-hyenadna":
        raise AssertionError("HyenaDNA embedding contract is incomplete")

    variant = run(
        [
            str(args.binary),
            "variant-score",
            "-m",
            str(converted),
            "--backend",
            "cpu",
            "--ctx",
            "16",
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
        ]
    )
    if len(json.loads(variant.stdout)["strands"]) != 2:
        raise AssertionError("HyenaDNA variant scoring omitted a strand")

    port = free_port()
    server = subprocess.Popen(
        [
            str(args.binary),
            "serve",
            "-m",
            str(converted),
            "--backend",
            "cpu",
            "--ctx",
            "16",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-sequence-bytes",
            "16",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        health: dict[str, object] | None = None
        for _ in range(100):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=0.2
                ) as response:
                    health = json.loads(response.read())
                break
            except OSError:
                time.sleep(0.02)
        if health is None:
            raise AssertionError("HyenaDNA server did not become healthy")
        scored = post(port, "/v1/score", {"sequence": "ACGTNACG"})
        if (
            health["architecture"] != "HyenaDNA"
            or health["profile"] != "hyenadna-runtime-v1"
            or scored["scored_tokens"] != 7
        ):
            raise AssertionError("server did not route through architecture tokenizer")
    finally:
        server.terminate()
        server.wait(timeout=10)
    report = {
        "architecture": "HyenaDNA",
        "artifact_profile": "hyenadna-runtime-v1",
        "maximum_logit_error": maximum_error,
        "oracle_shape": list(actual_shape),
    }
    (args.work_dir / "hyenadna-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
