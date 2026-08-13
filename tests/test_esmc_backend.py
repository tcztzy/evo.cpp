#!/usr/bin/env python3
"""Generate and execute the deterministic native ESMC CPU acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--c-api-binary", type=Path)
    parser.add_argument("--cli-binary", type=Path)
    parser.add_argument("--cuda-c-api-binary", type=Path)
    parser.add_argument("--cuda-cli-binary", type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.generator.parent)
    generated = subprocess.run(
        [sys.executable, str(args.generator), "--output-dir", str(args.work_dir)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    artifact = Path(generated.stdout.splitlines()[0])
    subprocess.run(
        [str(args.binary), str(artifact), str(args.work_dir)],
        check=True,
    )
    if args.c_api_binary is not None:
        subprocess.run([str(args.c_api_binary), str(artifact)], check=True)
    if args.cuda_c_api_binary is not None:
        subprocess.run(
            [str(args.cuda_c_api_binary), str(artifact), "cuda"], check=True
        )
    if args.cli_binary is not None:
        sequence_path = args.work_dir / "cli-input.fa"
        sequence_path.write_text(">protein\nLAGV\n", encoding="ascii")
        logits_dir = args.work_dir / "cli-logits"
        embedding_dir = args.work_dir / "cli-embedding"
        shutil.rmtree(logits_dir, ignore_errors=True)
        shutil.rmtree(embedding_dir, ignore_errors=True)
        subprocess.run(
            [
                str(args.cli_binary),
                "logits",
                "-m",
                str(artifact),
                "--input",
                str(sequence_path),
                "--output",
                str(logits_dir),
                "--ctx",
                "16",
                "--backend",
                "cpu",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(args.cli_binary),
                "embed",
                "-m",
                str(artifact),
                "--input",
                str(sequence_path),
                "--output",
                str(embedding_dir),
                "--layer",
                "2",
                "--pooling",
                "none",
                "--ctx",
                "16",
                "--backend",
                "cpu",
            ],
            check=True,
        )
        assert (logits_dir / "000000.npy").is_file()
        assert '"shape":[6,64]' in (
            logits_dir / "logits.jsonl"
        ).read_text(encoding="utf-8")
        assert '"shape":[6,4]' in (
            embedding_dir / "embeddings.jsonl"
        ).read_text(encoding="utf-8")
        revision = "e" * 40
        cache_home = args.work_dir / "esmc-hf-cache"
        hub = cache_home / "huggingface" / "hub"
        repo_root = hub / "models--owner--esmc-runtime"
        snapshot = repo_root / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        cached_artifact = snapshot / "esmc.safetensors"
        shutil.copyfile(artifact, cached_artifact)
        artifact_payload = cached_artifact.read_bytes()
        runtime_manifest = {
            "schema_version": 1,
            "artifact_profile": "esmc-runtime-v1",
            "model_id": "esmc_tiny_test",
            "load_path": cached_artifact.name,
            "files": [
                {
                    "path": cached_artifact.name,
                    "size": len(artifact_payload),
                    "sha256": hashlib.sha256(artifact_payload).hexdigest(),
                }
            ],
        }
        manifest_payload = json.dumps(runtime_manifest, sort_keys=True).encode()
        (snapshot / "evo-artifact.json").write_bytes(manifest_payload)
        (repo_root / "refs").mkdir(parents=True, exist_ok=True)
        (repo_root / "refs" / "main").write_text(
            revision + "\n", encoding="ascii"
        )
        receipt_dir = (
            hub
            / "evo-receipts"
            / "owner--esmc-runtime"
            / revision
        )
        receipt_dir.mkdir(parents=True, exist_ok=True)
        (receipt_dir / "runtime-artifact.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "runtime-artifact",
                    "repo": "owner/esmc-runtime",
                    "resolved_revision": revision,
                    "artifact_profile": "esmc-runtime-v1",
                    "model_id": "esmc_tiny_test",
                    "manifest_sha256": hashlib.sha256(
                        manifest_payload
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        cached_logits_dir = args.work_dir / "cli-hf-logits"
        shutil.rmtree(cached_logits_dir, ignore_errors=True)
        cache_environment = os.environ.copy()
        cache_environment["EVO_CACHE_HOME"] = str(cache_home)
        subprocess.run(
            [
                str(args.cli_binary),
                "logits",
                "-hf",
                "owner/esmc-runtime@main",
                "--input",
                str(sequence_path),
                "--output",
                str(cached_logits_dir),
                "--ctx",
                "16",
                "--backend",
                "cpu",
            ],
            check=True,
            env=cache_environment,
        )
        assert (cached_logits_dir / "000000.npy").read_bytes() == (
            logits_dir / "000000.npy"
        ).read_bytes()
        rejected = subprocess.run(
            [
                str(args.cli_binary),
                "run",
                "-m",
                str(artifact),
                "-p",
                "LAGV",
                "-n",
                "1",
                "--ctx",
                "16",
                "--backend",
                "cpu",
            ],
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0
        assert "unsupported" in rejected.stderr
    if args.cuda_cli_binary is not None:
        sequence_path = args.work_dir / "cuda-cli-input.fa"
        sequence_path.write_text(">protein\nLAGV\n", encoding="ascii")
        logits_dir = args.work_dir / "cuda-cli-logits"
        embedding_dir = args.work_dir / "cuda-cli-embedding"
        shutil.rmtree(logits_dir, ignore_errors=True)
        shutil.rmtree(embedding_dir, ignore_errors=True)
        common = ["--ctx", "16", "--gpu", "0"]
        cuda_logits = subprocess.run(
            [
                str(args.cuda_cli_binary),
                "logits",
                "-m",
                str(artifact),
                "--input",
                str(sequence_path),
                "--output",
                str(logits_dir),
                *common,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert cuda_logits.stderr.count("evo_record_metrics ") == 1
        subprocess.run(
            [
                str(args.cuda_cli_binary),
                "embed",
                "-m",
                str(artifact),
                "--input",
                str(sequence_path),
                "--output",
                str(embedding_dir),
                "--layer",
                "2",
                "--pooling",
                "none",
                *common,
            ],
            check=True,
        )
        assert '"shape":[6,64]' in (
            logits_dir / "logits.jsonl"
        ).read_text(encoding="utf-8")
        assert '"shape":[6,4]' in (
            embedding_dir / "embeddings.jsonl"
        ).read_text(encoding="utf-8")
    print("ESMC backend contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
