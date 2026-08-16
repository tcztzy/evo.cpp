#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract-test revision pinning, hashes, and runtime manifests offline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_fake_hub(root: Path) -> Path:
    package = root / "fake-python" / "huggingface_hub"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        """from pathlib import Path
import os
import shutil
from types import SimpleNamespace

class HfApi:
    def __init__(self, **kwargs):
        pass

    def model_info(self, *, repo_id, revision):
        log = Path(os.environ[\"FAKE_HF_LOG\"])
        with log.open(\"a\") as output:
            output.write(f\"resolve {repo_id} {revision}\\n\")
        return SimpleNamespace(sha=os.environ[\"FAKE_HF_RESOLVED\"])

def hf_hub_download(*, repo_id, filename, revision, cache_dir,
                    local_files_only=False, force_download=False, **kwargs):
    log = Path(os.environ[\"FAKE_HF_LOG\"])
    with log.open(\"a\") as output:
        output.write(
            f\"download {repo_id} {revision} {filename} \"
            f\"local={local_files_only} force={force_download}\\n\"
        )
    destination = (
        Path(cache_dir) / f\"models--{repo_id.replace('/', '--')}\"
        / \"snapshots\" / revision / filename
    )
    if destination.is_file() and not force_download:
        return str(destination)
    if local_files_only:
        raise FileNotFoundError(filename)
    source = Path(os.environ[\"FAKE_HF_REMOTE\"]) / repo_id / revision / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return str(destination)
""",
        encoding="utf-8",
    )
    return package.parent


def run_tool(
    tool: Path,
    python_path: Path,
    remote: Path,
    log: Path,
    resolved: str,
    arguments: list[str],
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(python_path),
            "FAKE_HF_REMOTE": str(remote),
            "FAKE_HF_LOG": str(log),
            "FAKE_HF_RESOLVED": resolved,
        }
    )
    return subprocess.run(
        [sys.executable, str(tool), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def require_success(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    if result.returncode != 0:
        raise AssertionError(f"fetch failed: stdout={result.stdout!r} stderr={result.stderr!r}")
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    help_result = subprocess.run(
        [sys.executable, str(args.tool), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if (
        help_result.returncode != 0
        or "biological-model checkpoints" not in help_result.stdout
    ):
        raise AssertionError("evo-fetch help has stale project scope")
    python_path = write_fake_hub(args.work_dir)
    remote = args.work_dir / "remote"
    cache = args.work_dir / "cache"
    log = args.work_dir / "hub.log"

    source_revision = "a" * 40
    source_payload = b"registered source checkpoint"
    source_path = remote / "owner" / "source" / source_revision / "source.pt"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_payload)
    source_registry = args.work_dir / "source-registry.json"
    source_registry.write_text(
        json.dumps(
            {
                "models": {
                    "tiny": {
                        "source_repo": "owner/source",
                        "source_revision": source_revision,
                        "checkpoint_files": [
                            {
                                "name": "source.pt",
                                "size": len(source_payload),
                                "sha256": digest(source_payload),
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    source_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    source_receipt = require_success(source_result)
    cached_source = Path(source_receipt["files"][0]["path"])
    if any(
        (
            source_receipt["kind"] != "source-checkpoint",
            source_receipt["resolved_revision"] != source_revision,
            source_receipt["files"][0]["sha256"] != digest(source_payload),
            cached_source.read_bytes() != source_payload,
            not Path(source_receipt["receipt"]).is_file(),
        )
    ):
        raise AssertionError("source receipt omitted immutable provenance")

    cached_source.write_bytes(b"x" * len(source_payload))
    repaired_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    require_success(repaired_result)
    if (
        cached_source.read_bytes() != source_payload
        or "failed verification; refreshing" not in repaired_result.stderr
        or "force=True" not in log.read_text(encoding="utf-8")
    ):
        raise AssertionError("corrupt cached checkpoint was not refreshed")

    log.write_text("", encoding="utf-8")
    local_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    require_success(local_result)
    if log.read_text(encoding="utf-8"):
        raise AssertionError("offline source lookup depended on Hugging Face metadata")
    cached_source.write_bytes(b"x" * len(source_payload))
    offline_tampered_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    if (
        offline_tampered_result.returncode == 0
        or "integrity mismatch" not in offline_tampered_result.stderr
        or log.read_text(encoding="utf-8")
    ):
        raise AssertionError("offline source snapshot did not fail closed")
    cached_source.write_bytes(source_payload)
    isolated_receipts = args.work_dir / "isolated-receipts"
    isolated_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "--receipt-dir",
            str(isolated_receipts),
            "--local-files-only",
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    isolated_receipt = require_success(isolated_result)
    if not Path(isolated_receipt["receipt"]).is_relative_to(isolated_receipts):
        raise AssertionError("custom receipt escaped its explicit directory")

    runtime_revision = "b" * 40
    runtime_root = remote / "owner" / "runtime" / runtime_revision
    runtime_root.mkdir(parents=True, exist_ok=True)
    index_payload = b'{"metadata":{},"weight_map":{}}\n'
    shard_payload = b"strict runtime shard"
    runtime_manifest = {
        "schema_version": 1,
        "artifact_profile": "esmc-runtime-v1",
        "model_id": "tiny_runtime",
        "load_path": "model.safetensors.index.json",
        "files": [
            {
                "path": "model.safetensors.index.json",
                "size": len(index_payload),
                "sha256": digest(index_payload),
            },
            {
                "path": "model-00001-of-00001.safetensors",
                "size": len(shard_payload),
                "sha256": digest(shard_payload),
            },
        ],
    }
    (runtime_root / "evo-artifact.json").write_text(
        json.dumps(runtime_manifest), encoding="utf-8"
    )
    (runtime_root / "model.safetensors.index.json").write_bytes(index_payload)
    (runtime_root / "model-00001-of-00001.safetensors").write_bytes(shard_payload)
    runtime_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        runtime_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            "owner/runtime@release-v1",
            "--registry",
            str(args.registry),
        ],
    )
    runtime_receipt = require_success(runtime_result)
    if any(
        (
            runtime_receipt["kind"] != "runtime-artifact",
            runtime_receipt["resolved_revision"] != runtime_revision,
            runtime_receipt["artifact_profile"] != "esmc-runtime-v1",
            Path(runtime_receipt["load_path"]).read_bytes() != index_payload,
            len(runtime_receipt["files"]) != 2,
        )
    ):
        raise AssertionError("runtime artifact was not resolved from its manifest")
    offline_runtime_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        runtime_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "runtime",
            f"owner/runtime@{runtime_revision}",
            "--registry",
            str(args.registry),
        ],
    )
    require_success(offline_runtime_result)
    cached_manifest = Path(runtime_receipt["manifest_path"])
    cached_manifest.write_bytes(b"{" + b" " * (cached_manifest.stat().st_size - 1))
    tampered_runtime_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        runtime_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "runtime",
            f"owner/runtime@{runtime_revision}",
            "--registry",
            str(args.registry),
        ],
    )
    if (
        tampered_runtime_result.returncode == 0
        or "manifest SHA256 differs" not in tampered_runtime_result.stderr
    ):
        raise AssertionError("offline runtime accepted a tampered cached manifest")

    invalid_revision = "c" * 40
    invalid_root = remote / "owner" / "invalid" / invalid_revision
    invalid_root.mkdir(parents=True, exist_ok=True)
    (invalid_root / "evo-artifact.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_profile": "evo2-runtime-v1",
                "model_id": "invalid",
                "load_path": "../escape.safetensors",
                "files": [
                    {"path": "../escape.safetensors", "size": 0, "sha256": "0" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )
    invalid_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        invalid_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            f"owner/invalid@{invalid_revision}",
            "--registry",
            str(args.registry),
        ],
    )
    if invalid_result.returncode == 0 or "normalized relative path" not in invalid_result.stderr:
        raise AssertionError("runtime manifest path traversal was not rejected")
    unknown_revision = "d" * 40
    unknown_root = remote / "owner" / "unknown" / unknown_revision
    unknown_root.mkdir(parents=True, exist_ok=True)
    unknown_manifest = copy.deepcopy(runtime_manifest)
    unknown_manifest["artifact_profile"] = "unknown-runtime-v1"
    (unknown_root / "evo-artifact.json").write_text(
        json.dumps(unknown_manifest), encoding="utf-8"
    )
    unknown_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        unknown_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            f"owner/unknown@{unknown_revision}",
            "--registry",
            str(args.registry),
        ],
    )
    if (
        unknown_result.returncode == 0
        or "unknown-runtime-v1' is not registered" not in unknown_result.stderr
    ):
        raise AssertionError("runtime fetch accepted an unregistered artifact profile")

    canonical_registry = json.loads(args.registry.read_text(encoding="utf-8"))
    extended_registry = copy.deepcopy(canonical_registry)
    extended_registry["artifact_profiles"].append(
        {
            "id": "fixture-runtime-v1",
            "metadata_key": "runtime.profile",
            "runtime_abi": "fixture-safetensors-v1",
        }
    )
    extended_registry["runtime_architectures"].append(
        {
            "id": "FixtureArchitecture",
            "artifact_profile": "fixture-runtime-v1",
            "runtime_abi": "fixture-safetensors-v1",
        }
    )
    extended_registry_path = args.work_dir / "extended-runtime-registry.json"
    extended_registry_path.write_text(
        json.dumps(extended_registry), encoding="utf-8"
    )
    custom_revision = "e" * 40
    custom_root = remote / "owner" / "custom" / custom_revision
    custom_root.mkdir(parents=True, exist_ok=True)
    custom_manifest = copy.deepcopy(runtime_manifest)
    custom_manifest["artifact_profile"] = "fixture-runtime-v1"
    (custom_root / "evo-artifact.json").write_text(
        json.dumps(custom_manifest), encoding="utf-8"
    )
    (custom_root / "model.safetensors.index.json").write_bytes(index_payload)
    (custom_root / "model-00001-of-00001.safetensors").write_bytes(shard_payload)
    custom_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        custom_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            f"owner/custom@{custom_revision}",
            "--registry",
            str(extended_registry_path),
        ],
    )
    custom_receipt = require_success(custom_result)
    if custom_receipt["artifact_profile"] != "fixture-runtime-v1":
        raise AssertionError("runtime fetch did not use its explicit profile registry")

    corrupt_registry = copy.deepcopy(canonical_registry)
    corrupt_registry["artifact_profiles"].append(
        copy.deepcopy(corrupt_registry["artifact_profiles"][0])
    )
    corrupt_registry_path = args.work_dir / "corrupt-runtime-registry.json"
    corrupt_registry_path.write_text(
        json.dumps(corrupt_registry), encoding="utf-8"
    )
    corrupt_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        custom_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            f"owner/custom@{custom_revision}",
            "--registry",
            str(corrupt_registry_path),
        ],
    )
    if (
        corrupt_result.returncode == 0
        or "duplicate artifact profile" not in corrupt_result.stderr
    ):
        raise AssertionError("runtime fetch accepted a corrupt profile registry")

    revision_traversal = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        runtime_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "runtime",
            "owner/runtime@../../escape",
            "--registry",
            str(args.registry),
        ],
    )
    if revision_traversal.returncode == 0 or "path traversal" not in revision_traversal.stderr:
        raise AssertionError("revision path traversal was not rejected locally")

    print("Hugging Face fetch contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
