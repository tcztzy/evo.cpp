#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check release metadata and workflow supply-chain contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    archive = args.work_dir / "evo-0.1.0-linux-x86_64-cuda12.8.tar.gz"
    payload = b"deterministic release payload"
    archive.write_bytes(payload)
    output = args.work_dir / f"{archive.name}.metadata.json"
    tool = args.source_dir / "tools" / "make_release_metadata.py"
    command = [
        sys.executable,
        str(tool),
        "--archive",
        str(archive),
        "--source-dir",
        str(args.source_dir),
        "--platform",
        "linux-x86_64",
        "--backend",
        "cuda-exact",
        "--cuda-version",
        "12.8",
        "--build-image",
        "nvidia/cuda@sha256:" + "b" * 64,
        "--commit",
        "a" * 40,
        "--tag",
        "v0.1.0",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise AssertionError(f"metadata tool failed: {result.stderr}")
    metadata = json.loads(output.read_text(encoding="utf-8"))
    registry = json.loads(
        (args.source_dir / "configs" / "model-registry.json").read_text(
            encoding="utf-8"
        )
    )
    registered_architectures = [
        entry["id"] for entry in registry["runtime_architectures"]
    ]
    runtime_profiles = list(
        dict.fromkeys(
            entry["artifact_profile"]
            for entry in registry["runtime_architectures"]
        )
    )
    expected_digest = hashlib.sha256(payload).hexdigest()
    if any(
        (
            metadata["schema_version"] != 2,
            metadata["artifact_kind"] != "runtime-binary",
            metadata["sha256"] != expected_digest,
            metadata["size"] != len(payload),
            metadata["version"] != "0.1.0",
            metadata["backend"] != "cuda-exact",
            metadata["cuda_version"] != "12.8",
            metadata["build_image"] != "nvidia/cuda@sha256:" + "b" * 64,
            metadata["registered_architectures"]
            != registered_architectures,
            metadata["runtime_profiles"]
            != runtime_profiles,
            metadata["execution_profiles"]
            != ["exact", "fast-q8-kv", "cpu-f32"],
            metadata["model_registry"]["installed_path"]
            != "share/evo/configs/model-registry.json",
            metadata["model_registry"]["sha256"]
            != hashlib.sha256(
                (args.source_dir / "configs" / "model-registry.json").read_bytes()
            ).hexdigest(),
            metadata["contains_model_weights"] is not False,
            result.stdout.strip() != f"{expected_digest}  {archive.name}",
        )
    ):
        raise AssertionError("release metadata is incomplete or noncanonical")
    mismatch_command = command.copy()
    mismatch_command[mismatch_command.index("--tag") + 1] = "v9.9.9"
    mismatch = subprocess.run(
        mismatch_command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if mismatch.returncode == 0 or "does not match project version" not in mismatch.stderr:
        raise AssertionError("release metadata accepted a tag/version mismatch")

    ci = (args.source_dir / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    release = (args.source_dir / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    for required in (
        "EVO_CUDA=OFF",
        "EVO_SANITIZE=ON",
        "EVO_WARNINGS_AS_ERRORS=ON",
        "ctest --test-dir",
    ):
        if required not in ci:
            raise AssertionError(f"CI workflow omitted {required}")
    for required in (
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        "sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7",
        "cmake==4.4.0",
        "conan==2.30.0",
        "actions/attest@v4",
        "subject-checksums",
        "SHA256SUMS",
        "softprops/action-gh-release@v3",
        "fail_on_unmatched_files: true",
    ):
        if required not in release:
            raise AssertionError(f"release workflow omitted {required}")
    print("release pipeline contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
