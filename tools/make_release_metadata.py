#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a canonical checksum and provenance sidecar for a release archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
VERSION_RE = re.compile(
    r"project\s*\(\s*evo_cpp\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", re.DOTALL
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def project_version(source_dir: Path) -> str:
    cmake = (source_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    match = VERSION_RE.search(cmake)
    if match is None:
        raise ValueError("cannot find evo_cpp semantic version in CMakeLists.txt")
    return match.group(1)


def runtime_contract(registry_path: Path) -> tuple[list[str], list[str]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry.get("runtime_architectures")
    if not isinstance(entries, list) or not entries:
        raise ValueError("model registry has no runtime_architectures")
    architectures: list[str] = []
    profiles: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("runtime architecture descriptor must be an object")
        architecture = entry.get("id")
        profile = entry.get("artifact_profile")
        runtime_abi = entry.get("runtime_abi")
        if not all(
            isinstance(value, str) and value
            for value in (architecture, profile, runtime_abi)
        ):
            raise ValueError("runtime architecture descriptor is incomplete")
        if architecture in architectures:
            raise ValueError(f"duplicate runtime architecture {architecture!r}")
        architectures.append(architecture)
        if profile not in profiles:
            profiles.append(profile)
    return architectures, profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--cuda-version")
    parser.add_argument("--build-image", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        archive = args.archive.resolve(strict=True)
        source_dir = args.source_dir.resolve(strict=True)
        if not archive.is_file():
            raise ValueError("release archive must be a regular file")
        if not COMMIT_RE.fullmatch(args.commit):
            raise ValueError("release commit must contain 40 hexadecimal characters")
        version = project_version(source_dir)
        if args.tag != f"v{version}":
            raise ValueError(
                f"release tag {args.tag!r} does not match project version v{version}"
            )
        if not args.platform or any(character.isspace() for character in args.platform):
            raise ValueError("release platform must be a nonempty token")
        if not args.backend or any(character.isspace() for character in args.backend):
            raise ValueError("release backend must be a nonempty token")
        registry = source_dir / "configs" / "model-registry.json"
        if not registry.is_file():
            raise ValueError("source tree does not contain the model registry")
        registered_architectures, runtime_profiles = runtime_contract(registry)
        metadata = {
            "schema_version": 2,
            "artifact_kind": "runtime-binary",
            "artifact": archive.name,
            "size": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "project": "evo.cpp",
            "version": version,
            "tag": args.tag,
            "commit": args.commit.lower(),
            "platform": args.platform,
            "backend": args.backend,
            "cuda_version": args.cuda_version,
            "build_image": args.build_image,
            "registered_architectures": registered_architectures,
            "runtime_profiles": runtime_profiles,
            "execution_profiles": ["exact", "fast-q8-kv", "cpu-f32"],
            "model_registry": {
                "installed_path": "share/evo/configs/model-registry.json",
                "sha256": sha256_file(registry),
            },
            "contains_model_weights": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{metadata['sha256']}  {archive.name}")
        return 0
    except (OSError, ValueError) as error:
        print(f"make_release_metadata: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
