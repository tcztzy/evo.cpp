#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Keep platform governance, compatibility, and release claims synchronized."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ABI_RE = re.compile(r"#define EVO_ABI_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)u")
PROJECT_RE = re.compile(
    r"project\s*\(\s*evo_cpp\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", re.DOTALL
)


def require(text: str, values: tuple[str, ...], label: str) -> None:
    for value in values:
        if value not in text:
            raise AssertionError(f"{label} omitted required contract: {value}")


def check_links(source_dir: Path, paths: list[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith(("#", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                relative = path.relative_to(source_dir)
                raise AssertionError(f"{relative}: broken local link {raw_target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    args = parser.parse_args()
    source_dir = args.source_dir.resolve(strict=True)

    required_paths = [
        source_dir / "CONTRIBUTING.md",
        source_dir / "README.md",
        source_dir / "README.zh_CN.md",
        source_dir / "docs" / "compatibility.md",
        source_dir / "docs" / "benchmark-matrix.md",
        source_dir / "docs" / "artifact-distribution.md",
    ]
    for path in required_paths:
        if not path.is_file():
            raise AssertionError(f"missing required project document: {path.name}")

    contributing = required_paths[0].read_text(encoding="utf-8")
    compatibility = required_paths[3].read_text(encoding="utf-8")
    benchmarks = required_paths[4].read_text(encoding="utf-8")
    releases = required_paths[5].read_text(encoding="utf-8")
    require(
        contributing,
        (
            "-DEVO_WARNINGS_AS_ERRORS=ON",
            "EVO_SANITIZE=ON",
            "Adding a model family",
            "real official checkpoint",
            "Large checkpoints",
        ),
        "CONTRIBUTING.md",
    )
    require(
        compatibility,
        (
            "C ABI",
            "Legacy CLI",
            "JSONL",
            "Artifact profiles",
            "Execution profiles",
            "linux-x86_64-cuda12.8",
            "Windows, HIP, Metal, Vulkan, SYCL",
        ),
        "compatibility policy",
    )
    require(
        benchmarks,
        (
            "StripedHyena2",
            "HyenaDNA",
            "`exact`",
            "`fast-q8-kv`",
            "`cpu-f32`",
            "hybrid",
            "Synthetic",
            "Real checkpoint",
            "source commit",
            "SHA256",
        ),
        "benchmark matrix",
    )
    require(
        releases,
        (
            "Release support matrix",
            "SHA256SUMS",
            "schema_version",
            "registered_architectures",
            "runtime_profiles",
            "contains_model_weights",
            "Release checklist",
        ),
        "release documentation",
    )

    header = (source_dir / "include" / "evo" / "evo.h").read_text(
        encoding="utf-8"
    )
    abi_parts = {name: value for name, value in ABI_RE.findall(header)}
    if set(abi_parts) != {"MAJOR", "MINOR", "PATCH"}:
        raise AssertionError("cannot parse C ABI version macros")
    abi = ".".join(abi_parts[name] for name in ("MAJOR", "MINOR", "PATCH"))
    if f"Current C ABI: **{abi}**" not in compatibility:
        raise AssertionError("compatibility policy C ABI version is stale")

    cmake = (source_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    project_match = PROJECT_RE.search(cmake)
    if project_match is None:
        raise AssertionError("cannot parse CMake project version")
    if f"Current project version: **{project_match.group(1)}**" not in compatibility:
        raise AssertionError("compatibility policy project version is stale")

    markdown = [source_dir / "CONTRIBUTING.md", source_dir / "README.md"]
    markdown.extend(sorted((source_dir / "docs").glob("*.md")))
    check_links(source_dir, markdown)
    print("documentation contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
