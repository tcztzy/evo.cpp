#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Keep platform governance, compatibility, and release claims synchronized."""

from __future__ import annotations

import argparse
import json
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
        source_dir / "docs" / "architectures.md",
        source_dir / "docs" / "checkpoint-conversion.md",
        source_dir / "docs" / "model-format.md",
        source_dir / "docs" / "server.md",
        source_dir / "NOTICE",
        source_dir / ".evo2-7b-hf-upload" / "README.md",
    ]
    for path in required_paths:
        if not path.is_file():
            raise AssertionError(f"missing required project document: {path.name}")

    contributing = required_paths[0].read_text(encoding="utf-8")
    compatibility = required_paths[3].read_text(encoding="utf-8")
    benchmarks = required_paths[4].read_text(encoding="utf-8")
    releases = required_paths[5].read_text(encoding="utf-8")
    architectures_document = required_paths[6].read_text(encoding="utf-8")
    conversion = required_paths[7].read_text(encoding="utf-8")
    model_format = required_paths[8].read_text(encoding="utf-8")
    server = required_paths[9].read_text(encoding="utf-8")
    notice = required_paths[10].read_text(encoding="utf-8")
    evo2_model_card = required_paths[11].read_text(encoding="utf-8")
    exactness = (source_dir / "docs" / "model-size-validation.md").read_text(
        encoding="utf-8"
    )
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
            "macOS arm64, C++17 CPU + MPS",
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
            "`mps-f32`",
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
            "architecture_matrix",
            "runtime_profiles",
            "contains_model_weights",
            "Release checklist",
        ),
        "release documentation",
    )
    registry = json.loads(
        (source_dir / "configs" / "model-registry.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_architectures = registry.get("runtime_architectures")
    if not isinstance(runtime_architectures, list) or not runtime_architectures:
        raise AssertionError("model registry omitted runtime_architectures")
    architecture_ids = tuple(entry["id"] for entry in runtime_architectures)
    artifact_profiles = tuple(
        dict.fromkeys(entry["artifact_profile"] for entry in runtime_architectures)
    )
    readme = required_paths[1].read_text(encoding="utf-8")
    readme_zh = required_paths[2].read_text(encoding="utf-8")
    for label, text in (
        ("README.md", readme),
        ("README.zh_CN.md", readme_zh),
        ("architecture registry", architectures_document),
        ("release documentation", releases),
    ):
        require(text, architecture_ids, label)
    architecture_rows = {}
    for line in architectures_document.splitlines():
        if not line.startswith("| `"):
            continue
        columns = [column.strip() for column in line.split("|")]
        if len(columns) == 7:
            architecture_rows[columns[1].strip("`")] = columns
    for entry in runtime_architectures:
        architecture_id = entry["id"]
        columns = architecture_rows.get(architecture_id)
        if columns is None:
            raise AssertionError(
                "architecture registry omitted matrix row for " + architecture_id
            )
        documented_backends = columns[4].lower()
        documented_surfaces = columns[5].lower()
        for backend in entry["backends"]:
            if backend not in documented_backends:
                raise AssertionError(
                    "architecture registry omitted {} backend for {}".format(
                        backend, architecture_id
                    )
                )
        for capability in entry["capabilities"]:
            if capability not in documented_surfaces:
                raise AssertionError(
                    "architecture registry omitted {} surface for {}".format(
                        capability, architecture_id
                    )
                )
    for label, text in (
        ("release documentation", releases),
        ("checkpoint conversion", conversion),
        ("model format", model_format),
    ):
        require(text, artifact_profiles, label)
    require(server, architecture_ids, "server capability boundary")
    require(notice, ("Evo 2", "HyenaDNA", "ESMC"), "NOTICE")
    require(
        evo2_model_card,
        ("biological-sequence inference runtime", *architecture_ids),
        "Evo 2 artifact model card",
    )
    stale_scope_claims = (
        "inference runtime for Evo 2.",
        "生产 runtime 只读取 `evo2-runtime-v1`",
        "loads one validated Evo 2 artifact",
        "inspect an Evo 2 runtime Safetensors file",
        "Fetch revision-pinned Evo source checkpoints",
    )
    scope_sources = {
        "README.md": readme,
        "README.zh_CN.md": readme_zh,
        "checkpoint conversion": conversion,
        "server": server,
        "Evo 2 artifact model card": evo2_model_card,
        "model inspector": (source_dir / "src" / "inspect_main.cpp").read_text(
            encoding="utf-8"
        ),
        "fetch helper": (source_dir / "tools" / "evo_fetch.py").read_text(
            encoding="utf-8"
        ),
    }
    for label, text in scope_sources.items():
        for stale in stale_scope_claims:
            if stale in text:
                raise AssertionError(f"{label} contains stale scope claim: {stale}")
    for model_id, model in registry["models"].items():
        if model_id not in exactness or model["exact_support"] not in exactness:
            raise AssertionError(
                f"model-size validation omitted exact support for {model_id}"
            )
        evidence = model["exact_evidence"]
        if evidence is not None and evidence not in exactness:
            raise AssertionError(
                f"model-size validation omitted evidence ID for {model_id}"
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
