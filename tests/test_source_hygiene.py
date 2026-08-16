#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check source text hygiene without assuming an unpinned formatter style."""

import argparse
import re
from pathlib import Path


TEXT_SUFFIXES = {
    ".cmake",
    ".cpp",
    ".cu",
    ".def",
    ".hpp",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yml",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    for path in sorted(args.source_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in TEXT_SUFFIXES
            or any(part == ".git" or part.startswith("build") for part in path.parts)
        ):
            continue
        payload = path.read_bytes()
        relative = path.relative_to(args.source_dir)
        if payload and not payload.endswith(b"\n"):
            failures.append(f"{relative}: missing final newline")
        for line_number, line in enumerate(payload.splitlines(), start=1):
            if line.endswith((b" ", b"\t")):
                failures.append(f"{relative}:{line_number}: trailing whitespace")
            if path.suffix == ".sh" and line.lstrip().startswith(b"status="):
                failures.append(
                    f"{relative}:{line_number}: use a task-specific exit "
                    "variable instead of shell-special 'status'"
                )
            if (
                path.suffix == ".sh"
                and b'awk "' in line
                and b"$" in line.split(b'awk "', maxsplit=1)[1]
            ):
                failures.append(
                    f"{relative}:{line_number}: single-quote awk programs so "
                    "the shell cannot expand their field variables"
                )
    sequence_header = args.source_dir / "include" / "evo" / "sequence_io.hpp"
    if "read_sequence_file" in sequence_header.read_text(encoding="utf-8"):
        failures.append(
            "include/evo/sequence_io.hpp: public sequence input must expose "
            "only the bounded callback streaming API"
        )
    for relative in (
        Path("src/c_api.cpp"),
        Path("src/cpu/inference_cli.cpp"),
        Path("src/main.cpp"),
    ):
        surface = (args.source_dir / relative).read_text(encoding="utf-8")
        if re.search(r"\bbool\s+esmc\b", surface):
            failures.append(
                f"{relative}: architecture dispatch must not use a bool esmc tag"
            )
        if re.search(r"(?:->|\.)tokenizer\s*==", surface):
            failures.append(
                f"{relative}: tokenizer identity must not select an architecture"
            )
        if "find_architecture_backend_factory" not in surface:
            failures.append(
                f"{relative}: backend dispatch must use the registered factory"
            )
    if failures:
        raise AssertionError("\n".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
