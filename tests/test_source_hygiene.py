#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check source text hygiene without assuming an unpinned formatter style."""

import argparse
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
    if failures:
        raise AssertionError("\n".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
