#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lock the registered GENEB backend and public-surface failure matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from evo.format import BytesTensorSource, write_model


UNSUPPORTED_EXIT = 5


def run_rejected(command: List[str], expected: str, label: str) -> None:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != UNSUPPORTED_EXIT or expected not in result.stderr:
        raise AssertionError(
            "{} did not fail with typed unsupported:\ncommand={}\nstdout={}\nstderr={}".format(
                label, command, result.stdout, result.stderr
            )
        )


def write_factory_probe(
    path: Path, descriptor: Dict[str, Any], registry_path: Path
) -> None:
    write_model(
        path,
        {
            "model.id": "t37-{}".format(descriptor["id"]),
            "model.architecture": descriptor["id"],
            "runtime.abi": descriptor["runtime_abi"],
        },
        [
            BytesTensorSource(
                "t37.factory_probe", "F32", (1,), struct.pack("<f", 0.0)
            )
        ],
        artifact_profile=descriptor["artifact_profile"],
        registry_path=registry_path,
        force=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--c-api", required=True, type=Path)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    descriptors = [
        entry
        for entry in registry["runtime_architectures"]
        if entry["artifact_profile"].startswith("geneb-")
    ]
    if not descriptors:
        raise AssertionError("model registry contains no production GENEB architecture")

    sequence = args.work_dir / "sequence.fasta"
    sequence.write_text(">t37\nACGT\n", encoding="ascii")
    public_surfaces = ("embed", "serve")
    backend_names = ("cuda", "mps")
    for descriptor in descriptors:
        architecture = descriptor["id"]
        if "cpu" not in descriptor["backends"]:
            raise AssertionError("{} omitted its portable CPU factory".format(architecture))
        if descriptor["capabilities"] != list(public_surfaces):
            raise AssertionError(
                "{} public surfaces differ from embed+serve".format(architecture)
            )
        artifact = args.work_dir / (architecture + ".safetensors")
        write_factory_probe(artifact, descriptor, args.registry)

        for backend in backend_names:
            if backend in descriptor["backends"]:
                continue
            c_api = subprocess.run(
                [
                    str(args.c_api),
                    "--geneb-unsupported",
                    backend,
                    str(artifact),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if c_api.returncode != 0:
                raise AssertionError(
                    "{} C ABI {} rejection drifted:\n{}{}".format(
                        architecture, backend, c_api.stdout, c_api.stderr
                    )
                )

            expected = "unsupported: backend {} is not supported by architecture {}".format(
                backend, architecture
            )
            output = args.work_dir / (architecture + "-" + backend + "-embed")
            if output.exists():
                shutil.rmtree(output)
            embed = [
                str(args.cli),
                "embed",
                "-m",
                str(artifact),
                "--input",
                str(sequence),
                "--output",
                str(output),
                "--preset",
                "geneb-v4-normalized",
                "--backend",
                backend,
            ]
            if backend == "cuda":
                embed.extend(("--gpu", "0"))
            run_rejected(embed, expected, "{} {} CLI embed".format(architecture, backend))
            if output.exists():
                raise AssertionError(
                    "{} {} CLI rejection created embedding output".format(
                        architecture, backend
                    )
                )

            serve = [
                str(args.cli),
                "serve",
                "-m",
                str(artifact),
                "--backend",
                backend,
            ]
            if backend == "cuda":
                serve.extend(("--gpu", "0"))
            run_rejected(serve, expected, "{} {} CLI serve".format(architecture, backend))

    print(
        "GENEB surface contract passed for {} registered architectures".format(
            len(descriptors)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
