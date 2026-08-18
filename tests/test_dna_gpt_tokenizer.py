#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline DNAGPT tokenizer generator/compiler/C++ closure."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


CASES = {
    "geneb-dna-gpt-0-1b-h": (
        ("<R>ACGTNAC", "21,5385,0"),
        ("<R>AXA", "21,0"),
        ("<R>AAAAAX", "21,0"),
        ("<R><M>", "21,18"),
    ),
    "geneb-dna-gpt-3b-m": (
        ("<R>ACGTNAC", "21,9290,37"),
        ("<R>AXA", "21,0"),
        ("<R>AAAAAX", "21,0"),
        ("<R><M>", "21,18"),
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: Any) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require_success(result: subprocess.CompletedProcess, label: str) -> None:
    if result.returncode != 0:
        raise AssertionError("%s failed:\n%s\n%s" % (label, result.stdout, result.stderr))


def compile_case(
    args: argparse.Namespace, model_id: str, root: Path
) -> Tuple[Path, Path, Path]:
    generated = run(
        [
            str(args.python),
            str(args.generator),
            "--model-id",
            model_id,
            "--output-dir",
            str(root),
        ]
    )
    require_success(generated, model_id + " tokenizer source generation")
    manifest = root / (model_id + ".tokenizer-manifest.json")
    receipt = root / (model_id + ".tokenizer-receipt.json")
    output = root / "tokenizer.json"
    descriptor = root / "tokenizer.descriptor.json"
    compiled = run(
        [
            str(args.python),
            str(args.compiler),
            "--manifest",
            str(manifest),
            "--receipt",
            str(receipt),
            "--output",
            str(output),
            "--descriptor",
            str(descriptor),
        ]
    )
    require_success(compiled, model_id + " tokenizer compilation")
    return manifest, output, descriptor


def profile_by_id(path: Path) -> Dict[str, Mapping[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {item["runtime_id"]: item for item in document["models"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    profiles = profile_by_id(args.profiles)
    if set(profiles) != set(CASES):
        raise AssertionError("DNA-GPT production tokenizer profiles differ")

    outputs = {}  # type: Dict[str, bytes]
    for model_id, vectors in CASES.items():
        root = args.work_dir / model_id
        manifest, output, descriptor = compile_case(args, model_id, root)
        payload = output.read_bytes()
        outputs[model_id] = payload
        expected = profiles[model_id]["tokenizer"]
        if (
            len(payload) != expected["compiled_asset_size"]
            or digest(output) != expected["compiled_asset_sha256"]
            or digest(manifest) != expected["compiler_manifest_sha256"]
        ):
            raise AssertionError(model_id + " compiled tokenizer identity differs")
        descriptor_value = json.loads(descriptor.read_text(encoding="utf-8"))
        if set(descriptor_value) != {
            "converter.schema",
            "converter.version",
            "compiler_manifest_sha256",
            "source_receipt_contract_sha256",
            "tokenizer.profile",
            "tokenizer.path",
            "tokenizer.sha256",
            "tokenizer.size",
        }:
            raise AssertionError(model_id + " tokenizer descriptor fields differ")
        for input_text, expected_ids in vectors:
            verified = run(
                [
                    str(args.runtime),
                    "--verify-asset-no-special",
                    str(root),
                    "tokenizer.json",
                    digest(output),
                    str(len(payload)),
                    input_text,
                    expected_ids,
                ]
            )
            require_success(verified, model_id + " C++ tokenizer closure")

        refused = run(
            [
                str(args.python),
                str(args.generator),
                "--model-id",
                model_id,
                "--output-dir",
                str(root),
            ]
        )
        if refused.returncode == 0 or output.read_bytes() != payload:
            raise AssertionError("generator overwrote an existing source kit")

        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        source_name = manifest_value["files"][0]["name"]
        with (root / source_name).open("ab") as source:
            source.write(b"X")
        corrupt_output = root / "corrupt-tokenizer.json"
        corrupt_descriptor = root / "corrupt-descriptor.json"
        rejected = run(
            [
                str(args.python),
                str(args.compiler),
                "--manifest",
                str(manifest),
                "--receipt",
                str(root / (model_id + ".tokenizer-receipt.json")),
                "--output",
                str(corrupt_output),
                "--descriptor",
                str(corrupt_descriptor),
            ]
        )
        if rejected.returncode == 0 or "hash/size mismatch" not in rejected.stderr:
            raise AssertionError("corrupted generated tokenizer source was accepted")
        if corrupt_output.exists() or corrupt_descriptor.exists():
            raise AssertionError("failed tokenizer compile published partial outputs")

    if outputs["geneb-dna-gpt-0-1b-h"] == outputs["geneb-dna-gpt-3b-m"]:
        raise AssertionError("static and dynamic DNAGPT tokenizer assets collapsed")
    print("DNA-GPT offline tokenizer generator/compiler/C++ closure passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
