#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract and corruption gates for the frozen GENEB v4 catalog."""

import argparse
import ast
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


Mutation = Tuple[str, Callable[[Dict[str, Any]], None]]


def run_validator(
    validator: Path,
    catalog: Path,
    source_dir: Path,
    expect_success: bool,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--catalog",
            str(catalog),
            "--source-dir",
            str(source_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            "GENEB validator rejected canonical catalog:\n{}{}".format(
                result.stdout, result.stderr
            )
        )
    if not expect_success:
        if result.returncode == 0:
            raise AssertionError("GENEB validator accepted a corrupted catalog")
        if "GENEB catalog validation failed:" not in result.stderr:
            raise AssertionError("validator failure omitted typed diagnostic")
    return result


def write_catalog(path: Path, catalog: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def first_huggingface(catalog: Dict[str, Any]) -> Dict[str, Any]:
    return next(
        model
        for model in catalog["models"]
        if model["source"]["kind"] == "huggingface"
    )


def enformer(catalog: Dict[str, Any]) -> Dict[str, Any]:
    return next(
        model for model in catalog["models"] if model["paper_name"] == "Enformer"
    )


def mutate_duplicate_model(catalog: Dict[str, Any]) -> None:
    catalog["models"][-1] = copy.deepcopy(catalog["models"][0])


def mutate_missing_model(catalog: Dict[str, Any]) -> None:
    catalog["models"].pop()


def mutate_paper_name(catalog: Dict[str, Any]) -> None:
    catalog["models"][0]["paper_name"] += "-typo"


def mutate_model_id(catalog: Dict[str, Any]) -> None:
    catalog["models"][0]["geneb_model_id"] = "not-in-model-meta"


def mutate_params(catalog: Dict[str, Any]) -> None:
    catalog["models"][0]["params"] += 1


def mutate_batching(catalog: Dict[str, Any]) -> None:
    catalog["suite"]["reference_batching"]["batch_size"] = 4


def mutate_alias_collision(catalog: Dict[str, Any]) -> None:
    runtime_id = catalog["models"][0]["runtime_id"]
    catalog["aliases"]["models"][runtime_id] = runtime_id


def mutate_false_support(catalog: Dict[str, Any]) -> None:
    support = catalog["models"][0]["runtime_support"]
    support["status"] = "supported"
    support["artifact_profile"] = "unverified-runtime-v1"
    support["reason"] = None


def mutate_normalization_hash(catalog: Dict[str, Any]) -> None:
    catalog["models"][0]["provenance"]["normalization_patch_sha256"] = "0" * 64


def mutate_reference_patch_hash(catalog: Dict[str, Any]) -> None:
    enformer(catalog)["provenance"]["reference_patch"]["sha256"] = "0" * 64


def mutate_weight_redistribution(catalog: Dict[str, Any]) -> None:
    catalog["models"][0]["licenses"]["redistribution"]["weights"] = True


def mutate_hf_revision(catalog: Dict[str, Any]) -> None:
    source = first_huggingface(catalog)["source"]
    first = "0" if source["revision"][0] != "0" else "1"
    source["revision"] = first + source["revision"][1:]


def mutate_false_reference_eligibility(catalog: Dict[str, Any]) -> None:
    catalog["models"][0]["benchmark_provenance"]["reference_status"] = "eligible"


def mutate_submission_hash(catalog: Dict[str, Any]) -> None:
    catalog["models"][0]["benchmark_provenance"][
        "official_submission_sha256"
    ] = "0" * 64


def mutate_nested_transform(catalog: Dict[str, Any]) -> None:
    model = next(
        item
        for item in catalog["models"]
        if item["input_transform"]["frame_trim"] is not None
    )
    model["input_transform"]["frame_trim"]["unregistered"] = True


def mutate_required_file_path(catalog: Dict[str, Any]) -> None:
    model = next(
        item for item in catalog["models"] if item["source"]["required_files"]
    )
    model["source"]["required_files"][0]["path"] = "../checkpoint.bin"


def check_python38_syntax(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path), feature_version=8)
    annotations = []  # type: List[ast.AST]
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
    for annotation in annotations:
        for node in ast.walk(annotation):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"list", "dict", "set", "tuple"}
            ):
                raise AssertionError(
                    "{} uses a PEP 585 annotation unavailable on Python 3.8"
                    .format(path.name)
                )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                raise AssertionError(
                    "{} uses a PEP 604 annotation unavailable on Python 3.8"
                    .format(path.name)
                )


def check_documentation(source_dir: Path) -> None:
    document = (source_dir / "docs" / "geneb.md").read_text(encoding="utf-8")
    for required in (
        "40 evaluated checkpoints",
        "catalog row is not, by itself, a supported runtime",
        "geneb-v4-reference",
        "geneb-v4-normalized",
        "normalization_patch_sha256",
        "ensure_ascii=False",
        "manual-source",
        "Python 3.8",
    ):
        if required not in document:
            raise AssertionError("docs/geneb.md omitted: " + required)
    for readme_name in ("README.md", "README.zh_CN.md"):
        readme = (source_dir / readme_name).read_text(encoding="utf-8")
        if "docs/geneb.md" not in readme:
            raise AssertionError(readme_name + " omitted the GENEB contract link")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve(strict=True)
    validator = args.validator.resolve(strict=True)
    catalog_path = args.catalog.resolve(strict=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    canonical = run_validator(validator, catalog_path, source_dir, True)
    summary = json.loads(canonical.stdout)
    if summary["valid"] is not True or summary["model_count"] != 40:
        raise AssertionError("canonical summary omitted the 40-model valid gate")
    if len(summary["models"]) != 40:
        raise AssertionError("canonical summary must contain 40 evidence records")
    if summary["runtime_support_counts"].get("supported", 0) != 0:
        raise AssertionError("catalog-only milestone must not claim supported runtimes")
    if summary["reference_provenance_counts"].get("eligible", 0) != 0:
        raise AssertionError("catalog-only milestone must not claim reference eligibility")
    for model in summary["models"]:
        for field in (
            "extractor_commit",
            "reference_patch_sha256",
            "normalization_patch_sha256",
            "oracle_env",
            "oracle_input_digest",
            "official_submission_path",
            "official_submission_sha256",
        ):
            if field not in model:
                raise AssertionError("summary evidence omitted " + field)
    repeated = run_validator(validator, catalog_path, source_dir, True)
    if repeated.stdout != canonical.stdout:
        raise AssertionError("validator summary is not deterministic")

    mutations = [
        ("duplicate-model", mutate_duplicate_model),
        ("missing-model", mutate_missing_model),
        ("paper-name", mutate_paper_name),
        ("model-id", mutate_model_id),
        ("params", mutate_params),
        ("batching", mutate_batching),
        ("alias-collision", mutate_alias_collision),
        ("false-support", mutate_false_support),
        ("normalization-hash", mutate_normalization_hash),
        ("reference-patch-hash", mutate_reference_patch_hash),
        ("weight-redistribution", mutate_weight_redistribution),
        ("hf-revision", mutate_hf_revision),
        ("false-reference-eligibility", mutate_false_reference_eligibility),
        ("submission-hash", mutate_submission_hash),
        ("nested-transform", mutate_nested_transform),
        ("required-file-path", mutate_required_file_path),
    ]  # type: List[Mutation]
    for label, mutate in mutations:
        candidate = copy.deepcopy(catalog)
        mutate(candidate)
        mutated_path = args.work_dir / (label + ".json")
        write_catalog(mutated_path, candidate)
        run_validator(validator, mutated_path, source_dir, False)

    duplicate_key_path = args.work_dir / "duplicate-key.json"
    duplicate_key_path.write_text(
        '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
    )
    run_validator(validator, duplicate_key_path, source_dir, False)

    check_python38_syntax(validator)
    check_python38_syntax(Path(__file__))
    check_documentation(source_dir)
    print("GENEB catalog contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
