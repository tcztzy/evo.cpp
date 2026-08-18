#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract and corruption tests for the locked GENEB probe runner."""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def expect_error(callable_value: Any, text: str) -> None:
    try:
        callable_value()
    except Exception as error:
        if text not in str(error):
            raise AssertionError(
                "failure omitted {!r}: {}".format(text, error)
            )
        return
    raise AssertionError("corrupted GENEB runner input was accepted")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path) -> str:
    payload = b"text,label\nACGT,0\nTGCA,1\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return digest(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--benchmark-spec", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    runner_path = args.runner.resolve(strict=True)
    syntax = runner_path.read_text(encoding="utf-8")
    ast.parse(syntax, filename=str(runner_path), feature_version=8)
    module_spec = importlib.util.spec_from_file_location("evo_run_geneb", runner_path)
    if module_spec is None or module_spec.loader is None:
        raise AssertionError("cannot import GENEB runner")
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)

    lock, lock_payload = runner.load_json(args.lock.resolve(), "probe lock")
    tasks, lock_digest = runner.validate_lock(
        lock, lock_payload, args.benchmark_spec.resolve()
    )
    if len(tasks) != 100 or len(set(tasks)) != 100 or len(lock_digest) != 64:
        raise AssertionError("locked benchmark did not expose 100 unique tasks")

    bad_lock = copy.deepcopy(lock)
    bad_lock["protocol"]["logistic_regression"]["solver"] = "liblinear"
    expect_error(
        lambda: runner.validate_lock(
            bad_lock, json.dumps(bad_lock).encode("utf-8"), args.benchmark_spec
        ),
        "LogisticRegression kwargs differ",
    )

    catalog, catalog_payload = runner.load_json(args.catalog.resolve(), "catalog")
    candidate = copy.deepcopy(catalog["models"][0])
    candidate["runtime_support"]["status"] = "supported"
    candidate["runtime_support"]["artifact_profile"] = "test-runtime-v1"
    candidate["runtime_support"]["reason"] = None
    candidate["embedding_presets"]["normalized"]["output_width"] = 2
    fake_catalog = copy.deepcopy(catalog)
    fake_catalog["models"] = [candidate]
    model, preset = runner.select_catalog_model(
        fake_catalog, candidate["runtime_id"], "geneb"
    )
    if preset != "geneb-v4-normalized":
        raise AssertionError("geneb alias did not resolve to normalized")
    expect_error(
        lambda: runner.select_catalog_model(
            fake_catalog, candidate["runtime_id"], "geneb-v4-reference"
        ),
        "not reference-eligible",
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.work_dir / "dataset"
    embedding_root = args.work_dir / "embeddings"
    embedding_root.mkdir(parents=True, exist_ok=True)
    embedding_payload = b"synthetic receipted NPY payload"
    embedding_path = embedding_root / "all.npy"
    embedding_path.write_bytes(embedding_payload)
    embedding_digest = digest(embedding_payload)
    manifest_tasks = {}  # type: Dict[str, Any]
    for task_id in tasks:
        splits = {}  # type: Dict[str, Any]
        for split in ("train", "test"):
            input_digest = write_csv(
                data_dir / "tasks" / task_id / (split + ".csv")
            )
            splits[split] = {
                "path": "all.npy",
                "size": len(embedding_payload),
                "sha256": embedding_digest,
                "rows": 2,
                "columns": 2,
                "input_sha256": input_digest,
            }
        manifest_tasks[task_id] = splits
    fake_catalog_payload = json.dumps(
        fake_catalog, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "geneb-embedding-set",
        "model_id": candidate["runtime_id"],
        "preset": preset,
        "catalog_sha256": digest(fake_catalog_payload),
        "dataset": {
            "repository": "darlednik/geneb-tasks",
            "revision": runner.DATASET_REVISION,
        },
        "output_width": 2,
        "tasks": manifest_tasks,
    }
    manifest_path = embedding_root / "embedding-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validated = runner.validate_embedding_manifest(
        manifest,
        manifest_path,
        model,
        preset,
        digest(fake_catalog_payload),
        tasks,
        data_dir,
    )
    if len(validated) != 100:
        raise AssertionError("embedding receipt did not validate 100 tasks")
    corrupted = copy.deepcopy(manifest)
    corrupted["tasks"][tasks[0]]["train"]["sha256"] = "0" * 64
    expect_error(
        lambda: runner.validate_embedding_manifest(
            corrupted,
            manifest_path,
            model,
            preset,
            digest(fake_catalog_payload),
            tasks,
            data_dir,
        ),
        "size/SHA256 differs",
    )
    traversal = copy.deepcopy(manifest)
    traversal["tasks"][tasks[0]]["train"]["path"] = "../all.npy"
    expect_error(
        lambda: runner.validate_embedding_manifest(
            traversal,
            manifest_path,
            model,
            preset,
            digest(fake_catalog_payload),
            tasks,
            data_dir,
        ),
        "normalized relative path",
    )

    complete_results = {
        regime: {
            task_id: {"MCC": 0.0, "Acc": 0.5, "F1": 0.5}
            for task_id in tasks
        }
        for regime in runner.REGIMES
    }
    runner.validate_complete_submission({"results": complete_results}, tasks)
    incomplete = copy.deepcopy(complete_results)
    incomplete["full"].pop(tasks[-1])
    expect_error(
        lambda: runner.validate_complete_submission(
            {"results": incomplete}, tasks
        ),
        "not complete for 100 tasks",
    )
    official_payload = json.dumps(
        {
            "model_id": candidate["geneb_model_id"],
            "results": complete_results,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    official_path = args.work_dir / "official.json"
    official_path.write_bytes(official_payload)
    reference_model = copy.deepcopy(candidate)
    reference_model["benchmark_provenance"][
        "official_submission_sha256"
    ] = digest(official_payload)
    if (
        runner.compare_reference_submission(
            {"results": complete_results}, official_path, reference_model
        )
        != digest(official_payload)
    ):
        raise AssertionError("reference comparison omitted the official digest")
    changed_results = copy.deepcopy(complete_results)
    changed_results["full"][tasks[0]]["MCC"] = 2.0e-6
    expect_error(
        lambda: runner.compare_reference_submission(
            {"results": changed_results}, official_path, reference_model
        ),
        "more than 1e-6",
    )

    unsupported_model = copy.deepcopy(catalog["models"][0])
    unsupported_model["oracle"]["status"] = "missing"
    unsupported_model["oracle"]["environment_lock"] = None
    unsupported_model["oracle"]["input_digest"] = None
    unsupported_model["oracle"]["tolerances"] = None
    unsupported_model["oracle"]["evidence"] = None
    unsupported_model["runtime_support"] = {
        "status": "cataloged",
        "artifact_profile": None,
        "reason": "test-only unpromoted row",
    }
    unsupported_model["backends"]["cpu"] = {
        "status": "not-promoted",
        "evidence": None,
    }
    unsupported_model["promotion_state"] = "cataloged"
    unsupported_catalog = copy.deepcopy(catalog)
    unsupported_catalog["models"][0] = unsupported_model
    unsupported_catalog_path = args.work_dir / "unsupported-catalog.json"
    unsupported_catalog_path.write_text(
        json.dumps(unsupported_catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    expected_output = args.work_dir / (
        str(unsupported_model["geneb_model_id"]) + ".json"
    )
    unsupported = subprocess.run(
        [
            sys.executable,
            str(runner_path),
            "--model",
            unsupported_model["runtime_id"],
            "--data-dir",
            str(data_dir),
            "--embeddings",
            str(manifest_path),
            "--output",
            str(expected_output),
            "--catalog",
            str(unsupported_catalog_path),
            "--benchmark-spec",
            str(args.benchmark_spec),
            "--lock",
            str(args.lock),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if unsupported.returncode != 2 or "runtime support gate" not in unsupported.stderr:
        raise AssertionError(
            "runner did not reject an unpromoted catalog row: " + unsupported.stderr
        )

    # The real catalog payload is intentionally distinct from the synthetic
    # contract catalog used above; retain this assertion as a hash-binding gate.
    if digest(catalog_payload) == digest(fake_catalog_payload):
        raise AssertionError("synthetic catalog mutation did not change its digest")
    print("GENEB runner contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
