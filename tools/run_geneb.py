#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the locked GENEB v4 linear probe on receipted native embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


BENCHMARK_SPEC_SHA256 = (
    "bf3ba7136f3448735ad5d1a1635ba3775097800938d3e1b4262c33a22d519e2f"
)
BENCHMARK_COMMIT = "b54d018903e7f6b874ee45b74e275936deff4cd3"
DATASET_REVISION = "4edd705be573e48c585c2cf79dc320f9f43c7b04"
LOCK_ID = "geneb-v4-probe-py311-v1"
REGIMES = ["full", "10shot", "1shot"]
METRICS = ["MCC", "Acc", "F1"]
SEEDS = [13, 17, 42, 123, 997]
KSHOTS = [1, 10]
MAX_JSON_BYTES = 64 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


class GenebRunError(RuntimeError):
    """Raised when a frozen GENEB probe contract is not satisfied."""


def default_config(name: str) -> Path:
    override = os.environ.get("EVO_GENEB_CONFIG_DIR")
    if override:
        return Path(override) / name
    source = Path(__file__).resolve().parents[1] / "configs" / name
    if source.is_file():
        return source
    return Path(__file__).resolve().parent.parent / "share" / "evo" / "configs" / name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, label: str) -> Tuple[Dict[str, Any], bytes]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise GenebRunError("cannot stat {} {}: {}".format(label, path, error))
    if size <= 0 or size > MAX_JSON_BYTES:
        raise GenebRunError("{} size is outside the supported range".format(label))

    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result = {}  # type: Dict[str, Any]
        for key, value in pairs:
            if key in result:
                raise GenebRunError("{} contains duplicate key {!r}".format(label, key))
            result[key] = value
        return result

    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except GenebRunError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenebRunError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise GenebRunError("{} root must be an object".format(label))
    return value, payload


def exact_keys(
    value: Mapping[str, Any], required: Sequence[str], optional: Sequence[str], label: str
) -> None:
    expected = set(required) | set(optional)
    actual = set(value)
    missing = set(required) - actual
    extra = actual - expected
    if missing or extra:
        raise GenebRunError(
            "{} keys differ (missing={}, extra={})".format(
                label, sorted(missing), sorted(extra)
            )
        )


def safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GenebRunError("{} must be a nonempty relative path".format(label))
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise GenebRunError("{} must be a normalized relative path".format(label))
    return value


def resolve_regular_below(root: Path, relative: str, label: str) -> Path:
    root = root.resolve(strict=True)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            if current.is_symlink():
                raise GenebRunError("{} contains a symbolic link".format(label))
        except OSError as error:
            raise GenebRunError("cannot inspect {}: {}".format(label, error))
    try:
        resolved = current.resolve(strict=True)
    except OSError as error:
        raise GenebRunError("cannot resolve {}: {}".format(label, error))
    try:
        common = Path(os.path.commonpath([str(root), str(resolved)]))
    except ValueError as error:
        raise GenebRunError("{} escapes its root: {}".format(label, error))
    if common != root or not resolved.is_file():
        raise GenebRunError("{} is not a regular file below its root".format(label))
    return resolved


def validate_benchmark_spec(spec: Mapping[str, Any]) -> List[str]:
    if spec.get("schema_version") != "1.0":
        raise GenebRunError("benchmark spec schema_version must be 1.0")
    if spec.get("harness_version") != "GENEB-0.1.0":
        raise GenebRunError("benchmark harness version differs")
    if spec.get("regimes") != REGIMES or spec.get("metrics") != METRICS:
        raise GenebRunError("benchmark regimes/metrics differ")
    protocol = spec.get("protocol")
    dataset = spec.get("dataset")
    tasks = spec.get("tasks")
    if not isinstance(protocol, dict) or not isinstance(dataset, dict):
        raise GenebRunError("benchmark protocol/dataset must be objects")
    if (
        protocol.get("seeds") != SEEDS
        or protocol.get("few_shot_k") != KSHOTS
        or protocol.get("logreg") != {"max_iter": 1000, "n_jobs": 32}
    ):
        raise GenebRunError("benchmark probe protocol differs")
    if (
        dataset.get("repo_id") != "darlednik/geneb-tasks"
        or dataset.get("revision") != DATASET_REVISION
    ):
        raise GenebRunError("benchmark dataset identity differs")
    if not isinstance(tasks, list) or len(tasks) != 100:
        raise GenebRunError("benchmark must contain exactly 100 tasks")
    task_ids = []  # type: List[str]
    category_counts = {}  # type: Dict[str, int]
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or set(task) != {"id", "label", "category"}:
            raise GenebRunError("benchmark task {} has invalid fields".format(index))
        task_id = task.get("id")
        category = task.get("category")
        if not isinstance(task_id, str) or not task_id or not isinstance(category, str):
            raise GenebRunError("benchmark task {} has invalid identity".format(index))
        task_ids.append(task_id)
        category_counts[category] = category_counts.get(category, 0) + 1
    if len(set(task_ids)) != 100:
        raise GenebRunError("benchmark task IDs are not unique")
    if category_counts != spec.get("categories") or len(category_counts) != 13:
        raise GenebRunError("benchmark category counts differ")
    return task_ids


def validate_lock(
    lock: Mapping[str, Any], lock_payload: bytes, spec_path: Path
) -> Tuple[List[str], str]:
    exact_keys(
        lock,
        ["schema_version", "id", "benchmark", "dataset", "environment", "thread_environment", "protocol"],
        [],
        "probe lock",
    )
    if lock.get("schema_version") != 1 or lock.get("id") != LOCK_ID:
        raise GenebRunError("probe lock identity differs")
    benchmark = lock.get("benchmark")
    dataset = lock.get("dataset")
    protocol = lock.get("protocol")
    if not isinstance(benchmark, dict) or not isinstance(dataset, dict) or not isinstance(protocol, dict):
        raise GenebRunError("probe lock nested records must be objects")
    if (
        benchmark.get("revision") != BENCHMARK_COMMIT
        or benchmark.get("spec_sha256") != BENCHMARK_SPEC_SHA256
        or dataset
        != {
            "repository": "darlednik/geneb-tasks",
            "revision": DATASET_REVISION,
            "tasks": 100,
        }
    ):
        raise GenebRunError("probe lock upstream identity differs")
    spec, spec_payload = load_json(spec_path, "benchmark spec")
    if hashlib.sha256(spec_payload).hexdigest() != BENCHMARK_SPEC_SHA256:
        raise GenebRunError("benchmark spec SHA256 differs from the pinned file")
    task_ids = validate_benchmark_spec(spec)
    if (
        protocol.get("regimes") != REGIMES
        or protocol.get("metrics") != METRICS
        or protocol.get("seeds") != SEEDS
        or protocol.get("few_shot_k") != KSHOTS
    ):
        raise GenebRunError("locked regimes/metrics/seeds/few-shot policy differs")
    kwargs = protocol.get("logistic_regression")
    expected_kwargs = {
        "penalty": "l2",
        "dual": False,
        "tol": 0.0001,
        "C": 1.0,
        "fit_intercept": True,
        "intercept_scaling": 1,
        "class_weight": None,
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "verbose": 0,
        "warm_start": False,
        "n_jobs": 32,
        "l1_ratio": None,
    }
    if kwargs != expected_kwargs:
        raise GenebRunError("locked LogisticRegression kwargs differ")
    return task_ids, hashlib.sha256(lock_payload).hexdigest()


def select_catalog_model(
    catalog: Mapping[str, Any], model_id: str, preset: str
) -> Tuple[Mapping[str, Any], str]:
    if catalog.get("schema_version") != 1:
        raise GenebRunError("GENEB catalog schema_version must be 1")
    suite = catalog.get("suite")
    aliases = catalog.get("aliases")
    models = catalog.get("models")
    if not isinstance(suite, dict) or not isinstance(aliases, dict) or not isinstance(models, list):
        raise GenebRunError("GENEB catalog root is invalid")
    dataset = suite.get("dataset")
    if (
        suite.get("id") != "geneb-v4"
        or not isinstance(dataset, dict)
        or dataset.get("revision") != DATASET_REVISION
        or dataset.get("tasks") != 100
    ):
        raise GenebRunError("GENEB catalog suite/dataset identity differs")
    model_aliases = aliases.get("models")
    if isinstance(model_aliases, dict) and isinstance(model_aliases.get(model_id), str):
        model_id = model_aliases[model_id]
    preset_aliases = aliases.get("presets")
    if isinstance(preset_aliases, dict) and isinstance(preset_aliases.get(preset), str):
        preset = preset_aliases[preset]
    if preset not in {"geneb-v4-reference", "geneb-v4-normalized"}:
        raise GenebRunError("preset must be geneb-v4-reference or geneb-v4-normalized")
    matches = [
        item
        for item in models
        if isinstance(item, dict) and item.get("runtime_id") == model_id
    ]
    if len(matches) != 1:
        raise GenebRunError("unknown GENEB runtime model ID: {}".format(model_id))
    model = matches[0]
    support = model.get("runtime_support")
    provenance = model.get("benchmark_provenance")
    if not isinstance(support, dict) or support.get("status") != "supported":
        raise GenebRunError("{} has not passed the native runtime support gate".format(model_id))
    if not isinstance(provenance, dict):
        raise GenebRunError("{} benchmark provenance is invalid".format(model_id))
    if preset == "geneb-v4-reference" and provenance.get("reference_status") != "eligible":
        raise GenebRunError("{} is not reference-eligible".format(model_id))
    return model, preset


def validate_environment(lock: Mapping[str, Any]) -> Dict[str, str]:
    expected = lock.get("environment")
    thread_environment = lock.get("thread_environment")
    if not isinstance(expected, dict) or not isinstance(thread_environment, dict):
        raise GenebRunError("probe environment lock is invalid")
    actual = {"python": platform.python_version()}  # type: Dict[str, str]
    modules = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib"),
        ("threadpoolctl", "threadpoolctl"),
    ]
    for label, module_name in modules:
        try:
            module = __import__(module_name)
        except ImportError as error:
            raise GenebRunError("locked environment is missing {}".format(label)) from error
        version = getattr(module, "__version__", None)
        if not isinstance(version, str):
            raise GenebRunError("cannot determine {} version".format(label))
        actual[label] = version
    if actual != expected:
        raise GenebRunError("environment differs: expected {}, got {}".format(expected, actual))
    wrong = {
        key: (value, os.environ.get(key))
        for key, value in thread_environment.items()
        if os.environ.get(key) != value
    }
    if wrong:
        raise GenebRunError("thread/locale environment differs: {}".format(wrong))
    return actual


def dataset_split_path(data_dir: Path, task_id: str, split: str) -> Path:
    candidates = [
        data_dir / "tasks" / task_id / (split + ".csv"),
        data_dir / task_id / (split + ".csv"),
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise GenebRunError(
            "dataset split {} {} must exist in exactly one pinned layout".format(task_id, split)
        )
    return present[0].resolve(strict=True)


def read_labels(path: Path, expected_rows: int) -> List[int]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
    except (OSError, UnicodeError, csv.Error) as error:
        raise GenebRunError("cannot read dataset CSV {}: {}".format(path, error))
    if not rows or set(rows[0]) != {"text", "label"}:
        raise GenebRunError("{} must have exactly text,label columns".format(path))
    if len(rows) != expected_rows:
        raise GenebRunError("{} row count differs from embedding receipt".format(path))
    labels = []  # type: List[int]
    for index, row in enumerate(rows):
        try:
            value = int(row["label"])
        except (KeyError, TypeError, ValueError) as error:
            raise GenebRunError("{} row {} has invalid label".format(path, index)) from error
        if str(value) != row["label"].strip():
            raise GenebRunError("{} row {} label is not canonical integer text".format(path, index))
        labels.append(value)
    return labels


def validate_embedding_manifest(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    model: Mapping[str, Any],
    preset: str,
    catalog_sha256: str,
    task_ids: Sequence[str],
    data_dir: Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    exact_keys(
        manifest,
        ["schema_version", "kind", "model_id", "preset", "catalog_sha256", "dataset", "output_width", "tasks"],
        [],
        "embedding manifest",
    )
    dataset = manifest.get("dataset")
    tasks = manifest.get("tasks")
    presets = model.get("embedding_presets")
    preset_key = "reference" if preset.endswith("reference") else "normalized"
    preset_record = presets.get(preset_key) if isinstance(presets, dict) else None
    output_width = preset_record.get("output_width") if isinstance(preset_record, dict) else None
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "geneb-embedding-set"
        or manifest.get("model_id") != model.get("runtime_id")
        or manifest.get("preset") != preset
        or manifest.get("catalog_sha256") != catalog_sha256
        or manifest.get("output_width") != output_width
        or not isinstance(dataset, dict)
        or dataset
        != {"repository": "darlednik/geneb-tasks", "revision": DATASET_REVISION}
        or not isinstance(tasks, dict)
        or set(tasks) != set(task_ids)
    ):
        raise GenebRunError("embedding manifest identity/completeness differs")
    root = manifest_path.resolve(strict=True).parent
    validated = {}  # type: Dict[str, Dict[str, Dict[str, Any]]]
    for task_id in task_ids:
        task = tasks[task_id]
        if not isinstance(task, dict) or set(task) != {"train", "test"}:
            raise GenebRunError("embedding task {} must contain train/test".format(task_id))
        validated[task_id] = {}
        for split in ("train", "test"):
            record = task[split]
            label = "embedding {} {}".format(task_id, split)
            if not isinstance(record, dict):
                raise GenebRunError("{} must be an object".format(label))
            exact_keys(
                record,
                ["path", "size", "sha256", "rows", "columns", "input_sha256"],
                [],
                label,
            )
            relative = safe_relative(record.get("path"), label + ".path")
            embedding_path = resolve_regular_below(root, relative, label + ".path")
            size = record.get("size")
            rows = record.get("rows")
            columns = record.get("columns")
            digest = record.get("sha256")
            input_digest = record.get("input_sha256")
            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
                or not isinstance(rows, int)
                or isinstance(rows, bool)
                or rows <= 0
                or not isinstance(columns, int)
                or isinstance(columns, bool)
                or columns != output_width
                or not isinstance(digest, str)
                or SHA256_RE.fullmatch(digest) is None
                or not isinstance(input_digest, str)
                or SHA256_RE.fullmatch(input_digest) is None
            ):
                raise GenebRunError("{} receipt fields are invalid".format(label))
            if embedding_path.suffix != ".npy":
                raise GenebRunError("{} must reference an NPY file".format(label))
            if embedding_path.stat().st_size != size or sha256_file(embedding_path) != digest:
                raise GenebRunError("{} size/SHA256 differs".format(label))
            csv_path = dataset_split_path(data_dir, task_id, split)
            if sha256_file(csv_path) != input_digest:
                raise GenebRunError("{} dataset input SHA256 differs".format(label))
            labels = read_labels(csv_path, rows)
            validated[task_id][split] = {
                "embedding_path": embedding_path,
                "labels": labels,
                "rows": rows,
                "columns": columns,
                "input_sha256": input_digest,
            }
    return validated


def load_embedding_matrix(record: Mapping[str, Any]) -> Any:
    import numpy as np

    try:
        matrix = np.load(str(record["embedding_path"]), mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise GenebRunError("cannot load embedding NPY: {}".format(error))
    if (
        matrix.ndim != 2
        or matrix.shape != (record["rows"], record["columns"])
        or matrix.dtype != np.dtype("float32")
        or not np.isfinite(matrix).all()
    ):
        raise GenebRunError("embedding NPY shape/dtype/finiteness differs")
    return matrix


def few_shot_indices(labels: Any, k: int, seed: int) -> Any:
    import numpy as np

    rng = np.random.RandomState(seed)
    return np.concatenate(
        [
            rng.choice(locations, size=min(k, len(locations)), replace=False)
            for class_id in np.unique(labels)
            for locations in [np.where(labels == class_id)[0]]
        ]
    )


def evaluate_task(
    train_embeddings: Any,
    train_labels: Sequence[int],
    test_embeddings: Any,
    test_labels: Sequence[int],
    kwargs: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    y_train = np.asarray(train_labels, dtype=np.int64)
    y_test = np.asarray(test_labels, dtype=np.int64)
    if len(np.unique(y_train)) < 2 or len(y_test) == 0:
        raise GenebRunError("probe task must contain at least two train classes and test rows")
    results = {}  # type: Dict[str, Dict[str, float]]
    for regime in REGIMES:
        mcc_values = []  # type: List[float]
        accuracy_values = []  # type: List[float]
        f1_values = []  # type: List[float]
        k = None  # type: Optional[int]
        if regime == "10shot":
            k = 10
        elif regime == "1shot":
            k = 1
        for seed in SEEDS:
            if k is None:
                selected = np.arange(len(y_train), dtype=np.int64)
            else:
                selected = few_shot_indices(y_train, k, seed)
            arguments = dict(kwargs)
            arguments["random_state"] = seed
            classifier = LogisticRegression(**arguments)
            classifier.fit(train_embeddings[selected], y_train[selected])
            predicted = classifier.predict(test_embeddings)
            mcc_values.append(float(matthews_corrcoef(y_test, predicted)))
            accuracy_values.append(float(accuracy_score(y_test, predicted)))
            f1_values.append(float(f1_score(y_test, predicted, average="macro")))
        record = {
            "MCC": float(np.mean(mcc_values)),
            "Acc": float(np.mean(accuracy_values)),
            "F1": float(np.mean(f1_values)),
        }
        if not all(math.isfinite(value) for value in record.values()):
            raise GenebRunError("probe produced a non-finite metric")
        results[regime] = record
    return results


def build_submission(
    model: Mapping[str, Any],
    preset: str,
    task_ids: Sequence[str],
    validated: Mapping[str, Mapping[str, Mapping[str, Any]]],
    lock: Mapping[str, Any],
    catalog_sha256: str,
    manifest_sha256: str,
    lock_sha256: str,
    environment: Mapping[str, str],
) -> Dict[str, Any]:
    results = {regime: {} for regime in REGIMES}  # type: Dict[str, Dict[str, Any]]
    protocol = lock["protocol"]
    kwargs = protocol["logistic_regression"]
    for index, task_id in enumerate(task_ids, 1):
        print("[{}/{}] {}".format(index, len(task_ids), task_id), file=sys.stderr, flush=True)
        train = validated[task_id]["train"]
        test = validated[task_id]["test"]
        per_regime = evaluate_task(
            load_embedding_matrix(train),
            train["labels"],
            load_embedding_matrix(test),
            test["labels"],
            kwargs,
        )
        for regime in REGIMES:
            results[regime][task_id] = per_regime[regime]
    reference = preset == "geneb-v4-reference"
    source = model.get("source")
    url = source.get("url") if isinstance(source, dict) else ""
    return {
        "schema_version": "1.0",
        "model_id": model["geneb_model_id"],
        "harness_version": "GENEB-0.1.0",
        "meta": {
            "display": model["paper_name"],
            "params": model["params"],
            "url": url if isinstance(url, str) else "",
            "zero_shot": False,
            "training_data": "",
            "provenance": "reproduced" if reference else "self-reported",
            "submitted_by": "evo.cpp",
        },
        "results": results,
        "evidence": {
            "namespace": "reference" if reference else "normalized",
            "claim": "reference-eligible" if reference else "protocol-compatible",
            "runtime_id": model["runtime_id"],
            "preset": preset,
            "catalog_sha256": catalog_sha256,
            "embedding_manifest_sha256": manifest_sha256,
            "benchmark_spec_sha256": BENCHMARK_SPEC_SHA256,
            "benchmark_commit": BENCHMARK_COMMIT,
            "dataset_revision": DATASET_REVISION,
            "environment_lock_id": LOCK_ID,
            "environment_lock_sha256": lock_sha256,
            "environment": dict(environment),
            "thread_environment": dict(lock["thread_environment"]),
            "logistic_regression": dict(lock["protocol"]["logistic_regression"]),
            "seeds": list(SEEDS),
        },
    }


def validate_complete_submission(
    submission: Mapping[str, Any], task_ids: Sequence[str]
) -> None:
    results = submission.get("results")
    if not isinstance(results, dict) or set(results) != set(REGIMES):
        raise GenebRunError("submission must contain exactly the three GENEB regimes")
    expected_tasks = set(task_ids)
    ranges = {"MCC": (-1.0, 1.0), "Acc": (0.0, 1.0), "F1": (0.0, 1.0)}
    for regime in REGIMES:
        per_task = results.get(regime)
        if not isinstance(per_task, dict) or set(per_task) != expected_tasks:
            raise GenebRunError(
                "submission {} regime is not complete for 100 tasks".format(regime)
            )
        for task_id in task_ids:
            scores = per_task[task_id]
            if not isinstance(scores, dict) or set(scores) != set(METRICS):
                raise GenebRunError(
                    "submission {}/{} metrics differ".format(regime, task_id)
                )
            for metric in METRICS:
                value = scores[metric]
                lower, upper = ranges[metric]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < lower - 1.0e-6
                    or float(value) > upper + 1.0e-6
                ):
                    raise GenebRunError(
                        "submission {}/{}/{} is invalid".format(
                            regime, task_id, metric
                        )
                    )


def compare_reference_submission(
    candidate: Mapping[str, Any], official_path: Path, model: Mapping[str, Any]
) -> str:
    provenance = model.get("benchmark_provenance")
    if not isinstance(provenance, dict):
        raise GenebRunError("reference model provenance is invalid")
    expected_sha256 = provenance.get("official_submission_sha256")
    if not isinstance(expected_sha256, str) or SHA256_RE.fullmatch(expected_sha256) is None:
        raise GenebRunError("reference submission SHA256 is not cataloged")
    official, payload = load_json(official_path, "official submission")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise GenebRunError("official submission SHA256 differs from the catalog")
    if official.get("model_id") != model.get("geneb_model_id"):
        raise GenebRunError("official submission model ID differs")
    candidate_results = candidate.get("results")
    official_results = official.get("results")
    if not isinstance(candidate_results, dict) or not isinstance(official_results, dict):
        raise GenebRunError("reference submission results are invalid")
    for regime in REGIMES:
        candidate_tasks = candidate_results.get(regime)
        official_tasks = official_results.get(regime)
        if (
            not isinstance(candidate_tasks, dict)
            or not isinstance(official_tasks, dict)
            or set(candidate_tasks) != set(official_tasks)
        ):
            raise GenebRunError("reference submission task set differs")
        for task_id, scores in candidate_tasks.items():
            official_scores = official_tasks.get(task_id)
            if not isinstance(scores, dict) or not isinstance(official_scores, dict):
                raise GenebRunError("reference submission metric record is invalid")
            for metric in METRICS:
                left = scores.get(metric)
                right = official_scores.get(metric)
                if (
                    isinstance(left, bool)
                    or isinstance(right, bool)
                    or not isinstance(left, (int, float))
                    or not isinstance(right, (int, float))
                    or abs(float(left) - float(right)) > 1.0e-6
                ):
                    raise GenebRunError(
                        "reference metric differs by more than 1e-6 at {}/{}/{}"
                        .format(regime, task_id, metric)
                    )
    return actual_sha256


def write_json_atomic(path: Path, value: Mapping[str, Any], force: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise GenebRunError("output already exists: {}".format(path))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if force:
            os.replace(str(temporary), str(path))
        else:
            os.link(str(temporary), str(path))
            temporary.unlink()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked 100-task GENEB v4 probe on receipted embeddings"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--preset",
        default="geneb",
        choices=["geneb", "geneb-v4-reference", "geneb-v4-normalized"],
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--official-submission",
        type=Path,
        help="required for reference runs; must match the cataloged official SHA256",
    )
    parser.add_argument("--catalog", type=Path, default=default_config("geneb-models.json"))
    parser.add_argument(
        "--benchmark-spec", type=Path, default=default_config("geneb-benchmark-spec.json")
    )
    parser.add_argument("--lock", type=Path, default=default_config("geneb-probe-lock.json"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        catalog_path = args.catalog.expanduser().resolve(strict=True)
        spec_path = args.benchmark_spec.expanduser().resolve(strict=True)
        lock_path = args.lock.expanduser().resolve(strict=True)
        manifest_path = args.embeddings.expanduser().resolve(strict=True)
        data_dir = args.data_dir.expanduser().resolve(strict=True)
        catalog, catalog_payload = load_json(catalog_path, "GENEB catalog")
        lock, lock_payload = load_json(lock_path, "GENEB probe lock")
        task_ids, lock_sha256 = validate_lock(lock, lock_payload, spec_path)
        model, preset = select_catalog_model(catalog, args.model, args.preset)
        expected_name = SAFE_FILENAME_RE.sub("_", str(model["geneb_model_id"])) + ".json"
        if args.output.name != expected_name:
            raise GenebRunError(
                "output filename must be the sanitized GENEB model ID: {}".format(
                    expected_name
                )
            )
        if preset == "geneb-v4-reference" and args.official_submission is None:
            raise GenebRunError("reference run requires --official-submission")
        if preset == "geneb-v4-normalized" and args.official_submission is not None:
            raise GenebRunError("normalized run must not compare an official submission")
        environment = validate_environment(lock)
        catalog_sha256 = hashlib.sha256(catalog_payload).hexdigest()
        manifest, manifest_payload = load_json(manifest_path, "embedding manifest")
        validated = validate_embedding_manifest(
            manifest,
            manifest_path,
            model,
            preset,
            catalog_sha256,
            task_ids,
            data_dir,
        )
        submission = build_submission(
            model,
            preset,
            task_ids,
            validated,
            lock,
            catalog_sha256,
            hashlib.sha256(manifest_payload).hexdigest(),
            lock_sha256,
            environment,
        )
        validate_complete_submission(submission, task_ids)
        if args.official_submission is not None:
            official_sha256 = compare_reference_submission(
                submission,
                args.official_submission.expanduser().resolve(strict=True),
                model,
            )
            submission["evidence"]["official_submission_sha256"] = official_sha256
            submission["evidence"]["metric_tolerance"] = 1.0e-6
        write_json_atomic(args.output.expanduser(), submission, args.force)
        print(str(args.output.expanduser().resolve()))
        return 0
    except (GenebRunError, OSError, ValueError) as error:
        print("run_geneb: error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
