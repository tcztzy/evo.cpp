#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract-test revision pinning, hashes, and runtime manifests offline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_TOOLS_DIRECTORY = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIRECTORY))

from evo.geneb_artifact import catalog_contract_sha256  # noqa: E402


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prepare_work_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name != "hf-fetch" and not resolved.name.startswith(
        "evo-hf-fetch-contract-"
    ):
        raise AssertionError("refusing to clean an unexpected fetch test directory")
    if path.is_symlink():
        raise AssertionError("fetch test work directory must not be a symlink")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)
    return resolved


def write_fake_hub(root: Path) -> Path:
    package = root / "fake-python" / "huggingface_hub"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        """from pathlib import Path
import os
import shutil
from types import SimpleNamespace

class HfApi:
    def __init__(self, **kwargs):
        pass

    def model_info(self, *, repo_id, revision, files_metadata=False):
        log = Path(os.environ[\"FAKE_HF_LOG\"])
        with log.open(\"a\") as output:
            output.write(f\"resolve {repo_id} {revision}\\n\")
        siblings = None
        if files_metadata:
            root = Path(os.environ[\"FAKE_HF_REMOTE\"]) / repo_id / revision
            override_name = os.environ.get(\"FAKE_HF_LARGE_METADATA_NAME\")
            override_size = int(os.environ.get(\"FAKE_HF_LARGE_METADATA_SIZE\", \"0\"))
            siblings = [
                SimpleNamespace(
                    rfilename=str(path.relative_to(root)),
                    size=(
                        override_size
                        if str(path.relative_to(root)) == override_name
                        else path.stat().st_size
                    ),
                )
                for path in sorted(root.rglob(\"*\"))
                if path.is_file()
            ]
        return SimpleNamespace(
            sha=os.environ[\"FAKE_HF_RESOLVED\"], siblings=siblings
        )

    def list_repo_files(self, *, repo_id, revision):
        log = Path(os.environ[\"FAKE_HF_LOG\"])
        with log.open(\"a\") as output:
            output.write(f\"list {repo_id} {revision}\\n\")
        root = Path(os.environ[\"FAKE_HF_REMOTE\"]) / repo_id / revision
        return sorted(
            str(path.relative_to(root))
            for path in root.rglob(\"*\")
            if path.is_file()
        )

def hf_hub_download(*, repo_id, filename, revision, cache_dir,
                    local_files_only=False, force_download=False, **kwargs):
    log = Path(os.environ[\"FAKE_HF_LOG\"])
    with log.open(\"a\") as output:
        output.write(
            f\"download {repo_id} {revision} {filename} \"
            f\"local={local_files_only} force={force_download}\\n\"
        )
    destination = (
        Path(cache_dir) / f\"models--{repo_id.replace('/', '--')}\"
        / \"snapshots\" / revision / filename
    )
    if (
        os.environ.get("FAKE_HF_FAIL_MIRROR_DOTFILE") == "1"
        and filename == ".gitattributes"
        and kwargs.get("endpoint") == "https://hf-mirror.example"
    ):
        raise RuntimeError("redirect metadata has no commit_hash/etag/size")
    if destination.is_file() and not force_download:
        return str(destination)
    if local_files_only:
        raise FileNotFoundError(filename)
    source = Path(os.environ[\"FAKE_HF_REMOTE\"]) / repo_id / revision / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return str(destination)
""",
        encoding="utf-8",
    )
    return package.parent


def run_tool(
    tool: Path,
    python_path: Path,
    remote: Path,
    log: Path,
    resolved: str,
    arguments: list[str],
    extra_environment=None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(python_path),
            "FAKE_HF_REMOTE": str(remote),
            "FAKE_HF_LOG": str(log),
            "FAKE_HF_RESOLVED": resolved,
        }
    )
    if extra_environment is not None:
        environment.update(extra_environment)
    return subprocess.run(
        [sys.executable, str(tool), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def require_success(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    if result.returncode != 0:
        raise AssertionError(f"fetch failed: stdout={result.stdout!r} stderr={result.stderr!r}")
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir = prepare_work_dir(args.work_dir)
    help_result = subprocess.run(
        [sys.executable, str(args.tool), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if (
        help_result.returncode != 0
        or "biological-model checkpoints" not in help_result.stdout
    ):
        raise AssertionError("evo-fetch help has stale project scope")
    python_path = write_fake_hub(args.work_dir)
    remote = args.work_dir / "remote"
    cache = args.work_dir / "cache"
    log = args.work_dir / "hub.log"

    source_revision = "a" * 40
    source_payload = b"registered source checkpoint"
    source_path = remote / "owner" / "source" / source_revision / "source.pt"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_payload)
    source_registry = args.work_dir / "source-registry.json"
    source_registry.write_text(
        json.dumps(
            {
                "models": {
                    "tiny": {
                        "source_repo": "owner/source",
                        "source_revision": source_revision,
                        "checkpoint_files": [
                            {
                                "name": "source.pt",
                                "size": len(source_payload),
                                "sha256": digest(source_payload),
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    source_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    source_receipt = require_success(source_result)
    cached_source = Path(source_receipt["files"][0]["path"])
    if any(
        (
            source_receipt["kind"] != "source-checkpoint",
            source_receipt["resolved_revision"] != source_revision,
            source_receipt["files"][0]["sha256"] != digest(source_payload),
            cached_source.read_bytes() != source_payload,
            not Path(source_receipt["receipt"]).is_file(),
        )
    ):
        raise AssertionError("source receipt omitted immutable provenance")

    cached_source.write_bytes(b"x" * len(source_payload))
    repaired_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    require_success(repaired_result)
    if (
        cached_source.read_bytes() != source_payload
        or "failed verification; refreshing" not in repaired_result.stderr
        or "force=True" not in log.read_text(encoding="utf-8")
    ):
        raise AssertionError("corrupt cached checkpoint was not refreshed")

    log.write_text("", encoding="utf-8")
    local_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    require_success(local_result)
    if log.read_text(encoding="utf-8"):
        raise AssertionError("offline source lookup depended on Hugging Face metadata")
    cached_source.write_bytes(b"x" * len(source_payload))
    offline_tampered_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    if (
        offline_tampered_result.returncode == 0
        or "integrity mismatch" not in offline_tampered_result.stderr
        or log.read_text(encoding="utf-8")
    ):
        raise AssertionError("offline source snapshot did not fail closed")
    cached_source.write_bytes(source_payload)
    isolated_receipts = args.work_dir / "isolated-receipts"
    isolated_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        source_revision,
        [
            "--cache-dir",
            str(cache),
            "--receipt-dir",
            str(isolated_receipts),
            "--local-files-only",
            "source",
            "tiny",
            "--registry",
            str(source_registry),
        ],
    )
    isolated_receipt = require_success(isolated_result)
    if not Path(isolated_receipt["receipt"]).is_relative_to(isolated_receipts):
        raise AssertionError("custom receipt escaped its explicit directory")

    geneb_revision = "c" * 40
    geneb_root = remote / "owner" / "geneb" / geneb_revision
    geneb_root.mkdir(parents=True, exist_ok=True)
    (geneb_root / ".gitattributes").write_bytes(b"*.safetensors filter=lfs\n")
    (geneb_root / "config.json").write_bytes(b'{"model":"geneb"}\n')
    (geneb_root / "model.safetensors").write_bytes(b"pinned geneb weights")
    (geneb_root / "tokenizer.json").write_bytes(b'{"tokenizer":true}\n')
    training_corpus = geneb_root / "tokenized_chromosomes" / "train.bin"
    training_corpus.parent.mkdir()
    training_corpus.write_bytes(b"must not be fetched")
    unbounded_root = remote / "owner" / "unbounded" / geneb_revision
    (unbounded_root / "tokenized_chromosomes").mkdir(parents=True)
    (unbounded_root / "config.json").write_bytes(b"{}\n")
    (unbounded_root / "tokenized_chromosomes" / "train.bin").write_bytes(
        b"metadata represents a very large corpus"
    )
    manual_id = "geneb-manual"
    geneb_catalog = args.work_dir / "geneb-catalog.json"
    geneb_catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "aliases": {"models": {"geneb-alias": "geneb-tiny"}},
                "models": [
                    {
                        "runtime_id": "geneb-tiny",
                        "tokenizer": {"kind": "fixture"},
                        "context": {"declared_max_tokens": 16},
                        "source": {
                            "kind": "huggingface",
                            "repo": "owner/geneb",
                            "requested_revision": "main",
                            "revision": geneb_revision,
                            "required_files": [
                                {"path": name, "size": None, "sha256": None}
                                for name in (
                                    ".gitattributes",
                                    "config.json",
                                    "model.safetensors",
                                    "tokenizer.json",
                                )
                            ],
                        },
                    },
                    {
                        "runtime_id": "geneb-unbounded",
                        "tokenizer": {"kind": "fixture"},
                        "context": {"declared_max_tokens": 16},
                        "source": {
                            "kind": "huggingface",
                            "repo": "owner/unbounded",
                            "requested_revision": "main",
                            "revision": geneb_revision,
                            "required_files": None,
                        },
                    },
                    {
                        "runtime_id": manual_id,
                        "source": {
                            "kind": "google-drive",
                            "url": "https://drive.example/manual",
                            "required_files": [
                                {
                                    "path": "manual.ckpt",
                                    "size": None,
                                    "sha256": None,
                                }
                            ],
                            "manual_instructions": (
                                "Download after accepting the provider terms."
                            ),
                        },
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    mirror_failure = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        geneb_revision,
        [
            "--cache-dir",
            str(cache),
            "--endpoint",
            "https://hf-mirror.example",
            "source",
            "geneb-alias",
            "--catalog",
            str(geneb_catalog),
        ],
        {"FAKE_HF_FAIL_MIRROR_DOTFILE": "1"},
    )
    if (
        mirror_failure.returncode == 0
        or "effective endpoint https://hf-mirror.example" not in mirror_failure.stderr
        or "--endpoint https://huggingface.co" not in mirror_failure.stderr
        or ".gitattributes" not in mirror_failure.stderr
    ):
        raise AssertionError(
            "mirror metadata failure was not actionable/fail-closed: "
            "returncode=%d stdout=%r stderr=%r"
            % (mirror_failure.returncode, mirror_failure.stdout, mirror_failure.stderr)
        )
    geneb_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        geneb_revision,
        [
            "--cache-dir",
            str(cache),
            "--endpoint",
            "https://huggingface.co",
            "source",
            "geneb-alias",
            "--catalog",
            str(geneb_catalog),
        ],
    )
    geneb_receipt = require_success(geneb_result)
    geneb_catalog_value = json.loads(geneb_catalog.read_text(encoding="utf-8"))
    geneb_model = geneb_catalog_value["models"][0]
    expected_contract = catalog_contract_sha256(geneb_catalog_value, geneb_model)
    if (
        geneb_receipt["model_id"] != "geneb-tiny"
        or geneb_receipt["requested_revision"] != "main"
        or geneb_receipt["resolved_revision"] != geneb_revision
        or geneb_receipt["source_kind"] != "huggingface"
        or geneb_receipt["catalog_contract_sha256"] != expected_contract
        or "catalog_sha256" in geneb_receipt
        or {entry["name"] for entry in geneb_receipt["files"]}
        != {".gitattributes", "config.json", "model.safetensors", "tokenizer.json"}
    ):
        raise AssertionError("GENEB discovery receipt omitted pinned provenance")
    if "download owner/geneb %s tokenized_chromosomes/train.bin" % geneb_revision in log.read_text(encoding="utf-8"):
        raise AssertionError("GENEB required_files allowlist downloaded training data")
    geneb_receipt_path = Path(geneb_receipt["receipt"])
    baseline_receipt = geneb_receipt_path.read_bytes()

    legacy_receipt = json.loads(baseline_receipt)
    legacy_receipt.pop("catalog_contract_sha256")
    legacy_receipt["catalog_sha256"] = digest(geneb_catalog.read_bytes())
    geneb_receipt_path.write_text(
        json.dumps(legacy_receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    migrated = require_success(
        run_tool(
            args.tool,
            python_path,
            remote,
            log,
            geneb_revision,
            [
                "--cache-dir",
                str(cache),
                "--endpoint",
                "https://huggingface.co",
                "source",
                "geneb-alias",
                "--catalog",
                str(geneb_catalog),
            ],
        )
    )
    if (
        migrated.get("catalog_contract_sha256") != expected_contract
        or "catalog_sha256" in migrated
        or geneb_receipt_path.read_bytes() != baseline_receipt
    ):
        raise AssertionError("legacy GENEB receipt did not migrate canonically")

    promotion_catalog = copy.deepcopy(geneb_catalog_value)
    for row in promotion_catalog["models"]:
        row["oracle"] = {"status": "passed"}
        row["runtime_support"] = {"status": "supported"}
        row["backends"] = {"cpu": {"status": "promoted"}}
        row["promotion_state"] = "runtime-supported"
    geneb_catalog.write_text(
        json.dumps(promotion_catalog, sort_keys=True) + "\n", encoding="utf-8"
    )
    promotion_receipt = require_success(
        run_tool(
            args.tool,
            python_path,
            remote,
            log,
            geneb_revision,
            [
                "--cache-dir",
                str(cache),
                "--endpoint",
                "https://huggingface.co",
                "source",
                "geneb-alias",
                "--catalog",
                str(geneb_catalog),
            ],
        )
    )
    if (
        promotion_receipt.get("catalog_contract_sha256") != expected_contract
        or geneb_receipt_path.read_bytes() != baseline_receipt
    ):
        raise AssertionError("promotion-only catalog changes altered source receipt")

    for label, mutate in (
        ("source", lambda row: row["source"].update({"url": "https://example.invalid"})),
        ("tokenizer", lambda row: row["tokenizer"].update({"kind": "changed"})),
        ("context", lambda row: row["context"].update({"declared_max_tokens": 17})),
    ):
        changed_catalog = copy.deepcopy(geneb_catalog_value)
        mutate(changed_catalog["models"][0])
        geneb_catalog.write_text(
            json.dumps(changed_catalog, sort_keys=True) + "\n", encoding="utf-8"
        )
        changed = run_tool(
            args.tool,
            python_path,
            remote,
            log,
            geneb_revision,
            [
                "--cache-dir",
                str(cache),
                "--endpoint",
                "https://huggingface.co",
                "source",
                "geneb-alias",
                "--catalog",
                str(geneb_catalog),
            ],
        )
        if changed.returncode == 0 or "does not match the catalog" not in changed.stderr:
            raise AssertionError("%s contract drift reused a prior receipt" % label)
    geneb_catalog.write_text(
        json.dumps(geneb_catalog_value, sort_keys=True) + "\n", encoding="utf-8"
    )
    log.write_text("", encoding="utf-8")
    unbounded = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        geneb_revision,
        [
            "--cache-dir",
            str(args.work_dir / "unbounded-cache"),
            "--endpoint",
            "https://huggingface.co",
            "source",
            "geneb-unbounded",
            "--catalog",
            str(geneb_catalog),
        ],
        {
            "FAKE_HF_LARGE_METADATA_NAME": "tokenized_chromosomes/train.bin",
            "FAKE_HF_LARGE_METADATA_SIZE": str(17 * 1024 * 1024 * 1024),
        },
    )
    if (
        unbounded.returncode == 0
        or "discovery safety cap" not in unbounded.stderr
        or "source.required_files" not in unbounded.stderr
        or "download owner/unbounded" in log.read_text(encoding="utf-8")
    ):
        raise AssertionError("unbounded GENEB snapshot was not rejected pre-download")
    config_payload = (geneb_root / "config.json").read_bytes()
    required_corruptions = (
        (
            "duplicate",
            [
                {"path": "config.json", "size": None, "sha256": None},
                {"path": "config.json", "size": None, "sha256": None},
            ],
            "duplicate paths",
        ),
        (
            "missing",
            [{"path": "missing.bin", "size": None, "sha256": None}],
            "missing or not regular files",
        ),
        (
            "directory",
            [
                {
                    "path": "tokenized_chromosomes",
                    "size": None,
                    "sha256": None,
                }
            ],
            "missing or not regular files",
        ),
        (
            "size",
            [
                {
                    "path": "config.json",
                    "size": len(config_payload) + 1,
                    "sha256": digest(config_payload),
                }
            ],
            "size/SHA256 differs",
        ),
        (
            "hash",
            [
                {
                    "path": "config.json",
                    "size": len(config_payload),
                    "sha256": "0" * 64,
                }
            ],
            "size/SHA256 differs",
        ),
        (
            "uppercase-hash",
            [
                {
                    "path": "config.json",
                    "size": len(config_payload),
                    "sha256": digest(config_payload).upper(),
                }
            ],
            "lowercase SHA256",
        ),
    )
    for label, required_files, expected_error in required_corruptions:
        corrupt_catalog_value = copy.deepcopy(geneb_catalog_value)
        corrupt_catalog_value["models"][0]["source"]["required_files"] = required_files
        corrupt_catalog = args.work_dir / ("required-%s.json" % label)
        corrupt_catalog.write_text(
            json.dumps(corrupt_catalog_value, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rejected_required = run_tool(
            args.tool,
            python_path,
            remote,
            log,
            geneb_revision,
            [
                "--cache-dir",
                str(args.work_dir / ("required-%s-cache" % label)),
                "--endpoint",
                "https://huggingface.co",
                "source",
                "geneb-tiny",
                "--catalog",
                str(corrupt_catalog),
            ],
        )
        if (
            rejected_required.returncode == 0
            or expected_error not in rejected_required.stderr
        ):
            raise AssertionError("required_files %s corruption was accepted" % label)
    log.write_text("", encoding="utf-8")
    geneb_offline = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        geneb_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "source",
            "geneb-tiny",
            "--catalog",
            str(geneb_catalog),
        ],
    )
    require_success(geneb_offline)
    if log.read_text(encoding="utf-8"):
        raise AssertionError("offline GENEB fetch contacted Hugging Face")
    manual_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        geneb_revision,
        ["source", manual_id, "--catalog", str(geneb_catalog)],
    )
    if (
        manual_result.returncode == 0
        or "manual-source" not in manual_result.stderr
        or "manual.ckpt" not in manual_result.stderr
        or "provider terms" not in manual_result.stderr
    ):
        raise AssertionError("manual GENEB source did not fail with instructions")

    runtime_revision = "b" * 40
    runtime_root = remote / "owner" / "runtime" / runtime_revision
    runtime_root.mkdir(parents=True, exist_ok=True)
    index_payload = b'{"metadata":{},"weight_map":{}}\n'
    shard_payload = b"strict runtime shard"
    runtime_manifest = {
        "schema_version": 1,
        "artifact_profile": "esmc-runtime-v1",
        "model_id": "tiny_runtime",
        "load_path": "model.safetensors.index.json",
        "files": [
            {
                "path": "model.safetensors.index.json",
                "size": len(index_payload),
                "sha256": digest(index_payload),
            },
            {
                "path": "model-00001-of-00001.safetensors",
                "size": len(shard_payload),
                "sha256": digest(shard_payload),
            },
        ],
    }
    (runtime_root / "evo-artifact.json").write_text(
        json.dumps(runtime_manifest), encoding="utf-8"
    )
    (runtime_root / "model.safetensors.index.json").write_bytes(index_payload)
    (runtime_root / "model-00001-of-00001.safetensors").write_bytes(shard_payload)
    runtime_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        runtime_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            "owner/runtime@release-v1",
            "--registry",
            str(args.registry),
        ],
    )
    runtime_receipt = require_success(runtime_result)
    if any(
        (
            runtime_receipt["kind"] != "runtime-artifact",
            runtime_receipt["resolved_revision"] != runtime_revision,
            runtime_receipt["artifact_profile"] != "esmc-runtime-v1",
            Path(runtime_receipt["load_path"]).read_bytes() != index_payload,
            len(runtime_receipt["files"]) != 2,
        )
    ):
        raise AssertionError("runtime artifact was not resolved from its manifest")
    offline_runtime_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        runtime_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "runtime",
            f"owner/runtime@{runtime_revision}",
            "--registry",
            str(args.registry),
        ],
    )
    require_success(offline_runtime_result)
    cached_manifest = Path(runtime_receipt["manifest_path"])
    cached_manifest.write_bytes(b"{" + b" " * (cached_manifest.stat().st_size - 1))
    tampered_runtime_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        runtime_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "runtime",
            f"owner/runtime@{runtime_revision}",
            "--registry",
            str(args.registry),
        ],
    )
    if (
        tampered_runtime_result.returncode == 0
        or "manifest SHA256 differs" not in tampered_runtime_result.stderr
    ):
        raise AssertionError("offline runtime accepted a tampered cached manifest")

    invalid_revision = "c" * 40
    invalid_root = remote / "owner" / "invalid" / invalid_revision
    invalid_root.mkdir(parents=True, exist_ok=True)
    (invalid_root / "evo-artifact.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_profile": "evo2-runtime-v1",
                "model_id": "invalid",
                "load_path": "../escape.safetensors",
                "files": [
                    {"path": "../escape.safetensors", "size": 0, "sha256": "0" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )
    invalid_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        invalid_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            f"owner/invalid@{invalid_revision}",
            "--registry",
            str(args.registry),
        ],
    )
    if invalid_result.returncode == 0 or "normalized relative path" not in invalid_result.stderr:
        raise AssertionError("runtime manifest path traversal was not rejected")
    unknown_revision = "d" * 40
    unknown_root = remote / "owner" / "unknown" / unknown_revision
    unknown_root.mkdir(parents=True, exist_ok=True)
    unknown_manifest = copy.deepcopy(runtime_manifest)
    unknown_manifest["artifact_profile"] = "unknown-runtime-v1"
    (unknown_root / "evo-artifact.json").write_text(
        json.dumps(unknown_manifest), encoding="utf-8"
    )
    unknown_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        unknown_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            f"owner/unknown@{unknown_revision}",
            "--registry",
            str(args.registry),
        ],
    )
    if (
        unknown_result.returncode == 0
        or "unknown-runtime-v1' is not registered" not in unknown_result.stderr
    ):
        raise AssertionError("runtime fetch accepted an unregistered artifact profile")

    canonical_registry = json.loads(args.registry.read_text(encoding="utf-8"))
    extended_registry = copy.deepcopy(canonical_registry)
    extended_registry["artifact_profiles"].append(
        {
            "id": "fixture-runtime-v1",
            "metadata_key": "runtime.profile",
            "runtime_abi": "fixture-safetensors-v1",
        }
    )
    extended_registry["runtime_architectures"].append(
        {
            "id": "FixtureArchitecture",
            "artifact_profile": "fixture-runtime-v1",
            "runtime_abi": "fixture-safetensors-v1",
        }
    )
    extended_registry_path = args.work_dir / "extended-runtime-registry.json"
    extended_registry_path.write_text(
        json.dumps(extended_registry), encoding="utf-8"
    )
    custom_revision = "e" * 40
    custom_root = remote / "owner" / "custom" / custom_revision
    custom_root.mkdir(parents=True, exist_ok=True)
    custom_manifest = copy.deepcopy(runtime_manifest)
    custom_manifest["artifact_profile"] = "fixture-runtime-v1"
    (custom_root / "evo-artifact.json").write_text(
        json.dumps(custom_manifest), encoding="utf-8"
    )
    (custom_root / "model.safetensors.index.json").write_bytes(index_payload)
    (custom_root / "model-00001-of-00001.safetensors").write_bytes(shard_payload)
    custom_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        custom_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            f"owner/custom@{custom_revision}",
            "--registry",
            str(extended_registry_path),
        ],
    )
    custom_receipt = require_success(custom_result)
    if custom_receipt["artifact_profile"] != "fixture-runtime-v1":
        raise AssertionError("runtime fetch did not use its explicit profile registry")

    corrupt_registry = copy.deepcopy(canonical_registry)
    corrupt_registry["artifact_profiles"].append(
        copy.deepcopy(corrupt_registry["artifact_profiles"][0])
    )
    corrupt_registry_path = args.work_dir / "corrupt-runtime-registry.json"
    corrupt_registry_path.write_text(
        json.dumps(corrupt_registry), encoding="utf-8"
    )
    corrupt_result = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        custom_revision,
        [
            "--cache-dir",
            str(cache),
            "runtime",
            f"owner/custom@{custom_revision}",
            "--registry",
            str(corrupt_registry_path),
        ],
    )
    if (
        corrupt_result.returncode == 0
        or "duplicate artifact profile" not in corrupt_result.stderr
    ):
        raise AssertionError("runtime fetch accepted a corrupt profile registry")

    revision_traversal = run_tool(
        args.tool,
        python_path,
        remote,
        log,
        runtime_revision,
        [
            "--cache-dir",
            str(cache),
            "--local-files-only",
            "runtime",
            "owner/runtime@../../escape",
            "--registry",
            str(args.registry),
        ],
    )
    if revision_traversal.returncode == 0 or "path traversal" not in revision_traversal.stderr:
        raise AssertionError("revision path traversal was not rejected locally")

    print("Hugging Face fetch contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
