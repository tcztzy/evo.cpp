#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch revision-pinned Evo source checkpoints or runtime artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
RUNTIME_PROFILES = {
    "evo2-runtime-v1",
    "hyenadna-runtime-v1",
    "esmc-runtime-v1",
}


class FetchError(RuntimeError):
    """Raised when acquisition or provenance validation fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_repo_spec(value: str) -> tuple[str, str]:
    repo, separator, revision = value.rpartition("@")
    if not separator:
        repo, revision = value, "main"
    if not REPO_RE.fullmatch(repo):
        raise FetchError("repository must be OWNER/NAME with optional @REVISION")
    return repo, validate_revision(revision)


def safe_repo_path(value: str) -> str:
    return value.replace("/", "--")


def validate_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FetchError(f"{label} must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FetchError(f"{label} must be a normalized relative path")
    return value


def validate_revision(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
    ):
        raise FetchError(
            "repository revision must be nonempty and contain no whitespace"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FetchError("repository revision must not contain path traversal")
    return value


def default_registry() -> Path:
    override = os.environ.get("EVO_MODEL_REGISTRY")
    if override:
        return Path(override)
    source_candidate = (
        Path(__file__).resolve().parents[1] / "configs" / "model-registry.json"
    )
    if source_candidate.is_file():
        return source_candidate
    installed_candidate = (
        Path(__file__).resolve().parent.parent
        / "share"
        / "evo"
        / "configs"
        / "model-registry.json"
    )
    return installed_candidate


def default_cache_dir() -> Path:
    evo_cache = os.environ.get("EVO_CACHE_HOME")
    if evo_cache:
        return Path(evo_cache) / "huggingface" / "hub"
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return root / "huggingface" / "hub"


def load_huggingface() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise FetchError(
            "evo-fetch requires huggingface-hub==1.24.0; install "
            "requirements-fetch.txt from the source or installed share/evo directory"
        ) from error
    return HfApi, hf_hub_download


def read_cached_ref(cache_dir: Path, repo: str, revision: str) -> str | None:
    reference = cache_dir / f"models--{safe_repo_path(repo)}" / "refs" / revision
    try:
        value = reference.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value.lower() if COMMIT_RE.fullmatch(value) else None


def resolve_revision(
    repo: str,
    requested_revision: str,
    cache_dir: Path,
    endpoint: str | None,
    local_files_only: bool,
) -> str:
    requested_revision = validate_revision(requested_revision)
    if COMMIT_RE.fullmatch(requested_revision):
        return requested_revision.lower()
    if local_files_only:
        cached = read_cached_ref(cache_dir, repo, requested_revision)
        if cached is None:
            raise FetchError(
                f"no cached immutable revision for {repo}@{requested_revision}"
            )
        return cached
    HfApi, _ = load_huggingface()
    arguments: dict[str, Any] = {}
    if endpoint:
        arguments["endpoint"] = endpoint
    api = HfApi(**arguments)
    try:
        info = api.model_info(repo_id=repo, revision=requested_revision)
    except Exception as error:
        raise FetchError(
            f"cannot resolve {repo}@{requested_revision}: {error}"
        ) from error
    resolved = getattr(info, "sha", "")
    if not isinstance(resolved, str) or not COMMIT_RE.fullmatch(resolved):
        raise FetchError("Hugging Face returned a non-commit model revision")
    return resolved.lower()


def download_verified(
    *,
    repo: str,
    resolved_revision: str,
    name: str,
    expected_size: int,
    expected_sha256: str,
    cache_dir: Path,
    endpoint: str | None,
    local_files_only: bool,
) -> Path:
    _, hf_hub_download = load_huggingface()
    arguments: dict[str, Any] = {
        "repo_id": repo,
        "filename": name,
        "revision": resolved_revision,
        "cache_dir": str(cache_dir),
        "local_files_only": local_files_only,
    }
    if endpoint:
        arguments["endpoint"] = endpoint

    def acquire(force: bool) -> Path:
        if force:
            arguments["force_download"] = True
        try:
            return Path(hf_hub_download(**arguments)).resolve()
        except Exception as error:
            raise FetchError(
                f"cannot fetch {repo}@{resolved_revision}/{name}: {error}"
            ) from error

    path = acquire(False)
    for attempt in range(2):
        try:
            size = path.stat().st_size
        except OSError as error:
            raise FetchError(f"downloaded file is unavailable: {path}") from error
        actual_sha256 = sha256_file(path)
        if size == expected_size and actual_sha256 == expected_sha256:
            return path
        if local_files_only or attempt != 0:
            raise FetchError(
                f"{name} integrity mismatch: size={size}, sha256={actual_sha256}; "
                f"expected size={expected_size}, sha256={expected_sha256}"
            )
        print(
            f"evo-fetch: cached {name} failed verification; refreshing", file=sys.stderr
        )
        path = acquire(True)
    raise AssertionError("unreachable verification loop")


def validate_file_entry(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FetchError(f"{label} must be an object")
    name = validate_relative_path(value.get("path", value.get("name")), f"{label}.path")
    size = value.get("size")
    digest = value.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise FetchError(f"{label}.size must be a nonnegative integer")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise FetchError(f"{label}.sha256 must contain 64 hexadecimal characters")
    return {"name": name, "size": size, "sha256": digest.lower()}


def receipt_destination(cache_dir: Path, repo: str, revision: str, kind: str) -> Path:
    directory = cache_dir / "evo-receipts" / safe_repo_path(repo) / revision
    return directory / f"{kind}.json"


def write_receipt(cache_dir: Path, receipt: dict[str, object]) -> Path:
    destination = receipt_destination(
        cache_dir,
        str(receipt["repo"]),
        str(receipt["resolved_revision"]),
        str(receipt["kind"]),
    )
    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{destination.name}.", dir=directory, delete=False
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, destination)
    return destination


def fetch_files(
    *,
    kind: str,
    repo: str,
    requested_revision: str,
    resolved_revision: str,
    entries: list[dict[str, object]],
    cache_dir: Path,
    endpoint: str | None,
    local_files_only: bool,
    extra: dict[str, object],
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for entry in entries:
        name = str(entry["name"])
        path = download_verified(
            repo=repo,
            resolved_revision=resolved_revision,
            name=name,
            expected_size=int(entry["size"]),
            expected_sha256=str(entry["sha256"]),
            cache_dir=cache_dir,
            endpoint=endpoint,
            local_files_only=local_files_only,
        )
        files.append({**entry, "path": str(path)})
    receipt: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "repo": repo,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "files": files,
        **extra,
    }
    receipt_path = write_receipt(cache_dir, receipt)
    receipt["receipt"] = str(receipt_path.resolve())
    return receipt


def load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FetchError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise FetchError(f"{label} root must be an object")
    return value


def source_command(args: argparse.Namespace) -> dict[str, object]:
    registry = load_json(args.registry, "model registry")
    models = registry.get("models")
    entry = models.get(args.model) if isinstance(models, dict) else None
    if not isinstance(entry, dict):
        raise FetchError(f"unknown registered model ID: {args.model}")
    files_value = entry.get("checkpoint_files")
    if not isinstance(files_value, list) or not files_value:
        raise FetchError(
            f"{args.model} has no Hugging Face source checkpoint; use its documented provider"
        )
    repo = entry.get("source_repo")
    requested_revision = entry.get("source_revision")
    if not isinstance(repo, str) or not REPO_RE.fullmatch(repo):
        raise FetchError(f"{args.model} registry source_repo is invalid")
    if not isinstance(requested_revision, str) or not COMMIT_RE.fullmatch(
        requested_revision
    ):
        raise FetchError(f"{args.model} registry source_revision is not immutable")
    entries = [
        validate_file_entry(value, f"checkpoint_files[{index}]")
        for index, value in enumerate(files_value)
    ]
    resolved = resolve_revision(
        repo, requested_revision, args.cache_dir, args.endpoint, args.local_files_only
    )
    if resolved != requested_revision.lower():
        raise FetchError("registered source revision resolved to a different commit")
    return fetch_files(
        kind="source-checkpoint",
        repo=repo,
        requested_revision=requested_revision,
        resolved_revision=resolved,
        entries=entries,
        cache_dir=args.cache_dir,
        endpoint=args.endpoint,
        local_files_only=args.local_files_only,
        extra={"model_id": args.model, "load_path": None},
    )


def runtime_command(args: argparse.Namespace) -> dict[str, object]:
    repo, requested_revision = parse_repo_spec(args.repository)
    manifest_name = validate_relative_path(args.manifest, "runtime manifest path")
    resolved = resolve_revision(
        repo, requested_revision, args.cache_dir, args.endpoint, args.local_files_only
    )
    _, hf_hub_download = load_huggingface()
    arguments: dict[str, Any] = {
        "repo_id": repo,
        "filename": manifest_name,
        "revision": resolved,
        "cache_dir": str(args.cache_dir),
        "local_files_only": args.local_files_only,
        "force_download": not args.local_files_only,
    }
    if args.endpoint:
        arguments["endpoint"] = args.endpoint
    try:
        manifest_path = Path(hf_hub_download(**arguments)).resolve()
    except Exception as error:
        raise FetchError(
            f"cannot fetch runtime manifest {repo}@{resolved}/{manifest_name}: {error}"
        ) from error
    manifest_sha256 = sha256_file(manifest_path)
    if args.local_files_only:
        prior_path = receipt_destination(
            args.cache_dir, repo, resolved, "runtime-artifact"
        )
        prior = load_json(prior_path, "cached runtime receipt")
        if prior.get("manifest_sha256") != manifest_sha256:
            raise FetchError(
                "cached runtime manifest SHA256 differs from its verified receipt"
            )
    manifest = load_json(manifest_path, "runtime artifact manifest")
    if manifest.get("schema_version") != 1:
        raise FetchError("runtime artifact manifest schema_version must be 1")
    artifact_profile = manifest.get("artifact_profile")
    if artifact_profile not in RUNTIME_PROFILES:
        raise FetchError(
            "runtime artifact profile must be one of "
            + ", ".join(sorted(RUNTIME_PROFILES))
        )
    files_value = manifest.get("files")
    if not isinstance(files_value, list) or not files_value:
        raise FetchError("runtime artifact manifest files must be a nonempty array")
    entries = [
        validate_file_entry(value, f"files[{index}]")
        for index, value in enumerate(files_value)
    ]
    names = [str(entry["name"]) for entry in entries]
    if len(set(names)) != len(names):
        raise FetchError("runtime artifact manifest contains duplicate file paths")
    load_path = validate_relative_path(manifest.get("load_path"), "load_path")
    if load_path not in names:
        raise FetchError("runtime artifact load_path must name one listed file")
    if not load_path.endswith((".safetensors", ".safetensors.index.json")):
        raise FetchError("runtime artifact load_path must be Safetensors or its index")
    model_id = manifest.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise FetchError("runtime artifact model_id must be a nonempty string")
    receipt = fetch_files(
        kind="runtime-artifact",
        repo=repo,
        requested_revision=requested_revision,
        resolved_revision=resolved,
        entries=entries,
        cache_dir=args.cache_dir,
        endpoint=args.endpoint,
        local_files_only=args.local_files_only,
        extra={
            "model_id": model_id,
            "artifact_profile": artifact_profile,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
        },
    )
    paths = {
        PurePosixPath(str(entry["name"])): str(entry["path"])
        for entry in receipt["files"]
    }
    receipt["load_path"] = paths[PurePosixPath(load_path)]
    write_receipt(args.cache_dir, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch hash-verified Evo checkpoints through the Hugging Face cache"
    )
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--print-path", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser(
        "source", help="fetch a registry-pinned source checkpoint"
    )
    source.add_argument("model")
    source.add_argument("--registry", type=Path, default=default_registry())
    runtime = commands.add_parser("runtime", help="fetch a manifested runtime artifact")
    runtime.add_argument("repository", help="OWNER/NAME[@REVISION]")
    runtime.add_argument("--manifest", default="evo-artifact.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        args.cache_dir = args.cache_dir.expanduser().resolve()
        if args.command == "source":
            args.registry = args.registry.expanduser().resolve()
            receipt = source_command(args)
        else:
            receipt = runtime_command(args)
        if args.print_path:
            if receipt.get("load_path"):
                print(receipt["load_path"])
            else:
                for entry in receipt["files"]:
                    print(entry["path"])
        else:
            print(json.dumps(receipt, sort_keys=True))
        return 0
    except (FetchError, OSError, ValueError) as error:
        print(f"evo-fetch: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
