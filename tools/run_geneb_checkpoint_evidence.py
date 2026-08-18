#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create and bind strict canonical-checkpoint evidence for GENEB runtimes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if not (_SCRIPT_DIRECTORY / "evo").is_dir():
    _INSTALLED_PYTHON = _SCRIPT_DIRECTORY.parent / "share" / "evo" / "python"
    if _INSTALLED_PYTHON.is_dir():
        sys.path.insert(0, str(_INSTALLED_PYTHON))

from evo.geneb_artifact import catalog_contract_sha256


EVIDENCE_KIND = "geneb-canonical-checkpoint-evidence"
EVIDENCE_REF_SCHEMA = "geneb-canonical-checkpoint-evidence-v1"
ORACLE_KIND = "geneb-independent-oracle-vector"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
BACKENDS = ("cpu", "cuda", "mps")
MAX_JSON_SIZE = 64 * 1024 * 1024
MAX_SAFETENSORS_HEADER = 16 * 1024 * 1024
TOKENIZER_DESCRIPTOR_KEYS = {
    "converter.schema",
    "converter.version",
    "compiler_manifest_sha256",
    "source_receipt_contract_sha256",
    "tokenizer.profile",
    "tokenizer.path",
    "tokenizer.sha256",
    "tokenizer.size",
}
THREAD_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}


class EvidenceError(RuntimeError):
    """Raised when evidence cannot be produced or verified."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(16 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise EvidenceError("cannot hash %s: %s" % (path, error))
    return digest.hexdigest()


def load_json(path: Path, label: str) -> Tuple[Dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EvidenceError("cannot read %s %s: %s" % (label, path, error))
    if not payload or len(payload) > MAX_JSON_SIZE:
        raise EvidenceError("%s must be nonempty and at most 64 MiB" % label)

    def reject_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result = {}  # type: Dict[str, Any]
        for key, value in pairs:
            if key in result:
                raise EvidenceError("%s contains duplicate key %r" % (label, key))
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except EvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError("cannot parse %s: %s" % (label, error))
    if not isinstance(value, dict):
        raise EvidenceError("%s root must be an object" % label)
    return value, payload


def exact_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str],
    label: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    actual = set(value)
    if not required_set <= actual or not actual <= allowed:
        raise EvidenceError(
            "%s fields differ: missing=%s extra=%s"
            % (label, sorted(required_set - actual), sorted(actual - allowed))
        )


def nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EvidenceError("%s must be a nonempty string" % label)
    return value


def digest_text(value: Any, label: str) -> str:
    text = nonempty_text(value, label)
    if not SHA256_RE.fullmatch(text):
        raise EvidenceError("%s must be a lowercase SHA256" % label)
    return text


def uint(value: Any, label: str, allow_zero: bool = True) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError("%s must be an unsigned integer" % label)
    return value


def normalized_relative(value: Any, label: str) -> str:
    text = nonempty_text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or str(path) != text or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise EvidenceError("%s must be a normalized relative path" % label)
    return text


def reject_local_absolute_paths(value: Any, label: str) -> None:
    """Reject non-portable local paths while preserving remote provenance URLs."""
    if isinstance(value, dict):
        for key, item in value.items():
            reject_local_absolute_paths(key, label + " key")
            reject_local_absolute_paths(item, "%s.%s" % (label, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            reject_local_absolute_paths(item, "%s[%d]" % (label, index))
        return
    if not isinstance(value, str):
        return
    parsed = urlsplit(value)
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or parsed.scheme.lower() == "file"
    ):
        raise EvidenceError("%s contains a local absolute filesystem path" % label)


def regular_file(path: Path, label: str, reject_symlink: bool = True) -> Path:
    if reject_symlink and path.is_symlink():
        raise EvidenceError("%s must not be a symlink" % label)
    try:
        if not path.is_file():
            raise EvidenceError("%s must be a regular file" % label)
    except OSError as error:
        raise EvidenceError("cannot stat %s: %s" % (label, error))
    return path


def strict_relative_file(root: Path, relative: str, label: str) -> Path:
    """Resolve a normalized relative file without traversing symlink components."""
    normalized_relative(relative, label + " path")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root
    for part in PurePosixPath(relative).parts:
        candidate /= part
        if candidate.is_symlink():
            raise EvidenceError("%s path contains a symlink" % label)
    regular_file(candidate, label)
    try:
        candidate.resolve(strict=True).relative_to(resolved_root)
    except ValueError:
        raise EvidenceError("%s path escapes its root" % label)
    return candidate


def write_atomic(path: Path, payload: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise EvidenceError("output already exists: %s" % path)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=".%s." % path.name, dir=str(path.parent), delete=False
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(str(temporary_path), str(path))
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def catalog_model(
    catalog: Mapping[str, Any], runtime_id: str
) -> Mapping[str, Any]:
    models = catalog.get("models")
    if catalog.get("schema_version") != 1 or not isinstance(models, list):
        raise EvidenceError("GENEB catalog must be schema v1 with a models array")
    matches = [
        item
        for item in models
        if isinstance(item, dict) and item.get("runtime_id") == runtime_id
    ]
    if len(matches) != 1:
        raise EvidenceError("catalog runtime_id is missing or duplicated")
    return matches[0]


def validate_receipt_files(raw: Any) -> Tuple[List[Dict[str, Any]], str]:
    if not isinstance(raw, list) or not raw:
        raise EvidenceError("source receipt files must be a nonempty array")
    files = []  # type: List[Dict[str, Any]]
    names = set()
    for index, item in enumerate(raw):
        label = "source receipt files[%d]" % index
        if not isinstance(item, dict):
            raise EvidenceError("%s must be an object" % label)
        exact_keys(item, ["name", "size", "sha256", "path"], [], label)
        name = normalized_relative(item["name"], label + ".name")
        if name in names:
            raise EvidenceError("source receipt file names must be unique")
        names.add(name)
        size = uint(item["size"], label + ".size")
        digest = digest_text(item["sha256"], label + ".sha256")
        source_path = regular_file(
            Path(nonempty_text(item["path"], label + ".path")), label + ".path"
        )
        try:
            actual_size = source_path.stat().st_size
        except OSError as error:
            raise EvidenceError("cannot stat %s: %s" % (label, error))
        if actual_size != size or sha256_file(source_path) != digest:
            raise EvidenceError("%s size/SHA256 differs" % label)
        files.append({"name": name, "size": size, "sha256": digest})
    files.sort(key=lambda value: value["name"])
    return files, sha256_bytes(canonical_json(files))


def validate_source_receipt(
    path: Path,
    model: Mapping[str, Any],
    catalog_path: Path,
    catalog_contract_digest: str,
) -> Dict[str, Any]:
    regular_file(path, "source receipt")
    receipt, payload = load_json(path, "source receipt")
    source = model.get("source")
    if not isinstance(source, dict):
        raise EvidenceError("catalog source must be an object")
    kind = receipt.get("kind")
    source_kind = receipt.get("source_kind")
    if kind == "source-checkpoint" and source_kind == "huggingface":
        exact_keys(
            receipt,
            [
                "schema_version",
                "kind",
                "model_id",
                "repo",
                "requested_revision",
                "resolved_revision",
                "files",
                "source_kind",
                "catalog_path",
                "catalog_contract_sha256",
                "load_path",
            ],
            [],
            "source receipt",
        )
        if (
            receipt["schema_version"] != 1
            or receipt["model_id"] != model.get("runtime_id")
            or receipt["repo"] != source.get("repo")
            or receipt["requested_revision"] != source.get("requested_revision")
            or receipt["resolved_revision"] != source.get("revision")
            or receipt["source_kind"] != "huggingface"
            or receipt["catalog_path"] != str(catalog_path.resolve())
            or receipt["catalog_contract_sha256"] != catalog_contract_digest
            or receipt["load_path"] is not None
        ):
            raise EvidenceError("Hugging Face source receipt provenance differs")
        provider = {
            "kind": "huggingface",
            "repo": receipt["repo"],
            "requested_revision": receipt["requested_revision"],
            "resolved_revision": receipt["resolved_revision"],
        }
    elif kind == "geneb-janusdna-manual-source" or (
        kind == "source-checkpoint" and source_kind != "huggingface"
    ):
        exact_keys(
            receipt,
            [
                "schema_version",
                "kind",
                "model_id",
                "source_kind",
                "source_url",
                "files",
            ],
            [],
            "manual source receipt",
        )
        if (
            receipt["schema_version"] != 1
            or receipt["model_id"] != model.get("runtime_id")
            or receipt["source_kind"] != source.get("kind")
            or receipt["source_url"] != source.get("url")
            or source.get("immutable") is not False
            or source.get("repo") is not None
            or source.get("revision") is not None
        ):
            raise EvidenceError("manual source receipt provenance differs")
        provider = {
            "kind": receipt["source_kind"],
            "url": receipt["source_url"],
        }
    else:
        raise EvidenceError("source receipt kind is unsupported by evidence schema")
    files, files_sha256 = validate_receipt_files(receipt["files"])
    if kind == "geneb-janusdna-manual-source" or source_kind != "huggingface":
        required = source.get("required_files")
        if not isinstance(required, list):
            raise EvidenceError("manual source catalog required_files is invalid")
        expected = []  # type: List[Tuple[str, Optional[int], Optional[str]]]
        expected_names = set()
        for index, item in enumerate(required):
            if not isinstance(item, dict):
                raise EvidenceError("manual source required file must be an object")
            exact_keys(
                item,
                ["path", "size", "sha256"],
                [],
                "manual source required_files[%d]" % index,
            )
            full_name = normalized_relative(
                item["path"], "manual source required file path"
            )
            name = PurePosixPath(full_name).name
            if not name or name in expected_names:
                raise EvidenceError(
                    "manual source required file basenames must be unique"
                )
            expected_names.add(name)
            size = (
                None
                if item["size"] is None
                else uint(item["size"], "manual source required file size", False)
            )
            digest = (
                None
                if item["sha256"] is None
                else digest_text(
                    item["sha256"], "manual source required file SHA256"
                )
            )
            expected.append((name, size, digest))
        actual = {item["name"]: item for item in files}
        if set(actual) != expected_names:
            raise EvidenceError("manual source receipt file set differs from catalog")
        for name, expected_size, expected_digest in expected:
            if expected_size is not None and actual[name]["size"] != expected_size:
                raise EvidenceError("manual source receipt file size differs from catalog")
            if expected_digest is not None and actual[name]["sha256"] != expected_digest:
                raise EvidenceError("manual source receipt file SHA256 differs from catalog")
    identity = {
        "model_id": model.get("runtime_id"),
        "provider": provider,
        "files": files,
    }
    return {
        "kind": kind,
        "receipt_sha256": sha256_bytes(payload),
        "snapshot_sha256": sha256_bytes(canonical_json(identity)),
        "file_manifest_sha256": files_sha256,
        "file_count": len(files),
        "files": files,
        "provider": provider,
    }


def validate_tokenizer_descriptor(
    path: Path, tokenizer_root: Optional[Path], artifact_parent: Path
) -> Dict[str, Any]:
    regular_file(path, "tokenizer descriptor")
    descriptor, payload = load_json(path, "tokenizer descriptor")
    exact_keys(descriptor, TOKENIZER_DESCRIPTOR_KEYS, [], "tokenizer descriptor")
    if (
        descriptor["converter.schema"] != "evo-tokenizer-conversion-receipt"
        or descriptor["converter.version"] != 1
        or descriptor["tokenizer.profile"] != "evo-tokenizer-v1"
    ):
        raise EvidenceError("tokenizer descriptor schema/profile differs")
    for key in (
        "compiler_manifest_sha256",
        "source_receipt_contract_sha256",
        "tokenizer.sha256",
    ):
        digest_text(descriptor[key], "tokenizer descriptor " + key)
    relative = normalized_relative(descriptor["tokenizer.path"], "tokenizer.path")
    size = uint(descriptor["tokenizer.size"], "tokenizer.size", False)
    root = (tokenizer_root if tokenizer_root is not None else path.parent).resolve()
    asset = root / relative
    component = root
    for part in PurePosixPath(relative).parts:
        component /= part
        if component.is_symlink():
            raise EvidenceError("tokenizer asset path contains a symlink")
    regular_file(asset, "tokenizer asset")
    resolved = asset.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise EvidenceError("tokenizer asset escapes tokenizer root")
    expected = artifact_parent.resolve() / relative
    if resolved != expected.resolve():
        raise EvidenceError(
            "tokenizer asset must live under artifact root at tokenizer.path"
        )
    if asset.stat().st_size != size or sha256_file(asset) != descriptor["tokenizer.sha256"]:
        raise EvidenceError("tokenizer asset size/SHA256 differs")
    return {
        "descriptor_sha256": sha256_bytes(payload),
        "compiler_manifest_sha256": descriptor["compiler_manifest_sha256"],
        "source_receipt_contract_sha256": descriptor[
            "source_receipt_contract_sha256"
        ],
        "profile": descriptor["tokenizer.profile"],
        "path": relative,
        "size": size,
        "sha256": descriptor["tokenizer.sha256"],
    }


def validate_input_transform_tokenizer(model: Mapping[str, Any]) -> Dict[str, Any]:
    tokenizer = model.get("tokenizer")
    transform = model.get("input_transform")
    if not isinstance(tokenizer, dict) or not isinstance(transform, dict):
        raise EvidenceError("catalog input-transform tokenizer contract is invalid")
    if (
        tokenizer.get("asset_source") != "input-transform"
        or tokenizer.get("kind") != "single-nucleotide"
        or tokenizer.get("assets") is not None
    ):
        raise EvidenceError("catalog input-transform tokenizer shape differs")
    contract = {"tokenizer": tokenizer, "input_transform": transform}
    return {
        "kind": "input-transform",
        "contract_sha256": sha256_bytes(canonical_json(contract)),
        "vocabulary_size": 4,
    }


def parse_safetensors_metadata(path: Path) -> Tuple[Mapping[str, Any], str]:
    try:
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise EvidenceError("artifact Safetensors prefix is truncated")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size == 0 or header_size > MAX_SAFETENSORS_HEADER:
                raise EvidenceError("artifact Safetensors header size is invalid")
            header = source.read(header_size)
            if len(header) != header_size:
                raise EvidenceError("artifact Safetensors header is truncated")
    except OSError as error:
        raise EvidenceError("cannot read artifact header: %s" % error)

    def reject_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result = {}  # type: Dict[str, Any]
        for key, value in pairs:
            if key in result:
                raise EvidenceError("artifact header contains duplicate key %r" % key)
            result[key] = value
        return result

    try:
        root = json.loads(header.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except EvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError("artifact Safetensors header is invalid: %s" % error)
    if not isinstance(root, dict) or not isinstance(root.get("__metadata__"), dict):
        raise EvidenceError("artifact Safetensors metadata is missing")
    return root["__metadata__"], sha256_bytes(header)


def typed_metadata(metadata: Mapping[str, Any], key: str, prefix: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.startswith(prefix):
        raise EvidenceError("artifact metadata %s has the wrong type" % key)
    return value[len(prefix) :]


def validate_artifact(
    path: Path,
    runtime_id: str,
    catalog_contract_digest: str,
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
) -> Dict[str, Any]:
    regular_file(path, "native artifact")
    metadata, header_sha256 = parse_safetensors_metadata(path)
    profile = typed_metadata(metadata, "runtime.profile", "s:")
    abi = typed_metadata(metadata, "runtime.abi", "s:")
    architecture = typed_metadata(metadata, "model.architecture", "s:")
    if "source.converter_manifest_sha256" in metadata:
        raise EvidenceError("artifact uses legacy whole-profile converter digest")
    converter_profile_contract = typed_metadata(
        metadata, "source.converter_profile_contract_sha256", "s:"
    )
    digest_text(
        converter_profile_contract, "artifact converter profile contract SHA256"
    )
    if typed_metadata(metadata, "model.id", "s:") != runtime_id:
        raise EvidenceError("artifact model.id differs from catalog")
    if typed_metadata(metadata, "source.receipt_sha256", "s:") != source["receipt_sha256"]:
        raise EvidenceError("artifact source receipt digest differs")
    if (
        typed_metadata(metadata, "source.catalog_contract_sha256", "s:")
        != catalog_contract_digest
    ):
        raise EvidenceError("artifact source catalog contract digest differs")
    if (
        typed_metadata(metadata, "geneb.catalog_contract_sha256", "s:")
        != catalog_contract_digest
    ):
        raise EvidenceError("artifact GENEB catalog contract digest differs")
    if tokenizer.get("kind") == "input-transform":
        forbidden = [
            key
            for key in metadata
            if key == "source.tokenizer_descriptor_sha256"
            or key.startswith("tokenizer.")
        ]
        if forbidden:
            raise EvidenceError("input-transform artifact contains tokenizer assets")
        if (
            typed_metadata(metadata, "runtime.tokenizer_vocabulary_size", "u:")
            != str(tokenizer["vocabulary_size"])
        ):
            raise EvidenceError("input-transform artifact vocabulary size differs")
    else:
        descriptor_key = "source.tokenizer_descriptor_sha256"
        if (
            typed_metadata(metadata, descriptor_key, "s:")
            != tokenizer["descriptor_sha256"]
        ):
            raise EvidenceError("artifact tokenizer descriptor digest differs")
        for key, expected, prefix in (
            ("tokenizer.profile", tokenizer["profile"], "s:"),
            ("tokenizer.path", tokenizer["path"], "s:"),
            ("tokenizer.sha256", tokenizer["sha256"], "s:"),
            ("tokenizer.size", str(tokenizer["size"]), "u:"),
        ):
            if typed_metadata(metadata, key, prefix) != expected:
                raise EvidenceError("artifact %s differs from descriptor" % key)
    return {
        "profile": profile,
        "abi": abi,
        "architecture": architecture,
        "converter_profile_contract_sha256": converter_profile_contract,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "header_sha256": header_sha256,
    }


def normalized_process_text(
    text: str, replacements: Sequence[str], successful: bool
) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for value in sorted(set(replacements), key=lambda item: (-len(item), item)):
        normalized = normalized.replace(value, "<PATH>")
    if not successful:
        return normalized

    lines = []  # type: List[str]
    for line in normalized.splitlines():
        if not line.startswith("evo_metrics "):
            lines.append(line)
            continue

        def reject_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
            value = {}  # type: Dict[str, Any]
            for key, item in pairs:
                if key in value:
                    raise EvidenceError("successful evo_metrics contains duplicate key")
                value[key] = item
            return value

        try:
            telemetry = json.loads(
                line[len("evo_metrics ") :], object_pairs_hook=reject_duplicates
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise EvidenceError("successful evo_metrics is invalid JSON: %s" % error)
        if not isinstance(telemetry, dict):
            raise EvidenceError("successful evo_metrics must be an object")
        stable = {
            key: value
            for key, value in telemetry.items()
            if not (
                key.endswith("_seconds")
                or key.endswith("_per_second")
                or key.endswith("_milliseconds")
                or key.endswith("_nanoseconds")
            )
        }
        lines.append(
            "evo_metrics "
            + json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        )
    suffix = "\n" if normalized.endswith("\n") else ""
    return "\n".join(lines) + suffix


def normalized_process_digest(
    text: str, replacements: Sequence[str], successful: bool
) -> str:
    return sha256_bytes(
        normalized_process_text(text, replacements, successful).encode("utf-8")
    )


def normalized_failure_summary(text: str, replacements: Sequence[str]) -> str:
    summary = normalized_process_text(text, replacements, False).strip()
    if not summary:
        return "process failed without stderr"
    maximum = 8192
    if len(summary) > maximum:
        digest = sha256_bytes(summary.encode("utf-8"))
        summary = summary[:maximum] + "\n<truncated sha256=%s>" % digest
    return summary


def execution_environment(
    native: Path,
    converter: Path,
    converter_python: Path,
    profiles: Path,
    endpoint: Optional[str],
) -> Dict[str, Any]:
    try:
        version = subprocess.run(
            [str(converter_python), "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise EvidenceError("cannot identify converter Python: %s" % error)
    value = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": version,
        "native_sha256": sha256_file(native),
        "converter_sha256": sha256_file(converter),
        "profiles_sha256": sha256_file(profiles),
        "source_endpoint": endpoint,
        "thread_environment": dict(sorted(THREAD_ENVIRONMENT.items())),
    }
    return {"value": value, "sha256": sha256_bytes(canonical_json(value))}


def stable_environment() -> Dict[str, str]:
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    return environment


def validate_endpoint(value: Optional[str], source_kind: Any) -> Optional[str]:
    if source_kind == "huggingface":
        if value is None:
            raise EvidenceError(
                "Hugging Face evidence requires explicit --source-endpoint; "
                "do not inherit HF_ENDPOINT"
            )
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise EvidenceError("source endpoint must be an HTTPS origin")
        return "%s://%s" % (parsed.scheme, parsed.netloc)
    if value is not None:
        raise EvidenceError("--source-endpoint is only valid for Hugging Face sources")
    return None


def run_converter(
    args: argparse.Namespace,
    replacements: Sequence[str],
    catalog_contract_digest: str,
) -> Dict[str, Any]:
    if args.reuse_artifact:
        if not args.artifact.is_file():
            raise EvidenceError("--reuse-artifact requires an existing artifact")
        return {
            "status": "validated-existing",
            "command_sha256": sha256_bytes(
                canonical_json(
                    {
                        "converter_sha256": sha256_file(args.converter),
                        "profiles_sha256": sha256_file(args.profiles),
                        "catalog_contract_sha256": catalog_contract_digest,
                        "mode": "validate-existing",
                    }
                )
            ),
            "stdout_sha256": sha256_bytes(b""),
            "stderr_sha256": sha256_bytes(b""),
        }
    if args.artifact.exists():
        raise EvidenceError("artifact output already exists")
    command = [
        str(args.converter_python),
        str(args.converter),
        "--receipt",
        str(args.source_receipt),
        "--catalog",
        str(args.catalog),
        "--profiles",
        str(args.profiles),
        "--output",
        str(args.artifact),
    ]
    if args.tokenizer_descriptor is not None:
        command.extend(
            [
                "--tokenizer-descriptor",
                str(args.tokenizer_descriptor),
                "--tokenizer-root",
                str(args.tokenizer_root),
            ]
        )
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=stable_environment(),
        )
    except OSError as error:
        raise EvidenceError("cannot execute family converter: %s" % error)
    if result.returncode != 0:
        raise EvidenceError(
            "family converter failed with exit %d: %s"
            % (result.returncode, result.stderr.strip())
        )
    semantic = {
        "converter_sha256": sha256_file(args.converter),
        "profiles_sha256": sha256_file(args.profiles),
        "catalog_contract_sha256": catalog_contract_digest,
        "receipt_sha256": sha256_file(args.source_receipt),
    }
    if args.tokenizer_descriptor is not None:
        semantic["tokenizer_descriptor_sha256"] = sha256_file(
            args.tokenizer_descriptor
        )
    return {
        "status": "passed",
        "command_sha256": sha256_bytes(canonical_json(semantic)),
        "stdout_sha256": normalized_process_digest(result.stdout, replacements, True),
        "stderr_sha256": normalized_process_digest(result.stderr, replacements, True),
    }


def read_npy_f32(path: Path) -> Tuple[List[int], List[float]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EvidenceError("cannot read embedding NPY: %s" % error)
    if len(payload) < 10 or payload[:6] != b"\x93NUMPY":
        raise EvidenceError("embedding output is not NPY")
    major = payload[6]
    if major == 1:
        if len(payload) < 10:
            raise EvidenceError("embedding NPY header is truncated")
        header_size = struct.unpack("<H", payload[8:10])[0]
        offset = 10
    elif major in (2, 3):
        if len(payload) < 12:
            raise EvidenceError("embedding NPY header is truncated")
        header_size = struct.unpack("<I", payload[8:12])[0]
        offset = 12
    else:
        raise EvidenceError("embedding NPY version is unsupported")
    end = offset + header_size
    if end > len(payload):
        raise EvidenceError("embedding NPY header exceeds file")
    try:
        header = ast.literal_eval(payload[offset:end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeError) as error:
        raise EvidenceError("embedding NPY header is invalid: %s" % error)
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise EvidenceError("embedding NPY header fields differ")
    if header["descr"] not in ("<f4", "=f4") or header["fortran_order"] is not False:
        raise EvidenceError("embedding NPY must be row-major little-endian F32")
    shape_value = header["shape"]
    if (
        not isinstance(shape_value, tuple)
        or not shape_value
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape_value)
    ):
        raise EvidenceError("embedding NPY shape is invalid")
    count = math.prod(shape_value)
    data = payload[end:]
    if len(data) != count * 4:
        raise EvidenceError("embedding NPY payload size differs")
    values = list(struct.unpack("<%df" % count, data))
    if any(not math.isfinite(value) for value in values):
        raise EvidenceError("embedding NPY contains non-finite values")
    return list(shape_value), values


def output_manifest(
    directory: Path,
    backend: str,
    profile: str,
    runtime_id: str,
    input_sha256: str,
) -> Tuple[Dict[str, Any], List[float]]:
    manifest_path = regular_file(directory / "embeddings.jsonl", "embedding manifest")
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvidenceError("cannot read embedding manifest: %s" % error)
    if len(lines) != 1:
        raise EvidenceError("canonical evidence input must produce exactly one record")
    try:
        def reject_duplicates(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
            result = {}  # type: Dict[str, Any]
            for key, value in pairs:
                if key in result:
                    raise EvidenceError("embedding manifest contains duplicate key")
                result[key] = value
            return result

        record = json.loads(lines[0], object_pairs_hook=reject_duplicates)
    except EvidenceError:
        raise
    except json.JSONDecodeError as error:
        raise EvidenceError("embedding manifest JSON is invalid: %s" % error)
    if not isinstance(record, dict):
        raise EvidenceError("embedding manifest row must be an object")
    record_name = nonempty_text(record.get("name"), "embedding manifest name")
    input_format = nonempty_text(
        record.get("input_format"), "embedding manifest input_format"
    )
    if (
        record.get("backend") != backend
        or record.get("profile") != profile
        or record.get("model_id") != runtime_id
        or record.get("dtype") != "float32"
        or record.get("record_index") != 0
    ):
        raise EvidenceError("embedding manifest backend/profile/model differs")
    relative = normalized_relative(record.get("file"), "embedding file")
    npy_path = regular_file(directory / relative, "embedding NPY")
    shape, values = read_npy_f32(npy_path)
    if record.get("shape") != shape:
        raise EvidenceError("embedding manifest/NPY shape differs")
    portable_record = dict(record)
    if (
        input_format == "raw"
        or PurePosixPath(record_name).is_absolute()
        or PureWindowsPath(record_name).is_absolute()
    ):
        source_label = "content-sha256:%s" % input_sha256
        portable_record["name"] = source_label
    else:
        source_label = record_name
    portable_manifest = canonical_json(portable_record)
    portable_manifest_sha256 = sha256_bytes(portable_manifest)
    files = []
    for file_path in sorted(path for path in directory.iterdir() if path.is_file()):
        relative_path = file_path.relative_to(directory).as_posix()
        if relative_path == "embeddings.jsonl":
            file_size = len(portable_manifest)
            file_sha256 = portable_manifest_sha256
        else:
            file_size = file_path.stat().st_size
            file_sha256 = sha256_file(file_path)
        files.append(
            {
                "path": relative_path,
                "size": file_size,
                "sha256": file_sha256,
            }
        )
    if {item["path"] for item in files} != {"embeddings.jsonl", relative}:
        raise EvidenceError("embedding output contains unexpected files")
    summary = {
        "manifest_sha256": portable_manifest_sha256,
        "output_sha256": sha256_bytes(canonical_json(files)),
        "files": files,
        "source_label": source_label,
        "shape": shape,
        "source_tokens": uint(record.get("source_tokens"), "source_tokens", False),
        "layer": uint(record.get("layer"), "embedding layer"),
        "point": nonempty_text(record.get("point"), "embedding point"),
        "pooling": nonempty_text(record.get("pooling"), "embedding pooling"),
        "preset": nonempty_text(record.get("preset"), "embedding preset"),
    }
    return summary, values


def run_backend(
    args: argparse.Namespace,
    backend: str,
    runtime_id: str,
    replacements: Sequence[str],
) -> Tuple[Dict[str, Any], Optional[List[float]]]:
    directory = args.embedding_dir / backend
    if directory.exists():
        try:
            if not directory.is_dir() or any(directory.iterdir()):
                raise EvidenceError("backend output directory must be absent or empty")
        except OSError as error:
            raise EvidenceError("cannot inspect backend output directory: %s" % error)
    command = [
        str(args.native),
        "embed",
        "-m",
        str(args.artifact),
        "--input",
        str(args.input),
        "--output",
        str(directory),
        "--preset",
        args.preset,
        "--ctx",
        str(args.context),
        "--backend",
        backend,
        "--profile",
        args.profile,
    ]
    semantic = {
        "native_sha256": sha256_file(args.native),
        "artifact_sha256": sha256_file(args.artifact),
        "input_sha256": sha256_file(args.input),
        "backend": backend,
        "profile": args.profile,
        "preset": args.preset,
        "context": args.context,
    }
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=stable_environment(),
        )
    except OSError as error:
        raise EvidenceError("cannot execute native backend: %s" % error)
    common = {
        "backend": backend,
        "profile": args.profile,
        "command_sha256": sha256_bytes(canonical_json(semantic)),
        "stdout_sha256": normalized_process_digest(
            result.stdout, replacements, result.returncode == 0
        ),
        "stderr_sha256": normalized_process_digest(
            result.stderr, replacements, result.returncode == 0
        ),
        "exit_code": result.returncode,
    }
    if result.returncode != 0:
        common.update(
            {
                "status": "failed",
                "reason": normalized_failure_summary(result.stderr, replacements),
            }
        )
        return common, None
    summary, values = output_manifest(
        directory, backend, args.profile, runtime_id, sha256_file(args.input)
    )
    common.update({"status": "passed", "output": summary})
    return common, values


def compare_oracle(
    args: argparse.Namespace,
    input_sha256: str,
    backend_values: Mapping[str, Optional[List[float]]],
) -> Dict[str, Any]:
    if args.oracle_vector is None:
        return {"status": "not-run", "reason": "no independent oracle supplied"}
    for name in ("max_abs", "mean_abs", "cosine"):
        if getattr(args, name) is None:
            raise EvidenceError("oracle comparison requires --%s" % name.replace("_", "-"))
    oracle, payload = load_json(args.oracle_vector, "oracle vector")
    exact_keys(
        oracle,
        [
            "schema_version",
            "kind",
            "runtime_id",
            "input_sha256",
            "backend",
            "profile",
            "values",
            "environment_lock",
            "provenance",
        ],
        [],
        "oracle vector",
    )
    backend = nonempty_text(oracle["backend"], "oracle backend")
    if (
        oracle["schema_version"] != 1
        or oracle["kind"] != ORACLE_KIND
        or oracle["runtime_id"] != args.model
        or oracle["input_sha256"] != input_sha256
        or oracle["profile"] != args.profile
        or backend not in backend_values
        or backend_values[backend] is None
        or not isinstance(oracle["environment_lock"], dict)
        or not isinstance(oracle["provenance"], dict)
    ):
        raise EvidenceError("oracle identity/environment differs")
    reject_local_absolute_paths(oracle["environment_lock"], "oracle environment_lock")
    reject_local_absolute_paths(oracle["provenance"], "oracle provenance")
    raw_values = oracle["values"]
    if not isinstance(raw_values, list) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw_values
    ):
        raise EvidenceError("oracle values must be finite numbers")
    expected = [float(value) for value in raw_values]
    actual = backend_values[backend]
    assert actual is not None
    if len(expected) != len(actual) or not expected:
        raise EvidenceError("oracle/native output shapes differ")
    differences = [abs(left - right) for left, right in zip(actual, expected)]
    maximum = max(differences)
    mean = sum(differences) / len(differences)
    dot = sum(left * right for left, right in zip(actual, expected))
    actual_norm = math.sqrt(sum(value * value for value in actual))
    expected_norm = math.sqrt(sum(value * value for value in expected))
    if actual == expected:
        cosine = 1.0
    elif actual_norm and expected_norm:
        cosine = dot / (actual_norm * expected_norm)
    else:
        cosine = 0.0
    tolerances = {
        "max_abs": float(args.max_abs),
        "mean_abs": float(args.mean_abs),
        "cosine": float(args.cosine),
    }
    if (
        tolerances["max_abs"] < 0
        or tolerances["mean_abs"] < 0
        or not -1.0 <= tolerances["cosine"] <= 1.0
    ):
        raise EvidenceError("oracle tolerances are outside their valid ranges")
    passed = (
        maximum <= tolerances["max_abs"]
        and mean <= tolerances["mean_abs"]
        and cosine >= tolerances["cosine"]
    )
    return {
        "status": "passed" if passed else "failed",
        "oracle_sha256": sha256_bytes(payload),
        "backend": backend,
        "profile": args.profile,
        "environment_lock": oracle["environment_lock"],
        "provenance": oracle["provenance"],
        "tolerances": tolerances,
        "metrics": {"max_abs": maximum, "mean_abs": mean, "cosine": cosine},
    }


def run_command(args: argparse.Namespace) -> int:
    args.catalog = args.catalog.expanduser().resolve(strict=True)
    args.source_receipt = args.source_receipt.expanduser().resolve(strict=True)
    args.tokenizer_descriptor = (
        args.tokenizer_descriptor.expanduser().resolve(strict=True)
        if args.tokenizer_descriptor is not None
        else None
    )
    args.converter = args.converter.expanduser().resolve(strict=True)
    args.converter_python = Path(
        os.path.abspath(str(args.converter_python.expanduser()))
    )
    args.profiles = args.profiles.expanduser().resolve(strict=True)
    args.native = args.native.expanduser().resolve(strict=True)
    args.input = args.input.expanduser().resolve(strict=True)
    args.artifact = args.artifact.expanduser().resolve()
    args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.evidence = args.evidence.expanduser().resolve()
    args.tokenizer_root = (
        args.tokenizer_root.expanduser().resolve()
        if args.tokenizer_root is not None
        else (
            args.tokenizer_descriptor.parent
            if args.tokenizer_descriptor is not None
            else None
        )
    )
    checked_paths = [
        (args.converter, "family converter", True),
        (args.converter_python, "converter Python", False),
        (args.profiles, "converter profiles", True),
        (args.native, "native runtime", False),
        (args.input, "canonical input", True),
    ]
    if args.tokenizer_descriptor is not None:
        checked_paths.append((args.tokenizer_descriptor, "tokenizer descriptor", True))
    for path, label, reject_symlink in checked_paths:
        regular_file(path, label, reject_symlink)
    if not os.access(str(args.native), os.X_OK):
        raise EvidenceError("native runtime must be executable")
    if not os.access(str(args.converter_python), os.X_OK):
        raise EvidenceError("converter Python must be executable")
    if args.context <= 0:
        raise EvidenceError("--ctx must be positive")
    requested = args.backend or ["cpu"]
    if len(requested) != len(set(requested)):
        raise EvidenceError("backend requests must be unique")

    catalog, catalog_payload = load_json(args.catalog, "GENEB catalog")
    model = catalog_model(catalog, args.model)
    source = model.get("source")
    if not isinstance(source, dict):
        raise EvidenceError("catalog source is invalid")
    endpoint = validate_endpoint(args.source_endpoint, source.get("kind"))
    catalog_contract_digest = catalog_contract_sha256(catalog, model)
    receipt = validate_source_receipt(
        args.source_receipt, model, args.catalog, catalog_contract_digest
    )
    tokenizer_spec = model.get("tokenizer")
    if not isinstance(tokenizer_spec, dict):
        raise EvidenceError("catalog tokenizer must be an object")
    if tokenizer_spec.get("asset_source") == "input-transform":
        if args.tokenizer_descriptor is not None or args.tokenizer_root is not None:
            raise EvidenceError(
                "input-transform model must not provide tokenizer descriptor/root"
            )
        tokenizer = validate_input_transform_tokenizer(model)
    else:
        if args.tokenizer_descriptor is None or args.tokenizer_root is None:
            raise EvidenceError("artifact tokenizer descriptor is required")
        tokenizer = validate_tokenizer_descriptor(
            args.tokenizer_descriptor, args.tokenizer_root, args.artifact.parent
        )
    input_size = args.input.stat().st_size
    input_sha256 = sha256_file(args.input)
    replacements = [
        str(args.catalog),
        str(args.source_receipt),
        str(args.converter),
        str(args.profiles),
        str(args.native),
        str(args.input),
        str(args.artifact),
        str(args.embedding_dir),
        str(args.evidence),
    ]
    if args.tokenizer_descriptor is not None:
        replacements.extend(
            [str(args.tokenizer_descriptor), str(args.tokenizer_root)]
        )
    converter = run_converter(args, replacements, catalog_contract_digest)
    artifact = validate_artifact(
        args.artifact, args.model, catalog_contract_digest, receipt, tokenizer
    )
    environment = execution_environment(
        args.native, args.converter, args.converter_python, args.profiles, endpoint
    )
    backend_results = {
        backend: {"status": "not-run", "reason": "not requested"}
        for backend in BACKENDS
    }  # type: Dict[str, Any]
    backend_values = {}  # type: Dict[str, Optional[List[float]]]
    failed = False
    for backend in requested:
        result, values = run_backend(args, backend, args.model, replacements)
        backend_results[backend] = result
        backend_values[backend] = values
        failed = failed or result["status"] != "passed"
    oracle = compare_oracle(args, input_sha256, backend_values)
    failed = failed or oracle["status"] == "failed"
    evidence = {
        "schema_version": 1,
        "kind": EVIDENCE_KIND,
        "runtime_id": args.model,
        "catalog": {"contract_sha256": catalog_contract_digest},
        "source": dict(receipt, endpoint=endpoint),
        "tokenizer": tokenizer,
        "converter": converter,
        "artifact": artifact,
        "input": {"size": input_size, "sha256": input_sha256},
        "environment": environment,
        "backends": backend_results,
        "oracle": oracle,
    }
    write_atomic(args.evidence, canonical_json(evidence), args.force_evidence)
    print(str(args.evidence))
    return 2 if failed else 0


def load_evidence(path: Path) -> Tuple[Dict[str, Any], bytes]:
    regular_file(path, "checkpoint evidence")
    evidence, payload = load_json(path, "checkpoint evidence")
    exact_keys(
        evidence,
        [
            "schema_version",
            "kind",
            "runtime_id",
            "catalog",
            "source",
            "tokenizer",
            "converter",
            "artifact",
            "input",
            "environment",
            "backends",
            "oracle",
        ],
        [],
        "checkpoint evidence",
    )
    if evidence["schema_version"] != 1 or evidence["kind"] != EVIDENCE_KIND:
        raise EvidenceError("checkpoint evidence schema/kind differs")
    if not isinstance(evidence["catalog"], dict):
        raise EvidenceError("checkpoint evidence catalog must be an object")
    exact_keys(evidence["catalog"], ["contract_sha256"], [], "evidence catalog")
    digest_text(
        evidence["catalog"]["contract_sha256"], "evidence contract SHA256"
    )
    runtime_id = nonempty_text(evidence["runtime_id"], "evidence runtime_id")
    source = evidence["source"]
    if not isinstance(source, dict):
        raise EvidenceError("checkpoint evidence source must be an object")
    exact_keys(
        source,
        [
            "kind",
            "receipt_sha256",
            "snapshot_sha256",
            "file_manifest_sha256",
            "file_count",
            "files",
            "provider",
            "endpoint",
        ],
        [],
        "evidence source",
    )
    if source["kind"] not in (
        "source-checkpoint",
        "geneb-janusdna-manual-source",
    ):
        raise EvidenceError("evidence source kind is unsupported")
    for key in ("receipt_sha256", "snapshot_sha256", "file_manifest_sha256"):
        digest_text(source[key], "evidence source " + key)
    files = source["files"]
    if not isinstance(files, list) or not files:
        raise EvidenceError("evidence source files must be a nonempty array")
    normalized_files = []
    names = set()
    for index, item in enumerate(files):
        label = "evidence source files[%d]" % index
        if not isinstance(item, dict):
            raise EvidenceError("%s must be an object" % label)
        exact_keys(item, ["name", "size", "sha256"], [], label)
        name = normalized_relative(item["name"], label + ".name")
        if name in names:
            raise EvidenceError("evidence source file names are duplicated")
        names.add(name)
        normalized_files.append(
            {
                "name": name,
                "size": uint(item["size"], label + ".size"),
                "sha256": digest_text(item["sha256"], label + ".sha256"),
            }
        )
    if files != sorted(normalized_files, key=lambda item: item["name"]):
        raise EvidenceError("evidence source files are not canonical")
    if source["file_count"] != len(files):
        raise EvidenceError("evidence source file count differs")
    if source["file_manifest_sha256"] != sha256_bytes(canonical_json(files)):
        raise EvidenceError("evidence source file manifest digest differs")
    provider = source["provider"]
    if not isinstance(provider, dict):
        raise EvidenceError("evidence source provider must be an object")
    provider_kind = provider.get("kind")
    if source["kind"] == "source-checkpoint" and provider_kind == "huggingface":
        exact_keys(
            provider,
            ["kind", "repo", "requested_revision", "resolved_revision"],
            [],
            "evidence source provider",
        )
        if provider["kind"] != "huggingface" or source["endpoint"] is None:
            raise EvidenceError("evidence Hugging Face endpoint/provider differs")
        nonempty_text(provider["repo"], "evidence source repo")
        nonempty_text(provider["requested_revision"], "evidence requested revision")
        revision = nonempty_text(
            provider["resolved_revision"], "evidence resolved revision"
        )
        if not COMMIT_RE.fullmatch(revision):
            raise EvidenceError("evidence resolved revision is not immutable")
        validate_endpoint(source["endpoint"], "huggingface")
    else:
        exact_keys(provider, ["kind", "url"], [], "evidence source provider")
        if (
            not isinstance(provider["kind"], str)
            or not provider["kind"]
            or provider["kind"] == "huggingface"
            or source["endpoint"] is not None
        ):
            raise EvidenceError("evidence manual source provider/endpoint differs")
        nonempty_text(provider["url"], "evidence source URL")
    snapshot_identity = {
        "model_id": runtime_id,
        "provider": provider,
        "files": files,
    }
    if source["snapshot_sha256"] != sha256_bytes(canonical_json(snapshot_identity)):
        raise EvidenceError("evidence source snapshot digest differs")

    tokenizer = evidence["tokenizer"]
    if not isinstance(tokenizer, dict):
        raise EvidenceError("checkpoint evidence tokenizer must be an object")
    if tokenizer.get("kind") == "input-transform":
        exact_keys(
            tokenizer,
            ["kind", "contract_sha256", "vocabulary_size"],
            [],
            "evidence input-transform tokenizer",
        )
        digest_text(
            tokenizer["contract_sha256"],
            "evidence input-transform tokenizer contract",
        )
        uint(
            tokenizer["vocabulary_size"],
            "evidence input-transform tokenizer vocabulary size",
            False,
        )
    else:
        exact_keys(
            tokenizer,
            [
                "descriptor_sha256",
                "compiler_manifest_sha256",
                "source_receipt_contract_sha256",
                "profile",
                "path",
                "size",
                "sha256",
            ],
            [],
            "evidence tokenizer",
        )
        for key in (
            "descriptor_sha256",
            "compiler_manifest_sha256",
            "source_receipt_contract_sha256",
            "sha256",
        ):
            digest_text(tokenizer[key], "evidence tokenizer " + key)
        if tokenizer["profile"] != "evo-tokenizer-v1":
            raise EvidenceError("evidence tokenizer profile differs")
        normalized_relative(tokenizer["path"], "evidence tokenizer path")
        uint(tokenizer["size"], "evidence tokenizer size", False)

    converter = evidence["converter"]
    if not isinstance(converter, dict):
        raise EvidenceError("checkpoint evidence converter must be an object")
    exact_keys(
        converter,
        ["status", "command_sha256", "stdout_sha256", "stderr_sha256"],
        [],
        "evidence converter",
    )
    if converter["status"] not in ("passed", "validated-existing"):
        raise EvidenceError("evidence converter status differs")
    for key in ("command_sha256", "stdout_sha256", "stderr_sha256"):
        digest_text(converter[key], "evidence converter " + key)

    artifact = evidence["artifact"]
    if not isinstance(artifact, dict):
        raise EvidenceError("checkpoint evidence artifact must be an object")
    exact_keys(
        artifact,
        [
            "profile",
            "abi",
            "architecture",
            "converter_profile_contract_sha256",
            "size",
            "sha256",
            "header_sha256",
        ],
        [],
        "evidence artifact",
    )
    for key in ("profile", "abi", "architecture"):
        nonempty_text(artifact[key], "evidence artifact " + key)
    digest_text(
        artifact["converter_profile_contract_sha256"],
        "evidence artifact converter profile contract SHA256",
    )
    uint(artifact["size"], "evidence artifact size", False)
    digest_text(artifact["sha256"], "evidence artifact SHA256")
    digest_text(artifact["header_sha256"], "evidence artifact header SHA256")

    input_value = evidence["input"]
    if not isinstance(input_value, dict):
        raise EvidenceError("checkpoint evidence input must be an object")
    exact_keys(input_value, ["size", "sha256"], [], "evidence input")
    uint(input_value["size"], "evidence input size", False)
    digest_text(input_value["sha256"], "evidence input SHA256")

    environment = evidence["environment"]
    if not isinstance(environment, dict):
        raise EvidenceError("checkpoint evidence environment must be an object")
    exact_keys(environment, ["value", "sha256"], [], "evidence environment")
    if not isinstance(environment["value"], dict):
        raise EvidenceError("evidence environment value must be an object")
    digest_text(environment["sha256"], "evidence environment SHA256")
    if environment["sha256"] != sha256_bytes(canonical_json(environment["value"])):
        raise EvidenceError("evidence environment digest differs")

    backends = evidence["backends"]
    if not isinstance(backends, dict) or set(backends) != set(BACKENDS):
        raise EvidenceError("checkpoint evidence backend set differs")
    for backend in BACKENDS:
        result = backends[backend]
        if not isinstance(result, dict) or result.get("status") not in (
            "not-run",
            "passed",
            "failed",
        ):
            raise EvidenceError("checkpoint evidence backend status is invalid")
        status = result["status"]
        if status == "not-run":
            exact_keys(result, ["status", "reason"], [], "evidence backend " + backend)
            nonempty_text(result["reason"], "evidence backend reason")
            continue
        common = [
            "status",
            "backend",
            "profile",
            "command_sha256",
            "stdout_sha256",
            "stderr_sha256",
            "exit_code",
        ]
        exact_keys(
            result,
            common + (["output"] if status == "passed" else ["reason"]),
            [],
            "evidence backend " + backend,
        )
        if result["backend"] != backend:
            raise EvidenceError("evidence backend identity differs")
        nonempty_text(result["profile"], "evidence backend profile")
        for key in ("command_sha256", "stdout_sha256", "stderr_sha256"):
            digest_text(result[key], "evidence backend " + key)
        exit_code = uint(result["exit_code"], "evidence backend exit code")
        if status == "failed":
            if exit_code == 0:
                raise EvidenceError("failed evidence backend has zero exit code")
            nonempty_text(result["reason"], "failed evidence backend reason")
            continue
        if exit_code != 0:
            raise EvidenceError("passed evidence backend has nonzero exit code")
        output = result["output"]
        if not isinstance(output, dict):
            raise EvidenceError("passed evidence backend output must be an object")
        exact_keys(
            output,
            [
                "manifest_sha256",
                "output_sha256",
                "files",
                "source_label",
                "shape",
                "source_tokens",
                "layer",
                "point",
                "pooling",
                "preset",
            ],
            [],
            "evidence backend output",
        )
        digest_text(output["manifest_sha256"], "evidence manifest SHA256")
        digest_text(output["output_sha256"], "evidence output SHA256")
        nonempty_text(output["source_label"], "evidence output source_label")
        output_files = output["files"]
        if not isinstance(output_files, list) or not output_files:
            raise EvidenceError("evidence output files must be nonempty")
        for index, item in enumerate(output_files):
            label = "evidence output files[%d]" % index
            if not isinstance(item, dict):
                raise EvidenceError("%s must be an object" % label)
            exact_keys(item, ["path", "size", "sha256"], [], label)
            normalized_relative(item["path"], label + ".path")
            uint(item["size"], label + ".size")
            digest_text(item["sha256"], label + ".sha256")
        if output_files != sorted(output_files, key=lambda item: item["path"]):
            raise EvidenceError("evidence output files are not canonical")
        if output["output_sha256"] != sha256_bytes(canonical_json(output_files)):
            raise EvidenceError("evidence output file digest differs")
        shape = output["shape"]
        if not isinstance(shape, list) or not shape:
            raise EvidenceError("evidence output shape is invalid")
        for index, dimension in enumerate(shape):
            uint(dimension, "evidence output shape[%d]" % index, False)
        uint(output["source_tokens"], "evidence output source_tokens", False)
        uint(output["layer"], "evidence output layer")
        for key in ("point", "pooling", "preset"):
            nonempty_text(output[key], "evidence output " + key)
    if not isinstance(evidence["oracle"], dict) or evidence["oracle"].get("status") not in (
        "not-run",
        "passed",
        "failed",
    ):
        raise EvidenceError("checkpoint evidence oracle status is invalid")
    oracle = evidence["oracle"]
    if oracle["status"] == "not-run":
        exact_keys(oracle, ["status", "reason"], [], "evidence oracle")
        nonempty_text(oracle["reason"], "evidence oracle reason")
    else:
        exact_keys(
            oracle,
            [
                "status",
                "oracle_sha256",
                "backend",
                "profile",
                "environment_lock",
                "provenance",
                "tolerances",
                "metrics",
            ],
            [],
            "evidence oracle",
        )
        digest_text(oracle["oracle_sha256"], "evidence oracle SHA256")
        oracle_backend = nonempty_text(oracle["backend"], "evidence oracle backend")
        nonempty_text(oracle["profile"], "evidence oracle profile")
        if (
            oracle_backend not in BACKENDS
            or backends[oracle_backend]["status"] != "passed"
            or not isinstance(oracle["environment_lock"], dict)
            or not isinstance(oracle["provenance"], dict)
        ):
            raise EvidenceError("evidence oracle binding differs")
        reject_local_absolute_paths(
            oracle["environment_lock"], "evidence oracle environment_lock"
        )
        reject_local_absolute_paths(
            oracle["provenance"], "evidence oracle provenance"
        )
        for section in ("tolerances", "metrics"):
            value = oracle[section]
            if not isinstance(value, dict):
                raise EvidenceError("evidence oracle %s must be an object" % section)
            exact_keys(
                value,
                ["max_abs", "mean_abs", "cosine"],
                [],
                "evidence oracle " + section,
            )
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in value.values()
            ):
                raise EvidenceError("evidence oracle %s must be finite" % section)
    return evidence, payload


def evidence_reference(path: Path, payload: bytes, catalog_parent: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(catalog_parent.resolve()).as_posix()
    except ValueError:
        raise EvidenceError("evidence must live below the catalog directory")
    normalized_relative(relative, "catalog evidence path")
    return {
        "schema": EVIDENCE_REF_SCHEMA,
        "path": relative,
        "sha256": sha256_bytes(payload),
    }


def apply_evidence_to_model(
    catalog: Mapping[str, Any],
    model: Dict[str, Any],
    evidence: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    if evidence["runtime_id"] != model.get("runtime_id"):
        raise EvidenceError("evidence runtime_id differs from catalog")
    if evidence["catalog"]["contract_sha256"] != catalog_contract_sha256(catalog, model):
        raise EvidenceError("evidence catalog contract digest differs")
    oracle = evidence["oracle"]
    if oracle.get("status") != "passed":
        raise EvidenceError("catalog promotion requires a passed independent oracle")
    cpu = evidence["backends"]["cpu"]
    if cpu.get("status") != "passed":
        raise EvidenceError("catalog promotion requires a passed CPU backend")
    artifact = evidence["artifact"]
    environment_lock = oracle.get("environment_lock")
    tolerances = oracle.get("tolerances")
    if not isinstance(environment_lock, dict) or not isinstance(tolerances, dict):
        raise EvidenceError("passed oracle evidence is incomplete")
    model["oracle"] = {
        "status": "passed",
        "environment_lock": environment_lock,
        "input_digest": evidence["input"]["sha256"],
        "tolerances": tolerances,
        "evidence": dict(
            reference,
            artifact_sha256=artifact["sha256"],
            output_sha256=cpu["output"]["output_sha256"],
        ),
    }
    model["runtime_support"] = {
        "status": "supported",
        "artifact_profile": artifact["profile"],
        "reason": None,
    }
    for backend in BACKENDS:
        result = evidence["backends"][backend]
        if result.get("status") == "passed":
            model["backends"][backend] = {
                "status": "promoted",
                "evidence": dict(
                    reference,
                    artifact_sha256=artifact["sha256"],
                    output_sha256=result["output"]["output_sha256"],
                    profile=result["profile"],
                ),
            }
    model["promotion_state"] = "runtime-supported"


def catalog_update_command(args: argparse.Namespace) -> int:
    catalog_path = args.catalog.expanduser().resolve(strict=True)
    evidence_path = args.evidence.expanduser().resolve(strict=True)
    catalog, _ = load_json(catalog_path, "GENEB catalog")
    evidence, payload = load_evidence(evidence_path)
    model = catalog_model(catalog, evidence["runtime_id"])
    assert isinstance(model, dict)
    reference = evidence_reference(evidence_path, payload, catalog_path.parent)
    apply_evidence_to_model(catalog, model, evidence, reference)
    if not args.in_place and args.output is None:
        raise EvidenceError("catalog-update requires --output or --in-place")
    output = (
        catalog_path
        if args.in_place
        else args.output.expanduser().resolve()
    )
    encoded = (json.dumps(catalog, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    write_atomic(output, encoded, args.in_place or args.force)
    print(str(output))
    return 0


def validate_reference(
    catalog: Mapping[str, Any], model: Mapping[str, Any], catalog_path: Path
) -> bool:
    oracle = model.get("oracle")
    if not isinstance(oracle, dict) or oracle.get("status") != "passed":
        return False
    reference = oracle.get("evidence")
    if not isinstance(reference, dict):
        raise EvidenceError("passed catalog oracle is missing evidence")
    exact_keys(
        reference,
        ["schema", "path", "sha256", "artifact_sha256", "output_sha256"],
        [],
        "catalog oracle evidence",
    )
    if reference["schema"] != EVIDENCE_REF_SCHEMA:
        raise EvidenceError("catalog oracle evidence schema differs")
    relative = normalized_relative(reference["path"], "catalog oracle evidence path")
    path = strict_relative_file(
        catalog_path.parent, relative, "catalog oracle evidence"
    )
    payload = path.read_bytes()
    if sha256_bytes(payload) != reference["sha256"]:
        raise EvidenceError("catalog oracle evidence SHA256 differs")
    evidence, _ = load_evidence(path)
    if evidence["runtime_id"] != model.get("runtime_id"):
        raise EvidenceError("catalog/evidence runtime_id differs")
    if evidence["catalog"]["contract_sha256"] != catalog_contract_sha256(catalog, model):
        raise EvidenceError("catalog/evidence contract SHA256 differs")
    if evidence["oracle"].get("status") != "passed":
        raise EvidenceError("catalog references non-passing oracle evidence")
    if evidence["artifact"].get("sha256") != reference["artifact_sha256"]:
        raise EvidenceError("catalog/evidence artifact SHA256 differs")
    if evidence["backends"]["cpu"].get("status") != "passed":
        raise EvidenceError("catalog evidence has no passed CPU backend")
    if evidence["backends"]["cpu"]["output"].get("output_sha256") != reference["output_sha256"]:
        raise EvidenceError("catalog/evidence CPU output SHA256 differs")
    if oracle.get("input_digest") != evidence["input"].get("sha256"):
        raise EvidenceError("catalog/evidence input digest differs")
    if oracle.get("environment_lock") != evidence["oracle"].get("environment_lock"):
        raise EvidenceError("catalog/evidence oracle environment differs")
    if oracle.get("tolerances") != evidence["oracle"].get("tolerances"):
        raise EvidenceError("catalog/evidence oracle tolerances differ")
    runtime = model.get("runtime_support")
    if (
        not isinstance(runtime, dict)
        or runtime.get("status") != "supported"
        or runtime.get("artifact_profile") != evidence["artifact"].get("profile")
        or model.get("promotion_state") != "runtime-supported"
    ):
        raise EvidenceError("catalog runtime promotion differs from evidence")
    backends = model.get("backends")
    if not isinstance(backends, dict):
        raise EvidenceError("catalog backends are invalid")
    for backend in BACKENDS:
        catalog_backend = backends.get(backend)
        evidence_backend = evidence["backends"][backend]
        if not isinstance(catalog_backend, dict):
            raise EvidenceError("catalog backend is invalid")
        if catalog_backend.get("status") == "promoted":
            backend_ref = catalog_backend.get("evidence")
            if not isinstance(backend_ref, dict):
                raise EvidenceError("promoted backend is missing evidence")
            exact_keys(
                backend_ref,
                [
                    "schema",
                    "path",
                    "sha256",
                    "artifact_sha256",
                    "output_sha256",
                    "profile",
                ],
                [],
                "catalog backend evidence",
            )
            if (
                backend_ref["schema"] != EVIDENCE_REF_SCHEMA
                or backend_ref["path"] != reference["path"]
                or backend_ref["sha256"] != reference["sha256"]
                or backend_ref["artifact_sha256"] != evidence["artifact"]["sha256"]
                or evidence_backend.get("status") != "passed"
                or backend_ref["output_sha256"]
                != evidence_backend["output"]["output_sha256"]
                or backend_ref["profile"] != evidence_backend["profile"]
            ):
                raise EvidenceError("catalog promoted backend evidence differs")
    return True


def catalog_validate_command(args: argparse.Namespace) -> int:
    catalog_path = args.catalog.expanduser().resolve(strict=True)
    catalog, _ = load_json(catalog_path, "GENEB catalog")
    models = catalog.get("models")
    if not isinstance(models, list):
        raise EvidenceError("GENEB catalog models must be an array")
    selected = [
        model
        for model in models
        if isinstance(model, dict)
        and (args.model is None or model.get("runtime_id") == args.model)
    ]
    if args.model is not None and len(selected) != 1:
        raise EvidenceError("catalog runtime_id is missing or duplicated")
    count = sum(validate_reference(catalog, model, catalog_path) for model in selected)
    print(json.dumps({"valid": True, "validated_evidence": count}, sort_keys=True))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and bind deterministic GENEB canonical-checkpoint evidence"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="convert, embed, and write evidence")
    run.add_argument("--catalog", required=True, type=Path)
    run.add_argument("--model", required=True)
    run.add_argument("--source-receipt", required=True, type=Path)
    run.add_argument("--source-endpoint")
    run.add_argument("--tokenizer-descriptor", type=Path)
    run.add_argument("--tokenizer-root", type=Path)
    run.add_argument("--converter", required=True, type=Path)
    run.add_argument("--converter-python", type=Path, default=Path(sys.executable))
    run.add_argument("--profiles", required=True, type=Path)
    run.add_argument("--artifact", required=True, type=Path)
    run.add_argument("--reuse-artifact", action="store_true")
    run.add_argument("--native", required=True, type=Path)
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--embedding-dir", required=True, type=Path)
    run.add_argument("--evidence", required=True, type=Path)
    run.add_argument("--backend", action="append", choices=BACKENDS)
    run.add_argument("--profile", default="cpu-f32")
    run.add_argument("--preset", default="geneb-v4-normalized")
    run.add_argument("--ctx", dest="context", type=int, default=4096)
    run.add_argument("--oracle-vector", type=Path)
    run.add_argument("--max-abs", type=float)
    run.add_argument("--mean-abs", type=float)
    run.add_argument("--cosine", type=float)
    run.add_argument("--force-evidence", action="store_true")

    update = commands.add_parser(
        "catalog-update", help="atomically bind passed evidence to a catalog"
    )
    update.add_argument("--catalog", required=True, type=Path)
    update.add_argument("--evidence", required=True, type=Path)
    update.add_argument("--output", type=Path)
    update.add_argument("--in-place", action="store_true")
    update.add_argument("--force", action="store_true")

    validate = commands.add_parser(
        "catalog-validate", help="validate catalog evidence references"
    )
    validate.add_argument("--catalog", required=True, type=Path)
    validate.add_argument("--model")
    args = parser.parse_args(argv)
    if args.command == "catalog-update" and args.in_place and args.output is not None:
        parser.error("--in-place and --output are mutually exclusive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        if args.command == "run":
            return run_command(args)
        if args.command == "catalog-update":
            return catalog_update_command(args)
        return catalog_validate_command(args)
    except (EvidenceError, OSError, ValueError) as error:
        print("geneb-checkpoint-evidence: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
