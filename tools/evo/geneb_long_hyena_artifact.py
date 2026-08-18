#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict shared conversion plumbing for GENEB long-Hyena artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .geneb_artifact import (
    GenebArtifactError,
    build_geneb_artifact_metadata,
    catalog_contract_sha256,
    converter_profile_contract_sha256,
    validate_hf_fetch_receipt_provenance,
)
from .hf_checkpoint import (
    CheckpointError,
    TensorSpec,
    load_json,
    normalized_relative_path,
    validate_tensor_manifest,
)


UINT64_MAX = (1 << 64) - 1
CHUNK_SIZE = 16 * 1024 * 1024
MAX_HEADER_SIZE = 16 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
REPO_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
KEY_RE = re.compile(r"[A-Za-z0-9._-]+")
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


class LongHyenaConversionError(ValueError):
    """Raised when a long-Hyena source is outside the closed contract."""


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
        raise LongHyenaConversionError(
            "%s fields differ: missing=%s extra=%s"
            % (label, sorted(required_set - actual), sorted(actual - allowed))
        )


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise LongHyenaConversionError("%s must be a nonempty string" % label)
    return value


def uint(value: Any, label: str, positive: bool = True) -> int:
    minimum = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > UINT64_MAX
    ):
        raise LongHyenaConversionError("%s must be uint64" % label)
    return value


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LongHyenaConversionError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise LongHyenaConversionError("%s must be finite and positive" % label)
    return result


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise LongHyenaConversionError("cannot hash %s: %s" % (path, error))
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def tensor_manifest_sha256(specs: Sequence[TensorSpec]) -> str:
    value = [
        {
            "name": spec.name,
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "nbytes": spec.nbytes,
        }
        for spec in sorted(specs, key=lambda item: item.name)
    ]
    return sha256_bytes(canonical_json(value))


def validate_source_files(raw: Any, label: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict) or not raw:
        raise LongHyenaConversionError("%s must be a nonempty object" % label)
    result = {}  # type: Dict[str, Dict[str, Any]]
    for raw_name, raw_descriptor in raw.items():
        name = normalized_relative_path(raw_name, label + ".name")
        if Path(name).name != name or name in result or not isinstance(raw_descriptor, dict):
            raise LongHyenaConversionError("%s contains an invalid source name" % label)
        exact_keys(raw_descriptor, ["size", "sha256"], [], label + "." + name)
        size = uint(raw_descriptor["size"], label + "." + name + ".size")
        digest = text(raw_descriptor["sha256"], label + "." + name + ".sha256")
        if not SHA256_RE.fullmatch(digest):
            raise LongHyenaConversionError("%s SHA256 is invalid" % name)
        result[name] = {"size": size, "sha256": digest}
    return result


def validate_tokenizer_profile(raw: Any, label: str) -> Dict[str, str]:
    if not isinstance(raw, dict):
        raise LongHyenaConversionError("%s must be an object" % label)
    exact_keys(
        raw,
        ["compiler_manifest", "compiler_manifest_sha256"],
        [],
        label,
    )
    relative = normalized_relative_path(raw["compiler_manifest"], label + ".path")
    digest = text(raw["compiler_manifest_sha256"], label + ".sha256")
    if not SHA256_RE.fullmatch(digest):
        raise LongHyenaConversionError("%s compiler hash is invalid" % label)
    return {"compiler_manifest": relative, "compiler_manifest_sha256": digest}


def validate_profile_identity(profile: Dict[str, Any], label: str) -> None:
    for key in (
        "runtime_id",
        "geneb_model_id",
        "paper_name",
        "catalog_architecture",
        "repo",
        "requested_revision",
        "revision",
        "checkpoint_manifest_sha256",
    ):
        profile[key] = text(profile[key], label + "." + key)
    if not REPO_RE.fullmatch(profile["repo"]):
        raise LongHyenaConversionError("%s.repo is not OWNER/NAME" % label)
    if not COMMIT_RE.fullmatch(profile["revision"]):
        raise LongHyenaConversionError("%s.revision is not an immutable commit" % label)
    if not SHA256_RE.fullmatch(profile["checkpoint_manifest_sha256"]):
        raise LongHyenaConversionError("%s manifest hash is invalid" % label)
    profile["source_files"] = validate_source_files(
        profile["source_files"], label + ".source_files"
    )
    raw_assets = profile["conversion_assets"]
    if not isinstance(raw_assets, list) or not raw_assets:
        raise LongHyenaConversionError(
            "%s.conversion_assets must be a nonempty array" % label
        )
    assets = []  # type: List[str]
    for index, raw_name in enumerate(raw_assets):
        name = normalized_relative_path(
            raw_name, "%s.conversion_assets[%d]" % (label, index)
        )
        if Path(name).name != name or name in assets or name not in profile["source_files"]:
            raise LongHyenaConversionError(
                "%s.conversion_assets contains an invalid/duplicated source" % label
            )
        assets.append(name)
    profile["conversion_assets"] = assets
    profile["tokenizer"] = validate_tokenizer_profile(
        profile["tokenizer"], label + ".tokenizer"
    )


def load_profile_document(
    path: Path,
    expected_format: str,
    profile_keys: Set[str],
    topology_keys: Set[str],
    topology_validator: Any,
) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    root, payload = load_json(path, "GENEB long-Hyena converter profiles")
    exact_keys(root, ["schema_version", "format", "models"], [], "profiles")
    if root["schema_version"] != 1 or root["format"] != expected_format:
        raise LongHyenaConversionError("profile schema/format is unsupported")
    if not isinstance(root["models"], list) or not root["models"]:
        raise LongHyenaConversionError("profiles.models must be nonempty")
    profiles = {}  # type: Dict[str, Dict[str, Any]]
    identities = set()  # type: Set[Tuple[str, str]]
    for index, raw in enumerate(root["models"]):
        label = "models[%d]" % index
        if not isinstance(raw, dict):
            raise LongHyenaConversionError("%s must be an object" % label)
        exact_keys(raw, profile_keys, [], label)
        profile = dict(raw)
        validate_profile_identity(profile, label)
        topology = profile["topology"]
        if not isinstance(topology, dict):
            raise LongHyenaConversionError("%s.topology must be an object" % label)
        exact_keys(topology, topology_keys, [], label + ".topology")
        profile["topology"] = topology_validator(dict(topology), label + ".topology")
        identity = (profile["repo"], profile["revision"])
        if profile["runtime_id"] in profiles or identity in identities:
            raise LongHyenaConversionError("profile identity is duplicated")
        profiles[profile["runtime_id"]] = profile
        identities.add(identity)
    return profiles, payload


def load_catalog_entry(
    path: Path, profile: Mapping[str, Any], expected_padding: str
) -> Tuple[Mapping[str, Any], Mapping[str, Any], bytes]:
    root, payload = load_json(path, "GENEB catalog")
    models = root.get("models")
    if not isinstance(models, list):
        raise LongHyenaConversionError("GENEB catalog models must be an array")
    matches = [item for item in models if isinstance(item, dict) and item.get("runtime_id") == profile["runtime_id"]]
    if len(matches) != 1:
        raise LongHyenaConversionError("catalog runtime_id is missing or duplicated")
    entry = matches[0]
    source = entry.get("source")
    tokenizer = entry.get("tokenizer")
    if not isinstance(source, dict) or not isinstance(tokenizer, dict):
        raise LongHyenaConversionError("catalog source/tokenizer is invalid")
    expected = {
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "architecture": profile["catalog_architecture"],
    }
    wrong = {key: (entry.get(key), value) for key, value in expected.items() if entry.get(key) != value}
    if wrong:
        raise LongHyenaConversionError("catalog identity differs: %s" % wrong)
    if (
        source.get("kind") != "huggingface"
        or source.get("repo") != profile["repo"]
        or source.get("requested_revision") != profile["requested_revision"]
        or source.get("revision") != profile["revision"]
        or source.get("immutable") is not True
    ):
        raise LongHyenaConversionError("catalog source provenance differs")
    if (
        tokenizer.get("add_special_tokens") is not False
        or tokenizer.get("padding_side") != expected_padding
    ):
        raise LongHyenaConversionError("catalog tokenizer special/padding policy differs")
    return entry, root, payload


def validate_catalog_runtime_contract(
    entry: Mapping[str, Any],
    topology: Mapping[str, Any],
    family: str,
    extractor_commit: str,
) -> None:
    if family not in ("hyena", "evo"):
        raise LongHyenaConversionError("catalog runtime family is unsupported")
    hyena = family == "hyena"
    expected_family = "hyena" if hyena else "striped-hyena"
    if entry.get("family") != expected_family:
        raise LongHyenaConversionError("catalog runtime family differs")

    tokenizer = entry.get("tokenizer")
    context = entry.get("context")
    transform = entry.get("input_transform")
    presets = entry.get("embedding_presets")
    provenance = entry.get("provenance")
    if not all(
        isinstance(value, dict)
        for value in (tokenizer, context, transform, presets, provenance)
    ):
        raise LongHyenaConversionError("catalog runtime contract is incomplete")
    maximum = topology["max_seqlen"]
    expected_tokenizer = {
        "kind": "single-nucleotide",
        "add_special_tokens": False,
        "padding_side": "left",
        "pad_to": "batch-max" if hyena else "none",
        "max_tokens": maximum,
    }
    wrong = {
        key: (tokenizer.get(key), value)
        for key, value in expected_tokenizer.items()
        if tokenizer.get(key) != value
    }
    if wrong:
        raise LongHyenaConversionError(
            "catalog tokenizer runtime contract differs: %s" % wrong
        )
    expected_context = {
        "unit": "tokens",
        "declared_max_tokens": maximum,
        "reference_max_tokens": maximum,
        "length_policy": "tokenizer-truncate" if hyena else "reject",
    }
    wrong = {
        key: (context.get(key), value)
        for key, value in expected_context.items()
        if context.get(key) != value
    }
    if wrong:
        raise LongHyenaConversionError("catalog context contract differs: %s" % wrong)
    expected_transform = {
        "case": "preserve",
        "strip_ascii_whitespace": False,
        "u_to_t": False,
        "invalid": "tokenizer-defined",
        "frame_trim": None,
        "raw_crop": None,
        "fixed_pad": None,
        "prefix": None,
        "special_tokens": "none",
        "token_truncation": "right" if hyena else "left",
    }
    wrong = {
        key: (transform.get(key), value)
        for key, value in expected_transform.items()
        if transform.get(key) != value
    }
    if wrong:
        raise LongHyenaConversionError(
            "catalog input transform differs: %s" % wrong
        )
    width = topology["hidden_size"]
    expected_presets = (
        {
            name: {
                "hidden_tap": "last-hidden-state",
                "pooling": "attention-mask-mean",
                "special_tokens": "none",
                "mask_domain": "attention-mask",
                "output_width": width,
            }
            for name in ("reference", "normalized")
        }
        if hyena
        else {
            "reference": {
                "hidden_tap": "model-final-hidden",
                "pooling": "mean-first-record-only",
                "special_tokens": "none",
                "mask_domain": "all-token-rows",
                "output_width": width,
            },
            "normalized": {
                "hidden_tap": "model-final-hidden",
                "pooling": "per-record-mean",
                "special_tokens": "none",
                "mask_domain": "record-token-rows",
                "output_width": width,
            },
        }
    )
    if presets != expected_presets:
        raise LongHyenaConversionError("catalog embedding preset contract differs")
    extractor = provenance.get("extractor")
    if not isinstance(extractor, dict) or extractor.get("commit") != extractor_commit:
        raise LongHyenaConversionError("catalog extractor provenance differs")


def validate_receipt(
    path: Path,
    profile: Mapping[str, Any],
    catalog_path: Path,
    catalog_payload: bytes,
    catalog_root: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
) -> Tuple[Dict[str, Path], bytes]:
    receipt, payload = load_json(path, "source checkpoint receipt")
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
        ],
        ["load_path", "source_kind", "catalog_path", "catalog_sha256", "catalog_contract_sha256"],
        "source receipt",
    )
    catalog_digests = {
        key for key in ("catalog_sha256", "catalog_contract_sha256") if key in receipt
    }
    has_catalog_provenance = any(
        key in receipt
        for key in (
            "source_kind",
            "catalog_path",
            "catalog_sha256",
            "catalog_contract_sha256",
        )
    )
    if has_catalog_provenance and (
        "source_kind" not in receipt
        or "catalog_path" not in receipt
        or len(catalog_digests) != 1
    ):
        raise LongHyenaConversionError(
            "source receipt catalog provenance is incomplete"
        )
    try:
        validate_hf_fetch_receipt_provenance(
            receipt, catalog_path, catalog_payload, catalog_entry
        )
    except GenebArtifactError as error:
        raise LongHyenaConversionError(
            "source receipt catalog provenance differs: %s" % error
        )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != profile["runtime_id"]
        or receipt["repo"] != profile["repo"]
        or receipt["requested_revision"] != profile["requested_revision"]
        or receipt["resolved_revision"] != profile["revision"]
        or ("load_path" in receipt and receipt["load_path"] is not None)
    ):
        raise LongHyenaConversionError("source receipt identity/revision is not pinned")
    if not isinstance(receipt["files"], list):
        raise LongHyenaConversionError("source receipt files must be an array")
    expected = profile["source_files"]
    required = set(profile["conversion_assets"])
    receipt_paths = {}  # type: Dict[str, Path]
    verified = {}  # type: Dict[str, Tuple[int, str]]
    for index, raw in enumerate(receipt["files"]):
        label = "receipt.files[%d]" % index
        if not isinstance(raw, dict):
            raise LongHyenaConversionError("%s must be an object" % label)
        exact_keys(raw, ["name", "size", "sha256", "path"], [], label)
        name = normalized_relative_path(raw["name"], label + ".name")
        if name in receipt_paths:
            raise LongHyenaConversionError("%s name is duplicated" % label)
        size = uint(raw["size"], label + ".size", False)
        digest = text(raw["sha256"], label + ".sha256")
        if not SHA256_RE.fullmatch(digest):
            raise LongHyenaConversionError("%s SHA256 is invalid" % label)
        source_path = Path(text(raw["path"], label + ".path"))
        if source_path.is_symlink():
            raise LongHyenaConversionError("source asset %s is a symlink" % name)
        try:
            actual_size = source_path.stat().st_size
        except OSError as error:
            raise LongHyenaConversionError("cannot stat %s: %s" % (name, error))
        actual_digest = sha256_file(source_path)
        if actual_size != size or actual_digest != digest:
            raise LongHyenaConversionError(
                "source receipt asset %s size/SHA differs" % name
            )
        receipt_paths[name] = source_path
        verified[name] = (size, digest)
    if not required <= set(receipt_paths):
        raise LongHyenaConversionError(
            "source conversion assets are missing: %s"
            % sorted(required - set(receipt_paths))
        )
    for name in set(expected) & set(receipt_paths):
        if verified[name] != (expected[name]["size"], expected[name]["sha256"]):
            raise LongHyenaConversionError(
                "pinned source asset %s size/SHA differs" % name
            )
    return {
        name: receipt_paths[name] for name in profile["conversion_assets"]
    }, payload


def validate_tokenizer_descriptor(
    descriptor_path: Path,
    tokenizer_root: Optional[Path],
    output_path: Path,
    expected_compiler_sha256: str,
) -> Tuple[Dict[str, Any], str]:
    descriptor, payload = load_json(descriptor_path, "tokenizer descriptor")
    exact_keys(descriptor, TOKENIZER_DESCRIPTOR_KEYS, [], "tokenizer descriptor")
    if (
        descriptor["converter.schema"] != "evo-tokenizer-conversion-receipt"
        or descriptor["converter.version"] != 1
        or descriptor["tokenizer.profile"] != "evo-tokenizer-v1"
        or descriptor["compiler_manifest_sha256"] != expected_compiler_sha256
    ):
        raise LongHyenaConversionError("tokenizer descriptor schema/compiler differs")
    for key in (
        "compiler_manifest_sha256",
        "source_receipt_contract_sha256",
        "tokenizer.sha256",
    ):
        if not isinstance(descriptor[key], str) or not SHA256_RE.fullmatch(descriptor[key]):
            raise LongHyenaConversionError("tokenizer descriptor %s is invalid" % key)
    relative = normalized_relative_path(descriptor["tokenizer.path"], "tokenizer.path")
    size = uint(descriptor["tokenizer.size"], "tokenizer.size")
    root = (tokenizer_root if tokenizer_root is not None else descriptor_path.parent).resolve()
    asset = root / relative
    component = root
    for part in Path(relative).parts:
        component /= part
        if component.is_symlink():
            raise LongHyenaConversionError(
                "tokenizer asset path contains a symlink"
            )
    resolved_asset = asset.resolve()
    try:
        resolved_asset.relative_to(root)
    except ValueError:
        raise LongHyenaConversionError("tokenizer.path escapes tokenizer root")
    expected_asset = output_path.resolve().parent / relative
    if resolved_asset != expected_asset.resolve():
        raise LongHyenaConversionError(
            "tokenizer asset must live under artifact output root at tokenizer.path"
        )
    try:
        actual_size = resolved_asset.stat().st_size
    except OSError as error:
        raise LongHyenaConversionError("cannot stat tokenizer asset: %s" % error)
    actual_digest = sha256_file(resolved_asset)
    if actual_size != size or actual_digest != descriptor["tokenizer.sha256"]:
        raise LongHyenaConversionError("tokenizer asset size/SHA differs")
    return (
        {
            key: descriptor[key]
            for key in (
                "tokenizer.profile",
                "tokenizer.path",
                "tokenizer.sha256",
                "tokenizer.size",
            )
        },
        sha256_bytes(payload),
    )


def _encode_metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "b:%d" % int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or value > UINT64_MAX:
            raise LongHyenaConversionError("metadata integer is outside uint64")
        return "u:%d" % value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LongHyenaConversionError("metadata float is not finite")
        return "f:%016x" % struct.unpack("<Q", struct.pack("<d", value))[0]
    if isinstance(value, str):
        return "s:" + value
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= UINT64_MAX
        for item in value
    ):
        if not value:
            raise LongHyenaConversionError("metadata uint64 list must be nonempty")
        return "l:" + ",".join(str(item) for item in value)
    raise LongHyenaConversionError(
        "unsupported metadata value type: %s" % type(value).__name__
    )


def encoded_header(
    artifact_profile: str, metadata: Mapping[str, Any], tensors: Sequence[Any]
) -> bytes:
    encoded = {}  # type: Dict[str, str]
    for key in sorted(metadata):
        if not KEY_RE.fullmatch(key) or len(key.encode("ascii")) > 255 or key == "runtime.profile":
            raise LongHyenaConversionError("invalid/reserved metadata key %r" % key)
        encoded[key] = _encode_metadata_value(metadata[key])
    encoded["runtime.profile"] = "s:" + artifact_profile
    root = {"__metadata__": dict(sorted(encoded.items()))}  # type: Dict[str, Any]
    offset = 0
    for tensor in tensors:
        end = offset + tensor.nbytes
        if end > UINT64_MAX:
            raise LongHyenaConversionError("artifact tensor data exceeds uint64")
        root[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    raw = json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    if not header or len(header) > MAX_HEADER_SIZE:
        raise LongHyenaConversionError("runtime Safetensors header exceeds 16 MiB")
    return header


def write_artifact(
    output_path: Path,
    artifact_profile: str,
    metadata: Mapping[str, Any],
    tensors: Sequence[Any],
    force: bool,
) -> None:
    if output_path.suffix != ".safetensors" or not tensors:
        raise LongHyenaConversionError("output must be nonempty .safetensors")
    header = encoded_header(artifact_profile, metadata, tensors)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError("output already exists: %s" % output_path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % output_path.name,
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as output:
            total_size = 8 + len(header) + sum(tensor.nbytes for tensor in tensors)
            if total_size > UINT64_MAX:
                raise LongHyenaConversionError("runtime artifact exceeds uint64")
            output.truncate(total_size)
            output.seek(0)
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for tensor in tensors:
                written = 0
                for raw_chunk in tensor.iter_chunks(CHUNK_SIZE):
                    chunk = memoryview(raw_chunk).cast("B")
                    if written + len(chunk) > tensor.nbytes:
                        raise LongHyenaConversionError("tensor yielded too many bytes")
                    output.write(chunk)
                    written += len(chunk)
                if written != tensor.nbytes:
                    raise LongHyenaConversionError("tensor yielded too few bytes")
            output.flush()
            os.fsync(output.fileno())
        if force:
            os.replace(str(temporary), str(output_path))
        else:
            os.link(str(temporary), str(output_path))
            temporary.unlink()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def common_metadata(
    catalog_root: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
    catalog_payload: bytes,
    profile: Mapping[str, Any],
    profile_format: str,
    receipt_payload: bytes,
    tokenizer: Mapping[str, Any],
    tokenizer_descriptor_sha256: str,
) -> Dict[str, Any]:
    try:
        metadata = build_geneb_artifact_metadata(
            catalog_root, catalog_entry, catalog_payload
        )
    except GenebArtifactError as error:
        raise LongHyenaConversionError(str(error))
    metadata.update(tokenizer)
    metadata.update(
        {
            "model.id": profile["runtime_id"],
            "source.repo": profile["repo"],
            "source.revision": profile["revision"],
            "source.immutable": True,
            "source.receipt_sha256": sha256_bytes(receipt_payload),
            "source.catalog_contract_sha256": catalog_contract_sha256(
                catalog_root, catalog_entry
            ),
            "source.converter_profile_contract_sha256": converter_profile_contract_sha256(
                1, profile_format, profile
            ),
            "source.checkpoint_manifest_sha256": profile[
                "checkpoint_manifest_sha256"
            ],
            "source.config_sha256": profile["source_files"]["config.json"][
                "sha256"
            ],
            "source.tokenizer_config_sha256": profile["source_files"][
                "tokenizer_config.json"
            ]["sha256"],
            "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        }
    )
    return metadata


__all__ = [
    "CheckpointError",
    "LongHyenaConversionError",
    "TensorSpec",
    "common_metadata",
    "exact_keys",
    "load_catalog_entry",
    "load_profile_document",
    "number",
    "sha256_bytes",
    "tensor_manifest_sha256",
    "text",
    "uint",
    "validate_receipt",
    "validate_catalog_runtime_contract",
    "validate_tensor_manifest",
    "validate_tokenizer_descriptor",
    "write_artifact",
]
