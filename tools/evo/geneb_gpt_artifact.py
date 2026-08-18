#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared strict helpers for the two canonical GENEB GPT artifact families."""

import hashlib
import itertools
import json
import math
import os
import re
import stat
import struct
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .hf_checkpoint import load_json, normalized_relative_path


CHUNK_SIZE = 16 * 1024 * 1024
MAX_HEADER_SIZE = 16 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
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
DNA_GPT_BASE_ALPHABET = "NAGCT"
DNA_GPT_RESERVED_TOKENS = (
    tuple(str(index) for index in range(10))
    + ("+", "-", "*", "/", "=", "&", "|", "!")
    + ("M", "B", "P", "R", "I", "K", "L", "O", "Q", "S", "U", "V", "W", "Y", "X", "Z")
)


class ConversionError(ValueError):
    """Raised when a source falls outside a closed GENEB GPT contract."""


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
        raise ConversionError(
            "%s fields differ: missing=%s extra=%s"
            % (label, sorted(required_set - actual), sorted(actual - allowed))
        )


def object_value(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConversionError("%s must be an object" % label)
    return value


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ConversionError("%s must be a nonempty string" % label)
    return value


def nullable_string(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return nonempty_string(value, label)


def uint64_value(value: Any, label: str, positive: bool = True) -> int:
    minimum = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value >= (1 << 64)
    ):
        raise ConversionError(
            "%s must be a%s uint64" % (label, " positive" if positive else "")
        )
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionError("%s must be finite numeric metadata" % label)
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConversionError("%s must be finite and positive" % label)
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
        raise ConversionError("cannot hash %s: %s" % (path, error))
    return digest.hexdigest()


def default_config_path(script_path: Path, name: str) -> Path:
    source = script_path.resolve().parent.parent / "configs" / name
    if source.is_file():
        return source
    return script_path.resolve().parent.parent / "share" / "evo" / "configs" / name


def canonical_manifest_sha256(sources: Sequence[Any]) -> str:
    manifest = sorted(
        [
        {"name": source.name, "dtype": source.dtype, "shape": list(source.shape)}
        for source in sources
        ],
        key=lambda item: item["name"],
    )
    payload = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    return sha256_bytes(payload)


def dna_gpt_vocab_pieces(kind: str) -> List[str]:
    if kind not in ("static-sixmer", "dynamic-sixmer"):
        raise ConversionError("DNA-GPT tokenizer kind is unsupported")
    pieces = ["<%s>" % token for token in DNA_GPT_RESERVED_TOKENS]
    start = 6 if kind == "static-sixmer" else 1
    for width in range(start, 7):
        pieces.extend(
            "".join(item)
            for item in itertools.product(DNA_GPT_BASE_ALPHABET, repeat=width)
        )
    return pieces


def validate_profile_manifest_identity(
    sources: Sequence[Any], profile: Mapping[str, Any]
) -> None:
    count = uint64_value(profile.get("source_tensor_count"), "source_tensor_count")
    nbytes = uint64_value(profile.get("source_tensor_bytes"), "source_tensor_bytes")
    digest = nonempty_string(
        profile.get("source_manifest_sha256"), "source_manifest_sha256"
    )
    if not SHA256_RE.fullmatch(digest):
        raise ConversionError("source_manifest_sha256 must be lowercase SHA256")
    actual_bytes = sum(source.nbytes for source in sources)
    actual_digest = canonical_manifest_sha256(sources)
    if len(sources) != count or actual_bytes != nbytes or actual_digest != digest:
        raise ConversionError(
            "source tensor header differs: count=%d bytes=%d manifest_sha256=%s"
            % (len(sources), actual_bytes, actual_digest)
        )


def _safe_tokenizer_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = root
    for component in PurePosixPath(relative).parts:
        candidate = candidate / component
        try:
            mode = os.lstat(str(candidate)).st_mode
        except OSError as error:
            raise ConversionError("tokenizer asset component is unavailable: %s" % error)
        if stat.S_ISLNK(mode):
            raise ConversionError("tokenizer asset path must not contain symlinks")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ConversionError("tokenizer asset escapes tokenizer root")
    if not stat.S_ISREG(os.stat(str(resolved), follow_symlinks=False).st_mode):
        raise ConversionError("tokenizer asset must be a regular file")
    return resolved


def validate_tokenizer_descriptor(
    descriptor_path: Path,
    tokenizer_root: Optional[Path],
    expected_compiler_manifest_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], str, Dict[str, Any], bytes]:
    descriptor, payload = load_json(descriptor_path, "tokenizer descriptor")
    if set(descriptor) != TOKENIZER_DESCRIPTOR_KEYS:
        raise ConversionError(
            "tokenizer descriptor fields differ: missing=%s extra=%s"
            % (
                sorted(TOKENIZER_DESCRIPTOR_KEYS - set(descriptor)),
                sorted(set(descriptor) - TOKENIZER_DESCRIPTOR_KEYS),
            )
        )
    if (
        descriptor["converter.schema"]
        != "evo-tokenizer-conversion-receipt"
        or descriptor["converter.version"] != 1
        or descriptor["tokenizer.profile"] != "evo-tokenizer-v1"
    ):
        raise ConversionError("tokenizer descriptor schema/profile is unsupported")
    for key in (
        "compiler_manifest_sha256",
        "source_receipt_contract_sha256",
        "tokenizer.sha256",
    ):
        if not isinstance(descriptor[key], str) or not SHA256_RE.fullmatch(
            descriptor[key]
        ):
            raise ConversionError("tokenizer descriptor %s is not lowercase SHA256" % key)
    if (
        expected_compiler_manifest_sha256 is not None
        and descriptor["compiler_manifest_sha256"]
        != expected_compiler_manifest_sha256
    ):
        raise ConversionError(
            "tokenizer compiler manifest SHA256 differs from pinned profile"
        )
    relative = normalized_relative_path(descriptor["tokenizer.path"], "tokenizer.path")
    size = uint64_value(descriptor["tokenizer.size"], "tokenizer.size")
    root = tokenizer_root if tokenizer_root is not None else descriptor_path.parent
    asset_path = _safe_tokenizer_path(root, relative)
    try:
        actual_size = asset_path.stat().st_size
    except OSError as error:
        raise ConversionError("cannot stat tokenizer asset: %s" % error)
    actual_digest = sha256_file(asset_path)
    if actual_size != size or actual_digest != descriptor["tokenizer.sha256"]:
        raise ConversionError("tokenizer asset size/SHA256 differs from descriptor")
    asset, asset_payload = load_json(asset_path, "runtime tokenizer asset")
    metadata = {
        "tokenizer.profile": descriptor["tokenizer.profile"],
        "tokenizer.path": relative,
        "tokenizer.sha256": descriptor["tokenizer.sha256"],
        "tokenizer.size": size,
    }
    return metadata, sha256_bytes(payload), asset, asset_payload


def validate_tokenizer_output_binding(
    descriptor_path: Path,
    tokenizer_root: Optional[Path],
    output_path: Path,
) -> None:
    """Require the validated tokenizer root to be the model artifact root."""
    root = (
        tokenizer_root.resolve()
        if tokenizer_root is not None
        else descriptor_path.resolve().parent
    )
    artifact_root = output_path.resolve().parent
    if root != artifact_root:
        raise ConversionError(
            "tokenizer root must equal the output artifact directory"
        )


def tokenizer_vocabulary(asset: Mapping[str, Any]) -> Tuple[Dict[int, str], Dict[str, int]]:
    vocab = asset.get("vocab")
    if not isinstance(vocab, list) or not vocab:
        raise ConversionError("runtime tokenizer vocab must be nonempty")
    by_id = {}  # type: Dict[int, str]
    by_piece = {}  # type: Dict[str, int]
    for index, raw in enumerate(vocab):
        if not isinstance(raw, dict) or set(raw) != {"id", "piece"}:
            raise ConversionError("runtime tokenizer vocab[%d] fields differ" % index)
        token_id = uint64_value(raw["id"], "runtime tokenizer vocab id", False)
        piece = nonempty_string(raw["piece"], "runtime tokenizer vocab piece")
        if token_id > 0xFFFFFFFF or token_id in by_id or piece in by_piece:
            raise ConversionError("runtime tokenizer vocab has duplicate/out-of-range ID")
        by_id[token_id] = piece
        by_piece[piece] = token_id
    return by_id, by_piece


def validate_common_tokenizer_asset(
    asset: Mapping[str, Any], expected_vocab_size: int
) -> Tuple[Dict[int, str], Dict[str, int]]:
    if set(asset) != {
        "format",
        "kind",
        "normalization",
        "pre_tokenizer",
        "model",
        "post_processor",
        "special_tokens",
        "vocab",
    } or asset.get("format") != "evo-tokenizer-v1":
        raise ConversionError("runtime tokenizer asset schema is unsupported")
    by_id, by_piece = tokenizer_vocabulary(asset)
    if len(by_id) != expected_vocab_size or set(by_id) != set(
        range(expected_vocab_size)
    ):
        raise ConversionError("runtime tokenizer vocabulary is not dense/model-sized")
    return by_id, by_piece


def validate_receipt_file_entry(
    raw: Any, label: str, allow_nested: bool = False
) -> Tuple[str, int, str, Path]:
    entry = object_value(raw, label)
    exact_keys(entry, ["name", "size", "sha256", "path"], [], label)
    name = normalized_relative_path(entry["name"], label + ".name")
    if not allow_nested and PurePosixPath(name).name != name:
        raise ConversionError("%s.name must identify a checkpoint-root file" % label)
    size = uint64_value(entry["size"], label + ".size", False)
    digest = nonempty_string(entry["sha256"], label + ".sha256")
    if not SHA256_RE.fullmatch(digest):
        raise ConversionError("%s.sha256 must be lowercase SHA256" % label)
    path = Path(nonempty_string(entry["path"], label + ".path")).resolve()
    try:
        actual_size = path.stat().st_size
    except OSError as error:
        raise ConversionError("cannot stat receipt asset %s: %s" % (name, error))
    # Integrity is deliberately established before any checkpoint parser is called.
    actual_digest = sha256_file(path)
    if actual_size != size or actual_digest != digest:
        raise ConversionError(
            "source receipt integrity mismatch for %s: size=%d sha256=%s"
            % (name, actual_size, actual_digest)
        )
    return name, size, digest, path


class CanonicalF32TensorSource:
    """Stream F32 unchanged or expand verified BF16 bits exactly to F32."""

    def __init__(self, source: Any) -> None:
        if source.dtype not in ("F32", "BF16"):
            raise ConversionError("canonical GPT source dtype must be F32 or BF16")
        self.name = source.name
        self.dtype = "F32"
        self.shape = tuple(source.shape)
        self.nbytes = source.nbytes if source.dtype == "F32" else source.nbytes * 2
        self._source = source

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        if self._source.dtype == "F32":
            for chunk in self._source.iter_chunks(chunk_size):
                yield memoryview(chunk).cast("B")
            return
        # BF16 is the high 16 bits of an IEEE-754 binary32 value.  Expanding
        # those bits is exact (including signed zero, infinities, and NaN
        # payloads) and avoids backend- or host-rounding behavior.
        try:
            import numpy as np
        except ModuleNotFoundError as error:
            raise ConversionError(
                "BF16 DNA-GPT conversion requires offline NumPy"
            ) from error
        source_chunk_size = max(2, (chunk_size // 2) & ~1)
        pending = b""
        for raw_chunk in self._source.iter_chunks(source_chunk_size):
            raw = pending + bytes(memoryview(raw_chunk).cast("B"))
            usable = len(raw) & ~1
            if usable:
                converted = np.frombuffer(raw, dtype="<u2", count=usable // 2).astype(
                    "<u4"
                )
                np.left_shift(converted, 16, out=converted)
                yield memoryview(converted).cast("B")
            pending = raw[usable:]
        if pending:
            raise ConversionError("BF16 tensor source ended on a partial element")


def _encode_metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "b:%d" % int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or value >= (1 << 64):
            raise ConversionError("metadata integer is outside uint64")
        return "u:%d" % value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConversionError("metadata float is non-finite")
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        return "f:%016x" % bits
    if isinstance(value, str):
        return "s:" + value
    raise ConversionError("unsupported metadata value type %s" % type(value).__name__)


def _header(
    metadata: Mapping[str, Any], tensors: Sequence[Any], artifact_profile: str
) -> bytes:
    encoded = {
        key: _encode_metadata_value(value) for key, value in sorted(metadata.items())
    }
    if "runtime.profile" in encoded:
        raise ConversionError("runtime.profile is reserved by the artifact writer")
    encoded["runtime.profile"] = "s:" + artifact_profile
    root = {"__metadata__": dict(sorted(encoded.items()))}  # type: Dict[str, Any]
    offset = 0
    names = set()
    for tensor in tensors:
        if tensor.name in names:
            raise ConversionError("canonical tensor %r is duplicated" % tensor.name)
        names.add(tensor.name)
        if tensor.dtype != "F32" or tensor.nbytes <= 0:
            raise ConversionError("runtime tensors must be nonempty F32")
        end = offset + tensor.nbytes
        if end >= (1 << 64):
            raise ConversionError("runtime tensor buffer exceeds uint64")
        root[tensor.name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    raw = json.dumps(root, sort_keys=True, separators=(",", ":")).encode("ascii")
    payload = raw + b" " * ((-len(raw)) % 8)
    if not payload or len(payload) > MAX_HEADER_SIZE:
        raise ConversionError("Safetensors header exceeds 16 MiB")
    return payload


def write_artifact(
    path: Path,
    metadata: Mapping[str, Any],
    tensors: Sequence[Any],
    artifact_profile: str,
    force: bool,
) -> None:
    if path.suffix != ".safetensors":
        raise ConversionError("output path must end in .safetensors")
    if not tensors:
        raise ConversionError("runtime tensor set must not be empty")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError("output already exists: %s" % path)
    header = _header(metadata, tensors, artifact_profile)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as output:
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for tensor in tensors:
                written = 0
                for raw_chunk in tensor.iter_chunks(CHUNK_SIZE):
                    chunk = memoryview(raw_chunk).cast("B")
                    if written + len(chunk) > tensor.nbytes:
                        raise ConversionError("tensor %r yielded too many bytes" % tensor.name)
                    output.write(chunk)
                    written += len(chunk)
                if written != tensor.nbytes:
                    raise ConversionError(
                        "tensor %r yielded %d bytes; expected %d"
                        % (tensor.name, written, tensor.nbytes)
                    )
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


__all__ = [
    "CanonicalF32TensorSource",
    "ConversionError",
    "DNA_GPT_BASE_ALPHABET",
    "DNA_GPT_RESERVED_TOKENS",
    "SHA256_RE",
    "canonical_manifest_sha256",
    "default_config_path",
    "dna_gpt_vocab_pieces",
    "exact_keys",
    "finite_float",
    "nonempty_string",
    "nullable_string",
    "object_value",
    "sha256_bytes",
    "sha256_file",
    "tokenizer_vocabulary",
    "uint64_value",
    "validate_common_tokenizer_asset",
    "validate_profile_manifest_identity",
    "validate_receipt_file_entry",
    "validate_tokenizer_descriptor",
    "validate_tokenizer_output_binding",
    "write_artifact",
]
