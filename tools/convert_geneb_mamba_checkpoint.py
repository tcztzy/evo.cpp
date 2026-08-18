#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile pinned GENEB Caduceus/eccDNAMamba checkpoints for native CPU."""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if not (_SCRIPT_DIRECTORY / "evo").is_dir():
    _INSTALLED_PYTHON = _SCRIPT_DIRECTORY.parent / "share" / "evo" / "python"
    if _INSTALLED_PYTHON.is_dir():
        sys.path.insert(0, str(_INSTALLED_PYTHON))

from evo.geneb_artifact import (  # noqa: E402
    GenebArtifactError,
    build_geneb_artifact_metadata,
    catalog_contract_sha256,
    converter_profile_contract_sha256,
    validate_hf_fetch_receipt_provenance,
)
from evo.hf_checkpoint import (  # noqa: E402
    CheckpointError,
    TensorSpec,
    load_hf_torch_checkpoints,
    load_json,
    normalized_relative_path,
    read_hf_safetensors,
    validate_tensor_manifest,
)


ARTIFACT_PROFILE = "geneb-mamba-runtime-v1"
RUNTIME_ABI = "geneb-mamba-safetensors-v1"
RUNTIME_ARCHITECTURE = "GenebMambaEncoder"
PROFILE_FORMAT = "geneb-mamba-converter-v1"
TOKENIZER_PROFILE = "evo-tokenizer-v1"
MAX_HEADER_SIZE = 16 * 1024 * 1024
CHUNK_SIZE = 16 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
KEY_RE = re.compile(r"[A-Za-z0-9._-]+")
PYTHON_FILE_RE = re.compile(r"[A-Za-z0-9_]+[.]py")

IMPLEMENTATION_CONTRACTS = {
    "caduceus-mamba1": "pinned-caduceus-remote-code-selective-scan-v1",
    "eccdna-mamba2": (
        "mamba-ssm-2.2.4@95d8aba8a8c75aedcaa6143713b11e745e7cd0d9;"
        "sdist-sha256=e4114c69302796c91b71e90032c2d974f611608fab331582a80de6eaf075efb9"
    ),
}
PROFILE_KEYS = {
    "runtime_id",
    "geneb_model_id",
    "paper_name",
    "catalog_architecture",
    "repo",
    "revision",
    "source",
    "config_sha256",
    "remote_code",
    "config_required",
    "tokenizer",
    "topology",
}
SOURCE_KEYS = {"format", "weights_name", "weights_size", "weights_sha256"}
TOKENIZER_KEYS = {
    "compiler_manifest_sha256",
    "compiled_asset_sha256",
    "compiled_asset_size",
    "kind",
    "emitted_vocab_size",
    "add_special_tokens",
    "padding_side",
}
TOPOLOGY_KEYS = {
    "variant",
    "vocab_size",
    "width",
    "output_width",
    "num_layers",
    "max_seqlen",
    "advertised_training_length",
    "inner_width",
    "state_width",
    "conv_width",
    "time_step_rank",
    "mlp_width",
    "head_width",
    "heads",
    "groups",
    "norm_epsilon",
    "rcps",
    "complement_map",
}
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


class ConversionError(ValueError):
    """Raised when source material falls outside the frozen Mamba ABI."""


def exact_keys(value: Mapping[str, Any], required: Iterable[str], optional: Iterable[str], label: str) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    actual = set(value)
    if not required_set <= actual or not actual <= allowed:
        raise ConversionError(
            "%s fields differ: missing=%s extra=%s"
            % (label, sorted(required_set - actual), sorted(actual - allowed))
        )


def object_value(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversionError("%s must be an object" % label)
    return dict(value)


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ConversionError("%s must be a nonempty string" % label)
    return value


def uint64_value(value: Any, label: str, positive: bool = True) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > UINT64_MAX:
        raise ConversionError("%s must be a%s uint64" % (label, " positive" if positive else ""))
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConversionError("%s must be positive finite" % label)
    return result


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(source_path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with source_path.open("rb") as source:
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ConversionError("cannot hash %s: %s" % (source_path, error))
    return digest.hexdigest()


def default_config_path(name: str) -> Path:
    source_path = _SCRIPT_DIRECTORY.parent / "configs" / name
    if source_path.is_file():
        return source_path
    return _SCRIPT_DIRECTORY.parent / "share" / "evo" / "configs" / name


def validate_topology(raw: Any, label: str) -> Dict[str, Any]:
    topology = object_value(raw, label)
    if set(topology) != TOPOLOGY_KEYS:
        raise ConversionError("%s fields differ" % label)
    if topology["variant"] not in IMPLEMENTATION_CONTRACTS:
        raise ConversionError("%s.variant is unsupported" % label)
    for key in (
        "vocab_size", "width", "output_width", "num_layers", "inner_width",
        "state_width", "conv_width",
    ):
        topology[key] = uint64_value(topology[key], "%s.%s" % (label, key))
    for key in (
        "max_seqlen", "advertised_training_length", "time_step_rank",
        "mlp_width", "head_width", "heads", "groups",
    ):
        topology[key] = uint64_value(topology[key], "%s.%s" % (label, key), False)
    topology["norm_epsilon"] = finite_float(topology["norm_epsilon"], label + ".norm_epsilon")
    if not isinstance(topology["rcps"], bool):
        raise ConversionError("%s.rcps must be boolean" % label)
    complement = topology["complement_map"]
    if not isinstance(complement, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in complement
    ):
        raise ConversionError("%s.complement_map must contain nonnegative integers" % label)
    width = topology["width"]
    if topology["variant"] == "caduceus-mamba1":
        exact = (
            topology["inner_width"] == width * 2
            and topology["time_step_rank"] == (width + 15) // 16
            and topology["mlp_width"] == 0
            and topology["head_width"] == 0
            and topology["heads"] == 0
            and topology["groups"] == 0
            and topology["output_width"] == width * (2 if topology["rcps"] else 1)
        )
        if topology["rcps"]:
            exact = exact and len(complement) == topology["vocab_size"] and all(
                value < len(complement) and complement[value] == index
                for index, value in enumerate(complement)
            )
        else:
            exact = exact and not complement
    else:
        exact = (
            not topology["rcps"] and not complement
            and topology["output_width"] == width
            and topology["inner_width"] == width * 2
            and topology["time_step_rank"] == 0
            and topology["mlp_width"] == width * 4
            and topology["head_width"] > 0
            and topology["heads"] > 0
            and topology["groups"] > 0
            and topology["inner_width"] == topology["head_width"] * topology["heads"]
            and topology["heads"] % topology["groups"] == 0
        )
    if not exact:
        raise ConversionError("%s geometry differs from the frozen variant" % label)
    return topology


def load_profiles(source_path: Path) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    root, payload = load_json(source_path, "GENEB Mamba converter profiles")
    exact_keys(root, ["schema_version", "format", "implementation_contracts", "models"], [], "profile manifest")
    if root["schema_version"] != 1 or root["format"] != PROFILE_FORMAT:
        raise ConversionError("Mamba profile schema/format is unsupported")
    if root["implementation_contracts"] != IMPLEMENTATION_CONTRACTS:
        raise ConversionError("Mamba implementation contracts differ")
    if not isinstance(root["models"], list) or not root["models"]:
        raise ConversionError("Mamba profiles must be a nonempty array")
    profiles = {}  # type: Dict[str, Dict[str, Any]]
    identities = set()  # type: Set[Tuple[str, str]]
    for index, raw in enumerate(root["models"]):
        label = "models[%d]" % index
        profile = object_value(raw, label)
        if set(profile) != PROFILE_KEYS:
            raise ConversionError("%s fields differ" % label)
        for key in ("runtime_id", "geneb_model_id", "paper_name", "catalog_architecture", "repo", "revision", "config_sha256"):
            profile[key] = nonempty_string(profile[key], "%s.%s" % (label, key))
        if not REPO_RE.fullmatch(profile["repo"]):
            raise ConversionError("%s.repo is not OWNER/NAME" % label)
        if not COMMIT_RE.fullmatch(profile["revision"]):
            raise ConversionError("%s.revision is not immutable" % label)
        if not SHA256_RE.fullmatch(profile["config_sha256"]):
            raise ConversionError("%s.config_sha256 is invalid" % label)
        source = object_value(profile["source"], label + ".source")
        if set(source) != SOURCE_KEYS or source["format"] not in ("safetensors", "pytorch-bin"):
            raise ConversionError("%s.source differs" % label)
        source["weights_name"] = normalized_relative_path(source["weights_name"], label + ".source.weights_name")
        if Path(source["weights_name"]).name != source["weights_name"]:
            raise ConversionError("%s source weights must be in checkpoint root" % label)
        source["weights_size"] = uint64_value(source["weights_size"], label + ".source.weights_size")
        if not isinstance(source["weights_sha256"], str) or not SHA256_RE.fullmatch(source["weights_sha256"]):
            raise ConversionError("%s.source.weights_sha256 is invalid" % label)
        expected_name = "model.safetensors" if source["format"] == "safetensors" else "pytorch_model.bin"
        if source["weights_name"] != expected_name:
            raise ConversionError("%s source format/name differ" % label)
        profile["source"] = source
        code = object_value(profile["remote_code"], label + ".remote_code")
        if not code or any(
            not PYTHON_FILE_RE.fullmatch(name)
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
            for name, digest in code.items()
        ):
            raise ConversionError("%s.remote_code must pin Python source hashes" % label)
        required = object_value(profile["config_required"], label + ".config_required")
        if not required:
            raise ConversionError("%s.config_required must be nonempty" % label)
        profile["config_required"] = required
        tokenizer = object_value(profile["tokenizer"], label + ".tokenizer")
        if set(tokenizer) != TOKENIZER_KEYS:
            raise ConversionError("%s.tokenizer fields differ" % label)
        for key in ("compiler_manifest_sha256", "compiled_asset_sha256"):
            if not isinstance(tokenizer[key], str) or not SHA256_RE.fullmatch(tokenizer[key]):
                raise ConversionError("%s.tokenizer.%s is invalid" % (label, key))
        tokenizer["compiled_asset_size"] = uint64_value(tokenizer["compiled_asset_size"], label + ".tokenizer.compiled_asset_size")
        tokenizer["emitted_vocab_size"] = uint64_value(tokenizer["emitted_vocab_size"], label + ".tokenizer.emitted_vocab_size")
        if tokenizer["kind"] not in ("bpe", "single-nucleotide") or tokenizer["padding_side"] not in ("left", "right") or not isinstance(tokenizer["add_special_tokens"], bool):
            raise ConversionError("%s.tokenizer behavior is unsupported" % label)
        profile["tokenizer"] = tokenizer
        profile["topology"] = validate_topology(profile["topology"], label + ".topology")
        if tokenizer["emitted_vocab_size"] > profile["topology"]["vocab_size"]:
            raise ConversionError("%s tokenizer emits outside model vocabulary" % label)
        identity = (profile["repo"], profile["revision"])
        if profile["runtime_id"] in profiles or identity in identities:
            raise ConversionError("Mamba profiles duplicate runtime/source identity")
        profiles[profile["runtime_id"]] = profile
        identities.add(identity)
    return profiles, payload


def load_catalog_entry(source_path: Path, profile: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    root, payload = load_json(source_path, "GENEB catalog")
    if root.get("schema_version") != 1 or not isinstance(root.get("models"), list):
        raise ConversionError("GENEB catalog schema is unsupported")
    matches = [value for value in root["models"] if isinstance(value, dict) and value.get("runtime_id") == profile["runtime_id"]]
    if len(matches) != 1:
        raise ConversionError("GENEB catalog must contain exactly one %s" % profile["runtime_id"])
    entry = matches[0]
    expected = {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "architecture": profile["catalog_architecture"],
        "family": "mamba",
    }
    wrong = {key: (entry.get(key), value) for key, value in expected.items() if entry.get(key) != value}
    if wrong:
        raise ConversionError("GENEB catalog Mamba identity differs: %s" % wrong)
    source = entry.get("source")
    if (
        not isinstance(source, dict) or source.get("kind") != "huggingface"
        or source.get("repo") != profile["repo"] or source.get("revision") != profile["revision"]
        or not isinstance(source.get("requested_revision"), str) or not source.get("requested_revision")
        or source.get("immutable") is not True
    ):
        raise ConversionError("GENEB catalog Mamba source differs")
    topology = profile["topology"]
    context = entry.get("context")
    declared = None if topology["max_seqlen"] == 0 else topology["max_seqlen"]
    presets = entry.get("embedding_presets")
    tokenizer = entry.get("tokenizer")
    special = "include" if profile["tokenizer"]["add_special_tokens"] else "none"
    if (
        not isinstance(context, dict) or context.get("declared_max_tokens") != declared
        or not isinstance(presets, dict)
        or any(
            not isinstance(presets.get(name), dict)
            or presets[name].get("output_width") != topology["output_width"]
            or presets[name].get("pooling") != "attention-mask-mean"
            or presets[name].get("special_tokens") != special
            or presets[name].get("mask_domain") != "attention-mask"
            for name in ("reference", "normalized")
        )
        or not isinstance(tokenizer, dict)
        or tokenizer.get("kind") != profile["tokenizer"]["kind"]
        or tokenizer.get("add_special_tokens") != profile["tokenizer"]["add_special_tokens"]
        or tokenizer.get("padding_side") != profile["tokenizer"]["padding_side"]
    ):
        raise ConversionError("GENEB catalog Mamba embedding/tokenizer contract differs")
    return entry, root, payload


def validate_receipt(
    source_path: Path,
    profile: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
    catalog_path: Path,
    catalog_payload: bytes,
) -> Tuple[Dict[str, Path], bytes]:
    receipt, payload = load_json(source_path, "source receipt")
    exact_keys(
        receipt,
        ["schema_version", "kind", "model_id", "repo", "requested_revision", "resolved_revision", "files"],
        ["load_path", "source_kind", "catalog_path", "catalog_sha256", "catalog_contract_sha256"],
        "source receipt",
    )
    validate_hf_fetch_receipt_provenance(
        receipt, catalog_path, catalog_payload, catalog_entry
    )
    catalog_source = catalog_entry.get("source")
    requested = catalog_source.get("requested_revision") if isinstance(catalog_source, dict) else None
    if (
        receipt["schema_version"] != 1 or receipt["kind"] != "source-checkpoint"
        or receipt["model_id"] != profile["runtime_id"] or receipt["repo"] != profile["repo"]
        or receipt["requested_revision"] != requested or receipt["resolved_revision"] != profile["revision"]
        or ("load_path" in receipt and receipt["load_path"] is not None)
        or ("source_kind" in receipt and receipt["source_kind"] != "huggingface")
    ):
        raise ConversionError("source receipt identity/revision is not pinned")
    if not isinstance(receipt["files"], list) or not receipt["files"]:
        raise ConversionError("source receipt files must be a nonempty array")
    paths = {}  # type: Dict[str, Path]
    verified = {}  # type: Dict[str, Tuple[int, str]]
    for index, raw in enumerate(receipt["files"]):
        label = "source receipt files[%d]" % index
        item = object_value(raw, label)
        exact_keys(item, ["name", "size", "sha256", "path"], [], label)
        name = normalized_relative_path(item["name"], label + ".name")
        if name in paths:
            raise ConversionError("%s name must be unique" % label)
        size = uint64_value(item["size"], label + ".size", False)
        digest = nonempty_string(item["sha256"], label + ".sha256")
        if not SHA256_RE.fullmatch(digest):
            raise ConversionError("%s.sha256 is invalid" % label)
        actual_path = Path(nonempty_string(item["path"], label + ".path")).resolve()
        try:
            actual_size = actual_path.stat().st_size
        except OSError as error:
            raise ConversionError("cannot stat receipt asset %s: %s" % (name, error))
        actual_digest = sha256_file(actual_path)
        if (actual_size, actual_digest) != (size, digest):
            raise ConversionError("source receipt integrity mismatch for %s" % name)
        paths[name] = actual_path
        verified[name] = (size, digest)
    expected_names = {"config.json", profile["source"]["weights_name"]} | set(profile["remote_code"])
    if not expected_names.issubset(paths):
        raise ConversionError(
            "Mamba source assets are missing: %s"
            % sorted(expected_names - set(paths))
        )
    if verified[profile["source"]["weights_name"]] != (
        profile["source"]["weights_size"], profile["source"]["weights_sha256"]
    ):
        raise ConversionError("Mamba source weight size/SHA256 differ from profile")
    for name, digest in profile["remote_code"].items():
        if verified[name][1] != digest:
            raise ConversionError("remote code SHA256 differs for %s" % name)
    return paths, payload


def validate_config(source_path: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
    config, payload = load_json(source_path, "HF config")
    digest = sha256_bytes(payload)
    if digest != profile["config_sha256"]:
        raise ConversionError("HF config SHA256 differs from pinned profile: got %s" % digest)
    wrong = {
        key: (config.get(key), expected)
        for key, expected in profile["config_required"].items()
        if key not in config or config[key] != expected
    }
    if wrong:
        raise ConversionError("HF config frozen semantic fields differ: %s" % wrong)
    topology = profile["topology"]
    direct = {
        "vocab_size": config.get("vocab_size"),
        "width": config.get("d_model"),
        "num_layers": config.get("n_layer"),
        "norm_epsilon": config.get("norm_epsilon", config.get("layer_norm_epsilon")),
    }
    expected_direct = {key: topology[key] for key in direct}
    if direct != expected_direct or config.get("torch_dtype") != "float32":
        raise ConversionError("HF config direct topology/dtype differs")
    if topology["variant"] == "caduceus-mamba1":
        ssm = config.get("ssm_cfg")
        derived = (
            isinstance(ssm, dict)
            and ssm.get("expand") == 2
            and ssm.get("d_state") == topology["state_width"]
            and ssm.get("d_conv") == topology["conv_width"]
            and ssm.get("dt_rank") == "auto"
            and config.get("bidirectional") is True
            and config.get("bidirectional_strategy") == "add"
            and config.get("bidirectional_weight_tie") is True
            and config.get("rcps") == topology["rcps"]
            and config.get("rms_norm") is True
            and config.get("fused_add_norm") is True
        )
    else:
        derived = (
            config.get("ssm_cfg") == {"layer": "Mamba2"}
            and config.get("expand") == 2
            and config.get("rms_norm") is True
            and config.get("fused_add_norm") is True
            and config.get("d_intermediate") == 0
        )
    if not derived:
        raise ConversionError("HF config does not select the frozen Mamba implementation")
    return config


def validate_tokenizer_descriptor(descriptor_path: Path, tokenizer_root: Optional[Path], artifact_root: Path, profile: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    descriptor, payload = load_json(descriptor_path, "tokenizer descriptor")
    if set(descriptor) != TOKENIZER_DESCRIPTOR_KEYS:
        raise ConversionError("tokenizer descriptor fields differ")
    expected = profile["tokenizer"]
    if (
        descriptor["converter.schema"] != "evo-tokenizer-conversion-receipt"
        or descriptor["converter.version"] != 1
        or descriptor["tokenizer.profile"] != TOKENIZER_PROFILE
        or descriptor["compiler_manifest_sha256"] != expected["compiler_manifest_sha256"]
        or descriptor["tokenizer.sha256"] != expected["compiled_asset_sha256"]
        or descriptor["tokenizer.size"] != expected["compiled_asset_size"]
    ):
        raise ConversionError("tokenizer descriptor differs from pinned compiler output")
    for key in ("source_receipt_contract_sha256", "tokenizer.sha256", "compiler_manifest_sha256"):
        if not isinstance(descriptor[key], str) or not SHA256_RE.fullmatch(descriptor[key]):
            raise ConversionError("tokenizer descriptor %s is invalid" % key)
    relative = normalized_relative_path(descriptor["tokenizer.path"], "tokenizer.path")
    root = (tokenizer_root if tokenizer_root is not None else descriptor_path.parent).resolve()
    asset = (root / relative).resolve()
    try:
        asset.relative_to(root)
    except ValueError:
        raise ConversionError("tokenizer.path escapes tokenizer root")
    if (artifact_root.resolve() / relative).resolve() != asset:
        raise ConversionError("tokenizer asset must already be staged relative to runtime artifact")
    try:
        actual_size = asset.stat().st_size
    except OSError as error:
        raise ConversionError("cannot stat tokenizer asset: %s" % error)
    if actual_size != descriptor["tokenizer.size"] or sha256_file(asset) != descriptor["tokenizer.sha256"]:
        raise ConversionError("tokenizer asset integrity mismatch")
    projected = {key: descriptor[key] for key in ("tokenizer.profile", "tokenizer.path", "tokenizer.sha256", "tokenizer.size")}
    return projected, sha256_bytes(payload)


def _mamba1_specs(prefix: str, topology: Mapping[str, Any], tied_projection: bool) -> List[TensorSpec]:
    inner = topology["inner_width"]
    width = topology["width"]
    state = topology["state_width"]
    rank = topology["time_step_rank"]
    specs = []  # type: List[TensorSpec]
    if not tied_projection:
        specs.append(TensorSpec(prefix + "in_proj.weight", "F32", (inner * 2, width)))
    specs.extend(
        [
            TensorSpec(prefix + "conv1d.weight", "F32", (inner, 1, topology["conv_width"])),
            TensorSpec(prefix + "conv1d.bias", "F32", (inner,)),
            TensorSpec(prefix + "x_proj.weight", "F32", (rank + state * 2, inner)),
            TensorSpec(prefix + "dt_proj.weight", "F32", (inner, rank)),
            TensorSpec(prefix + "dt_proj.bias", "F32", (inner,)),
            TensorSpec(prefix + "A_log", "F32", (inner, state)),
            TensorSpec(prefix + "D", "F32", (inner,)),
        ]
    )
    if not tied_projection:
        specs.append(TensorSpec(prefix + "out_proj.weight", "F32", (width, inner)))
    return specs


def canonical_tensor_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    width = topology["width"]
    specs = []  # type: List[TensorSpec]
    if topology["variant"] == "caduceus-mamba1":
        specs.append(TensorSpec("embedding.weight", "F32", (topology["vocab_size"], width)))
        for layer in range(topology["num_layers"]):
            prefix = "layers.%d." % layer
            specs.append(TensorSpec(prefix + "norm.weight", "F32", (width,)))
            specs.extend(_mamba1_specs(prefix + "forward.", topology, False))
            specs.extend(_mamba1_specs(prefix + "reverse.", topology, True))
        specs.append(TensorSpec("final_norm.weight", "F32", (width,)))
        return specs

    inner = topology["inner_width"]
    state = topology["state_width"]
    grouped_state = topology["groups"] * state
    specs.append(TensorSpec("token_embedding.weight", "F32", (topology["vocab_size"], width)))
    for direction in ("forward", "reverse"):
        for layer in range(topology["num_layers"]):
            prefix = "%s.layers.%d." % (direction, layer)
            specs.extend(
                [
                    TensorSpec(prefix + "norm.weight", "F32", (width,)),
                    TensorSpec(prefix + "mixer.in_proj.weight", "F32", (inner * 2 + grouped_state * 2 + topology["heads"], width)),
                    TensorSpec(prefix + "mixer.conv1d.weight", "F32", (inner + grouped_state * 2, 1, topology["conv_width"])),
                    TensorSpec(prefix + "mixer.conv1d.bias", "F32", (inner + grouped_state * 2,)),
                    TensorSpec(prefix + "mixer.dt_bias", "F32", (topology["heads"],)),
                    TensorSpec(prefix + "mixer.A_log", "F32", (topology["heads"],)),
                    TensorSpec(prefix + "mixer.D", "F32", (topology["heads"],)),
                    TensorSpec(prefix + "mixer.norm.weight", "F32", (inner,)),
                    TensorSpec(prefix + "mixer.out_proj.weight", "F32", (width, inner)),
                    TensorSpec(prefix + "norm2.weight", "F32", (width,)),
                    TensorSpec(prefix + "mlp.fc1.weight", "F32", (topology["mlp_width"] * 2, width)),
                    TensorSpec(prefix + "mlp.fc2.weight", "F32", (width, topology["mlp_width"])),
                ]
            )
        specs.append(TensorSpec(direction + ".final_norm.weight", "F32", (width,)))
    specs.append(TensorSpec("projection.weight", "F32", (width, width * 2)))
    return specs


def caduceus_source_contract(profile: Mapping[str, Any]) -> Tuple[List[TensorSpec], Dict[str, str], Dict[str, str], Dict[str, List[int]]]:
    topology = profile["topology"]
    width = topology["width"]
    inner = topology["inner_width"]
    state = topology["state_width"]
    rank = topology["time_step_rank"]
    rcps = topology["rcps"]
    source_specs = []  # type: List[TensorSpec]
    mapping = {}  # type: Dict[str, str]
    aliases = {}  # type: Dict[str, str]
    buffers = {}  # type: Dict[str, List[int]]
    embedding = (
        "caduceus.backbone.embeddings.word_embeddings.embedding.weight"
        if rcps else "caduceus.backbone.embeddings.word_embeddings.weight"
    )
    source_specs.append(TensorSpec(embedding, "F32", (topology["vocab_size"], width)))
    mapping[embedding] = "embedding.weight"
    mixer_suffixes = (
        ("in_proj.weight", (inner * 2, width)),
        ("conv1d.weight", (inner, 1, topology["conv_width"])),
        ("conv1d.bias", (inner,)),
        ("x_proj.weight", (rank + state * 2, inner)),
        ("dt_proj.weight", (inner, rank)),
        ("dt_proj.bias", (inner,)),
        ("A_log", (inner, state)),
        ("D", (inner,)),
        ("out_proj.weight", (width, inner)),
    )
    for layer in range(topology["num_layers"]):
        layer_source = "caduceus.backbone.layers.%d." % layer
        layer_runtime = "layers.%d." % layer
        norm_name = layer_source + "norm.weight"
        source_specs.append(TensorSpec(norm_name, "F32", (width,)))
        mapping[norm_name] = layer_runtime + "norm.weight"
        mixer_source = layer_source + ("mixer.submodule." if rcps else "mixer.")
        for direction, runtime_direction in (("mamba_fwd.", "forward."), ("mamba_rev.", "reverse.")):
            for suffix, shape in mixer_suffixes:
                source_name = mixer_source + direction + suffix
                tied = runtime_direction == "reverse." and suffix in ("in_proj.weight", "out_proj.weight")
                if tied and profile["source"]["format"] == "safetensors":
                    continue
                source_specs.append(TensorSpec(source_name, "F32", shape))
                if tied:
                    aliases[source_name] = mixer_source + "mamba_fwd." + suffix
                else:
                    mapping[source_name] = layer_runtime + runtime_direction + suffix
    final_name = "caduceus.backbone.norm_f.weight"
    source_specs.append(TensorSpec(final_name, "F32", (width,)))
    mapping[final_name] = "final_norm.weight"
    if rcps:
        for name in (
            "caduceus.backbone.embeddings.word_embeddings.complement_map",
            "lm_head.complement_map",
        ):
            source_specs.append(TensorSpec(name, "I64", (topology["vocab_size"],)))
            buffers[name] = list(topology["complement_map"])
    if profile["source"]["format"] == "pytorch-bin":
        alias = "lm_head.lm_head.weight"
        source_specs.append(TensorSpec(alias, "F32", (topology["vocab_size"], width)))
        aliases[alias] = embedding
    return source_specs, mapping, aliases, buffers


def ecc_source_contract(profile: Mapping[str, Any]) -> Tuple[List[TensorSpec], Dict[str, str], Dict[str, str], Dict[str, List[int]]]:
    topology = profile["topology"]
    width = topology["width"]
    inner = topology["inner_width"]
    state = topology["state_width"]
    grouped_state = topology["groups"] * state
    source_specs = []  # type: List[TensorSpec]
    mapping = {}  # type: Dict[str, str]
    embedding = "token_embedding.weight"
    source_specs.append(TensorSpec(embedding, "F32", (topology["vocab_size"], width)))
    mapping[embedding] = "token_embedding.weight"
    for source_direction, runtime_direction in (("mamba_forward", "forward"), ("mamba_backward", "reverse")):
        unused_embedding = source_direction + ".backbone.embedding.weight"
        source_specs.append(TensorSpec(unused_embedding, "F32", (topology["vocab_size"], width)))
        for layer in range(topology["num_layers"]):
            source_prefix = "%s.backbone.layers.%d." % (source_direction, layer)
            runtime_prefix = "%s.layers.%d." % (runtime_direction, layer)
            fields = (
                ("norm.weight", (width,)),
                ("mixer.in_proj.weight", (inner * 2 + grouped_state * 2 + topology["heads"], width)),
                ("mixer.conv1d.weight", (inner + grouped_state * 2, 1, topology["conv_width"])),
                ("mixer.conv1d.bias", (inner + grouped_state * 2,)),
                ("mixer.dt_bias", (topology["heads"],)),
                ("mixer.A_log", (topology["heads"],)),
                ("mixer.D", (topology["heads"],)),
                ("mixer.norm.weight", (inner,)),
                ("mixer.out_proj.weight", (width, inner)),
                ("norm2.weight", (width,)),
                ("mlp.fc1.weight", (topology["mlp_width"] * 2, width)),
                ("mlp.fc2.weight", (width, topology["mlp_width"])),
            )
            for suffix, shape in fields:
                name = source_prefix + suffix
                source_specs.append(TensorSpec(name, "F32", shape))
                mapping[name] = runtime_prefix + suffix
        final_name = source_direction + ".backbone.norm_f.weight"
        source_specs.append(TensorSpec(final_name, "F32", (width,)))
        mapping[final_name] = runtime_direction + ".final_norm.weight"
        source_specs.append(TensorSpec(source_direction + ".lm_head.weight", "F32", (topology["vocab_size"], width)))
    projection = "lm_head_proj.weight"
    source_specs.append(TensorSpec(projection, "F32", (width, width * 2)))
    mapping[projection] = "projection.weight"
    return source_specs, mapping, {}, {}


def source_contract(profile: Mapping[str, Any]) -> Tuple[List[TensorSpec], Dict[str, str], Dict[str, str], Dict[str, List[int]]]:
    if profile["topology"]["variant"] == "caduceus-mamba1":
        return caduceus_source_contract(profile)
    return ecc_source_contract(profile)


def read_source_tensors(paths: Mapping[str, Path], profile: Mapping[str, Any]) -> List[Any]:
    weight_name = profile["source"]["weights_name"]
    allowed = ("F32", "I64") if profile["topology"]["rcps"] else ("F32",)
    if profile["source"]["format"] == "safetensors":
        return read_hf_safetensors({weight_name: paths[weight_name]}, allowed_dtypes=allowed)
    return load_hf_torch_checkpoints({weight_name: paths[weight_name]}, allowed_dtypes=allowed)


def tensor_digest(tensor: Any) -> str:
    digest = hashlib.sha256()
    written = 0
    for raw in tensor.iter_chunks(CHUNK_SIZE):
        chunk = memoryview(raw).cast("B")
        digest.update(chunk)
        written += len(chunk)
    if written != tensor.nbytes:
        raise ConversionError("tensor %s yielded the wrong byte count" % tensor.name)
    return digest.hexdigest()


def tensor_bytes(tensor: Any, maximum: int) -> bytes:
    if tensor.nbytes > maximum:
        raise ConversionError("validation tensor %s is unexpectedly large" % tensor.name)
    result = bytearray()
    for raw in tensor.iter_chunks(CHUNK_SIZE):
        result.extend(memoryview(raw).cast("B"))
    if len(result) != tensor.nbytes:
        raise ConversionError("validation tensor %s is truncated" % tensor.name)
    return bytes(result)


@dataclasses.dataclass(frozen=True)
class AliasedTensorSource:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    nbytes: int
    source: Any

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        return self.source.iter_chunks(chunk_size)


def validated_runtime_tensors(source_tensors: Sequence[Any], profile: Mapping[str, Any]) -> List[AliasedTensorSource]:
    expected_source, mapping, aliases, buffers = source_contract(profile)
    ordered = validate_tensor_manifest(source_tensors, expected_source)
    by_name = {tensor.name: tensor for tensor in ordered}
    for alias_name, canonical_name in aliases.items():
        if tensor_digest(by_name[alias_name]) != tensor_digest(by_name[canonical_name]):
            raise ConversionError("tied source tensor differs: %s" % alias_name)
    for name, values in buffers.items():
        raw = tensor_bytes(by_name[name], len(values) * 8)
        actual = list(struct.unpack("<%dq" % len(values), raw))
        if actual != values:
            raise ConversionError("RCPS complement buffer differs: %s" % name)
    runtime = [
        AliasedTensorSource(mapping[tensor.name], tensor.dtype, tuple(tensor.shape), tensor.nbytes, tensor)
        for tensor in ordered if tensor.name in mapping
    ]
    expected_runtime = canonical_tensor_specs(profile["topology"])
    actual_by_name = {tensor.name: tensor for tensor in runtime}
    if len(actual_by_name) != len(runtime) or set(actual_by_name) != {spec.name for spec in expected_runtime}:
        raise ConversionError("internal runtime tensor mapping is not closed")
    for spec in expected_runtime:
        actual = actual_by_name[spec.name]
        if actual.dtype != spec.dtype or actual.shape != spec.shape:
            raise ConversionError("runtime tensor dtype/shape mapping differs: %s" % spec.name)
    return [actual_by_name[spec.name] for spec in expected_runtime]


def _encode_metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "b:%d" % int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or value > UINT64_MAX:
            raise ConversionError("metadata integer is outside uint64")
        return "u:%d" % value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConversionError("metadata float is not finite")
        return "f:%016x" % struct.unpack("<Q", struct.pack("<d", value))[0]
    if isinstance(value, str):
        return "s:" + value
    raise ConversionError("unsupported metadata value type: %s" % type(value).__name__)


def encoded_header(metadata: Mapping[str, Any], tensors: Sequence[Any]) -> bytes:
    encoded = {}  # type: Dict[str, str]
    for key in sorted(metadata):
        if not KEY_RE.fullmatch(key) or len(key.encode("ascii")) > 255 or key == "runtime.profile":
            raise ConversionError("invalid/reserved artifact metadata key %r" % key)
        encoded[key] = _encode_metadata_value(metadata[key])
    encoded["runtime.profile"] = "s:" + ARTIFACT_PROFILE
    root = {"__metadata__": dict(sorted(encoded.items()))}  # type: Dict[str, Any]
    offset = 0
    for tensor in tensors:
        if tensor.dtype != "F32":
            raise ConversionError("runtime tensor %s is not F32" % tensor.name)
        end = offset + tensor.nbytes
        if end > UINT64_MAX:
            raise ConversionError("artifact tensor data exceeds uint64")
        root[tensor.name] = {"dtype": tensor.dtype, "shape": list(tensor.shape), "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    if not header or len(header) > MAX_HEADER_SIZE:
        raise ConversionError("runtime Safetensors header exceeds 16 MiB")
    return header


def write_artifact(output_path: Path, metadata: Mapping[str, Any], tensors: Sequence[Any], force: bool) -> None:
    if output_path.suffix != ".safetensors" or not tensors:
        raise ConversionError("output must be a nonempty .safetensors artifact")
    header = encoded_header(metadata, tensors)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError("output already exists: %s" % output_path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % output_path.name, suffix=".tmp", dir=str(output_path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as output:
            total_size = 8 + len(header) + sum(tensor.nbytes for tensor in tensors)
            if total_size > UINT64_MAX:
                raise ConversionError("runtime artifact exceeds uint64")
            output.truncate(total_size)
            output.seek(0)
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for index, tensor in enumerate(tensors, start=1):
                print("[%d/%d] %s" % (index, len(tensors), tensor.name), file=sys.stderr)
                written = 0
                for raw in tensor.iter_chunks(CHUNK_SIZE):
                    chunk = memoryview(raw).cast("B")
                    if written + len(chunk) > tensor.nbytes:
                        raise ConversionError("tensor %s yielded too many bytes" % tensor.name)
                    output.write(chunk)
                    written += len(chunk)
                if written != tensor.nbytes:
                    raise ConversionError("tensor %s yielded the wrong byte count" % tensor.name)
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


def build_metadata(profile: Mapping[str, Any], tokenizer: Mapping[str, Any], receipt_sha256: str, catalog_sha256: str, profile_sha256: str, tokenizer_descriptor_sha256: str, geneb_metadata: Mapping[str, Any]) -> Dict[str, Any]:
    topology = profile["topology"]
    complement = ",".join(str(value) for value in topology["complement_map"])
    metadata = {
        "runtime.abi": RUNTIME_ABI,
        "runtime.embedding_layer_count": topology["num_layers"] + 1,
        "runtime.tokenizer_vocabulary_size": profile["tokenizer"]["emitted_vocab_size"],
        "model.id": profile["runtime_id"],
        "model.architecture": RUNTIME_ARCHITECTURE,
        "config.vocab_size": topology["vocab_size"],
        "config.hidden_size": topology["output_width"],
        "config.num_layers": topology["num_layers"],
        "config.max_seqlen": topology["max_seqlen"],
        "mamba.variant": topology["variant"],
        "mamba.vocab_size": topology["vocab_size"],
        "mamba.width": topology["width"],
        "mamba.output_width": topology["output_width"],
        "mamba.layers": topology["num_layers"],
        "mamba.max_sequence_length": topology["max_seqlen"],
        "mamba.advertised_training_length": topology["advertised_training_length"],
        "mamba.inner_width": topology["inner_width"],
        "mamba.state_width": topology["state_width"],
        "mamba.conv_width": topology["conv_width"],
        "mamba.time_step_rank": topology["time_step_rank"],
        "mamba.mlp_width": topology["mlp_width"],
        "mamba.head_width": topology["head_width"],
        "mamba.heads": topology["heads"],
        "mamba.groups": topology["groups"],
        "mamba.norm_epsilon": topology["norm_epsilon"],
        "mamba.rcps": topology["rcps"],
        "mamba.complement_map": complement,
        "mamba.hidden_tap": "post-final-norm",
        "mamba.pooling": "attention-mask-mean",
        "mamba.mask_domain": "attention-mask",
        "mamba.special_tokens": "include" if profile["tokenizer"]["add_special_tokens"] else "none",
        "source.repo": profile["repo"],
        "source.revision": profile["revision"],
        "source.weights.sha256": profile["source"]["weights_sha256"],
        "source.weights.size": profile["source"]["weights_size"],
        "source.receipt_sha256": receipt_sha256,
        "source.catalog_contract_sha256": catalog_sha256,
        "source.converter_profile_contract_sha256": profile_sha256,
        "source.config_sha256": profile["config_sha256"],
        "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        "source.implementation_contract": IMPLEMENTATION_CONTRACTS[topology["variant"]],
    }  # type: Dict[str, Any]
    for name, digest in sorted(profile["remote_code"].items()):
        metadata["source.remote." + name[:-3] + ".sha256"] = digest
    metadata.update(geneb_metadata)
    metadata.update(tokenizer)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=default_config_path("geneb-models.json"))
    parser.add_argument("--profiles", type=Path, default=default_config_path("geneb-mamba-models.json"))
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, _profile_payload = load_profiles(args.profiles.resolve())
        receipt_identity, _ = load_json(args.receipt.resolve(), "source receipt identity")
        model_id = receipt_identity.get("model_id")
        profile = profiles.get(model_id) if isinstance(model_id, str) else None
        if profile is None:
            raise ConversionError("source receipt does not identify a Mamba profile")
        catalog_entry, catalog_root, catalog_payload = load_catalog_entry(args.catalog.resolve(), profile)
        paths, receipt_payload = validate_receipt(
            args.receipt.resolve(),
            profile,
            catalog_entry,
            args.catalog.resolve(),
            catalog_payload,
        )
        validate_config(paths["config.json"], profile)
        tokenizer, tokenizer_descriptor_sha256 = validate_tokenizer_descriptor(
            args.tokenizer_descriptor.resolve(),
            args.tokenizer_root.resolve() if args.tokenizer_root is not None else None,
            args.output.resolve().parent,
            profile,
        )
        source_tensors = read_source_tensors(paths, profile)
        runtime_tensors = validated_runtime_tensors(source_tensors, profile)
        geneb_metadata = build_geneb_artifact_metadata(catalog_root, catalog_entry, catalog_payload)
        metadata = build_metadata(
            profile, tokenizer, sha256_bytes(receipt_payload),
            catalog_contract_sha256(catalog_root, catalog_entry),
            converter_profile_contract_sha256(
                1,
                PROFILE_FORMAT,
                profile,
                {"implementation_contracts": IMPLEMENTATION_CONTRACTS},
            ),
            tokenizer_descriptor_sha256,
            geneb_metadata,
        )
        write_artifact(args.output, metadata, runtime_tensors, args.force)
        print("wrote %s" % args.output)
        print("source_receipt_sha256=%s" % sha256_bytes(receipt_payload))
        print("variant=%s" % profile["topology"]["variant"])
        return 0
    except (
        CheckpointError, ConversionError, FileExistsError, GenebArtifactError,
        ImportError, OSError, ValueError,
    ) as error:
        print("convert_geneb_mamba_checkpoint: error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
