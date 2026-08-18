#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile pinned manual JanusDNA Lightning checkpoints for native CPU."""

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


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
)
from evo.hf_checkpoint import (  # noqa: E402
    CheckpointError,
    TensorSpec,
    load_json,
    normalized_relative_path,
    tensor_nbytes,
)


ARTIFACT_PROFILE = "geneb-janusdna-runtime-v1"
RUNTIME_ABI = "geneb-janusdna-lightning-v1"
RUNTIME_ARCHITECTURE = "GenebJanusDnaEncoder"
PROFILE_FORMAT = "geneb-janusdna-converter-v1"
TOKENIZER_PROFILE = "evo-tokenizer-v1"
DATAVERSE_URL = (
    "https://dataverse.harvard.edu/dataset.xhtml?"
    "persistentId=doi%3A10.7910%2FDVN%2FHDT0RN"
)
MAX_HEADER_SIZE = 16 * 1024 * 1024
CHUNK_SIZE = 16 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
KEY_RE = re.compile(r"[A-Za-z0-9._-]+")

IMPLEMENTATION_CONTRACT = {
    "artifact_profile": ARTIFACT_PROFILE,
    "runtime_abi": RUNTIME_ABI,
    "architecture": RUNTIME_ARCHITECTURE,
    "source_revision": "3f449542529a2948d73062514cf9844705a4277a",
    "modeling_sha256": "5e23060319e5ab8ef5ead566ceca6d49c7c89f9df0aae808f9399cacbbe00a68",
    "configuration_sha256": "431b37095d6d30df6ac4d5a73e6b1c9d00405c667fca132c7258e7ac378bdc7c",
    "tokenizer_sha256": "a85b0ee68a4764a3e27c11910972dc5ffa737204aa298b1e9cc798c18228da7e",
    "extractor_commit": "b465d2d6a11efbbc9a22c105e34832725ce50e05",
    "extractor_sha256": "2e5bb3f8f5c95bdaebdbe2730ed5590d64a10d298a40837760ec4d05dbf36cda",
}
EXPECTED_TOKENIZER = {
    "compiler_manifest_sha256": "081562dc14537f1570280cd389b77fd6af696f1e759351d2662622e50015ef46",
    "compiled_asset_sha256": "952e9c77e70ca90e76d9d6164b556b83ffcb9e74d67f82be79c123b4b1358e49",
    "compiled_asset_size": 659,
    "kind": "single-nucleotide",
    "emitted_vocab_size": 12,
    "add_special_tokens": False,
    "padding_side": "right",
    "pad_token_id": 4,
}
EXPECTED_PROFILE_ROWS = {
    "geneb-janusdna-72-w": {
        "geneb_model_id": "JanusDNA_72_w",
        "paper_name": "JanusDNA-72-w",
        "catalog_architecture": "janusdna-mamba-attention-moe",
        "source": {
            "format": "pytorch-lightning",
            "checkpoint_name": "72_with_midattn.ckpt",
            "checkpoint_size": 108242317,
            "checkpoint_sha256": "94790a3f6d6e719cb63e2c26190bb4968b50a06bdee5aa01ca48b57932c9bc3e",
            "config_name": "72_with_midattn_model_config.json",
            "config_size": 1608,
            "config_sha256": "ba89cced27cc2a868c3e79e11b2e3479f31c56e520d3d55c893d5b4274e6346f",
            "state_entries": 639,
            "clean_entries": 636,
            "shape_manifest_sha256": "5b8f64e8e74bbb86d3c2715b75c9ea46e61922f439a11855272a42c810369ecc",
            "metric_values": {
                "test_torchmetrics.num_tokens.count": 127139840,
                "train_torchmetrics.num_tokens.count": 1310720000,
                "val_torchmetrics.num_tokens.count": 144965632,
            },
        },
    },
    "geneb-janusdna-72-wo": {
        "geneb_model_id": "JanusDNA_72_wo",
        "paper_name": "JanusDNA-72-wo",
        "catalog_architecture": "janusdna-mamba-moe",
        "source": {
            "format": "pytorch-lightning",
            "checkpoint_name": "72_without_midattn.ckpt",
            "checkpoint_size": 108361237,
            "checkpoint_sha256": "015cd69eb0157eada2ed706677e64274a3cc53c2d76cfad6e1004e065ac033df",
            "config_name": "72_without_midattn_model_config.json",
            "config_size": 1610,
            "config_sha256": "0fe6f1e8e54f2a95cf045c511510ffb930d76ac2b93fdc657f7277864a58df9f",
            "state_entries": 655,
            "clean_entries": 652,
            "shape_manifest_sha256": "a3cc604234b268252fe59cb7302239016b3778ae2927ee0b5bfd89b2115a89d5",
            "metric_values": {
                "test_torchmetrics.num_tokens.count": 127139840,
                "train_torchmetrics.num_tokens.count": 1310720000,
                "val_torchmetrics.num_tokens.count": 144965632,
            },
        },
    },
}
EXPECTED_UNSAFE_GLOBALS = {
    "collections.defaultdict",
    "builtins.dict",
    "builtins.int",
    "omegaconf.base.Metadata",
    "omegaconf.dictconfig.DictConfig",
    "omegaconf.nodes.AnyNode",
    "omegaconf.base.ContainerMetadata",
    "omegaconf.listconfig.ListConfig",
    "builtins.list",
    "typing.Any",
}
TOP_LEVEL_TYPES = {
    "epoch": "builtins.int",
    "global_step": "builtins.int",
    "pytorch-lightning_version": "builtins.str",
    "state_dict": "collections.OrderedDict",
    "loops": "builtins.dict",
    "callbacks": "builtins.dict",
    "optimizer_states": "builtins.list",
    "lr_schedulers": "builtins.list",
    "hparams_name": "builtins.str",
    "hyper_parameters": "omegaconf.dictconfig.DictConfig",
    "hparams_type": "abc.ABCMeta",
}
PROFILE_KEYS = {
    "runtime_id",
    "geneb_model_id",
    "paper_name",
    "catalog_architecture",
    "source",
    "config_required",
    "tokenizer",
    "topology",
}
SOURCE_KEYS = {
    "format",
    "checkpoint_name",
    "checkpoint_size",
    "checkpoint_sha256",
    "config_name",
    "config_size",
    "config_sha256",
    "state_entries",
    "clean_entries",
    "shape_manifest_sha256",
    "metric_values",
}
TOKENIZER_KEYS = {
    "compiler_manifest_sha256",
    "compiled_asset_sha256",
    "compiled_asset_size",
    "kind",
    "emitted_vocab_size",
    "add_special_tokens",
    "padding_side",
    "pad_token_id",
}
TOPOLOGY_KEYS = {
    "variant",
    "vocab_size",
    "tokenizer_vocab_size",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "head_dim",
    "flex_head_dim",
    "inner_size",
    "state_size",
    "conv_size",
    "dt_rank",
    "mlp_size",
    "num_experts",
    "experts_per_token",
    "max_seqlen",
    "middle_attention_layer",
    "pad_token_id",
    "norm_epsilon",
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
    """Raised when source material falls outside the frozen JanusDNA ABI."""


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
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > UINT64_MAX
    ):
        raise ConversionError(
            "%s must be a%s uint64" % (label, " positive" if positive else "")
        )
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
    if topology["variant"] not in (
        "with-middle-attention",
        "without-middle-attention",
    ):
        raise ConversionError("%s.variant is unsupported" % label)
    for key in TOPOLOGY_KEYS - {"variant", "norm_epsilon"}:
        topology[key] = uint64_value(topology[key], "%s.%s" % (label, key))
    topology["norm_epsilon"] = finite_float(
        topology["norm_epsilon"], label + ".norm_epsilon"
    )
    production = (
        topology["vocab_size"] == 16
        and topology["tokenizer_vocab_size"] == 12
        and topology["hidden_size"] == 72
        and topology["num_layers"] == 8
        and topology["num_attention_heads"] == 4
        and topology["head_dim"] == 18
        and topology["flex_head_dim"] == 32
        and topology["inner_size"] == 144
        and topology["state_size"] == 16
        and topology["conv_size"] == 4
        and topology["dt_rank"] == 5
        and topology["mlp_size"] == 288
        and topology["num_experts"] == 16
        and topology["experts_per_token"] == 2
        and topology["max_seqlen"] == 1024
        and topology["middle_attention_layer"] == 4
        and topology["pad_token_id"] == 4
    )
    tiny = (
        topology["vocab_size"] == 16
        and topology["tokenizer_vocab_size"] == 12
        and topology["hidden_size"] == 4
        and topology["num_layers"] == 2
        and topology["num_attention_heads"] == 2
        and topology["head_dim"] == 2
        and topology["flex_head_dim"] == 4
        and topology["inner_size"] == 8
        and topology["state_size"] == 2
        and topology["conv_size"] == 2
        and topology["dt_rank"] == 2
        and topology["mlp_size"] == 8
        and topology["num_experts"] == 4
        and topology["experts_per_token"] == 2
        and topology["max_seqlen"] == 8
        and topology["middle_attention_layer"] == 0
        and topology["pad_token_id"] == 4
    )
    if (
        not (production or tiny)
        or topology["norm_epsilon"] != 1.0e-6
        or topology["num_attention_heads"] * topology["head_dim"]
        != topology["hidden_size"]
    ):
        raise ConversionError("%s differs from a closed JanusDNA tuple" % label)
    return topology


def add_mlp(specs: List[TensorSpec], prefix: str, topology: Mapping[str, Any]) -> None:
    width, mlp = topology["hidden_size"], topology["mlp_size"]
    specs.extend(
        [
            TensorSpec(prefix + "gate_proj.weight", "F32", (mlp, width)),
            TensorSpec(prefix + "up_proj.weight", "F32", (mlp, width)),
            TensorSpec(prefix + "down_proj.weight", "F32", (width, mlp)),
        ]
    )


def add_feed_forward(
    specs: List[TensorSpec],
    prefix: str,
    expert_layer: bool,
    topology: Mapping[str, Any],
) -> None:
    if not expert_layer:
        add_mlp(specs, prefix, topology)
        return
    specs.append(
        TensorSpec(
            prefix + "router.weight",
            "F32",
            (topology["num_experts"], topology["hidden_size"]),
        )
    )
    for expert in range(topology["num_experts"]):
        add_mlp(specs, prefix + "experts.%d." % expert, topology)


def add_mamba_direction(
    specs: List[TensorSpec],
    prefix: str,
    topology: Mapping[str, Any],
    tied: bool,
) -> None:
    width, inner = topology["hidden_size"], topology["inner_size"]
    state, rank = topology["state_size"], topology["dt_rank"]
    if not tied:
        specs.append(TensorSpec(prefix + "in_proj.weight", "F32", (inner * 2, width)))
    specs.extend(
        [
            TensorSpec(prefix + "A_log", "F32", (inner, state)),
            TensorSpec(prefix + "D", "F32", (inner,)),
            TensorSpec(
                prefix + "conv1d.weight", "F32", (inner, 1, topology["conv_size"])
            ),
            TensorSpec(prefix + "conv1d.bias", "F32", (inner,)),
            TensorSpec(prefix + "x_proj.weight", "F32", (rank + state * 2, inner)),
            TensorSpec(prefix + "dt_proj.weight", "F32", (inner, rank)),
            TensorSpec(prefix + "dt_proj.bias", "F32", (inner,)),
        ]
    )
    if not tied:
        specs.append(TensorSpec(prefix + "out_proj.weight", "F32", (width, inner)))
    specs.extend(
        [
            TensorSpec(prefix + "dt_layernorm.weight", "F32", (rank,)),
            TensorSpec(prefix + "b_layernorm.weight", "F32", (state,)),
            TensorSpec(prefix + "c_layernorm.weight", "F32", (state,)),
        ]
    )


def add_mamba_layer(
    specs: List[TensorSpec], layer: int, topology: Mapping[str, Any]
) -> None:
    prefix = "layers.%d.mamba_module." % layer
    add_mamba_direction(specs, prefix + "mamba_fwd.", topology, False)
    add_mamba_direction(specs, prefix + "mamba_rev.", topology, True)
    expert_layer = layer % 2 == 1
    for direction in ("fwd", "bwd"):
        specs.extend(
            [
                TensorSpec(
                    prefix + "input_layernorm_%s.weight" % direction,
                    "F32",
                    (topology["hidden_size"],),
                ),
                TensorSpec(
                    prefix + "pre_ff_layernorm_%s.weight" % direction,
                    "F32",
                    (topology["hidden_size"],),
                ),
            ]
        )
        add_feed_forward(
            specs,
            prefix + "feed_forward_%s." % direction,
            expert_layer,
            topology,
        )


def add_attention_layer(
    specs: List[TensorSpec], layer: int, topology: Mapping[str, Any]
) -> None:
    prefix = "layers.%d.attn." % layer
    width = topology["hidden_size"]
    expert_layer = layer % 2 == 1
    for direction in ("fwd", "bwd"):
        specs.extend(
            [
                TensorSpec(
                    prefix + "input_layernorm_%s.weight" % direction, "F32", (width,)
                ),
                TensorSpec(
                    prefix + "pre_ff_layernorm_%s.weight" % direction, "F32", (width,)
                ),
            ]
        )
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            specs.append(
                TensorSpec(
                    prefix + "self_attn_%s.%s.weight" % (direction, projection),
                    "F32",
                    (width, width),
                )
            )
        add_feed_forward(
            specs,
            prefix + "feed_forward_%s." % direction,
            expert_layer,
            topology,
        )


def canonical_tensor_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    width = topology["hidden_size"]
    specs = [TensorSpec("embed_tokens.weight", "F32", (topology["vocab_size"], width))]
    for layer in range(topology["num_layers"]):
        if (
            topology["variant"] == "with-middle-attention"
            and layer == topology["middle_attention_layer"]
        ):
            add_attention_layer(specs, layer, topology)
        else:
            add_mamba_layer(specs, layer, topology)
    specs.append(TensorSpec("final_layernorm.weight", "F32", (width,)))
    for projection in ("q_proj", "k_proj", "v_proj"):
        specs.append(
            TensorSpec(
                "final_attention.self_attn.%s.weight" % projection,
                "F32",
                (width, width),
            )
        )
    specs.append(
        TensorSpec(
            "final_attention.self_attn.o_projs.0.weight",
            "F32",
            (width, width),
        )
    )
    add_mlp(specs, "final_attention.feed_forward.", topology)
    specs.extend(
        [
            TensorSpec("final_attention.input_layernorm.weight", "F32", (width,)),
            TensorSpec("final_attention.pre_ff_layernorm.weight", "F32", (width,)),
        ]
    )
    return specs


def source_tensor_specs(topology: Mapping[str, Any]) -> List[TensorSpec]:
    # Expand only the two source aliases folded by the canonical artifact.
    runtime = canonical_tensor_specs(topology)
    by_name = {spec.name: spec for spec in runtime}
    source = list(runtime)
    for layer in range(topology["num_layers"]):
        if (
            topology["variant"] == "with-middle-attention"
            and layer == topology["middle_attention_layer"]
        ):
            continue
        prefix = "layers.%d.mamba_module." % layer
        for suffix in ("in_proj.weight", "out_proj.weight"):
            forward = by_name[prefix + "mamba_fwd." + suffix]
            source.append(
                TensorSpec(prefix + "mamba_rev." + suffix, "F32", forward.shape)
            )
    source.append(
        TensorSpec(
            "lm_head.weight",
            "F32",
            (topology["vocab_size"], topology["hidden_size"]),
        )
    )
    return sorted(source, key=lambda spec: spec.name)


def load_profiles(path: Path) -> Tuple[Dict[str, Dict[str, Any]], bytes]:
    root, payload = load_json(path, "JanusDNA profile manifest")
    exact_keys(
        root,
        ["schema_version", "format", "implementation_contract", "models"],
        [],
        "profile manifest",
    )
    if (
        root["schema_version"] != 1
        or root["format"] != PROFILE_FORMAT
        or root["implementation_contract"] != IMPLEMENTATION_CONTRACT
        or not isinstance(root["models"], list)
    ):
        raise ConversionError("JanusDNA profile manifest schema/contract differs")
    result = {}  # type: Dict[str, Dict[str, Any]]
    for index, raw in enumerate(root["models"]):
        label = "profile models[%d]" % index
        profile = object_value(raw, label)
        if set(profile) != PROFILE_KEYS:
            raise ConversionError("%s fields differ" % label)
        runtime_id = nonempty_string(profile["runtime_id"], label + ".runtime_id")
        if runtime_id in result:
            raise ConversionError("duplicate JanusDNA runtime_id")
        for key in ("geneb_model_id", "paper_name", "catalog_architecture"):
            profile[key] = nonempty_string(profile[key], label + "." + key)

        source = object_value(profile["source"], label + ".source")
        if set(source) != SOURCE_KEYS or source["format"] != "pytorch-lightning":
            raise ConversionError(label + ".source schema/format differs")
        for key in ("checkpoint_name", "config_name"):
            source[key] = normalized_relative_path(
                source[key], label + ".source." + key
            )
            if len(PurePosixPath(source[key]).parts) != 1:
                raise ConversionError(label + ".source." + key + " must be a basename")
        for key in (
            "checkpoint_size",
            "config_size",
            "state_entries",
            "clean_entries",
        ):
            source[key] = uint64_value(source[key], label + ".source." + key)
        for key in (
            "checkpoint_sha256",
            "config_sha256",
            "shape_manifest_sha256",
        ):
            source[key] = nonempty_string(source[key], label + ".source." + key)
            if not SHA256_RE.fullmatch(source[key]):
                raise ConversionError(label + ".source." + key + " is invalid")
        metrics = object_value(source["metric_values"], label + ".source.metric_values")
        expected_metric_names = {
            "train_torchmetrics.num_tokens.count",
            "val_torchmetrics.num_tokens.count",
            "test_torchmetrics.num_tokens.count",
        }
        if set(metrics) != expected_metric_names:
            raise ConversionError(label + ".source.metric_values fields differ")
        source["metric_values"] = {
            key: uint64_value(value, label + ".source.metric_values." + key)
            for key, value in metrics.items()
        }
        if source["state_entries"] != source["clean_entries"] + len(metrics):
            raise ConversionError(label + ".source entry counts differ")
        profile["source"] = source

        config_required = object_value(
            profile["config_required"], label + ".config_required"
        )
        if set(config_required) != {"config"} or not isinstance(
            config_required["config"], dict
        ):
            raise ConversionError(label + ".config_required schema differs")

        tokenizer = object_value(profile["tokenizer"], label + ".tokenizer")
        if set(tokenizer) != TOKENIZER_KEYS:
            raise ConversionError(label + ".tokenizer fields differ")
        for key in ("compiler_manifest_sha256", "compiled_asset_sha256"):
            tokenizer[key] = nonempty_string(
                tokenizer[key], label + ".tokenizer." + key
            )
            if not SHA256_RE.fullmatch(tokenizer[key]):
                raise ConversionError(label + ".tokenizer." + key + " is invalid")
        for key in ("compiled_asset_size", "emitted_vocab_size", "pad_token_id"):
            tokenizer[key] = uint64_value(
                tokenizer[key],
                label + ".tokenizer." + key,
                key != "pad_token_id",
            )
        if (
            tokenizer["kind"] != "single-nucleotide"
            or tokenizer["add_special_tokens"] is not False
            or tokenizer["padding_side"] != "right"
        ):
            raise ConversionError(label + ".tokenizer semantics differ")
        profile["tokenizer"] = tokenizer

        topology = validate_topology(profile["topology"], label + ".topology")
        profile["topology"] = topology
        expected_variant = (
            "with-middle-attention"
            if runtime_id == "geneb-janusdna-72-w"
            else "without-middle-attention"
        )
        if (
            runtime_id not in {"geneb-janusdna-72-w", "geneb-janusdna-72-wo"}
            or topology["variant"] != expected_variant
            or tokenizer["emitted_vocab_size"] != topology["tokenizer_vocab_size"]
            or tokenizer["pad_token_id"] != topology["pad_token_id"]
            or len(source_tensor_specs(topology)) != source["clean_entries"]
        ):
            raise ConversionError(label + " identity/topology/count contract differs")
        expected_row = EXPECTED_PROFILE_ROWS[runtime_id]
        if (
            any(
                profile[key] != expected_row[key]
                for key in ("geneb_model_id", "paper_name", "catalog_architecture")
            )
            or source != expected_row["source"]
            or tokenizer != EXPECTED_TOKENIZER
        ):
            raise ConversionError(label + " differs from the pinned profile row")
        result[runtime_id] = profile
    if set(result) != {"geneb-janusdna-72-w", "geneb-janusdna-72-wo"}:
        raise ConversionError(
            "profile manifest must contain exactly both JanusDNA rows"
        )
    return result, payload


def load_catalog(
    path: Path, profile: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    root, payload = load_json(path, "GENEB catalog")
    rows = root.get("models")
    matches = (
        [
            item
            for item in rows
            if isinstance(item, dict)
            and item.get("runtime_id") == profile["runtime_id"]
        ]
        if isinstance(rows, list)
        else []
    )
    if root.get("schema_version") != 1 or len(matches) != 1:
        raise ConversionError("GENEB catalog JanusDNA identity is missing")
    entry = matches[0]
    expected_identity = {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "family": "hybrid-mamba-moe",
        "architecture": profile["catalog_architecture"],
    }
    if any(entry.get(key) != value for key, value in expected_identity.items()):
        raise ConversionError("GENEB catalog JanusDNA identity differs")
    source = entry.get("source")
    expected_files = [
        {
            "path": profile["source"]["checkpoint_name"],
            "size": profile["source"]["checkpoint_size"],
            "sha256": profile["source"]["checkpoint_sha256"],
        },
        {
            "path": profile["source"]["config_name"],
            "size": profile["source"]["config_size"],
            "sha256": profile["source"]["config_sha256"],
        },
    ]
    if (
        not isinstance(source, dict)
        or source.get("kind") != "harvard-dataverse"
        or source.get("immutable") is not False
        or source.get("repo") is not None
        or source.get("requested_revision") is not None
        or source.get("revision") is not None
        or source.get("url") != DATAVERSE_URL
        or source.get("required_files") != expected_files
    ):
        raise ConversionError("GENEB catalog mutable source/required-files differ")
    tokenizer = entry.get("tokenizer")
    context = entry.get("context")
    transform = entry.get("input_transform")
    presets = entry.get("embedding_presets")
    normalized = presets.get("normalized") if isinstance(presets, dict) else None
    topology = profile["topology"]
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("kind") != "single-nucleotide"
        or tokenizer.get("add_special_tokens") is not False
        or tokenizer.get("padding_side") != "right"
        or tokenizer.get("max_tokens") != topology["max_seqlen"]
        or not isinstance(context, dict)
        or context.get("declared_max_tokens") != topology["max_seqlen"]
        or not isinstance(transform, dict)
        or transform.get("special_tokens") != "none"
        or transform.get("token_truncation") != "right"
        or transform.get("fixed_pad")
        != {
            "length": topology["max_seqlen"],
            "side": "right",
            "value": "tokenizer-pad",
            "balance": None,
        }
        or not isinstance(normalized, dict)
        or normalized.get("hidden_tap")
        != "twice-post-final-rmsnorm-after-identity-final-mlp-residual"
        or normalized.get("pooling") != "attention-mask-mean"
        or normalized.get("special_tokens") != "none"
        or normalized.get("mask_domain") != "attention-mask"
        or normalized.get("output_width") != topology["hidden_size"]
    ):
        raise ConversionError("GENEB catalog Janus tokenizer/context/pooling differs")
    return entry, root, payload


def validate_receipt(
    path: Path, profile: Mapping[str, Any]
) -> Tuple[Dict[str, Path], bytes]:
    receipt, payload = load_json(path, "manual source receipt")
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
        "source receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != "geneb-janusdna-manual-source"
        or receipt["model_id"] != profile["runtime_id"]
        or receipt["source_kind"] != "harvard-dataverse"
        or receipt["source_url"] != DATAVERSE_URL
        or not isinstance(receipt["files"], list)
    ):
        raise ConversionError("manual source receipt identity differs")
    expected = {
        profile["source"]["checkpoint_name"]: {
            "size": profile["source"]["checkpoint_size"],
            "sha256": profile["source"]["checkpoint_sha256"],
        },
        profile["source"]["config_name"]: {
            "size": profile["source"]["config_size"],
            "sha256": profile["source"]["config_sha256"],
        },
    }
    paths = {}  # type: Dict[str, Path]
    for index, raw in enumerate(receipt["files"]):
        label = "source receipt files[%d]" % index
        item = object_value(raw, label)
        exact_keys(item, ["name", "path", "size", "sha256"], [], label)
        name = normalized_relative_path(item["name"], label + ".name")
        if len(PurePosixPath(name).parts) != 1 or name in paths or name not in expected:
            raise ConversionError(label + " is duplicate, nested, or unexpected")
        registered = expected[name]
        if item["size"] != registered["size"] or item["sha256"] != registered["sha256"]:
            raise ConversionError(
                "manual receipt pinned size/SHA256 differs for " + name
            )
        source_path = Path(nonempty_string(item["path"], label + ".path")).resolve()
        try:
            actual_size = source_path.stat().st_size
        except OSError as error:
            raise ConversionError(
                "cannot stat manual source %s: %s" % (source_path, error)
            )
        # Full-file verification intentionally precedes torch import/pickle.
        if (
            actual_size != registered["size"]
            or sha256_file(source_path) != registered["sha256"]
        ):
            raise ConversionError("manual source payload integrity differs for " + name)
        paths[name] = source_path
    if set(paths) != set(expected):
        raise ConversionError("manual source receipt file set differs")
    return paths, payload


def validate_config(path: Path, profile: Mapping[str, Any]) -> None:
    config, payload = load_json(path, "Dataverse JanusDNA config")
    if (
        len(payload) != profile["source"]["config_size"]
        or sha256_bytes(payload) != profile["source"]["config_sha256"]
        or config != profile["config_required"]
    ):
        raise ConversionError("Dataverse JanusDNA config bytes/contents differ")


def type_name(value: Any) -> str:
    value_type = type(value)
    return value_type.__module__ + "." + value_type.__qualname__


def safe_load_checkpoint(path: Path) -> Mapping[str, Any]:
    # Deliberately deferred until validate_receipt completes full-file SHA256.
    if sys.byteorder != "little":
        raise ConversionError("JanusDNA conversion requires a little-endian host")
    torch = importlib.import_module("torch")
    base = importlib.import_module("omegaconf.base")
    dictconfig = importlib.import_module("omegaconf.dictconfig")
    listconfig = importlib.import_module("omegaconf.listconfig")
    nodes = importlib.import_module("omegaconf.nodes")
    typing_module = importlib.import_module("typing")
    safe_globals = [
        defaultdict,
        dict,
        int,
        base.Metadata,
        dictconfig.DictConfig,
        nodes.AnyNode,
        base.ContainerMetadata,
        listconfig.ListConfig,
        list,
        typing_module.Any,
    ]
    try:
        unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(str(path)))
    except Exception as error:
        raise ConversionError(
            "cannot inspect checkpoint unsafe-global manifest: %s" % error
        ) from error
    if unsafe != EXPECTED_UNSAFE_GLOBALS:
        raise ConversionError(
            "checkpoint unsafe-global manifest differs: missing=%s extra=%s"
            % (
                sorted(EXPECTED_UNSAFE_GLOBALS - unsafe),
                sorted(unsafe - EXPECTED_UNSAFE_GLOBALS),
            )
        )
    try:
        with torch.serialization.safe_globals(safe_globals):
            root = torch.load(
                str(path), weights_only=True, mmap=True, map_location="cpu"
            )
    except Exception as error:
        raise ConversionError(
            "weights-only checkpoint parse failed: %s" % error
        ) from error
    if not isinstance(root, dict) or set(root) != set(TOP_LEVEL_TYPES):
        raise ConversionError("Lightning checkpoint top-level fields differ")
    actual_types = {key: type_name(value) for key, value in root.items()}
    if actual_types != TOP_LEVEL_TYPES:
        raise ConversionError("Lightning checkpoint top-level types differ")
    if root["hparams_type"] is not dictconfig.DictConfig:
        raise ConversionError("Lightning checkpoint hparams_type differs")
    return root


def raw_source_name(name: str) -> str:
    if name == "lm_head.weight":
        return "model.lm_head.weight"
    return "model.model." + name


def shape_manifest_digest(specs: Sequence[TensorSpec]) -> str:
    lines = [
        "%s|torch.float32|%s" % (spec.name, ",".join(str(item) for item in spec.shape))
        for spec in sorted(specs, key=lambda item: item.name)
    ]
    return sha256_bytes("\n".join(lines).encode("ascii"))


def expected_alias_pairs(topology: Mapping[str, Any]) -> List[Tuple[str, str]]:
    result = []  # type: List[Tuple[str, str]]
    for layer in range(topology["num_layers"]):
        if (
            topology["variant"] == "with-middle-attention"
            and layer == topology["middle_attention_layer"]
        ):
            continue
        prefix = "layers.%d.mamba_module." % layer
        for suffix in ("in_proj.weight", "out_proj.weight"):
            result.append(
                (
                    prefix + "mamba_fwd." + suffix,
                    prefix + "mamba_rev." + suffix,
                )
            )
    return result


@dataclasses.dataclass(frozen=True)
class TorchTensorSource:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    nbytes: int
    tensor: Any

    def write_to(self, output: Any) -> None:
        # _write_file writes through the OS descriptor and bypasses Python's
        # buffered file object, so pending header/data bytes must be committed
        # before every storage write.
        output.flush()
        before = output.tell()
        # The audited PyTorch storage writer emits raw host-endian bytes without
        # a size prefix. This avoids a Python byte-by-byte storage conversion and
        # does not depend on NumPy being installed in the safe audit environment.
        self.tensor.untyped_storage()._write_file(output, True, False, 1)
        after = output.tell()
        if after - before != self.nbytes:
            raise ConversionError("tensor storage byte count differs: " + self.name)

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        payload = bytes(self.tensor.untyped_storage())
        if len(payload) != self.nbytes:
            raise ConversionError("tensor storage byte count differs: " + self.name)
        for offset in range(0, len(payload), chunk_size):
            yield memoryview(payload)[offset : offset + chunk_size]


def runtime_tensors(
    checkpoint: Mapping[str, Any], profile: Mapping[str, Any]
) -> List[TorchTensorSource]:
    state = checkpoint["state_dict"]
    if (
        not isinstance(state, Mapping)
        or len(state) != profile["source"]["state_entries"]
    ):
        raise ConversionError("Lightning state_dict count/type differs")
    source_specs = source_tensor_specs(profile["topology"])
    if (
        len(source_specs) != profile["source"]["clean_entries"]
        or shape_manifest_digest(source_specs)
        != profile["source"]["shape_manifest_sha256"]
    ):
        raise ConversionError("frozen source shape manifest differs")
    metric_values = profile["source"]["metric_values"]
    expected_raw = {raw_source_name(spec.name) for spec in source_specs} | set(
        metric_values
    )
    if set(state) != expected_raw:
        raise ConversionError(
            "Lightning state fields differ: missing=%s extra=%s"
            % (sorted(expected_raw - set(state)), sorted(set(state) - expected_raw))
        )
    torch = importlib.import_module("torch")
    for name, expected_value in metric_values.items():
        metric = state[name]
        if (
            not isinstance(metric, torch.Tensor)
            or metric.dtype != torch.int64
            or tuple(metric.shape) != ()
            or metric.device.type != "cpu"
            or metric.item() != expected_value
        ):
            raise ConversionError("exact torchmetrics scalar differs: " + name)

    clean = {}  # type: Dict[str, Any]
    spec_by_name = {spec.name: spec for spec in source_specs}
    for name, spec in spec_by_name.items():
        tensor = state[raw_source_name(name)]
        expected_bytes = tensor_nbytes(spec.dtype, spec.shape)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
            or not tensor.is_contiguous()
            or tensor.storage_offset() != 0
            or tuple(tensor.shape) != spec.shape
            or tensor.numel() * tensor.element_size() != expected_bytes
            or tensor.untyped_storage().nbytes() != expected_bytes
        ):
            raise ConversionError("source tensor dtype/shape/storage differs: " + name)
        clean[name] = tensor

    groups = {}  # type: Dict[int, List[str]]
    for name, tensor in clean.items():
        pointer = int(tensor.untyped_storage().data_ptr())
        groups.setdefault(pointer, []).append(name)
    actual_aliases = {
        tuple(sorted(names)) for names in groups.values() if len(names) > 1
    }
    expected_aliases = {
        tuple(sorted(pair)) for pair in expected_alias_pairs(profile["topology"])
    }
    if actual_aliases != expected_aliases:
        raise ConversionError(
            "source storage alias graph differs: missing=%s extra=%s"
            % (
                sorted(expected_aliases - actual_aliases),
                sorted(actual_aliases - expected_aliases),
            )
        )
    for forward, reverse in expected_alias_pairs(profile["topology"]):
        if not torch.equal(clean[forward], clean[reverse]):
            raise ConversionError("tied Mamba source tensors are not equal: " + forward)
    # LM head is strictly validated and independent, then intentionally dropped.
    result = []  # type: List[TorchTensorSource]
    for spec in canonical_tensor_specs(profile["topology"]):
        tensor = clean[spec.name]
        result.append(
            TorchTensorSource(spec.name, spec.dtype, spec.shape, spec.nbytes, tensor)
        )
    return result


def validate_tokenizer_descriptor(
    descriptor_path: Path,
    tokenizer_root: Optional[Path],
    artifact_root: Path,
    profile: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    descriptor, descriptor_payload = load_json(descriptor_path, "tokenizer descriptor")
    if set(descriptor) != TOKENIZER_DESCRIPTOR_KEYS:
        raise ConversionError("tokenizer descriptor fields differ")
    if (
        descriptor["converter.schema"] != "evo-tokenizer-conversion-receipt"
        or descriptor["converter.version"] != 1
        or descriptor["tokenizer.profile"] != TOKENIZER_PROFILE
    ):
        raise ConversionError("tokenizer descriptor schema/profile differs")
    for key in (
        "compiler_manifest_sha256",
        "source_receipt_contract_sha256",
        "tokenizer.sha256",
    ):
        if not isinstance(descriptor[key], str) or not SHA256_RE.fullmatch(
            descriptor[key]
        ):
            raise ConversionError("tokenizer descriptor digest is invalid: " + key)
    expected = profile["tokenizer"]
    if (
        descriptor["compiler_manifest_sha256"] != expected["compiler_manifest_sha256"]
        or descriptor["tokenizer.sha256"] != expected["compiled_asset_sha256"]
        or descriptor["tokenizer.size"] != expected["compiled_asset_size"]
    ):
        raise ConversionError("tokenizer descriptor differs from frozen profile")
    relative = normalized_relative_path(descriptor["tokenizer.path"], "tokenizer.path")
    root = (
        tokenizer_root if tokenizer_root is not None else descriptor_path.parent
    ).resolve()
    asset_path = (root / relative).resolve()
    try:
        asset_path.relative_to(root)
    except ValueError as error:
        raise ConversionError("tokenizer path escapes root") from error
    if (artifact_root.resolve() / relative).resolve() != asset_path:
        raise ConversionError("tokenizer asset must be staged beside output")
    payload = asset_path.read_bytes()
    if (
        len(payload) != expected["compiled_asset_size"]
        or sha256_bytes(payload) != expected["compiled_asset_sha256"]
    ):
        raise ConversionError("compiled tokenizer asset size/SHA256 differs")
    asset, _ = load_json(asset_path, "compiled JanusDNA tokenizer")
    exact_keys(
        asset,
        [
            "format",
            "kind",
            "normalization",
            "pre_tokenizer",
            "model",
            "post_processor",
            "special_tokens",
            "vocab",
        ],
        [],
        "compiled tokenizer",
    )
    expected_pieces = [
        "[CLS]",
        "[SEP]",
        "[BOS]",
        "[MASK]",
        "[PAD]",
        "[RESERVED]",
        "[UNK]",
        "A",
        "C",
        "G",
        "T",
        "N",
    ]
    expected_vocab = [
        {"id": index, "piece": piece} for index, piece in enumerate(expected_pieces)
    ]
    if (
        asset["format"] != "evo-tokenizer-v1"
        or asset["kind"] != "single-nucleotide"
        or asset["normalization"] != [{"op": "ascii-uppercase"}]
        or asset["pre_tokenizer"] != {"kind": "none"}
        or asset["model"] != {"unknown_policy": "unk", "match_special_literals": False}
        or asset["post_processor"]
        != {
            "prefix_ids": [],
            "suffix_ids": [],
            "padding": {"side": "right", "pad_id": 4},
        }
        or asset["special_tokens"]
        != {
            "unk": 6,
            "pad": 4,
            "bos": 2,
            "eos": 1,
            "cls": 0,
            "sep": 1,
            "mask": 3,
        }
        or asset["vocab"] != expected_vocab
    ):
        raise ConversionError("compiled JanusDNA tokenizer semantics differ")
    projected = {
        key: descriptor[key]
        for key in (
            "tokenizer.profile",
            "tokenizer.path",
            "tokenizer.sha256",
            "tokenizer.size",
        )
    }
    return projected, sha256_bytes(descriptor_payload)


def encode_metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "b:%d" % int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return "u:%d" % uint64_value(value, "metadata integer", False)
    if isinstance(value, float):
        parsed = finite_float(value, "metadata float")
        bits = struct.unpack("<Q", struct.pack("<d", parsed))[0]
        return "f:%016x" % bits
    if isinstance(value, str):
        if "\x00" in value:
            raise ConversionError("metadata string contains NUL")
        return "s:" + value
    raise ConversionError("unsupported metadata type")


def encoded_header(metadata: Mapping[str, Any], tensors: Sequence[Any]) -> bytes:
    encoded = {}  # type: Dict[str, str]
    for key in sorted(metadata):
        if not KEY_RE.fullmatch(key) or key == "runtime.profile":
            raise ConversionError("invalid/reserved metadata key: " + key)
        encoded[key] = encode_metadata_value(metadata[key])
    encoded["runtime.profile"] = "s:" + ARTIFACT_PROFILE
    root = {"__metadata__": dict(sorted(encoded.items()))}  # type: Dict[str, Any]
    offset = 0
    names = set()
    for tensor in tensors:
        if (
            tensor.name in names
            or not KEY_RE.fullmatch(tensor.name)
            or tensor.dtype != "F32"
            or tensor.nbytes != tensor_nbytes(tensor.dtype, tensor.shape)
        ):
            raise ConversionError(
                "invalid or duplicate artifact tensor: " + tensor.name
            )
        names.add(tensor.name)
        end = offset + tensor.nbytes
        if end > UINT64_MAX:
            raise ConversionError("artifact tensor payload exceeds uint64")
        root[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    raw = json.dumps(
        root, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    if not header or len(header) > MAX_HEADER_SIZE:
        raise ConversionError("artifact header exceeds 16 MiB")
    return header


def write_artifact(
    output_path: Path,
    metadata: Mapping[str, Any],
    tensors: Sequence[Any],
    force: bool,
) -> None:
    if output_path.suffix != ".safetensors" or not tensors:
        raise ConversionError("output must be a nonempty .safetensors artifact")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError("output already exists: %s" % output_path)
    header = encoded_header(metadata, tensors)
    total_bytes = 8 + len(header) + sum(tensor.nbytes for tensor in tensors)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + output_path.name + ".",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as output:
            output.truncate(total_bytes)
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for tensor in tensors:
                if isinstance(tensor, TorchTensorSource):
                    tensor.write_to(output)
                    continue
                written = 0
                for raw_chunk in tensor.iter_chunks(CHUNK_SIZE):
                    chunk = memoryview(raw_chunk).cast("B")
                    if written + len(chunk) > tensor.nbytes:
                        raise ConversionError(
                            "tensor yielded excess bytes: " + tensor.name
                        )
                    output.write(chunk)
                    written += len(chunk)
                if written != tensor.nbytes:
                    raise ConversionError(
                        "tensor yielded wrong byte count: " + tensor.name
                    )
            output.flush()
            os.fsync(output.fileno())
        if force:
            os.replace(str(temporary), str(output_path))
        else:
            os.link(str(temporary), str(output_path))
            temporary.unlink()
        try:
            directory_fd = os.open(str(output_path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not permit directory fsync; file fsync and
            # atomic publication above remain mandatory.
            pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_metadata(
    profile: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    receipt_sha256: str,
    catalog_sha256: str,
    profile_sha256: str,
    tokenizer_descriptor_sha256: str,
    geneb_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    topology = profile["topology"]
    middle_attention = (
        "causal-original-right-mask-both-directions"
        if topology["variant"] == "with-middle-attention"
        else "absent"
    )
    metadata = {
        "runtime.abi": RUNTIME_ABI,
        "runtime.embedding_layer_count": topology["num_layers"] + 1,
        "runtime.tokenizer_vocabulary_size": topology["tokenizer_vocab_size"],
        "model.id": profile["runtime_id"],
        "model.architecture": RUNTIME_ARCHITECTURE,
        "config.vocab_size": topology["vocab_size"],
        "config.hidden_size": topology["hidden_size"],
        "config.num_layers": topology["num_layers"],
        "config.max_seqlen": topology["max_seqlen"],
        "source.kind": "harvard-dataverse",
        "source.url": DATAVERSE_URL,
        "source.receipt_sha256": receipt_sha256,
        "source.catalog_contract_sha256": catalog_sha256,
        "source.converter_profile_contract_sha256": profile_sha256,
        "source.config_sha256": profile["source"]["config_sha256"],
        "source.checkpoint_sha256": profile["source"]["checkpoint_sha256"],
        "source.shape_manifest_sha256": profile["source"]["shape_manifest_sha256"],
        "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        "source.modeling_revision": IMPLEMENTATION_CONTRACT["source_revision"],
        "source.modeling_sha256": IMPLEMENTATION_CONTRACT["modeling_sha256"],
        "source.configuration_sha256": IMPLEMENTATION_CONTRACT["configuration_sha256"],
        "source.tokenizer_code_sha256": IMPLEMENTATION_CONTRACT["tokenizer_sha256"],
        "source.extractor_commit": IMPLEMENTATION_CONTRACT["extractor_commit"],
        "source.extractor_sha256": IMPLEMENTATION_CONTRACT["extractor_sha256"],
        "source.tensor_omit_policy": (
            "drop-exact-three-metrics-and-independent-lm-head;"
            "fold-exact-tied-reverse-in-out"
        ),
        "janus.variant": topology["variant"],
        "janus.vocab_size": topology["vocab_size"],
        "janus.tokenizer_vocab_size": topology["tokenizer_vocab_size"],
        "janus.hidden_size": topology["hidden_size"],
        "janus.num_layers": topology["num_layers"],
        "janus.num_attention_heads": topology["num_attention_heads"],
        "janus.head_dim": topology["head_dim"],
        "janus.flex_head_dim": topology["flex_head_dim"],
        "janus.inner_size": topology["inner_size"],
        "janus.state_size": topology["state_size"],
        "janus.conv_size": topology["conv_size"],
        "janus.dt_rank": topology["dt_rank"],
        "janus.mlp_size": topology["mlp_size"],
        "janus.num_experts": topology["num_experts"],
        "janus.experts_per_token": topology["experts_per_token"],
        "janus.max_seqlen": topology["max_seqlen"],
        "janus.middle_attention_layer": topology["middle_attention_layer"],
        "janus.pad_token_id": topology["pad_token_id"],
        "janus.norm_epsilon": topology["norm_epsilon"],
        "janus.weight_dtype": "F32",
        "janus.norm_placement": "pre-rms",
        "janus.activation": "silu",
        "janus.mamba_parameter_norm": "dt-b-c-rms",
        "janus.bidirectional_layout": "separate-concatenated-halves",
        "janus.tied_parameters": "mamba-in-out-only",
        "janus.expert_routing": "softmax-top2-no-renormalize",
        "janus.middle_attention": middle_attention,
        "janus.final_attention": "flex-four-condition-ignore-padding-mask",
        "janus.flex_scale": "padded-head-dimension",
        "janus.final_fusion": "repos-formasked-shift-1",
        "janus.final_mlp": "identity-residual-double",
        "janus.hidden_tap": "twice-post-final-rmsnorm",
        "janus.pooling": "attention-mask-mean",
        "janus.special_tokens": "none",
        "janus.mask_domain": "original-attention-mask",
        "janus.tokenizer_kind": "single-nucleotide-uppercase",
        "janus.official_reference_device": "gpu",
        "janus.alias_encoding": "canonical-shared-in-out",
    }  # type: Dict[str, Any]
    metadata.update(geneb_metadata)
    metadata.update(tokenizer)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--catalog", type=Path, default=default_config_path("geneb-models.json")
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=default_config_path("geneb-janusdna-models.json"),
    )
    parser.add_argument("--tokenizer-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        profiles, _profile_payload = load_profiles(args.profiles.resolve())
        receipt_identity, _ = load_json(
            args.receipt.resolve(), "manual source receipt identity"
        )
        model_id = receipt_identity.get("model_id")
        profile = profiles.get(model_id) if isinstance(model_id, str) else None
        if profile is None:
            raise ConversionError("receipt does not identify a JanusDNA profile")
        catalog_entry, catalog, catalog_payload = load_catalog(
            args.catalog.resolve(), profile
        )
        paths, receipt_payload = validate_receipt(args.receipt.resolve(), profile)
        validate_config(paths[profile["source"]["config_name"]], profile)
        tokenizer, tokenizer_descriptor_sha256 = validate_tokenizer_descriptor(
            args.tokenizer_descriptor.resolve(),
            args.tokenizer_root.resolve() if args.tokenizer_root is not None else None,
            args.output.resolve().parent,
            profile,
        )
        checkpoint = safe_load_checkpoint(paths[profile["source"]["checkpoint_name"]])
        tensors = runtime_tensors(checkpoint, profile)
        geneb_metadata = build_geneb_artifact_metadata(
            catalog, catalog_entry, catalog_payload
        )
        metadata = build_metadata(
            profile,
            tokenizer,
            sha256_bytes(receipt_payload),
            catalog_contract_sha256(catalog, catalog_entry),
            converter_profile_contract_sha256(
                1,
                PROFILE_FORMAT,
                profile,
                {"implementation_contract": IMPLEMENTATION_CONTRACT},
            ),
            tokenizer_descriptor_sha256,
            geneb_metadata,
        )
        write_artifact(args.output, metadata, tensors, args.force)
        print("wrote %s" % args.output)
        print("source_receipt_sha256=%s" % sha256_bytes(receipt_payload))
        print("variant=%s" % profile["topology"]["variant"])
        print("tensors=%d" % len(tensors))
        return 0
    except (
        CheckpointError,
        ConversionError,
        FileExistsError,
        GenebArtifactError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "convert_geneb_janusdna_checkpoint: error: %s" % error,
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
