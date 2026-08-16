#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict loader for the shared runtime artifact-profile registry."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


IDENTIFIER_RE = re.compile(r"[A-Za-z0-9._-]+")
PROFILE_METADATA_KEYS = frozenset({"evo2.profile", "runtime.profile"})


class ProfileRegistryError(ValueError):
    """Raised when the artifact-profile registry is missing or inconsistent."""


def _registry_object(pairs: List[Tuple[str, object]]) -> Dict[str, object]:
    value: Dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ProfileRegistryError(
                f"model registry contains duplicate key {key!r}"
            )
        value[key] = item
    return value


def load_artifact_profiles(path: Path) -> Dict[str, Dict[str, str]]:
    """Load profiles and verify every architecture/profile/ABI edge."""
    try:
        registry = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_registry_object
        )
    except ProfileRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProfileRegistryError(
            f"cannot read model registry {path}: {error}"
        ) from error
    schema_version = (
        registry.get("schema_version") if isinstance(registry, dict) else None
    )
    if (
        not isinstance(registry, dict)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ProfileRegistryError(f"{path}: unsupported model registry schema")
    values = registry.get("artifact_profiles")
    architectures = registry.get("runtime_architectures")
    if not isinstance(values, list) or not values:
        raise ProfileRegistryError(
            f"{path}: artifact_profiles must be a nonempty array"
        )
    if not isinstance(architectures, list) or not architectures:
        raise ProfileRegistryError(
            f"{path}: runtime_architectures must be a nonempty array"
        )

    profiles: Dict[str, Dict[str, str]] = {}
    runtime_abis = set()
    for index, raw in enumerate(values):
        label = f"artifact_profiles[{index}]"
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "metadata_key",
            "runtime_abi",
        }:
            raise ProfileRegistryError(
                f"{path}: {label} must contain exactly id, metadata_key, runtime_abi"
            )
        profile = raw["id"]
        metadata_key = raw["metadata_key"]
        runtime_abi = raw["runtime_abi"]
        if (
            not isinstance(profile, str)
            or not IDENTIFIER_RE.fullmatch(profile)
            or len(profile.encode("ascii")) > 255
        ):
            raise ProfileRegistryError(f"{path}: {label}.id is invalid")
        if (
            not isinstance(metadata_key, str)
            or metadata_key not in PROFILE_METADATA_KEYS
        ):
            raise ProfileRegistryError(
                f"{path}: {label}.metadata_key is not a registered metadata key"
            )
        if (
            not isinstance(runtime_abi, str)
            or not IDENTIFIER_RE.fullmatch(runtime_abi)
            or len(runtime_abi.encode("ascii")) > 255
        ):
            raise ProfileRegistryError(
                f"{path}: {label}.runtime_abi is invalid"
            )
        if profile in profiles:
            raise ProfileRegistryError(
                f"{path}: duplicate artifact profile {profile!r}"
            )
        if runtime_abi in runtime_abis:
            raise ProfileRegistryError(
                f"{path}: duplicate runtime ABI {runtime_abi!r}"
            )
        profiles[profile] = {
            "metadata_key": metadata_key,
            "runtime_abi": runtime_abi,
        }
        runtime_abis.add(runtime_abi)

    referenced = set()
    architecture_ids = set()
    for index, raw in enumerate(architectures):
        label = f"runtime_architectures[{index}]"
        if not isinstance(raw, dict):
            raise ProfileRegistryError(f"{path}: {label} must be an object")
        architecture = raw.get("id")
        profile = raw.get("artifact_profile")
        runtime_abi = raw.get("runtime_abi")
        if (
            not isinstance(architecture, str)
            or not IDENTIFIER_RE.fullmatch(architecture)
            or len(architecture.encode("ascii")) > 255
        ):
            raise ProfileRegistryError(
                f"{path}: {label}.id is invalid"
            )
        if architecture in architecture_ids:
            raise ProfileRegistryError(
                f"{path}: duplicate runtime architecture {architecture!r}"
            )
        architecture_ids.add(architecture)
        registered = profiles.get(profile) if isinstance(profile, str) else None
        if registered is None:
            raise ProfileRegistryError(
                f"{path}: {label} references unknown artifact profile {profile!r}"
            )
        if runtime_abi != registered["runtime_abi"]:
            raise ProfileRegistryError(
                f"{path}: {label}.runtime_abi disagrees with artifact profile"
            )
        referenced.add(profile)
    if referenced != set(profiles):
        orphaned = sorted(set(profiles) - referenced)
        raise ProfileRegistryError(
            f"{path}: artifact profiles are not referenced by an architecture: "
            + ", ".join(orphaned)
        )
    return profiles
