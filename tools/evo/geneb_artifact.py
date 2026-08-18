"""Canonical flat GENEB metadata shared by family-specific converters."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class GenebArtifactError(ValueError):
    """Raised when a catalog row cannot be represented in an artifact."""


_PROMOTION_FIELDS = ("oracle", "runtime_support", "backends", "promotion_state")
_PROFILE_SHARED_CONTRACT_FIELDS = frozenset(
    ("code_provenance", "implementation_contract", "implementation_contracts")
)


def converter_profile_contract_sha256(
    schema_version: Any,
    profile_format: Any,
    selected_profile: Mapping[str, Any],
    shared_contracts: Optional[Mapping[str, Any]] = None,
) -> str:
    """Hash one normalized converter profile and its family-wide contracts.

    The complete profile manifest must be strictly validated before this helper
    is called.  Sibling rows are deliberately absent from the projection so a
    promotion or correction for one model cannot perturb another model's
    artifact identity.
    """
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version <= 0
    ):
        raise GenebArtifactError(
            "converter profile schema_version must be a positive integer"
        )
    if not isinstance(profile_format, str) or not profile_format:
        raise GenebArtifactError("converter profile format must be nonempty")
    if not isinstance(selected_profile, dict) or not selected_profile:
        raise GenebArtifactError("selected converter profile must be an object")
    if shared_contracts is not None and not isinstance(shared_contracts, dict):
        raise GenebArtifactError("converter shared contracts must be an object")
    contracts = {} if shared_contracts is None else dict(shared_contracts)
    unknown = set(contracts) - _PROFILE_SHARED_CONTRACT_FIELDS
    if unknown:
        raise GenebArtifactError(
            "converter shared contract fields are unsupported: {}".format(
                sorted(unknown)
            )
        )
    if any(not isinstance(value, dict) or not value for value in contracts.values()):
        raise GenebArtifactError(
            "converter shared contract fields must be nonempty objects"
        )
    contract = {
        "schema_version": schema_version,
        "format": profile_format,
        "model": copy.deepcopy(dict(selected_profile)),
    }
    for key in sorted(contracts):
        contract[key] = copy.deepcopy(contracts[key])
    try:
        payload = (
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeError, ValueError) as error:
        raise GenebArtifactError(
            "converter profile contract is not canonical JSON: {}".format(error)
        )
    return hashlib.sha256(payload).hexdigest()


def catalog_contract_sha256(
    catalog: Mapping[str, Any], entry: Mapping[str, Any]
) -> str:
    """Hash one model's immutable runtime/source contract, excluding promotion state."""
    model = copy.deepcopy(dict(entry))
    for key in _PROMOTION_FIELDS:
        model.pop(key, None)
    contract = {
        "schema_version": catalog.get("schema_version"),
        "suite": catalog.get("suite"),
        "model": model,
    }
    try:
        payload = (
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeError, ValueError) as error:
        raise GenebArtifactError(
            "catalog model contract is not canonical JSON: {}".format(error)
        )
    return hashlib.sha256(payload).hexdigest()


def validate_hf_fetch_receipt_provenance(
    receipt: Mapping[str, Any],
    catalog_path: Path,
    catalog_payload: bytes,
    catalog_entry: Mapping[str, Any],
) -> None:
    """Validate the optional provenance fields emitted by ``evo-fetch``.

    Hand-authored test receipts predate these fields, so their absence remains
    valid. New receipts bind the stable per-model catalog contract. Legacy full
    catalog hashes remain readable only as a migration input.
    """

    source_kind = receipt.get("source_kind")
    if source_kind is not None and source_kind != "huggingface":
        raise GenebArtifactError("source receipt source_kind must be huggingface")
    has_path = "catalog_path" in receipt
    has_contract = "catalog_contract_sha256" in receipt
    has_legacy = "catalog_sha256" in receipt
    if has_contract and has_legacy:
        raise GenebArtifactError(
            "source receipt must not mix catalog contract and legacy hashes"
        )
    has_digest = has_contract or has_legacy
    if has_path != has_digest:
        raise GenebArtifactError(
            "source receipt catalog_path/catalog digest must appear together"
        )
    if not has_path:
        return
    raw_path = _string(receipt["catalog_path"], "source receipt.catalog_path")
    if Path(raw_path).expanduser().resolve() != catalog_path.expanduser().resolve():
        raise GenebArtifactError("source receipt catalog_path differs")
    if has_contract:
        raw_digest = _string(
            receipt["catalog_contract_sha256"],
            "source receipt.catalog_contract_sha256",
        )
        try:
            catalog = json.loads(catalog_payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise GenebArtifactError(
                "catalog payload cannot be decoded for contract validation"
            ) from error
        if not isinstance(catalog, dict):
            raise GenebArtifactError("catalog payload root must be an object")
        expected = catalog_contract_sha256(catalog, catalog_entry)
        if raw_digest != expected:
            raise GenebArtifactError(
                "source receipt catalog_contract_sha256 differs"
            )
    else:
        raw_digest = _string(
            receipt["catalog_sha256"], "source receipt.catalog_sha256"
        )
        expected = hashlib.sha256(catalog_payload).hexdigest()
        if raw_digest != expected:
            raise GenebArtifactError("source receipt catalog_sha256 differs")


def _string(value: Any, label: str, nullable: bool = False) -> Optional[str]:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value:
        raise GenebArtifactError("{} must be a nonempty string".format(label))
    return value


def _uint(value: Any, label: str, nullable: bool = False) -> Optional[int]:
    if value is None and nullable:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GenebArtifactError("{} must be a positive integer".format(label))
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise GenebArtifactError("{} must be boolean".format(label))
    return value


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GenebArtifactError("{} must be an object".format(label))
    return value


def _optional_transform(
    output: Dict[str, Any],
    prefix: str,
    value: Any,
    fields: Mapping[str, str],
) -> None:
    output[prefix + ".enabled"] = value is not None
    if value is None:
        return
    record = _object(value, prefix)
    if set(record) != set(fields):
        raise GenebArtifactError("{} keys differ".format(prefix))
    for field, kind in fields.items():
        raw = record[field]
        if kind == "uint":
            output[prefix + "." + field] = _uint(raw, prefix + "." + field)
        elif kind == "nullable-string":
            parsed = _string(raw, prefix + "." + field, True)
            output[prefix + "." + field] = "" if parsed is None else parsed
        else:
            output[prefix + "." + field] = _string(raw, prefix + "." + field)


def build_geneb_artifact_metadata(
    catalog: Mapping[str, Any], entry: Mapping[str, Any], catalog_payload: bytes
) -> Dict[str, Any]:
    """Compile one strict catalog row into scalar Safetensors metadata."""
    suite = _object(catalog.get("suite"), "catalog.suite")
    if suite.get("id") != "geneb-v4":
        raise GenebArtifactError("catalog suite must be geneb-v4")
    raw_cap = _uint(suite.get("raw_safety_cap_bytes"), "suite.raw_safety_cap_bytes")
    context = _object(entry.get("context"), "model.context")
    transform = _object(entry.get("input_transform"), "model.input_transform")
    presets = _object(entry.get("embedding_presets"), "model.embedding_presets")
    source = _object(entry.get("source"), "model.source")
    provenance = _object(entry.get("provenance"), "model.provenance")
    extractor = _object(provenance.get("extractor"), "model.provenance.extractor")
    reference_patch = _object(
        provenance.get("reference_patch"), "model.provenance.reference_patch"
    )
    output = {
        "geneb.schema_version": 1,
        "geneb.suite": "geneb-v4",
        "geneb.catalog_contract_sha256": catalog_contract_sha256(catalog, entry),
        "geneb.runtime_id": _string(entry.get("runtime_id"), "model.runtime_id"),
        "geneb.model_id": _string(entry.get("geneb_model_id"), "model.geneb_model_id"),
        "geneb.paper_name": _string(entry.get("paper_name"), "model.paper_name"),
        "geneb.source.kind": _string(source.get("kind"), "model.source.kind"),
        "geneb.source.immutable": _boolean(
            source.get("immutable"), "model.source.immutable"
        ),
        "geneb.raw_safety_cap_bytes": raw_cap,
        "geneb.context.unit": _string(context.get("unit"), "model.context.unit"),
        "geneb.context.length_policy": _string(
            context.get("length_policy"), "model.context.length_policy"
        ),
        "geneb.input.case": _string(transform.get("case"), "model.input.case"),
        "geneb.input.strip_ascii_whitespace": _boolean(
            transform.get("strip_ascii_whitespace"),
            "model.input.strip_ascii_whitespace",
        ),
        "geneb.input.u_to_t": _boolean(
            transform.get("u_to_t"), "model.input.u_to_t"
        ),
        "geneb.input.invalid": _string(
            transform.get("invalid"), "model.input.invalid"
        ),
        "geneb.input.prefix": ""
        if transform.get("prefix") is None
        else _string(transform.get("prefix"), "model.input.prefix"),
        "geneb.input.special_tokens": _string(
            transform.get("special_tokens"), "model.input.special_tokens"
        ),
        "geneb.input.token_truncation": _string(
            transform.get("token_truncation"), "model.input.token_truncation"
        ),
        "geneb.provenance.extractor_commit": _string(
            extractor.get("commit"), "model.provenance.extractor.commit"
        ),
        "geneb.provenance.reference_patch_sha256": _string(
            reference_patch.get("sha256"), "model.provenance.reference_patch.sha256"
        ),
        "geneb.provenance.normalization_patch_sha256": _string(
            provenance.get("normalization_patch_sha256"),
            "model.provenance.normalization_patch_sha256",
        ),
    }  # type: Dict[str, Any]
    revision = _string(source.get("revision"), "model.source.revision", True)
    output["geneb.source.revision"] = "" if revision is None else revision
    for name in ("declared_max_tokens", "reference_max_tokens"):
        parsed = _uint(context.get(name), "model.context." + name, True)
        output["geneb.context." + name + ".known"] = parsed is not None
        output["geneb.context." + name] = 0 if parsed is None else parsed
    _optional_transform(
        output,
        "geneb.input.frame_trim",
        transform.get("frame_trim"),
        {"multiple": "uint", "remove_from": "string"},
    )
    _optional_transform(
        output,
        "geneb.input.raw_crop",
        transform.get("raw_crop"),
        {"length": "uint", "keep": "string"},
    )
    _optional_transform(
        output,
        "geneb.input.fixed_pad",
        transform.get("fixed_pad"),
        {
            "length": "uint",
            "side": "string",
            "value": "string",
            "balance": "nullable-string",
        },
    )
    for preset_name in ("reference", "normalized"):
        preset = _object(presets.get(preset_name), "model.preset." + preset_name)
        if set(preset) != {
            "hidden_tap",
            "pooling",
            "special_tokens",
            "mask_domain",
            "output_width",
        }:
            raise GenebArtifactError("model preset {} keys differ".format(preset_name))
        prefix = "geneb.preset." + preset_name + "."
        for field in ("hidden_tap", "pooling", "special_tokens", "mask_domain"):
            output[prefix + field] = _string(
                preset.get(field), "model.preset.{}.{}".format(preset_name, field)
            )
        output[prefix + "output_width"] = _uint(
            preset.get("output_width"),
            "model.preset.{}.output_width".format(preset_name),
        )
    return output
