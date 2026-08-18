#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract-test deterministic canonical-checkpoint evidence and promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOOLS_DIRECTORY = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIRECTORY))

from evo.geneb_artifact import catalog_contract_sha256  # noqa: E402


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def prepare_work_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name != "geneb-checkpoint-evidence" and not resolved.name.startswith(
        "evo-geneb-evidence-contract-"
    ):
        raise AssertionError("refusing to clean an unexpected evidence test directory")
    if path.is_symlink():
        raise AssertionError("evidence test work directory must not be a symlink")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)
    return resolved


def write_fake_converter(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path
from evo.geneb_artifact import catalog_contract_sha256

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--receipt", required=True)
parser.add_argument("--catalog", required=True)
parser.add_argument("--profiles", required=True)
parser.add_argument("--tokenizer-descriptor", required=True)
parser.add_argument("--tokenizer-root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
descriptor = json.loads(Path(args.tokenizer_descriptor).read_text())
catalog = json.loads(Path(args.catalog).read_text())
catalog_model = next(
    item for item in catalog["models"] if item["runtime_id"] == "geneb-fixture"
)
catalog_contract = catalog_contract_sha256(catalog, catalog_model)
metadata = {
    "runtime.profile": "s:geneb-fixture-runtime-v1",
    "runtime.abi": "s:geneb-fixture-safetensors-v1",
    "model.architecture": "s:GenebFixtureEncoder",
    "model.id": "s:geneb-fixture",
    "source.receipt_sha256": "s:" + sha(args.receipt),
    "source.catalog_contract_sha256": "s:" + catalog_contract,
    "geneb.catalog_contract_sha256": "s:" + catalog_contract,
    "source.converter_profile_contract_sha256": "s:" + "e" * 64,
    "source.tokenizer_descriptor_sha256": "s:" + sha(args.tokenizer_descriptor),
    "tokenizer.profile": "s:" + descriptor["tokenizer.profile"],
    "tokenizer.path": "s:" + descriptor["tokenizer.path"],
    "tokenizer.sha256": "s:" + descriptor["tokenizer.sha256"],
    "tokenizer.size": "u:" + str(descriptor["tokenizer.size"]),
}
root = {"__metadata__": metadata}
raw = json.dumps(root, sort_keys=True, separators=(",", ":")).encode("ascii")
header = raw + b" " * ((-len(raw)) % 8)
Path(args.output).write_bytes(struct.pack("<Q", len(header)) + header)
print("wrote " + str(Path(args.output).resolve()))
''',
        encoding="utf-8",
    )


def write_fake_input_transform_converter(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path
from evo.geneb_artifact import catalog_contract_sha256

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--receipt", required=True)
parser.add_argument("--catalog", required=True)
parser.add_argument("--profiles", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
catalog = json.loads(Path(args.catalog).read_text())
catalog_model = next(
    item for item in catalog["models"] if item["runtime_id"] == "geneb-fixture"
)
catalog_contract = catalog_contract_sha256(catalog, catalog_model)
metadata = {
    "runtime.profile": "s:geneb-fixture-runtime-v1",
    "runtime.abi": "s:geneb-fixture-safetensors-v1",
    "runtime.tokenizer_vocabulary_size": "u:4",
    "model.architecture": "s:GenebFixtureEncoder",
    "model.id": "s:geneb-fixture",
    "source.receipt_sha256": "s:" + sha(args.receipt),
    "source.catalog_contract_sha256": "s:" + catalog_contract,
    "geneb.catalog_contract_sha256": "s:" + catalog_contract,
    "source.converter_profile_contract_sha256": "s:" + "e" * 64,
}
if "forbidden" in Path(args.output).name:
    metadata["tokenizer.profile"] = "s:evo-tokenizer-v1"
if "legacy-profile-digest" in Path(args.output).name:
    metadata.pop("source.converter_profile_contract_sha256")
    metadata["source.converter_manifest_sha256"] = "s:" + "e" * 64
root = {"__metadata__": metadata}
raw = json.dumps(root, sort_keys=True, separators=(",", ":")).encode("ascii")
header = raw + b" " * ((-len(raw)) % 8)
Path(args.output).write_bytes(struct.pack("<Q", len(header)) + header)
print("wrote " + str(Path(args.output).resolve()))
''',
        encoding="utf-8",
    )


def write_fake_native(path: Path) -> None:
    path.write_text(
        r'''#!/usr/bin/env python3
import argparse
import json
import struct
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
commands = parser.add_subparsers(dest="command", required=True)
embed = commands.add_parser("embed")
embed.add_argument("-m", required=True)
embed.add_argument("--input", required=True)
embed.add_argument("--output", required=True)
embed.add_argument("--preset", required=True)
embed.add_argument("--ctx", required=True)
embed.add_argument("--backend", required=True)
embed.add_argument("--profile", required=True)
args = parser.parse_args()
if args.backend == "mps":
    print("fixture backend failed for " + str(Path(args.input).resolve()), file=sys.stderr)
    raise SystemExit(7)
output = Path(args.output)
output.mkdir(parents=True)
header = repr({"descr": "<f4", "fortran_order": False, "shape": (1, 2)}).encode("latin1")
padding = (-(10 + len(header) + 1)) % 16
header += b" " * padding + b"\n"
npy = b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(header)) + header
npy += struct.pack("<2f", 1.0, 2.0)
(output / "000000.npy").write_bytes(npy)
record = {
    "record_index": 0,
    "name": str(Path(args.input).resolve()),
    "input_format": "raw",
    "file": "000000.npy",
    "source_tokens": 4,
    "shape": [1, 2],
    "layer": 1,
    "point": "fixture-final-hidden",
    "pooling": "attention-mask-mean",
    "preset": "geneb-v4-normalized",
    "dtype": "float32",
    "backend": args.backend,
    "profile": args.profile,
    "model_id": "geneb-fixture",
}
(output / "embeddings.jsonl").write_text(json.dumps(record, separators=(",", ":")) + "\n")
timing = 0.125 if "first" in str(output) else 19.75
print(
    "evo_metrics "
    + json.dumps(
        {
            "backend": args.backend,
            "model_load_seconds": timing,
            "profile": args.profile,
        },
        separators=(",", ":"),
    ),
    file=sys.stderr,
)
''',
        encoding="utf-8",
    )
    path.chmod(0o755)


def run(
    tool: Path, arguments: List[str], expected: int = 0
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(tool), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(
            os.environ,
            PYTHONPATH=str(tool.resolve().parent),
        ),
    )
    if result.returncode != expected:
        raise AssertionError(
            "unexpected evidence exit %d (expected %d):\nstdout=%s\nstderr=%s"
            % (result.returncode, expected, result.stdout, result.stderr)
        )
    return result


def evidence_arguments(
    root: Path,
    tool_inputs: Dict[str, Path],
    suffix: str,
    oracle: Optional[Path],
    endpoint: bool = True,
    receipt: Optional[Path] = None,
    backend: str = "cpu",
    include_tokenizer: bool = True,
) -> List[str]:
    arguments = [
        "run",
        "--catalog",
        str(tool_inputs["catalog"]),
        "--model",
        "geneb-fixture",
        "--source-receipt",
        str(receipt or tool_inputs["receipt"]),
    ]
    if endpoint:
        arguments.extend(["--source-endpoint", "https://huggingface.co"])
    if include_tokenizer:
        arguments.extend(
            [
                "--tokenizer-descriptor",
                str(tool_inputs["descriptor"]),
                "--tokenizer-root",
                str(root),
            ]
        )
    arguments.extend(
        [
        "--converter",
        str(tool_inputs["converter"]),
        "--profiles",
        str(tool_inputs["profiles"]),
        "--artifact",
        str(root / ("artifact-" + suffix + ".safetensors")),
        "--native",
        str(tool_inputs["native"]),
        "--input",
        str(tool_inputs["input"]),
        "--embedding-dir",
        str(root / ("embedding-" + suffix)),
        "--evidence",
        str(root / "evidence" / (suffix + ".json")),
        "--backend",
        backend,
        "--profile",
        "exact",
        ]
    )
    if oracle is not None:
        arguments.extend(
            [
                "--oracle-vector",
                str(oracle),
                "--max-abs",
                "0",
                "--mean-abs",
                "0",
                "--cosine",
                "1",
            ]
        )
    return arguments


def make_fixture(root: Path) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    source_dir = root / "source"
    source_dir.mkdir()
    attributes = source_dir / ".gitattributes"
    weights = source_dir / "model.safetensors"
    attributes.write_bytes(b"*.safetensors filter=lfs\n")
    weights.write_bytes(b"canonical checkpoint bytes\n")
    catalog = root / "catalog.json"
    model = {
        "runtime_id": "geneb-fixture",
        "geneb_model_id": "fixture",
        "paper_name": "Fixture",
        "family": "fixture",
        "architecture": "fixture",
        "tokenizer": {"kind": "character"},
        "context": {"declared_max_tokens": 16},
        "input_transform": {"case": "preserve"},
        "embedding_presets": {"normalized": {"output_width": 2}},
        "source": {
            "kind": "huggingface",
            "repo": "owner/fixture",
            "requested_revision": "main",
            "revision": "a" * 40,
        },
        "provenance": {},
        "licenses": {},
        "oracle": {
            "status": "missing",
            "environment_lock": None,
            "input_digest": None,
            "tolerances": None,
            "evidence": None,
        },
        "runtime_support": {
            "status": "cataloged",
            "artifact_profile": None,
            "reason": "fixture has no evidence",
        },
        "benchmark_provenance": {},
        "backends": {
            name: {"status": "not-promoted", "evidence": None}
            for name in ("cpu", "cuda", "mps")
        },
        "promotion_state": "cataloged",
    }
    write_json(
        catalog,
        {
            "schema_version": 1,
            "suite": {"id": "geneb-v4", "raw_safety_cap_bytes": 1024},
            "models": [model],
        },
    )
    receipt = root / "source-receipt.json"
    write_json(
        receipt,
        {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": "geneb-fixture",
            "repo": "owner/fixture",
            "requested_revision": "main",
            "resolved_revision": "a" * 40,
            "source_kind": "huggingface",
            "catalog_path": str(catalog.resolve()),
            "catalog_contract_sha256": catalog_contract_sha256(
                json.loads(catalog.read_text(encoding="utf-8")), model
            ),
            "load_path": None,
            "files": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": digest(path),
                    "path": str(path.resolve()),
                }
                for path in (attributes, weights)
            ],
        },
    )
    tokenizer = root / "tokenizers" / "fixture.json"
    write_json(tokenizer, {"format": "evo-tokenizer-v1", "fixture": True})
    descriptor = root / "tokenizer-descriptor.json"
    write_json(
        descriptor,
        {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": "b" * 64,
            "source_receipt_contract_sha256": "c" * 64,
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": "tokenizers/fixture.json",
            "tokenizer.sha256": digest(tokenizer),
            "tokenizer.size": tokenizer.stat().st_size,
        },
    )
    converter = root / "fake_converter.py"
    native = root / "fake_native.py"
    profiles = root / "profiles.json"
    input_path = root / "input.fa"
    write_fake_converter(converter)
    write_fake_native(native)
    write_json(profiles, {"schema_version": 1, "models": ["fixture"]})
    input_path.write_bytes(b">fixture\nACGT\n")
    oracle = root / "oracle.json"
    write_json(
        oracle,
        {
            "schema_version": 1,
            "kind": "geneb-independent-oracle-vector",
            "runtime_id": "geneb-fixture",
            "input_sha256": digest(input_path),
            "backend": "cpu",
            "profile": "exact",
            "values": [1.0, 2.0],
            "environment_lock": {"implementation": "independent-fixture-v1"},
            "provenance": {"kind": "independent", "generator_sha256": "d" * 64},
        },
    )
    return {
        "catalog": catalog,
        "receipt": receipt,
        "descriptor": descriptor,
        "converter": converter,
        "native": native,
        "profiles": profiles,
        "input": input_path,
        "oracle": oracle,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    args.work_dir = prepare_work_dir(args.work_dir)
    fixture = make_fixture(args.work_dir)

    missing_endpoint = run(
        args.tool,
        evidence_arguments(args.work_dir, fixture, "missing-endpoint", None, False),
        2,
    )
    if "explicit --source-endpoint" not in missing_endpoint.stderr:
        raise AssertionError("HF evidence accepted an inherited/implicit endpoint")

    original_catalog_bytes = fixture["catalog"].read_bytes()
    original_receipt_bytes = fixture["receipt"].read_bytes()
    manual_catalog = json.loads(original_catalog_bytes)
    manual_model = manual_catalog["models"][0]
    source_files = [
        args.work_dir / "source" / ".gitattributes",
        args.work_dir / "source" / "model.safetensors",
    ]
    manual_model["source"] = {
        "kind": "google-drive",
        "repo": None,
        "requested_revision": None,
        "revision": None,
        "immutable": False,
        "url": "https://drive.google.com/drive/folders/fixture",
        "required_files": [
            {
                "path": "nested/" + path.name,
                "size": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in source_files
        ],
    }
    write_json(fixture["catalog"], manual_catalog)
    write_json(
        fixture["receipt"],
        {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": "geneb-fixture",
            "source_kind": "google-drive",
            "source_url": manual_model["source"]["url"],
            "files": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": digest(path),
                    "path": str(path.resolve()),
                }
                for path in source_files
            ],
        },
    )
    manual_args = evidence_arguments(
        args.work_dir, fixture, "manual-source", None, endpoint=False
    )
    run(args.tool, manual_args)
    manual_evidence = json.loads(
        (args.work_dir / "evidence" / "manual-source.json").read_text(
            encoding="utf-8"
        )
    )
    if manual_evidence["source"]["provider"] != {
        "kind": "google-drive",
        "url": manual_model["source"]["url"],
    }:
        raise AssertionError("generic manual source provider differs")
    run(
        args.tool,
        [
            "catalog-update",
            "--catalog",
            str(fixture["catalog"]),
            "--evidence",
            str(args.work_dir / "evidence" / "manual-source.json"),
            "--output",
            str(args.work_dir / "manual-promoted.json"),
        ],
        2,
    )
    fixture["catalog"].write_bytes(original_catalog_bytes)
    fixture["receipt"].write_bytes(original_receipt_bytes)

    input_transform_converter = args.work_dir / "fake_input_transform_converter.py"
    write_fake_input_transform_converter(input_transform_converter)
    asset_converter = fixture["converter"]
    input_transform_catalog = json.loads(original_catalog_bytes)
    input_transform_model = input_transform_catalog["models"][0]
    input_transform_model["tokenizer"] = {
        "kind": "single-nucleotide",
        "asset_source": "input-transform",
        "assets": None,
    }
    input_transform_model["input_transform"] = {
        "case": "upper",
        "invalid": "unknown-to-N",
        "fixed_pad": {"length": 16, "side": "right", "value": "N"},
    }
    write_json(fixture["catalog"], input_transform_catalog)
    input_transform_receipt = json.loads(original_receipt_bytes)
    input_transform_receipt["catalog_contract_sha256"] = catalog_contract_sha256(
        input_transform_catalog, input_transform_model
    )
    write_json(fixture["receipt"], input_transform_receipt)
    fixture["converter"] = input_transform_converter
    run(
        args.tool,
        evidence_arguments(
            args.work_dir,
            fixture,
            "input-transform",
            None,
            include_tokenizer=False,
        ),
    )
    input_transform_evidence = json.loads(
        (args.work_dir / "evidence" / "input-transform.json").read_text(
            encoding="utf-8"
        )
    )
    expected_input_transform_contract = digest_bytes(
        canonical_json(
            {
                "tokenizer": input_transform_model["tokenizer"],
                "input_transform": input_transform_model["input_transform"],
            }
        )
    )
    if input_transform_evidence["tokenizer"] != {
        "kind": "input-transform",
        "contract_sha256": expected_input_transform_contract,
        "vocabulary_size": 4,
    }:
        raise AssertionError("input-transform tokenizer evidence differs")
    mixed = run(
        args.tool,
        evidence_arguments(
            args.work_dir,
            fixture,
            "input-transform-mixed",
            None,
            include_tokenizer=True,
        ),
        2,
    )
    if "must not provide tokenizer descriptor/root" not in mixed.stderr:
        raise AssertionError("input-transform model accepted an asset descriptor")
    forbidden = run(
        args.tool,
        evidence_arguments(
            args.work_dir,
            fixture,
            "input-transform-forbidden",
            None,
            include_tokenizer=False,
        ),
        2,
    )
    if "contains tokenizer assets" not in forbidden.stderr:
        raise AssertionError("input-transform artifact accepted tokenizer metadata")
    legacy_profile_digest = run(
        args.tool,
        evidence_arguments(
            args.work_dir,
            fixture,
            "input-transform-legacy-profile-digest",
            None,
            include_tokenizer=False,
        ),
        2,
    )
    if "legacy whole-profile converter digest" not in legacy_profile_digest.stderr:
        raise AssertionError("evidence accepted a legacy whole-profile digest")
    fixture["converter"] = asset_converter
    fixture["catalog"].write_bytes(original_catalog_bytes)
    fixture["receipt"].write_bytes(original_receipt_bytes)

    no_oracle_args = evidence_arguments(
        args.work_dir, fixture, "no-oracle", None
    )
    run(args.tool, no_oracle_args)
    no_oracle_path = args.work_dir / "evidence" / "no-oracle.json"
    no_oracle = json.loads(no_oracle_path.read_text(encoding="utf-8"))
    if (
        no_oracle["oracle"]["status"] != "not-run"
        or no_oracle["backends"]["cpu"]["status"] != "passed"
        or no_oracle["backends"]["cuda"]["status"] != "not-run"
        or no_oracle["backends"]["mps"]["status"] != "not-run"
        or no_oracle["source"]["endpoint"] != "https://huggingface.co"
        or no_oracle["source"]["file_count"] != 2
    ):
        raise AssertionError("typed backend/source/oracle evidence differs")
    refused = run(
        args.tool,
        [
            "catalog-update",
            "--catalog",
            str(fixture["catalog"]),
            "--evidence",
            str(no_oracle_path),
            "--output",
            str(args.work_dir / "must-not-promote.json"),
        ],
        2,
    )
    if "passed independent oracle" not in refused.stderr:
        raise AssertionError("catalog update promoted a not-run oracle")

    absolute_oracle = args.work_dir / "absolute-oracle.json"
    absolute_oracle_value = json.loads(
        fixture["oracle"].read_text(encoding="utf-8")
    )
    absolute_oracle_value["environment_lock"]["nested"] = {
        "cache": "/private/tmp/non-portable-oracle-cache",
        "upstream": "https://huggingface.co/owner/model",
        "class": "transformers.AutoModel",
        "revision": "a" * 40,
    }
    write_json(absolute_oracle, absolute_oracle_value)
    absolute_rejected = run(
        args.tool,
        evidence_arguments(
            args.work_dir, fixture, "absolute-oracle", absolute_oracle
        ),
        2,
    )
    if "local absolute filesystem path" not in absolute_rejected.stderr:
        raise AssertionError("oracle environment accepted a local absolute path")

    first_args = evidence_arguments(
        args.work_dir, fixture, "first", fixture["oracle"]
    )
    second_args = evidence_arguments(
        args.work_dir, fixture, "second", fixture["oracle"]
    )
    run(args.tool, first_args)
    run(args.tool, second_args)
    first = args.work_dir / "evidence" / "first.json"
    second = args.work_dir / "evidence" / "second.json"
    if first.read_bytes() != second.read_bytes():
        raise AssertionError("repeated canonical evidence is not byte-identical")

    relocated_input = args.work_dir / "relocated" / "same-content.fa"
    relocated_input.parent.mkdir()
    relocated_input.write_bytes(fixture["input"].read_bytes())
    relocated_fixture = dict(fixture)
    relocated_fixture["input"] = relocated_input
    relocated_args = evidence_arguments(
        args.work_dir, relocated_fixture, "relocated-input", fixture["oracle"]
    )
    run(args.tool, relocated_args)
    relocated = args.work_dir / "evidence" / "relocated-input.json"
    if first.read_bytes() != relocated.read_bytes():
        raise AssertionError(
            "canonical evidence changed across absolute paths for identical input"
        )

    failure_one_args = evidence_arguments(
        args.work_dir, fixture, "failure-one", None, backend="mps"
    )
    failure_relocated_args = evidence_arguments(
        args.work_dir,
        relocated_fixture,
        "failure-relocated",
        None,
        backend="mps",
    )
    run(args.tool, failure_one_args, 2)
    run(args.tool, failure_relocated_args, 2)
    failure_one = args.work_dir / "evidence" / "failure-one.json"
    failure_relocated = args.work_dir / "evidence" / "failure-relocated.json"
    if failure_one.read_bytes() != failure_relocated.read_bytes():
        raise AssertionError("normalized failure evidence changed across input paths")
    failure_value = json.loads(failure_one.read_text(encoding="utf-8"))
    if failure_value["backends"]["mps"]["reason"] != (
        "fixture backend failed for <PATH>"
    ):
        raise AssertionError("failed backend did not retain a stable diagnostic summary")
    original_catalog = json.loads(fixture["catalog"].read_text(encoding="utf-8"))
    promotion_catalog = json.loads(fixture["catalog"].read_text(encoding="utf-8"))
    promotion_model = promotion_catalog["models"][0]
    promotion_model["oracle"] = {"status": "passed", "evidence": {"fixture": True}}
    promotion_model["runtime_support"] = {"status": "supported"}
    promotion_model["backends"] = {
        "cpu": {"status": "promoted"},
        "cuda": {"status": "unsupported"},
        "mps": {"status": "unsupported"},
    }
    promotion_model["promotion_state"] = "runtime-supported"
    write_json(fixture["catalog"], promotion_catalog)
    third_args = evidence_arguments(
        args.work_dir, fixture, "promotion-only", fixture["oracle"]
    )
    run(args.tool, third_args)
    third = args.work_dir / "evidence" / "promotion-only.json"
    if first.read_bytes() != third.read_bytes():
        raise AssertionError("promotion-only catalog changes altered evidence bytes")
    write_json(fixture["catalog"], original_catalog)
    evidence = json.loads(first.read_text(encoding="utf-8"))
    if (
        evidence["oracle"]["status"] != "passed"
        or evidence["oracle"]["metrics"]
        != {"cosine": 1.0, "max_abs": 0.0, "mean_abs": 0.0}
        or evidence["converter"]["status"] != "passed"
        or evidence["artifact"]["profile"] != "geneb-fixture-runtime-v1"
        or evidence["backends"]["cpu"]["output"]["source_label"]
        != "content-sha256:" + digest(fixture["input"])
    ):
        raise AssertionError("passed evidence fields differ")

    absolute_evidence = args.work_dir / "evidence" / "absolute-path.json"
    absolute_evidence_value = json.loads(second.read_text(encoding="utf-8"))
    absolute_evidence_value["oracle"]["provenance"]["nested"] = [
        {"debug_path": "C:\\private\\oracle.json"}
    ]
    write_json(absolute_evidence, absolute_evidence_value)
    stored_absolute_rejected = run(
        args.tool,
        [
            "catalog-update",
            "--catalog",
            str(fixture["catalog"]),
            "--evidence",
            str(absolute_evidence),
            "--output",
            str(args.work_dir / "absolute-path-promoted.json"),
        ],
        2,
    )
    if "local absolute filesystem path" not in stored_absolute_rejected.stderr:
        raise AssertionError("stored oracle provenance accepted an absolute path")

    promoted = args.work_dir / "promoted.json"
    run(
        args.tool,
        [
            "catalog-update",
            "--catalog",
            str(fixture["catalog"]),
            "--evidence",
            str(first),
            "--output",
            str(promoted),
        ],
    )
    validated = run(
        args.tool,
        ["catalog-validate", "--catalog", str(promoted), "--model", "geneb-fixture"],
    )
    if json.loads(validated.stdout) != {"valid": True, "validated_evidence": 1}:
        raise AssertionError("catalog evidence validation summary differs")
    promoted_row = json.loads(promoted.read_text(encoding="utf-8"))["models"][0]
    if (
        promoted_row["runtime_support"]["status"] != "supported"
        or promoted_row["backends"]["cpu"]["status"] != "promoted"
        or promoted_row["promotion_state"] != "runtime-supported"
    ):
        raise AssertionError("catalog updater did not bind passed evidence")

    linked = args.work_dir / "evidence-link"
    linked.symlink_to(args.work_dir / "evidence", target_is_directory=True)
    symlink_catalog = args.work_dir / "promoted-symlink.json"
    symlink_value = json.loads(promoted.read_text(encoding="utf-8"))
    symlink_model = symlink_value["models"][0]
    symlink_model["oracle"]["evidence"]["path"] = "evidence-link/first.json"
    symlink_model["backends"]["cpu"]["evidence"]["path"] = (
        "evidence-link/first.json"
    )
    write_json(symlink_catalog, symlink_value)
    symlink_rejected = run(
        args.tool,
        ["catalog-validate", "--catalog", str(symlink_catalog)],
        2,
    )
    if "path contains a symlink" not in symlink_rejected.stderr:
        raise AssertionError("catalog evidence traversed an intermediate symlink")

    bad_receipt = args.work_dir / "bad-receipt.json"
    bad_value = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    bad_value["files"][0]["sha256"] = "0" * 64
    write_json(bad_receipt, bad_value)
    corrupted = run(
        args.tool,
        evidence_arguments(
            args.work_dir, fixture, "bad-receipt", None, True, bad_receipt
        ),
        2,
    )
    if "size/SHA256 differs" not in corrupted.stderr:
        raise AssertionError("source receipt corruption was not rejected")

    unknown = args.work_dir / "evidence" / "unknown-field.json"
    unknown_value = json.loads(second.read_text(encoding="utf-8"))
    unknown_value["fabricated"] = True
    write_json(unknown, unknown_value)
    rejected = run(
        args.tool,
        [
            "catalog-update",
            "--catalog",
            str(fixture["catalog"]),
            "--evidence",
            str(unknown),
            "--output",
            str(args.work_dir / "unknown-promoted.json"),
        ],
        2,
    )
    if "extra=['fabricated']" not in rejected.stderr:
        raise AssertionError("unknown evidence field was accepted")

    first.write_bytes(first.read_bytes() + b" ")
    tampered = run(
        args.tool,
        ["catalog-validate", "--catalog", str(promoted)],
        2,
    )
    if "evidence SHA256 differs" not in tampered.stderr:
        raise AssertionError("catalog evidence tampering was not rejected")
    print("GENEB canonical checkpoint evidence contracts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
