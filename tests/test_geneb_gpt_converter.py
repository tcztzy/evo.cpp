#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict converter, BF16, tokenizer, and native closure for GENEB GPTs."""

import argparse
import contextlib
import copy
import hashlib
import http.client
import importlib.util
import io
import json
import math
import shutil
import selectors
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_bytes(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (bits >> 16) & 0xFFFF)


def fixture_scalar(name: str, tensor_index: int, element_index: int) -> float:
    integer = ((tensor_index + 3) * 17 + (element_index + 7) * 13) % 47 - 23
    value = f32(f32(float(integer)) / f32(37.0))
    if "ln_" in name or name == "transformer.ln_f.weight":
        value = f32(f32(1.0) + f32(value * f32(0.04)))
    elif name.endswith(".bias"):
        value = f32(value * f32(0.035))
    elif name in ("transformer.wte.weight", "transformer.wpe.weight"):
        value = f32(value * f32(0.18))
    else:
        value = f32(value * f32(0.11))
    return value


def tensor_payload(name: str, tensor_index: int, count: int, dtype: str) -> bytes:
    output = bytearray()
    for element_index in range(count):
        value = fixture_scalar(name, tensor_index, element_index)
        output.extend(struct.pack("<f", value) if dtype == "F32" else bf16_bytes(value))
    return bytes(output)


def write_safetensors(
    path: Path,
    specs: Sequence[Any],
    omit: Optional[str] = None,
    extra: bool = False,
) -> Dict[str, bytes]:
    header = {"__metadata__": {"format": "pt"}}  # type: Dict[str, Any]
    payloads = {}  # type: Dict[str, bytes]
    offset = 0
    for tensor_index, spec in enumerate(specs):
        if spec.name == omit:
            continue
        count = math.prod(spec.shape)
        payload = tensor_payload(spec.name, tensor_index, count, spec.dtype)
        header[spec.name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads[spec.name] = payload
        offset += len(payload)
    if extra:
        payload = struct.pack("<f", 0.25)
        header["unexpected.weight"] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads["unexpected.weight"] = payload
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    padded = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(padded))
        + padded
        + b"".join(payloads[name] for name in header if name != "__metadata__")
    )
    return payloads


def manifest_digest(specs: Sequence[Any]) -> str:
    value = sorted(
        [
            {"name": spec.name, "dtype": spec.dtype, "shape": list(spec.shape)}
            for spec in specs
        ],
        key=lambda item: item["name"],
    )
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_catalog(source: Path, runtime_id: str, width: int, maximum: int) -> Dict[str, Any]:
    catalog = json.loads(source.read_text(encoding="utf-8"))
    matches = [item for item in catalog["models"] if item["runtime_id"] == runtime_id]
    if len(matches) != 1:
        raise AssertionError("fixture catalog model is missing")
    entry = copy.deepcopy(matches[0])
    entry["context"]["declared_max_tokens"] = maximum
    entry["context"]["reference_max_tokens"] = maximum
    entry["tokenizer"]["max_tokens"] = maximum
    for preset in entry["embedding_presets"].values():
        preset["output_width"] = width
    return {
        "schema_version": catalog["schema_version"],
        "suite": catalog["suite"],
        "models": [entry],
    }


def tokenizer_descriptor(root: Path, asset: Mapping[str, Any]) -> Tuple[Path, bytes]:
    asset_path = root / "assets" / "tokenizer.json"
    payload = canonical_json(asset)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(payload)
    descriptor = {
        "converter.schema": "evo-tokenizer-conversion-receipt",
        "converter.version": 1,
        "compiler_manifest_sha256": "1" * 64,
        "source_receipt_contract_sha256": "2" * 64,
        "tokenizer.profile": "evo-tokenizer-v1",
        "tokenizer.path": "assets/tokenizer.json",
        "tokenizer.sha256": hashlib.sha256(payload).hexdigest(),
        "tokenizer.size": len(payload),
    }
    descriptor_path = root / "tokenizer.descriptor.json"
    write_json(descriptor_path, descriptor)
    return descriptor_path, payload


def gpt2_tokenizer_asset() -> Dict[str, Any]:
    pieces = ["<|endoftext|>", "<pad>", "<unk>", "A", "C", "G", "AC"]
    pieces.extend("fixture-token-%d" % index for index in range(7, 115000))
    return {
        "format": "evo-tokenizer-v1",
        "kind": "bpe",
        "normalization": [],
        "pre_tokenizer": {"kind": "hf-whitespace-ascii"},
        "model": {"merges": [["A", "C"]]},
        "post_processor": {
            "prefix_ids": [],
            "suffix_ids": [],
            "padding": {"side": "right", "pad_id": 0},
        },
        "special_tokens": {
            "unk": None,
            "pad": 0,
            "bos": None,
            "eos": 0,
            "cls": None,
            "sep": None,
            "mask": None,
        },
        "vocab": [
            {"id": index, "piece": piece} for index, piece in enumerate(pieces)
        ],
    }


def dna_tokenizer_asset(helper: Any) -> Dict[str, Any]:
    pieces = helper.dna_gpt_vocab_pieces("dynamic-sixmer")
    return {
        "format": "evo-tokenizer-v1",
        "kind": "kmer",
        "normalization": [],
        "pre_tokenizer": {"kind": "none"},
        "model": {
            "k": 6,
            "stride": 6,
            "tail": "lookup",
            "unknown_policy": "unk",
            "match_special_literals": True,
        },
        "post_processor": {
            "prefix_ids": [21],
            "suffix_ids": [],
            "padding": {"side": "right", "pad_id": 20},
        },
        "special_tokens": {
            "unk": 0,
            "pad": 20,
            "bos": 21,
            "eos": None,
            "cls": None,
            "sep": None,
            "mask": None,
        },
        "vocab": [
            {"id": index, "piece": piece} for index, piece in enumerate(pieces)
        ],
    }


def receipt_entry(path: Path, name: Optional[str] = None) -> Dict[str, Any]:
    return {
        "name": path.name if name is None else name,
        "size": path.stat().st_size,
        "sha256": digest(path),
        "path": str(path.resolve()),
    }


def make_gpt2_case(root: Path, converter: Any, catalog_path: Path) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    topology = {
        "vocab_size": 115000,
        "hidden_size": 4,
        "num_layers": 2,
        "num_attention_heads": 2,
        "inner_width": 16,
        "max_seqlen": 8,
        "norm_epsilon": 0.00001,
    }
    config = root / "config.json"
    write_json(
        config,
        {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 115000,
            "n_embd": 4,
            "n_layer": 2,
            "n_head": 2,
            "n_positions": 8,
            "n_ctx": 8,
            "n_inner": None,
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 0.00001,
            "scale_attn_weights": True,
            "scale_attn_by_inverse_layer_idx": False,
            "reorder_and_upcast_attn": False,
            "torch_dtype": "float32",
            "transformers_version": "4.45.2",
            "attn_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "resid_pdrop": 0.1,
        },
    )
    specs = converter.canonical_tensor_specs(topology)
    weights = root / "model.safetensors"
    payloads = write_safetensors(weights, specs)
    descriptor_path, tokenizer_payload = tokenizer_descriptor(root, gpt2_tokenizer_asset())
    production = json.loads(
        (catalog_path.parent / "geneb-gpt2-models.json").read_text(encoding="utf-8")
    )
    profile = copy.deepcopy(production["models"][0])
    profile["topology"] = topology
    profile["config_sha256"] = digest(config)
    profile["weights_sha256"] = digest(weights)
    profile["weights_size"] = weights.stat().st_size
    profile["source_manifest_sha256"] = manifest_digest(specs)
    profile["source_tensor_count"] = len(specs)
    profile["source_tensor_bytes"] = sum(spec.nbytes for spec in specs)
    profile["tokenizer"]["compiled_asset_sha256"] = hashlib.sha256(
        tokenizer_payload
    ).hexdigest()
    profile["tokenizer"]["compiled_asset_size"] = len(tokenizer_payload)
    profile["tokenizer"]["compiler_manifest_sha256"] = "1" * 64
    profile["tokenizer"]["merge_count"] = 1
    profiles = root / "profiles.json"
    write_json(
        profiles,
        {
            "schema_version": 1,
            "format": converter.PROFILE_FORMAT,
            "models": [profile],
            "code_provenance": production["code_provenance"],
        },
    )
    catalog = root / "catalog.json"
    write_json(catalog, selected_catalog(catalog_path, profile["runtime_id"], 4, 8))
    receipt = root / "source-receipt.json"
    write_json(
        receipt,
        {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": profile["runtime_id"],
            "repo": profile["repo"],
            "requested_revision": profile["requested_revision"],
            "resolved_revision": profile["revision"],
            "files": [receipt_entry(config), receipt_entry(weights)],
        },
    )
    return {
        "root": root,
        "profiles": profiles,
        "catalog": catalog,
        "receipt": receipt,
        "descriptor": descriptor_path,
        "weights": weights,
        "config": config,
        "output": root / "model.evo.safetensors",
        "specs": specs,
        "payloads": payloads,
        "family": "gpt2",
    }


def make_dna_case(
    root: Path,
    converter: Any,
    helper: Any,
    catalog_path: Path,
    omit: Optional[str] = None,
) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    topology = {
        "vocab_size": 19564,
        "hidden_size": 4,
        "num_layers": 2,
        "num_attention_heads": 2,
        "inner_width": 16,
        "max_seqlen": 8,
        "norm_epsilon": 0.00001,
    }
    production = json.loads(
        (catalog_path.parent / "geneb-dna-gpt-models.json").read_text(encoding="utf-8")
    )
    profile = copy.deepcopy(production["models"][1])
    profile["topology"] = topology
    source_specs = converter.source_tensor_specs(profile)
    checkpoint = root / profile["source_filename"]
    payloads = write_safetensors(checkpoint, source_specs, omit=omit)
    if omit is not None:
        source_specs = [spec for spec in source_specs if spec.name != omit]
    profile["source_manifest_sha256"] = manifest_digest(source_specs)
    profile["source_tensor_count"] = len(source_specs)
    profile["source_tensor_bytes"] = sum(spec.nbytes for spec in source_specs)
    descriptor_path, tokenizer_payload = tokenizer_descriptor(
        root, dna_tokenizer_asset(helper)
    )
    profile["tokenizer"]["compiled_asset_sha256"] = hashlib.sha256(
        tokenizer_payload
    ).hexdigest()
    profile["tokenizer"]["compiled_asset_size"] = len(tokenizer_payload)
    profile["tokenizer"]["compiler_manifest_sha256"] = "1" * 64
    profiles = root / "profiles.json"
    write_json(
        profiles,
        {
            "schema_version": 1,
            "format": converter.PROFILE_FORMAT,
            "models": [profile],
            "code_provenance": production["code_provenance"],
        },
    )
    catalog = root / "catalog.json"
    catalog_value = selected_catalog(catalog_path, profile["runtime_id"], 4, 8)
    catalog_source = catalog_value["models"][0]["source"]
    catalog_source["required_files"] = [
        {
            "path": profile["source_filename"],
            "size": checkpoint.stat().st_size,
            "sha256": digest(checkpoint),
        }
    ]
    catalog_source["receipt"]["manifest_status"] = "pinned"
    write_json(catalog, catalog_value)
    receipt = root / "source-receipt.json"
    write_json(
        receipt,
        {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": profile["runtime_id"],
            "source_kind": "google-drive",
            "source_url": catalog_source["url"],
            "files": [receipt_entry(checkpoint)],
        },
    )
    runtime_names = {spec.name for spec in converter.backbone_tensor_specs(topology, "BF16")}
    return {
        "root": root,
        "profiles": profiles,
        "catalog": catalog,
        "receipt": receipt,
        "descriptor": descriptor_path,
        "weights": checkpoint,
        "output": root / "model.evo.safetensors",
        "specs": source_specs,
        "payloads": {name: payload for name, payload in payloads.items() if name in runtime_names},
        "family": "dna",
    }


def converter_arguments(case: Mapping[str, Any]) -> List[str]:
    return [
        "converter",
        "--receipt",
        str(case["receipt"]),
        "--catalog",
        str(case["catalog"]),
        "--profiles",
        str(case["profiles"]),
        "--tokenizer-descriptor",
        str(case["descriptor"]),
        "--tokenizer-root",
        str(case["root"]),
        "--output",
        str(case["output"]),
    ]


def invoke(
    converter: Any,
    case: Mapping[str, Any],
    torch_loader: Optional[Any] = None,
) -> Tuple[int, str, str]:
    old_argv = sys.argv
    old_loader = getattr(converter, "load_torch_checkpoint", None)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        sys.argv = converter_arguments(case)
        if torch_loader is not None:
            converter.load_torch_checkpoint = torch_loader
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = converter.main()
    finally:
        sys.argv = old_argv
        if old_loader is not None:
            converter.load_torch_checkpoint = old_loader
    return result, stdout.getvalue(), stderr.getvalue()


def expect_failure(
    converter: Any,
    case: Mapping[str, Any],
    needle: str,
    torch_loader: Optional[Any] = None,
) -> None:
    result, stdout, stderr = invoke(converter, case, torch_loader)
    if result == 0 or needle not in stderr:
        raise AssertionError(
            "expected failure containing %r:\n%s\n%s" % (needle, stdout, stderr)
        )
    if case["output"].exists():
        raise AssertionError("failed converter published a partial artifact")


def read_artifact(path: Path) -> Tuple[Dict[str, str], Dict[str, bytes], Dict[str, Any]]:
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        payload = source.read()
    metadata = header.pop("__metadata__")
    tensors = {
        name: payload[descriptor["data_offsets"][0] : descriptor["data_offsets"][1]]
        for name, descriptor in header.items()
    }
    return metadata, tensors, header


def expand_bf16(payload: bytes) -> bytes:
    if len(payload) % 2:
        raise AssertionError("BF16 fixture has a partial element")
    output = bytearray(len(payload) * 2)
    for source in range(0, len(payload), 2):
        bits = (payload[source] | (payload[source + 1] << 8)) << 16
        struct.pack_into("<I", output, source * 2, bits)
    return bytes(output)


def run_native(loader: Path, artifact: Path, expect_success: bool) -> None:
    result = subprocess.run(
        [str(loader), "--verify-artifact", str(artifact)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError("native artifact load failed:\n" + result.stderr)
    if not expect_success and result.returncode == 0:
        raise AssertionError("native artifact corruption was accepted")


def run_cpu_adapter(loader: Path, artifact: Path, family: str) -> None:
    result = subprocess.run(
        [str(loader), "--verify-cpu-adapter", str(artifact)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError("native CPU adapter failed:\n" + result.stderr)
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise AssertionError("native CPU adapter emitted invalid JSON") from error
    expected_tokens = [6, 5] if family == "gpt2" else [21, 106]
    expected = {
        "family": family,
        "tokens": expected_tokens,
        "rows": len(expected_tokens),
        "width": 4,
    }
    if observed != expected:
        raise AssertionError(
            "native CPU adapter result differs: %r != %r" % (observed, expected)
        )


def run_cli_server_closure(
    cli: Path, case: Mapping[str, Any], work_dir: Path
) -> None:
    family = str(case["family"])
    root = work_dir / (family + "-frontend")
    root.mkdir(parents=True)
    input_path = root / "input.fasta"
    input_path.write_text(">tiny\nACG\n", encoding="ascii")
    output = root / "embeddings"
    embedded = subprocess.run(
        [
            str(cli),
            "embed",
            "-m",
            str(case["output"]),
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--preset",
            "geneb",
            "--backend",
            "cpu",
            "--profile",
            "cpu-f32",
            "--ctx",
            "8",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if embedded.returncode != 0:
        raise AssertionError("GENEB GPT CLI embed failed:\n" + embedded.stderr)
    manifest = json.loads((output / "embeddings.jsonl").read_text(encoding="utf-8"))
    if (
        manifest.get("backend") != "cpu"
        or manifest.get("profile") != "cpu-f32"
        or manifest.get("preset") != "geneb-v4-normalized"
        or manifest.get("shape") != [1, 4]
        or manifest.get("token_count") != 2
    ):
        raise AssertionError("GENEB GPT CLI embedding manifest differs")

    server = subprocess.Popen(
        [
            str(cli),
            "serve",
            "-m",
            str(case["output"]),
            "--backend",
            "cpu",
            "--profile",
            "cpu-f32",
            "--ctx",
            "8",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if server.stderr is None:
            raise AssertionError("GENEB GPT server stderr is unavailable")
        selector = selectors.DefaultSelector()
        selector.register(server.stderr, selectors.EVENT_READ)
        port = None  # type: Optional[int]
        deadline = time.monotonic() + 20.0
        prefix = b"evo_server listening 127.0.0.1:"
        while time.monotonic() < deadline and port is None:
            if server.poll() is not None:
                raise AssertionError(
                    "GENEB GPT server failed:\n"
                    + server.stderr.read().decode(errors="replace")
                )
            for key, _ in selector.select(timeout=0.25):
                line = key.fileobj.readline()
                if line.startswith(prefix):
                    port = int(line[len(prefix) :])
        if port is None:
            raise AssertionError("GENEB GPT server did not announce its port")
        body = json.dumps(
            {"sequence": "ACG", "preset": "geneb-v4-normalized"}
        )
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request(
            "POST",
            "/v1/embeddings",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        document = json.loads(response.read())
        connection.close()
        if (
            response.status != 200
            or document.get("profile") != "cpu-f32"
            or document.get("preset") != "geneb-v4-normalized"
            or document.get("shape") != [1, 4]
            or document.get("input", {}).get("token_count") != 2
            or len(document.get("embedding", [])) != 4
        ):
            raise AssertionError("GENEB GPT server embedding response differs")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def rewrite_artifact(
    source_path: Path,
    destination: Path,
    metadata_mutation: Optional[Any] = None,
    tensor_mutation: Optional[Any] = None,
) -> None:
    with source_path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        payload = source.read()
    if metadata_mutation is not None:
        metadata_mutation(header["__metadata__"])
    if tensor_mutation is not None:
        tensor_mutation(header)
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    padded = raw + b" " * ((-len(raw)) % 8)
    destination.write_bytes(struct.pack("<Q", len(padded)) + padded + payload)


def validate_output(
    case: Mapping[str, Any], converter: Any, native_loader: Path, skip_native: bool
) -> None:
    metadata, tensors, _ = read_artifact(case["output"])
    expected = {
        "runtime.profile": "s:" + converter.ARTIFACT_PROFILE,
        "runtime.abi": "s:" + converter.RUNTIME_ABI,
        "runtime.embedding_layer_count": "u:3",
        "model.architecture": "s:" + converter.RUNTIME_ARCHITECTURE,
        "config.hidden_size": "u:4",
        "config.num_layers": "u:2",
        "config.max_seqlen": "u:8",
        "tokenizer.profile": "s:evo-tokenizer-v1",
        "tokenizer.path": "s:assets/tokenizer.json",
    }
    profiles, code, _ = converter.load_profiles(case["profiles"])
    expected["source.converter_profile_contract_sha256"] = (
        "s:"
        + converter.converter_profile_contract_sha256(
            1,
            converter.PROFILE_FORMAT,
            profiles[next(iter(profiles))],
            {"code_provenance": code},
        )
    )
    wrong = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if wrong:
        raise AssertionError("artifact metadata differs: %s" % wrong)
    if any(key.startswith("converter.") for key in metadata):
        raise AssertionError("tokenizer conversion receipt leaked into model metadata")
    if case["family"] == "gpt2":
        if set(tensors) != set(case["payloads"]):
            raise AssertionError("GPT-2 canonical tensor set differs")
        for name, payload in case["payloads"].items():
            if tensors[name] != payload:
                raise AssertionError("GPT-2 F32 tensor bytes changed: %s" % name)
    else:
        if set(tensors) != set(case["payloads"]):
            raise AssertionError("DNA-GPT task heads were not dropped exactly")
        for name, payload in case["payloads"].items():
            if tensors[name] != expand_bf16(payload):
                raise AssertionError("DNA-GPT BF16 expansion differs: %s" % name)
        if metadata.get("source.immutable") != "b:0" or metadata.get("source.revision") != "s:":
            raise AssertionError("manual DNA-GPT source provenance was falsified")
    if not skip_native:
        run_native(native_loader, case["output"], True)
        run_cpu_adapter(native_loader, case["output"], case["family"])


def copied_case(case: Mapping[str, Any], destination: Path) -> Dict[str, Any]:
    shutil.copytree(case["root"], destination)
    result = dict(case)
    result["root"] = destination
    for key in ("profiles", "catalog", "receipt", "descriptor", "weights", "output"):
        result[key] = destination / Path(case[key]).relative_to(case["root"])
    if "config" in case:
        result["config"] = destination / Path(case["config"]).relative_to(case["root"])
    if result["output"].exists():
        result["output"].unlink()
    receipt = json.loads(result["receipt"].read_text(encoding="utf-8"))
    for entry in receipt["files"]:
        entry["path"] = str((destination / Path(entry["path"]).name).resolve())
    write_json(result["receipt"], receipt)
    return result


def test_gpt2(
    args: argparse.Namespace, converter: Any, catalog_path: Path
) -> Mapping[str, Any]:
    case = make_gpt2_case(args.work_dir / "gpt2-valid", converter, catalog_path)
    result, stdout, stderr = invoke(converter, case)
    if result != 0:
        raise AssertionError("valid GPT-2 conversion failed:\n%s\n%s" % (stdout, stderr))
    validate_output(case, converter, args.native_loader, args.skip_native)

    original_profiles = case["profiles"].read_bytes()
    sibling = dict(case)
    sibling["output"] = case["root"] / "model-with-sibling.evo.safetensors"
    sibling_profiles = json.loads(
        sibling["profiles"].read_text(encoding="utf-8")
    )
    sibling_row = copy.deepcopy(sibling_profiles["models"][0])
    sibling_row["runtime_id"] = "geneb-gpt2-sibling-fixture"
    sibling_row["geneb_model_id"] = "GPT2SiblingFixture"
    sibling_row["paper_name"] = "GPT2 sibling fixture"
    sibling_row["repo"] = "fixture/gpt2-sibling"
    sibling_row["revision"] = "f" * 40
    sibling_profiles["models"].append(sibling_row)
    write_json(sibling["profiles"], sibling_profiles)
    result, stdout, stderr = invoke(converter, sibling)
    if result != 0:
        raise AssertionError(
            "GPT-2 sibling-row conversion failed:\n%s\n%s" % (stdout, stderr)
        )
    if sibling["output"].read_bytes() != case["output"].read_bytes():
        raise AssertionError("GPT-2 sibling profile changed selected artifact bytes")

    case["profiles"].write_bytes(original_profiles)
    own = dict(case)
    own["output"] = case["root"] / "model-with-own-change.evo.safetensors"
    own_profiles = json.loads(own["profiles"].read_text(encoding="utf-8"))
    own_profiles["models"][0]["paper_name"] += " changed"
    write_json(own["profiles"], own_profiles)
    expect_failure(converter, own, "catalog GPT-2 identity differs")
    case["profiles"].write_bytes(original_profiles)

    bad = copied_case(case, args.work_dir / "gpt2-bad-revision")
    receipt = json.loads(bad["receipt"].read_text(encoding="utf-8"))
    receipt["requested_revision"] = receipt["resolved_revision"]
    write_json(bad["receipt"], receipt)
    expect_failure(converter, bad, "identity/revision is not pinned")

    bad = copied_case(case, args.work_dir / "gpt2-bad-integrity")
    with bad["weights"].open("ab") as output:
        output.write(b"X")
    expect_failure(converter, bad, "source receipt integrity mismatch")

    bad = copied_case(case, args.work_dir / "gpt2-bad-config")
    config = json.loads(bad["config"].read_text(encoding="utf-8"))
    config["activation_function"] = "relu"
    write_json(bad["config"], config)
    receipt = json.loads(bad["receipt"].read_text(encoding="utf-8"))
    receipt["files"][0] = receipt_entry(bad["config"])
    write_json(bad["receipt"], receipt)
    profiles = json.loads(bad["profiles"].read_text(encoding="utf-8"))
    profiles["models"][0]["config_sha256"] = digest(bad["config"])
    write_json(bad["profiles"], profiles)
    expect_failure(converter, bad, "config semantics differ")

    bad = copied_case(case, args.work_dir / "gpt2-bad-tokenizer")
    with (bad["root"] / "assets" / "tokenizer.json").open("ab") as output:
        output.write(b"X")
    expect_failure(converter, bad, "tokenizer asset size/SHA256 differs")

    bad = copied_case(case, args.work_dir / "gpt2-detached-tokenizer-root")
    bad["output"] = args.work_dir / "gpt2-detached-output" / "model.evo.safetensors"
    bad["output"].parent.mkdir(parents=True)
    expect_failure(
        converter,
        bad,
        "tokenizer root must equal the output artifact directory",
    )
    return case


def test_dna(
    args: argparse.Namespace,
    converter: Any,
    helper: Any,
    checkpoint: Any,
    catalog_path: Path,
) -> Mapping[str, Any]:
    def reader(path: Path) -> Any:
        return checkpoint.read_safetensors(path)

    case = make_dna_case(args.work_dir / "dna-valid", converter, helper, catalog_path)
    result, stdout, stderr = invoke(converter, case, reader)
    if result != 0:
        raise AssertionError("valid DNA-GPT conversion failed:\n%s\n%s" % (stdout, stderr))
    validate_output(case, converter, args.native_loader, args.skip_native)

    bad = copied_case(case, args.work_dir / "dna-bad-provider-fields")
    receipt = json.loads(bad["receipt"].read_text(encoding="utf-8"))
    receipt["resolved_revision"] = "a" * 40
    write_json(bad["receipt"], receipt)
    calls = []  # type: List[Path]

    def spy(path: Path) -> Any:
        calls.append(path)
        return reader(path)

    expect_failure(converter, bad, "manual source receipt fields differ", spy)
    if calls:
        raise AssertionError("manual provider-field failure reached the checkpoint parser")

    bad = copied_case(case, args.work_dir / "dna-bad-integrity")
    with bad["weights"].open("ab") as output:
        output.write(b"X")
    calls = []
    expect_failure(converter, bad, "source receipt integrity mismatch", spy)
    if calls:
        raise AssertionError("manual receipt integrity failure reached the parser")

    missing_name = converter.source_tensor_specs(
        json.loads(case["profiles"].read_text(encoding="utf-8"))["models"][0]
    )[-1].name
    bad = make_dna_case(
        args.work_dir / "dna-missing-task-head",
        converter,
        helper,
        catalog_path,
        omit=missing_name,
    )
    expect_failure(converter, bad, "tensor manifest mismatch", reader)

    bad = copied_case(case, args.work_dir / "dna-bad-descriptor")
    descriptor = json.loads(bad["descriptor"].read_text(encoding="utf-8"))
    descriptor["tokenizer.extra"] = "forbidden"
    write_json(bad["descriptor"], descriptor)
    expect_failure(converter, bad, "tokenizer descriptor fields differ", reader)

    bad = copied_case(case, args.work_dir / "dna-bad-tokenizer-manifest")
    descriptor = json.loads(bad["descriptor"].read_text(encoding="utf-8"))
    descriptor["compiler_manifest_sha256"] = "3" * 64
    write_json(bad["descriptor"], descriptor)
    expect_failure(converter, bad, "compiler manifest SHA256 differs", reader)

    bad = copied_case(case, args.work_dir / "dna-bad-special-policy")
    catalog = json.loads(bad["catalog"].read_text(encoding="utf-8"))
    catalog["models"][0]["tokenizer"]["add_special_tokens"] = True
    write_json(bad["catalog"], catalog)
    expect_failure(
        converter,
        bad,
        "catalog DNA-GPT tokenizer/context/preset differs",
        reader,
    )

    bad = copied_case(case, args.work_dir / "dna-detached-tokenizer-root")
    bad["output"] = args.work_dir / "dna-detached-output" / "model.evo.safetensors"
    bad["output"].parent.mkdir(parents=True)
    expect_failure(
        converter,
        bad,
        "tokenizer root must equal the output artifact directory",
        reader,
    )
    return case


def test_bf16_bits(helper: Any) -> None:
    class Source:
        name = "probe"
        dtype = "BF16"
        shape = (5,)
        nbytes = 10

        def iter_chunks(self, chunk_size: int) -> Any:
            del chunk_size
            payload = struct.pack("<5H", 0x0000, 0x8000, 0x3F80, 0x7F80, 0x7FC1)
            yield memoryview(payload[:3])
            yield memoryview(payload[3:])

    converted = b"".join(
        bytes(chunk) for chunk in helper.CanonicalF32TensorSource(Source()).iter_chunks(7)
    )
    expected = struct.pack(
        "<5I", 0x00000000, 0x80000000, 0x3F800000, 0x7F800000, 0x7FC10000
    )
    if converted != expected:
        raise AssertionError("BF16 signed-zero/finite/inf/NaN bit expansion differs")


def test_native_corruption(
    args: argparse.Namespace, gpt2_case: Mapping[str, Any], dna_case: Mapping[str, Any]
) -> None:
    if args.skip_native:
        return
    corrupt = args.work_dir / "gpt2-bad-code.safetensors"
    rewrite_artifact(
        gpt2_case["output"],
        corrupt,
        metadata_mutation=lambda metadata: metadata.__setitem__(
            "source.transformers.activations_sha256", "s:" + "0" * 64
        ),
    )
    run_native(args.native_loader, corrupt, False)

    corrupt = args.work_dir / "dna-bad-kind.safetensors"
    rewrite_artifact(
        dna_case["output"],
        corrupt,
        metadata_mutation=lambda metadata: metadata.__setitem__(
            "dna_gpt.tokenizer_kind", "s:static-sixmer"
        ),
    )
    run_native(args.native_loader, corrupt, False)

    corrupt = args.work_dir / "dna-partial-tokenizer.safetensors"
    rewrite_artifact(
        dna_case["output"],
        corrupt,
        metadata_mutation=lambda metadata: metadata.pop("tokenizer.sha256"),
    )
    run_native(args.native_loader, corrupt, False)


def validate_production_profiles(gpt2: Any, dna: Any, args: argparse.Namespace) -> None:
    gpt_profiles, _, _ = gpt2.load_profiles(args.gpt2_profiles.resolve())
    if set(gpt_profiles) != {
        "geneb-gpt2-gene-multi-v2",
        "geneb-gpt2-gene-v1",
    }:
        raise AssertionError("production GPT2-Gene profile IDs differ")
    for profile in gpt_profiles.values():
        if profile["topology"]["vocab_size"] != 115000:
            raise AssertionError("GPT2-Gene vocabulary is not fixed at 115000")
        if len(gpt2.canonical_tensor_specs(profile["topology"])) != 148:
            raise AssertionError("GPT2-Gene exact tensor set is not 148")
    dna_profiles, _, _ = dna.load_profiles(args.dna_profiles.resolve())
    expected = {
        "geneb-dna-gpt-0-1b-h": ("F32", 84, 15659),
        "geneb-dna-gpt-3b-m": ("BF16", 372, 19564),
    }
    if set(dna_profiles) != set(expected):
        raise AssertionError("production DNA-GPT profile IDs differ")
    for runtime_id, profile in dna_profiles.items():
        dtype, count, vocabulary = expected[runtime_id]
        if (
            profile["source_dtype"] != dtype
            or len(dna.source_tensor_specs(profile)) != count
            or profile["topology"]["vocab_size"] != vocabulary
        ):
            raise AssertionError("%s production source contract differs" % runtime_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2-converter", required=True, type=Path)
    parser.add_argument("--dna-converter", required=True, type=Path)
    parser.add_argument("--gpt2-profiles", required=True, type=Path)
    parser.add_argument("--dna-profiles", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--native-loader", required=True, type=Path)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.gpt2_converter.resolve().parent))
    gpt2 = load_module("geneb_gpt2_converter", args.gpt2_converter.resolve())
    dna = load_module("geneb_dna_gpt_converter", args.dna_converter.resolve())
    from evo import geneb_gpt_artifact as helper
    from evo import hf_checkpoint as checkpoint

    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    validate_production_profiles(gpt2, dna, args)
    test_bf16_bits(helper)
    gpt2_case = test_gpt2(args, gpt2, args.catalog.resolve())
    dna_case = test_dna(args, dna, helper, checkpoint, args.catalog.resolve())
    test_native_corruption(args, gpt2_case, dna_case)
    if not args.skip_native:
        if args.cli is None:
            raise AssertionError("--cli is required for native frontend closure")
        run_cli_server_closure(args.cli, gpt2_case, args.work_dir)
        run_cli_server_closure(args.cli, dna_case, args.work_dir)
    print("GENEB GPT converter/BF16/tokenizer/native contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
