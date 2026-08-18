#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict converter/native closure tests for the GENEB Mamba artifact ABI."""

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def tensor_bytes(count: int, seed: int) -> bytes:
    values = []
    for index in range(count):
        integer = ((seed + 3) * 17 + (index + 7) * 13) % 41 - 20
        values.append(float(integer) / 37.0)
    return struct.pack("<%df" % count, *values)


def elements(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


Tensor = Tuple[str, Tuple[int, ...], bytes]


class MemoryTensor:
    def __init__(self, name: str, dtype: str, shape: Tuple[int, ...], raw: bytes) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.nbytes = len(raw)
        self.raw = raw

    def iter_chunks(self, chunk_size: int) -> Any:
        for offset in range(0, len(self.raw), chunk_size):
            yield memoryview(self.raw[offset : offset + chunk_size])


def write_safetensors(target: Path, tensors: Mapping[str, Tensor]) -> None:
    header = {}  # type: Dict[str, Any]
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        begin = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    encoded += b" " * ((-len(encoded)) % 8)
    write_bytes(target, struct.pack("<Q", len(encoded)) + encoded + payload)


def tiny_source_tensors() -> Dict[str, Tensor]:
    tensors = {}  # type: Dict[str, Tensor]
    seed = 0

    def add(name: str, shape: Tuple[int, ...], dtype: str = "F32", values: Optional[Sequence[int]] = None) -> None:
        nonlocal seed
        if dtype == "F32":
            raw = tensor_bytes(elements(shape), seed)
        else:
            if values is None:
                raise AssertionError("I64 fixture needs values")
            raw = struct.pack("<%dq" % len(values), *values)
        tensors[name] = (dtype, shape, raw)
        seed += 1

    add("caduceus.backbone.embeddings.word_embeddings.embedding.weight", (6, 2))
    prefix = "caduceus.backbone.layers.0."
    add(prefix + "norm.weight", (2,))
    mixer = prefix + "mixer.submodule."
    fields = (
        ("in_proj.weight", (8, 2)),
        ("conv1d.weight", (4, 1, 2)),
        ("conv1d.bias", (4,)),
        ("x_proj.weight", (5, 4)),
        ("dt_proj.weight", (4, 1)),
        ("dt_proj.bias", (4,)),
        ("A_log", (4, 2)),
        ("D", (4,)),
        ("out_proj.weight", (2, 4)),
    )
    for direction in ("mamba_fwd.", "mamba_rev."):
        for suffix, shape in fields:
            if direction == "mamba_rev." and suffix in ("in_proj.weight", "out_proj.weight"):
                continue
            add(mixer + direction + suffix, shape)
    add("caduceus.backbone.norm_f.weight", (2,))
    complement = (0, 1, 2, 4, 3, 5)
    add("caduceus.backbone.embeddings.word_embeddings.complement_map", (6,), "I64", complement)
    add("lm_head.complement_map", (6,), "I64", complement)
    return tensors


def load_converter(path: Path) -> Any:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location("geneb_mamba_converter", path)
    if specification is None or specification.loader is None:
        raise AssertionError("cannot import converter")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def production_contract(
    converter: Any, profiles_path: Path, catalog_path: Path
) -> None:
    profiles, _ = converter.load_profiles(profiles_path)
    expected_mamba2_dependency = (
        "mamba-ssm-2.2.4@95d8aba8a8c75aedcaa6143713b11e745e7cd0d9;"
        "sdist-sha256=e4114c69302796c91b71e90032c2d974f611608fab331582a80de6eaf075efb9"
    )
    if converter.IMPLEMENTATION_CONTRACTS.get("eccdna-mamba2") != expected_mamba2_dependency:
        raise AssertionError("Mamba2 dependency revision/sdist receipt differs")
    expected = {
        "geneb-eccdna-mamba": ("eccdna-mamba2", 0, 0, 4096, 584, 580),
        "geneb-plant-caduceus": ("caduceus-mamba1", 0, 512, 7, 613, 546),
        "geneb-caduceus-ps-131k": ("caduceus-mamba1", 131072, 131072, 12, 276, 274),
        "geneb-caduceus-ph-1k": ("caduceus-mamba1", 1024, 1024, 12, 70, 70),
    }
    if set(profiles) != set(expected):
        raise AssertionError("production Mamba profile IDs differ")
    for runtime_id, (variant, hard_max, advertised, tokenizer_vocab, source_count, runtime_count) in expected.items():
        profile = profiles[runtime_id]
        # This must use the checked-in production catalog. A catalog synthesized
        # from the converter profile cannot detect tokenizer/context drift.
        converter.load_catalog_entry(catalog_path, profile)
        topology = profile["topology"]
        source_specs, mapping, aliases, buffers = converter.source_contract(profile)
        if (
            topology["variant"] != variant
            or topology["max_seqlen"] != hard_max
            or topology["advertised_training_length"] != advertised
            or profile["tokenizer"]["emitted_vocab_size"] != tokenizer_vocab
            or len(source_specs) != source_count
            or len(converter.canonical_tensor_specs(topology)) != runtime_count
            or len(mapping) != runtime_count
        ):
            raise AssertionError("production profile topology/manifest differs for " + runtime_id)
        if runtime_id == "geneb-plant-caduceus" and (len(aliases) != 65 or len(buffers) != 2):
            raise AssertionError("Plant tied alias/RC buffer contract differs")
        if runtime_id == "geneb-plant-caduceus":
            manifest_path = (
                profiles_path.parent
                / "tokenizers"
                / "geneb-plant-caduceus-bpe-v1.json"
            )
            manifest_sha256 = digest(manifest_path.read_bytes())
            if (
                manifest_sha256
                != "348b2a93e9c035dba994cf3944a29d28ab193c1ae2539ab1e1048d2a907a7b69"
                or profile["tokenizer"]["compiler_manifest_sha256"]
                != manifest_sha256
            ):
                raise AssertionError(
                    "Plant tokenizer compiler manifest/profile pin differs"
                )
        if runtime_id == "geneb-eccdna-mamba":
            source_names = {spec.name for spec in source_specs}
            if not {
                "mamba_forward.backbone.embedding.weight",
                "mamba_forward.lm_head.weight",
                "mamba_backward.backbone.embedding.weight",
                "mamba_backward.lm_head.weight",
            } <= source_names:
                raise AssertionError("ecc omitted-source manifest is incomplete")

    plant = copy.deepcopy(profiles["geneb-plant-caduceus"])
    plant["topology"].update(
        {
            "vocab_size": 6,
            "width": 2,
            "output_width": 4,
            "num_layers": 1,
            "inner_width": 4,
            "state_width": 2,
            "conv_width": 2,
            "time_step_rank": 1,
            "complement_map": [0, 1, 2, 4, 3, 5],
        }
    )
    source_specs, _, aliases, buffers = converter.source_contract(plant)
    raw_by_name = {}  # type: Dict[str, bytes]
    for index, spec in enumerate(source_specs):
        if spec.name in buffers:
            raw_by_name[spec.name] = struct.pack("<%dq" % len(buffers[spec.name]), *buffers[spec.name])
        elif spec.name not in aliases:
            raw_by_name[spec.name] = tensor_bytes(elements(spec.shape), index)
    for alias, canonical_name in aliases.items():
        raw_by_name[alias] = raw_by_name[canonical_name]
    memory_tensors = [MemoryTensor(spec.name, spec.dtype, spec.shape, raw_by_name[spec.name]) for spec in source_specs]
    if len(converter.validated_runtime_tensors(memory_tensors, plant)) != 19:
        raise AssertionError("tiny Plant .bin alias closure differs")
    corrupt_name = sorted(aliases)[0]
    corrupt = list(memory_tensors)
    corrupt_index = next(index for index, tensor in enumerate(corrupt) if tensor.name == corrupt_name)
    original = corrupt[corrupt_index]
    damaged = bytearray(original.raw)
    damaged[0] ^= 1
    corrupt[corrupt_index] = MemoryTensor(original.name, original.dtype, original.shape, bytes(damaged))
    try:
        converter.validated_runtime_tensors(corrupt, plant)
    except converter.ConversionError:
        pass
    else:
        raise AssertionError("Plant unequal tied alias was accepted")


def run(command: Sequence[str], expect_ok: bool, label: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if (completed.returncode == 0) != expect_ok:
        raise AssertionError(
            "%s: return=%d\nstdout=%s\nstderr=%s"
            % (label, completed.returncode, completed.stdout, completed.stderr)
        )
    return completed


class TinyCase:
    def __init__(self, converter: Path, profiles_path: Path, catalog_path: Path, native_loader: Path, root: Path) -> None:
        self.converter = converter
        self.native_loader = native_loader
        self.root = root
        profiles_document = json.loads(profiles_path.read_text(encoding="utf-8"))
        self.profile_root = {
            "schema_version": profiles_document["schema_version"],
            "format": profiles_document["format"],
            "implementation_contracts": profiles_document["implementation_contracts"],
            "models": [],
        }
        source_profile = next(item for item in profiles_document["models"] if item["runtime_id"] == "geneb-caduceus-ps-131k")
        self.profile = copy.deepcopy(source_profile)
        self.profile["topology"] = {
            "variant": "caduceus-mamba1",
            "vocab_size": 6,
            "width": 2,
            "output_width": 4,
            "num_layers": 1,
            "max_seqlen": 3,
            "advertised_training_length": 512,
            "inner_width": 4,
            "state_width": 2,
            "conv_width": 2,
            "time_step_rank": 1,
            "mlp_width": 0,
            "head_width": 0,
            "heads": 0,
            "groups": 0,
            "norm_epsilon": 0.00001,
            "rcps": True,
            "complement_map": [0, 1, 2, 4, 3, 5],
        }
        self.config = copy.deepcopy(self.profile["config_required"])
        self.config.update({"d_model": 2, "n_layer": 1, "vocab_size": 6})
        self.config["ssm_cfg"] = copy.deepcopy(self.config["ssm_cfg"])
        self.config["ssm_cfg"].update({"d_state": 2, "d_conv": 2})
        self.profile["config_required"] = copy.deepcopy(self.config)
        self.profile["remote_code"] = {
            "configuration_caduceus.py": "0" * 64,
            "modeling_caduceus.py": "0" * 64,
            "modeling_rcps.py": "0" * 64,
        }
        self.tokenizer_asset = canonical({"format": "evo-tokenizer-v1", "fixture": "caduceus"})
        self.profile["tokenizer"].update(
            {
                "compiler_manifest_sha256": "1" * 64,
                "compiled_asset_sha256": digest(self.tokenizer_asset),
                "compiled_asset_size": len(self.tokenizer_asset),
                "emitted_vocab_size": 6,
            }
        )
        catalog_document = json.loads(catalog_path.read_text(encoding="utf-8"))
        entry = copy.deepcopy(next(item for item in catalog_document["models"] if item["runtime_id"] == self.profile["runtime_id"]))
        entry["context"]["declared_max_tokens"] = 3
        entry["context"]["reference_max_tokens"] = 3
        entry["context"]["unknown_fields"] = []
        entry["tokenizer"]["max_tokens"] = 3
        entry["tokenizer"]["unknown_fields"] = ["assets"]
        for name in ("reference", "normalized"):
            entry["embedding_presets"][name]["output_width"] = 4
        self.catalog_root = copy.deepcopy(catalog_document)
        self.catalog_root["models"] = [entry]
        self.tensors = tiny_source_tensors()

    def publish(self, tensors: Optional[Mapping[str, Tensor]] = None, config: Optional[Mapping[str, Any]] = None, extra_receipt_file: bool = False, wrong_tokenizer_manifest: bool = False) -> Tuple[Path, Path, Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        tensor_values = dict(self.tensors if tensors is None else tensors)
        config_value = dict(self.config if config is None else config)
        config_payload = canonical(config_value)
        config_path = self.root / "config.json"
        write_bytes(config_path, config_payload)
        weights_path = self.root / "model.safetensors"
        write_safetensors(weights_path, tensor_values)
        profile = copy.deepcopy(self.profile)
        profile["config_sha256"] = digest(config_payload)
        profile["source"].update(
            {
                "format": "safetensors",
                "weights_name": "model.safetensors",
                "weights_size": weights_path.stat().st_size,
                "weights_sha256": digest(weights_path.read_bytes()),
            }
        )
        files = []
        for name in ("configuration_caduceus.py", "modeling_caduceus.py", "modeling_rcps.py"):
            payload = ("# tiny pinned %s\n" % name).encode("ascii")
            source_path = self.root / name
            write_bytes(source_path, payload)
            profile["remote_code"][name] = digest(payload)
        profile_root = copy.deepcopy(self.profile_root)
        profile_root["models"] = [profile]
        profiles_path = self.root / "profiles.json"
        write_bytes(profiles_path, canonical(profile_root))
        catalog_path = self.root / "catalog.json"
        write_bytes(catalog_path, canonical(self.catalog_root))
        receipt_paths = [config_path, weights_path] + [self.root / name for name in sorted(profile["remote_code"])]
        if extra_receipt_file:
            extra = self.root / "unexpected.txt"
            write_bytes(extra, b"unexpected\n")
            receipt_paths.append(extra)
        for source_path in receipt_paths:
            payload = source_path.read_bytes()
            files.append({"name": source_path.name, "size": len(payload), "sha256": digest(payload), "path": str(source_path.resolve())})
        receipt = {
            "schema_version": 1,
            "kind": "source-checkpoint",
            "model_id": profile["runtime_id"],
            "repo": profile["repo"],
            "requested_revision": "main",
            "resolved_revision": profile["revision"],
            "files": files,
        }
        receipt_path = self.root / "receipt.json"
        write_bytes(receipt_path, canonical(receipt))
        asset_path = self.root / "tokenizer.asset.json"
        write_bytes(asset_path, self.tokenizer_asset)
        descriptor = {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": "2" * 64 if wrong_tokenizer_manifest else profile["tokenizer"]["compiler_manifest_sha256"],
            "source_receipt_contract_sha256": "3" * 64,
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": asset_path.name,
            "tokenizer.sha256": digest(self.tokenizer_asset),
            "tokenizer.size": len(self.tokenizer_asset),
        }
        descriptor_path = self.root / "tokenizer-descriptor.json"
        write_bytes(descriptor_path, canonical(descriptor))
        return profiles_path, catalog_path, receipt_path

    def command(self, profiles: Path, catalog: Path, receipt: Path, output: Path) -> Sequence[str]:
        return [
            sys.executable, str(self.converter), "--profiles", str(profiles),
            "--catalog", str(catalog), "--receipt", str(receipt),
            "--tokenizer-descriptor", str(self.root / "tokenizer-descriptor.json"),
            "--tokenizer-root", str(self.root), "--output", str(output),
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--native-loader", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    converter = load_converter(args.converter.resolve())
    production_contract(
        converter, args.profiles.resolve(), args.catalog.resolve()
    )
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    case = TinyCase(args.converter.resolve(), args.profiles.resolve(), args.catalog.resolve(), args.native_loader.resolve(), args.work_dir.resolve())

    profiles, catalog, receipt = case.publish()
    output_a = case.root / "tiny-a.safetensors"
    output_b = case.root / "tiny-b.safetensors"
    run(case.command(profiles, catalog, receipt, output_a), True, "valid conversion")
    run(case.command(profiles, catalog, receipt, output_b), True, "deterministic conversion")
    if output_a.read_bytes() != output_b.read_bytes():
        raise AssertionError("converter output is not deterministic")
    force_command = list(case.command(profiles, catalog, receipt, output_a)) + ["--force"]
    run(force_command, True, "atomic force conversion")
    run([str(args.native_loader.resolve()), "--artifact", str(output_a)], True, "native ModelFile closure")
    raw_header_size = struct.unpack("<Q", output_a.read_bytes()[:8])[0]
    header = json.loads(output_a.read_bytes()[8 : 8 + raw_header_size])
    metadata = header.get("__metadata__", {})
    if metadata.get("mamba.max_sequence_length") != "u:3" or metadata.get("mamba.advertised_training_length") != "u:512":
        raise AssertionError("hard/advertised context metadata are conflated")
    if metadata.get("runtime.tokenizer_vocabulary_size") != "u:6" or metadata.get("mamba.vocab_size") != "u:6":
        raise AssertionError("tokenizer/physical vocabulary metadata differ")

    missing = dict(case.tensors)
    missing.pop("lm_head.complement_map")
    profiles, catalog, receipt = case.publish(missing)
    run(case.command(profiles, catalog, receipt, case.root / "missing.safetensors"), False, "missing tensor")

    extra = dict(case.tensors)
    extra["unexpected.weight"] = ("F32", (1,), struct.pack("<f", 0.0))
    profiles, catalog, receipt = case.publish(extra)
    run(case.command(profiles, catalog, receipt, case.root / "extra.safetensors"), False, "extra tensor")

    wrong_dtype = dict(case.tensors)
    name = "caduceus.backbone.layers.0.norm.weight"
    wrong_dtype[name] = ("I64", (2,), struct.pack("<2q", 1, 1))
    profiles, catalog, receipt = case.publish(wrong_dtype)
    run(case.command(profiles, catalog, receipt, case.root / "dtype.safetensors"), False, "wrong dtype")

    wrong_map = dict(case.tensors)
    name = "lm_head.complement_map"
    wrong_map[name] = ("I64", (6,), struct.pack("<6q", 0, 1, 2, 3, 4, 5))
    profiles, catalog, receipt = case.publish(wrong_map)
    run(case.command(profiles, catalog, receipt, case.root / "map.safetensors"), False, "wrong complement map")

    bad_config = copy.deepcopy(case.config)
    bad_config["d_model"] = 3
    profiles, catalog, receipt = case.publish(config=bad_config)
    run(case.command(profiles, catalog, receipt, case.root / "config.safetensors"), False, "wrong config semantics")

    profiles, catalog, receipt = case.publish(extra_receipt_file=True)
    run(
        case.command(profiles, catalog, receipt, case.root / "receipt-with-extra.safetensors"),
        True,
        "verified non-critical receipt file",
    )
    write_bytes(case.root / "unexpected.txt", b"corrupt after receipt\n")
    run(
        case.command(profiles, catalog, receipt, case.root / "receipt-corrupt-extra.safetensors"),
        False,
        "receipt integrity mismatch",
    )

    profiles, catalog, receipt = case.publish(wrong_tokenizer_manifest=True)
    run(case.command(profiles, catalog, receipt, case.root / "tokenizer.safetensors"), False, "wrong tokenizer compiler")

    print("GENEB Mamba converter contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
