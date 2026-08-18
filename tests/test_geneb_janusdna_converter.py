#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict corruption and atomicity gates for the JanusDNA converter."""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_error(callable_value: Any, needle: str) -> None:
    try:
        callable_value()
    except Exception as error:  # noqa: BLE001 - assert converter typed boundary
        if needle not in str(error):
            raise AssertionError(
                "expected error containing %r, got %r" % (needle, error)
            ) from error
        return
    raise AssertionError("expected error containing %r" % needle)


def write_json(path: Path, value: Mapping[str, Any]) -> bytes:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    path.write_bytes(payload)
    return payload


def load_converter(path: Path) -> Any:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("geneb_janusdna_converter", path)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load converter module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStorage:
    next_pointer = 1000

    def __init__(self, nbytes: int, payload: bytes = b"") -> None:
        self._nbytes = nbytes
        self.payload = payload
        self.pointer = FakeStorage.next_pointer
        FakeStorage.next_pointer += 1

    def nbytes(self) -> int:
        return self._nbytes

    def data_ptr(self) -> int:
        return self.pointer

    def _write_file(
        self, output: Any, is_real_file: bool, save_size: bool, element_size: int
    ) -> None:
        check(is_real_file and not save_size and element_size == 1, "raw writer flags")
        os.write(output.fileno(), self.payload)


class FakeDevice:
    type = "cpu"


class FakeTensor:
    def __init__(
        self,
        shape: Sequence[int],
        dtype: str,
        layout: str,
        storage: FakeStorage,
        scalar: int = 0,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self.device = FakeDevice()
        self.storage = storage
        self.scalar = scalar
        self.contiguous = True
        self.offset = 0

    def is_contiguous(self) -> bool:
        return self.contiguous

    def storage_offset(self) -> int:
        return self.offset

    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    def element_size(self) -> int:
        return 8 if self.dtype == "I64" else 4

    def untyped_storage(self) -> FakeStorage:
        return self.storage

    def item(self) -> int:
        return self.scalar


class FakeTorch:
    Tensor = FakeTensor
    float32 = "F32"
    int64 = "I64"
    strided = "strided"

    @staticmethod
    def equal(left: FakeTensor, right: FakeTensor) -> bool:
        return left.storage is right.storage and left.shape == right.shape


def fake_checkpoint(converter: Any, profile: Mapping[str, Any]) -> Dict[str, Any]:
    topology = profile["topology"]
    alias_owner = {}  # type: Dict[str, str]
    for forward, reverse in converter.expected_alias_pairs(topology):
        alias_owner[reverse] = forward
    tensors = {}  # type: Dict[str, FakeTensor]
    storages = {}  # type: Dict[str, FakeStorage]
    for spec in converter.source_tensor_specs(topology):
        owner = alias_owner.get(spec.name, spec.name)
        if owner not in storages:
            storages[owner] = FakeStorage(spec.nbytes)
        tensors[converter.raw_source_name(spec.name)] = FakeTensor(
            spec.shape, "F32", "strided", storages[owner]
        )
    for name, value in profile["source"]["metric_values"].items():
        tensors[name] = FakeTensor((), "I64", "strided", FakeStorage(8), value)
    return {"state_dict": tensors}


@contextlib.contextmanager
def fake_torch_import(converter: Any) -> Any:
    original = converter.importlib.import_module

    def replacement(name: str) -> Any:
        if name == "torch":
            return FakeTorch
        return original(name)

    converter.importlib.import_module = replacement
    try:
        yield
    finally:
        converter.importlib.import_module = original


def check_manifest_corruptions(converter: Any, profile: Mapping[str, Any]) -> None:
    with fake_torch_import(converter):
        checkpoint = fake_checkpoint(converter, profile)
        tensors = converter.runtime_tensors(checkpoint, profile)
        expected = converter.canonical_tensor_specs(profile["topology"])
        check(len(tensors) == len(expected), "canonical tensor count")

        missing = fake_checkpoint(converter, profile)
        missing["state_dict"].pop(next(iter(missing["state_dict"])))
        expect_error(
            lambda: converter.runtime_tensors(missing, profile), "count/type differs"
        )

        extra = fake_checkpoint(converter, profile)
        extra["state_dict"]["model.model.unexpected.weight"] = FakeTensor(
            (1,), "F32", "strided", FakeStorage(4)
        )
        expect_error(
            lambda: converter.runtime_tensors(extra, profile), "count/type differs"
        )

        first_name = converter.raw_source_name(
            converter.source_tensor_specs(profile["topology"])[0].name
        )
        dtype = fake_checkpoint(converter, profile)
        dtype["state_dict"][first_name].dtype = "I64"
        expect_error(
            lambda: converter.runtime_tensors(dtype, profile),
            "dtype/shape/storage differs",
        )

        shape = fake_checkpoint(converter, profile)
        shape["state_dict"][first_name].shape = (1,)
        expect_error(
            lambda: converter.runtime_tensors(shape, profile),
            "dtype/shape/storage differs",
        )

        alias = fake_checkpoint(converter, profile)
        _, reverse = converter.expected_alias_pairs(profile["topology"])[0]
        reverse_raw = converter.raw_source_name(reverse)
        reverse_tensor = alias["state_dict"][reverse_raw]
        reverse_tensor.storage = FakeStorage(reverse_tensor.storage.nbytes())
        expect_error(
            lambda: converter.runtime_tensors(alias, profile), "alias graph differs"
        )

        lm_alias = fake_checkpoint(converter, profile)
        embed = lm_alias["state_dict"]["model.model.embed_tokens.weight"]
        lm_alias["state_dict"]["model.lm_head.weight"].storage = embed.storage
        expect_error(
            lambda: converter.runtime_tensors(lm_alias, profile),
            "alias graph differs",
        )

        metric = fake_checkpoint(converter, profile)
        metric_name = next(iter(profile["source"]["metric_values"]))
        metric["state_dict"][metric_name].scalar += 1
        expect_error(
            lambda: converter.runtime_tensors(metric, profile),
            "torchmetrics scalar differs",
        )


def check_profile_pin(converter: Any, source_path: Path, root: Path) -> None:
    profile = json.loads(source_path.read_text(encoding="utf-8"))
    profile["models"][0]["source"]["checkpoint_sha256"] = "0" * 64
    changed = root / "changed-profiles.json"
    write_json(changed, profile)
    expect_error(lambda: converter.load_profiles(changed), "pinned profile row")


def check_config_and_receipt(converter: Any, root: Path) -> None:
    config_path = root / "config.json"
    config_payload = write_json(config_path, {"config": {"frozen": True}})
    checkpoint_path = root / "weights.ckpt"
    checkpoint_path.write_bytes(b"verified-checkpoint")
    profile = {
        "runtime_id": "tiny",
        "source": {
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_size": checkpoint_path.stat().st_size,
            "checkpoint_sha256": hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest(),
            "config_name": config_path.name,
            "config_size": len(config_payload),
            "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        },
        "config_required": {"config": {"frozen": True}},
    }
    receipt_path = root / "receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "kind": "geneb-janusdna-manual-source",
            "model_id": "tiny",
            "source_kind": "harvard-dataverse",
            "source_url": converter.DATAVERSE_URL,
            "files": [
                {
                    "name": checkpoint_path.name,
                    "path": str(checkpoint_path),
                    "size": checkpoint_path.stat().st_size,
                    "sha256": profile["source"]["checkpoint_sha256"],
                },
                {
                    "name": config_path.name,
                    "path": str(config_path),
                    "size": len(config_payload),
                    "sha256": profile["source"]["config_sha256"],
                },
            ],
        },
    )
    converter.validate_receipt(receipt_path, profile)
    converter.validate_config(config_path, profile)
    checkpoint_path.write_bytes(b"verified-checkpoinu")
    expect_error(
        lambda: converter.validate_receipt(receipt_path, profile),
        "payload integrity differs",
    )
    checkpoint_path.write_bytes(b"verified-checkpoint")
    config_path.write_bytes(config_payload.replace(b"true", b"null"))
    expect_error(
        lambda: converter.validate_config(config_path, profile), "bytes/contents differ"
    )


def check_unsafe_global_gate(converter: Any, root: Path) -> None:
    called = {"load": False}

    class Serialization:
        @staticmethod
        def get_unsafe_globals_in_checkpoint(path: str) -> List[str]:
            return sorted(converter.EXPECTED_UNSAFE_GLOBALS | {"evil.payload"})

        @staticmethod
        def safe_globals(values: Sequence[Any]) -> Any:
            return contextlib.nullcontext()

    class UnsafeTorch:
        serialization = Serialization()

        @staticmethod
        def load(*args: Any, **kwargs: Any) -> None:
            called["load"] = True

    class DictConfig:
        pass

    mapping = {
        "torch": UnsafeTorch,
        "omegaconf.base": SimpleNamespace(
            Metadata=type("Metadata", (), {}),
            ContainerMetadata=type("ContainerMetadata", (), {}),
        ),
        "omegaconf.dictconfig": SimpleNamespace(DictConfig=DictConfig),
        "omegaconf.listconfig": SimpleNamespace(ListConfig=type("ListConfig", (), {})),
        "omegaconf.nodes": SimpleNamespace(AnyNode=type("AnyNode", (), {})),
    }
    original = converter.importlib.import_module

    def replacement(name: str) -> Any:
        if name in mapping:
            return mapping[name]
        return original(name)

    converter.importlib.import_module = replacement
    try:
        expect_error(
            lambda: converter.safe_load_checkpoint(root / "unsafe.ckpt"),
            "unsafe-global manifest differs",
        )
    finally:
        converter.importlib.import_module = original
    check(not called["load"], "unsafe manifest reached torch.load")


class ChunkTensor:
    def __init__(self, payload: bytes, fail: bool = False) -> None:
        self.name = "tiny.weight"
        self.dtype = "F32"
        self.shape = (1,)
        self.nbytes = 4
        self.payload = payload
        self.fail = fail

    def iter_chunks(self, chunk_size: int) -> Any:
        yield memoryview(self.payload[: 3 if self.fail else 4])


class ZeroTensor:
    def __init__(self, spec: Any) -> None:
        self.name = spec.name
        self.dtype = spec.dtype
        self.shape = spec.shape
        self.nbytes = spec.nbytes

    def iter_chunks(self, chunk_size: int) -> Any:
        block = bytes(min(chunk_size, 1024 * 1024))
        remaining = self.nbytes
        while remaining:
            size = min(remaining, len(block))
            yield memoryview(block)[:size]
            remaining -= size


def check_atomic_writer(converter: Any, root: Path) -> None:
    output = root / "atomic.safetensors"
    output.write_bytes(b"sentinel")
    expect_error(
        lambda: converter.write_artifact(
            output, {"model.id": "tiny"}, [ChunkTensor(b"\0\0\0\0")], False
        ),
        "already exists",
    )
    check(output.read_bytes() == b"sentinel", "no-force overwrote output")
    expect_error(
        lambda: converter.write_artifact(
            output,
            {"model.id": "tiny"},
            [ChunkTensor(b"\0\0\0\0", True)],
            True,
        ),
        "wrong byte count",
    )
    check(output.read_bytes() == b"sentinel", "failed force write lost old output")

    # Regression: direct OS-descriptor storage writes must not overwrite the
    # still-buffered Safetensors header.
    payload = struct.pack("<f", 1.25)
    direct = converter.TorchTensorSource(
        "tiny.weight",
        "F32",
        (1,),
        4,
        FakeTensor((1,), "F32", "strided", FakeStorage(4, payload)),
    )
    converter.write_artifact(output, {"model.id": "tiny"}, [direct], True)
    with output.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        data = source.read()
    check(header_size > 0 and header_size % 8 == 0, "invalid direct-write header")
    check(header["tiny.weight"]["data_offsets"] == [0, 4], "tensor offsets")
    check(data == payload, "direct raw tensor bytes")
    check(not list(root.glob(".atomic.safetensors.*.tmp")), "stale temp file")


def check_tokenizer(converter: Any, profile: Mapping[str, Any], root: Path) -> Path:
    tokenizers = root / "tokenizers"
    tokenizers.mkdir()
    asset_path = tokenizers / "janus.json"
    pieces = [
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
    asset_payload = write_json(
        asset_path,
        {
            "format": "evo-tokenizer-v1",
            "kind": "single-nucleotide",
            "normalization": [{"op": "ascii-uppercase"}],
            "pre_tokenizer": {"kind": "none"},
            "model": {"unknown_policy": "unk", "match_special_literals": False},
            "post_processor": {
                "prefix_ids": [],
                "suffix_ids": [],
                "padding": {"side": "right", "pad_id": 4},
            },
            "special_tokens": {
                "unk": 6,
                "pad": 4,
                "bos": 2,
                "eos": 1,
                "cls": 0,
                "sep": 1,
                "mask": 3,
            },
            "vocab": [
                {"id": index, "piece": piece} for index, piece in enumerate(pieces)
            ],
        },
    )
    check(len(asset_payload) == 659, "tokenizer fixture size")
    check(
        hashlib.sha256(asset_payload).hexdigest()
        == profile["tokenizer"]["compiled_asset_sha256"],
        "tokenizer fixture SHA",
    )
    descriptor_path = root / "descriptor.json"
    write_json(
        descriptor_path,
        {
            "converter.schema": "evo-tokenizer-conversion-receipt",
            "converter.version": 1,
            "compiler_manifest_sha256": profile["tokenizer"][
                "compiler_manifest_sha256"
            ],
            "source_receipt_contract_sha256": "1" * 64,
            "tokenizer.profile": "evo-tokenizer-v1",
            "tokenizer.path": "tokenizers/janus.json",
            "tokenizer.sha256": hashlib.sha256(asset_payload).hexdigest(),
            "tokenizer.size": len(asset_payload),
        },
    )
    converter.validate_tokenizer_descriptor(descriptor_path, root, root, profile)
    changed = json.loads(asset_payload)
    changed["post_processor"]["prefix_ids"] = [0]
    write_json(asset_path, changed)
    expect_error(
        lambda: converter.validate_tokenizer_descriptor(
            descriptor_path, root, root, profile
        ),
        "size/SHA256 differs",
    )
    asset_path.write_bytes(asset_payload)
    return descriptor_path


def check_cpu_adapter_artifacts(
    converter: Any,
    profiles: Mapping[str, Mapping[str, Any]],
    catalog_path: Path,
    native_loader: Path,
    root: Path,
    descriptor_path: Path,
) -> None:
    catalog, catalog_payload = converter.load_json(catalog_path, "GENEB catalog")
    profile_payload = converter.load_json(
        Path(converter.default_config_path("geneb-janusdna-models.json")),
        "Janus profile",
    )[1]
    for runtime_id in ("geneb-janusdna-72-w", "geneb-janusdna-72-wo"):
        profile = profiles[runtime_id]
        catalog_entry, _, verified_catalog_payload = converter.load_catalog(
            catalog_path, profile
        )
        check(
            catalog_payload == verified_catalog_payload,
            "catalog payload changed during adapter fixture",
        )
        tokenizer, descriptor_sha = converter.validate_tokenizer_descriptor(
            descriptor_path, root, root, profile
        )
        geneb_metadata = converter.build_geneb_artifact_metadata(
            catalog, catalog_entry, catalog_payload
        )
        metadata = converter.build_metadata(
            profile,
            tokenizer,
            "1" * 64,
            hashlib.sha256(catalog_payload).hexdigest(),
            hashlib.sha256(profile_payload).hexdigest(),
            descriptor_sha,
            geneb_metadata,
        )
        metadata["fixture.synthetic"] = True
        output = root / (runtime_id + ".safetensors")
        converter.write_artifact(
            output,
            metadata,
            [
                ZeroTensor(spec)
                for spec in converter.canonical_tensor_specs(profile["topology"])
            ],
            False,
        )
        result = subprocess.run(
            [str(native_loader), "--verify-cpu-adapter", str(output)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                "%s CPU adapter fixture failed: %s"
                % (runtime_id, result.stderr or result.stdout)
            )
        value = json.loads(result.stdout)
        expected_variant = profile["topology"]["variant"]
        check(
            value
            == {
                "variant": expected_variant,
                "model_vocab": 16,
                "tokenizer_vocab": 12,
                "rows": 1024,
                "width": 72,
                "layer": 8,
                "pooling": "attention-mask-mean",
            },
            runtime_id + " CPU adapter result differs",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--native-loader", required=True, type=Path)
    args = parser.parse_args()
    converter = load_converter(args.converter.resolve())
    profiles, _ = converter.load_profiles(args.profiles.resolve())
    with tempfile.TemporaryDirectory(prefix="geneb-janus-converter-") as directory:
        root = Path(directory)
        check_profile_pin(converter, args.profiles.resolve(), root)
        check_manifest_corruptions(converter, profiles["geneb-janusdna-72-w"])
        check_config_and_receipt(converter, root)
        check_unsafe_global_gate(converter, root)
        check_atomic_writer(converter, root)
        tokenizer_root = root / "tokenizer-case"
        tokenizer_root.mkdir()
        descriptor_path = check_tokenizer(
            converter, profiles["geneb-janusdna-72-w"], tokenizer_root
        )
        check_cpu_adapter_artifacts(
            converter,
            profiles,
            args.catalog.resolve(),
            args.native_loader.resolve(),
            tokenizer_root,
            descriptor_path,
        )
    print("GENEB JanusDNA converter tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
