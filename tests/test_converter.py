#!/usr/bin/env python3
"""Converter manifest, streaming writer, and native-reader roundtrip tests."""

from __future__ import annotations

import argparse
import dataclasses
import json
import struct
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evo.format import (
    DEFAULT_MAX_SHARD_SIZE,
    EVO2_PROFILE_VALUE,
    BytesTensorSource,
    FormatError,
    TensorSource,
    plan_shards,
    write_model,
)
from evo.model_config import (
    EXPECTED_BF16_TENSOR_COUNT,
    EXPECTED_F32_TENSOR_COUNT,
    EXPECTED_TENSOR_BYTES,
    EXPECTED_TENSOR_COUNT,
    ConfigError,
    bionemo_checkpoint_manifest,
    checkpoint_manifest,
    container_manifest,
    config_metadata,
    ignored_checkpoint_manifest,
    load_config,
    load_model_registry,
    runtime_manifest,
)


INSPECTOR: Path
CONFIG: Path
CONFIG_DIR: Path
WORK_DIR: Path


@dataclasses.dataclass(frozen=True, slots=True)
class BrokenTensorSource:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    chunks: tuple[bytes, ...]

    def iter_chunks(self, chunk_size: int):  # type: ignore[no-untyped-def]
        del chunk_size
        for chunk in self.chunks:
            yield memoryview(chunk)


class ConverterTests(unittest.TestCase):
    def test_every_registered_model_has_an_exact_manifest(self) -> None:
        registry = load_model_registry()
        models = {
            model_id: entry
            for model_id, entry in registry["models"].items()
            if "config" in entry
        }
        self.assertEqual(len(models), 8)
        for model_id, entry in models.items():
            with self.subTest(model_id=model_id):
                config = load_config(CONFIG_DIR / entry["config"])
                self.assertEqual(config.model_id, model_id)
                if model_id == "evo2_40b_bionemo_bf16":
                    source = bionemo_checkpoint_manifest(config)
                else:
                    source = checkpoint_manifest(config)
                runtime = runtime_manifest(config)
                container = container_manifest(config)
                self.assertEqual(len(source), entry["source_manifest"]["tensors"])
                self.assertEqual(sum(item.nbytes for item in source), entry["source_manifest"]["bytes"])
                self.assertEqual(len(runtime), entry["runtime_manifest"]["tensors"])
                self.assertEqual(sum(item.nbytes for item in runtime), entry["runtime_manifest"]["bytes"])
                self.assertEqual(len(container), entry["container_manifest"]["tensors"])
                self.assertEqual(
                    sum(item.nbytes for item in container),
                    entry["container_manifest"]["bytes"],
                )
                ignored = ignored_checkpoint_manifest(config)
                expected_ignored = len(config.hcl_layer_idxs) if entry["ignored_time_grid_length"] else 0
                self.assertEqual(len(ignored), expected_ignored)
                for item in ignored:
                    self.assertEqual(item.shape, (1, 1, entry["ignored_time_grid_length"]))

    def test_registered_7b_default_shard_plan_is_size_only(self) -> None:
        config = load_config(CONFIG_DIR / "evo2-7b-1m.yml")
        shards = plan_shards(container_manifest(config))
        self.assertEqual([len(shard) for shard in shards], [116, 109, 109, 11])
        self.assertEqual(
            [sum(tensor.nbytes for tensor in shard) for shard in shards],
            [4_255_968_256, 4_205_595_392, 4_205_595_392, 503_374_080],
        )
        self.assertTrue(
            all(
                sum(tensor.nbytes for tensor in shard) <= DEFAULT_MAX_SHARD_SIZE
                for shard in shards
            )
        )
        self.assertEqual(shards[0][0].name, "embedding_layer.weight")
        self.assertEqual(shards[-1][-1].name, "blocks.31.post_norm.scale")

    def test_registry_metadata_identifies_the_runtime_abi_and_model(self) -> None:
        one_b = load_config(CONFIG_DIR / "evo2-1b-8k.yml")
        metadata = config_metadata(one_b, "checkpoint.pt", 123)
        self.assertEqual(metadata["model.id"], "evo2_1b_base")
        self.assertEqual(metadata["checkpoint.source_norm_dtype"], "BF16")
        self.assertTrue(metadata["conversion.exact_widen_norm_to_f32"])

        forty_b = config_metadata(load_config(CONFIG), "evo2_40b.pt", 82_253_491_694)
        self.assertEqual(forty_b["model.id"], "evo2_40b")
        self.assertEqual(forty_b["runtime.abi"], "evo2-safetensors-v1")
        self.assertEqual(forty_b["hyena_projection_dtype"], "E4M3_SW")

    def test_every_registry_profile_rejects_dimension_corruption(self) -> None:
        registry = load_model_registry()
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as directory:
            temporary = Path(directory)
            for model_id, entry in registry["models"].items():
                if "config" not in entry:
                    continue
                with self.subTest(model_id=model_id):
                    source = CONFIG_DIR / entry["config"]
                    config = load_config(source)
                    changed = source.read_text(encoding="utf-8").replace(
                        f"hidden_size: {config.hidden_size}",
                        f"hidden_size: {config.hidden_size + 128}",
                        1,
                    )
                    path = temporary / f"{model_id}.yml"
                    path.write_text(changed, encoding="utf-8")
                    with self.assertRaisesRegex(ConfigError, "unsupported hidden_size"):
                        load_config(path)

    def test_official_40b_manifest_is_exact(self) -> None:
        config = load_config(CONFIG)
        manifest = checkpoint_manifest(config)
        counts = Counter(spec.dtype for spec in manifest)

        self.assertEqual(config.rotary_emb_base, 1e6)
        self.assertEqual(len(manifest), EXPECTED_TENSOR_COUNT)
        self.assertEqual(counts, {"BF16": EXPECTED_BF16_TENSOR_COUNT, "F32": EXPECTED_F32_TENSOR_COUNT})
        self.assertEqual(sum(spec.nbytes for spec in manifest), EXPECTED_TENSOR_BYTES)
        self.assertEqual(manifest[0].name, "embedding_layer.weight")
        self.assertEqual(manifest[-1].name, "norm.scale")

        by_name = {spec.name: spec for spec in manifest}
        self.assertEqual(by_name["blocks.3.inner_mha_cls.Wqkv.weight"].shape, (24576, 8192))
        self.assertEqual(by_name["blocks.2.filter.log_poles"].shape, (8192, 16, 1))
        self.assertEqual(by_name["blocks.1.filter.h"].shape, (512, 1, 128))
        self.assertNotIn("blocks.3.filter.h", by_name)

    def test_config_topology_drift_is_rejected(self) -> None:
        changed = CONFIG.read_text(encoding="utf-8").replace("hidden_size: 8192", "hidden_size: 4096")
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as directory:
            path = Path(directory) / "bad.yml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "unsupported hidden_size"):
                load_config(path)

        changed = CONFIG.read_text(encoding="utf-8").replace(
            "hyena_flip_x1x2: false", "hyena_flip_x1x2: true"
        )
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as directory:
            path = Path(directory) / "bad-semantics.yml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "hyena_flip_x1x2 must be false"):
                load_config(path)

        changed = CONFIG.read_text(encoding="utf-8").replace(
            "rotary_emb_base: 1000000", "rotary_emb_base: 100000000000"
        )
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as directory:
            path = Path(directory) / "bad-rope.yml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "unsupported rotary_emb_base"):
                load_config(path)

        changed = CONFIG.read_text(encoding="utf-8").replace(
            "model_id: evo2_40b", "model_id: made_up"
        )
        with tempfile.TemporaryDirectory(dir=WORK_DIR) as directory:
            path = Path(directory) / "unknown-id.yml"
            path.write_text(changed, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "unsupported model_id"):
                load_config(path)

    def test_streaming_writer_roundtrips_through_native_inspector(self) -> None:
        bf16 = bytes.fromhex("803f004040408040")
        f32 = struct.pack("<2f", 1.25, -2.5)
        tensors: list[TensorSource] = [
            BytesTensorSource("embedding_layer.weight", "BF16", (2, 2), bf16),
            BytesTensorSource("norm.scale", "F32", (2,), f32),
        ]
        metadata = {
            "model.name": "tiny-evo2",
            "config.hidden_size": 2,
            "config.eps": 1e-6,
            "config.tie_embeddings": True,
            "config.layers": [0, 3],
            "fixture.opaque": b"\x00\xff",
        }
        path = WORK_DIR / "roundtrip.safetensors"
        write_model(
            path,
            metadata,
            tensors,
            artifact_profile=EVO2_PROFILE_VALUE,
            chunk_size=3,
            force=True,
        )

        inspected = subprocess.run(
            [str(INSPECTOR), str(path)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        self.assertIn(
            "format=SAFETENSORS profile=evo2-runtime-v1", inspected
        )
        self.assertIn("validation=ok", inspected)
        self.assertIn("metadata model.name type=string value=tiny-evo2", inspected)
        self.assertIn("metadata config.tie_embeddings type=bool value=true", inspected)
        self.assertIn("tensor_count=2", inspected)
        self.assertIn("tensor embedding_layer.weight dtype=BF16 shape=[2,2]", inspected)
        self.assertIn("tensor norm.scale dtype=F32 shape=[2]", inspected)

        raw = path.read_bytes()
        header_size = struct.unpack_from("<Q", raw)[0]
        header = json.loads(raw[8 : 8 + header_size])
        data_offset = 8 + header_size
        first_begin, first_end = header["embedding_layer.weight"]["data_offsets"]
        second_begin, second_end = header["norm.scale"]["data_offsets"]
        self.assertEqual(header["embedding_layer.weight"]["dtype"], "BF16")
        self.assertEqual(header["norm.scale"]["dtype"], "F32")
        self.assertEqual(
            header["__metadata__"]["evo2.profile"], "s:evo2-runtime-v1"
        )
        self.assertEqual(raw[data_offset + first_begin : data_offset + first_end], bf16)
        self.assertEqual(raw[data_offset + second_begin : data_offset + second_end], f32)
        self.assertEqual(first_end, second_begin)

    def test_writer_refuses_overwrite_and_duplicate_names(self) -> None:
        source = BytesTensorSource("x", "BF16", (1,), b"\x00\x00")
        path = WORK_DIR / "overwrite.safetensors"
        with self.assertRaisesRegex(TypeError, "artifact_profile"):
            write_model(path, {"model.name": "implicit"}, [source])
        write_model(
            path,
            {"model.name": "first"},
            [source],
            artifact_profile=EVO2_PROFILE_VALUE,
            force=True,
        )
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_model(
                path,
                {"model.name": "second"},
                [source],
                artifact_profile=EVO2_PROFILE_VALUE,
            )
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(FormatError, "duplicate or reserved tensor name"):
            write_model(
                WORK_DIR / "duplicate.safetensors",
                {},
                [source, source],
                artifact_profile=EVO2_PROFILE_VALUE,
            )

    def test_writer_size_shards_use_the_standard_index_contract(self) -> None:
        tensors: list[TensorSource] = [
            BytesTensorSource("embedding_layer.weight", "BF16", (2,), b"\x00" * 4),
            BytesTensorSource("blocks.0.weight", "BF16", (3,), b"\x01" * 6),
            BytesTensorSource("blocks.1.weight", "F32", (2,), b"\x02" * 8),
        ]
        output = WORK_DIR / "sharded-model.safetensors"
        load_path = write_model(
            output,
            {"model.name": "sharded"},
            tensors,
            artifact_profile=EVO2_PROFILE_VALUE,
            max_shard_size=10,
            force=True,
        )

        self.assertEqual(load_path, output.with_name("sharded-model.safetensors.index.json"))
        index = json.loads(load_path.read_text(encoding="ascii"))
        self.assertEqual(index["metadata"], {"total_size": 18})
        self.assertEqual(
            index["weight_map"],
            {
                "embedding_layer.weight": "sharded-model-00001-of-00002.safetensors",
                "blocks.0.weight": "sharded-model-00001-of-00002.safetensors",
                "blocks.1.weight": "sharded-model-00002-of-00002.safetensors",
            },
        )
        self.assertFalse(output.exists())
        for shard in set(index["weight_map"].values()):
            self.assertTrue((output.parent / shard).is_file())

        inspected = subprocess.run(
            [str(INSPECTOR), str(load_path)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        self.assertIn("shards=2", inspected)
        self.assertIn("tensor_count=3", inspected)
        self.assertIn("tensor blocks.1.weight dtype=F32 shape=[2]", inspected)

        bad_index = json.loads(load_path.read_text(encoding="ascii"))
        bad_index["metadata"]["total_size"] += 1
        bad_total_path = output.with_name("bad-total.safetensors.index.json")
        bad_total_path.write_text(json.dumps(bad_index), encoding="ascii")
        rejected = subprocess.run(
            [str(INSPECTOR), str(bad_total_path)],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("total_size does not match", rejected.stderr)

        bad_index = json.loads(load_path.read_text(encoding="ascii"))
        first_name = next(iter(bad_index["weight_map"]))
        bad_index["weight_map"][first_name] = "../outside.safetensors"
        bad_path_path = output.with_name("bad-path.safetensors.index.json")
        bad_path_path.write_text(json.dumps(bad_index), encoding="ascii")
        rejected = subprocess.run(
            [str(INSPECTOR), str(bad_path_path)],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("invalid shard filename", rejected.stderr)

    def test_writer_rejects_short_and_long_streams(self) -> None:
        short = BrokenTensorSource("short", "BF16", (2,), 4, (b"\x00\x00",))
        long = BrokenTensorSource("long", "BF16", (2,), 4, (b"\x00" * 5,))
        with self.assertRaisesRegex(FormatError, "yielded 2 bytes"):
            write_model(
                WORK_DIR / "short.safetensors",
                {},
                [short],
                artifact_profile=EVO2_PROFILE_VALUE,
            )
        with self.assertRaisesRegex(FormatError, "yielded more bytes"):
            write_model(
                WORK_DIR / "long.safetensors",
                {},
                [long],
                artifact_profile=EVO2_PROFILE_VALUE,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspector", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments, unittest_args = parse_args(), [__file__]
    INSPECTOR = arguments.inspector.resolve()
    CONFIG = arguments.config.resolve()
    CONFIG_DIR = arguments.config_dir.resolve()
    WORK_DIR = arguments.work_dir.resolve()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    unittest.main(argv=unittest_args)
