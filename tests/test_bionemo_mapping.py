#!/usr/bin/env python3
"""Dependency-free BioNeMo 40B manifest and mapping contract."""

from __future__ import annotations

import argparse
import unittest
from collections import Counter
from pathlib import Path

from evo.bionemo_checkpoint import (
    DcpTensorMetadata,
    MappingGroup,
    build_mapping_groups,
    resolve_mapping,
)
from evo.checkpoint import CheckpointError
from evo.model_config import (
    EXPECTED_BIONEMO_BF16_TENSOR_COUNT,
    EXPECTED_BIONEMO_F32_TENSOR_COUNT,
    EXPECTED_BIONEMO_TENSOR_BYTES,
    EXPECTED_TENSOR_COUNT,
    bionemo_checkpoint_manifest,
    checkpoint_manifest,
    load_config,
)


BF16_CONFIG: Path
ARC_CONFIG: Path


def source_metadata(groups: list[MappingGroup]) -> dict[str, DcpTensorMetadata]:
    result: dict[str, DcpTensorMetadata] = {}
    for group in groups:
        sources = group.source_options[0]
        if not sources:
            continue
        output_shapes = [spec.shape for spec in group.outputs]
        if group.transform in {"identity", "cast_f32", "tied"}:
            shapes = [output_shapes[0]]
        elif group.transform == "unsqueeze_middle":
            output = output_shapes[0]
            shapes = [(output[0], output[2])]
        elif group.transform == "split_fc1":
            first, second = output_shapes
            shapes = [(first[0] + second[0], first[1])]
        elif group.transform == "hcm_filter":
            output = output_shapes[0]
            shapes = [(output[0], output[2]), (output[0], output[2])]
        elif group.transform == "hcl_filter":
            groups_count, state_size = output_shapes[1]
            shapes = [(groups_count, state_size)] * 3
        else:
            raise AssertionError(group.transform)

        dtypes = ["BF16"] * len(sources)
        for name, shape, dtype in zip(sources, shapes, dtypes, strict=True):
            result[name] = DcpTensorMetadata(
                physical_name=f"module.{name}",
                dtype=dtype,
                shape=shape,
            )
    return result


class BioNeMoMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(BF16_CONFIG)
        self.groups = build_mapping_groups(self.config)
        self.metadata = source_metadata(self.groups)

    def test_bf16_config_and_manifest_are_exact(self) -> None:
        self.assertFalse(self.config.use_fp8_input_projections)
        manifest = bionemo_checkpoint_manifest(self.config)
        counts = Counter(spec.dtype for spec in manifest)
        self.assertEqual(len(manifest), EXPECTED_TENSOR_COUNT)
        self.assertEqual(
            counts,
            {
                "BF16": EXPECTED_BIONEMO_BF16_TENSOR_COUNT,
                "F32": EXPECTED_BIONEMO_F32_TENSOR_COUNT,
            },
        )
        self.assertEqual(sum(spec.nbytes for spec in manifest), EXPECTED_BIONEMO_TENSOR_BYTES)
        by_name = {spec.name: spec for spec in manifest}
        self.assertEqual(by_name["blocks.1.filter.h"].dtype, "F32")
        self.assertEqual(by_name["blocks.0.filter.h"].dtype, "BF16")
        self.assertEqual(by_name["blocks.0.projections.weight"].dtype, "BF16")

        arc = load_config(ARC_CONFIG)
        self.assertTrue(arc.use_fp8_input_projections)
        self.assertEqual(
            [spec.name for spec in checkpoint_manifest(arc)],
            [spec.name for spec in manifest],
        )

    def test_mapping_covers_manifest_and_all_506_dcp_tensors(self) -> None:
        bound = resolve_mapping(self.groups, self.metadata)
        outputs = [
            spec.name for item in bound for spec in item.mapping.outputs
        ]
        expected = [
            spec.name for spec in bionemo_checkpoint_manifest(self.config)
        ]
        self.assertEqual(outputs, expected)
        self.assertEqual(len(self.metadata), 506)
        self.assertEqual(
            Counter(item.dtype for item in self.metadata.values()),
            {"BF16": 506},
        )
        self.assertEqual(
            next(
                item
                for item in bound
                if item.mapping.outputs[0].name == "blocks.3.inner_mha_cls.Wqkv.weight"
            ).sources,
            ("decoder.layers.3.self_attention.linear_qkv.weight",),
        )
        hcl = next(
            item
            for item in bound
            if item.mapping.outputs[0].name == "blocks.2.filter.log_poles"
        )
        self.assertEqual(hcl.mapping.transform, "hcl_filter")
        self.assertEqual(len(hcl.sources), 3)

    def test_nemo2_and_mbridge_names_bind_to_same_logical_mapping(self) -> None:
        bound = resolve_mapping(self.groups, self.metadata)
        self.assertTrue(
            all(
                item.physical_name.startswith("module.")
                for item in self.metadata.values()
            )
        )
        mbridge = {
            name: DcpTensorMetadata(name, item.dtype, item.shape)
            for name, item in self.metadata.items()
        }
        self.assertEqual(
            [item.sources for item in bound],
            [item.sources for item in resolve_mapping(self.groups, mbridge)],
        )

    def test_missing_unknown_ambiguous_shape_and_dtype_fail_closed(self) -> None:
        missing = dict(self.metadata)
        del missing["embedding.word_embeddings.weight"]
        with self.assertRaisesRegex(CheckpointError, "missing BioNeMo tensors"):
            resolve_mapping(self.groups, missing)

        extra = dict(self.metadata)
        extra["optimizer.step"] = DcpTensorMetadata(
            "module.optimizer.step", "F32", (1,)
        )
        with self.assertRaisesRegex(CheckpointError, "unknown BioNeMo data tensors"):
            resolve_mapping(self.groups, extra)

        ambiguous = dict(self.metadata)
        ambiguous["decoder.layers.0.norm.weight"] = DcpTensorMetadata(
            "module.decoder.layers.0.norm.weight", "F32", (8192,)
        )
        with self.assertRaisesRegex(CheckpointError, "ambiguous BioNeMo source"):
            resolve_mapping(self.groups, ambiguous)

        wrong_shape = dict(self.metadata)
        name = "decoder.layers.0.mixer.dense_projection.weight"
        wrong_shape[name] = DcpTensorMetadata(
            f"module.{name}", "BF16", (24575, 8192)
        )
        with self.assertRaisesRegex(CheckpointError, "shape="):
            resolve_mapping(self.groups, wrong_shape)

        wrong_dtype = dict(self.metadata)
        item = wrong_dtype[name]
        wrong_dtype[name] = DcpTensorMetadata(
            item.physical_name, "F32", item.shape
        )
        with self.assertRaisesRegex(CheckpointError, "bit-exact"):
            resolve_mapping(self.groups, wrong_dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-config", required=True, type=Path)
    parser.add_argument("--arc-config", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments, unittest_args = parse_args(), [__file__]
    BF16_CONFIG = arguments.bf16_config.resolve()
    ARC_CONFIG = arguments.arc_config.resolve()
    unittest.main(argv=unittest_args)
