#!/usr/bin/env python3
"""Small real-PyTorch integration test for mmap checkpoint loading."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    print("SKIP: PyTorch is not installed; install requirements-convert.txt", file=sys.stderr)
    raise SystemExit(77)

from evo.checkpoint import (
    CheckpointError,
    load_checkpoint,
    prepare_runtime_image_sources,
    prepare_runtime_sources,
)
from evo.model_config import TensorSpec


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="evo-torch-test-")
        self.directory = Path(self.temporary.name)
        self.manifest = [
            TensorSpec("bf16.weight", "BF16", (2, 3)),
            TensorSpec("f32.scale", "F32", (2,)),
        ]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def save(self, state: dict[str, object], name: str = "fixture.pt") -> Path:
        path = self.directory / name
        torch.save(state, path)
        return path

    def valid_state(self) -> dict[str, object]:
        return {
            "bf16.weight": torch.tensor(
                [[1.0, -2.0, 3.5], [4.0, 0.25, -0.5]], dtype=torch.bfloat16
            ),
            "f32.scale": torch.tensor([1.25, -2.5], dtype=torch.float32),
            "block._extra_state": io.BytesIO(b"documented Transformer Engine state"),
        }

    def fp8_payload(self, **changes: object) -> io.BytesIO:
        scale = torch.tensor([2.0, 4.0, 1.0], dtype=torch.float32)
        history = torch.zeros((16, 3), dtype=torch.float32)
        history[:, 0] = 224.0
        history[:, 1] = 112.0
        payload: dict[str, object] = {
            "scale_fwd": scale,
            "scale_inv_fwd": scale.reciprocal(),
            "amax_history_fwd": history,
            "scale_bwd": torch.ones(2, dtype=torch.float32),
            "scale_inv_bwd": torch.ones(2, dtype=torch.float32),
            "amax_history_bwd": torch.zeros((16, 2), dtype=torch.float32),
            "extra_fp8_variables": {
                "fp8_checkpoint": True,
                "num_gemms": 1,
                "fp8_max_fwd": 448.0,
                "fp8_max_bwd": 57344.0,
            },
        }
        payload.update(changes)
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        return stream

    def test_mmap_load_validates_and_exposes_bit_exact_chunks(self) -> None:
        state = self.valid_state()
        expected_bf16 = bytes(state["bf16.weight"].view(torch.uint8).reshape(-1).numpy())
        expected_f32 = bytes(state["f32.scale"].view(torch.uint8).reshape(-1).numpy())
        path = self.save(state)

        sources, extra_names, fp8_sources = load_checkpoint(
            path, self.manifest, expected_extra_states=1
        )
        self.assertEqual(extra_names, ["block._extra_state"])
        self.assertEqual(fp8_sources, [])
        self.assertEqual(b"".join(bytes(chunk) for chunk in sources[0].iter_chunks(3)), expected_bf16)
        self.assertEqual(b"".join(bytes(chunk) for chunk in sources[1].iter_chunks(3)), expected_f32)
        self.assertEqual(bytes(sources[0].read_range(3, 5)), expected_bf16[3:8])
        with self.assertRaisesRegex(CheckpointError, "out of range"):
            sources[0].read_range(len(expected_bf16), 1)

    def test_contiguous_storage_slice_is_streamed_from_its_offset(self) -> None:
        state = self.valid_state()
        state["bf16.weight"] = torch.arange(8, dtype=torch.bfloat16)[1:7].reshape(2, 3)
        self.assertTrue(state["bf16.weight"].is_contiguous())
        self.assertEqual(state["bf16.weight"].storage_offset(), 1)
        expected = bytes(state["bf16.weight"].view(torch.uint8).reshape(-1).numpy())

        sources, _, _ = load_checkpoint(
            self.save(state), self.manifest, expected_extra_states=1
        )
        self.assertGreater(sources[0].tensor.storage_offset(), 0)
        self.assertEqual(
            b"".join(bytes(chunk) for chunk in sources[0].iter_chunks(3)),
            expected,
        )

    def test_noncontiguous_storage_is_rejected(self) -> None:
        state = self.valid_state()
        state["bf16.weight"] = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2).T
        self.assertFalse(state["bf16.weight"].is_contiguous())
        with self.assertRaisesRegex(CheckpointError, "must be dense contiguous"):
            load_checkpoint(
                self.save(state), self.manifest, expected_extra_states=1
            )

    def test_projection_fp8_state_is_extracted_bit_exactly(self) -> None:
        state = self.valid_state()
        state["blocks.0.projections._extra_state"] = self.fp8_payload()
        del state["block._extra_state"]
        _, extra_names, fp8_sources = load_checkpoint(
            self.save(state),
            self.manifest,
            expected_extra_states=1,
            fp8_projection_layers=(0,),
        )
        self.assertEqual(extra_names, ["blocks.0.projections._extra_state"])
        self.assertEqual(
            [source.name for source in fp8_sources],
            [
                "blocks.0.projections.fp8_scale_fwd",
                "blocks.0.projections.fp8_scale_inv_fwd",
                "blocks.0.projections.fp8_amax_history_fwd",
            ],
        )
        self.assertEqual([source.shape for source in fp8_sources], [(2,), (2,), (16, 2)])
        expected = [
            torch.tensor([2.0, 4.0], dtype=torch.float32),
            torch.tensor([0.5, 0.25], dtype=torch.float32),
            torch.tensor([[224.0, 112.0]] * 16, dtype=torch.float32),
        ]
        for source, tensor in zip(fp8_sources, expected, strict=True):
            actual = b"".join(bytes(chunk) for chunk in source.iter_chunks(7))
            self.assertEqual(actual, bytes(tensor.view(torch.uint8).reshape(-1).numpy()))

    def test_runtime_image_stores_final_e4m3_codes_and_minimal_scales(self) -> None:
        float8 = getattr(torch, "float8_e4m3fn", None)
        if float8 is None:
            self.skipTest("PyTorch does not expose float8_e4m3fn")
        for source_dtype, torch_dtype in (
            ("BF16", torch.bfloat16),
            ("F32", torch.float32),
        ):
            with self.subTest(source_dtype=source_dtype):
                weight = torch.tensor(
                    [[-300.0, -2.0, 0.0], [0.2657, 100.0, 200.0]],
                    dtype=torch_dtype,
                )
                original = weight.clone()
                state = {
                    "blocks.0.projections.weight": weight,
                    "blocks.0.projections._extra_state": self.fp8_payload(),
                }
                manifest = [
                    TensorSpec(
                        "blocks.0.projections.weight", source_dtype, (2, 3)
                    )
                ]
                sources, _, fp8_sources = load_checkpoint(
                    self.save(state, f"fp8-runtime-{source_dtype}.pt"),
                    manifest,
                    expected_extra_states=1,
                    fp8_projection_layers=(0,),
                )
                prepared = prepare_runtime_image_sources(
                    sources, fp8_sources, (0,)
                )
                self.assertEqual(
                    [source.name for source in prepared],
                    [
                        "blocks.0.projections.fp8_runtime_scales",
                        "blocks.0.projections.weight",
                    ],
                )
                self.assertEqual(
                    [source.dtype for source in prepared], ["F32", "E4M3_SW"]
                )
                actual_scales = b"".join(
                    bytes(chunk) for chunk in prepared[0].iter_chunks(3)
                )
                expected_scales = torch.tensor(
                    [2.0, 0.125], dtype=torch.float32
                )
                self.assertEqual(
                    actual_scales,
                    bytes(
                        expected_scales.view(torch.uint8).reshape(-1).numpy()
                    ),
                )
                actual_codes = b"".join(
                    bytes(chunk) for chunk in prepared[1].iter_chunks(2)
                )
                runtime_weight = (
                    weight.to(torch.bfloat16)
                    if source_dtype == "F32"
                    else weight
                )
                expected_codes = (
                    runtime_weight.float()
                    .mul(4.0)
                    .clamp(-448.0, 448.0)
                    .to(float8)
                    .view(torch.uint8)
                    .reshape(-1)
                )
                self.assertEqual(actual_codes, bytes(expected_codes.numpy()))
                self.assertTrue(torch.equal(sources[0].tensor, original))

    def test_projection_fp8_state_rejects_invalid_metadata(self) -> None:
        invalid = {
            "inverse": {
                "scale_inv_fwd": torch.tensor([0.25, 0.25, 1.0], dtype=torch.float32)
            },
            "history": {
                "amax_history_fwd": torch.ones((15, 3), dtype=torch.float32)
            },
            "maximum": {
                "extra_fp8_variables": {
                    "fp8_checkpoint": True,
                    "num_gemms": 1,
                    "fp8_max_fwd": 240.0,
                    "fp8_max_bwd": 57344.0,
                }
            },
        }
        for name, changes in invalid.items():
            with self.subTest(name=name):
                state = self.valid_state()
                state["blocks.0.projections._extra_state"] = self.fp8_payload(**changes)
                del state["block._extra_state"]
                with self.assertRaisesRegex(CheckpointError, "blocks.0.projections"):
                    load_checkpoint(
                        self.save(state, f"{name}.pt"),
                        self.manifest,
                        expected_extra_states=1,
                        fp8_projection_layers=(0,),
                    )

    def test_projection_fp8_state_must_exist_for_every_hyena_layer(self) -> None:
        with self.assertRaisesRegex(
            CheckpointError, "missing FP8 projection extra state"
        ):
            load_checkpoint(
                self.save(self.valid_state()),
                self.manifest,
                expected_extra_states=1,
                fp8_projection_layers=(0,),
            )

    def test_missing_unknown_and_wrong_dtype_fail_closed(self) -> None:
        state = self.valid_state()
        del state["f32.scale"]
        with self.assertRaisesRegex(CheckpointError, "missing tensors: f32.scale"):
            load_checkpoint(self.save(state, "missing.pt"), self.manifest, expected_extra_states=1)

        state = self.valid_state()
        state["unknown.weight"] = torch.ones(1, dtype=torch.bfloat16)
        with self.assertRaisesRegex(CheckpointError, "unknown tensors: unknown.weight"):
            load_checkpoint(self.save(state, "unknown.pt"), self.manifest, expected_extra_states=1)

        state = self.valid_state()
        state["bf16.weight"] = state["bf16.weight"].float()
        with self.assertRaisesRegex(CheckpointError, "expected BF16"):
            load_checkpoint(self.save(state, "dtype.pt"), self.manifest, expected_extra_states=1)

    def test_only_documented_extra_state_is_skipped(self) -> None:
        state = self.valid_state()
        state["training.step"] = 10
        with self.assertRaisesRegex(CheckpointError, "unknown non-tensor checkpoint entries"):
            load_checkpoint(self.save(state), self.manifest, expected_extra_states=2)

    def test_documented_time_grid_is_validated_then_omitted(self) -> None:
        state = self.valid_state()
        state["blocks.2.mixer.mixer.filter.t"] = torch.arange(
            4, dtype=torch.float32
        ).reshape(1, 1, 4)
        ignored = [
            TensorSpec("blocks.2.mixer.mixer.filter.t", "F32", (1, 1, 4))
        ]
        sources, _, _ = load_checkpoint(
            self.save(state),
            self.manifest,
            expected_extra_states=1,
            ignored_manifest=ignored,
        )
        self.assertEqual([source.name for source in sources], ["bf16.weight", "f32.scale"])
        state["blocks.2.mixer.mixer.filter.t"] = state[
            "blocks.2.mixer.mixer.filter.t"
        ][:, :, :3]
        with self.assertRaisesRegex(CheckpointError, "shape="):
            load_checkpoint(
                self.save(state, "bad-time-grid.pt"),
                self.manifest,
                expected_extra_states=1,
                ignored_manifest=ignored,
            )

    def test_runtime_preparation_only_exactly_widens_bf16(self) -> None:
        sources, _, _ = load_checkpoint(
            self.save(self.valid_state()), self.manifest, expected_extra_states=1
        )
        runtime = [
            TensorSpec("bf16.weight", "F32", (2, 3)),
            TensorSpec("f32.scale", "F32", (2,)),
        ]
        prepared = prepare_runtime_sources(sources, runtime)
        actual = b"".join(bytes(chunk) for chunk in prepared[0].iter_chunks(5))
        expected = bytes(
            self.valid_state()["bf16.weight"].float().view(torch.uint8).reshape(-1).numpy()
        )
        self.assertEqual(actual, expected)
        unsafe = [
            TensorSpec("bf16.weight", "BF16", (2, 3)),
            TensorSpec("f32.scale", "BF16", (2,)),
        ]
        with self.assertRaisesRegex(CheckpointError, "unsafe runtime conversion"):
            prepare_runtime_sources(sources, unsafe)

    def test_part_file_is_rejected_with_merge_instruction(self) -> None:
        with self.assertRaisesRegex(CheckpointError, "merge all .partN files"):
            load_checkpoint(self.directory / "evo2_40b.pt.part0", self.manifest)


if __name__ == "__main__":
    unittest.main()
