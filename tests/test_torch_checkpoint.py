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

from evo2c.checkpoint import CheckpointError, load_checkpoint
from evo2c.model_config import TensorSpec


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="evo2c-torch-test-")
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

    def test_mmap_load_validates_and_exposes_bit_exact_chunks(self) -> None:
        state = self.valid_state()
        expected_bf16 = bytes(state["bf16.weight"].view(torch.uint8).reshape(-1).numpy())
        expected_f32 = bytes(state["f32.scale"].view(torch.uint8).reshape(-1).numpy())
        path = self.save(state)

        sources, extra_names = load_checkpoint(path, self.manifest, expected_extra_states=1)
        self.assertEqual(extra_names, ["block._extra_state"])
        self.assertEqual(b"".join(bytes(chunk) for chunk in sources[0].iter_chunks(3)), expected_bf16)
        self.assertEqual(b"".join(bytes(chunk) for chunk in sources[1].iter_chunks(3)), expected_f32)

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

    def test_part_file_is_rejected_with_merge_instruction(self) -> None:
        with self.assertRaisesRegex(CheckpointError, "merge all .partN files"):
            load_checkpoint(self.directory / "evo2_40b.pt.part0", self.manifest)


if __name__ == "__main__":
    unittest.main()
