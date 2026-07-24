#!/usr/bin/env python3
"""Small real-PyTorch BioNeMo DCP reader and transform integration tests."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.distributed.checkpoint as dcp
except ModuleNotFoundError:
    print("SKIP: PyTorch is not installed; install requirements-convert.txt", file=sys.stderr)
    raise SystemExit(77)

from evo2c.bionemo_checkpoint import (
    DcpReader,
    MappingGroup,
    _transform_tensors,
)
from evo2c.model_config import TensorSpec, load_config


CONFIG: Path


class BioNeMoDcpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="evo2c-bionemo-dcp-")
        self.directory = Path(self.temporary.name)
        self.config = load_config(CONFIG)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dcp_reader_normalizes_module_prefix_and_loads_bit_exact_bf16(self) -> None:
        expected = torch.tensor(
            [[1.0, -2.0, 3.5], [4.0, 0.25, -0.5]], dtype=torch.bfloat16
        )
        checkpoint = self.directory / "weights"
        writer = dcp.FileSystemWriter(
            str(checkpoint), single_file_per_rank=False, thread_count=1
        )
        dcp.save(
            state_dict={"module.test.weight": expected},
            storage_writer=writer,
            no_dist=True,
        )
        reader = DcpReader(checkpoint)
        self.assertEqual(set(reader.tensor_metadata), {"test.weight"})
        actual = reader.load("test.weight")
        self.assertEqual(
            bytes(actual.view(torch.uint8).reshape(-1).numpy()),
            bytes(expected.view(torch.uint8).reshape(-1).numpy()),
        )

    def transform(
        self,
        outputs: tuple[TensorSpec, ...],
        transform: str,
        inputs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        group = MappingGroup(outputs, (tuple(f"source.{i}" for i in range(len(inputs))),), transform)
        return _transform_tensors(torch, group, inputs, self.config)

    def test_identity_unsqueeze_and_fc1_split_preserve_bf16_bits(self) -> None:
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)
        identity = self.transform(
            (TensorSpec("identity", "BF16", (3, 4)),), "identity", [tensor]
        )["identity"]
        self.assertEqual(
            bytes(identity.view(torch.uint8).reshape(-1).numpy()),
            bytes(tensor.view(torch.uint8).reshape(-1).numpy()),
        )

        unsqueezed = self.transform(
            (TensorSpec("conv", "BF16", (3, 1, 4)),),
            "unsqueeze_middle",
            [tensor],
        )["conv"]
        self.assertTrue(torch.equal(unsqueezed[:, 0, :], tensor))

        merged = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
        split = self.transform(
            (
                TensorSpec("l1", "BF16", (3, 4)),
                TensorSpec("l2", "BF16", (3, 4)),
            ),
            "split_fc1",
            [merged],
        )
        self.assertTrue(torch.equal(split["l1"], merged[:3]))
        self.assertTrue(torch.equal(split["l2"], merged[3:]))

    def test_medium_and_long_filter_math_matches_bionemo_export(self) -> None:
        h = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        decay = torch.tensor([[0.5, 0.25, 0.125], [2.0, 3.0, 4.0]])
        medium = self.transform(
            (TensorSpec("h", "F32", (2, 1, 3)),),
            "hcm_filter",
            [h, decay],
        )["h"]
        self.assertTrue(torch.equal(medium[:, 0, :], h * decay))

        poles = torch.tensor([[-1.0, 0.0], [0.5, -0.5]])
        gamma = torch.tensor([[0.0, 0.25], [-0.25, 0.5]])
        residues = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        long_filter = self.transform(
            (
                TensorSpec("log_poles", "F32", (2, 2, 1)),
                TensorSpec("residues", "F32", (2, 2)),
            ),
            "hcl_filter",
            [poles, gamma, residues],
        )
        expected = -torch.exp(poles) * torch.exp(gamma)
        self.assertTrue(torch.equal(long_filter["log_poles"][:, :, 0], expected))
        self.assertTrue(torch.equal(long_filter["residues"], residues))

    def test_rope_inverse_frequency_uses_40b_interpolated_base(self) -> None:
        inv_freq = self.transform(
            (TensorSpec("inv", "F32", (64,)),),
            "inv_freq",
            [],
        )["inv"]
        positions = torch.arange(0, 128, 2, dtype=torch.float32)
        expected = 1.0 / (1_000_000.0 ** (positions / 128.0))
        self.assertTrue(torch.equal(inv_freq, expected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments, unittest_args = parse_args(), [__file__]
    CONFIG = arguments.config.resolve()
    unittest.main(argv=unittest_args)
