#!/usr/bin/env python3
"""Contracts for official Vortex reference-vector generation."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

try:
    import torch

    from generate_vortex_vectors import (
        OFFICIAL_FORCE_PROMPT_THRESHOLD,
        apply_official_quality_mode,
        official_midpoint_split,
        software_fp8_output_scale,
    )
except ModuleNotFoundError:
    print(
        "SKIP: reference-vector Python dependencies are not installed",
        file=sys.stderr,
    )
    raise SystemExit(77)


class VortexReferenceTests(unittest.TestCase):
    def test_v13_official_midpoint_split(self) -> None:
        prompt, target = official_midpoint_split("ABCDEFGHIJK")
        self.assertEqual(prompt, "ABCD")
        self.assertEqual(target, "EFGHIJK")

    def test_v13_official_quality_enables_cached_teacher_forcing(self) -> None:
        args = argparse.Namespace(
            official_quality=True,
            prompts_csv=Path("prompts.csv"),
            generate_tokens=500,
            midpoint_prompt=False,
            cached_generate=False,
            force_prompt_threshold=17,
        )
        apply_official_quality_mode(args)
        self.assertTrue(args.midpoint_prompt)
        self.assertTrue(args.cached_generate)
        self.assertEqual(
            args.force_prompt_threshold,
            OFFICIAL_FORCE_PROMPT_THRESHOLD,
        )

    def test_v13_official_quality_rejects_non_generation_inputs(self) -> None:
        for prompts, tokens in ((None, 500), (Path("prompts.csv"), 0)):
            with self.subTest(prompts=prompts, tokens=tokens):
                args = argparse.Namespace(
                    official_quality=True,
                    prompts_csv=prompts,
                    generate_tokens=tokens,
                    midpoint_prompt=False,
                    cached_generate=False,
                    force_prompt_threshold=17,
                )
                with self.assertRaisesRegex(ValueError, "--official-quality"):
                    apply_official_quality_mode(args)

    def test_t14_uses_stored_fp32_inverse_scales(self) -> None:
        scale = torch.tensor(
            [181.48834228515625, 255.98260498046875, 1.0],
            dtype=torch.float32,
        )
        scale_inv = scale.reciprocal()
        expected = float((scale_inv[0] * scale_inv[1]).item())
        self.assertEqual(software_fp8_output_scale(scale_inv), expected)
        self.assertNotEqual(expected, 1.0 / float((scale[0] * scale[1]).item()))

    def test_t14_rejects_invalid_inverse_scales(self) -> None:
        for value in (
            torch.ones(2, dtype=torch.float32),
            torch.ones(3, dtype=torch.float64),
            torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
            torch.tensor([1.0, float("nan"), 1.0], dtype=torch.float32),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "inverse scales"):
                    software_fp8_output_scale(value)


if __name__ == "__main__":
    unittest.main()
