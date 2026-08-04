#!/usr/bin/env python3
"""Contracts for official Vortex reference-vector generation."""

from __future__ import annotations

import argparse
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch

    from generate_vortex_vectors import (
        OFFICIAL_FORCE_PROMPT_THRESHOLD,
        apply_official_quality_mode,
        fixup_fp8_extra_states_on_module_device,
        official_midpoint_split,
        software_e4m3_quantize,
        software_e4m3_quantize_weight,
        software_fp8_output_scale,
        software_fp8_projection_bias,
        software_fp8_projection_layers,
    )
except ModuleNotFoundError:
    print(
        "SKIP: reference-vector Python dependencies are not installed",
        file=sys.stderr,
    )
    raise SystemExit(77)


class VortexReferenceTests(unittest.TestCase):
    def test_multigpu_fp8_fixup_recurses_over_unregistered_metadata(self) -> None:
        class FakeFp8Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.fp8_meta_tensors_initialized = True
                self.reloads = 0
                self.fp8_meta = {
                    key: SimpleNamespace(
                        amax_history=torch.ones(2, 3),
                        scale=torch.ones(3),
                        scale_inv=torch.ones(3),
                    )
                    for key in ("scaling_fwd", "scaling_bwd")
                }

            def get_extra_state(self) -> dict[str, bool]:
                return {"loaded": True}

            def set_extra_state(self, state: object) -> None:
                self.reloads += 1

        first = FakeFp8Module()
        second = FakeFp8Module()
        model = torch.nn.Sequential(first, torch.nn.Sequential(second))
        with patch("torch.cuda.device", return_value=nullcontext()):
            fixup_fp8_extra_states_on_module_device(model)
        for module in (first, second):
            self.assertEqual(module.reloads, 1)
            self.assertFalse(module.fp8_meta_tensors_initialized)
            for scaling in module.fp8_meta.values():
                self.assertEqual(scaling.scale_inv.device.type, "cpu")

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

    def test_software_fp8_weight_quantizer_rejects_non_cpu_source(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        with self.assertRaisesRegex(ValueError, "source must be on CPU"):
            software_e4m3_quantize_weight(
                torch.ones(1, device="cuda"),
                1.0,
                torch.device("cuda"),
            )

    def test_software_fp8_weight_quantizer_preserves_codes(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        source = torch.tensor(
            [-448.0, -1.0625, -0.001953125, 0.0, 0.001953125, 1.0625, 448.0],
            dtype=torch.bfloat16,
        )
        scale = 1.125
        expected = software_e4m3_quantize(source.cuda(), scale).to(
            torch.bfloat16
        )
        actual = software_e4m3_quantize_weight(
            source,
            scale,
            torch.device("cuda"),
            chunk_elements=3,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_software_fp8_projection_count_is_model_specific(self) -> None:
        blocks = [
            SimpleNamespace(projections=object()),
            SimpleNamespace(),
            SimpleNamespace(projections=object()),
        ]
        model = SimpleNamespace(blocks=blocks)
        self.assertEqual(software_fp8_projection_layers(model), (0, 2))

    def test_software_fp8_normalizes_empty_te_bias(self) -> None:
        self.assertIsNone(
            software_fp8_projection_bias(SimpleNamespace(bias=None))
        )
        self.assertIsNone(
            software_fp8_projection_bias(
                SimpleNamespace(bias=torch.empty(0, dtype=torch.bfloat16))
            )
        )
        bias = torch.ones(3, dtype=torch.bfloat16)
        self.assertIs(
            software_fp8_projection_bias(SimpleNamespace(bias=bias)),
            bias,
        )


if __name__ == "__main__":
    unittest.main()
