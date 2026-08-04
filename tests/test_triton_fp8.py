#!/usr/bin/env python3
"""CUDA oracle tests for the Ampere software-FP8 accumulator."""

from __future__ import annotations

import math
import sys
import unittest

try:
    import torch
    import triton  # noqa: F401
except ModuleNotFoundError:
    print("SKIP: PyTorch or Triton is not installed", file=sys.stderr)
    raise SystemExit(77)

from evo.triton_fp8 import e8m13_linear, h100_qgmma_linear


def truncate_e8m13(value: torch.Tensor, *, rne: bool) -> torch.Tensor:
    bits = value.contiguous().view(torch.int32)
    if rne:
        retained_lsb = (bits >> 10) & 1
        bits = bits + 0x1FF + retained_lsb
    return (bits & -1024).view(torch.float32)


def oracle(
    value: torch.Tensor,
    weight: torch.Tensor,
    *,
    rne: bool,
    promotion_interval: int,
) -> torch.Tensor:
    outer = torch.zeros(
        (value.shape[0], weight.shape[0]),
        dtype=torch.float32,
    )
    local = torch.zeros_like(outer)
    for start in range(0, value.shape[1], 32):
        local += value[:, start : start + 32].float() @ weight[
            :, start : start + 32
        ].float().T
        local = truncate_e8m13(local, rne=rne)
        if (start + 32) % promotion_interval == 0:
            outer += local
            local.zero_()
    return outer.to(torch.bfloat16)


def h100_qgmma_oracle(
    value: torch.Tensor,
    weight: torch.Tensor,
    output_scale: float,
) -> torch.Tensor:
    output = torch.empty(
        (value.shape[0], weight.shape[0]),
        dtype=torch.float32,
    )
    for row in range(value.shape[0]):
        for column in range(weight.shape[0]):
            accumulator = 0.0
            for start in range(0, value.shape[1], 32):
                products: list[tuple[float, int]] = []
                for offset in range(32):
                    left = float(value[row, start + offset])
                    right = float(weight[column, start + offset])
                    product = left * right
                    if product != 0.0:
                        product_exponent = (
                            math.frexp(abs(left))[1]
                            + math.frexp(abs(right))[1]
                            - 2
                        )
                        products.append((product, product_exponent))

                accumulator_exponent = (
                    math.frexp(abs(accumulator))[1] - 1
                    if accumulator != 0.0
                    else -1024
                )
                maximum_exponent = max(
                    [accumulator_exponent]
                    + [exponent for _, exponent in products]
                )
                aligned_sum = 0
                if accumulator != 0.0:
                    significand = math.ldexp(
                        abs(accumulator),
                        -accumulator_exponent,
                    )
                    aligned = int(significand * (1 << 13)) >> (
                        maximum_exponent - accumulator_exponent
                    )
                    aligned_sum += -aligned if accumulator < 0.0 else aligned
                for product, exponent in products:
                    significand = math.ldexp(abs(product), -exponent)
                    shift = maximum_exponent - exponent
                    aligned = (
                        int(significand * (1 << 13)) >> shift
                        if shift <= 31
                        else 0
                    )
                    aligned_sum += -aligned if product < 0.0 else aligned

                raw = math.ldexp(aligned_sum, maximum_exponent - 13)
                accumulator = float(
                    truncate_e8m13(
                        torch.tensor(raw, dtype=torch.float32),
                        rne=False,
                    )
                )
            output[row, column] = accumulator * output_scale
    return output.to(torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class TritonFp8Tests(unittest.TestCase):
    def test_matches_cpu_oracle_at_each_promotion_interval(self) -> None:
        generator = torch.Generator().manual_seed(1729)
        value = torch.randn(5, 64, generator=generator).to(torch.bfloat16)
        weight = torch.randn(7, 64, generator=generator).to(torch.bfloat16)

        for rounding, rne in (("rne", True), ("rtz", False)):
            for interval in (32, 64):
                with self.subTest(rounding=rounding, interval=interval):
                    expected = oracle(
                        value,
                        weight,
                        rne=rne,
                        promotion_interval=interval,
                    )
                    actual = e8m13_linear(
                        value.cuda(),
                        weight.cuda(),
                        1.0,
                        rounding=rounding,
                        promotion_interval=interval,
                    ).cpu()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_h100_qgmma_matches_global_alignment_oracle(self) -> None:
        generator = torch.Generator().manual_seed(2026)
        value = (
            torch.randn(3, 64, generator=generator)
            .mul_(3.0)
            .to(torch.float8_e4m3fn)
            .float()
            .to(torch.bfloat16)
        )
        weight = (
            torch.randn(5, 64, generator=generator)
            .mul_(2.0)
            .to(torch.float8_e4m3fn)
            .float()
            .to(torch.bfloat16)
        )
        expected = h100_qgmma_oracle(value, weight, 0.125)
        actual = h100_qgmma_linear(
            value.cuda(),
            weight.cuda(),
            0.125,
        ).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_invalid_promotion_interval(self) -> None:
        value = torch.zeros(1, 64, dtype=torch.bfloat16, device="cuda")
        weight = torch.zeros(1, 64, dtype=torch.bfloat16, device="cuda")
        for interval in (1, 16, 96):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, "promotion_interval"):
                    e8m13_linear(
                        value,
                        weight,
                        1.0,
                        promotion_interval=interval,
                    )


if __name__ == "__main__":
    unittest.main()
