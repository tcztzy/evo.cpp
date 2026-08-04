"""Ampere reference kernels for Hopper FP8 fast-accumulator semantics."""

from typing import Literal

import torch
import triton
import triton.language as tl


@triton.jit
def _h100_qgmma_linear_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    rows,
    columns: tl.constexpr,
    inner: tl.constexpr,
    output_scale,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
    group_rows: tl.constexpr,
):
    """Emulate H100 QGMMA's K=32 globally aligned FP8 block FMA."""
    program = tl.program_id(axis=0)
    row_programs = tl.cdiv(rows, block_rows)
    column_programs = tl.cdiv(columns, block_columns)
    programs_per_group = group_rows * column_programs
    group = program // programs_per_group
    first_row_program = group * group_rows
    group_size = tl.minimum(row_programs - first_row_program, group_rows)
    row_program = first_row_program + (program % group_size)
    column_program = (program % programs_per_group) // group_size

    row_offsets = row_program * block_rows + tl.arange(0, block_rows)
    column_offsets = (
        column_program * block_columns + tl.arange(0, block_columns)
    )
    accumulator = tl.zeros((block_rows, block_columns), dtype=tl.float32)

    for inner_start in tl.range(0, inner, 32):
        maximum_exponent = tl.full(
            (block_rows, block_columns),
            -1024,
            dtype=tl.int32,
        )

        # QGMMA aligns all 32 denormalized products and the incoming
        # accumulator to one shared exponent before summation.
        for offset in tl.static_range(0, 32):
            input_value = tl.load(
                input_pointer + row_offsets * inner + inner_start + offset,
                mask=row_offsets < rows,
                other=0.0,
            ).to(tl.float32)
            weight_value = tl.load(
                weight_pointer
                + column_offsets * inner
                + inner_start
                + offset,
                mask=column_offsets < columns,
                other=0.0,
            ).to(tl.float32)
            input_bits = tl.cast(input_value, tl.uint32, bitcast=True)
            weight_bits = tl.cast(weight_value, tl.uint32, bitcast=True)
            input_exponent = (
                ((input_bits >> 23) & 0xFF).to(tl.int32) - 127
            )
            weight_exponent = (
                ((weight_bits >> 23) & 0xFF).to(tl.int32) - 127
            )
            product_exponent = (
                input_exponent[:, None] + weight_exponent[None, :]
            )
            product_nonzero = (
                input_value[:, None] != 0.0
            ) & (weight_value[None, :] != 0.0)
            candidate = tl.where(
                product_nonzero,
                product_exponent,
                -1024,
            )
            maximum_exponent = tl.where(
                candidate > maximum_exponent,
                candidate,
                maximum_exponent,
            )

        accumulator_bits = tl.cast(
            accumulator,
            tl.uint32,
            bitcast=True,
        )
        accumulator_magnitude = accumulator_bits & 0x7FFFFFFF
        accumulator_nonzero = accumulator_magnitude != 0
        accumulator_exponent = (
            ((accumulator_magnitude >> 23) & 0xFF).to(tl.int32) - 127
        )
        accumulator_candidate = tl.where(
            accumulator_nonzero,
            accumulator_exponent,
            -1024,
        )
        maximum_exponent = tl.where(
            accumulator_candidate > maximum_exponent,
            accumulator_candidate,
            maximum_exponent,
        )

        accumulator_significand = (
            (accumulator_magnitude & 0x7FFFFF) | 0x800000
        ) >> 10
        accumulator_shift = maximum_exponent - accumulator_exponent
        accumulator_valid = accumulator_nonzero & (accumulator_shift <= 31)
        accumulator_shift = tl.minimum(
            tl.maximum(accumulator_shift, 0),
            31,
        )
        aligned_accumulator = tl.where(
            accumulator_valid,
            accumulator_significand >> accumulator_shift,
            0,
        ).to(tl.int32)
        aligned_accumulator = tl.where(
            (accumulator_bits & 0x80000000) != 0,
            -aligned_accumulator,
            aligned_accumulator,
        )
        aligned_sum = aligned_accumulator

        for offset in tl.static_range(0, 32):
            input_value = tl.load(
                input_pointer + row_offsets * inner + inner_start + offset,
                mask=row_offsets < rows,
                other=0.0,
            ).to(tl.float32)
            weight_value = tl.load(
                weight_pointer
                + column_offsets * inner
                + inner_start
                + offset,
                mask=column_offsets < columns,
                other=0.0,
            ).to(tl.float32)
            product = input_value[:, None] * weight_value[None, :]
            product_bits = tl.cast(product, tl.uint32, bitcast=True)
            product_magnitude = product_bits & 0x7FFFFFFF
            product_nonzero = product_magnitude != 0

            input_bits = tl.cast(input_value, tl.uint32, bitcast=True)
            weight_bits = tl.cast(weight_value, tl.uint32, bitcast=True)
            input_exponent = (
                ((input_bits >> 23) & 0xFF).to(tl.int32) - 127
            )
            weight_exponent = (
                ((weight_bits >> 23) & 0xFF).to(tl.int32) - 127
            )
            product_exponent = (
                input_exponent[:, None] + weight_exponent[None, :]
            )
            normalized_exponent = (
                ((product_magnitude >> 23) & 0xFF).to(tl.int32) - 127
            )
            product_significand = (
                (product_magnitude & 0x7FFFFF) | 0x800000
            )
            product_significand <<= (
                normalized_exponent - product_exponent
            )
            product_significand >>= 10
            product_shift = maximum_exponent - product_exponent
            product_valid = product_nonzero & (product_shift <= 31)
            product_shift = tl.minimum(tl.maximum(product_shift, 0), 31)
            aligned_product = tl.where(
                product_valid,
                product_significand >> product_shift,
                0,
            ).to(tl.int32)
            aligned_product = tl.where(
                (product_bits & 0x80000000) != 0,
                -aligned_product,
                aligned_product,
            )
            aligned_sum += aligned_product

        # The integer sum is exact at maximum_exponent - 13.  QGMMA then
        # normalizes and truncates it to 13 fractional bits.
        accumulator = aligned_sum.to(tl.float32) * tl.exp2(
            (maximum_exponent - 13).to(tl.float32)
        )
        accumulator_bits = tl.cast(
            accumulator,
            tl.uint32,
            bitcast=True,
        )
        accumulator_bits &= 0xFFFFFC00
        accumulator = tl.cast(
            accumulator_bits,
            tl.float32,
            bitcast=True,
        )

    output_offsets = row_offsets[:, None] * columns + column_offsets[None, :]
    tl.store(
        output_pointer + output_offsets,
        accumulator * output_scale,
        mask=(row_offsets[:, None] < rows)
        & (column_offsets[None, :] < columns),
    )


@triton.jit
def _e8m13_linear_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    rows,
    columns: tl.constexpr,
    inner: tl.constexpr,
    output_scale,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
    block_inner: tl.constexpr,
    promotion_interval: tl.constexpr,
    group_rows: tl.constexpr,
    round_to_nearest: tl.constexpr,
):
    program = tl.program_id(axis=0)
    row_programs = tl.cdiv(rows, block_rows)
    column_programs = tl.cdiv(columns, block_columns)
    programs_per_group = group_rows * column_programs
    group = program // programs_per_group
    first_row_program = group * group_rows
    group_size = tl.minimum(row_programs - first_row_program, group_rows)
    row_program = first_row_program + (program % group_size)
    column_program = (program % programs_per_group) // group_size

    row_offsets = row_program * block_rows + tl.arange(0, block_rows)
    column_offsets = (
        column_program * block_columns + tl.arange(0, block_columns)
    )
    inner_offsets = tl.arange(0, block_inner)
    accumulator = tl.zeros((block_rows, block_columns), dtype=tl.float32)
    local_accumulator = tl.zeros(
        (block_rows, block_columns),
        dtype=tl.float32,
    )

    for inner_start in tl.range(0, inner, block_inner):
        input_offsets = (
            row_offsets[:, None] * inner + inner_start + inner_offsets[None, :]
        )
        weight_offsets = (
            column_offsets[:, None] * inner
            + inner_start
            + inner_offsets[None, :]
        )
        input_tile = tl.load(
            input_pointer + input_offsets,
            mask=(row_offsets[:, None] < rows)
            & (inner_start + inner_offsets[None, :] < inner),
            other=0.0,
        )
        weight_tile = tl.load(
            weight_pointer + weight_offsets,
            mask=(column_offsets[:, None] < columns)
            & (inner_start + inner_offsets[None, :] < inner),
            other=0.0,
        )
        local_accumulator += tl.dot(
            input_tile,
            tl.trans(weight_tile),
            out_dtype=tl.float32,
        )

        # Hopper FP8 fast accumulation exposes an e8m13 accumulator: the
        # low ten FP32 mantissa bits are absent.  Quantize after each native
        # K=32 FP8 MMA step, before applying the two inverse scales.
        bits = tl.cast(local_accumulator, tl.uint32, bitcast=True)
        if round_to_nearest:
            retained_lsb = (bits >> 10) & 1
            bits += 0x1FF + retained_lsb
        bits &= 0xFFFFFC00
        local_accumulator = tl.cast(bits, tl.float32, bitcast=True)

        promote = (
            (inner_start + block_inner) % promotion_interval == 0
        ) | (inner_start + block_inner >= inner)
        accumulator += tl.where(promote, local_accumulator, 0.0)
        local_accumulator = tl.where(promote, 0.0, local_accumulator)

    output_offsets = row_offsets[:, None] * columns + column_offsets[None, :]
    tl.store(
        output_pointer + output_offsets,
        accumulator * output_scale,
        mask=(row_offsets[:, None] < rows)
        & (column_offsets[None, :] < columns),
    )


def e8m13_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    output_scale: float,
    *,
    rounding: Literal["rne", "rtz"] = "rne",
    promotion_interval: int | None = None,
) -> torch.Tensor:
    """Compute a scaled raw-E4M3 linear op with an emulated e8m13 accumulator."""
    if value.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("e8m13_linear inputs must be bfloat16")
    if value.device != weight.device or value.device.type != "cuda":
        raise ValueError("e8m13_linear inputs must share one CUDA device")
    if not value.is_contiguous() or not weight.is_contiguous():
        raise ValueError("e8m13_linear inputs must be contiguous")
    if value.shape[-1] != weight.shape[-1]:
        raise ValueError("e8m13_linear inner dimensions differ")
    if rounding not in {"rne", "rtz"}:
        raise ValueError(f"unsupported e8m13 rounding mode {rounding!r}")

    inner = int(value.shape[-1])
    if inner == 0 or inner % 32 != 0:
        raise ValueError("e8m13_linear inner dimension must be a positive multiple of 32")
    if promotion_interval is None:
        promotion_interval = inner
    if (
        promotion_interval < 32
        or promotion_interval > inner
        or promotion_interval % 32 != 0
    ):
        raise ValueError(
            "promotion_interval must be a multiple of 32 between 32 and the inner dimension"
        )
    columns = int(weight.shape[0])
    rows = value.numel() // inner
    output = torch.empty(
        (*value.shape[:-1], columns),
        dtype=value.dtype,
        device=value.device,
    )
    block_rows = 32
    block_columns = 32
    grid = (
        triton.cdiv(rows, block_rows)
        * triton.cdiv(columns, block_columns),
    )
    _e8m13_linear_kernel[grid](
        value,
        weight,
        output,
        rows,
        columns,
        inner,
        output_scale,
        block_rows,
        block_columns,
        32,
        promotion_interval,
        8,
        rounding == "rne",
        num_warps=8,
        num_stages=2,
    )
    return output


def h100_qgmma_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    output_scale: float,
) -> torch.Tensor:
    """Compute a scaled raw-E4M3 linear op with the H100 QGMMA bit model."""
    if value.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("h100_qgmma_linear inputs must be bfloat16")
    if value.device != weight.device or value.device.type != "cuda":
        raise ValueError("h100_qgmma_linear inputs must share one CUDA device")
    if not value.is_contiguous() or not weight.is_contiguous():
        raise ValueError("h100_qgmma_linear inputs must be contiguous")
    if value.shape[-1] != weight.shape[-1]:
        raise ValueError("h100_qgmma_linear inner dimensions differ")

    inner = int(value.shape[-1])
    if inner == 0 or inner % 32 != 0:
        raise ValueError(
            "h100_qgmma_linear inner dimension must be a positive multiple of 32"
        )
    columns = int(weight.shape[0])
    rows = value.numel() // inner
    output = torch.empty(
        (*value.shape[:-1], columns),
        dtype=value.dtype,
        device=value.device,
    )
    block_rows = 8
    block_columns = 8
    grid = (
        triton.cdiv(rows, block_rows)
        * triton.cdiv(columns, block_columns),
    )
    _h100_qgmma_linear_kernel[grid](
        value,
        weight,
        output,
        rows,
        columns,
        inner,
        output_scale,
        block_rows,
        block_columns,
        8,
        num_warps=4,
        num_stages=1,
    )
    return output
