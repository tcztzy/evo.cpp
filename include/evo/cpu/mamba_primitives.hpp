// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "evo/model_format.hpp"
#include "evo/status.hpp"

namespace evo::detail {
class LinearExecutor;
}

namespace evo::cpu {

// Non-owning F32 tensor view used by the reusable Mamba CPU primitives. The
// caller owns the backing artifact mapping for the complete call lifetime.
struct MambaTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct Mamba1Config final {
  std::size_t model_width{0};
  std::size_t inner_width{0};
  std::size_t state_width{0};
  std::size_t convolution_width{0};
  std::size_t time_step_rank{0};
  // JanusDNA normalizes the three x_projection slices independently before
  // dt_projection/selective scan. Caduceus leaves this disabled.
  bool parameter_projection_rms_norm{false};
  float parameter_projection_norm_epsilon{1.0e-6F};
};

struct Mamba1Weights final {
  MambaTensorView input_projection;
  MambaTensorView convolution_weight;
  MambaTensorView convolution_bias;
  MambaTensorView x_projection;
  MambaTensorView time_step_projection;
  MambaTensorView time_step_bias;
  MambaTensorView a_log;
  MambaTensorView skip;
  MambaTensorView output_projection;
  MambaTensorView projected_time_step_norm_scale;
  MambaTensorView projected_b_norm_scale;
  MambaTensorView projected_c_norm_scale;
};

struct Mamba2Config final {
  std::size_t model_width{0};
  std::size_t inner_width{0};
  std::size_t state_width{0};
  std::size_t convolution_width{0};
  std::size_t head_width{0};
  std::size_t heads{0};
  std::size_t groups{0};
  bool skip_has_inner_dimension{false};
  bool rms_norm{true};
  bool norm_before_gate{false};
  float norm_epsilon{1.0e-5F};
};

struct Mamba2Weights final {
  MambaTensorView input_projection;
  MambaTensorView convolution_weight;
  MambaTensorView convolution_bias;
  MambaTensorView time_step_bias;
  MambaTensorView a_log;
  MambaTensorView skip;
  MambaTensorView norm_scale;
  MambaTensorView output_projection;
};

// Row-major bias-free F32 projection with weight [output_width,input_width].
// The optional executor keeps model adapters on the shared optimized GEMM path.
[[nodiscard]] Status mamba_linear_f32(
    const std::vector<float> &input, std::size_t rows, std::size_t input_width,
    const MambaTensorView &weight, std::size_t output_width,
    evo::detail::LinearExecutor *linear_executor, std::vector<float> *output);

// input/output are row-major [rows,channels]. Weight is
// [channels,1,kernel_width], using PyTorch Conv1d correlation order and causal
// left padding. Bias may be null. SiLU is applied after the convolution.
[[nodiscard]] Status mamba_causal_depthwise_conv_silu(
    const std::vector<float> &input, std::size_t rows, std::size_t channels,
    std::size_t kernel_width, const MambaTensorView &weight,
    const MambaTensorView *bias, std::vector<float> *output);

[[nodiscard]] Status mamba_rms_norm(const std::vector<float> &input,
                                    std::size_t rows, std::size_t width,
                                    const MambaTensorView &scale, float epsilon,
                                    std::vector<float> *output);

// Implements Mamba2 RMSNormGated. group_width must divide width. When
// norm_before_gate=false, input*SiLU(gate) is normalized; otherwise normalized
// input is gated after scaling.
[[nodiscard]] Status
mamba_group_rms_norm_gated(const std::vector<float> &input,
                           const std::vector<float> &gate, std::size_t rows,
                           std::size_t width, std::size_t group_width,
                           const MambaTensorView &scale, float epsilon,
                           bool norm_before_gate, std::vector<float> *output);

// Mamba1 selective scan. x/gate are [rows,inner], raw_time_step is
// [rows,inner], and b/c are [rows,state]. The recurrence is the official
// exp(delta*A) discretization and uses O(inner*state) mutable state.
[[nodiscard]] Status mamba1_selective_scan_f32(
    const std::vector<float> &x, const std::vector<float> &raw_time_step,
    const std::vector<float> &b, const std::vector<float> &c,
    const std::vector<float> &gate, std::size_t rows,
    const Mamba1Config &config, const MambaTensorView &time_step_bias,
    const MambaTensorView &a_log, const MambaTensorView &skip,
    std::vector<float> *output);

// Complete bias-free Mamba1 mixer: projections, optional independent RMSNorm
// of the projected dt/B/C slices, causal conv, selective scan, gate, and
// output projection. This deliberately excludes residual/block norm,
// bidirectionality, RCPS, and model-specific masking.
[[nodiscard]] Status
mamba1_mixer_f32(const std::vector<float> &input, std::size_t rows,
                 const Mamba1Config &config, const Mamba1Weights &weights,
                 evo::detail::LinearExecutor *linear_executor,
                 std::vector<float> *output);

// Mamba2 SSD recurrence. x/gate are [rows,inner], b/c are
// [rows,groups*state], raw_time_step is [rows,heads]. State is
// O(heads*head_width*state); runtime is linear in rows.
[[nodiscard]] Status mamba2_selective_scan_f32(
    const std::vector<float> &x, const std::vector<float> &b,
    const std::vector<float> &c, const std::vector<float> &raw_time_step,
    const std::vector<float> &gate, std::size_t rows,
    const Mamba2Config &config, const MambaTensorView &time_step_bias,
    const MambaTensorView &a_log, const MambaTensorView &skip,
    const MambaTensorView *norm_scale, std::vector<float> *output);

// Complete Mamba2 mixer for the no-local-MLP layout used by eccDNAMamba.
// Residual/norm and the block GatedMLP remain model-level operations.
[[nodiscard]] Status
mamba2_mixer_f32(const std::vector<float> &input, std::size_t rows,
                 const Mamba2Config &config, const Mamba2Weights &weights,
                 evo::detail::LinearExecutor *linear_executor,
                 std::vector<float> *output);

} // namespace evo::cpu
