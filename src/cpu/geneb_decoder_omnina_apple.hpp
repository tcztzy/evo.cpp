// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

#include "evo/cpu/geneb_decoder.hpp"

namespace evo::cpu::detail {

// This backend is intentionally narrower than the portable decoder runtime.
// It reproduces the Apple-arm64 F32 operator order selected by PyTorch 2.7.1
// for the exact OmniNA topology.  Unsupported hosts may load the artifact, but
// must reject forward before reading activations.
[[nodiscard]] bool omnina_apple_f32_kernel_supported() noexcept;

[[nodiscard]] Status omnina_apple_f32_linear(
    const std::vector<float> &input, std::size_t rows, std::size_t input_width,
    const GenebDecoderTensorView &weight, std::size_t output_width,
    const GenebDecoderTensorView *bias, std::vector<float> *output);

[[nodiscard]] Status
omnina_apple_f32_rms_norm(const std::vector<float> &input, std::size_t rows,
                          const GenebDecoderTensorView &scale, float epsilon,
                          std::vector<float> *output);

[[nodiscard]] Status omnina_apple_f32_apply_rope(std::vector<float> *query,
                                                 std::vector<float> *key,
                                                 std::size_t rows,
                                                 std::size_t position_offset);

[[nodiscard]] Status
omnina_apple_f32_causal_attention(const std::vector<float> &query,
                                  const std::vector<float> &key,
                                  const std::vector<float> &value,
                                  std::size_t rows, std::vector<float> *output);

[[nodiscard]] Status omnina_apple_f32_swiglu(const std::vector<float> &gate,
                                             std::vector<float> *up);

} // namespace evo::cpu::detail
