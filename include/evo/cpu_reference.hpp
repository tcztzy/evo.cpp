// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

#include "evo/status.hpp"

namespace evo::cpu {

enum class MlpActivation { kGelu, kIdentity };
enum class FirOrientation { kCrossCorrelation, kCausalConvolution };
enum class FirBiasMode { kAdd, kMultiplyInput };

// All tensors are contiguous row-major F32 arrays. Sequence tensors use [time, width].
[[nodiscard]] Status rms_norm(const std::vector<float>& input,
                              std::size_t rows,
                              std::size_t width,
                              const std::vector<float>& scale,
                              float epsilon,
                              std::vector<float>* output);

// PyTorch nn.Linear layout: weight[out_features, in_features].
[[nodiscard]] Status linear(const std::vector<float>& input,
                            std::size_t rows,
                            std::size_t in_features,
                            const std::vector<float>& weight,
                            std::size_t out_features,
                            const std::vector<float>* bias,
                            std::vector<float>* output);

[[nodiscard]] Status gated_mlp(const std::vector<float>& input,
                               std::size_t rows,
                               std::size_t width,
                               std::size_t inner_width,
                               const std::vector<float>& l1_weight,
                               const std::vector<float>& l2_weight,
                               const std::vector<float>& l3_weight,
                               MlpActivation activation,
                               std::vector<float>* output);

// Exact PyTorch conv1d cross-correlation with left padding kernel_size-1.
[[nodiscard]] Status causal_depthwise_fir(const std::vector<float>& input,
                                          std::size_t length,
                                          std::size_t channels,
                                          const std::vector<float>& weight,
                                          std::size_t kernel_size,
                                          const std::vector<float>* bias,
                                          std::vector<float>* output,
                                          FirOrientation orientation = FirOrientation::kCrossCorrelation,
                                          FirBiasMode bias_mode = FirBiasMode::kAdd);

// Projection rows contain width interleaved triples [x2_i, x1_i, v_i].
[[nodiscard]] Status split_hyena_projection(const std::vector<float>& projection,
                                             std::size_t length,
                                             std::size_t width,
                                             std::vector<float>* x2,
                                             std::vector<float>* x1,
                                             std::vector<float>* value);

// state[width, state_size] is updated in place and remains F32.
[[nodiscard]] Status hcl_recurrence(const std::vector<float>& x2,
                                    const std::vector<float>& x1,
                                    const std::vector<float>& value,
                                    std::size_t length,
                                    std::size_t width,
                                    const std::vector<float>& direct,
                                    const std::vector<float>& log_poles,
                                    const std::vector<float>& residues,
                                    std::size_t state_size,
                                    std::vector<float>* state,
                                    std::vector<float>* output);

// GPT-NeoX half-pair RoPE. q and k use [time, heads, head_dim].
[[nodiscard]] Status apply_rope(std::vector<float>* query,
                                std::vector<float>* key,
                                std::size_t length,
                                std::size_t heads,
                                std::size_t head_dim,
                                const std::vector<float>& inverse_frequency,
                                std::size_t position_offset,
                                float position_scale);

// Full causal reference attention; production paths must use online/chunked attention.
[[nodiscard]] Status causal_attention(const std::vector<float>& query,
                                      const std::vector<float>& key,
                                      const std::vector<float>& value,
                                      std::size_t length,
                                      std::size_t heads,
                                      std::size_t head_dim,
                                      std::vector<float>* output);

}  // namespace evo::cpu
