// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include "evo2c/cuda/runtime.hpp"
#include "evo2c/status.hpp"

namespace evo2c::cuda {

// Contiguous BF16 caches [capacity,heads,head_dim]. Length is host metadata;
// appends are ordered on the supplied CUDA stream.
struct KvCache final {
  DeviceBuffer key;
  DeviceBuffer value;
  std::size_t capacity{0};
  std::size_t length{0};
  std::size_t heads{0};
  std::size_t head_dim{0};

  [[nodiscard]] Status allocate(int device, std::size_t token_capacity,
                                std::size_t head_count,
                                std::size_t dimensions_per_head);
  void reset_length() noexcept { length = 0; }
};

// Exact-shape BF16 Q/K/V scratch [tokens,heads,head_dim].
struct AttentionWorkspace final {
  DeviceBuffer query;
  DeviceBuffer key;
  DeviceBuffer value;
  std::size_t tokens{0};
  std::size_t heads{0};
  std::size_t head_dim{0};

  [[nodiscard]] Status allocate(int device, std::size_t token_count,
                                std::size_t head_count,
                                std::size_t dimensions_per_head);
};

// QKV input is BF16 [tokens,3,heads,head_dim]. Outputs are BF16
// [tokens,heads,head_dim].
[[nodiscard]] Status bf16_split_qkv(const DeviceBuffer &qkv, std::size_t tokens,
                                    std::size_t heads, std::size_t head_dim,
                                    DeviceBuffer *query, DeviceBuffer *key,
                                    DeviceBuffer *value, const Stream &stream);

// In-place GPT-NeoX half-pair RoPE on BF16 Q/K. inverse_frequency is F32
// [head_dim/2]; positions are (position_offset+token)/position_scale.
[[nodiscard]] Status bf16_apply_rope(DeviceBuffer *query, DeviceBuffer *key,
                                     const DeviceBuffer &inverse_frequency,
                                     std::size_t tokens, std::size_t heads,
                                     std::size_t head_dim,
                                     std::size_t position_offset,
                                     float position_scale,
                                     const Stream &stream);

[[nodiscard]] Status bf16_kv_append(const DeviceBuffer &key,
                                    const DeviceBuffer &value,
                                    std::size_t tokens, KvCache *cache,
                                    const Stream &stream);

// Queries are BF16 [query_tokens,heads,head_dim]. Query i attends to cache
// sources [0,prefix_tokens+i]. The F32 online softmax never materializes a
// score matrix and writes BF16 output with the same shape as query.
[[nodiscard]] Status bf16_online_causal_attention(const DeviceBuffer &query,
                                                  std::size_t query_tokens,
                                                  std::size_t prefix_tokens,
                                                  const KvCache &cache,
                                                  DeviceBuffer *output,
                                                  const Stream &stream);

// Full MHA inner operation. It splits QKV, applies RoPE at cache.length,
// appends K/V, and computes causal output for the newly appended chunk.
[[nodiscard]] Status
bf16_mha_prefill(const DeviceBuffer &qkv, const DeviceBuffer &inverse_frequency,
                 std::size_t tokens, std::size_t heads, std::size_t head_dim,
                 float position_scale, KvCache *cache,
                 AttentionWorkspace *workspace, DeviceBuffer *output,
                 const Stream &stream);

[[nodiscard]] Status
bf16_mha_decode(const DeviceBuffer &qkv, const DeviceBuffer &inverse_frequency,
                std::size_t heads, std::size_t head_dim, float position_scale,
                KvCache *cache, AttentionWorkspace *workspace,
                DeviceBuffer *output, const Stream &stream);

} // namespace evo2c::cuda
