// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "evo/cuda/runtime.hpp"
#include "evo/status.hpp"

namespace evo::cuda {

enum class QkvLayout {
  // [tokens,3,heads,head_dim], as produced by a projection-major weight.
  kProjectionMajor,
  // [tokens,heads,3,head_dim], as stored by the official Evo 2 checkpoint.
  kHeadMajor,
};

enum class KvCacheType { kBF16, kQ8Paged };

struct Q8KvPage final {
  DeviceBuffer key;
  DeviceBuffer value;
  DeviceBuffer key_scale;
  DeviceBuffer value_scale;
};

// Length is host metadata; appends are ordered on the supplied CUDA stream.
// BF16 uses contiguous [capacity,heads,head_dim] buffers. Q8 uses fixed token
// pages with int8 payloads and one F32 scale per (token,head).
struct KvCache final {
  DeviceBuffer key;
  DeviceBuffer value;
  std::vector<Q8KvPage> q8_pages;
  DeviceBuffer key_page_table;
  DeviceBuffer value_page_table;
  DeviceBuffer key_scale_page_table;
  DeviceBuffer value_scale_page_table;
  std::vector<std::int8_t *> host_key_pages;
  std::vector<std::int8_t *> host_value_pages;
  std::vector<float *> host_key_scale_pages;
  std::vector<float *> host_value_scale_pages;
  KvCacheType type{KvCacheType::kBF16};
  int device_id{-1};
  std::size_t capacity{0};
  std::size_t length{0};
  std::size_t heads{0};
  std::size_t head_dim{0};
  std::size_t page_tokens{0};

  [[nodiscard]] Status allocate(int device, std::size_t token_capacity,
                                std::size_t head_count,
                                std::size_t dimensions_per_head);
  [[nodiscard]] Status allocate_q8_paged(int device, std::size_t token_capacity,
                                         std::size_t head_count,
                                         std::size_t dimensions_per_head,
                                         std::size_t tokens_per_page,
                                         const Stream &stream);
  [[nodiscard]] std::size_t allocated_bytes() const noexcept;
  [[nodiscard]] bool quantized() const noexcept {
    return type == KvCacheType::kQ8Paged;
  }
  void reset_length() noexcept { length = 0; }
};

// Exact-shape BF16 Q/K/V scratch [tokens,heads,head_dim].
struct AttentionWorkspace final {
  DeviceBuffer query;
  DeviceBuffer key;
  DeviceBuffer value;
  DeviceBuffer softmax_lse;
  DeviceBuffer softmax_lse_accum;
  DeviceBuffer output_accum;
  DeviceBuffer scaled_key;
  DeviceBuffer scores;
  DeviceBuffer probabilities;
  std::size_t tokens{0};
  std::size_t heads{0};
  std::size_t head_dim{0};

  [[nodiscard]] Status allocate(int device, std::size_t token_count,
                                std::size_t head_count,
                                std::size_t dimensions_per_head);
};

// PyTorch FlashAttention-compatible BF16 causal attention. Inputs use
// [tokens,heads,128], key/value may include a prefix, and softmax_lse is F32
// scratch with at least query_tokens*heads elements. The two accumulator
// buffers are grown on demand when PyTorch's split-KV occupancy heuristic
// selects the two-stage forward kernel.
[[nodiscard]] Status bf16_flash_causal_attention(
    const DeviceBuffer &query, const DeviceBuffer &key,
    const DeviceBuffer &value, std::size_t query_tokens, std::size_t key_tokens,
    std::size_t heads, std::size_t head_dim, DeviceBuffer *softmax_lse,
    DeviceBuffer *softmax_lse_accum, DeviceBuffer *output_accum,
    DeviceBuffer *output, const Stream &stream);

// QKV input uses `layout`. Outputs are BF16 [tokens,heads,head_dim].
[[nodiscard]] Status
bf16_split_qkv(const DeviceBuffer &qkv, std::size_t tokens, std::size_t heads,
               std::size_t head_dim, DeviceBuffer *query, DeviceBuffer *key,
               DeviceBuffer *value, const Stream &stream,
               QkvLayout layout = QkvLayout::kProjectionMajor);

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

// Exact Vortex cached-generation fallback when use_flash_attn=false:
// BF16 K scaling, strided-batched QK GEMM, causal masked_fill, PyTorch's
// persistent warp softmax, then BF16 probability/value GEMM.
[[nodiscard]] Status bf16_cached_cross_attention(
    const Blas &blas, const DeviceBuffer &query, const DeviceBuffer &key,
    const DeviceBuffer &value, std::size_t query_tokens, std::size_t key_tokens,
    std::size_t heads, std::size_t head_dim, DeviceBuffer *scaled_key,
    DeviceBuffer *scores, DeviceBuffer *probabilities, DeviceBuffer *output,
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

} // namespace evo::cuda
