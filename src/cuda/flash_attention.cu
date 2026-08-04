// SPDX-License-Identifier: Apache-2.0
#include "evo/cuda/attention.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <cuda_bf16.h>

#define FLASH_NAMESPACE evo_flash
#define FLASHATTENTION_DISABLE_ALIBI 1
#define FLASHATTENTION_DISABLE_SOFTCAP 1
#define UNFUSE_FMA 1
#include "flash_fwd_kernel.h"

namespace evo::cuda {
namespace {

struct FlashParams final {
  using index_t = std::int64_t;

  void *q_ptr{};
  void *k_ptr{};
  void *v_ptr{};
  index_t q_batch_stride{};
  index_t k_batch_stride{};
  index_t v_batch_stride{};
  index_t q_row_stride{};
  index_t k_row_stride{};
  index_t v_row_stride{};
  index_t q_head_stride{};
  index_t k_head_stride{};
  index_t v_head_stride{};
  int h{};
  int h_k{};
  int h_h_k_ratio{};
  void *o_ptr{};
  void *oaccum_ptr{};
  index_t o_batch_stride{};
  index_t o_row_stride{};
  index_t o_head_stride{};
  void *p_ptr{};
  void *softmax_lse_ptr{};
  void *softmax_lseaccum_ptr{};
  int b{};
  int seqlen_q{};
  int seqlen_k{};
  int seqlen_knew{};
  int d{};
  int seqlen_q_rounded{};
  int seqlen_k_rounded{};
  int d_rounded{};
  int rotary_dim{};
  int total_q{};
  float scale_softmax{};
  float scale_softmax_log2{};
  int *cu_seqlens_q{};
  int *cu_seqlens_k{};
  int *leftpad_k{};
  int *seqused_k{};
  int *blockmask{};
  void *knew_ptr{};
  void *vnew_ptr{};
  index_t knew_batch_stride{};
  index_t vnew_batch_stride{};
  index_t knew_row_stride{};
  index_t vnew_row_stride{};
  index_t knew_head_stride{};
  index_t vnew_head_stride{};
  void *rotary_cos_ptr{};
  void *rotary_sin_ptr{};
  int *cache_batch_idx{};
  int *block_table{};
  index_t block_table_batch_stride{};
  int page_block_size{};
  float p_dropout{};
  std::uint8_t p_dropout_in_uint8_t{};
  float rp_dropout{};
  float scale_softmax_rp_dropout{};
  int window_size_left{};
  int window_size_right{};
  float softcap{};
  at::PhiloxCudaState philox_args{};
  std::uint64_t *rng_state{};
  bool is_bf16{};
  bool is_causal{};
  bool is_seqlens_k_cumulative{};
  bool is_rotary_interleaved{};
  int num_splits{};
  void *alibi_slopes_ptr{};
  index_t alibi_slopes_batch_stride{};
  bool unpadded_lse{};
  bool seqlenq_ngroups_swapped{};
};

using FlashTraits =
    Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t>;
using SplitFlashTraits =
    Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::bfloat16_t>;

template <bool IsEvenMN>
__global__ void flash_causal_kernel(const FlashParams params) {
  evo_flash::compute_attn<FlashTraits, false, true, false, false, IsEvenMN,
                            true, false, false>(params);
}

template <bool IsEvenMN>
__global__ void flash_causal_split_kernel(const FlashParams params) {
  evo_flash::compute_attn_splitkv<SplitFlashTraits, true, false, false,
                                    IsEvenMN, true, false, true, false>(params);
}

template <int LogMaxSplits>
__global__ void flash_causal_combine_kernel(const FlashParams params) {
  evo_flash::combine_attn_seqk_parallel<SplitFlashTraits, 4, LogMaxSplits,
                                          true>(params);
}

bool multiply(const std::size_t left, const std::size_t right,
              std::size_t *const result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *result = left * right;
  return true;
}

Status validate_buffer(const DeviceBuffer &buffer, const std::size_t bytes,
                       const int device, const char *const name) {
  if (!buffer.valid() || buffer.device() != device || buffer.bytes() < bytes) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " has invalid device or capacity"};
  }
  return Status::Ok();
}

int round_up_128(const std::size_t value) {
  return static_cast<int>((value + 127U) / 128U * 128U);
}

int num_splits_heuristic(const int batch_heads_m_blocks, const int num_sms,
                         const int num_n_blocks, int max_splits) {
  // This is the exact FlashAttention heuristic vendored by the pinned
  // PyTorch build. It treats a 128-thread split-KV CTA as half an SM slot.
  if (static_cast<float>(batch_heads_m_blocks) >=
      0.8F * static_cast<float>(num_sms)) {
    return 1;
  }
  max_splits = std::min({max_splits, num_sms, num_n_blocks});
  float max_efficiency = 0.0F;
  std::array<float, 128> efficiency{};
  const auto ceil_div = [](const int left, const int right) {
    return (left + right - 1) / right;
  };
  const auto eligible = [&](const int splits) {
    return splits == 1 ||
           ceil_div(num_n_blocks, splits) != ceil_div(num_n_blocks, splits - 1);
  };
  for (int splits = 1; splits <= max_splits; ++splits) {
    if (!eligible(splits))
      continue;
    const float waves = static_cast<float>(batch_heads_m_blocks * splits) /
                        static_cast<float>(num_sms);
    const float value = waves / std::ceil(waves);
    efficiency[static_cast<std::size_t>(splits - 1)] = value;
    max_efficiency = std::max(max_efficiency, value);
  }
  for (int splits = 1; splits <= max_splits; ++splits) {
    if (eligible(splits) && efficiency[static_cast<std::size_t>(splits - 1)] >=
                                0.85F * max_efficiency) {
      return splits;
    }
  }
  return 1;
}

Status ensure_accumulator(DeviceBuffer *const buffer, const int device,
                          const std::size_t bytes, const char *const name) {
  if (buffer == nullptr)
    return {ErrorCode::kInvalidArgument, std::string{name} + " is null"};
  if (buffer->valid() && buffer->device() == device &&
      buffer->bytes() >= bytes) {
    return Status::Ok();
  }
  buffer->reset();
  auto status = buffer->allocate(device, bytes);
  if (!status.ok())
    return {status.code(),
            std::string{"allocate "} + name + ": " + status.message()};
  return Status::Ok();
}

} // namespace

Status bf16_flash_causal_attention(
    const DeviceBuffer &query, const DeviceBuffer &key,
    const DeviceBuffer &value, const std::size_t query_tokens,
    const std::size_t key_tokens, const std::size_t heads,
    const std::size_t head_dim, DeviceBuffer *const softmax_lse,
    DeviceBuffer *const softmax_lse_accum, DeviceBuffer *const output_accum,
    DeviceBuffer *const output, const Stream &stream) {
  if (softmax_lse == nullptr || softmax_lse_accum == nullptr ||
      output_accum == nullptr || output == nullptr || !stream.valid() ||
      query_tokens == 0 || query_tokens > key_tokens || heads == 0 ||
      head_dim != 128 || query_tokens > static_cast<std::size_t>(INT_MAX) ||
      key_tokens > static_cast<std::size_t>(INT_MAX) ||
      heads > static_cast<std::size_t>(INT_MAX)) {
    return {ErrorCode::kInvalidArgument,
            "Flash causal attention arguments are invalid"};
  }
  std::size_t query_elements = 0;
  std::size_t key_elements = 0;
  std::size_t query_bytes = 0;
  std::size_t key_bytes = 0;
  std::size_t lse_bytes = 0;
  if (!multiply(query_tokens, heads, &query_elements) ||
      !multiply(query_elements, head_dim, &query_elements) ||
      !multiply(key_tokens, heads, &key_elements) ||
      !multiply(key_elements, head_dim, &key_elements) ||
      !multiply(query_elements, sizeof(__nv_bfloat16), &query_bytes) ||
      !multiply(key_elements, sizeof(__nv_bfloat16), &key_bytes) ||
      !multiply(query_tokens, heads, &lse_bytes) ||
      !multiply(lse_bytes, sizeof(float), &lse_bytes)) {
    return {ErrorCode::kInvalidArgument,
            "Flash causal attention dimensions overflow"};
  }
  const int device = query.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "Flash causal attention stream is on a different device"};
  }
  const std::array<std::pair<const DeviceBuffer *, std::size_t>, 5> buffers{{
      {&query, query_bytes},
      {output, query_bytes},
      {&key, key_bytes},
      {&value, key_bytes},
      {softmax_lse, lse_bytes},
  }};
  for (const auto &item : buffers) {
    auto status = validate_buffer(*item.first, item.second, device,
                                  "Flash causal attention buffer");
    if (!status.ok())
      return status;
  }
  auto status = select_device(device);
  if (!status.ok())
    return status;

  cudaDeviceProp properties{};
  status = cuda_status(cudaGetDeviceProperties(&properties, device),
                       "query FlashAttention device properties");
  if (!status.ok())
    return status;
  const std::size_t m_blocks = (query_tokens + 63U) / 64U;
  const std::size_t n_blocks = (key_tokens + 127U) / 128U;
  if (m_blocks > static_cast<std::size_t>(INT_MAX) ||
      n_blocks > static_cast<std::size_t>(INT_MAX) ||
      heads > static_cast<std::size_t>(INT_MAX) / m_blocks) {
    return {ErrorCode::kInvalidArgument,
            "FlashAttention split heuristic dimensions overflow"};
  }
  const int num_splits = num_splits_heuristic(
      static_cast<int>(heads * m_blocks), properties.multiProcessorCount * 2,
      static_cast<int>(n_blocks), 128);

  if (num_splits > 1) {
    std::size_t accumulator_elements = 0;
    std::size_t accumulator_bytes = 0;
    if (!multiply(static_cast<std::size_t>(num_splits), heads,
                  &accumulator_elements) ||
        !multiply(accumulator_elements, query_tokens, &accumulator_elements) ||
        !multiply(accumulator_elements, sizeof(float), &accumulator_bytes)) {
      return {ErrorCode::kInvalidArgument,
              "FlashAttention LSE accumulator dimensions overflow"};
    }
    status = ensure_accumulator(softmax_lse_accum, device, accumulator_bytes,
                                "FlashAttention LSE accumulator");
    if (!status.ok())
      return status;
    if (!multiply(accumulator_elements, head_dim, &accumulator_elements) ||
        !multiply(accumulator_elements, sizeof(float), &accumulator_bytes)) {
      return {ErrorCode::kInvalidArgument,
              "FlashAttention output accumulator dimensions overflow"};
    }
    status = ensure_accumulator(output_accum, device, accumulator_bytes,
                                "FlashAttention output accumulator");
    if (!status.ok())
      return status;
  }

  FlashParams params{};
  params.q_ptr = const_cast<void *>(query.data());
  params.k_ptr = const_cast<void *>(key.data());
  params.v_ptr = const_cast<void *>(value.data());
  params.o_ptr = output->data();
  params.softmax_lse_ptr = softmax_lse->data();
  params.softmax_lseaccum_ptr =
      num_splits > 1 ? softmax_lse_accum->data() : nullptr;
  params.oaccum_ptr = num_splits > 1 ? output_accum->data() : nullptr;
  const auto token_stride = static_cast<FlashParams::index_t>(heads * head_dim);
  params.q_batch_stride =
      static_cast<FlashParams::index_t>(query_tokens) * token_stride;
  params.k_batch_stride =
      static_cast<FlashParams::index_t>(key_tokens) * token_stride;
  params.v_batch_stride = params.k_batch_stride;
  params.o_batch_stride = params.q_batch_stride;
  params.q_row_stride = token_stride;
  params.k_row_stride = token_stride;
  params.v_row_stride = token_stride;
  params.o_row_stride = token_stride;
  params.q_head_stride = static_cast<FlashParams::index_t>(head_dim);
  params.k_head_stride = static_cast<FlashParams::index_t>(head_dim);
  params.v_head_stride = static_cast<FlashParams::index_t>(head_dim);
  params.o_head_stride = static_cast<FlashParams::index_t>(head_dim);
  params.b = 1;
  params.h = static_cast<int>(heads);
  params.h_k = params.h;
  params.h_h_k_ratio = 1;
  params.seqlen_q = static_cast<int>(query_tokens);
  params.seqlen_k = static_cast<int>(key_tokens);
  params.seqlen_q_rounded = round_up_128(query_tokens);
  params.seqlen_k_rounded = round_up_128(key_tokens);
  params.d = static_cast<int>(head_dim);
  params.d_rounded = params.d;
  params.total_q = params.seqlen_q;
  params.scale_softmax =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  params.scale_softmax_log2 =
      static_cast<float>(static_cast<double>(params.scale_softmax) * M_LOG2E);
  params.p_dropout = 1.0F;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0F;
  params.scale_softmax_rp_dropout = params.scale_softmax;
  params.window_size_left = -1;
  params.window_size_right = 0;
  params.is_bf16 = true;
  params.is_causal = true;
  params.is_seqlens_k_cumulative = true;
  params.num_splits = num_splits;

  if (num_splits > 1) {
    constexpr std::size_t shared_bytes = SplitFlashTraits::kSmemSize;
    const dim3 grid(static_cast<unsigned int>((query_tokens + 63U) / 64U),
                    static_cast<unsigned int>(num_splits),
                    static_cast<unsigned int>(heads));
    const bool even_mn = query_tokens % 64U == 0 && key_tokens % 128U == 0;
    if (even_mn) {
      status = cuda_status(
          cudaFuncSetAttribute(flash_causal_split_kernel<true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shared_bytes)),
          "cudaFuncSetAttribute Flash split-KV attention");
      if (!status.ok())
        return status;
      flash_causal_split_kernel<true>
          <<<grid, SplitFlashTraits::kNThreads, shared_bytes, stream.get()>>>(
              params);
    } else {
      status = cuda_status(
          cudaFuncSetAttribute(flash_causal_split_kernel<false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shared_bytes)),
          "cudaFuncSetAttribute Flash split-KV attention");
      if (!status.ok())
        return status;
      flash_causal_split_kernel<false>
          <<<grid, SplitFlashTraits::kNThreads, shared_bytes, stream.get()>>>(
              params);
    }
    status = cuda_status(cudaPeekAtLastError(),
                         "PyTorch-compatible Flash split-KV kernel");
    if (!status.ok())
      return status;

    const dim3 combine_grid(
        static_cast<unsigned int>((heads * query_tokens + 3U) / 4U));
    if (num_splits <= 2) {
      flash_causal_combine_kernel<1>
          <<<combine_grid, SplitFlashTraits::kNThreads, 0, stream.get()>>>(
              params);
    } else if (num_splits <= 4) {
      flash_causal_combine_kernel<2>
          <<<combine_grid, SplitFlashTraits::kNThreads, 0, stream.get()>>>(
              params);
    } else if (num_splits <= 8) {
      flash_causal_combine_kernel<3>
          <<<combine_grid, SplitFlashTraits::kNThreads, 0, stream.get()>>>(
              params);
    } else if (num_splits <= 16) {
      flash_causal_combine_kernel<4>
          <<<combine_grid, SplitFlashTraits::kNThreads, 0, stream.get()>>>(
              params);
    } else if (num_splits <= 32) {
      flash_causal_combine_kernel<5>
          <<<combine_grid, SplitFlashTraits::kNThreads, 0, stream.get()>>>(
              params);
    } else if (num_splits <= 64) {
      flash_causal_combine_kernel<6>
          <<<combine_grid, SplitFlashTraits::kNThreads, 0, stream.get()>>>(
              params);
    } else {
      flash_causal_combine_kernel<7>
          <<<combine_grid, SplitFlashTraits::kNThreads, 0, stream.get()>>>(
              params);
    }
    return cuda_status(cudaPeekAtLastError(),
                       "PyTorch-compatible Flash split-KV combine kernel");
  }

  constexpr std::size_t shared_bytes = FlashTraits::kSmemSize;
  const dim3 grid(static_cast<unsigned int>((query_tokens + 127U) / 128U), 1,
                  static_cast<unsigned int>(heads));
  const bool even_mn = query_tokens % 128U == 0 && key_tokens % 64U == 0;
  if (even_mn) {
    status = cuda_status(
        cudaFuncSetAttribute(flash_causal_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "cudaFuncSetAttribute Flash causal attention");
    if (!status.ok())
      return status;
    flash_causal_kernel<true>
        <<<grid, FlashTraits::kNThreads, shared_bytes, stream.get()>>>(params);
  } else {
    status = cuda_status(
        cudaFuncSetAttribute(flash_causal_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shared_bytes)),
        "cudaFuncSetAttribute Flash causal attention");
    if (!status.ok())
      return status;
    flash_causal_kernel<false>
        <<<grid, FlashTraits::kNThreads, shared_bytes, stream.get()>>>(params);
  }
  return cuda_status(cudaPeekAtLastError(),
                     "PyTorch-compatible Flash causal attention kernel");
}

} // namespace evo::cuda
