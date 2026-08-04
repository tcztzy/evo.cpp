// SPDX-License-Identifier: Apache-2.0
#include "evo/cuda/attention.hpp"

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <math_constants.h>

namespace evo::cuda {
namespace {

constexpr int kAttentionThreads = 256;
constexpr int kElementwiseThreads = 256;

bool multiply(const std::size_t left, const std::size_t right,
              std::size_t *const result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *result = left * right;
  return true;
}

Status tensor_elements(const std::size_t tokens, const std::size_t heads,
                       const std::size_t head_dim,
                       std::size_t *const elements) {
  std::size_t rows = 0;
  if (tokens == 0 || heads == 0 || head_dim == 0 ||
      !multiply(tokens, heads, &rows) || !multiply(rows, head_dim, elements)) {
    return {ErrorCode::kInvalidArgument,
            "attention dimensions are zero or overflow"};
  }
  return Status::Ok();
}

Status bytes_for(const std::size_t elements, const std::size_t element_size,
                 std::size_t *const bytes, const char *const name) {
  if (elements == 0 || !multiply(elements, element_size, bytes)) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " size is zero or overflows"};
  }
  return Status::Ok();
}

Status buffer_size(const DeviceBuffer &buffer, const std::size_t bytes,
                   const int device, const char *const name) {
  if (!buffer.valid()) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " is not allocated"};
  }
  if (buffer.device() != device) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " is on a different CUDA device"};
  }
  if (buffer.bytes() < bytes) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " has " + std::to_string(buffer.bytes()) +
                " bytes; requires at least " + std::to_string(bytes)};
  }
  return Status::Ok();
}

Status ensure_capacity(DeviceBuffer *const buffer, const int device,
                       const std::size_t bytes, const char *const name) {
  if (buffer == nullptr || bytes == 0)
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " has invalid storage requirements"};
  if (buffer->valid() && buffer->device() == device &&
      buffer->bytes() >= bytes) {
    return Status::Ok();
  }
  std::size_t allocation_bytes = bytes;
  if (buffer->valid() && buffer->device() == device) {
    if (buffer->bytes() <= std::numeric_limits<std::size_t>::max() / 2) {
      const std::size_t doubled = buffer->bytes() * 2;
      if (doubled > allocation_bytes)
        allocation_bytes = doubled;
    }
  }
  buffer->reset();
  auto status = buffer->allocate(device, allocation_bytes);
  if (!status.ok())
    return {status.code(),
            std::string{"allocate "} + name + ": " + status.message()};
  return Status::Ok();
}

Status launch_status(const char *const operation) {
  return cuda_status(cudaPeekAtLastError(), operation);
}

unsigned int grid_for(const std::size_t elements) {
  const auto blocks =
      (elements + static_cast<std::size_t>(kElementwiseThreads) - 1) /
      static_cast<std::size_t>(kElementwiseThreads);
  return static_cast<unsigned int>(
      blocks < 65535 ? blocks : static_cast<std::size_t>(65535));
}

Status validate_cache(const KvCache &cache, const int device) {
  if (cache.capacity == 0 || cache.heads == 0 || cache.head_dim == 0 ||
      cache.length > cache.capacity || cache.device_id != device) {
    return {ErrorCode::kInvalidArgument, "KV cache metadata is invalid"};
  }
  if (cache.type != KvCacheType::kBF16 && cache.type != KvCacheType::kQ8Paged) {
    return {ErrorCode::kInvalidArgument, "KV cache type is invalid"};
  }
  if (cache.type == KvCacheType::kQ8Paged) {
    if (cache.page_tokens == 0 || cache.key.valid() || cache.value.valid()) {
      return {ErrorCode::kInvalidArgument,
              "paged Q8 KV cache metadata is invalid"};
    }
    const std::size_t page_count = 1 + (cache.capacity - 1) / cache.page_tokens;
    if (cache.q8_pages.size() != page_count ||
        cache.host_key_pages.size() != page_count ||
        cache.host_value_pages.size() != page_count ||
        cache.host_key_scale_pages.size() != page_count ||
        cache.host_value_scale_pages.size() != page_count) {
      return {ErrorCode::kInvalidArgument,
              "paged Q8 KV cache page count is invalid"};
    }
    std::size_t page_elements = 0;
    std::size_t scale_elements = 0;
    std::size_t page_bytes = 0;
    std::size_t scale_bytes = 0;
    std::size_t table_bytes = 0;
    if (!multiply(cache.page_tokens, cache.heads, &scale_elements) ||
        !multiply(scale_elements, cache.head_dim, &page_elements) ||
        !multiply(page_elements, sizeof(std::int8_t), &page_bytes) ||
        !multiply(scale_elements, sizeof(float), &scale_bytes) ||
        !multiply(page_count, sizeof(void *), &table_bytes)) {
      return {ErrorCode::kInvalidArgument,
              "paged Q8 KV cache dimensions overflow"};
    }
    const std::size_t populated_pages =
        cache.length == 0 ? 0 : 1 + (cache.length - 1) / cache.page_tokens;
    for (std::size_t index = 0; index < page_count; ++index) {
      const auto &page = cache.q8_pages[index];
      const bool any_valid = page.key.valid() || page.value.valid() ||
                             page.key_scale.valid() || page.value_scale.valid();
      const bool all_valid = page.key.valid() && page.value.valid() &&
                             page.key_scale.valid() && page.value_scale.valid();
      if (any_valid != all_valid || (index < populated_pages && !all_valid)) {
        return {ErrorCode::kInvalidArgument,
                "paged Q8 KV physical page metadata is invalid"};
      }
      if (!all_valid) {
        if (cache.host_key_pages[index] != nullptr ||
            cache.host_value_pages[index] != nullptr ||
            cache.host_key_scale_pages[index] != nullptr ||
            cache.host_value_scale_pages[index] != nullptr) {
          return {ErrorCode::kInvalidArgument,
                  "unallocated Q8 KV page has non-null host metadata"};
        }
        continue;
      }
      for (const auto *const buffer : {&page.key, &page.value}) {
        auto status =
            buffer_size(*buffer, page_bytes, device, "paged Q8 KV payload");
        if (!status.ok())
          return status;
      }
      for (const auto *const buffer : {&page.key_scale, &page.value_scale}) {
        auto status =
            buffer_size(*buffer, scale_bytes, device, "paged Q8 KV scale");
        if (!status.ok())
          return status;
      }
      if (cache.host_key_pages[index] != page.key.data() ||
          cache.host_value_pages[index] != page.value.data() ||
          cache.host_key_scale_pages[index] != page.key_scale.data() ||
          cache.host_value_scale_pages[index] != page.value_scale.data()) {
        return {ErrorCode::kInvalidArgument,
                "paged Q8 KV host pointer metadata is invalid"};
      }
    }
    for (const auto *const table :
         {&cache.key_page_table, &cache.value_page_table,
          &cache.key_scale_page_table, &cache.value_scale_page_table}) {
      auto status =
          buffer_size(*table, table_bytes, device, "paged Q8 KV page table");
      if (!status.ok())
        return status;
    }
    return Status::Ok();
  }
  if (cache.page_tokens != 0 || !cache.q8_pages.empty()) {
    return {ErrorCode::kInvalidArgument, "BF16 KV cache metadata is invalid"};
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  auto status =
      tensor_elements(cache.capacity, cache.heads, cache.head_dim, &elements);
  if (!status.ok())
    return status;
  status = bytes_for(elements, sizeof(__nv_bfloat16), &bytes, "KV cache");
  if (!status.ok())
    return status;
  status = buffer_size(cache.key, bytes, device, "KV key cache");
  if (!status.ok())
    return status;
  return buffer_size(cache.value, bytes, device, "KV value cache");
}

__global__ void
split_qkv_kernel(const __nv_bfloat16 *const qkv, __nv_bfloat16 *const query,
                 __nv_bfloat16 *const key, __nv_bfloat16 *const value,
                 const std::size_t heads, const std::size_t head_dim,
                 const std::size_t token_width, const std::size_t elements,
                 const bool head_major) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t token = index / token_width;
    const std::size_t within = index % token_width;
    if (head_major) {
      const std::size_t head = within / head_dim;
      const std::size_t dimension = within % head_dim;
      const std::size_t source =
          token * token_width * 3 + head * 3 * head_dim + dimension;
      query[index] = qkv[source];
      key[index] = qkv[source + head_dim];
      value[index] = qkv[source + head_dim * 2];
    } else {
      const std::size_t source = token * token_width * 3 + within;
      query[index] = qkv[source];
      key[index] = qkv[source + token_width];
      value[index] = qkv[source + token_width * 2];
    }
  }
}

__global__ void
rope_kernel(__nv_bfloat16 *const query, __nv_bfloat16 *const key,
            const float *const inverse_frequency, const std::size_t tokens,
            const std::size_t heads, const std::size_t head_dim,
            const std::size_t position_offset, const float position_scale) {
  const std::size_t half = head_dim / 2;
  const std::size_t elements = tokens * heads * half;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t pair = index % half;
    const std::size_t row = index / half;
    const std::size_t token = row / heads;
    const std::size_t base = row * head_dim;
    const float position =
        static_cast<float>(position_offset + token) / position_scale;
    const float angle = position * inverse_frequency[pair];
    float sine = 0.0F;
    float cosine = 0.0F;
    sincosf(angle, &sine, &cosine);
    // Vortex materializes its RoPE cosine and sine caches in the QKV dtype
    // before the Triton rotation kernel reloads them as F32. Preserve that
    // BF16 rounding boundary instead of multiplying by the raw F32 trig
    // results.
    sine = __bfloat162float(__float2bfloat16_rn(sine));
    cosine = __bfloat162float(__float2bfloat16_rn(cosine));

    const float query_first = __bfloat162float(query[base + pair]);
    const float query_second = __bfloat162float(query[base + half + pair]);
    query[base + pair] =
        __float2bfloat16_rn(query_first * cosine - query_second * sine);
    query[base + half + pair] =
        __float2bfloat16_rn(query_second * cosine + query_first * sine);

    const float key_first = __bfloat162float(key[base + pair]);
    const float key_second = __bfloat162float(key[base + half + pair]);
    key[base + pair] =
        __float2bfloat16_rn(key_first * cosine - key_second * sine);
    key[base + half + pair] =
        __float2bfloat16_rn(key_second * cosine + key_first * sine);
  }
}

__global__ void cached_scale_key_kernel(const __nv_bfloat16 *const key,
                                        __nv_bfloat16 *const scaled_key,
                                        const std::size_t elements,
                                        const float scale) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    scaled_key[index] =
        __float2bfloat16_rn(__bfloat162float(key[index]) * scale);
  }
}

__global__ void cached_causal_mask_kernel(__nv_bfloat16 *const scores,
                                          const std::size_t query_tokens,
                                          const std::size_t key_tokens,
                                          const std::size_t heads) {
  const std::size_t elements = heads * query_tokens * key_tokens;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t within_head = index % (query_tokens * key_tokens);
    const std::size_t query = within_head / key_tokens;
    const std::size_t key = within_head % key_tokens;
    if (key > query + key_tokens - query_tokens)
      scores[index] = __float2bfloat16_rn(-10000.0F);
  }
}

template <int WarpBatch, int WarpSize>
__device__ __forceinline__ void cached_warp_max(float (&values)[WarpBatch]) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int batch = 0; batch < WarpBatch; ++batch) {
      const float other =
          __shfl_xor_sync(0xffffffffU, values[batch], offset, WarpSize);
      values[batch] = values[batch] < other ? other : values[batch];
    }
  }
}

template <int WarpBatch, int WarpSize>
__device__ __forceinline__ void cached_warp_sum(float (&values)[WarpBatch]) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int batch = 0; batch < WarpBatch; ++batch) {
      values[batch] +=
          __shfl_xor_sync(0xffffffffU, values[batch], offset, WarpSize);
    }
  }
}

template <int Log2Elements>
__global__ void cached_softmax_warp_kernel(__nv_bfloat16 *const output,
                                           const __nv_bfloat16 *const input,
                                           const int batch_size,
                                           const int stride,
                                           const int element_count) {
  constexpr int next_power_of_two = 1 << Log2Elements;
  constexpr int warp_size = next_power_of_two < 32 ? next_power_of_two : 32;
  constexpr int warp_iterations = next_power_of_two / warp_size;
  constexpr int warp_batch = next_power_of_two <= 128 ? 2 : 1;

  const int first_batch =
      (static_cast<int>(blockDim.y) * static_cast<int>(blockIdx.x) +
       static_cast<int>(threadIdx.y)) *
      warp_batch;
  int local_batches = batch_size - first_batch;
  if (local_batches > warp_batch)
    local_batches = warp_batch;
  const int local_index = static_cast<int>(threadIdx.x);
  const int base = first_batch * stride + local_index;

  float elements[warp_batch][warp_iterations];
#pragma unroll
  for (int batch = 0; batch < warp_batch; ++batch) {
    const int batch_elements = batch >= local_batches ? 0 : element_count;
#pragma unroll
    for (int iteration = 0; iteration < warp_iterations; ++iteration) {
      const int element = local_index + iteration * warp_size;
      elements[batch][iteration] =
          element < batch_elements
              ? __bfloat162float(
                    input[base + batch * element_count + iteration * warp_size])
              : -CUDART_INF_F;
    }
  }

  float maximum[warp_batch];
#pragma unroll
  for (int batch = 0; batch < warp_batch; ++batch) {
    maximum[batch] = elements[batch][0];
#pragma unroll
    for (int iteration = 0; iteration < warp_iterations; ++iteration) {
      maximum[batch] = maximum[batch] > elements[batch][iteration]
                           ? maximum[batch]
                           : elements[batch][iteration];
    }
  }
  cached_warp_max<warp_batch, warp_size>(maximum);

  float sum[warp_batch]{};
#pragma unroll
  for (int batch = 0; batch < warp_batch; ++batch) {
#pragma unroll
    for (int iteration = 0; iteration < warp_iterations; ++iteration) {
      elements[batch][iteration] =
          expf(elements[batch][iteration] - maximum[batch]);
      sum[batch] += elements[batch][iteration];
    }
  }
  cached_warp_sum<warp_batch, warp_size>(sum);

#pragma unroll
  for (int batch = 0; batch < warp_batch; ++batch) {
    if (batch >= local_batches)
      break;
#pragma unroll
    for (int iteration = 0; iteration < warp_iterations; ++iteration) {
      const int element = local_index + iteration * warp_size;
      if (element < element_count) {
        const float result = sum[batch] == 0.0F
                                 ? CUDART_NAN_F
                                 : elements[batch][iteration] / sum[batch];
        output[base + batch * element_count + iteration * warp_size] =
            __float2bfloat16_rn(result);
      }
    }
  }
}

template <int Log2Elements>
void launch_cached_softmax(__nv_bfloat16 *const output,
                           const __nv_bfloat16 *const input,
                           const int batch_size, const int element_count,
                           const cudaStream_t stream) {
  constexpr int next_power_of_two = 1 << Log2Elements;
  constexpr int warp_size = next_power_of_two < 32 ? next_power_of_two : 32;
  constexpr int batches_per_warp = next_power_of_two <= 128 ? 2 : 1;
  constexpr int threads_per_block = 128;
  constexpr int warps_per_block = threads_per_block / warp_size;
  constexpr int batches_per_block = warps_per_block * batches_per_warp;
  const int blocks = (batch_size + batches_per_block - 1) / batches_per_block;
  const dim3 threads(warp_size, warps_per_block, 1);
  cached_softmax_warp_kernel<Log2Elements><<<blocks, threads, 0, stream>>>(
      output, input, batch_size, element_count, element_count);
}

// PyTorch leaves its persistent warp softmax above 2048 elements.  The
// following kernels preserve the pinned SoftMax.cu launch geometry, per-thread
// accumulation order, vector alignment handling, and two-level warp reduction
// used by that larger contiguous-last-dimension path.
__device__ __forceinline__ float cached_block_max(const float left,
                                                  const float right) {
  return left < right ? right : left;
}

template <bool Maximum>
__device__ __forceinline__ float cached_block_reduce(float value,
                                                     float *const shared) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    const float other = __shfl_down_sync(0xffffffffU, value, offset, 32);
    if constexpr (Maximum)
      value = cached_block_max(value, other);
    else
      value += other;
  }
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) % 32;
  const int warp = static_cast<int>(threadIdx.x) / 32;
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();
  value = static_cast<int>(threadIdx.x) < static_cast<int>(blockDim.x) / 32
              ? shared[lane]
              : (Maximum ? -CUDART_INF_F : 0.0F);
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      const float other = __shfl_down_sync(0xffffffffU, value, offset, 32);
      if constexpr (Maximum)
        value = cached_block_max(value, other);
      else
        value += other;
    }
  }
  if (threadIdx.x == 0)
    shared[0] = value;
  __syncthreads();
  return shared[0];
}

template <int RegisterCount>
__global__ void cached_softmax_register_kernel(__nv_bfloat16 *const output,
                                               const __nv_bfloat16 *const input,
                                               const int columns) {
  extern __shared__ float reduction[];
  const auto *const row_input =
      input + static_cast<std::int64_t>(blockIdx.x) * columns;
  auto *const row_output =
      output + static_cast<std::int64_t>(blockIdx.x) * columns;
  __nv_bfloat16 values[RegisterCount];
  float thread_max = -CUDART_INF_F;
#pragma unroll
  for (int item = 0; item < RegisterCount; ++item) {
    const int column = static_cast<int>(threadIdx.x) + item * blockDim.x;
    if (column < columns) {
      values[item] = row_input[column];
      thread_max = cached_block_max(thread_max, __bfloat162float(values[item]));
    }
  }
  const float maximum = cached_block_reduce<true>(thread_max, reduction);
  float thread_sum = 0.0F;
#pragma unroll
  for (int item = 0; item < RegisterCount; ++item) {
    const int column = static_cast<int>(threadIdx.x) + item * blockDim.x;
    if (column < columns)
      thread_sum += expf(__bfloat162float(values[item]) - maximum);
  }
  const float sum = cached_block_reduce<false>(thread_sum, reduction);
#pragma unroll
  for (int item = 0; item < RegisterCount; ++item) {
    const int column = static_cast<int>(threadIdx.x) + item * blockDim.x;
    if (column < columns) {
      row_output[column] = __float2bfloat16_rn(
          expf(__bfloat162float(values[item]) - maximum) / sum);
    }
  }
}

struct alignas(16) CachedBf16Vector final {
  __nv_bfloat16 values[8];
};

__global__ void cached_softmax_shared_kernel(__nv_bfloat16 *const output,
                                             const __nv_bfloat16 *const input,
                                             const int columns) {
  extern __shared__ unsigned char storage[];
  auto *const cached = reinterpret_cast<__nv_bfloat16 *>(storage);
  auto *const reduction =
      reinterpret_cast<float *>(storage + columns * sizeof(__nv_bfloat16));
  const auto *const row_input =
      input + static_cast<std::int64_t>(blockIdx.x) * columns;
  auto *const row_output =
      output + static_cast<std::int64_t>(blockIdx.x) * columns;
  const auto *const vector_input =
      reinterpret_cast<const CachedBf16Vector *>(row_input);
  auto *const vector_cache = reinterpret_cast<CachedBf16Vector *>(cached);
  float thread_max = -CUDART_INF_F;
  for (int offset = static_cast<int>(threadIdx.x); offset * 8 < columns;
       offset += static_cast<int>(blockDim.x)) {
    const CachedBf16Vector current = vector_input[offset];
    vector_cache[offset] = current;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      thread_max =
          cached_block_max(thread_max, __bfloat162float(current.values[item]));
    }
  }
  const float maximum = cached_block_reduce<true>(thread_max, reduction);
  float thread_sum = 0.0F;
  for (int offset = static_cast<int>(threadIdx.x); offset * 8 < columns;
       offset += static_cast<int>(blockDim.x)) {
    const CachedBf16Vector current = vector_cache[offset];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      thread_sum += expf(__bfloat162float(current.values[item]) - maximum);
    }
  }
  const float sum = cached_block_reduce<false>(thread_sum, reduction);
  auto *const vector_output = reinterpret_cast<CachedBf16Vector *>(row_output);
  for (int offset = static_cast<int>(threadIdx.x); offset * 8 < columns;
       offset += static_cast<int>(blockDim.x)) {
    const CachedBf16Vector current = vector_cache[offset];
    CachedBf16Vector result{};
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      result.values[item] = __float2bfloat16_rn(
          expf(__bfloat162float(current.values[item]) - maximum) / sum);
    }
    vector_output[offset] = result;
  }
}

template <bool Maximum>
__device__ __forceinline__ float cached_ilp_reduce(const __nv_bfloat16 *input,
                                                   int columns,
                                                   const float maximum) {
  constexpr int kAlignmentBytes = 16;
  constexpr int kItems = 8;
  const int shift = static_cast<int>(reinterpret_cast<std::uintptr_t>(input) %
                                     kAlignmentBytes) /
                    static_cast<int>(sizeof(__nv_bfloat16));
  int offset = static_cast<int>(threadIdx.x);
  float result = Maximum ? -CUDART_INF_F : 0.0F;
  const auto accumulate = [&](const __nv_bfloat16 item) {
    const float value = __bfloat162float(item);
    if constexpr (Maximum)
      result = cached_block_max(result, value);
    else
      result += expf(value - maximum);
  };
  if (shift > 0) {
    input -= shift;
    columns += shift;
    if (offset >= shift && offset < columns)
      accumulate(input[offset]);
    columns -= blockDim.x > columns ? columns : static_cast<int>(blockDim.x);
    input += blockDim.x;
  }
  const int last = columns % (kItems * static_cast<int>(blockDim.x));
  const auto *const vectors = reinterpret_cast<const CachedBf16Vector *>(input);
  for (; offset * kItems < columns - last;
       offset += static_cast<int>(blockDim.x)) {
    const CachedBf16Vector current = vectors[offset];
#pragma unroll
    for (int item = 0; item < kItems; ++item)
      accumulate(current.values[item]);
  }
  offset = columns - last + static_cast<int>(threadIdx.x);
  for (; offset < columns; offset += static_cast<int>(blockDim.x))
    accumulate(input[offset]);
  return result;
}

__global__ void cached_softmax_general_kernel(__nv_bfloat16 *const output,
                                              const __nv_bfloat16 *const input,
                                              const int columns) {
  extern __shared__ float reduction[];
  const auto *const row_input =
      input + static_cast<std::int64_t>(blockIdx.x) * columns;
  auto *const row_output =
      output + static_cast<std::int64_t>(blockIdx.x) * columns;
  const float thread_max = cached_ilp_reduce<true>(row_input, columns, 0.0F);
  const float maximum = cached_block_reduce<true>(thread_max, reduction);
  const float thread_sum =
      cached_ilp_reduce<false>(row_input, columns, maximum);
  const float sum = cached_block_reduce<false>(thread_sum, reduction);

  constexpr int kAlignmentBytes = 16;
  constexpr int kItems = 8;
  const int input_shift =
      static_cast<int>(reinterpret_cast<std::uintptr_t>(row_input) %
                       kAlignmentBytes) /
      static_cast<int>(sizeof(__nv_bfloat16));
  const int output_shift =
      static_cast<int>(reinterpret_cast<std::uintptr_t>(row_output) %
                       kAlignmentBytes) /
      static_cast<int>(sizeof(__nv_bfloat16));
  if (input_shift != output_shift) {
    for (int column = static_cast<int>(threadIdx.x); column < columns;
         column += static_cast<int>(blockDim.x)) {
      row_output[column] = __float2bfloat16_rn(
          expf(__bfloat162float(row_input[column]) - maximum) / sum);
    }
    return;
  }

  const auto evaluate = [maximum, sum](const __nv_bfloat16 item) {
    return __float2bfloat16_rn(expf(__bfloat162float(item) - maximum) / sum);
  };
  auto *write_output = row_output;
  const auto *read_input = row_input;
  int size = columns;
  int offset = static_cast<int>(threadIdx.x);
  if (input_shift > 0) {
    read_input -= input_shift;
    write_output -= output_shift;
    size += input_shift;
    if (offset >= input_shift && offset < size)
      write_output[offset] = evaluate(read_input[offset]);
    size -= blockDim.x > size ? size : static_cast<int>(blockDim.x);
    read_input += blockDim.x;
    write_output += blockDim.x;
  }
  const int last = size % (kItems * static_cast<int>(blockDim.x));
  const auto *const vector_input =
      reinterpret_cast<const CachedBf16Vector *>(read_input);
  auto *const vector_output =
      reinterpret_cast<CachedBf16Vector *>(write_output);
  for (; offset * kItems < size - last;
       offset += static_cast<int>(blockDim.x)) {
    const CachedBf16Vector current = vector_input[offset];
    CachedBf16Vector result{};
#pragma unroll
    for (int item = 0; item < kItems; ++item)
      result.values[item] = evaluate(current.values[item]);
    vector_output[offset] = result;
  }
  offset = size - last + static_cast<int>(threadIdx.x);
  for (; offset < size; offset += static_cast<int>(blockDim.x))
    write_output[offset] = evaluate(read_input[offset]);
}

Status launch_cached_large_softmax(__nv_bfloat16 *const output,
                                   const __nv_bfloat16 *const input,
                                   const int rows, const int columns,
                                   const int device,
                                   const cudaStream_t stream) {
  constexpr int kThreads = 1024;
  constexpr int kItems = 8;
  const int register_count = (columns + kThreads - 1) / kThreads;
  const std::size_t reduction_bytes = (kThreads / 32) * sizeof(float);
  if (register_count < 10) {
#define EVO_REGISTER_SOFTMAX_CASE(value)                                     \
  case value:                                                                  \
    cached_softmax_register_kernel<value>                                      \
        <<<rows, kThreads, reduction_bytes, stream>>>(output, input, columns); \
    break
    switch (register_count) {
      EVO_REGISTER_SOFTMAX_CASE(3);
      EVO_REGISTER_SOFTMAX_CASE(4);
      EVO_REGISTER_SOFTMAX_CASE(5);
      EVO_REGISTER_SOFTMAX_CASE(6);
      EVO_REGISTER_SOFTMAX_CASE(7);
      EVO_REGISTER_SOFTMAX_CASE(8);
      EVO_REGISTER_SOFTMAX_CASE(9);
    default:
      return {ErrorCode::kUnsupported,
              "cached register softmax dispatch is unavailable"};
    }
#undef EVO_REGISTER_SOFTMAX_CASE
    return launch_status("PyTorch-compatible cached register softmax");
  }

  cudaDeviceProp properties{};
  auto status = cuda_status(cudaGetDeviceProperties(&properties, device),
                            "cudaGetDeviceProperties cached softmax");
  if (!status.ok())
    return status;
  const auto input_address = reinterpret_cast<std::uintptr_t>(input);
  const auto output_address = reinterpret_cast<std::uintptr_t>(output);
  const std::size_t shared_capacity_bytes =
      static_cast<std::size_t>(properties.sharedMemPerBlock);
  const std::size_t maximum_shared_elements =
      shared_capacity_bytes > reduction_bytes
          ? (shared_capacity_bytes - reduction_bytes) / sizeof(__nv_bfloat16)
          : 0;
  const bool use_shared =
      static_cast<std::size_t>(columns) < maximum_shared_elements &&
      input_address % 16 == 0 && output_address % 16 == 0 &&
      columns % kItems == 0;
  if (use_shared) {
    const std::size_t shared_bytes =
        static_cast<std::size_t>(columns) * sizeof(__nv_bfloat16) +
        reduction_bytes;
    cached_softmax_shared_kernel<<<rows, kThreads, shared_bytes, stream>>>(
        output, input, columns);
    return launch_status("PyTorch-compatible cached shared softmax");
  }
  cached_softmax_general_kernel<<<rows, kThreads, reduction_bytes, stream>>>(
      output, input, columns);
  return launch_status("PyTorch-compatible cached general softmax");
}

Status dispatch_cached_softmax(DeviceBuffer *const output,
                               const DeviceBuffer &input,
                               const std::size_t rows,
                               const std::size_t columns,
                               const Stream &stream) {
  if (columns == 0 || columns > static_cast<std::size_t>(INT_MAX) ||
      rows > static_cast<std::size_t>(INT_MAX))
    return {ErrorCode::kUnsupported,
            "cached PyTorch softmax dimensions exceed CUDA indexing"};
  auto *const destination = static_cast<__nv_bfloat16 *>(output->data());
  const auto *const source = static_cast<const __nv_bfloat16 *>(input.data());
  if (columns > 2048) {
    return launch_cached_large_softmax(
        destination, source, static_cast<int>(rows), static_cast<int>(columns),
        input.device(), stream.get());
  }
  int log2_elements = 0;
  while ((1ULL << log2_elements) < columns)
    ++log2_elements;
#define EVO_SOFTMAX_CASE(value)                                              \
  case value:                                                                  \
    launch_cached_softmax<value>(destination, source, static_cast<int>(rows),  \
                                 static_cast<int>(columns), stream.get());     \
    break
  switch (log2_elements) {
    EVO_SOFTMAX_CASE(0);
    EVO_SOFTMAX_CASE(1);
    EVO_SOFTMAX_CASE(2);
    EVO_SOFTMAX_CASE(3);
    EVO_SOFTMAX_CASE(4);
    EVO_SOFTMAX_CASE(5);
    EVO_SOFTMAX_CASE(6);
    EVO_SOFTMAX_CASE(7);
    EVO_SOFTMAX_CASE(8);
    EVO_SOFTMAX_CASE(9);
    EVO_SOFTMAX_CASE(10);
    EVO_SOFTMAX_CASE(11);
  default:
    return {ErrorCode::kUnsupported, "cached softmax dispatch is unavailable"};
  }
#undef EVO_SOFTMAX_CASE
  return launch_status("PyTorch-compatible cached softmax kernel");
}

__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

__inline__ __device__ float warp_max(float value) {
  for (int offset = 16; offset > 0; offset /= 2)
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  return value;
}

__global__ void q8_kv_append_kernel(
    const __nv_bfloat16 *const key, const __nv_bfloat16 *const value,
    std::int8_t *const *const key_pages, std::int8_t *const *const value_pages,
    float *const *const key_scale_pages, float *const *const value_scale_pages,
    const std::size_t token_offset, const std::size_t tokens,
    const std::size_t heads, const std::size_t head_dim,
    const std::size_t page_tokens) {
  const std::size_t token_head = blockIdx.x;
  const std::size_t token = token_head / heads;
  const std::size_t head = token_head % heads;
  if (token >= tokens)
    return;

  const std::size_t dimension = threadIdx.x;
  const std::size_t input_base = token_head * head_dim;
  const float key_value = dimension < head_dim
                              ? __bfloat162float(key[input_base + dimension])
                              : 0.0F;
  const float value_value =
      dimension < head_dim ? __bfloat162float(value[input_base + dimension])
                           : 0.0F;
  float key_maximum = warp_max(fabsf(key_value));
  float value_maximum = warp_max(fabsf(value_value));
  __shared__ float partial_key[32];
  __shared__ float partial_value[32];
  __shared__ float key_scale;
  __shared__ float value_scale;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int warp_count = (blockDim.x + 31) / 32;
  if (lane == 0) {
    partial_key[warp] = key_maximum;
    partial_value[warp] = value_maximum;
  }
  __syncthreads();
  if (warp == 0) {
    key_maximum = lane < warp_count ? partial_key[lane] : 0.0F;
    value_maximum = lane < warp_count ? partial_value[lane] : 0.0F;
    key_maximum = warp_max(key_maximum);
    value_maximum = warp_max(value_maximum);
    if (lane == 0) {
      key_scale = key_maximum == 0.0F ? 1.0F : key_maximum / 127.0F;
      value_scale = value_maximum == 0.0F ? 1.0F : value_maximum / 127.0F;
    }
  }
  __syncthreads();

  const std::size_t position = token_offset + token;
  const std::size_t page = position / page_tokens;
  const std::size_t within_page = position % page_tokens;
  const std::size_t scale_index = within_page * heads + head;
  const std::size_t output_base = scale_index * head_dim;
  if (dimension == 0) {
    key_scale_pages[page][scale_index] = key_scale;
    value_scale_pages[page][scale_index] = value_scale;
  }
  if (dimension < head_dim) {
    int key_quantized = __float2int_rn(key_value / key_scale);
    int value_quantized = __float2int_rn(value_value / value_scale);
    key_quantized = max(-127, min(127, key_quantized));
    value_quantized = max(-127, min(127, value_quantized));
    key_pages[page][output_base + dimension] =
        static_cast<std::int8_t>(key_quantized);
    value_pages[page][output_base + dimension] =
        static_cast<std::int8_t>(value_quantized);
  }
}

__global__ void set_q8_page_kernel(
    std::int8_t **const key_pages, std::int8_t **const value_pages,
    float **const key_scale_pages, float **const value_scale_pages,
    const std::size_t index, std::int8_t *const key, std::int8_t *const value,
    float *const key_scale, float *const value_scale) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    key_pages[index] = key;
    value_pages[index] = value;
    key_scale_pages[index] = key_scale;
    value_scale_pages[index] = value_scale;
  }
}

__global__ void online_attention_kernel(
    const __nv_bfloat16 *const query, const __nv_bfloat16 *const key_cache,
    const __nv_bfloat16 *const value_cache, __nv_bfloat16 *const output,
    const std::size_t query_tokens, const std::size_t prefix_tokens,
    const std::size_t heads, const std::size_t head_dim) {
  const std::size_t query_head = blockIdx.x;
  const std::size_t query_token = query_head / heads;
  const std::size_t head = query_head % heads;
  if (query_token >= query_tokens)
    return;

  const std::size_t query_base = (query_token * heads + head) * head_dim;
  const std::size_t dimension = threadIdx.x;
  constexpr int source_warps = kAttentionThreads / 32;
  __shared__ float shared_query[256];
  __shared__ float shared_scores[source_warps];
  __shared__ float shared_weights[source_warps];
  __shared__ float shared_maximum;
  __shared__ float shared_normalizer;
  __shared__ float shared_old_scale;
  if (dimension < head_dim)
    shared_query[dimension] = __bfloat162float(query[query_base + dimension]);
  if (dimension == 0) {
    shared_maximum = -CUDART_INF_F;
    shared_normalizer = 0.0F;
  }
  __syncthreads();

  float accumulator = 0.0F;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const std::size_t sources = prefix_tokens + query_token + 1;

  for (std::size_t source_base = 0; source_base < sources;
       source_base += source_warps) {
    const std::size_t source = source_base + static_cast<std::size_t>(warp);
    float dot = 0.0F;
    if (source < sources) {
      const std::size_t cache_base = (source * heads + head) * head_dim;
      for (std::size_t component = static_cast<std::size_t>(lane);
           component < head_dim; component += 32) {
        dot += shared_query[component] *
               __bfloat162float(key_cache[cache_base + component]);
      }
    }
    dot = warp_sum(dot);
    if (lane == 0)
      shared_scores[warp] = source < sources ? dot * scale : -CUDART_INF_F;
    __syncthreads();

    const std::size_t valid_sources =
        min(static_cast<std::size_t>(source_warps), sources - source_base);
    if (dimension == 0) {
      float next_maximum = shared_maximum;
      for (std::size_t offset = 0; offset < valid_sources; ++offset)
        next_maximum = fmaxf(next_maximum, shared_scores[offset]);
      shared_old_scale = shared_normalizer == 0.0F
                             ? 0.0F
                             : expf(shared_maximum - next_maximum);
      float next_normalizer = shared_normalizer * shared_old_scale;
      for (std::size_t offset = 0; offset < valid_sources; ++offset) {
        const float weight = expf(shared_scores[offset] - next_maximum);
        shared_weights[offset] = weight;
        next_normalizer += weight;
      }
      shared_maximum = next_maximum;
      shared_normalizer = next_normalizer;
    }
    __syncthreads();

    if (dimension < head_dim) {
      accumulator *= shared_old_scale;
      for (std::size_t offset = 0; offset < valid_sources; ++offset) {
        const std::size_t value_base =
            ((source_base + offset) * heads + head) * head_dim;
        accumulator += shared_weights[offset] *
                       __bfloat162float(value_cache[value_base + dimension]);
      }
    }
    __syncthreads();
  }
  if (dimension < head_dim) {
    output[query_base + dimension] =
        __float2bfloat16_rn(accumulator / shared_normalizer);
  }
}

// One warp owns one query while all eight warps reuse the same K/V tile. This
// keeps online-softmax storage linear in sequence length without materializing
// an attention matrix.
__global__ void bf16_causal_prefill_kernel(
    const __nv_bfloat16 *const query, const __nv_bfloat16 *const key_cache,
    const __nv_bfloat16 *const value_cache, __nv_bfloat16 *const output,
    const std::size_t query_tokens, const std::size_t prefix_tokens,
    const std::size_t heads) {
  constexpr std::size_t head_dim = 128;
  constexpr std::size_t queries_per_block = kAttentionThreads / 32;
  constexpr std::size_t sources_per_tile = 8;
  __shared__ float shared_key[sources_per_tile * head_dim];
  __shared__ float shared_value[sources_per_tile * head_dim];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const std::size_t query_tile = blockIdx.x / heads;
  const std::size_t head = blockIdx.x % heads;
  const std::size_t query_token =
      query_tile * queries_per_block + static_cast<std::size_t>(warp);
  const bool active_query = query_token < query_tokens;
  const std::size_t query_base =
      active_query ? (query_token * heads + head) * head_dim : 0;
  float query_values[4]{};
  float accumulators[4]{};
  float maximum = -CUDART_INF_F;
  float normalizer = 0.0F;
  if (active_query) {
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      const std::size_t dimension =
          static_cast<std::size_t>(lane + component * 32);
      query_values[component] = __bfloat162float(query[query_base + dimension]);
    }
  }
  constexpr float scale = 0.08838834764831845F;
  const std::size_t tile_query_end =
      min((query_tile + 1) * queries_per_block, query_tokens);
  const std::size_t maximum_sources = prefix_tokens + tile_query_end;

  for (std::size_t source_base = 0; source_base < maximum_sources;
       source_base += sources_per_tile) {
    const std::size_t valid_sources =
        min(sources_per_tile, maximum_sources - source_base);
    const std::size_t tile_elements = valid_sources * head_dim;
    for (std::size_t index = threadIdx.x; index < tile_elements;
         index += blockDim.x) {
      const std::size_t source_offset = index / head_dim;
      const std::size_t dimension = index % head_dim;
      const std::size_t cache_base =
          ((source_base + source_offset) * heads + head) * head_dim;
      shared_key[index] = __bfloat162float(key_cache[cache_base + dimension]);
      shared_value[index] =
          __bfloat162float(value_cache[cache_base + dimension]);
    }
    __syncthreads();

    if (active_query) {
      const std::size_t query_sources = prefix_tokens + query_token + 1;
      const std::size_t usable_sources =
          source_base >= query_sources
              ? 0
              : min(valid_sources, query_sources - source_base);
      for (std::size_t source_offset = 0; source_offset < usable_sources;
           ++source_offset) {
        const std::size_t shared_base = source_offset * head_dim;
        float dot = 0.0F;
#pragma unroll
        for (int component = 0; component < 4; ++component) {
          const std::size_t dimension =
              static_cast<std::size_t>(lane + component * 32);
          dot += query_values[component] * shared_key[shared_base + dimension];
        }
        dot = warp_sum(dot);
        const float score = __shfl_sync(0xffffffffU, dot * scale, 0);
        const float next_maximum = fmaxf(maximum, score);
        const float old_weight =
            normalizer == 0.0F ? 0.0F : expf(maximum - next_maximum);
        const float new_weight = expf(score - next_maximum);
#pragma unroll
        for (int component = 0; component < 4; ++component) {
          const std::size_t dimension =
              static_cast<std::size_t>(lane + component * 32);
          accumulators[component] =
              accumulators[component] * old_weight +
              new_weight * shared_value[shared_base + dimension];
        }
        normalizer = normalizer * old_weight + new_weight;
        maximum = next_maximum;
      }
    }
    __syncthreads();
  }

  if (active_query) {
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      const std::size_t dimension =
          static_cast<std::size_t>(lane + component * 32);
      output[query_base + dimension] =
          __float2bfloat16_rn(accumulators[component] / normalizer);
    }
  }
}

__global__ void q8_online_attention_kernel(
    const __nv_bfloat16 *const query, const std::int8_t *const *const key_pages,
    const std::int8_t *const *const value_pages,
    const float *const *const key_scale_pages,
    const float *const *const value_scale_pages, __nv_bfloat16 *const output,
    const std::size_t query_tokens, const std::size_t prefix_tokens,
    const std::size_t heads, const std::size_t head_dim,
    const std::size_t page_tokens) {
  const std::size_t query_head = blockIdx.x;
  const std::size_t query_token = query_head / heads;
  const std::size_t head = query_head % heads;
  if (query_token >= query_tokens)
    return;

  const std::size_t dimension = threadIdx.x;
  const std::size_t query_base = (query_token * heads + head) * head_dim;
  const float query_value =
      dimension < head_dim ? __bfloat162float(query[query_base + dimension])
                           : 0.0F;
  float accumulator = 0.0F;
  float maximum = -CUDART_INF_F;
  float normalizer = 0.0F;
  const float attention_scale = rsqrtf(static_cast<float>(head_dim));
  __shared__ float partial[32];
  __shared__ float shared_score;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int warp_count = (blockDim.x + 31) / 32;
  const std::size_t sources = prefix_tokens + query_token + 1;

  for (std::size_t source = 0; source < sources; ++source) {
    const std::size_t page = source / page_tokens;
    const std::size_t within_page = source % page_tokens;
    const std::size_t scale_index = within_page * heads + head;
    const std::size_t cache_base = scale_index * head_dim;
    const float key_scale = key_scale_pages[page][scale_index];
    const float value_scale = value_scale_pages[page][scale_index];
    float dot =
        dimension < head_dim
            ? query_value *
                  (static_cast<float>(key_pages[page][cache_base + dimension]) *
                   key_scale)
            : 0.0F;
    dot = warp_sum(dot);
    if (lane == 0)
      partial[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float total = lane < warp_count ? partial[lane] : 0.0F;
      total = warp_sum(total);
      if (lane == 0)
        shared_score = total * attention_scale;
    }
    __syncthreads();

    const float score = shared_score;
    const float next_maximum = fmaxf(maximum, score);
    const float old_weight =
        normalizer == 0.0F ? 0.0F : expf(maximum - next_maximum);
    const float new_weight = expf(score - next_maximum);
    if (dimension < head_dim) {
      const float cached_value =
          static_cast<float>(value_pages[page][cache_base + dimension]) *
          value_scale;
      accumulator = accumulator * old_weight + new_weight * cached_value;
    }
    normalizer = normalizer * old_weight + new_weight;
    maximum = next_maximum;
  }
  if (dimension < head_dim) {
    output[query_base + dimension] =
        __float2bfloat16_rn(accumulator / normalizer);
  }
}

} // namespace

Status KvCache::allocate(const int device, const std::size_t token_capacity,
                         const std::size_t head_count,
                         const std::size_t dimensions_per_head) {
  if (key.valid() || value.valid() || !q8_pages.empty()) {
    return {ErrorCode::kInvalidArgument, "KV cache is already allocated"};
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  auto status = tensor_elements(token_capacity, head_count, dimensions_per_head,
                                &elements);
  if (!status.ok())
    return status;
  status = bytes_for(elements, sizeof(__nv_bfloat16), &bytes, "KV cache");
  if (!status.ok())
    return status;
  status = key.allocate(device, bytes);
  if (!status.ok())
    return status;
  status = value.allocate(device, bytes);
  if (!status.ok()) {
    key.reset();
    return status;
  }
  type = KvCacheType::kBF16;
  device_id = device;
  capacity = token_capacity;
  length = 0;
  heads = head_count;
  head_dim = dimensions_per_head;
  page_tokens = 0;
  return Status::Ok();
}

Status KvCache::allocate_q8_paged(const int device,
                                  const std::size_t token_capacity,
                                  const std::size_t head_count,
                                  const std::size_t dimensions_per_head,
                                  const std::size_t tokens_per_page,
                                  const Stream &stream) {
  if (key.valid() || value.valid() || !q8_pages.empty() || !stream.valid() ||
      stream.device() != device || tokens_per_page == 0) {
    return {ErrorCode::kInvalidArgument,
            "paged Q8 KV cache allocation arguments are invalid"};
  }
  std::size_t page_elements = 0;
  std::size_t scale_elements = 0;
  std::size_t page_bytes = 0;
  std::size_t scale_bytes = 0;
  auto status = tensor_elements(tokens_per_page, head_count,
                                dimensions_per_head, &page_elements);
  if (!status.ok())
    return status;
  if (!multiply(tokens_per_page, head_count, &scale_elements)) {
    return {ErrorCode::kInvalidArgument,
            "paged Q8 KV scale dimensions overflow"};
  }
  status = bytes_for(page_elements, sizeof(std::int8_t), &page_bytes,
                     "paged Q8 KV payload");
  if (!status.ok())
    return status;
  status = bytes_for(scale_elements, sizeof(float), &scale_bytes,
                     "paged Q8 KV scale");
  if (!status.ok())
    return status;
  if (token_capacity == 0) {
    return {ErrorCode::kInvalidArgument, "paged Q8 KV token capacity is zero"};
  }
  const std::size_t page_count = 1 + (token_capacity - 1) / tokens_per_page;
  std::size_t table_bytes = 0;
  if (!multiply(page_count, sizeof(void *), &table_bytes)) {
    return {ErrorCode::kInvalidArgument,
            "paged Q8 KV page table dimensions overflow"};
  }

  q8_pages.resize(page_count);
  host_key_pages.assign(page_count, nullptr);
  host_value_pages.assign(page_count, nullptr);
  host_key_scale_pages.assign(page_count, nullptr);
  host_value_scale_pages.assign(page_count, nullptr);
  for (auto *const table : {&key_page_table, &value_page_table,
                            &key_scale_page_table, &value_scale_page_table}) {
    status = table->allocate(device, table_bytes);
    if (!status.ok())
      return status;
    status = table->zero(stream);
    if (!status.ok())
      return status;
  }

  type = KvCacheType::kQ8Paged;
  device_id = device;
  capacity = token_capacity;
  length = 0;
  heads = head_count;
  head_dim = dimensions_per_head;
  page_tokens = tokens_per_page;
  return Status::Ok();
}

Status ensure_q8_pages(KvCache *const cache, const std::size_t token_end,
                       const Stream &stream) {
  if (cache == nullptr || cache->type != KvCacheType::kQ8Paged ||
      token_end == 0 || token_end > cache->capacity ||
      stream.device() != cache->device_id) {
    return {ErrorCode::kInvalidArgument,
            "paged Q8 KV physical allocation arguments are invalid"};
  }
  std::size_t scale_elements = 0;
  std::size_t page_elements = 0;
  std::size_t page_bytes = 0;
  std::size_t scale_bytes = 0;
  if (!multiply(cache->page_tokens, cache->heads, &scale_elements) ||
      !multiply(scale_elements, cache->head_dim, &page_elements) ||
      !multiply(page_elements, sizeof(std::int8_t), &page_bytes) ||
      !multiply(scale_elements, sizeof(float), &scale_bytes)) {
    return {ErrorCode::kInvalidArgument,
            "paged Q8 KV physical page dimensions overflow"};
  }
  const std::size_t required_pages = 1 + (token_end - 1) / cache->page_tokens;
  for (std::size_t index = 0; index < required_pages; ++index) {
    if (cache->q8_pages[index].key.valid())
      continue;
    Q8KvPage candidate;
    auto status = candidate.key.allocate(cache->device_id, page_bytes);
    if (!status.ok())
      return status;
    status = candidate.value.allocate(cache->device_id, page_bytes);
    if (!status.ok())
      return status;
    status = candidate.key_scale.allocate(cache->device_id, scale_bytes);
    if (!status.ok())
      return status;
    status = candidate.value_scale.allocate(cache->device_id, scale_bytes);
    if (!status.ok())
      return status;
    auto *const key = static_cast<std::int8_t *>(candidate.key.data());
    auto *const value = static_cast<std::int8_t *>(candidate.value.data());
    auto *const key_scale = static_cast<float *>(candidate.key_scale.data());
    auto *const value_scale =
        static_cast<float *>(candidate.value_scale.data());
    set_q8_page_kernel<<<1, 1, 0, stream.get()>>>(
        static_cast<std::int8_t **>(cache->key_page_table.data()),
        static_cast<std::int8_t **>(cache->value_page_table.data()),
        static_cast<float **>(cache->key_scale_page_table.data()),
        static_cast<float **>(cache->value_scale_page_table.data()), index, key,
        value, key_scale, value_scale);
    status = launch_status("set paged Q8 KV page pointer kernel");
    if (!status.ok())
      return status;
    cache->q8_pages[index] = std::move(candidate);
    cache->host_key_pages[index] = key;
    cache->host_value_pages[index] = value;
    cache->host_key_scale_pages[index] = key_scale;
    cache->host_value_scale_pages[index] = value_scale;
  }
  return Status::Ok();
}

std::size_t KvCache::allocated_bytes() const noexcept {
  std::size_t total = key.bytes() + value.bytes() + key_page_table.bytes() +
                      value_page_table.bytes() + key_scale_page_table.bytes() +
                      value_scale_page_table.bytes();
  for (const auto &page : q8_pages) {
    total += page.key.bytes() + page.value.bytes() + page.key_scale.bytes() +
             page.value_scale.bytes();
  }
  return total;
}

Status AttentionWorkspace::allocate(const int device,
                                    const std::size_t token_count,
                                    const std::size_t head_count,
                                    const std::size_t dimensions_per_head) {
  if (query.valid() || key.valid() || value.valid() || softmax_lse.valid() ||
      softmax_lse_accum.valid() || output_accum.valid() || scaled_key.valid() ||
      scores.valid() || probabilities.valid()) {
    return {ErrorCode::kInvalidArgument,
            "attention workspace is already allocated"};
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  auto status =
      tensor_elements(token_count, head_count, dimensions_per_head, &elements);
  if (!status.ok())
    return status;
  status =
      bytes_for(elements, sizeof(__nv_bfloat16), &bytes, "attention workspace");
  if (!status.ok())
    return status;
  status = query.allocate(device, bytes);
  if (!status.ok())
    return status;
  status = key.allocate(device, bytes);
  if (!status.ok()) {
    query.reset();
    return status;
  }
  status = value.allocate(device, bytes);
  if (!status.ok()) {
    key.reset();
    query.reset();
    return status;
  }
  std::size_t lse_elements = 0;
  std::size_t lse_bytes = 0;
  if (!multiply(token_count, head_count, &lse_elements) ||
      !multiply(lse_elements, sizeof(float), &lse_bytes)) {
    value.reset();
    key.reset();
    query.reset();
    return {ErrorCode::kInvalidArgument,
            "attention softmax workspace dimensions overflow"};
  }
  status = softmax_lse.allocate(device, lse_bytes);
  if (!status.ok()) {
    value.reset();
    key.reset();
    query.reset();
    return status;
  }
  tokens = token_count;
  heads = head_count;
  head_dim = dimensions_per_head;
  return Status::Ok();
}

Status bf16_split_qkv(const DeviceBuffer &qkv, const std::size_t tokens,
                      const std::size_t heads, const std::size_t head_dim,
                      DeviceBuffer *const query, DeviceBuffer *const key,
                      DeviceBuffer *const value, const Stream &stream,
                      const QkvLayout layout) {
  if (query == nullptr || key == nullptr || value == nullptr ||
      !stream.valid() ||
      (layout != QkvLayout::kProjectionMajor &&
       layout != QkvLayout::kHeadMajor)) {
    return {ErrorCode::kInvalidArgument,
            "QKV split requires three outputs and an initialized stream"};
  }
  std::size_t elements = 0;
  std::size_t qkv_elements = 0;
  std::size_t tensor_bytes = 0;
  std::size_t qkv_bytes = 0;
  auto status = tensor_elements(tokens, heads, head_dim, &elements);
  if (!status.ok())
    return status;
  if (!multiply(elements, 3, &qkv_elements)) {
    return {ErrorCode::kInvalidArgument, "QKV dimensions overflow"};
  }
  status = bytes_for(elements, sizeof(__nv_bfloat16), &tensor_bytes,
                     "QKV split tensor");
  if (!status.ok())
    return status;
  status = bytes_for(qkv_elements, sizeof(__nv_bfloat16), &qkv_bytes, "QKV");
  if (!status.ok())
    return status;
  const int device = qkv.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "QKV split stream is on a different CUDA device"};
  }
  status = buffer_size(qkv, qkv_bytes, device, "QKV input");
  if (!status.ok())
    return status;
  for (const auto *const output : {query, key, value}) {
    status = buffer_size(*output, tensor_bytes, device, "QKV split output");
    if (!status.ok())
      return status;
  }
  status = select_device(device);
  if (!status.ok())
    return status;
  std::size_t token_width = 0;
  if (!multiply(heads, head_dim, &token_width)) {
    return {ErrorCode::kInvalidArgument, "QKV token width overflows"};
  }
  split_qkv_kernel<<<grid_for(elements), kElementwiseThreads, 0,
                     stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(qkv.data()),
      static_cast<__nv_bfloat16 *>(query->data()),
      static_cast<__nv_bfloat16 *>(key->data()),
      static_cast<__nv_bfloat16 *>(value->data()), heads, head_dim, token_width,
      elements, layout == QkvLayout::kHeadMajor);
  return launch_status("split QKV kernel");
}

Status bf16_apply_rope(DeviceBuffer *const query, DeviceBuffer *const key,
                       const DeviceBuffer &inverse_frequency,
                       const std::size_t tokens, const std::size_t heads,
                       const std::size_t head_dim,
                       const std::size_t position_offset,
                       const float position_scale, const Stream &stream) {
  if (query == nullptr || key == nullptr || !stream.valid() || head_dim == 0 ||
      head_dim % 2 != 0 || !std::isfinite(position_scale) ||
      position_scale <= 0.0F ||
      (tokens != 0 &&
       position_offset >
           std::numeric_limits<std::size_t>::max() - tokens + 1)) {
    return {ErrorCode::kInvalidArgument, "RoPE arguments are invalid"};
  }
  std::size_t elements = 0;
  std::size_t tensor_bytes = 0;
  std::size_t frequency_bytes = 0;
  auto status = tensor_elements(tokens, heads, head_dim, &elements);
  if (!status.ok())
    return status;
  status =
      bytes_for(elements, sizeof(__nv_bfloat16), &tensor_bytes, "RoPE tensor");
  if (!status.ok())
    return status;
  status = bytes_for(head_dim / 2, sizeof(float), &frequency_bytes,
                     "RoPE inverse frequency");
  if (!status.ok())
    return status;
  const int device = query->device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "RoPE stream is on a different CUDA device"};
  }
  status = buffer_size(*query, tensor_bytes, device, "RoPE query");
  if (!status.ok())
    return status;
  status = buffer_size(*key, tensor_bytes, device, "RoPE key");
  if (!status.ok())
    return status;
  status = buffer_size(inverse_frequency, frequency_bytes, device,
                       "RoPE inverse frequency");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  std::size_t pairs = 0;
  if (!multiply(tokens, heads, &pairs) ||
      !multiply(pairs, head_dim / 2, &pairs)) {
    return {ErrorCode::kInvalidArgument, "RoPE dimensions overflow"};
  }
  rope_kernel<<<grid_for(pairs), kElementwiseThreads, 0, stream.get()>>>(
      static_cast<__nv_bfloat16 *>(query->data()),
      static_cast<__nv_bfloat16 *>(key->data()),
      static_cast<const float *>(inverse_frequency.data()), tokens, heads,
      head_dim, position_offset, position_scale);
  return launch_status("RoPE kernel");
}

Status bf16_kv_append(const DeviceBuffer &key, const DeviceBuffer &value,
                      const std::size_t tokens, KvCache *const cache,
                      const Stream &stream) {
  if (cache == nullptr || !stream.valid() || tokens == 0) {
    return {ErrorCode::kInvalidArgument, "KV append arguments are invalid"};
  }
  const int device = cache->device_id;
  auto status = validate_cache(*cache, device);
  if (!status.ok())
    return status;
  if (stream.device() != device || key.device() != device ||
      value.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "KV append buffers and stream must share one CUDA device"};
  }
  if (tokens > cache->capacity - cache->length) {
    return {ErrorCode::kInvalidArgument, "KV cache capacity exceeded"};
  }
  std::size_t token_width = 0;
  std::size_t elements = 0;
  std::size_t bytes = 0;
  if (!multiply(cache->heads, cache->head_dim, &token_width) ||
      !multiply(tokens, token_width, &elements)) {
    return {ErrorCode::kInvalidArgument, "KV append dimensions overflow"};
  }
  status = bytes_for(elements, sizeof(__nv_bfloat16), &bytes, "KV append");
  if (!status.ok())
    return status;
  status = buffer_size(key, bytes, device, "KV append key");
  if (!status.ok())
    return status;
  status = buffer_size(value, bytes, device, "KV append value");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  if (cache->type == KvCacheType::kQ8Paged) {
    status = ensure_q8_pages(cache, cache->length + tokens, stream);
    if (!status.ok())
      return status;
    std::size_t blocks = 0;
    if (!multiply(tokens, cache->heads, &blocks) ||
        blocks > static_cast<std::size_t>(
                     std::numeric_limits<unsigned int>::max())) {
      return {ErrorCode::kInvalidArgument,
              "paged Q8 KV append grid dimensions overflow"};
    }
    q8_kv_append_kernel<<<static_cast<unsigned int>(blocks), kAttentionThreads,
                          0, stream.get()>>>(
        static_cast<const __nv_bfloat16 *>(key.data()),
        static_cast<const __nv_bfloat16 *>(value.data()),
        static_cast<std::int8_t *const *>(cache->key_page_table.data()),
        static_cast<std::int8_t *const *>(cache->value_page_table.data()),
        static_cast<float *const *>(cache->key_scale_page_table.data()),
        static_cast<float *const *>(cache->value_scale_page_table.data()),
        cache->length, tokens, cache->heads, cache->head_dim,
        cache->page_tokens);
    status = launch_status("paged Q8 KV append kernel");
    if (!status.ok())
      return status;
    cache->length += tokens;
    return Status::Ok();
  }

  std::size_t offset_elements = 0;
  std::size_t offset_bytes = 0;
  if (!multiply(cache->length, token_width, &offset_elements) ||
      !multiply(offset_elements, sizeof(__nv_bfloat16), &offset_bytes)) {
    return {ErrorCode::kInvalidArgument, "KV append offset overflows"};
  }
  auto *const key_destination =
      static_cast<std::uint8_t *>(cache->key.data()) + offset_bytes;
  auto *const value_destination =
      static_cast<std::uint8_t *>(cache->value.data()) + offset_bytes;
  status = cuda_status(cudaMemcpyAsync(key_destination, key.data(), bytes,
                                       cudaMemcpyDeviceToDevice, stream.get()),
                       "cudaMemcpyAsync KV key append");
  if (!status.ok())
    return status;
  status = cuda_status(cudaMemcpyAsync(value_destination, value.data(), bytes,
                                       cudaMemcpyDeviceToDevice, stream.get()),
                       "cudaMemcpyAsync KV value append");
  if (!status.ok())
    return status;
  cache->length += tokens;
  return Status::Ok();
}

Status bf16_online_causal_attention(const DeviceBuffer &query,
                                    const std::size_t query_tokens,
                                    const std::size_t prefix_tokens,
                                    const KvCache &cache,
                                    DeviceBuffer *const output,
                                    const Stream &stream) {
  if (output == nullptr || !stream.valid() || cache.head_dim > 256 ||
      prefix_tokens > cache.length ||
      query_tokens > cache.length - prefix_tokens) {
    return {ErrorCode::kInvalidArgument,
            "online causal attention arguments are invalid"};
  }
  const int device = query.device();
  auto status = validate_cache(cache, device);
  if (!status.ok())
    return status;
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "attention stream is on a different CUDA device"};
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  auto status_dimensions =
      tensor_elements(query_tokens, cache.heads, cache.head_dim, &elements);
  if (!status_dimensions.ok())
    return status_dimensions;
  status = bytes_for(elements, sizeof(__nv_bfloat16), &bytes,
                     "attention query/output");
  if (!status.ok())
    return status;
  status = buffer_size(query, bytes, device, "attention query");
  if (!status.ok())
    return status;
  status = buffer_size(*output, bytes, device, "attention output");
  if (!status.ok())
    return status;
  std::size_t blocks = 0;
  if (!multiply(query_tokens, cache.heads, &blocks) ||
      blocks >
          static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    return {ErrorCode::kInvalidArgument, "attention grid dimensions overflow"};
  }
  status = select_device(device);
  if (!status.ok())
    return status;
  if (cache.type == KvCacheType::kQ8Paged) {
    q8_online_attention_kernel<<<static_cast<unsigned int>(blocks),
                                 kAttentionThreads, 0, stream.get()>>>(
        static_cast<const __nv_bfloat16 *>(query.data()),
        static_cast<const std::int8_t *const *>(cache.key_page_table.data()),
        static_cast<const std::int8_t *const *>(cache.value_page_table.data()),
        static_cast<const float *const *>(cache.key_scale_page_table.data()),
        static_cast<const float *const *>(cache.value_scale_page_table.data()),
        static_cast<__nv_bfloat16 *>(output->data()), query_tokens,
        prefix_tokens, cache.heads, cache.head_dim, cache.page_tokens);
    return launch_status("paged Q8 online causal attention kernel");
  }
  if (cache.head_dim == 128 && query_tokens >= 4) {
    constexpr std::size_t queries_per_block = 8;
    const std::size_t query_tiles =
        (query_tokens + queries_per_block - 1) / queries_per_block;
    std::size_t tiled_blocks = 0;
    if (!multiply(query_tiles, cache.heads, &tiled_blocks) ||
        tiled_blocks > static_cast<std::size_t>(
                           std::numeric_limits<unsigned int>::max())) {
      return {ErrorCode::kInvalidArgument,
              "tiled attention grid dimensions overflow"};
    }
    bf16_causal_prefill_kernel<<<static_cast<unsigned int>(tiled_blocks),
                                 kAttentionThreads, 0, stream.get()>>>(
        static_cast<const __nv_bfloat16 *>(query.data()),
        static_cast<const __nv_bfloat16 *>(cache.key.data()),
        static_cast<const __nv_bfloat16 *>(cache.value.data()),
        static_cast<__nv_bfloat16 *>(output->data()), query_tokens,
        prefix_tokens, cache.heads);
    return launch_status("tiled BF16 causal prefill attention kernel");
  }
  online_attention_kernel<<<static_cast<unsigned int>(blocks),
                            kAttentionThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(query.data()),
      static_cast<const __nv_bfloat16 *>(cache.key.data()),
      static_cast<const __nv_bfloat16 *>(cache.value.data()),
      static_cast<__nv_bfloat16 *>(output->data()), query_tokens, prefix_tokens,
      cache.heads, cache.head_dim);
  return launch_status("online causal attention kernel");
}

Status bf16_cached_cross_attention(
    const Blas &blas, const DeviceBuffer &query, const DeviceBuffer &key,
    const DeviceBuffer &value, const std::size_t query_tokens,
    const std::size_t key_tokens, const std::size_t heads,
    const std::size_t head_dim, DeviceBuffer *const scaled_key,
    DeviceBuffer *const scores, DeviceBuffer *const probabilities,
    DeviceBuffer *const output, const Stream &stream) {
  if (!blas.valid() || scaled_key == nullptr || scores == nullptr ||
      probabilities == nullptr || output == nullptr || !stream.valid() ||
      query_tokens == 0 || key_tokens < query_tokens || heads == 0 ||
      head_dim == 0 || query_tokens > static_cast<std::size_t>(INT_MAX) ||
      key_tokens > static_cast<std::size_t>(INT_MAX) ||
      heads > static_cast<std::size_t>(INT_MAX) ||
      head_dim > static_cast<std::size_t>(INT_MAX) ||
      heads > static_cast<std::size_t>(INT_MAX) / query_tokens ||
      heads > static_cast<std::size_t>(INT_MAX) / head_dim) {
    return {ErrorCode::kInvalidArgument,
            "cached cross-attention arguments are invalid"};
  }
  std::size_t query_elements = 0;
  std::size_t key_elements = 0;
  std::size_t score_elements = 0;
  std::size_t query_bytes = 0;
  std::size_t key_bytes = 0;
  std::size_t score_bytes = 0;
  if (!multiply(query_tokens, heads, &query_elements) ||
      !multiply(query_elements, head_dim, &query_elements) ||
      !multiply(key_tokens, heads, &key_elements) ||
      !multiply(key_elements, head_dim, &key_elements) ||
      !multiply(heads, query_tokens, &score_elements) ||
      !multiply(score_elements, key_tokens, &score_elements) ||
      !multiply(query_elements, sizeof(__nv_bfloat16), &query_bytes) ||
      !multiply(key_elements, sizeof(__nv_bfloat16), &key_bytes) ||
      !multiply(score_elements, sizeof(__nv_bfloat16), &score_bytes)) {
    return {ErrorCode::kInvalidArgument,
            "cached cross-attention dimensions overflow"};
  }
  const int device = query.device();
  if (blas.device() != device || key.device() != device ||
      value.device() != device || stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "cached cross-attention buffers span CUDA devices"};
  }
  auto status = buffer_size(query, query_bytes, device, "cached query");
  if (!status.ok())
    return status;
  status = buffer_size(key, key_bytes, device, "cached key");
  if (!status.ok())
    return status;
  status = buffer_size(value, key_bytes, device, "cached value");
  if (!status.ok())
    return status;
  status = buffer_size(*output, query_bytes, device, "cached output");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  status = ensure_capacity(scaled_key, device, key_bytes,
                           "cached scaled-key workspace");
  if (!status.ok())
    return status;
  status =
      ensure_capacity(scores, device, score_bytes, "cached score workspace");
  if (!status.ok())
    return status;
  status = ensure_capacity(probabilities, device, score_bytes,
                           "cached probability workspace");
  if (!status.ok())
    return status;

  const float scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  cached_scale_key_kernel<<<grid_for(key_elements), kElementwiseThreads, 0,
                            stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(key.data()),
      static_cast<__nv_bfloat16 *>(scaled_key->data()), key_elements, scale);
  status = launch_status("cached BF16 key scale kernel");
  if (!status.ok())
    return status;

  const cublasHandle_t handle = blas.get();
  status = cublas_status(cublasSetStream(handle, stream.get()),
                         "cublasSetStream cached attention");
  if (!status.ok())
    return status;
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const int token_stride = static_cast<int>(heads * head_dim);
  const long long head_stride = static_cast<long long>(head_dim);
  const long long score_stride =
      static_cast<long long>(query_tokens * key_tokens);
  status = cublas_status(
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(key_tokens),
          static_cast<int>(query_tokens), static_cast<int>(head_dim), &alpha,
          scaled_key->data(), CUDA_R_16BF, token_stride, head_stride,
          query.data(), CUDA_R_16BF, token_stride, head_stride, &beta,
          scores->data(), CUDA_R_16BF, static_cast<int>(key_tokens),
          score_stride, static_cast<int>(heads), CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cached attention QK bmm");
  if (!status.ok())
    return status;

  if (query_tokens > 1) {
    cached_causal_mask_kernel<<<grid_for(score_elements), kElementwiseThreads,
                                0, stream.get()>>>(
        static_cast<__nv_bfloat16 *>(scores->data()), query_tokens, key_tokens,
        heads);
    status = launch_status("cached causal masked-fill kernel");
    if (!status.ok())
      return status;
  }
  status = dispatch_cached_softmax(probabilities, *scores, heads * query_tokens,
                                   key_tokens, stream);
  if (!status.ok())
    return status;

  status = cublas_status(
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(head_dim),
          static_cast<int>(query_tokens), static_cast<int>(key_tokens), &alpha,
          value.data(), CUDA_R_16BF, token_stride, head_stride,
          probabilities->data(), CUDA_R_16BF, static_cast<int>(key_tokens),
          score_stride, &beta, output->data(), CUDA_R_16BF, token_stride,
          head_stride, static_cast<int>(heads), CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cached attention probability/value bmm");
  return status;
}

Status bf16_mha_prefill(const DeviceBuffer &qkv,
                        const DeviceBuffer &inverse_frequency,
                        const std::size_t tokens, const std::size_t heads,
                        const std::size_t head_dim, const float position_scale,
                        KvCache *const cache,
                        AttentionWorkspace *const workspace,
                        DeviceBuffer *const output, const Stream &stream) {
  if (cache == nullptr || workspace == nullptr || output == nullptr ||
      workspace->tokens != tokens || workspace->heads != heads ||
      workspace->head_dim != head_dim || cache->heads != heads ||
      cache->head_dim != head_dim) {
    return {ErrorCode::kInvalidArgument,
            "MHA prefill dimensions or buffers do not match"};
  }
  const std::size_t prefix = cache->length;
  auto status = bf16_split_qkv(qkv, tokens, heads, head_dim, &workspace->query,
                               &workspace->key, &workspace->value, stream);
  if (!status.ok())
    return status;
  status =
      bf16_apply_rope(&workspace->query, &workspace->key, inverse_frequency,
                      tokens, heads, head_dim, prefix, position_scale, stream);
  if (!status.ok())
    return status;
  status =
      bf16_kv_append(workspace->key, workspace->value, tokens, cache, stream);
  if (!status.ok())
    return status;
  status = cache->type == KvCacheType::kBF16 && head_dim == 128
               ? bf16_flash_causal_attention(
                     workspace->query, cache->key, cache->value, tokens,
                     cache->length, heads, head_dim, &workspace->softmax_lse,
                     &workspace->softmax_lse_accum, &workspace->output_accum,
                     output, stream)
               : bf16_online_causal_attention(workspace->query, tokens, prefix,
                                              *cache, output, stream);
  if (!status.ok())
    cache->length = prefix;
  return status;
}

Status bf16_mha_decode(const DeviceBuffer &qkv,
                       const DeviceBuffer &inverse_frequency,
                       const std::size_t heads, const std::size_t head_dim,
                       const float position_scale, KvCache *const cache,
                       AttentionWorkspace *const workspace,
                       DeviceBuffer *const output, const Stream &stream) {
  if (workspace == nullptr || workspace->tokens != 1) {
    return {ErrorCode::kInvalidArgument,
            "MHA decode requires a one-token workspace"};
  }
  return bf16_mha_prefill(qkv, inverse_frequency, 1, heads, head_dim,
                          position_scale, cache, workspace, output, stream);
}

} // namespace evo::cuda
