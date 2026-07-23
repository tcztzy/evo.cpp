// SPDX-License-Identifier: Apache-2.0
#include "evo2c/cuda/attention.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include <cuda_bf16.h>
#include <math_constants.h>

namespace evo2c::cuda {
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
      cache.length > cache.capacity) {
    return {ErrorCode::kInvalidArgument, "KV cache metadata is invalid"};
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
                 const std::size_t token_width, const std::size_t elements) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t token = index / token_width;
    const std::size_t within = index % token_width;
    const std::size_t source = token * token_width * 3 + within;
    query[index] = qkv[source];
    key[index] = qkv[source + token_width];
    value[index] = qkv[source + token_width * 2];
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

__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
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

  const std::size_t dimension = threadIdx.x;
  const std::size_t query_base = (query_token * heads + head) * head_dim;
  const float query_value =
      dimension < head_dim ? __bfloat162float(query[query_base + dimension])
                           : 0.0F;
  float accumulator = 0.0F;
  float maximum = -CUDART_INF_F;
  float normalizer = 0.0F;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  __shared__ float partial[32];
  __shared__ float shared_score;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int warp_count = (blockDim.x + 31) / 32;
  const std::size_t sources = prefix_tokens + query_token + 1;

  for (std::size_t source = 0; source < sources; ++source) {
    const std::size_t cache_base = (source * heads + head) * head_dim;
    float dot =
        dimension < head_dim
            ? query_value * __bfloat162float(key_cache[cache_base + dimension])
            : 0.0F;
    dot = warp_sum(dot);
    if (lane == 0)
      partial[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float total = lane < warp_count ? partial[lane] : 0.0F;
      total = warp_sum(total);
      if (lane == 0)
        shared_score = total * scale;
    }
    __syncthreads();

    const float score = shared_score;
    const float next_maximum = fmaxf(maximum, score);
    const float old_weight =
        normalizer == 0.0F ? 0.0F : expf(maximum - next_maximum);
    const float new_weight = expf(score - next_maximum);
    if (dimension < head_dim) {
      accumulator =
          accumulator * old_weight +
          new_weight * __bfloat162float(value_cache[cache_base + dimension]);
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
  if (key.valid() || value.valid()) {
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
  capacity = token_capacity;
  length = 0;
  heads = head_count;
  head_dim = dimensions_per_head;
  return Status::Ok();
}

Status AttentionWorkspace::allocate(const int device,
                                    const std::size_t token_count,
                                    const std::size_t head_count,
                                    const std::size_t dimensions_per_head) {
  if (query.valid() || key.valid() || value.valid()) {
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
  tokens = token_count;
  heads = head_count;
  head_dim = dimensions_per_head;
  return Status::Ok();
}

Status bf16_split_qkv(const DeviceBuffer &qkv, const std::size_t tokens,
                      const std::size_t heads, const std::size_t head_dim,
                      DeviceBuffer *const query, DeviceBuffer *const key,
                      DeviceBuffer *const value, const Stream &stream) {
  if (query == nullptr || key == nullptr || value == nullptr ||
      !stream.valid()) {
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
      static_cast<__nv_bfloat16 *>(value->data()), token_width, elements);
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
  const int device = cache->key.device();
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
  std::size_t offset_elements = 0;
  std::size_t offset_bytes = 0;
  if (!multiply(cache->heads, cache->head_dim, &token_width) ||
      !multiply(tokens, token_width, &elements) ||
      !multiply(cache->length, token_width, &offset_elements)) {
    return {ErrorCode::kInvalidArgument, "KV append dimensions overflow"};
  }
  status = bytes_for(elements, sizeof(__nv_bfloat16), &bytes, "KV append");
  if (!status.ok())
    return status;
  if (!multiply(offset_elements, sizeof(__nv_bfloat16), &offset_bytes)) {
    return {ErrorCode::kInvalidArgument, "KV append offset overflows"};
  }
  status = buffer_size(key, bytes, device, "KV append key");
  if (!status.ok())
    return status;
  status = buffer_size(value, bytes, device, "KV append value");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
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
  online_attention_kernel<<<static_cast<unsigned int>(blocks),
                            kAttentionThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(query.data()),
      static_cast<const __nv_bfloat16 *>(cache.key.data()),
      static_cast<const __nv_bfloat16 *>(cache.value.data()),
      static_cast<__nv_bfloat16 *>(output->data()), query_tokens, prefix_tokens,
      cache.heads, cache.head_dim);
  return launch_status("online causal attention kernel");
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
  status = bf16_online_causal_attention(workspace->query, tokens, prefix,
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

} // namespace evo2c::cuda
