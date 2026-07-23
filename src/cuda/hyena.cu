// SPDX-License-Identifier: Apache-2.0
#include "evo2c/cuda/hyena.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <cuComplex.h>
#include <cuda_bf16.h>

#include "evo2c/cuda/ops.hpp"

namespace evo2c::cuda {
namespace {

constexpr int kThreads = 256;

bool multiply(const std::size_t left, const std::size_t right,
              std::size_t *const result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *result = left * right;
  return true;
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

Status cufft_status(const cufftResult error, const char *const operation) {
  if (error == CUFFT_SUCCESS)
    return Status::Ok();
  return {ErrorCode::kCuda, std::string{operation} + " failed with cuFFT " +
                                std::to_string(static_cast<int>(error))};
}

unsigned int grid_for(const std::size_t elements) {
  const auto blocks = (elements + static_cast<std::size_t>(kThreads) - 1) /
                      static_cast<std::size_t>(kThreads);
  return static_cast<unsigned int>(std::min<std::size_t>(blocks, 65535));
}

Status validate_fir_dimensions(const std::size_t length,
                               const std::size_t channels,
                               const std::size_t filter_groups,
                               const std::size_t kernel_size) {
  if (length == 0 || channels == 0 || filter_groups == 0 || kernel_size < 2) {
    return {ErrorCode::kInvalidArgument,
            "FIR dimensions must be positive and kernel_size at least two"};
  }
  if (filter_groups > channels || channels % filter_groups != 0) {
    return {ErrorCode::kInvalidArgument,
            "FIR filter_groups must divide channels"};
  }
  return Status::Ok();
}

Status validate_fir_buffers(const DeviceBuffer &input,
                            const DeviceBuffer &weight,
                            const DeviceBuffer *const bias,
                            const std::size_t length,
                            const std::size_t channels,
                            const std::size_t filter_groups,
                            const std::size_t kernel_size,
                            DeviceBuffer *const output, const Stream &stream) {
  if (output == nullptr || !stream.valid()) {
    return {ErrorCode::kInvalidArgument,
            "FIR requires an output and initialized stream"};
  }
  auto status =
      validate_fir_dimensions(length, channels, filter_groups, kernel_size);
  if (!status.ok())
    return status;
  std::size_t tensor_elements = 0;
  std::size_t weight_elements = 0;
  std::size_t tensor_bytes = 0;
  std::size_t weight_bytes = 0;
  std::size_t bias_bytes = 0;
  if (!multiply(length, channels, &tensor_elements) ||
      !multiply(filter_groups, kernel_size, &weight_elements)) {
    return {ErrorCode::kInvalidArgument, "FIR dimensions overflow"};
  }
  status = bytes_for(tensor_elements, sizeof(__nv_bfloat16), &tensor_bytes,
                     "FIR tensor");
  if (!status.ok())
    return status;
  status = bytes_for(weight_elements, sizeof(__nv_bfloat16), &weight_bytes,
                     "FIR weight");
  if (!status.ok())
    return status;
  status = bytes_for(channels, sizeof(__nv_bfloat16), &bias_bytes, "FIR bias");
  if (!status.ok())
    return status;
  const int device = input.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "FIR stream is on a different CUDA device"};
  }
  status = buffer_size(input, tensor_bytes, device, "FIR input");
  if (!status.ok())
    return status;
  status = buffer_size(weight, weight_bytes, device, "FIR weight");
  if (!status.ok())
    return status;
  status = buffer_size(*output, tensor_bytes, device, "FIR output");
  if (!status.ok())
    return status;
  if (bias != nullptr)
    return buffer_size(*bias, bias_bytes, device, "FIR bias");
  return Status::Ok();
}

Status validate_cache(const FirCache *const cache, const int device,
                      const std::size_t channels,
                      const std::size_t kernel_size) {
  if (cache == nullptr)
    return Status::Ok();
  if (cache->channels != channels || cache->kernel_size != kernel_size) {
    return {ErrorCode::kInvalidArgument,
            "FIR cache dimensions do not match the operation"};
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  if (!multiply(channels, kernel_size - 1, &elements)) {
    return {ErrorCode::kInvalidArgument, "FIR cache dimensions overflow"};
  }
  auto status = bytes_for(elements, sizeof(float), &bytes, "FIR cache");
  if (!status.ok())
    return status;
  return buffer_size(cache->state, bytes, device, "FIR cache");
}

Status validate_iir_cache(const IirCache *const cache, const int device,
                          const std::size_t width,
                          const std::size_t state_size) {
  if (cache == nullptr || cache->channels != width ||
      cache->state_size != state_size) {
    return {ErrorCode::kInvalidArgument,
            "HCL cache dimensions do not match the operation"};
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  if (!multiply(width, state_size, &elements)) {
    return {ErrorCode::kInvalidArgument, "HCL cache dimensions overflow"};
  }
  auto status = bytes_for(elements, sizeof(float), &bytes, "HCL cache");
  if (!status.ok())
    return status;
  return buffer_size(cache->state, bytes, device, "HCL cache");
}

__global__ void split_projection_kernel(const __nv_bfloat16 *const projection,
                                        __nv_bfloat16 *const x2,
                                        __nv_bfloat16 *const x1,
                                        __nv_bfloat16 *const value,
                                        const std::size_t elements) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t source = index * 3;
    x2[index] = projection[source];
    x1[index] = projection[source + 1];
    value[index] = projection[source + 2];
  }
}

__global__ void fill_fir_cache_kernel(const __nv_bfloat16 *const input,
                                      float *const cache,
                                      const std::size_t length,
                                      const std::size_t channels,
                                      const std::size_t cache_size) {
  const std::size_t elements = channels * cache_size;
  const std::size_t valid = length < cache_size ? length : cache_size;
  const std::size_t leading = cache_size - valid;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t channel = index / cache_size;
    const std::size_t state_index = index % cache_size;
    if (state_index < leading) {
      cache[index] = 0.0F;
    } else {
      const std::size_t time = length - valid + state_index - leading;
      cache[index] = __bfloat162float(input[time * channels + channel]);
    }
  }
}

__global__ void fir_prefill_kernel(
    const __nv_bfloat16 *const input, const __nv_bfloat16 *const weight,
    const __nv_bfloat16 *const bias, __nv_bfloat16 *const output,
    const std::size_t length, const std::size_t channels,
    const std::size_t channels_per_group, const std::size_t kernel_size,
    const bool cross_correlation, const bool multiply_bias) {
  const std::size_t elements = length * channels;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t time = index / channels;
    const std::size_t channel = index % channels;
    const std::size_t group = channel / channels_per_group;
    float total = 0.0F;
    for (std::size_t tap = 0; tap < kernel_size; ++tap) {
      const std::size_t delay = cross_correlation ? kernel_size - 1 - tap : tap;
      if (time >= delay) {
        total += __bfloat162float(input[(time - delay) * channels + channel]) *
                 __bfloat162float(weight[group * kernel_size + tap]);
      }
    }
    if (bias != nullptr) {
      const float bias_value = __bfloat162float(bias[channel]);
      total += multiply_bias ? bias_value * __bfloat162float(input[index])
                             : bias_value;
    }
    output[index] = __float2bfloat16_rn(total);
  }
}

__global__ void fir_continue_kernel(
    const __nv_bfloat16 *const input, const __nv_bfloat16 *const weight,
    const __nv_bfloat16 *const bias, const float *const cache,
    __nv_bfloat16 *const output, const std::size_t length,
    const std::size_t channels, const std::size_t channels_per_group,
    const std::size_t kernel_size, const bool cross_correlation,
    const bool multiply_bias) {
  const std::size_t elements = length * channels;
  const std::size_t cache_size = kernel_size - 1;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t time = index / channels;
    const std::size_t channel = index % channels;
    const std::size_t group = channel / channels_per_group;
    float total = 0.0F;
    for (std::size_t state_index = 0; state_index < cache_size;
         ++state_index) {
      const std::size_t history = time + state_index;
      const float source =
          history < cache_size
              ? cache[channel * cache_size + history]
              : __bfloat162float(
                    input[(history - cache_size) * channels + channel]);
      const std::size_t tap =
          cross_correlation ? state_index : cache_size - state_index;
      total += source *
               __bfloat162float(weight[group * kernel_size + tap]);
    }
    const float current = __bfloat162float(input[index]);
    const std::size_t current_tap = cross_correlation ? cache_size : 0;
    total += current *
             __bfloat162float(weight[group * kernel_size + current_tap]);
    if (bias != nullptr) {
      const float bias_value = __bfloat162float(bias[channel]);
      total += multiply_bias ? bias_value * current : bias_value;
    }
    output[index] = __float2bfloat16_rn(total);
  }
}

__global__ void advance_fir_cache_kernel(
    const __nv_bfloat16 *const input, float *const cache,
    const std::size_t length, const std::size_t channels,
    const std::size_t cache_size) {
  for (std::size_t channel =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       channel < channels;
       channel += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    if (length >= cache_size) {
      for (std::size_t state_index = 0; state_index < cache_size;
           ++state_index) {
        const std::size_t time = length - cache_size + state_index;
        cache[channel * cache_size + state_index] =
            __bfloat162float(input[time * channels + channel]);
      }
      continue;
    }
    const std::size_t retained = cache_size - length;
    for (std::size_t state_index = 0; state_index < retained; ++state_index) {
      cache[channel * cache_size + state_index] =
          cache[channel * cache_size + state_index + length];
    }
    for (std::size_t time = 0; time < length; ++time) {
      cache[channel * cache_size + retained + time] =
          __bfloat162float(input[time * channels + channel]);
    }
  }
}

__global__ void fir_decode_kernel(
    const __nv_bfloat16 *const input, const __nv_bfloat16 *const weight,
    const __nv_bfloat16 *const bias, float *const cache,
    __nv_bfloat16 *const output, const std::size_t channels,
    const std::size_t channels_per_group, const std::size_t kernel_size,
    const bool cross_correlation, const bool multiply_bias) {
  for (std::size_t channel =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       channel < channels;
       channel += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t cache_size = kernel_size - 1;
    const std::size_t group = channel / channels_per_group;
    const float current = __bfloat162float(input[channel]);
    float total = 0.0F;
    for (std::size_t state_index = 0; state_index < cache_size; ++state_index) {
      const std::size_t tap =
          cross_correlation ? state_index : cache_size - state_index;
      total += cache[channel * cache_size + state_index] *
               __bfloat162float(weight[group * kernel_size + tap]);
    }
    const std::size_t current_tap = cross_correlation ? cache_size : 0;
    total +=
        current * __bfloat162float(weight[group * kernel_size + current_tap]);
    if (bias != nullptr) {
      const float bias_value = __bfloat162float(bias[channel]);
      total += multiply_bias ? bias_value * current : bias_value;
    }
    output[channel] = __float2bfloat16_rn(total);
    for (std::size_t state_index = 1; state_index < cache_size; ++state_index) {
      cache[channel * cache_size + state_index - 1] =
          cache[channel * cache_size + state_index];
    }
    cache[channel * cache_size + cache_size - 1] = current;
  }
}

__global__ void pack_input_kernel(const __nv_bfloat16 *const input,
                                  float *const packed, const std::size_t length,
                                  const std::size_t channels,
                                  const std::size_t fft_size) {
  const std::size_t elements = channels * fft_size;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t channel = index / fft_size;
    const std::size_t time = index % fft_size;
    packed[index] = time < length
                        ? __bfloat162float(input[time * channels + channel])
                        : 0.0F;
  }
}

__global__ void pack_filter_kernel(const __nv_bfloat16 *const weight,
                                   float *const packed,
                                   const std::size_t filter_groups,
                                   const std::size_t kernel_size,
                                   const std::size_t fft_size,
                                   const bool cross_correlation) {
  const std::size_t elements = filter_groups * fft_size;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t group = index / fft_size;
    const std::size_t time = index % fft_size;
    if (time < kernel_size) {
      const std::size_t tap = cross_correlation ? kernel_size - 1 - time : time;
      packed[index] = __bfloat162float(weight[group * kernel_size + tap]);
    } else {
      packed[index] = 0.0F;
    }
  }
}

__global__ void
multiply_frequency_kernel(cufftComplex *const input_frequency,
                          const cufftComplex *const filter_frequency,
                          const std::size_t channels,
                          const std::size_t channels_per_group,
                          const std::size_t frequency_size, const float scale) {
  const std::size_t elements = channels * frequency_size;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t channel = index / frequency_size;
    const std::size_t frequency = index % frequency_size;
    const std::size_t group = channel / channels_per_group;
    const cufftComplex product =
        cuCmulf(input_frequency[index],
                filter_frequency[group * frequency_size + frequency]);
    input_frequency[index] =
        make_cuFloatComplex(product.x * scale, product.y * scale);
  }
}

__global__ void
unpack_fir_kernel(const float *const packed, const __nv_bfloat16 *const input,
                  const __nv_bfloat16 *const bias, __nv_bfloat16 *const output,
                  const std::size_t length, const std::size_t channels,
                  const std::size_t fft_size, const bool multiply_bias) {
  const std::size_t elements = length * channels;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t time = index / channels;
    const std::size_t channel = index % channels;
    float result = packed[channel * fft_size + time];
    if (bias != nullptr) {
      const float bias_value = __bfloat162float(bias[channel]);
      result += multiply_bias ? bias_value * __bfloat162float(input[index])
                              : bias_value;
    }
    output[index] = __float2bfloat16_rn(result);
  }
}

__global__ void hcl_gate_kernel(const __nv_bfloat16 *const x1,
                                const __nv_bfloat16 *const value,
                                __nv_bfloat16 *const gated,
                                const std::size_t elements) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    gated[index] = __float2bfloat16_rn(__bfloat162float(x1[index]) *
                                       __bfloat162float(value[index]));
  }
}

__global__ void hcl_recurrence_prefill_kernel(
    const __nv_bfloat16 *const x2, const __nv_bfloat16 *const gated,
    const __nv_bfloat16 *const direct, const float *const log_poles,
    const float *const residues, float *const state,
    __nv_bfloat16 *const output, const std::size_t length,
    const std::size_t width, const std::size_t state_size) {
  for (std::size_t channel =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       channel < width;
       channel += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    for (std::size_t time = 0; time < length; ++time) {
      const std::size_t data_index = time * width + channel;
      const float input_value = __bfloat162float(gated[data_index]);
      float modal = 0.0F;
      for (std::size_t mode = 0; mode < state_size; ++mode) {
        const std::size_t state_index = channel * state_size + mode;
        const float updated =
            expf(log_poles[state_index]) * state[state_index] + input_value;
        state[state_index] = updated;
        modal += residues[state_index] * updated;
      }
      output[data_index] = __float2bfloat16_rn(
          __bfloat162float(x2[data_index]) *
          (modal + __bfloat162float(direct[channel]) * input_value));
    }
  }
}

__global__ void hcl_filter_kernel(const float *const log_poles,
                                  const float *const residues,
                                  float *const filter, const std::size_t length,
                                  const std::size_t width,
                                  const std::size_t state_size,
                                  const std::size_t fft_size) {
  const std::size_t elements = width * fft_size;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t channel = index / fft_size;
    const std::size_t time = index % fft_size;
    float total = 0.0F;
    if (time < length) {
      for (std::size_t mode = 0; mode < state_size; ++mode) {
        const std::size_t parameter = channel * state_size + mode;
        const float exponential =
            expf(log_poles[parameter] * static_cast<float>(time));
        const float term = __fmul_rn(residues[parameter], exponential);
        total = __fadd_rn(total, term);
      }
    }
    filter[index] = total;
  }
}

__global__ void hcl_state_prefill_kernel(const __nv_bfloat16 *const gated,
                                         const float *const log_poles,
                                         float *const state,
                                         const std::size_t length,
                                         const std::size_t width,
                                         const std::size_t state_size) {
  const std::size_t elements = width * state_size;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t channel = index / state_size;
    const float pole = expf(log_poles[index]);
    float current = 0.0F;
    for (std::size_t time = 0; time < length; ++time) {
      current =
          pole * current + __bfloat162float(gated[time * width + channel]);
    }
    state[index] = current;
  }
}

__global__ void
unpack_hcl_kernel(const float *const convolution, const __nv_bfloat16 *const x2,
                  const __nv_bfloat16 *const gated,
                  const __nv_bfloat16 *const direct,
                  __nv_bfloat16 *const output, const std::size_t length,
                  const std::size_t width, const std::size_t fft_size) {
  const std::size_t elements = length * width;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t time = index / width;
    const std::size_t channel = index % width;
    // Vortex casts the FFT result to BF16 before applying the direct term,
    // then performs each BF16 pointwise operation separately.
    const __nv_bfloat16 convolved =
        __float2bfloat16_rn(convolution[channel * fft_size + time]);
    const __nv_bfloat16 direct_product = __float2bfloat16_rn(
        __bfloat162float(direct[channel]) * __bfloat162float(gated[index]));
    const __nv_bfloat16 mixed = __float2bfloat16_rn(
        __bfloat162float(convolved) + __bfloat162float(direct_product));
    output[index] = __float2bfloat16_rn(__bfloat162float(x2[index]) *
                                        __bfloat162float(mixed));
  }
}

__global__ void
hcl_decode_kernel(const __nv_bfloat16 *const x2, const __nv_bfloat16 *const x1,
                  const __nv_bfloat16 *const value,
                  const __nv_bfloat16 *const direct,
                  const float *const log_poles, const float *const residues,
                  float *const state, __nv_bfloat16 *const output,
                  const std::size_t width, const std::size_t state_size) {
  for (std::size_t channel =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       channel < width;
       channel += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const float gated = __bfloat162float(__float2bfloat16_rn(
        __bfloat162float(x1[channel]) * __bfloat162float(value[channel])));
    float modal = 0.0F;
    for (std::size_t mode = 0; mode < state_size; ++mode) {
      const std::size_t index = channel * state_size + mode;
      const float updated = expf(log_poles[index]) * state[index] + gated;
      state[index] = updated;
      modal += residues[index] * updated;
    }
    output[channel] = __float2bfloat16_rn(
        __bfloat162float(x2[channel]) *
        (modal + __bfloat162float(direct[channel]) * gated));
  }
}

Status fill_fir_cache(const DeviceBuffer &input, const std::size_t length,
                      const std::size_t channels, const std::size_t kernel_size,
                      FirCache *const cache, const Stream &stream) {
  if (cache == nullptr)
    return Status::Ok();
  const std::size_t cache_size = kernel_size - 1;
  std::size_t elements = 0;
  if (!multiply(channels, cache_size, &elements)) {
    return {ErrorCode::kInvalidArgument, "FIR cache dimensions overflow"};
  }
  fill_fir_cache_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<float *>(cache->state.data()), length, channels, cache_size);
  return launch_status("fill FIR cache kernel");
}

Status validate_hcl_buffers(const DeviceBuffer &x2, const DeviceBuffer &x1,
                            const DeviceBuffer &value,
                            const DeviceBuffer &direct,
                            const DeviceBuffer &log_poles,
                            const DeviceBuffer &residues,
                            const std::size_t length, const std::size_t width,
                            const std::size_t state_size, IirCache *const cache,
                            DeviceBuffer *const output, const Stream &stream) {
  if (length == 0 || width == 0 || state_size == 0 || output == nullptr ||
      !stream.valid()) {
    return {ErrorCode::kInvalidArgument,
            "HCL dimensions and output must be positive"};
  }
  std::size_t tensor_elements = 0;
  std::size_t parameter_elements = 0;
  std::size_t tensor_bytes = 0;
  std::size_t direct_bytes = 0;
  std::size_t parameter_bytes = 0;
  if (!multiply(length, width, &tensor_elements) ||
      !multiply(width, state_size, &parameter_elements)) {
    return {ErrorCode::kInvalidArgument, "HCL dimensions overflow"};
  }
  auto status = bytes_for(tensor_elements, sizeof(__nv_bfloat16), &tensor_bytes,
                          "HCL tensor");
  if (!status.ok())
    return status;
  status = bytes_for(width, sizeof(__nv_bfloat16), &direct_bytes, "HCL direct");
  if (!status.ok())
    return status;
  status = bytes_for(parameter_elements, sizeof(float), &parameter_bytes,
                     "HCL parameters");
  if (!status.ok())
    return status;
  const int device = x2.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "HCL stream is on a different CUDA device"};
  }
  const std::array<const DeviceBuffer *, 4> tensors{&x2, &x1, &value, output};
  for (const auto *const tensor : tensors) {
    status = buffer_size(*tensor, tensor_bytes, device, "HCL sequence tensor");
    if (!status.ok())
      return status;
  }
  status = buffer_size(direct, direct_bytes, device, "HCL direct");
  if (!status.ok())
    return status;
  status = buffer_size(log_poles, parameter_bytes, device, "HCL log_poles");
  if (!status.ok())
    return status;
  status = buffer_size(residues, parameter_bytes, device, "HCL residues");
  if (!status.ok())
    return status;
  return validate_iir_cache(cache, device, width, state_size);
}

} // namespace

Status FirCache::allocate(const int device, const std::size_t channel_count,
                          const std::size_t filter_length,
                          const Stream &stream) {
  if (state.valid() || stream.device() != device || filter_length < 2) {
    return {ErrorCode::kInvalidArgument,
            "FIR cache allocation received invalid arguments"};
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  if (!multiply(channel_count, filter_length - 1, &elements)) {
    return {ErrorCode::kInvalidArgument, "FIR cache dimensions overflow"};
  }
  auto status = bytes_for(elements, sizeof(float), &bytes, "FIR cache");
  if (!status.ok())
    return status;
  status = state.allocate(device, bytes);
  if (!status.ok())
    return status;
  status = state.zero(stream);
  if (!status.ok()) {
    state.reset();
    return status;
  }
  channels = channel_count;
  kernel_size = filter_length;
  return Status::Ok();
}

Status IirCache::allocate(const int device, const std::size_t channel_count,
                          const std::size_t mode_count, const Stream &stream) {
  if (state.valid() || stream.device() != device || channel_count == 0 ||
      mode_count == 0) {
    return {ErrorCode::kInvalidArgument,
            "IIR cache allocation received invalid arguments"};
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  if (!multiply(channel_count, mode_count, &elements)) {
    return {ErrorCode::kInvalidArgument, "IIR cache dimensions overflow"};
  }
  auto status = bytes_for(elements, sizeof(float), &bytes, "IIR cache");
  if (!status.ok())
    return status;
  status = state.allocate(device, bytes);
  if (!status.ok())
    return status;
  status = state.zero(stream);
  if (!status.ok()) {
    state.reset();
    return status;
  }
  channels = channel_count;
  state_size = mode_count;
  return Status::Ok();
}

FftWorkspace::~FftWorkspace() { reset(); }

FftWorkspace::FftWorkspace(FftWorkspace &&other) noexcept
    : input_time_(std::move(other.input_time_)),
      filter_time_(std::move(other.filter_time_)),
      output_time_(std::move(other.output_time_)),
      input_frequency_(std::move(other.input_frequency_)),
      filter_frequency_(std::move(other.filter_frequency_)),
      input_forward_(std::exchange(other.input_forward_, 0)),
      filter_forward_(std::exchange(other.filter_forward_, 0)),
      inverse_(std::exchange(other.inverse_, 0)),
      device_(std::exchange(other.device_, -1)),
      channels_(std::exchange(other.channels_, 0)),
      filter_groups_(std::exchange(other.filter_groups_, 0)),
      fft_size_(std::exchange(other.fft_size_, 0)) {}

FftWorkspace &FftWorkspace::operator=(FftWorkspace &&other) noexcept {
  if (this != &other) {
    reset();
    input_time_ = std::move(other.input_time_);
    filter_time_ = std::move(other.filter_time_);
    output_time_ = std::move(other.output_time_);
    input_frequency_ = std::move(other.input_frequency_);
    filter_frequency_ = std::move(other.filter_frequency_);
    input_forward_ = std::exchange(other.input_forward_, 0);
    filter_forward_ = std::exchange(other.filter_forward_, 0);
    inverse_ = std::exchange(other.inverse_, 0);
    device_ = std::exchange(other.device_, -1);
    channels_ = std::exchange(other.channels_, 0);
    filter_groups_ = std::exchange(other.filter_groups_, 0);
    fft_size_ = std::exchange(other.fft_size_, 0);
  }
  return *this;
}

Status FftWorkspace::allocate(const int device, const std::size_t channels,
                              const std::size_t filter_groups,
                              const std::size_t fft_size) {
  if (device_ >= 0 || channels == 0 || filter_groups == 0 || fft_size < 2 ||
      filter_groups > channels || channels % filter_groups != 0 ||
      channels > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      filter_groups >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      fft_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return {ErrorCode::kInvalidArgument,
            "FFT workspace allocation received invalid dimensions"};
  }
  auto status = select_device(device);
  if (!status.ok())
    return status;
  const std::size_t frequency_size = fft_size / 2 + 1;
  std::size_t channel_time_elements = 0;
  std::size_t filter_time_elements = 0;
  std::size_t channel_frequency_elements = 0;
  std::size_t filter_frequency_elements = 0;
  std::size_t channel_time_bytes = 0;
  std::size_t filter_time_bytes = 0;
  std::size_t channel_frequency_bytes = 0;
  std::size_t filter_frequency_bytes = 0;
  if (!multiply(channels, fft_size, &channel_time_elements) ||
      !multiply(filter_groups, fft_size, &filter_time_elements) ||
      !multiply(channels, frequency_size, &channel_frequency_elements) ||
      !multiply(filter_groups, frequency_size, &filter_frequency_elements)) {
    return {ErrorCode::kInvalidArgument, "FFT workspace dimensions overflow"};
  }
  status = bytes_for(channel_time_elements, sizeof(float), &channel_time_bytes,
                     "FFT channel time storage");
  if (!status.ok())
    return status;
  status = bytes_for(filter_time_elements, sizeof(float), &filter_time_bytes,
                     "FFT filter time storage");
  if (!status.ok())
    return status;
  status = bytes_for(channel_frequency_elements, sizeof(cufftComplex),
                     &channel_frequency_bytes, "FFT channel spectrum");
  if (!status.ok())
    return status;
  status = bytes_for(filter_frequency_elements, sizeof(cufftComplex),
                     &filter_frequency_bytes, "FFT filter spectrum");
  if (!status.ok())
    return status;

  status = input_time_.allocate(device, channel_time_bytes);
  if (!status.ok()) {
    reset();
    return status;
  }
  status = filter_time_.allocate(device, filter_time_bytes);
  if (!status.ok()) {
    reset();
    return status;
  }
  status = output_time_.allocate(device, channel_time_bytes);
  if (!status.ok()) {
    reset();
    return status;
  }
  status = input_frequency_.allocate(device, channel_frequency_bytes);
  if (!status.ok()) {
    reset();
    return status;
  }
  status = filter_frequency_.allocate(device, filter_frequency_bytes);
  if (!status.ok()) {
    reset();
    return status;
  }

  const int transform_size = static_cast<int>(fft_size);
  const int frequency_distance = static_cast<int>(frequency_size);
  int dimensions[1]{transform_size};
  status =
      cufft_status(cufftPlanMany(&input_forward_, 1, dimensions, nullptr, 1,
                                 transform_size, nullptr, 1, frequency_distance,
                                 CUFFT_R2C, static_cast<int>(channels)),
                   "cufftPlanMany input forward");
  if (!status.ok()) {
    reset();
    return status;
  }
  status =
      cufft_status(cufftPlanMany(&filter_forward_, 1, dimensions, nullptr, 1,
                                 transform_size, nullptr, 1, frequency_distance,
                                 CUFFT_R2C, static_cast<int>(filter_groups)),
                   "cufftPlanMany filter forward");
  if (!status.ok()) {
    reset();
    return status;
  }
  status =
      cufft_status(cufftPlanMany(&inverse_, 1, dimensions, nullptr, 1,
                                 frequency_distance, nullptr, 1, transform_size,
                                 CUFFT_C2R, static_cast<int>(channels)),
                   "cufftPlanMany inverse");
  if (!status.ok()) {
    reset();
    return status;
  }
  device_ = device;
  channels_ = channels;
  filter_groups_ = filter_groups;
  fft_size_ = fft_size;
  return Status::Ok();
}

void FftWorkspace::reset() noexcept {
  int previous = -1;
  const int workspace_device = device_;
  static_cast<void>(cudaGetDevice(&previous));
  if (workspace_device >= 0)
    static_cast<void>(cudaSetDevice(workspace_device));
  if (inverse_ != 0)
    static_cast<void>(cufftDestroy(inverse_));
  if (filter_forward_ != 0)
    static_cast<void>(cufftDestroy(filter_forward_));
  if (input_forward_ != 0)
    static_cast<void>(cufftDestroy(input_forward_));
  inverse_ = 0;
  filter_forward_ = 0;
  input_forward_ = 0;
  filter_frequency_.reset();
  input_frequency_.reset();
  output_time_.reset();
  filter_time_.reset();
  input_time_.reset();
  device_ = -1;
  channels_ = 0;
  filter_groups_ = 0;
  fft_size_ = 0;
  if (previous >= 0 && previous != workspace_device)
    static_cast<void>(cudaSetDevice(previous));
}

bool FftWorkspace::matches(const int device, const std::size_t channels,
                           const std::size_t filter_groups,
                           const std::size_t fft_size) const noexcept {
  return device_ == device && channels_ == channels &&
         filter_groups_ == filter_groups && fft_size_ == fft_size;
}

Status FftWorkspace::execute(const Stream &stream) {
  auto status = cufft_status(cufftSetStream(input_forward_, stream.get()),
                             "cufftSetStream input forward");
  if (!status.ok())
    return status;
  status = cufft_status(cufftSetStream(filter_forward_, stream.get()),
                        "cufftSetStream filter forward");
  if (!status.ok())
    return status;
  status = cufft_status(cufftSetStream(inverse_, stream.get()),
                        "cufftSetStream inverse");
  if (!status.ok())
    return status;
  status = cufft_status(
      cufftExecR2C(input_forward_, static_cast<cufftReal *>(input_time_.data()),
                   static_cast<cufftComplex *>(input_frequency_.data())),
      "cufftExecR2C input");
  if (!status.ok())
    return status;
  status = cufft_status(
      cufftExecR2C(filter_forward_,
                   static_cast<cufftReal *>(filter_time_.data()),
                   static_cast<cufftComplex *>(filter_frequency_.data())),
      "cufftExecR2C filter");
  if (!status.ok())
    return status;
  const std::size_t frequency_size = fft_size_ / 2 + 1;
  std::size_t frequency_elements = 0;
  if (!multiply(channels_, frequency_size, &frequency_elements)) {
    return {ErrorCode::kInvalidArgument, "FFT frequency dimensions overflow"};
  }
  multiply_frequency_kernel<<<grid_for(frequency_elements), kThreads, 0,
                              stream.get()>>>(
      static_cast<cufftComplex *>(input_frequency_.data()),
      static_cast<const cufftComplex *>(filter_frequency_.data()), channels_,
      channels_ / filter_groups_, frequency_size,
      1.0F / static_cast<float>(fft_size_));
  status = launch_status("FFT frequency multiply kernel");
  if (!status.ok())
    return status;
  return cufft_status(
      cufftExecC2R(inverse_,
                   static_cast<cufftComplex *>(input_frequency_.data()),
                   static_cast<cufftReal *>(output_time_.data())),
      "cufftExecC2R inverse");
}

std::size_t fir_fft_size(const std::size_t length,
                         const std::size_t kernel_size) noexcept {
  if (length == 0 || kernel_size == 0 ||
      length > std::numeric_limits<std::size_t>::max() - kernel_size + 1) {
    return 0;
  }
  const std::size_t required = length + kernel_size - 1;
  std::size_t result = 1;
  while (result < required) {
    if (result > std::numeric_limits<std::size_t>::max() / 2)
      return 0;
    result *= 2;
  }
  return result;
}

Status
bf16_split_hyena_projection(const DeviceBuffer &projection,
                            const std::size_t length, const std::size_t width,
                            DeviceBuffer *const x2, DeviceBuffer *const x1,
                            DeviceBuffer *const value, const Stream &stream) {
  if (x2 == nullptr || x1 == nullptr || value == nullptr || !stream.valid()) {
    return {ErrorCode::kInvalidArgument,
            "Hyena split requires three outputs and an initialized stream"};
  }
  std::size_t elements = 0;
  std::size_t projection_elements = 0;
  std::size_t output_bytes = 0;
  std::size_t projection_bytes = 0;
  if (!multiply(length, width, &elements) ||
      !multiply(elements, 3, &projection_elements)) {
    return {ErrorCode::kInvalidArgument,
            "Hyena projection dimensions are zero or overflow"};
  }
  auto status = bytes_for(elements, sizeof(__nv_bfloat16), &output_bytes,
                          "Hyena split output");
  if (!status.ok())
    return status;
  status = bytes_for(projection_elements, sizeof(__nv_bfloat16),
                     &projection_bytes, "Hyena projection");
  if (!status.ok())
    return status;
  const int device = projection.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "Hyena split stream is on a different CUDA device"};
  }
  status =
      buffer_size(projection, projection_bytes, device, "Hyena projection");
  if (!status.ok())
    return status;
  for (const auto *const output : {x2, x1, value}) {
    status = buffer_size(*output, output_bytes, device, "Hyena split output");
    if (!status.ok())
      return status;
  }
  status = select_device(device);
  if (!status.ok())
    return status;
  split_projection_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(projection.data()),
      static_cast<__nv_bfloat16 *>(x2->data()),
      static_cast<__nv_bfloat16 *>(x1->data()),
      static_cast<__nv_bfloat16 *>(value->data()), elements);
  return launch_status("split Hyena projection kernel");
}

Status bf16_fir_prefill_direct(
    const DeviceBuffer &input, const DeviceBuffer &weight,
    const DeviceBuffer *const bias, const std::size_t length,
    const std::size_t channels, const std::size_t filter_groups,
    const std::size_t kernel_size, const FirOrientation orientation,
    const FirBiasMode bias_mode, DeviceBuffer *const output,
    FirCache *const cache, const Stream &stream) {
  if ((orientation != FirOrientation::kCrossCorrelation &&
       orientation != FirOrientation::kCausalConvolution) ||
      (bias_mode != FirBiasMode::kAdd &&
       bias_mode != FirBiasMode::kMultiplyInput)) {
    return {ErrorCode::kInvalidArgument, "FIR mode is invalid"};
  }
  auto status =
      validate_fir_buffers(input, weight, bias, length, channels, filter_groups,
                           kernel_size, output, stream);
  if (!status.ok())
    return status;
  status = validate_cache(cache, input.device(), channels, kernel_size);
  if (!status.ok())
    return status;
  status = select_device(input.device());
  if (!status.ok())
    return status;
  std::size_t elements = 0;
  if (!multiply(length, channels, &elements)) {
    return {ErrorCode::kInvalidArgument, "FIR dimensions overflow"};
  }
  fir_prefill_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<const __nv_bfloat16 *>(weight.data()),
      bias == nullptr ? nullptr
                      : static_cast<const __nv_bfloat16 *>(bias->data()),
      static_cast<__nv_bfloat16 *>(output->data()), length, channels,
      channels / filter_groups, kernel_size,
      orientation == FirOrientation::kCrossCorrelation,
      bias_mode == FirBiasMode::kMultiplyInput);
  status = launch_status("direct FIR prefill kernel");
  if (!status.ok())
    return status;
  return fill_fir_cache(input, length, channels, kernel_size, cache, stream);
}

Status bf16_fir_continue_direct(
    const DeviceBuffer &input, const DeviceBuffer &weight,
    const DeviceBuffer *const bias, const std::size_t length,
    const std::size_t channels, const std::size_t filter_groups,
    const std::size_t kernel_size, const FirOrientation orientation,
    const FirBiasMode bias_mode, DeviceBuffer *const output,
    FirCache *const cache, const Stream &stream) {
  if (cache == nullptr ||
      (orientation != FirOrientation::kCrossCorrelation &&
       orientation != FirOrientation::kCausalConvolution) ||
      (bias_mode != FirBiasMode::kAdd &&
       bias_mode != FirBiasMode::kMultiplyInput)) {
    return {ErrorCode::kInvalidArgument,
            "continued FIR cache or mode is invalid"};
  }
  auto status =
      validate_fir_buffers(input, weight, bias, length, channels, filter_groups,
                           kernel_size, output, stream);
  if (!status.ok())
    return status;
  status = validate_cache(cache, input.device(), channels, kernel_size);
  if (!status.ok())
    return status;
  status = select_device(input.device());
  if (!status.ok())
    return status;
  std::size_t elements = 0;
  if (!multiply(length, channels, &elements)) {
    return {ErrorCode::kInvalidArgument,
            "continued FIR dimensions overflow"};
  }
  fir_continue_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<const __nv_bfloat16 *>(weight.data()),
      bias == nullptr ? nullptr
                      : static_cast<const __nv_bfloat16 *>(bias->data()),
      static_cast<const float *>(cache->state.data()),
      static_cast<__nv_bfloat16 *>(output->data()), length, channels,
      channels / filter_groups, kernel_size,
      orientation == FirOrientation::kCrossCorrelation,
      bias_mode == FirBiasMode::kMultiplyInput);
  status = launch_status("continued direct FIR kernel");
  if (!status.ok())
    return status;
  advance_fir_cache_kernel<<<grid_for(channels), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<float *>(cache->state.data()), length, channels,
      kernel_size - 1);
  return launch_status("advance continued FIR cache kernel");
}

Status bf16_fir_prefill_fft(
    const DeviceBuffer &input, const DeviceBuffer &weight,
    const DeviceBuffer *const bias, const std::size_t length,
    const std::size_t channels, const std::size_t filter_groups,
    const std::size_t kernel_size, const FirOrientation orientation,
    const FirBiasMode bias_mode, DeviceBuffer *const output,
    FirCache *const cache, FftWorkspace *const workspace,
    const Stream &stream) {
  if (workspace == nullptr ||
      (orientation != FirOrientation::kCrossCorrelation &&
       orientation != FirOrientation::kCausalConvolution) ||
      (bias_mode != FirBiasMode::kAdd &&
       bias_mode != FirBiasMode::kMultiplyInput)) {
    return {ErrorCode::kInvalidArgument,
            "FFT FIR workspace or mode is invalid"};
  }
  auto status =
      validate_fir_buffers(input, weight, bias, length, channels, filter_groups,
                           kernel_size, output, stream);
  if (!status.ok())
    return status;
  status = validate_cache(cache, input.device(), channels, kernel_size);
  if (!status.ok())
    return status;
  const std::size_t fft_size = fir_fft_size(length, kernel_size);
  if (fft_size == 0 ||
      !workspace->matches(input.device(), channels, filter_groups, fft_size)) {
    return {ErrorCode::kInvalidArgument,
            "FFT FIR workspace dimensions do not match"};
  }
  status = select_device(input.device());
  if (!status.ok())
    return status;
  std::size_t channel_time_elements = 0;
  std::size_t filter_time_elements = 0;
  std::size_t output_elements = 0;
  if (!multiply(channels, fft_size, &channel_time_elements) ||
      !multiply(filter_groups, fft_size, &filter_time_elements) ||
      !multiply(length, channels, &output_elements)) {
    return {ErrorCode::kInvalidArgument, "FFT FIR dimensions overflow"};
  }
  pack_input_kernel<<<grid_for(channel_time_elements), kThreads, 0,
                      stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<float *>(workspace->input_time_.data()), length, channels,
      fft_size);
  status = launch_status("FFT FIR input pack kernel");
  if (!status.ok())
    return status;
  pack_filter_kernel<<<grid_for(filter_time_elements), kThreads, 0,
                       stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(weight.data()),
      static_cast<float *>(workspace->filter_time_.data()), filter_groups,
      kernel_size, fft_size, orientation == FirOrientation::kCrossCorrelation);
  status = launch_status("FFT FIR filter pack kernel");
  if (!status.ok())
    return status;
  status = workspace->execute(stream);
  if (!status.ok())
    return status;
  unpack_fir_kernel<<<grid_for(output_elements), kThreads, 0, stream.get()>>>(
      static_cast<const float *>(workspace->output_time_.data()),
      static_cast<const __nv_bfloat16 *>(input.data()),
      bias == nullptr ? nullptr
                      : static_cast<const __nv_bfloat16 *>(bias->data()),
      static_cast<__nv_bfloat16 *>(output->data()), length, channels, fft_size,
      bias_mode == FirBiasMode::kMultiplyInput);
  status = launch_status("FFT FIR output unpack kernel");
  if (!status.ok())
    return status;
  return fill_fir_cache(input, length, channels, kernel_size, cache, stream);
}

Status bf16_fir_decode(const DeviceBuffer &input, const DeviceBuffer &weight,
                       const DeviceBuffer *const bias,
                       const std::size_t channels,
                       const std::size_t filter_groups,
                       const std::size_t kernel_size,
                       const FirOrientation orientation,
                       const FirBiasMode bias_mode, FirCache *const cache,
                       DeviceBuffer *const output, const Stream &stream) {
  auto status =
      validate_fir_buffers(input, weight, bias, 1, channels, filter_groups,
                           kernel_size, output, stream);
  if (!status.ok())
    return status;
  if (cache == nullptr ||
      (orientation != FirOrientation::kCrossCorrelation &&
       orientation != FirOrientation::kCausalConvolution) ||
      (bias_mode != FirBiasMode::kAdd &&
       bias_mode != FirBiasMode::kMultiplyInput)) {
    return {ErrorCode::kInvalidArgument, "FIR decode cache or mode is invalid"};
  }
  status = validate_cache(cache, input.device(), channels, kernel_size);
  if (!status.ok())
    return status;
  status = select_device(input.device());
  if (!status.ok())
    return status;
  fir_decode_kernel<<<grid_for(channels), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<const __nv_bfloat16 *>(weight.data()),
      bias == nullptr ? nullptr
                      : static_cast<const __nv_bfloat16 *>(bias->data()),
      static_cast<float *>(cache->state.data()),
      static_cast<__nv_bfloat16 *>(output->data()), channels,
      channels / filter_groups, kernel_size,
      orientation == FirOrientation::kCrossCorrelation,
      bias_mode == FirBiasMode::kMultiplyInput);
  return launch_status("FIR decode kernel");
}

Status bf16_hcs_prefill(const DeviceBuffer &x2, const DeviceBuffer &x1,
                        const DeviceBuffer &value, const DeviceBuffer &weight,
                        const std::size_t length, const std::size_t width,
                        const std::size_t filter_groups,
                        const std::size_t kernel_size, FirCache *const cache,
                        DeviceBuffer *const scratch, DeviceBuffer *const output,
                        const Stream &stream) {
  if (scratch == nullptr || output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "HCS prefill requires scratch and output buffers"};
  }
  std::size_t elements = 0;
  if (!multiply(length, width, &elements)) {
    return {ErrorCode::kInvalidArgument, "HCS dimensions overflow"};
  }
  auto status = bf16_gated_elementwise(
      x1, value, elements, GatedActivation::kIdentity, scratch, stream);
  if (!status.ok())
    return status;
  status = bf16_fir_prefill_direct(*scratch, weight, nullptr, length, width,
                                   filter_groups, kernel_size,
                                   FirOrientation::kCrossCorrelation,
                                   FirBiasMode::kAdd, output, cache, stream);
  if (!status.ok())
    return status;
  return bf16_gated_elementwise(*output, x2, elements,
                                GatedActivation::kIdentity, output, stream);
}

Status bf16_hcs_decode(const DeviceBuffer &x2, const DeviceBuffer &x1,
                       const DeviceBuffer &value, const DeviceBuffer &weight,
                       const std::size_t width, const std::size_t filter_groups,
                       const std::size_t kernel_size, FirCache *const cache,
                       DeviceBuffer *const scratch, DeviceBuffer *const output,
                       const Stream &stream) {
  if (scratch == nullptr || output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "HCS decode requires scratch and output buffers"};
  }
  auto status = bf16_gated_elementwise(
      x1, value, width, GatedActivation::kIdentity, scratch, stream);
  if (!status.ok())
    return status;
  status = bf16_fir_decode(*scratch, weight, nullptr, width, filter_groups,
                           kernel_size, FirOrientation::kCrossCorrelation,
                           FirBiasMode::kAdd, cache, output, stream);
  if (!status.ok())
    return status;
  return bf16_gated_elementwise(*output, x2, width, GatedActivation::kIdentity,
                                output, stream);
}

Status bf16_hcs_continue(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, const std::size_t length,
    const std::size_t width, const std::size_t filter_groups,
    const std::size_t kernel_size, FirCache *const cache,
    DeviceBuffer *const scratch, DeviceBuffer *const output,
    const Stream &stream) {
  if (scratch == nullptr || output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "HCS continuation requires scratch and output buffers"};
  }
  std::size_t elements = 0;
  if (!multiply(length, width, &elements)) {
    return {ErrorCode::kInvalidArgument,
            "HCS continuation dimensions overflow"};
  }
  auto status = bf16_gated_elementwise(
      x1, value, elements, GatedActivation::kIdentity, scratch, stream);
  if (!status.ok())
    return status;
  status = bf16_fir_continue_direct(
      *scratch, weight, nullptr, length, width, filter_groups, kernel_size,
      FirOrientation::kCrossCorrelation, FirBiasMode::kAdd, output, cache,
      stream);
  if (!status.ok())
    return status;
  return bf16_gated_elementwise(*output, x2, elements,
                                GatedActivation::kIdentity, output, stream);
}

Status bf16_hcm_prefill(const DeviceBuffer &x2, const DeviceBuffer &x1,
                        const DeviceBuffer &value, const DeviceBuffer &weight,
                        const DeviceBuffer &direct, const std::size_t length,
                        const std::size_t width,
                        const std::size_t filter_groups,
                        const std::size_t kernel_size, FirCache *const cache,
                        DeviceBuffer *const scratch, DeviceBuffer *const output,
                        FftWorkspace *const workspace, const Stream &stream) {
  if (scratch == nullptr || output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "HCM prefill requires scratch and output buffers"};
  }
  std::size_t elements = 0;
  if (!multiply(length, width, &elements)) {
    return {ErrorCode::kInvalidArgument, "HCM dimensions overflow"};
  }
  auto status = bf16_gated_elementwise(
      x1, value, elements, GatedActivation::kIdentity, scratch, stream);
  if (!status.ok())
    return status;
  status = bf16_fir_prefill_fft(
      *scratch, weight, &direct, length, width, filter_groups, kernel_size,
      FirOrientation::kCausalConvolution, FirBiasMode::kMultiplyInput, output,
      cache, workspace, stream);
  if (!status.ok())
    return status;
  return bf16_gated_elementwise(*output, x2, elements,
                                GatedActivation::kIdentity, output, stream);
}

Status bf16_hcm_decode(const DeviceBuffer &x2, const DeviceBuffer &x1,
                       const DeviceBuffer &value, const DeviceBuffer &weight,
                       const DeviceBuffer &direct, const std::size_t width,
                       const std::size_t filter_groups,
                       const std::size_t kernel_size, FirCache *const cache,
                       DeviceBuffer *const scratch, DeviceBuffer *const output,
                       const Stream &stream) {
  if (scratch == nullptr || output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "HCM decode requires scratch and output buffers"};
  }
  auto status = bf16_gated_elementwise(
      x1, value, width, GatedActivation::kIdentity, scratch, stream);
  if (!status.ok())
    return status;
  status = bf16_fir_decode(*scratch, weight, &direct, width, filter_groups,
                           kernel_size, FirOrientation::kCausalConvolution,
                           FirBiasMode::kMultiplyInput, cache, output, stream);
  if (!status.ok())
    return status;
  return bf16_gated_elementwise(*output, x2, width, GatedActivation::kIdentity,
                                output, stream);
}

Status bf16_hcm_continue(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, const DeviceBuffer &direct,
    const std::size_t length, const std::size_t width,
    const std::size_t filter_groups, const std::size_t kernel_size,
    FirCache *const cache, DeviceBuffer *const scratch,
    DeviceBuffer *const output, const Stream &stream) {
  if (scratch == nullptr || output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "HCM continuation requires scratch and output buffers"};
  }
  std::size_t elements = 0;
  if (!multiply(length, width, &elements)) {
    return {ErrorCode::kInvalidArgument,
            "HCM continuation dimensions overflow"};
  }
  auto status = bf16_gated_elementwise(
      x1, value, elements, GatedActivation::kIdentity, scratch, stream);
  if (!status.ok())
    return status;
  status = bf16_fir_continue_direct(
      *scratch, weight, &direct, length, width, filter_groups, kernel_size,
      FirOrientation::kCausalConvolution, FirBiasMode::kMultiplyInput, output,
      cache, stream);
  if (!status.ok())
    return status;
  return bf16_gated_elementwise(*output, x2, elements,
                                GatedActivation::kIdentity, output, stream);
}

Status bf16_hcl_prefill(const DeviceBuffer &x2, const DeviceBuffer &x1,
                        const DeviceBuffer &value, const DeviceBuffer &direct,
                        const DeviceBuffer &log_poles,
                        const DeviceBuffer &residues, const std::size_t length,
                        const std::size_t width, const std::size_t state_size,
                        const HclPrefillMode mode, IirCache *const cache,
                        DeviceBuffer *const scratch, DeviceBuffer *const output,
                        FftWorkspace *const workspace, const Stream &stream) {
  if (scratch == nullptr ||
      (mode != HclPrefillMode::kRecurrence &&
       mode != HclPrefillMode::kFft &&
       mode != HclPrefillMode::kRecurrenceContinue)) {
    return {ErrorCode::kInvalidArgument,
            "HCL prefill scratch or mode is invalid"};
  }
  auto status =
      validate_hcl_buffers(x2, x1, value, direct, log_poles, residues, length,
                           width, state_size, cache, output, stream);
  if (!status.ok())
    return status;
  std::size_t elements = 0;
  std::size_t tensor_bytes = 0;
  if (!multiply(length, width, &elements)) {
    return {ErrorCode::kInvalidArgument, "HCL dimensions overflow"};
  }
  status =
      bytes_for(elements, sizeof(__nv_bfloat16), &tensor_bytes, "HCL scratch");
  if (!status.ok())
    return status;
  status = buffer_size(*scratch, tensor_bytes, x2.device(), "HCL scratch");
  if (!status.ok())
    return status;
  status = select_device(x2.device());
  if (!status.ok())
    return status;
  if (mode != HclPrefillMode::kRecurrenceContinue) {
    status = cache->state.zero(stream);
    if (!status.ok())
      return status;
  }
  hcl_gate_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(x1.data()),
      static_cast<const __nv_bfloat16 *>(value.data()),
      static_cast<__nv_bfloat16 *>(scratch->data()), elements);
  status = launch_status("HCL input gate kernel");
  if (!status.ok())
    return status;

  if (mode != HclPrefillMode::kFft) {
    hcl_recurrence_prefill_kernel<<<grid_for(width), kThreads, 0,
                                    stream.get()>>>(
        static_cast<const __nv_bfloat16 *>(x2.data()),
        static_cast<const __nv_bfloat16 *>(scratch->data()),
        static_cast<const __nv_bfloat16 *>(direct.data()),
        static_cast<const float *>(log_poles.data()),
        static_cast<const float *>(residues.data()),
        static_cast<float *>(cache->state.data()),
        static_cast<__nv_bfloat16 *>(output->data()), length, width,
        state_size);
    return launch_status("HCL recurrence prefill kernel");
  }

  const std::size_t fft_size = fir_fft_size(length, length);
  if (workspace == nullptr || fft_size == 0 ||
      !workspace->matches(x2.device(), width, width, fft_size)) {
    return {ErrorCode::kInvalidArgument,
            "HCL FFT workspace dimensions do not match"};
  }
  std::size_t time_elements = 0;
  if (!multiply(width, fft_size, &time_elements)) {
    return {ErrorCode::kInvalidArgument, "HCL FFT dimensions overflow"};
  }
  pack_input_kernel<<<grid_for(time_elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(scratch->data()),
      static_cast<float *>(workspace->input_time_.data()), length, width,
      fft_size);
  status = launch_status("HCL FFT input pack kernel");
  if (!status.ok())
    return status;
  hcl_filter_kernel<<<grid_for(time_elements), kThreads, 0, stream.get()>>>(
      static_cast<const float *>(log_poles.data()),
      static_cast<const float *>(residues.data()),
      static_cast<float *>(workspace->filter_time_.data()), length, width,
      state_size, fft_size);
  status = launch_status("HCL modal filter kernel");
  if (!status.ok())
    return status;
  status = workspace->execute(stream);
  if (!status.ok())
    return status;
  unpack_hcl_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const float *>(workspace->output_time_.data()),
      static_cast<const __nv_bfloat16 *>(x2.data()),
      static_cast<const __nv_bfloat16 *>(scratch->data()),
      static_cast<const __nv_bfloat16 *>(direct.data()),
      static_cast<__nv_bfloat16 *>(output->data()), length, width, fft_size);
  status = launch_status("HCL FFT output kernel");
  if (!status.ok())
    return status;
  std::size_t state_elements = 0;
  if (!multiply(width, state_size, &state_elements)) {
    return {ErrorCode::kInvalidArgument, "HCL state dimensions overflow"};
  }
  hcl_state_prefill_kernel<<<grid_for(state_elements), kThreads, 0,
                             stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(scratch->data()),
      static_cast<const float *>(log_poles.data()),
      static_cast<float *>(cache->state.data()), length, width, state_size);
  return launch_status("HCL FFT state prefill kernel");
}

Status bf16_hcl_decode(const DeviceBuffer &x2, const DeviceBuffer &x1,
                       const DeviceBuffer &value, const DeviceBuffer &direct,
                       const DeviceBuffer &log_poles,
                       const DeviceBuffer &residues, const std::size_t width,
                       const std::size_t state_size, IirCache *const cache,
                       DeviceBuffer *const output, const Stream &stream) {
  auto status =
      validate_hcl_buffers(x2, x1, value, direct, log_poles, residues, 1, width,
                           state_size, cache, output, stream);
  if (!status.ok())
    return status;
  status = select_device(x2.device());
  if (!status.ok())
    return status;
  hcl_decode_kernel<<<grid_for(width), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(x2.data()),
      static_cast<const __nv_bfloat16 *>(x1.data()),
      static_cast<const __nv_bfloat16 *>(value.data()),
      static_cast<const __nv_bfloat16 *>(direct.data()),
      static_cast<const float *>(log_poles.data()),
      static_cast<const float *>(residues.data()),
      static_cast<float *>(cache->state.data()),
      static_cast<__nv_bfloat16 *>(output->data()), width, state_size);
  return launch_status("HCL decode kernel");
}

} // namespace evo2c::cuda
