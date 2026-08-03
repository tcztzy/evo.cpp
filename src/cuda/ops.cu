// SPDX-License-Identifier: Apache-2.0
#include "evo2c/cuda/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

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

Status required_bytes(const std::size_t elements,
                      const std::size_t element_size, std::size_t *const bytes,
                      const char *const name) {
  if (elements == 0 || !multiply(elements, element_size, bytes)) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " size is zero or overflows"};
  }
  return Status::Ok();
}

Status buffer_size(const DeviceBuffer &buffer, const std::size_t expected,
                   const int device, const char *const name) {
  if (!buffer.valid()) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " is not allocated"};
  }
  if (buffer.device() != device) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " is on a different CUDA device"};
  }
  if (buffer.bytes() < expected) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " has " + std::to_string(buffer.bytes()) +
                " bytes; requires at least " + std::to_string(expected)};
  }
  return Status::Ok();
}

Status launch_status(const char *const operation) {
  return cuda_status(cudaPeekAtLastError(), operation);
}

unsigned int grid_for(const std::size_t elements) {
  const auto blocks = (elements + static_cast<std::size_t>(kThreads) - 1) /
                      static_cast<std::size_t>(kThreads);
  return static_cast<unsigned int>(std::min<std::size_t>(blocks, 65535));
}

__inline__ __device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__global__ void rms_norm_kernel(const __nv_bfloat16 *const input,
                                const float *const scale,
                                __nv_bfloat16 *const output,
                                const std::size_t width, const float epsilon) {
  const auto row = static_cast<std::size_t>(blockIdx.x);
  float sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(input[row * width + column]);
    sum += value * value;
  }
  sum = warp_sum(sum);
  __shared__ float partial[32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  if (lane == 0)
    partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < (blockDim.x + 31) / 32 ? partial[lane] : 0.0F;
    block_sum = warp_sum(block_sum);
    if (lane == 0) {
      // Match Vortex's BF16 expression:
      // x.norm(2) * width**-0.5 + eps.
      const __nv_bfloat16 norm = __float2bfloat16_rn(sqrtf(block_sum));
      const __nv_bfloat16 scaled = __float2bfloat16_rn(
          __bfloat162float(norm) * rsqrtf(static_cast<float>(width)));
      const __nv_bfloat16 denominator =
          __float2bfloat16_rn(__bfloat162float(scaled) + epsilon);
      partial[0] = __bfloat162float(denominator);
    }
  }
  __syncthreads();
  const float denominator = partial[0];
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const float value = __bfloat162float(input[row * width + column]);
    const float bf16_scale =
        __bfloat162float(__float2bfloat16_rn(scale[column]));
    const __nv_bfloat16 normalized = __float2bfloat16_rn(value / denominator);
    output[row * width + column] =
        __float2bfloat16_rn(__bfloat162float(normalized) * bf16_scale);
  }
}

__device__ __forceinline__ __nv_bfloat16
pytorch_gelu_bf16(const __nv_bfloat16 input) {
  const float value = __bfloat162float(input);
  constexpr float kAlpha = 0.70710678118654752440F;
  const __nv_bfloat16 computed =
      __float2bfloat16_rn(value * 0.5F * (1.0F + ::erf(value * kAlpha)));
  // Exhaustive comparison of all 65,536 BF16 inputs found seven finite cases
  // where CUDA 12.8's libdevice erf rounds GELU differently from the CUDA
  // 13.3 libdevice used by pinned PyTorch.  Canonicalize those output bits so
  // exact inference is independent of the build toolkit.
  switch (__bfloat16_as_ushort(input)) {
  case 0xc049:
    return __ushort_as_bfloat16(0xbb2d);
  case 0xc081:
    return __ushort_as_bfloat16(0xb8eb);
  case 0xc089:
    return __ushort_as_bfloat16(0xb827);
  case 0xc092:
    return __ushort_as_bfloat16(0xb742);
  case 0xc0a0:
    return __ushort_as_bfloat16(0xb5c8);
  case 0xc0a8:
    return __ushort_as_bfloat16(0xb4fc);
  case 0xc0ab:
    return __ushort_as_bfloat16(0xb4ab);
  default:
    return computed;
  }
}

__global__ void gated_kernel(const __nv_bfloat16 *const first,
                             const __nv_bfloat16 *const second,
                             __nv_bfloat16 *const output,
                             const std::size_t elements, const bool gelu) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    float value = __bfloat162float(first[index]);
    if (gelu) {
      // PyTorch evaluates F.gelu(z1) and the following multiplication as two
      // distinct BF16 operations. Preserve the intermediate BF16 rounding.
      value = __bfloat162float(pytorch_gelu_bf16(first[index]));
    }
    output[index] =
        __float2bfloat16_rn(value * __bfloat162float(second[index]));
  }
}

__global__ void gelu_kernel(const __nv_bfloat16 *const input,
                            __nv_bfloat16 *const output,
                            const std::size_t elements) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = pytorch_gelu_bf16(input[index]);
  }
}

__global__ void add_kernel(__nv_bfloat16 *const output,
                           const __nv_bfloat16 *const residual,
                           const std::size_t elements) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = __float2bfloat16_rn(__bfloat162float(output[index]) +
                                        __bfloat162float(residual[index]));
  }
}

__global__ void add_bias_kernel(__nv_bfloat16 *const output,
                                const __nv_bfloat16 *const bias,
                                const std::size_t elements,
                                const std::size_t width) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    output[index] = __float2bfloat16_rn(__bfloat162float(output[index]) +
                                        __bfloat162float(bias[index % width]));
  }
}

__global__ void row_to_column_major_kernel(const __nv_bfloat16 *const input,
                                           __nv_bfloat16 *const output,
                                           const std::size_t rows,
                                           const std::size_t columns) {
  constexpr unsigned int tile_width = 32;
  constexpr unsigned int block_rows = 8;
  __shared__ __nv_bfloat16 tile[tile_width][tile_width + 1];

  const std::size_t input_column =
      static_cast<std::size_t>(blockIdx.x) * tile_width + threadIdx.x;
  const std::size_t input_row =
      static_cast<std::size_t>(blockIdx.y) * tile_width + threadIdx.y;
  for (unsigned int offset = 0; offset < tile_width; offset += block_rows) {
    if (input_column < columns && input_row + offset < rows) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[(input_row + offset) * columns + input_column];
    }
  }
  __syncthreads();

  const std::size_t output_row =
      static_cast<std::size_t>(blockIdx.x) * tile_width + threadIdx.y;
  const std::size_t output_column =
      static_cast<std::size_t>(blockIdx.y) * tile_width + threadIdx.x;
  for (unsigned int offset = 0; offset < tile_width; offset += block_rows) {
    if (output_row + offset < columns && output_column < rows) {
      output[(output_row + offset) * rows + output_column] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

__global__ void software_lowp_kernel(const __nv_bfloat16 *const input,
                                     std::uint8_t *const codes,
                                     float *const dequantized,
                                     const std::size_t elements,
                                     const float scale) {
  const float scale_inverse = __frcp_rn(scale);
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const float input_value = __bfloat162float(input[index]);
    const __nv_fp8_e4m3 quantized(input_value * scale);
    if (codes != nullptr)
      codes[index] = quantized.__x;
    dequantized[index] = static_cast<float>(quantized) * scale_inverse;
  }
}

__global__ void software_lowp_codes_kernel(const __nv_bfloat16 *const input,
                                           std::uint8_t *const codes,
                                           const std::size_t elements,
                                           const float scale) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const float input_value = __bfloat162float(input[index]);
    const __nv_fp8_e4m3 quantized(input_value * scale);
    codes[index] = quantized.__x;
  }
}

__global__ void software_lowp_f32_codes_kernel(const float *const input,
                                               std::uint8_t *const codes,
                                               const std::size_t elements,
                                               const float scale) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const __nv_fp8_e4m3 quantized(input[index] * scale);
    codes[index] = quantized.__x;
  }
}

__device__ int normal_exponent(const float value) {
  return static_cast<int>((__float_as_uint(fabsf(value)) >> 23U) & 0xffU) - 127;
}

__device__ std::int32_t align_significand(const float value,
                                          const int operand_exponent,
                                          const int maximum_exponent,
                                          const bool denormalized_product) {
  if (value == 0.0F)
    return 0;
  const std::uint32_t bits = __float_as_uint(value);
  const std::uint32_t magnitude = bits & 0x7fffffffU;
  std::uint32_t significand = (magnitude & 0x7fffffU) | 0x800000U;
  if (denormalized_product) {
    const int normalized_exponent =
        static_cast<int>((magnitude >> 23U) & 0xffU) - 127;
    significand <<=
        static_cast<unsigned int>(normalized_exponent - operand_exponent);
  }
  significand >>= 10U;
  const int shift = maximum_exponent - operand_exponent;
  if (shift > 31)
    significand = 0;
  else if (shift > 0)
    significand >>= static_cast<unsigned int>(shift);
  const auto aligned = static_cast<std::int32_t>(significand);
  return (bits & 0x80000000U) == 0U ? aligned : -aligned;
}

__device__ float truncate_e8m13(const float value) {
  return __uint_as_float(__float_as_uint(value) & 0xfffffc00U);
}

__device__ std::uint16_t pack_e4m3_integer(const std::uint8_t code) {
  const auto magnitude = static_cast<std::uint8_t>(code & 0x7fU);
  const auto exponent_field =
      static_cast<unsigned int>((magnitude >> 3U) & 0x0fU);
  const auto mantissa = static_cast<unsigned int>(magnitude & 0x07U);
  int exponent = 0;
  std::uint32_t significand = 0;
  if (exponent_field == 0U) {
    if (mantissa == 0U)
      return 0U;
    const int leading_bit = 31 - __clz(mantissa);
    exponent = -9 + leading_bit;
    significand = mantissa << static_cast<unsigned int>(3 - leading_bit);
  } else {
    exponent = static_cast<int>(exponent_field) - 7;
    significand = 8U + mantissa;
  }
  constexpr int exponent_bias = 16;
  return static_cast<std::uint16_t>(
      significand |
      (static_cast<std::uint32_t>(exponent + exponent_bias) << 5U) |
      ((code & 0x80U) != 0U ? 0x0400U : 0U));
}

__device__ std::int32_t aligned_e4m3_product(const std::uint16_t left,
                                             const std::uint16_t right,
                                             const int maximum_exponent) {
  constexpr int twice_exponent_bias = 32;
  const int exponent = static_cast<int>((left >> 5U) & 0x1fU) +
                       static_cast<int>((right >> 5U) & 0x1fU) -
                       twice_exponent_bias;
  const int shift = maximum_exponent - exponent;
  if (shift > 31)
    return 0;
  std::uint32_t value = static_cast<std::uint32_t>(left & 0x1fU) *
                        static_cast<std::uint32_t>(right & 0x1fU) * 128U;
  if (shift > 0)
    value >>= static_cast<unsigned int>(shift);
  const auto aligned = static_cast<std::int32_t>(value);
  return ((left ^ right) & 0x0400U) == 0U ? aligned : -aligned;
}

template <unsigned int TileRows, unsigned int TileColumns>
__global__ void software_h100_qgmma_tiled_kernel(
    const std::uint8_t *const input_codes,
    const std::uint8_t *const weight_codes, __nv_bfloat16 *const output,
    const std::size_t rows, const std::size_t in_features,
    const std::size_t out_features, const float output_scale) {
  constexpr unsigned int tile_inner = 32;
  constexpr unsigned int threads = TileRows * TileColumns;
  static_assert((TileRows == 1 && TileColumns == 32) ||
                (TileRows == 16 && TileColumns == 16));
  static_assert(threads <= 256);

  __shared__ std::uint16_t input_tile[TileRows][tile_inner];
  // K-major plus one padded column makes a warp's fixed-K column scan
  // contiguous and avoids the 16-way bank conflict of [column][K].
  __shared__ std::uint16_t weight_tile[tile_inner][TileColumns + 1];
  const unsigned int thread = threadIdx.x;
  const unsigned int local_row = thread / TileColumns;
  const unsigned int local_column = thread % TileColumns;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.y) * TileRows + local_row;
  const std::size_t column =
      static_cast<std::size_t>(blockIdx.x) * TileColumns + local_column;
  const bool valid_output = row < rows && column < out_features;
  float accumulator = 0.0F;

  for (std::size_t inner_start = 0; inner_start < in_features;
       inner_start += tile_inner) {
    for (unsigned int load = thread; load < TileRows * tile_inner;
         load += threads) {
      const unsigned int load_row = load / tile_inner;
      const unsigned int load_inner = load % tile_inner;
      const std::size_t global_row =
          static_cast<std::size_t>(blockIdx.y) * TileRows + load_row;
      input_tile[load_row][load_inner] = pack_e4m3_integer(
          global_row < rows
              ? input_codes[global_row * in_features + inner_start + load_inner]
              : 0U);
    }
    for (unsigned int load = thread; load < TileColumns * tile_inner;
         load += threads) {
      const unsigned int load_column = load / tile_inner;
      const unsigned int load_inner = load % tile_inner;
      const std::size_t global_column =
          static_cast<std::size_t>(blockIdx.x) * TileColumns + load_column;
      weight_tile[load_inner][load_column] =
          pack_e4m3_integer(global_column < out_features
                                ? weight_codes[global_column * in_features +
                                               inner_start + load_inner]
                                : 0U);
    }
    __syncthreads();

    if (valid_output) {
      int maximum_exponent =
          accumulator == 0.0F ? -1024 : normal_exponent(accumulator);
#pragma unroll
      for (unsigned int inner = 0; inner < tile_inner; ++inner) {
        const std::uint16_t left = input_tile[local_row][inner];
        const std::uint16_t right = weight_tile[inner][local_column];
        if ((left & 0x1fU) != 0U && (right & 0x1fU) != 0U)
          maximum_exponent =
              max(maximum_exponent,
                  static_cast<int>((left >> 5U) & 0x1fU) +
                      static_cast<int>((right >> 5U) & 0x1fU) - 32);
      }

      std::int32_t aligned_sum =
          accumulator == 0.0F
              ? 0
              : align_significand(accumulator, normal_exponent(accumulator),
                                  maximum_exponent, false);
#pragma unroll
      for (unsigned int inner = 0; inner < tile_inner; ++inner) {
        const std::uint16_t left = input_tile[local_row][inner];
        const std::uint16_t right = weight_tile[inner][local_column];
        if ((left & 0x1fU) != 0U && (right & 0x1fU) != 0U)
          aligned_sum += aligned_e4m3_product(left, right, maximum_exponent);
      }
      accumulator = aligned_sum == 0
                        ? 0.0F
                        : truncate_e8m13(ldexpf(static_cast<float>(aligned_sum),
                                                maximum_exponent - 13));
    }
    __syncthreads();
  }
  if (valid_output)
    output[row * out_features + column] =
        __float2bfloat16_rn(accumulator * output_scale);
}

struct MatmulObjects final {
  cublasLtMatmulDesc_t operation{nullptr};
  cublasLtMatrixLayout_t input{nullptr};
  cublasLtMatrixLayout_t weight{nullptr};
  cublasLtMatrixLayout_t output{nullptr};
  cublasLtMatmulPreference_t preference{nullptr};

  ~MatmulObjects() {
    if (preference != nullptr)
      static_cast<void>(cublasLtMatmulPreferenceDestroy(preference));
    if (output != nullptr)
      static_cast<void>(cublasLtMatrixLayoutDestroy(output));
    if (weight != nullptr)
      static_cast<void>(cublasLtMatrixLayoutDestroy(weight));
    if (input != nullptr)
      static_cast<void>(cublasLtMatrixLayoutDestroy(input));
    if (operation != nullptr)
      static_cast<void>(cublasLtMatmulDescDestroy(operation));
  }
};

} // namespace

struct Bf16LinearPlan::Impl final {
  MatmulObjects objects;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int device{-1};
  std::size_t rows{0};
  std::size_t in_features{0};
  std::size_t out_features{0};
  std::size_t workspace_bytes{0};
  bool has_bias{false};
  LinearWeightLayout weight_layout{LinearWeightLayout::kOutputMajor};
  LinearInputLayout input_layout{LinearInputLayout::kRowMajor};

  [[nodiscard]] bool
  matches(const int expected_device, const std::size_t expected_rows,
          const std::size_t expected_in_features,
          const std::size_t expected_out_features,
          const std::size_t expected_workspace_bytes, const bool expected_bias,
          const LinearWeightLayout expected_weight_layout,
          const LinearInputLayout expected_input_layout) const noexcept {
    return device == expected_device && rows == expected_rows &&
           in_features == expected_in_features &&
           out_features == expected_out_features &&
           workspace_bytes == expected_workspace_bytes &&
           has_bias == expected_bias &&
           weight_layout == expected_weight_layout &&
           input_layout == expected_input_layout;
  }
};

Bf16LinearPlan::Bf16LinearPlan() = default;
Bf16LinearPlan::~Bf16LinearPlan() = default;
Bf16LinearPlan::Bf16LinearPlan(Bf16LinearPlan &&other) noexcept
    : impl_(std::move(other.impl_)),
      build_count_(std::exchange(other.build_count_, 0)) {}
Bf16LinearPlan &Bf16LinearPlan::operator=(Bf16LinearPlan &&other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
    build_count_ = std::exchange(other.build_count_, 0);
  }
  return *this;
}

Status MlpWorkspace::allocate(const int device, const std::size_t rows,
                              const std::size_t inner_width,
                              const std::size_t blas_bytes) {
  std::size_t elements = 0;
  std::size_t temporary_bytes = 0;
  if (!multiply(rows, inner_width, &elements)) {
    return {ErrorCode::kInvalidArgument, "MLP workspace dimensions overflow"};
  }
  auto status = required_bytes(elements, sizeof(__nv_bfloat16),
                               &temporary_bytes, "MLP workspace");
  if (!status.ok())
    return status;
  status = first.allocate(device, temporary_bytes);
  if (!status.ok())
    return status;
  status = second.allocate(device, temporary_bytes);
  if (!status.ok())
    return status;
  status = gated.allocate(device, temporary_bytes);
  if (!status.ok())
    return status;
  status = activated.allocate(device, temporary_bytes);
  if (!status.ok())
    return status;
  return blas.allocate(device, blas_bytes);
}

Status bf16_linear(const BlasLt &handle, const DeviceBuffer &input,
                   const DeviceBuffer &weight, const DeviceBuffer *const bias,
                   const std::size_t rows, const std::size_t in_features,
                   const std::size_t out_features, DeviceBuffer *const output,
                   DeviceBuffer *const workspace, const Stream &stream,
                   Bf16LinearPlan *const plan,
                   const LinearWeightLayout weight_layout,
                   const LinearInputLayout input_layout) {
  if (!handle.valid() || output == nullptr || workspace == nullptr ||
      !stream.valid() ||
      (weight_layout != LinearWeightLayout::kOutputMajor &&
       weight_layout != LinearWeightLayout::kInputMajor) ||
      (input_layout != LinearInputLayout::kRowMajor &&
       input_layout != LinearInputLayout::kColumnMajor)) {
    return {ErrorCode::kInvalidArgument,
            "bf16_linear requires a handle, output, workspace, and stream"};
  }
  if (rows >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      in_features >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      out_features >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return {ErrorCode::kInvalidArgument, "bf16_linear dimensions exceed int64"};
  }
  std::size_t input_elements = 0;
  std::size_t weight_elements = 0;
  std::size_t output_elements = 0;
  if (!multiply(rows, in_features, &input_elements) ||
      !multiply(out_features, in_features, &weight_elements) ||
      !multiply(rows, out_features, &output_elements)) {
    return {ErrorCode::kInvalidArgument, "bf16_linear dimensions overflow"};
  }
  std::size_t input_bytes = 0;
  std::size_t weight_bytes = 0;
  std::size_t output_bytes = 0;
  auto status = required_bytes(input_elements, sizeof(__nv_bfloat16),
                               &input_bytes, "linear input");
  if (!status.ok())
    return status;
  status = required_bytes(weight_elements, sizeof(__nv_bfloat16), &weight_bytes,
                          "linear weight");
  if (!status.ok())
    return status;
  status = required_bytes(output_elements, sizeof(__nv_bfloat16), &output_bytes,
                          "linear output");
  if (!status.ok())
    return status;
  const int device = input.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "linear stream is on a different CUDA device"};
  }
  status = buffer_size(input, input_bytes, device, "linear input");
  if (!status.ok())
    return status;
  status = buffer_size(weight, weight_bytes, device, "linear weight");
  if (!status.ok())
    return status;
  status = buffer_size(*output, output_bytes, device, "linear output");
  if (!status.ok())
    return status;
  status = buffer_size(*workspace, 1, device, "linear workspace");
  if (!status.ok())
    return status;
  if (bias != nullptr) {
    std::size_t bias_bytes = 0;
    status = required_bytes(out_features, sizeof(__nv_bfloat16), &bias_bytes,
                            "linear bias");
    if (!status.ok())
      return status;
    status = buffer_size(*bias, bias_bytes, device, "linear bias");
    if (!status.ok())
      return status;
  }
  status = select_device(device);
  if (!status.ok())
    return status;

  const auto workspace_bytes = workspace->bytes();
  Bf16LinearPlan temporary_plan;
  Bf16LinearPlan *const active_plan = plan == nullptr ? &temporary_plan : plan;
  if (active_plan->impl_ == nullptr ||
      !active_plan->impl_->matches(device, rows, in_features, out_features,
                                   workspace_bytes, bias != nullptr,
                                   weight_layout, input_layout)) {
    auto candidate = std::make_unique<Bf16LinearPlan::Impl>();
    auto &objects = candidate->objects;
    status =
        cublas_status(cublasLtMatmulDescCreate(&objects.operation,
                                               CUBLAS_COMPUTE_32F, CUDA_R_32F),
                      "cublasLtMatmulDescCreate");
    if (!status.ok())
      return status;
    if (input_layout == LinearInputLayout::kColumnMajor) {
      const cublasOperation_t input_transpose = CUBLAS_OP_T;
      status = cublas_status(cublasLtMatmulDescSetAttribute(
                                 objects.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &input_transpose, sizeof(input_transpose)),
                             "cublasLtMatmulDescSetAttribute TRANSB");
      if (!status.ok())
        return status;
    }
    // Interpret row-major weight [N,K], input [M,K], and output [M,N] as
    // column-major [K,N], [K,M], and [N,M]. This is the native cuBLASLt layout
    // used by PyTorch and supports a fused BF16 bias epilogue.
    const cublasOperation_t transpose =
        weight_layout == LinearWeightLayout::kOutputMajor ? CUBLAS_OP_T
                                                          : CUBLAS_OP_N;
    status = cublas_status(cublasLtMatmulDescSetAttribute(
                               objects.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                               &transpose, sizeof(transpose)),
                           "cublasLtMatmulDescSetAttribute TRANSA");
    if (!status.ok())
      return status;
    if (bias != nullptr) {
      const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      status =
          cublas_status(cublasLtMatmulDescSetAttribute(
                            objects.operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                            &epilogue, sizeof(epilogue)),
                        "cublasLtMatmulDescSetAttribute EPILOGUE");
      if (!status.ok())
        return status;
    }
    status =
        input_layout == LinearInputLayout::kRowMajor
            ? cublas_status(cublasLtMatrixLayoutCreate(&objects.input,
                                                       CUDA_R_16BF, in_features,
                                                       rows, in_features),
                            "cublasLtMatrixLayoutCreate row-major input")
            : cublas_status(cublasLtMatrixLayoutCreate(&objects.input,
                                                       CUDA_R_16BF, rows,
                                                       in_features, rows),
                            "cublasLtMatrixLayoutCreate column-major input");
    if (!status.ok())
      return status;
    status = weight_layout == LinearWeightLayout::kOutputMajor
                 ? cublas_status(
                       cublasLtMatrixLayoutCreate(&objects.weight, CUDA_R_16BF,
                                                  in_features, out_features,
                                                  in_features),
                       "cublasLtMatrixLayoutCreate output-major weight view")
                 : cublas_status(
                       cublasLtMatrixLayoutCreate(&objects.weight, CUDA_R_16BF,
                                                  out_features, in_features,
                                                  out_features),
                       "cublasLtMatrixLayoutCreate input-major weight view");
    if (!status.ok())
      return status;
    status = cublas_status(cublasLtMatrixLayoutCreate(&objects.output,
                                                      CUDA_R_16BF, out_features,
                                                      rows, out_features),
                           "cublasLtMatrixLayoutCreate output transpose view");
    if (!status.ok())
      return status;
    status = cublas_status(cublasLtMatmulPreferenceCreate(&objects.preference),
                           "cublasLtMatmulPreferenceCreate");
    if (!status.ok())
      return status;
    status = cublas_status(cublasLtMatmulPreferenceSetAttribute(
                               objects.preference,
                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                               &workspace_bytes, sizeof(workspace_bytes)),
                           "cublasLtMatmulPreferenceSetAttribute workspace");
    if (!status.ok())
      return status;
    int returned = 0;
    status = cublas_status(cublasLtMatmulAlgoGetHeuristic(
                               handle.get(), objects.operation, objects.weight,
                               objects.input, objects.output, objects.output,
                               objects.preference, 1, &candidate->heuristic,
                               &returned),
                           "cublasLtMatmulAlgoGetHeuristic");
    if (!status.ok())
      return status;
    if (returned == 0) {
      return {ErrorCode::kCuda,
              "cuBLASLt found no BF16 row-major matmul algorithm"};
    }
    candidate->device = device;
    candidate->rows = rows;
    candidate->in_features = in_features;
    candidate->out_features = out_features;
    candidate->workspace_bytes = workspace_bytes;
    candidate->has_bias = bias != nullptr;
    candidate->weight_layout = weight_layout;
    candidate->input_layout = input_layout;
    active_plan->impl_ = std::move(candidate);
    ++active_plan->build_count_;
  }
  auto &cached = *active_plan->impl_;
  auto &objects = cached.objects;
  if (bias != nullptr) {
    const void *const bias_pointer = bias->data();
    status =
        cublas_status(cublasLtMatmulDescSetAttribute(
                          objects.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                          &bias_pointer, sizeof(bias_pointer)),
                      "cublasLtMatmulDescSetAttribute BIAS_POINTER");
    if (!status.ok())
      return status;
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  status = cublas_status(
      cublasLtMatmul(handle.get(), objects.operation, &alpha, weight.data(),
                     objects.weight, input.data(), objects.input, &beta,
                     output->data(), objects.output, output->data(),
                     objects.output, &cached.heuristic.algo, workspace->data(),
                     workspace_bytes, stream.get()),
      "cublasLtMatmul");
  return status;
}

Status bf16_row_to_column_major(const DeviceBuffer &input,
                                const std::size_t rows,
                                const std::size_t columns,
                                DeviceBuffer *const output,
                                const Stream &stream) {
  if (output == nullptr || !stream.valid()) {
    return {ErrorCode::kInvalidArgument,
            "bf16_row_to_column_major requires output and stream"};
  }
  std::size_t elements = 0;
  std::size_t bytes = 0;
  if (!multiply(rows, columns, &elements)) {
    return {ErrorCode::kInvalidArgument,
            "bf16_row_to_column_major dimensions overflow"};
  }
  auto status = required_bytes(elements, sizeof(__nv_bfloat16), &bytes,
                               "bf16 row-to-column-major");
  if (!status.ok())
    return status;
  const int device = input.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "bf16_row_to_column_major stream is on a different CUDA device"};
  }
  status = buffer_size(input, bytes, device, "bf16 row-to-column-major input");
  if (!status.ok())
    return status;
  status =
      buffer_size(*output, bytes, device, "bf16 row-to-column-major output");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  constexpr unsigned int tile_width = 32;
  constexpr unsigned int block_rows = 8;
  const dim3 blocks(
      static_cast<unsigned int>((columns + tile_width - 1) / tile_width),
      static_cast<unsigned int>((rows + tile_width - 1) / tile_width));
  const dim3 threads(tile_width, block_rows);
  row_to_column_major_kernel<<<blocks, threads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<__nv_bfloat16 *>(output->data()), rows, columns);
  return launch_status("row_to_column_major_kernel");
}

Status software_e4m3_quantize_bf16(const DeviceBuffer &input,
                                   const std::size_t elements,
                                   const float scale, DeviceBuffer *const codes,
                                   DeviceBuffer *const dequantized_f32,
                                   const Stream &stream) {
  if (dequantized_f32 == nullptr || !stream.valid() || !std::isfinite(scale) ||
      scale <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "software_e4m3_quantize_bf16 received invalid arguments"};
  }
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  auto status = required_bytes(elements, sizeof(__nv_bfloat16), &input_bytes,
                               "software E4M3 input");
  if (!status.ok())
    return status;
  status = required_bytes(elements, sizeof(float), &output_bytes,
                          "software E4M3 dequantized output");
  if (!status.ok())
    return status;
  const int device = input.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "software E4M3 stream is on a different CUDA device"};
  }
  status = buffer_size(input, input_bytes, device, "software E4M3 input");
  if (!status.ok())
    return status;
  status = buffer_size(*dequantized_f32, output_bytes, device,
                       "software E4M3 dequantized output");
  if (!status.ok())
    return status;
  if (codes != nullptr) {
    status = buffer_size(*codes, elements, device, "software E4M3 codes");
    if (!status.ok())
      return status;
  }
  status = select_device(device);
  if (!status.ok())
    return status;
  software_lowp_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      codes == nullptr ? nullptr : static_cast<std::uint8_t *>(codes->data()),
      static_cast<float *>(dequantized_f32->data()), elements, scale);
  return launch_status("software_lowp_kernel");
}

Status software_e4m3_quantize_bf16_codes(const DeviceBuffer &input,
                                         const std::size_t elements,
                                         const float scale,
                                         DeviceBuffer *const codes,
                                         const Stream &stream) {
  if (codes == nullptr || !stream.valid() || !std::isfinite(scale) ||
      scale <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "software_e4m3_quantize_bf16_codes received invalid arguments"};
  }
  std::size_t input_bytes = 0;
  auto status = required_bytes(elements, sizeof(__nv_bfloat16), &input_bytes,
                               "software E4M3 code input");
  if (!status.ok())
    return status;
  const int device = input.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "software E4M3 payload stream is on a different CUDA device"};
  }
  status = buffer_size(input, input_bytes, device, "software E4M3 code input");
  if (!status.ok())
    return status;
  status = buffer_size(*codes, elements, device, "software E4M3 code output");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  software_lowp_codes_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<std::uint8_t *>(codes->data()), elements, scale);
  return launch_status("software_lowp_codes_kernel");
}

Status software_e4m3_quantize_f32_codes(const DeviceBuffer &input,
                                        const std::size_t elements,
                                        const float scale,
                                        DeviceBuffer *const codes,
                                        const Stream &stream) {
  if (codes == nullptr || !stream.valid() || !std::isfinite(scale) ||
      scale <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "software_e4m3_quantize_f32_codes received invalid arguments"};
  }
  std::size_t input_bytes = 0;
  auto status = required_bytes(elements, sizeof(float), &input_bytes,
                               "software E4M3 F32 code input");
  if (!status.ok())
    return status;
  const int device = input.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "software E4M3 F32 stream is on a different CUDA device"};
  }
  status =
      buffer_size(input, input_bytes, device, "software E4M3 F32 code input");
  if (!status.ok())
    return status;
  status =
      buffer_size(*codes, elements, device, "software E4M3 F32 code output");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  software_lowp_f32_codes_kernel<<<grid_for(elements), kThreads, 0,
                                   stream.get()>>>(
      static_cast<const float *>(input.data()),
      static_cast<std::uint8_t *>(codes->data()), elements, scale);
  return launch_status("software_lowp_f32_codes_kernel");
}

Status software_e4m3_h100_linear(
    const DeviceBuffer &input, const DeviceBuffer &weight_codes,
    const std::size_t rows, const std::size_t in_features,
    const std::size_t out_features, const float input_scale,
    const float output_scale, DeviceBuffer *const input_codes_workspace,
    DeviceBuffer *const output, const Stream &stream) {
  if (input_codes_workspace == nullptr || output == nullptr ||
      !stream.valid() || rows == 0 || in_features == 0 ||
      in_features % 32U != 0 || out_features == 0 ||
      !std::isfinite(input_scale) || input_scale <= 0.0F ||
      !std::isfinite(output_scale) || output_scale <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "software_e4m3_h100_linear received invalid arguments"};
  }
  std::size_t input_elements = 0;
  std::size_t weight_elements = 0;
  std::size_t output_elements = 0;
  if (!multiply(rows, in_features, &input_elements) ||
      !multiply(out_features, in_features, &weight_elements) ||
      !multiply(rows, out_features, &output_elements)) {
    return {ErrorCode::kInvalidArgument,
            "software E4M3 linear dimensions overflow"};
  }
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
  auto status = required_bytes(input_elements, sizeof(__nv_bfloat16),
                               &input_bytes, "software E4M3 linear input");
  if (!status.ok())
    return status;
  status = required_bytes(output_elements, sizeof(__nv_bfloat16), &output_bytes,
                          "software E4M3 linear output");
  if (!status.ok())
    return status;
  const int device = input.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "software E4M3 linear stream is on a different CUDA device"};
  }
  status =
      buffer_size(input, input_bytes, device, "software E4M3 linear input");
  if (!status.ok())
    return status;
  status = buffer_size(weight_codes, weight_elements, device,
                       "software E4M3 linear weight codes");
  if (!status.ok())
    return status;
  status = buffer_size(*input_codes_workspace, input_elements, device,
                       "software E4M3 linear input-code workspace");
  if (!status.ok())
    return status;
  status =
      buffer_size(*output, output_bytes, device, "software E4M3 linear output");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  software_lowp_codes_kernel<<<grid_for(input_elements), kThreads, 0,
                               stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<std::uint8_t *>(input_codes_workspace->data()),
      input_elements, input_scale);
  auto status_launch = launch_status("software QGMMA input quantization");
  if (!status_launch.ok())
    return status_launch;
  constexpr unsigned int decode_columns = 32;
  constexpr unsigned int prefill_rows = 16;
  constexpr unsigned int prefill_columns = 16;
  const dim3 grid(
      static_cast<unsigned int>(
          (out_features + (rows == 1 ? decode_columns : prefill_columns) - 1U) /
          (rows == 1 ? decode_columns : prefill_columns)),
      static_cast<unsigned int>(
          rows == 1 ? 1 : (rows + prefill_rows - 1U) / prefill_rows));
  if (rows == 1) {
    software_h100_qgmma_tiled_kernel<1, decode_columns>
        <<<grid, decode_columns, 0, stream.get()>>>(
            static_cast<const std::uint8_t *>(input_codes_workspace->data()),
            static_cast<const std::uint8_t *>(weight_codes.data()),
            static_cast<__nv_bfloat16 *>(output->data()), rows, in_features,
            out_features, output_scale);
  } else {
    software_h100_qgmma_tiled_kernel<prefill_rows, prefill_columns>
        <<<grid, prefill_rows * prefill_columns, 0, stream.get()>>>(
            static_cast<const std::uint8_t *>(input_codes_workspace->data()),
            static_cast<const std::uint8_t *>(weight_codes.data()),
            static_cast<__nv_bfloat16 *>(output->data()), rows, in_features,
            out_features, output_scale);
  }
  return launch_status("software_h100_qgmma_tiled_kernel");
}

Status bf16_rms_norm(const DeviceBuffer &input, const DeviceBuffer &scale,
                     const std::size_t rows, const std::size_t width,
                     const float epsilon, DeviceBuffer *const output,
                     const Stream &stream) {
  if (output == nullptr || !stream.valid() || !std::isfinite(epsilon) ||
      epsilon <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "bf16_rms_norm received invalid arguments"};
  }
  if (rows > std::numeric_limits<unsigned int>::max()) {
    return {ErrorCode::kInvalidArgument,
            "bf16_rms_norm row count exceeds CUDA grid limit"};
  }
  std::size_t elements = 0;
  if (!multiply(rows, width, &elements)) {
    return {ErrorCode::kInvalidArgument, "bf16_rms_norm dimensions overflow"};
  }
  std::size_t tensor_bytes = 0;
  std::size_t scale_bytes = 0;
  auto status = required_bytes(elements, sizeof(__nv_bfloat16), &tensor_bytes,
                               "RMSNorm tensor");
  if (!status.ok())
    return status;
  status = required_bytes(width, sizeof(float), &scale_bytes, "RMSNorm scale");
  if (!status.ok())
    return status;
  const int device = input.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "RMSNorm stream is on a different CUDA device"};
  }
  status = buffer_size(input, tensor_bytes, device, "RMSNorm input");
  if (!status.ok())
    return status;
  status = buffer_size(scale, scale_bytes, device, "RMSNorm scale");
  if (!status.ok())
    return status;
  status = buffer_size(*output, tensor_bytes, device, "RMSNorm output");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  rms_norm_kernel<<<static_cast<unsigned int>(rows), kThreads, 0,
                    stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<const float *>(scale.data()),
      static_cast<__nv_bfloat16 *>(output->data()), width, epsilon);
  return launch_status("rms_norm_kernel");
}

Status bf16_gated_elementwise(const DeviceBuffer &first,
                              const DeviceBuffer &second,
                              const std::size_t elements,
                              const GatedActivation activation,
                              DeviceBuffer *const output,
                              const Stream &stream) {
  if (output == nullptr || !stream.valid() ||
      (activation != GatedActivation::kGelu &&
       activation != GatedActivation::kIdentity)) {
    return {ErrorCode::kInvalidArgument,
            "bf16_gated_elementwise received invalid arguments"};
  }
  std::size_t bytes = 0;
  auto status =
      required_bytes(elements, sizeof(__nv_bfloat16), &bytes, "gated tensor");
  if (!status.ok())
    return status;
  const int device = first.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "gated stream is on a different CUDA device"};
  }
  status = buffer_size(first, bytes, device, "gated first");
  if (!status.ok())
    return status;
  status = buffer_size(second, bytes, device, "gated second");
  if (!status.ok())
    return status;
  status = buffer_size(*output, bytes, device, "gated output");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  gated_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(first.data()),
      static_cast<const __nv_bfloat16 *>(second.data()),
      static_cast<__nv_bfloat16 *>(output->data()), elements,
      activation == GatedActivation::kGelu);
  return launch_status("gated_kernel");
}

Status bf16_gelu(const DeviceBuffer &input, const std::size_t elements,
                 DeviceBuffer *const output, const Stream &stream) {
  if (output == nullptr || !stream.valid()) {
    return {ErrorCode::kInvalidArgument,
            "bf16_gelu received invalid arguments"};
  }
  std::size_t bytes = 0;
  auto status =
      required_bytes(elements, sizeof(__nv_bfloat16), &bytes, "GELU tensor");
  if (!status.ok())
    return status;
  const int device = input.device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "GELU stream is on a different CUDA device"};
  }
  status = buffer_size(input, bytes, device, "GELU input");
  if (!status.ok())
    return status;
  status = buffer_size(*output, bytes, device, "GELU output");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  gelu_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(input.data()),
      static_cast<__nv_bfloat16 *>(output->data()), elements);
  return launch_status("gelu_kernel");
}

Status bf16_add_inplace(DeviceBuffer *const output,
                        const DeviceBuffer &residual,
                        const std::size_t elements, const Stream &stream) {
  if (output == nullptr || !stream.valid()) {
    return {ErrorCode::kInvalidArgument,
            "bf16_add_inplace received invalid arguments"};
  }
  std::size_t bytes = 0;
  auto status =
      required_bytes(elements, sizeof(__nv_bfloat16), &bytes, "add tensor");
  if (!status.ok())
    return status;
  const int device = output->device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "add stream is on a different CUDA device"};
  }
  status = buffer_size(*output, bytes, device, "add output");
  if (!status.ok())
    return status;
  status = buffer_size(residual, bytes, device, "add residual");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  add_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<__nv_bfloat16 *>(output->data()),
      static_cast<const __nv_bfloat16 *>(residual.data()), elements);
  return launch_status("add_kernel");
}

Status bf16_add_bias_inplace(DeviceBuffer *const output,
                             const DeviceBuffer &bias, const std::size_t rows,
                             const std::size_t width, const Stream &stream) {
  if (output == nullptr || !stream.valid() || rows == 0 || width == 0) {
    return {ErrorCode::kInvalidArgument,
            "bf16_add_bias_inplace received invalid arguments"};
  }
  std::size_t elements = 0;
  if (!multiply(rows, width, &elements)) {
    return {ErrorCode::kInvalidArgument, "bias add dimensions overflow"};
  }
  std::size_t output_bytes = 0;
  std::size_t bias_bytes = 0;
  auto status = required_bytes(elements, sizeof(__nv_bfloat16), &output_bytes,
                               "bias add output");
  if (!status.ok())
    return status;
  status = required_bytes(width, sizeof(__nv_bfloat16), &bias_bytes,
                          "bias add bias");
  if (!status.ok())
    return status;
  const int device = output->device();
  if (stream.device() != device) {
    return {ErrorCode::kInvalidArgument,
            "bias add stream is on a different CUDA device"};
  }
  status = buffer_size(*output, output_bytes, device, "bias add output");
  if (!status.ok())
    return status;
  status = buffer_size(bias, bias_bytes, device, "bias add bias");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  add_bias_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
      static_cast<__nv_bfloat16 *>(output->data()),
      static_cast<const __nv_bfloat16 *>(bias.data()), elements, width);
  return launch_status("add_bias_kernel");
}

Status bf16_mlp(const BlasLt &handle, const DeviceBuffer &input,
                const DeviceBuffer &l1_weight, const DeviceBuffer &l2_weight,
                const DeviceBuffer &l3_weight, const std::size_t rows,
                const std::size_t width, const std::size_t inner_width,
                const GatedActivation activation, MlpWorkspace *const workspace,
                DeviceBuffer *const output, const Stream &stream) {
  if (workspace == nullptr || output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "bf16_mlp requires workspace and output"};
  }
  auto status = bf16_linear(handle, input, l1_weight, nullptr, rows, width,
                            inner_width, &workspace->first, &workspace->blas,
                            stream, &workspace->up_plan);
  if (!status.ok())
    return status;
  status = bf16_linear(handle, input, l2_weight, nullptr, rows, width,
                       inner_width, &workspace->second, &workspace->blas,
                       stream, &workspace->up_plan);
  if (!status.ok())
    return status;
  std::size_t gated_elements = 0;
  if (!multiply(rows, inner_width, &gated_elements)) {
    return {ErrorCode::kInvalidArgument, "bf16_mlp dimensions overflow"};
  }
  const DeviceBuffer *gated_first = &workspace->first;
  if (activation == GatedActivation::kGelu) {
    status = bf16_gelu(workspace->first, gated_elements, &workspace->activated,
                       stream);
    if (!status.ok())
      return status;
    gated_first = &workspace->activated;
  }
  status = bf16_gated_elementwise(*gated_first, workspace->second,
                                  gated_elements, GatedActivation::kIdentity,
                                  &workspace->gated, stream);
  if (!status.ok())
    return status;
  return bf16_linear(handle, workspace->gated, l3_weight, nullptr, rows,
                     inner_width, width, output, &workspace->blas, stream,
                     &workspace->down_plan);
}

} // namespace evo2c::cuda
