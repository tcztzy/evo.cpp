// SPDX-License-Identifier: Apache-2.0
#include "evo2c/cuda/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include <cuda_bf16.h>

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
      value = 0.5F * value * (1.0F + erff(value * 0.70710678118654752440F));
    }
    output[index] =
        __float2bfloat16_rn(value * __bfloat162float(second[index]));
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
  return blas.allocate(device, blas_bytes);
}

Status bf16_linear(const BlasLt &handle, const DeviceBuffer &input,
                   const DeviceBuffer &weight, const DeviceBuffer *const bias,
                   const std::size_t rows, const std::size_t in_features,
                   const std::size_t out_features, DeviceBuffer *const output,
                   DeviceBuffer *const workspace, const Stream &stream) {
  if (!handle.valid() || output == nullptr || workspace == nullptr ||
      !stream.valid()) {
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

  MatmulObjects objects;
  status =
      cublas_status(cublasLtMatmulDescCreate(&objects.operation,
                                             CUBLAS_COMPUTE_32F, CUDA_R_32F),
                    "cublasLtMatmulDescCreate");
  if (!status.ok())
    return status;
  // Interpret row-major weight [N,K], input [M,K], and output [M,N] as
  // column-major [K,N], [K,M], and [N,M]. This is the native cuBLASLt layout
  // used by PyTorch and supports a fused BF16 bias epilogue.
  const cublasOperation_t transpose = CUBLAS_OP_T;
  status = cublas_status(cublasLtMatmulDescSetAttribute(
                             objects.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                             &transpose, sizeof(transpose)),
                         "cublasLtMatmulDescSetAttribute TRANSA");
  if (!status.ok())
    return status;
  if (bias != nullptr) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    status = cublas_status(cublasLtMatmulDescSetAttribute(
                               objects.operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                               &epilogue, sizeof(epilogue)),
                           "cublasLtMatmulDescSetAttribute EPILOGUE");
    if (!status.ok())
      return status;
    const void *const bias_pointer = bias->data();
    status =
        cublas_status(cublasLtMatmulDescSetAttribute(
                          objects.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                          &bias_pointer, sizeof(bias_pointer)),
                      "cublasLtMatmulDescSetAttribute BIAS_POINTER");
    if (!status.ok())
      return status;
  }
  status =
      cublas_status(cublasLtMatrixLayoutCreate(&objects.input, CUDA_R_16BF,
                                               in_features, rows, in_features),
                    "cublasLtMatrixLayoutCreate input transpose view");
  if (!status.ok())
    return status;
  status = cublas_status(cublasLtMatrixLayoutCreate(&objects.weight,
                                                    CUDA_R_16BF, in_features,
                                                    out_features, in_features),
                         "cublasLtMatrixLayoutCreate weight transpose view");
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
  const auto workspace_bytes = workspace->bytes();
  status = cublas_status(cublasLtMatmulPreferenceSetAttribute(
                             objects.preference,
                             CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                             &workspace_bytes, sizeof(workspace_bytes)),
                         "cublasLtMatmulPreferenceSetAttribute workspace");
  if (!status.ok())
    return status;
  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  status = cublas_status(cublasLtMatmulAlgoGetHeuristic(
                             handle.get(), objects.operation, objects.weight,
                             objects.input, objects.output, objects.output,
                             objects.preference, 1, &heuristic, &returned),
                         "cublasLtMatmulAlgoGetHeuristic");
  if (!status.ok())
    return status;
  if (returned == 0) {
    return {ErrorCode::kCuda,
            "cuBLASLt found no BF16 row-major matmul algorithm"};
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  status = cublas_status(
      cublasLtMatmul(handle.get(), objects.operation, &alpha, weight.data(),
                     objects.weight, input.data(), objects.input, &beta,
                     output->data(), objects.output, output->data(),
                     objects.output, &heuristic.algo, workspace->data(),
                     workspace_bytes, stream.get()),
      "cublasLtMatmul");
  return status;
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
  auto status =
      bf16_linear(handle, input, l1_weight, nullptr, rows, width, inner_width,
                  &workspace->first, &workspace->blas, stream);
  if (!status.ok())
    return status;
  status =
      bf16_linear(handle, input, l2_weight, nullptr, rows, width, inner_width,
                  &workspace->second, &workspace->blas, stream);
  if (!status.ok())
    return status;
  std::size_t gated_elements = 0;
  if (!multiply(rows, inner_width, &gated_elements)) {
    return {ErrorCode::kInvalidArgument, "bf16_mlp dimensions overflow"};
  }
  status = bf16_gated_elementwise(workspace->first, workspace->second,
                                  gated_elements, activation, &workspace->gated,
                                  stream);
  if (!status.ok())
    return status;
  return bf16_linear(handle, workspace->gated, l3_weight, nullptr, rows,
                     inner_width, width, output, &workspace->blas, stream);
}

} // namespace evo2c::cuda
