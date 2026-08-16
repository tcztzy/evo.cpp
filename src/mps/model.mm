// SPDX-License-Identifier: Apache-2.0
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "model.hpp"

#include "../linear_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace evo::mps {
namespace {

constexpr const char *kKernelName = "mps-f32-gemm+host-ops";

bool checked_product(const std::size_t first, const std::size_t second,
                     std::size_t *const output) noexcept {
  if (output == nullptr ||
      (first != 0 && second > std::numeric_limits<std::size_t>::max() / first))
    return false;
  *output = first * second;
  return true;
}

std::size_t dtype_size(const TensorDType dtype) noexcept {
  switch (dtype) {
  case TensorDType::kF32:
    return sizeof(float);
  case TensorDType::kBF16:
    return sizeof(std::uint16_t);
  case TensorDType::kE4M3Software:
    return sizeof(std::uint8_t);
  }
  return 0;
}

float decode_scalar(const evo::detail::LinearTensorView tensor,
                    const std::size_t index) noexcept {
  if (tensor.dtype == TensorDType::kBF16) {
    const auto *const source = tensor.data + index * sizeof(std::uint16_t);
    const std::uint32_t bits = static_cast<std::uint32_t>(source[0]) << 16U |
                               static_cast<std::uint32_t>(source[1]) << 24U;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  if (tensor.dtype == TensorDType::kE4M3Software) {
    const std::uint8_t bits = tensor.data[index];
    const std::uint8_t magnitude = bits & 0x7fU;
    if (magnitude == 0x7fU)
      return std::numeric_limits<float>::quiet_NaN();
    const unsigned exponent = (magnitude >> 3U) & 0x0fU;
    const unsigned mantissa = magnitude & 0x07U;
    const float value =
        exponent == 0U ? std::ldexp(static_cast<float>(mantissa), -9)
                       : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                                    static_cast<int>(exponent) - 7);
    return (bits & 0x80U) == 0U ? value : -value;
  }
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(float), sizeof(value));
  return value;
}

std::string error_message(NSError *const error) {
  if (error == nil)
    return "unknown Metal error";
  const char *const message = error.localizedDescription.UTF8String;
  return message == nullptr ? "unknown Metal error" : std::string{message};
}

constexpr const char *kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LinearShape {
  uint rows;
  uint input_width;
  uint output_width;
};

float decode_bf16(ushort value) {
  return as_type<float>(uint(value) << 16);
}

float decode_e4m3(uchar value) {
  const uint magnitude = uint(value & 0x7f);
  if (magnitude == 0x7f)
    return as_type<float>(0x7fc00000u);
  const uint exponent = (magnitude >> 3) & 0x0f;
  const uint mantissa = magnitude & 0x07;
  const float decoded = exponent == 0
      ? float(mantissa) * exp2(-9.0f)
      : (1.0f + float(mantissa) * 0.125f) *
            exp2(float(int(exponent) - 7));
  return (value & 0x80) == 0 ? decoded : -decoded;
}

kernel void linear_bf16(device const float *input [[buffer(0)]],
                        device const ushort *weight [[buffer(1)]],
                        device float *output [[buffer(2)]],
                        constant LinearShape &shape [[buffer(3)]],
                        uint2 position [[thread_position_in_grid]]) {
  if (position.x >= shape.output_width || position.y >= shape.rows)
    return;
  float total = 0.0f;
  const uint input_offset = position.y * shape.input_width;
  const uint weight_offset = position.x * shape.input_width;
  for (uint source = 0; source < shape.input_width; ++source)
    total += input[input_offset + source] *
             decode_bf16(weight[weight_offset + source]);
  output[position.y * shape.output_width + position.x] = total;
}

kernel void linear_e4m3(device const float *input [[buffer(0)]],
                        device const uchar *weight [[buffer(1)]],
                        device float *output [[buffer(2)]],
                        constant LinearShape &shape [[buffer(3)]],
                        uint2 position [[thread_position_in_grid]]) {
  if (position.x >= shape.output_width || position.y >= shape.rows)
    return;
  float total = 0.0f;
  const uint input_offset = position.y * shape.input_width;
  const uint weight_offset = position.x * shape.input_width;
  for (uint source = 0; source < shape.input_width; ++source)
    total += input[input_offset + source] *
             decode_e4m3(weight[weight_offset + source]);
  output[position.y * shape.output_width + position.x] = total;
}
)METAL";

struct LinearShape final {
  std::uint32_t rows;
  std::uint32_t input_width;
  std::uint32_t output_width;
};

class MpsLinearExecutor final : public evo::detail::LinearExecutor {
public:
  MpsLinearExecutor(id<MTLDevice> device, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> bf16_pipeline,
                    id<MTLComputePipelineState> e4m3_pipeline)
      : device_(device), queue_(queue), bf16_pipeline_(bf16_pipeline),
        e4m3_pipeline_(e4m3_pipeline),
        weight_cache_([[NSMutableDictionary alloc] init]) {}

  [[nodiscard]] const char *name() const noexcept override {
    return kKernelName;
  }

  [[nodiscard]] Status linear(const float *const input, const std::size_t rows,
                              const std::size_t input_width,
                              const evo::detail::LinearTensorView weight,
                              const std::size_t output_width,
                              const evo::detail::LinearTensorView *const bias,
                              std::vector<float> *const output) override {
    @autoreleasepool {
      std::size_t input_elements = 0;
      std::size_t output_elements = 0;
      std::size_t expected_weights = 0;
      if (input == nullptr || output == nullptr || rows == 0 ||
          input_width == 0 || output_width == 0 || weight.data == nullptr ||
          !checked_product(rows, input_width, &input_elements) ||
          !checked_product(rows, output_width, &output_elements) ||
          !checked_product(input_width, output_width, &expected_weights) ||
          weight.elements != expected_weights ||
          dtype_size(weight.dtype) == 0 ||
          (bias != nullptr &&
           (bias->data == nullptr || bias->elements != output_width ||
            dtype_size(bias->dtype) == 0))) {
        return {ErrorCode::kInvalidArgument,
                "MPS linear dimensions or tensor dtype are invalid"};
      }
      if (rows > std::numeric_limits<std::uint32_t>::max() ||
          input_width > std::numeric_limits<std::uint32_t>::max() ||
          output_width > std::numeric_limits<std::uint32_t>::max()) {
        return {ErrorCode::kUnsupported,
                "MPS linear dimensions exceed Metal kernel limits"};
      }

      std::lock_guard<std::mutex> lock{mutex_};
      Status status =
          weight.dtype == TensorDType::kF32
              ? encode_f32(input, rows, input_width, weight, output_width,
                           output_elements, output)
              : encode_quantized(input, rows, input_width, weight, output_width,
                                 input_elements, output_elements, output);
      if (!status.ok())
        return status;
      if (bias != nullptr) {
        for (std::size_t row = 0; row < rows; ++row) {
          for (std::size_t column = 0; column < output_width; ++column)
            (*output)[row * output_width + column] +=
                decode_scalar(*bias, column);
        }
      }
      return Status::Ok();
    }
  }

private:
  [[nodiscard]] id<MTLBuffer>
  cached_weight(const evo::detail::LinearTensorView weight,
                const std::size_t rows, const std::size_t columns,
                const std::size_t row_bytes) {
    NSValue *const key = [NSValue valueWithPointer:weight.data];
    id<MTLBuffer> buffer = [weight_cache_ objectForKey:key];
    if (buffer != nil)
      return buffer;
    std::size_t length = 0;
    if (!checked_product(rows, row_bytes, &length))
      return nil;
    buffer = [device_ newBufferWithLength:length
                                  options:MTLResourceStorageModeShared];
    if (buffer == nil)
      return nil;
    const std::size_t source_row_bytes = columns * dtype_size(weight.dtype);
    auto *const destination = static_cast<std::uint8_t *>(buffer.contents);
    for (std::size_t row = 0; row < rows; ++row) {
      std::memcpy(destination + row * row_bytes,
                  weight.data + row * source_row_bytes, source_row_bytes);
      if (row_bytes > source_row_bytes) {
        std::memset(destination + row * row_bytes + source_row_bytes, 0,
                    row_bytes - source_row_bytes);
      }
    }
    [weight_cache_ setObject:buffer forKey:key];
    return buffer;
  }

  [[nodiscard]] Status command_status(id<MTLCommandBuffer> command) const {
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
      return {ErrorCode::kMps,
              "MPS command failed: " + error_message(command.error)};
    }
    return Status::Ok();
  }

  [[nodiscard]] Status encode_f32(const float *const input,
                                  const std::size_t rows,
                                  const std::size_t input_width,
                                  const evo::detail::LinearTensorView weight,
                                  const std::size_t output_width,
                                  const std::size_t output_elements,
                                  std::vector<float> *const output) {
    const NSUInteger input_row_bytes =
        [MPSMatrixDescriptor rowBytesFromColumns:input_width
                                        dataType:MPSDataTypeFloat32];
    const NSUInteger output_row_bytes =
        [MPSMatrixDescriptor rowBytesFromColumns:output_width
                                        dataType:MPSDataTypeFloat32];
    std::size_t input_length = 0;
    std::size_t output_length = 0;
    if (!checked_product(rows, input_row_bytes, &input_length) ||
        !checked_product(rows, output_row_bytes, &output_length)) {
      return {ErrorCode::kUnsupported, "MPS matrix buffer dimensions overflow"};
    }
    id<MTLBuffer> input_buffer =
        [device_ newBufferWithLength:input_length
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [device_ newBufferWithLength:output_length
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_buffer =
        cached_weight(weight, output_width, input_width, input_row_bytes);
    if (input_buffer == nil || output_buffer == nil || weight_buffer == nil)
      return {ErrorCode::kMps, "MPS matrix buffer allocation failed"};
    auto *const input_bytes =
        static_cast<std::uint8_t *>(input_buffer.contents);
    for (std::size_t row = 0; row < rows; ++row) {
      std::memcpy(input_bytes + row * input_row_bytes,
                  input + row * input_width, input_width * sizeof(float));
    }

    MPSMatrixDescriptor *const left_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                              columns:input_width
                                             rowBytes:input_row_bytes
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *const right_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:output_width
                                              columns:input_width
                                             rowBytes:input_row_bytes
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *const result_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                              columns:output_width
                                             rowBytes:output_row_bytes
                                             dataType:MPSDataTypeFloat32];
    MPSMatrix *const left = [[MPSMatrix alloc] initWithBuffer:input_buffer
                                                   descriptor:left_descriptor];
    MPSMatrix *const right =
        [[MPSMatrix alloc] initWithBuffer:weight_buffer
                               descriptor:right_descriptor];
    MPSMatrix *const result =
        [[MPSMatrix alloc] initWithBuffer:output_buffer
                               descriptor:result_descriptor];
    MPSMatrixMultiplication *const multiplication =
        [[MPSMatrixMultiplication alloc] initWithDevice:device_
                                          transposeLeft:NO
                                         transposeRight:YES
                                             resultRows:rows
                                          resultColumns:output_width
                                        interiorColumns:input_width
                                                  alpha:1.0
                                                   beta:0.0];
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    if (command == nil)
      return {ErrorCode::kMps, "MPS command buffer allocation failed"};
    [multiplication encodeToCommandBuffer:command
                               leftMatrix:left
                              rightMatrix:right
                             resultMatrix:result];
    auto status = command_status(command);
    if (!status.ok())
      return status;
    output->resize(output_elements);
    const auto *const output_bytes =
        static_cast<const std::uint8_t *>(output_buffer.contents);
    for (std::size_t row = 0; row < rows; ++row) {
      std::memcpy(output->data() + row * output_width,
                  output_bytes + row * output_row_bytes,
                  output_width * sizeof(float));
    }
    return Status::Ok();
  }

  [[nodiscard]] Status encode_quantized(
      const float *const input, const std::size_t rows,
      const std::size_t input_width, const evo::detail::LinearTensorView weight,
      const std::size_t output_width, const std::size_t input_elements,
      const std::size_t output_elements, std::vector<float> *const output) {
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t weight_bytes = 0;
    if (!checked_product(input_elements, sizeof(float), &input_bytes) ||
        !checked_product(output_elements, sizeof(float), &output_bytes) ||
        !checked_product(weight.elements, dtype_size(weight.dtype),
                         &weight_bytes)) {
      return {ErrorCode::kUnsupported,
              "MPS compute buffer dimensions overflow"};
    }
    id<MTLBuffer> input_buffer =
        [device_ newBufferWithBytes:input
                             length:input_bytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [device_ newBufferWithLength:output_bytes
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_buffer =
        cached_weight(weight, 1, weight.elements, weight_bytes);
    if (input_buffer == nil || output_buffer == nil || weight_buffer == nil)
      return {ErrorCode::kMps, "MPS compute buffer allocation failed"};
    id<MTLComputePipelineState> pipeline =
        weight.dtype == TensorDType::kBF16 ? bf16_pipeline_ : e4m3_pipeline_;
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil)
      return {ErrorCode::kMps, "MPS compute encoder allocation failed"};
    const LinearShape shape{static_cast<std::uint32_t>(rows),
                            static_cast<std::uint32_t>(input_width),
                            static_cast<std::uint32_t>(output_width)};
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input_buffer offset:0 atIndex:0];
    [encoder setBuffer:weight_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    const NSUInteger group_width = std::min<NSUInteger>(16, output_width);
    const NSUInteger group_height = std::min<NSUInteger>(16, rows);
    [encoder dispatchThreads:MTLSizeMake(output_width, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(group_width, group_height, 1)];
    [encoder endEncoding];
    auto status = command_status(command);
    if (!status.ok())
      return status;
    output->resize(output_elements);
    std::memcpy(output->data(), output_buffer.contents, output_bytes);
    return Status::Ok();
  }

  id<MTLDevice> __strong device_;
  id<MTLCommandQueue> __strong queue_;
  id<MTLComputePipelineState> __strong bf16_pipeline_;
  id<MTLComputePipelineState> __strong e4m3_pipeline_;
  NSMutableDictionary<NSValue *, id<MTLBuffer>> *__strong weight_cache_;
  std::mutex mutex_;
};

} // namespace

Status create_linear_executor(
    std::shared_ptr<evo::detail::LinearExecutor> *const executor) {
  if (executor == nullptr)
    return {ErrorCode::kInvalidArgument, "MPS executor output is null"};
  executor->reset();
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil)
      return {ErrorCode::kMps, "no Metal device is available"};
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil)
      return {ErrorCode::kMps, "cannot create the Metal command queue"};
    NSError *library_error = nil;
    NSString *const source = [NSString stringWithUTF8String:kMetalSource];
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&library_error];
    if (library == nil) {
      return {ErrorCode::kMps, "cannot compile MPS linear kernels: " +
                                   error_message(library_error)};
    }
    NSError *pipeline_error = nil;
    id<MTLFunction> bf16_function =
        [library newFunctionWithName:@"linear_bf16"];
    id<MTLFunction> e4m3_function =
        [library newFunctionWithName:@"linear_e4m3"];
    id<MTLComputePipelineState> bf16_pipeline =
        [device newComputePipelineStateWithFunction:bf16_function
                                              error:&pipeline_error];
    if (bf16_pipeline == nil) {
      return {ErrorCode::kMps, "cannot create the BF16 MPS pipeline: " +
                                   error_message(pipeline_error)};
    }
    pipeline_error = nil;
    id<MTLComputePipelineState> e4m3_pipeline =
        [device newComputePipelineStateWithFunction:e4m3_function
                                              error:&pipeline_error];
    if (e4m3_pipeline == nil) {
      return {ErrorCode::kMps, "cannot create the E4M3 MPS pipeline: " +
                                   error_message(pipeline_error)};
    }
    *executor = std::make_shared<MpsLinearExecutor>(
        device, queue, bf16_pipeline, e4m3_pipeline);
    return Status::Ok();
  }
}

Status ModelLoader::load(const ModelFile &artifact,
                         const bool allow_test_fixture,
                         cpu::Model *const model) {
  if (model == nullptr)
    return {ErrorCode::kInvalidArgument, "MPS model output is null"};
  std::shared_ptr<evo::detail::LinearExecutor> executor;
  auto status = create_linear_executor(&executor);
  if (!status.ok())
    return status;
  return model->load_with_executor(artifact, std::move(executor),
                                   allow_test_fixture);
}

} // namespace evo::mps
