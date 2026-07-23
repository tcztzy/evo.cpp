// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "evo2c/cpu_reference.hpp"
#include "evo2c/cuda/ops.hpp"
#include "evo2c/cuda/runtime.hpp"
#include "evo2c/fp8.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

void require(const evo2c::Status &status, const std::string_view operation) {
  if (!status.ok()) {
    throw std::runtime_error(std::string{operation} + ": " + status.message());
  }
}

std::vector<float> values(const std::size_t count, const std::size_t offset,
                          const float divisor) {
  std::vector<float> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto raw = static_cast<int>(((index + offset) * 37 + 11) % 29) - 14;
    result.push_back(static_cast<float>(raw) / divisor);
  }
  return result;
}

std::vector<__nv_bfloat16> to_bf16(const std::vector<float> &input) {
  std::vector<__nv_bfloat16> output;
  output.reserve(input.size());
  for (const float value : input)
    output.push_back(__float2bfloat16(value));
  return output;
}

std::vector<float> to_float(const std::vector<__nv_bfloat16> &input) {
  std::vector<float> output;
  output.reserve(input.size());
  for (const auto value : input)
    output.push_back(__bfloat162float(value));
  return output;
}

evo2c::cuda::DeviceBuffer upload(const int device, const void *const source,
                                 const std::size_t bytes,
                                 const evo2c::cuda::Stream &stream) {
  evo2c::cuda::DeviceBuffer buffer;
  require(buffer.allocate(device, bytes), "allocate upload buffer");
  require(buffer.copy_from_host(source, bytes, stream), "upload buffer");
  return buffer;
}

std::vector<float> download_bf16(const evo2c::cuda::DeviceBuffer &buffer,
                                 const std::size_t elements,
                                 const evo2c::cuda::Stream &stream) {
  std::vector<__nv_bfloat16> raw(elements);
  require(buffer.copy_to_host(raw.data(), raw.size() * sizeof(raw[0]), stream),
          "download BF16");
  require(stream.synchronize(), "synchronize download");
  return to_float(raw);
}

float cosine(const std::vector<float> &left, const std::vector<float> &right) {
  double dot = 0.0;
  double left_norm = 0.0;
  double right_norm = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    dot += static_cast<double>(left[index]) * right[index];
    left_norm += static_cast<double>(left[index]) * left[index];
    right_norm += static_cast<double>(right[index]) * right[index];
  }
  return static_cast<float>(dot / std::sqrt(left_norm * right_norm));
}

std::uint32_t float_bits(const float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bits_float(const std::uint32_t bits) {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

int normal_exponent_reference(const float value) {
  return static_cast<int>((float_bits(std::abs(value)) >> 23U) & 0xffU) -
         127;
}

std::int32_t align_reference(const float value, const int operand_exponent,
                             const int maximum_exponent,
                             const bool denormalized_product) {
  if (value == 0.0F)
    return 0;
  const auto bits = float_bits(value);
  const auto magnitude = bits & 0x7fffffffU;
  std::uint32_t significand =
      (magnitude & 0x7fffffU) | 0x800000U;
  if (denormalized_product) {
    const int normalized_exponent =
        static_cast<int>((magnitude >> 23U) & 0xffU) - 127;
    significand <<= static_cast<unsigned int>(
        normalized_exponent - operand_exponent);
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

float h100_qgmma_reference(const std::vector<float> &input,
                           const std::vector<float> &weight,
                           const std::size_t offset,
                           const std::size_t inner) {
  float accumulator = 0.0F;
  for (std::size_t start = 0; start < inner; start += 32U) {
    int maximum_exponent =
        accumulator == 0.0F ? -1024
                            : normal_exponent_reference(accumulator);
    for (std::size_t index = 0; index < 32U; ++index) {
      const float left = input[start + index];
      const float right = weight[offset + start + index];
      if (left != 0.0F && right != 0.0F) {
        maximum_exponent =
            std::max(maximum_exponent,
                     normal_exponent_reference(left) +
                         normal_exponent_reference(right));
      }
    }
    std::int32_t aligned_sum =
        accumulator == 0.0F
            ? 0
            : align_reference(accumulator,
                              normal_exponent_reference(accumulator),
                              maximum_exponent, false);
    for (std::size_t index = 0; index < 32U; ++index) {
      const float left = input[start + index];
      const float right = weight[offset + start + index];
      if (left != 0.0F && right != 0.0F) {
        const int exponent = normal_exponent_reference(left) +
                             normal_exponent_reference(right);
        aligned_sum += align_reference(left * right, exponent,
                                       maximum_exponent, true);
      }
    }
    accumulator =
        aligned_sum == 0
            ? 0.0F
            : bits_float(
                  float_bits(std::ldexp(static_cast<float>(aligned_sum),
                                       maximum_exponent - 13)) &
                  0xfffffc00U);
  }
  return accumulator;
}

bool all_close(const std::vector<float> &actual,
               const std::vector<float> &expected, const float absolute,
               const float relative) {
  if (actual.size() != expected.size())
    return false;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float tolerance = absolute + relative * std::abs(expected[index]);
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance) {
      std::cerr << "mismatch at " << index << ": actual=" << actual[index]
                << " expected=" << expected[index] << " tolerance=" << tolerance
                << '\n';
      return false;
    }
  }
  return true;
}

void test_buffer(const int device, const evo2c::cuda::Stream &stream) {
  const std::vector<float> host{1.0F, -2.0F, 3.0F, 4.0F};
  auto buffer =
      upload(device, host.data(), host.size() * sizeof(float), stream);
  std::vector<float> roundtrip(host.size(), 0.0F);
  require(buffer.copy_to_host(roundtrip.data(),
                              roundtrip.size() * sizeof(float), stream),
          "buffer roundtrip");
  require(stream.synchronize(), "buffer synchronize");
  check(roundtrip == host, "DeviceBuffer copy is bit exact");
  check(!buffer.copy_to_host(roundtrip.data(), buffer.bytes() + 1, stream).ok(),
        "DeviceBuffer rejects out-of-bounds copies");
}

void test_linear(const int device, const evo2c::cuda::Stream &stream,
                 const evo2c::cuda::BlasLt &blas) {
  constexpr std::size_t rows = 5;
  constexpr std::size_t input_width = 7;
  constexpr std::size_t output_width = 11;
  const auto input_bf16 = to_bf16(values(rows * input_width, 3, 11.0F));
  const auto weight_bf16 =
      to_bf16(values(output_width * input_width, 17, 17.0F));
  const auto bias_bf16 = to_bf16(values(output_width, 29, 23.0F));
  const auto input_f32 = to_float(input_bf16);
  const auto weight_f32 = to_float(weight_bf16);
  const auto bias_f32 = to_float(bias_bf16);

  auto input = upload(device, input_bf16.data(),
                      input_bf16.size() * sizeof(input_bf16[0]), stream);
  auto weight = upload(device, weight_bf16.data(),
                       weight_bf16.size() * sizeof(weight_bf16[0]), stream);
  auto bias = upload(device, bias_bf16.data(),
                     bias_bf16.size() * sizeof(bias_bf16[0]), stream);
  evo2c::cuda::DeviceBuffer output;
  evo2c::cuda::DeviceBuffer workspace;
  require(output.allocate(device, rows * output_width * sizeof(__nv_bfloat16)),
          "linear output");
  require(workspace.allocate(device, evo2c::cuda::kDefaultBlasWorkspaceBytes),
          "linear workspace");

  require(evo2c::cuda::bf16_linear(blas, input, weight, nullptr, rows,
                                   input_width, output_width, &output,
                                   &workspace, stream),
          "BF16 linear");
  std::vector<float> expected;
  require(evo2c::cpu::linear(input_f32, rows, input_width, weight_f32,
                             output_width, nullptr, &expected),
          "CPU linear reference");
  auto actual = download_bf16(output, rows * output_width, stream);
  check(all_close(actual, expected, 0.02F, 0.02F),
        "cuBLASLt BF16 linear matches F32 reference");
  check(cosine(actual, expected) >= 0.999F,
        "cuBLASLt BF16 linear cosine is at least 0.999");

  require(evo2c::cuda::bf16_linear(blas, input, weight, &bias, rows,
                                   input_width, output_width, &output,
                                   &workspace, stream),
          "BF16 biased linear");
  require(evo2c::cpu::linear(input_f32, rows, input_width, weight_f32,
                             output_width, &bias_f32, &expected),
          "CPU biased linear reference");
  actual = download_bf16(output, rows * output_width, stream);
  check(all_close(actual, expected, 0.03F, 0.02F),
        "cuBLASLt BF16 biased linear matches F32 reference");
}

void test_software_e4m3(const int device,
                        const evo2c::cuda::Stream &stream) {
  const std::vector<float> values{
      -448.0F,       -432.0F,      -416.0F,       -1.25F,
      -1.1875F,      -1.125F,      -1.0625F,      -1.0F,
      -0.5F,         -0.015625F,   -0.00390625F,  -0.001953125F,
      -0.0009765625F, -0.0F,        0.0F,           0.0009765625F,
      0.001953125F,  0.0029296875F, 0.00390625F,   0.015625F,
      0.5F,          1.0F,         1.0625F,        1.125F,
      1.1875F,       1.25F,        416.0F,         432.0F,
      448.0F,
  };
  const std::vector<std::uint8_t> expected_codes{
      0xfe, 0xfe, 0xfd, 0xba, 0xba, 0xb9, 0xb8, 0xb8, 0xb0, 0x88,
      0x82, 0x81, 0x80, 0x80, 0x00, 0x00, 0x01, 0x02, 0x02, 0x08,
      0x30, 0x38, 0x38, 0x39, 0x3a, 0x3a, 0x7d, 0x7e, 0x7e,
  };
  const auto input_bf16 = to_bf16(values);
  auto input = upload(device, input_bf16.data(),
                      input_bf16.size() * sizeof(input_bf16[0]), stream);
  evo2c::cuda::DeviceBuffer codes;
  evo2c::cuda::DeviceBuffer dequantized;
  require(codes.allocate(device, values.size()), "E4M3 code output");
  require(dequantized.allocate(device, values.size() * sizeof(float)),
          "E4M3 dequantized output");
  require(evo2c::cuda::software_e4m3_quantize_bf16(
              input, values.size(), 1.0F, &codes, &dequantized, stream),
          "software E4M3 quantize");

  std::vector<std::uint8_t> actual_codes(values.size());
  std::vector<float> actual_values(values.size());
  require(codes.copy_to_host(actual_codes.data(), actual_codes.size(), stream),
          "download E4M3 codes");
  require(dequantized.copy_to_host(actual_values.data(),
                                   actual_values.size() * sizeof(float), stream),
          "download E4M3 values");
  require(stream.synchronize(), "synchronize E4M3 outputs");
  check(actual_codes == expected_codes,
        "sm80 software E4M3 codes match PyTorch bit vectors exactly");
  for (std::size_t index = 0; index < actual_values.size(); ++index) {
    check(actual_values[index] ==
              evo2c::fp8::decode_e4m3fn(expected_codes[index]),
          "sm80 software E4M3 dequantization is bit exact");
  }
  check(!evo2c::cuda::software_e4m3_quantize_bf16(
             input, values.size(), 0.0F, nullptr, &dequantized, stream)
             .ok(),
        "software E4M3 rejects a zero scale");
}

void test_software_h100_qgmma(const int device,
                              const evo2c::cuda::Stream &stream) {
  constexpr std::size_t rows = 3;
  constexpr std::size_t inner = 64;
  constexpr std::size_t columns = 5;
  constexpr float input_scale = 5.75F;
  constexpr float weight_scale = 3.25F;
  constexpr float output_scale = 0.03125F;
  const auto input_bf16 = to_bf16(values(rows * inner, 41, 13.0F));
  const auto weight_bf16 =
      to_bf16(values(columns * inner, 73, 17.0F));
  auto input = upload(device, input_bf16.data(),
                      input_bf16.size() * sizeof(input_bf16[0]), stream);
  auto weight = upload(device, weight_bf16.data(),
                       weight_bf16.size() * sizeof(weight_bf16[0]), stream);
  evo2c::cuda::DeviceBuffer payload;
  evo2c::cuda::DeviceBuffer input_codes;
  evo2c::cuda::DeviceBuffer output;
  require(payload.allocate(device, weight_bf16.size()),
          "QGMMA weight payload");
  require(input_codes.allocate(device, input_bf16.size()),
          "QGMMA input-code workspace");
  require(output.allocate(device,
                          rows * columns * sizeof(__nv_bfloat16)),
          "QGMMA output");
  require(evo2c::cuda::software_e4m3_quantize_bf16_codes(
              weight, weight_bf16.size(), weight_scale, &payload, stream),
          "QGMMA weight quantization");
  require(evo2c::cuda::software_e4m3_h100_linear(
              input, payload, rows, inner, columns, input_scale,
              output_scale, &input_codes, &output, stream),
          "software H100 QGMMA");
  const auto actual = download_bf16(output, rows * columns, stream);

  std::vector<float> quantized_weight;
  quantized_weight.reserve(weight_bf16.size());
  for (const auto value : weight_bf16) {
    quantized_weight.push_back(evo2c::fp8::decode_e4m3fn(
        evo2c::fp8::encode_e4m3fn(
            __bfloat162float(value) * weight_scale)));
  }
  std::vector<float> expected;
  expected.reserve(rows * columns);
  for (std::size_t row = 0; row < rows; ++row) {
    std::vector<float> quantized_input;
    quantized_input.reserve(inner);
    for (std::size_t index = 0; index < inner; ++index) {
      quantized_input.push_back(evo2c::fp8::decode_e4m3fn(
          evo2c::fp8::encode_e4m3fn(
              __bfloat162float(input_bf16[row * inner + index]) *
              input_scale)));
    }
    for (std::size_t column = 0; column < columns; ++column) {
      const float raw = h100_qgmma_reference(
          quantized_input, quantized_weight, column * inner, inner);
      expected.push_back(
          __bfloat162float(__float2bfloat16_rn(raw * output_scale)));
    }
  }
  check(actual == expected,
        "sm80 software QGMMA is bit exact against the H100 global-alignment oracle");
  require(evo2c::cuda::software_e4m3_h100_linear(
              input, payload, 1, inner, columns, input_scale,
              output_scale, &input_codes, &output, stream),
          "software H100 QGMMA decode tile");
  const auto decode_actual = download_bf16(output, columns, stream);
  check(decode_actual ==
            std::vector<float>(expected.begin(), expected.begin() + columns),
        "sm80 software QGMMA one-row decode tile is bit exact");
  check(!evo2c::cuda::software_e4m3_h100_linear(
             input, payload, rows, inner - 1, columns, input_scale,
             output_scale, &input_codes, &output, stream)
             .ok(),
        "software QGMMA rejects a non-K32 inner dimension");
}

void test_rms_norm(const int device, const evo2c::cuda::Stream &stream) {
  constexpr std::size_t rows = 3;
  constexpr std::size_t width = 257;
  const auto input_bf16 = to_bf16(values(rows * width, 7, 19.0F));
  const auto input_f32 = to_float(input_bf16);
  auto scale_f32 = values(width, 31, 101.0F);
  for (float &value : scale_f32)
    value = 1.0F + value * 0.1F;
  auto input = upload(device, input_bf16.data(),
                      input_bf16.size() * sizeof(input_bf16[0]), stream);
  auto scale = upload(device, scale_f32.data(),
                      scale_f32.size() * sizeof(float), stream);
  evo2c::cuda::DeviceBuffer output;
  require(output.allocate(device, input_bf16.size() * sizeof(input_bf16[0])),
          "RMSNorm output");
  require(evo2c::cuda::bf16_rms_norm(input, scale, rows, width, 1.0e-6F,
                                     &output, stream),
          "BF16 RMSNorm");
  std::vector<float> expected;
  require(evo2c::cpu::rms_norm(input_f32, rows, width, scale_f32, 1.0e-6F,
                               &expected),
          "CPU RMSNorm reference");
  const auto actual = download_bf16(output, rows * width, stream);
  check(all_close(actual, expected, 0.015F, 0.01F),
        "BF16 RMSNorm tail width matches after-sqrt epsilon reference");
  check(cosine(actual, expected) >= 0.999F,
        "BF16 RMSNorm cosine is at least 0.999");
}

void test_elementwise(const int device, const evo2c::cuda::Stream &stream) {
  constexpr std::size_t elements = 513;
  const auto first_bf16 = to_bf16(values(elements, 11, 13.0F));
  const auto second_bf16 = to_bf16(values(elements, 47, 17.0F));
  const auto first_f32 = to_float(first_bf16);
  const auto second_f32 = to_float(second_bf16);
  auto first = upload(device, first_bf16.data(),
                      first_bf16.size() * sizeof(first_bf16[0]), stream);
  auto second = upload(device, second_bf16.data(),
                       second_bf16.size() * sizeof(second_bf16[0]), stream);
  evo2c::cuda::DeviceBuffer output;
  require(output.allocate(device, elements * sizeof(__nv_bfloat16)),
          "gated output");
  require(evo2c::cuda::bf16_gated_elementwise(
              first, second, elements, evo2c::cuda::GatedActivation::kGelu,
              &output, stream),
          "GELU gate");
  std::vector<float> expected(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    const float gelu =
        0.5F * first_f32[index] *
        (1.0F + std::erf(first_f32[index] * 0.70710678118654752440F));
    expected[index] = gelu * second_f32[index];
  }
  auto actual = download_bf16(output, elements, stream);
  check(all_close(actual, expected, 0.015F, 0.01F),
        "fused GELU gate handles non-multiple tail");

  require(evo2c::cuda::bf16_gated_elementwise(
              first, second, elements, evo2c::cuda::GatedActivation::kIdentity,
              &output, stream),
          "identity gate");
  for (std::size_t index = 0; index < elements; ++index) {
    expected[index] = first_f32[index] * second_f32[index];
  }
  actual = download_bf16(output, elements, stream);
  check(all_close(actual, expected, 0.015F, 0.01F),
        "fused identity gate handles non-multiple tail");

  require(evo2c::cuda::bf16_add_inplace(&output, first, elements, stream),
          "residual add");
  for (std::size_t index = 0; index < elements; ++index)
    expected[index] += first_f32[index];
  actual = download_bf16(output, elements, stream);
  check(all_close(actual, expected, 0.02F, 0.015F),
        "BF16 residual add handles non-multiple tail");
}

void test_mlp(const int device, const evo2c::cuda::Stream &stream,
              const evo2c::cuda::BlasLt &blas) {
  constexpr std::size_t rows = 3;
  constexpr std::size_t width = 7;
  constexpr std::size_t inner = 13;
  const auto input_bf16 = to_bf16(values(rows * width, 5, 19.0F));
  const auto l1_bf16 = to_bf16(values(inner * width, 17, 29.0F));
  const auto l2_bf16 = to_bf16(values(inner * width, 61, 31.0F));
  const auto l3_bf16 = to_bf16(values(width * inner, 101, 37.0F));
  auto input = upload(device, input_bf16.data(),
                      input_bf16.size() * sizeof(input_bf16[0]), stream);
  auto l1 = upload(device, l1_bf16.data(), l1_bf16.size() * sizeof(l1_bf16[0]),
                   stream);
  auto l2 = upload(device, l2_bf16.data(), l2_bf16.size() * sizeof(l2_bf16[0]),
                   stream);
  auto l3 = upload(device, l3_bf16.data(), l3_bf16.size() * sizeof(l3_bf16[0]),
                   stream);
  evo2c::cuda::MlpWorkspace workspace;
  require(workspace.allocate(device, rows, inner), "MLP workspace");
  evo2c::cuda::DeviceBuffer output;
  require(output.allocate(device, rows * width * sizeof(__nv_bfloat16)),
          "MLP output");
  require(evo2c::cuda::bf16_mlp(blas, input, l1, l2, l3, rows, width, inner,
                                evo2c::cuda::GatedActivation::kGelu, &workspace,
                                &output, stream),
          "BF16 MLP");
  std::vector<float> expected;
  require(evo2c::cpu::gated_mlp(to_float(input_bf16), rows, width, inner,
                                to_float(l1_bf16), to_float(l2_bf16),
                                to_float(l3_bf16),
                                evo2c::cpu::MlpActivation::kGelu, &expected),
          "CPU MLP reference");
  const auto actual = download_bf16(output, rows * width, stream);
  check(cosine(actual, expected) >= 0.999F,
        "composed BF16 MLP cosine is at least 0.999");
  check(all_close(actual, expected, 0.05F, 0.05F),
        "composed BF16 MLP matches F32 reference");
}

int requested_device() {
  const char *environment = std::getenv("EVO2C_TEST_DEVICE");
  if (environment == nullptr)
    return 0;
  return std::stoi(environment);
}

} // namespace

int main() {
  try {
    const int device = requested_device();
    require(evo2c::cuda::select_device(device), "select CUDA device");
    cudaDeviceProp properties{};
    require(
        evo2c::cuda::cuda_status(cudaGetDeviceProperties(&properties, device),
                                 "cudaGetDeviceProperties"),
        "query CUDA device");
    int driver = 0;
    int runtime = 0;
    require(evo2c::cuda::cuda_status(cudaDriverGetVersion(&driver),
                                     "cudaDriverGetVersion"),
            "query CUDA driver");
    require(evo2c::cuda::cuda_status(cudaRuntimeGetVersion(&runtime),
                                     "cudaRuntimeGetVersion"),
            "query CUDA runtime");
    std::cout << "GPU=" << properties.name << " sm=" << properties.major
              << properties.minor << " driver=" << driver
              << " runtime=" << runtime << '\n';

    evo2c::cuda::Stream stream;
    evo2c::cuda::BlasLt blas;
    require(stream.create(), "create CUDA stream");
    require(blas.create(), "create cuBLASLt handle");
    test_buffer(device, stream);
    test_linear(device, stream, blas);
    test_software_e4m3(device, stream);
    test_software_h100_qgmma(device, stream);
    test_rms_norm(device, stream);
    test_elementwise(device, stream);
    test_mlp(device, stream, blas);
    require(stream.synchronize(), "final stream synchronize");
    require(evo2c::cuda::synchronize_device(), "final device synchronize");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " CUDA op test(s) failed\n";
    return 1;
  }
  std::cout << "CUDA op tests passed\n";
  return 0;
}
