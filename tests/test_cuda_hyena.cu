// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "evo2c/cpu_reference.hpp"
#include "evo2c/cuda/hyena.hpp"
#include "evo2c/cuda/runtime.hpp"

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

std::vector<float> quantize_bf16(const std::vector<float> &input) {
  return to_float(to_bf16(input));
}

evo2c::cuda::DeviceBuffer upload(const int device, const void *const source,
                                 const std::size_t bytes,
                                 const evo2c::cuda::Stream &stream) {
  evo2c::cuda::DeviceBuffer buffer;
  require(buffer.allocate(device, bytes), "allocate upload buffer");
  require(buffer.copy_from_host(source, bytes, stream), "upload buffer");
  return buffer;
}

evo2c::cuda::DeviceBuffer upload_bf16(const int device,
                                      const std::vector<float> &input,
                                      const evo2c::cuda::Stream &stream) {
  const auto converted = to_bf16(input);
  return upload(device, converted.data(),
                converted.size() * sizeof(converted[0]), stream);
}

evo2c::cuda::DeviceBuffer upload_f32(const int device,
                                     const std::vector<float> &input,
                                     const evo2c::cuda::Stream &stream) {
  return upload(device, input.data(), input.size() * sizeof(input[0]), stream);
}

std::vector<float> download_bf16(const evo2c::cuda::DeviceBuffer &buffer,
                                 const std::size_t elements,
                                 const evo2c::cuda::Stream &stream) {
  std::vector<__nv_bfloat16> raw(elements);
  require(buffer.copy_to_host(raw.data(), raw.size() * sizeof(raw[0]), stream),
          "download BF16");
  require(stream.synchronize(), "synchronize BF16 download");
  return to_float(raw);
}

std::vector<float> download_f32(const evo2c::cuda::DeviceBuffer &buffer,
                                const std::size_t elements,
                                const evo2c::cuda::Stream &stream) {
  std::vector<float> output(elements);
  require(buffer.copy_to_host(output.data(), output.size() * sizeof(output[0]),
                              stream),
          "download F32");
  require(stream.synchronize(), "synchronize F32 download");
  return output;
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

std::vector<float> expand_filter(const std::vector<float> &grouped,
                                 const std::size_t channels,
                                 const std::size_t groups,
                                 const std::size_t kernel) {
  std::vector<float> expanded(channels * kernel);
  const std::size_t channels_per_group = channels / groups;
  for (std::size_t channel = 0; channel < channels; ++channel) {
    const std::size_t group = channel / channels_per_group;
    std::copy_n(
        grouped.begin() + static_cast<std::ptrdiff_t>(group * kernel), kernel,
        expanded.begin() + static_cast<std::ptrdiff_t>(channel * kernel));
  }
  return expanded;
}

void test_projection_split(const int device,
                           const evo2c::cuda::Stream &stream) {
  constexpr std::size_t length = 3;
  constexpr std::size_t width = 5;
  const auto projection_f32 =
      quantize_bf16(values(length * width * 3, 3, 11.0F));
  auto projection = upload_bf16(device, projection_f32, stream);
  evo2c::cuda::DeviceBuffer x2;
  evo2c::cuda::DeviceBuffer x1;
  evo2c::cuda::DeviceBuffer value;
  const std::size_t bytes = length * width * sizeof(__nv_bfloat16);
  require(x2.allocate(device, bytes), "allocate split x2");
  require(x1.allocate(device, bytes), "allocate split x1");
  require(value.allocate(device, bytes), "allocate split value");
  require(evo2c::cuda::bf16_split_hyena_projection(projection, length, width,
                                                   &x2, &x1, &value, stream),
          "split Hyena projection");
  std::vector<float> expected_x2;
  std::vector<float> expected_x1;
  std::vector<float> expected_value;
  require(evo2c::cpu::split_hyena_projection(projection_f32, length, width,
                                             &expected_x2, &expected_x1,
                                             &expected_value),
          "CPU split reference");
  check(download_bf16(x2, length * width, stream) == expected_x2,
        "Hyena x2 split is bit exact");
  check(download_bf16(x1, length * width, stream) == expected_x1,
        "Hyena x1 split is bit exact");
  check(download_bf16(value, length * width, stream) == expected_value,
        "Hyena value split is bit exact");
}

void test_short_fir_cache(const int device, const evo2c::cuda::Stream &stream) {
  constexpr std::size_t length = 11;
  constexpr std::size_t channels = 7;
  constexpr std::size_t groups = 7;
  constexpr std::size_t kernel = 3;
  auto input_f32 = quantize_bf16(values(length * channels, 7, 13.0F));
  const auto weight_f32 = quantize_bf16(values(groups * kernel, 17, 19.0F));
  const auto bias_f32 = quantize_bf16(values(channels, 31, 23.0F));
  auto input = upload_bf16(device, input_f32, stream);
  auto weight = upload_bf16(device, weight_f32, stream);
  auto bias = upload_bf16(device, bias_f32, stream);
  evo2c::cuda::DeviceBuffer output;
  require(output.allocate(device, input_f32.size() * sizeof(__nv_bfloat16)),
          "allocate short FIR output");
  evo2c::cuda::FirCache cache;
  require(cache.allocate(device, channels, kernel, stream),
          "allocate short FIR cache");
  require(evo2c::cuda::bf16_fir_prefill_direct(
              input, weight, &bias, length, channels, groups, kernel,
              evo2c::cuda::FirOrientation::kCrossCorrelation,
              evo2c::cuda::FirBiasMode::kAdd, &output, &cache, stream),
          "short FIR prefill");
  std::vector<float> expected;
  require(evo2c::cpu::causal_depthwise_fir(
              input_f32, length, channels, weight_f32, kernel, &bias_f32,
              &expected, evo2c::cpu::FirOrientation::kCrossCorrelation,
              evo2c::cpu::FirBiasMode::kAdd),
          "CPU short FIR reference");
  const auto actual = download_bf16(output, input_f32.size(), stream);
  check(all_close(actual, expected, 0.02F, 0.01F),
        "short FIR prefill matches cross-correlation reference");

  const auto next_f32 = quantize_bf16(values(channels, 101, 17.0F));
  auto next = upload_bf16(device, next_f32, stream);
  evo2c::cuda::DeviceBuffer decoded;
  require(decoded.allocate(device, channels * sizeof(__nv_bfloat16)),
          "allocate short FIR decode output");
  require(evo2c::cuda::bf16_fir_decode(
              next, weight, &bias, channels, groups, kernel,
              evo2c::cuda::FirOrientation::kCrossCorrelation,
              evo2c::cuda::FirBiasMode::kAdd, &cache, &decoded, stream),
          "short FIR decode");
  input_f32.insert(input_f32.end(), next_f32.begin(), next_f32.end());
  require(evo2c::cpu::causal_depthwise_fir(
              input_f32, length + 1, channels, weight_f32, kernel, &bias_f32,
              &expected, evo2c::cpu::FirOrientation::kCrossCorrelation,
              evo2c::cpu::FirBiasMode::kAdd),
          "CPU short FIR decode reference");
  const std::vector<float> expected_last(
      expected.end() - static_cast<std::ptrdiff_t>(channels), expected.end());
  check(all_close(download_bf16(decoded, channels, stream), expected_last,
                  0.02F, 0.01F),
        "short FIR cache decode equals full causal last token");

  constexpr std::size_t chunk_length = 4;
  const auto chunk_f32 =
      quantize_bf16(values(chunk_length * channels, 131, 29.0F));
  auto chunk = upload_bf16(device, chunk_f32, stream);
  evo2c::cuda::DeviceBuffer chunk_output;
  require(chunk_output.allocate(device, chunk_f32.size() *
                                            sizeof(__nv_bfloat16)),
          "allocate continued FIR output");
  require(evo2c::cuda::bf16_fir_continue_direct(
              chunk, weight, &bias, chunk_length, channels, groups, kernel,
              evo2c::cuda::FirOrientation::kCrossCorrelation,
              evo2c::cuda::FirBiasMode::kAdd, &chunk_output, &cache, stream),
          "continue short FIR chunk");
  input_f32.insert(input_f32.end(), chunk_f32.begin(), chunk_f32.end());
  require(evo2c::cpu::causal_depthwise_fir(
              input_f32, length + 1 + chunk_length, channels, weight_f32,
              kernel, &bias_f32, &expected,
              evo2c::cpu::FirOrientation::kCrossCorrelation,
              evo2c::cpu::FirBiasMode::kAdd),
          "CPU continued short FIR reference");
  const std::vector<float> expected_chunk(
      expected.end() -
          static_cast<std::ptrdiff_t>(chunk_length * channels),
      expected.end());
  check(all_close(
            download_bf16(chunk_output, chunk_length * channels, stream),
            expected_chunk, 0.02F, 0.01F),
        "continued FIR chunk equals sequential causal reference");
}

void test_hcs(const int device, const evo2c::cuda::Stream &stream) {
  constexpr std::size_t length = 13;
  constexpr std::size_t width = 9;
  constexpr std::size_t groups = 3;
  constexpr std::size_t kernel = 7;
  const auto x2_f32 = quantize_bf16(values(length * width, 5, 17.0F));
  const auto x1_f32 = quantize_bf16(values(length * width, 41, 19.0F));
  const auto value_f32 = quantize_bf16(values(length * width, 73, 23.0F));
  const auto grouped_weight = quantize_bf16(values(groups * kernel, 97, 29.0F));
  auto x2 = upload_bf16(device, x2_f32, stream);
  auto x1 = upload_bf16(device, x1_f32, stream);
  auto value = upload_bf16(device, value_f32, stream);
  auto weight = upload_bf16(device, grouped_weight, stream);
  evo2c::cuda::DeviceBuffer gated;
  evo2c::cuda::DeviceBuffer filtered;
  const std::size_t bytes = length * width * sizeof(__nv_bfloat16);
  require(gated.allocate(device, bytes), "allocate HCS gated input");
  require(filtered.allocate(device, bytes), "allocate HCS output");
  evo2c::cuda::FirCache cache;
  require(cache.allocate(device, width, kernel, stream), "allocate HCS cache");
  require(evo2c::cuda::bf16_hcs_prefill(x2, x1, value, weight, length, width,
                                        groups, kernel, &cache, &gated,
                                        &filtered, stream),
          "HCS prefill");

  std::vector<float> gated_f32(length * width);
  for (std::size_t index = 0; index < gated_f32.size(); ++index)
    gated_f32[index] = x1_f32[index] * value_f32[index];
  gated_f32 = quantize_bf16(gated_f32);
  std::vector<float> expected;
  const auto expanded = expand_filter(grouped_weight, width, groups, kernel);
  require(evo2c::cpu::causal_depthwise_fir(
              gated_f32, length, width, expanded, kernel, nullptr, &expected,
              evo2c::cpu::FirOrientation::kCrossCorrelation),
          "CPU HCS reference");
  for (std::size_t index = 0; index < expected.size(); ++index)
    expected[index] *= x2_f32[index];
  const auto actual = download_bf16(filtered, length * width, stream);
  check(all_close(actual, expected, 0.03F, 0.02F),
        "HCS grouped FIR and outer gate match reference");
  check(cosine(actual, expected) >= 0.999F,
        "HCS output cosine is at least 0.999");

  const auto next_x2 = quantize_bf16(values(width, 151, 31.0F));
  const auto next_x1 = quantize_bf16(values(width, 173, 37.0F));
  const auto next_value = quantize_bf16(values(width, 197, 41.0F));
  std::vector<float> next_gated(width);
  for (std::size_t index = 0; index < width; ++index)
    next_gated[index] = next_x1[index] * next_value[index];
  next_gated = quantize_bf16(next_gated);
  auto next_x2_device = upload_bf16(device, next_x2, stream);
  auto next_x1_device = upload_bf16(device, next_x1, stream);
  auto next_value_device = upload_bf16(device, next_value, stream);
  evo2c::cuda::DeviceBuffer decode_scratch;
  evo2c::cuda::DeviceBuffer decoded;
  require(decode_scratch.allocate(device, width * sizeof(__nv_bfloat16)),
          "allocate HCS decode scratch");
  require(decoded.allocate(device, width * sizeof(__nv_bfloat16)),
          "allocate HCS decode output");
  require(evo2c::cuda::bf16_hcs_decode(
              next_x2_device, next_x1_device, next_value_device, weight, width,
              groups, kernel, &cache, &decode_scratch, &decoded, stream),
          "HCS decode");
  gated_f32.insert(gated_f32.end(), next_gated.begin(), next_gated.end());
  require(evo2c::cpu::causal_depthwise_fir(
              gated_f32, length + 1, width, expanded, kernel, nullptr,
              &expected, evo2c::cpu::FirOrientation::kCrossCorrelation),
          "CPU HCS decode reference");
  std::vector<float> expected_last(
      expected.end() - static_cast<std::ptrdiff_t>(width), expected.end());
  for (std::size_t index = 0; index < width; ++index)
    expected_last[index] *= next_x2[index];
  check(all_close(download_bf16(decoded, width, stream), expected_last, 0.03F,
                  0.02F),
        "HCS cached decode equals full causal last token");
}

void test_hcm_fft_decode(const int device, const evo2c::cuda::Stream &stream) {
  constexpr std::size_t length = 19;
  constexpr std::size_t width = 10;
  constexpr std::size_t groups = 5;
  constexpr std::size_t kernel = 128;
  const auto x2_f32 = quantize_bf16(values(length * width, 11, 31.0F));
  const auto x1_f32 = quantize_bf16(values(length * width, 43, 37.0F));
  const auto value_f32 = quantize_bf16(values(length * width, 79, 41.0F));
  const auto grouped_weight =
      quantize_bf16(values(groups * kernel, 101, 97.0F));
  const auto direct_f32 = quantize_bf16(values(width, 137, 53.0F));
  auto x2 = upload_bf16(device, x2_f32, stream);
  auto x1 = upload_bf16(device, x1_f32, stream);
  auto value = upload_bf16(device, value_f32, stream);
  auto weight = upload_bf16(device, grouped_weight, stream);
  auto direct = upload_bf16(device, direct_f32, stream);
  evo2c::cuda::DeviceBuffer gated;
  evo2c::cuda::DeviceBuffer filtered;
  const std::size_t bytes = length * width * sizeof(__nv_bfloat16);
  require(gated.allocate(device, bytes), "allocate HCM gated input");
  require(filtered.allocate(device, bytes), "allocate HCM output");
  evo2c::cuda::FirCache cache;
  require(cache.allocate(device, width, kernel, stream), "allocate HCM cache");
  const std::size_t fft_size = evo2c::cuda::fir_fft_size(length, kernel);
  evo2c::cuda::FftWorkspace fft;
  require(fft.allocate(device, width, groups, fft_size),
          "allocate HCM FFT workspace");
  require(evo2c::cuda::bf16_hcm_prefill(x2, x1, value, weight, direct, length,
                                        width, groups, kernel, &cache, &gated,
                                        &filtered, &fft, stream),
          "HCM FFT prefill");

  std::vector<float> gated_f32(length * width);
  for (std::size_t index = 0; index < gated_f32.size(); ++index)
    gated_f32[index] = x1_f32[index] * value_f32[index];
  gated_f32 = quantize_bf16(gated_f32);
  const auto expanded = expand_filter(grouped_weight, width, groups, kernel);
  std::vector<float> expected;
  require(evo2c::cpu::causal_depthwise_fir(
              gated_f32, length, width, expanded, kernel, &direct_f32,
              &expected, evo2c::cpu::FirOrientation::kCausalConvolution,
              evo2c::cpu::FirBiasMode::kMultiplyInput),
          "CPU HCM reference");
  for (std::size_t index = 0; index < expected.size(); ++index)
    expected[index] *= x2_f32[index];
  auto actual = download_bf16(filtered, length * width, stream);
  check(all_close(actual, expected, 0.04F, 0.03F),
        "HCM cuFFT prefill matches causal grouped reference");
  check(cosine(actual, expected) >= 0.999F,
        "HCM cuFFT output cosine is at least 0.999");

  const auto next_x2 = quantize_bf16(values(width, 173, 43.0F));
  const auto next_x1 = quantize_bf16(values(width, 191, 47.0F));
  const auto next_value = quantize_bf16(values(width, 211, 59.0F));
  std::vector<float> next_gated(width);
  for (std::size_t index = 0; index < width; ++index)
    next_gated[index] = next_x1[index] * next_value[index];
  next_gated = quantize_bf16(next_gated);
  auto next_x2_device = upload_bf16(device, next_x2, stream);
  auto next_x1_device = upload_bf16(device, next_x1, stream);
  auto next_value_device = upload_bf16(device, next_value, stream);
  evo2c::cuda::DeviceBuffer decode_scratch;
  evo2c::cuda::DeviceBuffer decoded;
  require(decode_scratch.allocate(device, width * sizeof(__nv_bfloat16)),
          "allocate HCM decode scratch");
  require(decoded.allocate(device, width * sizeof(__nv_bfloat16)),
          "allocate HCM decode output");
  require(evo2c::cuda::bf16_hcm_decode(
              next_x2_device, next_x1_device, next_value_device, weight, direct,
              width, groups, kernel, &cache, &decode_scratch, &decoded, stream),
          "HCM decode");
  gated_f32.insert(gated_f32.end(), next_gated.begin(), next_gated.end());
  require(evo2c::cpu::causal_depthwise_fir(
              gated_f32, length + 1, width, expanded, kernel, &direct_f32,
              &expected, evo2c::cpu::FirOrientation::kCausalConvolution,
              evo2c::cpu::FirBiasMode::kMultiplyInput),
          "CPU HCM decode reference");
  std::vector<float> expected_last(
      expected.end() - static_cast<std::ptrdiff_t>(width), expected.end());
  for (std::size_t index = 0; index < width; ++index)
    expected_last[index] *= next_x2[index];
  check(all_close(download_bf16(decoded, width, stream), expected_last, 0.04F,
                  0.03F),
        "HCM cached decode equals full causal last token");

  constexpr std::size_t chunk_length = 5;
  const auto chunk_x2 =
      quantize_bf16(values(chunk_length * width, 251, 67.0F));
  const auto chunk_x1 =
      quantize_bf16(values(chunk_length * width, 277, 71.0F));
  const auto chunk_value =
      quantize_bf16(values(chunk_length * width, 307, 73.0F));
  auto chunk_x2_device = upload_bf16(device, chunk_x2, stream);
  auto chunk_x1_device = upload_bf16(device, chunk_x1, stream);
  auto chunk_value_device = upload_bf16(device, chunk_value, stream);
  evo2c::cuda::DeviceBuffer chunk_scratch;
  evo2c::cuda::DeviceBuffer chunk_output;
  const std::size_t chunk_bytes =
      chunk_length * width * sizeof(__nv_bfloat16);
  require(chunk_scratch.allocate(device, chunk_bytes),
          "allocate HCM continuation scratch");
  require(chunk_output.allocate(device, chunk_bytes),
          "allocate HCM continuation output");
  require(evo2c::cuda::bf16_hcm_continue(
              chunk_x2_device, chunk_x1_device, chunk_value_device, weight,
              direct, chunk_length, width, groups, kernel, &cache,
              &chunk_scratch, &chunk_output, stream),
          "continue HCM chunk");
  std::vector<float> chunk_gated(chunk_length * width);
  for (std::size_t index = 0; index < chunk_gated.size(); ++index)
    chunk_gated[index] = chunk_x1[index] * chunk_value[index];
  chunk_gated = quantize_bf16(chunk_gated);
  gated_f32.insert(gated_f32.end(), chunk_gated.begin(), chunk_gated.end());
  require(evo2c::cpu::causal_depthwise_fir(
              gated_f32, length + 1 + chunk_length, width, expanded, kernel,
              &direct_f32, &expected,
              evo2c::cpu::FirOrientation::kCausalConvolution,
              evo2c::cpu::FirBiasMode::kMultiplyInput),
          "CPU HCM continuation reference");
  std::vector<float> expected_chunk(
      expected.end() -
          static_cast<std::ptrdiff_t>(chunk_length * width),
      expected.end());
  for (std::size_t index = 0; index < expected_chunk.size(); ++index)
    expected_chunk[index] *= chunk_x2[index];
  check(all_close(
            download_bf16(chunk_output, chunk_length * width, stream),
            expected_chunk, 0.04F, 0.03F),
        "HCM continuation equals sequential causal reference");
}

void test_hcl(const int device, const evo2c::cuda::Stream &stream) {
  constexpr std::size_t length = 9;
  constexpr std::size_t width = 7;
  constexpr std::size_t state_size = 3;
  const auto x2_f32 = quantize_bf16(values(length * width, 13, 31.0F));
  const auto x1_f32 = quantize_bf16(values(length * width, 47, 37.0F));
  const auto value_f32 = quantize_bf16(values(length * width, 83, 41.0F));
  const auto direct_f32 = quantize_bf16(values(width, 109, 43.0F));
  std::vector<float> log_poles(width * state_size);
  std::vector<float> residues(width * state_size);
  for (std::size_t index = 0; index < log_poles.size(); ++index) {
    log_poles[index] = -0.01F * static_cast<float>(index % state_size + 1);
    residues[index] = values(1, index + 131, 47.0F)[0];
  }
  auto x2 = upload_bf16(device, x2_f32, stream);
  auto x1 = upload_bf16(device, x1_f32, stream);
  auto value = upload_bf16(device, value_f32, stream);
  auto direct = upload_bf16(device, direct_f32, stream);
  auto poles = upload_f32(device, log_poles, stream);
  auto residue = upload_f32(device, residues, stream);
  const std::size_t bytes = length * width * sizeof(__nv_bfloat16);
  evo2c::cuda::DeviceBuffer scratch_recurrence;
  evo2c::cuda::DeviceBuffer scratch_fft;
  evo2c::cuda::DeviceBuffer output_recurrence;
  evo2c::cuda::DeviceBuffer output_fft;
  require(scratch_recurrence.allocate(device, bytes),
          "allocate HCL recurrence scratch");
  require(scratch_fft.allocate(device, bytes), "allocate HCL FFT scratch");
  require(output_recurrence.allocate(device, bytes),
          "allocate HCL recurrence output");
  require(output_fft.allocate(device, bytes), "allocate HCL FFT output");
  evo2c::cuda::IirCache recurrence_cache;
  evo2c::cuda::IirCache fft_cache;
  require(recurrence_cache.allocate(device, width, state_size, stream),
          "allocate HCL recurrence cache");
  require(fft_cache.allocate(device, width, state_size, stream),
          "allocate HCL FFT cache");
  require(evo2c::cuda::bf16_hcl_prefill(
              x2, x1, value, direct, poles, residue, length, width, state_size,
              evo2c::cuda::HclPrefillMode::kRecurrence, &recurrence_cache,
              &scratch_recurrence, &output_recurrence, nullptr, stream),
          "HCL recurrence prefill");
  const std::size_t fft_size = evo2c::cuda::fir_fft_size(length, length);
  evo2c::cuda::FftWorkspace fft;
  require(fft.allocate(device, width, width, fft_size),
          "allocate HCL FFT workspace");
  require(evo2c::cuda::bf16_hcl_prefill(
              x2, x1, value, direct, poles, residue, length, width, state_size,
              evo2c::cuda::HclPrefillMode::kFft, &fft_cache, &scratch_fft,
              &output_fft, &fft, stream),
          "HCL FFT prefill");

  std::vector<float> gated(length * width);
  for (std::size_t index = 0; index < gated.size(); ++index)
    gated[index] = x1_f32[index] * value_f32[index];
  gated = quantize_bf16(gated);
  std::vector<float> ones(gated.size(), 1.0F);
  std::vector<float> expected_state(width * state_size, 0.0F);
  std::vector<float> expected;
  require(evo2c::cpu::hcl_recurrence(x2_f32, gated, ones, length, width,
                                     direct_f32, log_poles, residues,
                                     state_size, &expected_state, &expected),
          "CPU HCL reference");
  const auto recurrence_actual =
      download_bf16(output_recurrence, length * width, stream);
  const auto fft_actual = download_bf16(output_fft, length * width, stream);
  check(all_close(recurrence_actual, expected, 0.03F, 0.02F),
        "HCL recurrent prefill matches modal reference");
  check(all_close(fft_actual, expected, 0.04F, 0.03F),
        "HCL cuFFT prefill matches modal reference");
  check(cosine(fft_actual, expected) >= 0.999F,
        "HCL cuFFT output cosine is at least 0.999");
  check(all_close(
            download_f32(recurrence_cache.state, width * state_size, stream),
            expected_state, 1.0e-5F, 1.0e-5F),
        "HCL recurrent prefill returns exact F32 final state");
  check(all_close(download_f32(fft_cache.state, width * state_size, stream),
                  expected_state, 1.0e-5F, 1.0e-5F),
        "HCL cuFFT prefill returns exact F32 final state");

  const auto next_x2 = quantize_bf16(values(width, 173, 53.0F));
  const auto next_x1 = quantize_bf16(values(width, 193, 59.0F));
  const auto next_value = quantize_bf16(values(width, 223, 61.0F));
  auto next_x2_device = upload_bf16(device, next_x2, stream);
  auto next_x1_device = upload_bf16(device, next_x1, stream);
  auto next_value_device = upload_bf16(device, next_value, stream);
  evo2c::cuda::DeviceBuffer decode_recurrence;
  evo2c::cuda::DeviceBuffer decode_fft;
  require(decode_recurrence.allocate(device, width * sizeof(__nv_bfloat16)),
          "allocate HCL recurrent decode output");
  require(decode_fft.allocate(device, width * sizeof(__nv_bfloat16)),
          "allocate HCL FFT decode output");
  require(evo2c::cuda::bf16_hcl_decode(
              next_x2_device, next_x1_device, next_value_device, direct, poles,
              residue, width, state_size, &recurrence_cache, &decode_recurrence,
              stream),
          "HCL recurrent-cache decode");
  require(evo2c::cuda::bf16_hcl_decode(
              next_x2_device, next_x1_device, next_value_device, direct, poles,
              residue, width, state_size, &fft_cache, &decode_fft, stream),
          "HCL FFT-cache decode");
  std::vector<float> next_gated(width);
  for (std::size_t index = 0; index < width; ++index)
    next_gated[index] = next_x1[index] * next_value[index];
  next_gated = quantize_bf16(next_gated);
  std::vector<float> expected_decode;
  require(evo2c::cpu::hcl_recurrence(
              next_x2, next_gated, std::vector<float>(width, 1.0F), 1, width,
              direct_f32, log_poles, residues, state_size, &expected_state,
              &expected_decode),
          "CPU HCL decode reference");
  check(all_close(download_bf16(decode_recurrence, width, stream),
                  expected_decode, 0.03F, 0.02F),
        "HCL recurrent cache decode matches continued reference");
  check(all_close(download_bf16(decode_fft, width, stream), expected_decode,
                  0.03F, 0.02F),
        "HCL FFT cache decode matches continued reference");

  constexpr std::size_t chunk_length = 4;
  const auto chunk_x2 =
      quantize_bf16(values(chunk_length * width, 257, 67.0F));
  const auto chunk_x1 =
      quantize_bf16(values(chunk_length * width, 281, 71.0F));
  const auto chunk_value =
      quantize_bf16(values(chunk_length * width, 313, 73.0F));
  auto chunk_x2_device = upload_bf16(device, chunk_x2, stream);
  auto chunk_x1_device = upload_bf16(device, chunk_x1, stream);
  auto chunk_value_device = upload_bf16(device, chunk_value, stream);
  const std::size_t chunk_bytes =
      chunk_length * width * sizeof(__nv_bfloat16);
  evo2c::cuda::DeviceBuffer chunk_scratch;
  evo2c::cuda::DeviceBuffer chunk_output;
  require(chunk_scratch.allocate(device, chunk_bytes),
          "allocate HCL continuation scratch");
  require(chunk_output.allocate(device, chunk_bytes),
          "allocate HCL continuation output");
  require(evo2c::cuda::bf16_hcl_prefill(
              chunk_x2_device, chunk_x1_device, chunk_value_device, direct,
              poles, residue, chunk_length, width, state_size,
              evo2c::cuda::HclPrefillMode::kRecurrenceContinue,
              &recurrence_cache, &chunk_scratch, &chunk_output, nullptr,
              stream),
          "continue HCL recurrence chunk");
  std::vector<float> chunk_gated(chunk_length * width);
  for (std::size_t index = 0; index < chunk_gated.size(); ++index)
    chunk_gated[index] = chunk_x1[index] * chunk_value[index];
  chunk_gated = quantize_bf16(chunk_gated);
  std::vector<float> expected_chunk;
  require(evo2c::cpu::hcl_recurrence(
              chunk_x2, chunk_gated,
              std::vector<float>(chunk_length * width, 1.0F), chunk_length,
              width, direct_f32, log_poles, residues, state_size,
              &expected_state, &expected_chunk),
          "CPU HCL continuation reference");
  check(all_close(
            download_bf16(chunk_output, chunk_length * width, stream),
            expected_chunk, 0.03F, 0.02F),
        "HCL recurrent chunk continues the existing F32 state");
  check(all_close(
            download_f32(recurrence_cache.state, width * state_size, stream),
            expected_state, 1.0e-5F, 1.0e-5F),
        "HCL recurrent chunk publishes the continued F32 final state");
}

int requested_device() {
  const char *environment = std::getenv("EVO2C_TEST_DEVICE");
  return environment == nullptr ? 0 : std::stoi(environment);
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
    std::cout << "GPU=" << properties.name << " sm=" << properties.major
              << properties.minor << '\n';
    evo2c::cuda::Stream stream;
    require(stream.create(), "create CUDA stream");
    test_projection_split(device, stream);
    test_short_fir_cache(device, stream);
    test_hcs(device, stream);
    test_hcm_fft_decode(device, stream);
    test_hcl(device, stream);
    require(stream.synchronize(), "final stream synchronize");
    require(evo2c::cuda::synchronize_device(), "final device synchronize");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " CUDA Hyena test(s) failed\n";
    return 1;
  }
  std::cout << "CUDA Hyena tests passed\n";
  return 0;
}
