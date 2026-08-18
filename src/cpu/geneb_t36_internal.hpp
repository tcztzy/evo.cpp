// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../linear_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "evo/model_format.hpp"
#include "evo/status.hpp"

namespace evo::cpu::t36 {

inline bool checked_product(const std::size_t left, const std::size_t right,
                            std::size_t *const output) noexcept {
  if (output == nullptr ||
      (left != 0 && right > std::numeric_limits<std::size_t>::max() / left))
    return false;
  *output = left * right;
  return true;
}

inline Status metadata_entry(const ModelFile &artifact,
                             const std::string_view key,
                             const MetadataType type,
                             const MetadataEntry **const output,
                             const std::string_view family) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr)
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " metadata is missing: " +
                                         std::string{key}};
  if (entry->type != type)
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " metadata has wrong type: " +
                                         std::string{key}};
  *output = entry;
  return Status::Ok();
}

inline Status metadata_string(const ModelFile &artifact,
                              const std::string_view key, std::string *output,
                              const std::string_view family) {
  const MetadataEntry *entry = nullptr;
  auto status =
      metadata_entry(artifact, key, MetadataType::kString, &entry, family);
  if (!status.ok())
    return status;
  output->assign(entry->value.begin(), entry->value.end());
  return Status::Ok();
}

inline Status metadata_literal(const ModelFile &artifact,
                               const std::string_view key,
                               const std::string_view expected,
                               const std::string_view family) {
  std::string actual;
  auto status = metadata_string(artifact, key, &actual, family);
  if (!status.ok())
    return status;
  if (actual != expected)
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " metadata differs: " +
                                         std::string{key}};
  return Status::Ok();
}

inline Status metadata_size(const ModelFile &artifact,
                            const std::string_view key, std::size_t *output,
                            const std::string_view family) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kU64, &entry, family);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(std::uint64_t))
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " u64 metadata is malformed"};
  std::uint64_t value = 0;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (value > std::numeric_limits<std::size_t>::max())
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " metadata exceeds size_t"};
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

inline Status metadata_float(const ModelFile &artifact,
                             const std::string_view key, float *output,
                             const std::string_view family) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kF64, &entry, family);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(double))
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " f64 metadata is malformed"};
  double value = 0.0;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (!std::isfinite(value) ||
      std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " f64 metadata is not finite F32"};
  *output = static_cast<float>(value);
  return Status::Ok();
}

inline Status metadata_bool(const ModelFile &artifact,
                            const std::string_view key, bool *output,
                            const std::string_view family) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kBool, &entry, family);
  if (!status.ok())
    return status;
  if (entry->value.size() != 1)
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " bool metadata is malformed"};
  *output = entry->value[0] != 0;
  return Status::Ok();
}

template <typename View>
inline float value(const View &tensor, const std::size_t index) noexcept {
  float result = 0.0F;
  std::memcpy(&result, tensor.data + index * sizeof(float), sizeof(result));
  return result;
}

template <typename View, typename Requirement>
inline Status tensor_requirement(const View &tensor,
                                 const Requirement &requirement,
                                 const std::string_view family) {
  if (tensor.dtype != TensorDType::kF32 ||
      requirement.dtype != TensorDType::kF32 ||
      tensor.shape != requirement.shape || tensor.data == nullptr)
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " tensor type/shape differs: " +
                                         requirement.name};
  std::size_t elements = 1;
  for (const std::size_t dimension : tensor.shape) {
    if (dimension == 0 || !checked_product(elements, dimension, &elements))
      return {ErrorCode::kModelFormat, std::string{family} +
                                           " tensor shape overflows: " +
                                           requirement.name};
  }
  std::size_t bytes = 0;
  if (!checked_product(elements, sizeof(float), &bytes) || bytes != tensor.bytes)
    return {ErrorCode::kModelFormat, std::string{family} +
                                         " tensor byte extent differs: " +
                                         requirement.name};
  return Status::Ok();
}

template <typename View>
inline Status linear(const std::vector<float> &input, const std::size_t rows,
                     const std::size_t input_width, const View &weight,
                     const std::size_t output_width, const View *const bias,
                     evo::detail::LinearExecutor *const executor,
                     std::vector<float> *const output,
                     const std::string_view family) {
  std::size_t input_elements = 0;
  std::size_t output_elements = 0;
  if (output == nullptr ||
      !checked_product(rows, input_width, &input_elements) ||
      input.size() != input_elements ||
      !checked_product(rows, output_width, &output_elements) ||
      weight.shape != std::vector<std::size_t>{output_width, input_width} ||
      weight.dtype != TensorDType::kF32 || weight.data == nullptr ||
      (bias != nullptr &&
       (bias->shape != std::vector<std::size_t>{output_width} ||
        bias->dtype != TensorDType::kF32 || bias->data == nullptr)))
    return {ErrorCode::kInvalidArgument,
            std::string{family} + " linear arguments are inconsistent"};
  if (executor != nullptr) {
    const evo::detail::LinearTensorView weight_view{
        weight.data, weight.dtype, output_width * input_width};
    evo::detail::LinearTensorView bias_view;
    const evo::detail::LinearTensorView *bias_pointer = nullptr;
    if (bias != nullptr) {
      bias_view = {bias->data, bias->dtype, output_width};
      bias_pointer = &bias_view;
    }
    auto status = executor->linear(input.data(), rows, input_width, weight_view,
                                   output_width, bias_pointer, output);
    if (!status.ok())
      return status;
  } else {
    output->assign(output_elements, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t out = 0; out < output_width; ++out) {
        float sum = bias == nullptr ? 0.0F : value(*bias, out);
        const std::size_t weight_base = out * input_width;
        const std::size_t input_base = row * input_width;
        for (std::size_t in = 0; in < input_width; ++in)
          sum += input[input_base + in] * value(weight, weight_base + in);
        (*output)[row * output_width + out] = sum;
      }
    }
  }
  for (const float item : *output) {
    if (!std::isfinite(item))
      return {ErrorCode::kInvalidArgument,
              std::string{family} + " linear output became non-finite"};
  }
  return Status::Ok();
}

template <typename View>
inline Status layer_norm(const std::vector<float> &input,
                         const std::size_t rows, const std::size_t width,
                         const View &scale, const View &bias,
                         const float epsilon, std::vector<float> *const output,
                         const std::string_view family) {
  std::size_t elements = 0;
  if (output == nullptr || !checked_product(rows, width, &elements) ||
      input.size() != elements ||
      scale.shape != std::vector<std::size_t>{width} ||
      bias.shape != std::vector<std::size_t>{width} ||
      scale.data == nullptr || bias.data == nullptr ||
      !std::isfinite(epsilon) || epsilon <= 0.0F)
    return {ErrorCode::kInvalidArgument,
            std::string{family} + " LayerNorm arguments are inconsistent"};
  output->resize(elements);
  for (std::size_t row = 0; row < rows; ++row) {
    double mean = 0.0;
    for (std::size_t column = 0; column < width; ++column)
      mean += input[row * width + column];
    mean /= static_cast<double>(width);
    double variance = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      const double centered = input[row * width + column] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<double>(width);
    const float inverse =
        1.0F / std::sqrt(static_cast<float>(variance) + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      const float normalized =
          (input[row * width + column] - static_cast<float>(mean)) * inverse;
      (*output)[row * width + column] =
          normalized * value(scale, column) + value(bias, column);
    }
  }
  return Status::Ok();
}

inline float erf_gelu(const float input) noexcept {
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  return 0.5F * input * (1.0F + std::erf(input * kInverseSqrtTwo));
}

inline bool finite(const std::vector<float> &values) noexcept {
  return std::all_of(values.begin(), values.end(),
                     [](const float value) { return std::isfinite(value); });
}

} // namespace evo::cpu::t36
