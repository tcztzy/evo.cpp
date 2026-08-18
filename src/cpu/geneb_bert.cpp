// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_bert.hpp"

#include "../linear_executor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace evo::cpu {
namespace {

constexpr std::size_t kMaximumTopologyLayers = 1024;
constexpr std::size_t kMaximumTopologyDimension = 1U << 24U;

Status metadata_entry(const ModelFile &artifact, const std::string_view key,
                      const MetadataType type,
                      const MetadataEntry **const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr) {
    return {ErrorCode::kModelFormat,
            "required GENEB BERT metadata is missing: " + std::string{key}};
  }
  if (entry->type != type) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT metadata has wrong type: " + std::string{key}};
  }
  *output = entry;
  return Status::Ok();
}

Status metadata_string(const ModelFile &artifact, const std::string_view key,
                       std::string *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kString, &entry);
  if (!status.ok())
    return status;
  output->assign(entry->value.begin(), entry->value.end());
  return Status::Ok();
}

Status metadata_size(const ModelFile &artifact, const std::string_view key,
                     std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(std::uint64_t)) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT u64 metadata is malformed: " + std::string{key}};
  }
  std::uint64_t value = 0;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (value > std::numeric_limits<std::size_t>::max()) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT metadata exceeds size_t: " + std::string{key}};
  }
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

Status metadata_float(const ModelFile &artifact, const std::string_view key,
                      float *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kF64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(double)) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT f64 metadata is malformed: " + std::string{key}};
  }
  double value = 0.0;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (!std::isfinite(value) ||
      value > static_cast<double>(std::numeric_limits<float>::max()) ||
      value < -static_cast<double>(std::numeric_limits<float>::max())) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT metadata is not finite F32: " + std::string{key}};
  }
  *output = static_cast<float>(value);
  return Status::Ok();
}

Status metadata_bool(const ModelFile &artifact, const std::string_view key,
                     bool *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kBool, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != 1) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT bool metadata is malformed: " + std::string{key}};
  }
  *output = entry->value[0] != 0;
  return Status::Ok();
}

Status metadata_literal(const ModelFile &artifact, const std::string_view key,
                        const std::string_view expected) {
  std::string actual;
  auto status = metadata_string(artifact, key, &actual);
  if (!status.ok())
    return status;
  if (actual != expected) {
    return {ErrorCode::kModelFormat, "GENEB BERT metadata mismatch for '" +
                                         std::string{key} + "': expected '" +
                                         std::string{expected} + "'"};
  }
  return Status::Ok();
}

Status metadata_dtype(const ModelFile &artifact, const std::string_view key,
                      TensorDType *const output) {
  std::string value;
  auto status = metadata_string(artifact, key, &value);
  if (!status.ok())
    return status;
  if (value == "F32")
    *output = TensorDType::kF32;
  else if (value == "BF16")
    *output = TensorDType::kBF16;
  else
    return {ErrorCode::kModelFormat,
            "GENEB BERT dtype metadata must be F32 or BF16: " +
                std::string{key}};
  return Status::Ok();
}

bool checked_product(const std::size_t left, const std::size_t right,
                     std::size_t *const output) noexcept {
  if (output == nullptr ||
      (left != 0 && right > std::numeric_limits<std::size_t>::max() / left))
    return false;
  *output = left * right;
  return true;
}

std::size_t dtype_size(const TensorDType dtype) noexcept {
  switch (dtype) {
  case TensorDType::kF32:
    return sizeof(float);
  case TensorDType::kBF16:
    return sizeof(std::uint16_t);
  case TensorDType::kE4M3Software:
    return 0;
  }
  return 0;
}

float tensor_value(const GenebBertTensorView &tensor,
                   const std::size_t index) noexcept {
  if (tensor.dtype == TensorDType::kBF16) {
    const auto *const source = tensor.data + index * sizeof(std::uint16_t);
    const std::uint32_t bits = static_cast<std::uint32_t>(source[0]) << 16U |
                               static_cast<std::uint32_t>(source[1]) << 24U;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(float), sizeof(value));
  return value;
}

float round_to_bf16(const float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) != 0x7f800000U)
    bits += 0x7fffU + ((bits >> 16U) & 1U);
  bits &= 0xffff0000U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float activation_value(const float value, const TensorDType dtype) noexcept {
  return dtype == TensorDType::kBF16 ? round_to_bf16(value) : value;
}

void round_activations(std::vector<float> *const values,
                       const TensorDType dtype) noexcept {
  if (dtype != TensorDType::kBF16)
    return;
  for (float &value : *values)
    value = round_to_bf16(value);
}

bool finite_vector(const std::vector<float> &values) noexcept {
  for (const float value : values) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

struct LayerNormMoments final {
  float mean{0.0F};
  float variance{0.0F};
};

// Match the F32 Welford/cascade reduction used by the pinned PyTorch CPU
// LayerNorm.  GENA-LM intentionally has no final normalization, so even a
// one-ULP embedding-normalization drift can be amplified by its residual
// stream into a visible checkpoint-oracle failure.
constexpr std::size_t kMomentLanes = 8U;
constexpr std::size_t kMomentChunk = 16U;
using MomentLanes = std::array<float, kMomentLanes>;

void add_moments(const std::size_t added_count, const float added_mean,
                 const float added_m2, std::size_t *const count,
                 float *const mean, float *const m2) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  const std::size_t combined = *count + added_count;
  const float ratio =
      combined == 0U ? 0.0F
                     : static_cast<float>(added_count) /
                           static_cast<float>(combined);
  const float delta = added_mean - *mean;
  *mean += ratio * delta;
  *m2 += added_m2 + delta * delta * ratio * static_cast<float>(*count);
  *count = combined;
}

void add_moment_lanes(const std::size_t added_count,
                      const MomentLanes &added_mean,
                      const MomentLanes &added_m2,
                      std::size_t *const count, MomentLanes *const mean,
                      MomentLanes *const m2) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  const std::size_t combined = *count + added_count;
  const float ratio =
      combined == 0U ? 0.0F
                     : static_cast<float>(added_count) /
                           static_cast<float>(combined);
#if defined(__aarch64__) || defined(_M_ARM64)
  const float32x4_t ratio_vector = vdupq_n_f32(ratio);
  const float32x4_t count_vector =
      vdupq_n_f32(static_cast<float>(*count));
  for (std::size_t half = 0; half < 2U; ++half) {
    const std::size_t offset = half * 4U;
    const float32x4_t added_mean_vector = vld1q_f32(added_mean.data() + offset);
    const float32x4_t mean_vector = vld1q_f32(mean->data() + offset);
    const float32x4_t delta =
        vsubq_f32(added_mean_vector, mean_vector);
    const float32x4_t next_mean =
        vaddq_f32(mean_vector, vmulq_f32(ratio_vector, delta));
    const float32x4_t delta_squared = vmulq_f32(delta, delta);
    const float32x4_t weighted_delta = vmulq_f32(
        vmulq_f32(delta_squared, ratio_vector), count_vector);
    const float32x4_t added_m2_vector = vld1q_f32(added_m2.data() + offset);
    const float32x4_t m2_vector = vld1q_f32(m2->data() + offset);
    const float32x4_t next_m2 = vaddq_f32(
        m2_vector, vaddq_f32(added_m2_vector, weighted_delta));
    vst1q_f32(mean->data() + offset, next_mean);
    vst1q_f32(m2->data() + offset, next_m2);
  }
#else
  for (std::size_t lane = 0; lane < kMomentLanes; ++lane) {
    const float delta = added_mean[lane] - (*mean)[lane];
    (*mean)[lane] += ratio * delta;
    (*m2)[lane] += added_m2[lane] + delta * delta * ratio *
                                             static_cast<float>(*count);
  }
#endif
  *count = combined;
}

void update_moment_lanes(const float *const values, const float ratio,
                         MomentLanes *const mean,
                         MomentLanes *const m2) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
  const float32x4_t ratio_vector = vdupq_n_f32(ratio);
  for (std::size_t half = 0; half < 2U; ++half) {
    const std::size_t offset = half * 4U;
    const float32x4_t value_vector = vld1q_f32(values + offset);
    const float32x4_t mean_vector = vld1q_f32(mean->data() + offset);
    const float32x4_t delta = vsubq_f32(value_vector, mean_vector);
    const float32x4_t next_mean =
        vaddq_f32(mean_vector, vmulq_f32(delta, ratio_vector));
    const float32x4_t m2_vector = vld1q_f32(m2->data() + offset);
    const float32x4_t next_m2 = vaddq_f32(
        m2_vector, vmulq_f32(delta, vsubq_f32(value_vector, next_mean)));
    vst1q_f32(mean->data() + offset, next_mean);
    vst1q_f32(m2->data() + offset, next_m2);
  }
#else
  for (std::size_t lane = 0; lane < kMomentLanes; ++lane) {
    const float value = values[lane];
    const float delta = value - (*mean)[lane];
    (*mean)[lane] += delta * ratio;
    (*m2)[lane] += delta * (value - (*mean)[lane]);
  }
#endif
}

LayerNormMoments rowwise_moments(const float *const input,
                                 const std::size_t width) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  const std::size_t vectors = width / kMomentLanes;
  const std::size_t chunks =
      (vectors + kMomentChunk - 1U) / kMomentChunk;
  if (chunks == 0U) {
    std::size_t count = 0U;
    float mean = 0.0F;
    float m2 = 0.0F;
    for (std::size_t index = 0; index < width; ++index) {
      const float delta = input[index] - mean;
      ++count;
      mean += delta / static_cast<float>(count);
      m2 += delta * (input[index] - mean);
    }
    return {mean, m2 / static_cast<float>(width)};
  }

  std::size_t depth = 1U;
  for (std::size_t extent = 2U; extent < chunks; extent *= 2U)
    ++depth;
  std::vector<std::size_t> counts(depth, 0U);
  std::vector<MomentLanes> means(depth, MomentLanes{});
  std::vector<MomentLanes> m2s(depth, MomentLanes{});

  for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
    const std::size_t chunk_vectors =
        std::min(kMomentChunk, vectors - chunk * kMomentChunk);
    MomentLanes chunk_mean{};
    MomentLanes chunk_m2{};
    for (std::size_t item = 0; item < chunk_vectors; ++item) {
      const float ratio = 1.0F / static_cast<float>(item + 1U);
      const std::size_t base =
          (chunk * kMomentChunk + item) * kMomentLanes;
      update_moment_lanes(input + base, ratio, &chunk_mean, &chunk_m2);
    }
    add_moment_lanes(chunk_vectors, chunk_mean, chunk_m2, &counts[0],
                     &means[0], &m2s[0]);

    std::size_t mask = chunk + 1U;
    for (std::size_t level = 1U;
         level < depth && (mask & 1U) == 0U; ++level) {
      add_moment_lanes(counts[level - 1U], means[level - 1U],
                       m2s[level - 1U], &counts[level], &means[level],
                       &m2s[level]);
      counts[level - 1U] = 0U;
      means[level - 1U].fill(0.0F);
      m2s[level - 1U].fill(0.0F);
      mask >>= 1U;
    }
  }
  for (std::size_t level = 1U; level < depth; ++level) {
    add_moment_lanes(counts[level], means[level], m2s[level], &counts[0],
                     &means[0], &m2s[0]);
  }

  std::size_t count = 0U;
  float mean = 0.0F;
  float m2 = 0.0F;
  for (std::size_t index = vectors * kMomentLanes; index < width; ++index) {
    const float delta = input[index] - mean;
    ++count;
    mean += delta / static_cast<float>(count);
    m2 += delta * (input[index] - mean);
  }
  const std::size_t lane_count =
      vectors * kMomentLanes / kMomentLanes;
  for (std::size_t lane = 0; lane < kMomentLanes; ++lane) {
    add_moments(lane_count, means[0][lane], m2s[0][lane], &count, &mean,
                &m2);
  }
  return {mean, m2 / static_cast<float>(width)};
}

Status matrix_elements(const std::size_t rows, const std::size_t columns,
                       const std::size_t actual, const std::string_view name) {
  std::size_t expected = 0;
  if (rows == 0 || columns == 0 || !checked_product(rows, columns, &expected)) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " dimensions are zero or overflow"};
  }
  if (expected != actual) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " has " + std::to_string(actual) +
                " elements; expected " + std::to_string(expected)};
  }
  return Status::Ok();
}

Status tensor_requirement(const GenebBertTensorView &tensor,
                          const GenebBertTensorRequirement &requirement) {
  if (tensor.dtype != requirement.dtype) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT tensor dtype mismatch: " + requirement.name};
  }
  if (tensor.shape != requirement.shape) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT tensor shape mismatch: " + requirement.name};
  }
  std::size_t elements = 1;
  for (const std::size_t dimension : tensor.shape) {
    if (dimension == 0 || !checked_product(elements, dimension, &elements)) {
      return {ErrorCode::kModelFormat,
              "GENEB BERT tensor dimensions overflow: " + requirement.name};
    }
  }
  std::size_t expected_bytes = 0;
  const std::size_t element_bytes = dtype_size(tensor.dtype);
  if (element_bytes == 0 ||
      !checked_product(elements, element_bytes, &expected_bytes) ||
      tensor.bytes != expected_bytes || tensor.data == nullptr) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT tensor payload mismatch: " + requirement.name};
  }
  return Status::Ok();
}

std::string layer_prefix(const std::size_t layer) {
  return "bert.encoder.layer." + std::to_string(layer) + ".";
}

void add_requirement(std::vector<GenebBertTensorRequirement> *const output,
                     std::string name, const TensorDType dtype,
                     std::vector<std::size_t> shape) {
  output->push_back({std::move(name), dtype, std::move(shape)});
}

Status runtime_linear(const std::vector<float> &input, const std::size_t rows,
                      const std::size_t input_width,
                      const GenebBertTensorView &weight,
                      const std::size_t output_width,
                      const GenebBertTensorView *const bias,
                      evo::detail::LinearExecutor *const executor,
                      const TensorDType activation_dtype,
                      std::vector<float> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT linear output is null"};
  auto status =
      matrix_elements(rows, input_width, input.size(), "BERT linear input");
  if (!status.ok())
    return status;
  std::size_t weight_elements = 0;
  std::size_t weight_bytes = 0;
  std::size_t bias_bytes = 0;
  std::size_t output_elements = 0;
  if (!checked_product(output_width, input_width, &weight_elements) ||
      !checked_product(weight_elements, dtype_size(weight.dtype),
                       &weight_bytes) ||
      !checked_product(output_width,
                       bias == nullptr ? 0 : dtype_size(bias->dtype),
                       &bias_bytes) ||
      !checked_product(rows, output_width, &output_elements) ||
      weight.bytes != weight_bytes || weight.data == nullptr ||
      (bias != nullptr &&
       (bias->data == nullptr || bias->bytes != bias_bytes))) {
    return {ErrorCode::kInternal,
            "GENEB BERT linear tensor view is inconsistent"};
  }
  if (executor != nullptr) {
    const evo::detail::LinearTensorView weight_view{weight.data, weight.dtype,
                                                    weight_elements};
    evo::detail::LinearTensorView bias_view;
    const evo::detail::LinearTensorView *bias_pointer = nullptr;
    if (bias != nullptr) {
      bias_view = {bias->data, bias->dtype, output_width};
      bias_pointer = &bias_view;
    }
    status = executor->linear(input.data(), rows, input_width, weight_view,
                              output_width, bias_pointer, output);
    if (!status.ok())
      return status;
    if (output->size() != output_elements || !finite_vector(*output)) {
      return {ErrorCode::kInternal,
              "GENEB BERT linear executor returned invalid output"};
    }
    round_activations(output, activation_dtype);
    return finite_vector(*output)
               ? Status::Ok()
               : Status{ErrorCode::kInvalidArgument,
                        "GENEB BERT linear output became non-finite"};
  }

#if defined(__APPLE__)
  if (activation_dtype == TensorDType::kF32 &&
      weight.dtype == TensorDType::kF32 &&
      (bias == nullptr || bias->dtype == TensorDType::kF32) &&
      rows <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      input_width <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      output_width <=
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    output->assign(output_elements, 0.0F);
    if (bias != nullptr) {
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t target = 0; target < output_width; ++target) {
          (*output)[row * output_width + target] =
              tensor_value(*bias, target);
        }
      }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(rows), static_cast<int>(output_width),
                static_cast<int>(input_width), 1.0F, input.data(),
                static_cast<int>(input_width),
                reinterpret_cast<const float *>(weight.data),
                static_cast<int>(input_width), bias == nullptr ? 0.0F : 1.0F,
                output->data(), static_cast<int>(output_width));
    return finite_vector(*output)
               ? Status::Ok()
               : Status{ErrorCode::kInvalidArgument,
                        "GENEB BERT Accelerate linear output became "
                        "non-finite"};
  }
#endif

  output->assign(output_elements, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t target = 0; target < output_width; ++target) {
      float total = bias == nullptr ? 0.0F : tensor_value(*bias, target);
      for (std::size_t source = 0; source < input_width; ++source) {
        total += input[row * input_width + source] *
                 tensor_value(weight, target * input_width + source);
      }
      (*output)[row * output_width + target] =
          activation_value(total, activation_dtype);
    }
  }
  return finite_vector(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB BERT linear accumulation became non-finite"};
}

Status layer_norm(const std::vector<float> &input, const std::size_t rows,
                  const std::size_t width, const GenebBertTensorView &scale,
                  const GenebBertTensorView &bias, const float epsilon,
                  const TensorDType activation_dtype,
                  std::vector<float> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT LayerNorm output is null"};
  auto status =
      matrix_elements(rows, width, input.size(), "BERT LayerNorm input");
  if (!status.ok())
    return status;
  if (scale.shape != std::vector<std::size_t>{width} ||
      bias.shape != std::vector<std::size_t>{width} || scale.data == nullptr ||
      bias.data == nullptr || scale.bytes != width * dtype_size(scale.dtype) ||
      bias.bytes != width * dtype_size(bias.dtype) || !std::isfinite(epsilon) ||
      epsilon <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT LayerNorm parameters are inconsistent"};
  }
  output->resize(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    const auto moments =
        rowwise_moments(input.data() + row * width, width);
    const float inverse =
        1.0F / std::sqrt(moments.variance + epsilon);
    const float normalization_bias = -inverse * moments.mean;
    if (!std::isfinite(inverse)) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT LayerNorm denominator is invalid"};
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    if (activation_dtype == TensorDType::kF32 &&
        scale.dtype == TensorDType::kF32 && bias.dtype == TensorDType::kF32) {
      const float32x4_t inverse_vector = vdupq_n_f32(inverse);
      const float32x4_t normalization_bias_vector =
          vdupq_n_f32(normalization_bias);
      std::size_t column = 0U;
      for (; column + 4U <= width; column += 4U) {
        std::array<float, 4U> scale_values{};
        std::array<float, 4U> bias_values{};
        for (std::size_t lane = 0; lane < 4U; ++lane) {
          scale_values[lane] = tensor_value(scale, column + lane);
          bias_values[lane] = tensor_value(bias, column + lane);
        }
        const float32x4_t input_vector =
            vld1q_f32(input.data() + row * width + column);
        const float32x4_t normalized = vaddq_f32(
            vmulq_f32(input_vector, inverse_vector),
            normalization_bias_vector);
        const float32x4_t scaled =
            vmulq_f32(normalized, vld1q_f32(scale_values.data()));
        const float32x4_t result =
            vaddq_f32(scaled, vld1q_f32(bias_values.data()));
        vst1q_f32(output->data() + row * width + column, result);
      }
      for (; column < width; ++column) {
        const float normalized =
            input[row * width + column] * inverse + normalization_bias;
        (*output)[row * width + column] =
            normalized * tensor_value(scale, column) +
            tensor_value(bias, column);
      }
      continue;
    }
#endif
    for (std::size_t column = 0; column < width; ++column) {
      const float normalized = activation_value(
          input[row * width + column] * inverse + normalization_bias,
          activation_dtype);
      const float scaled = activation_value(
          normalized * tensor_value(scale, column), activation_dtype);
      (*output)[row * width + column] = activation_value(
          scaled + tensor_value(bias, column), activation_dtype);
    }
  }
  return finite_vector(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB BERT LayerNorm output became non-finite"};
}

void add_in_place(std::vector<float> *const residual,
                  const std::vector<float> &update,
                  const TensorDType activation_dtype) noexcept {
  for (std::size_t index = 0; index < residual->size(); ++index) {
    (*residual)[index] =
        activation_value((*residual)[index] + update[index], activation_dtype);
  }
}

float gelu(const float value) noexcept {
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  return value * 0.5F * (1.0F + std::erf(value * kInverseSqrtTwo));
}

#if defined(__aarch64__) || defined(_M_ARM64)
float32x4_t pytorch_neon_erf(const float32x4_t value) noexcept {
  const float32x4_t one = vdupq_n_f32(1.0F);
  const float32x4_t parameter = vdupq_n_f32(0.3275911F);
  const float32x4_t coefficient1 = vdupq_n_f32(0.254829592F);
  const float32x4_t coefficient2 = vdupq_n_f32(-0.284496736F);
  const float32x4_t coefficient3 = vdupq_n_f32(1.421413741F);
  const float32x4_t coefficient4 = vdupq_n_f32(-1.453152027F);
  const float32x4_t coefficient5 = vdupq_n_f32(1.061405429F);
  const uint32x4_t sign = vandq_u32(
      vreinterpretq_u32_f32(value), vdupq_n_u32(0x80000000U));
  const float32x4_t absolute = vabsq_f32(value);
  const float32x4_t denominator =
      vfmaq_f32(one, parameter, absolute);
  const float32x4_t t = vdivq_f32(one, denominator);
  const float32x4_t intermediate1 =
      vfmaq_f32(coefficient4, coefficient5, t);
  const float32x4_t intermediate2 =
      vfmaq_f32(coefficient3, intermediate1, t);
  const float32x4_t intermediate3 =
      vfmaq_f32(coefficient2, intermediate2, t);
  const float32x4_t polynomial =
      vfmaq_f32(coefficient1, intermediate3, t);
  const float32x4_t squared = vmulq_f32(value, value);
  std::array<float, 4U> exponent_values{};
  vst1q_f32(exponent_values.data(), vnegq_f32(squared));
  for (float &item : exponent_values)
    item = std::exp(item);
  const float32x4_t negative_exponential =
      vnegq_f32(vld1q_f32(exponent_values.data()));
  const float32x4_t product = vmulq_f32(t, negative_exponential);
  const float32x4_t magnitude = vfmaq_f32(one, product, polynomial);
  return vreinterpretq_f32_u32(
      veorq_u32(vreinterpretq_u32_f32(magnitude), sign));
}

float32x4_t pytorch_neon_gelu(const float32x4_t value) noexcept {
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  const float32x4_t half = vdupq_n_f32(0.5F);
  const float32x4_t one = vdupq_n_f32(1.0F);
  const float32x4_t erf_value =
      pytorch_neon_erf(vmulq_n_f32(value, kInverseSqrtTwo));
  return vmulq_f32(vmulq_f32(value, half), vaddq_f32(one, erf_value));
}
#endif

std::vector<float> alibi_slopes(const std::size_t heads) {
  const auto power_of_two = [](const std::size_t count) {
    std::vector<float> result;
    result.reserve(count);
    const double start =
        std::pow(2.0, -std::pow(2.0, -(std::log2(count) - 3.0)));
    for (std::size_t index = 0; index < count; ++index)
      result.push_back(static_cast<float>(start * std::pow(start, index)));
    return result;
  };
  const double logarithm = std::log2(static_cast<double>(heads));
  if (std::floor(logarithm) == logarithm)
    return power_of_two(heads);
  const auto closest =
      static_cast<std::size_t>(std::pow(2.0, std::floor(logarithm)));
  auto result = power_of_two(closest);
  const auto expanded = power_of_two(closest * 2U);
  for (std::size_t index = 0; result.size() < heads; index += 2U)
    result.push_back(expanded[index]);
  return result;
}

Status apply_rope(std::vector<float> *const query,
                  std::vector<float> *const key, const std::size_t rows,
                  const std::size_t heads, const std::size_t head_dimension,
                  const float rope_base, const TensorDType activation_dtype) {
  std::size_t expected = 0;
  if (query == nullptr || key == nullptr || head_dimension % 2U != 0 ||
      !checked_product(rows, heads * head_dimension, &expected) ||
      query->size() != expected || key->size() != expected ||
      !std::isfinite(rope_base) || rope_base <= 1.0F) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT RoPE arguments are inconsistent"};
  }
  const std::size_t pairs = head_dimension / 2U;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t base = (row * heads + head) * head_dimension;
      for (std::size_t pair = 0; pair < pairs; ++pair) {
        const float exponent = 2.0F * static_cast<float>(pair) /
                               static_cast<float>(head_dimension);
        const float angle =
            static_cast<float>(row) / std::pow(rope_base, exponent);
        const float cosine =
            activation_value(std::cos(angle), activation_dtype);
        const float sine = activation_value(std::sin(angle), activation_dtype);
        const std::size_t first = base + pair;
        const std::size_t second = base + pair + pairs;
        const auto rotate = [&](std::vector<float> *const values) {
          const float first_value = (*values)[first];
          const float second_value = (*values)[second];
          (*values)[first] = activation_value(
              activation_value(first_value * cosine, activation_dtype) -
                  activation_value(second_value * sine, activation_dtype),
              activation_dtype);
          (*values)[second] = activation_value(
              activation_value(second_value * cosine, activation_dtype) +
                  activation_value(first_value * sine, activation_dtype),
              activation_dtype);
        };
        rotate(query);
        rotate(key);
      }
    }
  }
  return Status::Ok();
}

Status pytorch_softmax_f32(
    const std::vector<float> &scores,
    const std::vector<std::uint8_t> &attention_mask,
    std::vector<float> *const probabilities) {
  if (scores.size() != attention_mask.size() || scores.empty() ||
      probabilities == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT softmax inputs are inconsistent"};
  }
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < scores.size(); ++index) {
    if (attention_mask[index] != 0U)
      maximum = std::max(maximum, scores[index]);
  }
  if (!std::isfinite(maximum)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT softmax has no finite active score"};
  }

  probabilities->resize(scores.size());
  std::array<float, kMomentLanes> lane_sums{};
  for (std::size_t index = 0; index < scores.size(); ++index) {
    const float value = attention_mask[index] == 0U
                            ? 0.0F
                            : std::exp(scores[index] - maximum);
    (*probabilities)[index] = value;
    lane_sums[index % kMomentLanes] += value;
  }
  float total = lane_sums[0];
  for (std::size_t lane = 1U; lane < kMomentLanes; ++lane)
    total += lane_sums[lane];
  if (!std::isfinite(total) || total <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT softmax denominator is invalid"};
  }
  const float inverse = 1.0F / total;
  for (float &value : *probabilities)
    value *= inverse;
  return Status::Ok();
}

Status bidirectional_attention(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value,
    const std::vector<std::uint8_t> &attention_mask, const std::size_t rows,
    const std::size_t heads, const std::size_t head_dimension,
    const bool use_alibi, const bool unpad_masked_tokens,
    const TensorDType activation_dtype, std::vector<float> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT attention output is null"};
  std::size_t width = 0;
  std::size_t expected = 0;
  if (!checked_product(heads, head_dimension, &width) ||
      !checked_product(rows, width, &expected) || query.size() != expected ||
      key.size() != expected || value.size() != expected ||
      attention_mask.size() != rows) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT attention inputs are inconsistent"};
  }
  const auto slopes = use_alibi ? alibi_slopes(heads) : std::vector<float>{};
  const float divisor = std::sqrt(static_cast<float>(head_dimension));
  output->assign(expected, 0.0F);

#if defined(__APPLE__)
  if (activation_dtype == TensorDType::kF32 &&
      rows <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      width <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      head_dimension <=
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    std::vector<float> score_matrix(rows * rows, 0.0F);
    std::vector<float> probability_matrix(rows * rows, 0.0F);
    std::vector<float> score_row(rows, 0.0F);
    std::vector<float> probability_row;
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t head_offset = head * head_dimension;
      cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasTrans,
          static_cast<int>(rows), static_cast<int>(rows),
          static_cast<int>(head_dimension), 1.0F,
          query.data() + head_offset, static_cast<int>(width),
          key.data() + head_offset, static_cast<int>(width), 0.0F,
          score_matrix.data(), static_cast<int>(rows));
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t source_row = 0; source_row < rows; ++source_row) {
          float score =
              score_matrix[row * rows + source_row] / divisor;
          if (use_alibi) {
            const auto distance =
                row > source_row ? row - source_row : source_row - row;
            score -= slopes[head] * static_cast<float>(distance);
          }
          score_row[source_row] =
              attention_mask[source_row] == 0U
                  ? -std::numeric_limits<float>::infinity()
                  : score;
        }
        auto status =
            pytorch_softmax_f32(score_row, attention_mask, &probability_row);
        if (!status.ok())
          return status;
        std::copy(probability_row.begin(), probability_row.end(),
                  probability_matrix.begin() +
                      static_cast<std::ptrdiff_t>(row * rows));
      }
      cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans,
          static_cast<int>(rows), static_cast<int>(head_dimension),
          static_cast<int>(rows), 1.0F, probability_matrix.data(),
          static_cast<int>(rows), value.data() + head_offset,
          static_cast<int>(width), 0.0F, output->data() + head_offset,
          static_cast<int>(width));
    }
    if (unpad_masked_tokens) {
      for (std::size_t row = 0; row < rows; ++row) {
        if (attention_mask[row] == 0U) {
          std::fill(output->begin() +
                        static_cast<std::ptrdiff_t>(row * width),
                    output->begin() +
                        static_cast<std::ptrdiff_t>((row + 1U) * width),
                    0.0F);
        }
      }
    }
    return finite_vector(*output)
               ? Status::Ok()
               : Status{ErrorCode::kInvalidArgument,
                        "GENEB BERT Accelerate attention output became "
                        "non-finite"};
  }
#endif

  std::vector<float> scores(rows);
  std::vector<float> probabilities(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    if (unpad_masked_tokens && attention_mask[row] == 0)
      continue;
    for (std::size_t head = 0; head < heads; ++head) {
      for (std::size_t source_row = 0; source_row < rows; ++source_row) {
        if (attention_mask[source_row] == 0) {
          scores[source_row] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float score = 0.0F;
        const std::size_t query_base = (row * heads + head) * head_dimension;
        const std::size_t key_base =
            (source_row * heads + head) * head_dimension;
        for (std::size_t column = 0; column < head_dimension; ++column)
          score += query[query_base + column] * key[key_base + column];
        score /= divisor;
        if (use_alibi) {
          const auto distance =
              row > source_row ? row - source_row : source_row - row;
          score -= slopes[head] * static_cast<float>(distance);
        }
        scores[source_row] = score;
      }
      auto status =
          pytorch_softmax_f32(scores, attention_mask, &probabilities);
      if (!status.ok())
        return status;
      const std::size_t output_base = (row * heads + head) * head_dimension;
      for (std::size_t column = 0; column < head_dimension; ++column) {
        float sum = 0.0F;
        for (std::size_t source_row = 0; source_row < rows; ++source_row) {
          const std::size_t value_index =
              (source_row * heads + head) * head_dimension + column;
          sum += probabilities[source_row] * value[value_index];
        }
        (*output)[output_base + column] =
            activation_value(sum, activation_dtype);
      }
    }
  }
  return finite_vector(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB BERT attention output became non-finite"};
}

Status normalized_mask(const std::vector<std::uint8_t> &input,
                       const std::size_t rows,
                       std::vector<std::uint8_t> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT mask output is null"};
  if (input.empty())
    output->assign(rows, 1U);
  else
    *output = input;
  if (output->size() != rows)
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT attention mask length differs from input"};
  std::size_t active = 0;
  for (const std::uint8_t value : *output) {
    if (value > 1U)
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT attention mask values must be zero or one"};
    active += value;
  }
  if (active == 0)
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT attention mask must contain an active token"};
  return Status::Ok();
}

void zero_masked_rows(std::vector<float> *const values,
                      const std::vector<std::uint8_t> &attention_mask,
                      const std::size_t width) noexcept {
  for (std::size_t row = 0; row < attention_mask.size(); ++row) {
    if (attention_mask[row] == 0) {
      std::fill(values->begin() + static_cast<std::ptrdiff_t>(row * width),
                values->begin() +
                    static_cast<std::ptrdiff_t>((row + 1U) * width),
                0.0F);
    }
  }
}

} // namespace

Status validate_geneb_bert_topology(const GenebBertTopology &topology) {
  if (topology.vocabulary_size == 0 || topology.width == 0 ||
      topology.layers == 0 || topology.attention_heads == 0 ||
      topology.head_dimension == 0 || topology.inner_width == 0 ||
      topology.maximum_sequence_length == 0 ||
      topology.token_type_vocabulary_size == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT topology dimensions must be positive"};
  }
  if (topology.layers > kMaximumTopologyLayers ||
      topology.vocabulary_size > kMaximumTopologyDimension ||
      topology.width > kMaximumTopologyDimension ||
      topology.inner_width > kMaximumTopologyDimension ||
      topology.maximum_sequence_length > kMaximumTopologyDimension ||
      topology.token_type_vocabulary_size > kMaximumTopologyDimension) {
    return {ErrorCode::kUnsupported,
            "GENEB BERT topology exceeds reference-runtime limits"};
  }
  std::size_t attention_width = 0;
  if (!checked_product(topology.attention_heads, topology.head_dimension,
                       &attention_width) ||
      attention_width != topology.width) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT attention heads times head dimension must equal width"};
  }
  if (!std::isfinite(topology.layer_norm_epsilon) ||
      topology.layer_norm_epsilon <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT LayerNorm epsilon must be finite and positive"};
  }
  if (dtype_size(topology.embedding_dtype) == 0 ||
      dtype_size(topology.projection_dtype) == 0 ||
      dtype_size(topology.norm_dtype) == 0 ||
      dtype_size(topology.activation_dtype) == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT supports only F32/BF16 declared dtypes"};
  }
  if ((topology.position_encoding != GenebBertPositionEncoding::kAbsolute &&
       topology.position_encoding != GenebBertPositionEncoding::kAlibi &&
       topology.position_encoding != GenebBertPositionEncoding::kRope) ||
      (topology.norm_placement != GenebBertNormPlacement::kPre &&
       topology.norm_placement != GenebBertNormPlacement::kPost) ||
      (topology.mlp_kind != GenebBertMlpKind::kGelu &&
       topology.mlp_kind != GenebBertMlpKind::kGatedGelu) ||
      (topology.qkv_layout != GenebBertQkvLayout::kSeparate &&
       topology.qkv_layout != GenebBertQkvLayout::kFused) ||
      (topology.input_kind != GenebBertInputKind::kTokenIds &&
       topology.input_kind != GenebBertInputKind::kSoftVocabulary) ||
      (topology.pooling != GenebBertPooling::kAttentionMaskMean &&
       topology.pooling != GenebBertPooling::kClsToken)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT topology contains an unknown operation enum"};
  }
  if (topology.position_encoding == GenebBertPositionEncoding::kRope) {
    if (topology.head_dimension % 2U != 0 ||
        !std::isfinite(topology.rope_base) || topology.rope_base <= 1.0F) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT RoPE requires even head dimension and valid base"};
    }
  } else if (topology.rope_base != 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT non-RoPE topology must set rope_base to zero"};
  }
  const bool standard_absolute =
      topology.mlp_kind == GenebBertMlpKind::kGelu &&
      topology.qkv_layout == GenebBertQkvLayout::kSeparate &&
      topology.input_kind == GenebBertInputKind::kTokenIds &&
      !topology.unpad_masked_tokens;
  const bool genomics_fm_absolute =
      topology.norm_placement == GenebBertNormPlacement::kPost &&
      topology.mlp_kind == GenebBertMlpKind::kGatedGelu &&
      topology.qkv_layout == GenebBertQkvLayout::kFused &&
      topology.input_kind == GenebBertInputKind::kTokenIds &&
      topology.pooling == GenebBertPooling::kClsToken &&
      !topology.final_layer_norm && topology.unpad_masked_tokens &&
      topology.attention_bias && !topology.mlp_input_bias &&
      topology.mlp_output_bias && topology.token_type_vocabulary_size == 2U &&
      topology.layer_norm_epsilon == 1.0e-12F &&
      topology.embedding_dtype == TensorDType::kF32 &&
      topology.projection_dtype == TensorDType::kF32 &&
      topology.norm_dtype == TensorDType::kF32 &&
      topology.activation_dtype == TensorDType::kF32;
  if (topology.position_encoding == GenebBertPositionEncoding::kAbsolute &&
      !standard_absolute && !genomics_fm_absolute) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT absolute-position contract only admits pinned "
            "BERT/GENA or exact Genomics-FM topology"};
  }
  if (topology.norm_placement == GenebBertNormPlacement::kPre) {
    if (topology.position_encoding != GenebBertPositionEncoding::kAbsolute ||
        topology.mlp_kind != GenebBertMlpKind::kGelu ||
        topology.qkv_layout != GenebBertQkvLayout::kSeparate ||
        topology.input_kind != GenebBertInputKind::kTokenIds ||
        topology.unpad_masked_tokens) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT pre-LN contract only admits pinned GENA topology"};
    }
  } else if (topology.final_layer_norm) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT final LayerNorm is only valid for pre-LN"};
  }
  if (topology.position_encoding == GenebBertPositionEncoding::kAlibi) {
    if (topology.norm_placement != GenebBertNormPlacement::kPost ||
        topology.mlp_kind != GenebBertMlpKind::kGatedGelu ||
        topology.qkv_layout != GenebBertQkvLayout::kFused ||
        topology.input_kind != GenebBertInputKind::kTokenIds ||
        !topology.unpad_masked_tokens) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT ALiBi contract only admits pinned Mosaic topology"};
    }
  }
  if (topology.position_encoding == GenebBertPositionEncoding::kRope) {
    if (topology.norm_placement != GenebBertNormPlacement::kPost ||
        topology.mlp_kind != GenebBertMlpKind::kGelu ||
        topology.qkv_layout != GenebBertQkvLayout::kSeparate ||
        topology.input_kind != GenebBertInputKind::kSoftVocabulary ||
        topology.unpad_masked_tokens) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT RoPE contract only admits pinned MutBERT topology"};
    }
  }
  if (topology.mlp_kind == GenebBertMlpKind::kGatedGelu) {
    if (topology.mlp_input_bias || !topology.mlp_output_bias) {
      return {
          ErrorCode::kInvalidArgument,
          "GENEB BERT gated GELU requires biasless input and biased output"};
    }
  }
  return Status::Ok();
}

Status geneb_bert_topology_from_artifact(const ModelFile &artifact,
                                         GenebBertTopology *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT topology output is null"};
  if (artifact.profile() != kGenebBertArtifactProfile) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT artifact profile mismatch: expected '" +
                std::string{kGenebBertArtifactProfile} + "'"};
  }
  auto status = metadata_literal(artifact, "runtime.abi", kGenebBertRuntimeAbi);
  if (!status.ok())
    return status;
  status =
      metadata_literal(artifact, "model.architecture", kGenebBertArchitecture);
  if (!status.ok())
    return status;
  if (!artifact.tokenizer_asset_descriptor().has_value()) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT artifact requires a tokenizer descriptor"};
  }

  const std::set<std::string_view> encoder_metadata{
      "encoder.vocab_size",
      "encoder.hidden_size",
      "encoder.num_layers",
      "encoder.num_attention_heads",
      "encoder.head_dim",
      "encoder.inner_mlp_size",
      "encoder.max_seqlen",
      "encoder.type_vocab_size",
      "encoder.layer_norm_epsilon",
      "encoder.position_encoding",
      "encoder.norm_placement",
      "encoder.final_layer_norm",
      "encoder.unpad_masked_tokens",
      "encoder.mlp_kind",
      "encoder.qkv_layout",
      "encoder.input_kind",
      "encoder.pooling",
      "encoder.hidden_tap",
      "encoder.mask_domain",
      "encoder.special_tokens",
      "encoder.attention_bias",
      "encoder.mlp_input_bias",
      "encoder.mlp_output_bias",
      "encoder.rope_base",
      "encoder.embedding_dtype",
      "encoder.projection_dtype",
      "encoder.norm_dtype",
      "encoder.activation_dtype",
  };
  for (const auto &entry : artifact.metadata()) {
    if (entry.key.compare(0, 8, "encoder.") == 0 &&
        encoder_metadata.find(std::string_view{entry.key}) ==
            encoder_metadata.end()) {
      return {ErrorCode::kModelFormat,
              "unexpected GENEB BERT topology metadata: " + entry.key};
    }
  }

  GenebBertTopology topology;
  status =
      metadata_size(artifact, "encoder.vocab_size", &topology.vocabulary_size);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "encoder.hidden_size", &topology.width);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "encoder.num_layers", &topology.layers);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "encoder.num_attention_heads",
                         &topology.attention_heads);
  if (!status.ok())
    return status;
  status =
      metadata_size(artifact, "encoder.head_dim", &topology.head_dimension);
  if (!status.ok())
    return status;
  status =
      metadata_size(artifact, "encoder.inner_mlp_size", &topology.inner_width);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "encoder.max_seqlen",
                         &topology.maximum_sequence_length);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "encoder.type_vocab_size",
                         &topology.token_type_vocabulary_size);
  if (!status.ok())
    return status;
  status = metadata_float(artifact, "encoder.layer_norm_epsilon",
                          &topology.layer_norm_epsilon);
  if (!status.ok())
    return status;
  status = metadata_float(artifact, "encoder.rope_base", &topology.rope_base);
  if (!status.ok())
    return status;
  status = metadata_bool(artifact, "encoder.final_layer_norm",
                         &topology.final_layer_norm);
  if (!status.ok())
    return status;
  status = metadata_bool(artifact, "encoder.unpad_masked_tokens",
                         &topology.unpad_masked_tokens);
  if (!status.ok())
    return status;
  status = metadata_bool(artifact, "encoder.attention_bias",
                         &topology.attention_bias);
  if (!status.ok())
    return status;
  status = metadata_bool(artifact, "encoder.mlp_input_bias",
                         &topology.mlp_input_bias);
  if (!status.ok())
    return status;
  status = metadata_bool(artifact, "encoder.mlp_output_bias",
                         &topology.mlp_output_bias);
  if (!status.ok())
    return status;
  status = metadata_dtype(artifact, "encoder.embedding_dtype",
                          &topology.embedding_dtype);
  if (!status.ok())
    return status;
  status = metadata_dtype(artifact, "encoder.projection_dtype",
                          &topology.projection_dtype);
  if (!status.ok())
    return status;
  status = metadata_dtype(artifact, "encoder.norm_dtype", &topology.norm_dtype);
  if (!status.ok())
    return status;
  status = metadata_dtype(artifact, "encoder.activation_dtype",
                          &topology.activation_dtype);
  if (!status.ok())
    return status;

  std::string value;
  status = metadata_string(artifact, "encoder.position_encoding", &value);
  if (!status.ok())
    return status;
  if (value == "absolute")
    topology.position_encoding = GenebBertPositionEncoding::kAbsolute;
  else if (value == "alibi-symmetric")
    topology.position_encoding = GenebBertPositionEncoding::kAlibi;
  else if (value == "rope-split-half")
    topology.position_encoding = GenebBertPositionEncoding::kRope;
  else
    return {ErrorCode::kModelFormat,
            "GENEB BERT position_encoding metadata is invalid"};
  status = metadata_string(artifact, "encoder.norm_placement", &value);
  if (!status.ok())
    return status;
  if (value == "pre")
    topology.norm_placement = GenebBertNormPlacement::kPre;
  else if (value == "post")
    topology.norm_placement = GenebBertNormPlacement::kPost;
  else
    return {ErrorCode::kModelFormat,
            "GENEB BERT norm_placement metadata is invalid"};
  status = metadata_string(artifact, "encoder.mlp_kind", &value);
  if (!status.ok())
    return status;
  if (value == "gelu")
    topology.mlp_kind = GenebBertMlpKind::kGelu;
  else if (value == "gated-gelu")
    topology.mlp_kind = GenebBertMlpKind::kGatedGelu;
  else
    return {ErrorCode::kModelFormat, "GENEB BERT mlp_kind metadata is invalid"};
  status = metadata_string(artifact, "encoder.qkv_layout", &value);
  if (!status.ok())
    return status;
  if (value == "separate")
    topology.qkv_layout = GenebBertQkvLayout::kSeparate;
  else if (value == "fused-qkv")
    topology.qkv_layout = GenebBertQkvLayout::kFused;
  else
    return {ErrorCode::kModelFormat,
            "GENEB BERT qkv_layout metadata is invalid"};
  status = metadata_string(artifact, "encoder.input_kind", &value);
  if (!status.ok())
    return status;
  if (value == "token-ids")
    topology.input_kind = GenebBertInputKind::kTokenIds;
  else if (value == "soft-vocabulary")
    topology.input_kind = GenebBertInputKind::kSoftVocabulary;
  else
    return {ErrorCode::kModelFormat,
            "GENEB BERT input_kind metadata is invalid"};
  status = metadata_string(artifact, "encoder.pooling", &value);
  if (!status.ok())
    return status;
  if (value == "attention-mask-mean")
    topology.pooling = GenebBertPooling::kAttentionMaskMean;
  else if (value == "cls-token")
    topology.pooling = GenebBertPooling::kClsToken;
  else
    return {ErrorCode::kModelFormat, "GENEB BERT pooling metadata is invalid"};

  status =
      metadata_literal(artifact, "encoder.hidden_tap", "last-hidden-state");
  if (!status.ok())
    return status;
  status = metadata_literal(artifact, "encoder.special_tokens", "include");
  if (!status.ok())
    return status;
  status = metadata_literal(artifact, "encoder.mask_domain",
                            topology.pooling == GenebBertPooling::kClsToken
                                ? "cls-row"
                                : "attention-mask");
  if (!status.ok())
    return status;

  status = validate_geneb_bert_topology(topology);
  if (!status.ok()) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT artifact topology is invalid: " + status.message()};
  }
  const auto matching_size = [&](const std::string_view key,
                                 const std::size_t expected) {
    std::size_t actual = 0;
    auto match_status = metadata_size(artifact, key, &actual);
    if (!match_status.ok())
      return match_status;
    if (actual != expected) {
      return Status{ErrorCode::kModelFormat,
                    "GENEB BERT common metadata disagrees with encoder "
                    "topology: " +
                        std::string{key}};
    }
    return Status::Ok();
  };
  status = matching_size("config.vocab_size", topology.vocabulary_size);
  if (!status.ok())
    return status;
  status = matching_size("config.hidden_size", topology.width);
  if (!status.ok())
    return status;
  status = matching_size("config.num_layers", topology.layers);
  if (!status.ok())
    return status;
  status = matching_size("config.max_seqlen", topology.maximum_sequence_length);
  if (!status.ok())
    return status;
  status = matching_size("runtime.embedding_layer_count", topology.layers + 1U);
  if (!status.ok())
    return status;
  *output = topology;
  return Status::Ok();
}

Status canonical_geneb_bert_tensors(
    const GenebBertTopology &topology,
    std::vector<GenebBertTensorRequirement> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT tensor-requirement output is null"};
  auto status = validate_geneb_bert_topology(topology);
  if (!status.ok())
    return status;
  std::vector<GenebBertTensorRequirement> result;
  add_requirement(&result, "bert.embeddings.word_embeddings.weight",
                  topology.embedding_dtype,
                  {topology.vocabulary_size, topology.width});
  if (topology.position_encoding == GenebBertPositionEncoding::kAbsolute) {
    add_requirement(&result, "bert.embeddings.position_embeddings.weight",
                    topology.embedding_dtype,
                    {topology.maximum_sequence_length, topology.width});
  }
  add_requirement(&result, "bert.embeddings.token_type_embeddings.weight",
                  topology.embedding_dtype,
                  {topology.token_type_vocabulary_size, topology.width});
  add_requirement(&result, "bert.embeddings.LayerNorm.weight",
                  topology.norm_dtype, {topology.width});
  add_requirement(&result, "bert.embeddings.LayerNorm.bias",
                  topology.norm_dtype, {topology.width});
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    const std::string prefix = layer_prefix(layer);
    if (topology.norm_placement == GenebBertNormPlacement::kPre) {
      add_requirement(&result, prefix + "pre_attention_ln.weight",
                      topology.norm_dtype, {topology.width});
      add_requirement(&result, prefix + "pre_attention_ln.bias",
                      topology.norm_dtype, {topology.width});
    }
    if (topology.qkv_layout == GenebBertQkvLayout::kSeparate) {
      for (const std::string_view projection : {"query", "key", "value"}) {
        add_requirement(
            &result,
            prefix + "attention.self." + std::string{projection} + ".weight",
            topology.projection_dtype, {topology.width, topology.width});
        if (topology.attention_bias) {
          add_requirement(&result,
                          prefix + "attention.self." + std::string{projection} +
                              ".bias",
                          topology.projection_dtype, {topology.width});
        }
      }
    } else {
      add_requirement(&result, prefix + "attention.self.Wqkv.weight",
                      topology.projection_dtype,
                      {3U * topology.width, topology.width});
      if (topology.attention_bias) {
        add_requirement(&result, prefix + "attention.self.Wqkv.bias",
                        topology.projection_dtype, {3U * topology.width});
      }
    }
    add_requirement(&result, prefix + "attention.output.dense.weight",
                    topology.projection_dtype,
                    {topology.width, topology.width});
    if (topology.attention_bias) {
      add_requirement(&result, prefix + "attention.output.dense.bias",
                      topology.projection_dtype, {topology.width});
    }
    if (topology.norm_placement == GenebBertNormPlacement::kPost) {
      add_requirement(&result, prefix + "attention.output.LayerNorm.weight",
                      topology.norm_dtype, {topology.width});
      add_requirement(&result, prefix + "attention.output.LayerNorm.bias",
                      topology.norm_dtype, {topology.width});
    }
    if (topology.mlp_kind == GenebBertMlpKind::kGelu) {
      add_requirement(&result, prefix + "intermediate.dense.weight",
                      topology.projection_dtype,
                      {topology.inner_width, topology.width});
      if (topology.mlp_input_bias) {
        add_requirement(&result, prefix + "intermediate.dense.bias",
                        topology.projection_dtype, {topology.inner_width});
      }
      add_requirement(&result, prefix + "output.dense.weight",
                      topology.projection_dtype,
                      {topology.width, topology.inner_width});
      if (topology.mlp_output_bias) {
        add_requirement(&result, prefix + "output.dense.bias",
                        topology.projection_dtype, {topology.width});
      }
    } else {
      add_requirement(&result, prefix + "mlp.gated_layers.weight",
                      topology.projection_dtype,
                      {2U * topology.inner_width, topology.width});
      if (topology.mlp_input_bias) {
        add_requirement(&result, prefix + "mlp.gated_layers.bias",
                        topology.projection_dtype, {2U * topology.inner_width});
      }
      add_requirement(&result, prefix + "mlp.wo.weight",
                      topology.projection_dtype,
                      {topology.width, topology.inner_width});
      if (topology.mlp_output_bias) {
        add_requirement(&result, prefix + "mlp.wo.bias",
                        topology.projection_dtype, {topology.width});
      }
    }
    if (topology.norm_placement == GenebBertNormPlacement::kPre) {
      add_requirement(&result, prefix + "post_attention_ln.weight",
                      topology.norm_dtype, {topology.width});
      add_requirement(&result, prefix + "post_attention_ln.bias",
                      topology.norm_dtype, {topology.width});
    } else if (topology.mlp_kind == GenebBertMlpKind::kGatedGelu) {
      add_requirement(&result, prefix + "mlp.layernorm.weight",
                      topology.norm_dtype, {topology.width});
      add_requirement(&result, prefix + "mlp.layernorm.bias",
                      topology.norm_dtype, {topology.width});
    } else {
      add_requirement(&result, prefix + "output.LayerNorm.weight",
                      topology.norm_dtype, {topology.width});
      add_requirement(&result, prefix + "output.LayerNorm.bias",
                      topology.norm_dtype, {topology.width});
    }
  }
  if (topology.final_layer_norm) {
    add_requirement(&result, "bert.encoder.last_layer_ln.weight",
                    topology.norm_dtype, {topology.width});
    add_requirement(&result, "bert.encoder.last_layer_ln.bias",
                    topology.norm_dtype, {topology.width});
  }
  *output = std::move(result);
  return Status::Ok();
}

Status geneb_bert_pool(const GenebBertForwardResult &forward,
                       const std::vector<std::uint8_t> &attention_mask,
                       const GenebBertPooling pooling,
                       std::vector<float> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT pool output is null"};
  auto status =
      matrix_elements(forward.rows, forward.width, forward.final_hidden.size(),
                      "BERT final hidden state");
  if (!status.ok())
    return status;
  std::vector<std::uint8_t> mask;
  status = normalized_mask(attention_mask, forward.rows, &mask);
  if (!status.ok())
    return status;
  output->assign(forward.width, 0.0F);
  if (pooling == GenebBertPooling::kClsToken) {
    if (mask.front() == 0)
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT CLS pooling requires active row zero"};
    std::copy_n(forward.final_hidden.begin(), forward.width, output->begin());
    return Status::Ok();
  }
  if (pooling != GenebBertPooling::kAttentionMaskMean) {
    return {ErrorCode::kInvalidArgument, "GENEB BERT pooling mode is invalid"};
  }
  std::size_t count = 0;
  for (std::size_t row = 0; row < forward.rows; ++row) {
    if (mask[row] == 0)
      continue;
    ++count;
    for (std::size_t column = 0; column < forward.width; ++column)
      (*output)[column] += forward.final_hidden[row * forward.width + column];
  }
  for (float &value : *output)
    value /= static_cast<float>(count);
  return finite_vector(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB BERT pooled output became non-finite"};
}

struct GenebBertModel::Impl final {
  struct Layer final {
    GenebBertTensorView first_norm_weight;
    GenebBertTensorView first_norm_bias;
    GenebBertTensorView query_weight;
    GenebBertTensorView query_bias;
    GenebBertTensorView key_weight;
    GenebBertTensorView key_bias;
    GenebBertTensorView value_weight;
    GenebBertTensorView value_bias;
    GenebBertTensorView fused_qkv_weight;
    GenebBertTensorView fused_qkv_bias;
    GenebBertTensorView attention_output_weight;
    GenebBertTensorView attention_output_bias;
    GenebBertTensorView mlp_input_weight;
    GenebBertTensorView mlp_input_bias;
    GenebBertTensorView mlp_output_weight;
    GenebBertTensorView mlp_output_bias;
    GenebBertTensorView second_norm_weight;
    GenebBertTensorView second_norm_bias;
  };

  GenebBertTopology topology;
  GenebBertTensorView word_embedding;
  GenebBertTensorView position_embedding;
  GenebBertTensorView token_type_embedding;
  GenebBertTensorView embedding_norm_weight;
  GenebBertTensorView embedding_norm_bias;
  std::vector<Layer> layers;
  GenebBertTensorView final_norm_weight;
  GenebBertTensorView final_norm_bias;
  std::shared_ptr<evo::detail::LinearExecutor> linear_executor;

  Status run(const std::vector<TokenId> *tokens,
             const std::vector<float> *soft_vocabulary, std::size_t rows,
             const std::vector<std::uint8_t> &input_attention_mask,
             const std::vector<std::size_t> &capture_layers,
             GenebBertForwardResult *output) const;
};

GenebBertModel::GenebBertModel() = default;
GenebBertModel::~GenebBertModel() = default;
GenebBertModel::GenebBertModel(GenebBertModel &&) noexcept = default;
GenebBertModel &GenebBertModel::operator=(GenebBertModel &&) noexcept = default;

Status GenebBertModel::load(
    const GenebBertTopology &topology,
    const std::vector<GenebBertNamedTensorView> &tensors,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  std::vector<GenebBertTensorRequirement> requirements;
  auto status = canonical_geneb_bert_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  std::map<std::string, const GenebBertTensorView *, std::less<>> provided;
  for (const auto &named : tensors) {
    if (!provided.emplace(named.name, &named.tensor).second) {
      return {ErrorCode::kModelFormat,
              "GENEB BERT tensor is duplicated: " + named.name};
    }
  }
  if (provided.size() != requirements.size()) {
    return {ErrorCode::kModelFormat,
            "GENEB BERT tensor set has missing or extra entries"};
  }
  for (const auto &requirement : requirements) {
    const auto found = provided.find(requirement.name);
    if (found == provided.end()) {
      return {ErrorCode::kModelFormat,
              "GENEB BERT tensor is missing: " + requirement.name};
    }
    status = tensor_requirement(*found->second, requirement);
    if (!status.ok())
      return status;
  }
  const auto view = [&](const std::string &name) {
    return *provided.find(name)->second;
  };
  auto implementation = std::make_unique<Impl>();
  implementation->topology = topology;
  implementation->linear_executor = std::move(linear_executor);
  implementation->word_embedding =
      view("bert.embeddings.word_embeddings.weight");
  if (topology.position_encoding == GenebBertPositionEncoding::kAbsolute) {
    implementation->position_embedding =
        view("bert.embeddings.position_embeddings.weight");
  }
  implementation->token_type_embedding =
      view("bert.embeddings.token_type_embeddings.weight");
  implementation->embedding_norm_weight =
      view("bert.embeddings.LayerNorm.weight");
  implementation->embedding_norm_bias = view("bert.embeddings.LayerNorm.bias");
  implementation->layers.resize(topology.layers);
  for (std::size_t layer_index = 0; layer_index < topology.layers;
       ++layer_index) {
    auto &layer = implementation->layers[layer_index];
    const std::string prefix = layer_prefix(layer_index);
    if (topology.norm_placement == GenebBertNormPlacement::kPre) {
      layer.first_norm_weight = view(prefix + "pre_attention_ln.weight");
      layer.first_norm_bias = view(prefix + "pre_attention_ln.bias");
    } else {
      layer.first_norm_weight =
          view(prefix + "attention.output.LayerNorm.weight");
      layer.first_norm_bias = view(prefix + "attention.output.LayerNorm.bias");
    }
    if (topology.qkv_layout == GenebBertQkvLayout::kSeparate) {
      layer.query_weight = view(prefix + "attention.self.query.weight");
      layer.key_weight = view(prefix + "attention.self.key.weight");
      layer.value_weight = view(prefix + "attention.self.value.weight");
      if (topology.attention_bias) {
        layer.query_bias = view(prefix + "attention.self.query.bias");
        layer.key_bias = view(prefix + "attention.self.key.bias");
        layer.value_bias = view(prefix + "attention.self.value.bias");
      }
    } else {
      layer.fused_qkv_weight = view(prefix + "attention.self.Wqkv.weight");
      if (topology.attention_bias)
        layer.fused_qkv_bias = view(prefix + "attention.self.Wqkv.bias");
    }
    layer.attention_output_weight =
        view(prefix + "attention.output.dense.weight");
    if (topology.attention_bias) {
      layer.attention_output_bias =
          view(prefix + "attention.output.dense.bias");
    }
    if (topology.mlp_kind == GenebBertMlpKind::kGelu) {
      layer.mlp_input_weight = view(prefix + "intermediate.dense.weight");
      if (topology.mlp_input_bias)
        layer.mlp_input_bias = view(prefix + "intermediate.dense.bias");
      layer.mlp_output_weight = view(prefix + "output.dense.weight");
      if (topology.mlp_output_bias)
        layer.mlp_output_bias = view(prefix + "output.dense.bias");
    } else {
      layer.mlp_input_weight = view(prefix + "mlp.gated_layers.weight");
      if (topology.mlp_input_bias)
        layer.mlp_input_bias = view(prefix + "mlp.gated_layers.bias");
      layer.mlp_output_weight = view(prefix + "mlp.wo.weight");
      if (topology.mlp_output_bias)
        layer.mlp_output_bias = view(prefix + "mlp.wo.bias");
    }
    if (topology.norm_placement == GenebBertNormPlacement::kPre) {
      layer.second_norm_weight = view(prefix + "post_attention_ln.weight");
      layer.second_norm_bias = view(prefix + "post_attention_ln.bias");
    } else if (topology.mlp_kind == GenebBertMlpKind::kGatedGelu) {
      layer.second_norm_weight = view(prefix + "mlp.layernorm.weight");
      layer.second_norm_bias = view(prefix + "mlp.layernorm.bias");
    } else {
      layer.second_norm_weight = view(prefix + "output.LayerNorm.weight");
      layer.second_norm_bias = view(prefix + "output.LayerNorm.bias");
    }
  }
  if (topology.final_layer_norm) {
    implementation->final_norm_weight =
        view("bert.encoder.last_layer_ln.weight");
    implementation->final_norm_bias = view("bert.encoder.last_layer_ln.bias");
  }
  impl_ = std::move(implementation);
  return Status::Ok();
}

Status GenebBertModel::load(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  GenebBertTopology topology;
  auto status = geneb_bert_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebBertNamedTensorView> views;
  views.reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t index = 0; index < tensor.rank; ++index) {
      if (tensor.dimensions[index] > std::numeric_limits<std::size_t>::max()) {
        return {ErrorCode::kModelFormat,
                "GENEB BERT tensor dimension exceeds size_t: " + tensor.name};
      }
      shape.push_back(static_cast<std::size_t>(tensor.dimensions[index]));
    }
    if (tensor.data_size > std::numeric_limits<std::size_t>::max()) {
      return {ErrorCode::kModelFormat,
              "GENEB BERT tensor byte extent exceeds size_t: " + tensor.name};
    }
    const auto *const data = artifact.tensor_data(tensor);
    if (data == nullptr) {
      return {ErrorCode::kModelFormat,
              "GENEB BERT tensor payload is unavailable: " + tensor.name};
    }
    views.push_back({tensor.name,
                     {data, static_cast<std::size_t>(tensor.data_size),
                      tensor.dtype, std::move(shape)}});
  }
  return load(topology, views, std::move(linear_executor));
}

Status GenebBertModel::load_artifact(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  return load(artifact, std::move(linear_executor));
}

const GenebBertTopology *GenebBertModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebBertModel::linear_executor_name() const noexcept {
  if (impl_ == nullptr || impl_->linear_executor == nullptr)
    return "scalar-reference";
  return impl_->linear_executor->name();
}

Status
GenebBertModel::Impl::run(const std::vector<TokenId> *const tokens,
                          const std::vector<float> *const soft_vocabulary,
                          const std::size_t rows,
                          const std::vector<std::uint8_t> &input_attention_mask,
                          const std::vector<std::size_t> &capture_layers,
                          GenebBertForwardResult *const output) const {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT forward output is null"};
  if (rows == 0 || rows > topology.maximum_sequence_length) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT input length is zero or exceeds maximum sequence"};
  }
  if ((tokens == nullptr) == (soft_vocabulary == nullptr)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT forward requires exactly one input representation"};
  }
  if ((tokens != nullptr) !=
      (topology.input_kind == GenebBertInputKind::kTokenIds)) {
    return {ErrorCode::kUnsupported,
            "GENEB BERT input representation disagrees with artifact"};
  }
  std::vector<std::uint8_t> attention_mask;
  auto status = normalized_mask(input_attention_mask, rows, &attention_mask);
  if (!status.ok())
    return status;
  if (topology.unpad_masked_tokens) {
    bool saw_padding = false;
    for (const std::uint8_t value : attention_mask) {
      if (value == 0)
        saw_padding = true;
      else if (saw_padding) {
        return {ErrorCode::kInvalidArgument,
                "GENEB Mosaic unpadding requires a right-padded prefix mask"};
      }
    }
  }
  std::set<std::size_t> capture_set;
  for (const std::size_t layer : capture_layers) {
    if (layer > topology.layers || !capture_set.insert(layer).second) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT capture layers are duplicate or out of range"};
    }
  }
  GenebBertForwardResult result;
  result.rows = rows;
  result.width = topology.width;
  result.captures.reserve(capture_layers.size());
  for (const std::size_t layer : capture_layers)
    result.captures.push_back({layer, {}});
  const auto capture = [&](const std::size_t layer,
                           const std::vector<float> &values) {
    for (auto &item : result.captures) {
      if (item.layer == layer)
        item.values = values;
    }
  };

  std::size_t hidden_elements = 0;
  if (!checked_product(rows, topology.width, &hidden_elements)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB BERT hidden dimensions overflow"};
  }
  std::vector<float> embedding(hidden_elements, 0.0F);
  if (tokens != nullptr) {
    if (tokens->size() != rows)
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT token length differs from declared rows"};
    for (std::size_t row = 0; row < rows; ++row) {
      const std::size_t token = (*tokens)[row];
      if (token >= topology.vocabulary_size) {
        return {ErrorCode::kInvalidArgument,
                "GENEB BERT token ID is outside vocabulary"};
      }
      for (std::size_t column = 0; column < topology.width; ++column) {
        embedding[row * topology.width + column] =
            tensor_value(word_embedding, token * topology.width + column);
      }
    }
  } else {
    std::size_t expected_soft = 0;
    if (!checked_product(rows, topology.vocabulary_size, &expected_soft) ||
        soft_vocabulary->size() != expected_soft ||
        !finite_vector(*soft_vocabulary)) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT soft-vocabulary input is inconsistent"};
    }
    for (std::size_t row = 0; row < rows; ++row) {
      if (attention_mask[row] == 0)
        continue;
      for (std::size_t column = 0; column < topology.width; ++column) {
        float sum = 0.0F;
        for (std::size_t token = 0; token < topology.vocabulary_size; ++token) {
          sum += (*soft_vocabulary)[row * topology.vocabulary_size + token] *
                 tensor_value(word_embedding, token * topology.width + column);
        }
        embedding[row * topology.width + column] =
            activation_value(sum, topology.activation_dtype);
      }
    }
  }
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < topology.width; ++column) {
      float value = embedding[row * topology.width + column] +
                    tensor_value(token_type_embedding, column);
      if (topology.position_encoding == GenebBertPositionEncoding::kAbsolute) {
        value +=
            tensor_value(position_embedding, row * topology.width + column);
      }
      embedding[row * topology.width + column] =
          activation_value(value, topology.activation_dtype);
    }
  }
  std::vector<float> hidden;
  status = layer_norm(embedding, rows, topology.width, embedding_norm_weight,
                      embedding_norm_bias, topology.layer_norm_epsilon,
                      topology.activation_dtype, &hidden);
  if (!status.ok())
    return status;
  capture(0, hidden);

  for (std::size_t layer_index = 0; layer_index < topology.layers;
       ++layer_index) {
    const auto &layer = layers[layer_index];
    std::vector<float> attention_input;
    if (topology.norm_placement == GenebBertNormPlacement::kPre) {
      status = layer_norm(hidden, rows, topology.width, layer.first_norm_weight,
                          layer.first_norm_bias, topology.layer_norm_epsilon,
                          topology.activation_dtype, &attention_input);
      if (!status.ok())
        return status;
    } else {
      attention_input = hidden;
    }

    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    if (topology.qkv_layout == GenebBertQkvLayout::kSeparate) {
      const GenebBertTensorView *const query_bias =
          topology.attention_bias ? &layer.query_bias : nullptr;
      const GenebBertTensorView *const key_bias =
          topology.attention_bias ? &layer.key_bias : nullptr;
      const GenebBertTensorView *const value_bias =
          topology.attention_bias ? &layer.value_bias : nullptr;
      status = runtime_linear(attention_input, rows, topology.width,
                              layer.query_weight, topology.width, query_bias,
                              linear_executor.get(), topology.activation_dtype,
                              &query);
      if (!status.ok())
        return status;
      status = runtime_linear(attention_input, rows, topology.width,
                              layer.key_weight, topology.width, key_bias,
                              linear_executor.get(), topology.activation_dtype,
                              &key);
      if (!status.ok())
        return status;
      status = runtime_linear(attention_input, rows, topology.width,
                              layer.value_weight, topology.width, value_bias,
                              linear_executor.get(), topology.activation_dtype,
                              &value);
      if (!status.ok())
        return status;
    } else {
      std::vector<float> fused;
      const GenebBertTensorView *const fused_bias =
          topology.attention_bias ? &layer.fused_qkv_bias : nullptr;
      status = runtime_linear(attention_input, rows, topology.width,
                              layer.fused_qkv_weight, 3U * topology.width,
                              fused_bias, linear_executor.get(),
                              topology.activation_dtype, &fused);
      if (!status.ok())
        return status;
      query.resize(hidden_elements);
      key.resize(hidden_elements);
      value.resize(hidden_elements);
      for (std::size_t row = 0; row < rows; ++row) {
        const auto source = fused.begin() + static_cast<std::ptrdiff_t>(
                                                row * 3U * topology.width);
        std::copy_n(source, topology.width,
                    query.begin() +
                        static_cast<std::ptrdiff_t>(row * topology.width));
        std::copy_n(source + static_cast<std::ptrdiff_t>(topology.width),
                    topology.width,
                    key.begin() +
                        static_cast<std::ptrdiff_t>(row * topology.width));
        std::copy_n(source + static_cast<std::ptrdiff_t>(2U * topology.width),
                    topology.width,
                    value.begin() +
                        static_cast<std::ptrdiff_t>(row * topology.width));
      }
    }
    if (topology.position_encoding == GenebBertPositionEncoding::kRope) {
      status = apply_rope(&query, &key, rows, topology.attention_heads,
                          topology.head_dimension, topology.rope_base,
                          topology.activation_dtype);
      if (!status.ok())
        return status;
    }
    std::vector<float> attended;
    status = bidirectional_attention(
        query, key, value, attention_mask, rows, topology.attention_heads,
        topology.head_dimension,
        topology.position_encoding == GenebBertPositionEncoding::kAlibi,
        topology.unpad_masked_tokens, topology.activation_dtype, &attended);
    if (!status.ok())
      return status;
    std::vector<float> attention_output;
    const GenebBertTensorView *const attention_output_bias =
        topology.attention_bias ? &layer.attention_output_bias : nullptr;
    status = runtime_linear(attended, rows, topology.width,
                            layer.attention_output_weight, topology.width,
                            attention_output_bias, linear_executor.get(),
                            topology.activation_dtype, &attention_output);
    if (!status.ok())
      return status;
    add_in_place(&attention_output, hidden, topology.activation_dtype);
    if (topology.unpad_masked_tokens)
      zero_masked_rows(&attention_output, attention_mask, topology.width);
    if (topology.norm_placement == GenebBertNormPlacement::kPost) {
      std::vector<float> normalized;
      status = layer_norm(attention_output, rows, topology.width,
                          layer.first_norm_weight, layer.first_norm_bias,
                          topology.layer_norm_epsilon,
                          topology.activation_dtype, &normalized);
      if (!status.ok())
        return status;
      attention_output = std::move(normalized);
      if (topology.unpad_masked_tokens)
        zero_masked_rows(&attention_output, attention_mask, topology.width);
    }

    std::vector<float> mlp_input;
    if (topology.norm_placement == GenebBertNormPlacement::kPre) {
      status = layer_norm(attention_output, rows, topology.width,
                          layer.second_norm_weight, layer.second_norm_bias,
                          topology.layer_norm_epsilon,
                          topology.activation_dtype, &mlp_input);
      if (!status.ok())
        return status;
    } else {
      mlp_input = attention_output;
    }
    std::vector<float> expanded;
    const std::size_t expanded_width =
        topology.mlp_kind == GenebBertMlpKind::kGatedGelu
            ? 2U * topology.inner_width
            : topology.inner_width;
    const GenebBertTensorView *const mlp_input_bias =
        topology.mlp_input_bias ? &layer.mlp_input_bias : nullptr;
    status =
        runtime_linear(mlp_input, rows, topology.width, layer.mlp_input_weight,
                       expanded_width, mlp_input_bias, linear_executor.get(),
                       topology.activation_dtype, &expanded);
    if (!status.ok())
      return status;
    std::vector<float> activated(rows * topology.inner_width, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
      std::size_t column = 0U;
#if defined(__aarch64__) || defined(_M_ARM64)
      if (topology.activation_dtype == TensorDType::kF32) {
        for (; column + 4U <= topology.inner_width; column += 4U) {
          const std::size_t source = row * expanded_width + column;
          float32x4_t value_out =
              pytorch_neon_gelu(vld1q_f32(expanded.data() + source));
          if (topology.mlp_kind == GenebBertMlpKind::kGatedGelu) {
            value_out = vmulq_f32(
                value_out,
                vld1q_f32(expanded.data() + source + topology.inner_width));
          }
          vst1q_f32(activated.data() + row * topology.inner_width + column,
                    value_out);
        }
      }
#endif
      for (; column < topology.inner_width; ++column) {
        float value_out = gelu(expanded[row * expanded_width + column]);
        if (topology.mlp_kind == GenebBertMlpKind::kGatedGelu) {
          value_out *=
              expanded[row * expanded_width + topology.inner_width + column];
        }
        activated[row * topology.inner_width + column] =
            activation_value(value_out, topology.activation_dtype);
      }
    }
    std::vector<float> mlp_output;
    const GenebBertTensorView *const mlp_output_bias =
        topology.mlp_output_bias ? &layer.mlp_output_bias : nullptr;
    status = runtime_linear(activated, rows, topology.inner_width,
                            layer.mlp_output_weight, topology.width,
                            mlp_output_bias, linear_executor.get(),
                            topology.activation_dtype, &mlp_output);
    if (!status.ok())
      return status;
    add_in_place(&mlp_output, attention_output, topology.activation_dtype);
    if (topology.unpad_masked_tokens)
      zero_masked_rows(&mlp_output, attention_mask, topology.width);
    if (topology.norm_placement == GenebBertNormPlacement::kPost) {
      std::vector<float> normalized;
      status =
          layer_norm(mlp_output, rows, topology.width, layer.second_norm_weight,
                     layer.second_norm_bias, topology.layer_norm_epsilon,
                     topology.activation_dtype, &normalized);
      if (!status.ok())
        return status;
      hidden = std::move(normalized);
      if (topology.unpad_masked_tokens)
        zero_masked_rows(&hidden, attention_mask, topology.width);
    } else {
      hidden = std::move(mlp_output);
    }
    const std::size_t public_layer = layer_index + 1U;
    if (!(topology.final_layer_norm && public_layer == topology.layers))
      capture(public_layer, hidden);
  }
  if (topology.final_layer_norm) {
    std::vector<float> normalized;
    status = layer_norm(hidden, rows, topology.width, final_norm_weight,
                        final_norm_bias, topology.layer_norm_epsilon,
                        topology.activation_dtype, &normalized);
    if (!status.ok())
      return status;
    hidden = std::move(normalized);
    capture(topology.layers, hidden);
  }
  result.final_hidden = std::move(hidden);
  *output = std::move(result);
  return Status::Ok();
}

Status GenebBertModel::forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebBertForwardResult *const output) const {
  if (impl_ == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT model is not loaded"};
  return impl_->run(&tokens, nullptr, tokens.size(), attention_mask,
                    capture_layers, output);
}

Status
GenebBertModel::forward_soft(const std::vector<float> &soft_vocabulary,
                             const std::size_t rows,
                             const std::vector<std::uint8_t> &attention_mask,
                             const std::vector<std::size_t> &capture_layers,
                             GenebBertForwardResult *const output) const {
  if (impl_ == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT model is not loaded"};
  return impl_->run(nullptr, &soft_vocabulary, rows, attention_mask,
                    capture_layers, output);
}

Status GenebBertModel::pool(const GenebBertForwardResult &forward,
                            const std::vector<std::uint8_t> &attention_mask,
                            std::vector<float> *const output) const {
  if (impl_ == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB BERT model is not loaded"};
  return geneb_bert_pool(forward, attention_mask, impl_->topology.pooling,
                         output);
}

} // namespace evo::cpu
