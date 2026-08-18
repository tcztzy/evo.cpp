// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_olmo.hpp"

#include "../linear_executor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__) &&                                                      \
    (defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64))
#include <arm_neon.h>
#endif

namespace evo::cpu {
namespace {

constexpr std::size_t kMaximumWidth = 65536;
constexpr std::size_t kMaximumLayers = 1024;
constexpr std::size_t kMaximumSequenceLength = 1U << 20U;
constexpr std::size_t kMaximumFusedMlpWidth = 1U << 20U;

Status invalid(const std::string &message) {
  return {ErrorCode::kInvalidArgument, "GENEB OLMo: " + message};
}

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "GENEB OLMo artifact: " + message};
}

bool checked_mul(const std::size_t left, const std::size_t right,
                 std::size_t *const output) noexcept {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *output = left * right;
  return true;
}

bool metadata_u64(const ModelFile &artifact, const std::string_view key,
                  std::uint64_t *const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kU64 ||
      entry->value.size() != sizeof(std::uint64_t))
    return false;
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<std::uint64_t>(entry->value[byte]) << (byte * 8U);
  *output = value;
  return true;
}

bool metadata_f64(const ModelFile &artifact, const std::string_view key,
                  double *const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kF64 ||
      entry->value.size() != sizeof(std::uint64_t))
    return false;
  std::uint64_t bits = 0;
  for (std::size_t byte = 0; byte < sizeof(bits); ++byte)
    bits |= static_cast<std::uint64_t>(entry->value[byte]) << (byte * 8U);
  std::memcpy(output, &bits, sizeof(bits));
  return true;
}

bool metadata_bool(const ModelFile &artifact, const std::string_view key,
                   bool *const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kBool ||
      entry->value.size() != 1U || entry->value[0] > 1U)
    return false;
  *output = entry->value[0] != 0U;
  return true;
}

bool metadata_string(const ModelFile &artifact, const std::string_view key,
                     std::string *const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kString)
    return false;
  if (entry->value.empty())
    output->clear();
  else
    output->assign(reinterpret_cast<const char *>(entry->value.data()),
                   entry->value.size());
  return true;
}

Status require_literal(const ModelFile &artifact, const std::string_view key,
                       const std::string_view expected) {
  std::string value;
  if (!metadata_string(artifact, key, &value) || value != expected)
    return format_error("metadata '" + std::string{key} + "' must be '" +
                        std::string{expected} + "'");
  return Status::Ok();
}

Status require_bool(const ModelFile &artifact, const std::string_view key,
                    const bool expected) {
  bool value = false;
  if (!metadata_bool(artifact, key, &value) || value != expected)
    return format_error("metadata '" + std::string{key} + "' must be " +
                        (expected ? "true" : "false"));
  return Status::Ok();
}

float tensor_value(const GenebOlmoTensorView &tensor,
                   const std::size_t index) noexcept {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return value;
}

Status validate_tensor(const GenebOlmoTensorView &tensor,
                       const GenebOlmoTensorRequirement &requirement) {
  if (tensor.data == nullptr)
    return invalid("tensor '" + requirement.name + "' has null data");
  if (tensor.dtype != requirement.dtype)
    return invalid("tensor '" + requirement.name + "' has wrong dtype");
  if (tensor.shape != requirement.shape)
    return invalid("tensor '" + requirement.name + "' has wrong shape");
  std::size_t elements = 1;
  for (const auto dimension : tensor.shape) {
    if (dimension == 0 || !checked_mul(elements, dimension, &elements))
      return invalid("tensor '" + requirement.name +
                     "' has an overflowing shape");
  }
  std::size_t expected_bytes = 0;
  if (!checked_mul(elements, sizeof(float), &expected_bytes) ||
      tensor.bytes != expected_bytes)
    return invalid("tensor '" + requirement.name + "' has wrong byte size");
  return Status::Ok();
}

class ReferenceLinearExecutor final : public evo::detail::LinearExecutor {
public:
  [[nodiscard]] const char *name() const noexcept override {
    return "geneb-olmo-reference-f32";
  }

  [[nodiscard]] Status linear(const float *const input, const std::size_t rows,
                              const std::size_t input_width,
                              const evo::detail::LinearTensorView weight,
                              const std::size_t output_width,
                              const evo::detail::LinearTensorView *const bias,
                              std::vector<float> *const output) override {
    if (input == nullptr || output == nullptr || weight.data == nullptr ||
        rows == 0 || input_width == 0 || output_width == 0 || bias != nullptr)
      return invalid("reference linear received an invalid argument");
    std::size_t expected = 0;
    if (weight.dtype != TensorDType::kF32 ||
        !checked_mul(input_width, output_width, &expected) ||
        weight.elements != expected)
      return invalid("reference linear weight contract differs");
    std::size_t output_elements = 0;
    if (!checked_mul(rows, output_width, &output_elements))
      return invalid("reference linear output shape overflows");
    output->assign(output_elements, 0.0F);

    const bool aligned =
        reinterpret_cast<std::uintptr_t>(weight.data) % alignof(float) == 0U;
    const auto *const aligned_weight =
        aligned ? reinterpret_cast<const float *>(weight.data) : nullptr;
    for (std::size_t row = 0; row < rows; ++row) {
      const auto *const row_input = input + row * input_width;
      for (std::size_t target = 0; target < output_width; ++target) {
        float total = 0.0F;
        const std::size_t base = target * input_width;
        if (aligned_weight != nullptr) {
          const auto *const row_weight = aligned_weight + base;
          for (std::size_t source = 0; source < input_width; ++source)
            total += row_input[source] * row_weight[source];
        } else {
          for (std::size_t source = 0; source < input_width; ++source) {
            float value = 0.0F;
            std::memcpy(&value, weight.data + (base + source) * sizeof(float),
                        sizeof(value));
            total += row_input[source] * value;
          }
        }
        (*output)[row * output_width + target] = total;
      }
    }
    return Status::Ok();
  }
};

Status linear(const std::vector<float> &input, const std::size_t rows,
              const std::size_t input_width, const GenebOlmoTensorView &weight,
              const std::size_t output_width,
              evo::detail::LinearExecutor *const executor,
              std::vector<float> *const output) {
  std::size_t input_elements = 0;
  std::size_t weight_elements = 0;
  if (executor == nullptr || output == nullptr ||
      !checked_mul(rows, input_width, &input_elements) ||
      input.size() != input_elements ||
      !checked_mul(input_width, output_width, &weight_elements))
    return invalid("linear shape contract differs");
  std::size_t output_elements = 0;
  if (!checked_mul(rows, output_width, &output_elements))
    return invalid("linear output shape overflows");
  auto status = executor->linear(input.data(), rows, input_width,
                                 {weight.data, weight.dtype, weight_elements},
                                 output_width, nullptr, output);
  if (!status.ok())
    return status;
  if (output->size() != output_elements ||
      !std::all_of(output->begin(), output->end(),
                   [](const float value) { return std::isfinite(value); }))
    return invalid("linear executor returned a wrong-sized/non-finite output");
  return Status::Ok();
}

bool torch212_apple_arm64_layer_norm_supported() noexcept {
#if defined(__APPLE__) &&                                                      \
    (defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64))
  return true;
#else
  return false;
#endif
}

#if defined(__APPLE__) &&                                                      \
    (defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64))
struct LayerNormMoments final {
  float mean{0.0F};
  float variance{0.0F};
};

// These lane/chunk counts and the cascade order reproduce the pinned Torch
// 2.1.2 Apple-arm64 F32 LayerNorm reduction selected only by Omni-DNA-1B.
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
  const float ratio = combined == 0U ? 0.0F
                                     : static_cast<float>(added_count) /
                                           static_cast<float>(combined);
  const float delta = added_mean - *mean;
  *mean += ratio * delta;
  *m2 += added_m2 + delta * delta * ratio * static_cast<float>(*count);
  *count = combined;
}

void add_moment_lanes(const std::size_t added_count,
                      const MomentLanes &added_mean,
                      const MomentLanes &added_m2, std::size_t *const count,
                      MomentLanes *const mean, MomentLanes *const m2) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  const std::size_t combined = *count + added_count;
  const float ratio = combined == 0U ? 0.0F
                                     : static_cast<float>(added_count) /
                                           static_cast<float>(combined);
#if defined(__APPLE__) &&                                                      \
    (defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64))
  const float32x4_t ratio_vector = vdupq_n_f32(ratio);
  const float32x4_t count_vector = vdupq_n_f32(static_cast<float>(*count));
  for (std::size_t half = 0; half < 2U; ++half) {
    const std::size_t offset = half * 4U;
    const float32x4_t added_mean_vector = vld1q_f32(added_mean.data() + offset);
    const float32x4_t mean_vector = vld1q_f32(mean->data() + offset);
    const float32x4_t delta = vsubq_f32(added_mean_vector, mean_vector);
    const float32x4_t next_mean =
        vaddq_f32(mean_vector, vmulq_f32(ratio_vector, delta));
    const float32x4_t delta_squared = vmulq_f32(delta, delta);
    const float32x4_t weighted_delta =
        vmulq_f32(vmulq_f32(delta_squared, ratio_vector), count_vector);
    const float32x4_t added_m2_vector = vld1q_f32(added_m2.data() + offset);
    const float32x4_t m2_vector = vld1q_f32(m2->data() + offset);
    const float32x4_t next_m2 =
        vaddq_f32(m2_vector, vaddq_f32(added_m2_vector, weighted_delta));
    vst1q_f32(mean->data() + offset, next_mean);
    vst1q_f32(m2->data() + offset, next_m2);
  }
#else
  for (std::size_t lane = 0; lane < kMomentLanes; ++lane) {
    const float delta = added_mean[lane] - (*mean)[lane];
    (*mean)[lane] += ratio * delta;
    (*m2)[lane] +=
        added_m2[lane] + delta * delta * ratio * static_cast<float>(*count);
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
#if defined(__APPLE__) &&                                                      \
    (defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64))
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
  const std::size_t chunks = (vectors + kMomentChunk - 1U) / kMomentChunk;
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
      const std::size_t base = (chunk * kMomentChunk + item) * kMomentLanes;
      update_moment_lanes(input + base, ratio, &chunk_mean, &chunk_m2);
    }
    add_moment_lanes(chunk_vectors, chunk_mean, chunk_m2, &counts[0], &means[0],
                     &m2s[0]);

    std::size_t mask = chunk + 1U;
    for (std::size_t level = 1U; level < depth && (mask & 1U) == 0U; ++level) {
      add_moment_lanes(counts[level - 1U], means[level - 1U], m2s[level - 1U],
                       &counts[level], &means[level], &m2s[level]);
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
  const std::size_t lane_count = vectors;
  for (std::size_t lane = 0; lane < kMomentLanes; ++lane) {
    add_moments(lane_count, means[0][lane], m2s[0][lane], &count, &mean, &m2);
  }
  return {mean, m2 / static_cast<float>(width)};
}
#endif


Status layer_norm_no_affine_impl(const std::vector<float> &input,
                                 const std::size_t rows,
                                 const std::size_t width, const float epsilon,
                                 const GenebOlmoLayerNormKernel kernel,
                                 std::vector<float> *const output) {
  std::size_t elements = 0;
  if (output == nullptr || rows == 0U || width == 0U ||
      !checked_mul(rows, width, &elements) || input.size() != elements ||
      !std::isfinite(epsilon) || epsilon <= 0.0F)
    return invalid("LayerNorm shape/epsilon contract differs");
  if (kernel != GenebOlmoLayerNormKernel::kPortableTwoPass &&
      kernel != GenebOlmoLayerNormKernel::kTorch212AppleArm64ExactV1)
    return invalid("LayerNorm kernel is unsupported");
  if (kernel == GenebOlmoLayerNormKernel::kTorch212AppleArm64ExactV1 &&
      (width != 2048U || epsilon != 1.0e-5F))
    return invalid(
        "exact LayerNorm kernel requires width=2048 and epsilon=1e-5");
  if (kernel == GenebOlmoLayerNormKernel::kTorch212AppleArm64ExactV1 &&
      !torch212_apple_arm64_layer_norm_supported())
    return {ErrorCode::kUnsupported,
            "olmo.layer_norm_kernel=torch-2.1.2-apple-arm64-exact-v1 "
            "requires Apple arm64"};

  output->resize(elements);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t base = row * width;
    if (kernel == GenebOlmoLayerNormKernel::kPortableTwoPass) {
      float sum = 0.0F;
      for (std::size_t column = 0; column < width; ++column)
        sum += input[base + column];
      const float mean = sum / static_cast<float>(width);
      float sum_square = 0.0F;
      for (std::size_t column = 0; column < width; ++column) {
        const float centered = input[base + column] - mean;
        sum_square += centered * centered;
      }
      const float inverse =
          1.0F / std::sqrt(sum_square / static_cast<float>(width) + epsilon);
      for (std::size_t column = 0; column < width; ++column)
        (*output)[base + column] = (input[base + column] - mean) * inverse;
      continue;
    }

#if defined(__APPLE__) &&                                                      \
    (defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64))
    const auto moments = rowwise_moments(input.data() + base, width);
    const float inverse = 1.0F / std::sqrt(moments.variance + epsilon);
    const float normalization_bias = -inverse * moments.mean;
    const float32x4_t inverse_vector = vdupq_n_f32(inverse);
    const float32x4_t normalization_bias_vector =
        vdupq_n_f32(normalization_bias);
    std::size_t column = 0U;
    for (; column + 4U <= width; column += 4U) {
      const float32x4_t input_vector = vld1q_f32(input.data() + base + column);
      const float32x4_t normalized = vaddq_f32(
          vmulq_f32(input_vector, inverse_vector), normalization_bias_vector);
      vst1q_f32(output->data() + base + column, normalized);
    }
    for (; column < width; ++column)
      (*output)[base + column] =
          input[base + column] * inverse + normalization_bias;
#endif
  }
  return Status::Ok();
}

Status normalize(const std::vector<float> &input, const std::size_t rows,
                 const std::size_t width, const GenebOlmoNormType norm_type,
                 const GenebOlmoLayerNormKernel layer_norm_kernel,
                 const GenebOlmoTensorView *const scale, const float epsilon,
                 std::vector<float> *const output) {
  std::size_t elements = 0;
  if (output == nullptr || !checked_mul(rows, width, &elements) ||
      input.size() != elements)
    return invalid("normalization shape contract differs");
  if (norm_type == GenebOlmoNormType::kLayerNormNoAffine) {
    if (scale != nullptr)
      return invalid("LayerNorm topology/scale contract differs");
    return layer_norm_no_affine_impl(input, rows, width, epsilon,
                                     layer_norm_kernel, output);
  }
  if (norm_type == GenebOlmoNormType::kRmsNormAffine) {
    if (scale == nullptr || scale->dtype != TensorDType::kF32 ||
        scale->shape != std::vector<std::size_t>{width} ||
        scale->bytes != width * sizeof(float) || scale->data == nullptr)
      return invalid("affine RMSNorm scale contract differs");
    if (layer_norm_kernel != GenebOlmoLayerNormKernel::kPortableTwoPass)
      return invalid("RMSNorm cannot select a LayerNorm kernel");
  } else
    return invalid("normalization topology is unsupported");
  output->resize(elements);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t base = row * width;
    float sum_square = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      const float value = input[base + column];
      sum_square += value * value;
    }
    const float inverse =
        1.0F / std::sqrt(sum_square / static_cast<float>(width) + epsilon);
    for (std::size_t column = 0; column < width; ++column)
      (*output)[base + column] =
          input[base + column] * inverse * tensor_value(*scale, column);
  }
  return Status::Ok();
}

void apply_rope(std::vector<float> *const values, const std::size_t rows,
                const std::size_t heads, const std::size_t head_dimension,
                const float theta) {
  const std::size_t half = head_dimension / 2U;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t base = (row * heads + head) * head_dimension;
      for (std::size_t pair = 0; pair < half; ++pair) {
        const float exponent =
            static_cast<float>(pair * 2U) / static_cast<float>(head_dimension);
        const float angle = static_cast<float>(row) / std::pow(theta, exponent);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = (*values)[base + pair];
        const float second = (*values)[base + pair + half];
        (*values)[base + pair] = first * cosine - second * sine;
        (*values)[base + pair + half] = second * cosine + first * sine;
      }
    }
  }
}

Status causal_attention(const std::vector<float> &query,
                        const std::vector<float> &key,
                        const std::vector<float> &value,
                        const std::vector<std::uint8_t> &attention_mask,
                        const std::size_t rows, const std::size_t heads,
                        const std::size_t head_dimension,
                        std::vector<float> *const output) {
  std::size_t width = 0;
  std::size_t elements = 0;
  if (output == nullptr || !checked_mul(heads, head_dimension, &width) ||
      !checked_mul(rows, width, &elements) || query.size() != elements ||
      key.size() != elements || value.size() != elements ||
      attention_mask.size() != rows)
    return invalid("attention shape contract differs");
  output->assign(elements, 0.0F);
  std::vector<float> scores(rows, 0.0F);
  std::vector<float> probabilities(rows, 0.0F);
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t head = 0; head < heads; ++head) {
      std::size_t sources = 0;
      float maximum = -std::numeric_limits<float>::infinity();
      const std::size_t query_base = (row * heads + head) * head_dimension;
      for (std::size_t source = 0; source <= row; ++source) {
        if (attention_mask[source] == 0U)
          continue;
        const std::size_t key_base = (source * heads + head) * head_dimension;
        float score = 0.0F;
        for (std::size_t dimension = 0; dimension < head_dimension; ++dimension)
          score += query[query_base + dimension] * key[key_base + dimension];
        score *= scale;
        scores[sources++] = score;
        maximum = std::max(maximum, score);
      }
      if (sources == 0)
        return invalid("attention query has no visible valid key");
      float denominator = 0.0F;
      for (std::size_t source = 0; source < sources; ++source) {
        probabilities[source] = std::exp(scores[source] - maximum);
        denominator += probabilities[source];
      }
      for (std::size_t source = 0; source < sources; ++source)
        probabilities[source] /= denominator;
      for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        float total = 0.0F;
        std::size_t valid_index = 0;
        for (std::size_t source = 0; source <= row; ++source) {
          if (attention_mask[source] == 0U)
            continue;
          const std::size_t value_index =
              (source * heads + head) * head_dimension + dimension;
          total += probabilities[valid_index++] * value[value_index];
        }
        (*output)[query_base + dimension] = total;
      }
    }
  }
  return Status::Ok();
}

void add_in_place(std::vector<float> *const target,
                  const std::vector<float> &increment) {
  for (std::size_t index = 0; index < target->size(); ++index)
    (*target)[index] += increment[index];
}

void capture(const std::size_t layer, const std::vector<float> &values,
             const std::map<std::size_t, std::size_t> &capture_indices,
             std::vector<GenebOlmoHiddenCapture> *const captures) {
  const auto found = capture_indices.find(layer);
  if (found != capture_indices.end())
    (*captures)[found->second] = {layer, values};
}

} // namespace

Status geneb_olmo_layer_norm_no_affine(const std::vector<float> &input,
                                       const std::size_t rows,
                                       const std::size_t width,
                                       const float epsilon,
                                       const GenebOlmoLayerNormKernel kernel,
                                       std::vector<float> *const output) {
  return layer_norm_no_affine_impl(input, rows, width, epsilon, kernel, output);
}

Status validate_geneb_olmo_topology(const GenebOlmoTopology &topology) {
  if (topology.vocabulary_size == 0 ||
      !token_vocabulary_size_supported(topology.vocabulary_size) ||
      topology.width == 0 || topology.width > kMaximumWidth ||
      topology.layers == 0 || topology.layers > kMaximumLayers ||
      topology.heads == 0 || topology.width % topology.heads != 0 ||
      (topology.width / topology.heads) % 2U != 0 ||
      topology.fused_mlp_width == 0 ||
      topology.fused_mlp_width > kMaximumFusedMlpWidth ||
      topology.fused_mlp_width % 2U != 0 ||
      topology.maximum_sequence_length == 0 ||
      topology.maximum_sequence_length > kMaximumSequenceLength ||
      !std::isfinite(topology.norm_epsilon) || topology.norm_epsilon <= 0.0F ||
      !std::isfinite(topology.rope_theta) || topology.rope_theta <= 1.0F ||
      (topology.norm_type != GenebOlmoNormType::kLayerNormNoAffine &&
       topology.norm_type != GenebOlmoNormType::kRmsNormAffine) ||
      (topology.layer_norm_kernel !=
           GenebOlmoLayerNormKernel::kPortableTwoPass &&
       topology.layer_norm_kernel !=
           GenebOlmoLayerNormKernel::kTorch212AppleArm64ExactV1))
    return invalid("topology is outside the geneb-olmo-runtime-v1 contract");
  if (topology.layer_norm_kernel ==
          GenebOlmoLayerNormKernel::kTorch212AppleArm64ExactV1 &&
      (topology.norm_type != GenebOlmoNormType::kLayerNormNoAffine ||
       topology.width != 2048U || topology.layers != 16U ||
       topology.heads != 16U || topology.fused_mlp_width != 16384U ||
       topology.norm_epsilon != 1.0e-5F))
    return invalid("exact Apple-arm64 LayerNorm kernel requires the closed "
                   "Omni-DNA-1B F32 topology");
  std::size_t ignored = 0;
  if (!checked_mul(topology.vocabulary_size, topology.width, &ignored) ||
      !checked_mul(topology.width, topology.width, &ignored) ||
      !checked_mul(topology.fused_mlp_width, topology.width, &ignored))
    return invalid("topology tensor shapes overflow this process");
  return Status::Ok();
}

Status geneb_olmo_topology_from_artifact(const ModelFile &artifact,
                                         GenebOlmoTopology *const output) {
  if (output == nullptr)
    return invalid("topology output is null");
  if (artifact.profile() != kGenebOlmoArtifactProfile)
    return format_error("profile must be '" +
                        std::string{kGenebOlmoArtifactProfile} + "'");
  if (!artifact.tokenizer_asset_descriptor().has_value())
    return format_error("the four-key tokenizer descriptor is required");
  auto status = require_literal(artifact, "runtime.abi", kGenebOlmoRuntimeAbi);
  if (!status.ok())
    return status;
  status =
      require_literal(artifact, "model.architecture", kGenebOlmoArchitecture);
  if (!status.ok())
    return status;
  for (const auto &[key, expected] :
       std::vector<std::pair<std::string_view, std::string_view>>{
           {"olmo.block_type", "sequential"},
           {"olmo.activation", "swiglu"},
           {"olmo.qkv_layout", "q-k-v"},
           {"olmo.swiglu_layout", "x-gate"},
           {"olmo.rope_layout", "split-half"},
           {"olmo.weight_dtype", "F32"},
           {"source.olmo.repo", "allenai/OLMo"},
           {"source.olmo.revision", "6c3373fa182af2d57fe3c390ffc8420d5c5b325a"},
           {"source.olmo.version", "0.6.0"},
           {"source.olmo.model_py_sha256",
            "1986566f15ceaa1177604fc7404a3f51942b7c818b5ae3d3afb553e7dc2504bc"},
           {"source.olmo.config_py_sha256",
            "84a5265454458b5151a2ca0adfdc0536ba93451228605f96c4b3d89b8604cd74"},
           {"source.remote_modeling_sha256",
            "8e54a1c85cffe7eb9049549e2213d087cdfe56db4f7b0ae211171f98aeec1c21"},
           {"source.remote_configuration_sha256",
            "6e23d2c8ae0d9420670ee9a00de279d04e93c86d1a6b555dd62c1b49ecae12a"
            "5"}}) {
    status = require_literal(artifact, key, expected);
    if (!status.ok())
      return status;
  }
  for (const auto &[key, expected] :
       std::vector<std::pair<std::string_view, bool>>{
           {"olmo.norm_after", false},
           {"olmo.attention_layer_norm", false},
           {"olmo.include_bias", false},
           {"olmo.weight_tying", true},
           {"olmo.rope_full_precision", true}}) {
    status = require_bool(artifact, key, expected);
    if (!status.ok())
      return status;
  }

  std::uint64_t vocabulary = 0;
  std::uint64_t width = 0;
  std::uint64_t layers = 0;
  std::uint64_t max_sequence = 0;
  std::uint64_t heads = 0;
  std::uint64_t fused_mlp = 0;
  std::uint64_t embedding_layers = 0;
  if (!metadata_u64(artifact, "config.vocab_size", &vocabulary) ||
      !metadata_u64(artifact, "config.hidden_size", &width) ||
      !metadata_u64(artifact, "config.num_layers", &layers) ||
      !metadata_u64(artifact, "config.max_seqlen", &max_sequence) ||
      !metadata_u64(artifact, "olmo.num_attention_heads", &heads) ||
      !metadata_u64(artifact, "olmo.fused_mlp_width", &fused_mlp) ||
      !metadata_u64(artifact, "runtime.embedding_layer_count",
                    &embedding_layers))
    return format_error(
        "required integer topology metadata is missing/wrong-typed");
  if (layers == std::numeric_limits<std::uint64_t>::max() ||
      embedding_layers != layers + 1U)
    return format_error(
        "runtime.embedding_layer_count must equal num_layers+1");
  if (vocabulary > std::numeric_limits<std::size_t>::max() ||
      width > std::numeric_limits<std::size_t>::max() ||
      layers > std::numeric_limits<std::size_t>::max() ||
      max_sequence > std::numeric_limits<std::size_t>::max() ||
      heads > std::numeric_limits<std::size_t>::max() ||
      fused_mlp > std::numeric_limits<std::size_t>::max())
    return format_error("topology does not fit this process");

  double epsilon = 0.0;
  double theta = 0.0;
  if (!metadata_f64(artifact, "olmo.norm_epsilon", &epsilon) ||
      !metadata_f64(artifact, "olmo.rope_theta", &theta))
    return format_error(
        "required floating-point topology metadata is missing/wrong-typed");
  std::string norm_type;
  if (!metadata_string(artifact, "olmo.norm_type", &norm_type))
    return format_error("olmo.norm_type must be a string");
  std::string runtime_id;
  if (!metadata_string(artifact, "geneb.runtime_id", &runtime_id))
    return format_error("geneb.runtime_id must be a string");

  GenebOlmoTopology topology;
  topology.vocabulary_size = static_cast<std::size_t>(vocabulary);
  topology.width = static_cast<std::size_t>(width);
  topology.layers = static_cast<std::size_t>(layers);
  topology.heads = static_cast<std::size_t>(heads);
  topology.fused_mlp_width = static_cast<std::size_t>(fused_mlp);
  topology.maximum_sequence_length = static_cast<std::size_t>(max_sequence);
  topology.norm_epsilon = static_cast<float>(epsilon);
  topology.rope_theta = static_cast<float>(theta);
  if (norm_type == "layernorm-no-affine")
    topology.norm_type = GenebOlmoNormType::kLayerNormNoAffine;
  else if (norm_type == "rmsnorm-affine")
    topology.norm_type = GenebOlmoNormType::kRmsNormAffine;
  else
    return format_error("olmo.norm_type is unsupported");
  const auto *const layer_norm_kernel =
      artifact.find_metadata("olmo.layer_norm_kernel");
  if (runtime_id == "geneb-omni-dna-1b") {
    std::string value;
    if (layer_norm_kernel == nullptr ||
        !metadata_string(artifact, "olmo.layer_norm_kernel", &value) ||
        value != kGenebOlmoTorch212AppleArm64LayerNormKernel)
      return format_error(
          "Omni-DNA-1B requires olmo.layer_norm_kernel='" +
          std::string{kGenebOlmoTorch212AppleArm64LayerNormKernel} + "'");
    topology.layer_norm_kernel =
        GenebOlmoLayerNormKernel::kTorch212AppleArm64ExactV1;
  } else if (layer_norm_kernel != nullptr) {
    return format_error(
        "olmo.layer_norm_kernel is only valid for geneb-omni-dna-1b");
  }
  status = validate_geneb_olmo_topology(topology);
  if (!status.ok())
    return format_error(status.message());
  *output = topology;
  return Status::Ok();
}

Status canonical_geneb_olmo_tensors(
    const GenebOlmoTopology &topology,
    std::vector<GenebOlmoTensorRequirement> *const output) {
  if (output == nullptr)
    return invalid("tensor requirement output is null");
  auto status = validate_geneb_olmo_topology(topology);
  if (!status.ok())
    return status;
  std::vector<GenebOlmoTensorRequirement> result;
  result.reserve(
      1U +
      topology.layers *
          (topology.norm_type == GenebOlmoNormType::kRmsNormAffine ? 6U : 4U) +
      (topology.norm_type == GenebOlmoNormType::kRmsNormAffine ? 1U : 0U));
  result.push_back({"model.transformer.wte.weight",
                    TensorDType::kF32,
                    {topology.vocabulary_size, topology.width}});
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    const std::string prefix =
        "model.transformer.blocks." + std::to_string(layer) + ".";
    result.push_back({prefix + "att_proj.weight",
                      TensorDType::kF32,
                      {topology.width * 3U, topology.width}});
    if (topology.norm_type == GenebOlmoNormType::kRmsNormAffine)
      result.push_back(
          {prefix + "attn_norm.weight", TensorDType::kF32, {topology.width}});
    result.push_back({prefix + "attn_out.weight",
                      TensorDType::kF32,
                      {topology.width, topology.width}});
    if (topology.norm_type == GenebOlmoNormType::kRmsNormAffine)
      result.push_back(
          {prefix + "ff_norm.weight", TensorDType::kF32, {topology.width}});
    result.push_back({prefix + "ff_out.weight",
                      TensorDType::kF32,
                      {topology.width, topology.fused_mlp_width / 2U}});
    result.push_back({prefix + "ff_proj.weight",
                      TensorDType::kF32,
                      {topology.fused_mlp_width, topology.width}});
  }
  if (topology.norm_type == GenebOlmoNormType::kRmsNormAffine)
    result.push_back(
        {"model.transformer.ln_f.weight", TensorDType::kF32, {topology.width}});
  *output = std::move(result);
  return Status::Ok();
}

struct GenebOlmoModel::Impl final {
  struct Block final {
    GenebOlmoTensorView qkv;
    GenebOlmoTensorView attention_norm;
    GenebOlmoTensorView attention_output;
    GenebOlmoTensorView feed_forward_norm;
    GenebOlmoTensorView feed_forward_output;
    GenebOlmoTensorView feed_forward_projection;
  };

  GenebOlmoTopology topology;
  GenebOlmoTensorView embedding;
  std::vector<Block> blocks;
  GenebOlmoTensorView final_norm;
  std::shared_ptr<evo::detail::LinearExecutor> executor;
};

GenebOlmoModel::GenebOlmoModel() = default;
GenebOlmoModel::~GenebOlmoModel() = default;
GenebOlmoModel::GenebOlmoModel(GenebOlmoModel &&) noexcept = default;
GenebOlmoModel &GenebOlmoModel::operator=(GenebOlmoModel &&) noexcept = default;

Status GenebOlmoModel::load(
    const GenebOlmoTopology &topology,
    const std::vector<GenebOlmoNamedTensorView> &tensors,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  std::vector<GenebOlmoTensorRequirement> requirements;
  auto status = canonical_geneb_olmo_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  std::map<std::string, const GenebOlmoTensorView *, std::less<>> provided;
  for (const auto &named : tensors) {
    if (named.name.empty() ||
        !provided.emplace(named.name, &named.tensor).second)
      return invalid("tensor names must be nonempty and unique");
  }
  if (provided.size() != requirements.size())
    return invalid("tensor set count differs from exact OLMo manifest");
  for (const auto &requirement : requirements) {
    const auto found = provided.find(requirement.name);
    if (found == provided.end())
      return invalid("missing tensor '" + requirement.name + "'");
    status = validate_tensor(*found->second, requirement);
    if (!status.ok())
      return status;
  }

  auto candidate = std::make_unique<Impl>();
  candidate->topology = topology;
  candidate->embedding = *provided.at("model.transformer.wte.weight");
  candidate->blocks.resize(topology.layers);
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    const std::string prefix =
        "model.transformer.blocks." + std::to_string(layer) + ".";
    auto &block = candidate->blocks[layer];
    block.qkv = *provided.at(prefix + "att_proj.weight");
    block.attention_output = *provided.at(prefix + "attn_out.weight");
    block.feed_forward_output = *provided.at(prefix + "ff_out.weight");
    block.feed_forward_projection = *provided.at(prefix + "ff_proj.weight");
    if (topology.norm_type == GenebOlmoNormType::kRmsNormAffine) {
      block.attention_norm = *provided.at(prefix + "attn_norm.weight");
      block.feed_forward_norm = *provided.at(prefix + "ff_norm.weight");
    }
  }
  if (topology.norm_type == GenebOlmoNormType::kRmsNormAffine)
    candidate->final_norm = *provided.at("model.transformer.ln_f.weight");
  candidate->executor = linear_executor != nullptr
                            ? std::move(linear_executor)
                            : std::make_shared<ReferenceLinearExecutor>();
  impl_ = std::move(candidate);
  return Status::Ok();
}

Status GenebOlmoModel::load(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  GenebOlmoTopology topology;
  auto status = geneb_olmo_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebOlmoNamedTensorView> views;
  views.reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
      if (tensor.dimensions[dimension] >
          std::numeric_limits<std::size_t>::max())
        return format_error("tensor '" + tensor.name +
                            "' shape does not fit this process");
      shape.push_back(static_cast<std::size_t>(tensor.dimensions[dimension]));
    }
    if (tensor.data_size > std::numeric_limits<std::size_t>::max())
      return format_error("tensor '" + tensor.name +
                          "' byte size does not fit this process");
    views.push_back({tensor.name,
                     {artifact.tensor_data(tensor),
                      static_cast<std::size_t>(tensor.data_size), tensor.dtype,
                      std::move(shape)}});
  }
  status = load(topology, views, std::move(linear_executor));
  if (!status.ok())
    return format_error(status.message());
  return Status::Ok();
}

const GenebOlmoTopology *GenebOlmoModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebOlmoModel::linear_executor_name() const noexcept {
  return impl_ == nullptr || impl_->executor == nullptr
             ? "unloaded"
             : impl_->executor->name();
}

Status GenebOlmoModel::forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebOlmoForwardResult *const output) const {
  if (impl_ == nullptr)
    return invalid("model is not loaded");
  if (output == nullptr)
    return invalid("forward output is null");
  const auto &config = impl_->topology;
  if (config.layer_norm_kernel ==
          GenebOlmoLayerNormKernel::kTorch212AppleArm64ExactV1 &&
      !torch212_apple_arm64_layer_norm_supported())
    return {ErrorCode::kUnsupported,
            "olmo.layer_norm_kernel=torch-2.1.2-apple-arm64-exact-v1 "
            "requires Apple arm64"};
  if (tokens.empty() || tokens.size() > config.maximum_sequence_length ||
      attention_mask.size() != tokens.size())
    return invalid("token/mask length is outside the model context");
  bool saw_padding = false;
  std::size_t valid_tokens = 0;
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (static_cast<std::uint64_t>(tokens[index]) >= config.vocabulary_size)
      return invalid("token ID is outside the embedding vocabulary");
    const auto mask = attention_mask[index];
    if (mask > 1U || (mask == 1U && saw_padding))
      return invalid("attention mask must be right-padded binary data");
    if (mask == 0U)
      saw_padding = true;
    else
      ++valid_tokens;
  }
  if (valid_tokens == 0)
    return invalid("attention mask must contain a valid token");

  std::map<std::size_t, std::size_t> capture_indices;
  for (std::size_t index = 0; index < capture_layers.size(); ++index) {
    const auto layer = capture_layers[index];
    if (layer > config.layers || !capture_indices.emplace(layer, index).second)
      return invalid("capture layers must be unique and in [0,num_layers]");
  }

  std::size_t hidden_elements = 0;
  if (!checked_mul(tokens.size(), config.width, &hidden_elements))
    return invalid("hidden state shape overflows");
  std::vector<float> hidden(hidden_elements, 0.0F);
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    const auto token = static_cast<std::size_t>(tokens[row]);
    for (std::size_t column = 0; column < config.width; ++column)
      hidden[row * config.width + column] =
          tensor_value(impl_->embedding, token * config.width + column);
  }

  GenebOlmoForwardResult result;
  result.rows = tokens.size();
  result.width = config.width;
  result.captures.resize(capture_layers.size());
  capture(0, hidden, capture_indices, &result.captures);

  const std::size_t head_dimension = config.width / config.heads;
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    const auto &block = impl_->blocks[layer];
    const GenebOlmoTensorView *const attention_scale =
        config.norm_type == GenebOlmoNormType::kRmsNormAffine
            ? &block.attention_norm
            : nullptr;
    const GenebOlmoTensorView *const feed_forward_scale =
        config.norm_type == GenebOlmoNormType::kRmsNormAffine
            ? &block.feed_forward_norm
            : nullptr;
    std::vector<float> normalized;
    auto status = normalize(hidden, tokens.size(), config.width,
                            config.norm_type, config.layer_norm_kernel,
                            attention_scale, config.norm_epsilon, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> fused_qkv;
    status = linear(normalized, tokens.size(), config.width, block.qkv,
                    config.width * 3U, impl_->executor.get(), &fused_qkv);
    if (!status.ok())
      return status;
    std::vector<float> query(hidden_elements);
    std::vector<float> key(hidden_elements);
    std::vector<float> value(hidden_elements);
    for (std::size_t row = 0; row < tokens.size(); ++row) {
      const std::size_t fused_base = row * config.width * 3U;
      const std::size_t hidden_base = row * config.width;
      std::copy_n(fused_qkv.begin() + static_cast<std::ptrdiff_t>(fused_base),
                  config.width,
                  query.begin() + static_cast<std::ptrdiff_t>(hidden_base));
      std::copy_n(fused_qkv.begin() +
                      static_cast<std::ptrdiff_t>(fused_base + config.width),
                  config.width,
                  key.begin() + static_cast<std::ptrdiff_t>(hidden_base));
      std::copy_n(fused_qkv.begin() + static_cast<std::ptrdiff_t>(
                                          fused_base + config.width * 2U),
                  config.width,
                  value.begin() + static_cast<std::ptrdiff_t>(hidden_base));
    }
    apply_rope(&query, tokens.size(), config.heads, head_dimension,
               config.rope_theta);
    apply_rope(&key, tokens.size(), config.heads, head_dimension,
               config.rope_theta);
    std::vector<float> attended;
    status = causal_attention(query, key, value, attention_mask, tokens.size(),
                              config.heads, head_dimension, &attended);
    if (!status.ok())
      return status;
    std::vector<float> attention_output;
    status =
        linear(attended, tokens.size(), config.width, block.attention_output,
               config.width, impl_->executor.get(), &attention_output);
    if (!status.ok())
      return status;
    add_in_place(&hidden, attention_output);

    status = normalize(hidden, tokens.size(), config.width, config.norm_type,
                       config.layer_norm_kernel, feed_forward_scale,
                       config.norm_epsilon, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> fused_feed_forward;
    status = linear(normalized, tokens.size(), config.width,
                    block.feed_forward_projection, config.fused_mlp_width,
                    impl_->executor.get(), &fused_feed_forward);
    if (!status.ok())
      return status;
    const std::size_t activated_width = config.fused_mlp_width / 2U;
    std::vector<float> activated(tokens.size() * activated_width, 0.0F);
    for (std::size_t row = 0; row < tokens.size(); ++row) {
      const std::size_t fused_base = row * config.fused_mlp_width;
      const std::size_t activated_base = row * activated_width;
      for (std::size_t column = 0; column < activated_width; ++column) {
        const float x = fused_feed_forward[fused_base + column];
        const float gate =
            fused_feed_forward[fused_base + activated_width + column];
        const float silu = gate / (1.0F + std::exp(-gate));
        activated[activated_base + column] = silu * x;
      }
    }
    std::vector<float> feed_forward_output;
    status = linear(activated, tokens.size(), activated_width,
                    block.feed_forward_output, config.width,
                    impl_->executor.get(), &feed_forward_output);
    if (!status.ok())
      return status;
    add_in_place(&hidden, feed_forward_output);
    if (layer + 1U < config.layers)
      capture(layer + 1U, hidden, capture_indices, &result.captures);
  }

  const GenebOlmoTensorView *const final_scale =
      config.norm_type == GenebOlmoNormType::kRmsNormAffine ? &impl_->final_norm
                                                            : nullptr;
  std::vector<float> final_hidden;
  auto status = normalize(hidden, tokens.size(), config.width, config.norm_type,
                          config.layer_norm_kernel, final_scale,
                          config.norm_epsilon, &final_hidden);
  if (!status.ok())
    return status;
  capture(config.layers, final_hidden, capture_indices, &result.captures);
  result.pooled.assign(config.width, 0.0F);
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    if (attention_mask[row] == 0U)
      continue;
    for (std::size_t column = 0; column < config.width; ++column)
      result.pooled[column] += final_hidden[row * config.width + column];
  }
  for (auto &value : result.pooled)
    value /= static_cast<float>(valid_tokens);
  result.final_hidden = std::move(final_hidden);
  *output = std::move(result);
  return Status::Ok();
}

} // namespace evo::cpu
