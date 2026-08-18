// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_esm.hpp"

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
#include <string_view>
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
constexpr std::size_t kMaximumInnerWidth = 1U << 20U;
constexpr float kTokenDropoutTrainingRatio = 0.15F * 0.8F;

Status invalid(const std::string &message) {
  return {ErrorCode::kInvalidArgument, "GENEB ESM: " + message};
}

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "GENEB ESM artifact: " + message};
}

bool checked_mul(const std::size_t left, const std::size_t right,
                 std::size_t *const output) noexcept {
  if (output == nullptr ||
      (left != 0 && right > std::numeric_limits<std::size_t>::max() / left))
    return false;
  *output = left * right;
  return true;
}

bool finite_vector(const std::vector<float> &values) noexcept {
  for (const float value : values) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

Status metadata_entry(const ModelFile &artifact, const std::string_view key,
                      const MetadataType type,
                      const MetadataEntry **const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr)
    return format_error("required metadata is missing: " + std::string{key});
  if (entry->type != type)
    return format_error("metadata has wrong type: " + std::string{key});
  *output = entry;
  return Status::Ok();
}

Status metadata_string(const ModelFile &artifact, const std::string_view key,
                       std::string *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kString, &entry);
  if (!status.ok())
    return status;
  output->assign(reinterpret_cast<const char *>(entry->value.data()),
                 entry->value.size());
  return Status::Ok();
}

Status metadata_size(const ModelFile &artifact, const std::string_view key,
                     std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(std::uint64_t))
    return format_error("u64 metadata is malformed: " + std::string{key});
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<std::uint64_t>(entry->value[byte]) << (byte * 8U);
  if (value > std::numeric_limits<std::size_t>::max())
    return format_error("metadata exceeds size_t: " + std::string{key});
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

Status metadata_float(const ModelFile &artifact, const std::string_view key,
                      float *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kF64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(double))
    return format_error("f64 metadata is malformed: " + std::string{key});
  std::uint64_t bits = 0;
  for (std::size_t byte = 0; byte < sizeof(bits); ++byte)
    bits |= static_cast<std::uint64_t>(entry->value[byte]) << (byte * 8U);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  if (!std::isfinite(value) ||
      value > static_cast<double>(std::numeric_limits<float>::max()) ||
      value < -static_cast<double>(std::numeric_limits<float>::max()))
    return format_error("metadata is not finite F32: " + std::string{key});
  *output = static_cast<float>(value);
  return Status::Ok();
}

Status metadata_bool(const ModelFile &artifact, const std::string_view key,
                     bool *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kBool, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != 1U || entry->value[0] > 1U)
    return format_error("bool metadata is malformed: " + std::string{key});
  *output = entry->value[0] != 0U;
  return Status::Ok();
}

Status require_literal(const ModelFile &artifact, const std::string_view key,
                       const std::string_view expected) {
  std::string value;
  auto status = metadata_string(artifact, key, &value);
  if (!status.ok())
    return status;
  if (value != expected)
    return format_error("metadata '" + std::string{key} + "' must be '" +
                        std::string{expected} + "'");
  return Status::Ok();
}

Status require_bool(const ModelFile &artifact, const std::string_view key,
                    const bool expected) {
  bool value = false;
  auto status = metadata_bool(artifact, key, &value);
  if (!status.ok())
    return status;
  if (value != expected)
    return format_error("metadata '" + std::string{key} + "' must be " +
                        (expected ? "true" : "false"));
  return Status::Ok();
}

float tensor_value(const GenebEsmTensorView &tensor,
                   const std::size_t index) noexcept {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return value;
}

Status validate_tensor(const GenebEsmTensorView &tensor,
                       const GenebEsmTensorRequirement &requirement) {
  if (tensor.data == nullptr)
    return format_error("tensor '" + requirement.name + "' has null data");
  if (tensor.dtype != requirement.dtype)
    return format_error("tensor '" + requirement.name + "' has wrong dtype");
  if (tensor.shape != requirement.shape)
    return format_error("tensor '" + requirement.name + "' has wrong shape");
  std::size_t elements = 1;
  for (const std::size_t dimension : tensor.shape) {
    if (dimension == 0 || !checked_mul(elements, dimension, &elements))
      return format_error("tensor '" + requirement.name +
                          "' has an overflowing shape");
  }
  std::size_t expected_bytes = 0;
  if (!checked_mul(elements, sizeof(float), &expected_bytes) ||
      tensor.bytes != expected_bytes)
    return format_error("tensor '" + requirement.name +
                        "' has wrong byte size");
  return Status::Ok();
}

std::string layer_prefix(const std::size_t layer) {
  return "esm.encoder.layer." + std::to_string(layer) + ".";
}

void add_requirement(std::vector<GenebEsmTensorRequirement> *const output,
                     std::string name, std::vector<std::size_t> shape) {
  output->push_back({std::move(name), TensorDType::kF32, std::move(shape)});
}

Status runtime_linear(const std::vector<float> &input, const std::size_t rows,
                      const std::size_t input_width,
                      const GenebEsmTensorView &weight,
                      const std::size_t output_width,
                      const GenebEsmTensorView *const bias,
                      evo::detail::LinearExecutor *const executor,
                      std::vector<float> *const output) {
  std::size_t input_elements = 0;
  std::size_t weight_elements = 0;
  std::size_t output_elements = 0;
  if (output == nullptr || rows == 0 || input_width == 0 || output_width == 0 ||
      !checked_mul(rows, input_width, &input_elements) ||
      !checked_mul(output_width, input_width, &weight_elements) ||
      !checked_mul(rows, output_width, &output_elements) ||
      input.size() != input_elements || weight.data == nullptr ||
      weight.dtype != TensorDType::kF32 ||
      weight.bytes != weight_elements * sizeof(float) ||
      (bias != nullptr &&
       (bias->data == nullptr || bias->dtype != TensorDType::kF32 ||
        bias->bytes != output_width * sizeof(float))))
    return invalid("linear shape/tensor contract differs");

  if (executor != nullptr) {
    const evo::detail::LinearTensorView weight_view{weight.data, weight.dtype,
                                                    weight_elements};
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
    if (output->size() != output_elements || !finite_vector(*output))
      return invalid("linear executor returned invalid output");
    return Status::Ok();
  }

  output->assign(output_elements, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t target = 0; target < output_width; ++target) {
      float total = bias == nullptr ? 0.0F : tensor_value(*bias, target);
      const std::size_t weight_base = target * input_width;
      const std::size_t input_base = row * input_width;
      for (std::size_t source = 0; source < input_width; ++source)
        total += input[input_base + source] *
                 tensor_value(weight, weight_base + source);
      if (!std::isfinite(total))
        return invalid("linear accumulation became non-finite");
      (*output)[row * output_width + target] = total;
    }
  }
  return Status::Ok();
}

void add_in_place(std::vector<float> *const target,
                  const std::vector<float> &increment) noexcept {
  for (std::size_t index = 0; index < target->size(); ++index)
    (*target)[index] += increment[index];
}

float exact_gelu(const float value) noexcept {
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  return value * 0.5F * (1.0F + std::erf(value * kInverseSqrtTwo));
}

float silu(const float value) noexcept {
  if (value >= 0.0F)
    return value / (1.0F + std::exp(-value));
  const float exponential = std::exp(value);
  return value * exponential / (1.0F + exponential);
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
// 2.1.2 Apple-arm64 F32 LayerNorm reduction selected only by Agro-NT-1B.
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


void capture_hidden(const std::size_t layer, const std::vector<float> &values,
                    const std::map<std::size_t, std::size_t> &indices,
                    std::vector<GenebEsmHiddenCapture> *const captures) {
  const auto found = indices.find(layer);
  if (found != indices.end())
    (*captures)[found->second] = {layer, values};
}

} // namespace

Status validate_geneb_esm_topology(const GenebEsmTopology &topology) {
  if (topology.vocabulary_size == 0 ||
      !token_vocabulary_size_supported(topology.vocabulary_size) ||
      topology.width == 0 || topology.width > kMaximumWidth ||
      topology.layers == 0 || topology.layers > kMaximumLayers ||
      topology.heads == 0 || topology.head_dimension == 0 ||
      topology.inner_width == 0 || topology.inner_width > kMaximumInnerWidth ||
      topology.maximum_sequence_length == 0 ||
      topology.maximum_sequence_length > kMaximumSequenceLength ||
      topology.position_embedding_count == 0 ||
      topology.position_embedding_count > kMaximumSequenceLength ||
      !std::isfinite(topology.layer_norm_epsilon) ||
      topology.layer_norm_epsilon <= 0.0F ||
      !std::isfinite(topology.rope_base) || topology.rope_base <= 1.0F ||
      (topology.position_type != GenebEsmPositionType::kAbsolute &&
       topology.position_type != GenebEsmPositionType::kRotary) ||
      (topology.mlp_activation != GenebEsmMlpActivation::kGelu &&
       topology.mlp_activation != GenebEsmMlpActivation::kSwiGlu) ||
      (topology.layer_norm_kernel !=
           GenebEsmLayerNormKernel::kPortableTwoPass &&
       topology.layer_norm_kernel !=
           GenebEsmLayerNormKernel::kTorch212AppleArm64ExactV1))
    return invalid("topology is outside geneb-esm-runtime-v1 limits");

  if (topology.layer_norm_kernel ==
          GenebEsmLayerNormKernel::kTorch212AppleArm64ExactV1 &&
      (topology.width != 1500U || topology.layer_norm_epsilon != 1.0e-12F))
    return invalid(
        "exact LayerNorm kernel requires width=1500 and epsilon=1e-12");

  std::size_t attention_width = 0;
  if (!checked_mul(topology.heads, topology.head_dimension, &attention_width) ||
      attention_width != topology.width)
    return invalid("heads*head_dimension must equal hidden width");
  if (topology.position_type == GenebEsmPositionType::kRotary &&
      topology.head_dimension % 2U != 0U)
    return invalid("rotary head dimension must be even");
  if (topology.pad_token_id >= topology.vocabulary_size ||
      topology.mask_token_id >= topology.vocabulary_size ||
      topology.cls_token_id >= topology.vocabulary_size ||
      topology.pad_token_id == topology.mask_token_id ||
      topology.pad_token_id == topology.cls_token_id ||
      topology.mask_token_id == topology.cls_token_id)
    return invalid("pad/mask/CLS token IDs must be distinct and in vocabulary");
  if (topology.pad_token_id > std::numeric_limits<std::size_t>::max() -
                                  topology.maximum_sequence_length ||
      topology.pad_token_id + topology.maximum_sequence_length >=
          topology.position_embedding_count)
    return invalid(
        "position embedding table cannot represent maximum cumsum ID");
  std::size_t ignored = 0;
  const std::size_t fused_inner =
      topology.mlp_activation == GenebEsmMlpActivation::kSwiGlu
          ? topology.inner_width * 2U
          : topology.inner_width;
  if (!checked_mul(topology.vocabulary_size, topology.width, &ignored) ||
      !checked_mul(topology.position_embedding_count, topology.width,
                   &ignored) ||
      !checked_mul(topology.width, topology.width, &ignored) ||
      !checked_mul(fused_inner, topology.width, &ignored))
    return invalid("topology tensor shapes overflow this process");
  return Status::Ok();
}

Status geneb_esm_topology_from_artifact(const ModelFile &artifact,
                                        GenebEsmTopology *const output) {
  if (output == nullptr)
    return invalid("topology output is null");
  if (artifact.profile() != kGenebEsmArtifactProfile)
    return format_error("profile must be '" +
                        std::string{kGenebEsmArtifactProfile} + "'");
  if (!artifact.tokenizer_asset_descriptor().has_value())
    return format_error("the four-key tokenizer descriptor is required");
  auto status = require_literal(artifact, "runtime.abi", kGenebEsmRuntimeAbi);
  if (!status.ok())
    return status;
  status =
      require_literal(artifact, "model.architecture", kGenebEsmArchitecture);
  if (!status.ok())
    return status;

  const std::set<std::string_view> expected_esm_metadata{
      "esm.vocab_size",
      "esm.hidden_size",
      "esm.num_layers",
      "esm.num_attention_heads",
      "esm.head_dim",
      "esm.intermediate_size",
      "esm.max_seqlen",
      "esm.max_position_embeddings",
      "esm.layer_norm_epsilon",
      "esm.layer_norm_kernel",
      "esm.rope_base",
      "esm.position_embedding_type",
      "esm.rope_layout",
      "esm.mlp_activation",
      "esm.attention_bias",
      "esm.feed_forward_bias",
      "esm.embedding_layer_norm_before",
      "esm.final_layer_norm",
      "esm.token_dropout",
      "esm.token_dropout_training_ratio",
      "esm.pad_token_id",
      "esm.mask_token_id",
      "esm.cls_token_id",
      "esm.special_token_policy",
      "esm.position_id_mode",
      "esm.hidden_tap",
      "esm.pooling",
      "esm.weight_dtype",
      "esm.q_scale_placement",
      "esm.attention_mask",
      "esm.source_unused_position_embeddings",
  };
  for (const auto &entry : artifact.metadata()) {
    if (entry.key.compare(0, 4, "esm.") == 0 &&
        expected_esm_metadata.find(std::string_view{entry.key}) ==
            expected_esm_metadata.end())
      return format_error("unexpected esm.* metadata: " + entry.key);
  }

  for (const auto &[key, expected] :
       std::vector<std::pair<std::string_view, std::string_view>>{
           {"esm.rope_layout", "split-half"},
           {"esm.special_token_policy", "cls-only"},
           {"esm.position_id_mode", "padding-cumsum"},
           {"esm.hidden_tap", "last-hidden-state"},
           {"esm.pooling", "attention-mask-mean"},
           {"esm.weight_dtype", "F32"},
           {"esm.q_scale_placement", "before-rope"},
           {"esm.attention_mask", "bidirectional-key-mask"}}) {
    status = require_literal(artifact, key, expected);
    if (!status.ok())
      return status;
  }
  status = require_bool(artifact, "esm.embedding_layer_norm_before", false);
  if (!status.ok())
    return status;
  status = require_bool(artifact, "esm.final_layer_norm", true);
  if (!status.ok())
    return status;

  GenebEsmTopology topology;
  for (const auto &[key, target] :
       std::vector<std::pair<std::string_view, std::size_t *>>{
           {"esm.vocab_size", &topology.vocabulary_size},
           {"esm.hidden_size", &topology.width},
           {"esm.num_layers", &topology.layers},
           {"esm.num_attention_heads", &topology.heads},
           {"esm.head_dim", &topology.head_dimension},
           {"esm.intermediate_size", &topology.inner_width},
           {"esm.max_seqlen", &topology.maximum_sequence_length},
           {"esm.max_position_embeddings", &topology.position_embedding_count},
           {"esm.pad_token_id", &topology.pad_token_id},
           {"esm.mask_token_id", &topology.mask_token_id},
           {"esm.cls_token_id", &topology.cls_token_id}}) {
    status = metadata_size(artifact, key, target);
    if (!status.ok())
      return status;
  }
  status = metadata_float(artifact, "esm.layer_norm_epsilon",
                          &topology.layer_norm_epsilon);
  if (!status.ok())
    return status;
  status = metadata_float(artifact, "esm.rope_base", &topology.rope_base);
  if (!status.ok())
    return status;
  status =
      metadata_bool(artifact, "esm.attention_bias", &topology.attention_bias);
  if (!status.ok())
    return status;
  status = metadata_bool(artifact, "esm.feed_forward_bias",
                         &topology.feed_forward_bias);
  if (!status.ok())
    return status;
  status =
      metadata_bool(artifact, "esm.token_dropout", &topology.token_dropout);
  if (!status.ok())
    return status;

  float dropout_ratio = 0.0F;
  status = metadata_float(artifact, "esm.token_dropout_training_ratio",
                          &dropout_ratio);
  if (!status.ok())
    return status;
  if (std::abs(dropout_ratio - kTokenDropoutTrainingRatio) > 1.0e-7F)
    return format_error("token dropout training ratio must be 0.12");

  std::string value;
  status = metadata_string(artifact, "esm.position_embedding_type", &value);
  if (!status.ok())
    return status;
  if (value == "absolute")
    topology.position_type = GenebEsmPositionType::kAbsolute;
  else if (value == "rotary")
    topology.position_type = GenebEsmPositionType::kRotary;
  else
    return format_error("position_embedding_type is unsupported");
  status = metadata_string(artifact, "esm.mlp_activation", &value);
  if (!status.ok())
    return status;
  if (value == "gelu")
    topology.mlp_activation = GenebEsmMlpActivation::kGelu;
  else if (value == "swiglu")
    topology.mlp_activation = GenebEsmMlpActivation::kSwiGlu;
  else
    return format_error("mlp_activation is unsupported");

  std::string runtime_id;
  status = metadata_string(artifact, "geneb.runtime_id", &runtime_id);
  if (!status.ok())
    return status;
  const auto *const layer_norm_kernel =
      artifact.find_metadata("esm.layer_norm_kernel");
  if (runtime_id == "geneb-agro-nt-1b") {
    if (layer_norm_kernel == nullptr)
      return format_error(
          "Agro-NT-1B requires esm.layer_norm_kernel='" +
          std::string{kGenebEsmTorch212AppleArm64LayerNormKernel} + "'");
    status = metadata_string(artifact, "esm.layer_norm_kernel", &value);
    if (!status.ok())
      return status;
    if (value != kGenebEsmTorch212AppleArm64LayerNormKernel)
      return format_error(
          "Agro-NT-1B requires esm.layer_norm_kernel='" +
          std::string{kGenebEsmTorch212AppleArm64LayerNormKernel} + "'");
    topology.layer_norm_kernel =
        GenebEsmLayerNormKernel::kTorch212AppleArm64ExactV1;
  } else if (layer_norm_kernel != nullptr) {
    return format_error(
        "esm.layer_norm_kernel is only valid for geneb-agro-nt-1b");
  }

  bool unused_position = false;
  status = metadata_bool(artifact, "esm.source_unused_position_embeddings",
                         &unused_position);
  if (!status.ok())
    return status;
  if (unused_position !=
      (topology.position_type == GenebEsmPositionType::kRotary))
    return format_error(
        "unused position-embedding policy disagrees with position type");

  status = validate_geneb_esm_topology(topology);
  if (!status.ok())
    return format_error(status.message());

  const auto match_common = [&](const std::string_view key,
                                const std::size_t expected) {
    std::size_t actual = 0;
    auto match_status = metadata_size(artifact, key, &actual);
    if (!match_status.ok())
      return match_status;
    if (actual != expected)
      return format_error("common metadata disagrees with esm topology: " +
                          std::string{key});
    return Status::Ok();
  };
  status = match_common("config.vocab_size", topology.vocabulary_size);
  if (!status.ok())
    return status;
  status = match_common("config.hidden_size", topology.width);
  if (!status.ok())
    return status;
  status = match_common("config.num_layers", topology.layers);
  if (!status.ok())
    return status;
  status = match_common("config.max_seqlen", topology.maximum_sequence_length);
  if (!status.ok())
    return status;
  if (topology.layers == std::numeric_limits<std::size_t>::max())
    return format_error("layer count cannot be represented with embedding tap");
  status = match_common("runtime.embedding_layer_count", topology.layers + 1U);
  if (!status.ok())
    return status;
  *output = topology;
  return Status::Ok();
}

Status canonical_geneb_esm_tensors(
    const GenebEsmTopology &topology,
    std::vector<GenebEsmTensorRequirement> *const output) {
  if (output == nullptr)
    return invalid("tensor requirement output is null");
  auto status = validate_geneb_esm_topology(topology);
  if (!status.ok())
    return status;
  std::vector<GenebEsmTensorRequirement> result;
  result.reserve(3U + topology.layers * 16U);
  add_requirement(&result, "esm.embeddings.word_embeddings.weight",
                  {topology.vocabulary_size, topology.width});
  if (topology.position_type == GenebEsmPositionType::kAbsolute)
    add_requirement(&result, "esm.embeddings.position_embeddings.weight",
                    {topology.position_embedding_count, topology.width});
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    const std::string prefix = layer_prefix(layer);
    add_requirement(&result, prefix + "attention.LayerNorm.weight",
                    {topology.width});
    add_requirement(&result, prefix + "attention.LayerNorm.bias",
                    {topology.width});
    for (const std::string_view projection : {"query", "key", "value"}) {
      add_requirement(&result,
                      prefix + "attention.self." + std::string{projection} +
                          ".weight",
                      {topology.width, topology.width});
      if (topology.attention_bias)
        add_requirement(&result,
                        prefix + "attention.self." + std::string{projection} +
                            ".bias",
                        {topology.width});
    }
    add_requirement(&result, prefix + "attention.output.dense.weight",
                    {topology.width, topology.width});
    if (topology.attention_bias)
      add_requirement(&result, prefix + "attention.output.dense.bias",
                      {topology.width});
    add_requirement(&result, prefix + "LayerNorm.weight", {topology.width});
    add_requirement(&result, prefix + "LayerNorm.bias", {topology.width});
    const std::size_t fused_inner =
        topology.mlp_activation == GenebEsmMlpActivation::kSwiGlu
            ? topology.inner_width * 2U
            : topology.inner_width;
    add_requirement(&result, prefix + "intermediate.dense.weight",
                    {fused_inner, topology.width});
    if (topology.feed_forward_bias)
      add_requirement(&result, prefix + "intermediate.dense.bias",
                      {fused_inner});
    add_requirement(&result, prefix + "output.dense.weight",
                    {topology.width, topology.inner_width});
    if (topology.feed_forward_bias)
      add_requirement(&result, prefix + "output.dense.bias", {topology.width});
  }
  add_requirement(&result, "esm.encoder.emb_layer_norm_after.weight",
                  {topology.width});
  add_requirement(&result, "esm.encoder.emb_layer_norm_after.bias",
                  {topology.width});
  *output = std::move(result);
  return Status::Ok();
}

Status geneb_esm_layer_norm(const std::vector<float> &input,
                            const std::size_t rows, const std::size_t width,
                            const GenebEsmTensorView &scale,
                            const GenebEsmTensorView &bias, const float epsilon,
                            std::vector<float> *const output,
                            const GenebEsmLayerNormKernel kernel) {
  std::size_t elements = 0;
  if (output == nullptr || rows == 0 || width == 0 ||
      !checked_mul(rows, width, &elements) || input.size() != elements ||
      scale.data == nullptr || bias.data == nullptr ||
      scale.dtype != TensorDType::kF32 || bias.dtype != TensorDType::kF32 ||
      scale.shape != std::vector<std::size_t>{width} ||
      bias.shape != std::vector<std::size_t>{width} ||
      scale.bytes != width * sizeof(float) ||
      bias.bytes != width * sizeof(float) || !std::isfinite(epsilon) ||
      epsilon <= 0.0F)
    return invalid("LayerNorm shape/tensor/epsilon contract differs");
  if (kernel != GenebEsmLayerNormKernel::kPortableTwoPass &&
      kernel != GenebEsmLayerNormKernel::kTorch212AppleArm64ExactV1)
    return invalid("LayerNorm kernel is unsupported");
  if (kernel == GenebEsmLayerNormKernel::kTorch212AppleArm64ExactV1 &&
      (width != 1500U || epsilon != 1.0e-12F))
    return invalid(
        "exact LayerNorm kernel requires width=1500 and epsilon=1e-12");
  if (kernel == GenebEsmLayerNormKernel::kTorch212AppleArm64ExactV1 &&
      !torch212_apple_arm64_layer_norm_supported())
    return {ErrorCode::kUnsupported,
            "esm.layer_norm_kernel=torch-2.1.2-apple-arm64-exact-v1 "
            "requires Apple arm64"};
  output->resize(elements);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t base = row * width;
    if (kernel == GenebEsmLayerNormKernel::kPortableTwoPass) {
      float sum = 0.0F;
      for (std::size_t column = 0; column < width; ++column)
        sum += input[base + column];
      const float mean = sum / static_cast<float>(width);
      float sum_squares = 0.0F;
      for (std::size_t column = 0; column < width; ++column) {
        const float centered = input[base + column] - mean;
        sum_squares += centered * centered;
      }
      const float inverse =
          1.0F / std::sqrt(sum_squares / static_cast<float>(width) + epsilon);
      for (std::size_t column = 0; column < width; ++column) {
        (*output)[base + column] = (input[base + column] - mean) * inverse *
                                       tensor_value(scale, column) +
                                   tensor_value(bias, column);
      }
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
      const float32x4_t normalized = vaddq_f32(
          vmulq_f32(vld1q_f32(input.data() + base + column), inverse_vector),
          normalization_bias_vector);
      const float32x4_t affine = vaddq_f32(
          vmulq_f32(
              normalized,
              vld1q_f32(reinterpret_cast<const float *>(scale.data) + column)),
          vld1q_f32(reinterpret_cast<const float *>(bias.data) + column));
      vst1q_f32(output->data() + base + column, affine);
    }
    for (; column < width; ++column) {
      const float normalized =
          input[base + column] * inverse + normalization_bias;
      (*output)[base + column] =
          normalized * tensor_value(scale, column) + tensor_value(bias, column);
    }
#endif
  }
  return finite_vector(*output)
             ? Status::Ok()
             : invalid("LayerNorm produced non-finite output");
}

Status geneb_esm_apply_rotary(std::vector<float> *const query,
                              std::vector<float> *const key,
                              const std::size_t rows, const std::size_t heads,
                              const std::size_t head_dimension,
                              const float rope_base) {
  std::size_t width = 0;
  std::size_t elements = 0;
  if (query == nullptr || key == nullptr || rows == 0 || heads == 0 ||
      head_dimension == 0 || head_dimension % 2U != 0U ||
      !std::isfinite(rope_base) || rope_base <= 1.0F ||
      !checked_mul(heads, head_dimension, &width) ||
      !checked_mul(rows, width, &elements) || query->size() != elements ||
      key->size() != elements)
    return invalid("RoPE shape/scalar contract differs");
  const std::size_t half = head_dimension / 2U;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t pair = 0; pair < half; ++pair) {
      const float exponent =
          static_cast<float>(pair * 2U) / static_cast<float>(head_dimension);
      const float inverse_frequency = 1.0F / std::pow(rope_base, exponent);
      const float angle = static_cast<float>(row) * inverse_frequency;
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t base = (row * heads + head) * head_dimension;
        const float query_first = (*query)[base + pair];
        const float query_second = (*query)[base + pair + half];
        const float key_first = (*key)[base + pair];
        const float key_second = (*key)[base + pair + half];
        (*query)[base + pair] = query_first * cosine - query_second * sine;
        (*query)[base + pair + half] =
            query_second * cosine + query_first * sine;
        (*key)[base + pair] = key_first * cosine - key_second * sine;
        (*key)[base + pair + half] = key_second * cosine + key_first * sine;
      }
    }
  }
  return finite_vector(*query) && finite_vector(*key)
             ? Status::Ok()
             : invalid("RoPE produced non-finite output");
}

Status geneb_esm_bidirectional_attention(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value,
    const std::vector<std::uint8_t> &attention_mask, const std::size_t rows,
    const std::size_t heads, const std::size_t head_dimension,
    std::vector<float> *const output) {
  std::size_t width = 0;
  std::size_t elements = 0;
  if (output == nullptr || rows == 0 || heads == 0 || head_dimension == 0 ||
      !checked_mul(heads, head_dimension, &width) ||
      !checked_mul(rows, width, &elements) || query.size() != elements ||
      key.size() != elements || value.size() != elements ||
      attention_mask.size() != rows)
    return invalid("attention shape contract differs");
  std::size_t valid = 0;
  for (const auto mask : attention_mask) {
    if (mask > 1U)
      return invalid("attention mask must be binary");
    valid += mask;
  }
  if (valid == 0)
    return invalid("attention mask has no visible key");

  output->assign(elements, 0.0F);
  std::vector<float> scores(valid, 0.0F);
  std::vector<float> probabilities(valid, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t query_base = (row * heads + head) * head_dimension;
      std::size_t valid_index = 0;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t source = 0; source < rows; ++source) {
        if (attention_mask[source] == 0U)
          continue;
        const std::size_t key_base = (source * heads + head) * head_dimension;
        float score = 0.0F;
        for (std::size_t dimension = 0; dimension < head_dimension; ++dimension)
          score += query[query_base + dimension] * key[key_base + dimension];
        scores[valid_index++] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (std::size_t index = 0; index < valid; ++index) {
        probabilities[index] = std::exp(scores[index] - maximum);
        denominator += probabilities[index];
      }
      if (!std::isfinite(denominator) || denominator <= 0.0F)
        return invalid("attention softmax denominator is invalid");
      for (float &probability : probabilities)
        probability /= denominator;
      for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        float total = 0.0F;
        valid_index = 0;
        for (std::size_t source = 0; source < rows; ++source) {
          if (attention_mask[source] == 0U)
            continue;
          total += probabilities[valid_index++] *
                   value[(source * heads + head) * head_dimension + dimension];
        }
        (*output)[query_base + dimension] = total;
      }
    }
  }
  return finite_vector(*output)
             ? Status::Ok()
             : invalid("attention produced non-finite output");
}

struct GenebEsmModel::Impl final {
  struct Layer final {
    GenebEsmTensorView attention_norm_weight;
    GenebEsmTensorView attention_norm_bias;
    GenebEsmTensorView query_weight;
    GenebEsmTensorView query_bias;
    GenebEsmTensorView key_weight;
    GenebEsmTensorView key_bias;
    GenebEsmTensorView value_weight;
    GenebEsmTensorView value_bias;
    GenebEsmTensorView attention_output_weight;
    GenebEsmTensorView attention_output_bias;
    GenebEsmTensorView feed_forward_norm_weight;
    GenebEsmTensorView feed_forward_norm_bias;
    GenebEsmTensorView intermediate_weight;
    GenebEsmTensorView intermediate_bias;
    GenebEsmTensorView output_weight;
    GenebEsmTensorView output_bias;
  };

  GenebEsmTopology topology;
  GenebEsmTensorView word_embedding;
  GenebEsmTensorView position_embedding;
  std::vector<Layer> layers;
  GenebEsmTensorView final_norm_weight;
  GenebEsmTensorView final_norm_bias;
  std::shared_ptr<evo::detail::LinearExecutor> executor;
};

GenebEsmModel::GenebEsmModel() = default;
GenebEsmModel::~GenebEsmModel() = default;
GenebEsmModel::GenebEsmModel(GenebEsmModel &&) noexcept = default;
GenebEsmModel &GenebEsmModel::operator=(GenebEsmModel &&) noexcept = default;

Status GenebEsmModel::load(
    const GenebEsmTopology &topology,
    const std::vector<GenebEsmNamedTensorView> &tensors,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  std::vector<GenebEsmTensorRequirement> requirements;
  auto status = canonical_geneb_esm_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  std::map<std::string, const GenebEsmTensorView *, std::less<>> provided;
  for (const auto &named : tensors) {
    if (named.name.empty() ||
        !provided.emplace(named.name, &named.tensor).second)
      return format_error("tensor names must be nonempty and unique");
  }
  if (provided.size() != requirements.size())
    return format_error("tensor set count differs from exact ESM manifest");
  for (const auto &requirement : requirements) {
    const auto found = provided.find(requirement.name);
    if (found == provided.end())
      return format_error("required tensor is missing: " + requirement.name);
    status = validate_tensor(*found->second, requirement);
    if (!status.ok())
      return status;
  }

  const auto view = [&](const std::string &name) {
    return *provided.find(name)->second;
  };
  auto candidate = std::make_unique<Impl>();
  candidate->topology = topology;
  candidate->executor = std::move(linear_executor);
  candidate->word_embedding = view("esm.embeddings.word_embeddings.weight");
  if (topology.position_type == GenebEsmPositionType::kAbsolute)
    candidate->position_embedding =
        view("esm.embeddings.position_embeddings.weight");
  candidate->layers.resize(topology.layers);
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    const std::string prefix = layer_prefix(layer);
    auto &target = candidate->layers[layer];
    target.attention_norm_weight = view(prefix + "attention.LayerNorm.weight");
    target.attention_norm_bias = view(prefix + "attention.LayerNorm.bias");
    target.query_weight = view(prefix + "attention.self.query.weight");
    target.key_weight = view(prefix + "attention.self.key.weight");
    target.value_weight = view(prefix + "attention.self.value.weight");
    target.attention_output_weight =
        view(prefix + "attention.output.dense.weight");
    if (topology.attention_bias) {
      target.query_bias = view(prefix + "attention.self.query.bias");
      target.key_bias = view(prefix + "attention.self.key.bias");
      target.value_bias = view(prefix + "attention.self.value.bias");
      target.attention_output_bias =
          view(prefix + "attention.output.dense.bias");
    }
    target.feed_forward_norm_weight = view(prefix + "LayerNorm.weight");
    target.feed_forward_norm_bias = view(prefix + "LayerNorm.bias");
    target.intermediate_weight = view(prefix + "intermediate.dense.weight");
    target.output_weight = view(prefix + "output.dense.weight");
    if (topology.feed_forward_bias) {
      target.intermediate_bias = view(prefix + "intermediate.dense.bias");
      target.output_bias = view(prefix + "output.dense.bias");
    }
  }
  candidate->final_norm_weight =
      view("esm.encoder.emb_layer_norm_after.weight");
  candidate->final_norm_bias = view("esm.encoder.emb_layer_norm_after.bias");
  impl_ = std::move(candidate);
  return Status::Ok();
}

Status GenebEsmModel::load(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  GenebEsmTopology topology;
  auto status = geneb_esm_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebEsmNamedTensorView> views;
  views.reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    if (tensor.rank == 0 ||
        tensor.data_size > std::numeric_limits<std::size_t>::max())
      return format_error("tensor rank/size is invalid: " + tensor.name);
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
      if (tensor.dimensions[dimension] == 0 ||
          tensor.dimensions[dimension] >
              std::numeric_limits<std::size_t>::max())
        return format_error("tensor dimension is invalid: " + tensor.name);
      shape.push_back(static_cast<std::size_t>(tensor.dimensions[dimension]));
    }
    const auto *const data = artifact.tensor_data(tensor);
    if (data == nullptr)
      return format_error("tensor payload is unavailable: " + tensor.name);
    views.push_back({tensor.name,
                     {data, static_cast<std::size_t>(tensor.data_size),
                      tensor.dtype, std::move(shape)}});
  }
  status = load(topology, views, std::move(linear_executor));
  if (!status.ok())
    return format_error(status.message());
  return Status::Ok();
}

Status GenebEsmModel::load_artifact(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  return load(artifact, std::move(linear_executor));
}

const GenebEsmTopology *GenebEsmModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebEsmModel::linear_executor_name() const noexcept {
  if (impl_ == nullptr)
    return "unloaded";
  return impl_->executor == nullptr ? "cpu-reference" : impl_->executor->name();
}

Status GenebEsmModel::forward(const std::vector<TokenId> &tokens,
                              const std::vector<std::uint8_t> &attention_mask,
                              const std::vector<std::size_t> &capture_layers,
                              GenebEsmForwardResult *const output) const {
  if (impl_ == nullptr)
    return invalid("model is not loaded");
  if (output == nullptr)
    return invalid("forward output is null");
  const auto &config = impl_->topology;
  if (config.layer_norm_kernel ==
          GenebEsmLayerNormKernel::kTorch212AppleArm64ExactV1 &&
      !torch212_apple_arm64_layer_norm_supported())
    return {ErrorCode::kUnsupported,
            "esm.layer_norm_kernel=torch-2.1.2-apple-arm64-exact-v1 "
            "requires Apple arm64"};
  if (tokens.empty() || tokens.size() > config.maximum_sequence_length ||
      attention_mask.size() != tokens.size())
    return invalid("token/mask length is outside model context");

  bool saw_padding = false;
  std::size_t valid_tokens = 0;
  std::size_t mask_tokens = 0;
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const auto token = static_cast<std::size_t>(tokens[index]);
    if (token >= config.vocabulary_size)
      return invalid("token ID is outside embedding vocabulary");
    const auto mask = attention_mask[index];
    if (mask > 1U || (mask == 1U && saw_padding))
      return invalid("attention mask must be right-padded binary data");
    if (mask == 0U) {
      saw_padding = true;
      if (token != config.pad_token_id)
        return invalid("masked row token must equal pad_token_id");
    } else {
      if (token == config.pad_token_id)
        return invalid("valid row cannot contain pad_token_id");
      ++valid_tokens;
    }
    if (token == config.mask_token_id)
      ++mask_tokens;
  }
  if (valid_tokens == 0)
    return invalid("attention mask must contain a valid token");
  if (config.token_dropout && mask_tokens >= valid_tokens)
    return invalid("token dropout would divide by zero");

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
    const std::size_t token = static_cast<std::size_t>(tokens[row]);
    for (std::size_t column = 0; column < config.width; ++column)
      hidden[row * config.width + column] =
          tensor_value(impl_->word_embedding, token * config.width + column);
  }

  if (config.token_dropout) {
    for (std::size_t row = 0; row < tokens.size(); ++row) {
      if (static_cast<std::size_t>(tokens[row]) == config.mask_token_id)
        std::fill_n(hidden.begin() +
                        static_cast<std::ptrdiff_t>(row * config.width),
                    config.width, 0.0F);
    }
    const float observed =
        static_cast<float>(mask_tokens) / static_cast<float>(valid_tokens);
    const float scale = (1.0F - kTokenDropoutTrainingRatio) / (1.0F - observed);
    for (float &value : hidden)
      value *= scale;
  }

  if (config.position_type == GenebEsmPositionType::kAbsolute) {
    std::size_t non_padding = 0;
    for (std::size_t row = 0; row < tokens.size(); ++row) {
      std::size_t position = config.pad_token_id;
      if (attention_mask[row] != 0U)
        position = config.pad_token_id + (++non_padding);
      if (position >= config.position_embedding_count)
        return invalid("absolute position ID exceeds embedding table");
      for (std::size_t column = 0; column < config.width; ++column)
        hidden[row * config.width + column] += tensor_value(
            impl_->position_embedding, position * config.width + column);
    }
  }
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    if (attention_mask[row] == 0U)
      std::fill_n(hidden.begin() +
                      static_cast<std::ptrdiff_t>(row * config.width),
                  config.width, 0.0F);
  }
  if (!finite_vector(hidden))
    return invalid("embedding stage produced non-finite hidden state");

  GenebEsmForwardResult result;
  result.rows = tokens.size();
  result.width = config.width;
  result.captures.resize(capture_layers.size());
  capture_hidden(0, hidden, capture_indices, &result.captures);

  const float query_scale =
      1.0F / std::sqrt(static_cast<float>(config.head_dimension));
  for (std::size_t layer_index = 0; layer_index < config.layers;
       ++layer_index) {
    const auto &layer = impl_->layers[layer_index];
    std::vector<float> normalized;
    auto status = geneb_esm_layer_norm(
        hidden, tokens.size(), config.width, layer.attention_norm_weight,
        layer.attention_norm_bias, config.layer_norm_epsilon, &normalized,
        config.layer_norm_kernel);
    if (!status.ok())
      return status;
    const GenebEsmTensorView *const query_bias =
        config.attention_bias ? &layer.query_bias : nullptr;
    const GenebEsmTensorView *const key_bias =
        config.attention_bias ? &layer.key_bias : nullptr;
    const GenebEsmTensorView *const value_bias =
        config.attention_bias ? &layer.value_bias : nullptr;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    status = runtime_linear(normalized, tokens.size(), config.width,
                            layer.query_weight, config.width, query_bias,
                            impl_->executor.get(), &query);
    if (!status.ok())
      return status;
    status = runtime_linear(normalized, tokens.size(), config.width,
                            layer.key_weight, config.width, key_bias,
                            impl_->executor.get(), &key);
    if (!status.ok())
      return status;
    status = runtime_linear(normalized, tokens.size(), config.width,
                            layer.value_weight, config.width, value_bias,
                            impl_->executor.get(), &value);
    if (!status.ok())
      return status;
    for (float &item : query)
      item *= query_scale;
    if (config.position_type == GenebEsmPositionType::kRotary) {
      status = geneb_esm_apply_rotary(&query, &key, tokens.size(), config.heads,
                                      config.head_dimension, config.rope_base);
      if (!status.ok())
        return status;
    }
    std::vector<float> attended;
    status = geneb_esm_bidirectional_attention(
        query, key, value, attention_mask, tokens.size(), config.heads,
        config.head_dimension, &attended);
    if (!status.ok())
      return status;
    const GenebEsmTensorView *const attention_output_bias =
        config.attention_bias ? &layer.attention_output_bias : nullptr;
    std::vector<float> projected;
    status = runtime_linear(
        attended, tokens.size(), config.width, layer.attention_output_weight,
        config.width, attention_output_bias, impl_->executor.get(), &projected);
    if (!status.ok())
      return status;
    add_in_place(&hidden, projected);

    status = geneb_esm_layer_norm(
        hidden, tokens.size(), config.width, layer.feed_forward_norm_weight,
        layer.feed_forward_norm_bias, config.layer_norm_epsilon, &normalized,
        config.layer_norm_kernel);
    if (!status.ok())
      return status;
    const std::size_t fused_inner =
        config.mlp_activation == GenebEsmMlpActivation::kSwiGlu
            ? config.inner_width * 2U
            : config.inner_width;
    const GenebEsmTensorView *const intermediate_bias =
        config.feed_forward_bias ? &layer.intermediate_bias : nullptr;
    std::vector<float> intermediate;
    status = runtime_linear(
        normalized, tokens.size(), config.width, layer.intermediate_weight,
        fused_inner, intermediate_bias, impl_->executor.get(), &intermediate);
    if (!status.ok())
      return status;
    std::vector<float> activated(tokens.size() * config.inner_width, 0.0F);
    if (config.mlp_activation == GenebEsmMlpActivation::kGelu) {
      for (std::size_t index = 0; index < activated.size(); ++index)
        activated[index] = exact_gelu(intermediate[index]);
    } else {
      for (std::size_t row = 0; row < tokens.size(); ++row) {
        const std::size_t source_base = row * fused_inner;
        const std::size_t target_base = row * config.inner_width;
        for (std::size_t column = 0; column < config.inner_width; ++column) {
          const float first = intermediate[source_base + column];
          const float second =
              intermediate[source_base + config.inner_width + column];
          activated[target_base + column] = silu(first) * second;
        }
      }
    }
    const GenebEsmTensorView *const output_bias =
        config.feed_forward_bias ? &layer.output_bias : nullptr;
    std::vector<float> feed_forward;
    status = runtime_linear(activated, tokens.size(), config.inner_width,
                            layer.output_weight, config.width, output_bias,
                            impl_->executor.get(), &feed_forward);
    if (!status.ok())
      return status;
    add_in_place(&hidden, feed_forward);
    if (!finite_vector(hidden))
      return invalid("encoder block produced non-finite hidden state");
    if (layer_index + 1U < config.layers)
      capture_hidden(layer_index + 1U, hidden, capture_indices,
                     &result.captures);
  }

  auto status = geneb_esm_layer_norm(
      hidden, tokens.size(), config.width, impl_->final_norm_weight,
      impl_->final_norm_bias, config.layer_norm_epsilon, &result.final_hidden,
      config.layer_norm_kernel);
  if (!status.ok())
    return status;
  capture_hidden(config.layers, result.final_hidden, capture_indices,
                 &result.captures);
  result.pooled.assign(config.width, 0.0F);
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    if (attention_mask[row] == 0U)
      continue;
    for (std::size_t column = 0; column < config.width; ++column)
      result.pooled[column] += result.final_hidden[row * config.width + column];
  }
  for (float &value : result.pooled)
    value /= static_cast<float>(valid_tokens);
  *output = std::move(result);
  return Status::Ok();
}

} // namespace evo::cpu
