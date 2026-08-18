// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_custom_encoder.hpp"

#include "evo/cpu/geneb_bert.hpp"

#include "../linear_executor.hpp"

#include <algorithm>
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

namespace evo::cpu {
namespace {

constexpr std::size_t kMaximumWidth = 65536U;
constexpr std::size_t kMaximumLayers = 1024U;
constexpr std::size_t kMaximumSequenceLength = 1U << 20U;
constexpr std::size_t kMaximumInnerWidth = 1U << 20U;

Status invalid(const std::string &message) {
  return {ErrorCode::kInvalidArgument, "GENEB custom encoder: " + message};
}

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "GENEB custom encoder artifact: " + message};
}

bool checked_mul(const std::size_t left, const std::size_t right,
                 std::size_t *const output) noexcept {
  if (output == nullptr ||
      (left != 0U && right > std::numeric_limits<std::size_t>::max() / left))
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
  std::uint64_t value = 0U;
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
  std::uint64_t bits = 0U;
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
  if (entry->value.size() != 1U || entry->value.front() > 1U)
    return format_error("bool metadata is malformed: " + std::string{key});
  *output = entry->value.front() != 0U;
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

float tensor_value(const GenebCustomEncoderTensorView &tensor,
                   const std::size_t index) noexcept {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return value;
}

Status validate_tensor(const GenebCustomEncoderTensorView &tensor,
                       const GenebCustomEncoderTensorRequirement &requirement) {
  if (tensor.data == nullptr)
    return format_error("tensor '" + requirement.name + "' has null data");
  if (tensor.dtype != requirement.dtype)
    return format_error("tensor '" + requirement.name + "' has wrong dtype");
  if (tensor.shape != requirement.shape)
    return format_error("tensor '" + requirement.name + "' has wrong shape");
  std::size_t elements = 1U;
  for (const std::size_t dimension : tensor.shape) {
    if (dimension == 0U || !checked_mul(elements, dimension, &elements))
      return format_error("tensor '" + requirement.name +
                          "' has an overflowing shape");
  }
  std::size_t expected_bytes = 0U;
  if (!checked_mul(elements, sizeof(float), &expected_bytes) ||
      tensor.bytes != expected_bytes)
    return format_error("tensor '" + requirement.name +
                        "' has wrong byte size");
  return Status::Ok();
}

void add_requirement(
    std::vector<GenebCustomEncoderTensorRequirement> *const output,
    std::string name, std::vector<std::size_t> shape) {
  output->push_back({std::move(name), TensorDType::kF32, std::move(shape)});
}

std::string luca_layer_prefix(const std::size_t layer) {
  return "lucaone.encoder.layers." + std::to_string(layer) + ".";
}

GenebBertTopology
genomics_bert_topology(const GenebCustomEncoderTopology &topology) {
  GenebBertTopology result;
  result.vocabulary_size = topology.vocabulary_size;
  result.width = topology.width;
  result.layers = topology.layers;
  result.attention_heads = topology.attention_heads;
  result.head_dimension = topology.head_dimension;
  result.inner_width = topology.inner_width;
  result.maximum_sequence_length = topology.maximum_sequence_length;
  result.token_type_vocabulary_size = topology.token_type_vocabulary_size;
  result.layer_norm_epsilon = topology.layer_norm_epsilon;
  result.rope_base = 0.0F;
  result.position_encoding = GenebBertPositionEncoding::kAbsolute;
  result.norm_placement = GenebBertNormPlacement::kPost;
  result.mlp_kind = GenebBertMlpKind::kGatedGelu;
  result.qkv_layout = GenebBertQkvLayout::kFused;
  result.input_kind = GenebBertInputKind::kTokenIds;
  result.pooling = GenebBertPooling::kClsToken;
  result.final_layer_norm = false;
  result.unpad_masked_tokens = true;
  result.attention_bias = true;
  result.mlp_input_bias = false;
  result.mlp_output_bias = true;
  result.embedding_dtype = TensorDType::kF32;
  result.projection_dtype = TensorDType::kF32;
  result.norm_dtype = TensorDType::kF32;
  result.activation_dtype = TensorDType::kF32;
  return result;
}

Status layer_norm(const std::vector<float> &input, const std::size_t rows,
                  const std::size_t width,
                  const GenebCustomEncoderTensorView &scale,
                  const GenebCustomEncoderTensorView &bias, const float epsilon,
                  std::vector<float> *const output) {
  std::size_t elements = 0U;
  if (output == nullptr || rows == 0U || width == 0U ||
      !checked_mul(rows, width, &elements) || input.size() != elements ||
      scale.data == nullptr || bias.data == nullptr ||
      scale.dtype != TensorDType::kF32 || bias.dtype != TensorDType::kF32 ||
      scale.shape != std::vector<std::size_t>{width} ||
      bias.shape != std::vector<std::size_t>{width} ||
      scale.bytes != width * sizeof(float) ||
      bias.bytes != width * sizeof(float) || !std::isfinite(epsilon) ||
      epsilon <= 0.0F)
    return invalid("LayerNorm shape/tensor/epsilon contract differs");
  output->resize(elements);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t base = row * width;
    float sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column)
      sum += input[base + column];
    const float mean = sum / static_cast<float>(width);
    float square_sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      const float centered = input[base + column] - mean;
      square_sum += centered * centered;
    }
    const float inverse =
        1.0F / std::sqrt(square_sum / static_cast<float>(width) + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      (*output)[base + column] = (input[base + column] - mean) * inverse *
                                     tensor_value(scale, column) +
                                 tensor_value(bias, column);
    }
  }
  return finite_vector(*output) ? Status::Ok()
                                : invalid("LayerNorm produced non-finite data");
}

Status runtime_linear(const std::vector<float> &input, const std::size_t rows,
                      const std::size_t input_width,
                      const GenebCustomEncoderTensorView &weight,
                      const std::size_t output_width,
                      const GenebCustomEncoderTensorView *const bias,
                      evo::detail::LinearExecutor *const executor,
                      std::vector<float> *const output) {
  std::size_t input_elements = 0U;
  std::size_t weight_elements = 0U;
  std::size_t output_elements = 0U;
  if (output == nullptr || rows == 0U || input_width == 0U ||
      output_width == 0U || !checked_mul(rows, input_width, &input_elements) ||
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
      const std::size_t input_base = row * input_width;
      const std::size_t weight_base = target * input_width;
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

Status apply_luca_rotary(std::vector<float> *const query,
                         std::vector<float> *const key,
                         const GenebCustomEncoderTensorView &inverse_frequency,
                         const std::size_t rows, const std::size_t heads,
                         const std::size_t head_dimension) {
  std::size_t width = 0U;
  std::size_t elements = 0U;
  const std::size_t half = head_dimension / 2U;
  if (query == nullptr || key == nullptr || rows == 0U || heads == 0U ||
      head_dimension == 0U || head_dimension % 2U != 0U ||
      !checked_mul(heads, head_dimension, &width) ||
      !checked_mul(rows, width, &elements) || query->size() != elements ||
      key->size() != elements || inverse_frequency.data == nullptr ||
      inverse_frequency.dtype != TensorDType::kF32 ||
      inverse_frequency.shape != std::vector<std::size_t>{half} ||
      inverse_frequency.bytes != half * sizeof(float))
    return invalid("LucaOne RoPE contract differs");
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t pair = 0; pair < half; ++pair) {
      const float angle =
          static_cast<float>(row) * tensor_value(inverse_frequency, pair);
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t base = (row * heads + head) * head_dimension;
        const float q_first = (*query)[base + pair];
        const float q_second = (*query)[base + pair + half];
        const float k_first = (*key)[base + pair];
        const float k_second = (*key)[base + pair + half];
        (*query)[base + pair] = q_first * cosine - q_second * sine;
        (*query)[base + pair + half] = q_second * cosine + q_first * sine;
        (*key)[base + pair] = k_first * cosine - k_second * sine;
        (*key)[base + pair + half] = k_second * cosine + k_first * sine;
      }
    }
  }
  return finite_vector(*query) && finite_vector(*key)
             ? Status::Ok()
             : invalid("LucaOne RoPE produced non-finite data");
}

Status bidirectional_attention(const std::vector<float> &query,
                               const std::vector<float> &key,
                               const std::vector<float> &value,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::size_t rows, const std::size_t heads,
                               const std::size_t head_dimension,
                               std::vector<float> *const output) {
  std::size_t width = 0U;
  std::size_t elements = 0U;
  if (output == nullptr || rows == 0U || heads == 0U || head_dimension == 0U ||
      !checked_mul(heads, head_dimension, &width) ||
      !checked_mul(rows, width, &elements) || query.size() != elements ||
      key.size() != elements || value.size() != elements ||
      attention_mask.size() != rows)
    return invalid("attention shape contract differs");
  std::size_t visible = 0U;
  for (const std::uint8_t mask : attention_mask) {
    if (mask > 1U)
      return invalid("attention mask must be binary");
    visible += mask;
  }
  if (visible == 0U)
    return invalid("attention mask has no visible key");
  output->assign(elements, 0.0F);
  std::vector<float> scores(visible, 0.0F);
  std::vector<float> probabilities(visible, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t query_base = (row * heads + head) * head_dimension;
      float maximum = -std::numeric_limits<float>::infinity();
      std::size_t visible_index = 0U;
      for (std::size_t source = 0; source < rows; ++source) {
        if (attention_mask[source] == 0U)
          continue;
        const std::size_t key_base = (source * heads + head) * head_dimension;
        float score = 0.0F;
        for (std::size_t dimension = 0; dimension < head_dimension; ++dimension)
          score += query[query_base + dimension] * key[key_base + dimension];
        scores[visible_index++] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (std::size_t index = 0; index < visible; ++index) {
        probabilities[index] = std::exp(scores[index] - maximum);
        denominator += probabilities[index];
      }
      if (!std::isfinite(denominator) || denominator <= 0.0F)
        return invalid("attention softmax denominator is invalid");
      for (float &probability : probabilities)
        probability /= denominator;
      for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        float total = 0.0F;
        visible_index = 0U;
        for (std::size_t source = 0; source < rows; ++source) {
          if (attention_mask[source] == 0U)
            continue;
          total += probabilities[visible_index++] *
                   value[(source * heads + head) * head_dimension + dimension];
        }
        (*output)[query_base + dimension] = total;
      }
    }
  }
  return finite_vector(*output) ? Status::Ok()
                                : invalid("attention produced non-finite data");
}

float exact_gelu(const float value) noexcept {
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  return value * 0.5F * (1.0F + std::erf(value * kInverseSqrtTwo));
}

void add_in_place(std::vector<float> *const target,
                  const std::vector<float> &increment) noexcept {
  for (std::size_t index = 0; index < target->size(); ++index)
    (*target)[index] += increment[index];
}

void capture_hidden(
    const std::size_t layer, const std::vector<float> &values,
    const std::map<std::size_t, std::size_t> &capture_indices,
    std::vector<GenebCustomEncoderHiddenCapture> *const captures) {
  const auto found = capture_indices.find(layer);
  if (found != capture_indices.end())
    (*captures)[found->second] = {layer, values};
}

} // namespace

Status validate_geneb_custom_encoder_topology(
    const GenebCustomEncoderTopology &topology) {
  if ((topology.variant != GenebCustomEncoderVariant::kLucaOne &&
       topology.variant != GenebCustomEncoderVariant::kGenomicsFm) ||
      topology.vocabulary_size == 0U ||
      !token_vocabulary_size_supported(topology.vocabulary_size) ||
      topology.tokenizer_vocabulary_size == 0U ||
      !token_vocabulary_size_supported(topology.tokenizer_vocabulary_size) ||
      topology.tokenizer_vocabulary_size > topology.vocabulary_size ||
      topology.width == 0U || topology.width > kMaximumWidth ||
      topology.layers == 0U || topology.layers > kMaximumLayers ||
      topology.attention_heads == 0U || topology.head_dimension == 0U ||
      topology.inner_width == 0U || topology.inner_width > kMaximumInnerWidth ||
      topology.maximum_sequence_length == 0U ||
      topology.maximum_sequence_length > kMaximumSequenceLength ||
      topology.token_type_vocabulary_size == 0U ||
      !std::isfinite(topology.layer_norm_epsilon) ||
      topology.layer_norm_epsilon <= 0.0F ||
      topology.pad_token_id >= topology.tokenizer_vocabulary_size ||
      topology.cls_token_id >= topology.tokenizer_vocabulary_size ||
      topology.sep_token_id >= topology.tokenizer_vocabulary_size ||
      topology.pad_token_id == topology.cls_token_id ||
      topology.pad_token_id == topology.sep_token_id ||
      topology.cls_token_id == topology.sep_token_id ||
      topology.weight_dtype != TensorDType::kF32)
    return invalid(
        "topology is outside geneb-custom-encoder-runtime-v1 limits");
  std::size_t attention_width = 0U;
  std::size_t ignored = 0U;
  if (!checked_mul(topology.attention_heads, topology.head_dimension,
                   &attention_width) ||
      attention_width != topology.width ||
      !checked_mul(topology.vocabulary_size, topology.width, &ignored) ||
      !checked_mul(topology.inner_width, topology.width, &ignored))
    return invalid(
        "topology dimensions overflow or attention geometry differs");

  if (topology.variant == GenebCustomEncoderVariant::kLucaOne) {
    const bool exact =
        topology.tokenizer_vocabulary_size == topology.vocabulary_size &&
        topology.token_type_vocabulary_size == 2U &&
        topology.pad_token_id == 0U && topology.cls_token_id == 2U &&
        topology.sep_token_id == 3U &&
        topology.position_encoding ==
            GenebCustomPositionEncoding::kRopeSplitHalf &&
        topology.norm_placement == GenebCustomNormPlacement::kPre &&
        topology.qkv_layout == GenebCustomQkvLayout::kSeparate &&
        topology.mlp_kind == GenebCustomMlpKind::kGelu &&
        topology.pooling == GenebCustomPooling::kAttentionMaskMean &&
        topology.attention_bias && topology.mlp_input_bias &&
        topology.mlp_output_bias && !topology.embedding_layer_norm &&
        topology.final_layer_norm && !topology.unpad_masked_tokens &&
        topology.token_type_embeddings &&
        topology.layer_norm_epsilon == 1.0e-5F &&
        topology.rope_base == 10000.0F && topology.head_dimension % 2U == 0U;
    if (!exact)
      return invalid("LucaOne operation tuple differs from the pinned source");
  } else {
    const bool exact =
        topology.token_type_vocabulary_size == 2U &&
        topology.pad_token_id == 3U && topology.cls_token_id == 1U &&
        topology.sep_token_id == 2U &&
        topology.position_encoding == GenebCustomPositionEncoding::kAbsolute &&
        topology.norm_placement == GenebCustomNormPlacement::kPost &&
        topology.qkv_layout == GenebCustomQkvLayout::kFused &&
        topology.mlp_kind == GenebCustomMlpKind::kGatedGelu &&
        topology.pooling == GenebCustomPooling::kClsToken &&
        topology.attention_bias && !topology.mlp_input_bias &&
        topology.mlp_output_bias && topology.embedding_layer_norm &&
        !topology.final_layer_norm && topology.unpad_masked_tokens &&
        topology.token_type_embeddings &&
        topology.layer_norm_epsilon == 1.0e-12F && topology.rope_base == 0.0F;
    if (!exact)
      return invalid(
          "Genomics-FM operation tuple differs from the pinned source");
    const auto status =
        validate_geneb_bert_topology(genomics_bert_topology(topology));
    if (!status.ok())
      return invalid("Genomics-FM BERT tuple was rejected: " +
                     status.message());
  }
  return Status::Ok();
}

Status geneb_custom_encoder_topology_from_artifact(
    const ModelFile &artifact, GenebCustomEncoderTopology *const output) {
  if (output == nullptr)
    return invalid("topology output is null");
  if (artifact.profile() != kGenebCustomEncoderArtifactProfile)
    return format_error("profile must be '" +
                        std::string{kGenebCustomEncoderArtifactProfile} + "'");
  if (!artifact.tokenizer_asset_descriptor().has_value())
    return format_error("the four-key tokenizer descriptor is required");
  auto status =
      require_literal(artifact, "runtime.abi", kGenebCustomEncoderRuntimeAbi);
  if (!status.ok())
    return status;
  status = require_literal(artifact, "model.architecture",
                           kGenebCustomEncoderArchitecture);
  if (!status.ok())
    return status;

  const std::set<std::string_view> expected{
      "custom.variant",
      "custom.vocab_size",
      "custom.tokenizer_vocab_size",
      "custom.hidden_size",
      "custom.num_layers",
      "custom.num_attention_heads",
      "custom.head_dim",
      "custom.inner_size",
      "custom.max_seqlen",
      "custom.type_vocab_size",
      "custom.pad_token_id",
      "custom.cls_token_id",
      "custom.sep_token_id",
      "custom.layer_norm_epsilon",
      "custom.rope_base",
      "custom.position_encoding",
      "custom.norm_placement",
      "custom.qkv_layout",
      "custom.mlp_kind",
      "custom.pooling",
      "custom.attention_bias",
      "custom.mlp_input_bias",
      "custom.mlp_output_bias",
      "custom.embedding_layer_norm",
      "custom.final_layer_norm",
      "custom.unpad_masked_tokens",
      "custom.token_type_embeddings",
      "custom.weight_dtype",
      "custom.hidden_tap",
      "custom.special_tokens",
      "custom.mask_domain",
      "custom.attention_mask",
      "custom.rope_layout",
      "custom.gelu",
      "custom.tokenizer_kind",
      "custom.official_reference_device",
  };
  for (const auto &entry : artifact.metadata()) {
    if (entry.key.compare(0, 7, "custom.") == 0 &&
        expected.find(std::string_view{entry.key}) == expected.end())
      return format_error("unexpected custom.* metadata: " + entry.key);
  }

  GenebCustomEncoderTopology topology;
  std::string variant;
  status = metadata_string(artifact, "custom.variant", &variant);
  if (!status.ok())
    return status;
  if (variant == "lucaone")
    topology.variant = GenebCustomEncoderVariant::kLucaOne;
  else if (variant == "genomics-fm")
    topology.variant = GenebCustomEncoderVariant::kGenomicsFm;
  else
    return format_error("custom.variant is unsupported");

  for (const auto &[key, target] :
       std::vector<std::pair<std::string_view, std::size_t *>>{
           {"custom.vocab_size", &topology.vocabulary_size},
           {"custom.tokenizer_vocab_size", &topology.tokenizer_vocabulary_size},
           {"custom.hidden_size", &topology.width},
           {"custom.num_layers", &topology.layers},
           {"custom.num_attention_heads", &topology.attention_heads},
           {"custom.head_dim", &topology.head_dimension},
           {"custom.inner_size", &topology.inner_width},
           {"custom.max_seqlen", &topology.maximum_sequence_length},
           {"custom.type_vocab_size", &topology.token_type_vocabulary_size},
           {"custom.pad_token_id", &topology.pad_token_id},
           {"custom.cls_token_id", &topology.cls_token_id},
           {"custom.sep_token_id", &topology.sep_token_id}}) {
    status = metadata_size(artifact, key, target);
    if (!status.ok())
      return status;
  }
  status = metadata_float(artifact, "custom.layer_norm_epsilon",
                          &topology.layer_norm_epsilon);
  if (!status.ok())
    return status;
  status = metadata_float(artifact, "custom.rope_base", &topology.rope_base);
  if (!status.ok())
    return status;
  for (const auto &[key, target] :
       std::vector<std::pair<std::string_view, bool *>>{
           {"custom.attention_bias", &topology.attention_bias},
           {"custom.mlp_input_bias", &topology.mlp_input_bias},
           {"custom.mlp_output_bias", &topology.mlp_output_bias},
           {"custom.embedding_layer_norm", &topology.embedding_layer_norm},
           {"custom.final_layer_norm", &topology.final_layer_norm},
           {"custom.unpad_masked_tokens", &topology.unpad_masked_tokens},
           {"custom.token_type_embeddings", &topology.token_type_embeddings}}) {
    status = metadata_bool(artifact, key, target);
    if (!status.ok())
      return status;
  }
  std::string value;
  status = metadata_string(artifact, "custom.position_encoding", &value);
  if (!status.ok())
    return status;
  if (value == "absolute")
    topology.position_encoding = GenebCustomPositionEncoding::kAbsolute;
  else if (value == "rope-split-half")
    topology.position_encoding = GenebCustomPositionEncoding::kRopeSplitHalf;
  else
    return format_error("custom.position_encoding is unsupported");
  status = metadata_string(artifact, "custom.norm_placement", &value);
  if (!status.ok())
    return status;
  if (value == "pre")
    topology.norm_placement = GenebCustomNormPlacement::kPre;
  else if (value == "post")
    topology.norm_placement = GenebCustomNormPlacement::kPost;
  else
    return format_error("custom.norm_placement is unsupported");
  status = metadata_string(artifact, "custom.qkv_layout", &value);
  if (!status.ok())
    return status;
  if (value == "separate")
    topology.qkv_layout = GenebCustomQkvLayout::kSeparate;
  else if (value == "fused-qkv")
    topology.qkv_layout = GenebCustomQkvLayout::kFused;
  else
    return format_error("custom.qkv_layout is unsupported");
  status = metadata_string(artifact, "custom.mlp_kind", &value);
  if (!status.ok())
    return status;
  if (value == "gelu")
    topology.mlp_kind = GenebCustomMlpKind::kGelu;
  else if (value == "gated-gelu")
    topology.mlp_kind = GenebCustomMlpKind::kGatedGelu;
  else
    return format_error("custom.mlp_kind is unsupported");
  status = metadata_string(artifact, "custom.pooling", &value);
  if (!status.ok())
    return status;
  if (value == "attention-mask-mean")
    topology.pooling = GenebCustomPooling::kAttentionMaskMean;
  else if (value == "cls-token")
    topology.pooling = GenebCustomPooling::kClsToken;
  else
    return format_error("custom.pooling is unsupported");
  status = require_literal(artifact, "custom.weight_dtype", "F32");
  if (!status.ok())
    return status;
  topology.weight_dtype = TensorDType::kF32;

  const bool luca = topology.variant == GenebCustomEncoderVariant::kLucaOne;
  for (const auto &[key, expected_value] :
       std::vector<std::pair<std::string_view, std::string_view>>{
           {"custom.hidden_tap", "last-hidden-state"},
           {"custom.special_tokens", "include-added-boundaries"},
           {"custom.mask_domain", "attention-mask"},
           {"custom.attention_mask", "bidirectional-key-mask"},
           {"custom.rope_layout", luca ? "split-half" : "none"},
           {"custom.gelu", "exact-erf"},
           {"custom.tokenizer_kind", luca ? "character" : "bpe"},
           {"custom.official_reference_device", luca ? "cpu-or-gpu" : "gpu"}}) {
    status = require_literal(artifact, key, expected_value);
    if (!status.ok())
      return status;
  }

  status = validate_geneb_custom_encoder_topology(topology);
  if (!status.ok())
    return format_error(status.message());
  const auto match_common = [&](const std::string_view key,
                                const std::size_t expected_size) {
    std::size_t actual = 0U;
    auto match_status = metadata_size(artifact, key, &actual);
    if (!match_status.ok())
      return match_status;
    if (actual != expected_size)
      return format_error("common metadata disagrees with custom topology: " +
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
  status = match_common("runtime.tokenizer_vocabulary_size",
                        topology.tokenizer_vocabulary_size);
  if (!status.ok())
    return status;
  status = match_common("runtime.embedding_layer_count", topology.layers + 1U);
  if (!status.ok())
    return status;
  *output = topology;
  return Status::Ok();
}

Status canonical_geneb_custom_encoder_tensors(
    const GenebCustomEncoderTopology &topology,
    std::vector<GenebCustomEncoderTensorRequirement> *const output) {
  if (output == nullptr)
    return invalid("tensor requirement output is null");
  auto status = validate_geneb_custom_encoder_topology(topology);
  if (!status.ok())
    return status;
  std::vector<GenebCustomEncoderTensorRequirement> result;
  if (topology.variant == GenebCustomEncoderVariant::kGenomicsFm) {
    std::vector<GenebBertTensorRequirement> bert_requirements;
    status = canonical_geneb_bert_tensors(genomics_bert_topology(topology),
                                          &bert_requirements);
    if (!status.ok())
      return invalid("Genomics-FM BERT manifest failed: " + status.message());
    result.reserve(bert_requirements.size());
    for (auto &requirement : bert_requirements)
      result.push_back({std::move(requirement.name), requirement.dtype,
                        std::move(requirement.shape)});
    *output = std::move(result);
    return Status::Ok();
  }

  result.reserve(4U + topology.layers * 17U);
  add_requirement(&result, "lucaone.embeddings.embed_tokens.weight",
                  {topology.vocabulary_size, topology.width});
  add_requirement(&result, "lucaone.embeddings.embed_type.weight",
                  {topology.token_type_vocabulary_size, topology.width});
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    const std::string prefix = luca_layer_prefix(layer);
    add_requirement(&result, prefix + "pre_layer_norm.weight",
                    {topology.width});
    add_requirement(&result, prefix + "pre_layer_norm.bias", {topology.width});
    for (const std::string_view projection :
         {"q_proj", "k_proj", "v_proj", "out_proj"}) {
      add_requirement(
          &result, prefix + "self_attn." + std::string{projection} + ".weight",
          {topology.width, topology.width});
      add_requirement(&result,
                      prefix + "self_attn." + std::string{projection} + ".bias",
                      {topology.width});
    }
    add_requirement(&result, prefix + "self_attn.rot_emb.inv_freq",
                    {topology.head_dimension / 2U});
    add_requirement(&result, prefix + "post_layer_norm.weight",
                    {topology.width});
    add_requirement(&result, prefix + "post_layer_norm.bias", {topology.width});
    add_requirement(&result, prefix + "fc1.weight",
                    {topology.inner_width, topology.width});
    add_requirement(&result, prefix + "fc1.bias", {topology.inner_width});
    add_requirement(&result, prefix + "fc2.weight",
                    {topology.width, topology.inner_width});
    add_requirement(&result, prefix + "fc2.bias", {topology.width});
  }
  add_requirement(&result, "lucaone.encoder.last_layer_norm.weight",
                  {topology.width});
  add_requirement(&result, "lucaone.encoder.last_layer_norm.bias",
                  {topology.width});
  *output = std::move(result);
  return Status::Ok();
}

Status
geneb_custom_encoder_pool(const GenebCustomEncoderForwardResult &forward,
                          const std::vector<std::uint8_t> &attention_mask,
                          const GenebCustomPooling pooling,
                          std::vector<float> *const output) {
  std::size_t elements = 0U;
  if (output == nullptr || forward.rows == 0U || forward.width == 0U ||
      !checked_mul(forward.rows, forward.width, &elements) ||
      forward.final_hidden.size() != elements ||
      attention_mask.size() != forward.rows)
    return invalid("pool shape contract differs");
  std::size_t visible = 0U;
  bool saw_padding = false;
  for (const std::uint8_t mask : attention_mask) {
    if (mask > 1U || (mask == 1U && saw_padding))
      return invalid("pool attention mask must be right-padded binary data");
    if (mask == 0U)
      saw_padding = true;
    else
      ++visible;
  }
  if (visible == 0U)
    return invalid("pool attention mask has no visible row");
  output->assign(forward.width, 0.0F);
  if (pooling == GenebCustomPooling::kClsToken) {
    if (attention_mask.front() == 0U)
      return invalid("CLS pooling requires active row zero");
    std::copy_n(forward.final_hidden.begin(), forward.width, output->begin());
  } else if (pooling == GenebCustomPooling::kAttentionMaskMean) {
    for (std::size_t row = 0; row < forward.rows; ++row) {
      if (attention_mask[row] == 0U)
        continue;
      for (std::size_t column = 0; column < forward.width; ++column)
        (*output)[column] += forward.final_hidden[row * forward.width + column];
    }
    for (float &value : *output)
      value /= static_cast<float>(visible);
  } else {
    return invalid("pooling mode is unsupported");
  }
  return finite_vector(*output) ? Status::Ok()
                                : invalid("pooling produced non-finite data");
}

struct GenebCustomEncoderModel::Impl final {
  struct LucaLayer final {
    GenebCustomEncoderTensorView pre_norm_weight;
    GenebCustomEncoderTensorView pre_norm_bias;
    GenebCustomEncoderTensorView query_weight;
    GenebCustomEncoderTensorView query_bias;
    GenebCustomEncoderTensorView key_weight;
    GenebCustomEncoderTensorView key_bias;
    GenebCustomEncoderTensorView value_weight;
    GenebCustomEncoderTensorView value_bias;
    GenebCustomEncoderTensorView attention_output_weight;
    GenebCustomEncoderTensorView attention_output_bias;
    GenebCustomEncoderTensorView inverse_frequency;
    GenebCustomEncoderTensorView post_norm_weight;
    GenebCustomEncoderTensorView post_norm_bias;
    GenebCustomEncoderTensorView fc1_weight;
    GenebCustomEncoderTensorView fc1_bias;
    GenebCustomEncoderTensorView fc2_weight;
    GenebCustomEncoderTensorView fc2_bias;
  };

  GenebCustomEncoderTopology topology;
  GenebCustomEncoderTensorView word_embedding;
  GenebCustomEncoderTensorView token_type_embedding;
  std::vector<LucaLayer> luca_layers;
  GenebCustomEncoderTensorView final_norm_weight;
  GenebCustomEncoderTensorView final_norm_bias;
  std::unique_ptr<GenebBertModel> genomics_model;
  std::shared_ptr<evo::detail::LinearExecutor> executor;
};

GenebCustomEncoderModel::GenebCustomEncoderModel() = default;
GenebCustomEncoderModel::~GenebCustomEncoderModel() = default;
GenebCustomEncoderModel::GenebCustomEncoderModel(
    GenebCustomEncoderModel &&) noexcept = default;
GenebCustomEncoderModel &GenebCustomEncoderModel::operator=(
    GenebCustomEncoderModel &&) noexcept = default;

Status GenebCustomEncoderModel::load(
    const GenebCustomEncoderTopology &topology,
    const std::vector<GenebCustomEncoderNamedTensorView> &tensors,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  std::vector<GenebCustomEncoderTensorRequirement> requirements;
  auto status = canonical_geneb_custom_encoder_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  std::map<std::string, const GenebCustomEncoderTensorView *, std::less<>>
      provided;
  for (const auto &named : tensors) {
    if (named.name.empty() ||
        !provided.emplace(named.name, &named.tensor).second)
      return format_error("tensor names must be nonempty and unique");
  }
  if (provided.size() != requirements.size())
    return format_error("tensor set count differs from exact custom manifest");
  for (const auto &requirement : requirements) {
    const auto found = provided.find(requirement.name);
    if (found == provided.end())
      return format_error("required tensor is missing: " + requirement.name);
    status = validate_tensor(*found->second, requirement);
    if (!status.ok())
      return status;
  }

  auto candidate = std::make_unique<Impl>();
  candidate->topology = topology;
  candidate->executor = std::move(linear_executor);
  if (topology.variant == GenebCustomEncoderVariant::kGenomicsFm) {
    std::vector<GenebBertNamedTensorView> bert_views;
    bert_views.reserve(tensors.size());
    for (const auto &named : tensors) {
      bert_views.push_back({named.name,
                            {named.tensor.data, named.tensor.bytes,
                             named.tensor.dtype, named.tensor.shape}});
    }
    candidate->genomics_model = std::make_unique<GenebBertModel>();
    status = candidate->genomics_model->load(genomics_bert_topology(topology),
                                             bert_views, candidate->executor);
    if (!status.ok())
      return format_error("Genomics-FM BERT load failed: " + status.message());
    impl_ = std::move(candidate);
    return Status::Ok();
  }

  const auto view = [&](const std::string &name) {
    return *provided.find(name)->second;
  };
  candidate->word_embedding = view("lucaone.embeddings.embed_tokens.weight");
  candidate->token_type_embedding =
      view("lucaone.embeddings.embed_type.weight");
  candidate->luca_layers.resize(topology.layers);
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    auto &target = candidate->luca_layers[layer];
    const std::string prefix = luca_layer_prefix(layer);
    target.pre_norm_weight = view(prefix + "pre_layer_norm.weight");
    target.pre_norm_bias = view(prefix + "pre_layer_norm.bias");
    target.query_weight = view(prefix + "self_attn.q_proj.weight");
    target.query_bias = view(prefix + "self_attn.q_proj.bias");
    target.key_weight = view(prefix + "self_attn.k_proj.weight");
    target.key_bias = view(prefix + "self_attn.k_proj.bias");
    target.value_weight = view(prefix + "self_attn.v_proj.weight");
    target.value_bias = view(prefix + "self_attn.v_proj.bias");
    target.attention_output_weight = view(prefix + "self_attn.out_proj.weight");
    target.attention_output_bias = view(prefix + "self_attn.out_proj.bias");
    target.inverse_frequency = view(prefix + "self_attn.rot_emb.inv_freq");
    target.post_norm_weight = view(prefix + "post_layer_norm.weight");
    target.post_norm_bias = view(prefix + "post_layer_norm.bias");
    target.fc1_weight = view(prefix + "fc1.weight");
    target.fc1_bias = view(prefix + "fc1.bias");
    target.fc2_weight = view(prefix + "fc2.weight");
    target.fc2_bias = view(prefix + "fc2.bias");
  }
  candidate->final_norm_weight = view("lucaone.encoder.last_layer_norm.weight");
  candidate->final_norm_bias = view("lucaone.encoder.last_layer_norm.bias");
  impl_ = std::move(candidate);
  return Status::Ok();
}

Status GenebCustomEncoderModel::load(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  GenebCustomEncoderTopology topology;
  auto status =
      geneb_custom_encoder_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  const auto &descriptor = artifact.tokenizer_asset_descriptor();
  if (!descriptor.has_value())
    return format_error("tokenizer descriptor is unavailable");
  std::unique_ptr<ArtifactTokenizer> tokenizer;
  status = ArtifactTokenizer::Load(std::string{artifact.artifact_root()},
                                   *descriptor, &tokenizer);
  if (!status.ok())
    return format_error("tokenizer asset rejected: " + status.message());
  const auto expected_kind =
      topology.variant == GenebCustomEncoderVariant::kLucaOne
          ? ArtifactTokenizerKind::kCharacter
          : ArtifactTokenizerKind::kBpe;
  if (tokenizer->kind() != expected_kind ||
      tokenizer->vocabulary_size() != topology.tokenizer_vocabulary_size ||
      tokenizer->padding_side() != ArtifactTokenizerPaddingSide::kRight ||
      !tokenizer->pad_token_id().has_value() ||
      static_cast<std::size_t>(*tokenizer->pad_token_id()) !=
          topology.pad_token_id)
    return format_error("tokenizer kind/vocabulary/right-pad contract differs");

  std::vector<GenebCustomEncoderNamedTensorView> views;
  views.reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    if (tensor.rank == 0U ||
        tensor.data_size > std::numeric_limits<std::size_t>::max())
      return format_error("tensor rank/size is invalid: " + tensor.name);
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
      if (tensor.dimensions[dimension] == 0U ||
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
  return status.ok() ? Status::Ok() : format_error(status.message());
}

Status GenebCustomEncoderModel::load_artifact(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  return load(artifact, std::move(linear_executor));
}

const GenebCustomEncoderTopology *
GenebCustomEncoderModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebCustomEncoderModel::linear_executor_name() const noexcept {
  if (impl_ == nullptr)
    return "unloaded";
  if (impl_->genomics_model != nullptr)
    return impl_->genomics_model->linear_executor_name();
  return impl_->executor == nullptr ? "scalar-reference"
                                    : impl_->executor->name();
}

Status GenebCustomEncoderModel::forward(
    const std::vector<TokenId> &tokens,
    const std::vector<std::uint8_t> &attention_mask,
    const std::vector<std::size_t> &capture_layers,
    GenebCustomEncoderForwardResult *const output) const {
  if (impl_ == nullptr)
    return invalid("model is not loaded");
  if (output == nullptr)
    return invalid("forward output is null");
  const auto &topology = impl_->topology;
  if (tokens.empty() || tokens.size() > topology.maximum_sequence_length ||
      attention_mask.size() != tokens.size())
    return invalid("token/mask length is outside model context");
  bool saw_padding = false;
  std::size_t visible = 0U;
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (static_cast<std::size_t>(tokens[index]) >= topology.vocabulary_size)
      return invalid("token ID is outside embedding vocabulary");
    const std::uint8_t mask = attention_mask[index];
    if (mask > 1U || (mask == 1U && saw_padding))
      return invalid("attention mask must be right-padded binary data");
    if (mask == 0U) {
      saw_padding = true;
      if (static_cast<std::size_t>(tokens[index]) != topology.pad_token_id)
        return invalid("masked row token must equal pad_token_id");
    } else {
      if (static_cast<std::size_t>(tokens[index]) == topology.pad_token_id)
        return invalid("visible row cannot contain pad_token_id");
      ++visible;
    }
  }
  if (visible == 0U)
    return invalid("attention mask must contain a visible token");
  std::map<std::size_t, std::size_t> capture_indices;
  for (std::size_t index = 0; index < capture_layers.size(); ++index) {
    const std::size_t layer = capture_layers[index];
    if (layer > topology.layers ||
        !capture_indices.emplace(layer, index).second)
      return invalid("capture layers must be unique and in [0,num_layers]");
  }

  if (topology.variant == GenebCustomEncoderVariant::kGenomicsFm) {
    GenebBertForwardResult bert_result;
    auto status = impl_->genomics_model->forward(tokens, attention_mask,
                                                 capture_layers, &bert_result);
    if (!status.ok())
      return status;
    GenebCustomEncoderForwardResult result;
    result.rows = bert_result.rows;
    result.width = bert_result.width;
    result.final_hidden = std::move(bert_result.final_hidden);
    result.captures.reserve(bert_result.captures.size());
    for (auto &capture : bert_result.captures)
      result.captures.push_back({capture.layer, std::move(capture.values)});
    status = geneb_custom_encoder_pool(result, attention_mask, topology.pooling,
                                       &result.pooled);
    if (!status.ok())
      return status;
    *output = std::move(result);
    return Status::Ok();
  }

  std::size_t hidden_elements = 0U;
  if (!checked_mul(tokens.size(), topology.width, &hidden_elements))
    return invalid("hidden state shape overflows");
  std::vector<float> hidden(hidden_elements, 0.0F);
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    const std::size_t token = static_cast<std::size_t>(tokens[row]);
    if (token == topology.pad_token_id)
      continue;
    for (std::size_t column = 0; column < topology.width; ++column) {
      hidden[row * topology.width + column] =
          tensor_value(impl_->word_embedding, token * topology.width + column) +
          tensor_value(impl_->token_type_embedding, column);
    }
  }
  if (!finite_vector(hidden))
    return invalid("embedding stage produced non-finite data");
  GenebCustomEncoderForwardResult result;
  result.rows = tokens.size();
  result.width = topology.width;
  result.captures.resize(capture_layers.size());
  capture_hidden(0U, hidden, capture_indices, &result.captures);
  const float query_scale =
      1.0F / std::sqrt(static_cast<float>(topology.head_dimension));
  for (std::size_t layer_index = 0; layer_index < topology.layers;
       ++layer_index) {
    const auto &layer = impl_->luca_layers[layer_index];
    std::vector<float> normalized;
    auto status = layer_norm(hidden, tokens.size(), topology.width,
                             layer.pre_norm_weight, layer.pre_norm_bias,
                             topology.layer_norm_epsilon, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    status = runtime_linear(normalized, tokens.size(), topology.width,
                            layer.query_weight, topology.width,
                            &layer.query_bias, impl_->executor.get(), &query);
    if (!status.ok())
      return status;
    status = runtime_linear(normalized, tokens.size(), topology.width,
                            layer.key_weight, topology.width, &layer.key_bias,
                            impl_->executor.get(), &key);
    if (!status.ok())
      return status;
    status = runtime_linear(normalized, tokens.size(), topology.width,
                            layer.value_weight, topology.width,
                            &layer.value_bias, impl_->executor.get(), &value);
    if (!status.ok())
      return status;
    for (float &item : query)
      item *= query_scale;
    status =
        apply_luca_rotary(&query, &key, layer.inverse_frequency, tokens.size(),
                          topology.attention_heads, topology.head_dimension);
    if (!status.ok())
      return status;
    std::vector<float> attended;
    status = bidirectional_attention(query, key, value, attention_mask,
                                     tokens.size(), topology.attention_heads,
                                     topology.head_dimension, &attended);
    if (!status.ok())
      return status;
    std::vector<float> projected;
    status = runtime_linear(attended, tokens.size(), topology.width,
                            layer.attention_output_weight, topology.width,
                            &layer.attention_output_bias, impl_->executor.get(),
                            &projected);
    if (!status.ok())
      return status;
    add_in_place(&hidden, projected);

    status = layer_norm(hidden, tokens.size(), topology.width,
                        layer.post_norm_weight, layer.post_norm_bias,
                        topology.layer_norm_epsilon, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> intermediate;
    status =
        runtime_linear(normalized, tokens.size(), topology.width,
                       layer.fc1_weight, topology.inner_width, &layer.fc1_bias,
                       impl_->executor.get(), &intermediate);
    if (!status.ok())
      return status;
    for (float &value_item : intermediate)
      value_item = exact_gelu(value_item);
    std::vector<float> feed_forward;
    status = runtime_linear(intermediate, tokens.size(), topology.inner_width,
                            layer.fc2_weight, topology.width, &layer.fc2_bias,
                            impl_->executor.get(), &feed_forward);
    if (!status.ok())
      return status;
    add_in_place(&hidden, feed_forward);
    if (!finite_vector(hidden))
      return invalid("LucaOne encoder block produced non-finite data");
    if (layer_index + 1U < topology.layers)
      capture_hidden(layer_index + 1U, hidden, capture_indices,
                     &result.captures);
  }
  auto status = layer_norm(hidden, tokens.size(), topology.width,
                           impl_->final_norm_weight, impl_->final_norm_bias,
                           topology.layer_norm_epsilon, &result.final_hidden);
  if (!status.ok())
    return status;
  capture_hidden(topology.layers, result.final_hidden, capture_indices,
                 &result.captures);
  status = geneb_custom_encoder_pool(result, attention_mask, topology.pooling,
                                     &result.pooled);
  if (!status.ok())
    return status;
  *output = std::move(result);
  return Status::Ok();
}

} // namespace evo::cpu
