// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_dna_gpt.hpp"
#include "evo/cpu/geneb_gpt2.hpp"

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
#include <utility>
#include <vector>

namespace evo::cpu {
namespace {

constexpr std::size_t kMaximumWidth = 65536;
constexpr std::size_t kMaximumLayers = 1024;
constexpr std::size_t kMaximumInnerWidth = 1U << 20U;
constexpr std::size_t kMaximumSequenceLength = 1U << 20U;
constexpr std::string_view kTokenizerProfile = "evo-tokenizer-v1";

enum class MatrixLayout : std::uint8_t { kOutIn, kInOut };

struct TensorRef final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct NamedTensorRef final {
  std::string name;
  TensorRef tensor;
};

struct TensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct RuntimeConfig final {
  std::size_t vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t heads{0};
  std::size_t inner_width{0};
  std::size_t maximum_sequence_length{0};
  float norm_epsilon{0.0F};
  MatrixLayout matrix_layout{MatrixLayout::kOutIn};
  bool linear_bias{false};
  bool norm_bias{false};
  bool attention_uses_mask{false};
};

struct RuntimeLayer final {
  TensorRef attention_qkv_weight;
  TensorRef attention_qkv_bias;
  TensorRef attention_output_weight;
  TensorRef attention_output_bias;
  TensorRef attention_norm_weight;
  TensorRef attention_norm_bias;
  TensorRef mlp_input_weight;
  TensorRef mlp_input_bias;
  TensorRef mlp_output_weight;
  TensorRef mlp_output_bias;
  TensorRef mlp_norm_weight;
  TensorRef mlp_norm_bias;
};

struct Runtime final {
  RuntimeConfig config;
  TensorRef token_embedding;
  TensorRef position_embedding;
  std::vector<RuntimeLayer> blocks;
  TensorRef final_norm_weight;
  TensorRef final_norm_bias;
};

struct GenericCapture final {
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenericResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenericCapture> captures;
  std::vector<float> final_hidden;
  std::vector<float> pooled;
};

Status invalid(const std::string_view family, const std::string &message) {
  return {ErrorCode::kInvalidArgument,
          "GENEB " + std::string{family} + ": " + message};
}

Status format_error(const std::string_view family, const std::string &message) {
  return {ErrorCode::kModelFormat,
          "GENEB " + std::string{family} + " artifact: " + message};
}

bool checked_mul(const std::size_t left, const std::size_t right,
                 std::size_t *const output) noexcept {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
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
      entry->value.size() != 1U || entry->value.front() > 1U)
    return false;
  *output = entry->value.front() != 0U;
  return true;
}

bool metadata_string(const ModelFile &artifact, const std::string_view key,
                     std::string *const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kString)
    return false;
  output->assign(reinterpret_cast<const char *>(entry->value.data()),
                 entry->value.size());
  return true;
}

Status require_literal(const ModelFile &artifact, const std::string_view key,
                       const std::string_view expected,
                       const std::string_view family) {
  std::string actual;
  if (!metadata_string(artifact, key, &actual) || actual != expected)
    return format_error(family, "metadata '" + std::string{key} +
                                    "' must be '" + std::string{expected} +
                                    "'");
  return Status::Ok();
}

Status require_bool(const ModelFile &artifact, const std::string_view key,
                    const bool expected, const std::string_view family) {
  bool actual = false;
  if (!metadata_bool(artifact, key, &actual) || actual != expected)
    return format_error(family, "metadata '" + std::string{key} +
                                    "' has the wrong boolean value");
  return Status::Ok();
}

Status metadata_size(const ModelFile &artifact, const std::string_view key,
                     std::size_t *const output, const std::string_view family) {
  std::uint64_t value = 0;
  if (!metadata_u64(artifact, key, &value) || value == 0U ||
      value >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    return format_error(family, "metadata '" + std::string{key} +
                                    "' must be a positive size_t");
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

Status validate_runtime_config(const RuntimeConfig &config,
                               const std::string_view family) {
  if (config.vocabulary_size == 0U ||
      !token_vocabulary_size_supported(config.vocabulary_size) ||
      config.width == 0U || config.width > kMaximumWidth ||
      config.layers == 0U || config.layers > kMaximumLayers ||
      config.heads == 0U || config.width % config.heads != 0U ||
      config.inner_width == 0U || config.inner_width > kMaximumInnerWidth ||
      config.maximum_sequence_length == 0U ||
      config.maximum_sequence_length > kMaximumSequenceLength ||
      !std::isfinite(config.norm_epsilon) || config.norm_epsilon <= 0.0F)
    return invalid(family, "topology is outside the strict runtime contract");
  return Status::Ok();
}

float tensor_value(const TensorRef &tensor, const std::size_t index) noexcept {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return value;
}

Status validate_tensor(const TensorRef &tensor,
                       const TensorRequirement &requirement,
                       const std::string_view family) {
  if (tensor.data == nullptr)
    return invalid(family, "tensor '" + requirement.name + "' has null data");
  if (tensor.dtype != requirement.dtype)
    return invalid(family,
                   "tensor '" + requirement.name + "' has the wrong dtype");
  if (tensor.shape != requirement.shape)
    return invalid(family,
                   "tensor '" + requirement.name + "' has the wrong shape");
  std::size_t elements = 1;
  for (const auto dimension : tensor.shape) {
    if (dimension == 0U || !checked_mul(elements, dimension, &elements))
      return invalid(family,
                     "tensor '" + requirement.name + "' shape overflows");
  }
  std::size_t bytes = 0;
  if (!checked_mul(elements, sizeof(float), &bytes) || bytes != tensor.bytes)
    return invalid(family,
                   "tensor '" + requirement.name + "' has the wrong byte size");
  return Status::Ok();
}

std::vector<TensorRequirement>
canonical_requirements(const RuntimeConfig &config) {
  std::vector<TensorRequirement> result;
  result.reserve(3U + config.layers * (config.linear_bias ? 12U : 6U) +
                 (config.norm_bias ? 1U : 0U));
  result.push_back({"transformer.wte.weight",
                    TensorDType::kF32,
                    {config.vocabulary_size, config.width}});
  result.push_back({"transformer.wpe.weight",
                    TensorDType::kF32,
                    {config.maximum_sequence_length, config.width}});
  const auto matrix_shape = [&](const std::size_t input_width,
                                const std::size_t output_width) {
    return config.matrix_layout == MatrixLayout::kOutIn
               ? std::vector<std::size_t>{output_width, input_width}
               : std::vector<std::size_t>{input_width, output_width};
  };
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    const std::string prefix = "transformer.h." + std::to_string(layer) + ".";
    result.push_back({prefix + "attn.c_attn.weight", TensorDType::kF32,
                      matrix_shape(config.width, config.width * 3U)});
    if (config.linear_bias)
      result.push_back({prefix + "attn.c_attn.bias",
                        TensorDType::kF32,
                        {config.width * 3U}});
    result.push_back({prefix + "attn.c_proj.weight", TensorDType::kF32,
                      matrix_shape(config.width, config.width)});
    if (config.linear_bias)
      result.push_back(
          {prefix + "attn.c_proj.bias", TensorDType::kF32, {config.width}});
    result.push_back(
        {prefix + "ln_1.weight", TensorDType::kF32, {config.width}});
    if (config.norm_bias)
      result.push_back(
          {prefix + "ln_1.bias", TensorDType::kF32, {config.width}});
    result.push_back(
        {prefix + "ln_2.weight", TensorDType::kF32, {config.width}});
    if (config.norm_bias)
      result.push_back(
          {prefix + "ln_2.bias", TensorDType::kF32, {config.width}});
    result.push_back({prefix + "mlp.c_fc.weight", TensorDType::kF32,
                      matrix_shape(config.width, config.inner_width)});
    if (config.linear_bias)
      result.push_back(
          {prefix + "mlp.c_fc.bias", TensorDType::kF32, {config.inner_width}});
    result.push_back({prefix + "mlp.c_proj.weight", TensorDType::kF32,
                      matrix_shape(config.inner_width, config.width)});
    if (config.linear_bias)
      result.push_back(
          {prefix + "mlp.c_proj.bias", TensorDType::kF32, {config.width}});
  }
  result.push_back(
      {"transformer.ln_f.weight", TensorDType::kF32, {config.width}});
  if (config.norm_bias)
    result.push_back(
        {"transformer.ln_f.bias", TensorDType::kF32, {config.width}});
  return result;
}

const TensorRef &required_ref(
    const std::map<std::string, const TensorRef *, std::less<>> &by_name,
    const std::string &name) {
  return *by_name.find(name)->second;
}

Status load_runtime(const RuntimeConfig &config,
                    const std::vector<NamedTensorRef> &tensors,
                    const std::string_view family, Runtime *const output) {
  if (output == nullptr)
    return invalid(family, "runtime output is null");
  auto status = validate_runtime_config(config, family);
  if (!status.ok())
    return status;
  const auto requirements = canonical_requirements(config);
  std::map<std::string, const TensorRef *, std::less<>> by_name;
  for (const auto &item : tensors) {
    if (!by_name.emplace(item.name, &item.tensor).second)
      return invalid(family, "duplicate tensor '" + item.name + "'");
  }
  if (by_name.size() != requirements.size())
    return invalid(family, "tensor set has missing or extra entries");
  for (const auto &requirement : requirements) {
    const auto found = by_name.find(requirement.name);
    if (found == by_name.end())
      return invalid(family,
                     "required tensor '" + requirement.name + "' is missing");
    status = validate_tensor(*found->second, requirement, family);
    if (!status.ok())
      return status;
  }

  Runtime candidate;
  candidate.config = config;
  candidate.token_embedding = required_ref(by_name, "transformer.wte.weight");
  candidate.position_embedding =
      required_ref(by_name, "transformer.wpe.weight");
  candidate.blocks.resize(config.layers);
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    auto &block = candidate.blocks[layer];
    const std::string prefix = "transformer.h." + std::to_string(layer) + ".";
    block.attention_qkv_weight =
        required_ref(by_name, prefix + "attn.c_attn.weight");
    block.attention_output_weight =
        required_ref(by_name, prefix + "attn.c_proj.weight");
    block.attention_norm_weight = required_ref(by_name, prefix + "ln_1.weight");
    block.mlp_norm_weight = required_ref(by_name, prefix + "ln_2.weight");
    block.mlp_input_weight = required_ref(by_name, prefix + "mlp.c_fc.weight");
    block.mlp_output_weight =
        required_ref(by_name, prefix + "mlp.c_proj.weight");
    if (config.linear_bias) {
      block.attention_qkv_bias =
          required_ref(by_name, prefix + "attn.c_attn.bias");
      block.attention_output_bias =
          required_ref(by_name, prefix + "attn.c_proj.bias");
      block.mlp_input_bias = required_ref(by_name, prefix + "mlp.c_fc.bias");
      block.mlp_output_bias = required_ref(by_name, prefix + "mlp.c_proj.bias");
    }
    if (config.norm_bias) {
      block.attention_norm_bias = required_ref(by_name, prefix + "ln_1.bias");
      block.mlp_norm_bias = required_ref(by_name, prefix + "ln_2.bias");
    }
  }
  candidate.final_norm_weight =
      required_ref(by_name, "transformer.ln_f.weight");
  if (config.norm_bias)
    candidate.final_norm_bias = required_ref(by_name, "transformer.ln_f.bias");
  *output = std::move(candidate);
  return Status::Ok();
}

Status linear(const std::vector<float> &input, const std::size_t rows,
              const std::size_t input_width, const TensorRef &weight,
              const std::size_t output_width, const TensorRef *const bias,
              const MatrixLayout layout, const std::string_view family,
              std::vector<float> *const output) {
  std::size_t input_elements = 0;
  std::size_t output_elements = 0;
  if (output == nullptr || !checked_mul(rows, input_width, &input_elements) ||
      !checked_mul(rows, output_width, &output_elements) ||
      input.size() != input_elements)
    return invalid(family, "linear shape contract differs");
  output->assign(output_elements, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t input_base = row * input_width;
    for (std::size_t target = 0; target < output_width; ++target) {
      float total = bias == nullptr ? 0.0F : tensor_value(*bias, target);
      for (std::size_t source = 0; source < input_width; ++source) {
        const std::size_t weight_index = layout == MatrixLayout::kOutIn
                                             ? target * input_width + source
                                             : source * output_width + target;
        total +=
            input[input_base + source] * tensor_value(weight, weight_index);
      }
      (*output)[row * output_width + target] = total;
    }
  }
  return Status::Ok();
}

Status layer_norm(const std::vector<float> &input, const std::size_t rows,
                  const std::size_t width, const TensorRef &scale,
                  const TensorRef *const bias, const float epsilon,
                  const std::string_view family,
                  std::vector<float> *const output) {
  std::size_t elements = 0;
  if (output == nullptr || !checked_mul(rows, width, &elements) ||
      input.size() != elements)
    return invalid(family, "LayerNorm shape contract differs");
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
      float value =
          (input[base + column] - mean) * inverse * tensor_value(scale, column);
      if (bias != nullptr)
        value += tensor_value(*bias, column);
      (*output)[base + column] = value;
    }
  }
  return Status::Ok();
}

Status causal_attention(const std::vector<float> &query,
                        const std::vector<float> &key,
                        const std::vector<float> &value,
                        const std::vector<std::uint8_t> &mask,
                        const RuntimeConfig &config,
                        const std::string_view family,
                        std::vector<float> *const output) {
  const std::size_t rows = mask.size();
  const std::size_t head_width = config.width / config.heads;
  std::size_t elements = 0;
  if (output == nullptr || !checked_mul(rows, config.width, &elements) ||
      query.size() != elements || key.size() != elements ||
      value.size() != elements)
    return invalid(family, "attention shape contract differs");
  output->assign(elements, 0.0F);
  std::vector<float> scores(rows, 0.0F);
  std::vector<float> probabilities(rows, 0.0F);
  std::vector<std::size_t> visible(rows, 0U);
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_width));
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t head = 0; head < config.heads; ++head) {
      std::size_t count = 0;
      float maximum = -std::numeric_limits<float>::infinity();
      const std::size_t query_base = row * config.width + head * head_width;
      for (std::size_t source = 0; source <= row; ++source) {
        if (config.attention_uses_mask && mask[source] == 0U)
          continue;
        const std::size_t key_base = source * config.width + head * head_width;
        float score = 0.0F;
        for (std::size_t dimension = 0; dimension < head_width; ++dimension)
          score += query[query_base + dimension] * key[key_base + dimension];
        score *= scale;
        visible[count] = source;
        scores[count++] = score;
        maximum = std::max(maximum, score);
      }
      if (count == 0U)
        return invalid(family, "attention query has no visible causal key");
      float denominator = 0.0F;
      for (std::size_t index = 0; index < count; ++index) {
        probabilities[index] = std::exp(scores[index] - maximum);
        denominator += probabilities[index];
      }
      for (std::size_t index = 0; index < count; ++index)
        probabilities[index] /= denominator;
      for (std::size_t dimension = 0; dimension < head_width; ++dimension) {
        float total = 0.0F;
        for (std::size_t index = 0; index < count; ++index) {
          const std::size_t value_index =
              visible[index] * config.width + head * head_width + dimension;
          total += probabilities[index] * value[value_index];
        }
        (*output)[query_base + dimension] = total;
      }
    }
  }
  return Status::Ok();
}

float gelu_tanh(const float value) noexcept {
  constexpr float kCoefficient = 0.7978845608028654F;
  return 0.5F * value *
         (1.0F + std::tanh(kCoefficient *
                           (value + 0.044715F * value * value * value)));
}

void add_in_place(std::vector<float> *const target,
                  const std::vector<float> &increment) {
  for (std::size_t index = 0; index < target->size(); ++index)
    (*target)[index] += increment[index];
}

void capture(const std::size_t layer, const std::vector<float> &values,
             const std::map<std::size_t, std::size_t> &capture_indices,
             std::vector<GenericCapture> *const captures) {
  const auto found = capture_indices.find(layer);
  if (found != capture_indices.end())
    (*captures)[found->second] = {layer, values};
}

Status validate_forward_inputs(
    const Runtime &runtime, const std::vector<TokenId> &tokens,
    const std::vector<std::uint8_t> &mask,
    const std::vector<std::size_t> &capture_layers,
    const std::string_view family,
    std::map<std::size_t, std::size_t> *const capture_indices,
    std::size_t *const valid_count) {
  const auto &config = runtime.config;
  if (tokens.empty() || tokens.size() > config.maximum_sequence_length)
    return invalid(family, "token count is outside model context");
  if (mask.size() != tokens.size())
    return invalid(family, "mask length differs from token count");
  bool saw_padding = false;
  std::size_t count = 0;
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    if (!token_id_in_vocabulary(tokens[row], config.vocabulary_size))
      return invalid(family, "token ID is outside the model vocabulary");
    if (mask[row] > 1U)
      return invalid(family, "mask must be binary");
    if (mask[row] == 0U) {
      saw_padding = true;
    } else {
      if (saw_padding)
        return invalid(family, "mask must describe right padding");
      ++count;
    }
  }
  if (count == 0U)
    return invalid(family, "mask must contain at least one non-pad row");
  capture_indices->clear();
  for (std::size_t index = 0; index < capture_layers.size(); ++index) {
    if (capture_layers[index] > config.layers ||
        !capture_indices->emplace(capture_layers[index], index).second)
      return invalid(family,
                     "capture layers must be unique and in [0,num_layers]");
  }
  *valid_count = count;
  return Status::Ok();
}

Status forward_runtime(const Runtime &runtime,
                       const std::vector<TokenId> &tokens,
                       const std::vector<std::uint8_t> &mask,
                       const std::vector<std::size_t> &capture_layers,
                       const std::string_view family,
                       GenericResult *const output) {
  if (output == nullptr)
    return invalid(family, "forward output is null");
  std::map<std::size_t, std::size_t> capture_indices;
  std::size_t valid_count = 0;
  auto status = validate_forward_inputs(runtime, tokens, mask, capture_layers,
                                        family, &capture_indices, &valid_count);
  if (!status.ok())
    return status;
  const auto &config = runtime.config;
  std::size_t hidden_elements = 0;
  if (!checked_mul(tokens.size(), config.width, &hidden_elements))
    return invalid(family, "hidden state shape overflows");
  std::vector<float> hidden(hidden_elements, 0.0F);
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    const std::size_t token = static_cast<std::size_t>(tokens[row]);
    for (std::size_t column = 0; column < config.width; ++column) {
      hidden[row * config.width + column] =
          tensor_value(runtime.token_embedding, token * config.width + column) +
          tensor_value(runtime.position_embedding, row * config.width + column);
    }
  }

  GenericResult result;
  result.rows = tokens.size();
  result.width = config.width;
  result.captures.resize(capture_layers.size());
  capture(0U, hidden, capture_indices, &result.captures);
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    const auto &block = runtime.blocks[layer];
    const TensorRef *const attention_norm_bias =
        config.norm_bias ? &block.attention_norm_bias : nullptr;
    const TensorRef *const mlp_norm_bias =
        config.norm_bias ? &block.mlp_norm_bias : nullptr;
    const TensorRef *const qkv_bias =
        config.linear_bias ? &block.attention_qkv_bias : nullptr;
    const TensorRef *const attention_output_bias =
        config.linear_bias ? &block.attention_output_bias : nullptr;
    const TensorRef *const mlp_input_bias =
        config.linear_bias ? &block.mlp_input_bias : nullptr;
    const TensorRef *const mlp_output_bias =
        config.linear_bias ? &block.mlp_output_bias : nullptr;

    std::vector<float> normalized;
    status = layer_norm(hidden, tokens.size(), config.width,
                        block.attention_norm_weight, attention_norm_bias,
                        config.norm_epsilon, family, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> fused_qkv;
    status = linear(normalized, tokens.size(), config.width,
                    block.attention_qkv_weight, config.width * 3U, qkv_bias,
                    config.matrix_layout, family, &fused_qkv);
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
    std::vector<float> attended;
    status =
        causal_attention(query, key, value, mask, config, family, &attended);
    if (!status.ok())
      return status;
    std::vector<float> attention_output;
    status = linear(attended, tokens.size(), config.width,
                    block.attention_output_weight, config.width,
                    attention_output_bias, config.matrix_layout, family,
                    &attention_output);
    if (!status.ok())
      return status;
    add_in_place(&hidden, attention_output);

    status =
        layer_norm(hidden, tokens.size(), config.width, block.mlp_norm_weight,
                   mlp_norm_bias, config.norm_epsilon, family, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> feed_forward;
    status = linear(normalized, tokens.size(), config.width,
                    block.mlp_input_weight, config.inner_width, mlp_input_bias,
                    config.matrix_layout, family, &feed_forward);
    if (!status.ok())
      return status;
    for (auto &value_item : feed_forward)
      value_item = gelu_tanh(value_item);
    std::vector<float> feed_forward_output;
    status = linear(feed_forward, tokens.size(), config.inner_width,
                    block.mlp_output_weight, config.width, mlp_output_bias,
                    config.matrix_layout, family, &feed_forward_output);
    if (!status.ok())
      return status;
    add_in_place(&hidden, feed_forward_output);
    if (layer + 1U < config.layers)
      capture(layer + 1U, hidden, capture_indices, &result.captures);
  }

  const TensorRef *const final_bias =
      config.norm_bias ? &runtime.final_norm_bias : nullptr;
  status =
      layer_norm(hidden, tokens.size(), config.width, runtime.final_norm_weight,
                 final_bias, config.norm_epsilon, family, &result.final_hidden);
  if (!status.ok())
    return status;
  capture(config.layers, result.final_hidden, capture_indices,
          &result.captures);
  result.pooled.assign(config.width, 0.0F);
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    if (mask[row] == 0U)
      continue;
    for (std::size_t column = 0; column < config.width; ++column)
      result.pooled[column] += result.final_hidden[row * config.width + column];
  }
  for (auto &value_item : result.pooled)
    value_item /= static_cast<float>(valid_count);
  if (!std::all_of(
          result.final_hidden.begin(), result.final_hidden.end(),
          [](const float value_item) { return std::isfinite(value_item); }) ||
      !std::all_of(
          result.pooled.begin(), result.pooled.end(),
          [](const float value_item) { return std::isfinite(value_item); }))
    return invalid(family, "forward produced a non-finite value");
  *output = std::move(result);
  return Status::Ok();
}

Status common_topology_from_artifact(
    const ModelFile &artifact, const std::string_view profile,
    const std::string_view abi, const std::string_view architecture,
    const std::string_view metadata_prefix, const MatrixLayout layout,
    const bool linear_bias, const bool norm_bias,
    const bool attention_uses_mask, const std::string_view family,
    RuntimeConfig *const output) {
  if (output == nullptr)
    return invalid(family, "topology output is null");
  if (artifact.profile() != profile)
    return format_error(family,
                        "profile must be '" + std::string{profile} + "'");
  auto status = require_literal(artifact, "runtime.abi", abi, family);
  if (!status.ok())
    return status;
  status =
      require_literal(artifact, "model.architecture", architecture, family);
  if (!status.ok())
    return status;
  status =
      require_literal(artifact, "tokenizer.profile", kTokenizerProfile, family);
  if (!status.ok())
    return status;
  if (!artifact.tokenizer_asset_descriptor().has_value())
    return format_error(family, "complete tokenizer descriptor is required");

  RuntimeConfig config;
  for (const auto &field :
       {std::pair{"config.vocab_size", &config.vocabulary_size},
        std::pair{"config.hidden_size", &config.width},
        std::pair{"config.num_layers", &config.layers},
        std::pair{"config.max_seqlen", &config.maximum_sequence_length}}) {
    status = metadata_size(artifact, field.first, field.second, family);
    if (!status.ok())
      return status;
  }
  status = metadata_size(artifact,
                         std::string{metadata_prefix} + ".num_attention_heads",
                         &config.heads, family);
  if (!status.ok())
    return status;
  status =
      metadata_size(artifact, std::string{metadata_prefix} + ".inner_width",
                    &config.inner_width, family);
  if (!status.ok())
    return status;
  double epsilon = 0.0;
  if (!metadata_f64(artifact, std::string{metadata_prefix} + ".norm_epsilon",
                    &epsilon) ||
      !std::isfinite(epsilon) || epsilon <= 0.0 ||
      epsilon > std::numeric_limits<float>::max())
    return format_error(family, "norm epsilon metadata is invalid");
  config.norm_epsilon = static_cast<float>(epsilon);
  config.matrix_layout = layout;
  config.linear_bias = linear_bias;
  config.norm_bias = norm_bias;
  config.attention_uses_mask = attention_uses_mask;
  std::uint64_t layer_count = 0;
  if (!metadata_u64(artifact, "runtime.embedding_layer_count", &layer_count) ||
      layer_count != static_cast<std::uint64_t>(config.layers) + 1U)
    return format_error(
        family, "runtime.embedding_layer_count must equal config.num_layers+1");
  status = validate_runtime_config(config, family);
  if (!status.ok())
    return status;
  *output = config;
  return Status::Ok();
}

Status model_file_tensors(const ModelFile &artifact,
                          std::vector<NamedTensorRef> *const output) {
  output->clear();
  output->reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    if (tensor.data_size >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      return {ErrorCode::kModelFormat,
              "GENEB GPT tensor byte size exceeds size_t"};
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t index = 0; index < tensor.rank; ++index) {
      if (tensor.dimensions[index] >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return {ErrorCode::kModelFormat,
                "GENEB GPT tensor dimension exceeds size_t"};
      shape.push_back(static_cast<std::size_t>(tensor.dimensions[index]));
    }
    output->push_back({tensor.name,
                       {artifact.tensor_data(tensor),
                        static_cast<std::size_t>(tensor.data_size),
                        tensor.dtype, std::move(shape)}});
  }
  return Status::Ok();
}

RuntimeConfig gpt2_config(const GenebGpt2Topology &topology) {
  return {topology.vocabulary_size,
          topology.width,
          topology.layers,
          topology.heads,
          topology.inner_width,
          topology.maximum_sequence_length,
          topology.norm_epsilon,
          MatrixLayout::kInOut,
          true,
          true,
          true};
}

RuntimeConfig dna_gpt_config(const GenebDnaGptTopology &topology) {
  return {topology.vocabulary_size,
          topology.width,
          topology.layers,
          topology.heads,
          topology.inner_width,
          topology.maximum_sequence_length,
          topology.norm_epsilon,
          MatrixLayout::kOutIn,
          false,
          false,
          false};
}

template <typename Requirement>
void copy_requirements(const std::vector<TensorRequirement> &source,
                       std::vector<Requirement> *const output) {
  output->clear();
  output->reserve(source.size());
  for (const auto &item : source)
    output->push_back({item.name, item.dtype, item.shape});
}

} // namespace

struct GenebGpt2Model::Impl final {
  GenebGpt2Topology topology;
  Runtime runtime;
};

Status validate_geneb_gpt2_topology(const GenebGpt2Topology &topology) {
  return validate_runtime_config(gpt2_config(topology), "GPT-2");
}

Status geneb_gpt2_topology_from_artifact(const ModelFile &artifact,
                                         GenebGpt2Topology *const output) {
  RuntimeConfig config;
  auto status = common_topology_from_artifact(
      artifact, kGenebGpt2ArtifactProfile, kGenebGpt2RuntimeAbi,
      kGenebGpt2Architecture, "gpt2", MatrixLayout::kInOut, true, true, true,
      "GPT-2", &config);
  if (!status.ok())
    return status;
  for (const auto &literal :
       {std::pair{"gpt2.activation", "gelu-new"},
        std::pair{"gpt2.qkv_layout", "q-k-v"},
        std::pair{"gpt2.weight_layout", "conv1d-in-out"},
        std::pair{"gpt2.weight_dtype", "F32"},
        std::pair{"gpt2.hidden_tap", "post-final-layernorm"},
        std::pair{"gpt2.pooling", "attention-mask-mean"},
        std::pair{"source.kind", "huggingface"},
        std::pair{"source.transformers.version", "4.45.2"},
        std::pair{
            "source.transformers.modeling_gpt2_sha256",
            "57750b9faf5a72f596966d7f6541ad7a04294ea206c9af03a4702c60b1b4f3f6"},
        std::pair{
            "source.transformers.configuration_gpt2_sha256",
            "3689eab68d1e20b31caff38976f9f7b12484791321d84a4ff7bd791b205cad01"},
        std::pair{
            "source.transformers.activations_sha256",
            "10c37e915ce2b52d539a5b52ecaaf644e2b0c56d282db00778098d74342ebeee"},
        std::pair{
            "source.geneb.extractor_sha256",
            "f70ac97116ebd4971a0fdfe30cc5ddab18ea9c5c47b4f78e6b7dc046832de692"},
        std::pair{"geneb.provenance.extractor_commit",
                  "b465d2d6a11efbbc9a22c105e34832725ce50e05"}}) {
    status = require_literal(artifact, literal.first, literal.second, "GPT-2");
    if (!status.ok())
      return status;
  }
  for (const auto &flag : {std::pair{"gpt2.linear_bias", true},
                           std::pair{"gpt2.layer_norm_bias", true},
                           std::pair{"gpt2.absolute_position_embeddings", true},
                           std::pair{"gpt2.causal_attention", true},
                           std::pair{"gpt2.attention_uses_mask", true},
                           std::pair{"gpt2.eval_dropout_disabled", true},
                           std::pair{"source.immutable", true},
                           std::pair{"geneb.source.immutable", true}}) {
    status = require_bool(artifact, flag.first, flag.second, "GPT-2");
    if (!status.ok())
      return status;
  }
  *output = {config.vocabulary_size, config.width,
             config.layers,          config.heads,
             config.inner_width,     config.maximum_sequence_length,
             config.norm_epsilon};
  return Status::Ok();
}

Status canonical_geneb_gpt2_tensors(
    const GenebGpt2Topology &topology,
    std::vector<GenebGpt2TensorRequirement> *const output) {
  if (output == nullptr)
    return invalid("GPT-2", "tensor manifest output is null");
  auto status = validate_geneb_gpt2_topology(topology);
  if (!status.ok())
    return status;
  copy_requirements(canonical_requirements(gpt2_config(topology)), output);
  return Status::Ok();
}

GenebGpt2Model::GenebGpt2Model() = default;
GenebGpt2Model::~GenebGpt2Model() = default;
GenebGpt2Model::GenebGpt2Model(GenebGpt2Model &&) noexcept = default;
GenebGpt2Model &GenebGpt2Model::operator=(GenebGpt2Model &&) noexcept = default;

Status
GenebGpt2Model::load(const GenebGpt2Topology &topology,
                     const std::vector<GenebGpt2NamedTensorView> &tensors) {
  std::vector<NamedTensorRef> generic;
  generic.reserve(tensors.size());
  for (const auto &item : tensors)
    generic.push_back({item.name,
                       {item.tensor.data, item.tensor.bytes, item.tensor.dtype,
                        item.tensor.shape}});
  Runtime runtime;
  auto status = load_runtime(gpt2_config(topology), generic, "GPT-2", &runtime);
  if (!status.ok())
    return status;
  auto candidate = std::make_unique<Impl>();
  candidate->topology = topology;
  candidate->runtime = std::move(runtime);
  impl_ = std::move(candidate);
  return Status::Ok();
}

Status GenebGpt2Model::load(const ModelFile &artifact) {
  GenebGpt2Topology topology;
  auto status = geneb_gpt2_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<NamedTensorRef> generic;
  status = model_file_tensors(artifact, &generic);
  if (!status.ok())
    return status;
  Runtime runtime;
  status = load_runtime(gpt2_config(topology), generic, "GPT-2", &runtime);
  if (!status.ok())
    return status;
  auto candidate = std::make_unique<Impl>();
  candidate->topology = topology;
  candidate->runtime = std::move(runtime);
  impl_ = std::move(candidate);
  return Status::Ok();
}

const GenebGpt2Topology *GenebGpt2Model::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebGpt2Model::kernel_name() const noexcept {
  return impl_ == nullptr ? "unloaded" : "geneb-gpt2-reference-f32";
}

Status GenebGpt2Model::forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebGpt2ForwardResult *const output) const {
  if (impl_ == nullptr)
    return invalid("GPT-2", "model is not loaded");
  if (output == nullptr)
    return invalid("GPT-2", "forward output is null");
  GenericResult generic;
  auto status = forward_runtime(impl_->runtime, tokens, attention_mask,
                                capture_layers, "GPT-2", &generic);
  if (!status.ok())
    return status;
  GenebGpt2ForwardResult result;
  result.rows = generic.rows;
  result.width = generic.width;
  result.final_hidden = std::move(generic.final_hidden);
  result.pooled = std::move(generic.pooled);
  result.captures.reserve(generic.captures.size());
  for (auto &item : generic.captures)
    result.captures.push_back({item.layer, std::move(item.values)});
  *output = std::move(result);
  return Status::Ok();
}

struct GenebDnaGptModel::Impl final {
  GenebDnaGptTopology topology;
  Runtime runtime;
};

Status validate_geneb_dna_gpt_topology(const GenebDnaGptTopology &topology) {
  return validate_runtime_config(dna_gpt_config(topology), "DNA-GPT");
}

Status geneb_dna_gpt_topology_from_artifact(const ModelFile &artifact,
                                            GenebDnaGptTopology *const output) {
  RuntimeConfig config;
  auto status = common_topology_from_artifact(
      artifact, kGenebDnaGptArtifactProfile, kGenebDnaGptRuntimeAbi,
      kGenebDnaGptArchitecture, "dna_gpt", MatrixLayout::kOutIn, false, false,
      false, "DNA-GPT", &config);
  if (!status.ok())
    return status;
  for (const auto &literal :
       {std::pair{"dna_gpt.activation", "gelu-tanh"},
        std::pair{"dna_gpt.qkv_layout", "q-k-v"},
        std::pair{"dna_gpt.weight_layout", "linear-out-in"},
        std::pair{"dna_gpt.weight_dtype", "F32"},
        std::pair{"dna_gpt.layer_norm", "affine-weight-only"},
        std::pair{"dna_gpt.hidden_tap", "post-final-layernorm"},
        std::pair{"dna_gpt.pooling", "non-pad-token-mean"},
        std::pair{"source.kind", "google-drive"},
        std::pair{"source.requested_revision", ""},
        std::pair{"source.revision", ""},
        std::pair{"source.dnagpt.repo", "TencentAILabHealthcare/DNAGPT"},
        std::pair{"source.dnagpt.revision",
                  "9b6c0931e3b2011ee5bbb4b988be3e19d62953ae"},
        std::pair{
            "source.dnagpt.gpt_py_sha256",
            "e07ae8f5b7bed3ef38009a68d3699a215e4bc6c0e860a77fa873779a2123a6f0"},
        std::pair{
            "source.dnagpt.dna_gpt_py_sha256",
            "be7e4b7bfd249a1c56f1d2b282d756f62482e69daa81054d57d72afa414caa8f"},
        std::pair{
            "source.dnagpt.tokenizer_py_sha256",
            "799bba1f99977fe9b998d5e0a765ded7b973d7929481497dc741291d575dec16"},
        std::pair{
            "source.geneb.extractor_sha256",
            "3c9df4316dc049d447516527b378e6a4d944a63b83f4d656099d40e648656212"},
        std::pair{"geneb.provenance.extractor_commit",
                  "b465d2d6a11efbbc9a22c105e34832725ce50e05"}}) {
    status =
        require_literal(artifact, literal.first, literal.second, "DNA-GPT");
    if (!status.ok())
      return status;
  }
  for (const auto &flag :
       {std::pair{"dna_gpt.linear_bias", false},
        std::pair{"dna_gpt.layer_norm_bias", false},
        std::pair{"dna_gpt.absolute_position_embeddings", true},
        std::pair{"dna_gpt.causal_attention", true},
        std::pair{"dna_gpt.attention_uses_mask", false},
        std::pair{"dna_gpt.eval_dropout_disabled", true},
        std::pair{"source.immutable", false},
        std::pair{"geneb.source.immutable", false}}) {
    status = require_bool(artifact, flag.first, flag.second, "DNA-GPT");
    if (!status.ok())
      return status;
  }
  std::string source_dtype;
  std::string tokenizer_kind;
  std::uint64_t prefix_id = 0;
  std::uint64_t pad_id = 0;
  if (!metadata_string(artifact, "dna_gpt.source_weight_dtype",
                       &source_dtype) ||
      (source_dtype != "F32" && source_dtype != "BF16") ||
      !metadata_string(artifact, "dna_gpt.tokenizer_kind", &tokenizer_kind) ||
      (tokenizer_kind != "static-sixmer" &&
       tokenizer_kind != "dynamic-sixmer") ||
      !metadata_u64(artifact, "dna_gpt.tokenizer_prefix_id", &prefix_id) ||
      prefix_id != 21U ||
      !metadata_u64(artifact, "dna_gpt.tokenizer_pad_id", &pad_id) ||
      pad_id != 20U)
    return format_error(
        "DNA-GPT", "source/tokenizer metadata differs from the pinned ABI");
  if ((tokenizer_kind == "static-sixmer" && config.vocabulary_size != 15659U) ||
      (tokenizer_kind == "dynamic-sixmer" && config.vocabulary_size != 19564U))
    return format_error("DNA-GPT",
                        "tokenizer kind and vocabulary size disagree");
  *output = {config.vocabulary_size, config.width,
             config.layers,          config.heads,
             config.inner_width,     config.maximum_sequence_length,
             config.norm_epsilon};
  return Status::Ok();
}

Status canonical_geneb_dna_gpt_tensors(
    const GenebDnaGptTopology &topology,
    std::vector<GenebDnaGptTensorRequirement> *const output) {
  if (output == nullptr)
    return invalid("DNA-GPT", "tensor manifest output is null");
  auto status = validate_geneb_dna_gpt_topology(topology);
  if (!status.ok())
    return status;
  copy_requirements(canonical_requirements(dna_gpt_config(topology)), output);
  return Status::Ok();
}

GenebDnaGptModel::GenebDnaGptModel() = default;
GenebDnaGptModel::~GenebDnaGptModel() = default;
GenebDnaGptModel::GenebDnaGptModel(GenebDnaGptModel &&) noexcept = default;
GenebDnaGptModel &
GenebDnaGptModel::operator=(GenebDnaGptModel &&) noexcept = default;

Status
GenebDnaGptModel::load(const GenebDnaGptTopology &topology,
                       const std::vector<GenebDnaGptNamedTensorView> &tensors) {
  std::vector<NamedTensorRef> generic;
  generic.reserve(tensors.size());
  for (const auto &item : tensors)
    generic.push_back({item.name,
                       {item.tensor.data, item.tensor.bytes, item.tensor.dtype,
                        item.tensor.shape}});
  Runtime runtime;
  auto status =
      load_runtime(dna_gpt_config(topology), generic, "DNA-GPT", &runtime);
  if (!status.ok())
    return status;
  auto candidate = std::make_unique<Impl>();
  candidate->topology = topology;
  candidate->runtime = std::move(runtime);
  impl_ = std::move(candidate);
  return Status::Ok();
}

Status GenebDnaGptModel::load(const ModelFile &artifact) {
  GenebDnaGptTopology topology;
  auto status = geneb_dna_gpt_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<NamedTensorRef> generic;
  status = model_file_tensors(artifact, &generic);
  if (!status.ok())
    return status;
  Runtime runtime;
  status = load_runtime(dna_gpt_config(topology), generic, "DNA-GPT", &runtime);
  if (!status.ok())
    return status;
  auto candidate = std::make_unique<Impl>();
  candidate->topology = topology;
  candidate->runtime = std::move(runtime);
  impl_ = std::move(candidate);
  return Status::Ok();
}

const GenebDnaGptTopology *GenebDnaGptModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebDnaGptModel::kernel_name() const noexcept {
  return impl_ == nullptr ? "unloaded" : "geneb-dna-gpt-reference-f32";
}

Status GenebDnaGptModel::forward(const std::vector<TokenId> &tokens,
                                 const std::vector<std::uint8_t> &nonpad_mask,
                                 const std::vector<std::size_t> &capture_layers,
                                 GenebDnaGptForwardResult *const output) const {
  if (impl_ == nullptr)
    return invalid("DNA-GPT", "model is not loaded");
  if (output == nullptr)
    return invalid("DNA-GPT", "forward output is null");
  GenericResult generic;
  auto status = forward_runtime(impl_->runtime, tokens, nonpad_mask,
                                capture_layers, "DNA-GPT", &generic);
  if (!status.ok())
    return status;
  GenebDnaGptForwardResult result;
  result.rows = generic.rows;
  result.width = generic.width;
  result.final_hidden = std::move(generic.final_hidden);
  result.pooled = std::move(generic.pooled);
  result.captures.reserve(generic.captures.size());
  for (auto &item : generic.captures)
    result.captures.push_back({item.layer, std::move(item.values)});
  *output = std::move(result);
  return Status::Ok();
}

} // namespace evo::cpu
