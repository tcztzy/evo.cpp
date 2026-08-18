// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_roformer.hpp"

#include "geneb_t36_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace evo::cpu {
namespace {

constexpr std::string_view kFamily = "GENEB RoFormer";
constexpr std::size_t kMaximumTopologyDimension = 1U << 24U;
constexpr std::size_t kMaximumTopologyLayers = 1024U;

std::string layer_prefix(const std::size_t layer) {
  return "roformer.encoder.layer." + std::to_string(layer) + ".";
}

void add_requirement(std::vector<GenebRoformerTensorRequirement> *const output,
                     std::string name, std::vector<std::size_t> shape) {
  output->push_back(
      {std::move(name), TensorDType::kF32, std::move(shape)});
}

Status capture_request(const std::vector<std::size_t> &capture_layers,
                       const std::size_t layers,
                       std::set<std::size_t> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer capture output is null"};
  output->clear();
  for (const std::size_t layer : capture_layers) {
    if (layer > layers || !output->insert(layer).second)
      return {ErrorCode::kInvalidArgument,
              "GENEB RoFormer capture layers are invalid or duplicated"};
  }
  return Status::Ok();
}

void apply_adjacent_rope(std::vector<float> *const values,
                         const std::vector<std::size_t> &positions,
                         const std::size_t heads,
                         const std::size_t head_dimension,
                         const float rope_base) {
  const std::size_t rows = positions.size();
  const std::size_t pairs = head_dimension / 2U;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t base = (row * heads + head) * head_dimension;
      for (std::size_t pair = 0; pair < pairs; ++pair) {
        const float exponent = 2.0F * static_cast<float>(pair) /
                               static_cast<float>(head_dimension);
        const float angle = static_cast<float>(positions[row]) /
                            std::pow(rope_base, exponent);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const std::size_t even = base + pair * 2U;
        const std::size_t odd = even + 1U;
        const float first = (*values)[even];
        const float second = (*values)[odd];
        (*values)[even] = first * cosine - second * sine;
        (*values)[odd] = second * cosine + first * sine;
      }
    }
  }
}

Status online_attention(const std::vector<float> &query,
                        const std::vector<float> &key,
                        const std::vector<float> &value,
                        const std::vector<std::uint8_t> &attention_mask,
                        const std::size_t rows, const std::size_t heads,
                        const std::size_t head_dimension,
                        std::vector<float> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer attention output is null"};
  const std::size_t width = heads * head_dimension;
  std::size_t elements = 0;
  if (!t36::checked_product(rows, width, &elements) ||
      query.size() != elements || key.size() != elements ||
      value.size() != elements || attention_mask.size() != rows)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer attention arguments are inconsistent"};
  const bool any_key = std::any_of(attention_mask.begin(), attention_mask.end(),
                                   [](const std::uint8_t item) {
                                     return item != 0;
                                   });
  if (!any_key)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer attention mask has no active key"};
  output->assign(elements, 0.0F);
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
  for (std::size_t query_row = 0; query_row < rows; ++query_row) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t query_base =
          (query_row * heads + head) * head_dimension;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t key_row = 0; key_row < rows; ++key_row) {
        if (attention_mask[key_row] == 0)
          continue;
        const std::size_t key_base =
            (key_row * heads + head) * head_dimension;
        float score = 0.0F;
        for (std::size_t column = 0; column < head_dimension; ++column)
          score += query[query_base + column] * key[key_base + column];
        maximum = std::max(maximum, score * scale);
      }
      double denominator = 0.0;
      for (std::size_t key_row = 0; key_row < rows; ++key_row) {
        if (attention_mask[key_row] == 0)
          continue;
        const std::size_t key_base =
            (key_row * heads + head) * head_dimension;
        float score = 0.0F;
        for (std::size_t column = 0; column < head_dimension; ++column)
          score += query[query_base + column] * key[key_base + column];
        const float probability = std::exp(score * scale - maximum);
        denominator += probability;
        for (std::size_t column = 0; column < head_dimension; ++column)
          (*output)[query_base + column] +=
              probability * value[key_base + column];
      }
      if (!std::isfinite(denominator) || denominator <= 0.0)
        return {ErrorCode::kInvalidArgument,
                "GENEB RoFormer attention normalization is invalid"};
      const float inverse = 1.0F / static_cast<float>(denominator);
      for (std::size_t column = 0; column < head_dimension; ++column)
        (*output)[query_base + column] *= inverse;
    }
  }
  return t36::finite(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB RoFormer attention became non-finite"};
}

} // namespace

Status
validate_geneb_roformer_topology(const GenebRoformerTopology &topology) {
  const bool dimensions =
      topology.vocabulary_size > 0 &&
      topology.tokenizer_vocabulary_size > 0 &&
      topology.tokenizer_vocabulary_size <= topology.vocabulary_size &&
      topology.width > 0 && topology.layers > 0 &&
      topology.layers <= kMaximumTopologyLayers &&
      topology.attention_heads > 0 && topology.head_dimension > 0 &&
      topology.head_dimension % 2U == 0 &&
      topology.width == topology.attention_heads * topology.head_dimension &&
      topology.inner_width > 0 && topology.maximum_sequence_length >= 2U &&
      topology.token_type_vocabulary_size > 0 &&
      topology.vocabulary_size <= kMaximumTopologyDimension &&
      topology.width <= kMaximumTopologyDimension &&
      topology.inner_width <= kMaximumTopologyDimension &&
      topology.maximum_sequence_length <= kMaximumTopologyDimension;
  if (!dimensions)
    return {ErrorCode::kModelFormat,
            "GENEB RoFormer topology dimensions are invalid"};
  if (topology.pad_token_id >= topology.tokenizer_vocabulary_size ||
      topology.cls_token_id >= topology.tokenizer_vocabulary_size ||
      topology.sep_token_id >= topology.tokenizer_vocabulary_size ||
      topology.pad_token_id == topology.cls_token_id ||
      topology.pad_token_id == topology.sep_token_id ||
      topology.cls_token_id == topology.sep_token_id)
    return {ErrorCode::kModelFormat,
            "GENEB RoFormer special token IDs are invalid"};
  if (!std::isfinite(topology.layer_norm_epsilon) ||
      topology.layer_norm_epsilon <= 0.0F ||
      !std::isfinite(topology.rope_base) || topology.rope_base <= 1.0F ||
      topology.rotary_value || topology.weight_dtype != TensorDType::kF32)
    return {ErrorCode::kModelFormat,
            "GENEB RoFormer frozen numerical semantics differ"};
  return Status::Ok();
}

Status geneb_roformer_topology_from_artifact(
    const ModelFile &artifact, GenebRoformerTopology *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer topology output is null"};
  auto status = t36::metadata_literal(artifact, "runtime.profile",
                                      kGenebRoformerArtifactProfile, kFamily);
  if (!status.ok())
    return status;
  status = t36::metadata_literal(artifact, "runtime.abi",
                                 kGenebRoformerRuntimeAbi, kFamily);
  if (!status.ok())
    return status;
  status = t36::metadata_literal(artifact, "model.architecture",
                                 kGenebRoformerArchitecture, kFamily);
  if (!status.ok())
    return status;
  for (const auto &[key, expected] :
       std::initializer_list<std::pair<std::string_view, std::string_view>>{
           {"roformer.pooling", "attention-mask-mean"},
           {"roformer.manual_special_order", "cls-sep-payload"},
           {"roformer.pad_position_policy", "final-batch-position"},
           {"roformer.hidden_tap", "roformer-last-hidden-state"},
           {"roformer.mask_domain", "attention-mask"},
           {"roformer.special_tokens", "include-manual-cls-sep"}}) {
    status = t36::metadata_literal(artifact, key, expected, kFamily);
    if (!status.ok())
      return status;
  }
  GenebRoformerTopology topology;
  const auto size = [&](const std::string_view key, std::size_t *value) {
    return t36::metadata_size(artifact, key, value, kFamily);
  };
  status = size("roformer.vocab_size", &topology.vocabulary_size);
  if (!status.ok())
    return status;
  status = size("runtime.tokenizer_vocabulary_size",
                &topology.tokenizer_vocabulary_size);
  if (!status.ok())
    return status;
  status = size("roformer.width", &topology.width);
  if (!status.ok())
    return status;
  status = size("roformer.layers", &topology.layers);
  if (!status.ok())
    return status;
  status = size("roformer.heads", &topology.attention_heads);
  if (!status.ok())
    return status;
  status = size("roformer.head_width", &topology.head_dimension);
  if (!status.ok())
    return status;
  status = size("roformer.inner_width", &topology.inner_width);
  if (!status.ok())
    return status;
  status = size("roformer.max_sequence_length",
                &topology.maximum_sequence_length);
  if (!status.ok())
    return status;
  status = size("roformer.type_vocabulary_size",
                &topology.token_type_vocabulary_size);
  if (!status.ok())
    return status;
  std::size_t token = 0;
  status = size("roformer.pad_token_id", &token);
  if (!status.ok() || token > std::numeric_limits<TokenId>::max())
    return status.ok() ? Status{ErrorCode::kModelFormat,
                                "GENEB RoFormer pad token exceeds TokenId"}
                       : status;
  topology.pad_token_id = static_cast<TokenId>(token);
  status = size("roformer.cls_token_id", &token);
  if (!status.ok() || token > std::numeric_limits<TokenId>::max())
    return status.ok() ? Status{ErrorCode::kModelFormat,
                                "GENEB RoFormer CLS token exceeds TokenId"}
                       : status;
  topology.cls_token_id = static_cast<TokenId>(token);
  status = size("roformer.sep_token_id", &token);
  if (!status.ok() || token > std::numeric_limits<TokenId>::max())
    return status.ok() ? Status{ErrorCode::kModelFormat,
                                "GENEB RoFormer SEP token exceeds TokenId"}
                       : status;
  topology.sep_token_id = static_cast<TokenId>(token);
  status = t36::metadata_float(artifact, "roformer.layer_norm_epsilon",
                               &topology.layer_norm_epsilon, kFamily);
  if (!status.ok())
    return status;
  status = t36::metadata_float(artifact, "roformer.rope_base",
                               &topology.rope_base, kFamily);
  if (!status.ok())
    return status;
  status = t36::metadata_bool(artifact, "roformer.rotary_value",
                              &topology.rotary_value, kFamily);
  if (!status.ok())
    return status;
  topology.weight_dtype = TensorDType::kF32;
  status = validate_geneb_roformer_topology(topology);
  if (!status.ok())
    return status;
  *output = topology;
  return Status::Ok();
}

Status canonical_geneb_roformer_tensors(
    const GenebRoformerTopology &topology,
    std::vector<GenebRoformerTensorRequirement> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer tensor requirements output is null"};
  auto status = validate_geneb_roformer_topology(topology);
  if (!status.ok())
    return status;
  output->clear();
  add_requirement(output, "roformer.embeddings.word_embeddings.weight",
                  {topology.vocabulary_size, topology.width});
  add_requirement(output, "roformer.embeddings.token_type_embeddings.weight",
                  {topology.token_type_vocabulary_size, topology.width});
  add_requirement(output, "roformer.embeddings.LayerNorm.weight",
                  {topology.width});
  add_requirement(output, "roformer.embeddings.LayerNorm.bias",
                  {topology.width});
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    const std::string prefix = layer_prefix(layer);
    for (const std::string_view projection : {"query", "key", "value"}) {
      add_requirement(output,
                      prefix + "attention.self." + std::string{projection} +
                          ".weight",
                      {topology.width, topology.width});
      add_requirement(output,
                      prefix + "attention.self." + std::string{projection} +
                          ".bias",
                      {topology.width});
    }
    add_requirement(output, prefix + "attention.output.dense.weight",
                    {topology.width, topology.width});
    add_requirement(output, prefix + "attention.output.dense.bias",
                    {topology.width});
    add_requirement(output, prefix + "attention.output.LayerNorm.weight",
                    {topology.width});
    add_requirement(output, prefix + "attention.output.LayerNorm.bias",
                    {topology.width});
    add_requirement(output, prefix + "intermediate.dense.weight",
                    {topology.inner_width, topology.width});
    add_requirement(output, prefix + "intermediate.dense.bias",
                    {topology.inner_width});
    add_requirement(output, prefix + "output.dense.weight",
                    {topology.width, topology.inner_width});
    add_requirement(output, prefix + "output.dense.bias", {topology.width});
    add_requirement(output, prefix + "output.LayerNorm.weight",
                    {topology.width});
    add_requirement(output, prefix + "output.LayerNorm.bias",
                    {topology.width});
  }
  return Status::Ok();
}

Status geneb_roformer_pool(
    const GenebRoformerForwardResult &forward,
    const std::vector<std::uint8_t> &attention_mask,
    std::vector<float> *const output) {
  if (output == nullptr || forward.rows == 0 || forward.width == 0 ||
      attention_mask.size() != forward.rows ||
      forward.final_hidden.size() != forward.rows * forward.width)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer pooling arguments are inconsistent"};
  output->assign(forward.width, 0.0F);
  std::size_t count = 0;
  for (std::size_t row = 0; row < forward.rows; ++row) {
    if (attention_mask[row] == 0)
      continue;
    ++count;
    for (std::size_t column = 0; column < forward.width; ++column)
      (*output)[column] +=
          forward.final_hidden[row * forward.width + column];
  }
  if (count == 0)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer pooling mask is empty"};
  for (float &value : *output)
    value /= static_cast<float>(count);
  return t36::finite(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB RoFormer pooled output became non-finite"};
}

struct GenebRoformerModel::Impl final {
  struct Layer final {
    GenebRoformerTensorView query_weight;
    GenebRoformerTensorView query_bias;
    GenebRoformerTensorView key_weight;
    GenebRoformerTensorView key_bias;
    GenebRoformerTensorView value_weight;
    GenebRoformerTensorView value_bias;
    GenebRoformerTensorView attention_output_weight;
    GenebRoformerTensorView attention_output_bias;
    GenebRoformerTensorView attention_norm_weight;
    GenebRoformerTensorView attention_norm_bias;
    GenebRoformerTensorView intermediate_weight;
    GenebRoformerTensorView intermediate_bias;
    GenebRoformerTensorView output_weight;
    GenebRoformerTensorView output_bias;
    GenebRoformerTensorView output_norm_weight;
    GenebRoformerTensorView output_norm_bias;
  };

  GenebRoformerTopology topology;
  GenebRoformerTensorView word_embedding;
  GenebRoformerTensorView token_type_embedding;
  GenebRoformerTensorView embedding_norm_weight;
  GenebRoformerTensorView embedding_norm_bias;
  std::vector<Layer> layers;
  std::shared_ptr<evo::detail::LinearExecutor> linear_executor;
};

GenebRoformerModel::GenebRoformerModel() = default;
GenebRoformerModel::~GenebRoformerModel() = default;
GenebRoformerModel::GenebRoformerModel(GenebRoformerModel &&) noexcept =
    default;
GenebRoformerModel &
GenebRoformerModel::operator=(GenebRoformerModel &&) noexcept = default;

Status GenebRoformerModel::load(
    const GenebRoformerTopology &topology,
    const std::vector<GenebRoformerNamedTensorView> &tensors,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  std::vector<GenebRoformerTensorRequirement> requirements;
  auto status = canonical_geneb_roformer_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  std::map<std::string, const GenebRoformerTensorView *, std::less<>> provided;
  for (const auto &named : tensors) {
    if (!provided.emplace(named.name, &named.tensor).second)
      return {ErrorCode::kModelFormat,
              "GENEB RoFormer tensor is duplicated: " + named.name};
  }
  if (provided.size() != requirements.size())
    return {ErrorCode::kModelFormat,
            "GENEB RoFormer tensor set has missing or extra entries"};
  for (const auto &requirement : requirements) {
    const auto found = provided.find(requirement.name);
    if (found == provided.end())
      return {ErrorCode::kModelFormat,
              "GENEB RoFormer tensor is missing: " + requirement.name};
    status = t36::tensor_requirement(*found->second, requirement, kFamily);
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
      view("roformer.embeddings.word_embeddings.weight");
  implementation->token_type_embedding =
      view("roformer.embeddings.token_type_embeddings.weight");
  implementation->embedding_norm_weight =
      view("roformer.embeddings.LayerNorm.weight");
  implementation->embedding_norm_bias =
      view("roformer.embeddings.LayerNorm.bias");
  implementation->layers.resize(topology.layers);
  for (std::size_t index = 0; index < topology.layers; ++index) {
    auto &layer = implementation->layers[index];
    const std::string prefix = layer_prefix(index);
    layer.query_weight = view(prefix + "attention.self.query.weight");
    layer.query_bias = view(prefix + "attention.self.query.bias");
    layer.key_weight = view(prefix + "attention.self.key.weight");
    layer.key_bias = view(prefix + "attention.self.key.bias");
    layer.value_weight = view(prefix + "attention.self.value.weight");
    layer.value_bias = view(prefix + "attention.self.value.bias");
    layer.attention_output_weight =
        view(prefix + "attention.output.dense.weight");
    layer.attention_output_bias =
        view(prefix + "attention.output.dense.bias");
    layer.attention_norm_weight =
        view(prefix + "attention.output.LayerNorm.weight");
    layer.attention_norm_bias =
        view(prefix + "attention.output.LayerNorm.bias");
    layer.intermediate_weight = view(prefix + "intermediate.dense.weight");
    layer.intermediate_bias = view(prefix + "intermediate.dense.bias");
    layer.output_weight = view(prefix + "output.dense.weight");
    layer.output_bias = view(prefix + "output.dense.bias");
    layer.output_norm_weight = view(prefix + "output.LayerNorm.weight");
    layer.output_norm_bias = view(prefix + "output.LayerNorm.bias");
  }
  impl_ = std::move(implementation);
  return Status::Ok();
}

Status GenebRoformerModel::load(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  GenebRoformerTopology topology;
  auto status = geneb_roformer_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebRoformerNamedTensorView> views;
  views.reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t index = 0; index < tensor.rank; ++index) {
      if (tensor.dimensions[index] > std::numeric_limits<std::size_t>::max())
        return {ErrorCode::kModelFormat,
                "GENEB RoFormer tensor dimension exceeds size_t"};
      shape.push_back(static_cast<std::size_t>(tensor.dimensions[index]));
    }
    if (tensor.data_size > std::numeric_limits<std::size_t>::max())
      return {ErrorCode::kModelFormat,
              "GENEB RoFormer tensor extent exceeds size_t"};
    const auto *data = artifact.tensor_data(tensor);
    if (data == nullptr)
      return {ErrorCode::kModelFormat,
              "GENEB RoFormer tensor payload is unavailable"};
    views.push_back({tensor.name,
                     {data, static_cast<std::size_t>(tensor.data_size),
                      tensor.dtype, std::move(shape)}});
  }
  return load(topology, views, std::move(linear_executor));
}

Status GenebRoformerModel::load_artifact(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  return load(artifact, std::move(linear_executor));
}

const GenebRoformerTopology *GenebRoformerModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebRoformerModel::linear_executor_name() const noexcept {
  if (impl_ == nullptr || impl_->linear_executor == nullptr)
    return "scalar-reference";
  return impl_->linear_executor->name();
}

Status GenebRoformerModel::forward(
    const std::vector<TokenId> &tokens,
    const std::vector<std::uint8_t> &attention_mask,
    const std::vector<std::size_t> &capture_layers,
    GenebRoformerForwardResult *const output) const {
  if (impl_ == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer model is not loaded"};
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer forward output is null"};
  const auto &topology = impl_->topology;
  const std::size_t rows = tokens.size();
  if (rows == 0 || rows > topology.maximum_sequence_length ||
      attention_mask.size() != rows)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer input length/mask is invalid"};
  std::set<std::size_t> captures;
  auto status = capture_request(capture_layers, topology.layers, &captures);
  if (!status.ok())
    return status;
  const std::size_t width = topology.width;
  std::vector<float> hidden(rows * width, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    if (attention_mask[row] > 1 || tokens[row] >= topology.vocabulary_size)
      return {ErrorCode::kInvalidArgument,
              "GENEB RoFormer token/mask value is invalid"};
    for (std::size_t column = 0; column < width; ++column) {
      hidden[row * width + column] =
          t36::value(impl_->word_embedding,
                     static_cast<std::size_t>(tokens[row]) * width + column) +
          t36::value(impl_->token_type_embedding, column);
    }
  }
  std::vector<float> normalized;
  status = t36::layer_norm(hidden, rows, width,
                           impl_->embedding_norm_weight,
                           impl_->embedding_norm_bias,
                           topology.layer_norm_epsilon, &normalized, kFamily);
  if (!status.ok())
    return status;
  hidden.swap(normalized);
  output->rows = rows;
  output->width = width;
  output->captures.clear();
  if (captures.count(0) != 0)
    output->captures.push_back({0, hidden});
  std::vector<std::size_t> positions(rows, rows - 1U);
  for (std::size_t row = 0; row < rows; ++row) {
    if (attention_mask[row] != 0)
      positions[row] = row;
  }
  for (std::size_t layer_index = 0; layer_index < topology.layers;
       ++layer_index) {
    const auto &layer = impl_->layers[layer_index];
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    status = t36::linear(hidden, rows, width, layer.query_weight, width,
                         &layer.query_bias, impl_->linear_executor.get(),
                         &query, kFamily);
    if (!status.ok())
      return status;
    status = t36::linear(hidden, rows, width, layer.key_weight, width,
                         &layer.key_bias, impl_->linear_executor.get(), &key,
                         kFamily);
    if (!status.ok())
      return status;
    status = t36::linear(hidden, rows, width, layer.value_weight, width,
                         &layer.value_bias, impl_->linear_executor.get(),
                         &value, kFamily);
    if (!status.ok())
      return status;
    apply_adjacent_rope(&query, positions, topology.attention_heads,
                        topology.head_dimension, topology.rope_base);
    apply_adjacent_rope(&key, positions, topology.attention_heads,
                        topology.head_dimension, topology.rope_base);
    if (topology.rotary_value)
      apply_adjacent_rope(&value, positions, topology.attention_heads,
                          topology.head_dimension, topology.rope_base);
    std::vector<float> context;
    status = online_attention(query, key, value, attention_mask, rows,
                              topology.attention_heads,
                              topology.head_dimension, &context);
    if (!status.ok())
      return status;
    std::vector<float> attention_update;
    status = t36::linear(context, rows, width,
                         layer.attention_output_weight, width,
                         &layer.attention_output_bias,
                         impl_->linear_executor.get(), &attention_update,
                         kFamily);
    if (!status.ok())
      return status;
    for (std::size_t index = 0; index < hidden.size(); ++index)
      attention_update[index] += hidden[index];
    std::vector<float> attention_hidden;
    status = t36::layer_norm(attention_update, rows, width,
                             layer.attention_norm_weight,
                             layer.attention_norm_bias,
                             topology.layer_norm_epsilon, &attention_hidden,
                             kFamily);
    if (!status.ok())
      return status;
    std::vector<float> intermediate;
    status = t36::linear(attention_hidden, rows, width,
                         layer.intermediate_weight, topology.inner_width,
                         &layer.intermediate_bias,
                         impl_->linear_executor.get(), &intermediate, kFamily);
    if (!status.ok())
      return status;
    for (float &item : intermediate)
      item = t36::erf_gelu(item);
    std::vector<float> feed_forward;
    status = t36::linear(intermediate, rows, topology.inner_width,
                         layer.output_weight, width, &layer.output_bias,
                         impl_->linear_executor.get(), &feed_forward, kFamily);
    if (!status.ok())
      return status;
    for (std::size_t index = 0; index < feed_forward.size(); ++index)
      feed_forward[index] += attention_hidden[index];
    status = t36::layer_norm(feed_forward, rows, width,
                             layer.output_norm_weight,
                             layer.output_norm_bias,
                             topology.layer_norm_epsilon, &hidden, kFamily);
    if (!status.ok())
      return status;
    if (captures.count(layer_index + 1U) != 0)
      output->captures.push_back({layer_index + 1U, hidden});
  }
  output->final_hidden = std::move(hidden);
  return Status::Ok();
}

Status GenebRoformerModel::forward_payload(
    const std::vector<TokenId> &payload,
    const std::vector<std::size_t> &capture_layers,
    GenebRoformerForwardResult *const output) const {
  if (impl_ == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB RoFormer model is not loaded"};
  const std::size_t maximum_payload =
      impl_->topology.maximum_sequence_length - 2U;
  const std::size_t retained = std::min(payload.size(), maximum_payload);
  std::vector<TokenId> tokens;
  tokens.reserve(retained + 2U);
  tokens.push_back(impl_->topology.cls_token_id);
  tokens.push_back(impl_->topology.sep_token_id);
  tokens.insert(tokens.end(), payload.begin(),
                payload.begin() + static_cast<std::ptrdiff_t>(retained));
  std::vector<std::uint8_t> attention_mask(tokens.size(), 1);
  return forward(tokens, attention_mask, capture_layers, output);
}

Status GenebRoformerModel::pool(
    const GenebRoformerForwardResult &forward,
    const std::vector<std::uint8_t> &attention_mask,
    std::vector<float> *const output) const {
  return geneb_roformer_pool(forward, attention_mask, output);
}

} // namespace evo::cpu
