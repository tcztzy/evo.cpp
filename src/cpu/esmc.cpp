// SPDX-License-Identifier: Apache-2.0
#include "esmc_internal.hpp"

#include "evo/model_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace evo::cpu::detail {
namespace {

std::uint64_t read_u64(const std::uint8_t *const data) noexcept {
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<std::uint64_t>(data[byte]) << (byte * 8U);
  return value;
}

Status metadata_entry(const ModelFile &artifact, const std::string_view key,
                      const MetadataType type,
                      const MetadataEntry **const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr)
    return {ErrorCode::kModelFormat,
            "required ESMC metadata is missing: " + std::string{key}};
  if (entry->type != type)
    return {ErrorCode::kModelFormat,
            "ESMC metadata has wrong type: " + std::string{key}};
  *output = entry;
  return Status::Ok();
}

Status metadata_size(const ModelFile &artifact, const std::string_view key,
                     std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  const auto value = read_u64(entry->value.data());
  if (value > std::numeric_limits<std::size_t>::max())
    return {ErrorCode::kModelFormat,
            "ESMC metadata exceeds size_t: " + std::string{key}};
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

Status metadata_float(const ModelFile &artifact, const std::string_view key,
                      float *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kF64, &entry);
  if (!status.ok())
    return status;
  const auto bits = read_u64(entry->value.data());
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  if (!std::isfinite(value) || value > std::numeric_limits<float>::max() ||
      value < -std::numeric_limits<float>::max())
    return {ErrorCode::kModelFormat,
            "ESMC metadata is not finite F32: " + std::string{key}};
  *output = static_cast<float>(value);
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

Status metadata_bool(const ModelFile &artifact, const std::string_view key,
                     bool *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kBool, &entry);
  if (!status.ok())
    return status;
  *output = entry->value[0] != 0;
  return Status::Ok();
}

struct TensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t elements{0};

  [[nodiscard]] float at(const std::size_t index) const noexcept {
    float value = 0.0F;
    std::memcpy(&value, data + index * sizeof(float), sizeof(value));
    return value;
  }
};

Status tensor_view(const ModelFile &artifact, const std::string &name,
                   const std::vector<std::size_t> &shape,
                   TensorView *const output) {
  const auto *const tensor = artifact.find_tensor(name);
  if (tensor == nullptr)
    return {ErrorCode::kModelFormat,
            "required ESMC tensor is missing: " + name};
  if (tensor->dtype != TensorDType::kF32 || tensor->rank != shape.size())
    return {ErrorCode::kModelFormat,
            "ESMC tensor dtype/rank mismatch: " + name};
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (tensor->dimensions[index] != shape[index])
      return {ErrorCode::kModelFormat, "ESMC tensor shape mismatch: " + name};
  }
  const auto *const data = artifact.tensor_data(*tensor);
  if (data == nullptr ||
      tensor->element_count > std::numeric_limits<std::size_t>::max())
    return {ErrorCode::kModelFormat,
            "ESMC tensor payload is unavailable: " + name};
  *output = {data, static_cast<std::size_t>(tensor->element_count)};
  return Status::Ok();
}

Status linear(const std::vector<float> &input, const std::size_t rows,
              const std::size_t input_width, const TensorView &weight,
              const std::size_t output_width, const TensorView *const bias,
              std::vector<float> *const output) {
  if (output == nullptr || input.size() != rows * input_width ||
      weight.elements != output_width * input_width ||
      (bias != nullptr && bias->elements != output_width))
    return {ErrorCode::kInternal,
            "ESMC CPU linear dimensions are inconsistent"};
  output->resize(rows * output_width);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t target = 0; target < output_width; ++target) {
      float total = bias == nullptr ? 0.0F : bias->at(target);
      for (std::size_t source = 0; source < input_width; ++source) {
        total += input[row * input_width + source] *
                 weight.at(target * input_width + source);
      }
      (*output)[row * output_width + target] = total;
    }
  }
  return Status::Ok();
}

void layer_norm(const std::vector<float> &input, const std::size_t rows,
                const std::size_t width, const TensorView &weight,
                const TensorView *const bias, const float epsilon,
                std::vector<float> *const output) {
  output->resize(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    float mean = 0.0F;
    for (std::size_t column = 0; column < width; ++column)
      mean += input[row * width + column];
    mean /= static_cast<float>(width);
    float variance = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      const float centered = input[row * width + column] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<float>(width);
    const float inverse = 1.0F / std::sqrt(variance + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      float value =
          (input[row * width + column] - mean) * inverse * weight.at(column);
      if (bias != nullptr)
        value += bias->at(column);
      (*output)[row * width + column] = value;
    }
  }
}

void add_scaled(std::vector<float> *const residual,
                const std::vector<float> &update, const float scale) {
  for (std::size_t index = 0; index < residual->size(); ++index)
    (*residual)[index] += update[index] / scale;
}

} // namespace

struct EsmcModel::Impl final {
  struct Layer final {
    TensorView qkv_norm_weight;
    TensorView qkv_norm_bias;
    TensorView qkv_weight;
    TensorView q_norm_weight;
    TensorView k_norm_weight;
    TensorView attention_output_weight;
    TensorView ffn_norm_weight;
    TensorView ffn_norm_bias;
    TensorView ffn_input_weight;
    TensorView ffn_output_weight;
  };

  ModelConfig public_config;
  const ModelFile *file{nullptr};
  std::size_t heads{0};
  std::size_t inner_width{0};
  float epsilon{0.0F};
  float rope_base{0.0F};
  float residue_scale{0.0F};
  TensorView embedding;
  TensorView final_norm;
  TensorView head_input_weight;
  TensorView head_input_bias;
  TensorView head_norm_weight;
  TensorView head_norm_bias;
  TensorView head_output_weight;
  TensorView head_output_bias;
  std::vector<Layer> layer;

  [[nodiscard]] std::size_t head_width() const noexcept {
    return heads == 0 ? 0 : public_config.width / heads;
  }
};

struct EsmcContext::Impl final {
  std::shared_ptr<EsmcModel::Impl> weights;
  std::size_t capacity{0};
  std::size_t position{0};

  Status attention(const std::vector<float> &hidden, const std::size_t rows,
                   const EsmcModel::Impl::Layer &layer,
                   std::vector<float> *const output) const {
    const std::size_t width = weights->public_config.width;
    const std::size_t head_width = weights->head_width();
    std::vector<float> qkv_normalized;
    layer_norm(hidden, rows, width, layer.qkv_norm_weight, &layer.qkv_norm_bias,
               weights->epsilon, &qkv_normalized);
    std::vector<float> qkv;
    auto status = linear(qkv_normalized, rows, width, layer.qkv_weight,
                         width * 3, nullptr, &qkv);
    if (!status.ok())
      return status;
    std::vector<float> query(rows * width);
    std::vector<float> key(rows * width);
    std::vector<float> value(rows * width);
    for (std::size_t row = 0; row < rows; ++row) {
      std::copy_n(qkv.data() + row * width * 3, width,
                  query.data() + row * width);
      std::copy_n(qkv.data() + row * width * 3 + width, width,
                  key.data() + row * width);
      std::copy_n(qkv.data() + row * width * 3 + width * 2, width,
                  value.data() + row * width);
    }
    std::vector<float> normalized;
    layer_norm(query, rows, width, layer.q_norm_weight, nullptr,
               weights->epsilon, &normalized);
    query.swap(normalized);
    layer_norm(key, rows, width, layer.k_norm_weight, nullptr, weights->epsilon,
               &normalized);
    key.swap(normalized);
    const std::size_t half = head_width / 2;
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t head = 0; head < weights->heads; ++head) {
        const std::size_t base = row * width + head * head_width;
        for (std::size_t pair = 0; pair < half; ++pair) {
          const float exponent =
              static_cast<float>(pair * 2) / static_cast<float>(head_width);
          const float frequency = 1.0F / std::pow(weights->rope_base, exponent);
          const float angle = static_cast<float>(row) * frequency;
          const float cosine = std::cos(angle);
          const float sine = std::sin(angle);
          const auto rotate = [&](std::vector<float> *const tensor) {
            const float first = (*tensor)[base + pair];
            const float second = (*tensor)[base + half + pair];
            (*tensor)[base + pair] = first * cosine - second * sine;
            (*tensor)[base + half + pair] = second * cosine + first * sine;
          };
          rotate(&query);
          rotate(&key);
        }
      }
    }
    std::vector<float> context(rows * width, 0.0F);
    std::vector<float> scores(rows);
    const float score_scale = 1.0F / std::sqrt(static_cast<float>(head_width));
    for (std::size_t target = 0; target < rows; ++target) {
      for (std::size_t head = 0; head < weights->heads; ++head) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t source = 0; source < rows; ++source) {
          float score = 0.0F;
          for (std::size_t dimension = 0; dimension < head_width;
               ++dimension) {
            score += query[target * width + head * head_width + dimension] *
                     key[source * width + head * head_width + dimension];
          }
          score *= score_scale;
          scores[source] = score;
          maximum = std::max(maximum, score);
        }
        float denominator = 0.0F;
        for (float &score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (std::size_t dimension = 0; dimension < head_width; ++dimension) {
          float total = 0.0F;
          for (std::size_t source = 0; source < rows; ++source) {
            total += scores[source] / denominator *
                     value[source * width + head * head_width + dimension];
          }
          context[target * width + head * head_width + dimension] = total;
        }
      }
    }
    return linear(context, rows, width, layer.attention_output_weight, width,
                  nullptr, output);
  }

  Status forward(const std::vector<TokenId> &tokens,
                 std::vector<float> *const logits,
                 const std::size_t capture_layer,
                 std::vector<float> *const embedding_output) {
    if (!weights || tokens.empty() || position != 0 ||
        tokens.size() > capacity ||
        (logits == nullptr && embedding_output == nullptr) ||
        (embedding_output != nullptr &&
         capture_layer > weights->public_config.layers)) {
      return {ErrorCode::kInvalidArgument,
              "ESMC CPU forward requires one fresh full-sequence context"};
    }
    const std::size_t rows = tokens.size();
    const std::size_t width = weights->public_config.width;
    std::vector<float> hidden(rows * width);
    for (std::size_t row = 0; row < rows; ++row) {
      if (tokens[row] >= weights->public_config.vocab_size)
        return {ErrorCode::kInvalidArgument, "token exceeds ESMC vocabulary"};
      for (std::size_t column = 0; column < width; ++column) {
        hidden[row * width + column] = weights->embedding.at(
            static_cast<std::size_t>(tokens[row]) * width + column);
      }
    }
    if (embedding_output != nullptr && capture_layer == 0)
      *embedding_output = hidden;
    std::vector<float> update;
    std::vector<float> normalized;
    std::vector<float> projected;
    std::vector<float> gated;
    for (std::size_t index = 0; index < weights->layer.size(); ++index) {
      const auto &layer = weights->layer[index];
      auto status = attention(hidden, rows, layer, &update);
      if (!status.ok())
        return {status.code(), "ESMC attention block " + std::to_string(index) +
                                   ": " + status.message()};
      add_scaled(&hidden, update, weights->residue_scale);
      layer_norm(hidden, rows, width, layer.ffn_norm_weight,
                 &layer.ffn_norm_bias, weights->epsilon, &normalized);
      status = linear(normalized, rows, width, layer.ffn_input_weight,
                      weights->inner_width * 2, nullptr, &projected);
      if (!status.ok())
        return status;
      gated.resize(rows * weights->inner_width);
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < weights->inner_width; ++column) {
          const float first =
              projected[row * weights->inner_width * 2 + column];
          const float second = projected[row * weights->inner_width * 2 +
                                         weights->inner_width + column];
          gated[row * weights->inner_width + column] =
              first / (1.0F + std::exp(-first)) * second;
        }
      }
      status = linear(gated, rows, weights->inner_width,
                      layer.ffn_output_weight, width, nullptr, &update);
      if (!status.ok())
        return status;
      add_scaled(&hidden, update, weights->residue_scale);
      if (embedding_output != nullptr && capture_layer == index + 1 &&
          capture_layer < weights->public_config.layers) {
        *embedding_output = hidden;
      }
    }
    std::vector<float> final_hidden;
    layer_norm(hidden, rows, width, weights->final_norm, nullptr,
               weights->epsilon, &final_hidden);
    if (embedding_output != nullptr &&
        capture_layer == weights->public_config.layers) {
      *embedding_output = final_hidden;
    }
    if (logits != nullptr) {
      auto status =
          linear(final_hidden, rows, width, weights->head_input_weight, width,
                 &weights->head_input_bias, &projected);
      if (!status.ok())
        return status;
      for (float &value : projected) {
        value =
            0.5F * value * (1.0F + std::erf(value * 0.7071067811865475244F));
      }
      layer_norm(projected, rows, width, weights->head_norm_weight,
                 &weights->head_norm_bias, weights->epsilon, &normalized);
      status = linear(normalized, rows, width, weights->head_output_weight,
                      weights->public_config.vocab_size,
                      &weights->head_output_bias, logits);
      if (!status.ok())
        return status;
    }
    position = rows;
    return Status::Ok();
  }
};

EsmcModel::EsmcModel() : impl_(std::make_shared<Impl>()) {}
EsmcModel::~EsmcModel() = default;

Status EsmcModel::load(const ModelFile &artifact,
                       const bool allow_test_fixture) {
  if (!impl_ || impl_->file != nullptr)
    return {ErrorCode::kInvalidArgument, "ESMC model is already loaded"};
  auto candidate = std::make_shared<Impl>();
  std::string architecture;
  std::string runtime_abi;
  auto status = metadata_string(artifact, "model.architecture", &architecture);
  if (status.ok())
    status = metadata_string(artifact, "runtime.abi", &runtime_abi);
  if (!status.ok())
    return status;
  const auto *const registered = find_architecture(architecture);
  if (registered == nullptr ||
      registered->tokenizer != ArchitectureTokenizer::kEsmcProtein ||
      registered->artifact_profile != artifact.profile() ||
      registered->runtime_abi != runtime_abi) {
    return {ErrorCode::kUnsupported,
            "artifact is not a registered ESMC CPU architecture"};
  }
  candidate->public_config.architecture = architecture;
  candidate->public_config.artifact_profile = std::string{artifact.profile()};
  candidate->public_config.tokenizer = registered->tokenizer;
  candidate->public_config.test_fixture = registered->synthetic_fixture;
  if (registered->synthetic_fixture) {
    bool marked = false;
    status = metadata_bool(artifact, "fixture.synthetic", &marked);
    if (!status.ok())
      return status;
    if (!marked || !allow_test_fixture)
      return {ErrorCode::kUnsupported,
              "synthetic ESMC fixtures require explicit test permission"};
  }
  status =
      metadata_string(artifact, "model.id", &candidate->public_config.model_id);
  if (!status.ok())
    return status;
  for (const auto &item :
       {std::pair{"config.vocab_size", &candidate->public_config.vocab_size},
        std::pair{"config.hidden_size", &candidate->public_config.width},
        std::pair{"config.num_layers", &candidate->public_config.layers},
        std::pair{"config.max_seqlen", &candidate->public_config.max_seqlen},
        std::pair{"config.num_attention_heads", &candidate->heads},
        std::pair{"config.inner_mlp_size", &candidate->inner_width}}) {
    status = metadata_size(artifact, item.first, item.second);
    if (!status.ok())
      return status;
  }
  status = metadata_float(artifact, "config.layer_norm_epsilon",
                          &candidate->epsilon);
  if (status.ok())
    status =
        metadata_float(artifact, "config.rope_base", &candidate->rope_base);
  if (status.ok())
    status = metadata_float(artifact, "config.residue_scaling_factor",
                            &candidate->residue_scale);
  if (!status.ok())
    return status;
  std::size_t embedding_layers = 0;
  status = metadata_size(artifact, "runtime.embedding_layer_count",
                         &embedding_layers);
  if (!status.ok())
    return status;
  if (candidate->public_config.vocab_size != 64 ||
      candidate->public_config.width == 0 ||
      candidate->public_config.layers == 0 || candidate->heads == 0 ||
      candidate->public_config.width % candidate->heads != 0 ||
      candidate->head_width() % 2 != 0 || candidate->inner_width == 0 ||
      candidate->public_config.max_seqlen == 0 ||
      candidate->public_config.max_seqlen > 2048 ||
      embedding_layers != candidate->public_config.layers + 1 ||
      !std::isfinite(candidate->epsilon) || candidate->epsilon <= 0.0F ||
      !std::isfinite(candidate->rope_base) || candidate->rope_base <= 1.0F ||
      !std::isfinite(candidate->residue_scale) ||
      candidate->residue_scale <= 0.0F) {
    return {ErrorCode::kUnsupported, "ESMC CPU dimensions are unsupported"};
  }
  if (!registered->synthetic_fixture) {
    const auto *const official =
        find_official_esmc_model(candidate->public_config.model_id);
    std::string source_repo;
    std::string source_revision;
    status = metadata_string(artifact, "source.repo", &source_repo);
    if (status.ok())
      status = metadata_string(artifact, "source.revision", &source_revision);
    if (!status.ok())
      return status;
    if (official == nullptr || official->huggingface_repo != source_repo ||
        official->huggingface_revision != source_revision ||
        official->vocab_size != candidate->public_config.vocab_size ||
        official->hidden_size != candidate->public_config.width ||
        official->layers != candidate->public_config.layers ||
        official->heads != candidate->heads ||
        official->inner_width != candidate->inner_width ||
        official->max_seqlen != candidate->public_config.max_seqlen) {
      return {ErrorCode::kUnsupported,
              "ESMC artifact does not match a pinned official topology"};
    }
  }
  const std::size_t width = candidate->public_config.width;
  const std::size_t vocabulary = candidate->public_config.vocab_size;
  status = tensor_view(artifact, "esmc.embed.weight", {vocabulary, width},
                       &candidate->embedding);
  if (status.ok())
    status = tensor_view(artifact, "esmc.transformer.norm.weight", {width},
                         &candidate->final_norm);
  if (status.ok())
    status = tensor_view(artifact, "lm_head.0.weight", {width, width},
                         &candidate->head_input_weight);
  if (status.ok())
    status = tensor_view(artifact, "lm_head.0.bias", {width},
                         &candidate->head_input_bias);
  if (status.ok())
    status = tensor_view(artifact, "lm_head.2.weight", {width},
                         &candidate->head_norm_weight);
  if (status.ok())
    status = tensor_view(artifact, "lm_head.2.bias", {width},
                         &candidate->head_norm_bias);
  if (status.ok())
    status = tensor_view(artifact, "lm_head.3.weight", {vocabulary, width},
                         &candidate->head_output_weight);
  if (status.ok())
    status = tensor_view(artifact, "lm_head.3.bias", {vocabulary},
                         &candidate->head_output_bias);
  if (!status.ok())
    return status;
  candidate->layer.resize(candidate->public_config.layers);
  for (std::size_t index = 0; index < candidate->layer.size(); ++index) {
    auto &layer = candidate->layer[index];
    const std::string block =
        "esmc.transformer.blocks." + std::to_string(index);
    status =
        tensor_view(artifact, block + ".attn.layernorm_qkv.layer_norm_weight",
                    {width}, &layer.qkv_norm_weight);
    if (status.ok())
      status =
          tensor_view(artifact, block + ".attn.layernorm_qkv.layer_norm_bias",
                      {width}, &layer.qkv_norm_bias);
    if (status.ok())
      status = tensor_view(artifact, block + ".attn.layernorm_qkv.weight",
                           {width * 3, width}, &layer.qkv_weight);
    if (status.ok())
      status = tensor_view(artifact, block + ".attn.q_ln.weight", {width},
                           &layer.q_norm_weight);
    if (status.ok())
      status = tensor_view(artifact, block + ".attn.k_ln.weight", {width},
                           &layer.k_norm_weight);
    if (status.ok())
      status = tensor_view(artifact, block + ".attn.out_proj.weight",
                           {width, width}, &layer.attention_output_weight);
    if (status.ok())
      status = tensor_view(artifact, block + ".ffn.layer_norm_weight", {width},
                           &layer.ffn_norm_weight);
    if (status.ok())
      status = tensor_view(artifact, block + ".ffn.layer_norm_bias", {width},
                           &layer.ffn_norm_bias);
    if (status.ok())
      status = tensor_view(artifact, block + ".ffn.fc1_weight",
                           {candidate->inner_width * 2, width},
                           &layer.ffn_input_weight);
    if (status.ok())
      status = tensor_view(artifact, block + ".ffn.fc2_weight",
                           {width, candidate->inner_width},
                           &layer.ffn_output_weight);
    if (!status.ok())
      return status;
  }
  candidate->file = &artifact;
  impl_ = std::move(candidate);
  return Status::Ok();
}

const ModelConfig &EsmcModel::config() const noexcept {
  return impl_->public_config;
}

const char *EsmcModel::kernel_name() const noexcept {
  return "scalar-f32-bidirectional";
}

EsmcContext::EsmcContext() : impl_(std::make_unique<Impl>()) {}
EsmcContext::~EsmcContext() = default;

Status EsmcContext::initialize_shared(const EsmcModel &model,
                                      const std::size_t capacity) {
  if (!impl_ || impl_->weights || !model.impl_ ||
      model.impl_->file == nullptr || capacity == 0 ||
      capacity > model.impl_->public_config.max_seqlen) {
    return {ErrorCode::kInvalidArgument,
            "ESMC CPU context requires a loaded model and valid capacity"};
  }
  impl_->weights = model.impl_;
  impl_->capacity = capacity;
  return Status::Ok();
}

Status EsmcContext::prefill(const std::vector<TokenId> &tokens,
                            std::vector<float> *const logits) {
  return impl_->forward(tokens, logits, std::numeric_limits<std::size_t>::max(),
                        nullptr);
}

Status EsmcContext::prefill_embedding(const std::vector<TokenId> &tokens,
                                      const std::size_t layer,
                                      std::vector<float> *const embedding) {
  return impl_->forward(tokens, nullptr, layer, embedding);
}

std::size_t EsmcContext::position() const noexcept { return impl_->position; }

std::size_t EsmcContext::activation_capacity() const noexcept {
  return impl_->capacity;
}

const ModelConfig &EsmcContext::config() const noexcept {
  return impl_->weights->public_config;
}

} // namespace evo::cpu::detail
