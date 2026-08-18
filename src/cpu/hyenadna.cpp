// SPDX-License-Identifier: Apache-2.0
#include "hyenadna_internal.hpp"

#include "../linear_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "evo/model_registry.hpp"

namespace evo::cpu::detail {
namespace {

constexpr std::size_t kMaximumDirectContext = 4096;
constexpr std::size_t kActivationChunk = 256;

Status metadata_entry(const ModelFile &model, const std::string_view key,
                      const MetadataType type,
                      const MetadataEntry **const output) {
  const auto *const entry = model.find_metadata(key);
  if (entry == nullptr)
    return {ErrorCode::kModelFormat,
            "HyenaDNA required metadata is missing: " + std::string{key}};
  if (entry->type != type)
    return {ErrorCode::kModelFormat,
            "HyenaDNA metadata has wrong type: " + std::string{key}};
  *output = entry;
  return Status::Ok();
}

Status metadata_size(const ModelFile &model, const std::string_view key,
                     std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(std::uint64_t))
    return {ErrorCode::kModelFormat, "HyenaDNA u64 metadata is malformed"};
  std::uint64_t value = 0;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (value > std::numeric_limits<std::size_t>::max())
    return {ErrorCode::kModelFormat, "HyenaDNA dimension exceeds size_t"};
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

Status metadata_float(const ModelFile &model, const std::string_view key,
                      float *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kF64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(std::uint64_t))
    return {ErrorCode::kModelFormat, "HyenaDNA f64 metadata is malformed"};
  double value = 0.0;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (!std::isfinite(value) || value < -std::numeric_limits<float>::max() ||
      value > std::numeric_limits<float>::max()) {
    return {ErrorCode::kModelFormat, "HyenaDNA f64 metadata is not finite F32"};
  }
  *output = static_cast<float>(value);
  return Status::Ok();
}

Status metadata_string(const ModelFile &model, const std::string_view key,
                       std::string *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kString, &entry);
  if (!status.ok())
    return status;
  output->assign(entry->value.begin(), entry->value.end());
  return Status::Ok();
}

Status metadata_bool(const ModelFile &model, const std::string_view key,
                     bool *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kBool, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != 1)
    return {ErrorCode::kModelFormat, "HyenaDNA bool metadata is malformed"};
  *output = entry->value[0] != 0;
  return Status::Ok();
}

struct TensorView final {
  const float *data{nullptr};
  std::size_t count{0};

  [[nodiscard]] float at(const std::size_t index) const noexcept {
    return data[index];
  }
};

Status tensor_view(const ModelFile &model, const std::string &name,
                   const std::vector<std::size_t> &shape,
                   TensorView *const output) {
  const auto *const tensor = model.find_tensor(name);
  if (tensor == nullptr)
    return {ErrorCode::kModelFormat,
            "HyenaDNA tensor is missing: " + name};
  if (tensor->dtype != TensorDType::kF32 || tensor->rank != shape.size())
    return {ErrorCode::kModelFormat,
            "HyenaDNA tensor dtype/rank mismatch: " + name};
  std::size_t count = 1;
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (tensor->dimensions[index] != shape[index] ||
        shape[index] > std::numeric_limits<std::size_t>::max() / count) {
      return {ErrorCode::kModelFormat,
              "HyenaDNA tensor shape mismatch: " + name};
    }
    count *= shape[index];
  }
  const auto *const bytes = model.tensor_data(*tensor);
  if (bytes == nullptr || tensor->data_size != count * sizeof(float))
    return {ErrorCode::kModelFormat,
            "HyenaDNA tensor payload mismatch: " + name};
  output->data = reinterpret_cast<const float *>(bytes);
  output->count = count;
  return Status::Ok();
}

void linear(const std::vector<float> &input, const TensorView &weight,
            const std::size_t outputs, const TensorView *const bias,
            std::vector<float> *const result) {
  const std::size_t inputs = input.size();
  result->assign(outputs, 0.0F);
  for (std::size_t target = 0; target < outputs; ++target) {
    float value = bias == nullptr ? 0.0F : bias->at(target);
    const std::size_t offset = target * inputs;
    for (std::size_t source = 0; source < inputs; ++source)
      value += input[source] * weight.at(offset + source);
    (*result)[target] = value;
  }
}

Status runtime_linear(const std::vector<float> &input,
                      const TensorView &weight, const std::size_t outputs,
                      const TensorView *const bias,
                      evo::detail::LinearExecutor *const executor,
                      std::vector<float> *const result) {
  if (result == nullptr || input.empty() ||
      weight.count != outputs * input.size() ||
      (bias != nullptr && bias->count != outputs)) {
    return {ErrorCode::kInternal,
            "HyenaDNA linear dimensions are inconsistent"};
  }
  if (executor != nullptr) {
    const evo::detail::LinearTensorView weight_view{
        reinterpret_cast<const std::uint8_t *>(weight.data),
        TensorDType::kF32, weight.count};
    evo::detail::LinearTensorView bias_view;
    const evo::detail::LinearTensorView *bias_pointer = nullptr;
    if (bias != nullptr) {
      bias_view = {reinterpret_cast<const std::uint8_t *>(bias->data),
                   TensorDType::kF32, bias->count};
      bias_pointer = &bias_view;
    }
    return executor->linear(input.data(), 1, input.size(), weight_view, outputs,
                            bias_pointer, result);
  }
  linear(input, weight, outputs, bias, result);
  return Status::Ok();
}

void layer_norm(const std::vector<float> &input, const TensorView &weight,
                const TensorView &bias, const float epsilon,
                std::vector<float> *const output) {
  float mean = 0.0F;
  for (const float value : input)
    mean += value;
  mean /= static_cast<float>(input.size());
  float variance = 0.0F;
  for (const float value : input) {
    const float centered = value - mean;
    variance += centered * centered;
  }
  variance /= static_cast<float>(input.size());
  const float inverse = 1.0F / std::sqrt(variance + epsilon);
  output->resize(input.size());
  for (std::size_t index = 0; index < input.size(); ++index)
    (*output)[index] =
        (input[index] - mean) * inverse * weight.at(index) + bias.at(index);
}

float gelu(const float value) noexcept {
  constexpr float factor = 0.7978845608028654F;
  return 0.5F * value *
         (1.0F + std::tanh(factor * (value + 0.044715F * value * value * value)));
}

} // namespace

struct HyenaDnaModel::Impl final {
  struct Layer final {
    TensorView norm1_weight;
    TensorView norm1_bias;
    TensorView norm2_weight;
    TensorView norm2_bias;
    TensorView in_weight;
    TensorView in_bias;
    TensorView short_weight;
    TensorView short_bias;
    TensorView filter_bias;
    TensorView filter_input_weight;
    TensorView filter_input_bias;
    TensorView filter_frequency;
    std::vector<TensorView> filter_inner_weight;
    std::vector<TensorView> filter_inner_bias;
    TensorView filter_output_weight;
    TensorView filter_deltas;
    TensorView filter_time;
    TensorView filter_position;
    TensorView out_weight;
    TensorView out_bias;
    TensorView mlp_up_weight;
    TensorView mlp_up_bias;
    TensorView mlp_down_weight;
    TensorView mlp_down_bias;
    std::vector<float> kernel;
  };

  const ModelFile *file{nullptr};
  ModelConfig public_config;
  std::size_t inner_width{0};
  std::size_t embedding_dimension{0};
  std::size_t filter_order{0};
  std::size_t inner_filter_layers{0};
  std::size_t short_filter_length{0};
  float epsilon{0.0F};
  TensorView embedding;
  TensorView final_norm_weight;
  TensorView final_norm_bias;
  TensorView head;
  std::vector<Layer> layer;
  std::shared_ptr<evo::detail::LinearExecutor> executor;

  Status build_kernel(Layer *const target) const {
    const std::size_t width = public_config.width;
    target->kernel.resize(public_config.max_seqlen * width);
    std::vector<float> activation;
    std::vector<float> next;
    for (std::size_t position = 0; position < public_config.max_seqlen;
         ++position) {
      std::vector<float> encoded(embedding_dimension);
      for (std::size_t index = 0; index < embedding_dimension; ++index) {
        encoded[index] = target->filter_position.at(
            position * embedding_dimension + index);
      }
      linear(encoded, target->filter_input_weight, filter_order,
             &target->filter_input_bias, &activation);
      for (std::size_t index = 0; index < filter_order; ++index)
        activation[index] =
            std::sin(target->filter_frequency.at(index) * activation[index]);
      for (std::size_t inner = 0; inner < inner_filter_layers; ++inner) {
        linear(activation, target->filter_inner_weight[inner], filter_order,
               &target->filter_inner_bias[inner], &next);
        for (std::size_t index = 0; index < filter_order; ++index)
          next[index] =
              std::sin(target->filter_frequency.at(index) * next[index]);
        activation.swap(next);
      }
      linear(activation, target->filter_output_weight, width, nullptr, &next);
      const float time = target->filter_time.at(position);
      for (std::size_t channel = 0; channel < width; ++channel) {
        const float modulation =
            std::exp(-time * std::abs(target->filter_deltas.at(channel))) +
            0.05F;
        target->kernel[position * width + channel] =
            next[channel] * modulation;
      }
    }
    return Status::Ok();
  }
};

struct HyenaDnaContext::Impl final {
  struct LayerState final {
    std::vector<float> short_history;
    std::vector<float> gated_history;
  };

  std::shared_ptr<HyenaDnaModel::Impl> weights;
  std::size_t capacity{0};
  std::size_t position{0};
  std::vector<LayerState> state;

  Status step(const TokenId token, const std::size_t capture_layer,
              float *const logits, float *const capture) {
    if (!weights ||
        !token_id_in_vocabulary(token, weights->public_config.vocab_size) ||
        position >= capacity) {
      return {ErrorCode::kInvalidArgument,
              "HyenaDNA token/context is outside its configured bounds"};
    }
    const std::size_t width = weights->public_config.width;
    const std::size_t projected_width = width * 3;
    std::vector<float> hidden(width);
    for (std::size_t index = 0; index < width; ++index)
      hidden[index] = weights->embedding.at(static_cast<std::size_t>(token) *
                                                width +
                                            index);
    std::vector<float> normalized;
    std::vector<float> projected;
    std::vector<float> filtered(projected_width);
    std::vector<float> gated(width);
    std::vector<float> mixed(width);
    std::vector<float> output;
    std::vector<float> mlp;
    for (std::size_t layer_index = 0; layer_index < weights->layer.size();
         ++layer_index) {
      const auto &layer = weights->layer[layer_index];
      auto &layer_state = state[layer_index];
      const std::vector<float> residual = hidden;
      layer_norm(hidden, layer.norm1_weight, layer.norm1_bias,
                 weights->epsilon, &normalized);
      auto status = runtime_linear(normalized, layer.in_weight,
                                   projected_width, &layer.in_bias,
                                   weights->executor.get(), &projected);
      if (!status.ok())
        return status;
      for (std::size_t channel = 0; channel < projected_width; ++channel) {
        float value = layer.short_bias.at(channel) +
                      projected[channel] *
                          layer.short_weight.at(channel * 3 + 2);
        if (position != 0) {
          const std::size_t previous = ((position - 1) % 2) * projected_width;
          value += layer_state.short_history[previous + channel] *
                   layer.short_weight.at(channel * 3 + 1);
        }
        if (position > 1) {
          const std::size_t older = ((position - 2) % 2) * projected_width;
          value += layer_state.short_history[older + channel] *
                   layer.short_weight.at(channel * 3);
        }
        filtered[channel] = value;
      }
      std::copy(projected.begin(), projected.end(),
                layer_state.short_history.begin() +
                    static_cast<std::ptrdiff_t>((position % 2) *
                                                projected_width));
      for (std::size_t channel = 0; channel < width; ++channel) {
        gated[channel] = filtered[width * 2 + channel] *
                         filtered[width + channel];
      }
      layer_state.gated_history.insert(layer_state.gated_history.end(),
                                       gated.begin(), gated.end());
      for (std::size_t channel = 0; channel < width; ++channel) {
        float value = layer.filter_bias.at(channel) * gated[channel];
        for (std::size_t lag = 0; lag <= position; ++lag) {
          value += layer.kernel[lag * width + channel] *
                   layer_state.gated_history[(position - lag) * width +
                                             channel];
        }
        mixed[channel] = value * filtered[channel];
      }
      status = runtime_linear(mixed, layer.out_weight, width, &layer.out_bias,
                              weights->executor.get(), &output);
      if (!status.ok())
        return status;
      for (std::size_t index = 0; index < width; ++index)
        hidden[index] = output[index] + residual[index];
      const std::vector<float> second_residual = hidden;
      layer_norm(hidden, layer.norm2_weight, layer.norm2_bias,
                 weights->epsilon, &normalized);
      status = runtime_linear(normalized, layer.mlp_up_weight,
                              weights->inner_width, &layer.mlp_up_bias,
                              weights->executor.get(), &mlp);
      if (!status.ok())
        return status;
      for (float &value : mlp)
        value = gelu(value);
      status = runtime_linear(mlp, layer.mlp_down_weight, width,
                              &layer.mlp_down_bias, weights->executor.get(),
                              &output);
      if (!status.ok())
        return status;
      for (std::size_t index = 0; index < width; ++index)
        hidden[index] = output[index] + second_residual[index];
      if (capture != nullptr && capture_layer == layer_index)
        std::copy(hidden.begin(), hidden.end(), capture);
    }
    if (logits != nullptr) {
      layer_norm(hidden, weights->final_norm_weight, weights->final_norm_bias,
                 weights->epsilon, &normalized);
      const std::size_t vocabulary = weights->public_config.vocab_size;
      std::vector<float> head_output;
      const auto status = runtime_linear(normalized, weights->head, vocabulary,
                                         nullptr, weights->executor.get(),
                                         &head_output);
      if (!status.ok())
        return status;
      std::copy(head_output.begin(), head_output.end(), logits);
    }
    ++position;
    return Status::Ok();
  }

  Status forward(const std::vector<TokenId> &tokens, const bool initial,
                 std::vector<float> *const logits,
                 const std::size_t capture_layer,
                 std::vector<float> *const embedding) {
    if (!weights || tokens.empty() || position + tokens.size() > capacity ||
        (initial && position != 0) || (!initial && position == 0) ||
        (embedding != nullptr && capture_layer >= weights->layer.size())) {
      return {ErrorCode::kInvalidArgument,
              "HyenaDNA forward arguments are invalid"};
    }
    const std::size_t width = weights->public_config.width;
    const std::size_t vocabulary = weights->public_config.vocab_size;
    if (logits != nullptr)
      logits->resize(tokens.size() * vocabulary);
    if (embedding != nullptr)
      embedding->resize(tokens.size() * width);
    for (std::size_t row = 0; row < tokens.size(); ++row) {
      auto status = step(tokens[row], capture_layer,
                         logits == nullptr ? nullptr
                                           : logits->data() + row * vocabulary,
                         embedding == nullptr
                             ? nullptr
                             : embedding->data() + row * width);
      if (!status.ok())
        return status;
    }
    return Status::Ok();
  }
};

HyenaDnaModel::HyenaDnaModel() : impl_(std::make_shared<Impl>()) {}
HyenaDnaModel::~HyenaDnaModel() = default;

Status HyenaDnaModel::load(const ModelFile &artifact,
                           const bool allow_test_fixture,
                           std::shared_ptr<evo::detail::LinearExecutor> executor) {
  if (!impl_ || impl_->file != nullptr)
    return {ErrorCode::kInvalidArgument, "HyenaDNA model is already loaded"};
  auto candidate = std::make_shared<Impl>();
  candidate->executor = std::move(executor);
  std::string architecture;
  std::string runtime_abi;
  auto status = metadata_string(artifact, "model.architecture", &architecture);
  if (status.ok())
    status = metadata_string(artifact, "runtime.abi", &runtime_abi);
  if (!status.ok())
    return status;
  const auto *const registered = find_architecture(architecture);
  if (registered == nullptr ||
      registered->implementation != ArchitectureImplementation::kHyenaDna ||
      registered->artifact_profile != artifact.profile() ||
      registered->runtime_abi != runtime_abi) {
    return {ErrorCode::kUnsupported,
            "artifact is not a registered HyenaDNA host architecture"};
  }
  candidate->public_config.architecture = architecture;
  candidate->public_config.artifact_profile = std::string{artifact.profile()};
  candidate->public_config.implementation = registered->implementation;
  candidate->public_config.tokenizer = registered->tokenizer;
  candidate->public_config.test_fixture = registered->synthetic_fixture;
  if (registered->synthetic_fixture) {
    bool marked = false;
    status = metadata_bool(artifact, "fixture.synthetic", &marked);
    if (!status.ok())
      return status;
    if (!marked || !allow_test_fixture)
      return {ErrorCode::kUnsupported,
              "synthetic HyenaDNA fixtures require explicit test permission"};
  }
  if (artifact.find_metadata("model.id") != nullptr) {
    status = metadata_string(artifact, "model.id",
                             &candidate->public_config.model_id);
  } else {
    status = metadata_string(artifact, "model.name",
                             &candidate->public_config.model_id);
  }
  if (!status.ok())
    return status;
  for (const auto &item :
       {std::pair{"config.vocab_size", &candidate->public_config.vocab_size},
        std::pair{"config.hidden_size", &candidate->public_config.width},
        std::pair{"config.num_layers", &candidate->public_config.layers},
        std::pair{"config.max_seqlen", &candidate->public_config.max_seqlen},
        std::pair{"config.inner_mlp_size", &candidate->inner_width},
        std::pair{"config.embedding_dim", &candidate->embedding_dimension},
        std::pair{"config.filter_order", &candidate->filter_order},
        std::pair{"config.num_inner_mlps", &candidate->inner_filter_layers},
        std::pair{"config.short_filter_length",
                  &candidate->short_filter_length}}) {
    status = metadata_size(artifact, item.first, item.second);
    if (!status.ok())
      return status;
  }
  status = metadata_float(artifact, "config.layer_norm_epsilon",
                          &candidate->epsilon);
  if (!status.ok())
    return status;
  std::size_t hyena_order = 0;
  status = metadata_size(artifact, "config.hyena_order", &hyena_order);
  if (!status.ok())
    return status;
  if (candidate->public_config.vocab_size != 16 ||
      candidate->public_config.width == 0 ||
      candidate->public_config.layers == 0 || candidate->inner_width == 0 ||
      candidate->embedding_dimension < 3 ||
      candidate->embedding_dimension % 2 == 0 ||
      candidate->filter_order == 0 || candidate->inner_filter_layers == 0 ||
      candidate->short_filter_length != 3 || hyena_order != 2 ||
      candidate->public_config.max_seqlen == 0 ||
      candidate->public_config.max_seqlen > kMaximumDirectContext ||
      !std::isfinite(candidate->epsilon) || candidate->epsilon <= 0.0F) {
    return {ErrorCode::kUnsupported,
            "HyenaDNA host direct-convolution dimensions are unsupported"};
  }
  const std::size_t width = candidate->public_config.width;
  const std::size_t vocabulary = candidate->public_config.vocab_size;
  status = tensor_view(artifact,
                       "hyena.backbone.embeddings.word_embeddings.weight",
                       {vocabulary, width}, &candidate->embedding);
  if (status.ok())
    status = tensor_view(artifact, "hyena.backbone.ln_f.weight", {width},
                         &candidate->final_norm_weight);
  if (status.ok())
    status = tensor_view(artifact, "hyena.backbone.ln_f.bias", {width},
                         &candidate->final_norm_bias);
  if (status.ok())
    status = tensor_view(artifact, "lm_head.weight", {vocabulary, width},
                         &candidate->head);
  if (!status.ok())
    return status;
  candidate->layer.resize(candidate->public_config.layers);
  for (std::size_t index = 0; index < candidate->layer.size(); ++index) {
    auto &layer = candidate->layer[index];
    const std::string prefix =
        "hyena.backbone.layers." + std::to_string(index);
    const std::string mixer = prefix + ".mixer";
    const std::string filter = mixer + ".filter_fn";
    for (const auto &item :
         {std::pair{prefix + ".norm1.weight", &layer.norm1_weight},
          std::pair{prefix + ".norm1.bias", &layer.norm1_bias},
          std::pair{prefix + ".norm2.weight", &layer.norm2_weight},
          std::pair{prefix + ".norm2.bias", &layer.norm2_bias}}) {
      status = tensor_view(artifact, item.first, {width}, item.second);
      if (!status.ok())
        return status;
    }
    const std::size_t projected = width * 3;
    for (const auto &item :
         {std::pair{mixer + ".in_proj.weight", &layer.in_weight},
          std::pair{mixer + ".out_proj.weight", &layer.out_weight}}) {
      const auto shape = item.second == &layer.in_weight
                             ? std::vector<std::size_t>{projected, width}
                             : std::vector<std::size_t>{width, width};
      status = tensor_view(artifact, item.first, shape, item.second);
      if (!status.ok())
        return status;
    }
    for (const auto &item :
         {std::pair{mixer + ".in_proj.bias", &layer.in_bias},
          std::pair{mixer + ".out_proj.bias", &layer.out_bias}}) {
      const auto size = item.second == &layer.in_bias ? projected : width;
      status = tensor_view(artifact, item.first, {size}, item.second);
      if (!status.ok())
        return status;
    }
    status = tensor_view(artifact, mixer + ".short_filter.weight",
                         {projected, 1, 3}, &layer.short_weight);
    if (status.ok())
      status = tensor_view(artifact, mixer + ".short_filter.bias", {projected},
                           &layer.short_bias);
    if (status.ok())
      status = tensor_view(artifact, filter + ".bias", {width},
                           &layer.filter_bias);
    if (status.ok())
      status = tensor_view(
          artifact, filter + ".implicit_filter.0.weight",
          {candidate->filter_order, candidate->embedding_dimension},
          &layer.filter_input_weight);
    if (status.ok())
      status = tensor_view(artifact, filter + ".implicit_filter.0.bias",
                           {candidate->filter_order},
                           &layer.filter_input_bias);
    if (status.ok())
      status = tensor_view(artifact, filter + ".implicit_filter.1.freq",
                           {1, candidate->filter_order},
                           &layer.filter_frequency);
    if (!status.ok())
      return status;
    layer.filter_inner_weight.resize(candidate->inner_filter_layers);
    layer.filter_inner_bias.resize(candidate->inner_filter_layers);
    for (std::size_t inner = 0; inner < candidate->inner_filter_layers;
         ++inner) {
      const std::string module = std::to_string(2 + inner * 2);
      status = tensor_view(artifact,
                           filter + ".implicit_filter." + module + ".weight",
                           {candidate->filter_order, candidate->filter_order},
                           &layer.filter_inner_weight[inner]);
      if (status.ok())
        status = tensor_view(
            artifact, filter + ".implicit_filter." + module + ".bias",
            {candidate->filter_order}, &layer.filter_inner_bias[inner]);
      if (!status.ok())
        return status;
    }
    const std::string output_module =
        std::to_string(2 + candidate->inner_filter_layers * 2);
    status = tensor_view(
        artifact, filter + ".implicit_filter." + output_module + ".weight",
        {width, candidate->filter_order}, &layer.filter_output_weight);
    if (status.ok())
      status = tensor_view(artifact, filter + ".modulation.deltas",
                           {1, 1, width}, &layer.filter_deltas);
    if (status.ok())
      status = tensor_view(artifact, filter + ".pos_emb.t",
                           {1, candidate->public_config.max_seqlen, 1},
                           &layer.filter_time);
    if (status.ok())
      status = tensor_view(
          artifact, filter + ".pos_emb.z",
          {1, candidate->public_config.max_seqlen,
           candidate->embedding_dimension},
          &layer.filter_position);
    if (status.ok())
      status = tensor_view(artifact, prefix + ".mlp.fc1.weight",
                           {candidate->inner_width, width},
                           &layer.mlp_up_weight);
    if (status.ok())
      status = tensor_view(artifact, prefix + ".mlp.fc1.bias",
                           {candidate->inner_width}, &layer.mlp_up_bias);
    if (status.ok())
      status = tensor_view(artifact, prefix + ".mlp.fc2.weight",
                           {width, candidate->inner_width},
                           &layer.mlp_down_weight);
    if (status.ok())
      status = tensor_view(artifact, prefix + ".mlp.fc2.bias", {width},
                           &layer.mlp_down_bias);
    if (!status.ok())
      return status;
    status = candidate->build_kernel(&layer);
    if (!status.ok())
      return status;
  }
  candidate->file = &artifact;
  impl_ = std::move(candidate);
  return Status::Ok();
}

const ModelConfig &HyenaDnaModel::config() const noexcept {
  return impl_->public_config;
}

const char *HyenaDnaModel::kernel_name() const noexcept {
  return impl_->executor ? impl_->executor->name() : "direct-convolution-f32";
}

Status HyenaDnaModel::encode(const std::string_view sequence,
                             std::vector<TokenId> *const tokens) const {
  if (tokens == nullptr)
    return {ErrorCode::kInvalidArgument, "HyenaDNA token output is null"};
  tokens->clear();
  tokens->reserve(sequence.size());
  for (const char character : sequence) {
    TokenId token = 6;
    switch (character) {
    case 'A':
      token = 7;
      break;
    case 'C':
      token = 8;
      break;
    case 'G':
      token = 9;
      break;
    case 'T':
      token = 10;
      break;
    case 'N':
      token = 11;
      break;
    default:
      break;
    }
    tokens->push_back(token);
  }
  return Status::Ok();
}

Status HyenaDnaModel::decode_token(const TokenId token,
                                   std::uint8_t *const byte) const {
  if (byte == nullptr)
    return {ErrorCode::kInvalidArgument, "HyenaDNA byte output is null"};
  constexpr char alphabet[] = "ACGTN";
  if (token < 7 || token > 11)
    return {ErrorCode::kModelFormat,
            "HyenaDNA sampled a special or padded token without a DNA byte"};
  *byte = static_cast<std::uint8_t>(alphabet[token - 7]);
  return Status::Ok();
}

HyenaDnaContext::HyenaDnaContext() : impl_(std::make_unique<Impl>()) {}
HyenaDnaContext::~HyenaDnaContext() = default;

Status HyenaDnaContext::initialize_shared(const HyenaDnaModel &model,
                                          const std::size_t capacity) {
  if (!impl_ || impl_->weights || !model.impl_ ||
      model.impl_->file == nullptr || capacity == 0 ||
      capacity > model.impl_->public_config.max_seqlen) {
    return {ErrorCode::kInvalidArgument,
            "HyenaDNA context requires loaded weights and valid capacity"};
  }
  impl_->weights = model.impl_;
  impl_->capacity = capacity;
  impl_->state.resize(model.impl_->layer.size());
  const std::size_t projected = model.impl_->public_config.width * 3;
  for (auto &state : impl_->state) {
    state.short_history.assign(projected * 2, 0.0F);
    state.gated_history.reserve(capacity * model.impl_->public_config.width);
  }
  return Status::Ok();
}

Status HyenaDnaContext::prefill(const std::vector<TokenId> &tokens,
                                std::vector<float> *const logits) {
  return impl_->forward(tokens, true, logits,
                        std::numeric_limits<std::size_t>::max(), nullptr);
}

Status HyenaDnaContext::prefill_chunk(const std::vector<TokenId> &tokens,
                                      std::vector<float> *const logits) {
  return impl_->forward(tokens, false, logits,
                        std::numeric_limits<std::size_t>::max(), nullptr);
}

Status HyenaDnaContext::decode(const TokenId token,
                               std::vector<float> *const logits) {
  return impl_->forward({token}, false, logits,
                        std::numeric_limits<std::size_t>::max(), nullptr);
}

Status HyenaDnaContext::prefill_embedding(
    const std::vector<TokenId> &tokens, const std::size_t layer,
    std::vector<float> *const embedding) {
  return impl_->forward(tokens, true, nullptr, layer, embedding);
}

Status HyenaDnaContext::prefill_chunk_embedding(
    const std::vector<TokenId> &tokens, const std::size_t layer,
    std::vector<float> *const embedding) {
  return impl_->forward(tokens, false, nullptr, layer, embedding);
}

std::size_t HyenaDnaContext::position() const noexcept {
  return impl_->position;
}

std::size_t HyenaDnaContext::activation_capacity() const noexcept {
  return std::min(impl_->capacity, kActivationChunk);
}

const ModelConfig &HyenaDnaContext::config() const noexcept {
  return impl_->weights->public_config;
}

const char *HyenaDnaContext::kernel_name() const noexcept {
  return impl_->weights->executor ? impl_->weights->executor->name()
                                  : "direct-convolution-f32";
}

} // namespace evo::cpu::detail
