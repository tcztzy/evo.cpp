// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_sequence_cnn.hpp"

#include "geneb_t36_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>

namespace evo::cpu {
namespace {

constexpr std::string_view kFamily = "GENEB sequence CNN";
constexpr std::size_t kMaximumLayers = 1024U;
constexpr std::size_t kMaximumWidth = 1U << 20U;
constexpr std::size_t kMaximumInputLength = 1U << 30U;

void add_requirement(
    std::vector<GenebSequenceCnnTensorRequirement> *const output,
    std::string name, std::vector<std::size_t> shape) {
  output->push_back(
      {std::move(name), TensorDType::kF32, std::move(shape)});
}

std::string transformer_prefix(const GenebSequenceCnnTopology &topology,
                               const std::size_t layer) {
  if (topology.variant == GenebSequenceCnnVariant::kEnformer)
    return "transformer." + std::to_string(layer) + ".";
  return "transformer.transformer." + std::to_string(layer) + ".";
}

float sigmoid_gelu(const float input, const float scale) noexcept {
  return input / (1.0F + std::exp(-scale * input));
}

Status batch_norm(const std::vector<float> &input, const std::size_t rows,
                  const std::size_t width,
                  const GenebSequenceCnnTensorView &scale,
                  const GenebSequenceCnnTensorView &bias,
                  const GenebSequenceCnnTensorView &running_mean,
                  const GenebSequenceCnnTensorView &running_variance,
                  const float epsilon, std::vector<float> *const output) {
  std::size_t elements = 0;
  const std::vector<std::size_t> expected{width};
  if (output == nullptr || !t36::checked_product(rows, width, &elements) ||
      input.size() != elements || scale.shape != expected ||
      bias.shape != expected || running_mean.shape != expected ||
      running_variance.shape != expected || scale.data == nullptr ||
      bias.data == nullptr || running_mean.data == nullptr ||
      running_variance.data == nullptr || !std::isfinite(epsilon) ||
      epsilon <= 0.0F)
    return {ErrorCode::kInvalidArgument,
            "GENEB sequence CNN BatchNorm arguments are inconsistent"};
  output->resize(elements);
  for (std::size_t column = 0; column < width; ++column) {
    const float variance = t36::value(running_variance, column);
    if (!std::isfinite(variance) || variance + epsilon <= 0.0F)
      return {ErrorCode::kInvalidArgument,
              "GENEB sequence CNN BatchNorm variance is invalid"};
    const float multiplier =
        t36::value(scale, column) / std::sqrt(variance + epsilon);
    const float mean = t36::value(running_mean, column);
    const float offset = t36::value(bias, column);
    for (std::size_t row = 0; row < rows; ++row)
      (*output)[row * width + column] =
          (input[row * width + column] - mean) * multiplier + offset;
  }
  return t36::finite(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB sequence CNN BatchNorm became non-finite"};
}

Status same_convolution(const std::vector<float> &input,
                        const std::size_t rows,
                        const std::size_t input_width,
                        const GenebSequenceCnnTensorView &weight,
                        const GenebSequenceCnnTensorView &bias,
                        const std::size_t output_width,
                        const std::size_t kernel,
                        std::vector<float> *const output) {
  std::size_t elements = 0;
  if (output == nullptr || rows == 0 || kernel == 0 || kernel % 2U == 0 ||
      !t36::checked_product(rows, input_width, &elements) ||
      input.size() != elements ||
      weight.shape !=
          std::vector<std::size_t>{output_width, input_width, kernel} ||
      bias.shape != std::vector<std::size_t>{output_width} ||
      weight.data == nullptr || bias.data == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB sequence CNN convolution arguments are inconsistent"};
  output->assign(rows * output_width, 0.0F);
  const std::size_t padding = kernel / 2U;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < output_width; ++out) {
      float sum = t36::value(bias, out);
      for (std::size_t in = 0; in < input_width; ++in) {
        const std::size_t weight_base = (out * input_width + in) * kernel;
        for (std::size_t tap = 0; tap < kernel; ++tap) {
          const std::ptrdiff_t source = static_cast<std::ptrdiff_t>(row) +
                                        static_cast<std::ptrdiff_t>(tap) -
                                        static_cast<std::ptrdiff_t>(padding);
          if (source >= 0 &&
              static_cast<std::size_t>(source) < rows) {
            sum += input[static_cast<std::size_t>(source) * input_width + in] *
                   t36::value(weight, weight_base + tap);
          }
        }
      }
      (*output)[row * output_width + out] = sum;
    }
  }
  return t36::finite(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB sequence CNN convolution became non-finite"};
}

Status attention_pool(const std::vector<float> &input,
                      const std::size_t rows, const std::size_t width,
                      const GenebSequenceCnnTensorView &logit_weight,
                      std::vector<float> *const output) {
  if (output == nullptr || rows == 0 || input.size() != rows * width ||
      logit_weight.shape !=
          std::vector<std::size_t>{width, width, 1, 1} ||
      logit_weight.data == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB sequence CNN attention-pool arguments are inconsistent"};
  const std::size_t pooled_rows = (rows + 1U) / 2U;
  output->assign(pooled_rows * width, 0.0F);
  for (std::size_t pooled = 0; pooled < pooled_rows; ++pooled) {
    const std::size_t first_row = pooled * 2U;
    const bool has_second = first_row + 1U < rows;
    for (std::size_t out = 0; out < width; ++out) {
      float first_logit = 0.0F;
      float second_logit = 0.0F;
      for (std::size_t in = 0; in < width; ++in) {
        const float coefficient = t36::value(logit_weight, out * width + in);
        first_logit += input[first_row * width + in] * coefficient;
        if (has_second)
          second_logit += input[(first_row + 1U) * width + in] * coefficient;
      }
      if (!has_second) {
        (*output)[pooled * width + out] = input[first_row * width + out];
        continue;
      }
      const float maximum = std::max(first_logit, second_logit);
      const float first_exp = std::exp(first_logit - maximum);
      const float second_exp = std::exp(second_logit - maximum);
      const float first_probability = first_exp / (first_exp + second_exp);
      (*output)[pooled * width + out] =
          input[first_row * width + out] * first_probability +
          input[(first_row + 1U) * width + out] *
              (1.0F - first_probability);
    }
  }
  return t36::finite(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB sequence CNN attention pool became non-finite"};
}

std::vector<float> relative_features(const std::size_t rows,
                                     const std::size_t feature_width) {
  const std::size_t basis = feature_width / 6U;
  const std::size_t distances = rows * 2U - 1U;
  std::vector<float> result(distances * feature_width, 0.0F);
  const double maximum_exponent = std::log2(static_cast<double>(rows));
  const double standard_deviation =
      static_cast<double>(rows) / (2.0 * static_cast<double>(basis));
  for (std::size_t index = 0; index < distances; ++index) {
    const std::ptrdiff_t signed_distance =
        static_cast<std::ptrdiff_t>(index) -
        static_cast<std::ptrdiff_t>(rows - 1U);
    const double distance = std::abs(static_cast<double>(signed_distance));
    std::vector<double> gamma_values(basis, 0.0);
    double gamma_maximum = 0.0;
    for (std::size_t component = 0; component < basis; ++component) {
      const double fraction =
          basis == 1U ? 0.0 : static_cast<double>(component) /
                                    static_cast<double>(basis - 1U);
      const double half_life =
          std::pow(2.0, 3.0 + (maximum_exponent - 3.0) * fraction);
      result[index * feature_width + component] = static_cast<float>(
          std::exp(-std::log(2.0) * distance / half_life));
      const double center_width =
          std::pow(2.0, static_cast<double>(component) + 1.0) - 1.0;
      result[index * feature_width + basis + component] =
          center_width > distance ? 1.0F : 0.0F;
      const double start_mean =
          static_cast<double>(rows) / static_cast<double>(basis);
      const double mean =
          basis == 1U
              ? start_mean
              : start_mean +
                    (static_cast<double>(rows) - start_mean) * fraction;
      const double concentration =
          (mean / standard_deviation) * (mean / standard_deviation);
      const double rate = mean /
                          (standard_deviation * standard_deviation);
      double probability = 0.0;
      if (distance > 0.0) {
        const double log_probability =
            (concentration - 1.0) * std::log(distance) - rate * distance -
            (std::lgamma(concentration) - concentration * std::log(rate));
        probability = std::exp(log_probability);
      }
      gamma_values[component] = probability + 1.0e-8;
      gamma_maximum = std::max(gamma_maximum, gamma_values[component]);
    }
    for (std::size_t component = 0; component < basis; ++component)
      result[index * feature_width + basis * 2U + component] =
          static_cast<float>(gamma_values[component] / gamma_maximum);
    const float sign = signed_distance < 0 ? -1.0F
                       : signed_distance > 0 ? 1.0F
                                             : 0.0F;
    for (std::size_t component = 0; component < feature_width / 2U;
         ++component)
      result[index * feature_width + feature_width / 2U + component] =
          sign * result[index * feature_width + component];
  }
  return result;
}

Status relative_attention(
    const std::vector<float> &input, const std::size_t rows,
    const GenebSequenceCnnTopology &topology,
    const GenebSequenceCnnTensorView &query_weight,
    const GenebSequenceCnnTensorView &key_weight,
    const GenebSequenceCnnTensorView &value_weight,
    const GenebSequenceCnnTensorView &output_weight,
    const GenebSequenceCnnTensorView &output_bias,
    const GenebSequenceCnnTensorView &relative_key_weight,
    const GenebSequenceCnnTensorView &content_bias,
    const GenebSequenceCnnTensorView &position_bias,
    evo::detail::LinearExecutor *const executor,
    std::vector<float> *const output) {
  const std::size_t heads = topology.attention_heads;
  const std::size_t key_dimension = topology.key_dimension;
  const std::size_t value_dimension = topology.value_dimension;
  const std::size_t width = topology.width;
  const std::size_t projected_key = heads * key_dimension;
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  auto status = t36::linear(
      input, rows, width, query_weight, projected_key,
      static_cast<const GenebSequenceCnnTensorView *>(nullptr), executor,
      &query, kFamily);
  if (!status.ok())
    return status;
  status = t36::linear(
      input, rows, width, key_weight, projected_key,
      static_cast<const GenebSequenceCnnTensorView *>(nullptr), executor, &key,
      kFamily);
  if (!status.ok())
    return status;
  status = t36::linear(
      input, rows, width, value_weight, heads * value_dimension,
      static_cast<const GenebSequenceCnnTensorView *>(nullptr), executor,
      &value, kFamily);
  if (!status.ok())
    return status;
  const float scale = 1.0F / std::sqrt(static_cast<float>(key_dimension));
  for (float &item : query)
    item *= scale;
  std::vector<float> features =
      relative_features(rows, topology.relative_feature_width);
  std::vector<float> relative_key;
  status = t36::linear(
      features, rows * 2U - 1U, topology.relative_feature_width,
      relative_key_weight, projected_key,
      static_cast<const GenebSequenceCnnTensorView *>(nullptr), executor,
      &relative_key, kFamily);
  if (!status.ok())
    return status;
  std::vector<float> context(rows * heads * value_dimension, 0.0F);
  for (std::size_t query_row = 0; query_row < rows; ++query_row) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t query_base =
          (query_row * heads + head) * key_dimension;
      float maximum = -std::numeric_limits<float>::infinity();
      const auto score = [&](const std::size_t key_row) {
        const std::size_t key_base =
            (key_row * heads + head) * key_dimension;
        const std::size_t relative_row = rows - 1U - query_row + key_row;
        const std::size_t relative_base =
            (relative_row * heads + head) * key_dimension;
        float result = 0.0F;
        for (std::size_t column = 0; column < key_dimension; ++column) {
          result +=
              (query[query_base + column] +
               t36::value(content_bias, head * key_dimension + column)) *
                  key[key_base + column] +
              (query[query_base + column] +
               t36::value(position_bias, head * key_dimension + column)) *
                  relative_key[relative_base + column];
        }
        return result;
      };
      for (std::size_t key_row = 0; key_row < rows; ++key_row)
        maximum = std::max(maximum, score(key_row));
      double denominator = 0.0;
      for (std::size_t key_row = 0; key_row < rows; ++key_row) {
        const float probability = std::exp(score(key_row) - maximum);
        denominator += probability;
        const std::size_t value_base =
            (key_row * heads + head) * value_dimension;
        const std::size_t context_base =
            (query_row * heads + head) * value_dimension;
        for (std::size_t column = 0; column < value_dimension; ++column)
          context[context_base + column] +=
              probability * value[value_base + column];
      }
      if (!std::isfinite(denominator) || denominator <= 0.0)
        return {ErrorCode::kInvalidArgument,
                "GENEB sequence CNN attention denominator is invalid"};
      const float inverse = 1.0F / static_cast<float>(denominator);
      const std::size_t context_base =
          (query_row * heads + head) * value_dimension;
      for (std::size_t column = 0; column < value_dimension; ++column)
        context[context_base + column] *= inverse;
    }
  }
  return t36::linear(context, rows, heads * value_dimension, output_weight,
                     width, &output_bias, executor, output, kFamily);
}

} // namespace

Status validate_geneb_sequence_cnn_topology(
    const GenebSequenceCnnTopology &topology) {
  if (topology.input_length == 0 ||
      topology.input_length > kMaximumInputLength || topology.stem_width == 0 ||
      topology.tower_widths.empty() || topology.tower_widths.size() > 32U ||
      topology.width == 0 || topology.width > kMaximumWidth ||
      topology.output_width == 0 || topology.output_width > kMaximumWidth ||
      topology.layers == 0 || topology.layers > kMaximumLayers ||
      topology.attention_heads == 0 || topology.key_dimension == 0 ||
      topology.value_dimension == 0 ||
      topology.attention_heads * topology.value_dimension != topology.width ||
      topology.relative_feature_width == 0 ||
      topology.relative_feature_width % 6U != 0 ||
      topology.target_length == 0 || topology.weight_dtype != TensorDType::kF32)
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN topology dimensions are invalid"};
  if (topology.tower_widths.back() != topology.width ||
      std::any_of(topology.tower_widths.begin(), topology.tower_widths.end(),
                  [](const std::size_t value) {
                    return value == 0 || value > kMaximumWidth;
                  }))
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN tower widths are invalid"};
  std::size_t rows = topology.input_length;
  for (std::size_t index = 0; index <= topology.tower_widths.size(); ++index)
    rows = (rows + 1U) / 2U;
  if (rows < topology.target_length)
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN target exceeds downsampled length"};
  if (!std::isfinite(topology.batch_norm_epsilon) ||
      topology.batch_norm_epsilon <= 0.0F ||
      !std::isfinite(topology.gelu_sigmoid_scale) ||
      topology.gelu_sigmoid_scale <= 0.0F || topology.use_tf_gamma)
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN frozen numerical semantics differ"};
  if (topology.variant == GenebSequenceCnnVariant::kEnformer) {
    if (topology.species_num_experts != 0 || topology.top_k != 0 ||
        topology.gate_negative_slope != 0.0F || !topology.species.empty())
      return {ErrorCode::kModelFormat,
              "GENEB Enformer unexpectedly enables SPACE MoE"};
  } else if (topology.species_num_experts == 0 || topology.top_k == 0 ||
             topology.top_k > topology.species_num_experts ||
             !std::isfinite(topology.gate_negative_slope) ||
             topology.gate_negative_slope < 0.0F || topology.species.empty()) {
    return {ErrorCode::kModelFormat,
            "GENEB SPACE MoE topology is invalid"};
  }
  return Status::Ok();
}

Status geneb_sequence_cnn_topology_from_artifact(
    const ModelFile &artifact, GenebSequenceCnnTopology *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB sequence CNN topology output is null"};
  auto status = t36::metadata_literal(artifact, "runtime.profile",
                                      kGenebSequenceCnnArtifactProfile, kFamily);
  if (!status.ok())
    return status;
  status = t36::metadata_literal(artifact, "runtime.abi",
                                 kGenebSequenceCnnRuntimeAbi, kFamily);
  if (!status.ok())
    return status;
  status = t36::metadata_literal(artifact, "model.architecture",
                                 kGenebSequenceCnnArchitecture, kFamily);
  if (!status.ok())
    return status;
  for (const auto &[key, expected] :
       std::initializer_list<std::pair<std::string_view, std::string_view>>{
           {"sequence_cnn.pooling", "spatial-mean"},
           {"sequence_cnn.mask_domain", "all-896-spatial-rows"},
           {"sequence_cnn.special_tokens", "none"}}) {
    status = t36::metadata_literal(artifact, key, expected, kFamily);
    if (!status.ok())
      return status;
  }
  GenebSequenceCnnTopology topology;
  std::string variant;
  status = t36::metadata_string(artifact, "sequence_cnn.variant", &variant,
                                kFamily);
  if (!status.ok())
    return status;
  if (variant == "enformer")
    topology.variant = GenebSequenceCnnVariant::kEnformer;
  else if (variant == "space")
    topology.variant = GenebSequenceCnnVariant::kSpace;
  else
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN variant is unsupported"};
  const auto size = [&](const std::string_view key, std::size_t *value) {
    return t36::metadata_size(artifact, key, value, kFamily);
  };
  status = size("sequence_cnn.input_length", &topology.input_length);
  if (!status.ok())
    return status;
  topology.stem_width = 768;
  topology.tower_widths = {768, 896, 1024, 1152, 1280, 1536};
  status = size("sequence_cnn.trunk_width", &topology.width);
  if (!status.ok())
    return status;
  status = size("sequence_cnn.output_width", &topology.output_width);
  if (!status.ok())
    return status;
  status = size("sequence_cnn.depth", &topology.layers);
  if (!status.ok())
    return status;
  status = size("sequence_cnn.heads", &topology.attention_heads);
  if (!status.ok())
    return status;
  status = size("sequence_cnn.key_width", &topology.key_dimension);
  if (!status.ok())
    return status;
  status = size("sequence_cnn.value_width", &topology.value_dimension);
  if (!status.ok())
    return status;
  status = size("sequence_cnn.relative_feature_width",
                &topology.relative_feature_width);
  if (!status.ok())
    return status;
  status = size("sequence_cnn.target_length", &topology.target_length);
  if (!status.ok())
    return status;
  std::size_t downsamples = 0;
  status = size("sequence_cnn.num_downsamples", &downsamples);
  if (!status.ok())
    return status;
  if (downsamples != topology.tower_widths.size() + 1U)
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN downsample count differs"};
  status = t36::metadata_float(artifact, "sequence_cnn.batch_norm_epsilon",
                               &topology.batch_norm_epsilon, kFamily);
  if (!status.ok())
    return status;
  status = t36::metadata_float(artifact, "sequence_cnn.gelu_sigmoid_scale",
                               &topology.gelu_sigmoid_scale, kFamily);
  if (!status.ok())
    return status;
  status = t36::metadata_bool(artifact, "sequence_cnn.use_tf_gamma",
                              &topology.use_tf_gamma, kFamily);
  if (!status.ok())
    return status;
  status = size("sequence_cnn.species_num_experts",
                &topology.species_num_experts);
  if (!status.ok())
    return status;
  status = size("sequence_cnn.top_k", &topology.top_k);
  if (!status.ok())
    return status;
  status = t36::metadata_float(artifact, "sequence_cnn.gate_negative_slope",
                               &topology.gate_negative_slope, kFamily);
  if (!status.ok())
    return status;
  status = t36::metadata_string(artifact, "sequence_cnn.species",
                                &topology.species, kFamily);
  if (!status.ok())
    return status;
  topology.weight_dtype = TensorDType::kF32;
  if (topology.width != 1536 || topology.output_width != 3072 ||
      topology.stem_width != topology.width / 2U)
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN production topology differs"};
  status = validate_geneb_sequence_cnn_topology(topology);
  if (!status.ok())
    return status;
  *output = topology;
  return Status::Ok();
}

Status canonical_geneb_sequence_cnn_tensors(
    const GenebSequenceCnnTopology &topology,
    std::vector<GenebSequenceCnnTensorRequirement> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB sequence CNN tensor requirements output is null"};
  auto status = validate_geneb_sequence_cnn_topology(topology);
  if (!status.ok())
    return status;
  output->clear();
  add_requirement(output, "stem.0.weight", {topology.stem_width, 4, 15});
  add_requirement(output, "stem.0.bias", {topology.stem_width});
  for (const std::string_view suffix : {"weight", "bias", "running_mean",
                                        "running_var"})
    add_requirement(output, "stem.1.fn.0." + std::string{suffix},
                    {topology.stem_width});
  add_requirement(output, "stem.1.fn.2.weight",
                  {topology.stem_width, topology.stem_width, 1});
  add_requirement(output, "stem.1.fn.2.bias", {topology.stem_width});
  add_requirement(output, "stem.2.to_attn_logits.weight",
                  {topology.stem_width, topology.stem_width, 1, 1});
  std::size_t previous = topology.stem_width;
  for (std::size_t index = 0; index < topology.tower_widths.size(); ++index) {
    const std::size_t width = topology.tower_widths[index];
    const std::string prefix = "conv_tower." + std::to_string(index) + ".";
    for (const std::string_view suffix : {"weight", "bias", "running_mean",
                                          "running_var"})
      add_requirement(output, prefix + "0.0." + std::string{suffix},
                      {previous});
    add_requirement(output, prefix + "0.2.weight", {width, previous, 5});
    add_requirement(output, prefix + "0.2.bias", {width});
    for (const std::string_view suffix : {"weight", "bias", "running_mean",
                                          "running_var"})
      add_requirement(output, prefix + "1.fn.0." + std::string{suffix},
                      {width});
    add_requirement(output, prefix + "1.fn.2.weight", {width, width, 1});
    add_requirement(output, prefix + "1.fn.2.bias", {width});
    add_requirement(output, prefix + "2.to_attn_logits.weight",
                    {width, width, 1, 1});
    previous = width;
  }
  if (topology.variant == GenebSequenceCnnVariant::kSpace) {
    add_requirement(output, "transformer.species_embedding.human",
                    {1, 1, topology.width});
    add_requirement(output, "transformer.species_embedding.mouse",
                    {1, 1, topology.width});
  }
  for (std::size_t index = 0; index < topology.layers; ++index) {
    const std::string prefix = transformer_prefix(topology, index);
    const std::string attention =
        topology.variant == GenebSequenceCnnVariant::kEnformer ? "0.fn."
                                                               : "attention.fn.";
    add_requirement(output, prefix + attention + "0.weight", {topology.width});
    add_requirement(output, prefix + attention + "0.bias", {topology.width});
    add_requirement(output, prefix + attention + "1.rel_content_bias",
                    {1, topology.attention_heads, 1,
                     topology.key_dimension});
    add_requirement(output, prefix + attention + "1.rel_pos_bias",
                    {1, topology.attention_heads, 1,
                     topology.key_dimension});
    add_requirement(output, prefix + attention + "1.to_q.weight",
                    {topology.attention_heads * topology.key_dimension,
                     topology.width});
    add_requirement(output, prefix + attention + "1.to_k.weight",
                    {topology.attention_heads * topology.key_dimension,
                     topology.width});
    add_requirement(output, prefix + attention + "1.to_v.weight",
                    {topology.width, topology.width});
    add_requirement(output, prefix + attention + "1.to_out.weight",
                    {topology.width, topology.width});
    add_requirement(output, prefix + attention + "1.to_out.bias",
                    {topology.width});
    add_requirement(output, prefix + attention + "1.to_rel_k.weight",
                    {topology.attention_heads * topology.key_dimension,
                     topology.relative_feature_width});
    if (topology.variant == GenebSequenceCnnVariant::kEnformer) {
      add_requirement(output, prefix + "1.fn.0.weight", {topology.width});
      add_requirement(output, prefix + "1.fn.0.bias", {topology.width});
      add_requirement(output, prefix + "1.fn.1.weight",
                      {topology.width * 2U, topology.width});
      add_requirement(output, prefix + "1.fn.1.bias", {topology.width * 2U});
      add_requirement(output, prefix + "1.fn.4.weight",
                      {topology.width, topology.width * 2U});
      add_requirement(output, prefix + "1.fn.4.bias", {topology.width});
    } else {
      const std::string feed = prefix + "feed_forward.";
      for (const std::string_view species : {"human", "mouse"}) {
        add_requirement(output,
                        feed + "gates." + std::string{species} + ".0.weight",
                        {topology.species_num_experts, topology.width});
        add_requirement(output,
                        feed + "gates." + std::string{species} + ".0.bias",
                        {topology.species_num_experts});
      }
      add_requirement(output, feed + "layer_norm.weight", {topology.width});
      add_requirement(output, feed + "layer_norm.bias", {topology.width});
      add_requirement(output, feed + "input.weight",
                      {topology.species_num_experts, topology.width,
                       topology.width * 2U});
      add_requirement(output, feed + "input.bias",
                      {topology.species_num_experts, topology.width * 2U});
      add_requirement(output, feed + "output.weight",
                      {topology.species_num_experts, topology.width * 2U,
                       topology.width});
      add_requirement(output, feed + "output.bias",
                      {topology.species_num_experts, topology.width});
    }
  }
  for (const std::string_view suffix : {"weight", "bias", "running_mean",
                                        "running_var"})
    add_requirement(output, "final_pointwise.1.0." + std::string{suffix},
                    {topology.width});
  add_requirement(output, "final_pointwise.1.2.weight",
                  {topology.output_width, topology.width, 1});
  add_requirement(output, "final_pointwise.1.2.bias",
                  {topology.output_width});
  return Status::Ok();
}

Status geneb_sequence_cnn_pool(
    const GenebSequenceCnnForwardResult &forward,
    std::vector<float> *const output) {
  if (output == nullptr || forward.rows == 0 || forward.width == 0 ||
      forward.final_hidden.size() != forward.rows * forward.width)
    return {ErrorCode::kInvalidArgument,
            "GENEB sequence CNN pooling arguments are inconsistent"};
  output->assign(forward.width, 0.0F);
  for (std::size_t row = 0; row < forward.rows; ++row)
    for (std::size_t column = 0; column < forward.width; ++column)
      (*output)[column] +=
          forward.final_hidden[row * forward.width + column];
  for (float &value : *output)
    value /= static_cast<float>(forward.rows);
  return t36::finite(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB sequence CNN pooled output became non-finite"};
}

struct GenebSequenceCnnModel::Impl final {
  struct BatchNorm final {
    GenebSequenceCnnTensorView weight;
    GenebSequenceCnnTensorView bias;
    GenebSequenceCnnTensorView running_mean;
    GenebSequenceCnnTensorView running_variance;
  };
  struct Tower final {
    BatchNorm input_norm;
    GenebSequenceCnnTensorView convolution_weight;
    GenebSequenceCnnTensorView convolution_bias;
    BatchNorm residual_norm;
    GenebSequenceCnnTensorView residual_weight;
    GenebSequenceCnnTensorView residual_bias;
    GenebSequenceCnnTensorView pool_weight;
  };
  struct Transformer final {
    BatchNorm unused;
    GenebSequenceCnnTensorView attention_norm_weight;
    GenebSequenceCnnTensorView attention_norm_bias;
    GenebSequenceCnnTensorView content_bias;
    GenebSequenceCnnTensorView position_bias;
    GenebSequenceCnnTensorView query_weight;
    GenebSequenceCnnTensorView key_weight;
    GenebSequenceCnnTensorView value_weight;
    GenebSequenceCnnTensorView attention_output_weight;
    GenebSequenceCnnTensorView attention_output_bias;
    GenebSequenceCnnTensorView relative_key_weight;
    GenebSequenceCnnTensorView feed_norm_weight;
    GenebSequenceCnnTensorView feed_norm_bias;
    GenebSequenceCnnTensorView feed_input_weight;
    GenebSequenceCnnTensorView feed_input_bias;
    GenebSequenceCnnTensorView feed_output_weight;
    GenebSequenceCnnTensorView feed_output_bias;
    GenebSequenceCnnTensorView human_gate_weight;
    GenebSequenceCnnTensorView human_gate_bias;
    GenebSequenceCnnTensorView mouse_gate_weight;
    GenebSequenceCnnTensorView mouse_gate_bias;
  };

  GenebSequenceCnnTopology topology;
  GenebSequenceCnnTensorView stem_weight;
  GenebSequenceCnnTensorView stem_bias;
  BatchNorm stem_norm;
  GenebSequenceCnnTensorView stem_residual_weight;
  GenebSequenceCnnTensorView stem_residual_bias;
  GenebSequenceCnnTensorView stem_pool_weight;
  std::vector<Tower> towers;
  GenebSequenceCnnTensorView human_species_embedding;
  GenebSequenceCnnTensorView mouse_species_embedding;
  std::vector<Transformer> transformers;
  BatchNorm final_norm;
  GenebSequenceCnnTensorView final_weight;
  GenebSequenceCnnTensorView final_bias;
  std::shared_ptr<evo::detail::LinearExecutor> linear_executor;
};

GenebSequenceCnnModel::GenebSequenceCnnModel() = default;
GenebSequenceCnnModel::~GenebSequenceCnnModel() = default;
GenebSequenceCnnModel::GenebSequenceCnnModel(
    GenebSequenceCnnModel &&) noexcept = default;
GenebSequenceCnnModel &GenebSequenceCnnModel::operator=(
    GenebSequenceCnnModel &&) noexcept = default;

Status GenebSequenceCnnModel::load(
    const GenebSequenceCnnTopology &topology,
    const std::vector<GenebSequenceCnnNamedTensorView> &tensors,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  std::vector<GenebSequenceCnnTensorRequirement> requirements;
  auto status = canonical_geneb_sequence_cnn_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  std::map<std::string, const GenebSequenceCnnTensorView *, std::less<>>
      provided;
  for (const auto &named : tensors) {
    if (!provided.emplace(named.name, &named.tensor).second)
      return {ErrorCode::kModelFormat,
              "GENEB sequence CNN tensor is duplicated: " + named.name};
  }
  if (provided.size() != requirements.size())
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN tensor set has missing or extra entries"};
  for (const auto &requirement : requirements) {
    const auto found = provided.find(requirement.name);
    if (found == provided.end())
      return {ErrorCode::kModelFormat,
              "GENEB sequence CNN tensor is missing: " + requirement.name};
    status = t36::tensor_requirement(*found->second, requirement, kFamily);
    if (!status.ok())
      return status;
  }
  const auto view = [&](const std::string &name) {
    return *provided.find(name)->second;
  };
  const auto norm = [&](const std::string &prefix) {
    return Impl::BatchNorm{view(prefix + "weight"), view(prefix + "bias"),
                           view(prefix + "running_mean"),
                           view(prefix + "running_var")};
  };
  auto implementation = std::make_unique<Impl>();
  implementation->topology = topology;
  implementation->linear_executor = std::move(linear_executor);
  implementation->stem_weight = view("stem.0.weight");
  implementation->stem_bias = view("stem.0.bias");
  implementation->stem_norm = norm("stem.1.fn.0.");
  implementation->stem_residual_weight = view("stem.1.fn.2.weight");
  implementation->stem_residual_bias = view("stem.1.fn.2.bias");
  implementation->stem_pool_weight = view("stem.2.to_attn_logits.weight");
  implementation->towers.resize(topology.tower_widths.size());
  for (std::size_t index = 0; index < topology.tower_widths.size(); ++index) {
    auto &tower = implementation->towers[index];
    const std::string prefix = "conv_tower." + std::to_string(index) + ".";
    tower.input_norm = norm(prefix + "0.0.");
    tower.convolution_weight = view(prefix + "0.2.weight");
    tower.convolution_bias = view(prefix + "0.2.bias");
    tower.residual_norm = norm(prefix + "1.fn.0.");
    tower.residual_weight = view(prefix + "1.fn.2.weight");
    tower.residual_bias = view(prefix + "1.fn.2.bias");
    tower.pool_weight = view(prefix + "2.to_attn_logits.weight");
  }
  if (topology.variant == GenebSequenceCnnVariant::kSpace) {
    implementation->human_species_embedding =
        view("transformer.species_embedding.human");
    implementation->mouse_species_embedding =
        view("transformer.species_embedding.mouse");
  }
  implementation->transformers.resize(topology.layers);
  for (std::size_t index = 0; index < topology.layers; ++index) {
    auto &layer = implementation->transformers[index];
    const std::string prefix = transformer_prefix(topology, index);
    const std::string attention =
        topology.variant == GenebSequenceCnnVariant::kEnformer ? "0.fn."
                                                               : "attention.fn.";
    layer.attention_norm_weight = view(prefix + attention + "0.weight");
    layer.attention_norm_bias = view(prefix + attention + "0.bias");
    layer.content_bias = view(prefix + attention + "1.rel_content_bias");
    layer.position_bias = view(prefix + attention + "1.rel_pos_bias");
    layer.query_weight = view(prefix + attention + "1.to_q.weight");
    layer.key_weight = view(prefix + attention + "1.to_k.weight");
    layer.value_weight = view(prefix + attention + "1.to_v.weight");
    layer.attention_output_weight =
        view(prefix + attention + "1.to_out.weight");
    layer.attention_output_bias =
        view(prefix + attention + "1.to_out.bias");
    layer.relative_key_weight =
        view(prefix + attention + "1.to_rel_k.weight");
    if (topology.variant == GenebSequenceCnnVariant::kEnformer) {
      layer.feed_norm_weight = view(prefix + "1.fn.0.weight");
      layer.feed_norm_bias = view(prefix + "1.fn.0.bias");
      layer.feed_input_weight = view(prefix + "1.fn.1.weight");
      layer.feed_input_bias = view(prefix + "1.fn.1.bias");
      layer.feed_output_weight = view(prefix + "1.fn.4.weight");
      layer.feed_output_bias = view(prefix + "1.fn.4.bias");
    } else {
      const std::string feed = prefix + "feed_forward.";
      layer.human_gate_weight = view(feed + "gates.human.0.weight");
      layer.human_gate_bias = view(feed + "gates.human.0.bias");
      layer.mouse_gate_weight = view(feed + "gates.mouse.0.weight");
      layer.mouse_gate_bias = view(feed + "gates.mouse.0.bias");
      layer.feed_norm_weight = view(feed + "layer_norm.weight");
      layer.feed_norm_bias = view(feed + "layer_norm.bias");
      layer.feed_input_weight = view(feed + "input.weight");
      layer.feed_input_bias = view(feed + "input.bias");
      layer.feed_output_weight = view(feed + "output.weight");
      layer.feed_output_bias = view(feed + "output.bias");
    }
  }
  implementation->final_norm = norm("final_pointwise.1.0.");
  implementation->final_weight = view("final_pointwise.1.2.weight");
  implementation->final_bias = view("final_pointwise.1.2.bias");
  impl_ = std::move(implementation);
  return Status::Ok();
}

Status GenebSequenceCnnModel::load(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  GenebSequenceCnnTopology topology;
  auto status = geneb_sequence_cnn_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebSequenceCnnNamedTensorView> views;
  views.reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t index = 0; index < tensor.rank; ++index) {
      if (tensor.dimensions[index] > std::numeric_limits<std::size_t>::max())
        return {ErrorCode::kModelFormat,
                "GENEB sequence CNN tensor dimension exceeds size_t"};
      shape.push_back(static_cast<std::size_t>(tensor.dimensions[index]));
    }
    if (tensor.data_size > std::numeric_limits<std::size_t>::max())
      return {ErrorCode::kModelFormat,
              "GENEB sequence CNN tensor extent exceeds size_t"};
    const auto *data = artifact.tensor_data(tensor);
    if (data == nullptr)
      return {ErrorCode::kModelFormat,
              "GENEB sequence CNN tensor payload is unavailable"};
    views.push_back({tensor.name,
                     {data, static_cast<std::size_t>(tensor.data_size),
                      tensor.dtype, std::move(shape)}});
  }
  return load(topology, views, std::move(linear_executor));
}

Status GenebSequenceCnnModel::load_artifact(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  return load(artifact, std::move(linear_executor));
}

const GenebSequenceCnnTopology *
GenebSequenceCnnModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebSequenceCnnModel::linear_executor_name() const noexcept {
  if (impl_ == nullptr || impl_->linear_executor == nullptr)
    return "scalar-reference";
  return impl_->linear_executor->name();
}

Status GenebSequenceCnnModel::forward(
    const std::string_view sequence,
    GenebSequenceCnnForwardResult *const output) const {
  if (impl_ == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB sequence CNN model is not loaded"};
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB sequence CNN forward output is null"};
  const auto &topology = impl_->topology;
  const std::size_t length = topology.input_length;
  std::vector<float> hidden(length * 4U, 0.0F);
  std::size_t source_start = 0;
  std::size_t destination_start = 0;
  std::size_t retained = std::min(sequence.size(), length);
  if (topology.variant == GenebSequenceCnnVariant::kSpace) {
    if (sequence.size() > length)
      source_start = (sequence.size() - length) / 2U;
    else
      destination_start = (length - sequence.size()) / 2U;
  }
  for (std::size_t index = 0; index < retained; ++index) {
    unsigned char raw =
        static_cast<unsigned char>(sequence[source_start + index]);
    char base = static_cast<char>(std::toupper(raw));
    if (topology.variant == GenebSequenceCnnVariant::kSpace && base == 'U')
      base = 'T';
    std::size_t channel = 4U;
    if (base == 'A')
      channel = 0;
    else if (base == 'C')
      channel = 1;
    else if (base == 'G')
      channel = 2;
    else if (base == 'T')
      channel = 3;
    if (channel < 4U)
      hidden[(destination_start + index) * 4U + channel] = 1.0F;
  }
  std::vector<float> stage;
  auto status = same_convolution(hidden, length, 4U, impl_->stem_weight,
                                 impl_->stem_bias, topology.stem_width, 15U,
                                 &stage);
  if (!status.ok())
    return status;
  std::vector<float> normalized;
  status = batch_norm(stage, length, topology.stem_width,
                      impl_->stem_norm.weight, impl_->stem_norm.bias,
                      impl_->stem_norm.running_mean,
                      impl_->stem_norm.running_variance,
                      topology.batch_norm_epsilon, &normalized);
  if (!status.ok())
    return status;
  for (float &item : normalized)
    item = sigmoid_gelu(item, topology.gelu_sigmoid_scale);
  std::vector<float> residual;
  status = same_convolution(normalized, length, topology.stem_width,
                            impl_->stem_residual_weight,
                            impl_->stem_residual_bias, topology.stem_width, 1U,
                            &residual);
  if (!status.ok())
    return status;
  for (std::size_t index = 0; index < stage.size(); ++index)
    stage[index] += residual[index];
  std::size_t rows = length;
  status = attention_pool(stage, rows, topology.stem_width,
                          impl_->stem_pool_weight, &hidden);
  if (!status.ok())
    return status;
  rows = (rows + 1U) / 2U;
  std::size_t width = topology.stem_width;
  for (std::size_t tower_index = 0;
       tower_index < topology.tower_widths.size(); ++tower_index) {
    const auto &tower = impl_->towers[tower_index];
    const std::size_t output_width = topology.tower_widths[tower_index];
    status = batch_norm(hidden, rows, width, tower.input_norm.weight,
                        tower.input_norm.bias, tower.input_norm.running_mean,
                        tower.input_norm.running_variance,
                        topology.batch_norm_epsilon, &normalized);
    if (!status.ok())
      return status;
    for (float &item : normalized)
      item = sigmoid_gelu(item, topology.gelu_sigmoid_scale);
    status = same_convolution(normalized, rows, width,
                              tower.convolution_weight,
                              tower.convolution_bias, output_width, 5U,
                              &stage);
    if (!status.ok())
      return status;
    status = batch_norm(stage, rows, output_width,
                        tower.residual_norm.weight, tower.residual_norm.bias,
                        tower.residual_norm.running_mean,
                        tower.residual_norm.running_variance,
                        topology.batch_norm_epsilon, &normalized);
    if (!status.ok())
      return status;
    for (float &item : normalized)
      item = sigmoid_gelu(item, topology.gelu_sigmoid_scale);
    status = same_convolution(normalized, rows, output_width,
                              tower.residual_weight, tower.residual_bias,
                              output_width, 1U, &residual);
    if (!status.ok())
      return status;
    for (std::size_t index = 0; index < stage.size(); ++index)
      stage[index] += residual[index];
    status = attention_pool(stage, rows, output_width, tower.pool_weight,
                            &hidden);
    if (!status.ok())
      return status;
    rows = (rows + 1U) / 2U;
    width = output_width;
  }
  if (topology.variant == GenebSequenceCnnVariant::kSpace) {
    const auto &embedding = topology.species == "human"
                                ? impl_->human_species_embedding
                                : impl_->mouse_species_embedding;
    hidden.resize((rows + 1U) * width);
    for (std::size_t column = 0; column < width; ++column)
      hidden[rows * width + column] = t36::value(embedding, column);
    ++rows;
  }
  for (std::size_t layer_index = 0; layer_index < topology.layers;
       ++layer_index) {
    const auto &layer = impl_->transformers[layer_index];
    status = t36::layer_norm(hidden, rows, width,
                             layer.attention_norm_weight,
                             layer.attention_norm_bias,
                             topology.batch_norm_epsilon, &normalized,
                             kFamily);
    if (!status.ok())
      return status;
    status = relative_attention(
        normalized, rows, topology, layer.query_weight, layer.key_weight,
        layer.value_weight, layer.attention_output_weight,
        layer.attention_output_bias, layer.relative_key_weight,
        layer.content_bias, layer.position_bias,
        impl_->linear_executor.get(), &stage);
    if (!status.ok())
      return status;
    for (std::size_t index = 0; index < hidden.size(); ++index)
      hidden[index] += stage[index];
    if (topology.variant == GenebSequenceCnnVariant::kEnformer) {
      status = t36::layer_norm(hidden, rows, width,
                               layer.feed_norm_weight, layer.feed_norm_bias,
                               topology.batch_norm_epsilon, &normalized,
                               kFamily);
      if (!status.ok())
        return status;
      status = t36::linear(normalized, rows, width,
                           layer.feed_input_weight, width * 2U,
                           &layer.feed_input_bias,
                           impl_->linear_executor.get(), &stage, kFamily);
      if (!status.ok())
        return status;
      for (float &item : stage)
        item = std::max(item, 0.0F);
      status = t36::linear(stage, rows, width * 2U,
                           layer.feed_output_weight, width,
                           &layer.feed_output_bias,
                           impl_->linear_executor.get(), &residual, kFamily);
      if (!status.ok())
        return status;
      for (std::size_t index = 0; index < hidden.size(); ++index)
        hidden[index] += residual[index];
    } else {
      const auto &gate_weight = topology.species == "human"
                                    ? layer.human_gate_weight
                                    : layer.mouse_gate_weight;
      const auto &gate_bias = topology.species == "human"
                                  ? layer.human_gate_bias
                                  : layer.mouse_gate_bias;
      std::vector<float> gates;
      status = t36::linear(hidden, rows, width, gate_weight,
                           topology.species_num_experts, &gate_bias,
                           impl_->linear_executor.get(), &gates, kFamily);
      if (!status.ok())
        return status;
      for (float &item : gates) {
        if (item < 0.0F)
          item *= topology.gate_negative_slope;
      }
      status = t36::layer_norm(hidden, rows, width,
                               layer.feed_norm_weight, layer.feed_norm_bias,
                               topology.batch_norm_epsilon, &normalized,
                               kFamily);
      if (!status.ok())
        return status;
      residual.assign(hidden.size(), 0.0F);
      for (std::size_t row = 0; row < rows; ++row) {
        std::vector<std::size_t> selected(topology.species_num_experts);
        std::iota(selected.begin(), selected.end(), 0U);
        std::partial_sort(
            selected.begin(),
            selected.begin() +
                static_cast<std::ptrdiff_t>(topology.top_k),
            selected.end(), [&](const std::size_t left,
                                const std::size_t right) {
              return gates[row * topology.species_num_experts + left] >
                     gates[row * topology.species_num_experts + right];
            });
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t rank = 0; rank < topology.top_k; ++rank)
          maximum = std::max(
              maximum,
              gates[row * topology.species_num_experts + selected[rank]]);
        std::vector<float> probabilities(topology.top_k, 0.0F);
        float denominator = 0.0F;
        for (std::size_t rank = 0; rank < topology.top_k; ++rank) {
          probabilities[rank] = std::exp(
              gates[row * topology.species_num_experts + selected[rank]] -
              maximum);
          denominator += probabilities[rank];
        }
        for (float &probability : probabilities)
          probability /= denominator;
        for (std::size_t rank = 0; rank < topology.top_k; ++rank) {
          const std::size_t expert = selected[rank];
          std::vector<float> intermediate(width * 2U, 0.0F);
          for (std::size_t out = 0; out < width * 2U; ++out) {
            float sum = t36::value(layer.feed_input_bias,
                                    expert * width * 2U + out);
            for (std::size_t in = 0; in < width; ++in) {
              const std::size_t weight_index =
                  (expert * width + in) * width * 2U + out;
              sum += normalized[row * width + in] *
                     t36::value(layer.feed_input_weight, weight_index);
            }
            intermediate[out] = std::max(sum, 0.0F);
          }
          for (std::size_t out = 0; out < width; ++out) {
            float sum =
                t36::value(layer.feed_output_bias, expert * width + out);
            for (std::size_t in = 0; in < width * 2U; ++in) {
              const std::size_t weight_index =
                  (expert * width * 2U + in) * width + out;
              sum += intermediate[in] *
                     t36::value(layer.feed_output_weight, weight_index);
            }
            residual[row * width + out] += probabilities[rank] * sum;
          }
        }
      }
      for (std::size_t index = 0; index < hidden.size(); ++index)
        hidden[index] += residual[index];
    }
  }
  if (topology.variant == GenebSequenceCnnVariant::kSpace) {
    --rows;
    hidden.resize(rows * width);
  }
  const std::size_t trim = (rows - topology.target_length) / 2U;
  std::vector<float> cropped(topology.target_length * width);
  for (std::size_t row = 0; row < topology.target_length; ++row)
    std::copy_n(
        hidden.begin() +
            static_cast<std::ptrdiff_t>((trim + row) * width),
        width,
        cropped.begin() + static_cast<std::ptrdiff_t>(row * width));
  rows = topology.target_length;
  status = batch_norm(cropped, rows, width, impl_->final_norm.weight,
                      impl_->final_norm.bias, impl_->final_norm.running_mean,
                      impl_->final_norm.running_variance,
                      topology.batch_norm_epsilon, &normalized);
  if (!status.ok())
    return status;
  for (float &item : normalized)
    item = sigmoid_gelu(item, topology.gelu_sigmoid_scale);
  status = same_convolution(normalized, rows, width, impl_->final_weight,
                            impl_->final_bias, topology.output_width, 1U,
                            &stage);
  if (!status.ok())
    return status;
  for (float &item : stage)
    item = sigmoid_gelu(item, topology.gelu_sigmoid_scale);
  output->rows = rows;
  output->width = topology.output_width;
  output->final_hidden = std::move(stage);
  return Status::Ok();
}

Status GenebSequenceCnnModel::pool(
    const GenebSequenceCnnForwardResult &forward,
    std::vector<float> *const output) const {
  return geneb_sequence_cnn_pool(forward, output);
}

} // namespace evo::cpu
