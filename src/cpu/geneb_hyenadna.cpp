// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_hyenadna.hpp"

#include "fft_convolution.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace evo::cpu {
namespace {

constexpr std::size_t kMaximumDimension = 1U << 20U;
constexpr std::size_t kMaximumLayers = 1024U;
constexpr std::size_t kMaximumSequenceLength = 1U << 24U;
constexpr std::string_view kTokenizerProfile = "evo-tokenizer-v1";
constexpr std::string_view kPinnedModelingSha =
    "d78029cacea2c9259f1293fdb50b60f90872089469185e27d5c769086c475607";
constexpr std::string_view kPinnedTokenizationSha =
    "d424da2f794958b0360e033bbf5edc9ee4e3e3126e883093621348f07761e408";
constexpr std::string_view kPinnedExtractorSha =
    "b449660cc7c2f0efb06e771ca0206f890b7cd7a523bbf24b877e7494449f8941";
constexpr std::string_view kPinnedExtractorCommit =
    "b465d2d6a11efbbc9a22c105e34832725ce50e05";

Status invalid(const std::string &message) {
  return {ErrorCode::kInvalidArgument, "GENEB HyenaDNA: " + message};
}

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "GENEB HyenaDNA artifact: " + message};
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
  std::uint64_t value = 0U;
  for (std::size_t index = 0U; index < sizeof(value); ++index)
    value |= static_cast<std::uint64_t>(entry->value[index]) << (index * 8U);
  *output = value;
  return true;
}

bool metadata_f64(const ModelFile &artifact, const std::string_view key,
                  double *const output) {
  std::uint64_t bits = 0U;
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kF64 ||
      entry->value.size() != sizeof(bits))
    return false;
  for (std::size_t index = 0U; index < sizeof(bits); ++index)
    bits |= static_cast<std::uint64_t>(entry->value[index]) << (index * 8U);
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

bool is_lower_hex_sha256(const std::string &value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

Status require_sha(const ModelFile &artifact, const std::string_view key) {
  std::string value;
  if (!metadata_string(artifact, key, &value) || !is_lower_hex_sha256(value))
    return format_error("metadata '" + std::string{key} +
                        "' must be a lowercase SHA256");
  return Status::Ok();
}

Status metadata_size(const ModelFile &artifact, const std::string_view key,
                     std::size_t *const output) {
  std::uint64_t value = 0U;
  if (!metadata_u64(artifact, key, &value) || value == 0U ||
      value >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    return format_error("metadata '" + std::string{key} +
                        "' must be a positive size");
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

float value_at(const GenebHyenaDnaTensorView &tensor,
               const std::size_t index) noexcept {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(float), sizeof(value));
  return value;
}

Status expected_bytes(const std::vector<std::size_t> &shape,
                      const TensorDType dtype, std::size_t *const output) {
  std::size_t elements = 1U;
  for (const auto dimension : shape) {
    if (dimension == 0U || !checked_mul(elements, dimension, &elements))
      return invalid("tensor shape overflows");
  }
  const std::size_t scalar = dtype == TensorDType::kF32 ? 4U : 0U;
  if (scalar == 0U || !checked_mul(elements, scalar, output))
    return invalid("tensor dtype/size is unsupported");
  return Status::Ok();
}

void requirement(std::vector<GenebHyenaDnaTensorRequirement> *const output,
                 std::string name, std::vector<std::size_t> shape) {
  output->push_back({std::move(name), TensorDType::kF32, std::move(shape)});
}

Status validate_tensor(const GenebHyenaDnaTensorView &tensor,
                       const GenebHyenaDnaTensorRequirement &required) {
  if (tensor.data == nullptr)
    return invalid("tensor '" + required.name + "' has null data");
  if (tensor.dtype != required.dtype)
    return invalid("tensor '" + required.name + "' has wrong dtype");
  if (tensor.shape != required.shape)
    return invalid("tensor '" + required.name + "' has wrong shape");
  std::size_t bytes = 0U;
  auto status = expected_bytes(required.shape, required.dtype, &bytes);
  if (!status.ok())
    return status;
  if (tensor.bytes != bytes)
    return invalid("tensor '" + required.name + "' has wrong byte size");
  return Status::Ok();
}

Status linear(const std::vector<float> &input, const std::size_t rows,
              const std::size_t input_width,
              const GenebHyenaDnaTensorView &weight,
              const GenebHyenaDnaTensorView *const bias,
              const std::size_t output_width,
              std::vector<float> *const output) {
  std::size_t input_elements = 0U;
  if (output == nullptr || !checked_mul(rows, input_width, &input_elements) ||
      input.size() != input_elements || weight.data == nullptr ||
      weight.dtype != TensorDType::kF32 ||
      weight.shape != std::vector<std::size_t>{output_width, input_width})
    return invalid("linear dimensions differ");
  if (bias != nullptr &&
      (bias->data == nullptr || bias->dtype != TensorDType::kF32 ||
       bias->shape != std::vector<std::size_t>{output_width}))
    return invalid("linear bias dimensions differ");
  std::size_t output_elements = 0U;
  if (!checked_mul(rows, output_width, &output_elements))
    return invalid("linear output overflows");
  output->assign(output_elements, 0.0F);
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t target = 0U; target < output_width; ++target) {
      float total = bias == nullptr ? 0.0F : value_at(*bias, target);
      const std::size_t base = target * input_width;
      for (std::size_t source = 0U; source < input_width; ++source)
        total +=
            input[row * input_width + source] * value_at(weight, base + source);
      (*output)[row * output_width + target] = total;
    }
  }
  return Status::Ok();
}

Status layer_norm(const std::vector<float> &input, const std::size_t rows,
                  const std::size_t width,
                  const GenebHyenaDnaTensorView &weight,
                  const GenebHyenaDnaTensorView &bias, const float epsilon,
                  std::vector<float> *const output) {
  if (output == nullptr || input.size() != rows * width ||
      weight.shape != std::vector<std::size_t>{width} ||
      bias.shape != std::vector<std::size_t>{width})
    return invalid("LayerNorm dimensions differ");
  output->resize(input.size());
  for (std::size_t row = 0U; row < rows; ++row) {
    const std::size_t base = row * width;
    float sum = 0.0F;
    for (std::size_t column = 0U; column < width; ++column)
      sum += input[base + column];
    const float mean = sum / static_cast<float>(width);
    float squared = 0.0F;
    for (std::size_t column = 0U; column < width; ++column) {
      const float centered = input[base + column] - mean;
      squared += centered * centered;
    }
    const float inverse =
        1.0F / std::sqrt(squared / static_cast<float>(width) + epsilon);
    for (std::size_t column = 0U; column < width; ++column)
      (*output)[base + column] =
          (input[base + column] - mean) * inverse * value_at(weight, column) +
          value_at(bias, column);
  }
  return Status::Ok();
}

float gelu_tanh(const float value) {
  constexpr float kScale = 0.7978845608028654F;
  return 0.5F * value *
         (1.0F +
          std::tanh(kScale * (value + 0.044715F * value * value * value)));
}

void add_in_place(std::vector<float> *const destination,
                  const std::vector<float> &source) {
  for (std::size_t index = 0U; index < destination->size(); ++index)
    (*destination)[index] += source[index];
}

struct Layer final {
  GenebHyenaDnaTensorView filter_bias;
  GenebHyenaDnaTensorView implicit0_bias;
  GenebHyenaDnaTensorView implicit0_weight;
  GenebHyenaDnaTensorView frequency;
  GenebHyenaDnaTensorView implicit2_bias;
  GenebHyenaDnaTensorView implicit2_weight;
  GenebHyenaDnaTensorView implicit4_bias;
  GenebHyenaDnaTensorView implicit4_weight;
  GenebHyenaDnaTensorView implicit6_weight;
  GenebHyenaDnaTensorView deltas;
  GenebHyenaDnaTensorView position_t;
  GenebHyenaDnaTensorView position_z;
  GenebHyenaDnaTensorView in_bias;
  GenebHyenaDnaTensorView in_weight;
  GenebHyenaDnaTensorView out_bias;
  GenebHyenaDnaTensorView out_weight;
  GenebHyenaDnaTensorView short_bias;
  GenebHyenaDnaTensorView short_weight;
  GenebHyenaDnaTensorView fc1_bias;
  GenebHyenaDnaTensorView fc1_weight;
  GenebHyenaDnaTensorView fc2_bias;
  GenebHyenaDnaTensorView fc2_weight;
  GenebHyenaDnaTensorView norm1_bias;
  GenebHyenaDnaTensorView norm1_weight;
  GenebHyenaDnaTensorView norm2_bias;
  GenebHyenaDnaTensorView norm2_weight;
};

struct Runtime final {
  GenebHyenaDnaTensorView embedding;
  std::vector<Layer> layers;
  GenebHyenaDnaTensorView final_norm_bias;
  GenebHyenaDnaTensorView final_norm_weight;
  GenebHyenaDnaTensorView lm_head;
};

const GenebHyenaDnaTensorView &
find_required(const std::map<std::string, GenebHyenaDnaTensorView> &tensors,
              const std::string &name) {
  return tensors.find(name)->second;
}

Status bind_runtime(const GenebHyenaDnaTopology &topology,
                    const std::vector<GenebHyenaDnaNamedTensorView> &tensors,
                    Runtime *const output) {
  std::vector<GenebHyenaDnaTensorRequirement> requirements;
  auto status = canonical_geneb_hyenadna_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  if (tensors.size() != requirements.size())
    return invalid("tensor set size differs from canonical manifest");
  std::map<std::string, GenebHyenaDnaTensorView> by_name;
  for (const auto &item : tensors) {
    if (item.name.empty() || !by_name.emplace(item.name, item.tensor).second)
      return invalid("tensor names are empty or duplicated");
  }
  for (const auto &required : requirements) {
    const auto found = by_name.find(required.name);
    if (found == by_name.end())
      return invalid("required tensor '" + required.name + "' is missing");
    status = validate_tensor(found->second, required);
    if (!status.ok())
      return status;
  }

  Runtime runtime;
  runtime.embedding = find_required(
      by_name, "hyena.backbone.embeddings.word_embeddings.weight");
  runtime.layers.resize(topology.layers);
  for (std::size_t index = 0U; index < topology.layers; ++index) {
    const std::string prefix =
        "hyena.backbone.layers." + std::to_string(index) + ".";
    auto &layer = runtime.layers[index];
    layer.filter_bias = find_required(by_name, prefix + "mixer.filter_fn.bias");
    layer.implicit0_bias = find_required(
        by_name, prefix + "mixer.filter_fn.implicit_filter.0.bias");
    layer.implicit0_weight = find_required(
        by_name, prefix + "mixer.filter_fn.implicit_filter.0.weight");
    layer.frequency = find_required(
        by_name, prefix + "mixer.filter_fn.implicit_filter.1.freq");
    layer.implicit2_bias = find_required(
        by_name, prefix + "mixer.filter_fn.implicit_filter.2.bias");
    layer.implicit2_weight = find_required(
        by_name, prefix + "mixer.filter_fn.implicit_filter.2.weight");
    layer.implicit4_bias = find_required(
        by_name, prefix + "mixer.filter_fn.implicit_filter.4.bias");
    layer.implicit4_weight = find_required(
        by_name, prefix + "mixer.filter_fn.implicit_filter.4.weight");
    layer.implicit6_weight = find_required(
        by_name, prefix + "mixer.filter_fn.implicit_filter.6.weight");
    layer.deltas =
        find_required(by_name, prefix + "mixer.filter_fn.modulation.deltas");
    layer.position_t =
        find_required(by_name, prefix + "mixer.filter_fn.pos_emb.t");
    layer.position_z =
        find_required(by_name, prefix + "mixer.filter_fn.pos_emb.z");
    layer.in_bias = find_required(by_name, prefix + "mixer.in_proj.bias");
    layer.in_weight = find_required(by_name, prefix + "mixer.in_proj.weight");
    layer.out_bias = find_required(by_name, prefix + "mixer.out_proj.bias");
    layer.out_weight = find_required(by_name, prefix + "mixer.out_proj.weight");
    layer.short_bias =
        find_required(by_name, prefix + "mixer.short_filter.bias");
    layer.short_weight =
        find_required(by_name, prefix + "mixer.short_filter.weight");
    layer.fc1_bias = find_required(by_name, prefix + "mlp.fc1.bias");
    layer.fc1_weight = find_required(by_name, prefix + "mlp.fc1.weight");
    layer.fc2_bias = find_required(by_name, prefix + "mlp.fc2.bias");
    layer.fc2_weight = find_required(by_name, prefix + "mlp.fc2.weight");
    layer.norm1_bias = find_required(by_name, prefix + "norm1.bias");
    layer.norm1_weight = find_required(by_name, prefix + "norm1.weight");
    layer.norm2_bias = find_required(by_name, prefix + "norm2.bias");
    layer.norm2_weight = find_required(by_name, prefix + "norm2.weight");
  }
  runtime.final_norm_bias = find_required(by_name, "hyena.backbone.ln_f.bias");
  runtime.final_norm_weight =
      find_required(by_name, "hyena.backbone.ln_f.weight");
  runtime.lm_head = find_required(by_name, "lm_head.weight");
  *output = std::move(runtime);
  return Status::Ok();
}

void accumulate_stats(const detail::FftWorkStats &source,
                      GenebLongContextStats *const destination) {
  destination->maximum_fft_transform_length = std::max(
      destination->maximum_fft_transform_length, source.transform_length);
  destination->maximum_fft_radix2_length =
      std::max(destination->maximum_fft_radix2_length, source.radix2_length);
  destination->peak_fft_complex_values = std::max(
      destination->peak_fft_complex_values, source.peak_complex_values);
  if (source.butterfly_count > std::numeric_limits<std::uint64_t>::max() -
                                   destination->fft_butterfly_count)
    destination->fft_butterfly_count =
        std::numeric_limits<std::uint64_t>::max();
  else
    destination->fft_butterfly_count += source.butterfly_count;
}

Status implicit_filter(const Layer &layer,
                       const GenebHyenaDnaTopology &topology,
                       const std::size_t rows,
                       std::vector<float> *const output) {
  std::vector<float> position(rows * topology.positional_width, 0.0F);
  for (std::size_t index = 0U; index < position.size(); ++index)
    position[index] = value_at(layer.position_z, index);
  std::vector<float> hidden;
  auto status =
      linear(position, rows, topology.positional_width, layer.implicit0_weight,
             &layer.implicit0_bias, topology.filter_width, &hidden);
  if (!status.ok())
    return status;
  const auto activate = [&](std::vector<float> *const values) {
    for (std::size_t row = 0U; row < rows; ++row)
      for (std::size_t column = 0U; column < topology.filter_width; ++column)
        (*values)[row * topology.filter_width + column] =
            std::sin(value_at(layer.frequency, column) *
                     (*values)[row * topology.filter_width + column]);
  };
  activate(&hidden);
  std::vector<float> next;
  status = linear(hidden, rows, topology.filter_width, layer.implicit2_weight,
                  &layer.implicit2_bias, topology.filter_width, &next);
  if (!status.ok())
    return status;
  activate(&next);
  status = linear(next, rows, topology.filter_width, layer.implicit4_weight,
                  &layer.implicit4_bias, topology.filter_width, &hidden);
  if (!status.ok())
    return status;
  activate(&hidden);
  status = linear(hidden, rows, topology.filter_width, layer.implicit6_weight,
                  nullptr, topology.width, output);
  if (!status.ok())
    return status;
  for (std::size_t row = 0U; row < rows; ++row) {
    const float time = value_at(layer.position_t, row);
    for (std::size_t channel = 0U; channel < topology.width; ++channel) {
      const float decay =
          std::exp(-time * std::abs(value_at(layer.deltas, channel))) + 0.05F;
      (*output)[row * topology.width + channel] *= decay;
    }
  }
  return Status::Ok();
}

Status hyena_mixer(const Layer &layer, const GenebHyenaDnaTopology &topology,
                   const std::vector<float> &input,
                   GenebLongContextStats *const work,
                   std::vector<float> *const output) {
  const std::size_t rows = input.size() / topology.width;
  std::vector<float> projected;
  auto status = linear(input, rows, topology.width, layer.in_weight,
                       &layer.in_bias, topology.width * 3U, &projected);
  if (!status.ok())
    return status;
  std::vector<float> filtered(projected.size(), 0.0F);
  const std::size_t channels = topology.width * 3U;
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t channel = 0U; channel < channels; ++channel) {
      float total = value_at(layer.short_bias, channel);
      for (std::size_t tap = 0U; tap < topology.short_filter_width; ++tap) {
        const std::ptrdiff_t source =
            static_cast<std::ptrdiff_t>(row) +
            static_cast<std::ptrdiff_t>(tap) -
            static_cast<std::ptrdiff_t>(topology.short_filter_width - 1U);
        if (source >= 0)
          total +=
              projected[static_cast<std::size_t>(source) * channels + channel] *
              value_at(layer.short_weight,
                       channel * topology.short_filter_width + tap);
      }
      filtered[row * channels + channel] = total;
    }
  }
  std::vector<float> kernel;
  status = implicit_filter(layer, topology, rows, &kernel);
  if (!status.ok())
    return status;
  std::vector<float> mixed(rows * topology.width, 0.0F);
  std::vector<float> signal(rows);
  std::vector<float> channel_kernel(rows);
  std::vector<float> convolved;
  for (std::size_t channel = 0U; channel < topology.width; ++channel) {
    for (std::size_t row = 0U; row < rows; ++row) {
      const std::size_t base = row * channels;
      signal[row] = filtered[base + topology.width + channel] *
                    filtered[base + topology.width * 2U + channel];
      channel_kernel[row] = kernel[row * topology.width + channel];
    }
    detail::FftWorkStats stats;
    status = detail::causal_fft_convolution(signal, channel_kernel, &convolved,
                                            &stats);
    if (!status.ok())
      return status;
    accumulate_stats(stats, work);
    const float diagonal = value_at(layer.filter_bias, channel);
    for (std::size_t row = 0U; row < rows; ++row) {
      const std::size_t base = row * channels;
      mixed[row * topology.width + channel] =
          filtered[base + channel] * (convolved[row] + signal[row] * diagonal);
    }
  }
  return linear(mixed, rows, topology.width, layer.out_weight, &layer.out_bias,
                topology.width, output);
}

bool wants_capture(const std::set<std::size_t> &captures,
                   const std::size_t layer) {
  return captures.find(layer) != captures.end();
}

void capture(const std::size_t layer, const std::vector<float> &values,
             const std::set<std::size_t> &captures,
             std::vector<GenebHyenaDnaHiddenCapture> *const output) {
  if (wants_capture(captures, layer))
    output->push_back({layer, values});
}

Status
model_file_tensors(const ModelFile &artifact,
                   std::vector<GenebHyenaDnaNamedTensorView> *const output) {
  output->clear();
  output->reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    if (tensor.data_size >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      return format_error("tensor byte size exceeds size_t");
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t index = 0U; index < tensor.rank; ++index) {
      if (tensor.dimensions[index] >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return format_error("tensor dimension exceeds size_t");
      shape.push_back(static_cast<std::size_t>(tensor.dimensions[index]));
    }
    output->push_back({tensor.name,
                       {artifact.tensor_data(tensor),
                        static_cast<std::size_t>(tensor.data_size),
                        tensor.dtype, std::move(shape)}});
  }
  return Status::Ok();
}

} // namespace

struct GenebHyenaDnaModel::Impl final {
  GenebHyenaDnaTopology topology;
  Runtime runtime;
};

Status validate_geneb_hyenadna_topology(const GenebHyenaDnaTopology &topology) {
  if (topology.vocabulary_size == 0U ||
      !token_vocabulary_size_supported(topology.vocabulary_size) ||
      topology.embedding_rows < topology.vocabulary_size ||
      topology.embedding_rows > kMaximumDimension || topology.width == 0U ||
      topology.width > kMaximumDimension || topology.layers == 0U ||
      topology.layers > kMaximumLayers || topology.inner_width == 0U ||
      topology.inner_width > kMaximumDimension || topology.filter_width == 0U ||
      topology.filter_width > kMaximumDimension ||
      topology.positional_width < 3U ||
      topology.positional_width > kMaximumDimension ||
      topology.positional_width % 2U == 0U ||
      topology.short_filter_width == 0U ||
      topology.short_filter_width > 4096U ||
      topology.maximum_sequence_length == 0U ||
      topology.maximum_sequence_length > kMaximumSequenceLength ||
      !std::isfinite(topology.norm_epsilon) || topology.norm_epsilon <= 0.0F)
    return invalid("topology is outside the bounded runtime contract");
  return Status::Ok();
}

Status
geneb_hyenadna_topology_from_artifact(const ModelFile &artifact,
                                      GenebHyenaDnaTopology *const output) {
  if (output == nullptr)
    return invalid("topology output is null");
  if (artifact.profile() != kGenebHyenaDnaArtifactProfile)
    return format_error("profile must be '" +
                        std::string{kGenebHyenaDnaArtifactProfile} + "'");
  for (const auto &literal :
       {std::pair{"runtime.abi", kGenebHyenaDnaRuntimeAbi},
        std::pair{"model.architecture", kGenebHyenaDnaArchitecture},
        std::pair{"tokenizer.profile", kTokenizerProfile},
        std::pair{"hyenadna.weight_dtype", std::string_view{"F32"}},
        std::pair{"hyenadna.hidden_tap",
                  std::string_view{"post-final-layernorm"}},
        std::pair{"hyenadna.pooling", std::string_view{"attention-mask-mean"}},
        std::pair{"hyenadna.padding_side", std::string_view{"left"}},
        std::pair{"hyenadna.activation", std::string_view{"gelu-tanh"}},
        std::pair{"hyenadna.long_convolution",
                  std::string_view{"exact-2l-fft"}},
        std::pair{"source.modeling_hyena_sha256", kPinnedModelingSha},
        std::pair{"source.modeling_hyena_path",
                  std::string_view{"modeling_hyena.py"}},
        std::pair{"source.modeling_hyena_lines",
                  std::string_view{"14-20,86-160,214-235,250-348"}},
        std::pair{"source.tokenization_hyena_sha256", kPinnedTokenizationSha},
        std::pair{"source.tokenization_hyena_path",
                  std::string_view{"tokenization_hyena.py"}},
        std::pair{"source.tokenization_hyena_lines", std::string_view{"8-76"}},
        std::pair{"source.tokenizer_config_path",
                  std::string_view{"tokenizer_config.json"}},
        std::pair{"source.geneb.extractor_sha256", kPinnedExtractorSha},
        std::pair{
            "source.geneb.extractor_path",
            std::string_view{"embedding_pipeline/extractors/hyenadna.py"}},
        std::pair{"source.geneb.extractor_lines",
                  std::string_view{"14,16-24,40-56"}},
        std::pair{"geneb.provenance.extractor_commit",
                  kPinnedExtractorCommit}}) {
    auto status = require_literal(artifact, literal.first, literal.second);
    if (!status.ok())
      return status;
  }
  for (const auto &flag :
       {std::pair{"hyenadna.model_receives_attention_mask", false},
        std::pair{"hyenadna.add_special_tokens", false},
        std::pair{"source.immutable", true},
        std::pair{"geneb.source.immutable", true}}) {
    auto status = require_bool(artifact, flag.first, flag.second);
    if (!status.ok())
      return status;
  }
  for (const auto key :
       {"source.config_sha256", "source.tokenizer_config_sha256",
        "source.checkpoint_manifest_sha256"}) {
    auto status = require_sha(artifact, key);
    if (!status.ok())
      return status;
  }
  if (!artifact.tokenizer_asset_descriptor().has_value())
    return format_error("complete tokenizer descriptor is required");

  GenebHyenaDnaTopology topology;
  for (const auto &field :
       {std::pair{"config.vocab_size", &topology.vocabulary_size},
        std::pair{"config.hidden_size", &topology.width},
        std::pair{"config.num_layers", &topology.layers},
        std::pair{"config.max_seqlen", &topology.maximum_sequence_length},
        std::pair{"hyenadna.embedding_rows", &topology.embedding_rows},
        std::pair{"hyenadna.inner_width", &topology.inner_width},
        std::pair{"hyenadna.filter_width", &topology.filter_width},
        std::pair{"hyenadna.positional_width", &topology.positional_width},
        std::pair{"hyenadna.short_filter_width",
                  &topology.short_filter_width}}) {
    auto status = metadata_size(artifact, field.first, field.second);
    if (!status.ok())
      return status;
  }
  double epsilon = 0.0;
  if (!metadata_f64(artifact, "hyenadna.norm_epsilon", &epsilon) ||
      !std::isfinite(epsilon) || epsilon <= 0.0 ||
      epsilon > std::numeric_limits<float>::max())
    return format_error("norm epsilon metadata is invalid");
  topology.norm_epsilon = static_cast<float>(epsilon);
  std::uint64_t embedding_layers = 0U;
  std::uint64_t tokenizer_vocabulary_size = 0U;
  if (!metadata_u64(artifact, "runtime.embedding_layer_count",
                    &embedding_layers) ||
      embedding_layers != static_cast<std::uint64_t>(topology.layers) + 1U)
    return format_error(
        "runtime.embedding_layer_count must equal config.num_layers+1");
  if (!metadata_u64(artifact, "runtime.tokenizer_vocabulary_size",
                    &tokenizer_vocabulary_size) ||
      tokenizer_vocabulary_size != topology.vocabulary_size)
    return format_error(
        "runtime.tokenizer_vocabulary_size must equal config.vocab_size");
  auto status = validate_geneb_hyenadna_topology(topology);
  if (!status.ok())
    return status;
  *output = topology;
  return Status::Ok();
}

Status canonical_geneb_hyenadna_tensors(
    const GenebHyenaDnaTopology &topology,
    std::vector<GenebHyenaDnaTensorRequirement> *const output) {
  if (output == nullptr)
    return invalid("tensor manifest output is null");
  auto status = validate_geneb_hyenadna_topology(topology);
  if (!status.ok())
    return status;
  output->clear();
  requirement(output, "hyena.backbone.embeddings.word_embeddings.weight",
              {topology.embedding_rows, topology.width});
  for (std::size_t index = 0U; index < topology.layers; ++index) {
    const std::string prefix =
        "hyena.backbone.layers." + std::to_string(index) + ".";
    requirement(output, prefix + "mixer.filter_fn.bias", {topology.width});
    requirement(output, prefix + "mixer.filter_fn.implicit_filter.0.bias",
                {topology.filter_width});
    requirement(output, prefix + "mixer.filter_fn.implicit_filter.0.weight",
                {topology.filter_width, topology.positional_width});
    requirement(output, prefix + "mixer.filter_fn.implicit_filter.1.freq",
                {1U, topology.filter_width});
    requirement(output, prefix + "mixer.filter_fn.implicit_filter.2.bias",
                {topology.filter_width});
    requirement(output, prefix + "mixer.filter_fn.implicit_filter.2.weight",
                {topology.filter_width, topology.filter_width});
    requirement(output, prefix + "mixer.filter_fn.implicit_filter.4.bias",
                {topology.filter_width});
    requirement(output, prefix + "mixer.filter_fn.implicit_filter.4.weight",
                {topology.filter_width, topology.filter_width});
    requirement(output, prefix + "mixer.filter_fn.implicit_filter.6.weight",
                {topology.width, topology.filter_width});
    requirement(output, prefix + "mixer.filter_fn.modulation.deltas",
                {1U, 1U, topology.width});
    requirement(output, prefix + "mixer.filter_fn.pos_emb.t",
                {1U, topology.maximum_sequence_length, 1U});
    requirement(
        output, prefix + "mixer.filter_fn.pos_emb.z",
        {1U, topology.maximum_sequence_length, topology.positional_width});
    requirement(output, prefix + "mixer.in_proj.bias", {topology.width * 3U});
    requirement(output, prefix + "mixer.in_proj.weight",
                {topology.width * 3U, topology.width});
    requirement(output, prefix + "mixer.out_proj.bias", {topology.width});
    requirement(output, prefix + "mixer.out_proj.weight",
                {topology.width, topology.width});
    requirement(output, prefix + "mixer.short_filter.bias",
                {topology.width * 3U});
    requirement(output, prefix + "mixer.short_filter.weight",
                {topology.width * 3U, 1U, topology.short_filter_width});
    requirement(output, prefix + "mlp.fc1.bias", {topology.inner_width});
    requirement(output, prefix + "mlp.fc1.weight",
                {topology.inner_width, topology.width});
    requirement(output, prefix + "mlp.fc2.bias", {topology.width});
    requirement(output, prefix + "mlp.fc2.weight",
                {topology.width, topology.inner_width});
    requirement(output, prefix + "norm1.bias", {topology.width});
    requirement(output, prefix + "norm1.weight", {topology.width});
    requirement(output, prefix + "norm2.bias", {topology.width});
    requirement(output, prefix + "norm2.weight", {topology.width});
  }
  requirement(output, "hyena.backbone.ln_f.bias", {topology.width});
  requirement(output, "hyena.backbone.ln_f.weight", {topology.width});
  requirement(output, "lm_head.weight",
              {topology.embedding_rows, topology.width});
  return Status::Ok();
}

Status geneb_long_causal_convolution(const std::vector<float> &input,
                                     const std::vector<float> &kernel,
                                     std::vector<float> *const output,
                                     GenebLongContextStats *const stats) {
  detail::FftWorkStats internal;
  auto status =
      detail::causal_fft_convolution(input, kernel, output, &internal);
  if (!status.ok())
    return status;
  if (stats != nullptr) {
    *stats = {};
    accumulate_stats(internal, stats);
  }
  return Status::Ok();
}

GenebHyenaDnaModel::GenebHyenaDnaModel() = default;
GenebHyenaDnaModel::~GenebHyenaDnaModel() = default;
GenebHyenaDnaModel::GenebHyenaDnaModel(GenebHyenaDnaModel &&) noexcept =
    default;
GenebHyenaDnaModel &
GenebHyenaDnaModel::operator=(GenebHyenaDnaModel &&) noexcept = default;

Status GenebHyenaDnaModel::load(
    const GenebHyenaDnaTopology &topology,
    const std::vector<GenebHyenaDnaNamedTensorView> &tensors) {
  auto status = validate_geneb_hyenadna_topology(topology);
  if (!status.ok())
    return status;
  auto implementation = std::make_unique<Impl>();
  implementation->topology = topology;
  status = bind_runtime(topology, tensors, &implementation->runtime);
  if (!status.ok())
    return status;
  impl_ = std::move(implementation);
  return Status::Ok();
}

Status GenebHyenaDnaModel::load(const ModelFile &artifact) {
  GenebHyenaDnaTopology topology;
  auto status = geneb_hyenadna_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebHyenaDnaNamedTensorView> tensors;
  status = model_file_tensors(artifact, &tensors);
  if (!status.ok())
    return status;
  return load(topology, tensors);
}

const GenebHyenaDnaTopology *GenebHyenaDnaModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebHyenaDnaModel::kernel_name() const noexcept {
  return impl_ == nullptr ? "unloaded" : "geneb-hyenadna-f32-bluestein-fft";
}

Status
GenebHyenaDnaModel::forward(const std::vector<TokenId> &tokens,
                            const std::vector<std::uint8_t> &attention_mask,
                            const std::vector<std::size_t> &capture_layers,
                            GenebHyenaDnaForwardResult *const output) const {
  if (impl_ == nullptr)
    return invalid("model is not loaded");
  if (output == nullptr)
    return invalid("forward output is null");
  const auto &topology = impl_->topology;
  if (tokens.empty() || tokens.size() > topology.maximum_sequence_length ||
      attention_mask.size() != tokens.size())
    return invalid("token/mask length is outside the model context");
  bool saw_valid = false;
  std::size_t valid_count = 0U;
  for (std::size_t row = 0U; row < tokens.size(); ++row) {
    if (!token_id_in_vocabulary(tokens[row], topology.vocabulary_size))
      return invalid("token ID is outside the logical vocabulary");
    if (attention_mask[row] > 1U)
      return invalid("attention mask must be binary");
    if (attention_mask[row] != 0U) {
      saw_valid = true;
      ++valid_count;
    } else if (saw_valid) {
      return invalid("attention mask must be left padded");
    }
  }
  if (valid_count == 0U)
    return invalid("attention mask has no valid token");
  std::set<std::size_t> captures;
  for (const auto layer : capture_layers) {
    if (layer > topology.layers || !captures.insert(layer).second)
      return invalid("capture layers are out of range or duplicated");
  }

  try {
    GenebHyenaDnaForwardResult result;
    result.rows = tokens.size();
    result.width = topology.width;
    std::vector<float> hidden(tokens.size() * topology.width, 0.0F);
    for (std::size_t row = 0U; row < tokens.size(); ++row)
      for (std::size_t column = 0U; column < topology.width; ++column)
        hidden[row * topology.width + column] = value_at(
            impl_->runtime.embedding,
            static_cast<std::size_t>(tokens[row]) * topology.width + column);
    capture(0U, hidden, captures, &result.captures);

    for (std::size_t layer_index = 0U; layer_index < topology.layers;
         ++layer_index) {
      const auto &layer = impl_->runtime.layers[layer_index];
      std::vector<float> normalized;
      auto status =
          layer_norm(hidden, tokens.size(), topology.width, layer.norm1_weight,
                     layer.norm1_bias, topology.norm_epsilon, &normalized);
      if (!status.ok())
        return status;
      std::vector<float> mixed;
      status = hyena_mixer(layer, topology, normalized, &result.work, &mixed);
      if (!status.ok())
        return status;
      add_in_place(&hidden, mixed);
      status =
          layer_norm(hidden, tokens.size(), topology.width, layer.norm2_weight,
                     layer.norm2_bias, topology.norm_epsilon, &normalized);
      if (!status.ok())
        return status;
      std::vector<float> feed_forward;
      status =
          linear(normalized, tokens.size(), topology.width, layer.fc1_weight,
                 &layer.fc1_bias, topology.inner_width, &feed_forward);
      if (!status.ok())
        return status;
      for (auto &value : feed_forward)
        value = gelu_tanh(value);
      std::vector<float> feed_forward_output;
      status = linear(feed_forward, tokens.size(), topology.inner_width,
                      layer.fc2_weight, &layer.fc2_bias, topology.width,
                      &feed_forward_output);
      if (!status.ok())
        return status;
      add_in_place(&hidden, feed_forward_output);
      if (layer_index + 1U < topology.layers)
        capture(layer_index + 1U, hidden, captures, &result.captures);
    }
    auto status = layer_norm(hidden, tokens.size(), topology.width,
                             impl_->runtime.final_norm_weight,
                             impl_->runtime.final_norm_bias,
                             topology.norm_epsilon, &result.final_hidden);
    if (!status.ok())
      return status;
    capture(topology.layers, result.final_hidden, captures, &result.captures);
    result.pooled.assign(topology.width, 0.0F);
    for (std::size_t row = 0U; row < tokens.size(); ++row) {
      if (attention_mask[row] == 0U)
        continue;
      for (std::size_t column = 0U; column < topology.width; ++column)
        result.pooled[column] +=
            result.final_hidden[row * topology.width + column];
    }
    for (auto &value : result.pooled)
      value /= static_cast<float>(valid_count);
    if (!std::all_of(result.final_hidden.begin(), result.final_hidden.end(),
                     [](const float value) { return std::isfinite(value); }) ||
        !std::all_of(result.pooled.begin(), result.pooled.end(),
                     [](const float value) { return std::isfinite(value); }))
      return invalid("forward produced a non-finite value");
    *output = std::move(result);
  } catch (const std::bad_alloc &) {
    return {ErrorCode::kInternal, "GENEB HyenaDNA: forward allocation failed"};
  }
  return Status::Ok();
}

} // namespace evo::cpu
