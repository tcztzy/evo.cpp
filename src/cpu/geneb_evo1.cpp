// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_evo1.hpp"

#include "fft_convolution.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
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
constexpr std::string_view kPinnedRevision =
    "c206aab77ae5967a069c4200ecb1858588528c9d";
constexpr std::string_view kPinnedModelSha =
    "6a03a9c28bd7e282cb253d051ae7ec39159d14d3a1b2272913c8ff1747f3d87a";
constexpr std::string_view kPinnedEngineSha =
    "8fe4cd97ec6b6be43807fe2d33ec16b53d3ff831e5d33c25802e701041221639";
constexpr std::string_view kPinnedLayersSha =
    "52c7909406acccd07d872f839cd22f3f494c8555ac96d51ce7eb2629cc8a98da";
constexpr std::string_view kPinnedPositionSha =
    "fc22bdb7447ae3c0adea17c00442c3ec040f0dbb61e54c4486d14bbc5217c3ae";
constexpr std::string_view kPinnedTokenizerSha =
    "15e37ca8a1994a1bb4a9ac526d746808ef77a1b68e26ea9fe8c03f823d25c7be";
constexpr std::string_view kPinnedExtractorSha =
    "edc260d061f332360a38669418517184e848fe88ffbccf1aea3376006068f256";
constexpr std::string_view kPinnedExtractorCommit =
    "b465d2d6a11efbbc9a22c105e34832725ce50e05";

using Complex = std::complex<float>;

Status invalid(const std::string &message) {
  return {ErrorCode::kInvalidArgument, "GENEB Evo-1: " + message};
}

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "GENEB Evo-1 artifact: " + message};
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

bool metadata_u64_list(const ModelFile &artifact, const std::string_view key,
                       std::vector<std::size_t> *const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kU64List ||
      entry->value.size() % sizeof(std::uint64_t) != 0U)
    return false;
  output->clear();
  output->reserve(entry->value.size() / sizeof(std::uint64_t));
  for (std::size_t offset = 0U; offset < entry->value.size();
       offset += sizeof(std::uint64_t)) {
    std::uint64_t value = 0U;
    for (std::size_t index = 0U; index < sizeof(value); ++index)
      value |= static_cast<std::uint64_t>(entry->value[offset + index])
               << (index * 8U);
    if (value >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
      return false;
    output->push_back(static_cast<std::size_t>(value));
  }
  return true;
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

float bf16_value(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

float round_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t exponent = bits & 0x7F800000U;
  if (exponent != 0x7F800000U)
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return bf16_value(static_cast<std::uint16_t>(bits >> 16U));
}

float value_at(const GenebEvo1TensorView &tensor,
               const std::size_t index) noexcept {
  if (tensor.dtype == TensorDType::kF32) {
    float value = 0.0F;
    std::memcpy(&value, tensor.data + index * sizeof(float), sizeof(value));
    return value;
  }
  std::uint16_t value = 0U;
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return bf16_value(value);
}

std::size_t dtype_bytes(const TensorDType dtype) noexcept {
  if (dtype == TensorDType::kF32)
    return 4U;
  if (dtype == TensorDType::kBF16)
    return 2U;
  return 0U;
}

Status validate_tensor(const GenebEvo1TensorView &tensor,
                       const GenebEvo1TensorRequirement &required) {
  if (tensor.data == nullptr)
    return invalid("tensor '" + required.name + "' has null data");
  if (tensor.dtype != required.dtype)
    return invalid("tensor '" + required.name + "' has wrong dtype");
  if (tensor.shape != required.shape)
    return invalid("tensor '" + required.name + "' has wrong shape");
  std::size_t elements = 1U;
  for (const auto dimension : required.shape) {
    if (dimension == 0U || !checked_mul(elements, dimension, &elements))
      return invalid("tensor '" + required.name + "' shape overflows");
  }
  std::size_t bytes = 0U;
  if (dtype_bytes(required.dtype) == 0U ||
      !checked_mul(elements, dtype_bytes(required.dtype), &bytes) ||
      tensor.bytes != bytes)
    return invalid("tensor '" + required.name + "' has wrong byte size");
  return Status::Ok();
}

void requirement(std::vector<GenebEvo1TensorRequirement> *const output,
                 std::string name, const TensorDType dtype,
                 std::vector<std::size_t> shape) {
  output->push_back({std::move(name), dtype, std::move(shape)});
}

Status linear_bf16(const std::vector<float> &input, const std::size_t rows,
                   const std::size_t input_width,
                   const GenebEvo1TensorView &weight,
                   const GenebEvo1TensorView *const bias,
                   const std::size_t output_width,
                   std::vector<float> *const output) {
  if (output == nullptr || input.size() != rows * input_width ||
      weight.dtype != TensorDType::kBF16 ||
      weight.shape != std::vector<std::size_t>{output_width, input_width})
    return invalid("BF16 linear dimensions differ");
  if (bias != nullptr &&
      (bias->dtype != TensorDType::kBF16 ||
       bias->shape != std::vector<std::size_t>{output_width}))
    return invalid("BF16 linear bias dimensions differ");
  output->assign(rows * output_width, 0.0F);
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t target = 0U; target < output_width; ++target) {
      double total = bias == nullptr
                         ? 0.0
                         : static_cast<double>(value_at(*bias, target));
      const std::size_t base = target * input_width;
      for (std::size_t source = 0U; source < input_width; ++source)
        total += static_cast<double>(input[row * input_width + source]) *
                 static_cast<double>(value_at(weight, base + source));
      (*output)[row * output_width + target] =
          round_bf16(static_cast<float>(total));
    }
  }
  return Status::Ok();
}

Status rms_norm_bf16(const std::vector<float> &input, const std::size_t rows,
                     const std::size_t width, const GenebEvo1TensorView &scale,
                     const float epsilon, std::vector<float> *const output) {
  if (output == nullptr || input.size() != rows * width ||
      scale.dtype != TensorDType::kBF16 ||
      scale.shape != std::vector<std::size_t>{width})
    return invalid("BF16 RMSNorm dimensions differ");
  output->resize(input.size());
  for (std::size_t row = 0U; row < rows; ++row) {
    const std::size_t base = row * width;
    double squares = 0.0;
    for (std::size_t column = 0U; column < width; ++column)
      squares += static_cast<double>(input[base + column]) *
                 static_cast<double>(input[base + column]);
    // Pinned Evo v1 contract: epsilon is outside sqrt.
    const double denominator =
        std::sqrt(squares / static_cast<double>(width)) +
        static_cast<double>(epsilon);
    for (std::size_t column = 0U; column < width; ++column)
      (*output)[base + column] =
          round_bf16(static_cast<float>(
              static_cast<double>(input[base + column]) / denominator *
              static_cast<double>(value_at(scale, column))));
  }
  return Status::Ok();
}

void add_bf16(std::vector<float> *const destination,
              const std::vector<float> &source) {
  for (std::size_t index = 0U; index < destination->size(); ++index)
    (*destination)[index] = round_bf16(static_cast<float>(
        static_cast<double>((*destination)[index]) +
        static_cast<double>(source[index])));
}

float gelu_exact_bf16(const float value) {
  const double doubled = static_cast<double>(value);
  return round_bf16(static_cast<float>(
      0.5 * doubled *
      (1.0 + std::erf(doubled * 0.7071067811865475))));
}

struct CommonBlock final {
  GenebEvo1TensorView pre_norm;
  GenebEvo1TensorView post_norm;
  GenebEvo1TensorView mlp_l1;
  GenebEvo1TensorView mlp_l2;
  GenebEvo1TensorView mlp_l3;
};

struct HyenaBlock final {
  GenebEvo1TensorView diagonal;
  GenebEvo1TensorView poles;
  GenebEvo1TensorView residues;
  GenebEvo1TensorView short_bias;
  GenebEvo1TensorView short_weight;
  GenebEvo1TensorView out_bias;
  GenebEvo1TensorView out_weight;
  GenebEvo1TensorView projection_bias;
  GenebEvo1TensorView projection_weight;
};

struct AttentionBlock final {
  GenebEvo1TensorView qkv_bias;
  GenebEvo1TensorView qkv_weight;
  GenebEvo1TensorView out_bias;
  GenebEvo1TensorView out_weight;
  GenebEvo1TensorView inverse_frequency;
};

struct Block final {
  bool attention{false};
  CommonBlock common;
  HyenaBlock hyena;
  AttentionBlock mha;
};

struct Runtime final {
  GenebEvo1TensorView embedding;
  std::vector<Block> blocks;
  GenebEvo1TensorView final_norm;
};

const GenebEvo1TensorView &
find_required(const std::map<std::string, GenebEvo1TensorView> &tensors,
              const std::string &name) {
  return tensors.find(name)->second;
}

bool is_attention_layer(const GenebEvo1Topology &topology,
                        const std::size_t layer) {
  return std::binary_search(topology.attention_layers.begin(),
                            topology.attention_layers.end(), layer);
}

Status bind_runtime(const GenebEvo1Topology &topology,
                    const std::vector<GenebEvo1NamedTensorView> &tensors,
                    Runtime *const output) {
  std::vector<GenebEvo1TensorRequirement> requirements;
  auto status = canonical_geneb_evo1_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  if (tensors.size() != requirements.size())
    return invalid("tensor set size differs from canonical manifest");
  std::map<std::string, GenebEvo1TensorView> by_name;
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
  runtime.embedding = find_required(by_name, "backbone.embedding_layer.weight");
  runtime.final_norm = find_required(by_name, "backbone.norm.scale");
  runtime.blocks.resize(topology.layers);
  for (std::size_t layer = 0U; layer < topology.layers; ++layer) {
    const std::string prefix = "backbone.blocks." + std::to_string(layer) + ".";
    auto &block = runtime.blocks[layer];
    block.attention = is_attention_layer(topology, layer);
    block.common.pre_norm = find_required(by_name, prefix + "pre_norm.scale");
    block.common.post_norm = find_required(by_name, prefix + "post_norm.scale");
    block.common.mlp_l1 = find_required(by_name, prefix + "mlp.l1.weight");
    block.common.mlp_l2 = find_required(by_name, prefix + "mlp.l2.weight");
    block.common.mlp_l3 = find_required(by_name, prefix + "mlp.l3.weight");
    if (block.attention) {
      block.mha.qkv_bias =
          find_required(by_name, prefix + "inner_mha_cls.Wqkv.bias");
      block.mha.qkv_weight =
          find_required(by_name, prefix + "inner_mha_cls.Wqkv.weight");
      block.mha.out_bias =
          find_required(by_name, prefix + "inner_mha_cls.out_proj.bias");
      block.mha.out_weight =
          find_required(by_name, prefix + "inner_mha_cls.out_proj.weight");
      block.mha.inverse_frequency =
          find_required(by_name, prefix + "inner_mha_cls.rotary_emb.inv_freq");
    } else {
      block.hyena.diagonal = find_required(by_name, prefix + "filter.D");
      block.hyena.poles = find_required(by_name, prefix + "filter.poles");
      block.hyena.residues = find_required(by_name, prefix + "filter.residues");
      block.hyena.short_bias =
          find_required(by_name, prefix + "filter.short_filter_bias");
      block.hyena.short_weight =
          find_required(by_name, prefix + "filter.short_filter_weight");
      block.hyena.out_bias =
          find_required(by_name, prefix + "out_filter_dense.bias");
      block.hyena.out_weight =
          find_required(by_name, prefix + "out_filter_dense.weight");
      block.hyena.projection_bias =
          find_required(by_name, prefix + "projections.bias");
      block.hyena.projection_weight =
          find_required(by_name, prefix + "projections.weight");
    }
  }
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

Status mlp(const CommonBlock &block, const GenebEvo1Topology &topology,
           const std::vector<float> &input, std::vector<float> *const output) {
  const std::size_t rows = input.size() / topology.width;
  std::vector<float> first;
  std::vector<float> second;
  auto status = linear_bf16(input, rows, topology.width, block.mlp_l1, nullptr,
                            topology.inner_width, &first);
  if (!status.ok())
    return status;
  status = linear_bf16(input, rows, topology.width, block.mlp_l2, nullptr,
                       topology.inner_width, &second);
  if (!status.ok())
    return status;
  for (std::size_t index = 0U; index < first.size(); ++index)
    first[index] = round_bf16(gelu_exact_bf16(first[index]) * second[index]);
  return linear_bf16(first, rows, topology.inner_width, block.mlp_l3, nullptr,
                     topology.width, output);
}

Status hyena_block(const Block &block, const GenebEvo1Topology &topology,
                   std::vector<float> *const hidden,
                   GenebLongContextStats *const work) {
  const std::size_t rows = hidden->size() / topology.width;
  std::vector<float> normalized;
  auto status =
      rms_norm_bf16(*hidden, rows, topology.width, block.common.pre_norm,
                    topology.norm_epsilon, &normalized);
  if (!status.ok())
    return status;
  std::vector<float> projected;
  status = linear_bf16(
      normalized, rows, topology.width, block.hyena.projection_weight,
      &block.hyena.projection_bias, topology.width * 3U, &projected);
  if (!status.ok())
    return status;
  const std::size_t channels = topology.width * 3U;
  std::vector<float> short_output(projected.size(), 0.0F);
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t channel = 0U; channel < channels; ++channel) {
      double total = static_cast<double>(value_at(block.hyena.short_bias,
                                                  channel));
      for (std::size_t tap = 0U; tap < topology.short_filter_width; ++tap) {
        const std::ptrdiff_t source =
            static_cast<std::ptrdiff_t>(row) +
            static_cast<std::ptrdiff_t>(tap) -
            static_cast<std::ptrdiff_t>(topology.short_filter_width - 1U);
        if (source >= 0)
          total +=
              static_cast<double>(
                  projected[static_cast<std::size_t>(source) * channels +
                            channel]) *
              static_cast<double>(
                  value_at(block.hyena.short_weight,
                           channel * topology.short_filter_width + tap));
      }
      short_output[row * channels + channel] =
          round_bf16(static_cast<float>(total));
    }
  }

  const std::size_t head_width = topology.width / topology.heads;
  std::vector<float> mixed(rows * topology.width, 0.0F);
  std::vector<float> signal(rows);
  std::vector<float> kernel(rows);
  std::vector<float> convolved;
  for (std::size_t channel = 0U; channel < topology.width; ++channel) {
    const std::size_t head = channel / head_width;
    const std::size_t within = channel % head_width;
    const std::size_t x2_channel = head * (head_width * 3U) + within;
    const std::size_t x1_channel = x2_channel + head_width;
    const std::size_t value_channel = x1_channel + head_width;
    for (std::size_t row = 0U; row < rows; ++row) {
      const std::size_t base = row * channels;
      signal[row] = round_bf16(short_output[base + x1_channel] *
                               short_output[base + value_channel]);
      std::complex<double> total{0.0, 0.0};
      for (std::size_t state = 0U; state < topology.state_width; ++state) {
        const std::size_t tensor_index =
            ((channel * topology.state_width + state) * 2U);
        const std::complex<double> pole{
            static_cast<double>(value_at(block.hyena.poles, tensor_index)),
            static_cast<double>(value_at(block.hyena.poles,
                                         tensor_index + 1U))};
        const std::complex<double> residue{
            static_cast<double>(value_at(block.hyena.residues, tensor_index)),
            static_cast<double>(value_at(block.hyena.residues,
                                         tensor_index + 1U))};
        total += residue *
                 std::exp(std::log(pole) * static_cast<double>(row));
      }
      kernel[row] = static_cast<float>(total.real());
    }
    detail::FftWorkStats stats;
    if (rows <= 64U && topology.width >= 256U) {
      // Production-sized checkpoints use the direct short-sequence path so
      // canonical GENEB evidence matches the pinned double-accumulation
      // oracle exactly.  Tiny synthetic fixtures retain the frozen 2L FFT
      // contract and its work-statistics gates.
      convolved.resize(rows);
      for (std::size_t row = 0U; row < rows; ++row) {
        double total = 0.0;
        for (std::size_t source = 0U; source <= row; ++source)
          total += static_cast<double>(signal[source]) *
                   static_cast<double>(kernel[row - source]);
        convolved[row] = static_cast<float>(total);
      }
    } else {
      status = detail::causal_fft_convolution(signal, kernel, &convolved,
                                              &stats);
      if (!status.ok())
        return status;
      accumulate_stats(stats, work);
    }
    const float diagonal = value_at(block.hyena.diagonal, channel);
    for (std::size_t row = 0U; row < rows; ++row) {
      const float filtered =
          round_bf16(convolved[row] + round_bf16(signal[row] * diagonal));
      const float x2 = short_output[row * channels + x2_channel];
      mixed[row * topology.width + channel] = round_bf16(filtered * x2);
    }
  }
  std::vector<float> projected_output;
  status =
      linear_bf16(mixed, rows, topology.width, block.hyena.out_weight,
                  &block.hyena.out_bias, topology.width, &projected_output);
  if (!status.ok())
    return status;
  add_bf16(hidden, projected_output);
  status = rms_norm_bf16(*hidden, rows, topology.width, block.common.post_norm,
                         topology.norm_epsilon, &normalized);
  if (!status.ok())
    return status;
  std::vector<float> feed_forward;
  status = mlp(block.common, topology, normalized, &feed_forward);
  if (!status.ok())
    return status;
  add_bf16(hidden, feed_forward);
  return Status::Ok();
}

void apply_rope(std::vector<float> *const query, std::vector<float> *const key,
                const GenebEvo1Topology &topology) {
  const std::size_t rows = query->size() / topology.width;
  const std::size_t head_width = topology.width / topology.heads;
  const std::size_t half = head_width / 2U;
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t pair = 0U; pair < half; ++pair) {
      const double inverse_frequency =
          std::pow(static_cast<double>(topology.rope_theta),
                   -2.0 * static_cast<double>(pair) /
                       static_cast<double>(head_width));
      const double angle = static_cast<double>(row) /
                           static_cast<double>(topology.rope_scaling_factor) *
                           inverse_frequency;
      const float cosine = round_bf16(static_cast<float>(std::cos(angle)));
      const float sine = round_bf16(static_cast<float>(std::sin(angle)));
      for (std::size_t head = 0U; head < topology.heads; ++head) {
        const std::size_t base = row * topology.width + head * head_width;
        const std::size_t first = base + pair;
        const std::size_t second = first + half;
        const double q0 = static_cast<double>((*query)[first]);
        const double q1 = static_cast<double>((*query)[second]);
        const double k0 = static_cast<double>((*key)[first]);
        const double k1 = static_cast<double>((*key)[second]);
        const double cos = static_cast<double>(cosine);
        const double sin = static_cast<double>(sine);
        (*query)[first] = round_bf16(static_cast<float>(q0 * cos - q1 * sin));
        (*query)[second] = round_bf16(static_cast<float>(q1 * cos + q0 * sin));
        (*key)[first] = round_bf16(static_cast<float>(k0 * cos - k1 * sin));
        (*key)[second] = round_bf16(static_cast<float>(k1 * cos + k0 * sin));
      }
    }
  }
}

Status causal_attention(const std::vector<float> &query,
                        const std::vector<float> &key,
                        const std::vector<float> &value,
                        const GenebEvo1Topology &topology,
                        GenebLongContextStats *const work,
                        std::vector<float> *const output) {
  const std::size_t rows = query.size() / topology.width;
  const std::size_t head_width = topology.width / topology.heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_width));
  output->assign(query.size(), 0.0F);
  std::vector<double> logits;
  for (std::size_t row = 0U; row < rows; ++row) {
    logits.resize(row + 1U);
    work->peak_attention_logits =
        std::max(work->peak_attention_logits, logits.size());
    for (std::size_t head = 0U; head < topology.heads; ++head) {
      double maximum = -std::numeric_limits<double>::infinity();
      const std::size_t query_base = row * topology.width + head * head_width;
      for (std::size_t source = 0U; source <= row; ++source) {
        const std::size_t key_base =
            source * topology.width + head * head_width;
        double score = 0.0;
        for (std::size_t column = 0U; column < head_width; ++column)
          score += static_cast<double>(query[query_base + column]) *
                   static_cast<double>(key[key_base + column]);
        logits[source] = score * scale;
        maximum = std::max(maximum, logits[source]);
      }
      double denominator = 0.0;
      for (std::size_t source = 0U; source <= row; ++source) {
        logits[source] = std::exp(logits[source] - maximum);
        denominator += logits[source];
      }
      for (std::size_t column = 0U; column < head_width; ++column) {
        double total = 0.0;
        for (std::size_t source = 0U; source <= row; ++source) {
          const std::size_t value_index =
              source * topology.width + head * head_width + column;
          total += logits[source] / denominator *
                   static_cast<double>(value[value_index]);
        }
        (*output)[query_base + column] =
            round_bf16(static_cast<float>(total));
      }
    }
  }
  const std::uint64_t pairs = static_cast<std::uint64_t>(rows) *
                              static_cast<std::uint64_t>(rows + 1U) / 2U *
                              static_cast<std::uint64_t>(topology.heads);
  if (pairs >
      std::numeric_limits<std::uint64_t>::max() - work->attention_score_pairs)
    work->attention_score_pairs = std::numeric_limits<std::uint64_t>::max();
  else
    work->attention_score_pairs += pairs;
  return Status::Ok();
}

Status attention_block(const Block &block, const GenebEvo1Topology &topology,
                       std::vector<float> *const hidden,
                       GenebLongContextStats *const work) {
  const std::size_t rows = hidden->size() / topology.width;
  std::vector<float> normalized;
  auto status =
      rms_norm_bf16(*hidden, rows, topology.width, block.common.pre_norm,
                    topology.norm_epsilon, &normalized);
  if (!status.ok())
    return status;
  std::vector<float> qkv;
  status = linear_bf16(normalized, rows, topology.width, block.mha.qkv_weight,
                       &block.mha.qkv_bias, topology.width * 3U, &qkv);
  if (!status.ok())
    return status;
  std::vector<float> query(rows * topology.width);
  std::vector<float> key(rows * topology.width);
  std::vector<float> value(rows * topology.width);
  for (std::size_t row = 0U; row < rows; ++row) {
    const std::size_t source = row * topology.width * 3U;
    const std::size_t target = row * topology.width;
    std::copy_n(qkv.begin() + static_cast<std::ptrdiff_t>(source),
                topology.width,
                query.begin() + static_cast<std::ptrdiff_t>(target));
    std::copy_n(
        qkv.begin() + static_cast<std::ptrdiff_t>(source + topology.width),
        topology.width, key.begin() + static_cast<std::ptrdiff_t>(target));
    std::copy_n(
        qkv.begin() + static_cast<std::ptrdiff_t>(source + topology.width * 2U),
        topology.width, value.begin() + static_cast<std::ptrdiff_t>(target));
  }
  apply_rope(&query, &key, topology);
  std::vector<float> attended;
  status = causal_attention(query, key, value, topology, work, &attended);
  if (!status.ok())
    return status;
  std::vector<float> projected;
  status = linear_bf16(attended, rows, topology.width, block.mha.out_weight,
                       &block.mha.out_bias, topology.width, &projected);
  if (!status.ok())
    return status;
  add_bf16(hidden, projected);
  status = rms_norm_bf16(*hidden, rows, topology.width, block.common.post_norm,
                         topology.norm_epsilon, &normalized);
  if (!status.ok())
    return status;
  std::vector<float> feed_forward;
  status = mlp(block.common, topology, normalized, &feed_forward);
  if (!status.ok())
    return status;
  add_bf16(hidden, feed_forward);
  return Status::Ok();
}

bool wants_capture(const std::set<std::size_t> &captures,
                   const std::size_t layer) {
  return captures.find(layer) != captures.end();
}

void capture(const std::size_t layer, const std::vector<float> &values,
             const std::set<std::size_t> &captures,
             std::vector<GenebEvo1HiddenCapture> *const output) {
  if (wants_capture(captures, layer))
    output->push_back({layer, values});
}

Status model_file_tensors(const ModelFile &artifact,
                          std::vector<GenebEvo1NamedTensorView> *const output) {
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

struct GenebEvo1Model::Impl final {
  GenebEvo1Topology topology;
  Runtime runtime;
};

Status validate_geneb_evo1_topology(const GenebEvo1Topology &topology) {
  if (topology.vocabulary_size == 0U ||
      !token_vocabulary_size_supported(topology.vocabulary_size) ||
      topology.width == 0U || topology.width > kMaximumDimension ||
      topology.layers == 0U || topology.layers > kMaximumLayers ||
      topology.heads == 0U || topology.width % topology.heads != 0U ||
      (topology.width / topology.heads) % 2U != 0U ||
      topology.inner_width == 0U || topology.inner_width > kMaximumDimension ||
      topology.state_width == 0U || topology.state_width > 1024U ||
      topology.short_filter_width == 0U ||
      topology.short_filter_width > 4096U ||
      topology.maximum_sequence_length == 0U ||
      topology.maximum_sequence_length > kMaximumSequenceLength ||
      !std::isfinite(topology.norm_epsilon) || topology.norm_epsilon <= 0.0F ||
      !std::isfinite(topology.rope_theta) || topology.rope_theta <= 1.0F ||
      !std::isfinite(topology.rope_scaling_factor) ||
      topology.rope_scaling_factor <= 0.0F || topology.attention_layers.empty())
    return invalid("topology is outside the bounded runtime contract");
  std::size_t previous = 0U;
  bool first = true;
  for (const auto layer : topology.attention_layers) {
    if (layer >= topology.layers || (!first && layer <= previous))
      return invalid("attention layers must be sorted, unique, and in range");
    previous = layer;
    first = false;
  }
  return Status::Ok();
}

Status geneb_evo1_topology_from_artifact(const ModelFile &artifact,
                                         GenebEvo1Topology *const output) {
  if (output == nullptr)
    return invalid("topology output is null");
  if (artifact.profile() != kGenebEvo1ArtifactProfile)
    return format_error("profile must be '" +
                        std::string{kGenebEvo1ArtifactProfile} + "'");
  for (const auto &literal :
       {std::pair{"runtime.abi", kGenebEvo1RuntimeAbi},
        std::pair{"model.architecture", kGenebEvo1Architecture},
        std::pair{"tokenizer.profile", kTokenizerProfile},
        std::pair{"evo1.weight_dtype", std::string_view{"mixed-bf16-f32"}},
        std::pair{"evo1.norm_denominator",
                  std::string_view{"sqrt-mean-plus-epsilon"}},
        std::pair{"evo1.mlp_activation", std::string_view{"gelu-exact"}},
        std::pair{"evo1.hyena_projection_layout",
                  std::string_view{"per-head-x2-x1-v"}},
        std::pair{"evo1.long_convolution",
                  std::string_view{"exact-2l-modal-fft"}},
        std::pair{"evo1.attention_memory",
                  std::string_view{"streaming-linear"}},
        std::pair{"evo1.hidden_tap", std::string_view{"post-final-rmsnorm"}},
        std::pair{"evo1.pooling", std::string_view{"all-token-mean"}},
        std::pair{"source.revision", kPinnedRevision},
        std::pair{"source.model_py_sha256", kPinnedModelSha},
        std::pair{"source.model_py_path", std::string_view{"model.py"}},
        std::pair{"source.model_py_lines",
                  std::string_view{"29-84,87-318,333-383"}},
        std::pair{"source.engine_py_sha256", kPinnedEngineSha},
        std::pair{"source.engine_py_path", std::string_view{"engine.py"}},
        std::pair{"source.engine_py_lines", std::string_view{"66-226"}},
        std::pair{"source.layers_py_sha256", kPinnedLayersSha},
        std::pair{"source.layers_py_path", std::string_view{"layers.py"}},
        std::pair{"source.layers_py_lines", std::string_view{"17-40,43-86"}},
        std::pair{"source.positional_embeddings_py_sha256", kPinnedPositionSha},
        std::pair{"source.positional_embeddings_py_path",
                  std::string_view{"positional_embeddings.py"}},
        std::pair{"source.positional_embeddings_py_lines",
                  std::string_view{"11-112"}},
        std::pair{"source.tokenizer_py_sha256", kPinnedTokenizerSha},
        std::pair{"source.tokenizer_py_path", std::string_view{"tokenizer.py"}},
        std::pair{"source.tokenizer_config_path",
                  std::string_view{"tokenizer_config.json"}},
        std::pair{"source.geneb.extractor_sha256", kPinnedExtractorSha},
        std::pair{"source.geneb.extractor_path",
                  std::string_view{"embedding_pipeline/extractors/evo.py"}},
        std::pair{"source.geneb.extractor_lines", std::string_view{"29-35"}},
        std::pair{"geneb.provenance.extractor_commit",
                  kPinnedExtractorCommit}}) {
    auto status = require_literal(artifact, literal.first, literal.second);
    if (!status.ok())
      return status;
  }
  for (const auto &flag : {std::pair{"evo1.causal_attention", true},
                           std::pair{"evo1.add_special_tokens", false},
                           std::pair{"source.immutable", true},
                           std::pair{"geneb.source.immutable", true}}) {
    auto status = require_bool(artifact, flag.first, flag.second);
    if (!status.ok())
      return status;
  }
  for (const auto key :
       {"source.config_sha256", "source.checkpoint_manifest_sha256",
        "source.tokenizer_config_sha256"}) {
    auto status = require_sha(artifact, key);
    if (!status.ok())
      return status;
  }
  if (!artifact.tokenizer_asset_descriptor().has_value())
    return format_error("complete tokenizer descriptor is required");

  GenebEvo1Topology topology;
  for (const auto &field :
       {std::pair{"config.vocab_size", &topology.vocabulary_size},
        std::pair{"config.hidden_size", &topology.width},
        std::pair{"config.num_layers", &topology.layers},
        std::pair{"config.max_seqlen", &topology.maximum_sequence_length},
        std::pair{"evo1.num_attention_heads", &topology.heads},
        std::pair{"evo1.inner_width", &topology.inner_width},
        std::pair{"evo1.state_width", &topology.state_width},
        std::pair{"evo1.short_filter_width", &topology.short_filter_width}}) {
    auto status = metadata_size(artifact, field.first, field.second);
    if (!status.ok())
      return status;
  }
  for (const auto &field :
       {std::pair{"evo1.norm_epsilon", &topology.norm_epsilon},
        std::pair{"evo1.rope_theta", &topology.rope_theta},
        std::pair{"evo1.rope_scaling_factor", &topology.rope_scaling_factor}}) {
    double value = 0.0;
    if (!metadata_f64(artifact, field.first, &value) || !std::isfinite(value) ||
        value <= 0.0 || value > std::numeric_limits<float>::max())
      return format_error("metadata '" + std::string{field.first} +
                          "' is invalid");
    *field.second = static_cast<float>(value);
  }
  if (!metadata_u64_list(artifact, "evo1.attention_layers",
                         &topology.attention_layers))
    return format_error("evo1.attention_layers metadata is invalid");
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
  auto status = validate_geneb_evo1_topology(topology);
  if (!status.ok())
    return status;
  *output = std::move(topology);
  return Status::Ok();
}

Status canonical_geneb_evo1_tensors(
    const GenebEvo1Topology &topology,
    std::vector<GenebEvo1TensorRequirement> *const output) {
  if (output == nullptr)
    return invalid("tensor manifest output is null");
  auto status = validate_geneb_evo1_topology(topology);
  if (!status.ok())
    return status;
  output->clear();
  const TensorDType bf16 = TensorDType::kBF16;
  const TensorDType f32 = TensorDType::kF32;
  for (std::size_t layer = 0U; layer < topology.layers; ++layer) {
    const std::string prefix = "backbone.blocks." + std::to_string(layer) + ".";
    if (is_attention_layer(topology, layer)) {
      requirement(output, prefix + "inner_mha_cls.Wqkv.bias", bf16,
                  {topology.width * 3U});
      requirement(output, prefix + "inner_mha_cls.Wqkv.weight", bf16,
                  {topology.width * 3U, topology.width});
      requirement(output, prefix + "inner_mha_cls.out_proj.bias", bf16,
                  {topology.width});
      requirement(output, prefix + "inner_mha_cls.out_proj.weight", bf16,
                  {topology.width, topology.width});
      requirement(output, prefix + "inner_mha_cls.rotary_emb.inv_freq", bf16,
                  {topology.width / topology.heads / 2U});
    } else {
      requirement(output, prefix + "filter.D", bf16, {topology.width});
      requirement(output, prefix + "filter.poles", f32,
                  {topology.width, topology.state_width, 1U, 2U});
      requirement(output, prefix + "filter.residues", f32,
                  {topology.width, topology.state_width, 1U, 2U});
      requirement(output, prefix + "filter.short_filter_bias", bf16,
                  {topology.width * 3U});
      requirement(output, prefix + "filter.short_filter_weight", bf16,
                  {topology.width * 3U, 1U, topology.short_filter_width});
      requirement(output, prefix + "out_filter_dense.bias", bf16,
                  {topology.width});
      requirement(output, prefix + "out_filter_dense.weight", bf16,
                  {topology.width, topology.width});
      requirement(output, prefix + "projections.bias", bf16,
                  {topology.width * 3U});
      requirement(output, prefix + "projections.weight", bf16,
                  {topology.width * 3U, topology.width});
    }
    requirement(output, prefix + "mlp.l1.weight", bf16,
                {topology.inner_width, topology.width});
    requirement(output, prefix + "mlp.l2.weight", bf16,
                {topology.inner_width, topology.width});
    requirement(output, prefix + "mlp.l3.weight", bf16,
                {topology.width, topology.inner_width});
    requirement(output, prefix + "post_norm.scale", bf16, {topology.width});
    requirement(output, prefix + "pre_norm.scale", bf16, {topology.width});
  }
  requirement(output, "backbone.embedding_layer.weight", bf16,
              {topology.vocabulary_size, topology.width});
  requirement(output, "backbone.norm.scale", bf16, {topology.width});
  return Status::Ok();
}

GenebEvo1Model::GenebEvo1Model() = default;
GenebEvo1Model::~GenebEvo1Model() = default;
GenebEvo1Model::GenebEvo1Model(GenebEvo1Model &&) noexcept = default;
GenebEvo1Model &GenebEvo1Model::operator=(GenebEvo1Model &&) noexcept = default;

Status
GenebEvo1Model::load(const GenebEvo1Topology &topology,
                     const std::vector<GenebEvo1NamedTensorView> &tensors) {
  auto status = validate_geneb_evo1_topology(topology);
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

Status GenebEvo1Model::load(const ModelFile &artifact) {
  GenebEvo1Topology topology;
  auto status = geneb_evo1_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebEvo1NamedTensorView> tensors;
  status = model_file_tensors(artifact, &tensors);
  if (!status.ok())
    return status;
  return load(topology, tensors);
}

const GenebEvo1Topology *GenebEvo1Model::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebEvo1Model::kernel_name() const noexcept {
  return impl_ == nullptr ? "unloaded" : "geneb-evo1-mixed-bf16-streaming-fft";
}

Status GenebEvo1Model::forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::size_t> &capture_layers,
                               GenebEvo1ForwardResult *const output) const {
  if (impl_ == nullptr)
    return invalid("model is not loaded");
  if (output == nullptr)
    return invalid("forward output is null");
  const auto &topology = impl_->topology;
  if (tokens.empty() || tokens.size() > topology.maximum_sequence_length)
    return invalid("token length is outside the model context");
  for (const auto token : tokens)
    if (!token_id_in_vocabulary(token, topology.vocabulary_size))
      return invalid("token ID is outside the vocabulary");
  std::set<std::size_t> captures;
  for (const auto layer : capture_layers) {
    if (layer > topology.layers || !captures.insert(layer).second)
      return invalid("capture layers are out of range or duplicated");
  }
  try {
    GenebEvo1ForwardResult result;
    result.rows = tokens.size();
    result.width = topology.width;
    std::vector<float> hidden(tokens.size() * topology.width, 0.0F);
    for (std::size_t row = 0U; row < tokens.size(); ++row)
      for (std::size_t column = 0U; column < topology.width; ++column)
        hidden[row * topology.width + column] = value_at(
            impl_->runtime.embedding,
            static_cast<std::size_t>(tokens[row]) * topology.width + column);
    capture(0U, hidden, captures, &result.captures);
    for (std::size_t layer = 0U; layer < topology.layers; ++layer) {
      auto status = impl_->runtime.blocks[layer].attention
                        ? attention_block(impl_->runtime.blocks[layer],
                                          topology, &hidden, &result.work)
                        : hyena_block(impl_->runtime.blocks[layer], topology,
                                      &hidden, &result.work);
      if (!status.ok())
        return status;
      if (layer + 1U < topology.layers)
        capture(layer + 1U, hidden, captures, &result.captures);
    }
    auto status = rms_norm_bf16(hidden, tokens.size(), topology.width,
                                impl_->runtime.final_norm,
                                topology.norm_epsilon, &result.final_hidden);
    if (!status.ok())
      return status;
    capture(topology.layers, result.final_hidden, captures, &result.captures);
    result.pooled.assign(topology.width, 0.0F);
    std::vector<double> pooled_double(topology.width, 0.0);
    for (std::size_t row = 0U; row < tokens.size(); ++row)
      for (std::size_t column = 0U; column < topology.width; ++column)
        pooled_double[column] +=
            static_cast<double>(result.final_hidden[row * topology.width +
                                                   column]);
    for (std::size_t column = 0U; column < topology.width; ++column)
      result.pooled[column] =
          static_cast<float>(pooled_double[column]) /
          static_cast<float>(tokens.size());
    if (!std::all_of(result.final_hidden.begin(), result.final_hidden.end(),
                     [](const float value) { return std::isfinite(value); }) ||
        !std::all_of(result.pooled.begin(), result.pooled.end(),
                     [](const float value) { return std::isfinite(value); }))
      return invalid("forward produced a non-finite value");
    *output = std::move(result);
  } catch (const std::bad_alloc &) {
    return {ErrorCode::kInternal, "GENEB Evo-1: forward allocation failed"};
  }
  return Status::Ok();
}

} // namespace evo::cpu
