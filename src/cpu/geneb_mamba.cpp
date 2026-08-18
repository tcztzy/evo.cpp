// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_mamba.hpp"

#include "../linear_executor.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace evo::cpu {
namespace {

constexpr std::size_t kMaximumLayers = 1024U;
constexpr std::size_t kMaximumDimension = 1U << 24U;
constexpr std::size_t kMaximumConvolutionWidth = 1024U;

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "GENEB Mamba: " + message};
}

Status invalid(const std::string &message) {
  return {ErrorCode::kInvalidArgument, "GENEB Mamba: " + message};
}

bool checked_multiply(const std::size_t left, const std::size_t right,
                      std::size_t *const output) noexcept {
  if (output == nullptr ||
      (left != 0U && right > std::numeric_limits<std::size_t>::max() / left))
    return false;
  *output = left * right;
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
  output->assign(entry->value.begin(), entry->value.end());
  return Status::Ok();
}

Status metadata_literal(const ModelFile &artifact, const std::string_view key,
                        const std::string_view expected) {
  std::string actual;
  auto status = metadata_string(artifact, key, &actual);
  if (!status.ok())
    return status;
  if (actual != expected)
    return format_error("metadata '" + std::string{key} +
                        "' differs from the frozen literal");
  return Status::Ok();
}

Status metadata_size(const ModelFile &artifact, const std::string_view key,
                     const bool positive, std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(std::uint64_t))
    return format_error("u64 metadata is malformed: " + std::string{key});
  std::uint64_t value = 0U;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if ((positive && value == 0U) ||
      value > std::numeric_limits<std::size_t>::max())
    return format_error("u64 metadata is outside size_t: " + std::string{key});
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
  double value = 0.0;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (!std::isfinite(value) || value <= 0.0 ||
      value > static_cast<double>(std::numeric_limits<float>::max()))
    return format_error("f64 metadata is not positive finite F32: " +
                        std::string{key});
  *output = static_cast<float>(value);
  return Status::Ok();
}

Status metadata_bool(const ModelFile &artifact, const std::string_view key,
                     bool *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kBool, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != 1U)
    return format_error("bool metadata is malformed: " + std::string{key});
  *output = entry->value.front() != 0U;
  return Status::Ok();
}

Status parse_complement_map(const std::string_view text,
                            std::vector<TokenId> *const output) {
  if (output == nullptr)
    return invalid("complement-map output is null");
  std::vector<TokenId> result;
  if (text.empty()) {
    *output = std::move(result);
    return Status::Ok();
  }
  std::size_t begin = 0U;
  while (begin < text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::size_t extent =
        end == std::string_view::npos ? text.size() - begin : end - begin;
    if (extent == 0U)
      return format_error("complement map contains an empty ID");
    std::uint64_t value = 0U;
    const char *const first = text.data() + begin;
    const char *const last = first + extent;
    const auto parsed = std::from_chars(first, last, value);
    if (parsed.ec != std::errc{} || parsed.ptr != last ||
        value > std::numeric_limits<TokenId>::max())
      return format_error("complement map contains an invalid token ID");
    result.push_back(static_cast<TokenId>(value));
    if (end == std::string_view::npos)
      break;
    begin = end + 1U;
  }
  *output = std::move(result);
  return Status::Ok();
}

void add_requirement(std::vector<GenebMambaTensorRequirement> *const output,
                     std::string name, std::vector<std::size_t> shape) {
  output->push_back({std::move(name), TensorDType::kF32, std::move(shape)});
}

std::string layer_prefix(const std::size_t layer) {
  return "layers." + std::to_string(layer) + ".";
}

std::string direction_prefix(const std::string_view direction,
                             const std::size_t layer) {
  return std::string{direction} + ".layers." + std::to_string(layer) + ".";
}

void add_mamba1_direction(
    std::vector<GenebMambaTensorRequirement> *const output,
    const std::string &prefix, const GenebMambaTopology &topology,
    const bool tied_projection) {
  if (!tied_projection) {
    add_requirement(output, prefix + "in_proj.weight",
                    {topology.inner_width * 2U, topology.width});
  }
  add_requirement(output, prefix + "conv1d.weight",
                  {topology.inner_width, 1U, topology.convolution_width});
  add_requirement(output, prefix + "conv1d.bias", {topology.inner_width});
  add_requirement(output, prefix + "x_proj.weight",
                  {topology.time_step_rank + topology.state_width * 2U,
                   topology.inner_width});
  add_requirement(output, prefix + "dt_proj.weight",
                  {topology.inner_width, topology.time_step_rank});
  add_requirement(output, prefix + "dt_proj.bias", {topology.inner_width});
  add_requirement(output, prefix + "A_log",
                  {topology.inner_width, topology.state_width});
  add_requirement(output, prefix + "D", {topology.inner_width});
  if (!tied_projection) {
    add_requirement(output, prefix + "out_proj.weight",
                    {topology.width, topology.inner_width});
  }
}

void add_mamba2_direction(
    std::vector<GenebMambaTensorRequirement> *const output,
    const std::string_view direction, const GenebMambaTopology &topology) {
  const std::size_t grouped_state = topology.groups * topology.state_width;
  for (std::size_t layer = 0U; layer < topology.layers; ++layer) {
    const std::string prefix = direction_prefix(direction, layer);
    add_requirement(output, prefix + "norm.weight", {topology.width});
    add_requirement(
        output, prefix + "mixer.in_proj.weight",
        {topology.inner_width * 2U + grouped_state * 2U + topology.heads,
         topology.width});
    add_requirement(output, prefix + "mixer.conv1d.weight",
                    {topology.inner_width + grouped_state * 2U, 1U,
                     topology.convolution_width});
    add_requirement(output, prefix + "mixer.conv1d.bias",
                    {topology.inner_width + grouped_state * 2U});
    add_requirement(output, prefix + "mixer.dt_bias", {topology.heads});
    add_requirement(output, prefix + "mixer.A_log", {topology.heads});
    add_requirement(output, prefix + "mixer.D", {topology.heads});
    add_requirement(output, prefix + "mixer.norm.weight",
                    {topology.inner_width});
    add_requirement(output, prefix + "mixer.out_proj.weight",
                    {topology.width, topology.inner_width});
    add_requirement(output, prefix + "norm2.weight", {topology.width});
    add_requirement(output, prefix + "mlp.fc1.weight",
                    {topology.mlp_width * 2U, topology.width});
    add_requirement(output, prefix + "mlp.fc2.weight",
                    {topology.width, topology.mlp_width});
  }
  add_requirement(output, std::string{direction} + ".final_norm.weight",
                  {topology.width});
}

Status validate_tensor(const MambaTensorView &tensor,
                       const GenebMambaTensorRequirement &requirement) {
  if (tensor.dtype != requirement.dtype)
    return format_error("tensor dtype differs: " + requirement.name);
  if (tensor.shape != requirement.shape)
    return format_error("tensor shape differs: " + requirement.name);
  std::size_t elements = 1U;
  for (const auto dimension : tensor.shape) {
    if (dimension == 0U || !checked_multiply(elements, dimension, &elements))
      return format_error("tensor shape overflows: " + requirement.name);
  }
  std::size_t bytes = 0U;
  if (!checked_multiply(elements, sizeof(float), &bytes) ||
      tensor.data == nullptr || tensor.bytes != bytes)
    return format_error("tensor payload extent differs: " + requirement.name);
  return Status::Ok();
}

float tensor_value(const MambaTensorView &tensor,
                   const std::size_t index) noexcept {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return value;
}

void reverse_rows(const std::vector<float> &input, const std::size_t rows,
                  const std::size_t width, const bool reverse_channels,
                  std::vector<float> *const output) {
  output->resize(input.size());
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t column = 0U; column < width; ++column) {
      const std::size_t target_column =
          reverse_channels ? width - 1U - column : column;
      (*output)[(rows - 1U - row) * width + target_column] =
          input[row * width + column];
    }
  }
}

Status add_rms_norm(const std::vector<float> &hidden,
                    const std::vector<float> *const previous_residual,
                    const std::size_t rows, const std::size_t width,
                    const MambaTensorView &scale, const float epsilon,
                    std::vector<float> *const normalized,
                    std::vector<float> *const residual) {
  if (normalized == nullptr || residual == nullptr ||
      hidden.size() != rows * width ||
      (previous_residual != nullptr &&
       previous_residual->size() != hidden.size()))
    return invalid("add-RMSNorm dimensions differ");
  residual->resize(hidden.size());
  for (std::size_t index = 0U; index < hidden.size(); ++index) {
    (*residual)[index] =
        hidden[index] +
        (previous_residual == nullptr ? 0.0F : (*previous_residual)[index]);
  }
  return mamba_rms_norm(*residual, rows, width, scale, epsilon, normalized);
}

Status gated_mlp(const std::vector<float> &input, const std::size_t rows,
                 const std::size_t width, const std::size_t mlp_width,
                 const MambaTensorView &first_weight,
                 const MambaTensorView &second_weight,
                 evo::detail::LinearExecutor *const executor,
                 std::vector<float> *const output) {
  std::vector<float> projected;
  auto status = mamba_linear_f32(input, rows, width, first_weight,
                                 mlp_width * 2U, executor, &projected);
  if (!status.ok())
    return status;
  std::vector<float> activated(rows * mlp_width, 0.0F);
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t column = 0U; column < mlp_width; ++column) {
      const float gate = projected[row * mlp_width * 2U + mlp_width + column];
      const float exponential = gate >= 0.0F ? std::exp(-gate) : std::exp(gate);
      const float silu_gate = gate >= 0.0F
                                  ? gate / (1.0F + exponential)
                                  : gate * exponential / (1.0F + exponential);
      activated[row * mlp_width + column] =
          projected[row * mlp_width * 2U + column] * silu_gate;
    }
  }
  return mamba_linear_f32(activated, rows, mlp_width, second_weight, width,
                          executor, output);
}

bool wants_capture(const std::set<std::size_t> &captures,
                   const std::size_t layer) {
  return captures.find(layer) != captures.end();
}

} // namespace

Status validate_geneb_mamba_topology(const GenebMambaTopology &topology) {
  if (topology.vocabulary_size == 0U ||
      !token_vocabulary_size_supported(topology.vocabulary_size) ||
      topology.tokenizer_vocabulary_size == 0U ||
      topology.tokenizer_vocabulary_size > topology.vocabulary_size ||
      topology.width == 0U || topology.width > kMaximumDimension ||
      topology.output_width == 0U ||
      topology.output_width > kMaximumDimension || topology.layers == 0U ||
      topology.layers > kMaximumLayers || topology.inner_width == 0U ||
      topology.inner_width > kMaximumDimension || topology.state_width == 0U ||
      topology.state_width > kMaximumDimension ||
      topology.convolution_width == 0U ||
      topology.convolution_width > kMaximumConvolutionWidth ||
      !std::isfinite(topology.norm_epsilon) || topology.norm_epsilon <= 0.0F)
    return invalid("topology is outside the bounded runtime contract");

  if (topology.variant == GenebMambaVariant::kCaduceusMamba1) {
    if (topology.inner_width != topology.width * 2U ||
        topology.time_step_rank != (topology.width + 15U) / 16U ||
        topology.mlp_width != 0U || topology.head_width != 0U ||
        topology.heads != 0U || topology.groups != 0U ||
        topology.output_width !=
            topology.width *
                (topology.reverse_complement_parameter_sharing ? 2U : 1U))
      return invalid("Caduceus geometry differs from Mamba1");
    if (topology.reverse_complement_parameter_sharing) {
      if (topology.complement_map.size() != topology.vocabulary_size)
        return invalid("RCPS complement map size differs from vocabulary");
      for (std::size_t index = 0U; index < topology.complement_map.size();
           ++index) {
        const auto complement = topology.complement_map[index];
        if (!token_id_in_vocabulary(complement, topology.vocabulary_size) ||
            topology.complement_map[complement] != index)
          return invalid("RCPS complement map is not an involution");
      }
    } else if (!topology.complement_map.empty()) {
      return invalid("non-RCPS topology must not carry a complement map");
    }
  } else {
    if (topology.reverse_complement_parameter_sharing ||
        !topology.complement_map.empty() ||
        topology.output_width != topology.width ||
        topology.time_step_rank != 0U || topology.mlp_width == 0U ||
        topology.head_width == 0U || topology.heads == 0U ||
        topology.groups == 0U ||
        topology.inner_width != topology.head_width * topology.heads ||
        topology.heads % topology.groups != 0U)
      return invalid("eccDNAMamba geometry differs from Mamba2");
  }
  return Status::Ok();
}

Status geneb_mamba_topology_from_artifact(const ModelFile &artifact,
                                          GenebMambaTopology *const output) {
  if (output == nullptr)
    return invalid("topology output is null");
  if (artifact.profile() != kGenebMambaArtifactProfile)
    return format_error("artifact profile differs");
  auto status =
      metadata_literal(artifact, "runtime.abi", kGenebMambaRuntimeAbi);
  if (!status.ok())
    return status;
  status =
      metadata_literal(artifact, "model.architecture", kGenebMambaArchitecture);
  if (!status.ok())
    return status;
  if (!artifact.tokenizer_asset_descriptor().has_value())
    return format_error("artifact tokenizer descriptor is missing");

  const std::set<std::string_view> allowed{"mamba.variant",
                                           "mamba.vocab_size",
                                           "mamba.width",
                                           "mamba.output_width",
                                           "mamba.layers",
                                           "mamba.max_sequence_length",
                                           "mamba.advertised_training_length",
                                           "mamba.inner_width",
                                           "mamba.state_width",
                                           "mamba.conv_width",
                                           "mamba.time_step_rank",
                                           "mamba.mlp_width",
                                           "mamba.head_width",
                                           "mamba.heads",
                                           "mamba.groups",
                                           "mamba.norm_epsilon",
                                           "mamba.rcps",
                                           "mamba.complement_map",
                                           "mamba.hidden_tap",
                                           "mamba.pooling",
                                           "mamba.mask_domain",
                                           "mamba.special_tokens"};
  for (const auto &entry : artifact.metadata()) {
    if (entry.key.compare(0U, 6U, "mamba.") == 0U &&
        allowed.find(entry.key) == allowed.end())
      return format_error("unexpected topology metadata: " + entry.key);
  }

  GenebMambaTopology topology;
  std::string variant;
  status = metadata_string(artifact, "mamba.variant", &variant);
  if (!status.ok())
    return status;
  if (variant == "caduceus-mamba1")
    topology.variant = GenebMambaVariant::kCaduceusMamba1;
  else if (variant == "eccdna-mamba2")
    topology.variant = GenebMambaVariant::kEccDnaMamba2;
  else
    return format_error("mamba.variant is unsupported");
  status = metadata_size(artifact, "runtime.tokenizer_vocabulary_size", true,
                         &topology.tokenizer_vocabulary_size);
  if (!status.ok())
    return status;
  for (const auto &field :
       {std::pair{"mamba.vocab_size", &topology.vocabulary_size},
        std::pair{"mamba.width", &topology.width},
        std::pair{"mamba.output_width", &topology.output_width},
        std::pair{"mamba.layers", &topology.layers},
        std::pair{"mamba.inner_width", &topology.inner_width},
        std::pair{"mamba.state_width", &topology.state_width},
        std::pair{"mamba.conv_width", &topology.convolution_width}}) {
    status = metadata_size(artifact, field.first, true, field.second);
    if (!status.ok())
      return status;
  }
  status = metadata_size(artifact, "mamba.max_sequence_length", false,
                         &topology.maximum_sequence_length);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "mamba.advertised_training_length", false,
                         &topology.advertised_training_sequence_length);
  if (!status.ok())
    return status;
  for (const auto &field :
       {std::pair{"mamba.time_step_rank", &topology.time_step_rank},
        std::pair{"mamba.mlp_width", &topology.mlp_width},
        std::pair{"mamba.head_width", &topology.head_width},
        std::pair{"mamba.heads", &topology.heads},
        std::pair{"mamba.groups", &topology.groups}}) {
    status = metadata_size(artifact, field.first, false, field.second);
    if (!status.ok())
      return status;
  }
  status =
      metadata_float(artifact, "mamba.norm_epsilon", &topology.norm_epsilon);
  if (!status.ok())
    return status;
  status = metadata_bool(artifact, "mamba.rcps",
                         &topology.reverse_complement_parameter_sharing);
  if (!status.ok())
    return status;
  std::string complement;
  status = metadata_string(artifact, "mamba.complement_map", &complement);
  if (!status.ok())
    return status;
  status = parse_complement_map(complement, &topology.complement_map);
  if (!status.ok())
    return status;
  for (const auto &literal :
       {std::pair{"mamba.hidden_tap", "post-final-norm"},
        std::pair{"mamba.pooling", "attention-mask-mean"},
        std::pair{"mamba.mask_domain", "attention-mask"}}) {
    status = metadata_literal(artifact, literal.first, literal.second);
    if (!status.ok())
      return status;
  }
  status = metadata_literal(artifact, "mamba.special_tokens",
                            topology.variant == GenebMambaVariant::kEccDnaMamba2
                                ? "include"
                                : "none");
  if (!status.ok())
    return status;
  status = validate_geneb_mamba_topology(topology);
  if (!status.ok())
    return status;
  *output = std::move(topology);
  return Status::Ok();
}

Status canonical_geneb_mamba_tensors(
    const GenebMambaTopology &topology,
    std::vector<GenebMambaTensorRequirement> *const output) {
  if (output == nullptr)
    return invalid("tensor-requirement output is null");
  auto status = validate_geneb_mamba_topology(topology);
  if (!status.ok())
    return status;
  std::vector<GenebMambaTensorRequirement> result;
  if (topology.variant == GenebMambaVariant::kCaduceusMamba1) {
    add_requirement(&result, "embedding.weight",
                    {topology.vocabulary_size, topology.width});
    for (std::size_t layer = 0U; layer < topology.layers; ++layer) {
      const std::string prefix = layer_prefix(layer);
      add_requirement(&result, prefix + "norm.weight", {topology.width});
      add_mamba1_direction(&result, prefix + "forward.", topology, false);
      add_mamba1_direction(&result, prefix + "reverse.", topology, true);
    }
    add_requirement(&result, "final_norm.weight", {topology.width});
  } else {
    add_requirement(&result, "token_embedding.weight",
                    {topology.vocabulary_size, topology.width});
    add_mamba2_direction(&result, "forward", topology);
    add_mamba2_direction(&result, "reverse", topology);
    add_requirement(&result, "projection.weight",
                    {topology.width, topology.width * 2U});
  }
  *output = std::move(result);
  return Status::Ok();
}

Status geneb_mamba_pool(const GenebMambaForwardResult &forward,
                        const std::vector<std::uint8_t> &attention_mask,
                        std::vector<float> *const output) {
  if (output == nullptr)
    return invalid("pool output is null");
  if (forward.rows == 0U || forward.width == 0U ||
      forward.final_hidden.size() != forward.rows * forward.width ||
      attention_mask.size() != forward.rows)
    return invalid("pool dimensions differ");
  output->assign(forward.width, 0.0F);
  std::size_t count = 0U;
  for (std::size_t row = 0U; row < forward.rows; ++row) {
    if (attention_mask[row] > 1U)
      return invalid("attention mask must contain only zero/one");
    if (attention_mask[row] == 0U)
      continue;
    ++count;
    for (std::size_t column = 0U; column < forward.width; ++column)
      (*output)[column] += forward.final_hidden[row * forward.width + column];
  }
  if (count == 0U)
    return invalid("attention mask has no effective token");
  for (float &value : *output)
    value /= static_cast<float>(count);
  return Status::Ok();
}

struct GenebMambaModel::Impl final {
  struct CaduceusLayer final {
    MambaTensorView norm;
    Mamba1Weights forward;
    Mamba1Weights reverse;
  };
  struct EccLayer final {
    MambaTensorView norm;
    Mamba2Weights mixer;
    MambaTensorView norm2;
    MambaTensorView mlp_first;
    MambaTensorView mlp_second;
  };
  struct EccDirection final {
    std::vector<EccLayer> layers;
    MambaTensorView final_norm;
  };

  GenebMambaTopology topology;
  MambaTensorView embedding;
  std::vector<CaduceusLayer> caduceus_layers;
  MambaTensorView final_norm;
  EccDirection ecc_forward;
  EccDirection ecc_reverse;
  MambaTensorView projection;
  std::shared_ptr<evo::detail::LinearExecutor> linear_executor;

  Status bidirectional_mamba1(const std::vector<float> &input,
                              const CaduceusLayer &layer,
                              std::vector<float> *const output) const {
    const Mamba1Config config{topology.width,
                              topology.inner_width,
                              topology.state_width,
                              topology.convolution_width,
                              topology.time_step_rank,
                              false,
                              1.0e-6F};
    std::vector<float> forward;
    auto status =
        mamba1_mixer_f32(input, input.size() / topology.width, config,
                         layer.forward, linear_executor.get(), &forward);
    if (!status.ok())
      return status;
    std::vector<float> reversed_input;
    reverse_rows(input, input.size() / topology.width, topology.width, false,
                 &reversed_input);
    std::vector<float> reverse;
    status =
        mamba1_mixer_f32(reversed_input, input.size() / topology.width, config,
                         layer.reverse, linear_executor.get(), &reverse);
    if (!status.ok())
      return status;
    std::vector<float> aligned;
    reverse_rows(reverse, input.size() / topology.width, topology.width, false,
                 &aligned);
    output->resize(forward.size());
    for (std::size_t index = 0U; index < forward.size(); ++index)
      (*output)[index] = forward[index] + aligned[index];
    return Status::Ok();
  }

  Status run_caduceus(const std::vector<TokenId> &tokens,
                      const std::set<std::size_t> &captures,
                      GenebMambaForwardResult *const output) const {
    const std::size_t rows = tokens.size();
    const std::size_t width = topology.width;
    const std::size_t hidden_width = topology.output_width;
    std::vector<float> hidden(rows * hidden_width, 0.0F);
    for (std::size_t row = 0U; row < rows; ++row) {
      const std::size_t token = tokens[row];
      for (std::size_t column = 0U; column < width; ++column)
        hidden[row * hidden_width + column] =
            tensor_value(embedding, token * width + column);
      if (topology.reverse_complement_parameter_sharing) {
        const std::size_t complement = topology.complement_map[token];
        for (std::size_t column = 0U; column < width; ++column) {
          hidden[row * hidden_width + width + column] = tensor_value(
              embedding, complement * width + (width - 1U - column));
        }
      }
    }
    if (wants_capture(captures, 0U))
      output->captures.push_back({0U, hidden});

    std::vector<float> residual;
    for (std::size_t layer_index = 0U; layer_index < topology.layers;
         ++layer_index) {
      const auto &layer = caduceus_layers[layer_index];
      std::vector<float> normalized;
      std::vector<float> next_residual;
      if (!topology.reverse_complement_parameter_sharing) {
        const std::vector<float> *const previous =
            residual.empty() ? nullptr : &residual;
        auto status =
            add_rms_norm(hidden, previous, rows, width, layer.norm,
                         topology.norm_epsilon, &normalized, &next_residual);
        if (!status.ok())
          return status;
        status = bidirectional_mamba1(normalized, layer, &hidden);
        if (!status.ok())
          return status;
      } else {
        std::vector<float> first(rows * width);
        std::vector<float> second(rows * width);
        std::vector<float> first_residual;
        std::vector<float> second_residual;
        for (std::size_t row = 0U; row < rows; ++row) {
          std::copy_n(
              hidden.begin() + static_cast<std::ptrdiff_t>(row * hidden_width),
              width, first.begin() + static_cast<std::ptrdiff_t>(row * width));
          std::copy_n(
              hidden.begin() +
                  static_cast<std::ptrdiff_t>(row * hidden_width + width),
              width, second.begin() + static_cast<std::ptrdiff_t>(row * width));
        }
        if (!residual.empty()) {
          first_residual.resize(rows * width);
          second_residual.resize(rows * width);
          for (std::size_t row = 0U; row < rows; ++row) {
            std::copy_n(residual.begin() +
                            static_cast<std::ptrdiff_t>(row * hidden_width),
                        width,
                        first_residual.begin() +
                            static_cast<std::ptrdiff_t>(row * width));
            std::copy_n(residual.begin() + static_cast<std::ptrdiff_t>(
                                               row * hidden_width + width),
                        width,
                        second_residual.begin() +
                            static_cast<std::ptrdiff_t>(row * width));
          }
        }
        std::vector<float> normalized_first;
        std::vector<float> residual_first;
        auto status =
            add_rms_norm(second, residual.empty() ? nullptr : &second_residual,
                         rows, width, layer.norm, topology.norm_epsilon,
                         &normalized_first, &residual_first);
        if (!status.ok())
          return status;
        std::vector<float> reversed_first;
        reverse_rows(first, rows, width, true, &reversed_first);
        std::vector<float> reversed_first_residual;
        if (!residual.empty()) {
          reverse_rows(first_residual, rows, width, true,
                       &reversed_first_residual);
        }
        std::vector<float> normalized_second_reversed;
        std::vector<float> residual_second_reversed;
        status = add_rms_norm(
            reversed_first,
            residual.empty() ? nullptr : &reversed_first_residual, rows, width,
            layer.norm, topology.norm_epsilon, &normalized_second_reversed,
            &residual_second_reversed);
        if (!status.ok())
          return status;
        std::vector<float> normalized_second;
        std::vector<float> residual_second;
        reverse_rows(normalized_second_reversed, rows, width, true,
                     &normalized_second);
        reverse_rows(residual_second_reversed, rows, width, true,
                     &residual_second);
        std::vector<float> mixed_first;
        status = bidirectional_mamba1(normalized_first, layer, &mixed_first);
        if (!status.ok())
          return status;
        std::vector<float> wrapper_input;
        reverse_rows(normalized_second, rows, width, true, &wrapper_input);
        std::vector<float> wrapper_output;
        status = bidirectional_mamba1(wrapper_input, layer, &wrapper_output);
        if (!status.ok())
          return status;
        std::vector<float> mixed_second;
        reverse_rows(wrapper_output, rows, width, true, &mixed_second);
        hidden.resize(rows * hidden_width);
        next_residual.resize(rows * hidden_width);
        for (std::size_t row = 0U; row < rows; ++row) {
          for (std::size_t column = 0U; column < width; ++column) {
            hidden[row * hidden_width + column] =
                mixed_first[row * width + column];
            hidden[row * hidden_width + width + column] =
                mixed_second[row * width + column];
            next_residual[row * hidden_width + column] =
                residual_first[row * width + column];
            next_residual[row * hidden_width + width + column] =
                residual_second[row * width + column];
          }
        }
      }
      residual = std::move(next_residual);
      const std::size_t public_layer = layer_index + 1U;
      if (public_layer < topology.layers &&
          wants_capture(captures, public_layer))
        output->captures.push_back({public_layer, hidden});
    }

    std::vector<float> final_hidden;
    if (!topology.reverse_complement_parameter_sharing) {
      auto status =
          add_rms_norm(hidden, &residual, rows, width, final_norm,
                       topology.norm_epsilon, &final_hidden, &residual);
      if (!status.ok())
        return status;
    } else {
      final_hidden.resize(rows * hidden_width);
      for (std::size_t half = 0U; half < 2U; ++half) {
        std::vector<float> hidden_half(rows * width);
        std::vector<float> residual_half(rows * width);
        for (std::size_t row = 0U; row < rows; ++row) {
          for (std::size_t column = 0U; column < width; ++column) {
            hidden_half[row * width + column] =
                hidden[row * hidden_width + half * width + column];
            residual_half[row * width + column] =
                residual[row * hidden_width + half * width + column];
          }
        }
        const bool reverse = half == 1U;
        if (reverse) {
          std::vector<float> transformed;
          reverse_rows(hidden_half, rows, width, true, &transformed);
          hidden_half = std::move(transformed);
          reverse_rows(residual_half, rows, width, true, &transformed);
          residual_half = std::move(transformed);
        }
        std::vector<float> normalized;
        std::vector<float> summed;
        auto status =
            add_rms_norm(hidden_half, &residual_half, rows, width, final_norm,
                         topology.norm_epsilon, &normalized, &summed);
        if (!status.ok())
          return status;
        if (reverse) {
          std::vector<float> transformed;
          reverse_rows(normalized, rows, width, true, &transformed);
          normalized = std::move(transformed);
        }
        for (std::size_t row = 0U; row < rows; ++row) {
          for (std::size_t column = 0U; column < width; ++column) {
            final_hidden[row * hidden_width + half * width + column] =
                normalized[row * width + column];
          }
        }
      }
    }
    output->final_hidden = std::move(final_hidden);
    if (wants_capture(captures, topology.layers))
      output->captures.push_back({topology.layers, output->final_hidden});
    return Status::Ok();
  }

  Status run_ecc_direction(const std::vector<float> &embedding_input,
                           const std::vector<std::uint8_t> &attention_mask,
                           const EccDirection &direction,
                           std::vector<float> *const output) const {
    const std::size_t rows = attention_mask.size();
    std::vector<float> hidden = embedding_input;
    std::vector<float> residual;
    const Mamba2Config mixer_config{topology.width,
                                    topology.inner_width,
                                    topology.state_width,
                                    topology.convolution_width,
                                    topology.head_width,
                                    topology.heads,
                                    topology.groups,
                                    false,
                                    true,
                                    false,
                                    1.0e-5F};
    for (const auto &layer : direction.layers) {
      std::vector<float> normalized;
      std::vector<float> next_residual;
      auto status = add_rms_norm(
          hidden, residual.empty() ? nullptr : &residual, rows, topology.width,
          layer.norm, topology.norm_epsilon, &normalized, &next_residual);
      if (!status.ok())
        return status;
      status = mamba2_mixer_f32(normalized, rows, mixer_config, layer.mixer,
                                linear_executor.get(), &hidden);
      if (!status.ok())
        return status;
      residual = std::move(next_residual);
      status =
          add_rms_norm(hidden, &residual, rows, topology.width, layer.norm2,
                       topology.norm_epsilon, &normalized, &next_residual);
      if (!status.ok())
        return status;
      residual = std::move(next_residual);
      status = gated_mlp(normalized, rows, topology.width, topology.mlp_width,
                         layer.mlp_first, layer.mlp_second,
                         linear_executor.get(), &hidden);
      if (!status.ok())
        return status;
      for (std::size_t row = 0U; row < rows; ++row) {
        const float mask = static_cast<float>(attention_mask[row]);
        for (std::size_t column = 0U; column < topology.width; ++column) {
          hidden[row * topology.width + column] *= mask;
          residual[row * topology.width + column] *= mask;
        }
      }
    }
    std::vector<float> ignored;
    return add_rms_norm(hidden, &residual, rows, topology.width,
                        direction.final_norm, topology.norm_epsilon, output,
                        &ignored);
  }

  Status run_ecc(const std::vector<TokenId> &tokens,
                 const std::vector<std::uint8_t> &attention_mask,
                 const std::set<std::size_t> &captures,
                 GenebMambaForwardResult *const output) const {
    for (const auto layer : captures) {
      if (layer != 0U && layer != topology.layers)
        return {ErrorCode::kUnsupported,
                "eccDNAMamba exposes only token embedding and projected final "
                "hidden"};
    }
    const std::size_t rows = tokens.size();
    std::vector<float> embeddings(rows * topology.width);
    for (std::size_t row = 0U; row < rows; ++row) {
      for (std::size_t column = 0U; column < topology.width; ++column) {
        embeddings[row * topology.width + column] =
            tensor_value(embedding, tokens[row] * topology.width + column);
      }
    }
    if (wants_capture(captures, 0U))
      output->captures.push_back({0U, embeddings});
    std::vector<float> forward;
    auto status =
        run_ecc_direction(embeddings, attention_mask, ecc_forward, &forward);
    if (!status.ok())
      return status;
    std::vector<float> reverse_input;
    reverse_rows(embeddings, rows, topology.width, false, &reverse_input);
    std::vector<std::uint8_t> reverse_mask(attention_mask.rbegin(),
                                           attention_mask.rend());
    std::vector<float> reverse;
    status =
        run_ecc_direction(reverse_input, reverse_mask, ecc_reverse, &reverse);
    if (!status.ok())
      return status;
    std::vector<float> aligned_reverse;
    reverse_rows(reverse, rows, topology.width, false, &aligned_reverse);
    std::vector<float> combined(rows * topology.width * 2U);
    for (std::size_t row = 0U; row < rows; ++row) {
      for (std::size_t column = 0U; column < topology.width; ++column) {
        combined[row * topology.width * 2U + column] =
            forward[row * topology.width + column];
        combined[row * topology.width * 2U + topology.width + column] =
            aligned_reverse[row * topology.width + column];
      }
    }
    status = mamba_linear_f32(combined, rows, topology.width * 2U, projection,
                              topology.width, linear_executor.get(),
                              &output->final_hidden);
    if (!status.ok())
      return status;
    if (wants_capture(captures, topology.layers))
      output->captures.push_back({topology.layers, output->final_hidden});
    return Status::Ok();
  }
};

GenebMambaModel::GenebMambaModel() = default;
GenebMambaModel::~GenebMambaModel() = default;
GenebMambaModel::GenebMambaModel(GenebMambaModel &&) noexcept = default;
GenebMambaModel &
GenebMambaModel::operator=(GenebMambaModel &&) noexcept = default;

Status GenebMambaModel::load(
    const GenebMambaTopology &topology,
    const std::vector<GenebMambaNamedTensorView> &tensors,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  std::vector<GenebMambaTensorRequirement> requirements;
  auto status = canonical_geneb_mamba_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  std::map<std::string, const MambaTensorView *, std::less<>> provided;
  for (const auto &named : tensors) {
    if (!provided.emplace(named.name, &named.tensor).second)
      return format_error("tensor is duplicated: " + named.name);
  }
  if (provided.size() != requirements.size())
    return format_error("tensor set has missing or extra entries");
  for (const auto &requirement : requirements) {
    const auto found = provided.find(requirement.name);
    if (found == provided.end())
      return format_error("tensor is missing: " + requirement.name);
    status = validate_tensor(*found->second, requirement);
    if (!status.ok())
      return status;
  }
  const auto view = [&](const std::string &name) {
    return *provided.find(name)->second;
  };
  auto implementation = std::make_unique<Impl>();
  implementation->topology = topology;
  implementation->linear_executor = std::move(linear_executor);
  if (topology.variant == GenebMambaVariant::kCaduceusMamba1) {
    implementation->embedding = view("embedding.weight");
    implementation->caduceus_layers.resize(topology.layers);
    for (std::size_t index = 0U; index < topology.layers; ++index) {
      auto &layer = implementation->caduceus_layers[index];
      const std::string prefix = layer_prefix(index);
      layer.norm = view(prefix + "norm.weight");
      const auto direction = [&](const std::string &direction_prefix,
                                 Mamba1Weights *const weights,
                                 const Mamba1Weights *const tied) {
        weights->input_projection =
            tied == nullptr ? view(direction_prefix + "in_proj.weight")
                            : tied->input_projection;
        weights->convolution_weight = view(direction_prefix + "conv1d.weight");
        weights->convolution_bias = view(direction_prefix + "conv1d.bias");
        weights->x_projection = view(direction_prefix + "x_proj.weight");
        weights->time_step_projection =
            view(direction_prefix + "dt_proj.weight");
        weights->time_step_bias = view(direction_prefix + "dt_proj.bias");
        weights->a_log = view(direction_prefix + "A_log");
        weights->skip = view(direction_prefix + "D");
        weights->output_projection =
            tied == nullptr ? view(direction_prefix + "out_proj.weight")
                            : tied->output_projection;
      };
      direction(prefix + "forward.", &layer.forward, nullptr);
      direction(prefix + "reverse.", &layer.reverse, &layer.forward);
    }
    implementation->final_norm = view("final_norm.weight");
  } else {
    implementation->embedding = view("token_embedding.weight");
    const auto load_direction = [&](const std::string_view name,
                                    Impl::EccDirection *const direction) {
      direction->layers.resize(topology.layers);
      for (std::size_t index = 0U; index < topology.layers; ++index) {
        auto &layer = direction->layers[index];
        const std::string prefix = direction_prefix(name, index);
        layer.norm = view(prefix + "norm.weight");
        layer.mixer.input_projection = view(prefix + "mixer.in_proj.weight");
        layer.mixer.convolution_weight = view(prefix + "mixer.conv1d.weight");
        layer.mixer.convolution_bias = view(prefix + "mixer.conv1d.bias");
        layer.mixer.time_step_bias = view(prefix + "mixer.dt_bias");
        layer.mixer.a_log = view(prefix + "mixer.A_log");
        layer.mixer.skip = view(prefix + "mixer.D");
        layer.mixer.norm_scale = view(prefix + "mixer.norm.weight");
        layer.mixer.output_projection = view(prefix + "mixer.out_proj.weight");
        layer.norm2 = view(prefix + "norm2.weight");
        layer.mlp_first = view(prefix + "mlp.fc1.weight");
        layer.mlp_second = view(prefix + "mlp.fc2.weight");
      }
      direction->final_norm = view(std::string{name} + ".final_norm.weight");
    };
    load_direction("forward", &implementation->ecc_forward);
    load_direction("reverse", &implementation->ecc_reverse);
    implementation->projection = view("projection.weight");
  }
  impl_ = std::move(implementation);
  return Status::Ok();
}

Status GenebMambaModel::load(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  GenebMambaTopology topology;
  auto status = geneb_mamba_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebMambaNamedTensorView> views;
  views.reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t index = 0U; index < tensor.rank; ++index) {
      if (tensor.dimensions[index] > std::numeric_limits<std::size_t>::max())
        return format_error("tensor dimension exceeds size_t: " + tensor.name);
      shape.push_back(static_cast<std::size_t>(tensor.dimensions[index]));
    }
    if (tensor.data_size > std::numeric_limits<std::size_t>::max())
      return format_error("tensor byte extent exceeds size_t: " + tensor.name);
    const auto *const data = artifact.tensor_data(tensor);
    if (data == nullptr)
      return format_error("tensor payload is unavailable: " + tensor.name);
    views.push_back({tensor.name,
                     {data, static_cast<std::size_t>(tensor.data_size),
                      tensor.dtype, std::move(shape)}});
  }
  return load(topology, views, std::move(linear_executor));
}

Status GenebMambaModel::load_artifact(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  return load(artifact, std::move(linear_executor));
}

const GenebMambaTopology *GenebMambaModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebMambaModel::linear_executor_name() const noexcept {
  if (impl_ == nullptr || impl_->linear_executor == nullptr)
    return "scalar-reference";
  return impl_->linear_executor->name();
}

Status GenebMambaModel::forward(const std::vector<TokenId> &tokens,
                                const std::vector<std::uint8_t> &attention_mask,
                                const std::vector<std::size_t> &capture_layers,
                                GenebMambaForwardResult *const output) const {
  if (impl_ == nullptr)
    return invalid("model is not loaded");
  if (output == nullptr)
    return invalid("forward output is null");
  if (tokens.empty() || attention_mask.size() != tokens.size())
    return invalid("tokens/attention mask dimensions differ");
  if (impl_->topology.maximum_sequence_length != 0U &&
      tokens.size() > impl_->topology.maximum_sequence_length)
    return invalid("token count exceeds the declared checkpoint context");
  for (std::size_t index = 0U; index < tokens.size(); ++index) {
    if (!token_id_in_vocabulary(tokens[index], impl_->topology.vocabulary_size))
      return invalid("token ID exceeds vocabulary");
    if (attention_mask[index] > 1U)
      return invalid("attention mask must contain only zero/one");
  }
  if (std::none_of(attention_mask.begin(), attention_mask.end(),
                   [](const std::uint8_t value) { return value != 0U; }))
    return invalid("attention mask has no effective token");
  std::set<std::size_t> captures;
  for (const auto layer : capture_layers) {
    if (layer > impl_->topology.layers || !captures.insert(layer).second)
      return invalid("capture layer is out of range or duplicated");
  }
  GenebMambaForwardResult result;
  result.rows = tokens.size();
  result.width = impl_->topology.output_width;
  auto status = impl_->topology.variant == GenebMambaVariant::kCaduceusMamba1
                    ? impl_->run_caduceus(tokens, captures, &result)
                    : impl_->run_ecc(tokens, attention_mask, captures, &result);
  if (!status.ok())
    return status;
  *output = std::move(result);
  return Status::Ok();
}

Status GenebMambaModel::pool(const GenebMambaForwardResult &forward,
                             const std::vector<std::uint8_t> &attention_mask,
                             std::vector<float> *const output) const {
  if (impl_ == nullptr)
    return invalid("model is not loaded");
  return geneb_mamba_pool(forward, attention_mask, output);
}

} // namespace evo::cpu
