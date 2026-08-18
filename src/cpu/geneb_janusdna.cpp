// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_janusdna.hpp"

#include "../linear_executor.hpp"

#include <algorithm>
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

constexpr std::size_t kProductionVocabulary = 16U;
constexpr std::size_t kTokenizerVocabulary = 12U;
constexpr std::size_t kProductionWidth = 72U;
constexpr std::size_t kProductionLayers = 8U;
constexpr std::size_t kProductionHeads = 4U;
constexpr std::size_t kProductionHeadDimension = 18U;
constexpr std::size_t kProductionFlexHeadDimension = 32U;
constexpr std::size_t kProductionInnerWidth = 144U;
constexpr std::size_t kProductionStateWidth = 16U;
constexpr std::size_t kProductionConvolutionWidth = 4U;
constexpr std::size_t kProductionTimeStepRank = 5U;
constexpr std::size_t kProductionMlpWidth = 288U;
constexpr std::size_t kProductionExperts = 16U;
constexpr std::size_t kProductionMaximumSequenceLength = 1024U;
constexpr std::size_t kProductionMiddleAttentionLayer = 4U;
constexpr std::size_t kPadToken = 4U;

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "GENEB JanusDNA: " + message};
}

Status invalid(const std::string &message) {
  return {ErrorCode::kInvalidArgument, "GENEB JanusDNA: " + message};
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

Status require_literal(const ModelFile &artifact, const std::string_view key,
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
                     std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(std::uint64_t))
    return format_error("u64 metadata is malformed: " + std::string{key});
  std::uint64_t value = 0U;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (value == 0U || value > std::numeric_limits<std::size_t>::max())
    return format_error("u64 metadata is outside positive size_t: " +
                        std::string{key});
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

void add_requirement(std::vector<GenebJanusDnaTensorRequirement> *const output,
                     std::string name, std::vector<std::size_t> shape) {
  output->push_back({std::move(name), TensorDType::kF32, std::move(shape)});
}

std::string layer_prefix(const std::size_t layer) {
  return "layers." + std::to_string(layer) + ".";
}

void add_mlp(std::vector<GenebJanusDnaTensorRequirement> *const output,
             const std::string &prefix, const GenebJanusDnaTopology &topology) {
  add_requirement(output, prefix + "gate_proj.weight",
                  {topology.mlp_width, topology.width});
  add_requirement(output, prefix + "up_proj.weight",
                  {topology.mlp_width, topology.width});
  add_requirement(output, prefix + "down_proj.weight",
                  {topology.width, topology.mlp_width});
}

void add_feed_forward(std::vector<GenebJanusDnaTensorRequirement> *const output,
                      const std::string &prefix, const bool expert_layer,
                      const GenebJanusDnaTopology &topology) {
  if (!expert_layer) {
    add_mlp(output, prefix, topology);
    return;
  }
  add_requirement(output, prefix + "router.weight",
                  {topology.experts, topology.width});
  for (std::size_t expert = 0U; expert < topology.experts; ++expert)
    add_mlp(output, prefix + "experts." + std::to_string(expert) + ".",
            topology);
}

void add_mamba_direction(
    std::vector<GenebJanusDnaTensorRequirement> *const output,
    const std::string &prefix, const bool tied_projection,
    const GenebJanusDnaTopology &topology) {
  if (!tied_projection) {
    add_requirement(output, prefix + "in_proj.weight",
                    {topology.inner_width * 2U, topology.width});
  }
  add_requirement(output, prefix + "A_log",
                  {topology.inner_width, topology.state_width});
  add_requirement(output, prefix + "D", {topology.inner_width});
  add_requirement(output, prefix + "conv1d.weight",
                  {topology.inner_width, 1U, topology.convolution_width});
  add_requirement(output, prefix + "conv1d.bias", {topology.inner_width});
  add_requirement(output, prefix + "x_proj.weight",
                  {topology.time_step_rank + topology.state_width * 2U,
                   topology.inner_width});
  add_requirement(output, prefix + "dt_proj.weight",
                  {topology.inner_width, topology.time_step_rank});
  add_requirement(output, prefix + "dt_proj.bias", {topology.inner_width});
  if (!tied_projection) {
    add_requirement(output, prefix + "out_proj.weight",
                    {topology.width, topology.inner_width});
  }
  add_requirement(output, prefix + "dt_layernorm.weight",
                  {topology.time_step_rank});
  add_requirement(output, prefix + "b_layernorm.weight",
                  {topology.state_width});
  add_requirement(output, prefix + "c_layernorm.weight",
                  {topology.state_width});
}

void add_middle_attention_direction(
    std::vector<GenebJanusDnaTensorRequirement> *const output,
    const std::string &prefix, const std::string_view direction,
    const bool expert_layer, const GenebJanusDnaTopology &topology) {
  const std::string suffix{direction};
  add_requirement(output, prefix + "input_layernorm_" + suffix + ".weight",
                  {topology.width});
  add_requirement(output, prefix + "pre_ff_layernorm_" + suffix + ".weight",
                  {topology.width});
  for (const std::string_view projection :
       {"q_proj", "k_proj", "v_proj", "o_proj"}) {
    add_requirement(output,
                    prefix + "self_attn_" + suffix + "." +
                        std::string{projection} + ".weight",
                    {topology.width, topology.width});
  }
  add_feed_forward(output, prefix + "feed_forward_" + suffix + ".",
                   expert_layer, topology);
}

void add_mamba_layer(std::vector<GenebJanusDnaTensorRequirement> *const output,
                     const std::size_t layer,
                     const GenebJanusDnaTopology &topology) {
  const std::string prefix = layer_prefix(layer) + "mamba_module.";
  add_mamba_direction(output, prefix + "mamba_fwd.", false, topology);
  // The converter proves the source reverse in/out entries storage-alias the
  // forward pair. The artifact carries one canonical payload for each pair.
  add_mamba_direction(output, prefix + "mamba_rev.", true, topology);
  const bool expert_layer = layer % 2U == 1U;
  for (const std::string_view direction : {"fwd", "bwd"}) {
    const std::string suffix{direction};
    add_requirement(output, prefix + "input_layernorm_" + suffix + ".weight",
                    {topology.width});
    add_requirement(output, prefix + "pre_ff_layernorm_" + suffix + ".weight",
                    {topology.width});
    add_feed_forward(output, prefix + "feed_forward_" + suffix + ".",
                     expert_layer, topology);
  }
}

void add_middle_attention_layer(
    std::vector<GenebJanusDnaTensorRequirement> *const output,
    const std::size_t layer, const GenebJanusDnaTopology &topology) {
  const std::string prefix = layer_prefix(layer) + "attn.";
  const bool expert_layer = layer % 2U == 1U;
  add_middle_attention_direction(output, prefix, "fwd", expert_layer, topology);
  add_middle_attention_direction(output, prefix, "bwd", expert_layer, topology);
}

bool production_topology(const GenebJanusDnaTopology &topology) noexcept {
  return topology.vocabulary_size == kProductionVocabulary &&
         topology.tokenizer_vocabulary_size == kTokenizerVocabulary &&
         topology.width == kProductionWidth &&
         topology.layers == kProductionLayers &&
         topology.attention_heads == kProductionHeads &&
         topology.head_dimension == kProductionHeadDimension &&
         topology.flex_attention_head_dimension ==
             kProductionFlexHeadDimension &&
         topology.inner_width == kProductionInnerWidth &&
         topology.state_width == kProductionStateWidth &&
         topology.convolution_width == kProductionConvolutionWidth &&
         topology.time_step_rank == kProductionTimeStepRank &&
         topology.mlp_width == kProductionMlpWidth &&
         topology.experts == kProductionExperts &&
         topology.experts_per_token == 2U &&
         topology.maximum_sequence_length == kProductionMaximumSequenceLength &&
         topology.middle_attention_layer == kProductionMiddleAttentionLayer &&
         topology.pad_token_id == kPadToken;
}

bool tiny_topology(const GenebJanusDnaTopology &topology) noexcept {
  return topology.vocabulary_size == kProductionVocabulary &&
         topology.tokenizer_vocabulary_size == kTokenizerVocabulary &&
         topology.width == 4U && topology.layers == 2U &&
         topology.attention_heads == 2U && topology.head_dimension == 2U &&
         topology.flex_attention_head_dimension == 4U &&
         topology.inner_width == 8U && topology.state_width == 2U &&
         topology.convolution_width == 2U && topology.time_step_rank == 2U &&
         topology.mlp_width == 8U && topology.experts == 4U &&
         topology.experts_per_token == 2U &&
         topology.maximum_sequence_length == 8U &&
         topology.middle_attention_layer == 0U &&
         topology.pad_token_id == kPadToken;
}

Status validate_tensor(const MambaTensorView &tensor,
                       const GenebJanusDnaTensorRequirement &requirement) {
  if (tensor.dtype != requirement.dtype)
    return format_error("tensor dtype differs: " + requirement.name);
  if (tensor.shape != requirement.shape)
    return format_error("tensor shape differs: " + requirement.name);
  std::size_t elements = 1U;
  for (const auto dimension : tensor.shape) {
    if (!checked_multiply(elements, dimension, &elements))
      return format_error("tensor shape overflows: " + requirement.name);
  }
  std::size_t expected_bytes = 0U;
  if (!checked_multiply(elements, sizeof(float), &expected_bytes) ||
      tensor.bytes != expected_bytes || tensor.data == nullptr)
    return format_error("tensor byte extent differs: " + requirement.name);
  return Status::Ok();
}

float tensor_value(const MambaTensorView &tensor,
                   const std::size_t index) noexcept {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(float), sizeof(value));
  return value;
}

void add_in_place(std::vector<float> *const target,
                  const std::vector<float> &increment) noexcept {
  for (std::size_t index = 0U; index < target->size(); ++index)
    (*target)[index] += increment[index];
}

float silu(const float value) noexcept {
  return value / (1.0F + std::exp(-value));
}

struct MlpWeights final {
  MambaTensorView gate;
  MambaTensorView up;
  MambaTensorView down;
};

struct FeedForwardWeights final {
  bool expert_layer{false};
  MlpWeights dense;
  MambaTensorView router;
  std::vector<MlpWeights> experts;
};

Status run_mlp(const std::vector<float> &input, const std::size_t rows,
               const GenebJanusDnaTopology &topology, const MlpWeights &weights,
               evo::detail::LinearExecutor *const executor,
               std::vector<float> *const output) {
  std::vector<float> gate;
  auto status = mamba_linear_f32(input, rows, topology.width, weights.gate,
                                 topology.mlp_width, executor, &gate);
  if (!status.ok())
    return status;
  std::vector<float> up;
  status = mamba_linear_f32(input, rows, topology.width, weights.up,
                            topology.mlp_width, executor, &up);
  if (!status.ok())
    return status;
  for (std::size_t index = 0U; index < gate.size(); ++index)
    gate[index] = silu(gate[index]) * up[index];
  return mamba_linear_f32(gate, rows, topology.mlp_width, weights.down,
                          topology.width, executor, output);
}

struct ExpertAssignment final {
  std::size_t row{0};
  float weight{0.0F};
};

Status run_feed_forward(const std::vector<float> &input, const std::size_t rows,
                        const GenebJanusDnaTopology &topology,
                        const FeedForwardWeights &weights,
                        evo::detail::LinearExecutor *const executor,
                        std::vector<float> *const output) {
  if (!weights.expert_layer)
    return run_mlp(input, rows, topology, weights.dense, executor, output);

  std::vector<float> logits;
  auto status = mamba_linear_f32(input, rows, topology.width, weights.router,
                                 topology.experts, executor, &logits);
  if (!status.ok())
    return status;
  std::vector<std::vector<ExpertAssignment>> assignments(topology.experts);
  for (std::size_t row = 0U; row < rows; ++row) {
    const std::size_t offset = row * topology.experts;
    const auto begin = logits.begin() + static_cast<std::ptrdiff_t>(offset);
    const float maximum = *std::max_element(
        begin, begin + static_cast<std::ptrdiff_t>(topology.experts));
    std::vector<float> probabilities(topology.experts);
    float denominator = 0.0F;
    for (std::size_t expert = 0U; expert < topology.experts; ++expert) {
      probabilities[expert] = std::exp(logits[offset + expert] - maximum);
      denominator += probabilities[expert];
    }
    for (float &probability : probabilities)
      probability /= denominator;

    // torch.topk does not promise stable tied indices across devices. The
    // native ABI freezes the audited all-equal behavior: the lower expert ID
    // wins an exact tie. Tied weights commute in the source's weighted sum.
    std::size_t first = 0U;
    for (std::size_t expert = 1U; expert < topology.experts; ++expert) {
      if (probabilities[expert] > probabilities[first])
        first = expert;
    }
    std::size_t second = first == 0U ? 1U : 0U;
    for (std::size_t expert = 0U; expert < topology.experts; ++expert) {
      if (expert == first)
        continue;
      if (probabilities[expert] > probabilities[second] ||
          (probabilities[expert] == probabilities[second] && expert < second))
        second = expert;
    }
    assignments[first].push_back({row, probabilities[first]});
    assignments[second].push_back({row, probabilities[second]});
  }

  output->assign(rows * topology.width, 0.0F);
  for (std::size_t expert = 0U; expert < topology.experts; ++expert) {
    const auto &selected = assignments[expert];
    if (selected.empty())
      continue;
    std::vector<float> gathered(selected.size() * topology.width);
    for (std::size_t item = 0U; item < selected.size(); ++item) {
      std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(
                                      selected[item].row * topology.width),
                  topology.width,
                  gathered.begin() +
                      static_cast<std::ptrdiff_t>(item * topology.width));
    }
    std::vector<float> expert_output;
    status = run_mlp(gathered, selected.size(), topology,
                     weights.experts[expert], executor, &expert_output);
    if (!status.ok())
      return status;
    for (std::size_t item = 0U; item < selected.size(); ++item) {
      for (std::size_t column = 0U; column < topology.width; ++column) {
        (*output)[selected[item].row * topology.width + column] +=
            expert_output[item * topology.width + column] *
            selected[item].weight;
      }
    }
  }
  return Status::Ok();
}

Status normalize(const std::vector<float> &input, const std::size_t rows,
                 const GenebJanusDnaTopology &topology,
                 const MambaTensorView &scale,
                 std::vector<float> *const output) {
  return mamba_rms_norm(input, rows, topology.width, scale,
                        topology.norm_epsilon, output);
}

struct DirectionBlockWeights final {
  MambaTensorView input_norm;
  MambaTensorView pre_ff_norm;
  FeedForwardWeights feed_forward;
};

struct MambaLayerWeights final {
  Mamba1Weights forward_mamba;
  Mamba1Weights reverse_mamba;
  DirectionBlockWeights forward;
  DirectionBlockWeights reverse;
};

struct AttentionWeights final {
  MambaTensorView query;
  MambaTensorView key;
  MambaTensorView value;
  MambaTensorView output;
};

struct AttentionLayerWeights final {
  AttentionWeights attention;
  DirectionBlockWeights block;
};

struct MiddleAttentionLayerWeights final {
  AttentionLayerWeights forward;
  AttentionLayerWeights reverse;
};

struct FinalAttentionWeights final {
  MambaTensorView input_norm;
  AttentionWeights attention;
  MambaTensorView pre_ff_norm;
  MlpWeights feed_forward;
};

Status run_middle_attention(const std::vector<float> &input,
                            const std::size_t rows,
                            const std::vector<std::uint8_t> &attention_mask,
                            const GenebJanusDnaTopology &topology,
                            const AttentionWeights &weights,
                            evo::detail::LinearExecutor *const executor,
                            std::vector<float> *const output) {
  std::vector<float> query;
  auto status = mamba_linear_f32(input, rows, topology.width, weights.query,
                                 topology.width, executor, &query);
  if (!status.ok())
    return status;
  std::vector<float> key;
  status = mamba_linear_f32(input, rows, topology.width, weights.key,
                            topology.width, executor, &key);
  if (!status.ok())
    return status;
  std::vector<float> value;
  status = mamba_linear_f32(input, rows, topology.width, weights.value,
                            topology.width, executor, &value);
  if (!status.ok())
    return status;

  std::vector<float> attended(rows * topology.width, 0.0F);
  std::vector<float> scores(rows);
  const float scale =
      1.0F / std::sqrt(static_cast<float>(topology.head_dimension));
  for (std::size_t head = 0U; head < topology.attention_heads; ++head) {
    const std::size_t head_offset = head * topology.head_dimension;
    for (std::size_t query_row = 0U; query_row < rows; ++query_row) {
      if (attention_mask[query_row] == 0U)
        continue;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t key_row = 0U; key_row <= query_row; ++key_row) {
        if (attention_mask[key_row] == 0U) {
          scores[key_row] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float dot = 0.0F;
        for (std::size_t column = 0U; column < topology.head_dimension;
             ++column) {
          dot += query[query_row * topology.width + head_offset + column] *
                 key[key_row * topology.width + head_offset + column];
        }
        scores[key_row] = dot * scale;
        maximum = std::max(maximum, scores[key_row]);
      }
      float denominator = 0.0F;
      for (std::size_t key_row = 0U; key_row <= query_row; ++key_row) {
        if (attention_mask[key_row] == 0U)
          continue;
        scores[key_row] = std::exp(scores[key_row] - maximum);
        denominator += scores[key_row];
      }
      for (std::size_t column = 0U; column < topology.head_dimension;
           ++column) {
        float total = 0.0F;
        for (std::size_t key_row = 0U; key_row <= query_row; ++key_row) {
          if (attention_mask[key_row] != 0U) {
            total += scores[key_row] / denominator *
                     value[key_row * topology.width + head_offset + column];
          }
        }
        attended[query_row * topology.width + head_offset + column] = total;
      }
    }
  }
  return mamba_linear_f32(attended, rows, topology.width, weights.output,
                          topology.width, executor, output);
}

bool flex_mask(const std::size_t query, const std::size_t key,
               const std::size_t original_rows) noexcept {
  const bool first =
      key < original_rows && query < original_rows && query >= key;
  const bool second = key >= original_rows && query < original_rows &&
                      key >= original_rows + query + 2U;
  const bool third = key < original_rows && query >= original_rows &&
                     query >= key + original_rows + 2U;
  const bool fourth =
      key >= original_rows && query >= original_rows && query <= key;
  return first || second || third || fourth;
}

Status run_final_flex_attention(const std::vector<float> &input,
                                const std::size_t original_rows,
                                const GenebJanusDnaTopology &topology,
                                const AttentionWeights &weights,
                                evo::detail::LinearExecutor *const executor,
                                std::vector<float> *const output) {
  const std::size_t rows = original_rows * 2U;
  std::vector<float> query;
  auto status = mamba_linear_f32(input, rows, topology.width, weights.query,
                                 topology.width, executor, &query);
  if (!status.ok())
    return status;
  std::vector<float> key;
  status = mamba_linear_f32(input, rows, topology.width, weights.key,
                            topology.width, executor, &key);
  if (!status.ok())
    return status;
  std::vector<float> value;
  status = mamba_linear_f32(input, rows, topology.width, weights.value,
                            topology.width, executor, &value);
  if (!status.ok())
    return status;

  // The source physically zero-pads Q/K/V from 18 to 32 dimensions. Zero
  // padding leaves the dot/value components unchanged, but flex_attention's
  // default scale is therefore exactly 1/sqrt(32), not 1/sqrt(18).
  const float scale =
      1.0F /
      std::sqrt(static_cast<float>(topology.flex_attention_head_dimension));
  std::vector<float> attended(rows * topology.width, 0.0F);
  std::vector<float> scores(rows);
  for (std::size_t head = 0U; head < topology.attention_heads; ++head) {
    const std::size_t head_offset = head * topology.head_dimension;
    for (std::size_t query_row = 0U; query_row < rows; ++query_row) {
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t key_row = 0U; key_row < rows; ++key_row) {
        if (!flex_mask(query_row, key_row, original_rows)) {
          scores[key_row] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float dot = 0.0F;
        for (std::size_t column = 0U; column < topology.head_dimension;
             ++column) {
          dot += query[query_row * topology.width + head_offset + column] *
                 key[key_row * topology.width + head_offset + column];
        }
        scores[key_row] = dot * scale;
        maximum = std::max(maximum, scores[key_row]);
      }
      float denominator = 0.0F;
      for (std::size_t key_row = 0U; key_row < rows; ++key_row) {
        if (!flex_mask(query_row, key_row, original_rows))
          continue;
        scores[key_row] = std::exp(scores[key_row] - maximum);
        denominator += scores[key_row];
      }
      for (std::size_t column = 0U; column < topology.head_dimension;
           ++column) {
        float total = 0.0F;
        for (std::size_t key_row = 0U; key_row < rows; ++key_row) {
          if (flex_mask(query_row, key_row, original_rows)) {
            total += scores[key_row] / denominator *
                     value[key_row * topology.width + head_offset + column];
          }
        }
        attended[query_row * topology.width + head_offset + column] = total;
      }
    }
  }
  return mamba_linear_f32(attended, rows, topology.width, weights.output,
                          topology.width, executor, output);
}

} // namespace

Status validate_geneb_janusdna_topology(const GenebJanusDnaTopology &topology) {
  if ((topology.variant != GenebJanusDnaVariant::kWithMiddleAttention &&
       topology.variant != GenebJanusDnaVariant::kWithoutMiddleAttention) ||
      topology.weight_dtype != TensorDType::kF32 ||
      topology.norm_epsilon != 1.0e-6F ||
      topology.attention_heads * topology.head_dimension != topology.width ||
      topology.flex_attention_head_dimension < topology.head_dimension ||
      topology.experts_per_token != 2U || topology.pad_token_id != kPadToken ||
      (!production_topology(topology) && !tiny_topology(topology))) {
    return invalid(
        "topology differs from the production or independent tiny tuple");
  }
  return Status::Ok();
}

Status canonical_geneb_janusdna_tensors(
    const GenebJanusDnaTopology &topology,
    std::vector<GenebJanusDnaTensorRequirement> *const output) {
  if (output == nullptr)
    return invalid("tensor-requirement output is null");
  auto status = validate_geneb_janusdna_topology(topology);
  if (!status.ok())
    return status;
  std::vector<GenebJanusDnaTensorRequirement> result;
  add_requirement(&result, "embed_tokens.weight",
                  {topology.vocabulary_size, topology.width});
  for (std::size_t layer = 0U; layer < topology.layers; ++layer) {
    if (topology.variant == GenebJanusDnaVariant::kWithMiddleAttention &&
        layer == topology.middle_attention_layer)
      add_middle_attention_layer(&result, layer, topology);
    else
      add_mamba_layer(&result, layer, topology);
  }
  add_requirement(&result, "final_layernorm.weight", {topology.width});
  for (const std::string_view projection : {"q_proj", "k_proj", "v_proj"}) {
    add_requirement(&result,
                    "final_attention.self_attn." + std::string{projection} +
                        ".weight",
                    {topology.width, topology.width});
  }
  add_requirement(&result, "final_attention.self_attn.o_projs.0.weight",
                  {topology.width, topology.width});
  add_mlp(&result, "final_attention.feed_forward.", topology);
  add_requirement(&result, "final_attention.input_layernorm.weight",
                  {topology.width});
  add_requirement(&result, "final_attention.pre_ff_layernorm.weight",
                  {topology.width});
  *output = std::move(result);
  return Status::Ok();
}

Status
geneb_janusdna_topology_from_artifact(const ModelFile &artifact,
                                      GenebJanusDnaTopology *const output) {
  if (output == nullptr)
    return invalid("topology output is null");
  if (artifact.profile() != kGenebJanusDnaArtifactProfile)
    return format_error("profile must be '" +
                        std::string{kGenebJanusDnaArtifactProfile} + "'");
  if (!artifact.tokenizer_asset_descriptor().has_value())
    return format_error("the four-key tokenizer descriptor is required");
  auto status =
      require_literal(artifact, "runtime.abi", kGenebJanusDnaRuntimeAbi);
  if (!status.ok())
    return status;
  status = require_literal(artifact, "model.architecture",
                           kGenebJanusDnaArchitecture);
  if (!status.ok())
    return status;

  const std::set<std::string_view> expected{
      "janus.variant",
      "janus.vocab_size",
      "janus.tokenizer_vocab_size",
      "janus.hidden_size",
      "janus.num_layers",
      "janus.num_attention_heads",
      "janus.head_dim",
      "janus.flex_head_dim",
      "janus.inner_size",
      "janus.state_size",
      "janus.conv_size",
      "janus.dt_rank",
      "janus.mlp_size",
      "janus.num_experts",
      "janus.experts_per_token",
      "janus.max_seqlen",
      "janus.middle_attention_layer",
      "janus.pad_token_id",
      "janus.norm_epsilon",
      "janus.weight_dtype",
      "janus.norm_placement",
      "janus.activation",
      "janus.mamba_parameter_norm",
      "janus.bidirectional_layout",
      "janus.tied_parameters",
      "janus.expert_routing",
      "janus.middle_attention",
      "janus.final_attention",
      "janus.flex_scale",
      "janus.final_fusion",
      "janus.final_mlp",
      "janus.hidden_tap",
      "janus.pooling",
      "janus.special_tokens",
      "janus.mask_domain",
      "janus.tokenizer_kind",
      "janus.official_reference_device",
      "janus.alias_encoding",
  };
  for (const auto &entry : artifact.metadata()) {
    if (entry.key.compare(0U, 6U, "janus.") == 0U &&
        expected.find(std::string_view{entry.key}) == expected.end()) {
      return format_error("unexpected janus.* metadata: " + entry.key);
    }
  }

  GenebJanusDnaTopology topology;
  std::string variant;
  status = metadata_string(artifact, "janus.variant", &variant);
  if (!status.ok())
    return status;
  if (variant == "with-middle-attention")
    topology.variant = GenebJanusDnaVariant::kWithMiddleAttention;
  else if (variant == "without-middle-attention")
    topology.variant = GenebJanusDnaVariant::kWithoutMiddleAttention;
  else
    return format_error("janus.variant is unsupported");

  for (const auto &[key, target] :
       std::vector<std::pair<std::string_view, std::size_t *>>{
           {"janus.vocab_size", &topology.vocabulary_size},
           {"janus.tokenizer_vocab_size", &topology.tokenizer_vocabulary_size},
           {"janus.hidden_size", &topology.width},
           {"janus.num_layers", &topology.layers},
           {"janus.num_attention_heads", &topology.attention_heads},
           {"janus.head_dim", &topology.head_dimension},
           {"janus.flex_head_dim", &topology.flex_attention_head_dimension},
           {"janus.inner_size", &topology.inner_width},
           {"janus.state_size", &topology.state_width},
           {"janus.conv_size", &topology.convolution_width},
           {"janus.dt_rank", &topology.time_step_rank},
           {"janus.mlp_size", &topology.mlp_width},
           {"janus.num_experts", &topology.experts},
           {"janus.experts_per_token", &topology.experts_per_token},
           {"janus.max_seqlen", &topology.maximum_sequence_length},
           {"janus.middle_attention_layer", &topology.middle_attention_layer},
           {"janus.pad_token_id", &topology.pad_token_id}}) {
    status = metadata_size(artifact, key, target);
    if (!status.ok())
      return status;
  }
  status =
      metadata_float(artifact, "janus.norm_epsilon", &topology.norm_epsilon);
  if (!status.ok())
    return status;
  topology.weight_dtype = TensorDType::kF32;

  const std::string_view middle =
      topology.variant == GenebJanusDnaVariant::kWithMiddleAttention
          ? "causal-original-right-mask-both-directions"
          : "absent";
  for (const auto &[key, expected_value] :
       std::vector<std::pair<std::string_view, std::string_view>>{
           {"janus.weight_dtype", "F32"},
           {"janus.norm_placement", "pre-rms"},
           {"janus.activation", "silu"},
           {"janus.mamba_parameter_norm", "dt-b-c-rms"},
           {"janus.bidirectional_layout", "separate-concatenated-halves"},
           {"janus.tied_parameters", "mamba-in-out-only"},
           {"janus.expert_routing", "softmax-top2-no-renormalize"},
           {"janus.middle_attention", middle},
           {"janus.final_attention", "flex-four-condition-ignore-padding-mask"},
           {"janus.flex_scale", "padded-head-dimension"},
           {"janus.final_fusion", "repos-formasked-shift-1"},
           {"janus.final_mlp", "identity-residual-double"},
           {"janus.hidden_tap", "twice-post-final-rmsnorm"},
           {"janus.pooling", "attention-mask-mean"},
           {"janus.special_tokens", "none"},
           {"janus.mask_domain", "original-attention-mask"},
           {"janus.tokenizer_kind", "single-nucleotide-uppercase"},
           {"janus.official_reference_device", "gpu"},
           {"janus.alias_encoding", "canonical-shared-in-out"}}) {
    status = require_literal(artifact, key, expected_value);
    if (!status.ok())
      return status;
  }

  status = validate_geneb_janusdna_topology(topology);
  if (!status.ok())
    return format_error(status.message());
  const auto match_common = [&](const std::string_view key,
                                const std::size_t expected_value) {
    std::size_t actual = 0U;
    auto match_status = metadata_size(artifact, key, &actual);
    if (!match_status.ok())
      return match_status;
    if (actual != expected_value)
      return format_error("common metadata disagrees with Janus topology: " +
                          std::string{key});
    return Status::Ok();
  };
  for (const auto &[key, value] :
       std::vector<std::pair<std::string_view, std::size_t>>{
           {"config.vocab_size", topology.vocabulary_size},
           {"config.hidden_size", topology.width},
           {"config.num_layers", topology.layers},
           {"config.max_seqlen", topology.maximum_sequence_length},
           {"runtime.tokenizer_vocabulary_size",
            topology.tokenizer_vocabulary_size},
           {"runtime.embedding_layer_count", topology.layers + 1U}}) {
    status = match_common(key, value);
    if (!status.ok())
      return status;
  }
  *output = topology;
  return Status::Ok();
}

Status geneb_janusdna_pool(const GenebJanusDnaForwardResult &forward,
                           const std::vector<std::uint8_t> &attention_mask,
                           std::vector<float> *const output) {
  if (output == nullptr)
    return invalid("pool output is null");
  if (forward.rows == 0U || forward.width == 0U ||
      forward.final_hidden.size() != forward.rows * forward.width ||
      attention_mask.size() != forward.rows)
    return invalid("pool dimensions differ");
  float denominator = 0.0F;
  output->assign(forward.width, 0.0F);
  for (std::size_t row = 0U; row < forward.rows; ++row) {
    if (attention_mask[row] > 1U)
      return invalid("attention mask must contain only zero/one");
    if (attention_mask[row] == 0U)
      continue;
    denominator += 1.0F;
    for (std::size_t column = 0U; column < forward.width; ++column) {
      (*output)[column] += forward.final_hidden[row * forward.width + column];
    }
  }
  if (denominator == 0.0F)
    return invalid("attention mask has no effective token");
  for (float &value : *output)
    value /= denominator;
  return Status::Ok();
}

struct GenebJanusDnaModel::Impl final {
  GenebJanusDnaTopology topology;
  std::shared_ptr<evo::detail::LinearExecutor> linear_executor;
  MambaTensorView embedding;
  std::vector<MambaLayerWeights> mamba_layers;
  std::vector<MiddleAttentionLayerWeights> attention_layers;
  std::vector<bool> is_attention_layer;
  FinalAttentionWeights final_attention;
  MambaTensorView final_norm;

  Mamba1Config mamba_config() const {
    Mamba1Config config;
    config.model_width = topology.width;
    config.inner_width = topology.inner_width;
    config.state_width = topology.state_width;
    config.convolution_width = topology.convolution_width;
    config.time_step_rank = topology.time_step_rank;
    config.parameter_projection_rms_norm = true;
    config.parameter_projection_norm_epsilon = topology.norm_epsilon;
    return config;
  }

  Status run_block_feed_forward(std::vector<float> *const hidden,
                                const std::size_t rows,
                                const DirectionBlockWeights &weights) const {
    std::vector<float> normalized;
    auto status =
        normalize(*hidden, rows, topology, weights.pre_ff_norm, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> feed_forward;
    status = run_feed_forward(normalized, rows, topology, weights.feed_forward,
                              linear_executor.get(), &feed_forward);
    if (!status.ok())
      return status;
    add_in_place(hidden, feed_forward);
    return Status::Ok();
  }

  Status run_mamba_direction(std::vector<float> *const hidden,
                             const std::size_t rows, const Mamba1Weights &mamba,
                             const DirectionBlockWeights &block) const {
    std::vector<float> normalized;
    auto status =
        normalize(*hidden, rows, topology, block.input_norm, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> mixed;
    status = mamba1_mixer_f32(normalized, rows, mamba_config(), mamba,
                              linear_executor.get(), &mixed);
    if (!status.ok())
      return status;
    add_in_place(hidden, mixed);
    return run_block_feed_forward(hidden, rows, block);
  }

  Status
  run_attention_direction(std::vector<float> *const hidden,
                          const std::size_t rows,
                          const std::vector<std::uint8_t> &attention_mask,
                          const AttentionLayerWeights &weights) const {
    std::vector<float> normalized;
    auto status = normalize(*hidden, rows, topology, weights.block.input_norm,
                            &normalized);
    if (!status.ok())
      return status;
    std::vector<float> attended;
    status = run_middle_attention(normalized, rows, attention_mask, topology,
                                  weights.attention, linear_executor.get(),
                                  &attended);
    if (!status.ok())
      return status;
    add_in_place(hidden, attended);
    return run_block_feed_forward(hidden, rows, weights.block);
  }

  Status run_final_attention(std::vector<float> *const hidden,
                             const std::size_t rows) const {
    std::vector<float> normalized;
    auto status = normalize(*hidden, rows * 2U, topology,
                            final_attention.input_norm, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> attended;
    status = run_final_flex_attention(normalized, rows, topology,
                                      final_attention.attention,
                                      linear_executor.get(), &attended);
    if (!status.ok())
      return status;
    add_in_place(hidden, attended);
    status = normalize(*hidden, rows * 2U, topology,
                       final_attention.pre_ff_norm, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> feed_forward;
    status =
        run_mlp(normalized, rows * 2U, topology, final_attention.feed_forward,
                linear_executor.get(), &feed_forward);
    if (!status.ok())
      return status;
    add_in_place(hidden, feed_forward);
    return Status::Ok();
  }

  Status final_fusion(const std::vector<float> &hidden, const std::size_t rows,
                      std::vector<float> *const output) const {
    if (rows <= 2U)
      return invalid("final fusion requires more than two token rows");
    output->assign(rows * topology.width, 0.0F);
    const auto copy_row = [&](const std::size_t destination,
                              const std::size_t source) {
      std::copy_n(
          hidden.begin() + static_cast<std::ptrdiff_t>(source * topology.width),
          topology.width,
          output->begin() +
              static_cast<std::ptrdiff_t>(destination * topology.width));
    };
    copy_row(0U, rows + 1U);
    for (std::size_t row = 0U; row < rows - 2U; ++row) {
      for (std::size_t column = 0U; column < topology.width; ++column) {
        (*output)[(row + 1U) * topology.width + column] =
            hidden[row * topology.width + column] +
            hidden[(rows + 2U + row) * topology.width + column];
      }
    }
    copy_row(rows - 1U, rows - 2U);
    return Status::Ok();
  }

  Status run(const std::vector<TokenId> &tokens,
             const std::vector<std::uint8_t> &attention_mask,
             const std::set<std::size_t> &captures,
             GenebJanusDnaForwardResult *const output) const {
    const std::size_t rows = tokens.size();
    std::vector<float> original_embedding(rows * topology.width);
    for (std::size_t row = 0U; row < rows; ++row) {
      for (std::size_t column = 0U; column < topology.width; ++column) {
        original_embedding[row * topology.width + column] =
            tensor_value(embedding, tokens[row] * topology.width + column);
      }
    }
    if (captures.find(0U) != captures.end())
      output->captures.push_back({0U, original_embedding});

    std::vector<float> forward = original_embedding;
    std::vector<float> reverse(original_embedding.size());
    for (std::size_t row = 0U; row < rows; ++row) {
      std::copy_n(
          original_embedding.begin() +
              static_cast<std::ptrdiff_t>((rows - 1U - row) * topology.width),
          topology.width,
          reverse.begin() + static_cast<std::ptrdiff_t>(row * topology.width));
    }

    std::size_t mamba_index = 0U;
    std::size_t attention_index = 0U;
    for (std::size_t layer = 0U; layer < topology.layers; ++layer) {
      Status status = Status::Ok();
      if (is_attention_layer[layer]) {
        const auto &weights = attention_layers[attention_index++];
        status = run_attention_direction(&forward, rows, attention_mask,
                                         weights.forward);
        if (status.ok()) {
          // Pinned source bug: the unmodified original right-padding mask is
          // applied to the already-reversed half.
          status = run_attention_direction(&reverse, rows, attention_mask,
                                           weights.reverse);
        }
      } else {
        const auto &weights = mamba_layers[mamba_index++];
        status = run_mamba_direction(&forward, rows, weights.forward_mamba,
                                     weights.forward);
        if (status.ok()) {
          status = run_mamba_direction(&reverse, rows, weights.reverse_mamba,
                                       weights.reverse);
        }
      }
      if (!status.ok())
        return status;
    }

    // The final attention sees both halves in original coordinate order.
    std::vector<float> doubled(rows * topology.width * 2U);
    std::copy(forward.begin(), forward.end(), doubled.begin());
    for (std::size_t row = 0U; row < rows; ++row) {
      std::copy_n(reverse.begin() + static_cast<std::ptrdiff_t>(
                                        (rows - 1U - row) * topology.width),
                  topology.width,
                  doubled.begin() + static_cast<std::ptrdiff_t>(
                                        (rows + row) * topology.width));
    }
    auto status = run_final_attention(&doubled, rows);
    if (!status.ok())
      return status;
    std::vector<float> fused;
    status = final_fusion(doubled, rows, &fused);
    if (!status.ok())
      return status;
    status =
        normalize(fused, rows, topology, final_norm, &output->final_hidden);
    if (!status.ok())
      return status;
    // GENEB replaces final_mlp with Identity but the source retains the
    // residual addition: residual + Identity(residual) == exactly 2*x.
    for (float &value : output->final_hidden)
      value *= 2.0F;
    if (captures.find(topology.layers) != captures.end())
      output->captures.push_back({topology.layers, output->final_hidden});
    return geneb_janusdna_pool(*output, attention_mask, &output->pooled);
  }
};

GenebJanusDnaModel::GenebJanusDnaModel() = default;
GenebJanusDnaModel::~GenebJanusDnaModel() = default;
GenebJanusDnaModel::GenebJanusDnaModel(GenebJanusDnaModel &&) noexcept =
    default;
GenebJanusDnaModel &
GenebJanusDnaModel::operator=(GenebJanusDnaModel &&) noexcept = default;

Status GenebJanusDnaModel::load(
    const GenebJanusDnaTopology &topology,
    const std::vector<GenebJanusDnaNamedTensorView> &tensors,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  std::vector<GenebJanusDnaTensorRequirement> requirements;
  auto status = canonical_geneb_janusdna_tensors(topology, &requirements);
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
  const auto load_mlp = [&](const std::string &prefix,
                            MlpWeights *const weights) {
    weights->gate = view(prefix + "gate_proj.weight");
    weights->up = view(prefix + "up_proj.weight");
    weights->down = view(prefix + "down_proj.weight");
  };
  const auto load_feed_forward = [&](const std::string &prefix,
                                     const bool expert_layer,
                                     FeedForwardWeights *const weights) {
    weights->expert_layer = expert_layer;
    if (!expert_layer) {
      load_mlp(prefix, &weights->dense);
      return;
    }
    weights->router = view(prefix + "router.weight");
    weights->experts.resize(topology.experts);
    for (std::size_t expert = 0U; expert < topology.experts; ++expert) {
      load_mlp(prefix + "experts." + std::to_string(expert) + ".",
               &weights->experts[expert]);
    }
  };
  const auto load_block = [&](const std::string &prefix,
                              const std::string_view direction,
                              const bool expert_layer,
                              DirectionBlockWeights *const block) {
    const std::string suffix{direction};
    block->input_norm = view(prefix + "input_layernorm_" + suffix + ".weight");
    block->pre_ff_norm =
        view(prefix + "pre_ff_layernorm_" + suffix + ".weight");
    load_feed_forward(prefix + "feed_forward_" + suffix + ".", expert_layer,
                      &block->feed_forward);
  };
  const auto load_mamba = [&](const std::string &prefix,
                              const Mamba1Weights *const tied,
                              Mamba1Weights *const weights) {
    weights->input_projection = tied == nullptr
                                    ? view(prefix + "in_proj.weight")
                                    : tied->input_projection;
    weights->a_log = view(prefix + "A_log");
    weights->skip = view(prefix + "D");
    weights->convolution_weight = view(prefix + "conv1d.weight");
    weights->convolution_bias = view(prefix + "conv1d.bias");
    weights->x_projection = view(prefix + "x_proj.weight");
    weights->time_step_projection = view(prefix + "dt_proj.weight");
    weights->time_step_bias = view(prefix + "dt_proj.bias");
    weights->output_projection = tied == nullptr
                                     ? view(prefix + "out_proj.weight")
                                     : tied->output_projection;
    weights->projected_time_step_norm_scale =
        view(prefix + "dt_layernorm.weight");
    weights->projected_b_norm_scale = view(prefix + "b_layernorm.weight");
    weights->projected_c_norm_scale = view(prefix + "c_layernorm.weight");
  };
  const auto load_attention = [&](const std::string &prefix,
                                  AttentionWeights *const weights) {
    weights->query = view(prefix + "q_proj.weight");
    weights->key = view(prefix + "k_proj.weight");
    weights->value = view(prefix + "v_proj.weight");
    weights->output = view(prefix + "o_proj.weight");
  };

  auto implementation = std::make_unique<Impl>();
  implementation->topology = topology;
  implementation->linear_executor = std::move(linear_executor);
  implementation->embedding = view("embed_tokens.weight");
  implementation->is_attention_layer.assign(topology.layers, false);
  for (std::size_t layer = 0U; layer < topology.layers; ++layer) {
    const bool middle_attention =
        topology.variant == GenebJanusDnaVariant::kWithMiddleAttention &&
        layer == topology.middle_attention_layer;
    implementation->is_attention_layer[layer] = middle_attention;
    const bool expert_layer = layer % 2U == 1U;
    if (middle_attention) {
      implementation->attention_layers.emplace_back();
      auto &weights = implementation->attention_layers.back();
      const std::string prefix = layer_prefix(layer) + "attn.";
      load_attention(prefix + "self_attn_fwd.", &weights.forward.attention);
      load_attention(prefix + "self_attn_bwd.", &weights.reverse.attention);
      load_block(prefix, "fwd", expert_layer, &weights.forward.block);
      load_block(prefix, "bwd", expert_layer, &weights.reverse.block);
    } else {
      implementation->mamba_layers.emplace_back();
      auto &weights = implementation->mamba_layers.back();
      const std::string prefix = layer_prefix(layer) + "mamba_module.";
      load_mamba(prefix + "mamba_fwd.", nullptr, &weights.forward_mamba);
      load_mamba(prefix + "mamba_rev.", &weights.forward_mamba,
                 &weights.reverse_mamba);
      load_block(prefix, "fwd", expert_layer, &weights.forward);
      load_block(prefix, "bwd", expert_layer, &weights.reverse);
    }
  }
  implementation->final_norm = view("final_layernorm.weight");
  implementation->final_attention.input_norm =
      view("final_attention.input_layernorm.weight");
  implementation->final_attention.pre_ff_norm =
      view("final_attention.pre_ff_layernorm.weight");
  implementation->final_attention.attention.query =
      view("final_attention.self_attn.q_proj.weight");
  implementation->final_attention.attention.key =
      view("final_attention.self_attn.k_proj.weight");
  implementation->final_attention.attention.value =
      view("final_attention.self_attn.v_proj.weight");
  implementation->final_attention.attention.output =
      view("final_attention.self_attn.o_projs.0.weight");
  load_mlp("final_attention.feed_forward.",
           &implementation->final_attention.feed_forward);
  impl_ = std::move(implementation);
  return Status::Ok();
}

Status GenebJanusDnaModel::load(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  GenebJanusDnaTopology topology;
  auto status = geneb_janusdna_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebJanusDnaNamedTensorView> views;
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

Status GenebJanusDnaModel::load_artifact(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  return load(artifact, std::move(linear_executor));
}

const GenebJanusDnaTopology *GenebJanusDnaModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebJanusDnaModel::linear_executor_name() const noexcept {
  if (impl_ == nullptr || impl_->linear_executor == nullptr)
    return "scalar-reference";
  return impl_->linear_executor->name();
}

Status
GenebJanusDnaModel::forward(const std::vector<TokenId> &tokens,
                            const std::vector<std::uint8_t> &attention_mask,
                            const std::vector<std::size_t> &capture_layers,
                            GenebJanusDnaForwardResult *const output) const {
  if (impl_ == nullptr)
    return invalid("model is not loaded");
  if (output == nullptr)
    return invalid("forward output is null");
  if (tokens.size() <= 2U ||
      tokens.size() > impl_->topology.maximum_sequence_length ||
      attention_mask.size() != tokens.size())
    return invalid("tokens/mask dimensions or sequence length differ");

  bool padding_started = false;
  for (std::size_t row = 0U; row < tokens.size(); ++row) {
    if (tokens[row] >= impl_->topology.tokenizer_vocabulary_size)
      return invalid("token ID exceeds tokenizer vocabulary");
    if (attention_mask[row] > 1U)
      return invalid("attention mask must contain only zero/one");
    if (attention_mask[row] == 0U)
      padding_started = true;
    else if (padding_started)
      return invalid("attention mask must be a right-padded prefix");
    if ((attention_mask[row] == 0U) !=
        (tokens[row] == impl_->topology.pad_token_id))
      return invalid("padding mask and PAD token positions differ");
  }
  if (attention_mask.front() == 0U)
    return invalid("attention mask has no effective token");

  std::set<std::size_t> captures;
  for (const auto layer : capture_layers) {
    if ((layer != 0U && layer != impl_->topology.layers) ||
        !captures.insert(layer).second) {
      return {ErrorCode::kUnsupported,
              "GENEB JanusDNA exposes only token embedding and final hidden"};
    }
  }
  GenebJanusDnaForwardResult result;
  result.rows = tokens.size();
  result.width = impl_->topology.width;
  auto status = impl_->run(tokens, attention_mask, captures, &result);
  if (!status.ok())
    return status;
  *output = std::move(result);
  return Status::Ok();
}

Status GenebJanusDnaModel::pool(const GenebJanusDnaForwardResult &forward,
                                const std::vector<std::uint8_t> &attention_mask,
                                std::vector<float> *const output) const {
  if (impl_ == nullptr)
    return invalid("model is not loaded");
  return geneb_janusdna_pool(forward, attention_mask, output);
}

} // namespace evo::cpu
