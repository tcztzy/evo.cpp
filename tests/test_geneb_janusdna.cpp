// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_janusdna.hpp"
#include "evo/cpu/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using evo::cpu::GenebJanusDnaNamedTensorView;
using evo::cpu::GenebJanusDnaTensorRequirement;
using evo::cpu::GenebJanusDnaTopology;

void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void check(const bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

bool close(const float left, const float right,
           const float tolerance = 2.0e-5F) {
  return std::isfinite(left) && std::isfinite(right) &&
         std::abs(left - right) <= tolerance;
}

bool ends_with(const std::string_view value,
               const std::string_view suffix) noexcept {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

GenebJanusDnaTopology tiny_topology(const bool with_attention) {
  GenebJanusDnaTopology topology;
  topology.variant =
      with_attention ? evo::cpu::GenebJanusDnaVariant::kWithMiddleAttention
                     : evo::cpu::GenebJanusDnaVariant::kWithoutMiddleAttention;
  topology.vocabulary_size = 16U;
  topology.tokenizer_vocabulary_size = 12U;
  topology.width = 4U;
  topology.layers = 2U;
  topology.attention_heads = 2U;
  topology.head_dimension = 2U;
  topology.flex_attention_head_dimension = 4U;
  topology.inner_width = 8U;
  topology.state_width = 2U;
  topology.convolution_width = 2U;
  topology.time_step_rank = 2U;
  topology.mlp_width = 8U;
  topology.experts = 4U;
  topology.experts_per_token = 2U;
  topology.maximum_sequence_length = 8U;
  topology.middle_attention_layer = 0U;
  topology.pad_token_id = 4U;
  topology.norm_epsilon = 1.0e-6F;
  topology.weight_dtype = evo::TensorDType::kF32;
  return topology;
}

GenebJanusDnaTopology production_topology(const bool with_attention) {
  auto topology = tiny_topology(with_attention);
  topology.width = 72U;
  topology.layers = 8U;
  topology.attention_heads = 4U;
  topology.head_dimension = 18U;
  topology.flex_attention_head_dimension = 32U;
  topology.inner_width = 144U;
  topology.state_width = 16U;
  topology.convolution_width = 4U;
  topology.time_step_rank = 5U;
  topology.mlp_width = 288U;
  topology.experts = 16U;
  topology.maximum_sequence_length = 1024U;
  topology.middle_attention_layer = 4U;
  return topology;
}

float fixture_scalar(const std::string &name, const std::size_t tensor_index,
                     const std::size_t element_index) {
  const int integer =
      static_cast<int>(
          ((tensor_index + 5U) * 19U + (element_index + 3U) * 11U) % 47U) -
      23;
  const float value = static_cast<float>(integer) / 41.0F;
  if (name.find("layernorm.weight") != std::string::npos)
    return 1.0F + value * 0.025F;
  if (ends_with(name, "A_log"))
    return -0.55F + value * 0.04F;
  if (ends_with(name, "dt_proj.bias"))
    return -1.0F + value * 0.07F;
  if (ends_with(name, ".D"))
    return 0.45F + value * 0.05F;
  if (name == "embed_tokens.weight")
    return value * 0.18F;
  if (ends_with(name, "router.weight"))
    return value * 0.09F;
  return value * 0.075F;
}

struct TinyWeights final {
  std::vector<std::vector<float>> storage;
  std::vector<GenebJanusDnaNamedTensorView> views;
};

TinyWeights tiny_weights(const GenebJanusDnaTopology &topology) {
  std::vector<GenebJanusDnaTensorRequirement> requirements;
  check(
      evo::cpu::canonical_geneb_janusdna_tensors(topology, &requirements).ok(),
      "tiny topology has a canonical manifest");
  TinyWeights result;
  result.storage.reserve(requirements.size());
  result.views.reserve(requirements.size());
  for (std::size_t tensor_index = 0U; tensor_index < requirements.size();
       ++tensor_index) {
    const auto &requirement = requirements[tensor_index];
    std::size_t elements = 1U;
    for (const auto dimension : requirement.shape)
      elements *= dimension;
    std::vector<float> values(elements);
    for (std::size_t element = 0U; element < elements; ++element)
      values[element] = fixture_scalar(requirement.name, tensor_index, element);
    result.storage.push_back(std::move(values));
    const auto &stored = result.storage.back();
    result.views.push_back(
        {requirement.name,
         {reinterpret_cast<const std::uint8_t *>(stored.data()),
          stored.size() * sizeof(float), evo::TensorDType::kF32,
          requirement.shape}});
  }
  return result;
}

void check_production_manifests() {
  for (const bool with_attention : {true, false}) {
    const auto topology = production_topology(with_attention);
    check(evo::cpu::validate_geneb_janusdna_topology(topology).ok(),
          "production topology is accepted");
    std::vector<GenebJanusDnaTensorRequirement> requirements;
    check(evo::cpu::canonical_geneb_janusdna_tensors(topology, &requirements)
              .ok(),
          "production manifest is accepted");
    check(requirements.size() == (with_attention ? 621U : 635U),
          "production canonical tensor count is exact");
    for (std::size_t layer = 0U; layer < topology.layers; ++layer) {
      if (with_attention && layer == topology.middle_attention_layer)
        continue;
      const std::string reverse_prefix =
          "layers." + std::to_string(layer) + ".mamba_module.mamba_rev.";
      for (const auto &requirement : requirements) {
        check(requirement.name != reverse_prefix + "in_proj.weight" &&
                  requirement.name != reverse_prefix + "out_proj.weight",
              "canonical artifact folds only reverse tied projections");
      }
    }
  }
}

void check_topology_and_tensor_gates() {
  auto topology = tiny_topology(true);
  check(evo::cpu::validate_geneb_janusdna_topology(topology).ok(),
        "tiny with-attention topology is accepted");
  check(evo::cpu::validate_geneb_janusdna_topology(tiny_topology(false)).ok(),
        "tiny without-attention topology is accepted");
  auto drift = topology;
  drift.flex_attention_head_dimension = drift.head_dimension;
  check(!evo::cpu::validate_geneb_janusdna_topology(drift).ok(),
        "logical-head Flex scale drift is rejected");
  drift = topology;
  drift.norm_epsilon = 1.0e-5F;
  check(!evo::cpu::validate_geneb_janusdna_topology(drift).ok(),
        "RMS epsilon drift is rejected");
  drift = topology;
  drift.experts_per_token = 1U;
  check(!evo::cpu::validate_geneb_janusdna_topology(drift).ok(),
        "top-2 drift is rejected");
  drift = topology;
  drift.tokenizer_vocabulary_size = drift.vocabulary_size;
  check(!evo::cpu::validate_geneb_janusdna_topology(drift).ok(),
        "tokenizer/model vocabulary collapse is rejected");

  auto weights = tiny_weights(topology);
  evo::cpu::GenebJanusDnaModel model;
  auto missing = weights.views;
  missing.pop_back();
  check(!model.load(topology, missing).ok(), "missing tensor is rejected");
  auto extra = weights.views;
  extra.push_back(extra.front());
  extra.back().name = "unexpected.weight";
  check(!model.load(topology, extra).ok(), "extra tensor is rejected");
  auto duplicated = weights.views;
  duplicated.push_back(duplicated.front());
  check(!model.load(topology, duplicated).ok(), "duplicate tensor is rejected");
  auto wrong_shape = weights.views;
  wrong_shape.front().tensor.shape.front() += 1U;
  check(!model.load(topology, wrong_shape).ok(),
        "wrong tensor shape is rejected");
  auto wrong_dtype = weights.views;
  wrong_dtype.front().tensor.dtype = evo::TensorDType::kBF16;
  check(!model.load(topology, wrong_dtype).ok(),
        "wrong tensor dtype is rejected");
}

void check_forward(const bool with_attention) {
  const auto topology = tiny_topology(with_attention);
  auto weights = tiny_weights(topology);
  evo::cpu::GenebJanusDnaModel model;
  check(model.load(topology, weights.views).ok(), "tiny Janus model loads");
  const std::vector<evo::TokenId> tokens{7U, 8U, 11U, 4U};
  const std::vector<std::uint8_t> mask{1U, 1U, 1U, 0U};
  evo::cpu::GenebJanusDnaForwardResult result;
  const auto status =
      model.forward(tokens, mask, {0U, topology.layers}, &result);
  check(status.ok(), "tiny forward succeeds: " + status.message());
  check(result.rows == 4U && result.width == 4U &&
            result.final_hidden.size() == 16U && result.pooled.size() == 4U &&
            result.captures.size() == 2U &&
            result.captures.front().layer == 0U &&
            result.captures.back().layer == topology.layers,
        "tiny forward/capture geometry is exact");
  for (const float value : result.final_hidden)
    check(std::isfinite(value), "tiny final hidden is finite");
  for (std::size_t column = 0U; column < topology.width; ++column) {
    float expected = 0.0F;
    for (std::size_t row = 0U; row < 3U; ++row)
      expected += result.final_hidden[row * topology.width + column];
    check(close(result.pooled[column], expected / 3.0F),
          "pool is the original attention-mask mean");
  }
  check(!model.forward(tokens, {1U, 0U, 1U, 0U}, {}, &result).ok(),
        "non-prefix mask is rejected");
  check(!model.forward(tokens, {1U, 1U, 1U, 1U}, {}, &result).ok(),
        "PAD/mask disagreement is rejected");
  auto bad_tokens = tokens;
  bad_tokens.front() = 12U;
  check(!model.forward(bad_tokens, mask, {}, &result).ok(),
        "IDs in padded model-only vocabulary are rejected");
  check(!model.forward(tokens, mask, {1U}, &result).ok(),
        "unexposed internal doubled-sequence layer is rejected");
}

void check_top2_tie_order() {
  const auto topology = tiny_topology(false);
  auto baseline_weights = tiny_weights(topology);
  for (std::size_t index = 0U; index < baseline_weights.views.size(); ++index) {
    if (ends_with(baseline_weights.views[index].name, "router.weight"))
      std::fill(baseline_weights.storage[index].begin(),
                baseline_weights.storage[index].end(), 0.0F);
  }
  const std::vector<evo::TokenId> tokens{7U, 8U, 11U, 4U};
  const std::vector<std::uint8_t> mask{1U, 1U, 1U, 0U};
  evo::cpu::GenebJanusDnaModel baseline_model;
  check(baseline_model.load(topology, baseline_weights.views).ok(),
        "tie-order baseline loads");
  evo::cpu::GenebJanusDnaForwardResult baseline;
  check(baseline_model.forward(tokens, mask, {}, &baseline).ok(),
        "tie-order baseline runs");

  auto unselected = tiny_weights(topology);
  for (std::size_t index = 0U; index < unselected.views.size(); ++index) {
    const std::string &name = unselected.views[index].name;
    if (ends_with(name, "router.weight")) {
      std::fill(unselected.storage[index].begin(),
                unselected.storage[index].end(), 0.0F);
    } else if (name.find(".experts.2.") != std::string::npos ||
               name.find(".experts.3.") != std::string::npos) {
      std::fill(unselected.storage[index].begin(),
                unselected.storage[index].end(), 9.0F);
    }
  }
  evo::cpu::GenebJanusDnaModel unselected_model;
  check(unselected_model.load(topology, unselected.views).ok(),
        "unselected-expert mutation loads");
  evo::cpu::GenebJanusDnaForwardResult unchanged;
  check(unselected_model.forward(tokens, mask, {}, &unchanged).ok() &&
            unchanged.final_hidden == baseline.final_hidden,
        "all-equal top-2 deterministically selects experts 0 then 1");
}

void check_1024_pool_contract() {
  const auto topology = production_topology(true);
  evo::cpu::GenebJanusDnaForwardResult forward;
  forward.rows = 1024U;
  forward.width = topology.width;
  forward.final_hidden.resize(forward.rows * forward.width);
  for (std::size_t row = 0U; row < forward.rows; ++row) {
    for (std::size_t column = 0U; column < forward.width; ++column)
      forward.final_hidden[row * forward.width + column] =
          static_cast<float>((row + column) % 17U) * 0.125F;
  }
  std::vector<std::uint8_t> mask(1024U, 0U);
  std::fill_n(mask.begin(), 17, 1U);
  std::vector<float> pooled;
  check(evo::cpu::geneb_janusdna_pool(forward, mask, &pooled).ok() &&
            pooled.size() == topology.width,
        "1024-row fixed-pad pooling contract succeeds");
  for (std::size_t column = 0U; column < topology.width; ++column) {
    float expected = 0.0F;
    for (std::size_t row = 0U; row < 17U; ++row)
      expected += forward.final_hidden[row * topology.width + column];
    check(close(pooled[column], expected / 17.0F),
          "1024-row pool excludes padded rows");
  }
}

void dump_vector(const std::vector<float> &values) {
  std::cout << '[';
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::cout.precision(std::numeric_limits<float>::max_digits10);
    std::cout << values[index];
  }
  std::cout << ']';
}

int dump_tiny(const std::string_view variant) {
  const bool with_attention = variant == "with-middle-attention";
  if (!with_attention && variant != "without-middle-attention") {
    std::cerr << "unknown tiny Janus variant\n";
    return 2;
  }
  const auto topology = tiny_topology(with_attention);
  auto weights = tiny_weights(topology);
  evo::cpu::GenebJanusDnaModel model;
  auto status = model.load(topology, weights.views);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebJanusDnaForwardResult result;
  status = model.forward({7U, 8U, 11U, 4U}, {1U, 1U, 1U, 0U},
                         {0U, topology.layers}, &result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::cout << "{\"variant\":\"" << variant << "\",\"rows\":" << result.rows
            << ",\"width\":" << result.width << ",\"captures\":[";
  for (std::size_t index = 0U; index < result.captures.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::cout << "{\"layer\":" << result.captures[index].layer
              << ",\"values\":";
    dump_vector(result.captures[index].values);
    std::cout << '}';
  }
  std::cout << "],\"final_hidden\":";
  dump_vector(result.final_hidden);
  std::cout << ",\"pooled\":";
  dump_vector(result.pooled);
  std::cout << "}\n";
  return 0;
}

int run_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebJanusDnaModel model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  const auto *const topology = model.topology();
  std::vector<GenebJanusDnaTensorRequirement> requirements;
  status = evo::cpu::canonical_geneb_janusdna_tensors(*topology, &requirements);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  const bool with_attention =
      topology->variant == evo::cpu::GenebJanusDnaVariant::kWithMiddleAttention;
  std::cout << "{\"variant\":\""
            << (with_attention ? "with-middle-attention"
                               : "without-middle-attention")
            << "\",\"model_vocab\":" << topology->vocabulary_size
            << ",\"tokenizer_vocab\":" << topology->tokenizer_vocabulary_size
            << ",\"max_seqlen\":" << topology->maximum_sequence_length
            << ",\"tensors\":" << requirements.size() << "}\n";
  return 0;
}

int verify_cpu_adapter(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << "Janus adapter artifact open failed: " << status.message()
              << '\n';
    return 2;
  }
  evo::cpu::GenebJanusDnaModel direct;
  status = direct.load(artifact);
  if (!status.ok() || direct.topology() == nullptr) {
    std::cerr << (status.ok() ? "Janus typed topology is missing"
                              : status.message())
              << '\n';
    return 2;
  }
  const auto *const topology = direct.topology();
  const bool with_attention =
      topology->variant == evo::cpu::GenebJanusDnaVariant::kWithMiddleAttention;

  evo::cpu::Model model;
  status = model.load(artifact, true);
  if (!status.ok()) {
    std::cerr << "Janus public CPU Model load failed: " << status.message()
              << '\n';
    return 2;
  }
  const auto &config = model.config();
  if (config.architecture != evo::cpu::kGenebJanusDnaArchitecture ||
      config.artifact_profile != evo::cpu::kGenebJanusDnaArtifactProfile ||
      config.implementation !=
          evo::ArchitectureImplementation::kGenebJanusDnaEncoder ||
      config.tokenizer != evo::ArchitectureTokenizer::kArtifact ||
      config.vocab_size != 16U || topology->tokenizer_vocabulary_size != 12U ||
      config.width != 72U || config.layers != 8U ||
      config.max_seqlen != 1024U) {
    std::cerr << "Janus public CPU Model contract differs\n";
    return 2;
  }
  std::vector<evo::TokenId> encoded;
  status = model.encode("acgtnx", &encoded);
  if (!status.ok() ||
      encoded != std::vector<evo::TokenId>{7U, 8U, 9U, 10U, 11U, 6U}) {
    std::cerr << (status.ok() ? "Janus tokenizer IDs differ" : status.message())
              << '\n';
    return 2;
  }

  evo::GenebPreparedEmbeddingInput prepared;
  status = model.prepare_geneb_embedding_input("acgtn", &prepared);
  const std::vector<evo::TokenId> expected_prefix{7U, 8U, 9U, 10U, 11U};
  if (!status.ok() || prepared.tokens.size() != 1024U ||
      prepared.attention_mask.size() != 1024U ||
      !std::equal(expected_prefix.begin(), expected_prefix.end(),
                  prepared.tokens.begin()) ||
      !std::all_of(prepared.tokens.begin() + 5, prepared.tokens.end(),
                   [](const evo::TokenId token) { return token == 4U; }) ||
      !std::all_of(prepared.attention_mask.begin(),
                   prepared.attention_mask.begin() + 5,
                   [](const std::uint8_t value) { return value == 1U; }) ||
      !std::all_of(prepared.attention_mask.begin() + 5,
                   prepared.attention_mask.end(),
                   [](const std::uint8_t value) { return value == 0U; }) ||
      prepared.token_plan.original_token_count != 5U ||
      prepared.token_plan.effective_token_count != 1024U ||
      prepared.token_plan.retained_token_count != 5U ||
      prepared.token_plan.exceeds_context) {
    std::cerr << (status.ok() ? "Janus fixed-pad preparation differs"
                              : status.message())
              << " tokens=" << prepared.tokens.size()
              << " mask=" << prepared.attention_mask.size()
              << " original=" << prepared.token_plan.original_token_count
              << " effective=" << prepared.token_plan.effective_token_count
              << " retained=" << prepared.token_plan.retained_token_count
              << " exceeds=" << prepared.token_plan.exceeds_context << '\n';
    return 2;
  }

  evo::cpu::GenebJanusDnaForwardResult expected;
  status = direct.forward(prepared.tokens, prepared.attention_mask,
                          {topology->layers}, &expected);
  if (!status.ok() || expected.captures.size() != 1U ||
      expected.captures.front().layer != topology->layers) {
    std::cerr << (status.ok() ? "Janus typed final capture is incomplete"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::cpu::Context context;
  status = context.initialize_shared(model, prepared.tokens.size());
  if (!status.ok()) {
    std::cerr << "Janus public CPU Context initialize failed: "
              << status.message() << '\n';
    return 2;
  }
  std::vector<float> actual_hidden;
  status =
      context.prefill_embedding_masked(prepared.tokens, prepared.attention_mask,
                                       topology->layers, &actual_hidden);
  if (!status.ok() || actual_hidden != expected.captures.front().values ||
      context.position() != 1024U || context.activation_capacity() != 1024U) {
    std::cerr << (status.ok()
                      ? "Janus CPU adapter hidden differs from typed runtime"
                      : status.message())
              << '\n';
    return 2;
  }
  const auto *const spec = model.geneb_embedding_spec();
  if (spec == nullptr || spec->runtime_id != config.model_id ||
      spec->reference.output_width != 72U ||
      spec->reference.hidden_tap !=
          "twice-post-final-rmsnorm-after-identity-final-mlp-residual" ||
      spec->reference.pooling != "attention-mask-mean" ||
      spec->reference.special_tokens != "none" ||
      spec->reference.mask_domain != "attention-mask") {
    std::cerr << "Janus GENEB embedding preset differs\n";
    return 2;
  }
  std::vector<float> actual_pooled;
  status = evo::pool_geneb_embedding(actual_hidden, prepared.tokens.size(),
                                     config.width, prepared.attention_mask,
                                     spec->reference.pooling, &actual_pooled);
  const bool pooled_matches =
      actual_pooled.size() == expected.pooled.size() &&
      std::equal(actual_pooled.begin(), actual_pooled.end(),
                 expected.pooled.begin(),
                 [](const float left, const float right) {
                   return close(left, right, 1.0e-4F);
                 });
  if (!status.ok() || !pooled_matches) {
    std::cerr << (status.ok()
                      ? "Janus public pooling differs from typed runtime"
                      : status.message())
              << '\n';
    return 2;
  }

  evo::GenebPreparedEmbeddingInput truncated;
  status =
      model.prepare_geneb_embedding_input(std::string(1025U, 'A'), &truncated);
  if (!status.ok() || truncated.tokens.size() != 1024U ||
      truncated.attention_mask != std::vector<std::uint8_t>(1024U, 1U) ||
      !truncated.token_plan.exceeds_context ||
      truncated.token_plan.original_token_count != 1025U ||
      truncated.token_plan.retained_token_count != 1024U ||
      truncated.token_plan.effective_token_count != 1024U ||
      truncated.token_plan.truncated_left != 0U ||
      truncated.token_plan.truncated_right != 1U) {
    std::cerr << (status.ok() ? "Janus right-truncation metadata differs"
                              : status.message())
              << '\n';
    return 2;
  }

  std::cout << "{\"variant\":\""
            << (with_attention ? "with-middle-attention"
                               : "without-middle-attention")
            << "\",\"model_vocab\":16,\"tokenizer_vocab\":12,"
               "\"rows\":1024,\"width\":72,\"layer\":8,"
               "\"pooling\":\"attention-mask-mean\"}\n";
  return 0;
}

} // namespace

int main(const int argc, char **argv) {
  if (argc == 3 && std::string_view{argv[1]} == "--dump-tiny")
    return dump_tiny(argv[2]);
  if (argc == 3 && std::string_view{argv[1]} == "--verify-cpu-adapter")
    return verify_cpu_adapter(argv[2]);
  if (argc == 2)
    return run_artifact(argv[1]);
  if (argc != 1) {
    std::cerr << "usage: test_geneb_janusdna [artifact | --dump-tiny VARIANT "
                 "| --verify-cpu-adapter ARTIFACT]\n";
    return 2;
  }
  check_production_manifests();
  check_topology_and_tensor_gates();
  check_forward(true);
  check_forward(false);
  check_top2_tie_order();
  check_1024_pool_contract();
  std::cout << "GENEB JanusDNA tests passed\n";
  return 0;
}
