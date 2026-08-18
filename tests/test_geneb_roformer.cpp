// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_roformer.hpp"
#include "evo/cpu/model.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void check(const bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

bool ends_with(const std::string_view value,
               const std::string_view suffix) noexcept {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

evo::cpu::GenebRoformerTopology tiny_topology() {
  evo::cpu::GenebRoformerTopology topology;
  topology.vocabulary_size = 8U;
  topology.tokenizer_vocabulary_size = 8U;
  topology.width = 4U;
  topology.layers = 2U;
  topology.attention_heads = 2U;
  topology.head_dimension = 2U;
  topology.inner_width = 6U;
  topology.maximum_sequence_length = 5120U;
  topology.token_type_vocabulary_size = 2U;
  topology.pad_token_id = 3U;
  topology.cls_token_id = 1U;
  topology.sep_token_id = 2U;
  topology.layer_norm_epsilon = 1.0e-6F;
  topology.rope_base = 10000.0F;
  topology.rotary_value = false;
  topology.weight_dtype = evo::TensorDType::kF32;
  return topology;
}

float fixture_scalar(const std::string &name, const std::size_t tensor_index,
                     const std::size_t element_index) {
  const int integer =
      static_cast<int>(((tensor_index + 3U) * 13U +
                        (element_index + 5U) * 7U) %
                       31U) -
      15;
  const float base = static_cast<float>(integer) / 100.0F;
  if (name.find("LayerNorm.weight") != std::string::npos)
    return 1.0F + base * 0.1F;
  if (name == "roformer.embeddings.word_embeddings.weight")
    return base * 0.6F;
  if (name == "roformer.embeddings.token_type_embeddings.weight")
    return base * 0.2F;
  if (ends_with(name, ".bias"))
    return base * 0.15F;
  return base * 0.35F;
}

struct TinyWeights final {
  std::vector<std::vector<float>> storage;
  std::vector<evo::cpu::GenebRoformerNamedTensorView> views;
};

TinyWeights tiny_weights(const evo::cpu::GenebRoformerTopology &topology) {
  std::vector<evo::cpu::GenebRoformerTensorRequirement> requirements;
  check(evo::cpu::canonical_geneb_roformer_tensors(topology, &requirements)
            .ok(),
        "tiny RoFormer manifest is valid");
  TinyWeights result;
  result.storage.reserve(requirements.size());
  result.views.reserve(requirements.size());
  for (std::size_t tensor_index = 0; tensor_index < requirements.size();
       ++tensor_index) {
    const auto &requirement = requirements[tensor_index];
    std::size_t elements = 1U;
    for (const std::size_t dimension : requirement.shape)
      elements *= dimension;
    std::vector<float> values(elements);
    for (std::size_t element = 0; element < elements; ++element)
      values[element] =
          fixture_scalar(requirement.name, tensor_index, element);
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

void json_vector(const std::vector<float> &values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    std::cout << std::setprecision(9) << values[index];
  }
  std::cout << ']';
}

void dump_tiny() {
  const auto topology = tiny_topology();
  auto weights = tiny_weights(topology);
  evo::cpu::GenebRoformerModel model;
  check(model.load(topology, weights.views).ok(), "tiny RoFormer loads");
  const std::vector<evo::TokenId> tokens{1U, 2U, 5U, 6U, 3U};
  const std::vector<std::uint8_t> mask{1U, 1U, 1U, 1U, 0U};
  evo::cpu::GenebRoformerForwardResult result;
  const auto status = model.forward(tokens, mask, {0U, 1U, 2U}, &result);
  check(status.ok(), "tiny RoFormer forward: " + status.message());
  std::vector<float> pooled;
  check(model.pool(result, mask, &pooled).ok(), "tiny RoFormer pools");
  evo::cpu::GenebRoformerForwardResult payload;
  check(model.forward_payload({5U, 6U}, {}, &payload).ok(),
        "manual CLS-SEP-payload path succeeds");
  std::cout << "{\"rows\":" << result.rows << ",\"width\":"
            << result.width << ",\"captures\":[";
  for (std::size_t index = 0; index < result.captures.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    std::cout << "{\"layer\":" << result.captures[index].layer
              << ",\"values\":";
    json_vector(result.captures[index].values);
    std::cout << '}';
  }
  std::cout << "],\"final_hidden\":";
  json_vector(result.final_hidden);
  std::cout << ",\"pooled\":";
  json_vector(pooled);
  std::cout << ",\"payload_first_tokens\":[1,2,5,6]}\n";
}

void check_contracts() {
  auto topology = tiny_topology();
  check(evo::cpu::validate_geneb_roformer_topology(topology).ok(),
        "tiny topology is accepted");
  auto drift = topology;
  drift.rotary_value = true;
  check(!evo::cpu::validate_geneb_roformer_topology(drift).ok(),
        "rotary-value drift is rejected");
  drift = topology;
  drift.head_dimension = 3U;
  check(!evo::cpu::validate_geneb_roformer_topology(drift).ok(),
        "odd adjacent-pair head dimension is rejected");
  auto weights = tiny_weights(topology);
  evo::cpu::GenebRoformerModel model;
  auto missing = weights.views;
  missing.pop_back();
  check(!model.load(topology, missing).ok(), "missing tensor is rejected");
  auto extra = weights.views;
  extra.push_back(extra.front());
  extra.back().name = "unexpected.weight";
  check(!model.load(topology, extra).ok(), "extra tensor is rejected");
  auto duplicate = weights.views;
  duplicate.push_back(duplicate.front());
  check(!model.load(topology, duplicate).ok(),
        "duplicate tensor is rejected");
  auto wrong_shape = weights.views;
  wrong_shape.front().tensor.shape.front() += 1U;
  check(!model.load(topology, wrong_shape).ok(),
        "wrong tensor shape is rejected");
  auto wrong_dtype = weights.views;
  wrong_dtype.front().tensor.dtype = evo::TensorDType::kBF16;
  check(!model.load(topology, wrong_dtype).ok(),
        "wrong tensor dtype is rejected");

  evo::cpu::GenebRoformerTopology production = topology;
  production.vocabulary_size = 4096U;
  production.tokenizer_vocabulary_size = 4096U;
  production.width = 768U;
  production.layers = 12U;
  production.attention_heads = 12U;
  production.head_dimension = 64U;
  production.inner_width = 3072U;
  production.layer_norm_epsilon = 1.0e-12F;
  std::vector<evo::cpu::GenebRoformerTensorRequirement> requirements;
  check(evo::cpu::canonical_geneb_roformer_tensors(production, &requirements)
            .ok(),
        "production topology is accepted");
  std::size_t bytes = 0U;
  for (const auto &requirement : requirements) {
    std::size_t elements = 1U;
    for (const std::size_t dimension : requirement.shape)
      elements *= dimension;
    bytes += elements * sizeof(float);
  }
  check(requirements.size() == 196U && bytes == 352813056U,
        "production canonical manifest is exact");
}

void check_complexity() {
  auto topology = tiny_topology();
  topology.width = 2U;
  topology.layers = 1U;
  topology.attention_heads = 1U;
  topology.head_dimension = 2U;
  topology.inner_width = 2U;
  auto weights = tiny_weights(topology);
  evo::cpu::GenebRoformerModel model;
  check(model.load(topology, weights.views).ok(),
        "4097-token topology loads");
  std::vector<evo::TokenId> tokens(4097U, 5U);
  std::vector<std::uint8_t> mask(tokens.size(), 1U);
  evo::cpu::GenebRoformerForwardResult result;
  const auto status = model.forward(tokens, mask, {}, &result);
  check(status.ok(), "4097-token online attention succeeds: " +
                         status.message());
  check(result.final_hidden.size() == tokens.size() * topology.width,
        "4097-token result has no hidden 4096 cap");
}

void load_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  check(status.ok(), "ModelFile opens RoFormer artifact: " + status.message());
  evo::cpu::GenebRoformerModel model;
  status = model.load(artifact);
  check(status.ok(), "typed RoFormer loads artifact: " + status.message());
  check(model.topology() != nullptr && model.topology()->width == 768U &&
            model.topology()->layers == 12U,
        "typed RoFormer artifact topology is exact");
  evo::cpu::Model public_model;
  status = public_model.load(artifact);
  check(status.ok(), "public CPU RoFormer adapter loads artifact: " +
                         status.message());
  check(public_model.config().width == 768U &&
            public_model.config().layers == 12U,
        "public CPU RoFormer config is exact");
  std::cout << "loaded geneb-deepgene tensors=" << artifact.tensors().size()
            << '\n';
}

} // namespace

int main(const int argc, char **argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--dump-tiny") {
    dump_tiny();
    return 0;
  }
  if (argc == 2 && std::string_view{argv[1]} == "--complexity") {
    check_complexity();
    return 0;
  }
  if (argc == 3 && std::string_view{argv[1]} == "--load-artifact") {
    load_artifact(argv[2]);
    return 0;
  }
  check_contracts();
  dump_tiny();
  return 0;
}
