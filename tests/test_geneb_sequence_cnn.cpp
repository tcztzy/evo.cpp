// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_sequence_cnn.hpp"
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

evo::cpu::GenebSequenceCnnTopology tiny_topology(
    const evo::cpu::GenebSequenceCnnVariant variant) {
  evo::cpu::GenebSequenceCnnTopology topology;
  topology.variant = variant;
  topology.input_length = 8U;
  topology.stem_width = 4U;
  topology.tower_widths = {4U};
  topology.width = 4U;
  topology.output_width = 6U;
  topology.layers = 1U;
  topology.attention_heads = 2U;
  topology.key_dimension = 2U;
  topology.value_dimension = 2U;
  topology.relative_feature_width = 6U;
  topology.target_length = 2U;
  topology.batch_norm_epsilon = 1.0e-5F;
  topology.gelu_sigmoid_scale = 1.702F;
  topology.use_tf_gamma = false;
  topology.weight_dtype = evo::TensorDType::kF32;
  if (variant == evo::cpu::GenebSequenceCnnVariant::kSpace) {
    topology.species_num_experts = 4U;
    topology.top_k = 3U;
    topology.gate_negative_slope = 0.01F;
    topology.species = "human";
  }
  return topology;
}

float fixture_scalar(
    const evo::cpu::GenebSequenceCnnTensorRequirement &requirement,
    const std::size_t tensor_index, const std::size_t element_index) {
  const int integer =
      static_cast<int>(((tensor_index + 7U) * 11U +
                        (element_index + 3U) * 5U) %
                       29U) -
      14;
  const float base = static_cast<float>(integer) / 100.0F;
  if (ends_with(requirement.name, "running_var"))
    return 1.0F + base * 0.1F;
  if (ends_with(requirement.name, "running_mean"))
    return base * 0.05F;
  if (requirement.shape.size() == 1U &&
      ends_with(requirement.name, ".weight"))
    return 1.0F + base * 0.1F;
  if (ends_with(requirement.name, ".bias"))
    return base * 0.08F;
  if (requirement.name.find("species_embedding") != std::string::npos)
    return base * 0.3F;
  if (requirement.name.find("gates.") != std::string::npos)
    return base * 0.4F;
  return base * 0.2F;
}

struct TinyWeights final {
  std::vector<std::vector<float>> storage;
  std::vector<evo::cpu::GenebSequenceCnnNamedTensorView> views;
};

TinyWeights tiny_weights(
    const evo::cpu::GenebSequenceCnnTopology &topology) {
  std::vector<evo::cpu::GenebSequenceCnnTensorRequirement> requirements;
  check(evo::cpu::canonical_geneb_sequence_cnn_tensors(topology,
                                                        &requirements)
            .ok(),
        "tiny sequence-CNN manifest is valid");
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
          fixture_scalar(requirement, tensor_index, element);
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
    if (index != 0U)
      std::cout << ',';
    std::cout << std::setprecision(9) << values[index];
  }
  std::cout << ']';
}

void dump_variant(const evo::cpu::GenebSequenceCnnVariant variant) {
  const auto topology = tiny_topology(variant);
  auto weights = tiny_weights(topology);
  evo::cpu::GenebSequenceCnnModel model;
  check(model.load(topology, weights.views).ok(),
        "tiny sequence-CNN loads");
  evo::cpu::GenebSequenceCnnForwardResult result;
  const auto status = model.forward("acguNXTTAA", &result);
  check(status.ok(), "tiny sequence-CNN forward: " + status.message());
  std::vector<float> pooled;
  check(model.pool(result, &pooled).ok(), "tiny sequence-CNN pools");
  std::cout << "{\"rows\":" << result.rows << ",\"width\":"
            << result.width << ",\"final_hidden\":";
  json_vector(result.final_hidden);
  std::cout << ",\"pooled\":";
  json_vector(pooled);
  std::cout << '}';
}

void dump_tiny() {
  std::cout << "{\"enformer\":";
  dump_variant(evo::cpu::GenebSequenceCnnVariant::kEnformer);
  std::cout << ",\"space\":";
  dump_variant(evo::cpu::GenebSequenceCnnVariant::kSpace);
  std::cout << "}\n";
}

evo::cpu::GenebSequenceCnnTopology production_topology(
    const evo::cpu::GenebSequenceCnnVariant variant) {
  auto topology = tiny_topology(variant);
  topology.input_length =
      variant == evo::cpu::GenebSequenceCnnVariant::kEnformer ? 196608U
                                                              : 131072U;
  topology.stem_width = 768U;
  topology.tower_widths = {768U, 896U, 1024U, 1152U, 1280U, 1536U};
  topology.width = 1536U;
  topology.output_width = 3072U;
  topology.layers = 11U;
  topology.attention_heads = 8U;
  topology.key_dimension = 64U;
  topology.value_dimension = 192U;
  topology.relative_feature_width = 192U;
  topology.target_length = 896U;
  return topology;
}

void check_manifest(const evo::cpu::GenebSequenceCnnTopology &topology,
                    const std::size_t expected_count,
                    const std::size_t expected_bytes) {
  std::vector<evo::cpu::GenebSequenceCnnTensorRequirement> requirements;
  check(evo::cpu::canonical_geneb_sequence_cnn_tensors(topology,
                                                        &requirements)
            .ok(),
        "production sequence-CNN topology is accepted");
  std::size_t bytes = 0U;
  for (const auto &requirement : requirements) {
    std::size_t elements = 1U;
    for (const std::size_t dimension : requirement.shape)
      elements *= dimension;
    bytes += elements * sizeof(float);
  }
  check(requirements.size() == expected_count && bytes == expected_bytes,
        "production canonical sequence-CNN manifest is exact");
}

void check_contracts() {
  for (const auto variant : {evo::cpu::GenebSequenceCnnVariant::kEnformer,
                             evo::cpu::GenebSequenceCnnVariant::kSpace}) {
    auto topology = tiny_topology(variant);
    check(evo::cpu::validate_geneb_sequence_cnn_topology(topology).ok(),
          "tiny sequence-CNN topology is accepted");
    auto drift = topology;
    drift.relative_feature_width = 8U;
    check(!evo::cpu::validate_geneb_sequence_cnn_topology(drift).ok(),
          "relative-feature drift is rejected");
    drift = topology;
    drift.target_length = 3U;
    check(!evo::cpu::validate_geneb_sequence_cnn_topology(drift).ok(),
          "oversized target is rejected");
    auto weights = tiny_weights(topology);
    evo::cpu::GenebSequenceCnnModel model;
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
  }
  check_manifest(
      production_topology(evo::cpu::GenebSequenceCnnVariant::kEnformer),
      269U, 919500800U);
  check_manifest(
      production_topology(evo::cpu::GenebSequenceCnnVariant::kSpace), 315U,
      2166370656U);
}

void run_long_context(const evo::cpu::GenebSequenceCnnVariant variant) {
  auto topology = tiny_topology(variant);
  topology.input_length =
      variant == evo::cpu::GenebSequenceCnnVariant::kEnformer ? 196608U
                                                              : 131072U;
  topology.stem_width = 2U;
  topology.width = 2U;
  topology.output_width = 2U;
  topology.attention_heads = 1U;
  topology.key_dimension = 2U;
  topology.value_dimension = 2U;
  topology.target_length = 4U;
  topology.tower_widths.assign(
      variant == evo::cpu::GenebSequenceCnnVariant::kEnformer ? 14U : 13U,
      2U);
  auto weights = tiny_weights(topology);
  evo::cpu::GenebSequenceCnnModel model;
  check(model.load(topology, weights.views).ok(),
        "advertised-context tiny sequence-CNN loads");
  std::string sequence(topology.input_length + 1U, 'A');
  evo::cpu::GenebSequenceCnnForwardResult result;
  const auto status = model.forward(sequence, &result);
  check(status.ok(), "advertised-context forward succeeds: " +
                         status.message());
  check(result.rows == topology.target_length &&
            result.width == topology.output_width,
        "advertised context is not silently capped");
}

void check_complexity() {
  run_long_context(evo::cpu::GenebSequenceCnnVariant::kEnformer);
  run_long_context(evo::cpu::GenebSequenceCnnVariant::kSpace);
}

void load_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  check(status.ok(), "ModelFile opens sequence-CNN artifact: " +
                         status.message());
  evo::cpu::GenebSequenceCnnModel model;
  status = model.load(artifact);
  check(status.ok(), "typed sequence-CNN loads artifact: " +
                         status.message());
  check(model.topology() != nullptr && model.topology()->layers == 11U &&
            model.topology()->width == 1536U,
        "typed sequence-CNN artifact topology is exact");
  evo::cpu::Model public_model;
  status = public_model.load(artifact);
  check(status.ok(), "public CPU sequence-CNN adapter loads artifact: " +
                         status.message());
  check(public_model.config().width == 3072U &&
            public_model.config().layers == 0U,
        "public CPU sequence-CNN config is exact (width=" +
            std::to_string(public_model.config().width) + ", layers=" +
            std::to_string(public_model.config().layers) + ")");
  std::cout << "loaded sequence-cnn tensors=" << artifact.tensors().size()
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
