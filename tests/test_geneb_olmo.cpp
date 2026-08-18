// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "evo/cpu/geneb_olmo.hpp"
#include "evo/cpu/model.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

bool close(const float left, const float right,
           const float tolerance = 2.0e-5F) {
  return std::abs(left - right) <= tolerance;
}

bool finite(const std::vector<float> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](const float value) { return std::isfinite(value); });
}

std::size_t elements(const std::vector<std::size_t> &shape) {
  std::size_t result = 1;
  for (const auto dimension : shape)
    result *= dimension;
  return result;
}

evo::cpu::GenebOlmoTopology
fixture_topology(const evo::cpu::GenebOlmoNormType norm_type) {
  evo::cpu::GenebOlmoTopology topology;
  topology.vocabulary_size = 7;
  topology.width = 4;
  topology.layers = 2;
  topology.heads = 2;
  topology.fused_mlp_width = 12;
  topology.maximum_sequence_length = 8;
  topology.norm_epsilon = 1.0e-5F;
  topology.rope_theta = 10000.0F;
  topology.norm_type = norm_type;
  return topology;
}

float fixture_value(const std::string &name, const std::size_t tensor_index,
                    const std::size_t element_index) {
  const auto integer =
      static_cast<int>(
          ((tensor_index + 2U) * 19U + (element_index + 5U) * 11U) % 43U) -
      21;
  float value = static_cast<float>(integer) / 31.0F;
  if (name.find("norm.weight") != std::string::npos ||
      name == "model.transformer.ln_f.weight")
    value = 1.0F + value * 0.04F;
  else if (name == "model.transformer.wte.weight")
    value *= 0.3F;
  else
    value *= 0.12F;
  return value;
}

struct StoredTensor final {
  std::string name;
  std::vector<std::uint8_t> bytes;
  std::vector<std::size_t> shape;
};

struct Fixture final {
  evo::cpu::GenebOlmoTopology topology;
  std::vector<StoredTensor> storage;
  std::vector<evo::cpu::GenebOlmoNamedTensorView> views;
};

void build_fixture(const evo::cpu::GenebOlmoNormType norm_type,
                   Fixture *const output) {
  output->topology = fixture_topology(norm_type);
  std::vector<evo::cpu::GenebOlmoTensorRequirement> requirements;
  const auto status =
      evo::cpu::canonical_geneb_olmo_tensors(output->topology, &requirements);
  if (!status.ok()) {
    std::cerr << "cannot compile fixture: " << status.message() << '\n';
    std::abort();
  }
  output->storage.clear();
  output->storage.resize(requirements.size());
  for (std::size_t tensor_index = 0; tensor_index < requirements.size();
       ++tensor_index) {
    auto &stored = output->storage[tensor_index];
    stored.name = requirements[tensor_index].name;
    stored.shape = requirements[tensor_index].shape;
    stored.bytes.resize(elements(stored.shape) * sizeof(float));
    for (std::size_t index = 0; index < elements(stored.shape); ++index) {
      const float value = fixture_value(stored.name, tensor_index, index);
      std::memcpy(stored.bytes.data() + index * sizeof(value), &value,
                  sizeof(value));
    }
  }
  output->views.clear();
  output->views.reserve(output->storage.size());
  for (const auto &stored : output->storage)
    output->views.push_back({stored.name,
                             {stored.bytes.data(), stored.bytes.size(),
                              evo::TensorDType::kF32, stored.shape}});
}

evo::cpu::GenebOlmoForwardResult run_fixture(const Fixture &fixture) {
  evo::cpu::GenebOlmoModel model;
  auto status = model.load(fixture.topology, fixture.views);
  if (!status.ok()) {
    std::cerr << "cannot load fixture: " << status.message() << '\n';
    std::abort();
  }
  evo::cpu::GenebOlmoForwardResult result;
  status = model.forward({1, 4, 2, 6}, {1, 1, 1, 0}, {0, 1, 2}, &result);
  if (!status.ok()) {
    std::cerr << "cannot run fixture: " << status.message() << '\n';
    std::abort();
  }
  return result;
}

void dump_vector(const std::vector<float> &values) {
  std::cout << '[' << std::setprecision(9);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

void dump_result(const std::string_view name,
                 const evo::cpu::GenebOlmoForwardResult &result,
                 bool *const first) {
  const auto emit = [&](const std::string &suffix,
                        const std::vector<float> &values) {
    if (!*first)
      std::cout << ',';
    *first = false;
    std::cout << '\"' << name << '.' << suffix << "\":";
    dump_vector(values);
  };
  for (const auto &capture : result.captures)
    emit("capture" + std::to_string(capture.layer), capture.values);
  emit("final", result.final_hidden);
  emit("pooled", result.pooled);
}

int dump_fixture_json() {
  Fixture layer_norm;
  Fixture rms_norm;
  build_fixture(evo::cpu::GenebOlmoNormType::kLayerNormNoAffine, &layer_norm);
  build_fixture(evo::cpu::GenebOlmoNormType::kRmsNormAffine, &rms_norm);
  const auto layer_result = run_fixture(layer_norm);
  const auto rms_result = run_fixture(rms_norm);
  bool first = true;
  std::cout << "{\"vectors\":{";
  dump_result("layernorm", layer_result, &first);
  dump_result("rmsnorm", rms_result, &first);
  std::cout << "}}\n";
  return 0;
}

int dump_exact_layer_norm_bits() {
  constexpr std::size_t kRows = 2U;
  constexpr std::size_t kWidth = 2048U;
  std::vector<float> input(kRows * kWidth, 0.0F);
  for (std::size_t index = 0; index < input.size(); ++index) {
    const int integer = static_cast<int>(((index + 3U) * 37U) % 257U) - 128;
    input[index] = static_cast<float>(integer) / 31.0F;
  }
  std::vector<float> output;
  const auto status = evo::cpu::geneb_olmo_layer_norm_no_affine(
      input, kRows, kWidth, 1.0e-5F,
      evo::cpu::GenebOlmoLayerNormKernel::kTorch212AppleArm64ExactV1, &output);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return status.code() == evo::ErrorCode::kUnsupported ? 77 : 2;
  }
  std::cout << "{\"bits\":[";
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &output[index], sizeof(bits));
    std::cout << bits;
  }
  std::cout << "]}\n";
  return 0;
}

int verify_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebOlmoModel model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebOlmoForwardResult result;
  status = model.forward({1, 4, 2, 6}, {1, 1, 1, 0}, {0, 1, 2}, &result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  bool first = true;
  std::cout << "{\"rows\":" << result.rows << ",\"width\":" << result.width
            << ",\"vectors\":{";
  dump_result("artifact", result, &first);
  std::cout << "}}\n";
  return 0;
}

int verify_cpu_adapter(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::Model model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  if (model.config().architecture != evo::cpu::kGenebOlmoArchitecture ||
      model.config().artifact_profile != evo::cpu::kGenebOlmoArtifactProfile ||
      model.config().implementation !=
          evo::ArchitectureImplementation::kGenebOlmoDecoder ||
      model.config().tokenizer != evo::ArchitectureTokenizer::kArtifact) {
    std::cerr << "CPU Model did not retain the OLMo registry contract\n";
    return 2;
  }
  const auto *const runtime_entry = artifact.find_metadata("geneb.runtime_id");
  if (runtime_entry == nullptr ||
      runtime_entry->type != evo::MetadataType::kString) {
    std::cerr << "OLMo artifact runtime identity is missing\n";
    return 2;
  }
  const std::string runtime_id{runtime_entry->value.begin(),
                               runtime_entry->value.end()};
  std::vector<evo::TokenId> expected_tokens;
  if (runtime_id == "geneb-omni-dna-1b" ||
      runtime_id == "geneb-omni-dna-300m") {
    expected_tokens = {1, 5, 3206, 2};
  } else if (runtime_id.rfind("geneb-olmo-tiny-", 0U) == 0U) {
    expected_tokens = {1, 4, 2};
  } else {
    std::cerr << "OLMo CPU adapter received an unknown runtime ID\n";
    return 2;
  }
  std::vector<evo::TokenId> tokens;
  status = model.encode("ACG", &tokens);
  if (!status.ok() || tokens != expected_tokens) {
    std::cerr << (status.ok() ? "CPU artifact tokenizer IDs differ"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::cpu::Context context;
  status = context.initialize_shared(model, model.config().max_seqlen);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::vector<float> embedding;
  status = context.prefill_embedding(tokens, model.config().layers, &embedding);
  if (!status.ok() ||
      embedding.size() != tokens.size() * model.config().width ||
      !finite(embedding)) {
    std::cerr << (status.ok() ? "CPU adapter embedding shape/value differs"
                              : status.message())
              << '\n';
    return 2;
  }
  std::cout << "{\"tokens\":[";
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::cout << tokens[index];
  }
  std::cout << "],\"rows\":" << tokens.size()
            << ",\"width\":" << model.config().width << "}\n";
  return 0;
}

void test_topology_and_manifests() {
  auto layer_norm =
      fixture_topology(evo::cpu::GenebOlmoNormType::kLayerNormNoAffine);
  auto rms_norm = fixture_topology(evo::cpu::GenebOlmoNormType::kRmsNormAffine);
  check(evo::cpu::validate_geneb_olmo_topology(layer_norm).ok(),
        "parameterless LayerNorm topology is accepted");
  check(evo::cpu::validate_geneb_olmo_topology(rms_norm).ok(),
        "affine RMSNorm topology is accepted");
  std::vector<evo::cpu::GenebOlmoTensorRequirement> requirements;
  check(
      evo::cpu::canonical_geneb_olmo_tensors(layer_norm, &requirements).ok() &&
          requirements.size() == 9,
      "LayerNorm topology has no norm tensors");
  check(std::none_of(requirements.begin(), requirements.end(),
                     [](const auto &item) {
                       return item.name.find("norm.weight") !=
                                  std::string::npos ||
                              item.name == "model.transformer.ln_f.weight";
                     }),
        "non-affine LayerNorm manifest excludes every scale");
  check(evo::cpu::canonical_geneb_olmo_tensors(rms_norm, &requirements).ok() &&
            requirements.size() == 14,
        "RMSNorm topology has per-block and final scales");
  check(requirements.back().name == "model.transformer.ln_f.weight",
        "RMSNorm manifest ends with the final scale");

  auto invalid = layer_norm;
  invalid.width = 5;
  check(!evo::cpu::validate_geneb_olmo_topology(invalid).ok(),
        "head divisibility is strict");
  invalid = layer_norm;
  invalid.fused_mlp_width = 11;
  check(!evo::cpu::validate_geneb_olmo_topology(invalid).ok(),
        "fused SwiGLU width must split evenly");
  invalid = layer_norm;
  invalid.norm_epsilon = 0.0F;
  check(!evo::cpu::validate_geneb_olmo_topology(invalid).ok(),
        "zero norm epsilon is rejected");

  auto exact = layer_norm;
  exact.vocabulary_size = 4096;
  exact.width = 2048;
  exact.layers = 16;
  exact.heads = 16;
  exact.fused_mlp_width = 16384;
  exact.maximum_sequence_length = 250;
  exact.layer_norm_kernel =
      evo::cpu::GenebOlmoLayerNormKernel::kTorch212AppleArm64ExactV1;
  check(evo::cpu::validate_geneb_olmo_topology(exact).ok(),
        "Omni-DNA-1B exact LayerNorm kernel accepts its closed topology");
  invalid = exact;
  invalid.width = 1024;
  check(!evo::cpu::validate_geneb_olmo_topology(invalid).ok(),
        "exact LayerNorm kernel rejects width drift");
  invalid = exact;
  invalid.norm_type = evo::cpu::GenebOlmoNormType::kRmsNormAffine;
  check(!evo::cpu::validate_geneb_olmo_topology(invalid).ok(),
        "exact LayerNorm kernel rejects norm topology drift");
  invalid = layer_norm;
  invalid.layer_norm_kernel =
      static_cast<evo::cpu::GenebOlmoLayerNormKernel>(255);
  check(!evo::cpu::validate_geneb_olmo_topology(invalid).ok(),
        "unknown LayerNorm kernel enum is rejected");
}

void test_load_and_forward(const evo::cpu::GenebOlmoNormType norm_type) {
  Fixture fixture;
  build_fixture(norm_type, &fixture);
  evo::cpu::GenebOlmoModel model;
  auto status = model.load(fixture.topology, fixture.views);
  check(status.ok(), "exact typed tensor set loads");
  check(model.topology() != nullptr && model.topology()->norm_type == norm_type,
        "loaded topology is exposed");
  check(std::string_view{model.linear_executor_name()} ==
            "geneb-olmo-reference-f32",
        "portable F32 executor is explicit");

  evo::cpu::GenebOlmoForwardResult padded;
  status = model.forward({1, 4, 2, 6}, {1, 1, 1, 0}, {2, 0, 1}, &padded);
  check(status.ok(), "right-padded forward succeeds");
  check(padded.rows == 4 && padded.width == 4 &&
            padded.final_hidden.size() == 16 && padded.pooled.size() == 4 &&
            padded.captures.size() == 3,
        "forward exposes all requested shapes");
  check(padded.captures[0].layer == 2 && padded.captures[1].layer == 0 &&
            padded.captures[2].layer == 1,
        "capture order follows the caller and final index is post-norm");
  check(finite(padded.final_hidden) && finite(padded.pooled),
        "forward values are finite");
  bool distinguishes_direct_division = false;
  for (std::size_t column = 0; column < padded.width; ++column) {
    float sum = 0.0F;
    for (std::size_t row = 0; row < 3; ++row)
      sum += padded.final_hidden[row * padded.width + column];
    const float expected = sum / 3.0F;
    const float reciprocal_product = sum * (1.0F / 3.0F);
    distinguishes_direct_division =
        distinguishes_direct_division || expected != reciprocal_product;
    check(expected == padded.pooled[column],
          "pool uses exact direct F32 division after ordered sum");
  }
  check(distinguishes_direct_division,
        "pool fixture distinguishes direct division from reciprocal multiply");

  evo::cpu::GenebOlmoForwardResult unpadded;
  status = model.forward({1, 4, 2}, {1, 1, 1}, {}, &unpadded);
  check(status.ok(), "unpadded forward succeeds");
  for (std::size_t index = 0; index < unpadded.final_hidden.size(); ++index)
    check(close(unpadded.final_hidden[index], padded.final_hidden[index]),
          "right padding does not contaminate valid hidden states");
  for (std::size_t index = 0; index < unpadded.pooled.size(); ++index)
    check(close(unpadded.pooled[index], padded.pooled[index]),
          "right padding does not contaminate pooled embedding");

  evo::cpu::GenebOlmoForwardResult ignored;
  check(!model.forward({1, 4}, {1, 0, 0}, {}, &ignored).ok(),
        "mask length mismatch is rejected");
  check(!model.forward({1, 4, 2}, {1, 0, 1}, {}, &ignored).ok(),
        "non-right-padded mask is rejected");
  check(!model.forward({1, 4}, {0, 0}, {}, &ignored).ok(),
        "all-padding mask is rejected");
  check(!model.forward({1, 7}, {1, 1}, {}, &ignored).ok(),
        "out-of-vocabulary token is rejected");
  check(!model.forward({1, 2}, {1, 1}, {1, 1}, &ignored).ok(),
        "duplicate capture layer is rejected");
  check(!model.forward({1, 2}, {1, 1}, {3}, &ignored).ok(),
        "out-of-range capture layer is rejected");

  auto missing = fixture.views;
  missing.pop_back();
  check(!model.load(fixture.topology, missing).ok(),
        "missing tensor is rejected");
  auto extra = fixture.views;
  extra.push_back(extra.front());
  extra.back().name = "extra.weight";
  check(!model.load(fixture.topology, extra).ok(), "extra tensor is rejected");
  auto duplicate = fixture.views;
  duplicate.push_back(duplicate.front());
  check(!model.load(fixture.topology, duplicate).ok(),
        "duplicate tensor is rejected");
  auto wrong_shape = fixture.views;
  wrong_shape.front().tensor.shape[0] += 1;
  check(!model.load(fixture.topology, wrong_shape).ok(),
        "wrong tensor shape is rejected");
  auto wrong_dtype = fixture.views;
  wrong_dtype.front().tensor.dtype = evo::TensorDType::kBF16;
  check(!model.load(fixture.topology, wrong_dtype).ok(),
        "non-F32 tensor is rejected");
}

} // namespace

int main(const int argc, char **argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--dump-json")
    return dump_fixture_json();
  if (argc == 2 && std::string_view{argv[1]} == "--dump-exact-layer-norm-bits")
    return dump_exact_layer_norm_bits();
  if (argc == 3 && std::string_view{argv[1]} == "--verify-artifact")
    return verify_artifact(argv[2]);
  if (argc == 3 && std::string_view{argv[1]} == "--verify-cpu-adapter")
    return verify_cpu_adapter(argv[2]);
  if (argc != 1) {
    std::cerr << "usage: test_geneb_olmo [--dump-json | "
                 "--dump-exact-layer-norm-bits | --verify-artifact PATH | "
                 "--verify-cpu-adapter PATH]\n";
    return 2;
  }
  test_topology_and_manifests();
  test_load_and_forward(evo::cpu::GenebOlmoNormType::kLayerNormNoAffine);
  test_load_and_forward(evo::cpu::GenebOlmoNormType::kRmsNormAffine);
  if (failures != 0) {
    std::cerr << failures << " GENEB OLMo test(s) failed\n";
    return 1;
  }
  std::cout << "GENEB OLMo tests passed\n";
  return 0;
}
