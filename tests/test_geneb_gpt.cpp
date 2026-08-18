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

#include "evo/cpu/geneb_dna_gpt.hpp"
#include "evo/cpu/geneb_gpt2.hpp"
#include "evo/cpu/model.hpp"
#include "evo/tokenizer.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

bool close(const float left, const float right,
           const float tolerance = 4.0e-5F) {
  return std::abs(left - right) <= tolerance;
}

bool finite(const std::vector<float> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](const float value) { return std::isfinite(value); });
}

bool ends_with(const std::string &value, const std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

std::size_t elements(const std::vector<std::size_t> &shape) {
  std::size_t result = 1;
  for (const auto dimension : shape)
    result *= dimension;
  return result;
}

struct StoredTensor final {
  std::string name;
  std::vector<std::uint8_t> bytes;
  std::vector<std::size_t> shape;
};

float fixture_value(const std::string &name, const std::size_t tensor_index,
                    const std::size_t element_index) {
  const auto integer =
      static_cast<int>(
          ((tensor_index + 3U) * 17U + (element_index + 7U) * 13U) % 47U) -
      23;
  float value = static_cast<float>(integer) / 37.0F;
  if (name.find("ln_") != std::string::npos ||
      name == "transformer.ln_f.weight") {
    if (ends_with(name, ".weight"))
      value = 1.0F + value * 0.04F;
    else
      value *= 0.025F;
  } else if (ends_with(name, ".bias")) {
    value *= 0.035F;
  } else if (name == "transformer.wte.weight" ||
             name == "transformer.wpe.weight") {
    value *= 0.18F;
  } else {
    value *= 0.11F;
  }
  return value;
}

template <typename Requirement>
std::vector<StoredTensor>
store_fixture(const std::vector<Requirement> &requirements) {
  std::vector<StoredTensor> storage(requirements.size());
  for (std::size_t tensor_index = 0; tensor_index < requirements.size();
       ++tensor_index) {
    auto &stored = storage[tensor_index];
    stored.name = requirements[tensor_index].name;
    stored.shape = requirements[tensor_index].shape;
    stored.bytes.resize(elements(stored.shape) * sizeof(float));
    for (std::size_t index = 0; index < elements(stored.shape); ++index) {
      const float value = fixture_value(stored.name, tensor_index, index);
      std::memcpy(stored.bytes.data() + index * sizeof(value), &value,
                  sizeof(value));
    }
  }
  return storage;
}

evo::cpu::GenebGpt2Topology gpt2_topology() {
  return {7, 4, 2, 2, 8, 8, 1.0e-5F};
}

evo::cpu::GenebDnaGptTopology dna_topology() {
  return {7, 4, 2, 2, 8, 8, 1.0e-5F};
}

struct Gpt2Fixture final {
  evo::cpu::GenebGpt2Topology topology;
  std::vector<StoredTensor> storage;
  std::vector<evo::cpu::GenebGpt2NamedTensorView> views;
};

struct DnaFixture final {
  evo::cpu::GenebDnaGptTopology topology;
  std::vector<StoredTensor> storage;
  std::vector<evo::cpu::GenebDnaGptNamedTensorView> views;
};

Gpt2Fixture make_gpt2_fixture() {
  Gpt2Fixture fixture;
  fixture.topology = gpt2_topology();
  std::vector<evo::cpu::GenebGpt2TensorRequirement> requirements;
  const auto status =
      evo::cpu::canonical_geneb_gpt2_tensors(fixture.topology, &requirements);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  fixture.storage = store_fixture(requirements);
  fixture.views.reserve(fixture.storage.size());
  for (const auto &stored : fixture.storage)
    fixture.views.push_back({stored.name,
                             {stored.bytes.data(), stored.bytes.size(),
                              evo::TensorDType::kF32, stored.shape}});
  return fixture;
}

DnaFixture make_dna_fixture() {
  DnaFixture fixture;
  fixture.topology = dna_topology();
  std::vector<evo::cpu::GenebDnaGptTensorRequirement> requirements;
  const auto status = evo::cpu::canonical_geneb_dna_gpt_tensors(
      fixture.topology, &requirements);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  fixture.storage = store_fixture(requirements);
  fixture.views.reserve(fixture.storage.size());
  for (const auto &stored : fixture.storage)
    fixture.views.push_back({stored.name,
                             {stored.bytes.data(), stored.bytes.size(),
                              evo::TensorDType::kF32, stored.shape}});
  return fixture;
}

evo::cpu::GenebGpt2ForwardResult run_gpt2(const Gpt2Fixture &fixture) {
  evo::cpu::GenebGpt2Model model;
  auto status = model.load(fixture.topology, fixture.views);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  evo::cpu::GenebGpt2ForwardResult result;
  status = model.forward({1, 4, 2, 6}, {1, 1, 1, 0}, {0, 1, 2}, &result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  return result;
}

evo::cpu::GenebDnaGptForwardResult run_dna(const DnaFixture &fixture) {
  evo::cpu::GenebDnaGptModel model;
  auto status = model.load(fixture.topology, fixture.views);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  evo::cpu::GenebDnaGptForwardResult result;
  status = model.forward({1, 4, 2, 6}, {1, 1, 1, 0}, {0, 1, 2}, &result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  return result;
}

void dump_vector(const std::vector<float> &values) {
  std::cout << '[' << std::setprecision(9);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

template <typename Result>
void dump_result(const std::string_view prefix, const Result &result,
                 bool *const first) {
  const auto emit = [&](const std::string &name,
                        const std::vector<float> &values) {
    if (!*first)
      std::cout << ',';
    *first = false;
    std::cout << '"' << prefix << '.' << name << "\":";
    dump_vector(values);
  };
  for (const auto &capture : result.captures)
    emit("capture" + std::to_string(capture.layer), capture.values);
  emit("final", result.final_hidden);
  emit("pooled", result.pooled);
}

int dump_json() {
  const auto gpt2 = run_gpt2(make_gpt2_fixture());
  const auto dna = run_dna(make_dna_fixture());
  bool first = true;
  std::cout << "{\"vectors\":{";
  dump_result("gpt2", gpt2, &first);
  dump_result("dna", dna, &first);
  std::cout << "}}\n";
  return 0;
}

int verify_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::vector<float> pooled;
  std::size_t rows = 0;
  std::size_t width = 0;
  std::string family;
  if (artifact.profile() == evo::cpu::kGenebGpt2ArtifactProfile) {
    evo::cpu::GenebGpt2Model model;
    status = model.load(artifact);
    evo::cpu::GenebGpt2ForwardResult result;
    if (status.ok())
      status = model.forward({1, 4, 2, 6}, {1, 1, 1, 0}, {}, &result);
    if (status.ok()) {
      rows = result.rows;
      width = result.width;
      pooled = std::move(result.pooled);
      family = "gpt2";
    }
  } else if (artifact.profile() == evo::cpu::kGenebDnaGptArtifactProfile) {
    evo::cpu::GenebDnaGptModel model;
    status = model.load(artifact);
    evo::cpu::GenebDnaGptForwardResult result;
    if (status.ok())
      status = model.forward({1, 4, 2, 6}, {1, 1, 1, 0}, {}, &result);
    if (status.ok()) {
      rows = result.rows;
      width = result.width;
      pooled = std::move(result.pooled);
      family = "dna";
    }
  } else {
    status = {evo::ErrorCode::kUnsupported,
              "artifact is not a GENEB GPT profile"};
  }
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  const auto &descriptor = artifact.tokenizer_asset_descriptor();
  if (!descriptor.has_value()) {
    std::cerr << "artifact lacks tokenizer descriptor\n";
    return 2;
  }
  std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
  status = evo::ArtifactTokenizer::Load(std::string{artifact.artifact_root()},
                                        *descriptor, &tokenizer);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::vector<evo::TokenId> tokens;
  status = tokenizer->encode("ACG", {}, &tokens);
  if (!status.ok() || tokens.empty()) {
    std::cerr << (status.ok() ? "tokenizer returned no IDs" : status.message())
              << '\n';
    return 2;
  }
  std::cout << "{\"family\":\"" << family << "\",\"rows\":" << rows
            << ",\"width\":" << width << ",\"token_count\":" << tokens.size()
            << ",\"pooled\":";
  dump_vector(pooled);
  std::cout << "}\n";
  return 0;
}

int verify_cpu_adapter(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << "GENEB GPT adapter artifact open failed: " << status.message()
              << '\n';
    return 2;
  }

  const bool is_gpt2 =
      artifact.profile() == evo::cpu::kGenebGpt2ArtifactProfile;
  const bool is_dna =
      artifact.profile() == evo::cpu::kGenebDnaGptArtifactProfile;
  if (!is_gpt2 && !is_dna) {
    std::cerr << "artifact is not a GENEB GPT profile\n";
    return 2;
  }
  const auto expected_architecture = is_gpt2
                                         ? evo::cpu::kGenebGpt2Architecture
                                         : evo::cpu::kGenebDnaGptArchitecture;
  const auto expected_profile = is_gpt2 ? evo::cpu::kGenebGpt2ArtifactProfile
                                        : evo::cpu::kGenebDnaGptArtifactProfile;
  const auto expected_abi = is_gpt2 ? evo::cpu::kGenebGpt2RuntimeAbi
                                    : evo::cpu::kGenebDnaGptRuntimeAbi;
  const auto expected_implementation =
      is_gpt2 ? evo::ArchitectureImplementation::kGenebGpt2Decoder
              : evo::ArchitectureImplementation::kGenebDnaGptDecoder;

  evo::cpu::Model model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << "GENEB GPT CPU Model load failed: " << status.message()
              << '\n';
    return 2;
  }
  const auto &config = model.config();
  const auto *const registered = evo::find_architecture(config.architecture);
  if (config.architecture != expected_architecture ||
      config.artifact_profile != expected_profile ||
      config.implementation != expected_implementation ||
      config.tokenizer != evo::ArchitectureTokenizer::kArtifact ||
      registered == nullptr || registered->id != expected_architecture ||
      registered->artifact_profile != expected_profile ||
      registered->runtime_abi != expected_abi ||
      registered->implementation != expected_implementation ||
      registered->tokenizer != evo::ArchitectureTokenizer::kArtifact ||
      (registered->backends & evo::kArchitectureBackendCpu) == 0U ||
      (registered->capabilities & evo::kArchitectureEmbed) == 0U) {
    std::cerr << "GENEB GPT CPU Model did not retain its registry contract\n";
    return 2;
  }

  const bool is_static_dna = is_dna && config.vocab_size == 15659U;
  const std::vector<evo::TokenId> expected_tokens =
      is_gpt2         ? std::vector<evo::TokenId>{6, 5}
      : is_static_dna ? std::vector<evo::TokenId>{21, 0}
                      : std::vector<evo::TokenId>{21, 106};
  std::vector<evo::TokenId> direct_tokens;
  status = model.encode("ACG", &direct_tokens);
  if (!status.ok() || direct_tokens != expected_tokens) {
    std::cerr << (status.ok() ? "GENEB GPT artifact tokenizer IDs differ"
                              : status.message())
              << '\n';
    return 2;
  }

  evo::GenebPreparedEmbeddingInput prepared;
  status = model.prepare_geneb_embedding_input("ACG", &prepared);
  const auto expected_policy = is_gpt2
                                   ? evo::GenebSpecialTokenPolicy::kNone
                                   : evo::GenebSpecialTokenPolicy::kPrefixOnly;
  if (!status.ok() || prepared.tokens != expected_tokens ||
      prepared.attention_mask != std::vector<std::uint8_t>{1, 1} ||
      prepared.transform.sequence != "ACG" ||
      prepared.transform.prefix != (is_gpt2 ? "" : "<R>") ||
      prepared.transform.special_token_policy != expected_policy ||
      prepared.token_plan.original_token_count != expected_tokens.size() ||
      prepared.token_plan.effective_token_count != expected_tokens.size()) {
    std::cerr << (status.ok() ? "GENEB GPT prepared input semantics differ"
                              : status.message())
              << '\n';
    return 2;
  }
  const auto *const geneb_spec = model.geneb_embedding_spec();
  if (geneb_spec == nullptr || geneb_spec->runtime_id != config.model_id ||
      geneb_spec->reference.output_width != config.width ||
      geneb_spec->normalized.output_width != config.width) {
    std::cerr << "GENEB GPT embedding preset metadata differs\n";
    return 2;
  }

  evo::cpu::Context context;
  status = context.initialize_shared(model, config.max_seqlen);
  if (!status.ok()) {
    std::cerr << "GENEB GPT CPU context failed: " << status.message() << '\n';
    return 2;
  }
  std::vector<float> adapter_hidden;
  status = context.prefill_embedding_masked(
      prepared.tokens, prepared.attention_mask, config.layers, &adapter_hidden);
  if (!status.ok() ||
      adapter_hidden.size() != prepared.tokens.size() * config.width ||
      !finite(adapter_hidden) || context.position() != prepared.tokens.size()) {
    std::cerr << (status.ok() ? "GENEB GPT CPU adapter shape/value differs"
                              : status.message())
              << '\n';
    return 2;
  }

  std::vector<float> direct_hidden;
  if (is_gpt2) {
    evo::cpu::GenebGpt2Model direct;
    evo::cpu::GenebGpt2ForwardResult result;
    status = direct.load(artifact);
    if (status.ok())
      status = direct.forward(prepared.tokens, prepared.attention_mask,
                              {config.layers}, &result);
    if (status.ok() && result.captures.size() == 1U)
      direct_hidden = std::move(result.captures.front().values);
  } else {
    evo::cpu::GenebDnaGptModel direct;
    evo::cpu::GenebDnaGptForwardResult result;
    status = direct.load(artifact);
    if (status.ok())
      status = direct.forward(prepared.tokens, prepared.attention_mask,
                              {config.layers}, &result);
    if (status.ok() && result.captures.size() == 1U)
      direct_hidden = std::move(result.captures.front().values);
  }
  if (!status.ok() || direct_hidden != adapter_hidden) {
    std::cerr << (status.ok()
                      ? "GENEB GPT CPU adapter differs from typed runtime"
                      : status.message())
              << '\n';
    return 2;
  }

  std::cout << "{\"family\":\"" << (is_gpt2 ? "gpt2" : "dna")
            << "\",\"tokens\":[" << expected_tokens[0] << ','
            << expected_tokens[1] << "],\"rows\":" << prepared.tokens.size()
            << ",\"width\":" << config.width << "}\n";
  return 0;
}

template <typename Result>
void check_pool(const Result &result, const std::string_view family) {
  check(result.rows == 4U && result.width == 4U &&
            result.final_hidden.size() == 16U && result.pooled.size() == 4U &&
            result.captures.size() == 3U,
        std::string{family} + " forward shapes are exact");
  check(result.captures[0].layer == 0U && result.captures[1].layer == 1U &&
            result.captures[2].layer == 2U,
        std::string{family} + " capture order/last tap are exact");
  check(finite(result.final_hidden) && finite(result.pooled),
        std::string{family} + " result is finite");
  for (std::size_t column = 0; column < result.width; ++column) {
    float sum = 0.0F;
    for (std::size_t row = 0; row < 3U; ++row)
      sum += result.final_hidden[row * result.width + column];
    const float expected = sum / 3.0F;
    check(expected == result.pooled[column],
          std::string{family} + " pool uses exact direct F32 division");
  }
}

void test_topologies_and_manifests() {
  auto gpt2 = gpt2_topology();
  auto dna = dna_topology();
  check(evo::cpu::validate_geneb_gpt2_topology(gpt2).ok(),
        "GPT-2 tiny topology is accepted");
  check(evo::cpu::validate_geneb_dna_gpt_topology(dna).ok(),
        "DNA-GPT tiny topology is accepted");
  std::vector<evo::cpu::GenebGpt2TensorRequirement> gpt2_requirements;
  std::vector<evo::cpu::GenebDnaGptTensorRequirement> dna_requirements;
  check(evo::cpu::canonical_geneb_gpt2_tensors(gpt2, &gpt2_requirements).ok() &&
            gpt2_requirements.size() == 28U,
        "GPT-2 exact tensor set includes all affine biases");
  check(
      evo::cpu::canonical_geneb_dna_gpt_tensors(dna, &dna_requirements).ok() &&
          dna_requirements.size() == 15U,
      "DNA-GPT exact backbone set is bias-free");
  const auto gpt2_qkv = std::find_if(
      gpt2_requirements.begin(), gpt2_requirements.end(), [](const auto &item) {
        return item.name == "transformer.h.0.attn.c_attn.weight";
      });
  const auto dna_qkv = std::find_if(
      dna_requirements.begin(), dna_requirements.end(), [](const auto &item) {
        return item.name == "transformer.h.0.attn.c_attn.weight";
      });
  check(gpt2_qkv != gpt2_requirements.end() &&
            gpt2_qkv->shape == std::vector<std::size_t>({4, 12}),
        "GPT-2 Conv1D QKV stays [in,out]");
  check(dna_qkv != dna_requirements.end() &&
            dna_qkv->shape == std::vector<std::size_t>({12, 4}),
        "DNA-GPT Linear QKV stays [out,in]");
  gpt2.width = 5;
  check(!evo::cpu::validate_geneb_gpt2_topology(gpt2).ok(),
        "GPT-2 head divisibility is strict");
  dna.norm_epsilon = 0.0F;
  check(!evo::cpu::validate_geneb_dna_gpt_topology(dna).ok(),
        "DNA-GPT zero epsilon is rejected");
}

void test_gpt2_runtime() {
  auto fixture = make_gpt2_fixture();
  evo::cpu::GenebGpt2Model model;
  auto status = model.load(fixture.topology, fixture.views);
  check(status.ok(), "GPT-2 exact typed tensor set loads");
  check(model.topology() != nullptr && model.topology()->vocabulary_size ==
                                           fixture.topology.vocabulary_size,
        "GPT-2 topology is exposed");
  check(std::string_view{model.kernel_name()} == "geneb-gpt2-reference-f32",
        "GPT-2 portable kernel identity is explicit");
  evo::cpu::GenebGpt2ForwardResult padded;
  status = model.forward({1, 4, 2, 6}, {1, 1, 1, 0}, {0, 1, 2}, &padded);
  check(status.ok(), "GPT-2 padded forward succeeds");
  check_pool(padded, "GPT-2");
  evo::cpu::GenebGpt2ForwardResult unpadded;
  status = model.forward({1, 4, 2}, {1, 1, 1}, {}, &unpadded);
  check(status.ok(), "GPT-2 unpadded forward succeeds");
  for (std::size_t index = 0; index < unpadded.final_hidden.size(); ++index)
    check(close(unpadded.final_hidden[index], padded.final_hidden[index]),
          "GPT-2 right padding preserves valid rows");
  auto missing = fixture.views;
  missing.pop_back();
  check(!model.load(fixture.topology, missing).ok(),
        "GPT-2 missing tensor is rejected");
  auto duplicate = fixture.views;
  duplicate.push_back(duplicate.front());
  check(!model.load(fixture.topology, duplicate).ok(),
        "GPT-2 duplicate tensor is rejected");
  auto wrong_shape = fixture.views;
  wrong_shape.front().tensor.shape.front() += 1U;
  check(!model.load(fixture.topology, wrong_shape).ok(),
        "GPT-2 wrong tensor shape is rejected");
}

void test_dna_runtime() {
  auto fixture = make_dna_fixture();
  evo::cpu::GenebDnaGptModel model;
  auto status = model.load(fixture.topology, fixture.views);
  check(status.ok(), "DNA-GPT exact typed tensor set loads");
  check(std::string_view{model.kernel_name()} == "geneb-dna-gpt-reference-f32",
        "DNA-GPT portable kernel identity is explicit");
  evo::cpu::GenebDnaGptForwardResult padded;
  status = model.forward({1, 4, 2, 6}, {1, 1, 1, 0}, {0, 1, 2}, &padded);
  check(status.ok(), "DNA-GPT padded forward succeeds");
  check_pool(padded, "DNA-GPT");
  evo::cpu::GenebDnaGptForwardResult unpadded;
  status = model.forward({1, 4, 2}, {1, 1, 1}, {}, &unpadded);
  check(status.ok(), "DNA-GPT unpadded forward succeeds");
  for (std::size_t index = 0; index < unpadded.final_hidden.size(); ++index)
    check(close(unpadded.final_hidden[index], padded.final_hidden[index]),
          "DNA-GPT causal right padding preserves valid rows");
  evo::cpu::GenebDnaGptForwardResult all_nonpad;
  status = model.forward({1, 4, 2, 6}, {1, 1, 1, 1}, {}, &all_nonpad);
  check(status.ok(), "DNA-GPT all-nonpad forward succeeds");
  for (std::size_t column = 0; column < padded.width; ++column)
    check(close(all_nonpad.final_hidden[3U * padded.width + column],
                padded.final_hidden[3U * padded.width + column]),
          "DNA-GPT attention deliberately ignores the pooling mask");
  evo::cpu::GenebDnaGptForwardResult ignored;
  check(!model.forward({1, 4}, {1, 0, 0}, {}, &ignored).ok(),
        "DNA-GPT mask length mismatch is rejected");
  check(!model.forward({1, 4, 2}, {1, 0, 1}, {}, &ignored).ok(),
        "DNA-GPT non-right padding is rejected");
  check(!model.forward({1, 7}, {1, 1}, {}, &ignored).ok(),
        "DNA-GPT out-of-vocabulary token is rejected");
  check(!model.forward({1, 2}, {1, 1}, {2, 2}, &ignored).ok(),
        "DNA-GPT duplicate capture layer is rejected");
  auto extra = fixture.views;
  extra.push_back(extra.front());
  extra.back().name = "unexpected.weight";
  check(!model.load(fixture.topology, extra).ok(),
        "DNA-GPT extra tensor is rejected");
  auto wrong_dtype = fixture.views;
  wrong_dtype.front().tensor.dtype = evo::TensorDType::kBF16;
  check(!model.load(fixture.topology, wrong_dtype).ok(),
        "DNA-GPT runtime artifact must be canonical F32");
}

} // namespace

int main(const int argc, char **argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--dump-json")
    return dump_json();
  if (argc == 3 && std::string_view{argv[1]} == "--verify-artifact")
    return verify_artifact(argv[2]);
  if (argc == 3 && std::string_view{argv[1]} == "--verify-cpu-adapter")
    return verify_cpu_adapter(argv[2]);
  if (argc != 1) {
    std::cerr << "usage: test_geneb_gpt [--dump-json|--verify-artifact PATH|"
                 "--verify-cpu-adapter PATH]\n";
    return 2;
  }
  test_topologies_and_manifests();
  test_gpt2_runtime();
  test_dna_runtime();
  if (failures != 0) {
    std::cerr << failures << " GENEB GPT test(s) failed\n";
    return 1;
  }
  std::cout << "GENEB GPT tests passed\n";
  return 0;
}
