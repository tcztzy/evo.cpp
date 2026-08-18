// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "evo/cpu/geneb_evo1.hpp"
#include "evo/cpu/geneb_hyenadna.hpp"
#include "evo/cpu/model.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

std::size_t elements(const std::vector<std::size_t> &shape) {
  std::size_t result = 1U;
  for (const auto dimension : shape)
    result *= dimension;
  return result;
}

std::uint16_t bf16_bits(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7F800000U) != 0x7F800000U)
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

evo::cpu::GenebHyenaDnaTopology hyena_topology() {
  return {12U, 16U, 4U, 1U, 6U, 3U, 3U, 2U, 4U, 1.0e-5F};
}

evo::cpu::GenebEvo1Topology evo_topology() {
  evo::cpu::GenebEvo1Topology topology;
  topology.vocabulary_size = 8U;
  topology.width = 4U;
  topology.layers = 2U;
  topology.heads = 2U;
  topology.inner_width = 6U;
  topology.state_width = 2U;
  topology.short_filter_width = 2U;
  topology.maximum_sequence_length = 8U;
  topology.norm_epsilon = 1.0e-4F;
  topology.rope_theta = 10000.0F;
  topology.rope_scaling_factor = 2.0F;
  topology.attention_layers = {1U};
  return topology;
}

float fixture_value(const std::string &name, const std::size_t tensor_index,
                    const std::size_t element_index) {
  const auto integer =
      static_cast<int>(
          ((tensor_index + 5U) * 17U + (element_index + 3U) * 11U) % 37U) -
      18;
  float value = static_cast<float>(integer) / 41.0F;
  if (name == "hyena.backbone.embeddings.word_embeddings.weight") {
    const std::size_t row = element_index / 4U;
    const std::size_t column = element_index % 4U;
    if (row == 4U)
      return 0.35F - static_cast<float>(column) * 0.11F;
    return value * 0.24F;
  }
  if (name.find("hyena.backbone.layers") != std::string::npos) {
    if (name.find("norm") != std::string::npos)
      return name.find("bias") == std::string::npos ? 1.0F : 0.0F;
    if (name.find("mixer.in_proj.weight") != std::string::npos)
      return element_index % 4U == (element_index / 4U) % 4U ? 0.55F : 0.0F;
    if (name.find("mixer.in_proj.bias") != std::string::npos ||
        name.find("mixer.short_filter.bias") != std::string::npos ||
        name.find("mixer.out_proj.bias") != std::string::npos ||
        name.find("mlp.fc1.bias") != std::string::npos ||
        name.find("mlp.fc2.bias") != std::string::npos)
      return 0.0F;
    if (name.find("mixer.short_filter.weight") != std::string::npos)
      return element_index % 2U == 0U ? 0.35F : 0.65F;
    if (name.find("mixer.out_proj.weight") != std::string::npos)
      return element_index % 4U == element_index / 4U ? 0.45F : 0.0F;
    if (name.find("mixer.filter_fn.bias") != std::string::npos)
      return 0.4F;
  }
  if (name.find("norm") != std::string::npos &&
      name.find("bias") == std::string::npos)
    return 1.0F + value * 0.03F;
  if (name == "backbone.norm.scale" || name.find(".scale") != std::string::npos)
    return 1.0F + value * 0.03F;
  if (name.find("filter.poles") != std::string::npos)
    return element_index % 2U == 0U ? 0.72F + value * 0.02F
                                    : 0.04F + value * 0.01F;
  if (name.find("filter.residues") != std::string::npos)
    return value * 0.08F;
  if (name.find("modulation.deltas") != std::string::npos)
    return -1.1F + value * 0.1F;
  if (name.find("implicit_filter.1.freq") != std::string::npos)
    return 1.7F + value * 0.05F;
  if (name.find("pos_emb.t") != std::string::npos)
    return static_cast<float>(element_index) / 3.0F;
  if (name.find("pos_emb.z") != std::string::npos) {
    const std::size_t row = element_index / 3U;
    const std::size_t column = element_index % 3U;
    if (column == 0U)
      return static_cast<float>(row) / 3.0F;
    return column == 1U ? std::cos(static_cast<float>(row))
                        : std::sin(static_cast<float>(row));
  }
  if (name.find("embedding") != std::string::npos)
    return value * 0.24F;
  return value * 0.09F;
}

struct StoredTensor final {
  std::string name;
  evo::TensorDType dtype{evo::TensorDType::kF32};
  std::vector<std::size_t> shape;
  std::vector<std::uint8_t> bytes;
};

struct HyenaFixture final {
  evo::cpu::GenebHyenaDnaTopology topology;
  std::vector<StoredTensor> storage;
  std::vector<evo::cpu::GenebHyenaDnaNamedTensorView> views;
};

struct EvoFixture final {
  evo::cpu::GenebEvo1Topology topology;
  std::vector<StoredTensor> storage;
  std::vector<evo::cpu::GenebEvo1NamedTensorView> views;
};

void build_hyena_fixture(HyenaFixture *const output) {
  output->topology = hyena_topology();
  std::vector<evo::cpu::GenebHyenaDnaTensorRequirement> requirements;
  const auto status = evo::cpu::canonical_geneb_hyenadna_tensors(
      output->topology, &requirements);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  output->storage.resize(requirements.size());
  for (std::size_t tensor_index = 0U; tensor_index < requirements.size();
       ++tensor_index) {
    auto &stored = output->storage[tensor_index];
    stored.name = requirements[tensor_index].name;
    stored.dtype = requirements[tensor_index].dtype;
    stored.shape = requirements[tensor_index].shape;
    stored.bytes.resize(elements(stored.shape) * sizeof(float));
    for (std::size_t index = 0U; index < elements(stored.shape); ++index) {
      const float value = fixture_value(stored.name, tensor_index, index);
      std::memcpy(stored.bytes.data() + index * sizeof(float), &value,
                  sizeof(value));
    }
  }
  for (const auto &stored : output->storage)
    output->views.push_back({stored.name,
                             {stored.bytes.data(), stored.bytes.size(),
                              stored.dtype, stored.shape}});
}

void build_evo_fixture(EvoFixture *const output) {
  output->topology = evo_topology();
  std::vector<evo::cpu::GenebEvo1TensorRequirement> requirements;
  const auto status =
      evo::cpu::canonical_geneb_evo1_tensors(output->topology, &requirements);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  output->storage.resize(requirements.size());
  for (std::size_t tensor_index = 0U; tensor_index < requirements.size();
       ++tensor_index) {
    auto &stored = output->storage[tensor_index];
    stored.name = requirements[tensor_index].name;
    stored.dtype = requirements[tensor_index].dtype;
    stored.shape = requirements[tensor_index].shape;
    const std::size_t scalar = stored.dtype == evo::TensorDType::kF32 ? 4U : 2U;
    stored.bytes.resize(elements(stored.shape) * scalar);
    for (std::size_t index = 0U; index < elements(stored.shape); ++index) {
      const float value = fixture_value(stored.name, tensor_index, index);
      if (stored.dtype == evo::TensorDType::kF32) {
        std::memcpy(stored.bytes.data() + index * sizeof(float), &value,
                    sizeof(value));
      } else {
        const std::uint16_t bits = bf16_bits(value);
        std::memcpy(stored.bytes.data() + index * sizeof(bits), &bits,
                    sizeof(bits));
      }
    }
  }
  for (const auto &stored : output->storage)
    output->views.push_back({stored.name,
                             {stored.bytes.data(), stored.bytes.size(),
                              stored.dtype, stored.shape}});
}

void dump_vector(const std::vector<float> &values) {
  std::cout << '[' << std::setprecision(9);
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

int dump_vectors() {
  HyenaFixture hyena_fixture;
  EvoFixture evo_fixture;
  build_hyena_fixture(&hyena_fixture);
  build_evo_fixture(&evo_fixture);
  evo::cpu::GenebHyenaDnaModel hyena;
  evo::cpu::GenebEvo1Model evo;
  auto status = hyena.load(hyena_fixture.topology, hyena_fixture.views);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  status = evo.load(evo_fixture.topology, evo_fixture.views);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebHyenaDnaForwardResult reference;
  evo::cpu::GenebHyenaDnaForwardResult normalized;
  evo::cpu::GenebEvo1ForwardResult evo_result;
  status = hyena.forward({4U, 7U, 8U}, {0U, 1U, 1U}, {0U, 1U}, &reference);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  status = hyena.forward({7U, 8U}, {1U, 1U}, {0U, 1U}, &normalized);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  status = evo.forward({1U, 2U, 3U}, {0U, 1U, 2U}, &evo_result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::cout << "{\"vectors\":{";
  std::cout << "\"hyena.reference.final\":";
  dump_vector(reference.final_hidden);
  std::cout << ",\"hyena.reference.pooled\":";
  dump_vector(reference.pooled);
  std::cout << ",\"hyena.normalized.final\":";
  dump_vector(normalized.final_hidden);
  std::cout << ",\"hyena.normalized.pooled\":";
  dump_vector(normalized.pooled);
  std::cout << ",\"evo.final\":";
  dump_vector(evo_result.final_hidden);
  std::cout << ",\"evo.pooled\":";
  dump_vector(evo_result.pooled);
  std::cout << "},\"work\":{\"evo_attention_pairs\":"
            << evo_result.work.attention_score_pairs
            << ",\"evo_peak_logits\":" << evo_result.work.peak_attention_logits
            << "}}\n";
  return 0;
}

void test_fft_vectors() {
  std::vector<float> output;
  evo::cpu::GenebLongContextStats stats;
  auto status = evo::cpu::geneb_long_causal_convolution(
      {1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}, &output, &stats);
  check(status.ok(), "Bluestein causal convolution succeeds");
  const std::vector<float> expected{4.0F, 13.0F, 28.0F};
  check(output.size() == expected.size(), "convolution output size matches");
  for (std::size_t index = 0U; index < std::min(output.size(), expected.size());
       ++index)
    check(std::abs(output[index] - expected[index]) < 2.0e-4F,
          "causal convolution matches direct overlap vector");
  check(stats.maximum_fft_transform_length == 6U,
        "mathematical FFT length remains exactly 2L");
  check(stats.maximum_fft_radix2_length == 16U,
        "Bluestein scratch uses bounded radix-2 length");
  check(stats.fft_butterfly_count > 0U && stats.peak_fft_complex_values <= 96U,
        "FFT work and memory counters are bounded");

  status = evo::cpu::geneb_long_causal_convolution(
      {1.0F, -1.0F, 1.0F, -1.0F}, {1.0F, 1.0F, 1.0F, 1.0F}, &output, &stats);
  check(status.ok(), "radix-2 causal convolution succeeds");
  const std::vector<float> repeated{1.0F, 0.0F, 1.0F, 0.0F};
  for (std::size_t index = 0U; index < repeated.size(); ++index)
    check(std::abs(output[index] - repeated[index]) < 2.0e-4F,
          "repeated overlap convolution is deterministic");
}

void test_topologies_and_manifests() {
  auto hyena = hyena_topology();
  auto evo_value = evo_topology();
  check(evo::cpu::validate_geneb_hyenadna_topology(hyena).ok(),
        "tiny HyenaDNA topology is accepted");
  check(evo::cpu::validate_geneb_evo1_topology(evo_value).ok(),
        "tiny Evo-1 topology is accepted");
  evo_value.attention_layers = {1U, 1U};
  check(!evo::cpu::validate_geneb_evo1_topology(evo_value).ok(),
        "duplicate Evo attention layers are rejected");

  evo::cpu::GenebHyenaDnaTopology production_hyena{
      12U, 16U, 256U, 8U, 1024U, 64U, 5U, 3U, 1000002U, 1.0e-5F};
  std::vector<evo::cpu::GenebHyenaDnaTensorRequirement> hyena_requirements;
  auto status = evo::cpu::canonical_geneb_hyenadna_tensors(production_hyena,
                                                           &hyena_requirements);
  check(status.ok() && hyena_requirements.size() == 212U,
        "pinned HyenaDNA exact 212-tensor set is frozen");
  evo::cpu::GenebEvo1Topology production_evo;
  production_evo.vocabulary_size = 512U;
  production_evo.width = 4096U;
  production_evo.layers = 32U;
  production_evo.heads = 32U;
  production_evo.inner_width = 10928U;
  production_evo.state_width = 8U;
  production_evo.short_filter_width = 3U;
  production_evo.maximum_sequence_length = 131072U;
  production_evo.norm_epsilon = 1.0e-6F;
  production_evo.rope_theta = 10000.0F;
  production_evo.rope_scaling_factor = 16.0F;
  production_evo.attention_layers = {8U, 16U, 24U};
  std::vector<evo::cpu::GenebEvo1TensorRequirement> evo_requirements;
  status =
      evo::cpu::canonical_geneb_evo1_tensors(production_evo, &evo_requirements);
  check(status.ok() && evo_requirements.size() == 438U,
        "pinned Evo-1 exact 438-tensor mixed-dtype set is frozen");
  const auto f32_count = static_cast<std::size_t>(std::count_if(
      evo_requirements.begin(), evo_requirements.end(),
      [](const auto &item) { return item.dtype == evo::TensorDType::kF32; }));
  check(f32_count == 58U, "only 29 Evo poles/residues pairs remain F32");
}

void test_tiny_forward_contracts() {
  HyenaFixture hyena_fixture;
  EvoFixture evo_fixture;
  build_hyena_fixture(&hyena_fixture);
  build_evo_fixture(&evo_fixture);
  evo::cpu::GenebHyenaDnaModel hyena;
  evo::cpu::GenebEvo1Model evo;
  auto status = hyena.load(hyena_fixture.topology, hyena_fixture.views);
  check(status.ok(), "HyenaDNA typed fixture loads");
  status = evo.load(evo_fixture.topology, evo_fixture.views);
  check(status.ok(), "Evo-1 typed fixture loads");
  evo::cpu::GenebHyenaDnaForwardResult reference;
  evo::cpu::GenebHyenaDnaForwardResult normalized;
  status = hyena.forward({4U, 7U, 8U}, {0U, 1U, 1U}, {1U}, &reference);
  check(status.ok(), "HyenaDNA reference left-pad path runs");
  status = hyena.forward({7U, 8U}, {1U, 1U}, {1U}, &normalized);
  check(status.ok(), "HyenaDNA normalized no-pad path runs");
  check(reference.pooled != normalized.pooled,
        "reference left-pad contamination remains distinct from normalized");
  evo::cpu::GenebHyenaDnaForwardResult three_visible;
  status = hyena.forward({7U, 8U, 9U}, {1U, 1U, 1U}, {}, &three_visible);
  check(status.ok(), "three-visible-token HyenaDNA fixture runs");
  bool hyena_distinguishes_direct_division = false;
  for (std::size_t column = 0; status.ok() && column < three_visible.width;
       ++column) {
    float sum = 0.0F;
    for (std::size_t row = 0; row < 3U; ++row)
      sum += three_visible.final_hidden[row * three_visible.width + column];
    const float expected = sum / 3.0F;
    hyena_distinguishes_direct_division =
        hyena_distinguishes_direct_division || expected != sum * (1.0F / 3.0F);
    check(three_visible.pooled[column] == expected,
          "HyenaDNA private pool uses exact direct F32 division");
  }
  check(hyena_distinguishes_direct_division,
        "HyenaDNA fixture distinguishes division from reciprocal multiply");
  status = hyena.forward({7U, 4U}, {1U, 0U}, {}, &reference);
  check(!status.ok(), "right-padded Hyena mask is rejected");
  evo::cpu::GenebEvo1ForwardResult evo_result;
  status = evo.forward({1U, 2U, 3U}, {0U, 1U, 2U}, &evo_result);
  check(status.ok(), "Evo-1 modal+attention fixture runs");
  bool evo_distinguishes_direct_division = false;
  for (std::size_t column = 0; status.ok() && column < evo_result.width;
       ++column) {
    float sum = 0.0F;
    for (std::size_t row = 0; row < 3U; ++row)
      sum += evo_result.final_hidden[row * evo_result.width + column];
    const float expected = sum / 3.0F;
    evo_distinguishes_direct_division =
        evo_distinguishes_direct_division || expected != sum * (1.0F / 3.0F);
    check(evo_result.pooled[column] == expected,
          "Evo-1 private pool uses exact direct F32 division");
  }
  check(evo_distinguishes_direct_division,
        "Evo-1 fixture distinguishes division from reciprocal multiply");
  check(evo_result.work.attention_score_pairs == 12U &&
            evo_result.work.peak_attention_logits == 3U,
        "Evo attention compute and linear-memory counters differ explicitly");
  check(evo_result.work.maximum_fft_transform_length == 6U,
        "Evo modal Hyena uses exact 2L FFT");
}

int complexity_gate() {
  constexpr std::size_t length = 1000002U;
  std::vector<float> input(length, 0.0F);
  std::vector<float> kernel(length, 0.0F);
  input[0] = 1.0F;
  kernel[0] = 1.0F;
  std::vector<float> output;
  evo::cpu::GenebLongContextStats stats;
  const auto status =
      evo::cpu::geneb_long_causal_convolution(input, kernel, &output, &stats);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  check(output.size() == length && std::abs(output[0] - 1.0F) < 5.0e-3F,
        "one-million-token FFT returns a valid causal impulse");
  check(stats.maximum_fft_transform_length == length * 2U,
        "one-million-token gate keeps exact 2L transform");
  check(stats.maximum_fft_radix2_length <= length * 8U,
        "Bluestein radix scratch is linear in L");
  check(
      stats.peak_fft_complex_values ==
          length * 10U + stats.maximum_fft_radix2_length * 2U,
      "FFT peak resident counter includes outer arrays and Bluestein scratch");
  check(stats.peak_fft_complex_values <= length * 24U,
        "FFT peak resident complex storage is O(L)");
  const double nlogn =
      static_cast<double>(length) * std::log2(static_cast<double>(length));
  check(static_cast<double>(stats.fft_butterfly_count) <= 32.0 * nlogn,
        "FFT butterfly counter is O(L log L)");
  if (failures != 0)
    return 1;
  std::cout << "{\"length\":" << length
            << ",\"transform_length\":" << stats.maximum_fft_transform_length
            << ",\"radix2_length\":" << stats.maximum_fft_radix2_length
            << ",\"butterflies\":" << stats.fft_butterfly_count
            << ",\"peak_complex\":" << stats.peak_fft_complex_values << "}\n";
  return 0;
}

int verify_hyena_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebHyenaDnaModel model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebHyenaDnaForwardResult result;
  status = model.forward({4U, 7U, 8U}, {0U, 1U, 1U}, {model.topology()->layers},
                         &result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::cout << "{\"rows\":" << result.rows << ",\"width\":" << result.width
            << ",\"pooled\":";
  dump_vector(result.pooled);
  std::cout << "}\n";
  return 0;
}

int verify_evo_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebEvo1Model model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebEvo1ForwardResult result;
  status = model.forward({1U, 2U, 3U}, {model.topology()->layers}, &result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::cout << "{\"rows\":" << result.rows << ",\"width\":" << result.width
            << ",\"pooled\":";
  dump_vector(result.pooled);
  std::cout << "}\n";
  return 0;
}

bool finite_values(const std::vector<float> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](const float value) { return std::isfinite(value); });
}

int verify_hyena_cpu_adapter(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebHyenaDnaModel direct;
  status = direct.load(artifact);
  if (!status.ok() || direct.topology() == nullptr) {
    std::cerr << (status.ok() ? "HyenaDNA typed topology is missing"
                              : status.message())
              << '\n';
    return 2;
  }
  const auto *const topology = direct.topology();
  evo::cpu::Model model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  const auto &config = model.config();
  if (config.architecture != evo::cpu::kGenebHyenaDnaArchitecture ||
      config.artifact_profile != evo::cpu::kGenebHyenaDnaArtifactProfile ||
      config.implementation !=
          evo::ArchitectureImplementation::kGenebHyenaDnaDecoder ||
      config.tokenizer != evo::ArchitectureTokenizer::kArtifact ||
      config.vocab_size != topology->vocabulary_size ||
      config.width != topology->width || config.layers != topology->layers ||
      config.max_seqlen != topology->maximum_sequence_length) {
    std::cerr << "HyenaDNA public CPU Model contract differs\n";
    return 2;
  }

  std::vector<evo::TokenId> encoded;
  status = model.encode("AC", &encoded);
  if (!status.ok() || encoded != std::vector<evo::TokenId>{7U, 8U}) {
    std::cerr << (status.ok() ? "HyenaDNA artifact tokenizer IDs differ"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::GenebPreparedEmbeddingInput prepared;
  status = model.prepare_geneb_embedding_input("AC", &prepared);
  if (!status.ok() || prepared.tokens != encoded ||
      prepared.attention_mask != std::vector<std::uint8_t>{1U, 1U} ||
      prepared.transform.sequence != "AC") {
    std::cerr << (status.ok() ? "HyenaDNA GENEB preparation differs"
                              : status.message())
              << '\n';
    return 2;
  }
  std::vector<evo::TokenId> padded_tokens{4U};
  padded_tokens.insert(padded_tokens.end(), prepared.tokens.begin(),
                       prepared.tokens.end());
  const std::vector<std::uint8_t> padded_mask{0U, 1U, 1U};
  evo::cpu::GenebHyenaDnaForwardResult expected;
  status =
      direct.forward(padded_tokens, padded_mask, {topology->layers}, &expected);
  if (!status.ok() || expected.captures.size() != 1U) {
    std::cerr << (status.ok() ? "HyenaDNA typed capture is incomplete"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::cpu::Context context;
  status = context.initialize_shared(model, padded_tokens.size());
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::vector<float> actual_hidden;
  status = context.prefill_embedding_masked(padded_tokens, padded_mask,
                                            topology->layers, &actual_hidden);
  if (!status.ok() || actual_hidden != expected.captures.front().values ||
      !finite_values(actual_hidden) ||
      context.position() != padded_tokens.size() ||
      context.activation_capacity() != padded_tokens.size()) {
    std::cerr << (status.ok()
                      ? "HyenaDNA CPU adapter differs from typed runtime"
                      : status.message())
              << '\n';
    return 2;
  }
  const auto *const spec = model.geneb_embedding_spec();
  if (spec == nullptr || spec->runtime_id != config.model_id ||
      spec->reference.pooling != "attention-mask-mean" ||
      spec->reference.output_width != config.width) {
    std::cerr << "HyenaDNA GENEB preset differs\n";
    return 2;
  }
  std::vector<float> actual_pooled;
  status = evo::pool_geneb_embedding(actual_hidden, padded_tokens.size(),
                                     config.width, padded_mask,
                                     spec->reference.pooling, &actual_pooled);
  if (!status.ok() || actual_pooled != expected.pooled) {
    std::cerr << (status.ok() ? "HyenaDNA masked pooling differs"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::cpu::GenebHyenaDnaForwardResult normalized;
  status = direct.forward(prepared.tokens, prepared.attention_mask,
                          {topology->layers}, &normalized);
  if (!status.ok() || normalized.pooled == expected.pooled) {
    std::cerr << (status.ok() ? "HyenaDNA left-pad contamination was erased"
                              : status.message())
              << '\n';
    return 2;
  }
  std::cout << "{\"family\":\"hyena\",\"tokens\":[7,8],"
               "\"padded_tokens\":[4,7,8],\"mask\":[0,1,1],\"rows\":3,"
               "\"width\":"
            << config.width << ",\"pooling\":\"attention-mask-mean\"}\n";
  return 0;
}

int verify_evo_cpu_adapter(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebEvo1Model direct;
  status = direct.load(artifact);
  if (!status.ok() || direct.topology() == nullptr) {
    std::cerr << (status.ok() ? "Evo-1 typed topology is missing"
                              : status.message())
              << '\n';
    return 2;
  }
  const auto *const topology = direct.topology();
  evo::cpu::Model model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  const auto &config = model.config();
  if (config.architecture != evo::cpu::kGenebEvo1Architecture ||
      config.artifact_profile != evo::cpu::kGenebEvo1ArtifactProfile ||
      config.implementation !=
          evo::ArchitectureImplementation::kGenebStripedHyenaV1 ||
      config.tokenizer != evo::ArchitectureTokenizer::kArtifact ||
      config.vocab_size != topology->vocabulary_size ||
      config.width != topology->width || config.layers != topology->layers ||
      config.max_seqlen != topology->maximum_sequence_length) {
    std::cerr << "Evo-1 public CPU Model contract differs\n";
    return 2;
  }
  std::vector<evo::TokenId> encoded;
  status = model.encode("AN", &encoded);
  if (!status.ok() || encoded != std::vector<evo::TokenId>{7U, 6U}) {
    std::cerr << (status.ok() ? "Evo-1 artifact tokenizer IDs differ"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::GenebPreparedEmbeddingInput prepared;
  status = model.prepare_geneb_embedding_input("AN", &prepared);
  if (!status.ok() || prepared.tokens != encoded ||
      prepared.attention_mask != std::vector<std::uint8_t>{1U, 1U} ||
      prepared.transform.sequence != "AN") {
    std::cerr << (status.ok() ? "Evo-1 GENEB preparation differs"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::cpu::GenebEvo1ForwardResult expected;
  status = direct.forward(prepared.tokens, {topology->layers}, &expected);
  if (!status.ok() || expected.captures.size() != 1U) {
    std::cerr << (status.ok() ? "Evo-1 typed capture is incomplete"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::cpu::Context context;
  status = context.initialize_shared(model, prepared.tokens.size());
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::vector<float> actual_hidden;
  status =
      context.prefill_embedding_masked(prepared.tokens, prepared.attention_mask,
                                       topology->layers, &actual_hidden);
  if (!status.ok() || actual_hidden != expected.captures.front().values ||
      !finite_values(actual_hidden) ||
      context.position() != prepared.tokens.size() ||
      context.activation_capacity() != prepared.tokens.size()) {
    std::cerr << (status.ok() ? "Evo-1 CPU adapter differs from typed runtime"
                              : status.message())
              << '\n';
    return 2;
  }
  const auto *const spec = model.geneb_embedding_spec();
  if (spec == nullptr || spec->runtime_id != config.model_id ||
      spec->reference.pooling != "mean-first-record-only" ||
      spec->normalized.pooling != "per-record-mean" ||
      spec->reference.output_width != config.width) {
    std::cerr << "Evo-1 GENEB preset differs\n";
    return 2;
  }
  std::vector<float> actual_pooled;
  status = evo::pool_geneb_embedding(actual_hidden, prepared.tokens.size(),
                                     config.width, prepared.attention_mask,
                                     spec->reference.pooling, &actual_pooled);
  if (!status.ok() || actual_pooled != expected.pooled) {
    std::cerr << (status.ok() ? "Evo-1 all-token pooling differs"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::cpu::Context rejects_padding;
  status = rejects_padding.initialize_shared(model, prepared.tokens.size());
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::vector<float> ignored;
  if (rejects_padding
          .prefill_embedding_masked(prepared.tokens, {0U, 1U}, topology->layers,
                                    &ignored)
          .ok()) {
    std::cerr << "Evo-1 CPU adapter accepted a padded reference record\n";
    return 2;
  }
  std::cout << "{\"family\":\"evo\",\"tokens\":[7,6],\"mask\":[1,1],"
               "\"rows\":2,\"width\":"
            << config.width << ",\"pooling\":\"mean-first-record-only\"}\n";
  return 0;
}

int verify_cpu_adapter(const std::string_view family, const std::string &path) {
  if (family == "hyena")
    return verify_hyena_cpu_adapter(path);
  if (family == "evo")
    return verify_evo_cpu_adapter(path);
  std::cerr << "unknown GENEB long-Hyena CPU adapter family\n";
  return 2;
}

} // namespace

int main(const int argc, char **const argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--dump-vectors")
    return dump_vectors();
  if (argc == 2 && std::string_view{argv[1]} == "--complexity")
    return complexity_gate();
  if (argc == 3 && std::string_view{argv[1]} == "--verify-hyena-artifact")
    return verify_hyena_artifact(argv[2]);
  if (argc == 3 && std::string_view{argv[1]} == "--verify-evo-artifact")
    return verify_evo_artifact(argv[2]);
  if (argc == 4 && std::string_view{argv[1]} == "--verify-cpu-adapter")
    return verify_cpu_adapter(argv[2], argv[3]);
  if (argc != 1) {
    std::cerr << "usage: test_geneb_long_hyena [--dump-vectors|--complexity|"
                 "--verify-hyena-artifact PATH|--verify-evo-artifact PATH|"
                 "--verify-cpu-adapter {hyena|evo} PATH]\n";
    return 2;
  }
  test_fft_vectors();
  test_topologies_and_manifests();
  test_tiny_forward_contracts();
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "GENEB long Hyena tests passed\n";
  return 0;
}
