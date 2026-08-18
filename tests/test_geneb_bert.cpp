// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_bert.hpp"
#include "evo/cpu/model.hpp"
#include "evo/geneb_embedding.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using evo::cpu::GenebBertInputKind;
using evo::cpu::GenebBertMlpKind;
using evo::cpu::GenebBertNormPlacement;
using evo::cpu::GenebBertPooling;
using evo::cpu::GenebBertPositionEncoding;
using evo::cpu::GenebBertQkvLayout;

void require(const bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::uint32_t fnv1a(const std::string_view text) {
  std::uint32_t hash = 2166136261U;
  for (const char character : text) {
    hash ^= static_cast<std::uint8_t>(character);
    hash *= 16777619U;
  }
  return hash;
}

bool is_norm_weight(const std::string &name) {
  return name.find("LayerNorm.weight") != std::string::npos ||
         name.find("_ln.weight") != std::string::npos ||
         name.find("layernorm.weight") != std::string::npos;
}

bool is_norm_bias(const std::string &name) {
  return name.find("LayerNorm.bias") != std::string::npos ||
         name.find("_ln.bias") != std::string::npos ||
         name.find("layernorm.bias") != std::string::npos;
}

evo::cpu::GenebBertTopology base_topology() {
  evo::cpu::GenebBertTopology topology;
  topology.vocabulary_size = 9;
  topology.width = 4;
  topology.layers = 2;
  topology.attention_heads = 2;
  topology.head_dimension = 2;
  topology.inner_width = 6;
  topology.maximum_sequence_length = 8;
  topology.token_type_vocabulary_size = 2;
  topology.layer_norm_epsilon = 1.0e-5F;
  topology.rope_base = 0.0F;
  topology.position_encoding = GenebBertPositionEncoding::kAbsolute;
  topology.norm_placement = GenebBertNormPlacement::kPost;
  topology.mlp_kind = GenebBertMlpKind::kGelu;
  topology.qkv_layout = GenebBertQkvLayout::kSeparate;
  topology.input_kind = GenebBertInputKind::kTokenIds;
  topology.pooling = GenebBertPooling::kAttentionMaskMean;
  topology.attention_bias = true;
  topology.mlp_input_bias = true;
  topology.mlp_output_bias = true;
  return topology;
}

evo::cpu::GenebBertTopology topology_for(const std::string_view kind) {
  auto topology = base_topology();
  if (kind == "standard")
    return topology;
  if (kind == "gena-final" || kind == "gena-no-final") {
    topology.norm_placement = GenebBertNormPlacement::kPre;
    topology.final_layer_norm = kind == "gena-final";
    return topology;
  }
  if (kind == "mosaic") {
    topology.position_encoding = GenebBertPositionEncoding::kAlibi;
    topology.mlp_kind = GenebBertMlpKind::kGatedGelu;
    topology.qkv_layout = GenebBertQkvLayout::kFused;
    topology.pooling = GenebBertPooling::kClsToken;
    topology.unpad_masked_tokens = true;
    topology.mlp_input_bias = false;
    return topology;
  }
  if (kind == "genomics-fm") {
    topology.layer_norm_epsilon = 1.0e-12F;
    topology.mlp_kind = GenebBertMlpKind::kGatedGelu;
    topology.qkv_layout = GenebBertQkvLayout::kFused;
    topology.pooling = GenebBertPooling::kClsToken;
    topology.unpad_masked_tokens = true;
    topology.mlp_input_bias = false;
    return topology;
  }
  if (kind == "mutbert") {
    topology.position_encoding = GenebBertPositionEncoding::kRope;
    topology.input_kind = GenebBertInputKind::kSoftVocabulary;
    topology.rope_base = 10000.0F;
    return topology;
  }
  throw std::runtime_error("unknown test topology: " + std::string{kind});
}

struct Fixture final {
  evo::cpu::GenebBertTopology topology;
  std::vector<std::vector<std::uint8_t>> storage;
  std::vector<evo::cpu::GenebBertNamedTensorView> views;
  evo::cpu::GenebBertModel model;
};

void build_fixture(const std::string_view kind, Fixture *const output) {
  require(output != nullptr, "fixture output is null");
  Fixture fixture;
  fixture.topology = topology_for(kind);
  std::vector<evo::cpu::GenebBertTensorRequirement> requirements;
  auto status =
      evo::cpu::canonical_geneb_bert_tensors(fixture.topology, &requirements);
  require(status.ok(), "canonical tensor manifest failed: " + status.message());
  fixture.storage.resize(requirements.size());
  fixture.views.reserve(requirements.size());
  for (std::size_t tensor_index = 0; tensor_index < requirements.size();
       ++tensor_index) {
    const auto &requirement = requirements[tensor_index];
    std::size_t elements = 1;
    for (const std::size_t dimension : requirement.shape)
      elements *= dimension;
    auto &bytes = fixture.storage[tensor_index];
    bytes.resize(elements * sizeof(float));
    const std::uint32_t seed = fnv1a(requirement.name);
    for (std::size_t index = 0; index < elements; ++index) {
      const float oscillation =
          std::sin(static_cast<float>((seed % 997U) + index * 17U) * 0.013F);
      float value = oscillation * 0.075F;
      if (is_norm_weight(requirement.name))
        value = 1.0F + oscillation * 0.08F;
      else if (is_norm_bias(requirement.name))
        value = oscillation * 0.015F;
      std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(value));
    }
  }
  for (std::size_t index = 0; index < requirements.size(); ++index) {
    const auto &requirement = requirements[index];
    fixture.views.push_back(
        {requirement.name,
         {fixture.storage[index].data(), fixture.storage[index].size(),
          requirement.dtype, requirement.shape}});
  }
  status = fixture.model.load(fixture.topology, fixture.views);
  require(status.ok(), "fixture model load failed: " + status.message());
  *output = std::move(fixture);
}

std::vector<float> manual_layer_norm(const std::vector<float> &values,
                                     const float *const scale,
                                     const float *const bias,
                                     const float epsilon) {
  float mean = 0.0F;
  for (const float value : values)
    mean += value;
  mean /= static_cast<float>(values.size());
  float variance = 0.0F;
  for (const float value : values) {
    const float centered = value - mean;
    variance += centered * centered;
  }
  variance /= static_cast<float>(values.size());
  const float inverse = 1.0F / std::sqrt(variance + epsilon);
  std::vector<float> output(values.size());
  for (std::size_t index = 0; index < values.size(); ++index)
    output[index] =
        (values[index] - mean) * inverse * scale[index] + bias[index];
  return output;
}

const evo::cpu::GenebBertTensorView &view(const Fixture &fixture,
                                          const std::string_view name) {
  for (const auto &named : fixture.views) {
    if (named.name == name)
      return named.tensor;
  }
  throw std::runtime_error("missing fixture tensor: " + std::string{name});
}

float f32_value(const evo::cpu::GenebBertTensorView &tensor,
                const std::size_t index) {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(float), sizeof(value));
  return value;
}

void require_close(const std::vector<float> &actual,
                   const std::vector<float> &expected, const float tolerance,
                   const std::string &label) {
  require(actual.size() == expected.size(), label + " size differs");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (std::fabs(actual[index] - expected[index]) > tolerance) {
      throw std::runtime_error(label + " differs at " + std::to_string(index));
    }
  }
}

void test_topology_contracts() {
  for (const std::string_view kind : {"standard", "gena-final", "gena-no-final",
                                      "mosaic", "mutbert", "genomics-fm"}) {
    auto topology = topology_for(kind);
    require(evo::cpu::validate_geneb_bert_topology(topology).ok(),
            std::string{kind} + " topology must validate");
    std::vector<evo::cpu::GenebBertTensorRequirement> manifest;
    require(evo::cpu::canonical_geneb_bert_tensors(topology, &manifest).ok() &&
                !manifest.empty(),
            std::string{kind} + " manifest must compile");
  }
  auto invalid = topology_for("mosaic");
  invalid.position_encoding = GenebBertPositionEncoding::kAbsolute;
  require(!evo::cpu::validate_geneb_bert_topology(invalid).ok(),
          "Mosaic topology must not masquerade as absolute BERT");
  invalid = topology_for("mutbert");
  invalid.input_kind = GenebBertInputKind::kTokenIds;
  require(!evo::cpu::validate_geneb_bert_topology(invalid).ok(),
          "MutBERT topology must retain soft-vocabulary input");
  invalid = topology_for("genomics-fm");
  invalid.pooling = GenebBertPooling::kAttentionMaskMean;
  require(!evo::cpu::validate_geneb_bert_topology(invalid).ok(),
          "Genomics-FM topology must retain raw CLS pooling");
  invalid = topology_for("genomics-fm");
  invalid.layer_norm_epsilon = 1.0e-5F;
  require(!evo::cpu::validate_geneb_bert_topology(invalid).ok(),
          "Genomics-FM topology must retain its audited LayerNorm epsilon");
  invalid = topology_for("genomics-fm");
  invalid.activation_dtype = evo::TensorDType::kBF16;
  require(!evo::cpu::validate_geneb_bert_topology(invalid).ok(),
          "Genomics-FM topology must retain its F32 activation contract");
}

void test_load_corruption() {
  Fixture fixture;
  build_fixture("standard", &fixture);
  auto missing = fixture.views;
  missing.pop_back();
  evo::cpu::GenebBertModel model;
  require(!model.load(fixture.topology, missing).ok(),
          "missing tensor must fail load");
  auto extra = fixture.views;
  extra.push_back(extra.front());
  extra.back().name = "unexpected.weight";
  require(!model.load(fixture.topology, extra).ok(),
          "extra tensor must fail load");
  auto wrong_shape = fixture.views;
  wrong_shape.front().tensor.shape[0] += 1U;
  require(!model.load(fixture.topology, wrong_shape).ok(),
          "wrong tensor shape must fail load");
  auto wrong_dtype = fixture.views;
  wrong_dtype.front().tensor.dtype = evo::TensorDType::kBF16;
  require(!model.load(fixture.topology, wrong_dtype).ok(),
          "wrong tensor dtype must fail load");
}

void test_forward_families() {
  for (const std::string_view kind :
       {"standard", "gena-final", "gena-no-final", "mosaic", "genomics-fm"}) {
    Fixture fixture;
    build_fixture(kind, &fixture);
    const std::vector<evo::TokenId> tokens{1U, 5U, 2U};
    const bool unpadded = kind == "mosaic" || kind == "genomics-fm";
    const std::vector<std::uint8_t> mask =
        unpadded ? std::vector<std::uint8_t>{1U, 1U, 0U}
                 : std::vector<std::uint8_t>{1U, 1U, 1U};
    evo::cpu::GenebBertForwardResult result;
    auto status = fixture.model.forward(tokens, mask, {2U, 0U, 1U}, &result);
    require(status.ok(),
            std::string{kind} + " forward failed: " + status.message());
    require(
        result.rows == 3U && result.width == 4U &&
            result.final_hidden.size() == 12U && result.captures.size() == 3U &&
            result.captures[0].layer == 2U && result.captures[1].layer == 0U &&
            result.captures[2].layer == 1U,
        std::string{kind} + " hidden capture contract differs");
    if (unpadded) {
      for (std::size_t column = 0; column < result.width; ++column) {
        require(result.final_hidden[2U * result.width + column] == 0.0F,
                std::string{kind} +
                    " masked query row must be padded back as zeros");
      }
      require(!fixture.model.forward(tokens, {1U, 0U, 1U}, {}, &result).ok(),
              std::string{kind} +
                  " must reject a non-prefix mask it cannot unpad exactly");
    }
    std::vector<float> pooled;
    status = fixture.model.pool(result, mask, &pooled);
    require(status.ok() && pooled.size() == 4U,
            std::string{kind} + " pool failed");
  }
}

void test_mutbert_one_hot_times_embedding() {
  Fixture fixture;
  build_fixture("mutbert", &fixture);
  const std::size_t rows = 2;
  std::vector<float> soft(rows * fixture.topology.vocabulary_size, 0.0F);
  soft[5] = 0.25F;
  soft[6] = 0.75F;
  // This row is deliberately nonzero; the pinned GENEB transform must zero it
  // before the vocabulary-by-embedding multiplication because mask[1] == 0.
  soft[fixture.topology.vocabulary_size + 8U] = 1.0F;
  const std::vector<std::uint8_t> mask{1U, 0U};
  evo::cpu::GenebBertForwardResult result;
  auto status = fixture.model.forward_soft(soft, rows, mask, {0U}, &result);
  require(status.ok(), "MutBERT soft forward failed: " + status.message());
  require(result.captures.size() == 1U &&
              result.captures[0].values.size() == rows * fixture.topology.width,
          "MutBERT embedding capture is missing");

  const auto &word = view(fixture, "bert.embeddings.word_embeddings.weight");
  const auto &type =
      view(fixture, "bert.embeddings.token_type_embeddings.weight");
  const auto &scale = view(fixture, "bert.embeddings.LayerNorm.weight");
  const auto &bias = view(fixture, "bert.embeddings.LayerNorm.bias");
  std::vector<float> scale_values(4U);
  std::vector<float> bias_values(4U);
  for (std::size_t column = 0; column < 4U; ++column) {
    scale_values[column] = f32_value(scale, column);
    bias_values[column] = f32_value(bias, column);
  }
  std::vector<float> first_raw(4U);
  std::vector<float> padded_raw(4U);
  for (std::size_t column = 0; column < 4U; ++column) {
    first_raw[column] = 0.25F * f32_value(word, 5U * 4U + column) +
                        0.75F * f32_value(word, 6U * 4U + column) +
                        f32_value(type, column);
    padded_raw[column] = f32_value(type, column);
  }
  const auto first_expected =
      manual_layer_norm(first_raw, scale_values.data(), bias_values.data(),
                        fixture.topology.layer_norm_epsilon);
  const auto padded_expected =
      manual_layer_norm(padded_raw, scale_values.data(), bias_values.data(),
                        fixture.topology.layer_norm_epsilon);
  const std::vector<float> first_actual(result.captures[0].values.begin(),
                                        result.captures[0].values.begin() + 4);
  const std::vector<float> padded_actual(result.captures[0].values.begin() + 4,
                                         result.captures[0].values.end());
  require_close(first_actual, first_expected, 2.0e-6F,
                "MutBERT weighted one-hot embedding");
  require_close(padded_actual, padded_expected, 2.0e-6F,
                "MutBERT zeroed padded one-hot embedding");

  std::vector<float> invalid_soft = soft;
  invalid_soft[0] = std::numeric_limits<float>::quiet_NaN();
  require(
      !fixture.model.forward_soft(invalid_soft, rows, mask, {}, &result).ok(),
      "MutBERT non-finite soft input must fail");
  const std::vector<evo::TokenId> tokens{5U, 8U};
  require(!fixture.model.forward(tokens, mask, {}, &result).ok(),
          "MutBERT artifact must reject token-ID forward API");
}

void test_attention_mask_mean_uses_direct_f32_division() {
  // PyTorch/NumPy sum active F32 rows in order and then divide by the active
  // count.  Precomputing a reciprocal and multiplying changes this case by
  // one ULP, which is observable for GENA-LM because it has no final norm.
  std::vector<float> hidden(11U, 0.0F);
  hidden[0] = 1.0e8F;
  hidden[1] = 1.0F;
  hidden[2] = -1.0e8F;
  hidden[3] = -1000.0F;
  const std::vector<std::uint8_t> mask(11U, 1U);
  std::vector<float> pooled;
  const auto status = evo::pool_geneb_embedding(
      hidden, 11U, 1U, mask, "attention-mask-mean", &pooled);
  require(status.ok() && pooled.size() == 1U,
          "GENEB direct-division pooling failed");
  std::uint32_t bits = 0U;
  std::memcpy(&bits, pooled.data(), sizeof(bits));
  require(bits == 0xc2b5d174U,
          "GENEB attention-mask mean must use direct F32 division");
}

void dump_array(const std::vector<float> &values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    std::cout << std::setprecision(9) << values[index];
  }
  std::cout << ']';
}

void dump_case(const std::string_view kind) {
  Fixture fixture;
  build_fixture(kind, &fixture);
  const std::vector<std::uint8_t> mask{1U, 1U, 1U};
  evo::cpu::GenebBertForwardResult result;
  auto status = evo::Status::Ok();
  if (kind == "mutbert") {
    std::vector<float> soft(3U * fixture.topology.vocabulary_size, 0.0F);
    soft[1U] = 1.0F;
    soft[fixture.topology.vocabulary_size + 5U] = 1.0F;
    soft[2U * fixture.topology.vocabulary_size + 2U] = 1.0F;
    status = fixture.model.forward_soft(soft, 3U, mask, {0U, 1U, 2U}, &result);
  } else {
    status = fixture.model.forward({1U, 5U, 2U}, mask, {0U, 1U, 2U}, &result);
  }
  require(status.ok(), "vector dump forward failed: " + status.message());
  std::vector<float> pooled;
  status = fixture.model.pool(result, mask, &pooled);
  require(status.ok(), "vector dump pool failed: " + status.message());
  std::cout << "{\"case\":\"" << kind << "\",\"rows\":" << result.rows
            << ",\"width\":" << result.width << ",\"captures\":{";
  for (std::size_t index = 0; index < result.captures.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    std::cout << '\"' << result.captures[index].layer << "\":";
    dump_array(result.captures[index].values);
  }
  std::cout << "},\"final_hidden\":";
  dump_array(result.final_hidden);
  std::cout << ",\"pooled\":";
  dump_array(pooled);
  std::cout << "}\n";
}

void verify_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  require(status.ok(), "artifact open failed: " + status.message());
  evo::cpu::GenebBertModel model;
  status = model.load(artifact);
  require(status.ok(), "artifact load failed: " + status.message());
}

void verify_adapter_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  require(status.ok(), "adapter artifact open failed: " + status.message());

  evo::cpu::GenebBertModel direct;
  status = direct.load(artifact);
  require(status.ok(), "direct artifact load failed: " + status.message());
  const auto *const topology = direct.topology();
  require(topology != nullptr, "direct artifact topology is missing");

  const std::vector<evo::TokenId> tokens{1U, 5U, 2U};
  const std::vector<std::uint8_t> attention_mask{1U, 1U, 0U};
  evo::cpu::GenebBertForwardResult expected;
  if (topology->input_kind == GenebBertInputKind::kSoftVocabulary) {
    std::vector<float> one_hot(tokens.size() * topology->vocabulary_size, 0.0F);
    for (std::size_t row = 0; row < tokens.size(); ++row) {
      if (attention_mask[row] != 0U)
        one_hot[row * topology->vocabulary_size + tokens[row]] = 1.0F;
    }
    status = direct.forward_soft(one_hot, tokens.size(), attention_mask,
                                 {topology->layers}, &expected);
  } else {
    status =
        direct.forward(tokens, attention_mask, {topology->layers}, &expected);
  }
  require(status.ok(), "direct adapter oracle failed: " + status.message());
  require(expected.captures.size() == 1U,
          "direct adapter oracle capture is incomplete");

  evo::cpu::Model model;
  status = model.load(artifact);
  require(status.ok(), "public CPU model load failed: " + status.message());
  require(model.config().implementation ==
                  evo::ArchitectureImplementation::kGenebBertEncoder &&
              model.config().vocab_size == topology->vocabulary_size &&
              model.config().width == topology->width &&
              model.config().layers == topology->layers,
          "public CPU model configuration differs from the artifact");

  evo::cpu::Context context;
  status = context.initialize_shared(model, tokens.size());
  require(status.ok(),
          "public CPU context initialization failed: " + status.message());
  std::vector<float> actual;
  status = context.prefill_embedding_masked(tokens, attention_mask,
                                            topology->layers, &actual);
  require(status.ok(),
          "public CPU masked embedding failed: " + status.message());
  require(actual == expected.captures.front().values,
          "public CPU adapter differs from the direct runtime");
  require(context.position() == tokens.size() &&
              context.activation_capacity() == tokens.size(),
          "public CPU context state differs after embedding");
}

} // namespace

int main(const int argc, const char *const *const argv) {
  try {
    if (argc == 3 && std::string_view{argv[1]} == "--dump-case") {
      dump_case(argv[2]);
      return 0;
    }
    if (argc == 3 && std::string_view{argv[1]} == "--verify-artifact") {
      verify_artifact(argv[2]);
      return 0;
    }
    if (argc == 3 && std::string_view{argv[1]} == "--verify-adapter") {
      verify_adapter_artifact(argv[2]);
      return 0;
    }
    if (argc != 1)
      throw std::runtime_error("invalid GENEB BERT test arguments");
    test_topology_contracts();
    test_load_corruption();
    test_forward_families();
    test_mutbert_one_hot_times_embedding();
    test_attention_mask_mean_uses_direct_f32_division();
    std::cout << "GENEB BERT tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
