// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_custom_encoder.hpp"
#include "evo/cpu/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using evo::cpu::GenebCustomEncoderNamedTensorView;
using evo::cpu::GenebCustomEncoderTensorRequirement;
using evo::cpu::GenebCustomEncoderTopology;

void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void check(const bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

bool close(const float left, const float right, const float tolerance = 2e-5F) {
  return std::isfinite(left) && std::isfinite(right) &&
         std::abs(left - right) <= tolerance;
}

GenebCustomEncoderTopology luca_topology() {
  GenebCustomEncoderTopology topology;
  topology.variant = evo::cpu::GenebCustomEncoderVariant::kLucaOne;
  topology.vocabulary_size = 10U;
  topology.tokenizer_vocabulary_size = 10U;
  topology.width = 4U;
  topology.layers = 2U;
  topology.attention_heads = 2U;
  topology.head_dimension = 2U;
  topology.inner_width = 6U;
  topology.maximum_sequence_length = 8U;
  topology.token_type_vocabulary_size = 2U;
  topology.pad_token_id = 0U;
  topology.cls_token_id = 2U;
  topology.sep_token_id = 3U;
  topology.layer_norm_epsilon = 1.0e-5F;
  topology.rope_base = 10000.0F;
  topology.position_encoding =
      evo::cpu::GenebCustomPositionEncoding::kRopeSplitHalf;
  topology.norm_placement = evo::cpu::GenebCustomNormPlacement::kPre;
  topology.qkv_layout = evo::cpu::GenebCustomQkvLayout::kSeparate;
  topology.mlp_kind = evo::cpu::GenebCustomMlpKind::kGelu;
  topology.pooling = evo::cpu::GenebCustomPooling::kAttentionMaskMean;
  topology.attention_bias = true;
  topology.mlp_input_bias = true;
  topology.mlp_output_bias = true;
  topology.embedding_layer_norm = false;
  topology.final_layer_norm = true;
  topology.unpad_masked_tokens = false;
  topology.token_type_embeddings = true;
  topology.weight_dtype = evo::TensorDType::kF32;
  return topology;
}

GenebCustomEncoderTopology genomics_topology() {
  GenebCustomEncoderTopology topology;
  topology.variant = evo::cpu::GenebCustomEncoderVariant::kGenomicsFm;
  topology.vocabulary_size = 12U;
  topology.tokenizer_vocabulary_size = 10U;
  topology.width = 4U;
  topology.layers = 2U;
  topology.attention_heads = 2U;
  topology.head_dimension = 2U;
  topology.inner_width = 6U;
  topology.maximum_sequence_length = 8U;
  topology.token_type_vocabulary_size = 2U;
  topology.pad_token_id = 3U;
  topology.cls_token_id = 1U;
  topology.sep_token_id = 2U;
  topology.layer_norm_epsilon = 1.0e-12F;
  topology.rope_base = 0.0F;
  topology.position_encoding = evo::cpu::GenebCustomPositionEncoding::kAbsolute;
  topology.norm_placement = evo::cpu::GenebCustomNormPlacement::kPost;
  topology.qkv_layout = evo::cpu::GenebCustomQkvLayout::kFused;
  topology.mlp_kind = evo::cpu::GenebCustomMlpKind::kGatedGelu;
  topology.pooling = evo::cpu::GenebCustomPooling::kClsToken;
  topology.attention_bias = true;
  topology.mlp_input_bias = false;
  topology.mlp_output_bias = true;
  topology.embedding_layer_norm = true;
  topology.final_layer_norm = false;
  topology.unpad_masked_tokens = true;
  topology.token_type_embeddings = true;
  topology.weight_dtype = evo::TensorDType::kF32;
  return topology;
}

struct TinyWeights final {
  std::vector<std::vector<float>> storage;
  std::vector<GenebCustomEncoderNamedTensorView> views;
};

TinyWeights tiny_weights(const GenebCustomEncoderTopology &topology) {
  std::vector<GenebCustomEncoderTensorRequirement> requirements;
  check(
      evo::cpu::canonical_geneb_custom_encoder_tensors(topology, &requirements)
          .ok(),
      "tiny topology has a canonical manifest");
  TinyWeights result;
  result.storage.reserve(requirements.size());
  result.views.reserve(requirements.size());
  for (std::size_t tensor_index = 0; tensor_index < requirements.size();
       ++tensor_index) {
    const auto &requirement = requirements[tensor_index];
    std::size_t elements = 1U;
    for (const std::size_t dimension : requirement.shape)
      elements *= dimension;
    std::vector<float> values(elements, 0.0F);
    if (requirement.name.find("LayerNorm.weight") != std::string::npos ||
        requirement.name.find("layer_norm.weight") != std::string::npos ||
        requirement.name.find("layernorm.weight") != std::string::npos) {
      for (std::size_t index = 0; index < elements; ++index)
        values[index] = 0.93F + 0.02F * static_cast<float>(index % 5U);
    } else if (requirement.name.find("rot_emb.inv_freq") != std::string::npos) {
      for (std::size_t index = 0; index < elements; ++index)
        values[index] =
            1.0F /
            std::pow(10000.0F, static_cast<float>(2U * index) /
                                   static_cast<float>(topology.head_dimension));
    } else if (requirement.name.find(".bias") != std::string::npos) {
      for (std::size_t index = 0; index < elements; ++index)
        values[index] = static_cast<float>(
                            static_cast<int>((tensor_index + index) % 7U) - 3) *
                        0.004F;
    } else {
      for (std::size_t index = 0; index < elements; ++index)
        values[index] =
            static_cast<float>(
                static_cast<int>((tensor_index * 11U + index * 7U) % 19U) - 9) *
            0.013F;
    }
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

void check_forward(const GenebCustomEncoderTopology &topology,
                   const std::vector<evo::TokenId> &tokens,
                   const std::vector<std::uint8_t> &mask) {
  auto weights = tiny_weights(topology);
  evo::cpu::GenebCustomEncoderModel model;
  check(model.load(topology, weights.views).ok(), "tiny model loads");
  evo::cpu::GenebCustomEncoderForwardResult result;
  auto status = model.forward(tokens, mask, {0U, 1U, 2U}, &result);
  check(status.ok(), "tiny forward succeeds: " + status.message());
  check(result.rows == tokens.size() && result.width == topology.width &&
            result.final_hidden.size() == tokens.size() * topology.width &&
            result.pooled.size() == topology.width &&
            result.captures.size() == 3U,
        "forward result shapes are exact");
  check(result.captures[0].layer == 0U && result.captures[1].layer == 1U &&
            result.captures[2].layer == 2U,
        "capture order is caller order");
  for (const float value : result.final_hidden)
    check(std::isfinite(value), "final hidden is finite");
  if (topology.pooling == evo::cpu::GenebCustomPooling::kClsToken) {
    for (std::size_t column = 0; column < topology.width; ++column)
      check(close(result.pooled[column], result.final_hidden[column]),
            "Genomics-FM pool is raw row zero");
  } else {
    const std::size_t count =
        static_cast<std::size_t>(std::count(mask.begin(), mask.end(), 1U));
    for (std::size_t column = 0; column < topology.width; ++column) {
      float expected = 0.0F;
      for (std::size_t row = 0; row < tokens.size(); ++row) {
        if (mask[row] != 0U)
          expected += result.final_hidden[row * topology.width + column];
      }
      check(close(result.pooled[column], expected / static_cast<float>(count)),
            "LucaOne pool is attention-mask mean");
    }
  }
  check(!model.forward(tokens, {1U, 0U, 1U, 0U}, {}, &result).ok(),
        "non-prefix attention mask is rejected");
  check(!model.forward(tokens, mask, {0U, 0U}, &result).ok(),
        "duplicate capture is rejected");
  auto invalid_tokens = tokens;
  invalid_tokens.front() = static_cast<evo::TokenId>(topology.vocabulary_size);
  check(!model.forward(invalid_tokens, mask, {}, &result).ok(),
        "out-of-range token ID is rejected");
}

void check_topology_and_manifest_gates() {
  auto luca = luca_topology();
  auto genomics = genomics_topology();
  check(evo::cpu::validate_geneb_custom_encoder_topology(luca).ok(),
        "LucaOne topology is accepted");
  check(evo::cpu::validate_geneb_custom_encoder_topology(genomics).ok(),
        "Genomics-FM topology is accepted");
  auto drift = luca;
  drift.layer_norm_epsilon = 1.0e-12F;
  check(!evo::cpu::validate_geneb_custom_encoder_topology(drift).ok(),
        "LucaOne source/config epsilon confusion is rejected");
  drift = genomics;
  drift.position_encoding =
      evo::cpu::GenebCustomPositionEncoding::kRopeSplitHalf;
  check(!evo::cpu::validate_geneb_custom_encoder_topology(drift).ok(),
        "Genomics-FM APE drift is rejected");
  drift = genomics;
  drift.mlp_input_bias = true;
  check(!evo::cpu::validate_geneb_custom_encoder_topology(drift).ok(),
        "Genomics-FM bias drift is rejected");
  drift = genomics;
  drift.pooling = evo::cpu::GenebCustomPooling::kAttentionMaskMean;
  check(!evo::cpu::validate_geneb_custom_encoder_topology(drift).ok(),
        "Genomics-FM CLS pooling drift is rejected");

  auto weights = tiny_weights(luca);
  auto missing = weights.views;
  missing.pop_back();
  evo::cpu::GenebCustomEncoderModel model;
  check(!model.load(luca, missing).ok(), "missing tensor is rejected");
  auto duplicated = weights.views;
  duplicated.push_back(duplicated.front());
  check(!model.load(luca, duplicated).ok(), "duplicate tensor is rejected");
  auto wrong_shape = weights.views;
  wrong_shape.front().tensor.shape.front() += 1U;
  check(!model.load(luca, wrong_shape).ok(), "wrong tensor shape is rejected");
}

void dump_vector(const std::vector<float> &values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::cout.precision(std::numeric_limits<float>::max_digits10);
    std::cout << values[index];
  }
  std::cout << ']';
}

int run_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebCustomEncoderModel model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  const auto *const topology = model.topology();
  std::vector<evo::TokenId> tokens;
  if (topology->variant == evo::cpu::GenebCustomEncoderVariant::kLucaOne)
    tokens = {2U, 5U, 6U, 3U, 0U};
  else
    tokens = {1U, 4U, 5U, 2U, 3U};
  evo::cpu::GenebCustomEncoderForwardResult result;
  status = model.forward(tokens, {1U, 1U, 1U, 1U, 0U}, {0U, topology->layers},
                         &result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::cout << "{\"variant\":\""
            << (topology->variant ==
                        evo::cpu::GenebCustomEncoderVariant::kLucaOne
                    ? "lucaone"
                    : "genomics-fm")
            << "\",\"rows\":" << result.rows << ",\"width\":" << result.width
            << ",\"pooled\":";
  dump_vector(result.pooled);
  std::cout << "}\n";
  return 0;
}

int verify_cpu_adapter(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << "custom encoder adapter artifact open failed: "
              << status.message() << '\n';
    return 2;
  }

  evo::cpu::GenebCustomEncoderModel direct;
  status = direct.load(artifact);
  if (!status.ok()) {
    std::cerr << "custom encoder typed load failed: " << status.message()
              << '\n';
    return 2;
  }
  const auto *const topology = direct.topology();
  if (topology == nullptr) {
    std::cerr << "custom encoder typed topology is missing\n";
    return 2;
  }
  const bool luca =
      topology->variant == evo::cpu::GenebCustomEncoderVariant::kLucaOne;
  const bool production_luca =
      luca && topology->vocabulary_size == 39U && topology->width == 2560U &&
      topology->layers == 20U && topology->maximum_sequence_length == 4096U;

  evo::cpu::Model model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << "custom encoder public CPU Model load failed: "
              << status.message() << '\n';
    return 2;
  }
  const auto &config = model.config();
  if (config.architecture != evo::cpu::kGenebCustomEncoderArchitecture ||
      config.artifact_profile != evo::cpu::kGenebCustomEncoderArtifactProfile ||
      config.implementation !=
          evo::ArchitectureImplementation::kGenebCustomEncoder ||
      config.tokenizer != evo::ArchitectureTokenizer::kArtifact ||
      config.vocab_size != topology->vocabulary_size ||
      config.width != topology->width || config.layers != topology->layers ||
      config.max_seqlen != topology->maximum_sequence_length) {
    std::cerr << "custom encoder public CPU Model contract differs\n";
    return 2;
  }
  std::vector<evo::TokenId> direct_tokens;
  status = model.encode(luca ? "Atu?" : "[BPE]", &direct_tokens);
  const std::vector<evo::TokenId> expected_direct_tokens =
      luca ? std::vector<evo::TokenId>{2U, 5U, 6U, 6U, 9U, 3U}
           : std::vector<evo::TokenId>{1U, 4096U, 2U};
  if (!status.ok() || direct_tokens != expected_direct_tokens) {
    std::cerr << (status.ok() ? "custom encoder tokenizer IDs differ"
                              : status.message())
              << ":";
    for (const auto token : direct_tokens)
      std::cerr << ' ' << token;
    std::cerr << '\n';
    return 2;
  }

  evo::GenebPreparedEmbeddingInput prepared;
  status = model.prepare_geneb_embedding_input("acgtx", &prepared);
  if (!status.ok()) {
    std::cerr << "custom encoder GENEB preparation failed: " << status.message()
              << '\n';
    return 2;
  }
  const std::vector<evo::TokenId> expected_boundaries =
      luca ? std::vector<evo::TokenId>{2U, 3U}
           : std::vector<evo::TokenId>{1U, 2U};
  if (prepared.tokens.size() < 2U ||
      prepared.tokens.front() != expected_boundaries.front() ||
      prepared.tokens.back() != expected_boundaries.back() ||
      prepared.attention_mask !=
          std::vector<std::uint8_t>(prepared.tokens.size(), 1U) ||
      prepared.transform.sequence != (luca ? "acgtx" : "ACGTN")) {
    std::cerr << "custom encoder prepared token/input semantics differ\n";
    return 2;
  }

  std::size_t maximum_forward_rows = 0U;
  const auto verify_masked_adapter =
      [&](const std::vector<evo::TokenId> &tokens,
          const std::vector<std::uint8_t> &attention_mask,
          const std::string_view label,
          evo::cpu::GenebCustomEncoderForwardResult *const expected,
          std::vector<float> *const actual_hidden) {
        constexpr std::size_t kProductionLucaForwardLimit = 16U;
        if (production_luca && tokens.size() > kProductionLucaForwardLimit) {
          std::cerr << label
                    << " attempted an unbounded production LucaOne forward\n";
          return false;
        }
        maximum_forward_rows = std::max(maximum_forward_rows, tokens.size());
        status = direct.forward(tokens, attention_mask, {topology->layers},
                                expected);
        if (!status.ok() || expected->captures.size() != 1U) {
          std::cerr << label << ' '
                    << (status.ok() ? "typed capture is incomplete"
                                    : status.message())
                    << '\n';
          return false;
        }
        evo::cpu::Context context;
        status = context.initialize_shared(model, tokens.size());
        if (!status.ok()) {
          std::cerr << label
                    << " public CPU Context failed: " << status.message()
                    << '\n';
          return false;
        }
        status = context.prefill_embedding_masked(
            tokens, attention_mask, topology->layers, actual_hidden);
        if (!status.ok() ||
            *actual_hidden != expected->captures.front().values ||
            context.position() != tokens.size() ||
            context.activation_capacity() != tokens.size()) {
          std::cerr << label << ' '
                    << (status.ok() ? "CPU adapter differs from typed runtime"
                                    : status.message())
                    << '\n';
          return false;
        }
        std::vector<float> actual_pooled;
        status = evo::pool_geneb_embedding(
            *actual_hidden, tokens.size(), config.width, attention_mask,
            luca ? "attention-mask-mean" : "cls-token", &actual_pooled);
        if (!status.ok() || actual_pooled != expected->pooled) {
          std::cerr << label << ' '
                    << (status.ok()
                            ? "public pooling differs from typed runtime"
                            : status.message())
                    << '\n';
          return false;
        }
        return true;
      };

  evo::cpu::GenebCustomEncoderForwardResult expected;
  std::vector<float> actual_hidden;
  if (!verify_masked_adapter(prepared.tokens, prepared.attention_mask,
                             "custom encoder", &expected, &actual_hidden))
    return 2;

  const auto *const spec = model.geneb_embedding_spec();
  if (spec == nullptr || spec->runtime_id != config.model_id ||
      spec->reference.output_width != config.width ||
      spec->reference.pooling != (luca ? "attention-mask-mean" : "cls-token")) {
    std::cerr << "custom encoder GENEB embedding preset differs\n";
    return 2;
  }
  std::vector<float> actual_pooled;
  status = evo::pool_geneb_embedding(actual_hidden, prepared.tokens.size(),
                                     config.width, prepared.attention_mask,
                                     spec->reference.pooling, &actual_pooled);
  if (!status.ok() || actual_pooled != expected.pooled) {
    std::cerr
        << (status.ok()
                ? "custom encoder public pooling differs from typed runtime"
                : status.message())
        << '\n';
    return 2;
  }

  auto padded_tokens = prepared.tokens;
  auto padded_mask = prepared.attention_mask;
  padded_tokens.resize(topology->maximum_sequence_length,
                       static_cast<evo::TokenId>(topology->pad_token_id));
  padded_mask.resize(topology->maximum_sequence_length, 0U);
  evo::cpu::GenebCustomEncoderForwardResult padded_expected;
  std::vector<float> padded_hidden;
  if (production_luca) {
    if (padded_tokens.size() != topology->maximum_sequence_length ||
        padded_mask.size() != topology->maximum_sequence_length ||
        !std::equal(prepared.tokens.begin(), prepared.tokens.end(),
                    padded_tokens.begin()) ||
        !std::equal(prepared.attention_mask.begin(),
                    prepared.attention_mask.end(), padded_mask.begin()) ||
        !std::all_of(padded_mask.begin() + static_cast<std::ptrdiff_t>(
                                               prepared.attention_mask.size()),
                     padded_mask.end(),
                     [](const std::uint8_t value) { return value == 0U; })) {
      std::cerr << "production LucaOne full-padding metadata differs\n";
      return 2;
    }
    auto bounded_tokens = prepared.tokens;
    auto bounded_mask = prepared.attention_mask;
    bounded_tokens.resize(prepared.tokens.size() + 2U,
                          static_cast<evo::TokenId>(topology->pad_token_id));
    bounded_mask.resize(bounded_tokens.size(), 0U);
    if (!verify_masked_adapter(
            bounded_tokens, bounded_mask,
            "production LucaOne bounded right-padding closure",
            &padded_expected, &padded_hidden))
      return 2;
  } else if (!verify_masked_adapter(padded_tokens, padded_mask,
                                    "custom encoder right-padding closure",
                                    &padded_expected, &padded_hidden)) {
    return 2;
  }

  if (!luca) {
    evo::GenebPreparedEmbeddingInput whitespace_prepared;
    status = model.prepare_geneb_embedding_input(" \tacgtx\r\n",
                                                 &whitespace_prepared);
    if (!status.ok() || whitespace_prepared.tokens != prepared.tokens ||
        whitespace_prepared.attention_mask != prepared.attention_mask ||
        whitespace_prepared.transform.sequence != prepared.transform.sequence ||
        whitespace_prepared.transform.trim_left != 2U ||
        whitespace_prepared.transform.trim_right != 2U) {
      std::cerr << (status.ok()
                        ? "Genomics-FM outer-strip token closure differs"
                        : status.message())
                << '\n';
      return 2;
    }
    evo::cpu::GenebCustomEncoderForwardResult whitespace_expected;
    std::vector<float> whitespace_hidden;
    if (!verify_masked_adapter(whitespace_prepared.tokens,
                               whitespace_prepared.attention_mask,
                               "Genomics-FM outer-strip vector closure",
                               &whitespace_expected, &whitespace_hidden) ||
        whitespace_hidden != actual_hidden ||
        whitespace_expected.pooled != expected.pooled) {
      std::cerr << "Genomics-FM stripped/unstripped vectors differ\n";
      return 2;
    }
  }

  evo::GenebPreparedEmbeddingInput truncated;
  std::string truncation_input;
  for (std::size_t multiplier = 2U; multiplier <= 1024U; multiplier *= 2U) {
    const std::size_t target = topology->maximum_sequence_length * multiplier;
    truncation_input.clear();
    truncation_input.reserve(target);
    constexpr std::string_view pattern{"ACGTN"};
    while (truncation_input.size() < target)
      truncation_input.append(pattern);
    truncation_input.resize(target);
    status = model.prepare_geneb_embedding_input(truncation_input, &truncated);
    if (!status.ok() || truncated.token_plan.exceeds_context)
      break;
  }
  const auto &plan = truncated.token_plan;
  if (!status.ok() ||
      truncated.tokens.size() != topology->maximum_sequence_length ||
      truncated.tokens.front() != expected_boundaries.front() ||
      truncated.tokens.back() != expected_boundaries.back() ||
      truncated.attention_mask !=
          std::vector<std::uint8_t>(truncated.tokens.size(), 1U) ||
      !plan.exceeds_context || plan.prefix_token_count != 1U ||
      plan.suffix_token_count != 1U || plan.source_offset != 1U ||
      plan.retained_payload_token_count !=
          topology->maximum_sequence_length - 2U ||
      plan.retained_token_count != topology->maximum_sequence_length ||
      plan.truncated_left != 0U ||
      plan.truncated_right !=
          plan.original_token_count - plan.retained_token_count ||
      plan.effective_token_count != topology->maximum_sequence_length) {
    std::cerr << (status.ok() ? "custom encoder boundary-preserving truncation "
                                "metadata differs"
                              : status.message())
              << '\n';
    return 2;
  }
  evo::cpu::GenebCustomEncoderForwardResult truncated_expected;
  std::vector<float> truncated_hidden;
  if (!production_luca &&
      !verify_masked_adapter(truncated.tokens, truncated.attention_mask,
                             "custom encoder truncation closure",
                             &truncated_expected, &truncated_hidden))
    return 2;

  if (production_luca && maximum_forward_rows > 16U) {
    std::cerr << "production LucaOne adapter exceeded its bounded forward gate\n";
    return 2;
  }

  std::cout << "{\"variant\":\"" << (luca ? "lucaone" : "genomics-fm")
            << "\",\"model_vocab\":" << topology->vocabulary_size
            << ",\"tokenizer_vocab\":" << topology->tokenizer_vocabulary_size
            << ",\"rows\":" << prepared.tokens.size()
            << ",\"max_forward_rows\":" << maximum_forward_rows
            << ",\"width\":" << config.width << ",\"pooling\":\""
            << spec->reference.pooling << "\"}\n";
  return 0;
}

int dump_tiny(const std::string_view variant) {
  const bool luca = variant == "lucaone";
  if (!luca && variant != "genomics-fm") {
    std::cerr << "unknown tiny variant\n";
    return 2;
  }
  const auto topology = luca ? luca_topology() : genomics_topology();
  auto weights = tiny_weights(topology);
  evo::cpu::GenebCustomEncoderModel model;
  auto status = model.load(topology, weights.views);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  const std::vector<evo::TokenId> tokens =
      luca ? std::vector<evo::TokenId>{2U, 5U, 6U, 3U, 0U}
           : std::vector<evo::TokenId>{1U, 4U, 5U, 2U, 3U};
  evo::cpu::GenebCustomEncoderForwardResult result;
  status = model.forward(tokens, {1U, 1U, 1U, 1U, 0U}, {0U, 1U, 2U}, &result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::cout << "{\"variant\":\"" << variant << "\",\"rows\":" << result.rows
            << ",\"width\":" << result.width << ",\"captures\":[";
  for (std::size_t index = 0; index < result.captures.size(); ++index) {
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

} // namespace

int main(const int argc, char **argv) {
  if (argc == 3 && std::string_view{argv[1]} == "--dump-tiny")
    return dump_tiny(argv[2]);
  if (argc == 3 && std::string_view{argv[1]} == "--verify-cpu-adapter")
    return verify_cpu_adapter(argv[2]);
  if (argc == 2)
    return run_artifact(argv[1]);
  if (argc != 1) {
    std::cerr << "usage: test_geneb_custom_encoder [artifact]\n";
    return 2;
  }
  check_topology_and_manifest_gates();
  check_forward(luca_topology(), {2U, 5U, 6U, 3U}, {1U, 1U, 1U, 1U});
  check_forward(genomics_topology(), {1U, 4U, 5U, 3U}, {1U, 1U, 1U, 0U});
  std::cout << "GENEB custom encoder tests passed\n";
  return 0;
}
