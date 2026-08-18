// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "evo/backend.hpp"
#include "evo/profile.hpp"
#include "evo/sampler.hpp"
#include "evo/status.hpp"
#include "evo/variant.hpp"

namespace evo {

enum class RunMode {
  kGenerate,
  kScore,
  kLogits,
  kEmbed,
  kVariantScore,
  kServe,
  kBench
};

enum class EmbeddingPooling { kNone, kMean, kLast };

enum class GenerationOutputFormat { kRaw, kFasta };

struct DumpLayerSpec final {
  std::size_t layer{0};
  std::string path;
};

struct CliOptions final {
  RunMode mode{RunMode::kGenerate};
  std::string model_path;
  std::string hf_repo;
  std::string prompt;
  std::string score_path;
  std::string benchmark_path;
  std::size_t benchmark_warmup{1};
  std::size_t benchmark_repetitions{5};
  GenerationOutputFormat generation_output_format{GenerationOutputFormat::kRaw};
  std::string generation_name{"generated"};
  std::string embed_path;
  std::string embed_output_dir;
  std::string embed_preset;
  std::size_t embed_layer{0};
  EmbeddingPooling embedding_pooling{EmbeddingPooling::kNone};
  std::string variant_sequence;
  std::string variant_vcf_path;
  std::string variant_reference_path;
  std::size_t variant_position_1based{0};
  std::string variant_reference;
  std::string variant_alternate;
  std::size_t variant_window_tokens{0};
  VariantStrand variant_strand{VariantStrand::kBoth};
  VariantNormalization variant_normalization{VariantNormalization::kSum};
  std::string server_host{"127.0.0.1"};
  std::uint16_t server_port{8080};
  std::size_t server_max_queue{64};
  std::size_t server_max_batch{4};
  std::size_t server_batch_window_ms{2};
  std::size_t server_max_request_bytes{1U << 20U};
  std::size_t server_max_sequence_bytes{0};
  std::size_t server_max_embedding_values{1U << 20U};
  ExecutionBackend backend{ExecutionBackend::kAuto};
  InferenceProfile inference_profile{InferenceProfile::kExact};
  bool profile_explicit{false};
  std::size_t generated_tokens{0};
  std::size_t context_size{8192};
  bool context_size_explicit{false};
  std::optional<std::size_t> force_prompt_threshold;
  std::optional<std::size_t> gpu_layers;
  std::vector<int> gpu_ids;
  SamplingConfig sampling;
  bool dump_tokens{false};
  std::optional<std::string> dump_logits_path;
  std::optional<DumpLayerSpec> dump_layer;
};

[[nodiscard]] std::string_view cli_usage() noexcept;
[[nodiscard]] Status parse_cli(int argc, char *const argv[],
                               CliOptions *options);

} // namespace evo
