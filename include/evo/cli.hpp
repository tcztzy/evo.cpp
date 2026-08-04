// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "evo/sampler.hpp"
#include "evo/status.hpp"

namespace evo {

enum class RunMode { kGenerate, kScore };

struct DumpLayerSpec final {
  std::size_t layer{0};
  std::string path;
};

struct CliOptions final {
  RunMode mode{RunMode::kGenerate};
  std::string model_path;
  std::string prompt;
  std::string score_path;
  std::size_t generated_tokens{0};
  std::size_t context_size{8192};
  std::optional<std::size_t> force_prompt_threshold;
  std::vector<int> gpu_ids;
  SamplingConfig sampling;
  bool dump_tokens{false};
  std::optional<std::string> dump_logits_path;
  std::optional<DumpLayerSpec> dump_layer;
};

[[nodiscard]] std::string_view cli_usage() noexcept;
[[nodiscard]] Status parse_cli(int argc, char* const argv[], CliOptions* options);

}  // namespace evo
