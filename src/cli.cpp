// SPDX-License-Identifier: Apache-2.0
#include "evo2c/cli.hpp"

#include <cerrno>
#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <string_view>

namespace evo2c {
namespace {

template <typename Integer>
bool parse_unsigned(const std::string_view text, Integer* const value) {
  std::uint64_t parsed = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      parsed > static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
    return false;
  }
  *value = static_cast<Integer>(parsed);
  return true;
}

bool parse_float(const std::string_view text, float* const value) {
  if (text.empty() || std::isspace(static_cast<unsigned char>(text.front())) != 0 ||
      std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    return false;
  }
  const std::string owned{text};
  char* end = nullptr;
  errno = 0;
  const float parsed = std::strtof(owned.c_str(), &end);
  if (errno == ERANGE || end != owned.c_str() + owned.size() || !std::isfinite(parsed)) {
    return false;
  }
  *value = parsed;
  return true;
}

Status duplicate(const std::string_view option) {
  return {ErrorCode::kInvalidArgument, "option '" + std::string{option} + "' was specified twice"};
}

Status value_after(const int argc,
                   char* const argv[],
                   int* const index,
                   const std::string_view option,
                   std::string_view* const value) {
  if (*index + 1 >= argc) {
    return {ErrorCode::kInvalidArgument,
            "option '" + std::string{option} + "' requires a value"};
  }
  ++*index;
  *value = argv[*index];
  return Status::Ok();
}

Status parse_gpu_list(const std::string_view text, std::vector<int>* const ids) {
  std::set<int> unique;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto comma = text.find(',', start);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    const auto part = text.substr(start, end - start);
    int id = 0;
    if (part.empty() || !parse_unsigned(part, &id) || id > 63) {
      return {ErrorCode::kInvalidArgument,
              "--gpu must be 1 to 4 unique IDs in [0, 63], for example 0,1,2,3"};
    }
    if (!unique.insert(id).second) {
      return {ErrorCode::kInvalidArgument, "--gpu contains duplicate ID " + std::to_string(id)};
    }
    ids->push_back(id);
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  if (ids->empty() || ids->size() > 4) {
    return {ErrorCode::kInvalidArgument, "--gpu requires between 1 and 4 device IDs"};
  }
  return Status::Ok();
}

}  // namespace

std::string_view cli_usage() noexcept {
  return "evo2c - native Evo 2 inference engine (1B, 7B, 20B, 40B)\n\n"
         "Usage:\n"
         "  evo2c --help\n"
         "  evo2c --version\n"
         "  evo2c -m MODEL -p DNA -n TOKENS --ctx N --gpu 0,1,2,3 [sampling]\n"
         "  evo2c -m MODEL --score FASTA_OR_TEXT --gpu 0,1,2,3\n\n"
         "Sampling:\n"
         "  --temp F       Temperature > 0 (default: 1)\n"
         "  --top-k K      Keep K logits; 0 disables, 1 is greedy (default: 1)\n"
         "  --top-p P      Nucleus probability in (0,1] (default: 1)\n"
         "  --seed N       Reproducible unsigned 64-bit seed (default: 0)\n\n"
         "Prompt execution:\n"
         "  Prompts are split into bounded activation chunks automatically\n"
         "  --force-prompt-threshold N\n"
         "                 Chunk-prefill N bytes, then teacher-force the rest\n\n"
         "Debug:\n"
         "  --dump-tokens\n"
         "  --dump-logits PATH\n"
         "  --dump-layer INDEX:PATH\n";
}

Status parse_cli(const int argc, char* const argv[], CliOptions* const options) {
  if (options == nullptr) {
    return {ErrorCode::kInvalidArgument, "CLI output pointer is null"};
  }
  *options = CliOptions{};
  bool seen_model = false;
  bool seen_prompt = false;
  bool seen_score = false;
  bool seen_tokens = false;
  bool seen_context = false;
  bool seen_force_prompt_threshold = false;
  bool seen_gpu = false;
  bool seen_temperature = false;
  bool seen_top_k = false;
  bool seen_top_p = false;
  bool seen_seed = false;
  bool seen_dump_tokens = false;

  for (int index = 1; index < argc; ++index) {
    const std::string_view option{argv[index]};
    std::string_view value;
    auto status = Status::Ok();
    if (option == "-m" || option == "--model") {
      if (seen_model) return duplicate(option);
      seen_model = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      options->model_path = value;
    } else if (option == "-p" || option == "--prompt") {
      if (seen_prompt) return duplicate(option);
      seen_prompt = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      options->prompt = value;
    } else if (option == "--score") {
      if (seen_score) return duplicate(option);
      seen_score = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      options->score_path = value;
    } else if (option == "-n" || option == "--tokens") {
      if (seen_tokens) return duplicate(option);
      seen_tokens = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      if (!parse_unsigned(value, &options->generated_tokens) || options->generated_tokens == 0) {
        return {ErrorCode::kInvalidArgument, "--tokens must be a positive integer"};
      }
      if (options->generated_tokens > 1'048'576) {
        return {ErrorCode::kInvalidArgument, "--tokens must not exceed 1048576"};
      }
    } else if (option == "--ctx") {
      if (seen_context) return duplicate(option);
      seen_context = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      if (!parse_unsigned(value, &options->context_size) || options->context_size == 0 ||
          options->context_size > 1'048'576) {
        return {ErrorCode::kInvalidArgument, "--ctx must be an integer in [1, 1048576]"};
      }
    } else if (option == "--force-prompt-threshold") {
      if (seen_force_prompt_threshold) return duplicate(option);
      seen_force_prompt_threshold = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      std::size_t threshold = 0;
      if (!parse_unsigned(value, &threshold) || threshold == 0 ||
          threshold > 1'048'576) {
        return {ErrorCode::kInvalidArgument,
                "--force-prompt-threshold must be an integer in [1, 1048576]"};
      }
      options->force_prompt_threshold = threshold;
    } else if (option == "--gpu") {
      if (seen_gpu) return duplicate(option);
      seen_gpu = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      status = parse_gpu_list(value, &options->gpu_ids);
      if (!status.ok()) return status;
    } else if (option == "--temp") {
      if (seen_temperature) return duplicate(option);
      seen_temperature = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      if (!parse_float(value, &options->sampling.temperature)) {
        return {ErrorCode::kInvalidArgument, "--temp must be a finite number"};
      }
    } else if (option == "--top-k") {
      if (seen_top_k) return duplicate(option);
      seen_top_k = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      if (!parse_unsigned(value, &options->sampling.top_k)) {
        return {ErrorCode::kInvalidArgument, "--top-k must be a nonnegative integer"};
      }
    } else if (option == "--top-p") {
      if (seen_top_p) return duplicate(option);
      seen_top_p = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      if (!parse_float(value, &options->sampling.top_p)) {
        return {ErrorCode::kInvalidArgument, "--top-p must be a finite number"};
      }
    } else if (option == "--seed") {
      if (seen_seed) return duplicate(option);
      seen_seed = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      if (!parse_unsigned(value, &options->sampling.seed)) {
        return {ErrorCode::kInvalidArgument, "--seed must be an unsigned 64-bit integer"};
      }
    } else if (option == "--dump-tokens") {
      if (seen_dump_tokens) return duplicate(option);
      seen_dump_tokens = true;
      options->dump_tokens = true;
    } else if (option == "--dump-logits") {
      if (options->dump_logits_path.has_value()) return duplicate(option);
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      if (value.empty()) {
        return {ErrorCode::kInvalidArgument, "--dump-logits path must not be empty"};
      }
      options->dump_logits_path = std::string{value};
    } else if (option == "--dump-layer") {
      if (options->dump_layer.has_value()) return duplicate(option);
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok()) return status;
      const auto colon = value.find(':');
      std::size_t layer = 0;
      if (colon == std::string_view::npos || colon + 1 == value.size() ||
          !parse_unsigned(value.substr(0, colon), &layer)) {
        return {ErrorCode::kInvalidArgument,
                "--dump-layer must be a nonnegative INDEX:PATH; the model validates INDEX"};
      }
      options->dump_layer = DumpLayerSpec{layer, std::string{value.substr(colon + 1)}};
    } else {
      return {ErrorCode::kInvalidArgument, "unknown option '" + std::string{option} + "'"};
    }
  }

  if (!seen_model || options->model_path.empty()) {
    return {ErrorCode::kInvalidArgument, "a nonempty model path is required with -m MODEL"};
  }
  if (seen_prompt == seen_score) {
    return {ErrorCode::kInvalidArgument, "specify exactly one of -p PROMPT or --score INPUT"};
  }
  if (!seen_gpu) {
    return {ErrorCode::kInvalidArgument, "--gpu is required"};
  }
  if (seen_prompt) {
    options->mode = RunMode::kGenerate;
    if (options->prompt.empty()) {
      return {ErrorCode::kInvalidArgument, "generation prompt must not be empty"};
    }
    if (!seen_tokens) {
      return {ErrorCode::kInvalidArgument, "generation requires -n TOKENS"};
    }
  } else {
    options->mode = RunMode::kScore;
    if (options->score_path.empty()) {
      return {ErrorCode::kInvalidArgument, "score input path must not be empty"};
    }
    if (seen_tokens) {
      return {ErrorCode::kInvalidArgument, "--tokens is only valid for generation"};
    }
    if (seen_temperature || seen_top_k || seen_top_p || seen_seed) {
      return {ErrorCode::kInvalidArgument, "sampling options are only valid for generation"};
    }
    if (seen_force_prompt_threshold) {
      return {ErrorCode::kInvalidArgument,
              "--force-prompt-threshold is only valid for generation"};
    }
  }
  return validate_sampling_config(options->sampling);
}

}  // namespace evo2c
