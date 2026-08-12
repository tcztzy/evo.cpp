// SPDX-License-Identifier: Apache-2.0
#include "evo/cli.hpp"

#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <string_view>

namespace evo {
namespace {

template <typename Integer>
bool parse_unsigned(const std::string_view text, Integer *const value) {
  std::uint64_t parsed = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      parsed >
          static_cast<std::uint64_t>(std::numeric_limits<Integer>::max())) {
    return false;
  }
  *value = static_cast<Integer>(parsed);
  return true;
}

bool parse_float(const std::string_view text, float *const value) {
  if (text.empty() ||
      std::isspace(static_cast<unsigned char>(text.front())) != 0 ||
      std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    return false;
  }
  const std::string owned{text};
  char *end = nullptr;
  errno = 0;
  const float parsed = std::strtof(owned.c_str(), &end);
  if (errno == ERANGE || end != owned.c_str() + owned.size() ||
      !std::isfinite(parsed)) {
    return false;
  }
  *value = parsed;
  return true;
}

Status duplicate(const std::string_view option) {
  return {ErrorCode::kInvalidArgument,
          "option '" + std::string{option} + "' was specified twice"};
}

Status value_after(const int argc, char *const argv[], int *const index,
                   const std::string_view option,
                   std::string_view *const value) {
  if (*index + 1 >= argc) {
    return {ErrorCode::kInvalidArgument,
            "option '" + std::string{option} + "' requires a value"};
  }
  ++*index;
  *value = argv[*index];
  return Status::Ok();
}

Status parse_gpu_list(const std::string_view text,
                      std::vector<int> *const ids) {
  std::set<int> unique;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto comma = text.find(',', start);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    const auto part = text.substr(start, end - start);
    int id = 0;
    if (part.empty() || !parse_unsigned(part, &id) || id > 63) {
      return {
          ErrorCode::kInvalidArgument,
          "--gpu must be 1 to 4 unique IDs in [0, 63], for example 0,1,2,3"};
    }
    if (!unique.insert(id).second) {
      return {ErrorCode::kInvalidArgument,
              "--gpu contains duplicate ID " + std::to_string(id)};
    }
    ids->push_back(id);
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  if (ids->empty() || ids->size() > 4) {
    return {ErrorCode::kInvalidArgument,
            "--gpu requires between 1 and 4 device IDs"};
  }
  return Status::Ok();
}

} // namespace

std::string_view cli_usage() noexcept {
  return "evo - native Evo 2 inference engine (1B, 7B, 20B, 40B)\n\n"
         "Usage:\n"
         "  evo --help\n"
         "  evo --version\n"
         "  evo -m MODEL.safetensors[.index.json] -p DNA -n TOKENS --ctx N "
         "--gpu 0,1,2,3 [sampling]\n"
         "  evo -m MODEL.safetensors[.index.json] --score "
         "FASTA_FASTQ_OR_TEXT --gpu "
         "0,1,2,3\n"
         "  evo embed -m MODEL --input FASTA_FASTQ_OR_TEXT --output DIR "
         "--layer N "
         "--pooling none|mean|last --gpu 0,1,2,3\n"
         "  evo variant-score -m MODEL --sequence DNA --position POS --ref "
         "REF --alt ALT --window N --strand forward|reverse|both "
         "--normalization sum|mean --gpu 0,1,2,3\n"
         "  evo variant-score -m MODEL --vcf VARIANTS.vcf[.gz] --reference "
         "REFERENCE.fa[.gz] --window N --gpu 0,1,2,3\n\n"
         "  evo serve -m MODEL --ctx N --gpu 0,1,2,3 [server options]\n\n"
         "Server:\n"
         "  --host ADDRESS              IPv4 bind address (default: "
         "127.0.0.1)\n"
         "  --port N                    TCP port; 0 selects a free port "
         "(default: 8080)\n"
         "  --max-queue N               Pending request limit (default: "
         "64)\n"
         "  --max-batch N               Isolated contexts per microbatch "
         "(default: 4)\n"
         "  --batch-window-ms N         Coalescing window in milliseconds "
         "(default: 2)\n"
         "  --max-request-bytes N       HTTP body limit (default: 1048576)\n"
         "  --max-sequence-bytes N      Per-sequence limit (default: --ctx)\n"
         "  --max-embedding-values N    Embedding response value limit "
         "(default: 1048576)\n\n"
         "Execution profile:\n"
         "  --backend auto|cpu|cuda     Runtime backend (default: auto)\n"
         "  --gpu-layers N              Explicit CUDA prefix + CPU suffix\n"
         "  --profile exact|fast-q8-kv|cpu-f32\n"
         "                 Explicit arithmetic/cache semantics\n\n"
         "Sampling:\n"
         "  --temp F       Temperature > 0 (default: 1)\n"
         "  --top-k K      Keep K logits; 0 disables, 1 is greedy (default: "
         "1)\n"
         "  --top-p P      Nucleus probability in (0,1] (default: 1)\n"
         "  --seed N       Reproducible unsigned 64-bit seed (default: 0)\n\n"
         "Prompt execution:\n"
         "  Prompts are split into bounded activation chunks automatically\n"
         "  --force-prompt-threshold N\n"
         "                 Chunk-prefill N bytes, then teacher-force the "
         "rest\n\n"
         "Debug:\n"
         "  --dump-tokens\n"
         "  --dump-logits PATH\n"
         "  --dump-layer INDEX:PATH\n";
}

Status parse_cli(const int argc, char *const argv[],
                 CliOptions *const options) {
  if (options == nullptr) {
    return {ErrorCode::kInvalidArgument, "CLI output pointer is null"};
  }
  *options = CliOptions{};
  const bool embed_command = argc > 1 && std::string_view{argv[1]} == "embed";
  const bool variant_command =
      argc > 1 && std::string_view{argv[1]} == "variant-score";
  const bool server_command = argc > 1 && std::string_view{argv[1]} == "serve";
  if (embed_command) {
    options->mode = RunMode::kEmbed;
  } else if (variant_command) {
    options->mode = RunMode::kVariantScore;
  } else if (server_command) {
    options->mode = RunMode::kServe;
  }
  bool seen_model = false;
  bool seen_prompt = false;
  bool seen_score = false;
  bool seen_tokens = false;
  bool seen_context = false;
  bool seen_force_prompt_threshold = false;
  bool seen_gpu = false;
  bool seen_gpu_layers = false;
  bool seen_temperature = false;
  bool seen_top_k = false;
  bool seen_top_p = false;
  bool seen_seed = false;
  bool seen_dump_tokens = false;
  bool seen_embed_input = false;
  bool seen_embed_output = false;
  bool seen_embed_layer = false;
  bool seen_embedding_pooling = false;
  bool seen_variant_sequence = false;
  bool seen_variant_vcf = false;
  bool seen_variant_reference_path = false;
  bool seen_variant_position = false;
  bool seen_variant_reference = false;
  bool seen_variant_alternate = false;
  bool seen_variant_window = false;
  bool seen_variant_strand = false;
  bool seen_variant_normalization = false;
  bool seen_server_host = false;
  bool seen_server_port = false;
  bool seen_server_max_queue = false;
  bool seen_server_max_batch = false;
  bool seen_server_batch_window = false;
  bool seen_server_max_request = false;
  bool seen_server_max_sequence = false;
  bool seen_server_max_embedding = false;
  bool seen_profile = false;
  bool seen_backend = false;

  const int option_start =
      embed_command || variant_command || server_command ? 2 : 1;
  for (int index = option_start; index < argc; ++index) {
    const std::string_view option{argv[index]};
    std::string_view value;
    auto status = Status::Ok();
    if (option == "-m" || option == "--model") {
      if (seen_model)
        return duplicate(option);
      seen_model = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->model_path = value;
    } else if (option == "-p" || option == "--prompt") {
      if (seen_prompt)
        return duplicate(option);
      seen_prompt = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->prompt = value;
    } else if (option == "--score") {
      if (seen_score)
        return duplicate(option);
      seen_score = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->score_path = value;
    } else if (option == "--input") {
      if (seen_embed_input)
        return duplicate(option);
      seen_embed_input = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->embed_path = value;
    } else if (option == "--output") {
      if (seen_embed_output)
        return duplicate(option);
      seen_embed_output = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->embed_output_dir = value;
    } else if (option == "--layer") {
      if (seen_embed_layer)
        return duplicate(option);
      seen_embed_layer = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->embed_layer)) {
        return {ErrorCode::kInvalidArgument,
                "--layer must be a nonnegative integer; the model validates "
                "its range"};
      }
    } else if (option == "--pooling") {
      if (seen_embedding_pooling)
        return duplicate(option);
      seen_embedding_pooling = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (value == "none") {
        options->embedding_pooling = EmbeddingPooling::kNone;
      } else if (value == "mean") {
        options->embedding_pooling = EmbeddingPooling::kMean;
      } else if (value == "last") {
        options->embedding_pooling = EmbeddingPooling::kLast;
      } else {
        return {ErrorCode::kInvalidArgument,
                "--pooling must be one of none, mean, or last"};
      }
    } else if (option == "--sequence") {
      if (seen_variant_sequence)
        return duplicate(option);
      seen_variant_sequence = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->variant_sequence = value;
    } else if (option == "--vcf") {
      if (seen_variant_vcf)
        return duplicate(option);
      seen_variant_vcf = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->variant_vcf_path = value;
    } else if (option == "--reference") {
      if (seen_variant_reference_path)
        return duplicate(option);
      seen_variant_reference_path = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->variant_reference_path = value;
    } else if (option == "--position") {
      if (seen_variant_position)
        return duplicate(option);
      seen_variant_position = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->variant_position_1based) ||
          options->variant_position_1based == 0) {
        return {ErrorCode::kInvalidArgument,
                "--position must be a positive 1-based integer"};
      }
    } else if (option == "--ref") {
      if (seen_variant_reference)
        return duplicate(option);
      seen_variant_reference = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->variant_reference = value;
    } else if (option == "--alt") {
      if (seen_variant_alternate)
        return duplicate(option);
      seen_variant_alternate = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      options->variant_alternate = value;
    } else if (option == "--window") {
      if (seen_variant_window)
        return duplicate(option);
      seen_variant_window = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->variant_window_tokens) ||
          options->variant_window_tokens == 0 ||
          options->variant_window_tokens > 1'048'576) {
        return {ErrorCode::kInvalidArgument,
                "--window must be an integer in [1, 1048576]"};
      }
    } else if (option == "--strand") {
      if (seen_variant_strand)
        return duplicate(option);
      seen_variant_strand = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (value == "forward") {
        options->variant_strand = VariantStrand::kForward;
      } else if (value == "reverse") {
        options->variant_strand = VariantStrand::kReverse;
      } else if (value == "both") {
        options->variant_strand = VariantStrand::kBoth;
      } else {
        return {ErrorCode::kInvalidArgument,
                "--strand must be one of forward, reverse, or both"};
      }
    } else if (option == "--normalization") {
      if (seen_variant_normalization)
        return duplicate(option);
      seen_variant_normalization = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (value == "sum") {
        options->variant_normalization = VariantNormalization::kSum;
      } else if (value == "mean") {
        options->variant_normalization = VariantNormalization::kMean;
      } else {
        return {ErrorCode::kInvalidArgument,
                "--normalization must be one of sum or mean"};
      }
    } else if (option == "--host") {
      if (seen_server_host)
        return duplicate(option);
      seen_server_host = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (value.empty())
        return {ErrorCode::kInvalidArgument, "--host must not be empty"};
      options->server_host = value;
    } else if (option == "--port") {
      if (seen_server_port)
        return duplicate(option);
      seen_server_port = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->server_port))
        return {ErrorCode::kInvalidArgument,
                "--port must be an integer in [0, 65535]"};
    } else if (option == "--max-queue") {
      if (seen_server_max_queue)
        return duplicate(option);
      seen_server_max_queue = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->server_max_queue) ||
          options->server_max_queue == 0 || options->server_max_queue > 4096) {
        return {ErrorCode::kInvalidArgument,
                "--max-queue must be an integer in [1, 4096]"};
      }
    } else if (option == "--max-batch") {
      if (seen_server_max_batch)
        return duplicate(option);
      seen_server_max_batch = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->server_max_batch) ||
          options->server_max_batch == 0 || options->server_max_batch > 256) {
        return {ErrorCode::kInvalidArgument,
                "--max-batch must be an integer in [1, 256]"};
      }
    } else if (option == "--batch-window-ms") {
      if (seen_server_batch_window)
        return duplicate(option);
      seen_server_batch_window = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->server_batch_window_ms) ||
          options->server_batch_window_ms > 1000) {
        return {ErrorCode::kInvalidArgument,
                "--batch-window-ms must be an integer in [0, 1000]"};
      }
    } else if (option == "--max-request-bytes") {
      if (seen_server_max_request)
        return duplicate(option);
      seen_server_max_request = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->server_max_request_bytes) ||
          options->server_max_request_bytes < 256 ||
          options->server_max_request_bytes > (64U << 20U)) {
        return {ErrorCode::kInvalidArgument,
                "--max-request-bytes must be an integer in [256, 67108864]"};
      }
    } else if (option == "--max-sequence-bytes") {
      if (seen_server_max_sequence)
        return duplicate(option);
      seen_server_max_sequence = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->server_max_sequence_bytes) ||
          options->server_max_sequence_bytes < 2 ||
          options->server_max_sequence_bytes > 1'048'576) {
        return {ErrorCode::kInvalidArgument,
                "--max-sequence-bytes must be an integer in [2, 1048576]"};
      }
    } else if (option == "--max-embedding-values") {
      if (seen_server_max_embedding)
        return duplicate(option);
      seen_server_max_embedding = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->server_max_embedding_values) ||
          options->server_max_embedding_values == 0 ||
          options->server_max_embedding_values > (1U << 28U)) {
        return {ErrorCode::kInvalidArgument,
                "--max-embedding-values must be an integer in [1, 268435456]"};
      }
    } else if (option == "--backend") {
      if (seen_backend)
        return duplicate(option);
      seen_backend = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      status = parse_execution_backend(value, &options->backend);
      if (!status.ok())
        return status;
    } else if (option == "--profile") {
      if (seen_profile)
        return duplicate(option);
      seen_profile = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      status = parse_inference_profile(value, &options->inference_profile);
      if (!status.ok())
        return status;
      options->profile_explicit = true;
    } else if (option == "-n" || option == "--tokens") {
      if (seen_tokens)
        return duplicate(option);
      seen_tokens = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->generated_tokens) ||
          options->generated_tokens == 0) {
        return {ErrorCode::kInvalidArgument,
                "--tokens must be a positive integer"};
      }
      if (options->generated_tokens > 1'048'576) {
        return {ErrorCode::kInvalidArgument,
                "--tokens must not exceed 1048576"};
      }
    } else if (option == "--ctx") {
      if (seen_context)
        return duplicate(option);
      seen_context = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->context_size) ||
          options->context_size == 0 || options->context_size > 1'048'576) {
        return {ErrorCode::kInvalidArgument,
                "--ctx must be an integer in [1, 1048576]"};
      }
    } else if (option == "--force-prompt-threshold") {
      if (seen_force_prompt_threshold)
        return duplicate(option);
      seen_force_prompt_threshold = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      std::size_t threshold = 0;
      if (!parse_unsigned(value, &threshold) || threshold == 0 ||
          threshold > 1'048'576) {
        return {ErrorCode::kInvalidArgument,
                "--force-prompt-threshold must be an integer in [1, 1048576]"};
      }
      options->force_prompt_threshold = threshold;
    } else if (option == "--gpu") {
      if (seen_gpu)
        return duplicate(option);
      seen_gpu = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      status = parse_gpu_list(value, &options->gpu_ids);
      if (!status.ok())
        return status;
    } else if (option == "--gpu-layers") {
      if (seen_gpu_layers)
        return duplicate(option);
      seen_gpu_layers = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      std::size_t layers = 0;
      if (!parse_unsigned(value, &layers) || layers == 0 || layers > 4096)
        return {ErrorCode::kInvalidArgument,
                "--gpu-layers must be an integer in [1, 4096]"};
      options->gpu_layers = layers;
    } else if (option == "--temp") {
      if (seen_temperature)
        return duplicate(option);
      seen_temperature = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_float(value, &options->sampling.temperature)) {
        return {ErrorCode::kInvalidArgument, "--temp must be a finite number"};
      }
    } else if (option == "--top-k") {
      if (seen_top_k)
        return duplicate(option);
      seen_top_k = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->sampling.top_k)) {
        return {ErrorCode::kInvalidArgument,
                "--top-k must be a nonnegative integer"};
      }
    } else if (option == "--top-p") {
      if (seen_top_p)
        return duplicate(option);
      seen_top_p = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_float(value, &options->sampling.top_p)) {
        return {ErrorCode::kInvalidArgument, "--top-p must be a finite number"};
      }
    } else if (option == "--seed") {
      if (seen_seed)
        return duplicate(option);
      seen_seed = true;
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (!parse_unsigned(value, &options->sampling.seed)) {
        return {ErrorCode::kInvalidArgument,
                "--seed must be an unsigned 64-bit integer"};
      }
    } else if (option == "--dump-tokens") {
      if (seen_dump_tokens)
        return duplicate(option);
      seen_dump_tokens = true;
      options->dump_tokens = true;
    } else if (option == "--dump-logits") {
      if (options->dump_logits_path.has_value())
        return duplicate(option);
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      if (value.empty()) {
        return {ErrorCode::kInvalidArgument,
                "--dump-logits path must not be empty"};
      }
      options->dump_logits_path = std::string{value};
    } else if (option == "--dump-layer") {
      if (options->dump_layer.has_value())
        return duplicate(option);
      status = value_after(argc, argv, &index, option, &value);
      if (!status.ok())
        return status;
      const auto colon = value.find(':');
      std::size_t layer = 0;
      if (colon == std::string_view::npos || colon + 1 == value.size() ||
          !parse_unsigned(value.substr(0, colon), &layer)) {
        return {ErrorCode::kInvalidArgument,
                "--dump-layer must be a nonnegative INDEX:PATH; the model "
                "validates INDEX"};
      }
      options->dump_layer =
          DumpLayerSpec{layer, std::string{value.substr(colon + 1)}};
    } else {
      return {ErrorCode::kInvalidArgument,
              "unknown option '" + std::string{option} + "'"};
    }
  }

  if (!seen_model || options->model_path.empty()) {
    return {ErrorCode::kInvalidArgument,
            "a nonempty model path is required with -m MODEL"};
  }
  if (options->backend == ExecutionBackend::kCpu) {
    if (seen_gpu || seen_gpu_layers)
      return {ErrorCode::kInvalidArgument,
              "--backend cpu cannot be combined with GPU options"};
    if (seen_profile && options->inference_profile != InferenceProfile::kCpuF32)
      return {
          ErrorCode::kInvalidArgument,
          "--backend cpu requires --profile cpu-f32 when profile is explicit"};
    options->inference_profile = InferenceProfile::kCpuF32;
  } else if (seen_gpu_layers) {
    if (!seen_gpu)
      return {ErrorCode::kInvalidArgument,
              "--gpu-layers requires an explicit --gpu device list"};
    if (seen_profile && options->inference_profile != InferenceProfile::kCpuF32)
      return {
          ErrorCode::kInvalidArgument,
          "hybrid offload requires --profile cpu-f32 when profile is explicit"};
    options->inference_profile = InferenceProfile::kCpuF32;
  } else if (options->inference_profile == InferenceProfile::kCpuF32) {
    return {ErrorCode::kInvalidArgument,
            "--profile cpu-f32 requires --backend cpu"};
  }
  const bool requires_gpu = options->backend != ExecutionBackend::kCpu;
  const bool seen_server_option =
      seen_server_host || seen_server_port || seen_server_max_queue ||
      seen_server_max_batch || seen_server_batch_window ||
      seen_server_max_request || seen_server_max_sequence ||
      seen_server_max_embedding;
  if (server_command) {
    if (seen_prompt || seen_score || seen_tokens ||
        seen_force_prompt_threshold || seen_temperature || seen_top_k ||
        seen_top_p || seen_seed || seen_dump_tokens ||
        options->dump_logits_path.has_value() ||
        options->dump_layer.has_value() || seen_embed_input ||
        seen_embed_output || seen_embed_layer || seen_embedding_pooling ||
        seen_variant_sequence || seen_variant_position ||
        seen_variant_vcf || seen_variant_reference_path ||
        seen_variant_reference || seen_variant_alternate ||
        seen_variant_window || seen_variant_strand ||
        seen_variant_normalization || seen_gpu_layers) {
      return {ErrorCode::kInvalidArgument,
              "serve accepts server options, --ctx, and --gpu only"};
    }
    if (!seen_gpu && requires_gpu)
      return {ErrorCode::kInvalidArgument, "--gpu is required"};
    if (options->server_max_batch > options->server_max_queue) {
      return {ErrorCode::kInvalidArgument,
              "--max-batch must not exceed --max-queue"};
    }
    if (!seen_server_max_sequence)
      options->server_max_sequence_bytes = options->context_size;
    if (options->server_max_sequence_bytes > options->context_size) {
      return {ErrorCode::kInvalidArgument,
              "--max-sequence-bytes must not exceed --ctx"};
    }
    return Status::Ok();
  }
  if (seen_server_option) {
    return {ErrorCode::kInvalidArgument, "server options require 'evo serve'"};
  }
  if (embed_command) {
    if (seen_prompt || seen_score || seen_tokens ||
        seen_force_prompt_threshold || seen_temperature || seen_top_k ||
        seen_top_p || seen_seed || options->dump_logits_path.has_value() ||
        options->dump_layer.has_value() || seen_variant_sequence ||
        seen_variant_vcf || seen_variant_reference_path ||
        seen_variant_position || seen_variant_reference ||
        seen_variant_alternate || seen_variant_window || seen_variant_strand ||
        seen_variant_normalization) {
      return {ErrorCode::kInvalidArgument,
              "embed accepts --input/--output/--layer/--pooling, --ctx, "
              "--gpu, and --dump-tokens; generation/scoring options are "
              "invalid"};
    }
    if (!seen_embed_input || options->embed_path.empty()) {
      return {ErrorCode::kInvalidArgument,
              "embed requires a nonempty --input path"};
    }
    if (!seen_embed_output || options->embed_output_dir.empty()) {
      return {ErrorCode::kInvalidArgument,
              "embed requires a nonempty --output directory"};
    }
    if (!seen_embed_layer) {
      return {ErrorCode::kInvalidArgument, "embed requires --layer INDEX"};
    }
    if (!seen_gpu && requires_gpu)
      return {ErrorCode::kInvalidArgument, "--gpu is required"};
    return Status::Ok();
  }
  if (variant_command) {
    if (seen_prompt || seen_score || seen_tokens ||
        seen_force_prompt_threshold || seen_temperature || seen_top_k ||
        seen_top_p || seen_seed || seen_embed_input || seen_embed_output ||
        seen_embed_layer || seen_embedding_pooling ||
        options->dump_logits_path.has_value() ||
        options->dump_layer.has_value()) {
      return {ErrorCode::kInvalidArgument,
              "variant-score accepts variant arguments, --ctx, --gpu, and "
              "--dump-tokens; generation/scoring/embedding options are "
              "invalid"};
    }
    const bool vcf_mode = seen_variant_vcf || seen_variant_reference_path;
    if (vcf_mode) {
      if (!seen_variant_vcf || options->variant_vcf_path.empty() ||
          !seen_variant_reference_path ||
          options->variant_reference_path.empty()) {
        return {ErrorCode::kInvalidArgument,
                "VCF variant scoring requires nonempty --vcf and --reference "
                "paths"};
      }
      if (seen_variant_sequence || seen_variant_position ||
          seen_variant_reference || seen_variant_alternate) {
        return {ErrorCode::kInvalidArgument,
                "--vcf/--reference cannot be combined with inline "
                "--sequence/--position/--ref/--alt"};
      }
    } else {
      if (!seen_variant_sequence || options->variant_sequence.empty()) {
        return {ErrorCode::kInvalidArgument,
                "variant-score requires a nonempty --sequence or --vcf"};
      }
      if (!seen_variant_position) {
        return {ErrorCode::kInvalidArgument,
                "variant-score requires --position POS"};
      }
      if (!seen_variant_reference || options->variant_reference.empty()) {
        return {ErrorCode::kInvalidArgument,
                "variant-score requires a nonempty --ref allele"};
      }
      if (!seen_variant_alternate || options->variant_alternate.empty()) {
        return {ErrorCode::kInvalidArgument,
                "variant-score requires a nonempty --alt allele"};
      }
    }
    if (!seen_gpu && requires_gpu)
      return {ErrorCode::kInvalidArgument, "--gpu is required"};
    if (!seen_variant_window)
      options->variant_window_tokens = options->context_size;
    if (options->variant_window_tokens > options->context_size) {
      return {ErrorCode::kInvalidArgument, "--window must not exceed --ctx"};
    }
    return Status::Ok();
  }
  if (seen_embed_input || seen_embed_output || seen_embed_layer ||
      seen_embedding_pooling) {
    return {ErrorCode::kInvalidArgument,
            "--input, --output, --layer, and --pooling require 'evo embed'"};
  }
  if (seen_variant_sequence || seen_variant_position ||
      seen_variant_vcf || seen_variant_reference_path ||
      seen_variant_reference || seen_variant_alternate || seen_variant_window ||
      seen_variant_strand || seen_variant_normalization) {
    return {ErrorCode::kInvalidArgument,
            "variant arguments require 'evo variant-score'"};
  }
  if (seen_prompt == seen_score) {
    return {ErrorCode::kInvalidArgument,
            "specify exactly one of -p PROMPT or --score INPUT"};
  }
  if (!seen_gpu && requires_gpu) {
    return {ErrorCode::kInvalidArgument, "--gpu is required"};
  }
  if (seen_prompt) {
    options->mode = RunMode::kGenerate;
    if (options->prompt.empty()) {
      return {ErrorCode::kInvalidArgument,
              "generation prompt must not be empty"};
    }
    if (!seen_tokens) {
      return {ErrorCode::kInvalidArgument, "generation requires -n TOKENS"};
    }
  } else {
    options->mode = RunMode::kScore;
    if (options->score_path.empty()) {
      return {ErrorCode::kInvalidArgument,
              "score input path must not be empty"};
    }
    if (seen_tokens) {
      return {ErrorCode::kInvalidArgument,
              "--tokens is only valid for generation"};
    }
    if (seen_temperature || seen_top_k || seen_top_p || seen_seed) {
      return {ErrorCode::kInvalidArgument,
              "sampling options are only valid for generation"};
    }
    if (seen_force_prompt_threshold) {
      return {ErrorCode::kInvalidArgument,
              "--force-prompt-threshold is only valid for generation"};
    }
  }
  return validate_sampling_config(options->sampling);
}

} // namespace evo
