// SPDX-License-Identifier: Apache-2.0
#include "evo/cuda/inference_cli.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

#include "evo/cuda/model.hpp"
#include "evo/cuda/runtime.hpp"
#include "evo/model_format.hpp"
#include "evo/npy.hpp"
#include "evo/sampler.hpp"
#include "evo/sequence_io.hpp"
#include "evo/tokenizer.hpp"
#include "evo/variant.hpp"
#include "evo/variant_io.hpp"

namespace evo::cuda {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct DeviceMemory final {
  int device{-1};
  std::size_t baseline_used_bytes{0};
  std::size_t observed_peak_delta_bytes{0};
  std::size_t total_bytes{0};
};

class MemoryTracker final {
public:
  [[nodiscard]] Status initialize(const std::vector<int> &devices) {
    entries_.clear();
    entries_.reserve(devices.size());
    for (const int device : devices) {
      auto status = select_device(device);
      if (!status.ok()) {
        return status;
      }
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      status = cuda_status(cudaMemGetInfo(&free_bytes, &total_bytes),
                           "cudaMemGetInfo before model load");
      if (!status.ok()) {
        return status;
      }
      entries_.push_back({device, total_bytes - free_bytes, 0, total_bytes});
    }
    return Status::Ok();
  }

  [[nodiscard]] Status observe() {
    for (auto &entry : entries_) {
      auto status = select_device(entry.device);
      if (!status.ok()) {
        return status;
      }
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      status = cuda_status(cudaMemGetInfo(&free_bytes, &total_bytes),
                           "cudaMemGetInfo after inference operation");
      if (!status.ok()) {
        return status;
      }
      const std::size_t used_bytes = total_bytes - free_bytes;
      const std::size_t delta = used_bytes > entry.baseline_used_bytes
                                    ? used_bytes - entry.baseline_used_bytes
                                    : 0;
      entry.observed_peak_delta_bytes =
          std::max(entry.observed_peak_delta_bytes, delta);
      entry.total_bytes = total_bytes;
    }
    return Status::Ok();
  }

  [[nodiscard]] const std::vector<DeviceMemory> &entries() const noexcept {
    return entries_;
  }

private:
  std::vector<DeviceMemory> entries_;
};

struct Metrics final {
  double model_load_seconds{0.0};
  double prefill_seconds{0.0};
  double teacher_force_seconds{0.0};
  double decode_seconds{0.0};
  std::size_t prefill_tokens{0};
  std::size_t teacher_force_tokens{0};
  std::size_t decode_tokens{0};
  std::size_t retained_logits_peak_bytes{0};
};

using LogitsChunkConsumer =
    std::function<Status(std::size_t offset, const std::vector<float> &logits)>;

Status validate_devices(const std::vector<int> &devices) {
  if (devices.empty() || devices.size() > 4) {
    return {ErrorCode::kInvalidArgument,
            "Evo 2 inference requires one to four CUDA devices; "
            "for example pass --gpu 0 or --gpu 0,1,2,3"};
  }
  int count = 0;
  auto status = cuda_status(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (!status.ok()) {
    return status;
  }
  for (const int device : devices) {
    if (device < 0 || device >= count) {
      return {ErrorCode::kInvalidArgument,
              "CUDA device " + std::to_string(device) +
                  " is unavailable; visible device count is " +
                  std::to_string(count)};
    }
  }
  return Status::Ok();
}

std::optional<LayerDump> make_layer_dump(const CliOptions &options) {
  if (!options.dump_layer.has_value()) {
    return std::nullopt;
  }
  return LayerDump{options.dump_layer->layer, options.dump_layer->path,
                   LayerDumpPoint::kBlockOutput};
}

void write_json_string(std::ostream &output, const std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  output.put('"');
  for (const char raw : value) {
    const auto byte = static_cast<unsigned char>(raw);
    switch (byte) {
    case '"':
      output << "\\\"";
      break;
    case '\\':
      output << "\\\\";
      break;
    case '\b':
      output << "\\b";
      break;
    case '\f':
      output << "\\f";
      break;
    case '\n':
      output << "\\n";
      break;
    case '\r':
      output << "\\r";
      break;
    case '\t':
      output << "\\t";
      break;
    default:
      if (byte < 0x20U || byte >= 0x7fU) {
        output << "\\u00" << hex[byte >> 4U] << hex[byte & 0x0fU];
      } else {
        output.put(static_cast<char>(byte));
      }
      break;
    }
  }
  output.put('"');
}

Status log_probability(const float *const logits, const std::size_t vocab_size,
                       const TokenId target, double *const result) {
  if (logits == nullptr || result == nullptr ||
      static_cast<std::size_t>(target) >= vocab_size) {
    return {ErrorCode::kInvalidArgument,
            "score target is outside the model vocabulary"};
  }
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t token = 0; token < vocab_size; ++token) {
    if (!std::isfinite(logits[token])) {
      return {ErrorCode::kInternal, "model produced a non-finite score logit"};
    }
    maximum = std::max(maximum, logits[token]);
  }
  double normalizer = 0.0;
  for (std::size_t token = 0; token < vocab_size; ++token) {
    normalizer += std::exp(static_cast<double>(logits[token]) -
                           static_cast<double>(maximum));
  }
  if (!std::isfinite(normalizer) || normalizer <= 0.0) {
    return {ErrorCode::kInternal, "score log-softmax normalization failed"};
  }
  *result = static_cast<double>(logits[target]) - static_cast<double>(maximum) -
            std::log(normalizer);
  return Status::Ok();
}

void print_score_record(const SequenceRecord &record,
                        const std::vector<double> &token_scores,
                        const RuntimeModelConfig &config) {
  double total = 0.0;
  for (const double value : token_scores) {
    total += value;
  }
  const double mean = total / static_cast<double>(token_scores.size());
  const double perplexity = std::exp(-mean);

  std::cout << "{\"name\":";
  write_json_string(std::cout, record.name);
  std::cout << ",\"input_format\":\""
            << sequence_format_name(record.format)
            << "\",\"backend\":\"cuda\",\"profile\":";
  write_json_string(std::cout,
                    inference_profile_name(config.inference_profile));
  std::cout << ",\"model_id\":";
  if (config.model_id.empty())
    std::cout << "null";
  else
    write_json_string(std::cout, config.model_id);
  std::cout << ",\"tokens\":" << record.bytes.size()
            << ",\"scored_tokens\":" << token_scores.size()
            << ",\"log_likelihood\":" << std::setprecision(17) << total
            << ",\"mean_log_likelihood\":" << mean << ",\"perplexity\":";
  if (std::isfinite(perplexity)) {
    std::cout << perplexity;
  } else {
    std::cout << "null";
  }
  std::cout << ",\"token_log_likelihoods\":[";
  for (std::size_t index = 0; index < token_scores.size(); ++index) {
    if (index != 0) {
      std::cout.put(',');
    }
    std::cout << token_scores[index];
  }
  std::cout << "]}\n";
}

const char *pooling_name(const EmbeddingPooling pooling) noexcept {
  switch (pooling) {
  case EmbeddingPooling::kNone:
    return "none";
  case EmbeddingPooling::kMean:
    return "mean";
  case EmbeddingPooling::kLast:
    return "last";
  }
  return "unknown";
}

void write_embedding_metadata(
    std::ostream &output, const SequenceRecord &record,
    const std::size_t record_index, const std::string_view filename,
    const std::size_t source_tokens, const std::size_t rows,
    const std::size_t columns, const std::size_t layer,
    const EmbeddingPooling pooling, const RuntimeModelConfig &config) {
  output << "{\"record_index\":" << record_index << ",\"name\":";
  write_json_string(output, record.name);
  output << ",\"input_format\":\""
         << sequence_format_name(record.format) << "\",\"file\":";
  write_json_string(output, filename);
  output << ",\"source_tokens\":" << source_tokens << ",\"shape\":[" << rows
         << ',' << columns << "],\"layer\":" << layer
         << ",\"point\":\"block_output\",\"pooling\":\""
         << pooling_name(pooling)
         << "\",\"dtype\":\"float32\",\"backend\":\"cuda\",\"profile\":\""
         << inference_profile_name(config.inference_profile)
         << "\",\"test_fixture\":" << (config.test_fixture ? "true" : "false")
         << ",\"model_id\":";
  if (config.model_id.empty()) {
    output << "null";
  } else {
    write_json_string(output, config.model_id);
  }
  output << "}\n";
}

Status prepare_embedding_output(const std::string &path,
                                std::ofstream *const manifest) {
  if (manifest == nullptr)
    return {ErrorCode::kInvalidArgument, "embedding manifest output is null"};
  const std::filesystem::path directory{path};
  std::error_code error;
  if (std::filesystem::exists(directory, error)) {
    if (error || !std::filesystem::is_directory(directory, error)) {
      return {ErrorCode::kInvalidArgument,
              "embedding output exists and is not a directory: '" + path + "'"};
    }
    const std::filesystem::directory_iterator begin{directory, error};
    if (error)
      return {ErrorCode::kIo,
              "cannot inspect embedding output directory '" + path + "'"};
    if (begin != std::filesystem::directory_iterator{}) {
      return {ErrorCode::kInvalidArgument,
              "embedding output directory must be empty: '" + path + "'"};
    }
  } else {
    if (error || !std::filesystem::create_directories(directory, error) ||
        error) {
      return {ErrorCode::kIo,
              "cannot create embedding output directory '" + path + "'"};
    }
  }
  const auto manifest_path = directory / "embeddings.jsonl";
  manifest->open(manifest_path, std::ios::binary | std::ios::trunc);
  if (!*manifest) {
    return {ErrorCode::kIo,
            "cannot open embedding manifest '" + manifest_path.string() + "'"};
  }
  return Status::Ok();
}

Status prefill_in_chunks(
    const std::vector<TokenId> &tokens, const std::size_t token_count,
    const std::optional<LayerDump> &dump, const bool collect_all_logits,
    const bool cached_initial, const bool stateless_initial,
    PipelineModel *const model, std::vector<float> *const logits,
    Metrics *const metrics, const LogitsChunkConsumer &consumer = {}) {
  if (model == nullptr || logits == nullptr || metrics == nullptr ||
      token_count == 0 || token_count > tokens.size() ||
      model->activation_capacity() == 0) {
    return {ErrorCode::kInvalidArgument,
            "chunked prefill arguments are invalid"};
  }
  const std::size_t chunk_capacity = model->activation_capacity();
  if (cached_initial && token_count > chunk_capacity) {
    return {ErrorCode::kInvalidArgument,
            "exact cached prefill must fit one activation chunk; lower "
            "--force-prompt-threshold to at most " +
                std::to_string(chunk_capacity)};
  }
  if (stateless_initial &&
      (cached_initial || dump.has_value() || token_count > chunk_capacity)) {
    return {ErrorCode::kInvalidArgument,
            "stateless prefill requires one non-cached chunk without a layer "
            "dump"};
  }
  if (dump.has_value() && token_count > chunk_capacity) {
    return {ErrorCode::kInvalidArgument,
            "--dump-layer is unavailable when prefill spans multiple "
            "activation chunks"};
  }
  const std::size_t vocab_size = model->config().vocab_size;
  if (collect_all_logits &&
      (vocab_size == 0 ||
       token_count > std::numeric_limits<std::size_t>::max() / vocab_size)) {
    return {ErrorCode::kInvalidArgument,
            "chunked prefill logit dimensions overflow"};
  }

  logits->clear();
  if (collect_all_logits) {
    logits->reserve(token_count * vocab_size);
  }
  std::size_t offset = 0;
  bool first = true;
  while (offset < token_count) {
    const std::size_t chunk_size =
        std::min(chunk_capacity, token_count - offset);
    const std::vector<TokenId> chunk(
        tokens.begin() + static_cast<std::ptrdiff_t>(offset),
        tokens.begin() + static_cast<std::ptrdiff_t>(offset + chunk_size));
    std::vector<float> chunk_logits;
    const auto start = Clock::now();
    const auto status = [&]() {
      if (first && cached_initial) {
        return dump.has_value() ? model->prefill_cached_with_dumps(
                                      chunk, &chunk_logits, {*dump})
                                : model->prefill_cached(chunk, &chunk_logits);
      }
      if (first && stateless_initial)
        return model->prefill_stateless(chunk, &chunk_logits);
      return first ? model->prefill(chunk, &chunk_logits, dump)
                   : model->prefill_chunk(chunk, &chunk_logits);
    }();
    metrics->prefill_seconds += seconds_since(start);
    metrics->prefill_tokens += chunk_size;
    if (!status.ok()) {
      return {status.code(), "activation chunk at token " +
                                 std::to_string(offset) + ": " +
                                 status.message()};
    }
    if (chunk_logits.size() != chunk_size * vocab_size) {
      return {ErrorCode::kInternal,
              "chunked prefill returned an incomplete logit matrix"};
    }
    if (consumer) {
      const auto consume_status = consumer(offset, chunk_logits);
      if (!consume_status.ok()) {
        return {consume_status.code(), "consume logits at token " +
                                           std::to_string(offset) + ": " +
                                           consume_status.message()};
      }
    }
    if (collect_all_logits) {
      logits->insert(logits->end(), chunk_logits.begin(), chunk_logits.end());
    } else {
      *logits = std::move(chunk_logits);
    }
    metrics->retained_logits_peak_bytes = std::max(
        metrics->retained_logits_peak_bytes, logits->size() * sizeof(float));
    offset += chunk_size;
    first = false;
  }
  return Status::Ok();
}

Status run_generate(const CliOptions &options, PipelineModel *const model,
                    MemoryTracker *const memory, Metrics *const metrics) {
  constexpr std::size_t kOfficialForcePromptThreshold = 3000;
  const auto prompt = encode_bytes(options.prompt);
  const std::size_t default_threshold =
      std::min(kOfficialForcePromptThreshold, model->activation_capacity());
  const std::size_t prefill_tokens =
      std::min(prompt.size(),
               options.force_prompt_threshold.value_or(default_threshold));
  std::vector<float> logits;
  auto status =
      prefill_in_chunks(prompt, prefill_tokens, make_layer_dump(options), false,
                        true, false, model, &logits, metrics);
  if (!status.ok()) {
    return {status.code(), "prompt prefill: " + status.message()};
  }
  status = memory->observe();
  if (!status.ok()) {
    return status;
  }

  const std::size_t vocab_size = model->config().vocab_size;
  if (vocab_size != kTokenizerVocabSize || logits.size() < vocab_size ||
      logits.size() % vocab_size != 0) {
    return {ErrorCode::kModelFormat,
            "generation requires a 512-token vocabulary and complete final "
            "prefill-chunk logits"};
  }
  std::vector<float> current(
      logits.end() - static_cast<std::ptrdiff_t>(vocab_size), logits.end());
  for (std::size_t index = prefill_tokens; index < prompt.size(); ++index) {
    const auto force_start = Clock::now();
    status = model->decode(prompt[index], &current);
    metrics->teacher_force_seconds += seconds_since(force_start);
    ++metrics->teacher_force_tokens;
    if (!status.ok()) {
      return {status.code(), "teacher-forced prompt token " +
                                 std::to_string(index) + ": " +
                                 status.message()};
    }
  }
  status = memory->observe();
  if (!status.ok()) {
    return status;
  }
  Sampler sampler(options.sampling);
  std::string generated;
  generated.reserve(options.generated_tokens);
  std::vector<float> dumped_logits;
  if (options.dump_logits_path.has_value()) {
    if (options.generated_tokens >
        std::numeric_limits<std::size_t>::max() / vocab_size) {
      return {ErrorCode::kInvalidArgument,
              "generated logit dump dimensions overflow"};
    }
    dumped_logits.reserve(options.generated_tokens * vocab_size);
  }

  for (std::size_t step = 0; step < options.generated_tokens; ++step) {
    if (options.dump_logits_path.has_value()) {
      dumped_logits.insert(dumped_logits.end(), current.begin(), current.end());
    }
    TokenId sampled = 0;
    status = sampler.sample(current, &sampled);
    if (!status.ok()) {
      return status;
    }
    std::uint8_t byte = 0;
    status = token_to_byte(sampled, &byte);
    if (!status.ok()) {
      return {status.code(), "generated token " + std::to_string(sampled) +
                                 " has no byte representation"};
    }
    generated.push_back(static_cast<char>(byte));
    if (step + 1 == options.generated_tokens) {
      continue;
    }
    const auto decode_start = Clock::now();
    status = model->decode(sampled, &current);
    metrics->decode_seconds += seconds_since(decode_start);
    ++metrics->decode_tokens;
    if (!status.ok()) {
      return {status.code(), "cached decode step " + std::to_string(step + 1) +
                                 ": " + status.message()};
    }
  }
  status = memory->observe();
  if (!status.ok()) {
    return status;
  }
  if (options.dump_logits_path.has_value()) {
    status = npy::write_f32(*options.dump_logits_path, dumped_logits,
                            options.generated_tokens, vocab_size);
    if (!status.ok()) {
      return status;
    }
  }
  std::cout.write(generated.data(),
                  static_cast<std::streamsize>(generated.size()));
  std::cout.flush();
  if (!std::cout) {
    return {ErrorCode::kIo, "failed to write generated bytes to stdout"};
  }
  return Status::Ok();
}

Status run_score(const CliOptions &options, PipelineModel *const model,
                 MemoryTracker *const memory, Metrics *const metrics) {
  const std::size_t vocab_size = model->config().vocab_size;
  if (vocab_size != kTokenizerVocabSize) {
    return {ErrorCode::kModelFormat, "scoring requires a 512-token vocabulary"};
  }
  std::size_t processed_records = 0;
  const auto status = stream_sequence_file(
      options.score_path, options.context_size,
      [&](const SequenceRecord &record) -> Status {
        if (processed_records != 0 && (options.dump_logits_path.has_value() ||
                                       options.dump_layer.has_value())) {
          return {ErrorCode::kInvalidArgument,
                  "--dump-logits and --dump-layer require a score input with "
                  "exactly one record"};
        }
        if (options.dump_tokens) {
          const auto dumped = encode_bytes(record.bytes);
          std::cerr << "tokens " << record.name << "=[";
          for (std::size_t index = 0; index < dumped.size(); ++index) {
            if (index != 0) {
              std::cerr << ',';
            }
            std::cerr << dumped[index];
          }
          std::cerr << "]\n";
        }
        const auto tokens = encode_bytes(record.bytes);
        if (tokens.size() < 2) {
          return {ErrorCode::kInvalidArgument,
                  "score record '" + record.name +
                      "' must contain at least two bytes"};
        }
        std::vector<float> logits;
        const auto dump = make_layer_dump(options);
        const bool stateless =
            !dump.has_value() && tokens.size() <= model->activation_capacity();
        const bool retain_all_logits = options.dump_logits_path.has_value();
        std::vector<double> token_scores;
        token_scores.reserve(tokens.size() - 1);
        const LogitsChunkConsumer score_chunk =
            [&](const std::size_t offset,
                const std::vector<float> &chunk_logits) -> Status {
          const std::size_t chunk_tokens = chunk_logits.size() / vocab_size;
          for (std::size_t local = 0; local < chunk_tokens; ++local) {
            const std::size_t row = offset + local;
            if (row + 1 >= tokens.size()) {
              break;
            }
            double value = 0.0;
            const auto score_status =
                log_probability(chunk_logits.data() + local * vocab_size,
                                vocab_size, tokens[row + 1], &value);
            if (!score_status.ok()) {
              return score_status;
            }
            token_scores.push_back(value);
          }
          return Status::Ok();
        };
        auto record_status = prefill_in_chunks(
            tokens, tokens.size(), dump, retain_all_logits, false, stateless,
            model, &logits, metrics, score_chunk);
        if (!record_status.ok()) {
          return {record_status.code(), "score prefill for '" + record.name +
                                            "': " + record_status.message()};
        }
        record_status = memory->observe();
        if (!record_status.ok()) {
          return record_status;
        }
        if (retain_all_logits && logits.size() != tokens.size() * vocab_size) {
          return {ErrorCode::kInternal,
                  "score prefill returned an incomplete logit matrix"};
        }
        if (options.dump_logits_path.has_value()) {
          record_status = npy::write_f32(*options.dump_logits_path, logits,
                                         tokens.size(), vocab_size);
          if (!record_status.ok()) {
            return record_status;
          }
        }

        if (token_scores.size() != tokens.size() - 1) {
          return {ErrorCode::kInternal,
                  "score prefill returned incomplete token log likelihoods"};
        }
        print_score_record(record, token_scores, model->config());
        ++processed_records;
        return Status::Ok();
      });
  if (!status.ok()) {
    return status;
  }
  std::cout.flush();
  if (!std::cout) {
    return {ErrorCode::kIo, "failed to write score JSONL to stdout"};
  }
  return Status::Ok();
}

Status score_sequence(const std::string_view sequence,
                      PipelineModel *const model, Metrics *const metrics,
                      std::vector<double> *const token_scores) {
  if (model == nullptr || metrics == nullptr || token_scores == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "variant sequence scoring arguments are invalid"};
  }
  const auto tokens = encode_bytes(sequence);
  if (tokens.size() < 2) {
    return {ErrorCode::kInvalidArgument,
            "variant sequence must contain at least two bytes"};
  }
  const std::size_t vocab_size = model->config().vocab_size;
  if (vocab_size != kTokenizerVocabSize) {
    return {ErrorCode::kModelFormat,
            "variant scoring requires a 512-token vocabulary"};
  }
  token_scores->clear();
  token_scores->reserve(tokens.size() - 1);
  const LogitsChunkConsumer score_chunk =
      [&](const std::size_t offset,
          const std::vector<float> &chunk_logits) -> Status {
    const std::size_t chunk_tokens = chunk_logits.size() / vocab_size;
    for (std::size_t local = 0; local < chunk_tokens; ++local) {
      const std::size_t row = offset + local;
      if (row + 1 >= tokens.size())
        break;
      double value = 0.0;
      const auto status =
          log_probability(chunk_logits.data() + local * vocab_size, vocab_size,
                          tokens[row + 1], &value);
      if (!status.ok())
        return status;
      token_scores->push_back(value);
    }
    return Status::Ok();
  };
  std::vector<float> logits;
  const bool stateless = tokens.size() <= model->activation_capacity();
  auto status =
      prefill_in_chunks(tokens, tokens.size(), std::nullopt, false, false,
                        stateless, model, &logits, metrics, score_chunk);
  if (!status.ok())
    return status;
  if (token_scores->size() != tokens.size() - 1) {
    return {ErrorCode::kInternal,
            "variant prefill returned incomplete token log likelihoods"};
  }
  return Status::Ok();
}

struct VariantStrandScore final {
  const char *strand{"unknown"};
  double reference_log_likelihood{0.0};
  double alternate_log_likelihood{0.0};
  std::size_t reference_scored_tokens{0};
  std::size_t alternate_scored_tokens{0};
  double delta{0.0};
};

Status score_variant_strand(const std::string_view reference,
                            const std::string_view alternate,
                            const VariantStrand strand,
                            const VariantNormalization normalization,
                            PipelineModel *const model, Metrics *const metrics,
                            VariantStrandScore *const result) {
  if (result == nullptr) {
    return {ErrorCode::kInvalidArgument, "variant strand score output is null"};
  }
  std::vector<double> reference_scores;
  auto status = score_sequence(reference, model, metrics, &reference_scores);
  if (!status.ok()) {
    return {status.code(), std::string{variant_strand_name(strand)} +
                               " reference window: " + status.message()};
  }
  std::vector<double> alternate_scores;
  status = score_sequence(alternate, model, metrics, &alternate_scores);
  if (!status.ok()) {
    return {status.code(), std::string{variant_strand_name(strand)} +
                               " alternate window: " + status.message()};
  }
  result->strand = variant_strand_name(strand);
  result->reference_scored_tokens = reference_scores.size();
  result->alternate_scored_tokens = alternate_scores.size();
  result->reference_log_likelihood =
      std::accumulate(reference_scores.begin(), reference_scores.end(), 0.0);
  result->alternate_log_likelihood =
      std::accumulate(alternate_scores.begin(), alternate_scores.end(), 0.0);
  if (normalization == VariantNormalization::kMean) {
    result->reference_log_likelihood /=
        static_cast<double>(result->reference_scored_tokens);
    result->alternate_log_likelihood /=
        static_cast<double>(result->alternate_scored_tokens);
  }
  result->delta =
      result->alternate_log_likelihood - result->reference_log_likelihood;
  return Status::Ok();
}

Status run_variant_one(const CliOptions &options,
                       const std::string_view sequence,
                       const std::size_t position_1based,
                       const std::string_view reference,
                       const std::string_view alternate,
                       const std::size_t coordinate_offset,
                       const VcfRecord *const vcf, PipelineModel *const model,
                       MemoryTracker *const memory, Metrics *const metrics) {
  VariantWindow window;
  auto status = make_variant_window(sequence, position_1based, reference,
                                    alternate, options.variant_window_tokens,
                                    &window);
  if (!status.ok())
    return status;

  if (options.dump_tokens) {
    const auto dump = [](const std::string_view label,
                         const std::string_view window_sequence) {
      const auto tokens = encode_bytes(window_sequence);
      std::cerr << "tokens " << label << "=[";
      for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0)
          std::cerr.put(',');
        std::cerr << tokens[index];
      }
      std::cerr << "]\n";
    };
    dump("variant-reference", window.reference);
    dump("variant-alternate", window.alternate);
  }

  std::vector<VariantStrandScore> scores;
  scores.reserve(options.variant_strand == VariantStrand::kBoth ? 2 : 1);
  if (options.variant_strand == VariantStrand::kForward ||
      options.variant_strand == VariantStrand::kBoth) {
    VariantStrandScore result;
    status = score_variant_strand(
        window.reference, window.alternate, VariantStrand::kForward,
        options.variant_normalization, model, metrics, &result);
    if (!status.ok())
      return status;
    scores.push_back(result);
  }
  if (options.variant_strand == VariantStrand::kReverse ||
      options.variant_strand == VariantStrand::kBoth) {
    std::string reverse_reference_sequence;
    status =
        reverse_complement(window.reference, &reverse_reference_sequence);
    if (!status.ok())
      return status;
    std::string reverse_alternate_sequence;
    status = reverse_complement(window.alternate, &reverse_alternate_sequence);
    if (!status.ok())
      return status;
    VariantStrandScore result;
    status = score_variant_strand(
        reverse_reference_sequence, reverse_alternate_sequence,
        VariantStrand::kReverse,
        options.variant_normalization, model, metrics, &result);
    if (!status.ok())
      return status;
    scores.push_back(result);
  }
  status = memory->observe();
  if (!status.ok())
    return status;

  double aggregate = 0.0;
  for (const auto &score : scores)
    aggregate += score.delta;
  aggregate /= static_cast<double>(scores.size());
  std::ostringstream output;
  output << "{\"source\":\"" << (vcf == nullptr ? "inline" : "vcf")
         << "\"";
  if (vcf != nullptr) {
    output << ",\"record_index\":" << vcf->record_index
           << ",\"allele_index\":" << vcf->allele_index
           << ",\"vcf_line\":" << vcf->line_number << ",\"id\":";
    write_json_string(output, vcf->id);
    output << ",\"contig\":";
    write_json_string(output, vcf->contig);
  }
  output << ",\"position\":"
         << (vcf == nullptr ? position_1based : vcf->position_1based)
         << ",\"position_coordinate_system\":\""
         << (vcf == nullptr ? "1-based" : "VCF-1-based")
         << "\",\"reference\":";
  write_json_string(output, reference);
  output << ",\"alternate\":";
  write_json_string(output, alternate);
  output << ",\"window\":{\"start\":"
         << coordinate_offset + window.reference_start << ",\"end\":"
         << coordinate_offset + window.reference_end
         << ",\"coordinate_system\":\"0-based-half-open\"";
  if (vcf != nullptr) {
    output << ",\"contig\":";
    write_json_string(output, vcf->contig);
  }
  output << ",\"reference_sequence\":";
  write_json_string(output, window.reference);
  output << ",\"alternate_sequence\":";
  write_json_string(output, window.alternate);
  output << ",\"reference_tokens\":" << window.reference.size()
         << ",\"alternate_tokens\":" << window.alternate.size() << '}'
         << ",\"normalization\":\""
         << variant_normalization_name(options.variant_normalization)
         << "\",\"backend\":\"cuda\",\"profile\":\""
         << inference_profile_name(model->config().inference_profile)
         << "\",\"test_fixture\":"
         << (model->config().test_fixture ? "true" : "false")
         << ",\"model_id\":";
  if (model->config().model_id.empty()) {
    output << "null";
  } else {
    write_json_string(output, model->config().model_id);
  }
  output << ",\"strands\":[";
  for (std::size_t index = 0; index < scores.size(); ++index) {
    if (index != 0)
      output.put(',');
    const auto &score = scores[index];
    output << "{\"strand\":\"" << score.strand
           << "\",\"reference_log_likelihood\":" << std::setprecision(17)
           << score.reference_log_likelihood
           << ",\"alternate_log_likelihood\":"
           << score.alternate_log_likelihood
           << ",\"reference_scored_tokens\":"
           << score.reference_scored_tokens
           << ",\"alternate_scored_tokens\":"
           << score.alternate_scored_tokens << ",\"delta\":" << score.delta
           << '}';
  }
  output << "],\"aggregation\":\""
         << (scores.size() == 1 ? "single_strand" : "mean_across_strands")
         << "\",\"score\":" << aggregate << "}\n";
  const auto document = output.str();
  std::cout.write(document.data(), static_cast<std::streamsize>(document.size()));
  if (!std::cout) {
    return {ErrorCode::kIo, "failed to write variant score JSON to stdout"};
  }
  return Status::Ok();
}

Status run_variant_score(const CliOptions &options, PipelineModel *const model,
                         MemoryTracker *const memory, Metrics *const metrics) {
  if (options.variant_vcf_path.empty()) {
    return run_variant_one(
        options, options.variant_sequence, options.variant_position_1based,
        options.variant_reference, options.variant_alternate, 0, nullptr,
        model, memory, metrics);
  }
  const auto status = stream_vcf_file(
      options.variant_vcf_path, [&](const VcfRecord &record) -> Status {
        ReferenceSlice slice;
        auto inner = fetch_reference_slice(
            options.variant_reference_path, record.contig,
            record.position_1based, record.reference, record.alternate,
            options.variant_window_tokens, &slice);
        if (!inner.ok()) {
          return {inner.code(), "VCF line " +
                                    std::to_string(record.line_number) + ": " +
                                    inner.message()};
        }
        const std::size_t local_position =
            record.position_1based - slice.start;
        return run_variant_one(options, slice.sequence, local_position,
                               record.reference, record.alternate, slice.start,
                               &record, model, memory, metrics);
      });
  if (!status.ok())
    return status;
  std::cout.flush();
  return std::cout ? Status::Ok()
                   : Status{ErrorCode::kIo,
                            "failed writing variant score JSONL"};
}

Status run_embed(const CliOptions &options, PipelineModel *const model,
                 MemoryTracker *const memory, Metrics *const metrics) {
  if (options.embed_layer >= model->config().layers) {
    return {ErrorCode::kInvalidArgument,
            "embedding layer " + std::to_string(options.embed_layer) +
                " is outside [0, " + std::to_string(model->config().layers) +
                ")"};
  }
  std::ofstream manifest;
  auto status = prepare_embedding_output(options.embed_output_dir, &manifest);
  if (!status.ok())
    return status;
  const std::filesystem::path output_directory{options.embed_output_dir};
  std::size_t record_index = 0;
  status = stream_sequence_file(
      options.embed_path, options.context_size,
      [&](const SequenceRecord &record) -> Status {
        if (options.dump_tokens) {
          const auto dumped = encode_bytes(record.bytes);
          std::cerr << "tokens " << record.name << "=[";
          for (std::size_t index = 0; index < dumped.size(); ++index) {
            if (index != 0)
              std::cerr.put(',');
            std::cerr << dumped[index];
          }
          std::cerr << "]\n";
        }
        const auto tokens = encode_bytes(record.bytes);
        if (tokens.empty()) {
          return {ErrorCode::kInvalidArgument,
                  "embedding record '" + record.name + "' is empty"};
        }
        std::ostringstream filename_builder;
        filename_builder << std::setw(6) << std::setfill('0') << record_index
                         << ".npy";
        const std::string filename = filename_builder.str();
        const auto output_path = output_directory / filename;
        npy::F32MatrixWriter writer;
        const std::size_t width = model->config().width;
        std::vector<double> mean;
        std::vector<float> last;
        if (options.embedding_pooling == EmbeddingPooling::kNone) {
          auto writer_status =
              writer.open(output_path.string(), tokens.size(), width);
          if (!writer_status.ok())
            return writer_status;
        } else if (options.embedding_pooling == EmbeddingPooling::kMean) {
          mean.assign(width, 0.0);
        } else {
          last.assign(width, 0.0F);
        }

        std::size_t offset = 0;
        bool first = true;
        while (offset < tokens.size()) {
          const std::size_t rows =
              std::min(model->activation_capacity(), tokens.size() - offset);
          const std::vector<TokenId> chunk(
              tokens.begin() + static_cast<std::ptrdiff_t>(offset),
              tokens.begin() + static_cast<std::ptrdiff_t>(offset + rows));
          std::vector<float> embedding;
          const auto start = Clock::now();
          auto chunk_status =
              first ? model->prefill_embedding(chunk, options.embed_layer,
                                               &embedding)
                    : model->prefill_chunk_embedding(chunk, options.embed_layer,
                                                     &embedding);
          metrics->prefill_seconds += seconds_since(start);
          metrics->prefill_tokens += rows;
          if (!chunk_status.ok()) {
            return {chunk_status.code(),
                    "embed record '" + record.name + "' at token " +
                        std::to_string(offset) + ": " + chunk_status.message()};
          }
          if (embedding.size() != rows * width) {
            return {ErrorCode::kInternal,
                    "embedding chunk has an incomplete matrix"};
          }
          if (options.embedding_pooling == EmbeddingPooling::kNone) {
            chunk_status = writer.append(embedding.data(), embedding.size());
            if (!chunk_status.ok())
              return chunk_status;
          } else if (options.embedding_pooling == EmbeddingPooling::kMean) {
            for (std::size_t row = 0; row < rows; ++row) {
              for (std::size_t column = 0; column < width; ++column) {
                mean[column] += embedding[row * width + column];
              }
            }
          } else {
            std::copy_n(embedding.end() - static_cast<std::ptrdiff_t>(width),
                        width, last.begin());
          }
          offset += rows;
          first = false;
        }

        if (options.embedding_pooling != EmbeddingPooling::kNone) {
          std::vector<float> pooled(width);
          if (options.embedding_pooling == EmbeddingPooling::kMean) {
            for (std::size_t column = 0; column < width; ++column) {
              pooled[column] = static_cast<float>(
                  mean[column] / static_cast<double>(tokens.size()));
            }
          } else {
            pooled = std::move(last);
          }
          auto writer_status = writer.open(output_path.string(), 1, width);
          if (writer_status.ok())
            writer_status = writer.append(pooled.data(), pooled.size());
          if (!writer_status.ok())
            return writer_status;
        }
        auto writer_status = writer.close();
        if (!writer_status.ok())
          return writer_status;
        auto observe_status = memory->observe();
        if (!observe_status.ok())
          return observe_status;

        const std::size_t output_rows =
            options.embedding_pooling == EmbeddingPooling::kNone ? tokens.size()
                                                                 : 1;
        write_embedding_metadata(manifest, record, record_index, filename,
                                 tokens.size(), output_rows, width,
                                 options.embed_layer, options.embedding_pooling,
                                 model->config());
        manifest.flush();
        if (!manifest) {
          return {ErrorCode::kIo,
                  "failed to write embedding manifest for record '" +
                      record.name + "'"};
        }
        write_embedding_metadata(std::cout, record, record_index, filename,
                                 tokens.size(), output_rows, width,
                                 options.embed_layer, options.embedding_pooling,
                                 model->config());
        std::cout.flush();
        if (!std::cout) {
          return {ErrorCode::kIo,
                  "failed to write embedding metadata to stdout"};
        }
        ++record_index;
        return Status::Ok();
      });
  return status;
}

void print_metrics(const Metrics &metrics, const MemoryTracker &memory,
                   const std::vector<StageAssignment> &stages,
                   const bool q8_kv_cache,
                   const InferenceProfile inference_profile) {
  const double prefill_rate =
      metrics.prefill_seconds > 0.0
          ? static_cast<double>(metrics.prefill_tokens) /
                metrics.prefill_seconds
          : 0.0;
  const double decode_rate =
      metrics.decode_seconds > 0.0
          ? static_cast<double>(metrics.decode_tokens) / metrics.decode_seconds
          : 0.0;
  const double teacher_force_rate =
      metrics.teacher_force_seconds > 0.0
          ? static_cast<double>(metrics.teacher_force_tokens) /
                metrics.teacher_force_seconds
          : 0.0;
  std::cerr << "evo_metrics {\"model_load_seconds\":" << std::setprecision(9)
            << metrics.model_load_seconds
            << ",\"prefill_tokens\":" << metrics.prefill_tokens
            << ",\"prefill_seconds\":" << metrics.prefill_seconds
            << ",\"prefill_tokens_per_second\":" << prefill_rate
            << ",\"teacher_force_tokens\":" << metrics.teacher_force_tokens
            << ",\"teacher_force_seconds\":" << metrics.teacher_force_seconds
            << ",\"teacher_force_tokens_per_second\":" << teacher_force_rate
            << ",\"decode_tokens\":" << metrics.decode_tokens
            << ",\"decode_seconds\":" << metrics.decode_seconds
            << ",\"decode_tokens_per_second\":" << decode_rate
            << ",\"kv_cache\":\""
            << (q8_kv_cache ? "q8_paged" : "bf16_contiguous")
            << "\",\"profile\":\"" << inference_profile_name(inference_profile)
            << "\",\"retained_logits_peak_bytes\":"
            << metrics.retained_logits_peak_bytes << ",\"gpus\":[";
  for (std::size_t index = 0; index < memory.entries().size(); ++index) {
    if (index != 0) {
      std::cerr.put(',');
    }
    const auto &entry = memory.entries()[index];
    const auto &stage = stages[index];
    std::cerr << "{\"device\":" << entry.device << ",\"layers\":["
              << stage.layer_begin << ',' << stage.layer_end << ']'
              << ",\"weights_bytes\":" << stage.weight_bytes
              << ",\"cache_bytes\":" << stage.cache_bytes
              << ",\"arena_bytes\":" << stage.arena_bytes
              << ",\"observed_peak_delta_bytes\":"
              << entry.observed_peak_delta_bytes
              << ",\"total_bytes\":" << entry.total_bytes << '}';
  }
  std::cerr << "]}\n";
}

} // namespace

Status run_inference_cli(const CliOptions &options,
                         const bool allow_test_fixture) {
  auto status = validate_devices(options.gpu_ids);
  if (!status.ok()) {
    return status;
  }
  MemoryTracker memory;
  status = memory.initialize(options.gpu_ids);
  if (!status.ok()) {
    return status;
  }

  Metrics metrics;
  const auto load_start = Clock::now();
  std::cerr << "evo: validating and opening model '" << options.model_path
            << "'\n";
  ModelFile model_file;
  status = model_file.open(options.model_path);
  if (!status.ok()) {
    return status;
  }
  std::cerr
      << "evo: loading native BF16/software-E4M3 pipeline on CUDA devices";
  for (const int device : options.gpu_ids) {
    std::cerr << ' ' << device;
  }
  std::cerr << '\n';
  PipelineModel model;
  status = model.load(model_file, options.gpu_ids, options.context_size,
                      allow_test_fixture, options.inference_profile);
  metrics.model_load_seconds = seconds_since(load_start);
  if (!status.ok()) {
    return status;
  }
  status = memory.observe();
  if (!status.ok()) {
    return status;
  }

  if (options.mode == RunMode::kGenerate) {
    status = run_generate(options, &model, &memory, &metrics);
  } else if (options.mode == RunMode::kScore) {
    status = run_score(options, &model, &memory, &metrics);
  } else if (options.mode == RunMode::kEmbed) {
    status = run_embed(options, &model, &memory, &metrics);
  } else {
    status = run_variant_score(options, &model, &memory, &metrics);
  }
  if (!status.ok()) {
    return status;
  }
  model.refresh_cache_bytes();
  print_metrics(metrics, memory, model.stages(), model.uses_q8_kv_cache(),
                model.config().inference_profile);
  return Status::Ok();
}

} // namespace evo::cuda
