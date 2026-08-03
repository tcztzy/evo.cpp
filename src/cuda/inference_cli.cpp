// SPDX-License-Identifier: Apache-2.0
#include "evo2c/cuda/inference_cli.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

#include "evo2c/cuda/model.hpp"
#include "evo2c/cuda/runtime.hpp"
#include "evo2c/model_format.hpp"
#include "evo2c/sampler.hpp"
#include "evo2c/sequence_io.hpp"
#include "evo2c/tokenizer.hpp"

namespace evo2c::cuda {
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
};

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

Status write_npy_f32(const std::string &path, const std::vector<float> &values,
                     const std::size_t rows, const std::size_t columns) {
  if (rows == 0 || columns == 0 ||
      rows > std::numeric_limits<std::size_t>::max() / columns ||
      rows * columns != values.size()) {
    return {ErrorCode::kInvalidArgument,
            "logit dump dimensions do not match its F32 payload"};
  }
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       std::to_string(rows) + ", " + std::to_string(columns) +
                       "), }";
  constexpr std::size_t prefix_size = 10;
  const std::size_t padding =
      (64 - ((prefix_size + header.size() + 1) % 64)) % 64;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
    return {ErrorCode::kInternal, "NPY header exceeds version-1 capacity"};
  }
  if (values.size() >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) /
          sizeof(float)) {
    return {ErrorCode::kInvalidArgument,
            "logit dump exceeds the platform stream size"};
  }

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return {ErrorCode::kIo, "cannot open logit dump '" + path + "'"};
  }
  const std::array<char, 8> magic{
      static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y', 1, 0};
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  const auto header_size = static_cast<std::uint16_t>(header.size());
  const std::array<char, 2> encoded_size{
      static_cast<char>(header_size & 0xffU),
      static_cast<char>((header_size >> 8U) & 0xffU)};
  output.write(encoded_size.data(),
               static_cast<std::streamsize>(encoded_size.size()));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!output) {
    return {ErrorCode::kIo, "failed to write logit dump '" + path + "'"};
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
                        const std::vector<double> &token_scores) {
  double total = 0.0;
  for (const double value : token_scores) {
    total += value;
  }
  const double mean = total / static_cast<double>(token_scores.size());
  const double perplexity = std::exp(-mean);

  std::cout << "{\"name\":";
  write_json_string(std::cout, record.name);
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

Status
prefill_in_chunks(const std::vector<TokenId> &tokens,
                  const std::size_t token_count,
                  const std::optional<LayerDump> &dump,
                  const bool collect_all_logits, const bool cached_initial,
                  const bool stateless_initial, PipelineModel *const model,
                  std::vector<float> *const logits, Metrics *const metrics) {
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
    if (collect_all_logits) {
      logits->insert(logits->end(), chunk_logits.begin(), chunk_logits.end());
    } else {
      *logits = std::move(chunk_logits);
    }
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
    status = write_npy_f32(*options.dump_logits_path, dumped_logits,
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
  std::vector<SequenceRecord> records;
  auto status = read_sequence_file(options.score_path, &records);
  if (!status.ok()) {
    return status;
  }
  if (records.size() != 1 && (options.dump_logits_path.has_value() ||
                              options.dump_layer.has_value())) {
    return {ErrorCode::kInvalidArgument,
            "--dump-logits and --dump-layer require a score input with "
            "exactly one record"};
  }

  const std::size_t vocab_size = model->config().vocab_size;
  if (vocab_size != kTokenizerVocabSize) {
    return {ErrorCode::kModelFormat, "scoring requires a 512-token vocabulary"};
  }
  for (const auto &record : records) {
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
    status = prefill_in_chunks(tokens, tokens.size(), dump, true, false,
                               stateless, model, &logits, metrics);
    if (!status.ok()) {
      return {status.code(),
              "score prefill for '" + record.name + "': " + status.message()};
    }
    status = memory->observe();
    if (!status.ok()) {
      return status;
    }
    if (logits.size() != tokens.size() * vocab_size) {
      return {ErrorCode::kInternal,
              "score prefill returned an incomplete logit matrix"};
    }
    if (options.dump_logits_path.has_value()) {
      status = write_npy_f32(*options.dump_logits_path, logits, tokens.size(),
                             vocab_size);
      if (!status.ok()) {
        return status;
      }
    }

    std::vector<double> token_scores;
    token_scores.reserve(tokens.size() - 1);
    for (std::size_t index = 1; index < tokens.size(); ++index) {
      double value = 0.0;
      status = log_probability(logits.data() + (index - 1) * vocab_size,
                               vocab_size, tokens[index], &value);
      if (!status.ok()) {
        return status;
      }
      token_scores.push_back(value);
    }
    print_score_record(record, token_scores);
  }
  std::cout.flush();
  if (!std::cout) {
    return {ErrorCode::kIo, "failed to write score JSONL to stdout"};
  }
  return Status::Ok();
}

void print_metrics(const Metrics &metrics, const MemoryTracker &memory,
                   const std::vector<StageAssignment> &stages,
                   const bool q8_kv_cache) {
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
  std::cerr << "evo2c_metrics {\"model_load_seconds\":" << std::setprecision(9)
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
            << "\",\"gpus\":[";
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
  std::cerr << "evo2c: validating and opening model '" << options.model_path
            << "'\n";
  ModelFile model_file;
  status = model_file.open(options.model_path);
  if (!status.ok()) {
    return status;
  }
  std::cerr
      << "evo2c: loading native BF16/software-E4M3 pipeline on CUDA devices";
  for (const int device : options.gpu_ids) {
    std::cerr << ' ' << device;
  }
  std::cerr << '\n';
  PipelineModel model;
  status = model.load(model_file, options.gpu_ids, options.context_size,
                      allow_test_fixture);
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
  } else {
    status = run_score(options, &model, &memory, &metrics);
  }
  if (!status.ok()) {
    return status;
  }
  model.refresh_cache_bytes();
  print_metrics(metrics, memory, model.stages(), model.uses_q8_kv_cache());
  return Status::Ok();
}

} // namespace evo2c::cuda
