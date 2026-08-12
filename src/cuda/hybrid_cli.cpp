// SPDX-License-Identifier: Apache-2.0
#include "evo/cuda/hybrid_cli.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "evo/cpu/model.hpp"
#include "evo/cuda/model.hpp"
#include "evo/model_format.hpp"
#include "evo/sampler.hpp"
#include "evo/sequence_io.hpp"
#include "evo/tokenizer.hpp"

namespace evo::cuda {
namespace {

void write_json_string(std::ostream &output, const std::string_view value) {
  output.put('"');
  for (const char item : value) {
    const auto byte = static_cast<unsigned char>(item);
    if (byte == '"')
      output << "\\\"";
    else if (byte == '\\')
      output << "\\\\";
    else if (byte >= 0x20U && byte < 0x7fU)
      output.put(item);
    else
      output << "?";
  }
  output.put('"');
}

Status log_probability(const float *const logits, const TokenId target,
                       double *const result) {
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < kTokenizerVocabSize; ++index) {
    if (!std::isfinite(logits[index]))
      return {ErrorCode::kInternal,
              "hybrid backend produced non-finite logits"};
    maximum = std::max(maximum, logits[index]);
  }
  double denominator = 0.0;
  for (std::size_t index = 0; index < kTokenizerVocabSize; ++index)
    denominator += std::exp(static_cast<double>(logits[index] - maximum));
  *result =
      static_cast<double>(logits[target] - maximum) - std::log(denominator);
  return Status::Ok();
}

struct HybridWeights final {
  ModelFile artifact;
  PipelineModel cuda;
  cpu::Model cpu;
  std::size_t gpu_layers{0};

  Status load(const CliOptions &options, const bool allow_test_fixture) {
    gpu_layers = *options.gpu_layers;
    auto status = artifact.open(options.model_path);
    if (!status.ok())
      return status;
    status = cpu.load(artifact, allow_test_fixture);
    if (!status.ok())
      return status;
    if (gpu_layers >= cpu.config().layers)
      return {ErrorCode::kInvalidArgument,
              "--gpu-layers must be smaller than the model layer count; omit "
              "it for an all-CUDA model"};
    return cuda.load(artifact, options.gpu_ids, options.context_size,
                     allow_test_fixture, InferenceProfile::kExact, gpu_layers);
  }
};

class HybridContext final {
public:
  Status initialize(const HybridWeights &weights, const std::size_t capacity) {
    auto status = cuda_.initialize_shared(weights.cuda, capacity,
                                          InferenceProfile::kExact);
    if (!status.ok())
      return status;
    status = cpu_.initialize_shared(weights.cpu, capacity, weights.gpu_layers);
    if (!status.ok())
      return status;
    capacity_ =
        std::min(cuda_.activation_capacity(), cpu_.activation_capacity());
    return Status::Ok();
  }

  Status prefill(const std::vector<TokenId> &tokens,
                 std::vector<float> *const logits) {
    std::vector<float> hidden;
    auto status = cuda_.prefill_prefix(tokens, &hidden);
    if (!status.ok())
      return status;
    return cpu_.prefill_from_hidden(hidden, tokens.size(), logits);
  }

  Status prefill_chunk(const std::vector<TokenId> &tokens,
                       std::vector<float> *const logits) {
    std::vector<float> hidden;
    auto status = cuda_.prefill_chunk_prefix(tokens, &hidden);
    if (!status.ok())
      return status;
    return cpu_.prefill_chunk_from_hidden(hidden, tokens.size(), logits);
  }

  Status decode(const TokenId token, std::vector<float> *const logits) {
    std::vector<float> hidden;
    auto status = cuda_.decode_prefix(token, &hidden);
    if (!status.ok())
      return status;
    return cpu_.prefill_chunk_from_hidden(hidden, 1, logits);
  }

  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }

private:
  PipelineModel cuda_;
  cpu::Context cpu_;
  std::size_t capacity_{0};
};

Status prefill(HybridContext *const context, const std::vector<TokenId> &tokens,
               std::vector<float> *const last,
               std::vector<double> *const scores = nullptr) {
  std::size_t offset = 0;
  bool first = true;
  while (offset < tokens.size()) {
    const auto rows =
        std::min(context->activation_capacity(), tokens.size() - offset);
    const std::vector<TokenId> chunk(
        tokens.begin() + static_cast<std::ptrdiff_t>(offset),
        tokens.begin() + static_cast<std::ptrdiff_t>(offset + rows));
    std::vector<float> logits;
    const auto status = first ? context->prefill(chunk, &logits)
                              : context->prefill_chunk(chunk, &logits);
    if (!status.ok())
      return status;
    if (logits.size() != rows * kTokenizerVocabSize)
      return {ErrorCode::kInternal,
              "hybrid prefill returned incomplete logits"};
    if (scores != nullptr) {
      for (std::size_t row = 0; row < rows && offset + row + 1 < tokens.size();
           ++row) {
        double value = 0.0;
        const auto inner =
            log_probability(logits.data() + row * kTokenizerVocabSize,
                            tokens[offset + row + 1], &value);
        if (!inner.ok())
          return inner;
        scores->push_back(value);
      }
    }
    *last = std::move(logits);
    offset += rows;
    first = false;
  }
  return Status::Ok();
}

Status run_generate(const CliOptions &options, const HybridWeights &weights) {
  HybridContext context;
  auto status = context.initialize(weights, options.context_size);
  if (!status.ok())
    return status;
  const auto prompt = encode_bytes(options.prompt);
  std::vector<float> logits;
  status = prefill(&context, prompt, &logits);
  if (!status.ok())
    return status;
  std::vector<float> current(
      logits.end() - static_cast<std::ptrdiff_t>(kTokenizerVocabSize),
      logits.end());
  Sampler sampler(options.sampling);
  std::string generated;
  generated.reserve(options.generated_tokens);
  for (std::size_t step = 0; step < options.generated_tokens; ++step) {
    TokenId token = 0;
    status = sampler.sample(current, &token);
    if (!status.ok())
      return status;
    std::uint8_t byte = 0;
    status = token_to_byte(token, &byte);
    if (!status.ok())
      return status;
    generated.push_back(static_cast<char>(byte));
    if (step + 1 < options.generated_tokens) {
      status = context.decode(token, &current);
      if (!status.ok())
        return status;
    }
  }
  std::cout.write(generated.data(),
                  static_cast<std::streamsize>(generated.size()));
  std::cout.flush();
  return std::cout ? Status::Ok()
                   : Status{ErrorCode::kIo, "failed writing hybrid generation"};
}

Status run_score(const CliOptions &options, const HybridWeights &weights) {
  return stream_sequence_file(
      options.score_path, options.context_size,
      [&](const SequenceRecord &record) -> Status {
        const auto tokens = encode_bytes(record.bytes);
        if (tokens.size() < 2)
          return {ErrorCode::kInvalidArgument,
                  "hybrid score record must contain at least two bytes"};
        HybridContext context;
        auto status = context.initialize(weights, options.context_size);
        if (!status.ok())
          return status;
        std::vector<float> logits;
        std::vector<double> scores;
        status = prefill(&context, tokens, &logits, &scores);
        if (!status.ok())
          return status;
        const double total = std::accumulate(scores.begin(), scores.end(), 0.0);
        const double mean = total / static_cast<double>(scores.size());
        std::cout << "{\"name\":";
        write_json_string(std::cout, record.name);
        std::cout << ",\"backend\":\"cpu+cuda\",\"profile\":\"cpu-f32\","
                     "\"gpu_layers\":"
                  << weights.gpu_layers << ",\"tokens\":" << tokens.size()
                  << ",\"scored_tokens\":" << scores.size()
                  << ",\"log_likelihood\":" << std::setprecision(17) << total
                  << ",\"mean_log_likelihood\":" << mean
                  << ",\"perplexity\":" << std::exp(-mean)
                  << ",\"token_log_likelihoods\":[";
        for (std::size_t index = 0; index < scores.size(); ++index) {
          if (index != 0)
            std::cout.put(',');
          std::cout << scores[index];
        }
        std::cout << "]}\n";
        return Status::Ok();
      });
}

} // namespace

Status run_hybrid_cli(const CliOptions &options,
                      const bool allow_test_fixture) {
  if (!options.gpu_layers.has_value() ||
      options.inference_profile != InferenceProfile::kCpuF32)
    return {ErrorCode::kInvalidArgument,
            "hybrid execution requires --gpu-layers and cpu-f32 profile"};
  if (options.dump_layer.has_value() || options.dump_logits_path.has_value())
    return {ErrorCode::kUnsupported,
            "hybrid tensor dumps are unavailable; choose CPU or CUDA"};
  if (options.mode != RunMode::kGenerate && options.mode != RunMode::kScore)
    return {ErrorCode::kUnsupported,
            "hybrid offload currently supports generation and scoring"};
  HybridWeights weights;
  auto status = weights.load(options, allow_test_fixture);
  if (!status.ok())
    return status;
  status = options.mode == RunMode::kGenerate ? run_generate(options, weights)
                                              : run_score(options, weights);
  std::cerr << "evo_metrics {\"backend\":\"cpu+cuda\","
               "\"profile\":\"cpu-f32\",\"gpu_layers\":"
            << weights.gpu_layers << ",\"cpu_kernel\":\""
            << weights.cpu.kernel_name() << "\"}\n";
  return status;
}

} // namespace evo::cuda
