// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/inference_cli.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "evo/cpu/model.hpp"
#include "evo/model_format.hpp"
#include "evo/sampler.hpp"
#include "evo/sequence_io.hpp"
#include "evo/tokenizer.hpp"
#include "evo/variant.hpp"

namespace evo::cpu {
namespace {

using Clock = std::chrono::steady_clock;

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
      if (byte < 0x20U || byte >= 0x7fU)
        output << "\\u00" << hex[byte >> 4U] << hex[byte & 0x0fU];
      else
        output.put(static_cast<char>(byte));
      break;
    }
  }
  output.put('"');
}

Status write_npy_f32(const std::filesystem::path &path,
                     const std::vector<float> &values, const std::size_t rows,
                     const std::size_t columns) {
  if (rows == 0 || columns == 0 || values.size() != rows * columns)
    return {ErrorCode::kInvalidArgument, "NPY dimensions are inconsistent"};
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       std::to_string(rows) + ", " + std::to_string(columns) +
                       "), }";
  const std::size_t prefix = 10;
  const std::size_t padding = (16 - ((prefix + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<std::uint16_t>::max())
    return {ErrorCode::kInternal, "NPY header exceeds version-1 capacity"};
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    return {ErrorCode::kIo, "cannot open NPY output '" + path.string() + "'"};
  const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
  output.write(magic, sizeof(magic));
  const auto length = static_cast<std::uint16_t>(header.size());
  const char length_bytes[] = {static_cast<char>(length & 0xffU),
                               static_cast<char>(length >> 8U)};
  output.write(length_bytes, sizeof(length_bytes));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!output)
    return {ErrorCode::kIo,
            "failed writing NPY output '" + path.string() + "'"};
  return Status::Ok();
}

Status log_probability(const float *const logits, const std::size_t columns,
                       const TokenId target, double *const result) {
  if (logits == nullptr || result == nullptr || target >= columns)
    return {ErrorCode::kInvalidArgument, "score target exceeds vocabulary"};
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < columns; ++index) {
    if (!std::isfinite(logits[index]))
      return {ErrorCode::kInternal, "CPU backend produced non-finite logits"};
    maximum = std::max(maximum, logits[index]);
  }
  double denominator = 0.0;
  for (std::size_t index = 0; index < columns; ++index)
    denominator += std::exp(static_cast<double>(logits[index] - maximum));
  *result =
      static_cast<double>(logits[target] - maximum) - std::log(denominator);
  return Status::Ok();
}

Status prefill(Context *const context, const std::vector<TokenId> &tokens,
               const bool retain, std::vector<float> *const output,
               std::vector<double> *const scores = nullptr) {
  if (context == nullptr || output == nullptr || tokens.empty())
    return {ErrorCode::kInvalidArgument, "CPU chunked prefill is invalid"};
  const auto columns = context->config().vocab_size;
  output->clear();
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
    if (logits.size() != rows * columns)
      return {ErrorCode::kInternal, "CPU prefill returned incomplete logits"};
    if (scores != nullptr) {
      for (std::size_t row = 0; row < rows && offset + row + 1 < tokens.size();
           ++row) {
        double value = 0.0;
        const auto score_status =
            log_probability(logits.data() + row * columns, columns,
                            tokens[offset + row + 1], &value);
        if (!score_status.ok())
          return score_status;
        scores->push_back(value);
      }
    }
    if (retain)
      output->insert(output->end(), logits.begin(), logits.end());
    else
      *output = std::move(logits);
    offset += rows;
    first = false;
  }
  return Status::Ok();
}

void print_score(const SequenceRecord &record,
                 const std::vector<double> &scores, const ModelConfig &config) {
  const double total = std::accumulate(scores.begin(), scores.end(), 0.0);
  const double mean = total / static_cast<double>(scores.size());
  std::cout << "{\"name\":";
  write_json_string(std::cout, record.name);
  std::cout << ",\"backend\":\"cpu\",\"profile\":\"cpu-f32\",\"model_id\":";
  write_json_string(std::cout, config.model_id);
  std::cout << ",\"tokens\":" << record.bytes.size()
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
}

Status run_generate(const CliOptions &options, const Model &model) {
  Context context;
  auto status = context.initialize_shared(model, options.context_size);
  if (!status.ok())
    return status;
  const auto prompt = encode_bytes(options.prompt);
  const auto prefill_count =
      std::min(prompt.size(),
               options.force_prompt_threshold.value_or(
                   std::min<std::size_t>(3000, context.activation_capacity())));
  std::vector<TokenId> initial(prompt.begin(),
                               prompt.begin() +
                                   static_cast<std::ptrdiff_t>(prefill_count));
  std::vector<float> logits;
  status = prefill(&context, initial, false, &logits);
  if (!status.ok())
    return {status.code(), "CPU prompt prefill: " + status.message()};
  const auto columns = model.config().vocab_size;
  std::vector<float> current(
      logits.end() - static_cast<std::ptrdiff_t>(columns), logits.end());
  for (std::size_t index = prefill_count; index < prompt.size(); ++index) {
    status = context.decode(prompt[index], &current);
    if (!status.ok())
      return status;
  }
  Sampler sampler(options.sampling);
  std::string generated;
  std::vector<float> dump;
  generated.reserve(options.generated_tokens);
  if (options.dump_logits_path.has_value())
    dump.reserve(options.generated_tokens * columns);
  for (std::size_t step = 0; step < options.generated_tokens; ++step) {
    if (options.dump_logits_path.has_value())
      dump.insert(dump.end(), current.begin(), current.end());
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
  if (options.dump_logits_path.has_value()) {
    status = write_npy_f32(*options.dump_logits_path, dump,
                           options.generated_tokens, columns);
    if (!status.ok())
      return status;
  }
  std::cout.write(generated.data(),
                  static_cast<std::streamsize>(generated.size()));
  std::cout.flush();
  return std::cout ? Status::Ok()
                   : Status{ErrorCode::kIo, "failed writing generated bytes"};
}

Status run_score(const CliOptions &options, const Model &model) {
  std::size_t record_count = 0;
  auto status = stream_sequence_file(
      options.score_path, options.context_size,
      [&](const SequenceRecord &record) -> Status {
        if (record.bytes.size() < 2)
          return {ErrorCode::kInvalidArgument,
                  "score record '" + record.name + "' must contain two bytes"};
        if (record_count != 0 && options.dump_logits_path.has_value())
          return {ErrorCode::kInvalidArgument,
                  "--dump-logits requires exactly one input record"};
        Context context;
        auto inner = context.initialize_shared(model, options.context_size);
        if (!inner.ok())
          return inner;
        const auto tokens = encode_bytes(record.bytes);
        std::vector<float> logits;
        std::vector<double> scores;
        inner = prefill(&context, tokens, options.dump_logits_path.has_value(),
                        &logits, &scores);
        if (!inner.ok())
          return inner;
        if (options.dump_logits_path.has_value()) {
          inner = write_npy_f32(*options.dump_logits_path, logits,
                                tokens.size(), model.config().vocab_size);
          if (!inner.ok())
            return inner;
        }
        print_score(record, scores, model.config());
        ++record_count;
        return Status::Ok();
      });
  if (!status.ok())
    return status;
  std::cout.flush();
  return std::cout ? Status::Ok()
                   : Status{ErrorCode::kIo, "failed writing CPU score JSONL"};
}

Status sequence_score(const std::string_view sequence, const Model &model,
                      const std::size_t capacity, double *const result) {
  Context context;
  auto status = context.initialize_shared(model, capacity);
  if (!status.ok())
    return status;
  const auto tokens = encode_bytes(sequence);
  std::vector<float> logits;
  std::vector<double> scores;
  status = prefill(&context, tokens, false, &logits, &scores);
  if (!status.ok())
    return status;
  *result = std::accumulate(scores.begin(), scores.end(), 0.0);
  return Status::Ok();
}

struct StrandScore final {
  const char *strand{nullptr};
  double reference{0.0};
  double alternate{0.0};
  double delta{0.0};
};

Status run_variant(const CliOptions &options, const Model &model) {
  VariantWindow window;
  auto status = make_variant_window(
      options.variant_sequence, options.variant_position_1based,
      options.variant_reference, options.variant_alternate,
      options.variant_window_tokens, &window);
  if (!status.ok())
    return status;
  std::vector<StrandScore> scores;
  const auto add = [&](const std::string_view reference,
                       const std::string_view alternate,
                       const VariantStrand strand) -> Status {
    StrandScore score;
    score.strand = variant_strand_name(strand);
    auto inner = sequence_score(reference, model, options.context_size,
                                &score.reference);
    if (!inner.ok())
      return inner;
    inner = sequence_score(alternate, model, options.context_size,
                           &score.alternate);
    if (!inner.ok())
      return inner;
    if (options.variant_normalization == VariantNormalization::kMean) {
      score.reference /= static_cast<double>(reference.size() - 1);
      score.alternate /= static_cast<double>(alternate.size() - 1);
    }
    score.delta = score.alternate - score.reference;
    scores.push_back(score);
    return Status::Ok();
  };
  if (options.variant_strand != VariantStrand::kReverse) {
    status = add(window.reference, window.alternate, VariantStrand::kForward);
    if (!status.ok())
      return status;
  }
  if (options.variant_strand != VariantStrand::kForward) {
    std::string reference;
    std::string alternate;
    status = reverse_complement(window.reference, &reference);
    if (status.ok())
      status = reverse_complement(window.alternate, &alternate);
    if (status.ok())
      status = add(reference, alternate, VariantStrand::kReverse);
    if (!status.ok())
      return status;
  }
  double aggregate = 0.0;
  for (const auto &score : scores)
    aggregate += score.delta;
  aggregate /= static_cast<double>(scores.size());
  std::cout << "{\"position\":" << options.variant_position_1based
            << ",\"position_coordinate_system\":\"1-based\",\"reference\":";
  write_json_string(std::cout, options.variant_reference);
  std::cout << ",\"alternate\":";
  write_json_string(std::cout, options.variant_alternate);
  std::cout << ",\"window\":{\"start\":" << window.reference_start
            << ",\"end\":" << window.reference_end
            << ",\"coordinate_system\":\"0-based-half-open\"}"
            << ",\"normalization\":\""
            << variant_normalization_name(options.variant_normalization)
            << "\",\"backend\":\"cpu\",\"profile\":\"cpu-f32\",\"strands\":[";
  for (std::size_t index = 0; index < scores.size(); ++index) {
    if (index != 0)
      std::cout.put(',');
    std::cout << "{\"strand\":\"" << scores[index].strand
              << "\",\"reference_log_likelihood\":" << std::setprecision(17)
              << scores[index].reference
              << ",\"alternate_log_likelihood\":" << scores[index].alternate
              << ",\"delta\":" << scores[index].delta << '}';
  }
  std::cout << "],\"aggregation\":\""
            << (scores.size() == 1 ? "single_strand" : "mean_across_strands")
            << "\",\"score\":" << aggregate << "}\n";
  return Status::Ok();
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

Status run_embed(const CliOptions &options, const Model &model) {
  if (options.embed_layer >= model.config().layers)
    return {ErrorCode::kInvalidArgument, "embedding layer exceeds CPU model"};
  const std::filesystem::path directory{options.embed_output_dir};
  std::error_code error;
  if (std::filesystem::exists(directory, error)) {
    if (error || !std::filesystem::is_directory(directory, error) ||
        std::filesystem::directory_iterator{directory, error} !=
            std::filesystem::directory_iterator{}) {
      return {ErrorCode::kInvalidArgument,
              "embedding output directory must be empty"};
    }
  } else if (!std::filesystem::create_directories(directory, error) || error) {
    return {ErrorCode::kIo, "cannot create embedding output directory"};
  }
  std::ofstream manifest(directory / "embeddings.jsonl", std::ios::binary);
  if (!manifest)
    return {ErrorCode::kIo, "cannot create embedding manifest"};
  std::size_t record_index = 0;
  return stream_sequence_file(
      options.embed_path, options.context_size,
      [&](const SequenceRecord &record) -> Status {
        Context context;
        auto status = context.initialize_shared(model, options.context_size);
        if (!status.ok())
          return status;
        const auto tokens = encode_bytes(record.bytes);
        std::vector<float> all;
        std::size_t offset = 0;
        bool first = true;
        while (offset < tokens.size()) {
          const auto rows =
              std::min(context.activation_capacity(), tokens.size() - offset);
          const std::vector<TokenId> chunk(
              tokens.begin() + static_cast<std::ptrdiff_t>(offset),
              tokens.begin() + static_cast<std::ptrdiff_t>(offset + rows));
          std::vector<float> values;
          status = first ? context.prefill_embedding(chunk, options.embed_layer,
                                                     &values)
                         : context.prefill_chunk_embedding(
                               chunk, options.embed_layer, &values);
          if (!status.ok())
            return status;
          all.insert(all.end(), values.begin(), values.end());
          offset += rows;
          first = false;
        }
        const auto width = model.config().width;
        std::size_t rows = tokens.size();
        if (options.embedding_pooling == EmbeddingPooling::kMean) {
          std::vector<float> pooled(width, 0.0F);
          for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < width; ++column)
              pooled[column] +=
                  all[row * width + column] / static_cast<float>(rows);
          all = std::move(pooled);
          rows = 1;
        } else if (options.embedding_pooling == EmbeddingPooling::kLast) {
          all = std::vector<float>(
              all.end() - static_cast<std::ptrdiff_t>(width), all.end());
          rows = 1;
        }
        std::ostringstream filename;
        filename << std::setw(6) << std::setfill('0') << record_index << ".npy";
        status = write_npy_f32(directory / filename.str(), all, rows, width);
        if (!status.ok())
          return status;
        manifest << "{\"record_index\":" << record_index << ",\"name\":";
        write_json_string(manifest, record.name);
        manifest << ",\"file\":";
        write_json_string(manifest, filename.str());
        manifest << ",\"source_tokens\":" << tokens.size() << ",\"shape\":["
                 << rows << ',' << width
                 << "],\"layer\":" << options.embed_layer
                 << ",\"point\":\"block_output\",\"pooling\":\""
                 << pooling_name(options.embedding_pooling)
                 << "\",\"dtype\":\"float32\",\"backend\":\"cpu\","
                    "\"profile\":\"cpu-f32\",\"model_id\":";
        write_json_string(manifest, model.config().model_id);
        manifest << "}\n";
        ++record_index;
        return manifest ? Status::Ok()
                        : Status{ErrorCode::kIo,
                                 "failed writing embedding manifest"};
      });
}

} // namespace

Status run_inference_cli(const CliOptions &options,
                         const bool allow_test_fixture) {
  if (options.inference_profile != InferenceProfile::kCpuF32)
    return {ErrorCode::kInvalidArgument,
            "CPU backend requires the cpu-f32 execution profile"};
  if (options.dump_layer.has_value())
    return {ErrorCode::kUnsupported,
            "--dump-layer is not available in the cpu-f32 CLI; use embed"};
  const auto load_start = Clock::now();
  ModelFile artifact;
  auto status = artifact.open(options.model_path);
  if (!status.ok())
    return status;
  Model model;
  status = model.load(artifact, allow_test_fixture);
  if (!status.ok())
    return status;
  const double load_seconds =
      std::chrono::duration<double>(Clock::now() - load_start).count();
  if (options.mode == RunMode::kGenerate)
    status = run_generate(options, model);
  else if (options.mode == RunMode::kScore)
    status = run_score(options, model);
  else if (options.mode == RunMode::kEmbed)
    status = run_embed(options, model);
  else if (options.mode == RunMode::kVariantScore)
    status = run_variant(options, model);
  else
    status = {ErrorCode::kUnsupported, "CPU server dispatch is separate"};
  std::cerr << "evo_metrics {\"backend\":\"cpu\",\"profile\":\"cpu-f32\","
               "\"kernel\":\""
            << model.kernel_name()
            << "\",\"model_load_seconds\":" << std::setprecision(9)
            << load_seconds << "}\n";
  return status;
}

} // namespace evo::cpu
