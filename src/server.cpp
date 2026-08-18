// SPDX-License-Identifier: Apache-2.0
#include "evo/server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "evo/cli.hpp"
#include "evo/evo.h"
#include "evo/json.hpp"
#include "evo/variant.hpp"

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace evo {
namespace {

using ModelPtr = std::unique_ptr<evo_model, decltype(&evo_model_free)>;
using ContextPtr = std::unique_ptr<evo_context, decltype(&evo_context_free)>;
using BatchPtr = std::unique_ptr<evo_batch, decltype(&evo_batch_free)>;
using SamplerPtr = std::unique_ptr<evo_sampler, decltype(&evo_sampler_free)>;

ServerResponse error_response(const int http_status,
                              const std::string_view code,
                              const std::string_view message) {
  std::string body{"{\"error\":{\"code\":"};
  append_json_string(&body, code);
  body.append(",\"message\":");
  append_json_string(&body, message);
  body.append("}}");
  return {http_status, "application/json", std::move(body)};
}

int http_status_for(const evo_status status) noexcept {
  switch (status) {
  case EVO_STATUS_INVALID_ARGUMENT:
    return 400;
  case EVO_STATUS_IO:
  case EVO_STATUS_MODEL_FORMAT:
    return 422;
  case EVO_STATUS_UNSUPPORTED:
    return 501;
  case EVO_STATUS_CUDA:
  case EVO_STATUS_MPS:
    return 503;
  case EVO_STATUS_INTERNAL:
  case EVO_STATUS_OK:
    return 500;
  }
  return 500;
}

Status status_from_c(const evo_status status, const std::string_view prefix) {
  if (status == EVO_STATUS_OK)
    return Status::Ok();
  const char *const detail = evo_last_error();
  return {static_cast<ErrorCode>(status),
          std::string{prefix} + (detail == nullptr || detail[0] == '\0'
                                     ? std::string{evo_status_name(status)}
                                     : std::string{detail})};
}

ServerResponse c_error(const evo_status status,
                       const std::string_view fallback) {
  const char *const detail = evo_last_error();
  return error_response(http_status_for(status), evo_status_name(status),
                        detail == nullptr || detail[0] == '\0'
                            ? fallback
                            : std::string_view{detail});
}

void append_number(std::string *const output, const double value) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(17) << value;
  output->append(stream.str());
}

bool integer_value(const JsonValue &value, const std::size_t maximum,
                   std::size_t *const output) noexcept {
  if (value.type != JsonType::kNumber || !std::isfinite(value.number) ||
      value.number < 0.0 || value.number > static_cast<double>(maximum)) {
    return false;
  }
  double integral = 0.0;
  if (std::modf(value.number, &integral) != 0.0)
    return false;
  *output = static_cast<std::size_t>(integral);
  return true;
}

Status require_object(const JsonValue &root,
                      const std::set<std::string_view> &allowed) {
  if (root.type != JsonType::kObject)
    return {ErrorCode::kInvalidArgument, "request body must be a JSON object"};
  for (const auto &member : root.object) {
    if (allowed.find(member.first) == allowed.end()) {
      return {ErrorCode::kInvalidArgument,
              "unknown request field '" + member.first + "'"};
    }
  }
  return Status::Ok();
}

Status required_string(const JsonValue &root, const std::string_view key,
                       std::string *const output) {
  const auto *const value = root.find(key);
  if (value == nullptr || value->type != JsonType::kString ||
      value->string.empty()) {
    return {ErrorCode::kInvalidArgument,
            "field '" + std::string{key} + "' must be a nonempty JSON string"};
  }
  *output = value->string;
  return Status::Ok();
}

Status optional_string(const JsonValue &root, const std::string_view key,
                       const std::string_view fallback, std::string *output) {
  const auto *const value = root.find(key);
  if (value == nullptr) {
    *output = fallback;
    return Status::Ok();
  }
  if (value->type != JsonType::kString || value->string.empty()) {
    return {ErrorCode::kInvalidArgument,
            "field '" + std::string{key} + "' must be a nonempty JSON string"};
  }
  *output = value->string;
  return Status::Ok();
}

Status required_size(const JsonValue &root, const std::string_view key,
                     const std::size_t minimum, const std::size_t maximum,
                     std::size_t *const output) {
  const auto *const value = root.find(key);
  if (value == nullptr || !integer_value(*value, maximum, output) ||
      *output < minimum) {
    return {ErrorCode::kInvalidArgument,
            "field '" + std::string{key} + "' must be an integer in [" +
                std::to_string(minimum) + ", " + std::to_string(maximum) + "]"};
  }
  return Status::Ok();
}

Status optional_size(const JsonValue &root, const std::string_view key,
                     const std::size_t fallback, const std::size_t minimum,
                     const std::size_t maximum, std::size_t *const output) {
  const auto *const value = root.find(key);
  if (value == nullptr) {
    *output = fallback;
    return Status::Ok();
  }
  if (!integer_value(*value, maximum, output) || *output < minimum) {
    return {ErrorCode::kInvalidArgument,
            "field '" + std::string{key} + "' must be an integer in [" +
                std::to_string(minimum) + ", " + std::to_string(maximum) + "]"};
  }
  return Status::Ok();
}

Status optional_float(const JsonValue &root, const std::string_view key,
                      const float fallback, float *const output) {
  const auto *const value = root.find(key);
  if (value == nullptr) {
    *output = fallback;
    return Status::Ok();
  }
  if (value->type != JsonType::kNumber || !std::isfinite(value->number) ||
      value->number < -static_cast<double>(std::numeric_limits<float>::max()) ||
      value->number > static_cast<double>(std::numeric_limits<float>::max())) {
    return {ErrorCode::kInvalidArgument,
            "field '" + std::string{key} + "' must be a finite F32 number"};
  }
  *output = static_cast<float>(value->number);
  return Status::Ok();
}

class Runtime final {
public:
  Runtime(ModelPtr model, const std::size_t context_size,
          const std::size_t maximum_sequence,
          const std::size_t maximum_embedding_values,
          const InferenceProfile inference_profile)
      : model_(std::move(model)), context_size_(context_size),
        maximum_sequence_(maximum_sequence),
        maximum_embedding_values_(maximum_embedding_values),
        backend_(evo_backend_name(evo_model_backend(model_.get()))),
        model_id_(evo_model_id(model_.get())),
        architecture_(evo_model_architecture(model_.get())),
        profile_(evo_model_profile(model_.get())),
        inference_profile_(inference_profile),
        vocabulary_(evo_model_vocab_size(model_.get())),
        embedding_width_(evo_model_embedding_width(model_.get())),
        layer_count_(evo_model_layer_count(model_.get())) {}

  [[nodiscard]] const std::string &model_id() const noexcept {
    return model_id_;
  }
  [[nodiscard]] const std::string &backend() const noexcept { return backend_; }
  [[nodiscard]] const std::string &architecture() const noexcept {
    return architecture_;
  }
  [[nodiscard]] const std::string &profile() const noexcept { return profile_; }
  [[nodiscard]] InferenceProfile inference_profile() const noexcept {
    return inference_profile_;
  }
  [[nodiscard]] std::size_t context_size() const noexcept {
    return context_size_;
  }
  [[nodiscard]] std::size_t maximum_sequence() const noexcept {
    return maximum_sequence_;
  }
  [[nodiscard]] std::size_t maximum_embedding_values() const noexcept {
    return maximum_embedding_values_;
  }

  ServerResponse dispatch(const std::string_view path,
                          const std::string_view body,
                          const CancellationToken &cancel) const {
    if (cancel.cancelled())
      return error_response(499, "cancelled", "request was cancelled");
    JsonValue root;
    auto status = parse_json(body, &root);
    if (!status.ok())
      return error_response(400, "invalid_json", status.message());
    if (path == "/v1/generate")
      return generate(root, cancel);
    if (path == "/v1/score")
      return score(root, cancel);
    if (path == "/v1/embeddings")
      return embeddings(root, cancel);
    if (path == "/v1/variants")
      return variants(root, cancel);
    return error_response(404, "not_found", "route does not exist");
  }

private:
  struct LogitsState final {
    const CancellationToken *cancel{nullptr};
    std::vector<float> last;
  };

  struct ScoreState final {
    const CancellationToken *cancel{nullptr};
    const std::vector<std::uint32_t> *tokens{nullptr};
    std::vector<double> scores;
    std::string error;
  };

  struct EmbeddingState final {
    const CancellationToken *cancel{nullptr};
    std::string pooling;
    std::size_t limit{0};
    std::size_t rows{0};
    std::size_t columns{0};
    std::vector<double> values;
    std::string error;
  };

  struct EmbeddingExState final {
    const CancellationToken *cancel{nullptr};
    std::size_t limit{0};
    std::vector<double> values;
    std::string resolved_preset;
    std::string hidden_tap;
    std::string pooling;
    std::size_t layer{0};
    std::size_t original_length{0};
    std::size_t effective_length{0};
    std::size_t crop_left{0};
    std::size_t crop_right{0};
    std::size_t pad_left{0};
    std::size_t pad_right{0};
    std::size_t token_count{0};
    std::size_t rows{0};
    std::size_t columns{0};
    std::string error;
  };

  static evo_status capture_logits(const float *const logits,
                                   const std::size_t rows,
                                   const std::size_t columns, const std::size_t,
                                   void *const user_data) {
    auto *const state = static_cast<LogitsState *>(user_data);
    if (state == nullptr || logits == nullptr || rows == 0 || columns == 0)
      return EVO_STATUS_INTERNAL;
    if (state->cancel != nullptr && state->cancel->cancelled())
      return EVO_STATUS_INVALID_ARGUMENT;
    const float *const final = logits + (rows - 1) * columns;
    state->last.assign(final, final + columns);
    return EVO_STATUS_OK;
  }

  static evo_status score_logits(const float *const logits,
                                 const std::size_t rows,
                                 const std::size_t columns,
                                 const std::size_t token_offset,
                                 void *const user_data) {
    auto *const state = static_cast<ScoreState *>(user_data);
    if (state == nullptr || logits == nullptr || columns == 0)
      return EVO_STATUS_INTERNAL;
    if (state->cancel != nullptr && state->cancel->cancelled())
      return EVO_STATUS_INVALID_ARGUMENT;
    for (std::size_t local = 0; local < rows; ++local) {
      const std::size_t row = token_offset + local;
      if (state->tokens == nullptr || row + 1 >= state->tokens->size())
        break;
      const auto target = (*state->tokens)[row + 1];
      if (target >= columns) {
        state->error = "target token is outside the model vocabulary";
        return EVO_STATUS_MODEL_FORMAT;
      }
      const float *const values = logits + local * columns;
      double maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t column = 0; column < columns; ++column) {
        if (!std::isfinite(values[column])) {
          state->error = "model returned a non-finite score logit";
          return EVO_STATUS_INTERNAL;
        }
        maximum = std::max(maximum, static_cast<double>(values[column]));
      }
      double denominator = 0.0;
      for (std::size_t column = 0; column < columns; ++column)
        denominator += std::exp(static_cast<double>(values[column]) - maximum);
      if (!(denominator > 0.0) || !std::isfinite(denominator)) {
        state->error = "score log-softmax normalization failed";
        return EVO_STATUS_INTERNAL;
      }
      state->scores.push_back(static_cast<double>(values[target]) - maximum -
                              std::log(denominator));
    }
    return EVO_STATUS_OK;
  }

  static evo_status capture_embeddings(const float *const embedding,
                                       const std::size_t rows,
                                       const std::size_t columns,
                                       const std::size_t,
                                       void *const user_data) {
    auto *const state = static_cast<EmbeddingState *>(user_data);
    if (state == nullptr || embedding == nullptr || rows == 0 || columns == 0)
      return EVO_STATUS_INTERNAL;
    if (state->cancel != nullptr && state->cancel->cancelled())
      return EVO_STATUS_INVALID_ARGUMENT;
    if (state->columns != 0 && state->columns != columns) {
      state->error = "embedding width changed between chunks";
      return EVO_STATUS_INTERNAL;
    }
    state->columns = columns;
    if (state->pooling == "none") {
      if (rows > (state->limit - state->values.size()) / columns) {
        state->error = "embedding response exceeds configured value limit";
        return EVO_STATUS_INVALID_ARGUMENT;
      }
      state->values.reserve(state->values.size() + rows * columns);
      for (std::size_t index = 0; index < rows * columns; ++index) {
        if (!std::isfinite(embedding[index])) {
          state->error = "model returned a non-finite embedding";
          return EVO_STATUS_INTERNAL;
        }
        state->values.push_back(embedding[index]);
      }
    } else if (state->pooling == "mean") {
      if (state->values.empty())
        state->values.assign(columns, 0.0);
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
          const float value = embedding[row * columns + column];
          if (!std::isfinite(value)) {
            state->error = "model returned a non-finite embedding";
            return EVO_STATUS_INTERNAL;
          }
          state->values[column] += value;
        }
      }
    } else {
      state->values.resize(columns);
      const float *const last = embedding + (rows - 1) * columns;
      for (std::size_t column = 0; column < columns; ++column) {
        if (!std::isfinite(last[column])) {
          state->error = "model returned a non-finite embedding";
          return EVO_STATUS_INTERNAL;
        }
        state->values[column] = last[column];
      }
    }
    state->rows += rows;
    return EVO_STATUS_OK;
  }

  static evo_status capture_embeddings_ex(
      const float *const embedding, const std::size_t rows,
      const std::size_t columns, const std::size_t,
      const evo_embedding_result_info *const info, void *const user_data) {
    auto *const state = static_cast<EmbeddingExState *>(user_data);
    if (state == nullptr || embedding == nullptr || info == nullptr ||
        info->struct_size < sizeof(evo_embedding_result_info) || rows == 0 ||
        columns == 0 || info->rows != rows || info->columns != columns ||
        info->resolved_preset == nullptr || info->hidden_tap == nullptr ||
        info->pooling == nullptr) {
      return EVO_STATUS_INTERNAL;
    }
    if (state->cancel != nullptr && state->cancel->cancelled())
      return EVO_STATUS_INVALID_ARGUMENT;
    if (rows > state->limit / columns) {
      state->error = "embedding response exceeds configured value limit";
      return EVO_STATUS_INVALID_ARGUMENT;
    }
    state->values.reserve(rows * columns);
    for (std::size_t index = 0; index < rows * columns; ++index) {
      if (!std::isfinite(embedding[index])) {
        state->error = "model returned a non-finite embedding";
        return EVO_STATUS_INTERNAL;
      }
      state->values.push_back(embedding[index]);
    }
    state->resolved_preset = info->resolved_preset;
    state->hidden_tap = info->hidden_tap;
    state->pooling = info->pooling;
    state->layer = info->layer;
    state->original_length = info->original_length;
    state->effective_length = info->effective_length;
    state->crop_left = info->crop_left;
    state->crop_right = info->crop_right;
    state->pad_left = info->pad_left;
    state->pad_right = info->pad_right;
    state->token_count = info->token_count;
    state->rows = info->rows;
    state->columns = info->columns;
    return EVO_STATUS_OK;
  }

  ServerResponse validation_error(const Status &status) const {
    return error_response(400, "invalid_argument", status.message());
  }

  ServerResponse cancelled_or_c_error(const evo_status status,
                                      const CancellationToken &cancel,
                                      const std::string &callback_error,
                                      const std::string_view fallback) const {
    if (cancel.cancelled())
      return error_response(499, "cancelled", "request was cancelled");
    if (!callback_error.empty())
      return error_response(http_status_for(status), evo_status_name(status),
                            callback_error);
    return c_error(status, fallback);
  }

  evo_status make_context(ContextPtr *const output) const {
    evo_context_params params = evo_context_default_params();
    params.context_size = context_size_;
    if (inference_profile_ == InferenceProfile::kFastQ8Kv)
      params.flags |= EVO_CONTEXT_FLAG_FAST_Q8_KV;
    evo_context *raw = nullptr;
    const evo_status status = evo_context_create(model_.get(), &params, &raw);
    output->reset(raw);
    return status;
  }

  static evo_status make_batch(const std::string_view sequence,
                               BatchPtr *const output) {
    evo_batch *raw = nullptr;
    evo_status status = evo_batch_create(1, &raw);
    output->reset(raw);
    if (status != EVO_STATUS_OK)
      return status;
    return evo_batch_add_sequence(
        output->get(), reinterpret_cast<const std::uint8_t *>(sequence.data()),
        sequence.size());
  }

  evo_status tokenize(const std::string_view sequence,
                      std::vector<std::uint32_t> *const tokens) const {
    std::size_t count = 0;
    evo_status status = evo_model_encode(
        model_.get(), reinterpret_cast<const std::uint8_t *>(sequence.data()),
        sequence.size(), nullptr, 0, &count);
    if (status != EVO_STATUS_OK)
      return status;
    tokens->resize(count);
    return evo_model_encode(
        model_.get(), reinterpret_cast<const std::uint8_t *>(sequence.data()),
        sequence.size(), tokens->data(), tokens->size(), &count);
  }

  ServerResponse generate(const JsonValue &root,
                          const CancellationToken &cancel) const {
    auto status = require_object(root, {"prompt", "max_tokens", "temperature",
                                        "top_k", "top_p", "seed"});
    if (!status.ok())
      return validation_error(status);
    std::string prompt;
    status = required_string(root, "prompt", &prompt);
    if (!status.ok())
      return validation_error(status);
    std::size_t maximum_tokens = 0;
    status =
        required_size(root, "max_tokens", 1, context_size_, &maximum_tokens);
    if (!status.ok())
      return validation_error(status);
    if (prompt.size() > maximum_sequence_ ||
        maximum_tokens >
            context_size_ - std::min(prompt.size(), context_size_)) {
      return error_response(413, "request_too_large",
                            "prompt and generated tokens exceed server limits");
    }
    evo_sampler_params sampler_params = evo_sampler_default_params();
    status = optional_float(root, "temperature", sampler_params.temperature,
                            &sampler_params.temperature);
    if (!status.ok())
      return validation_error(status);
    status = optional_float(root, "top_p", sampler_params.top_p,
                            &sampler_params.top_p);
    if (!status.ok())
      return validation_error(status);
    status = optional_size(root, "top_k", sampler_params.top_k, 0, vocabulary_,
                           &sampler_params.top_k);
    if (!status.ok())
      return validation_error(status);
    std::size_t seed = 0;
    status = optional_size(root, "seed", 0, 0,
                           static_cast<std::size_t>(1ULL << 53U), &seed);
    if (!status.ok())
      return validation_error(status);
    sampler_params.seed = static_cast<std::uint64_t>(seed);

    ContextPtr context{nullptr, &evo_context_free};
    evo_status c_status = make_context(&context);
    if (c_status != EVO_STATUS_OK)
      return c_error(c_status, "failed to create request context");
    BatchPtr batch{nullptr, &evo_batch_free};
    c_status = make_batch(prompt, &batch);
    if (c_status != EVO_STATUS_OK)
      return c_error(c_status, "failed to create prompt batch");
    SamplerPtr sampler{nullptr, &evo_sampler_free};
    evo_sampler *raw_sampler = nullptr;
    c_status = evo_sampler_create(&sampler_params, &raw_sampler);
    sampler.reset(raw_sampler);
    if (c_status != EVO_STATUS_OK)
      return c_error(c_status, "invalid sampler parameters");

    LogitsState logits{&cancel, {}};
    c_status = evo_context_prefill(context.get(), batch.get(), capture_logits,
                                   &logits);
    if (c_status != EVO_STATUS_OK)
      return cancelled_or_c_error(c_status, cancel, {}, "prefill failed");
    if (logits.last.size() != vocabulary_)
      return error_response(500, "internal", "prefill returned wrong logits");

    std::string generated;
    generated.reserve(maximum_tokens);
    std::vector<std::uint32_t> tokens;
    tokens.reserve(maximum_tokens);
    for (std::size_t step = 0; step < maximum_tokens; ++step) {
      if (cancel.cancelled())
        return error_response(499, "cancelled", "request was cancelled");
      std::uint32_t token = 0;
      c_status = evo_sampler_sample(sampler.get(), logits.last.data(),
                                    logits.last.size(), &token);
      if (c_status != EVO_STATUS_OK)
        return c_error(c_status, "sampling failed");
      std::uint8_t byte = 0;
      c_status = evo_model_decode_token(model_.get(), token, &byte);
      if (c_status != EVO_STATUS_OK)
        return c_error(c_status,
                       "sampled token has no raw sequence representation");
      tokens.push_back(token);
      generated.push_back(static_cast<char>(byte));
      if (step + 1 == maximum_tokens)
        break;
      c_status =
          evo_context_decode(context.get(), token, capture_logits, &logits);
      if (c_status != EVO_STATUS_OK)
        return cancelled_or_c_error(c_status, cancel, {}, "decode failed");
    }
    std::string body{"{\"model\":"};
    append_json_string(&body, model_id_);
    body.append(",\"profile\":");
    append_json_string(&body, inference_profile_name(inference_profile_));
    body.append(",\"generated\":");
    append_json_string(&body, generated);
    body.append(",\"tokens\":[");
    for (std::size_t index = 0; index < tokens.size(); ++index) {
      if (index != 0)
        body.push_back(',');
      body.append(std::to_string(tokens[index]));
    }
    body.append("],\"generated_tokens\":");
    body.append(std::to_string(tokens.size()));
    body.push_back('}');
    return {200, "application/json", std::move(body)};
  }

  ServerResponse score_sequence(const std::string_view sequence,
                                const CancellationToken &cancel,
                                std::vector<double> *const scores) const {
    ContextPtr context{nullptr, &evo_context_free};
    evo_status c_status = make_context(&context);
    if (c_status != EVO_STATUS_OK)
      return c_error(c_status, "failed to create request context");
    BatchPtr batch{nullptr, &evo_batch_free};
    c_status = make_batch(sequence, &batch);
    if (c_status != EVO_STATUS_OK)
      return c_error(c_status, "failed to create score batch");
    std::vector<std::uint32_t> target_tokens;
    c_status = tokenize(sequence, &target_tokens);
    if (c_status != EVO_STATUS_OK)
      return c_error(c_status, "failed to tokenize score sequence");
    ScoreState state{&cancel, &target_tokens, {}, {}};
    state.scores.reserve(sequence.size() - 1);
    c_status =
        evo_context_prefill(context.get(), batch.get(), score_logits, &state);
    if (c_status != EVO_STATUS_OK)
      return cancelled_or_c_error(c_status, cancel, state.error,
                                  "score prefill failed");
    if (state.scores.size() != sequence.size() - 1) {
      return error_response(500, "internal",
                            "score prefill returned incomplete results");
    }
    *scores = std::move(state.scores);
    return {200, "application/json", {}};
  }

  ServerResponse score(const JsonValue &root,
                       const CancellationToken &cancel) const {
    auto status = require_object(root, {"sequence"});
    if (!status.ok())
      return validation_error(status);
    std::string sequence;
    status = required_string(root, "sequence", &sequence);
    if (!status.ok())
      return validation_error(status);
    if (sequence.size() < 2)
      return error_response(400, "invalid_argument",
                            "sequence must contain at least two bytes");
    if (sequence.size() > maximum_sequence_)
      return error_response(413, "request_too_large",
                            "sequence exceeds server limit");
    std::vector<double> scores;
    auto response = score_sequence(sequence, cancel, &scores);
    if (response.http_status != 200)
      return response;
    double total = 0.0;
    for (const double value : scores)
      total += value;
    const double mean = total / static_cast<double>(scores.size());
    std::string body{"{\"model\":"};
    append_json_string(&body, model_id_);
    body.append(",\"profile\":");
    append_json_string(&body, inference_profile_name(inference_profile_));
    body.append(",\"sequence_bytes\":");
    body.append(std::to_string(sequence.size()));
    body.append(",\"scored_tokens\":");
    body.append(std::to_string(scores.size()));
    body.append(",\"total_log_likelihood\":");
    append_number(&body, total);
    body.append(",\"mean_log_likelihood\":");
    append_number(&body, mean);
    body.append(",\"perplexity\":");
    append_number(&body, std::exp(-mean));
    body.append(",\"token_log_likelihoods\":[");
    for (std::size_t index = 0; index < scores.size(); ++index) {
      if (index != 0)
        body.push_back(',');
      append_number(&body, scores[index]);
    }
    body.append("]}");
    return {200, "application/json", std::move(body)};
  }

  ServerResponse embeddings(const JsonValue &root,
                            const CancellationToken &cancel) const {
    auto status =
        require_object(root, {"sequence", "layer", "pooling", "preset"});
    if (!status.ok())
      return validation_error(status);
    std::string sequence;
    status = required_string(root, "sequence", &sequence);
    if (!status.ok())
      return validation_error(status);
    if (sequence.size() > maximum_sequence_)
      return error_response(413, "request_too_large",
                            "sequence exceeds server limit");
    const bool preset_requested = root.find("preset") != nullptr;
    if (preset_requested) {
      if (root.find("layer") != nullptr || root.find("pooling") != nullptr) {
        return error_response(
            400, "invalid_argument",
            "preset is mutually exclusive with layer and pooling");
      }
      std::string preset;
      status = required_string(root, "preset", &preset);
      if (!status.ok())
        return validation_error(status);
      if (preset != "geneb-v4-reference" &&
          preset != "geneb-v4-normalized") {
        return error_response(
            400, "invalid_argument",
            "preset must be geneb-v4-reference or geneb-v4-normalized");
      }
      if (embedding_width_ == 0 ||
          embedding_width_ > maximum_embedding_values_) {
        return error_response(413, "response_too_large",
                              "embedding exceeds configured value limit");
      }
      ContextPtr context{nullptr, &evo_context_free};
      evo_status c_status = make_context(&context);
      if (c_status != EVO_STATUS_OK)
        return c_error(c_status, "failed to create request context");
      BatchPtr batch{nullptr, &evo_batch_free};
      c_status = make_batch(sequence, &batch);
      if (c_status != EVO_STATUS_OK)
        return c_error(c_status, "failed to create embedding batch");
      evo_embedding_options options = evo_embedding_default_options();
      options.preset = preset.c_str();
      EmbeddingExState state;
      state.cancel = &cancel;
      state.limit = maximum_embedding_values_;
      c_status = evo_context_embed_ex(context.get(), batch.get(), &options,
                                      capture_embeddings_ex, &state);
      if (c_status != EVO_STATUS_OK) {
        return cancelled_or_c_error(c_status, cancel, state.error,
                                    "embedding inference failed");
      }
      std::string body{"{\"model\":"};
      append_json_string(&body, model_id_);
      body.append(",\"profile\":");
      append_json_string(&body, inference_profile_name(inference_profile_));
      body.append(",\"preset\":");
      append_json_string(&body, state.resolved_preset);
      body.append(",\"hidden_tap\":");
      append_json_string(&body, state.hidden_tap);
      body.append(",\"layer\":");
      body.append(std::to_string(state.layer));
      body.append(",\"pooling\":");
      append_json_string(&body, state.pooling);
      body.append(",\"input\":{\"original_length\":");
      body.append(std::to_string(state.original_length));
      body.append(",\"effective_length\":");
      body.append(std::to_string(state.effective_length));
      body.append(",\"crop_left\":");
      body.append(std::to_string(state.crop_left));
      body.append(",\"crop_right\":");
      body.append(std::to_string(state.crop_right));
      body.append(",\"pad_left\":");
      body.append(std::to_string(state.pad_left));
      body.append(",\"pad_right\":");
      body.append(std::to_string(state.pad_right));
      body.append(",\"token_count\":");
      body.append(std::to_string(state.token_count));
      body.append("},\"shape\":[");
      body.append(std::to_string(state.rows));
      body.push_back(',');
      body.append(std::to_string(state.columns));
      body.append("],\"embedding\":[");
      for (std::size_t index = 0; index < state.values.size(); ++index) {
        if (index != 0)
          body.push_back(',');
        append_number(&body, state.values[index]);
      }
      body.append("]}");
      return {200, "application/json", std::move(body)};
    }
    std::size_t layer = 0;
    status = required_size(root, "layer", 0,
                           layer_count_ == 0 ? 0 : layer_count_ - 1, &layer);
    if (!status.ok())
      return validation_error(status);
    std::string pooling;
    status = optional_string(root, "pooling", "none", &pooling);
    if (!status.ok())
      return validation_error(status);
    if (pooling != "none" && pooling != "mean" && pooling != "last") {
      return error_response(400, "invalid_argument",
                            "pooling must be none, mean, or last");
    }
    if (embedding_width_ == 0 ||
        (pooling == "none" &&
         sequence.size() > maximum_embedding_values_ / embedding_width_) ||
        (pooling != "none" && embedding_width_ > maximum_embedding_values_)) {
      return error_response(413, "response_too_large",
                            "embedding exceeds configured value limit");
    }
    ContextPtr context{nullptr, &evo_context_free};
    evo_status c_status = make_context(&context);
    if (c_status != EVO_STATUS_OK)
      return c_error(c_status, "failed to create request context");
    BatchPtr batch{nullptr, &evo_batch_free};
    c_status = make_batch(sequence, &batch);
    if (c_status != EVO_STATUS_OK)
      return c_error(c_status, "failed to create embedding batch");
    EmbeddingState state{&cancel, pooling, maximum_embedding_values_, 0, 0,
                         {},      {}};
    c_status = evo_context_embed(context.get(), batch.get(), layer,
                                 capture_embeddings, &state);
    if (c_status != EVO_STATUS_OK)
      return cancelled_or_c_error(c_status, cancel, state.error,
                                  "embedding inference failed");
    if (pooling == "mean") {
      for (double &value : state.values)
        value /= static_cast<double>(state.rows);
    }
    std::string body{"{\"model\":"};
    append_json_string(&body, model_id_);
    body.append(",\"profile\":");
    append_json_string(&body, inference_profile_name(inference_profile_));
    body.append(",\"layer\":");
    body.append(std::to_string(layer));
    body.append(",\"pooling\":");
    append_json_string(&body, pooling);
    body.append(",\"shape\":[");
    if (pooling == "none") {
      body.append(std::to_string(state.rows));
      body.push_back(',');
    }
    body.append(std::to_string(state.columns));
    body.append("],\"embedding\":[");
    for (std::size_t index = 0; index < state.values.size(); ++index) {
      if (index != 0)
        body.push_back(',');
      append_number(&body, state.values[index]);
    }
    body.append("]}");
    return {200, "application/json", std::move(body)};
  }

  struct StrandScore final {
    const char *strand{"?"};
    double reference{0.0};
    double alternate{0.0};
    std::size_t reference_tokens{0};
    std::size_t alternate_tokens{0};
    double delta{0.0};
  };

  ServerResponse variant_strand(const std::string_view reference,
                                const std::string_view alternate,
                                const VariantStrand strand,
                                const VariantNormalization normalization,
                                const CancellationToken &cancel,
                                StrandScore *const output) const {
    std::vector<double> reference_scores;
    auto response = score_sequence(reference, cancel, &reference_scores);
    if (response.http_status != 200)
      return response;
    std::vector<double> alternate_scores;
    response = score_sequence(alternate, cancel, &alternate_scores);
    if (response.http_status != 200)
      return response;
    output->strand = variant_strand_name(strand);
    output->reference_tokens = reference_scores.size();
    output->alternate_tokens = alternate_scores.size();
    for (const double value : reference_scores)
      output->reference += value;
    for (const double value : alternate_scores)
      output->alternate += value;
    if (normalization == VariantNormalization::kMean) {
      output->reference /= static_cast<double>(output->reference_tokens);
      output->alternate /= static_cast<double>(output->alternate_tokens);
    }
    output->delta = output->alternate - output->reference;
    return {200, "application/json", {}};
  }

  ServerResponse variants(const JsonValue &root,
                          const CancellationToken &cancel) const {
    auto status = require_object(root, {"sequence", "position", "ref", "alt",
                                        "window", "strand", "normalization"});
    if (!status.ok())
      return validation_error(status);
    std::string sequence;
    std::string reference;
    std::string alternate;
    status = required_string(root, "sequence", &sequence);
    if (!status.ok())
      return validation_error(status);
    status = required_string(root, "ref", &reference);
    if (!status.ok())
      return validation_error(status);
    status = required_string(root, "alt", &alternate);
    if (!status.ok())
      return validation_error(status);
    if (sequence.size() > maximum_sequence_)
      return error_response(413, "request_too_large",
                            "reference sequence exceeds server limit");
    std::size_t position = 0;
    status = required_size(root, "position", 1, sequence.size(), &position);
    if (!status.ok())
      return validation_error(status);
    std::size_t window_size = 0;
    status = optional_size(root, "window", maximum_sequence_, 2,
                           maximum_sequence_, &window_size);
    if (!status.ok())
      return validation_error(status);
    std::string strand_name;
    status = optional_string(root, "strand", "both", &strand_name);
    if (!status.ok())
      return validation_error(status);
    VariantStrand strand = VariantStrand::kBoth;
    if (strand_name == "forward")
      strand = VariantStrand::kForward;
    else if (strand_name == "reverse")
      strand = VariantStrand::kReverse;
    else if (strand_name != "both")
      return error_response(400, "invalid_argument",
                            "strand must be forward, reverse, or both");
    std::string normalization_name;
    status = optional_string(root, "normalization", "sum", &normalization_name);
    if (!status.ok())
      return validation_error(status);
    VariantNormalization normalization = VariantNormalization::kSum;
    if (normalization_name == "mean")
      normalization = VariantNormalization::kMean;
    else if (normalization_name != "sum")
      return error_response(400, "invalid_argument",
                            "normalization must be sum or mean");

    VariantWindow window;
    status = make_variant_window(sequence, position, reference, alternate,
                                 window_size, &window);
    if (!status.ok())
      return validation_error(status);
    std::vector<StrandScore> scores;
    if (strand == VariantStrand::kForward || strand == VariantStrand::kBoth) {
      StrandScore result;
      auto response = variant_strand(window.reference, window.alternate,
                                     VariantStrand::kForward, normalization,
                                     cancel, &result);
      if (response.http_status != 200)
        return response;
      scores.push_back(result);
    }
    if (strand == VariantStrand::kReverse || strand == VariantStrand::kBoth) {
      std::string reverse_reference;
      std::string reverse_alternate;
      status = reverse_complement(window.reference, &reverse_reference);
      if (!status.ok())
        return validation_error(status);
      status = reverse_complement(window.alternate, &reverse_alternate);
      if (!status.ok())
        return validation_error(status);
      StrandScore result;
      auto response = variant_strand(reverse_reference, reverse_alternate,
                                     VariantStrand::kReverse, normalization,
                                     cancel, &result);
      if (response.http_status != 200)
        return response;
      scores.push_back(result);
    }
    double aggregate = 0.0;
    for (const auto &entry : scores)
      aggregate += entry.delta;
    aggregate /= static_cast<double>(scores.size());

    std::string body{"{\"model\":"};
    append_json_string(&body, model_id_);
    body.append(",\"profile\":");
    append_json_string(&body, inference_profile_name(inference_profile_));
    body.append(",\"coordinate_system\":\"input_position_1_based;"
                "intervals_0_based_half_open\",\"position_1based\":");
    body.append(std::to_string(position));
    body.append(",\"window_start\":");
    body.append(std::to_string(window.reference_start));
    body.append(",\"window_end\":");
    body.append(std::to_string(window.reference_end));
    body.append(",\"variant_offset\":");
    body.append(std::to_string(window.variant_offset));
    body.append(",\"reference\":");
    append_json_string(&body, reference);
    body.append(",\"alternate\":");
    append_json_string(&body, alternate);
    body.append(",\"normalization\":");
    append_json_string(&body, variant_normalization_name(normalization));
    body.append(",\"strands\":[");
    for (std::size_t index = 0; index < scores.size(); ++index) {
      if (index != 0)
        body.push_back(',');
      const auto &entry = scores[index];
      body.append("{\"strand\":");
      append_json_string(&body, entry.strand);
      body.append(",\"reference_log_likelihood\":");
      append_number(&body, entry.reference);
      body.append(",\"alternate_log_likelihood\":");
      append_number(&body, entry.alternate);
      body.append(",\"reference_scored_tokens\":");
      body.append(std::to_string(entry.reference_tokens));
      body.append(",\"alternate_scored_tokens\":");
      body.append(std::to_string(entry.alternate_tokens));
      body.append(",\"delta\":");
      append_number(&body, entry.delta);
      body.push_back('}');
    }
    body.append("],\"aggregation\":");
    append_json_string(&body, scores.size() == 1 ? "single_strand"
                                                 : "mean_across_strands");
    body.append(",\"score\":");
    append_number(&body, aggregate);
    body.push_back('}');
    return {200, "application/json", std::move(body)};
  }

  ModelPtr model_;
  std::size_t context_size_{0};
  std::size_t maximum_sequence_{0};
  std::size_t maximum_embedding_values_{0};
  std::string backend_;
  std::string model_id_;
  std::string architecture_;
  std::string profile_;
  InferenceProfile inference_profile_{InferenceProfile::kExact};
  std::size_t vocabulary_{0};
  std::size_t embedding_width_{0};
  std::size_t layer_count_{0};
};

#if !defined(_WIN32)

volatile std::sig_atomic_t stop_requested = 0;

void request_stop(const int) noexcept { stop_requested = 1; }

struct HttpRequest final {
  std::string method;
  std::string path;
  std::string body;
  std::optional<std::size_t> timeout_ms;
};

bool send_all(const int socket_fd, const std::string_view bytes) noexcept {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
#if defined(MSG_NOSIGNAL)
    const ssize_t count = ::send(socket_fd, bytes.data() + offset,
                                 bytes.size() - offset, MSG_NOSIGNAL);
#else
    const ssize_t count =
        ::send(socket_fd, bytes.data() + offset, bytes.size() - offset, 0);
#endif
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return false;
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

void send_response(const int socket_fd, const ServerResponse &response) {
  std::string reason = "Internal Server Error";
  switch (response.http_status) {
  case 200:
    reason = "OK";
    break;
  case 400:
    reason = "Bad Request";
    break;
  case 404:
    reason = "Not Found";
    break;
  case 408:
    reason = "Request Timeout";
    break;
  case 413:
    reason = "Content Too Large";
    break;
  case 422:
    reason = "Unprocessable Content";
    break;
  case 429:
    reason = "Too Many Requests";
    break;
  case 499:
    reason = "Client Closed Request";
    break;
  case 501:
    reason = "Not Implemented";
    break;
  case 503:
    reason = "Service Unavailable";
    break;
  default:
    break;
  }
  std::string header{"HTTP/1.1 "};
  header.append(std::to_string(response.http_status));
  header.push_back(' ');
  header.append(reason);
  header.append("\r\nContent-Type: ");
  header.append(response.content_type);
  header.append("\r\nContent-Length: ");
  header.append(std::to_string(response.body.size()));
  header.append(
      "\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n");
  if (send_all(socket_fd, header))
    static_cast<void>(send_all(socket_fd, response.body));
}

std::string lowercase(std::string value) {
  for (char &byte : value) {
    if (byte >= 'A' && byte <= 'Z')
      byte = static_cast<char>(byte - 'A' + 'a');
  }
  return value;
}

bool parse_decimal(const std::string_view text,
                   std::size_t *const output) noexcept {
  if (text.empty())
    return false;
  std::size_t value = 0;
  for (const char byte : text) {
    if (byte < '0' || byte > '9')
      return false;
    const std::size_t digit = static_cast<std::size_t>(byte - '0');
    if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10)
      return false;
    value = value * 10 + digit;
  }
  *output = value;
  return true;
}

ServerResponse read_request(const int socket_fd, const std::size_t maximum_body,
                            HttpRequest *const output) {
  constexpr std::size_t maximum_header = 16U << 10U;
  std::string bytes;
  bytes.reserve(4096);
  std::size_t split = std::string::npos;
  while ((split = bytes.find("\r\n\r\n")) == std::string::npos) {
    char buffer[4096];
    const ssize_t count = ::recv(socket_fd, buffer, sizeof(buffer), 0);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return error_response(400, "invalid_http", "incomplete HTTP headers");
    bytes.append(buffer, static_cast<std::size_t>(count));
    if (bytes.size() > maximum_header)
      return error_response(413, "headers_too_large",
                            "HTTP headers exceed 16384 bytes");
  }
  const std::string_view head{bytes.data(), split};
  const auto first_end = head.find("\r\n");
  const std::string_view request_line = head.substr(0, first_end);
  const auto first_space = request_line.find(' ');
  const auto second_space = first_space == std::string_view::npos
                                ? std::string_view::npos
                                : request_line.find(' ', first_space + 1);
  if (first_space == std::string_view::npos ||
      second_space == std::string_view::npos ||
      request_line.substr(second_space + 1) != "HTTP/1.1") {
    return error_response(400, "invalid_http",
                          "request line must use HTTP/1.1");
  }
  output->method = request_line.substr(0, first_space);
  output->path =
      request_line.substr(first_space + 1, second_space - first_space - 1);
  if (output->path.find('?') != std::string::npos)
    return error_response(400, "invalid_http", "query strings are unsupported");

  std::map<std::string, std::string, std::less<>> headers;
  std::size_t line_start =
      first_end == std::string_view::npos ? head.size() : first_end + 2;
  while (line_start < head.size()) {
    const auto line_end = head.find("\r\n", line_start);
    const std::string_view line = head.substr(
        line_start,
        (line_end == std::string_view::npos ? head.size() : line_end) -
            line_start);
    const auto colon = line.find(':');
    if (colon == std::string_view::npos)
      return error_response(400, "invalid_http", "malformed HTTP header");
    std::string name = lowercase(std::string{line.substr(0, colon)});
    std::size_t value_start = colon + 1;
    while (value_start < line.size() &&
           (line[value_start] == ' ' || line[value_start] == '\t'))
      ++value_start;
    std::string value{line.substr(value_start)};
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t'))
      value.pop_back();
    if (!headers.emplace(std::move(name), std::move(value)).second)
      return error_response(400, "invalid_http", "duplicate HTTP header");
    if (line_end == std::string_view::npos)
      break;
    line_start = line_end + 2;
  }
  if (headers.find("transfer-encoding") != headers.end())
    return error_response(400, "invalid_http",
                          "Transfer-Encoding is unsupported");
  std::size_t content_length = 0;
  const auto length = headers.find("content-length");
  if (output->method == "POST") {
    if (length == headers.end() ||
        !parse_decimal(length->second, &content_length))
      return error_response(400, "invalid_http",
                            "POST requires one decimal Content-Length");
    if (content_length > maximum_body)
      return error_response(413, "request_too_large",
                            "HTTP body exceeds configured limit");
  } else if (length != headers.end() &&
             (!parse_decimal(length->second, &content_length) ||
              content_length != 0)) {
    return error_response(400, "invalid_http", "GET body is unsupported");
  }
  const auto timeout = headers.find("x-evo-timeout-ms");
  if (timeout != headers.end()) {
    std::size_t timeout_value = 0;
    if (!parse_decimal(timeout->second, &timeout_value) ||
        timeout_value > 3'600'000) {
      return error_response(400, "invalid_http",
                            "X-Evo-Timeout-Ms must be in [0, 3600000]");
    }
    output->timeout_ms = timeout_value;
  }
  const std::size_t body_start = split + 4;
  if (bytes.size() - body_start > content_length)
    return error_response(400, "invalid_http",
                          "HTTP pipelining is unsupported");
  while (bytes.size() - body_start < content_length) {
    char buffer[8192];
    const std::size_t remaining = content_length - (bytes.size() - body_start);
    const ssize_t count =
        ::recv(socket_fd, buffer, std::min(sizeof(buffer), remaining), 0);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return error_response(400, "invalid_http", "incomplete HTTP body");
    bytes.append(buffer, static_cast<std::size_t>(count));
  }
  output->body.assign(bytes.data() + body_start, content_length);
  return {200, "application/json", {}};
}

bool client_disconnected(const int socket_fd) noexcept {
  pollfd descriptor{};
  descriptor.fd = socket_fd;
  descriptor.events = POLLIN;
  const int ready = ::poll(&descriptor, 1, 0);
  if (ready <= 0)
    return false;
  if ((descriptor.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0)
    return true;
  if ((descriptor.revents & POLLIN) != 0) {
    char byte = 0;
    const ssize_t count = ::recv(socket_fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    return count == 0;
  }
  return false;
}

std::string health_body(const Runtime &runtime, const CliOptions &options) {
  std::string body{"{\"status\":\"ok\",\"model\":"};
  append_json_string(&body, runtime.model_id());
  body.append(",\"architecture\":");
  append_json_string(&body, runtime.architecture());
  body.append(",\"backend\":");
  append_json_string(&body, runtime.backend());
  body.append(",\"profile\":");
  append_json_string(&body, runtime.profile());
  body.append(",\"execution_profile\":");
  append_json_string(&body,
                     inference_profile_name(runtime.inference_profile()));
  body.append(",\"context_size\":");
  body.append(std::to_string(runtime.context_size()));
  body.append(",\"max_sequence_bytes\":");
  body.append(std::to_string(runtime.maximum_sequence()));
  body.append(",\"max_request_bytes\":");
  body.append(std::to_string(options.server_max_request_bytes));
  body.append(",\"max_embedding_values\":");
  body.append(std::to_string(runtime.maximum_embedding_values()));
  body.append(",\"batching\":\"isolated_context_microbatch\"}");
  return body;
}

struct HttpMetrics final {
  std::atomic<std::uint64_t> connections{0};
  std::atomic<std::uint64_t> responses{0};
  std::atomic<std::uint64_t> client_errors{0};
  std::atomic<std::uint64_t> server_errors{0};
  std::atomic<std::size_t> active_connections{0};
};

std::string metrics_body(const DynamicScheduler &scheduler,
                         const HttpMetrics &http) {
  const auto metrics = scheduler.metrics();
  std::string body;
  const auto line = [&](const std::string_view name, const auto value) {
    body.append(name);
    body.push_back(' ');
    body.append(std::to_string(value));
    body.push_back('\n');
  };
  line("evo_http_connections_total", http.connections.load());
  line("evo_http_responses_total", http.responses.load());
  line("evo_http_client_errors_total", http.client_errors.load());
  line("evo_http_server_errors_total", http.server_errors.load());
  line("evo_http_active_connections", http.active_connections.load());
  line("evo_scheduler_submitted_total", metrics.submitted);
  line("evo_scheduler_rejected_total", metrics.rejected);
  line("evo_scheduler_batches_total", metrics.batches);
  line("evo_scheduler_batch_items_total", metrics.batch_items);
  line("evo_scheduler_completed_total", metrics.completed);
  line("evo_scheduler_failed_total", metrics.failed);
  line("evo_scheduler_cancelled_total", metrics.cancelled);
  line("evo_scheduler_queued", metrics.queued);
  line("evo_scheduler_queue_peak", metrics.queue_peak);
  line("evo_scheduler_active", metrics.active);
  line("evo_scheduler_active_peak", metrics.active_peak);
  return body;
}

void record_response(const ServerResponse &response,
                     HttpMetrics *const metrics) {
  ++metrics->responses;
  if (response.http_status >= 400 && response.http_status < 500)
    ++metrics->client_errors;
  else if (response.http_status >= 500)
    ++metrics->server_errors;
}

void handle_connection(const int socket_fd, const Runtime &runtime,
                       const CliOptions &options, DynamicScheduler &scheduler,
                       HttpMetrics *const metrics) {
  struct Close final {
    int descriptor{-1};
    ~Close() {
      if (descriptor >= 0)
        ::close(descriptor);
    }
  } close{socket_fd};
  struct Active final {
    HttpMetrics *metrics{nullptr};
    ~Active() { --metrics->active_connections; }
  } active{metrics};

  timeval timeout{};
  timeout.tv_sec = 30;
  static_cast<void>(::setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                                 sizeof(timeout)));
  HttpRequest request;
  auto response =
      read_request(socket_fd, options.server_max_request_bytes, &request);
  if (response.http_status != 200) {
    record_response(response, metrics);
    send_response(socket_fd, response);
    return;
  }
  if (request.method == "GET") {
    if (request.path == "/health") {
      response = {200, "application/json", health_body(runtime, options)};
    } else if (request.path == "/metrics") {
      response = {200, "text/plain; version=0.0.4",
                  metrics_body(scheduler, *metrics)};
    } else {
      response = error_response(404, "not_found", "route does not exist");
    }
    record_response(response, metrics);
    send_response(socket_fd, response);
    return;
  }
  if (request.method != "POST") {
    response = error_response(400, "invalid_http", "method is unsupported");
    record_response(response, metrics);
    send_response(socket_fd, response);
    return;
  }
  if (request.path != "/v1/generate" && request.path != "/v1/score" &&
      request.path != "/v1/embeddings" && request.path != "/v1/variants") {
    response = error_response(404, "not_found", "route does not exist");
    record_response(response, metrics);
    send_response(socket_fd, response);
    return;
  }
  auto scheduled =
      scheduler.submit([&runtime, path = request.path,
                        body = request.body](const CancellationToken &cancel) {
        return runtime.dispatch(path, body, cancel);
      });
  if (!scheduled) {
    response = error_response(503, "queue_full", "server queue is full");
    record_response(response, metrics);
    send_response(socket_fd, response);
    return;
  }
  if (request.timeout_ms == 0) {
    scheduled->cancel();
    static_cast<void>(scheduled->future().wait());
    response =
        error_response(408, "deadline_exceeded", "request deadline expired");
    record_response(response, metrics);
    send_response(socket_fd, response);
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  while (scheduled->future().wait_for(std::chrono::milliseconds{10}) !=
         std::future_status::ready) {
    if (client_disconnected(socket_fd)) {
      scheduled->cancel();
      return;
    }
    if (request.timeout_ms.has_value() &&
        std::chrono::steady_clock::now() - start >=
            std::chrono::milliseconds{*request.timeout_ms}) {
      scheduled->cancel();
      static_cast<void>(scheduled->future().wait());
      response =
          error_response(408, "deadline_exceeded", "request deadline expired");
      record_response(response, metrics);
      send_response(socket_fd, response);
      return;
    }
  }
  response = scheduled->future().get();
  record_response(response, metrics);
  send_response(socket_fd, response);
}

#endif

} // namespace

Status run_server(const CliOptions &options, const bool allow_test_fixture) {
#if defined(_WIN32)
  (void)options;
  (void)allow_test_fixture;
  return {ErrorCode::kUnsupported,
          "the native HTTP server is not yet available on Windows"};
#else
  std::vector<std::int32_t> devices;
  devices.reserve(options.gpu_ids.size());
  for (const int device : options.gpu_ids)
    devices.push_back(static_cast<std::int32_t>(device));
  evo_model_params model_params = evo_model_default_params();
  switch (options.backend) {
  case ExecutionBackend::kCpu:
    model_params.backend = EVO_BACKEND_CPU;
    break;
  case ExecutionBackend::kMps:
    model_params.backend = EVO_BACKEND_MPS;
    break;
  case ExecutionBackend::kAuto:
  case ExecutionBackend::kCuda:
    model_params.backend = EVO_BACKEND_CUDA;
    break;
  }
  if (model_params.backend == EVO_BACKEND_CUDA) {
    model_params.devices = devices.data();
    model_params.device_count = devices.size();
  }
  if (allow_test_fixture)
    model_params.flags |= EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC;
  evo_model *raw_model = nullptr;
  const evo_status load_status =
      evo_model_load(options.model_path.c_str(), &model_params, &raw_model);
  ModelPtr model{raw_model, &evo_model_free};
  if (load_status != EVO_STATUS_OK)
    return status_from_c(load_status, "server model load: ");
  if (evo_model_vocab_size(model.get()) == 0 ||
      !token_vocabulary_size_supported(evo_model_vocab_size(model.get()))) {
    return {ErrorCode::kModelFormat,
            "server model vocabulary does not fit the public token type"};
  }
  const std::size_t model_context = evo_model_max_context(model.get());
  const bool fixed_shape_sequence_cnn =
      std::string_view{evo_model_architecture(model.get())} ==
      "GenebSequenceCnnEncoder";
  const std::size_t context_size =
      !options.context_size_explicit && model_context != 0
          ? (fixed_shape_sequence_cnn
                 ? model_context
                 : std::min(options.context_size, model_context))
          : options.context_size;
  if (model_context != 0 && context_size > model_context)
    return {ErrorCode::kInvalidArgument,
            "server context exceeds the model maximum"};
  Runtime runtime{
      std::move(model), context_size, options.server_max_sequence_bytes,
      options.server_max_embedding_values, options.inference_profile};
  DynamicScheduler scheduler{
      options.server_max_queue, options.server_max_batch,
      std::chrono::milliseconds{options.server_batch_window_ms}};

  const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0)
    return {ErrorCode::kIo, "failed to create server socket"};
  struct Listener final {
    int descriptor{-1};
    ~Listener() {
      if (descriptor >= 0)
        ::close(descriptor);
    }
  } listener_guard{listener};
  int reuse = 1;
  static_cast<void>(
      ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(options.server_port);
  if (::inet_pton(AF_INET, options.server_host.c_str(), &address.sin_addr) != 1)
    return {ErrorCode::kInvalidArgument,
            "--host must be a numeric IPv4 address"};
  if (::bind(listener, reinterpret_cast<const sockaddr *>(&address),
             sizeof(address)) != 0) {
    return {ErrorCode::kIo, "failed to bind " + options.server_host + ":" +
                                std::to_string(options.server_port) + ": " +
                                std::strerror(errno)};
  }
  if (::listen(listener, static_cast<int>(options.server_max_queue)) != 0)
    return {ErrorCode::kIo, "failed to listen on server socket"};
  sockaddr_in bound{};
  socklen_t bound_size = sizeof(bound);
  if (::getsockname(listener, reinterpret_cast<sockaddr *>(&bound),
                    &bound_size) != 0)
    return {ErrorCode::kIo, "failed to inspect bound server socket"};
  std::cerr << "evo_server listening " << options.server_host << ':'
            << ntohs(bound.sin_port) << '\n';
  std::cerr.flush();

  stop_requested = 0;
  const auto previous_interrupt = std::signal(SIGINT, request_stop);
  const auto previous_terminate = std::signal(SIGTERM, request_stop);
  HttpMetrics metrics;
  struct ConnectionThread final {
    std::thread worker;
    std::shared_ptr<std::atomic<bool>> complete;
  };
  std::vector<ConnectionThread> connections;
  while (stop_requested == 0) {
    for (auto iterator = connections.begin(); iterator != connections.end();) {
      if (!iterator->complete->load(std::memory_order_acquire)) {
        ++iterator;
        continue;
      }
      if (iterator->worker.joinable())
        iterator->worker.join();
      iterator = connections.erase(iterator);
    }
    pollfd descriptor{};
    descriptor.fd = listener;
    descriptor.events = POLLIN;
    const int ready = ::poll(&descriptor, 1, 250);
    if (ready < 0 && errno == EINTR)
      continue;
    if (ready < 0) {
      std::signal(SIGINT, previous_interrupt);
      std::signal(SIGTERM, previous_terminate);
      return {ErrorCode::kIo, "server accept poll failed"};
    }
    if (ready == 0)
      continue;
    sockaddr_in peer{};
    socklen_t peer_size = sizeof(peer);
    const int connection =
        ::accept(listener, reinterpret_cast<sockaddr *>(&peer), &peer_size);
    if (connection < 0 && errno == EINTR)
      continue;
    if (connection < 0)
      continue;
    ++metrics.connections;
    const std::size_t active = ++metrics.active_connections;
    if (active > options.server_max_queue + options.server_max_batch) {
      --metrics.active_connections;
      const auto response =
          error_response(503, "connection_limit", "too many connections");
      record_response(response, &metrics);
      send_response(connection, response);
      ::close(connection);
      continue;
    }
    auto complete = std::make_shared<std::atomic<bool>>(false);
    connections.push_back(
        ConnectionThread{std::thread{[connection, &runtime, &options,
                                      &scheduler, &metrics, complete] {
                           try {
                             handle_connection(connection, runtime, options,
                                               scheduler, &metrics);
                           } catch (...) {
                             // A malformed or resource-exhausted connection
                             // must not terminate the long-running server.
                             // DynamicScheduler separately converts inference
                             // task exceptions into structured responses.
                           }
                           complete->store(true, std::memory_order_release);
                         }},
                         std::move(complete)});
  }
  for (auto &connection : connections) {
    if (connection.worker.joinable())
      connection.worker.join();
  }
  std::signal(SIGINT, previous_interrupt);
  std::signal(SIGTERM, previous_terminate);
  return Status::Ok();
#endif
}

} // namespace evo
