// SPDX-License-Identifier: Apache-2.0
#include "evo/evo.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "evo/model_format.hpp"
#include "evo/sampler.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"
#include "evo/version.hpp"

#if defined(EVO_HAS_CUDA)
#include "evo/cuda/model.hpp"
#endif

namespace {

thread_local std::string last_error;

static_assert(static_cast<int>(evo::ErrorCode::kOk) == EVO_STATUS_OK);
static_assert(static_cast<int>(evo::ErrorCode::kInvalidArgument) ==
              EVO_STATUS_INVALID_ARGUMENT);
static_assert(static_cast<int>(evo::ErrorCode::kIo) == EVO_STATUS_IO);
static_assert(static_cast<int>(evo::ErrorCode::kModelFormat) ==
              EVO_STATUS_MODEL_FORMAT);
static_assert(static_cast<int>(evo::ErrorCode::kUnsupported) ==
              EVO_STATUS_UNSUPPORTED);
static_assert(static_cast<int>(evo::ErrorCode::kCuda) == EVO_STATUS_CUDA);
static_assert(static_cast<int>(evo::ErrorCode::kInternal) ==
              EVO_STATUS_INTERNAL);

evo_status publish(const evo::Status &status) {
  if (status.ok()) {
    last_error.clear();
    return EVO_STATUS_OK;
  }
  last_error = std::string{evo::error_code_name(status.code())} + ": " +
               status.message();
  return static_cast<evo_status>(status.code());
}

evo_status publish(const evo_status status, std::string message) {
  if (status == EVO_STATUS_OK) {
    last_error.clear();
    return status;
  }
  last_error = std::move(message);
  return status;
}

template <typename Function> evo_status protect(Function &&function) noexcept {
  try {
    return function();
  } catch (const std::bad_alloc &) {
    return publish(EVO_STATUS_INTERNAL, "internal: allocation failed");
  } catch (const std::exception &error) {
    return publish(EVO_STATUS_INTERNAL, "internal: unexpected exception: " +
                                            std::string{error.what()});
  } catch (...) {
    return publish(EVO_STATUS_INTERNAL, "internal: unknown exception");
  }
}

std::string metadata_string(const evo::ModelFile &model,
                            const std::string_view key,
                            const std::string_view fallback = {}) {
  const auto *const entry = model.find_metadata(key);
  if (entry == nullptr || entry->type != evo::MetadataType::kString)
    return std::string{fallback};
  return {entry->value.begin(), entry->value.end()};
}

std::size_t metadata_size(const evo::ModelFile &model,
                          const std::string_view key,
                          const std::size_t fallback = 0) {
  const auto *const entry = model.find_metadata(key);
  if (entry == nullptr || entry->type != evo::MetadataType::kU64 ||
      entry->value.size() != sizeof(std::uint64_t)) {
    return fallback;
  }
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    value |= static_cast<std::uint64_t>(entry->value[byte]) << (byte * 8U);
  }
  return value <= std::numeric_limits<std::size_t>::max()
             ? static_cast<std::size_t>(value)
             : fallback;
}

bool metadata_bool(const evo::ModelFile &model, const std::string_view key) {
  const auto *const entry = model.find_metadata(key);
  return entry != nullptr && entry->type == evo::MetadataType::kBool &&
         entry->value.size() == 1 && entry->value[0] != 0;
}

bool valid_status(const evo_status status) noexcept {
  return status == EVO_STATUS_OK || status == EVO_STATUS_INVALID_ARGUMENT ||
         status == EVO_STATUS_IO || status == EVO_STATUS_MODEL_FORMAT ||
         status == EVO_STATUS_UNSUPPORTED || status == EVO_STATUS_CUDA ||
         status == EVO_STATUS_INTERNAL;
}

struct ModelState final {
  evo::ModelFile file;
  evo_backend backend{EVO_BACKEND_AUTO};
  std::vector<int> devices;
  std::string model_id;
  std::string architecture;
  std::size_t vocab_size{0};
  std::size_t max_context{0};
  std::size_t embedding_width{0};
  std::size_t layer_count{0};
  bool allow_synthetic{false};
#if defined(EVO_HAS_CUDA)
  std::unique_ptr<evo::cuda::PipelineModel> cuda_weights;
#endif
};

} // namespace

struct evo_model final {
  std::shared_ptr<ModelState> state;
};

struct evo_context final {
  std::shared_ptr<ModelState> state;
  std::size_t capacity{0};
  bool failed{false};
#if defined(EVO_HAS_CUDA)
  std::unique_ptr<evo::cuda::PipelineModel> cuda;
#endif
};

struct evo_batch final {
  std::size_t sequence_capacity{0};
  std::size_t token_count{0};
  std::vector<std::vector<evo::TokenId>> sequences;
};

struct evo_sampler final {
  explicit evo_sampler(const evo::SamplingConfig &config) : value(config) {}
  evo::Sampler value;
};

extern "C" {

uint32_t evo_abi_version(void) { return EVO_ABI_VERSION_CURRENT; }

const char *evo_version_string(void) { return evo::version().data(); }

const char *evo_status_name(const evo_status status) {
  if (!valid_status(status))
    return "unknown";
  return evo::error_code_name(static_cast<evo::ErrorCode>(status));
}

const char *evo_backend_name(const evo_backend backend) {
  switch (backend) {
  case EVO_BACKEND_AUTO:
    return "auto";
  case EVO_BACKEND_CUDA:
    return "cuda";
  case EVO_BACKEND_CPU:
    return "cpu";
  }
  return "unknown";
}

const char *evo_last_error(void) { return last_error.c_str(); }

evo_model_params evo_model_default_params(void) {
  return {sizeof(evo_model_params), EVO_BACKEND_AUTO, nullptr, 0, 0};
}

evo_context_params evo_context_default_params(void) {
  return {sizeof(evo_context_params), 4096, 0};
}

evo_sampler_params evo_sampler_default_params(void) {
  return {sizeof(evo_sampler_params), 1.0F, 1, 1.0F, 0};
}

evo_status evo_model_load(const char *const path,
                          const evo_model_params *const params,
                          evo_model **const model_out) {
  if (model_out != nullptr)
    *model_out = nullptr;
  return protect([&]() -> evo_status {
    if (path == nullptr || path[0] == '\0' || model_out == nullptr) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: model path/output is null or empty");
    }
    if (params != nullptr && params->struct_size < sizeof(evo_model_params)) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: evo_model_params is too small");
    }
    const evo_model_params options =
        params == nullptr ? evo_model_default_params() : *params;
    if (options.backend != EVO_BACKEND_AUTO &&
        options.backend != EVO_BACKEND_CUDA &&
        options.backend != EVO_BACKEND_CPU) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: unknown model backend");
    }
    constexpr std::uint32_t known_flags =
        EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC;
    if ((options.flags & ~known_flags) != 0 || options.device_count > 4 ||
        (options.device_count != 0 && options.devices == nullptr)) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: invalid model flags/device list");
    }
    auto state = std::make_shared<ModelState>();
    state->devices.reserve(options.device_count == 0 ? 1
                                                     : options.device_count);
    if (options.device_count == 0) {
      state->devices.push_back(0);
    } else {
      std::set<int> unique_devices;
      for (std::size_t index = 0; index < options.device_count; ++index) {
        if (options.devices[index] < 0 ||
            !unique_devices.insert(options.devices[index]).second) {
          return publish(EVO_STATUS_INVALID_ARGUMENT,
                         "invalid_argument: CUDA devices must be unique and "
                         "nonnegative");
        }
        state->devices.push_back(options.devices[index]);
      }
    }
    state->backend = options.backend;
    if (state->backend == EVO_BACKEND_AUTO) {
#if defined(EVO_HAS_CUDA)
      state->backend = EVO_BACKEND_CUDA;
#else
      state->backend = EVO_BACKEND_CPU;
#endif
    }
#if !defined(EVO_HAS_CUDA)
    if (state->backend == EVO_BACKEND_CUDA) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: this evo library was built without CUDA");
    }
#endif
    auto status = state->file.open(path);
    if (!status.ok())
      return publish(status);
    state->model_id = metadata_string(state->file, "model.id");
    if (state->model_id.empty())
      state->model_id = metadata_string(state->file, "model.name", "unknown");
    state->architecture =
        metadata_string(state->file, "model.architecture", "unknown");
    state->vocab_size = metadata_size(state->file, "config.vocab_size");
    state->max_context = metadata_size(state->file, "config.max_seqlen");
    state->embedding_width = metadata_size(state->file, "config.hidden_size");
    state->layer_count = metadata_size(state->file, "config.num_layers");
    if (state->max_context == 0 &&
        metadata_bool(state->file, "fixture.synthetic")) {
      state->max_context = std::numeric_limits<std::size_t>::max();
    }
    state->allow_synthetic =
        (options.flags & EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC) != 0;
#if defined(EVO_HAS_CUDA)
    if (state->backend == EVO_BACKEND_CUDA) {
      state->cuda_weights = std::make_unique<evo::cuda::PipelineModel>();
      status = state->cuda_weights->load(state->file, state->devices, 1,
                                         state->allow_synthetic);
      if (!status.ok())
        return publish(status);
    }
#endif
    auto handle = std::make_unique<evo_model>();
    handle->state = std::move(state);
    *model_out = handle.release();
    return publish(evo::Status::Ok());
  });
}

void evo_model_free(evo_model *const model) { delete model; }

evo_backend evo_model_backend(const evo_model *const model) {
  return model == nullptr || !model->state ? EVO_BACKEND_AUTO
                                           : model->state->backend;
}

const char *evo_model_id(const evo_model *const model) {
  return model == nullptr || !model->state ? ""
                                           : model->state->model_id.c_str();
}

const char *evo_model_architecture(const evo_model *const model) {
  return model == nullptr || !model->state ? ""
                                           : model->state->architecture.c_str();
}

const char *evo_model_profile(const evo_model *const model) {
  return model == nullptr || !model->state ? "" : "evo2-runtime-v1";
}

size_t evo_model_vocab_size(const evo_model *const model) {
  return model == nullptr || !model->state ? 0 : model->state->vocab_size;
}

size_t evo_model_max_context(const evo_model *const model) {
  return model == nullptr || !model->state ? 0 : model->state->max_context;
}

size_t evo_model_embedding_width(const evo_model *const model) {
  return model == nullptr || !model->state ? 0 : model->state->embedding_width;
}

size_t evo_model_layer_count(const evo_model *const model) {
  return model == nullptr || !model->state ? 0 : model->state->layer_count;
}

evo_status evo_context_create(const evo_model *const model,
                              const evo_context_params *const params,
                              evo_context **const context_out) {
  if (context_out != nullptr)
    *context_out = nullptr;
  return protect([&]() -> evo_status {
    if (model == nullptr || !model->state || context_out == nullptr) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: model/context output is null");
    }
    if (params != nullptr && params->struct_size < sizeof(evo_context_params)) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: evo_context_params is too small");
    }
    const evo_context_params options =
        params == nullptr ? evo_context_default_params() : *params;
    if (options.context_size == 0 || options.flags != 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: invalid context parameters");
    }
    if (model->state->max_context != 0 &&
        options.context_size > model->state->max_context) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: requested context exceeds model "
                     "maximum");
    }
    if (model->state->backend == EVO_BACKEND_CPU) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CPU inference backend is not implemented");
    }
#if defined(EVO_HAS_CUDA)
    if (model->state->backend != EVO_BACKEND_CUDA) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: selected inference backend is unavailable");
    }
    if (!model->state->cuda_weights) {
      return publish(EVO_STATUS_INTERNAL,
                     "internal: CUDA model weights are unavailable");
    }
    auto cuda = std::make_unique<evo::cuda::PipelineModel>();
    const auto status = cuda->initialize_shared(*model->state->cuda_weights,
                                                options.context_size);
    if (!status.ok())
      return publish(status);
    auto handle = std::make_unique<evo_context>();
    handle->state = model->state;
    handle->capacity = options.context_size;
    handle->cuda = std::move(cuda);
    *context_out = handle.release();
    return publish(evo::Status::Ok());
#else
    return publish(EVO_STATUS_UNSUPPORTED,
                   "unsupported: this evo library was built without CUDA");
#endif
  });
}

void evo_context_free(evo_context *const context) { delete context; }

size_t evo_context_position(const evo_context *const context) {
#if defined(EVO_HAS_CUDA)
  return context == nullptr || !context->cuda ? 0 : context->cuda->position();
#else
  (void)context;
  return 0;
#endif
}

size_t evo_context_capacity(const evo_context *const context) {
  return context == nullptr ? 0 : context->capacity;
}

evo_status evo_context_prefill(evo_context *const context,
                               const evo_batch *const batch,
                               const evo_logits_callback callback,
                               void *const user_data) {
  return protect([&]() -> evo_status {
    if (context == nullptr || batch == nullptr || callback == nullptr) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: prefill argument is null");
    }
    if (context->failed) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: context is invalid after a prior "
                     "failure");
    }
    if (batch->sequences.size() != 1) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: exact backend currently requires batch "
                     "size one");
    }
    const auto &tokens = batch->sequences.front();
    if (tokens.empty() || tokens.size() > context->capacity ||
        evo_context_position(context) != 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: prefill requires a fresh context and "
                     "a nonempty sequence within capacity");
    }
    // From here onward, any status/exception can follow a partial state
    // advance. Keep the context poisoned unless every chunk and callback
    // completes.
    context->failed = true;
#if defined(EVO_HAS_CUDA)
    const std::size_t chunk_capacity = context->cuda->activation_capacity();
    const std::size_t columns = context->cuda->config().vocab_size;
    std::size_t offset = 0;
    bool first = true;
    while (offset < tokens.size()) {
      const std::size_t rows = std::min(chunk_capacity, tokens.size() - offset);
      const std::vector<evo::TokenId> chunk(
          tokens.begin() + static_cast<std::ptrdiff_t>(offset),
          tokens.begin() + static_cast<std::ptrdiff_t>(offset + rows));
      std::vector<float> logits;
      const auto status = first ? context->cuda->prefill(chunk, &logits)
                                : context->cuda->prefill_chunk(chunk, &logits);
      if (!status.ok())
        return publish(status);
      if (logits.size() != rows * columns) {
        return publish(EVO_STATUS_INTERNAL,
                       "internal: backend returned incomplete prefill logits");
      }
      const evo_status callback_status =
          callback(logits.data(), rows, columns, offset, user_data);
      if (callback_status != EVO_STATUS_OK) {
        return publish(valid_status(callback_status) ? callback_status
                                                     : EVO_STATUS_INTERNAL,
                       "callback: logits consumer aborted prefill");
      }
      offset += rows;
      first = false;
    }
    context->failed = false;
    return publish(evo::Status::Ok());
#else
    (void)user_data;
    return publish(EVO_STATUS_UNSUPPORTED,
                   "unsupported: this evo library was built without CUDA");
#endif
  });
}

evo_status evo_context_decode(evo_context *const context, const uint32_t token,
                              const evo_logits_callback callback,
                              void *const user_data) {
  return protect([&]() -> evo_status {
    if (context == nullptr || callback == nullptr) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: decode argument is null");
    }
    if (context->failed) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: context is invalid after a prior "
                     "failure");
    }
    const std::size_t offset = evo_context_position(context);
    if (offset == 0 || offset >= context->capacity || !context->state ||
        (context->state->vocab_size != 0 &&
         token >= context->state->vocab_size) ||
        token > std::numeric_limits<evo::TokenId>::max()) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: decode token/context is invalid");
    }
    context->failed = true;
#if defined(EVO_HAS_CUDA)
    std::vector<float> logits;
    const auto status =
        context->cuda->decode(static_cast<evo::TokenId>(token), &logits);
    if (!status.ok())
      return publish(status);
    const std::size_t columns = context->cuda->config().vocab_size;
    if (logits.size() != columns) {
      return publish(EVO_STATUS_INTERNAL,
                     "internal: backend returned incomplete decode logits");
    }
    const evo_status callback_status =
        callback(logits.data(), 1, columns, offset, user_data);
    if (callback_status != EVO_STATUS_OK) {
      return publish(valid_status(callback_status) ? callback_status
                                                   : EVO_STATUS_INTERNAL,
                     "callback: logits consumer aborted decode");
    }
    context->failed = false;
    return publish(evo::Status::Ok());
#else
    (void)user_data;
    return publish(EVO_STATUS_UNSUPPORTED,
                   "unsupported: this evo library was built without CUDA");
#endif
  });
}

evo_status evo_context_embed(evo_context *const context,
                             const evo_batch *const batch, const size_t layer,
                             const evo_embedding_callback callback,
                             void *const user_data) {
  return protect([&]() -> evo_status {
    if (context == nullptr || batch == nullptr || callback == nullptr) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: embedding argument is null");
    }
    if (context->failed) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: context is invalid after a prior "
                     "failure");
    }
    if (!context->state || layer >= context->state->layer_count) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: embedding layer is outside the model");
    }
    if (batch->sequences.size() != 1) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: exact backend currently requires batch "
                     "size one");
    }
    const auto &tokens = batch->sequences.front();
    if (tokens.empty() || tokens.size() > context->capacity ||
        evo_context_position(context) != 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: embedding requires a fresh context and "
                     "a nonempty sequence within capacity");
    }
    context->failed = true;
#if defined(EVO_HAS_CUDA)
    const std::size_t chunk_capacity = context->cuda->activation_capacity();
    const std::size_t columns = context->cuda->config().width;
    std::size_t offset = 0;
    bool first = true;
    while (offset < tokens.size()) {
      const std::size_t rows = std::min(chunk_capacity, tokens.size() - offset);
      const std::vector<evo::TokenId> chunk(
          tokens.begin() + static_cast<std::ptrdiff_t>(offset),
          tokens.begin() + static_cast<std::ptrdiff_t>(offset + rows));
      std::vector<float> embedding;
      const auto status =
          first ? context->cuda->prefill_embedding(chunk, layer, &embedding)
                : context->cuda->prefill_chunk_embedding(chunk, layer,
                                                         &embedding);
      if (!status.ok())
        return publish(status);
      if (embedding.size() != rows * columns) {
        return publish(EVO_STATUS_INTERNAL,
                       "internal: backend returned incomplete embeddings");
      }
      const evo_status callback_status =
          callback(embedding.data(), rows, columns, offset, user_data);
      if (callback_status != EVO_STATUS_OK) {
        return publish(valid_status(callback_status) ? callback_status
                                                     : EVO_STATUS_INTERNAL,
                       "callback: embedding consumer aborted inference");
      }
      offset += rows;
      first = false;
    }
    context->failed = false;
    return publish(evo::Status::Ok());
#else
    (void)user_data;
    return publish(EVO_STATUS_UNSUPPORTED,
                   "unsupported: this evo library was built without CUDA");
#endif
  });
}

evo_status evo_batch_create(const size_t sequence_capacity,
                            evo_batch **const batch_out) {
  if (batch_out != nullptr)
    *batch_out = nullptr;
  return protect([&]() -> evo_status {
    if (batch_out == nullptr || sequence_capacity == 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: batch capacity/output is invalid");
    }
    auto handle = std::make_unique<evo_batch>();
    handle->sequence_capacity = sequence_capacity;
    handle->sequences.reserve(sequence_capacity);
    *batch_out = handle.release();
    return publish(evo::Status::Ok());
  });
}

void evo_batch_free(evo_batch *const batch) { delete batch; }

void evo_batch_clear(evo_batch *const batch) {
  if (batch == nullptr)
    return;
  batch->sequences.clear();
  batch->token_count = 0;
}

evo_status evo_batch_add_sequence(evo_batch *const batch,
                                  const uint8_t *const sequence,
                                  const size_t sequence_length) {
  return protect([&]() -> evo_status {
    if (batch == nullptr || sequence == nullptr || sequence_length == 0 ||
        batch->sequences.size() >= batch->sequence_capacity ||
        sequence_length >
            std::numeric_limits<std::size_t>::max() - batch->token_count) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: sequence does not fit batch");
    }
    std::vector<evo::TokenId> tokens;
    tokens.reserve(sequence_length);
    for (std::size_t index = 0; index < sequence_length; ++index)
      tokens.push_back(static_cast<evo::TokenId>(sequence[index]));
    batch->sequences.push_back(std::move(tokens));
    batch->token_count += sequence_length;
    return publish(evo::Status::Ok());
  });
}

size_t evo_batch_sequence_count(const evo_batch *const batch) {
  return batch == nullptr ? 0 : batch->sequences.size();
}

size_t evo_batch_token_count(const evo_batch *const batch) {
  return batch == nullptr ? 0 : batch->token_count;
}

evo_status evo_sampler_create(const evo_sampler_params *const params,
                              evo_sampler **const sampler_out) {
  if (sampler_out != nullptr)
    *sampler_out = nullptr;
  return protect([&]() -> evo_status {
    if (sampler_out == nullptr) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: sampler output is null");
    }
    if (params != nullptr && params->struct_size < sizeof(evo_sampler_params)) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: evo_sampler_params is too small");
    }
    const evo_sampler_params options =
        params == nullptr ? evo_sampler_default_params() : *params;
    const evo::SamplingConfig config{options.temperature, options.top_k,
                                     options.top_p, options.seed};
    const auto status = evo::validate_sampling_config(config);
    if (!status.ok())
      return publish(status);
    *sampler_out = std::make_unique<evo_sampler>(config).release();
    return publish(evo::Status::Ok());
  });
}

void evo_sampler_free(evo_sampler *const sampler) { delete sampler; }

evo_status evo_sampler_sample(evo_sampler *const sampler,
                              const float *const logits,
                              const size_t logits_count,
                              uint32_t *const token_out) {
  return protect([&]() -> evo_status {
    if (sampler == nullptr || logits == nullptr || token_out == nullptr) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: sampler argument is null");
    }
    const std::vector<float> values(logits, logits + logits_count);
    evo::TokenId token = 0;
    const auto status = sampler->value.sample(values, &token);
    if (!status.ok())
      return publish(status);
    *token_out = token;
    return publish(evo::Status::Ok());
  });
}

} // extern "C"
