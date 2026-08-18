// SPDX-License-Identifier: Apache-2.0
#include "evo/evo.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "evo/cpu/model.hpp"
#include "evo/geneb_embedding.hpp"
#include "evo/model_format.hpp"
#include "evo/model_registry.hpp"
#include "evo/profile.hpp"
#include "evo/sampler.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"
#include "evo/version.hpp"

#if defined(EVO_HAS_CUDA)
#include "evo/cuda/esmc.hpp"
#include "evo/cuda/model.hpp"
#endif

#if defined(EVO_HAS_MPS)
#include "mps/model.hpp"
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
static_assert(static_cast<int>(evo::ErrorCode::kMps) == EVO_STATUS_MPS);
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
         status == EVO_STATUS_MPS || status == EVO_STATUS_INTERNAL;
}

const evo::ArchitectureBackendFactorySpec *
architecture_backend_factory(const evo::ArchitectureSpec &architecture,
                             const evo_backend backend) noexcept {
  switch (backend) {
  case EVO_BACKEND_CPU:
    return evo::find_architecture_backend_factory(architecture,
                                                  evo::kArchitectureBackendCpu);
  case EVO_BACKEND_CUDA:
    return evo::find_architecture_backend_factory(
        architecture, evo::kArchitectureBackendCuda);
  case EVO_BACKEND_MPS:
    return evo::find_architecture_backend_factory(architecture,
                                                  evo::kArchitectureBackendMps);
  case EVO_BACKEND_AUTO:
    return nullptr;
  }
  return nullptr;
}

struct ModelState final {
  evo::ModelFile file;
  evo_backend backend{EVO_BACKEND_AUTO};
  std::vector<int> devices;
  std::string model_id;
  std::string architecture;
  std::string artifact_profile;
  const evo::ArchitectureSpec *architecture_spec{nullptr};
  std::size_t vocab_size{0};
  std::size_t max_context{0};
  std::size_t embedding_width{0};
  std::size_t layer_count{0};
  bool allow_synthetic{false};
  std::shared_ptr<evo::ArtifactTokenizer> artifact_tokenizer;
  std::optional<evo::GenebEmbeddingArtifactSpec> geneb_embedding;
  std::mutex cpu_mutex;
  std::unique_ptr<evo::cpu::Model> cpu_weights;
#if defined(EVO_HAS_CUDA)
  std::unique_ptr<evo::cuda::PipelineModel> cuda_weights;
  std::unique_ptr<evo::cuda::EsmcModel> cuda_esmc_weights;
#endif
};

evo::Status encode_model_sequence(const ModelState &model,
                                  const std::string_view sequence,
                                  std::vector<evo::TokenId> *const tokens) {
  if (model.artifact_tokenizer)
    return model.artifact_tokenizer->encode(sequence, {}, tokens);
  return evo::encode_sequence(model.architecture_spec->tokenizer, sequence,
                              tokens);
}

evo::Status decode_model_token(const ModelState &model,
                               const evo::TokenId token,
                               std::uint8_t *const byte) {
  if (model.artifact_tokenizer) {
    return {evo::ErrorCode::kUnsupported,
            "artifact tokenizer detokenization is not supported"};
  }
  return evo::decode_sequence_token(model.architecture_spec->tokenizer, token,
                                    byte);
}

struct LegacyEmbeddingCapture final {
  std::vector<float> values;
  std::size_t rows{0};
  std::size_t columns{0};
  std::size_t next_offset{0};
};

evo_status capture_legacy_embedding(const float *const values,
                                    const std::size_t rows,
                                    const std::size_t columns,
                                    const std::size_t token_offset,
                                    void *const user_data) {
  auto *const capture = static_cast<LegacyEmbeddingCapture *>(user_data);
  if (capture == nullptr || values == nullptr || rows == 0 || columns == 0 ||
      token_offset != capture->next_offset ||
      (capture->columns != 0 && capture->columns != columns) ||
      rows > std::numeric_limits<std::size_t>::max() / columns ||
      rows * columns >
          std::numeric_limits<std::size_t>::max() - capture->values.size()) {
    return EVO_STATUS_INTERNAL;
  }
  capture->columns = columns;
  capture->rows += rows;
  capture->next_offset += rows;
  capture->values.insert(capture->values.end(), values,
                         values + rows * columns);
  return EVO_STATUS_OK;
}

} // namespace

struct evo_model final {
  std::shared_ptr<ModelState> state;
};

struct evo_context final {
  std::shared_ptr<ModelState> state;
  std::size_t capacity{0};
  evo::InferenceProfile profile{evo::InferenceProfile::kExact};
  bool failed{false};
  std::unique_ptr<evo::cpu::Context> cpu;
#if defined(EVO_HAS_CUDA)
  std::unique_ptr<evo::cuda::PipelineModel> cuda;
  std::unique_ptr<evo::cuda::EsmcContext> cuda_esmc;
#endif
};

struct evo_batch final {
  std::size_t sequence_capacity{0};
  std::size_t token_count{0};
  std::vector<std::vector<std::uint8_t>> sequences;
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
  case EVO_BACKEND_MPS:
    return "mps";
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

evo_embedding_options evo_embedding_default_options(void) {
  return {sizeof(evo_embedding_options), nullptr,
          std::numeric_limits<std::size_t>::max(), nullptr};
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
        options.backend != EVO_BACKEND_CPU &&
        options.backend != EVO_BACKEND_MPS) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: unknown model backend");
    }
    if (options.backend == EVO_BACKEND_MPS && options.device_count != 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: MPS uses the system default Metal "
                     "device and does not accept a device list");
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
    auto status = state->file.open(path);
    if (!status.ok())
      return publish(status);
    state->model_id = metadata_string(state->file, "model.id");
    if (state->model_id.empty())
      state->model_id = metadata_string(state->file, "model.name", "unknown");
    state->architecture =
        metadata_string(state->file, "model.architecture", "unknown");
    state->artifact_profile = std::string{state->file.profile()};
    const std::string runtime_abi =
        metadata_string(state->file, "runtime.abi", "unknown");
    const auto *const architecture =
        evo::find_architecture(state->architecture);
    if (architecture == nullptr ||
        architecture->artifact_profile != state->artifact_profile ||
        architecture->runtime_abi != runtime_abi) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: artifact architecture/profile/ABI is not "
                     "registered");
    }
    state->architecture_spec = architecture;
    if (state->backend == EVO_BACKEND_AUTO) {
#if defined(EVO_HAS_CUDA)
      state->backend = architecture_backend_factory(*architecture,
                                                    EVO_BACKEND_CUDA) != nullptr
                           ? EVO_BACKEND_CUDA
                           : EVO_BACKEND_CPU;
#else
      state->backend = EVO_BACKEND_CPU;
#endif
    }
    if (state->backend == EVO_BACKEND_AUTO) {
      return publish(EVO_STATUS_INTERNAL,
                     "internal: automatic backend selection was unresolved");
    }
    if (architecture_backend_factory(*architecture, state->backend) ==
        nullptr) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: selected architecture/backend factory is "
                     "not registered");
    }
#if !defined(EVO_HAS_CUDA)
    if (state->backend == EVO_BACKEND_CUDA) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: this evo library was built without CUDA");
    }
#endif
#if !defined(EVO_HAS_MPS)
    if (state->backend == EVO_BACKEND_MPS) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: this evo library was built without MPS");
    }
#endif
    state->vocab_size = metadata_size(state->file, "config.vocab_size");
    state->max_context = metadata_size(state->file, "config.max_seqlen");
    state->embedding_width = metadata_size(state->file, "config.hidden_size");
    const std::size_t transformer_layers =
        metadata_size(state->file, "config.num_layers");
    const auto implementation = state->architecture_spec->implementation;
    const bool includes_post_final_norm =
        implementation == evo::ArchitectureImplementation::kEsmc ||
        implementation ==
            evo::ArchitectureImplementation::kGenebTransformerDecoder ||
        implementation == evo::ArchitectureImplementation::kGenebOlmoDecoder ||
        implementation == evo::ArchitectureImplementation::kGenebEsmEncoder ||
        implementation == evo::ArchitectureImplementation::kGenebBertEncoder ||
        implementation == evo::ArchitectureImplementation::kGenebGpt2Decoder ||
        implementation ==
            evo::ArchitectureImplementation::kGenebDnaGptDecoder ||
        implementation ==
            evo::ArchitectureImplementation::kGenebCustomEncoder ||
        implementation == evo::ArchitectureImplementation::kGenebMambaEncoder ||
        implementation ==
            evo::ArchitectureImplementation::kGenebHyenaDnaDecoder ||
        implementation ==
            evo::ArchitectureImplementation::kGenebStripedHyenaV1 ||
        implementation ==
            evo::ArchitectureImplementation::kGenebJanusDnaEncoder ||
        implementation ==
            evo::ArchitectureImplementation::kGenebRoformerEncoder;
    if (implementation ==
        evo::ArchitectureImplementation::kGenebSequenceCnnEncoder) {
      state->vocab_size = 256U;
    }
    const std::size_t default_embedding_layers =
        includes_post_final_norm &&
                transformer_layers != std::numeric_limits<std::size_t>::max()
            ? transformer_layers + 1U
            : transformer_layers;
    state->layer_count = metadata_size(
        state->file, "runtime.embedding_layer_count", default_embedding_layers);
    if (state->file.tokenizer_asset_descriptor().has_value()) {
      std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
      status = evo::ArtifactTokenizer::Load(
          std::string{state->file.artifact_root()},
          *state->file.tokenizer_asset_descriptor(), &tokenizer);
      if (!status.ok())
        return publish(status);
      const auto tokenizer_vocabulary_size = metadata_size(
          state->file, "runtime.tokenizer_vocabulary_size", state->vocab_size);
      if (tokenizer->vocabulary_size() != tokenizer_vocabulary_size) {
        return publish(EVO_STATUS_MODEL_FORMAT,
                       "model_format: tokenizer vocabulary size differs from "
                       "model vocabulary");
      }
      state->artifact_tokenizer = std::move(tokenizer);
    }
    if (evo::architecture_requires_artifact_tokenizer(
            *state->architecture_spec) &&
        !state->artifact_tokenizer) {
      return publish(EVO_STATUS_MODEL_FORMAT,
                     "model_format: architecture requires a verified "
                     "artifact tokenizer descriptor");
    }
    if (state->artifact_profile.rfind("geneb-", 0) == 0) {
      evo::GenebEmbeddingArtifactSpec spec;
      status = evo::geneb_embedding_spec_from_artifact(state->file, &spec);
      if (!status.ok())
        return publish(status);
      if (spec.runtime_id != state->model_id ||
          spec.reference.output_width != state->embedding_width ||
          spec.normalized.output_width != state->embedding_width) {
        return publish(EVO_STATUS_MODEL_FORMAT,
                       "model_format: GENEB preset identity/width differs "
                       "from runtime config");
      }
      state->geneb_embedding = std::move(spec);
    }
    if (state->max_context == 0 &&
        metadata_bool(state->file, "fixture.synthetic")) {
      state->max_context = std::numeric_limits<std::size_t>::max();
    }
    state->allow_synthetic =
        (options.flags & EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC) != 0;
#if defined(EVO_HAS_CUDA)
    if (state->backend == EVO_BACKEND_CUDA) {
      switch (state->architecture_spec->implementation) {
      case evo::ArchitectureImplementation::kUnknown:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture implementation is "
                       "unknown");
      case evo::ArchitectureImplementation::kEsmc:
        state->cuda_esmc_weights = std::make_unique<evo::cuda::EsmcModel>();
        status = state->cuda_esmc_weights->load(state->file, state->devices,
                                                state->allow_synthetic);
        break;
      case evo::ArchitectureImplementation::kStripedHyena2:
        state->cuda_weights = std::make_unique<evo::cuda::PipelineModel>();
        status = state->cuda_weights->load(state->file, state->devices, 1,
                                           state->allow_synthetic);
        break;
      case evo::ArchitectureImplementation::kHyenaDna:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "HyenaDNA implementation");
      case evo::ArchitectureImplementation::kGenebTransformerDecoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB transformer decoder implementation");
      case evo::ArchitectureImplementation::kGenebOlmoDecoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB OLMo decoder implementation");
      case evo::ArchitectureImplementation::kGenebEsmEncoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB ESM encoder implementation");
      case evo::ArchitectureImplementation::kGenebBertEncoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB BERT encoder implementation");
      case evo::ArchitectureImplementation::kGenebGpt2Decoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB GPT-2 decoder implementation");
      case evo::ArchitectureImplementation::kGenebDnaGptDecoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB DNA-GPT decoder implementation");
      case evo::ArchitectureImplementation::kGenebCustomEncoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB custom encoder implementation");
      case evo::ArchitectureImplementation::kGenebMambaEncoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB Mamba encoder implementation");
      case evo::ArchitectureImplementation::kGenebHyenaDnaDecoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB HyenaDNA decoder implementation");
      case evo::ArchitectureImplementation::kGenebStripedHyenaV1:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB Evo-1 implementation");
      case evo::ArchitectureImplementation::kGenebJanusDnaEncoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB JanusDNA encoder implementation");
      case evo::ArchitectureImplementation::kGenebSequenceCnnEncoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB sequence CNN encoder implementation");
      case evo::ArchitectureImplementation::kGenebRoformerEncoder:
        return publish(EVO_STATUS_UNSUPPORTED,
                       "unsupported: CUDA architecture factory has no "
                       "GENEB RoFormer encoder implementation");
      }
      if (!status.ok())
        return publish(status);
    }
#endif
#if defined(EVO_HAS_MPS)
    if (state->backend == EVO_BACKEND_MPS) {
      state->cpu_weights = std::make_unique<evo::cpu::Model>();
      status = evo::mps::ModelLoader::load(state->file, state->allow_synthetic,
                                           state->cpu_weights.get());
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
  return model == nullptr || !model->state
             ? ""
             : model->state->artifact_profile.c_str();
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

evo_status evo_model_encode(const evo_model *const model,
                            const uint8_t *const sequence,
                            const size_t sequence_length,
                            uint32_t *const tokens, const size_t token_capacity,
                            size_t *const token_count) {
  if (token_count != nullptr)
    *token_count = 0;
  return protect([&]() -> evo_status {
    if (model == nullptr || !model->state || sequence == nullptr ||
        sequence_length == 0 || token_count == nullptr ||
        (tokens == nullptr && token_capacity != 0)) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: model tokenizer arguments are invalid");
    }
    const std::string_view text{reinterpret_cast<const char *>(sequence),
                                sequence_length};
    std::vector<evo::TokenId> encoded;
    const auto status = encode_model_sequence(*model->state, text, &encoded);
    if (!status.ok())
      return publish(status);
    *token_count = encoded.size();
    if (tokens == nullptr)
      return publish(evo::Status::Ok());
    if (token_capacity < encoded.size()) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: token buffer is too small");
    }
    for (std::size_t index = 0; index < encoded.size(); ++index)
      tokens[index] = encoded[index];
    return publish(evo::Status::Ok());
  });
}

evo_status evo_model_decode_token(const evo_model *const model,
                                  const uint32_t token,
                                  uint8_t *const byte_out) {
  return protect([&]() -> evo_status {
    if (model == nullptr || !model->state || byte_out == nullptr) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: model detokenizer arguments are "
                     "invalid");
    }
    return publish(decode_model_token(
        *model->state, static_cast<evo::TokenId>(token), byte_out));
  });
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
    constexpr std::uint32_t known_flags = EVO_CONTEXT_FLAG_FAST_Q8_KV;
    if (options.context_size == 0 || (options.flags & ~known_flags) != 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: invalid context parameters");
    }
    evo::InferenceProfile profile =
        (options.flags & EVO_CONTEXT_FLAG_FAST_Q8_KV) != 0
            ? evo::InferenceProfile::kFastQ8Kv
            : evo::InferenceProfile::kExact;
    if (model->state->backend == EVO_BACKEND_CUDA &&
        model->state->architecture_spec->implementation ==
            evo::ArchitectureImplementation::kEsmc &&
        options.flags != 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: ESMC supports only the exact CUDA "
                     "profile");
    }
    if (model->state->max_context != 0 &&
        options.context_size > model->state->max_context) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: requested context exceeds model "
                     "maximum");
    }
    if (model->state->backend == EVO_BACKEND_CPU ||
        model->state->backend == EVO_BACKEND_MPS) {
      if (options.flags != 0) {
        return publish(EVO_STATUS_INVALID_ARGUMENT,
                       "invalid_argument: CPU and MPS contexts do not accept "
                       "CUDA KV profile flags");
      }
      profile = model->state->backend == EVO_BACKEND_MPS
                    ? evo::InferenceProfile::kMpsF32
                    : evo::InferenceProfile::kCpuF32;
      std::lock_guard<std::mutex> lock{model->state->cpu_mutex};
      if (!model->state->cpu_weights) {
        if (model->state->backend == EVO_BACKEND_MPS) {
          return publish(EVO_STATUS_INTERNAL,
                         "internal: MPS model weights are unavailable");
        }
        auto weights = std::make_unique<evo::cpu::Model>();
        const auto load_status =
            weights->load(model->state->file, model->state->allow_synthetic);
        if (!load_status.ok())
          return publish(load_status);
        model->state->cpu_weights = std::move(weights);
      }
      auto cpu = std::make_unique<evo::cpu::Context>();
      const auto status = cpu->initialize_shared(*model->state->cpu_weights,
                                                 options.context_size);
      if (!status.ok())
        return publish(status);
      auto handle = std::make_unique<evo_context>();
      handle->state = model->state;
      handle->capacity = options.context_size;
      handle->profile = profile;
      handle->cpu = std::move(cpu);
      *context_out = handle.release();
      return publish(evo::Status::Ok());
    }
#if defined(EVO_HAS_CUDA)
    if (model->state->backend != EVO_BACKEND_CUDA) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: selected inference backend is unavailable");
    }
    auto handle = std::make_unique<evo_context>();
    handle->state = model->state;
    handle->capacity = options.context_size;
    handle->profile = profile;
    switch (model->state->architecture_spec->implementation) {
    case evo::ArchitectureImplementation::kUnknown:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA context implementation is unknown");
    case evo::ArchitectureImplementation::kEsmc: {
      if (!model->state->cuda_esmc_weights) {
        return publish(EVO_STATUS_INTERNAL,
                       "internal: ESMC CUDA model weights are unavailable");
      }
      auto cuda = std::make_unique<evo::cuda::EsmcContext>();
      const auto status = cuda->initialize_shared(
          *model->state->cuda_esmc_weights, options.context_size);
      if (!status.ok())
        return publish(status);
      handle->cuda_esmc = std::move(cuda);
      break;
    }
    case evo::ArchitectureImplementation::kStripedHyena2: {
      if (!model->state->cuda_weights) {
        return publish(EVO_STATUS_INTERNAL,
                       "internal: CUDA model weights are unavailable");
      }
      auto cuda = std::make_unique<evo::cuda::PipelineModel>();
      const auto status = cuda->initialize_shared(
          *model->state->cuda_weights, options.context_size, profile);
      if (!status.ok())
        return publish(status);
      handle->cuda = std::move(cuda);
      break;
    }
    case evo::ArchitectureImplementation::kHyenaDna:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "HyenaDNA context implementation");
    case evo::ArchitectureImplementation::kGenebTransformerDecoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB transformer decoder context implementation");
    case evo::ArchitectureImplementation::kGenebOlmoDecoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB OLMo decoder context implementation");
    case evo::ArchitectureImplementation::kGenebEsmEncoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB ESM encoder context implementation");
    case evo::ArchitectureImplementation::kGenebBertEncoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB BERT encoder context implementation");
    case evo::ArchitectureImplementation::kGenebGpt2Decoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB GPT-2 decoder context implementation");
    case evo::ArchitectureImplementation::kGenebDnaGptDecoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB DNA-GPT decoder context implementation");
    case evo::ArchitectureImplementation::kGenebCustomEncoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB custom encoder context implementation");
    case evo::ArchitectureImplementation::kGenebMambaEncoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB Mamba encoder context implementation");
    case evo::ArchitectureImplementation::kGenebHyenaDnaDecoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB HyenaDNA decoder context implementation");
    case evo::ArchitectureImplementation::kGenebStripedHyenaV1:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB Evo-1 context implementation");
    case evo::ArchitectureImplementation::kGenebJanusDnaEncoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB JanusDNA encoder context implementation");
    case evo::ArchitectureImplementation::kGenebSequenceCnnEncoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB sequence CNN encoder context implementation");
    case evo::ArchitectureImplementation::kGenebRoformerEncoder:
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: CUDA architecture factory has no "
                     "GENEB RoFormer encoder context implementation");
    }
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
  if (context != nullptr && context->cpu)
    return context->cpu->position();
#if defined(EVO_HAS_CUDA)
  if (context != nullptr && context->cuda_esmc)
    return context->cuda_esmc->position();
  return context == nullptr || !context->cuda ? 0 : context->cuda->position();
#else
  (void)context;
  return 0;
#endif
}

size_t evo_context_capacity(const evo_context *const context) {
  return context == nullptr ? 0 : context->capacity;
}

const char *evo_context_profile(const evo_context *const context) {
  return context == nullptr ? ""
                            : evo::inference_profile_name(context->profile);
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
    if (!context->state || (context->state->architecture_spec->capabilities &
                            evo::kArchitectureLogits) == 0) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: architecture does not expose logits");
    }
    if (batch->sequences.size() != 1) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: exact backend currently requires batch "
                     "size one");
    }
    const auto &sequence = batch->sequences.front();
    if (sequence.empty() || evo_context_position(context) != 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: prefill requires a fresh context and "
                     "a nonempty sequence within capacity");
    }
    std::vector<evo::TokenId> tokens;
    const std::string_view sequence_view{
        reinterpret_cast<const char *>(sequence.data()), sequence.size()};
    const auto encode_status =
        encode_model_sequence(*context->state, sequence_view, &tokens);
    if (!encode_status.ok())
      return publish(encode_status);
    if (tokens.size() > context->capacity) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: encoded sequence exceeds context "
                     "capacity");
    }
    // From here onward, any status/exception can follow a partial state
    // advance. Keep the context poisoned unless every chunk and callback
    // completes.
    context->failed = true;
    if (context->cpu) {
      const std::size_t chunk_capacity = context->cpu->activation_capacity();
      const std::size_t columns = context->cpu->config().vocab_size;
      std::size_t offset = 0;
      bool first = true;
      while (offset < tokens.size()) {
        const std::size_t rows =
            std::min(chunk_capacity, tokens.size() - offset);
        const std::vector<evo::TokenId> chunk(
            tokens.begin() + static_cast<std::ptrdiff_t>(offset),
            tokens.begin() + static_cast<std::ptrdiff_t>(offset + rows));
        std::vector<float> logits;
        const auto status = first ? context->cpu->prefill(chunk, &logits)
                                  : context->cpu->prefill_chunk(chunk, &logits);
        if (!status.ok())
          return publish(status);
        if (logits.size() != rows * columns)
          return publish(EVO_STATUS_INTERNAL,
                         "internal: host backend returned incomplete logits");
        const evo_status callback_status =
            callback(logits.data(), rows, columns, offset, user_data);
        if (callback_status != EVO_STATUS_OK) {
          return publish(valid_status(callback_status) ? callback_status
                                                       : EVO_STATUS_INTERNAL,
                         "callback: logits consumer aborted host prefill");
        }
        offset += rows;
        first = false;
      }
      context->failed = false;
      return publish(evo::Status::Ok());
    }
#if defined(EVO_HAS_CUDA)
    if (context->cuda_esmc) {
      std::vector<float> logits;
      const auto status = context->cuda_esmc->prefill(tokens, &logits);
      const std::size_t columns = context->cuda_esmc->config().vocab_size;
      if (!status.ok())
        return publish(status);
      if (logits.size() != tokens.size() * columns) {
        return publish(EVO_STATUS_INTERNAL,
                       "internal: ESMC CUDA backend returned incomplete "
                       "prefill logits");
      }
      const evo_status callback_status =
          callback(logits.data(), tokens.size(), columns, 0, user_data);
      if (callback_status != EVO_STATUS_OK) {
        return publish(valid_status(callback_status) ? callback_status
                                                     : EVO_STATUS_INTERNAL,
                       "callback: logits consumer aborted ESMC prefill");
      }
      context->failed = false;
      return publish(evo::Status::Ok());
    }
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
    if (!context->state || (context->state->architecture_spec->capabilities &
                            evo::kArchitectureGenerate) == 0) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: architecture does not support incremental "
                     "decode");
    }
    const std::size_t offset = evo_context_position(context);
    if (offset == 0 || offset >= context->capacity || !context->state ||
        !evo::token_id_in_vocabulary(token, context->state->vocab_size)) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: decode token/context is invalid");
    }
    context->failed = true;
    if (context->cpu) {
      std::vector<float> logits;
      const auto status =
          context->cpu->decode(static_cast<evo::TokenId>(token), &logits);
      if (!status.ok())
        return publish(status);
      const std::size_t columns = context->cpu->config().vocab_size;
      if (logits.size() != columns)
        return publish(
            EVO_STATUS_INTERNAL,
            "internal: host backend returned incomplete decode logits");
      const evo_status callback_status =
          callback(logits.data(), 1, columns, offset, user_data);
      if (callback_status != EVO_STATUS_OK) {
        return publish(valid_status(callback_status) ? callback_status
                                                     : EVO_STATUS_INTERNAL,
                       "callback: logits consumer aborted host decode");
      }
      context->failed = false;
      return publish(evo::Status::Ok());
    }
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
    if (!context->state || (context->state->architecture_spec->capabilities &
                            evo::kArchitectureEmbed) == 0) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: architecture does not expose embeddings");
    }
    if (context->state->architecture_spec->implementation ==
        evo::ArchitectureImplementation::kGenebSequenceCnnEncoder) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: GENEB sequence CNN exposes its returned "
                     "sequence embedding through evo_context_embed_ex with "
                     "a GENEB preset");
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
    const auto &sequence = batch->sequences.front();
    if (sequence.empty() || evo_context_position(context) != 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: embedding requires a fresh context and "
                     "a nonempty sequence within capacity");
    }
    std::vector<evo::TokenId> tokens;
    const std::string_view sequence_view{
        reinterpret_cast<const char *>(sequence.data()), sequence.size()};
    const auto encode_status =
        encode_model_sequence(*context->state, sequence_view, &tokens);
    if (!encode_status.ok())
      return publish(encode_status);
    if (tokens.size() > context->capacity) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: encoded sequence exceeds context "
                     "capacity");
    }
    context->failed = true;
    if (context->cpu) {
      const std::size_t chunk_capacity = context->cpu->activation_capacity();
      const std::size_t columns = context->cpu->config().width;
      std::size_t offset = 0;
      bool first = true;
      while (offset < tokens.size()) {
        const std::size_t rows =
            std::min(chunk_capacity, tokens.size() - offset);
        const std::vector<evo::TokenId> chunk(
            tokens.begin() + static_cast<std::ptrdiff_t>(offset),
            tokens.begin() + static_cast<std::ptrdiff_t>(offset + rows));
        std::vector<float> embedding;
        const auto status =
            first ? context->cpu->prefill_embedding(chunk, layer, &embedding)
                  : context->cpu->prefill_chunk_embedding(chunk, layer,
                                                          &embedding);
        if (!status.ok())
          return publish(status);
        if (embedding.size() != rows * columns)
          return publish(
              EVO_STATUS_INTERNAL,
              "internal: host backend returned incomplete embeddings");
        const evo_status callback_status =
            callback(embedding.data(), rows, columns, offset, user_data);
        if (callback_status != EVO_STATUS_OK) {
          return publish(valid_status(callback_status) ? callback_status
                                                       : EVO_STATUS_INTERNAL,
                         "callback: embedding consumer aborted host inference");
        }
        offset += rows;
        first = false;
      }
      context->failed = false;
      return publish(evo::Status::Ok());
    }
#if defined(EVO_HAS_CUDA)
    if (context->cuda_esmc) {
      std::vector<float> embedding;
      const auto status =
          context->cuda_esmc->prefill_embedding(tokens, layer, &embedding);
      const std::size_t columns = context->cuda_esmc->config().width;
      if (!status.ok())
        return publish(status);
      if (embedding.size() != tokens.size() * columns) {
        return publish(EVO_STATUS_INTERNAL,
                       "internal: ESMC CUDA backend returned incomplete "
                       "embeddings");
      }
      const evo_status callback_status =
          callback(embedding.data(), tokens.size(), columns, 0, user_data);
      if (callback_status != EVO_STATUS_OK) {
        return publish(valid_status(callback_status) ? callback_status
                                                     : EVO_STATUS_INTERNAL,
                       "callback: embedding consumer aborted ESMC inference");
      }
      context->failed = false;
      return publish(evo::Status::Ok());
    }
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

evo_status evo_context_embed_ex(evo_context *const context,
                                const evo_batch *const batch,
                                const evo_embedding_options *const options,
                                const evo_embedding_ex_callback callback,
                                void *const user_data) {
  return protect([&]() -> evo_status {
    if (context == nullptr || batch == nullptr || callback == nullptr) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: extended embedding argument is null");
    }
    if (options != nullptr &&
        options->struct_size < sizeof(evo_embedding_options)) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: evo_embedding_options is too small");
    }
    const evo_embedding_options selected =
        options == nullptr ? evo_embedding_default_options() : *options;
    const std::string_view preset =
        selected.preset == nullptr ? std::string_view{} : selected.preset;
    const std::string_view pooling =
        selected.pooling == nullptr ? std::string_view{} : selected.pooling;
    if (batch->sequences.size() != 1 || batch->sequences.front().empty()) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: extended embeddings require one nonempty "
                     "sequence");
    }
    if (preset.empty()) {
      if (selected.layer == std::numeric_limits<std::size_t>::max()) {
        return publish(EVO_STATUS_INVALID_ARGUMENT,
                       "invalid_argument: explicit embedding requires layer");
      }
      const std::string resolved_pooling =
          pooling.empty() ? std::string{"none"} : std::string{pooling};
      if (resolved_pooling != "none" && resolved_pooling != "mean" &&
          resolved_pooling != "last") {
        return publish(EVO_STATUS_INVALID_ARGUMENT,
                       "invalid_argument: pooling must be none, mean, or last");
      }
      std::vector<evo::TokenId> tokens;
      const auto &sequence = batch->sequences.front();
      const std::string_view sequence_view{
          reinterpret_cast<const char *>(sequence.data()), sequence.size()};
      auto status =
          encode_model_sequence(*context->state, sequence_view, &tokens);
      if (!status.ok())
        return publish(status);
      LegacyEmbeddingCapture capture;
      const auto embedded = evo_context_embed(
          context, batch, selected.layer, capture_legacy_embedding, &capture);
      if (embedded != EVO_STATUS_OK)
        return embedded;
      std::vector<float> result;
      std::size_t rows = capture.rows;
      if (resolved_pooling == "none") {
        result = std::move(capture.values);
      } else if (resolved_pooling == "last") {
        if (capture.rows == 0)
          return publish(EVO_STATUS_INTERNAL,
                         "internal: embedding callback returned no rows");
        const auto begin =
            capture.values.end() - static_cast<std::ptrdiff_t>(capture.columns);
        result.assign(begin, capture.values.end());
        rows = 1;
      } else {
        const std::vector<std::uint8_t> mask(capture.rows, 1U);
        status = evo::pool_geneb_embedding(capture.values, capture.rows,
                                           capture.columns, mask,
                                           "attention-mask-mean", &result);
        if (!status.ok())
          return publish(status);
        rows = 1;
      }
      const evo_embedding_result_info info{
          sizeof(evo_embedding_result_info),
          "",
          "explicit-layer",
          resolved_pooling.c_str(),
          selected.layer,
          sequence.size(),
          sequence.size(),
          0,
          0,
          0,
          0,
          tokens.size(),
          rows,
          capture.columns,
      };
      const evo_status callback_status =
          callback(result.data(), rows, capture.columns, 0, &info, user_data);
      if (callback_status != EVO_STATUS_OK) {
        context->failed = true;
        return publish(valid_status(callback_status) ? callback_status
                                                     : EVO_STATUS_INTERNAL,
                       "callback: extended embedding consumer aborted");
      }
      return publish(evo::Status::Ok());
    }

    if (selected.layer != std::numeric_limits<std::size_t>::max() ||
        !pooling.empty()) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: preset is mutually exclusive with "
                     "layer and pooling");
    }
    if (!context->state || !context->state->geneb_embedding.has_value() ||
        !context->cpu || !context->state->cpu_weights) {
      return publish(EVO_STATUS_UNSUPPORTED,
                     "unsupported: GENEB presets require a registered CPU "
                     "artifact and preprocessing contract");
    }
    evo::GenebEmbeddingPresetKind preset_kind;
    if (preset == "geneb-v4-reference") {
      preset_kind = evo::GenebEmbeddingPresetKind::kReference;
    } else if (preset == "geneb-v4-normalized" || preset == "geneb") {
      preset_kind = evo::GenebEmbeddingPresetKind::kNormalized;
    } else {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: unknown embedding preset");
    }
    if (context->failed || evo_context_position(context) != 0) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: embedding preset requires a fresh "
                     "context");
    }
    const auto &sequence = batch->sequences.front();
    const std::string_view sequence_view{
        reinterpret_cast<const char *>(sequence.data()), sequence.size()};
    evo::GenebPreparedEmbeddingInput prepared;
    auto status = context->state->cpu_weights->prepare_geneb_embedding_input(
        sequence_view, &prepared);
    if (!status.ok())
      return publish(status);
    if (prepared.tokens.size() > context->capacity) {
      return publish(EVO_STATUS_INVALID_ARGUMENT,
                     "invalid_argument: GENEB effective token count exceeds "
                     "context capacity");
    }
    const auto &resolved = evo::geneb_embedding_preset(
        *context->state->geneb_embedding, preset_kind);
    const std::size_t layer = context->state->layer_count - 1U;
    context->failed = true;
    std::vector<float> hidden;
    status = context->cpu->prefill_embedding_masked(
        prepared.tokens, prepared.attention_mask, layer, &hidden);
    if (!status.ok())
      return publish(status);
    const std::size_t columns = context->cpu->config().width;
    if (columns == 0U || hidden.empty() || hidden.size() % columns != 0U ||
        resolved.output_width != columns) {
      return publish(EVO_STATUS_INTERNAL,
                     "internal: GENEB embedding shape/preset width differs");
    }
    const std::size_t hidden_rows = hidden.size() / columns;
    std::vector<std::uint8_t> spatial_mask;
    const std::vector<std::uint8_t> *pooling_mask = &prepared.attention_mask;
    if (hidden_rows != prepared.attention_mask.size()) {
      if (resolved.pooling != "spatial-mean") {
        return publish(EVO_STATUS_INTERNAL,
                       "internal: GENEB hidden rows differ from token mask");
      }
      spatial_mask.assign(hidden_rows, 1U);
      pooling_mask = &spatial_mask;
    }
    std::vector<float> result;
    status = evo::pool_geneb_embedding(hidden, hidden_rows, columns,
                                       *pooling_mask,
                                       resolved.pooling, &result);
    if (!status.ok())
      return publish(status);
    if (result.size() != columns) {
      return publish(EVO_STATUS_INTERNAL,
                     "internal: GENEB preset did not produce one vector");
    }
    if (prepared.transform.pad_left > std::numeric_limits<std::size_t>::max() -
                                          prepared.token_plan.pad_left ||
        prepared.transform.pad_right > std::numeric_limits<std::size_t>::max() -
                                           prepared.token_plan.pad_right) {
      return publish(EVO_STATUS_INTERNAL,
                     "internal: GENEB padding metadata overflowed");
    }
    const evo_embedding_result_info info{
        sizeof(evo_embedding_result_info),
        resolved.name.c_str(),
        resolved.hidden_tap.c_str(),
        resolved.pooling.c_str(),
        layer,
        prepared.transform.original_length,
        prepared.transform.effective_length,
        prepared.transform.crop_left,
        prepared.transform.crop_right,
        prepared.transform.pad_left + prepared.token_plan.pad_left,
        prepared.transform.pad_right + prepared.token_plan.pad_right,
        prepared.tokens.size(),
        1,
        columns,
    };
    const evo_status callback_status =
        callback(result.data(), 1, columns, 0, &info, user_data);
    if (callback_status != EVO_STATUS_OK) {
      return publish(valid_status(callback_status) ? callback_status
                                                   : EVO_STATUS_INTERNAL,
                     "callback: GENEB embedding consumer aborted");
    }
    context->failed = false;
    return publish(evo::Status::Ok());
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
    batch->sequences.emplace_back(sequence, sequence + sequence_length);
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
