// SPDX-License-Identifier: Apache-2.0
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "evo/cli.hpp"
#include "evo/cpu/inference_cli.hpp"
#include "evo/model_format.hpp"
#include "evo/model_registry.hpp"
#include "evo/server.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"
#include "evo/version.hpp"

#include "artifact_cache.hpp"

#if defined(EVO_HAS_CUDA)
#include "evo/cuda/hybrid_cli.hpp"
#include "evo/cuda/inference_cli.hpp"
#endif

#if defined(EVO_HAS_MPS)
#include "mps/inference_cli.hpp"
#endif

namespace {

void print_help() { std::cout << evo::cli_usage(); }

evo::Status dump_tokens(const std::string_view label,
                        const std::string_view bytes,
                        const evo::ArchitectureTokenizer tokenizer) {
  std::vector<evo::TokenId> tokens;
  const auto status = evo::encode_sequence(tokenizer, bytes, &tokens);
  if (!status.ok())
    return status;
  std::cerr << "tokens " << label << "=[";
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (index != 0) {
      std::cerr << ',';
    }
    std::cerr << tokens[index];
  }
  std::cerr << "]\n";
  return evo::Status::Ok();
}

void dump_byte_tokens(const std::string_view label,
                      const std::string_view bytes) {
  const auto tokens = evo::encode_bytes(bytes);
  std::cerr << "tokens " << label << "=[";
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (index != 0)
      std::cerr << ',';
    std::cerr << tokens[index];
  }
  std::cerr << "]\n";
}

int fail(const evo::Status &status) {
  std::cerr << "evo: " << evo::error_code_name(status.code()) << ": "
            << status.message() << '\n';
  return evo::exit_code(status.code());
}

const evo::ArchitectureBackendFactorySpec *architecture_backend_factory(
    const evo::ArchitectureSpec &architecture,
    const evo::ExecutionBackend backend) noexcept {
  switch (backend) {
  case evo::ExecutionBackend::kCpu:
    return evo::find_architecture_backend_factory(
        architecture, evo::kArchitectureBackendCpu);
  case evo::ExecutionBackend::kMps:
    return evo::find_architecture_backend_factory(
        architecture, evo::kArchitectureBackendMps);
  case evo::ExecutionBackend::kAuto:
    return nullptr;
  case evo::ExecutionBackend::kCuda:
    return evo::find_architecture_backend_factory(
        architecture, evo::kArchitectureBackendCuda);
  }
  return nullptr;
}

evo::Status resolve_automatic_backend(evo::CliOptions *const options) {
  if (options == nullptr)
    return {evo::ErrorCode::kInvalidArgument, "CLI options are null"};
  if (options->backend != evo::ExecutionBackend::kAuto)
    return evo::Status::Ok();
  const bool cuda_placement =
      !options->gpu_ids.empty() || options->gpu_layers.has_value();
  options->backend = cuda_placement ? evo::ExecutionBackend::kCuda
                                    : evo::ExecutionBackend::kCpu;
  if (options->backend == evo::ExecutionBackend::kCpu) {
    if (options->profile_explicit &&
        options->inference_profile != evo::InferenceProfile::kCpuF32) {
      return {evo::ErrorCode::kInvalidArgument,
              "automatic CPU backend requires --profile cpu-f32 when the "
              "profile is explicit"};
    }
    options->inference_profile = evo::InferenceProfile::kCpuF32;
  } else if (!options->gpu_layers.has_value() &&
             options->inference_profile == evo::InferenceProfile::kCpuF32) {
    return {evo::ErrorCode::kInvalidArgument,
            "automatic CUDA backend does not accept --profile cpu-f32"};
  }
  return evo::Status::Ok();
}

evo::Status validate_model_operation(evo::CliOptions *const options) {
  if (options == nullptr)
    return {evo::ErrorCode::kInvalidArgument, "CLI options are null"};
  evo::ModelFile artifact;
  auto status = artifact.open(options->model_path);
  if (!status.ok()) {
    // Preserve the original diagnostic contract: --dump-tokens remains useful
    // even when the artifact itself cannot be opened.  A valid artifact uses
    // its registered tokenizer below.
    if (options->mode == evo::RunMode::kGenerate && options->dump_tokens)
      dump_byte_tokens("prompt", options->prompt);
    return status;
  }
  const auto *const entry = artifact.find_metadata("model.architecture");
  if (entry == nullptr || entry->type != evo::MetadataType::kString) {
    return {evo::ErrorCode::kModelFormat,
            "model.architecture metadata is missing"};
  }
  const std::string architecture{entry->value.begin(), entry->value.end()};
  const auto *const spec = evo::find_architecture(architecture);
  if (spec == nullptr) {
    return {evo::ErrorCode::kUnsupported,
            "artifact architecture is not registered: " + architecture};
  }
  unsigned required = 0;
  switch (options->mode) {
  case evo::RunMode::kGenerate:
    required = evo::kArchitectureGenerate;
    break;
  case evo::RunMode::kScore:
  case evo::RunMode::kBench:
    required = evo::kArchitectureScore;
    break;
  case evo::RunMode::kLogits:
    required = evo::kArchitectureLogits;
    break;
  case evo::RunMode::kEmbed:
    required = evo::kArchitectureEmbed;
    break;
  case evo::RunMode::kVariantScore:
    required = evo::kArchitectureVariant;
    break;
  case evo::RunMode::kServe:
    required = evo::kArchitectureServe;
    break;
  }
  if ((spec->capabilities & required) == 0) {
    return {evo::ErrorCode::kUnsupported,
            "operation is not supported by architecture " + architecture};
  }
  status = resolve_automatic_backend(options);
  if (!status.ok())
    return status;
  const auto *const backend_factory =
      architecture_backend_factory(*spec, options->backend);
  if (backend_factory == nullptr) {
    return {evo::ErrorCode::kUnsupported,
            "backend " +
                std::string{evo::execution_backend_name(options->backend)} +
                " is not supported by architecture " + architecture};
  }
  if (spec->implementation == evo::ArchitectureImplementation::kEsmc &&
      backend_factory->backend == evo::kArchitectureBackendCuda &&
      options->inference_profile != evo::InferenceProfile::kExact) {
    return {evo::ErrorCode::kUnsupported,
            "ESMC CUDA inference supports only --profile exact"};
  }
  if (spec->implementation == evo::ArchitectureImplementation::kEsmc &&
      backend_factory->backend == evo::kArchitectureBackendCuda &&
      options->gpu_ids.size() > 1) {
    return {evo::ErrorCode::kUnsupported,
            "ESMC CUDA inference currently supports exactly one GPU"};
  }
  if (options->mode == evo::RunMode::kGenerate && options->dump_tokens) {
    status = dump_tokens("prompt", options->prompt, spec->tokenizer);
    if (!status.ok())
      return status;
  }
  return evo::Status::Ok();
}

} // namespace

int main(const int argc, char **argv) {
  try {
    if (argc == 2) {
      const std::string_view arg{argv[1]};
      if (arg == "--help" || arg == "-h") {
        print_help();
        return 0;
      }
      if (arg == "--version") {
        std::cout << "evo " << evo::version() << '\n';
        return 0;
      }
    }
    if (argc == 3) {
      const std::string_view command{argv[1]};
      const std::string_view argument{argv[2]};
      if ((command == "run" || command == "score" || command == "logits" ||
           command == "embed" || command == "variant-score" ||
           command == "serve" || command == "bench") &&
          (argument == "--help" || argument == "-h")) {
        print_help();
        return 0;
      }
    }

    evo::CliOptions options;
    auto status = evo::parse_cli(argc, argv, &options);
    if (!status.ok()) {
      return fail(status);
    }
    if (!options.hf_repo.empty()) {
      status =
          evo::resolve_cached_hf_artifact(options.hf_repo, &options.model_path);
      if (!status.ok()) {
        return fail(status);
      }
    }
    if (options.mode == evo::RunMode::kGenerate) {
      if (options.prompt.size() > options.context_size ||
          options.generated_tokens >
              options.context_size - options.prompt.size()) {
        return fail({evo::ErrorCode::kInvalidArgument,
                     "prompt bytes plus generated tokens exceed --ctx"});
      }
    }
    status = validate_model_operation(&options);
    if (!status.ok()) {
      return fail(status);
    }

#if defined(EVO_ALLOW_TEST_FIXTURE)
    constexpr bool allow_test_fixture = true;
#else
    constexpr bool allow_test_fixture = false;
#endif
    if (options.mode == evo::RunMode::kServe) {
      status = evo::run_server(options, allow_test_fixture);
      return status.ok() ? 0 : fail(status);
    }

    if (options.backend == evo::ExecutionBackend::kCpu) {
      status = evo::cpu::run_inference_cli(options, allow_test_fixture);
      return status.ok() ? 0 : fail(status);
    }
    if (options.backend == evo::ExecutionBackend::kMps) {
#if defined(EVO_HAS_MPS)
      status = evo::mps::run_inference_cli(options, allow_test_fixture);
      return status.ok() ? 0 : fail(status);
#else
      return fail({evo::ErrorCode::kUnsupported,
                   "this evo binary was built without MPS support; rebuild "
                   "on macOS arm64 with -DEVO_MPS=ON"});
#endif
    }
#if defined(EVO_HAS_CUDA)
    if (options.gpu_layers.has_value()) {
      status = evo::cuda::run_hybrid_cli(options, allow_test_fixture);
      return status.ok() ? 0 : fail(status);
    }
    status = evo::cuda::run_inference_cli(options, allow_test_fixture);
    return status.ok() ? 0 : fail(status);
#else
    return fail({evo::ErrorCode::kUnsupported,
                 "this evo binary was built without CUDA support; pass "
                 "--backend cpu or rebuild with -DEVO_CUDA=ON"});
#endif
  } catch (const std::exception &error) {
    return fail({evo::ErrorCode::kInternal, error.what()});
  } catch (...) {
    return fail({evo::ErrorCode::kInternal, "unknown exception"});
  }
}
