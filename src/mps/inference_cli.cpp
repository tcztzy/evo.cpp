// SPDX-License-Identifier: Apache-2.0
#include "inference_cli.hpp"

#include "../cpu/inference_cli_internal.hpp"
#include "model.hpp"

#include <chrono>

#include "evo/cpu/model.hpp"
#include "evo/model_format.hpp"

namespace evo::mps {

Status run_inference_cli(const CliOptions &options,
                         const bool allow_test_fixture) {
  if (options.backend != ExecutionBackend::kMps ||
      options.inference_profile != InferenceProfile::kMpsF32) {
    return {ErrorCode::kInvalidArgument,
            "MPS backend requires --backend mps and --profile mps-f32"};
  }
  const auto load_start = std::chrono::steady_clock::now();
  ModelFile artifact;
  auto status = artifact.open(options.model_path);
  if (!status.ok())
    return status;
  cpu::Model model;
  status = ModelLoader::load(artifact, allow_test_fixture, &model);
  if (!status.ok())
    return status;
  const double load_seconds = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - load_start)
                                  .count();
  return cpu::run_inference_cli_loaded(options, model, load_seconds);
}

} // namespace evo::mps
