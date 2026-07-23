// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "evo2c/cli.hpp"
#include "evo2c/status.hpp"

namespace evo2c::cuda {

// Runs the CUDA-backed score or generation command after common CLI validation.
// Synthetic models are accepted only by the dedicated test executable.
[[nodiscard]] Status run_inference_cli(const CliOptions &options,
                                       bool allow_test_fixture = false);

} // namespace evo2c::cuda
