// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "evo/cli.hpp"
#include "evo/status.hpp"

namespace evo::mps {

[[nodiscard]] Status run_inference_cli(const CliOptions &options,
                                       bool allow_test_fixture = false);

} // namespace evo::mps
