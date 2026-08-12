// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "evo/cli.hpp"
#include "evo/status.hpp"

namespace evo::cuda {

[[nodiscard]] Status run_hybrid_cli(const CliOptions &options,
                                    bool allow_test_fixture = false);

} // namespace evo::cuda
