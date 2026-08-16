// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "evo/cli.hpp"
#include "evo/cpu/model.hpp"
#include "evo/status.hpp"

namespace evo::cpu {

[[nodiscard]] Status run_inference_cli_loaded(const CliOptions &options,
                                              const Model &model,
                                              double load_seconds);

} // namespace evo::cpu
