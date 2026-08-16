// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>

#include "evo/status.hpp"

namespace evo {

enum class ExecutionBackend { kAuto, kCpu, kCuda, kMps };

[[nodiscard]] const char *
execution_backend_name(ExecutionBackend backend) noexcept;
[[nodiscard]] Status parse_execution_backend(std::string_view text,
                                             ExecutionBackend *backend);

} // namespace evo
