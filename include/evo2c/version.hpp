// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>

namespace evo2c {

inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;

[[nodiscard]] std::string_view version() noexcept;

}  // namespace evo2c

