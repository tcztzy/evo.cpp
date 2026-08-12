// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>

#include "evo/status.hpp"

namespace evo {

enum class InferenceProfile { kExact, kFastQ8Kv, kCpuF32 };

[[nodiscard]] const char *
inference_profile_name(InferenceProfile profile) noexcept;
[[nodiscard]] Status parse_inference_profile(std::string_view text,
                                             InferenceProfile *profile);
[[nodiscard]] bool
inference_profile_is_exact(InferenceProfile profile) noexcept;

} // namespace evo
