// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <string_view>

#include "evo/status.hpp"

namespace evo {

// Resolve an artifact that evo-fetch already placed in the local Hugging Face
// cache. The manifest receipt, manifest, every file size, and every SHA256 are
// verified without network access or a downloader subprocess.
[[nodiscard]] Status resolve_cached_hf_artifact(std::string_view repository,
                                                std::string *model_path);

} // namespace evo
