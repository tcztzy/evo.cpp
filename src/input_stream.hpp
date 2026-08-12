// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <functional>
#include <istream>
#include <string>

#include "evo/status.hpp"

namespace evo::detail {

using InputStreamCallback = std::function<Status(std::istream &input)>;

// zlib's transparent read mode accepts ordinary and gzip-compressed bytes.
// A path of "-" reads a duplicate of stdin so ownership remains with the CLI.
[[nodiscard]] Status with_input_stream(const std::string &path,
                                       const InputStreamCallback &callback);

[[nodiscard]] Status read_bounded_line(std::istream &input,
                                       std::size_t max_bytes,
                                       std::string *line, bool *has_line);

} // namespace evo::detail
