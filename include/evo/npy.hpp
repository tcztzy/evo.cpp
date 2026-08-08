// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "evo/status.hpp"

namespace evo::npy {

// Writes a nonempty row-major, little-endian F32 matrix in NumPy's NPY format.
[[nodiscard]] Status write_f32(const std::string &path,
                               const std::vector<float> &values,
                               std::size_t rows, std::size_t columns);

}  // namespace evo::npy
