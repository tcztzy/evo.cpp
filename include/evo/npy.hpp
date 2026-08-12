// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "evo/status.hpp"

namespace evo::npy {

class F32MatrixWriter final {
public:
  F32MatrixWriter() = default;
  ~F32MatrixWriter();

  F32MatrixWriter(const F32MatrixWriter &) = delete;
  F32MatrixWriter &operator=(const F32MatrixWriter &) = delete;

  [[nodiscard]] Status open(const std::string &path, std::size_t rows,
                            std::size_t columns);
  [[nodiscard]] Status append(const float *values, std::size_t elements);
  [[nodiscard]] Status close();

private:
  void discard() noexcept;

  std::ofstream output_;
  std::string path_;
  std::size_t expected_elements_{0};
  std::size_t written_elements_{0};
  bool completed_{false};
};

// Writes a nonempty row-major, little-endian F32 matrix in NumPy's NPY format.
[[nodiscard]] Status write_f32(const std::string &path,
                               const std::vector<float> &values,
                               std::size_t rows, std::size_t columns);

} // namespace evo::npy
