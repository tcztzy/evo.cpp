// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "evo/model_format.hpp"
#include "evo/status.hpp"

namespace evo::detail {

struct LinearTensorView final {
  const std::uint8_t *data{nullptr};
  TensorDType dtype{TensorDType::kF32};
  std::size_t elements{0};
};

class LinearExecutor {
public:
  virtual ~LinearExecutor() = default;

  [[nodiscard]] virtual const char *name() const noexcept = 0;
  [[nodiscard]] virtual Status
  linear(const float *input, std::size_t rows, std::size_t input_width,
         LinearTensorView weight, std::size_t output_width,
         const LinearTensorView *bias, std::vector<float> *output) = 0;
};

} // namespace evo::detail
