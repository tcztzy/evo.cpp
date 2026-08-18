// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "evo/status.hpp"

namespace evo::cpu::detail {

// Deterministic counters used by the long-context complexity gate.  They are
// deliberately independent of wall-clock timing and count only the FFT work
// and peak complex scratch owned by one convolution call.
struct FftWorkStats final {
  std::size_t signal_length{0};
  std::size_t transform_length{0};
  std::size_t radix2_length{0};
  std::uint64_t butterfly_count{0};
  std::size_t peak_complex_values{0};
};

// Computes the first input.size() rows of the linear causal convolution.
// The mathematical transform length is exactly 2*L, matching the pinned
// PyTorch implementations.  A radix-2 FFT is used directly when possible;
// other lengths use Bluestein and remain O(L log L), never direct O(L^2).
[[nodiscard]] Status
causal_fft_convolution(const std::vector<float> &input,
                       const std::vector<float> &kernel,
                       std::vector<float> *output,
                       FftWorkStats *stats = nullptr);

} // namespace evo::cpu::detail
