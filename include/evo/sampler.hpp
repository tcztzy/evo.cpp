// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo {

struct SamplingConfig final {
  float temperature{1.0F};
  std::size_t top_k{1};
  float top_p{1.0F};
  std::uint64_t seed{0};
};

[[nodiscard]] Status validate_sampling_config(const SamplingConfig& config);

class Sampler final {
 public:
  explicit Sampler(SamplingConfig config);

  [[nodiscard]] Status sample(const std::vector<float>& logits, TokenId* token);

 private:
  SamplingConfig config_;
  std::mt19937_64 random_;
};

}  // namespace evo
