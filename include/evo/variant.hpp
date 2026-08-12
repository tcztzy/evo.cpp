// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "evo/status.hpp"

namespace evo {

enum class VariantStrand { kForward, kReverse, kBoth };
enum class VariantNormalization { kSum, kMean };

struct VariantWindow final {
  std::string reference;
  std::string alternate;
  std::size_t reference_start{0};
  std::size_t reference_end{0};
  std::size_t variant_offset{0};
};

[[nodiscard]] Status validate_iupac_dna(std::string_view sequence,
                                        std::string_view label);
[[nodiscard]] Status reverse_complement(std::string_view sequence,
                                        std::string *output);
[[nodiscard]] Status
make_variant_window(std::string_view sequence, std::size_t position_1based,
                    std::string_view reference, std::string_view alternate,
                    std::size_t maximum_tokens, VariantWindow *output);
[[nodiscard]] const char *variant_strand_name(VariantStrand strand) noexcept;
[[nodiscard]] const char *
variant_normalization_name(VariantNormalization normalization) noexcept;

} // namespace evo
