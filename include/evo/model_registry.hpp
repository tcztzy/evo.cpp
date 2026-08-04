// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace evo {

enum class OfficialProjectionPrecision { kBF16, kE4M3Software };
enum class OfficialProjectionWeightDType { kBF16, kF32 };
enum class OfficialHcmFilterDType { kBF16, kF32 };

struct OfficialModelSpec final {
  std::string_view id;
  std::size_t vocab_size;
  std::size_t hidden_size;
  std::size_t layers;
  std::size_t heads;
  std::size_t inner_width;
  std::size_t hcs_groups;
  std::size_t hcm_groups;
  std::size_t hcl_groups;
  std::size_t max_seqlen;
  double rope_base;
  double rope_scale;
  bool interpolated_rope;
  OfficialProjectionPrecision projection_precision;
  OfficialProjectionWeightDType projection_weight_dtype;
  OfficialHcmFilterDType hcm_filter_dtype;
  std::vector<std::size_t> hcs;
  std::vector<std::size_t> hcm;
  std::vector<std::size_t> hcl;
  std::vector<std::size_t> attention;
};

[[nodiscard]] const std::vector<OfficialModelSpec> &official_model_specs();
[[nodiscard]] const OfficialModelSpec *
find_official_model(std::string_view model_id) noexcept;

} // namespace evo
