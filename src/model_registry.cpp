// SPDX-License-Identifier: Apache-2.0
#include "evo/model_registry.hpp"

#include <algorithm>
#include <utility>

namespace evo {
namespace {

OfficialModelSpec spec(
    const std::string_view id, const std::size_t width,
    const std::size_t layers, const std::size_t heads,
    const std::size_t inner, const std::size_t hcs_groups,
    const std::size_t hcm_groups, const std::size_t hcl_groups,
    const std::size_t max_seqlen, const double rope_base,
    const double rope_scale, const bool interpolated,
    const OfficialProjectionPrecision precision,
    const OfficialProjectionWeightDType weight_dtype,
    const OfficialHcmFilterDType hcm_dtype, std::vector<std::size_t> hcs,
    std::vector<std::size_t> hcm, std::vector<std::size_t> hcl,
    std::vector<std::size_t> attention) {
  return {id, 512, width, layers, heads, inner, hcs_groups, hcm_groups,
          hcl_groups, max_seqlen, rope_base, rope_scale, interpolated,
          precision, weight_dtype, hcm_dtype, std::move(hcs), std::move(hcm),
          std::move(hcl), std::move(attention)};
}

} // namespace

const std::vector<OfficialModelSpec> &official_model_specs() {
  using H = OfficialHcmFilterDType;
  using P = OfficialProjectionPrecision;
  using W = OfficialProjectionWeightDType;
  static const std::vector<OfficialModelSpec> specs{
      spec("evo2_1b_base", 1920, 25, 15, 5120, 128, 128, 1920, 8192,
           10000.0, 1.0, false, P::kE4M3Software, W::kBF16, H::kBF16,
           {0, 4, 7, 11, 14, 18, 21}, {1, 5, 8, 12, 15, 19, 22},
           {2, 6, 9, 13, 16, 20, 23}, {3, 10, 17, 24}),
      spec("evo2_7b", 4096, 32, 32, 11264, 256, 256, 4096, 1048576,
           10000.0, 128.0, true, P::kBF16, W::kBF16, H::kBF16,
           {0, 4, 7, 11, 14, 18, 21, 25, 28},
           {1, 5, 8, 12, 15, 19, 22, 26, 29},
           {2, 6, 9, 13, 16, 20, 23, 27, 30}, {3, 10, 17, 24, 31}),
      spec("evo2_7b_base", 4096, 32, 32, 11008, 256, 256, 4096, 32768,
           10000.0, 1.0, false, P::kBF16, W::kBF16, H::kBF16,
           {0, 4, 7, 11, 14, 18, 21, 25, 28},
           {1, 5, 8, 12, 15, 19, 22, 26, 29},
           {2, 6, 9, 13, 16, 20, 23, 27, 30}, {3, 10, 17, 24, 31}),
      spec("evo2_7b_262k", 4096, 32, 32, 11008, 256, 256, 4096, 262144,
           10000.0, 32.0, true, P::kBF16, W::kBF16, H::kBF16,
           {0, 4, 7, 11, 14, 18, 21, 25, 28},
           {1, 5, 8, 12, 15, 19, 22, 26, 29},
           {2, 6, 9, 13, 16, 20, 23, 27, 30}, {3, 10, 17, 24, 31}),
      spec("evo2_20b", 8192, 24, 64, 22528, 512, 512, 8192, 1048576,
           1000000.0, 128.0, true, P::kE4M3Software, W::kF32, H::kBF16,
           {0, 4, 7, 11, 14, 18, 21}, {1, 5, 8, 12, 15, 19, 22},
           {2, 6, 9, 13, 16, 20, 23}, {3, 10, 17}),
      spec("evo2_40b", 8192, 50, 64, 22528, 512, 512, 8192, 1048576,
           1000000.0, 128.0, true, P::kE4M3Software, W::kBF16, H::kBF16,
           {0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46},
           {1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47},
           {2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48},
           {3, 10, 17, 24, 31, 35, 42, 49}),
      spec("evo2_40b_base", 8192, 50, 64, 21888, 512, 512, 8192, 8192,
           1000000.0, 1.0, false, P::kE4M3Software, W::kBF16, H::kBF16,
           {0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46},
           {1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47},
           {2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48},
           {3, 10, 17, 24, 31, 35, 42, 49}),
      spec("evo2_40b_bionemo_bf16", 8192, 50, 64, 22528, 512, 512, 8192,
           1048576, 1000000.0, 128.0, true, P::kBF16, W::kBF16, H::kF32,
           {0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46},
           {1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47},
           {2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48},
           {3, 10, 17, 24, 31, 35, 42, 49})};
  return specs;
}

const OfficialModelSpec *
find_official_model(const std::string_view model_id) noexcept {
  const auto &specs = official_model_specs();
  const auto found =
      std::find_if(specs.begin(), specs.end(), [model_id](const auto &item) {
        return item.id == model_id;
      });
  return found == specs.end() ? nullptr : &*found;
}

} // namespace evo
