// SPDX-License-Identifier: Apache-2.0
#include "evo/model_registry.hpp"

#include <algorithm>
#include <string>
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
    const OfficialHcmFilterDType hcm_dtype,
    const OfficialExactSupport exact_support,
    const std::string_view exact_evidence, std::vector<std::size_t> hcs,
    std::vector<std::size_t> hcm, std::vector<std::size_t> hcl,
    std::vector<std::size_t> attention) {
  return {id, 512, width, layers, heads, inner, hcs_groups, hcm_groups,
          hcl_groups, max_seqlen, rope_base, rope_scale, interpolated,
          precision, weight_dtype, hcm_dtype, exact_support, exact_evidence,
          std::move(hcs), std::move(hcm), std::move(hcl),
          std::move(attention)};
}

} // namespace

const std::vector<OfficialModelSpec> &official_model_specs() {
  using H = OfficialHcmFilterDType;
  using P = OfficialProjectionPrecision;
  using S = OfficialExactSupport;
  using W = OfficialProjectionWeightDType;
  static const std::vector<OfficialModelSpec> specs{
      spec("evo2_1b_base", 1920, 25, 15, 5120, 128, 128, 1920, 8192,
           10000.0, 1.0, false, P::kE4M3Software, W::kBF16, H::kBF16,
           S::kValidated, "model-size-validation/2026-08-04/evo2_1b_base",
           {0, 4, 7, 11, 14, 18, 21}, {1, 5, 8, 12, 15, 19, 22},
           {2, 6, 9, 13, 16, 20, 23}, {3, 10, 17, 24}),
      spec("evo2_7b", 4096, 32, 32, 11264, 256, 256, 4096, 1048576,
           10000.0, 128.0, true, P::kBF16, W::kBF16, H::kBF16,
           S::kValidated, "model-size-validation/2026-08-04/evo2_7b",
           {0, 4, 7, 11, 14, 18, 21, 25, 28},
           {1, 5, 8, 12, 15, 19, 22, 26, 29},
           {2, 6, 9, 13, 16, 20, 23, 27, 30}, {3, 10, 17, 24, 31}),
      spec("evo2_7b_base", 4096, 32, 32, 11008, 256, 256, 4096, 32768,
           10000.0, 1.0, false, P::kBF16, W::kBF16, H::kBF16,
           S::kUnsupported, {},
           {0, 4, 7, 11, 14, 18, 21, 25, 28},
           {1, 5, 8, 12, 15, 19, 22, 26, 29},
           {2, 6, 9, 13, 16, 20, 23, 27, 30}, {3, 10, 17, 24, 31}),
      spec("evo2_7b_262k", 4096, 32, 32, 11008, 256, 256, 4096, 262144,
           10000.0, 32.0, true, P::kBF16, W::kBF16, H::kBF16,
           S::kUnsupported, {},
           {0, 4, 7, 11, 14, 18, 21, 25, 28},
           {1, 5, 8, 12, 15, 19, 22, 26, 29},
           {2, 6, 9, 13, 16, 20, 23, 27, 30}, {3, 10, 17, 24, 31}),
      spec("evo2_20b", 8192, 24, 64, 22528, 512, 512, 8192, 1048576,
           1000000.0, 128.0, true, P::kE4M3Software, W::kF32, H::kBF16,
           S::kValidated, "model-size-validation/2026-08-04/evo2_20b",
           {0, 4, 7, 11, 14, 18, 21}, {1, 5, 8, 12, 15, 19, 22},
           {2, 6, 9, 13, 16, 20, 23}, {3, 10, 17}),
      spec("evo2_40b", 8192, 50, 64, 22528, 512, 512, 8192, 1048576,
           1000000.0, 128.0, true, P::kE4M3Software, W::kBF16, H::kBF16,
           S::kValidated, "model-size-validation/2026-08-04/evo2_40b",
           {0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46},
           {1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47},
           {2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48},
           {3, 10, 17, 24, 31, 35, 42, 49}),
      spec("evo2_40b_base", 8192, 50, 64, 21888, 512, 512, 8192, 8192,
           1000000.0, 1.0, false, P::kE4M3Software, W::kBF16, H::kBF16,
           S::kUnsupported, {},
           {0, 4, 7, 11, 14, 18, 21, 25, 28, 32, 36, 39, 43, 46},
           {1, 5, 8, 12, 15, 19, 22, 26, 29, 33, 37, 40, 44, 47},
           {2, 6, 9, 13, 16, 20, 23, 27, 30, 34, 38, 41, 45, 48},
           {3, 10, 17, 24, 31, 35, 42, 49}),
      spec("evo2_40b_bionemo_bf16", 8192, 50, 64, 22528, 512, 512, 8192,
           1048576, 1000000.0, 128.0, true, P::kBF16, W::kBF16, H::kF32,
           S::kUnsupported, {},
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

const char *
official_exact_support_name(const OfficialExactSupport support) noexcept {
  switch (support) {
  case OfficialExactSupport::kValidated:
    return "validated";
  case OfficialExactSupport::kUnsupported:
    return "unsupported";
  }
  return "unsupported";
}

Status require_official_exact_support(const std::string_view model_id) {
  const auto *const model = find_official_model(model_id);
  if (model == nullptr)
    return {ErrorCode::kUnsupported,
            "exact execution is unsupported for unknown model.id '" +
                std::string{model_id} + "'"};
  if (model->exact_support != OfficialExactSupport::kValidated) {
    return {ErrorCode::kUnsupported,
            "exact execution is unsupported for model.id '" +
                std::string{model_id} +
                "': no pinned real-checkpoint raw-bit gate exists; use an "
                "explicit approximate profile/backend or contribute an "
                "independent exact evidence record"};
  }
  return Status::Ok();
}

const std::vector<OfficialEsmcModelSpec> &official_esmc_model_specs() {
  static const std::vector<OfficialEsmcModelSpec> specs{
      {"esmc_300m", "biohub/ESMC-300M",
       "a59b831785f907e96e6a246b1d142bfb76df31ee", "esmc-300m-2024-12",
       64, 960, 30, 15, 2560, 2048, OfficialExactSupport::kValidated,
       "esmc-official-oracle/2026-08-12/esmc_300m"},
      {"esmc_600m", "biohub/ESMC-600M",
       "a7e82012c83126b9eedb055fea9fa84b6c02f094", "esmc-600m-2024-12",
       64, 1152, 36, 18, 3072, 2048, OfficialExactSupport::kValidated,
       "esmc-official-oracle/2026-08-12/esmc_600m"},
      {"esmc_6b", "biohub/ESMC-6B",
       "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a", "esmc-6b-2024-12",
       64, 2560, 80, 40, 6912, 2048, OfficialExactSupport::kValidated,
       "esmc-official-oracle/2026-08-12/esmc_6b"},
  };
  return specs;
}

const OfficialEsmcModelSpec *
find_official_esmc_model(const std::string_view model_id) noexcept {
  const auto &specs = official_esmc_model_specs();
  const auto found =
      std::find_if(specs.begin(), specs.end(), [model_id](const auto &item) {
        return item.id == model_id;
      });
  return found == specs.end() ? nullptr : &*found;
}

const std::vector<ArchitectureSpec> &architecture_specs() {
  constexpr unsigned all_capabilities =
      kArchitectureGenerate | kArchitectureScore | kArchitectureEmbed |
      kArchitectureVariant | kArchitectureServe | kArchitectureLogits;
  static const std::vector<ArchitectureSpec> specs{
      {"StripedHyena2", "evo2-runtime-v1", "evo2-safetensors-v1",
       ArchitectureImplementation::kStripedHyena2,
       ArchitectureTokenizer::kByteIdentity,
       kArchitectureBackendCpu | kArchitectureBackendCuda |
           kArchitectureBackendMps,
       all_capabilities,
       false},
      {"StripedHyena2Test", "evo2-runtime-v1", "evo2-safetensors-v1",
       ArchitectureImplementation::kStripedHyena2,
       ArchitectureTokenizer::kByteIdentity,
       kArchitectureBackendCpu | kArchitectureBackendCuda |
           kArchitectureBackendMps,
       all_capabilities,
       true},
      {"HyenaDNA", "hyenadna-runtime-v1", "hyenadna-safetensors-v1",
       ArchitectureImplementation::kHyenaDna,
       ArchitectureTokenizer::kHyenaDnaCharacter,
       kArchitectureBackendCpu | kArchitectureBackendMps,
       all_capabilities, false},
      {"HyenaDNATest", "hyenadna-runtime-v1", "hyenadna-safetensors-v1",
       ArchitectureImplementation::kHyenaDna,
       ArchitectureTokenizer::kHyenaDnaCharacter,
       kArchitectureBackendCpu | kArchitectureBackendMps,
       all_capabilities, true},
      {"ESMC", "esmc-runtime-v1", "esmc-safetensors-v1",
       ArchitectureImplementation::kEsmc,
       ArchitectureTokenizer::kEsmcProtein,
       kArchitectureBackendCpu | kArchitectureBackendCuda |
           kArchitectureBackendMps,
       kArchitectureEmbed | kArchitectureLogits, false},
      {"ESMCTest", "esmc-runtime-v1", "esmc-safetensors-v1",
       ArchitectureImplementation::kEsmc,
       ArchitectureTokenizer::kEsmcProtein,
       kArchitectureBackendCpu | kArchitectureBackendCuda |
           kArchitectureBackendMps,
       kArchitectureEmbed | kArchitectureLogits, true},
  };
  return specs;
}

const ArchitectureSpec *
find_architecture(const std::string_view architecture) noexcept {
  const auto &specs = architecture_specs();
  const auto found = std::find_if(
      specs.begin(), specs.end(), [architecture](const auto &item) {
        return item.id == architecture;
      });
  return found == specs.end() ? nullptr : &*found;
}

const ArchitectureSpec *
find_artifact_profile(const std::string_view profile) noexcept {
  const auto &specs = architecture_specs();
  const auto found =
      std::find_if(specs.begin(), specs.end(), [profile](const auto &item) {
        return item.artifact_profile == profile;
      });
  return found == specs.end() ? nullptr : &*found;
}

const char *architecture_implementation_name(
    const ArchitectureImplementation implementation) noexcept {
  switch (implementation) {
  case ArchitectureImplementation::kUnknown:
    return "unknown";
  case ArchitectureImplementation::kStripedHyena2:
    return "striped-hyena-2";
  case ArchitectureImplementation::kHyenaDna:
    return "hyenadna";
  case ArchitectureImplementation::kEsmc:
    return "esmc";
  }
  return "unknown";
}

const ArchitectureBackendFactorySpec *find_architecture_backend_factory(
    const ArchitectureSpec &architecture,
    const ArchitectureBackend backend) noexcept {
  static const std::vector<ArchitectureBackendFactorySpec> factories{
      {ArchitectureImplementation::kStripedHyena2,
       kArchitectureBackendCpu},
      {ArchitectureImplementation::kStripedHyena2,
       kArchitectureBackendCuda},
      {ArchitectureImplementation::kStripedHyena2,
       kArchitectureBackendMps},
      {ArchitectureImplementation::kHyenaDna, kArchitectureBackendCpu},
      {ArchitectureImplementation::kHyenaDna, kArchitectureBackendMps},
      {ArchitectureImplementation::kEsmc, kArchitectureBackendCpu},
      {ArchitectureImplementation::kEsmc, kArchitectureBackendCuda},
      {ArchitectureImplementation::kEsmc, kArchitectureBackendMps},
  };
  const auto backend_mask = static_cast<unsigned>(backend);
  if ((backend != kArchitectureBackendCpu &&
       backend != kArchitectureBackendCuda &&
       backend != kArchitectureBackendMps) ||
      (architecture.backends & backend_mask) == 0U) {
    return nullptr;
  }
  const auto found =
      std::find_if(factories.begin(), factories.end(),
                   [&architecture, backend](const auto &factory) {
                     return factory.implementation ==
                                architecture.implementation &&
                            factory.backend == backend;
                   });
  return found == factories.end() ? nullptr : &*found;
}

} // namespace evo
