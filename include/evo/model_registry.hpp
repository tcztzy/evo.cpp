// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "evo/status.hpp"

namespace evo {

enum class ArchitectureTokenizer {
  kByteIdentity,
  kHyenaDnaCharacter,
  kEsmcProtein,
  // The model artifact must carry a verified evo-tokenizer-v1 descriptor.
  kArtifact,
};

// The native implementation selected by an artifact architecture.  This is
// deliberately independent from tokenizer and capability metadata: two
// architectures may share a tokenizer without sharing an execution graph.
enum class ArchitectureImplementation {
  kUnknown,
  kStripedHyena2,
  kHyenaDna,
  kEsmc,
  kGenebTransformerDecoder,
  kGenebOlmoDecoder,
  kGenebEsmEncoder,
  kGenebBertEncoder,
  kGenebGpt2Decoder,
  kGenebDnaGptDecoder,
  kGenebCustomEncoder,
  kGenebMambaEncoder,
  kGenebHyenaDnaDecoder,
  kGenebStripedHyenaV1,
  kGenebJanusDnaEncoder,
  kGenebSequenceCnnEncoder,
  kGenebRoformerEncoder,
};

enum ArchitectureBackend : unsigned {
  kArchitectureBackendCpu = 1U << 0U,
  kArchitectureBackendCuda = 1U << 1U,
  kArchitectureBackendMps = 1U << 2U,
};

enum ArchitectureCapability : unsigned {
  kArchitectureGenerate = 1U << 0U,
  kArchitectureScore = 1U << 1U,
  kArchitectureEmbed = 1U << 2U,
  kArchitectureVariant = 1U << 3U,
  kArchitectureServe = 1U << 4U,
  kArchitectureLogits = 1U << 5U,
};

struct ArchitectureSpec final {
  std::string_view id;
  std::string_view artifact_profile;
  std::string_view runtime_abi;
  ArchitectureImplementation implementation;
  ArchitectureTokenizer tokenizer;
  unsigned backends;
  unsigned capabilities;
  bool synthetic_fixture;
};

struct ArchitectureBackendFactorySpec final {
  ArchitectureImplementation implementation;
  ArchitectureBackend backend;
};

enum class OfficialProjectionPrecision { kBF16, kE4M3Software };
enum class OfficialProjectionWeightDType { kBF16, kF32 };
enum class OfficialHcmFilterDType { kBF16, kF32 };
enum class OfficialExactSupport { kValidated, kUnsupported };

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
  OfficialExactSupport exact_support;
  std::string_view exact_evidence;
  std::vector<std::size_t> hcs;
  std::vector<std::size_t> hcm;
  std::vector<std::size_t> hcl;
  std::vector<std::size_t> attention;
};

struct OfficialEsmcModelSpec final {
  std::string_view id;
  std::string_view huggingface_repo;
  std::string_view huggingface_revision;
  std::string_view hosted_alias;
  std::size_t vocab_size;
  std::size_t hidden_size;
  std::size_t layers;
  std::size_t heads;
  std::size_t inner_width;
  std::size_t max_seqlen;
  OfficialExactSupport exact_support;
  std::string_view exact_evidence;
};

[[nodiscard]] const std::vector<OfficialModelSpec> &official_model_specs();
[[nodiscard]] const OfficialModelSpec *
find_official_model(std::string_view model_id) noexcept;
[[nodiscard]] const char *
official_exact_support_name(OfficialExactSupport support) noexcept;
[[nodiscard]] Status require_official_exact_support(std::string_view model_id);
[[nodiscard]] const std::vector<OfficialEsmcModelSpec> &
official_esmc_model_specs();
[[nodiscard]] const OfficialEsmcModelSpec *
find_official_esmc_model(std::string_view model_id) noexcept;
[[nodiscard]] const std::vector<ArchitectureSpec> &architecture_specs();
[[nodiscard]] const ArchitectureSpec *
find_architecture(std::string_view architecture) noexcept;
[[nodiscard]] const ArchitectureSpec *
find_artifact_profile(std::string_view profile) noexcept;
[[nodiscard]] bool architecture_requires_artifact_tokenizer(
    const ArchitectureSpec &architecture) noexcept;
[[nodiscard]] const char *architecture_implementation_name(
    ArchitectureImplementation implementation) noexcept;
[[nodiscard]] const ArchitectureBackendFactorySpec *
find_architecture_backend_factory(const ArchitectureSpec &architecture,
                                  ArchitectureBackend backend) noexcept;

} // namespace evo
