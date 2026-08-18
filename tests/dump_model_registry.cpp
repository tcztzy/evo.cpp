// SPDX-License-Identifier: Apache-2.0
#include <iomanip>
#include <iostream>
#include <vector>

#include "evo/model_registry.hpp"

namespace {

void indices(const std::vector<std::size_t> &values) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    std::cout << values[index];
  }
}

} // namespace

int main() {
  const auto *const striped = evo::find_architecture("StripedHyena2");
  const auto *const esmc = evo::find_architecture("ESMC");
  if (striped == nullptr || esmc == nullptr)
    return 2;
  const evo::ArchitectureSpec shared_tokenizer_probe{
      "SharedTokenizerProbe", "probe-runtime-v1", "probe-abi-v1",
      evo::ArchitectureImplementation::kEsmc, striped->tokenizer,
      evo::kArchitectureBackendCpu, evo::kArchitectureEmbed, true};
  const auto *const shared_tokenizer_factory =
      evo::find_architecture_backend_factory(
          shared_tokenizer_probe, evo::kArchitectureBackendCpu);
  if (shared_tokenizer_probe.tokenizer != striped->tokenizer ||
      shared_tokenizer_probe.implementation == striped->implementation ||
      shared_tokenizer_factory == nullptr ||
      shared_tokenizer_factory->implementation !=
          evo::ArchitectureImplementation::kEsmc) {
    return 3;
  }
  auto unknown_probe = shared_tokenizer_probe;
  unknown_probe.implementation = evo::ArchitectureImplementation::kUnknown;
  if (evo::find_architecture("UnregisteredArchitecture") != nullptr ||
      evo::find_architecture_backend_factory(
          unknown_probe, evo::kArchitectureBackendCpu) != nullptr) {
    return 4;
  }

  std::cout << std::setprecision(17);
  for (const auto &model : evo::official_model_specs()) {
    std::cout
        << model.id << '|' << model.vocab_size << '|' << model.hidden_size
        << '|' << model.layers << '|' << model.heads << '|'
        << model.inner_width << '|' << model.hcs_groups << '|'
        << model.hcm_groups << '|' << model.hcl_groups << '|'
        << model.max_seqlen << '|' << model.rope_base << '|'
        << model.rope_scale << '|' << (model.interpolated_rope ? 1 : 0) << '|'
        << (model.projection_precision ==
                    evo::OfficialProjectionPrecision::kBF16
                ? "BF16"
                : "E4M3_SW")
        << '|'
        << (model.projection_weight_dtype ==
                    evo::OfficialProjectionWeightDType::kBF16
                ? "BF16"
                : "F32")
        << '|'
        << (model.hcm_filter_dtype ==
                    evo::OfficialHcmFilterDType::kBF16
                ? "BF16"
                : "F32")
        << '|';
    indices(model.hcs);
    std::cout << '|';
    indices(model.hcm);
    std::cout << '|';
    indices(model.hcl);
    std::cout << '|';
    indices(model.attention);
    std::cout << '|'
              << evo::official_exact_support_name(model.exact_support) << '|'
              << model.exact_evidence << '\n';
  }
  for (const auto &model : evo::official_esmc_model_specs()) {
    std::cout << "$|" << model.id << '|' << model.huggingface_repo << '|'
              << model.huggingface_revision << '|' << model.hosted_alias << '|'
              << model.vocab_size << '|' << model.hidden_size << '|'
              << model.layers << '|' << model.heads << '|' << model.inner_width
              << '|' << model.max_seqlen << '|'
              << evo::official_exact_support_name(model.exact_support) << '|'
              << model.exact_evidence << '\n';
  }
  for (const auto &architecture : evo::architecture_specs()) {
    const char *tokenizer = "esmc-protein";
    if (architecture.tokenizer == evo::ArchitectureTokenizer::kByteIdentity)
      tokenizer = "byte";
    else if (architecture.tokenizer ==
             evo::ArchitectureTokenizer::kHyenaDnaCharacter)
      tokenizer = "hyenadna-character";
    else if (architecture.tokenizer == evo::ArchitectureTokenizer::kArtifact)
      tokenizer = "artifact";
    std::cout << "@|" << architecture.id << '|'
              << architecture.artifact_profile << '|'
              << architecture.runtime_abi << '|'
              << evo::architecture_implementation_name(
                     architecture.implementation)
              << '|' << tokenizer << '|'
              << architecture.backends << '|' << architecture.capabilities
              << '|' << (architecture.synthetic_fixture ? 1 : 0) << '\n';

    unsigned factory_backends = 0;
    for (const evo::ArchitectureBackend backend : {
             evo::kArchitectureBackendCpu, evo::kArchitectureBackendCuda,
             evo::kArchitectureBackendMps}) {
      if (evo::find_architecture_backend_factory(architecture, backend) !=
          nullptr) {
        factory_backends |= backend;
      }
    }
    std::cout << "&|" << architecture.id << '|' << factory_backends << '\n';
  }
}
