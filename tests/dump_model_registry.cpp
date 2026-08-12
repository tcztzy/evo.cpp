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
    std::cout << '\n';
  }
  for (const auto &architecture : evo::architecture_specs()) {
    const char *tokenizer =
        architecture.tokenizer == evo::ArchitectureTokenizer::kByteIdentity
            ? "byte"
            : "hyenadna-character";
    std::cout << "@|" << architecture.id << '|'
              << architecture.artifact_profile << '|'
              << architecture.runtime_abi << '|' << tokenizer << '|'
              << architecture.backends << '|' << architecture.capabilities
              << '|' << (architecture.synthetic_fixture ? 1 : 0) << '\n';
  }
}
