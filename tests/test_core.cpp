// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "evo/model_registry.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"
#include "evo/version.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

}  // namespace

int main() {
  const auto ok = evo::Status::Ok();
  check(ok.ok(), "Status::Ok is successful");
  check(static_cast<bool>(ok), "Status boolean conversion preserves success");
  check(ok.message().empty(), "successful status has no error text");

  const evo::Status invalid{evo::ErrorCode::kInvalidArgument, "missing model"};
  check(!invalid.ok(), "error status is not successful");
  check(invalid.message() == "missing model", "error status preserves actionable text");
  check(std::string_view{evo::error_code_name(invalid.code())} == "invalid_argument",
        "error status has stable machine-readable name");
  check(evo::exit_code(invalid.code()) != 0, "error maps to nonzero exit status");

  check(evo::version() == "0.1.0", "runtime version matches project version");
  check(evo::kVersionMajor == 0 && evo::kVersionMinor == 1 && evo::kVersionPatch == 0,
        "version components match runtime version");

  const auto &models = evo::official_model_specs();
  check(models.size() == 8, "registry contains every supported official variant");
  for (const auto &model : models) {
    check(evo::find_official_model(model.id) == &model,
          "registry lookup returns stable model record");
    std::vector<std::size_t> layers = model.hcs;
    layers.insert(layers.end(), model.hcm.begin(), model.hcm.end());
    layers.insert(layers.end(), model.hcl.begin(), model.hcl.end());
    layers.insert(layers.end(), model.attention.begin(), model.attention.end());
    std::sort(layers.begin(), layers.end());
    std::vector<std::size_t> expected(model.layers);
    std::iota(expected.begin(), expected.end(), 0);
    check(layers == expected, "official stripe layers are disjoint and complete");
    check(model.hidden_size / model.heads == 128,
          "official registry fixes the attention head dimension");
    if (model.exact_support == evo::OfficialExactSupport::kValidated) {
      check(!model.exact_evidence.empty(),
            "validated exact model IDs carry a pinned evidence record");
      check(evo::require_official_exact_support(model.id).ok(),
            "validated exact model IDs pass the runtime support gate");
    } else {
      const auto unsupported = evo::require_official_exact_support(model.id);
      check(model.exact_evidence.empty() && !unsupported.ok() &&
                unsupported.code() == evo::ErrorCode::kUnsupported &&
                unsupported.message().find(std::string{model.id}) !=
                    std::string::npos,
            "ungated model IDs fail exact execution with a typed diagnostic");
    }
  }
  check(evo::find_official_model("evo2_20b")->projection_weight_dtype ==
            evo::OfficialProjectionWeightDType::kF32,
        "20B preserves checkpoint-native F32 projection weights");
  check(evo::find_official_model("evo2_7b")->projection_precision ==
            evo::OfficialProjectionPrecision::kBF16,
        "7B retains BF16 projection semantics");
  check(evo::find_official_model("unknown") == nullptr,
        "unknown model IDs fail registry lookup");
  const auto unknown_exact = evo::require_official_exact_support("unknown");
  check(!unknown_exact.ok() &&
            unknown_exact.code() == evo::ErrorCode::kUnsupported,
        "unknown exact model IDs fail closed");

  const auto &esmc_models = evo::official_esmc_model_specs();
  check(esmc_models.size() == 3,
        "registry contains all three canonical ESMC sizes");
  std::vector<evo::TokenId> protein;
  const auto protein_status = evo::encode_sequence(
      evo::ArchitectureTokenizer::kEsmcProtein,
      "LAGV<mask>|Z?", &protein);
  check(protein_status.ok() &&
            protein == std::vector<evo::TokenId>{0, 4, 5, 6, 7, 32, 31, 27,
                                                3, 2},
        "ESMC tokenizer matches the pinned special-token and BPE IDs");
  std::vector<evo::TokenId> explicit_specials;
  check(evo::encode_sequence(evo::ArchitectureTokenizer::kEsmcProtein,
                             "<cls><pad><eos><unk>",
                             &explicit_specials)
                .ok() &&
            explicit_specials ==
                std::vector<evo::TokenId>{0, 0, 1, 2, 3, 2},
        "ESMC tokenizer preserves explicit registered special tokens");
  std::uint8_t residue = 0;
  check(evo::decode_sequence_token(evo::ArchitectureTokenizer::kEsmcProtein,
                                   23, &residue)
                .ok() &&
            residue == static_cast<std::uint8_t>('C'),
        "ESMC residue token decodes through the canonical vocabulary");

  if (failures != 0) {
    std::cerr << failures << " core test(s) failed\n";
    return 1;
  }
  std::cout << "core tests passed\n";
  return 0;
}
