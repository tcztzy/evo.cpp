// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

#include "evo2c/model_registry.hpp"
#include "evo2c/status.hpp"
#include "evo2c/version.hpp"

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
  const auto ok = evo2c::Status::Ok();
  check(ok.ok(), "Status::Ok is successful");
  check(static_cast<bool>(ok), "Status boolean conversion preserves success");
  check(ok.message().empty(), "successful status has no error text");

  const evo2c::Status invalid{evo2c::ErrorCode::kInvalidArgument, "missing model"};
  check(!invalid.ok(), "error status is not successful");
  check(invalid.message() == "missing model", "error status preserves actionable text");
  check(std::string_view{evo2c::error_code_name(invalid.code())} == "invalid_argument",
        "error status has stable machine-readable name");
  check(evo2c::exit_code(invalid.code()) != 0, "error maps to nonzero exit status");

  check(evo2c::version() == "0.1.0", "runtime version matches project version");
  check(evo2c::kVersionMajor == 0 && evo2c::kVersionMinor == 1 && evo2c::kVersionPatch == 0,
        "version components match runtime version");

  const auto &models = evo2c::official_model_specs();
  check(models.size() == 8, "registry contains every supported official variant");
  for (const auto &model : models) {
    check(evo2c::find_official_model(model.id) == &model,
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
  }
  check(evo2c::find_official_model("evo2_20b")->projection_weight_dtype ==
            evo2c::OfficialProjectionWeightDType::kF32,
        "20B preserves checkpoint-native F32 projection weights");
  check(evo2c::find_official_model("evo2_7b")->projection_precision ==
            evo2c::OfficialProjectionPrecision::kBF16,
        "7B retains BF16 projection semantics");
  check(evo2c::find_official_model("unknown") == nullptr,
        "unknown model IDs fail registry lookup");

  if (failures != 0) {
    std::cerr << failures << " core test(s) failed\n";
    return 1;
  }
  std::cout << "core tests passed\n";
  return 0;
}
