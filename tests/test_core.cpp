// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <string_view>

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

  if (failures != 0) {
    std::cerr << failures << " core test(s) failed\n";
    return 1;
  }
  std::cout << "core tests passed\n";
  return 0;
}

