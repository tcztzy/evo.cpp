// SPDX-License-Identifier: Apache-2.0
#include <exception>
#include <iostream>
#include <string_view>

#include "evo2c/status.hpp"
#include "evo2c/version.hpp"

namespace {

void print_help() {
  std::cout << "evo2c-inspect - inspect an EVO2C model without CUDA\n\n"
            << "Usage:\n"
            << "  evo2c-inspect --help\n"
            << "  evo2c-inspect --version\n"
            << "  evo2c-inspect MODEL [--tensor NAME]\n";
}

int fail(const evo2c::Status& status) {
  std::cerr << "evo2c-inspect: " << evo2c::error_code_name(status.code()) << ": "
            << status.message() << '\n';
  return evo2c::exit_code(status.code());
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    if (argc == 2) {
      const std::string_view arg{argv[1]};
      if (arg == "--help" || arg == "-h") {
        print_help();
        return 0;
      }
      if (arg == "--version") {
        std::cout << "evo2c-inspect " << evo2c::version() << '\n';
        return 0;
      }
    }

    return fail({evo2c::ErrorCode::kUnsupported,
                 "model inspection is not implemented yet; complete SPEC task T2"});
  } catch (const std::exception& error) {
    return fail({evo2c::ErrorCode::kInternal, error.what()});
  } catch (...) {
    return fail({evo2c::ErrorCode::kInternal, "unknown exception"});
  }
}

