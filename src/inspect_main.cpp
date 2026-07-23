// SPDX-License-Identifier: Apache-2.0
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

#include "evo2c/model_format.hpp"
#include "evo2c/status.hpp"
#include "evo2c/version.hpp"

namespace {

void print_help() {
  std::cout << "evo2c-inspect - inspect and verify an EVO2C model without CUDA\n\n"
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

void print_tensor(const evo2c::TensorInfo& tensor) {
  std::cout << "tensor " << tensor.name << " dtype=" << evo2c::tensor_dtype_name(tensor.dtype)
            << " shape=[";
  for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
    if (dimension != 0) {
      std::cout << ',';
    }
    std::cout << tensor.dimensions[dimension];
  }
  std::cout << "] offset=" << tensor.data_offset << " bytes=" << tensor.data_size
            << " crc32=" << std::hex << tensor.data_crc32 << std::dec << '\n';
}

int inspect(const std::string& path, const std::string_view selected_tensor) {
  evo2c::ModelFile model;
  const auto status = model.open(path);
  if (!status.ok()) {
    return fail(status);
  }

  std::cout << "format=EVO2C version=" << model.version() << " file_size=" << model.file_size()
            << " checksum=ok\n";
  std::cout << "metadata_count=" << model.metadata().size() << '\n';
  for (const auto& entry : model.metadata()) {
    std::cout << "metadata " << entry.key << " type=" << evo2c::metadata_type_name(entry.type)
              << " value=" << evo2c::metadata_value_text(entry) << '\n';
  }

  if (!selected_tensor.empty()) {
    const auto* tensor = model.find_tensor(selected_tensor);
    if (tensor == nullptr) {
      return fail({evo2c::ErrorCode::kInvalidArgument,
                   "tensor '" + std::string(selected_tensor) + "' not found"});
    }
    print_tensor(*tensor);
    return 0;
  }

  std::cout << "tensor_count=" << model.tensors().size() << '\n';
  for (const auto& tensor : model.tensors()) {
    print_tensor(tensor);
  }
  return 0;
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
      return inspect(argv[1], {});
    }
    if (argc == 4 && std::string_view{argv[2]} == "--tensor") {
      return inspect(argv[1], argv[3]);
    }
    return fail({evo2c::ErrorCode::kInvalidArgument,
                 "usage: evo2c-inspect MODEL [--tensor NAME]"});
  } catch (const std::exception& error) {
    return fail({evo2c::ErrorCode::kInternal, error.what()});
  } catch (...) {
    return fail({evo2c::ErrorCode::kInternal, "unknown exception"});
  }
}

