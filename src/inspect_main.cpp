// SPDX-License-Identifier: Apache-2.0
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

#include "evo/model_format.hpp"
#include "evo/model_registry.hpp"
#include "evo/status.hpp"
#include "evo/version.hpp"

namespace {

void print_help() {
  std::cout << "evo-inspect - inspect an Evo 2 runtime Safetensors file\n\n"
            << "Usage:\n"
            << "  evo-inspect --help\n"
            << "  evo-inspect --version\n"
            << "  evo-inspect MODEL.safetensors[.index.json] [--tensor NAME]\n";
}

int fail(const evo::Status &status) {
  std::cerr << "evo-inspect: " << evo::error_code_name(status.code())
            << ": " << status.message() << '\n';
  return evo::exit_code(status.code());
}

void print_tensor(const evo::TensorInfo &tensor) {
  std::cout << "tensor " << tensor.name
            << " dtype=" << evo::tensor_dtype_name(tensor.dtype)
            << " shape=[";
  for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
    if (dimension != 0)
      std::cout << ',';
    std::cout << tensor.dimensions[dimension];
  }
  std::cout << "] shard=" << tensor.shard_index
            << " offset=" << tensor.data_offset
            << " bytes=" << tensor.data_size << '\n';
}

int inspect(const std::string &path, const std::string_view selected_tensor) {
  evo::ModelFile model;
  const auto status = model.open(path);
  if (!status.ok())
    return fail(status);

  std::cout << "format=" << model.format_name()
            << " profile=" << model.profile()
            << " shards=" << model.shard_count()
            << " file_size=" << model.file_size() << " validation=ok\n";
  std::cout << "metadata_count=" << model.metadata().size() << '\n';
  for (const auto &entry : model.metadata()) {
    std::cout << "metadata " << entry.key
              << " type=" << evo::metadata_type_name(entry.type)
              << " value=" << evo::metadata_value_text(entry) << '\n';
  }
  const auto *const model_id = model.find_metadata("model.id");
  const auto *const official =
      model_id != nullptr && model_id->type == evo::MetadataType::kString
          ? evo::find_official_model(evo::metadata_value_text(*model_id))
          : nullptr;
  if (model.profile() == evo::kModelProfile) {
    std::cout << "exact_support="
              << (official == nullptr
                      ? "unknown"
                      : evo::official_exact_support_name(
                            official->exact_support))
              << " evidence="
              << (official == nullptr || official->exact_evidence.empty()
                      ? "none"
                      : official->exact_evidence)
              << '\n';
  }

  if (!selected_tensor.empty()) {
    const auto *const tensor = model.find_tensor(selected_tensor);
    if (tensor == nullptr) {
      return fail({evo::ErrorCode::kInvalidArgument,
                   "tensor '" + std::string{selected_tensor} + "' not found"});
    }
    print_tensor(*tensor);
    return 0;
  }

  std::cout << "tensor_count=" << model.tensors().size() << '\n';
  for (const auto &tensor : model.tensors())
    print_tensor(tensor);
  return 0;
}

} // namespace

int main(const int argc, char **argv) {
  try {
    if (argc == 2) {
      const std::string_view argument{argv[1]};
      if (argument == "--help" || argument == "-h") {
        print_help();
        return 0;
      }
      if (argument == "--version") {
        std::cout << "evo-inspect " << evo::version() << '\n';
        return 0;
      }
      return inspect(argv[1], {});
    }
    if (argc == 4 && std::string_view{argv[2]} == "--tensor" &&
        std::string_view{argv[3]}.empty() == false) {
      return inspect(argv[1], argv[3]);
    }
    return fail(
        {evo::ErrorCode::kInvalidArgument,
         "usage: evo-inspect MODEL.safetensors[.index.json] [--tensor NAME]"});
  } catch (const std::exception &error) {
    return fail({evo::ErrorCode::kInternal, error.what()});
  } catch (...) {
    return fail({evo::ErrorCode::kInternal, "unknown exception"});
  }
}
