// SPDX-License-Identifier: Apache-2.0
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "evo2c/cli.hpp"
#include "evo2c/sequence_io.hpp"
#include "evo2c/status.hpp"
#include "evo2c/tokenizer.hpp"
#include "evo2c/version.hpp"

namespace {

void print_help() {
  std::cout << evo2c::cli_usage();
}

void dump_tokens(const std::string_view label, const std::string_view bytes) {
  const auto tokens = evo2c::encode_bytes(bytes);
  std::cerr << "tokens " << label << "=[";
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (index != 0) {
      std::cerr << ',';
    }
    std::cerr << tokens[index];
  }
  std::cerr << "]\n";
}

int fail(const evo2c::Status& status) {
  std::cerr << "evo2c: " << evo2c::error_code_name(status.code()) << ": "
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
        std::cout << "evo2c " << evo2c::version() << '\n';
        return 0;
      }
    }

    evo2c::CliOptions options;
    auto status = evo2c::parse_cli(argc, argv, &options);
    if (!status.ok()) {
      return fail(status);
    }

    if (options.mode == evo2c::RunMode::kGenerate) {
      if (options.prompt.size() > options.context_size ||
          options.generated_tokens > options.context_size - options.prompt.size()) {
        return fail({evo2c::ErrorCode::kInvalidArgument,
                     "prompt bytes plus generated tokens exceed --ctx"});
      }
      if (options.dump_tokens) {
        dump_tokens("prompt", options.prompt);
      }
    } else {
      std::vector<evo2c::SequenceRecord> records;
      status = evo2c::read_sequence_file(options.score_path, &records);
      if (!status.ok()) {
        return fail(status);
      }
      for (const auto& record : records) {
        if (record.bytes.size() > options.context_size) {
          return fail({evo2c::ErrorCode::kInvalidArgument,
                       "score record '" + record.name + "' exceeds --ctx"});
        }
      }
      if (options.dump_tokens) {
        for (const auto& record : records) {
          dump_tokens(record.name, record.bytes);
        }
      }
    }

    return fail({evo2c::ErrorCode::kUnsupported,
                 "model execution is not implemented yet; complete SPEC tasks T5-T13"});
  } catch (const std::exception& error) {
    return fail({evo2c::ErrorCode::kInternal, error.what()});
  } catch (...) {
    return fail({evo2c::ErrorCode::kInternal, "unknown exception"});
  }
}
