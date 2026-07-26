// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "evo2c/cli.hpp"
#include "evo2c/sampler.hpp"
#include "evo2c/sequence_io.hpp"
#include "evo2c/tokenizer.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

class TemporaryFile final {
 public:
  explicit TemporaryFile(const std::string_view contents) {
    auto pattern = (std::filesystem::temp_directory_path() / "evo2c-frontend-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int descriptor = ::mkstemp(writable.data());
    if (descriptor < 0) {
      std::abort();
    }
    ::close(descriptor);
    path_ = writable.data();
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output) {
      std::abort();
    }
  }

  ~TemporaryFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  TemporaryFile(const TemporaryFile&) = delete;
  TemporaryFile& operator=(const TemporaryFile&) = delete;

  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  std::string path_;
};

evo2c::Status parse(std::vector<std::string> arguments, evo2c::CliOptions* const options) {
  std::vector<char*> argv;
  argv.reserve(arguments.size());
  for (auto& argument : arguments) {
    argv.push_back(argument.data());
  }
  return evo2c::parse_cli(static_cast<int>(argv.size()), argv.data(), options);
}

void test_tokenizer() {
  const std::string bytes{"A\xC3\xA9\0\xFF", 5};
  const auto tokens = evo2c::encode_bytes(bytes);
  check(tokens == std::vector<evo2c::TokenId>({65, 195, 169, 0, 255}),
        "UTF-8 and arbitrary bytes map to identical unsigned IDs");
  check(evo2c::kEosToken == 0 && evo2c::kPadToken == 1 &&
            evo2c::kTokenizerVocabSize == 512,
        "tokenizer constants match Vortex");
  std::uint8_t byte = 0;
  check(evo2c::token_to_byte(255, &byte).ok() && byte == 255,
        "byte token decodes without sign extension");
  check(!evo2c::token_to_byte(256, &byte).ok(), "non-byte vocabulary ID is rejected");
  check(!evo2c::token_to_byte(1, nullptr).ok(), "null tokenizer output is rejected");
}

void test_sequence_reader() {
  TemporaryFile fasta{">one\nAC\nGT\n\n>two description\r\nN\r\n"};
  std::vector<evo2c::SequenceRecord> records;
  auto status = evo2c::read_sequence_file(fasta.path(), &records);
  check(status.ok(), std::string{"valid FASTA parses: "} + status.message());
  check(records.size() == 2, "multi-record FASTA preserves record boundaries");
  if (records.size() == 2) {
    check(records[0].name == "one" && records[0].bytes == "ACGT",
          "wrapped FASTA sequence is concatenated");
    check(records[1].name == "two description" && records[1].bytes == "N",
          "CRLF FASTA header and sequence are normalized");
  }

  const std::string raw{"A\xC3\xA9\n\0", 5};
  TemporaryFile text{raw};
  status = evo2c::read_sequence_file(text.path(), &records);
  check(status.ok() && records.size() == 1 && records[0].bytes == raw,
        "raw text input is byte-exact including newline and NUL");

  TemporaryFile empty_record{">one\n>two\nAC\n"};
  status = evo2c::read_sequence_file(empty_record.path(), &records);
  check(!status.ok() && status.message().find("has no sequence") != std::string::npos,
        "empty FASTA record is rejected");

  TemporaryFile whitespace{">one\nAC GT\n"};
  status = evo2c::read_sequence_file(whitespace.path(), &records);
  check(!status.ok() && status.message().find("whitespace") != std::string::npos,
        "ambiguous FASTA whitespace is rejected");
}

void test_sampler() {
  std::vector<float> logits(evo2c::kTokenizerVocabSize, -10.0F);
  logits[7] = 4.0F;
  logits[9] = 4.0F;
  evo2c::TokenId token = 0;
  evo2c::Sampler greedy{{1.0F, 1, 1.0F, 123}};
  auto status = greedy.sample(logits, &token);
  check(status.ok() && token == 7, "greedy sampler uses stable lowest-ID tie break");

  evo2c::Sampler first{{1.0F, 8, 0.95F, 42}};
  evo2c::Sampler second{{1.0F, 8, 0.95F, 42}};
  for (std::size_t iteration = 0; iteration < 64; ++iteration) {
    evo2c::TokenId left = 0;
    evo2c::TokenId right = 0;
    check(first.sample(logits, &left).ok() && second.sample(logits, &right).ok() && left == right,
          "identical sampler seeds reproduce every draw");
  }

  std::fill(logits.begin(), logits.end(), 0.0F);
  logits[2] = 3.0F;
  logits[3] = 2.0F;
  evo2c::Sampler top_k{{1.0F, 2, 1.0F, 9}};
  for (std::size_t iteration = 0; iteration < 128; ++iteration) {
    check(top_k.sample(logits, &token).ok() && (token == 2 || token == 3),
          "top-k sampler never selects a filtered token");
  }

  std::fill(logits.begin(), logits.end(), 0.0F);
  logits[11] = 10.0F;
  evo2c::Sampler top_p{{1.0F, 0, 0.5F, 1}};
  for (std::size_t iteration = 0; iteration < 16; ++iteration) {
    check(top_p.sample(logits, &token).ok() && token == 11,
          "top-p retains the minimal probability prefix");
  }

  logits[1] = std::numeric_limits<float>::quiet_NaN();
  check(!top_k.sample(logits, &token).ok(), "non-finite logits fail closed");
  check(!evo2c::Sampler({0.0F, 1, 1.0F, 0}).sample(logits, &token).ok(),
        "invalid sampling configuration fails closed");
}

void test_cli() {
  evo2c::CliOptions options;
  auto status = parse({"evo2c", "-m", "model.evo2", "-p", "ACGT", "-n", "8",
                       "--ctx", "128", "--gpu", "0,1,2,3", "--temp", "0.8",
                       "--top-k", "32", "--top-p", "0.9", "--seed", "99",
                       "--force-prompt-threshold", "3",
                       "--dump-tokens", "--dump-logits", "logits.npy", "--dump-layer",
                       "49:layer.npy"},
                      &options);
  check(status.ok(), std::string{"valid generation CLI parses: "} + status.message());
  check(options.mode == evo2c::RunMode::kGenerate && options.prompt == "ACGT" &&
            options.generated_tokens == 8 && options.context_size == 128,
        "generation CLI values are retained");
  check(options.gpu_ids == std::vector<int>({0, 1, 2, 3}) && options.sampling.top_k == 32 &&
            options.sampling.seed == 99,
        "GPU and sampling CLI values are retained");
  check(options.dump_layer.has_value() && options.dump_layer->layer == 49,
        "debug layer CLI parses");
  check(options.force_prompt_threshold == 3,
        "teacher-forced prompt threshold CLI parses");

  status = parse({"evo2c", "-m", "model.evo2", "-p", "A", "-n", "1",
                  "--gpu", "0", "--dump-layer", "50:layer.npy"},
                 &options);
  check(status.ok() && options.dump_layer.has_value() &&
            options.dump_layer->layer == 50,
        "CLI defers model-specific layer bounds to the runtime");

  status = parse({"evo2c", "-m", "model.evo2", "--score", "input.fa", "--gpu", "2"},
                 &options);
  check(status.ok() && options.mode == evo2c::RunMode::kScore,
        "score CLI accepts one GPU and no sampling options");

  status = parse({"evo2c", "-m", "a", "-m", "b", "-p", "A", "-n", "1", "--gpu", "0"},
                 &options);
  check(!status.ok() && status.message().find("specified twice") != std::string::npos,
        "duplicate CLI options are rejected");
  status = parse({"evo2c", "-m", "a", "-p", "A", "--score", "x", "-n", "1", "--gpu", "0"},
                 &options);
  check(!status.ok() && status.message().find("exactly one") != std::string::npos,
        "conflicting run modes are rejected");
  status = parse({"evo2c", "-m", "a", "-p", "A", "-n", "1", "--gpu", "0,0"},
                 &options);
  check(!status.ok() && status.message().find("duplicate ID") != std::string::npos,
        "duplicate GPU IDs are rejected");
  status = parse({"evo2c", "-m", "a", "-p", "A", "-n", "1", "--gpu", "0", "--top-p", "0"},
                 &options);
  check(!status.ok() && status.message().find("top-p") != std::string::npos,
        "invalid top-p is rejected");
  status = parse({"evo2c", "-m", "a", "-p", "A", "-n", "1", "--gpu", "0",
                  "--force-prompt-threshold", "0"},
                 &options);
  check(!status.ok() &&
            status.message().find("force-prompt-threshold") != std::string::npos,
        "zero teacher-forcing threshold is rejected");
  status = parse({"evo2c", "-m", "a", "--score", "input.fa", "--gpu", "0",
                  "--force-prompt-threshold", "3"},
                 &options);
  check(!status.ok() &&
            status.message().find("only valid for generation") != std::string::npos,
        "teacher-forcing threshold is rejected for score mode");
}

}  // namespace

int main() {
  test_tokenizer();
  test_sequence_reader();
  test_sampler();
  test_cli();
  if (failures != 0) {
    std::cerr << failures << " frontend test(s) failed\n";
    return 1;
  }
  std::cout << "frontend tests passed\n";
  return 0;
}
