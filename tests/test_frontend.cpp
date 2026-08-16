// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "evo/cli.hpp"
#include "evo/generation_io.hpp"
#include "evo/sampler.hpp"
#include "evo/sequence_io.hpp"
#include "evo/tokenizer.hpp"
#include "evo/variant.hpp"
#include "evo/variant_io.hpp"

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
    auto pattern =
        (std::filesystem::temp_directory_path() / "evo-frontend-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int descriptor = ::mkstemp(writable.data());
    if (descriptor < 0) {
      std::abort();
    }
    ::close(descriptor);
    path_ = writable.data();
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
    if (!output) {
      std::abort();
    }
  }

  ~TemporaryFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;

  [[nodiscard]] const std::string &path() const noexcept { return path_; }

private:
  std::string path_;
};

class TemporaryGzipFile final {
public:
  explicit TemporaryGzipFile(const std::string_view contents) {
    auto pattern =
        (std::filesystem::temp_directory_path() / "evo-frontend-gzip-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int descriptor = ::mkstemp(writable.data());
    if (descriptor < 0)
      std::abort();
    ::close(descriptor);
    path_ = writable.data();
    std::vector<unsigned char> compressed{0x1fU, 0x8bU, 0x08U, 0x00U, 0x00U,
                                          0x00U, 0x00U, 0x00U, 0x00U, 0xffU};
    std::size_t offset = 0;
    do {
      const std::size_t size =
          std::min<std::size_t>(65535, contents.size() - offset);
      const bool final = offset + size == contents.size();
      compressed.push_back(final ? 0x01U : 0x00U);
      const auto length = static_cast<std::uint16_t>(size);
      const auto inverted = static_cast<std::uint16_t>(~length);
      compressed.push_back(static_cast<unsigned char>(length & 0xffU));
      compressed.push_back(static_cast<unsigned char>(length >> 8U));
      compressed.push_back(static_cast<unsigned char>(inverted & 0xffU));
      compressed.push_back(static_cast<unsigned char>(inverted >> 8U));
      for (std::size_t index = 0; index < size; ++index) {
        compressed.push_back(
            static_cast<unsigned char>(contents[offset + index]));
      }
      offset += size;
    } while (offset < contents.size());
    std::uint32_t crc = 0xffffffffU;
    for (const char character : contents) {
      crc ^= static_cast<unsigned char>(character);
      for (unsigned bit = 0; bit < 8; ++bit)
        crc = (crc >> 1U) ^ ((crc & 1U) != 0 ? 0xedb88320U : 0U);
    }
    crc ^= 0xffffffffU;
    const auto append_u32 = [&compressed](const std::uint32_t value) {
      for (unsigned shift = 0; shift < 32; shift += 8) {
        compressed.push_back(
            static_cast<unsigned char>((value >> shift) & 0xffU));
      }
    };
    append_u32(crc);
    append_u32(static_cast<std::uint32_t>(contents.size()));
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(compressed.data()),
                 static_cast<std::streamsize>(compressed.size()));
    if (!output) {
      std::abort();
    }
  }

  ~TemporaryGzipFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  TemporaryGzipFile(const TemporaryGzipFile &) = delete;
  TemporaryGzipFile &operator=(const TemporaryGzipFile &) = delete;

  [[nodiscard]] const std::string &path() const noexcept { return path_; }

private:
  std::string path_;
};

class ScopedStdin final {
public:
  explicit ScopedStdin(const std::string &path) {
    saved_ = ::dup(STDIN_FILENO);
    const int input = ::open(path.c_str(), O_RDONLY);
    if (saved_ < 0 || input < 0 || ::dup2(input, STDIN_FILENO) < 0)
      std::abort();
    ::close(input);
  }

  ~ScopedStdin() {
    if (::dup2(saved_, STDIN_FILENO) < 0)
      std::abort();
    ::close(saved_);
  }

  ScopedStdin(const ScopedStdin &) = delete;
  ScopedStdin &operator=(const ScopedStdin &) = delete;

private:
  int saved_{-1};
};

evo::Status parse(std::vector<std::string> arguments,
                  evo::CliOptions *const options) {
  std::vector<char *> argv;
  argv.reserve(arguments.size());
  for (auto &argument : arguments) {
    argv.push_back(argument.data());
  }
  return evo::parse_cli(static_cast<int>(argv.size()), argv.data(), options);
}

evo::Status
collect_test_records(const std::string &path,
                     std::vector<evo::SequenceRecord> *const records) {
  records->clear();
  constexpr std::size_t kTestRecordLimit = 1U << 20U;
  return evo::stream_sequence_file(path, kTestRecordLimit,
                                   [&](const evo::SequenceRecord &record) {
                                     records->push_back(record);
                                     return evo::Status::Ok();
                                   });
}

void test_tokenizer() {
  const std::string bytes{"A\xC3\xA9\0\xFF", 5};
  const auto tokens = evo::encode_bytes(bytes);
  check(tokens == std::vector<evo::TokenId>({65, 195, 169, 0, 255}),
        "UTF-8 and arbitrary bytes map to identical unsigned IDs");
  check(evo::kEvo2EosToken == 0 && evo::kEvo2PadToken == 1 &&
            evo::kEvo2TokenizerVocabSize == 512,
        "tokenizer constants match Vortex");
  std::uint8_t byte = 0;
  check(evo::token_to_byte(255, &byte).ok() && byte == 255,
        "byte token decodes without sign extension");
  check(!evo::token_to_byte(256, &byte).ok(),
        "non-byte vocabulary ID is rejected");
  check(!evo::token_to_byte(1, nullptr).ok(),
        "null tokenizer output is rejected");
}

void test_sequence_reader() {
  TemporaryFile fasta{">one\nAC\nGT\n\n>two description\r\nN\r\n"};
  std::vector<evo::SequenceRecord> records;
  auto status = collect_test_records(fasta.path(), &records);
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
  status = collect_test_records(text.path(), &records);
  check(status.ok() && records.size() == 1 && records[0].bytes == raw,
        "raw text input is byte-exact including newline and NUL");

  TemporaryFile empty_record{">one\n>two\nAC\n"};
  status = collect_test_records(empty_record.path(), &records);
  check(!status.ok() &&
            status.message().find("has no sequence") != std::string::npos,
        "empty FASTA record is rejected");

  TemporaryFile whitespace{">one\nAC GT\n"};
  status = collect_test_records(whitespace.path(), &records);
  check(!status.ok() &&
            status.message().find("whitespace") != std::string::npos,
        "ambiguous FASTA whitespace is rejected");

  std::vector<std::string> streamed_names;
  std::vector<std::string> streamed_bytes;
  status = evo::stream_sequence_file(fasta.path(), 4,
                                     [&](const evo::SequenceRecord &record) {
                                       streamed_names.push_back(record.name);
                                       streamed_bytes.push_back(record.bytes);
                                       return evo::Status::Ok();
                                     });
  check(status.ok() &&
            streamed_names ==
                std::vector<std::string>({"one", "two description"}) &&
            streamed_bytes == std::vector<std::string>({"ACGT", "N"}),
        "streaming FASTA delivers one ordered byte-exact record at a time");

  TemporaryFile oversized_fasta{">one\nACGTA\n"};
  std::size_t delivered = 0;
  status = evo::stream_sequence_file(oversized_fasta.path(), 4,
                                     [&](const evo::SequenceRecord &) {
                                       ++delivered;
                                       return evo::Status::Ok();
                                     });
  check(!status.ok() && delivered == 0 &&
            status.message().find("exceeds maximum 4 bytes") !=
                std::string::npos,
        "FASTA record limit is enforced before callback delivery");

  TemporaryFile oversized_raw{"ACGTA"};
  status = evo::stream_sequence_file(oversized_raw.path(), 4,
                                     [&](const evo::SequenceRecord &) {
                                       ++delivered;
                                       return evo::Status::Ok();
                                     });
  check(!status.ok() && delivered == 0 &&
            status.message().find("exceeds maximum 4 bytes") !=
                std::string::npos,
        "raw record limit is enforced while bytes are appended");

  TemporaryFile late_error{">one\nAC\n>two\nA C\n"};
  status = evo::stream_sequence_file(
      late_error.path(), 8, [&](const evo::SequenceRecord &record) {
        ++delivered;
        check(record.name == "one" && record.bytes == "AC",
              "record preceding a later parse error remains complete");
        return evo::Status::Ok();
      });
  check(!status.ok() && delivered == 1 &&
            status.message().find("whitespace at line 4") != std::string::npos,
        "later FASTA errors preserve completed callbacks and fail the stream");

  TemporaryFile fastq{"@read-one\nACGT\n+\nIIII\n"
                      "@read-two description\r\nN\r\n"
                      "+read-two description\r\n!\r\n"};
  status = collect_test_records(fastq.path(), &records);
  check(status.ok() && records.size() == 2 && records[0].name == "read-one" &&
            records[0].bytes == "ACGT" &&
            records[0].format == evo::SequenceFormat::kFastq &&
            records[1].name == "read-two description" &&
            records[1].bytes == "N",
        "strict FASTQ preserves names, CRLF normalization, and records");

  TemporaryFile late_fastq{"@one\nAC\n+\nII\n@two\nACG\n+\nII\n"};
  delivered = 0;
  status = evo::stream_sequence_file(
      late_fastq.path(), 8, [&](const evo::SequenceRecord &record) {
        ++delivered;
        check(record.name == "one" && record.bytes == "AC",
              "FASTQ delivers a complete record before a later error");
        return evo::Status::Ok();
      });
  check(!status.ok() && delivered == 1 &&
            status.message().find("quality length") != std::string::npos,
        "FASTQ late validation errors preserve completed callbacks");

  TemporaryGzipFile gzip_fasta{">gzip-one\nACGT\n>gzip-two\nNN\n"};
  status = collect_test_records(gzip_fasta.path(), &records);
  check(status.ok() && records.size() == 2 && records[0].name == "gzip-one" &&
            records[0].bytes == "ACGT" &&
            records[0].format == evo::SequenceFormat::kFasta,
        "gzip FASTA is transparently decompressed without extension guessing");

  std::vector<std::string> stdin_names;
  {
    ScopedStdin redirected{gzip_fasta.path()};
    status = evo::stream_sequence_file("-", 8,
                                       [&](const evo::SequenceRecord &record) {
                                         stdin_names.push_back(record.name);
                                         return evo::Status::Ok();
                                       });
  }
  check(status.ok() &&
            stdin_names == std::vector<std::string>({"gzip-one", "gzip-two"}),
        "stdin accepts the same transparently compressed sequence stream");
  check(std::string_view{evo::sequence_format_name(
            evo::SequenceFormat::kFastq)} == "fastq",
        "sequence format names are stable output metadata");
}

void test_sampler() {
  std::vector<float> logits(evo::kEvo2TokenizerVocabSize, -10.0F);
  logits[7] = 4.0F;
  logits[9] = 4.0F;
  evo::TokenId token = 0;
  evo::Sampler greedy{{1.0F, 1, 1.0F, 123}};
  auto status = greedy.sample(logits, &token);
  check(status.ok() && token == 7,
        "greedy sampler uses stable lowest-ID tie break");

  std::vector<float> small_logits(16, -10.0F);
  small_logits[10] = 4.0F;
  status = greedy.sample(small_logits, &token);
  check(status.ok() && token == 10,
        "sampler accepts a registered non-Evo vocabulary");

  evo::Sampler first{{1.0F, 8, 0.95F, 42}};
  evo::Sampler second{{1.0F, 8, 0.95F, 42}};
  for (std::size_t iteration = 0; iteration < 64; ++iteration) {
    evo::TokenId left = 0;
    evo::TokenId right = 0;
    check(first.sample(logits, &left).ok() &&
              second.sample(logits, &right).ok() && left == right,
          "identical sampler seeds reproduce every draw");
  }

  std::fill(logits.begin(), logits.end(), 0.0F);
  logits[2] = 3.0F;
  logits[3] = 2.0F;
  evo::Sampler top_k{{1.0F, 2, 1.0F, 9}};
  for (std::size_t iteration = 0; iteration < 128; ++iteration) {
    check(top_k.sample(logits, &token).ok() && (token == 2 || token == 3),
          "top-k sampler never selects a filtered token");
  }

  std::fill(logits.begin(), logits.end(), 0.0F);
  logits[11] = 10.0F;
  evo::Sampler top_p{{1.0F, 0, 0.5F, 1}};
  for (std::size_t iteration = 0; iteration < 16; ++iteration) {
    check(top_p.sample(logits, &token).ok() && token == 11,
          "top-p retains the minimal probability prefix");
  }

  logits[1] = std::numeric_limits<float>::quiet_NaN();
  check(!top_k.sample(logits, &token).ok(), "non-finite logits fail closed");
  check(!evo::Sampler({0.0F, 1, 1.0F, 0}).sample(logits, &token).ok(),
        "invalid sampling configuration fails closed");
}

void test_variant() {
  std::string reverse;
  auto status = evo::reverse_complement("ACGTN", &reverse);
  check(status.ok() && reverse == "NACGT",
        "reverse complement preserves canonical IUPAC DNA");
  status = evo::reverse_complement("RYSWKMBDHVN", &reverse);
  check(status.ok() && reverse == "NBDHVKMWSRY",
        "reverse complement supports ambiguous IUPAC DNA");
  status = evo::reverse_complement("acgt", &reverse);
  check(status.ok() && reverse == "acgt",
        "reverse complement preserves input case");
  status = evo::reverse_complement("ACGU", &reverse);
  check(!status.ok() && status.message().find("offset 3") != std::string::npos,
        "reverse complement rejects non-IUPAC DNA with an offset");

  evo::VariantWindow window;
  status = evo::make_variant_window("AACCGGTT", 3, "C", "T", 6, &window);
  check(status.ok() && window.reference == "AACCGG" &&
            window.alternate == "AATCGG" && window.reference_start == 0 &&
            window.reference_end == 6 && window.variant_offset == 2,
        "SNV window uses 1-based input and 0-based half-open coordinates");
  status = evo::make_variant_window("AACCGG", 3, "C", "CTT", 5, &window);
  check(status.ok() && window.reference == "ACC" &&
            window.alternate == "ACTTC" && window.reference_start == 1 &&
            window.reference_end == 4 && window.variant_offset == 1,
        "indel windows fit the longer allele without shifting the locus");
  status = evo::make_variant_window("AACCGG", 3, "G", "T", 5, &window);
  check(!status.ok() &&
            status.message().find("does not match") != std::string::npos,
        "reference mismatch fails before inference");
  status = evo::make_variant_window("A", 1, "A", "T", 2, &window);
  check(
      !status.ok() && status.message().find("two tokens") != std::string::npos,
      "variant likelihood requires two-token reference and alternate windows");

  TemporaryGzipFile reference{">chr1 primary contig\nAACC\nGGTT\n"
                              ">chr2\nTTTT\n"};
  evo::ReferenceSlice slice;
  status = evo::fetch_reference_slice(reference.path(), "chr1", 3, "C", "T", 6,
                                      &slice);
  check(status.ok() && slice.contig == "chr1" && slice.sequence == "AACCGG" &&
            slice.start == 0 && slice.end == 6 && slice.contig_length == 8,
        "reference FASTA lookup returns a global 0-based half-open slice");
  status = evo::fetch_reference_slice(reference.path(), "chr1", 3, "G", "T", 6,
                                      &slice);
  check(!status.ok() &&
            status.message().find("does not match") != std::string::npos,
        "VCF REF mismatch against reference FASTA fails explicitly");

  TemporaryGzipFile vcf{"##fileformat=VCFv4.3\n"
                        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                        "chr1\t3\trs1\tC\tT,G\t.\tPASS\t.\n"};
  std::vector<evo::VcfRecord> variants;
  status = evo::stream_vcf_file(vcf.path(), [&](const evo::VcfRecord &record) {
    variants.push_back(record);
    return evo::Status::Ok();
  });
  check(status.ok() && variants.size() == 2 && variants[0].record_index == 0 &&
            variants[0].allele_index == 0 && variants[0].contig == "chr1" &&
            variants[0].position_1based == 3 && variants[1].allele_index == 1 &&
            variants[1].alternate == "G",
        "gzip VCF streams one ordered event per alternate allele");

  TemporaryFile late_vcf{"##fileformat=VCFv4.3\n"
                         "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                         "chr1\t3\trs1\tC\tT\t.\tPASS\t.\n"
                         "chr1\tbad\trs2\tC\tA\t.\tPASS\t.\n"};
  std::size_t delivered = 0;
  status = evo::stream_vcf_file(late_vcf.path(), [&](const evo::VcfRecord &) {
    ++delivered;
    return evo::Status::Ok();
  });
  check(!status.ok() && delivered == 1 &&
            status.message().find("VCF POS") != std::string::npos,
        "later VCF errors preserve complete prior variant callbacks");
}

void test_cli() {
  evo::CliOptions options;
  auto status = parse({"evo",
                       "-m",
                       "model.safetensors",
                       "-p",
                       "ACGT",
                       "-n",
                       "8",
                       "--ctx",
                       "128",
                       "--gpu",
                       "0,1,2,3",
                       "--temp",
                       "0.8",
                       "--top-k",
                       "32",
                       "--top-p",
                       "0.9",
                       "--seed",
                       "99",
                       "--force-prompt-threshold",
                       "3",
                       "--dump-tokens",
                       "--dump-logits",
                       "logits.npy",
                       "--dump-layer",
                       "49:layer.npy",
                       "--profile",
                       "fast-q8-kv"},
                      &options);
  check(status.ok(),
        std::string{"valid generation CLI parses: "} + status.message());
  check(options.mode == evo::RunMode::kGenerate && options.prompt == "ACGT" &&
            options.generated_tokens == 8 && options.context_size == 128,
        "generation CLI values are retained");
  check(options.gpu_ids == std::vector<int>({0, 1, 2, 3}) &&
            options.sampling.top_k == 32 && options.sampling.seed == 99,
        "GPU and sampling CLI values are retained");
  check(options.dump_layer.has_value() && options.dump_layer->layer == 49,
        "debug layer CLI parses");
  check(options.force_prompt_threshold == 3,
        "teacher-forced prompt threshold CLI parses");
  check(options.inference_profile == evo::InferenceProfile::kFastQ8Kv,
        "explicit approximate execution profile parses");
  status = parse({"evo", "run", "-m", "model.safetensors", "-p", "ACGT", "-n",
                  "2", "--output-format", "fasta", "--name", "candidate",
                  "--backend", "cpu"},
                 &options);
  check(status.ok() && options.mode == evo::RunMode::kGenerate &&
            options.generation_output_format ==
                evo::GenerationOutputFormat::kFasta &&
            options.generation_name == "candidate",
        "run subcommand retains FASTA generation output contract");
  status = parse({"evo", "score", "-m", "model.safetensors", "--input",
                  "input.fa", "--backend", "cpu"},
                 &options);
  check(status.ok() && options.mode == evo::RunMode::kScore &&
            options.score_path == "input.fa",
        "score subcommand maps --input to the streaming score path");
  status = parse({"evo", "score", "-hf", "owner/runtime@main", "--input",
                  "input.fa", "--backend", "cpu"},
                 &options);
  check(status.ok() && options.model_path.empty() &&
            options.hf_repo == "owner/runtime@main",
        "Hugging Face model source is retained for offline cache resolution");
  status = parse({"evo", "score", "-m", "model.safetensors", "-hf",
                  "owner/runtime", "--input", "input.fa", "--backend", "cpu"},
                 &options);
  check(!status.ok() &&
            status.message().find("exactly one") != std::string::npos,
        "CLI rejects ambiguous local and Hugging Face model sources");
  status =
      parse({"evo", "bench", "-m", "model.safetensors", "--input", "input.fa",
             "--warmup", "2", "--repetitions", "7", "--backend", "cpu"},
            &options);
  check(status.ok() && options.mode == evo::RunMode::kBench &&
            options.benchmark_path == "input.fa" &&
            options.benchmark_warmup == 2 && options.benchmark_repetitions == 7,
        "bench subcommand retains input and sample policy");
  status = parse({"evo", "bench", "-m", "model.safetensors", "--input",
                  "input.fa", "--repetitions", "0", "--backend", "cpu"},
                 &options);
  check(!status.ok() &&
            status.message().find("repetitions") != std::string::npos,
        "bench rejects an empty measured sample set");
  status = parse({"evo", "run", "-m", "model.safetensors", "-p", "ACGT", "-n",
                  "2", "--output-format", "json", "--backend", "cpu"},
                 &options);
  check(!status.ok() &&
            status.message().find("output-format") != std::string::npos,
        "generation rejects an unknown output encoding");
  status = parse({"evo", "-m", "model.safetensors", "--score", "input.fa",
                  "--backend", "cpu"},
                 &options);
  check(status.ok() && options.backend == evo::ExecutionBackend::kCpu &&
            options.inference_profile == evo::InferenceProfile::kCpuF32 &&
            options.gpu_ids.empty(),
        "CPU backend selects explicit non-exact profile without a GPU list");
  status = parse({"evo", "-m", "model.safetensors", "--score", "input.fa",
                  "--backend", "cpu", "--gpu", "0"},
                 &options);
  check(!status.ok() &&
            status.message().find("GPU options") != std::string::npos,
        "CPU backend rejects contradictory GPU placement");
  status = parse({"evo", "score", "-m", "model.safetensors", "--input",
                  "input.fa", "--backend", "mps"},
                 &options);
  check(status.ok() && options.backend == evo::ExecutionBackend::kMps &&
            options.inference_profile == evo::InferenceProfile::kMpsF32 &&
            options.gpu_ids.empty(),
        "MPS backend selects its explicit non-exact profile without CUDA placement");
  status = parse({"evo", "score", "-m", "model.safetensors", "--input",
                  "input.fa", "--backend", "mps", "--gpu", "0"},
                 &options);
  check(!status.ok() && status.message().find("MPS") != std::string::npos,
        "MPS backend rejects CUDA device placement");
  status = parse({"evo", "score", "-m", "model.safetensors", "--input",
                  "input.fa", "--backend", "mps", "--profile", "exact"},
                 &options);
  check(!status.ok() && status.message().find("mps-f32") != std::string::npos,
        "MPS backend rejects contradictory execution profiles");
  status = parse({"evo", "-m", "model.safetensors", "--score", "input.fa",
                  "--gpu", "0", "--gpu-layers", "17"},
                 &options);
  check(status.ok() && options.gpu_layers == 17 &&
            options.inference_profile == evo::InferenceProfile::kCpuF32,
        "hybrid offload retains its explicit CUDA prefix length");
  status = parse({"evo", "-m", "model.safetensors", "--score", "input.fa",
                  "--gpu-layers", "17"},
                 &options);
  check(!status.ok() && status.message().find("requires an explicit --gpu") !=
                            std::string::npos,
        "hybrid offload never auto-selects a CUDA device");
  status = parse({"evo", "-m", "model.safetensors", "-p", "A", "-n", "1",
                  "--gpu", "0", "--dump-layer", "50:layer.npy"},
                 &options);
  check(status.ok() && options.dump_layer.has_value() &&
            options.dump_layer->layer == 50,
        "CLI defers model-specific layer bounds to the runtime");

  status = parse(
      {"evo", "-m", "model.safetensors", "--score", "input.fa", "--gpu", "2"},
      &options);
  check(status.ok() && options.mode == evo::RunMode::kScore,
        "score CLI accepts one GPU and no sampling options");

  status =
      parse({"evo", "logits", "-m", "model.safetensors", "--input", "input.fa",
             "--output", "logits", "--ctx", "2048", "--gpu", "0"},
            &options);
  check(status.ok() && options.mode == evo::RunMode::kLogits &&
            options.embed_path == "input.fa" &&
            options.embed_output_dir == "logits",
        "logits subcommand retains its streaming input and output directory");
  status =
      parse({"evo", "logits", "-m", "model.safetensors", "--input", "input.fa",
             "--output", "logits", "--layer", "1", "--gpu", "0"},
            &options);
  check(!status.ok() &&
            status.message().find("logits accepts") != std::string::npos,
        "logits rejects embedding-only layer selection");

  status = parse({"evo", "embed", "-m", "model.safetensors", "--input",
                  "input.fa", "--output", "embeddings", "--layer", "17",
                  "--pooling", "mean", "--ctx", "4096", "--gpu", "0,1"},
                 &options);
  check(status.ok() && options.mode == evo::RunMode::kEmbed &&
            options.embed_path == "input.fa" &&
            options.embed_output_dir == "embeddings" &&
            options.embed_layer == 17 &&
            options.embedding_pooling == evo::EmbeddingPooling::kMean,
        "embedding subcommand retains input, layer, pooling, and output");
  status = parse({"evo", "embed", "-m", "model.safetensors", "--input",
                  "input.fa", "--output", "embeddings", "--layer", "17",
                  "--pooling", "median", "--gpu", "0"},
                 &options);
  check(!status.ok() && status.message().find("pooling") != std::string::npos,
        "embedding rejects an unknown pooling mode");
  status = parse({"evo", "embed", "-m", "model.safetensors", "--input",
                  "input.fa", "--output", "embeddings", "--layer", "17",
                  "--gpu", "0", "--top-k", "1"},
                 &options);
  check(!status.ok() &&
            status.message().find("generation/scoring") != std::string::npos,
        "embedding rejects sampling and generation options");

  status = parse({"evo",
                  "variant-score",
                  "-m",
                  "model.safetensors",
                  "--sequence",
                  "AACCGGTT",
                  "--position",
                  "3",
                  "--ref",
                  "C",
                  "--alt",
                  "T",
                  "--window",
                  "6",
                  "--strand",
                  "both",
                  "--normalization",
                  "mean",
                  "--ctx",
                  "8",
                  "--gpu",
                  "0,1"},
                 &options);
  check(
      status.ok() && options.mode == evo::RunMode::kVariantScore &&
          options.variant_sequence == "AACCGGTT" &&
          options.variant_position_1based == 3 &&
          options.variant_reference == "C" &&
          options.variant_alternate == "T" &&
          options.variant_window_tokens == 6 &&
          options.variant_strand == evo::VariantStrand::kBoth &&
          options.variant_normalization == evo::VariantNormalization::kMean,
      "variant-score retains coordinates, alleles, strand, and normalization");
  status = parse({"evo", "variant-score", "-m", "model.safetensors",
                  "--sequence", "AACCGGTT", "--position", "3", "--ref", "C",
                  "--alt", "T", "--ctx", "8", "--gpu", "0"},
                 &options);
  check(status.ok() && options.variant_window_tokens == 8 &&
            options.variant_strand == evo::VariantStrand::kBoth &&
            options.variant_normalization == evo::VariantNormalization::kSum,
        "variant-score defaults to context window, both strands, and sum");
  status = parse({"evo", "variant-score", "-m", "model.safetensors", "--vcf",
                  "variants.vcf.gz", "--reference", "reference.fa.gz",
                  "--window", "8", "--backend", "cpu"},
                 &options);
  check(status.ok() && options.variant_vcf_path == "variants.vcf.gz" &&
            options.variant_reference_path == "reference.fa.gz" &&
            options.variant_sequence.empty(),
        "variant-score retains gzip VCF and reference FASTA paths");
  status = parse({"evo", "variant-score", "-m", "model.safetensors", "--vcf",
                  "variants.vcf", "--reference", "reference.fa", "--sequence",
                  "AACCGGTT", "--position", "3", "--ref", "C", "--alt", "T",
                  "--backend", "cpu"},
                 &options);
  check(!status.ok() &&
            status.message().find("cannot be combined") != std::string::npos,
        "VCF batch mode rejects contradictory inline coordinates");
  status = parse({"evo", "variant-score", "-m", "model.safetensors", "--vcf",
                  "variants.vcf", "--backend", "cpu"},
                 &options);
  check(!status.ok() &&
            status.message().find("--reference") != std::string::npos,
        "VCF batch mode requires a reference FASTA path");
  status = parse({"evo", "variant-score", "-m", "model.safetensors",
                  "--sequence", "AACCGGTT", "--position", "3", "--ref", "C",
                  "--alt", "T", "--window", "9", "--ctx", "8", "--gpu", "0"},
                 &options);
  check(!status.ok() &&
            status.message().find("must not exceed") != std::string::npos,
        "variant window cannot exceed the inference context");
  status = parse({"evo", "variant-score", "-m", "model.safetensors",
                  "--sequence", "AACCGGTT", "--position", "3", "--ref", "C",
                  "--alt", "T", "--strand", "sense", "--gpu", "0"},
                 &options);
  check(!status.ok() && status.message().find("strand") != std::string::npos,
        "variant-score rejects unknown strand semantics");
  status = parse({"evo", "-m", "model.safetensors", "--sequence", "AACCGGTT",
                  "--position", "3", "--ref", "C", "--alt", "T", "--gpu", "0"},
                 &options);
  check(!status.ok() &&
            status.message().find("variant-score") != std::string::npos,
        "variant arguments require the variant-score subcommand");

  status =
      parse({"evo", "-m", "a", "-m", "b", "-p", "A", "-n", "1", "--gpu", "0"},
            &options);
  check(!status.ok() &&
            status.message().find("specified twice") != std::string::npos,
        "duplicate CLI options are rejected");
  status = parse(
      {"evo", "-m", "a", "-p", "A", "--score", "x", "-n", "1", "--gpu", "0"},
      &options);
  check(!status.ok() &&
            status.message().find("exactly one") != std::string::npos,
        "conflicting run modes are rejected");
  status =
      parse({"evo", "-m", "a", "-p", "A", "-n", "1", "--gpu", "0,0"}, &options);
  check(!status.ok() &&
            status.message().find("duplicate ID") != std::string::npos,
        "duplicate GPU IDs are rejected");
  status = parse(
      {"evo", "-m", "a", "-p", "A", "-n", "1", "--gpu", "0", "--top-p", "0"},
      &options);
  check(!status.ok() && status.message().find("top-p") != std::string::npos,
        "invalid top-p is rejected");
  status = parse({"evo", "-m", "a", "-p", "A", "-n", "1", "--gpu", "0",
                  "--force-prompt-threshold", "0"},
                 &options);
  check(!status.ok() && status.message().find("force-prompt-threshold") !=
                            std::string::npos,
        "zero teacher-forcing threshold is rejected");
  status = parse({"evo", "-m", "a", "--score", "input.fa", "--gpu", "0",
                  "--force-prompt-threshold", "3"},
                 &options);
  check(!status.ok() && status.message().find("only valid for generation") !=
                            std::string::npos,
        "teacher-forcing threshold is rejected for score mode");

  status = parse({"evo",
                  "serve",
                  "-m",
                  "model.safetensors",
                  "--ctx",
                  "4096",
                  "--gpu",
                  "0,1",
                  "--host",
                  "0.0.0.0",
                  "--port",
                  "0",
                  "--max-queue",
                  "32",
                  "--max-batch",
                  "8",
                  "--batch-window-ms",
                  "5",
                  "--max-request-bytes",
                  "8192",
                  "--max-sequence-bytes",
                  "2048",
                  "--max-embedding-values",
                  "65536"},
                 &options);
  check(status.ok() && options.mode == evo::RunMode::kServe &&
            options.server_host == "0.0.0.0" && options.server_port == 0 &&
            options.server_max_queue == 32 && options.server_max_batch == 8 &&
            options.server_batch_window_ms == 5 &&
            options.server_max_request_bytes == 8192 &&
            options.server_max_sequence_bytes == 2048 &&
            options.server_max_embedding_values == 65536,
        "serve retains explicit network, batching, and resource limits");
  status = parse({"evo", "serve", "-m", "model.safetensors", "--ctx", "16",
                  "--gpu", "0", "--profile", "fast-q8-kv"},
                 &options);
  check(status.ok() && options.server_max_sequence_bytes == 16 &&
            options.inference_profile == evo::InferenceProfile::kFastQ8Kv,
        "serve defaults sequence limit and retains its execution profile");
  status = parse({"evo", "serve", "-m", "model.safetensors", "--gpu", "0",
                  "--profile", "approximate"},
                 &options);
  check(!status.ok() &&
            status.message().find("fast-q8-kv") != std::string::npos,
        "unknown execution profiles fail closed");
  status = parse({"evo", "serve", "-m", "model.safetensors", "--ctx", "8",
                  "--gpu", "0", "--max-sequence-bytes", "9"},
                 &options);
  check(!status.ok() &&
            status.message().find("must not exceed --ctx") != std::string::npos,
        "serve rejects a sequence limit larger than context capacity");
  status = parse({"evo", "serve", "-m", "model.safetensors", "--gpu", "0",
                  "--max-queue", "2", "--max-batch", "3"},
                 &options);
  check(!status.ok() &&
            status.message().find("--max-batch") != std::string::npos,
        "serve rejects a batch larger than the bounded queue");
  status = parse(
      {"evo", "-m", "a", "-p", "A", "-n", "1", "--gpu", "0", "--port", "8080"},
      &options);
  check(!status.ok() && status.message().find("evo serve") != std::string::npos,
        "server-only options require the serve subcommand");

  std::ostringstream raw;
  status = evo::write_generated_sequence(
      raw, "ACGT", evo::GenerationOutputFormat::kRaw, "ignored");
  check(status.ok() && raw.str() == "ACGT",
        "raw generation output is byte exact");
  std::ostringstream fasta;
  status = evo::write_generated_sequence(
      fasta, "ACGTN", evo::GenerationOutputFormat::kFasta, "candidate");
  check(status.ok() && fasta.str() == ">candidate\nACGTN\n",
        "FASTA generation output is named and round-trippable");
  std::ostringstream invalid_fasta;
  status = evo::write_generated_sequence(
      invalid_fasta, "AC GT", evo::GenerationOutputFormat::kFasta, "candidate");
  check(!status.ok() && invalid_fasta.str().empty(),
        "FASTA generation fails before emitting a partial invalid record");
}

} // namespace

int main() {
  test_tokenizer();
  test_sequence_reader();
  test_sampler();
  test_variant();
  test_cli();
  if (failures != 0) {
    std::cerr << failures << " frontend test(s) failed\n";
    return 1;
  }
  std::cout << "frontend tests passed\n";
  return 0;
}
