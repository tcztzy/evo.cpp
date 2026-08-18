// SPDX-License-Identifier: Apache-2.0
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>

#include "evo/tokenizer.hpp"
#include "evo/geneb_embedding.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

std::string
manifest(const std::string_view kind, const std::string_view normalization,
         const std::string_view pretokenizer, const std::string_view model,
         const std::string_view post_processor,
         const std::string_view special_tokens, const std::string_view vocab) {
  return "{\"format\":\"evo-tokenizer-v1\",\"kind\":\"" + std::string{kind} +
         "\",\"normalization\":" + std::string{normalization} +
         ",\"pre_tokenizer\":" + std::string{pretokenizer} +
         ",\"model\":" + std::string{model} +
         ",\"post_processor\":" + std::string{post_processor} +
         ",\"special_tokens\":" + std::string{special_tokens} +
         ",\"vocab\":" + std::string{vocab} + "}";
}

constexpr std::string_view kNoSpecials =
    R"({"unk":null,"pad":null,"bos":null,"eos":null,"cls":null,"sep":null,"mask":null})";
constexpr std::string_view kNoPost =
    R"({"prefix_ids":[],"suffix_ids":[],"padding":{"side":"none","pad_id":null}})";

std::string character_manifest() {
  return manifest(
      "character", "[]", R"({"kind":"none"})",
      R"({"unknown_policy":"unk","match_special_literals":false})", kNoPost,
      R"({"unk":2,"pad":null,"bos":null,"eos":null,"cls":null,"sep":null,"mask":null})",
      R"([{"id":0,"piece":"A"},{"id":1,"piece":"C"},{"id":2,"piece":"[UNK]"}])");
}

std::string single_nucleotide_manifest() {
  return manifest(
      "single-nucleotide", R"([{"op":"ascii-uppercase"},{"op":"u-to-t"}])",
      R"({"kind":"none"})",
      R"({"unknown_policy":"unk","match_special_literals":false})",
      R"({"prefix_ids":[7],"suffix_ids":[8],"padding":{"side":"right","pad_id":6}})",
      R"({"unk":5,"pad":6,"bos":null,"eos":8,"cls":7,"sep":null,"mask":null})",
      R"([{"id":0,"piece":"A"},{"id":1,"piece":"C"},{"id":2,"piece":"G"},{"id":3,"piece":"T"},{"id":4,"piece":"N"},{"id":5,"piece":"[UNK]"},{"id":6,"piece":"[PAD]"},{"id":7,"piece":"[CLS]"},{"id":8,"piece":"[EOS]"}])");
}

std::string fixed_kmer_manifest() {
  return manifest(
      "kmer", "[]", R"({"kind":"none"})",
      R"({"k":2,"stride":2,"tail":"error","unknown_policy":"unk"})", kNoPost,
      R"({"unk":2,"pad":null,"bos":null,"eos":null,"cls":null,"sep":null,"mask":null})",
      R"([{"id":0,"piece":"AA"},{"id":1,"piece":"CG"},{"id":2,"piece":"[UNK]"}])");
}

std::string overlapping_kmer_manifest() {
  return manifest(
      "kmer", "[]", R"({"kind":"none"})",
      R"({"k":2,"stride":1,"tail":"drop","unknown_policy":"error"})", kNoPost,
      kNoSpecials,
      R"([{"id":0,"piece":"AA"},{"id":1,"piece":"AC"},{"id":2,"piece":"CG"}])");
}

std::string wordpiece_manifest() {
  return manifest(
      "wordpiece", "[]", R"({"kind":"ascii-whitespace"})",
      R"({"continuation_prefix":"##","max_input_chars_per_word":32})",
      R"({"prefix_ids":[2],"suffix_ids":[3],"padding":{"side":"right","pad_id":1}})",
      R"({"unk":0,"pad":1,"bos":null,"eos":null,"cls":2,"sep":3,"mask":null})",
      R"([{"id":0,"piece":"[UNK]"},{"id":1,"piece":"[PAD]"},{"id":2,"piece":"[CLS]"},{"id":3,"piece":"[SEP]"},{"id":4,"piece":"A"},{"id":5,"piece":"AC"},{"id":6,"piece":"##G"},{"id":7,"piece":"##GT"}])");
}

std::string gena_bpe_manifest() {
  return manifest(
      "bpe",
      R"([{"op":"strip-ascii-whitespace"},{"op":"replace-byte-run","byte":"N","min_count":3,"replacement":"-"}])",
      R"({"kind":"split-isolated","literal":"-"})",
      R"({"merges":[["A","C"],["A","A"],["AA","AA"]]})", kNoPost,
      R"({"unk":4,"pad":null,"bos":null,"eos":null,"cls":null,"sep":null,"mask":null})",
      R"([{"id":0,"piece":"A"},{"id":1,"piece":"C"},{"id":2,"piece":"-"},{"id":3,"piece":"AC"},{"id":4,"piece":"[UNK]"},{"id":5,"piece":"AA"},{"id":6,"piece":"AAAA"}])");
}

std::string omnina_bpe_manifest() {
  return manifest(
      "bpe",
      R"([{"op":"prepend-literal","value":"▁"},{"op":"replace-literal","from":" ","to":"▁"}])",
      R"({"kind":"whole-input"})", R"({"merges":[["▁","A"],["▁A","C"]]})",
      kNoPost, kNoSpecials,
      R"([{"id":0,"piece":"▁"},{"id":1,"piece":"A"},{"id":2,"piece":"C"},{"id":3,"piece":"▁A"},{"id":4,"piece":"▁AC"}])");
}

std::string longest_match_manifest() {
  return manifest(
      "longest-match", "[]", R"({"kind":"whole-input"})",
      R"({"unknown_policy":"unk","match_special_literals":false})", kNoPost,
      R"({"unk":4,"pad":null,"bos":null,"eos":null,"cls":null,"sep":null,"mask":null})",
      R"([{"id":0,"piece":"A"},{"id":1,"piece":"AC"},{"id":2,"piece":"ACG"},{"id":3,"piece":"G"},{"id":4,"piece":"[UNK]"}])");
}

std::string kmer_bpe_manifest() {
  return manifest(
      "kmer-bpe", "[]", R"({"kind":"none"})",
      R"({"k":2,"stride":2,"tail":"error","unknown_policy":"unk","merges":[["AA","CG"]]})",
      kNoPost,
      R"({"unk":3,"pad":null,"bos":null,"eos":null,"cls":null,"sep":null,"mask":null})",
      R"([{"id":0,"piece":"AA"},{"id":1,"piece":"CG"},{"id":2,"piece":"AACG"},{"id":3,"piece":"[UNK]"}])");
}

std::string hf_whitespace_bpe_manifest() {
  return manifest(
      "bpe", "[]", R"({"kind":"hf-whitespace-ascii"})",
      R"({"merges":[["A","C"]],"literal_token_ids":[6]})", kNoPost,
      R"({"unk":5,"pad":null,"bos":null,"eos":null,"cls":null,"sep":null,"mask":null})",
      R"([{"id":0,"piece":"A"},{"id":1,"piece":"C"},{"id":2,"piece":"!"},{"id":3,"piece":"?"},{"id":4,"piece":"AC"},{"id":5,"piece":"[UNK]"},{"id":6,"piece":"[BPE]"}])");
}

std::string dna_fixed_kmer_manifest() {
  return manifest(
      "kmer", "[]", R"({"kind":"none"})",
      R"({"k":2,"stride":2,"tail":"lookup","unknown_policy":"unk","match_special_literals":true})",
      kNoPost,
      R"({"unk":1,"pad":null,"bos":0,"eos":null,"cls":null,"sep":null,"mask":null})",
      R"([{"id":0,"piece":"<R>"},{"id":1,"piece":"[UNK]"},{"id":2,"piece":"AA"},{"id":3,"piece":"A"}])");
}

std::string manual_bos_kmer_manifest() {
  return manifest(
      "kmer", "[]", R"({"kind":"none"})",
      R"({"k":6,"stride":6,"tail":"lookup","unknown_policy":"unk","match_special_literals":true})",
      R"({"prefix_ids":[1],"suffix_ids":[],"padding":{"side":"none","pad_id":null}})",
      R"({"unk":0,"pad":null,"bos":1,"eos":null,"cls":null,"sep":null,"mask":null})",
      R"([{"id":0,"piece":"<oov>"},{"id":1,"piece":"<s>"},{"id":2,"piece":"AAAAAA"},{"id":3,"piece":"TTTTTT"}])");
}

std::string byte_bpe_manifest() {
  std::string vocab = "[";
  std::string encoder = "[";
  for (std::size_t index = 0; index < 256; ++index) {
    if (index != 0) {
      vocab.push_back(',');
      encoder.push_back(',');
    }
    const auto piece = "b" + std::to_string(1000 + index).substr(1);
    vocab +=
        "{\"id\":" + std::to_string(index) + ",\"piece\":\"" + piece + "\"}";
    encoder += "\"" + piece + "\"";
  }
  vocab += R"(,{"id":256,"piece":"b065b067"}])";
  encoder += "]";
  const auto model = "{\"add_prefix_space\":false,\"byte_encoder\":" + encoder +
                     ",\"merges\":[[\"b065\",\"b067\"]]}";
  return manifest("byte-bpe", "[]", R"({"kind":"whole-input"})", model, kNoPost,
                  kNoSpecials, vocab);
}

std::string unknown_kind_manifest() {
  return manifest("mystery", "[]", R"({"kind":"none"})", "{}", kNoPost,
                  kNoSpecials, R"([{"id":0,"piece":"A"}])");
}

std::string duplicate_vocab_manifest() {
  return manifest(
      "character", "[]", R"({"kind":"none"})",
      R"({"unknown_policy":"error","match_special_literals":false})", kNoPost,
      kNoSpecials, R"([{"id":0,"piece":"A"},{"id":0,"piece":"C"}])");
}

std::string duplicate_piece_manifest() {
  return manifest(
      "character", "[]", R"({"kind":"none"})",
      R"({"unknown_policy":"error","match_special_literals":false})", kNoPost,
      kNoSpecials, R"([{"id":0,"piece":"A"},{"id":1,"piece":"A"}])");
}

struct Case final {
  std::string name;
  std::string payload;
  std::string sha256;
};

std::vector<Case> cases() {
  return {
      {"character", character_manifest(),
       "5fc03a22324a14e25441db938d1e9727c2c636fc259156c667ab567781a09bc2"},
      {"single-nucleotide", single_nucleotide_manifest(),
       "0fcf0ef318e9f257ad6e6bf8a4974edc942fd491b9db45c313b07a0be7a9f558"},
      {"fixed-kmer", fixed_kmer_manifest(),
       "6811e1e93b89f55b4e5cc9429d830a84e2e9d30208778f090e4386749cfb26ab"},
      {"overlapping-kmer", overlapping_kmer_manifest(),
       "5ffb86eb7738007654026372bd7c3325ac095110d0cc213925b864daa8cc82ad"},
      {"wordpiece", wordpiece_manifest(),
       "08b1676af34036f543d37e92d8633dc7ea42585ec1628512e9866329ee15e06c"},
      {"gena-bpe", gena_bpe_manifest(),
       "7d31182d300b44d9c962f9f80e4d6f368cec58c1af7b1afc65ca0d1476a1f9a5"},
      {"omnina-bpe", omnina_bpe_manifest(),
       "0959a519d0ecbc1f8f910e32bd647a1a9577c78acf6347018df893417a91cd56"},
      {"byte-bpe", byte_bpe_manifest(),
       "d4dc51edfd85c42b576f4696d146165ce7db29ef9fd579b12b3009810dcdc68b"},
      {"longest-match", longest_match_manifest(),
       "34c02667e6053f2ae5ad9a913c129fe19a78b1ae2d79f77e18ad5a8b286f9b47"},
      {"kmer-bpe", kmer_bpe_manifest(),
       "aa582787e95512dccf8e520be5daa829d10d9326a8152da4147068b9dbedde63"},
      {"unknown-kind", unknown_kind_manifest(),
       "db3ec9258f6848358975ef1cbddf0aed85b44f9784a79bfba14c35a88f9a1ad7"},
      {"duplicate-vocab", duplicate_vocab_manifest(),
       "13290f48127c0a257622b093cadded82cf03830a88c52b188eef85e52a21e3fe"},
      {"duplicate-piece", duplicate_piece_manifest(),
       "36044c03ebbd61f8483d8b6784166b3c833e0f8c015099d36386f9848c74547a"},
      {"hf-whitespace-bpe", hf_whitespace_bpe_manifest(),
       "d3bb493b75ceb2f907d318f7f1770d531433221cefef5f1d92606f6065b521e2"},
      {"dna-fixed-kmer", dna_fixed_kmer_manifest(),
       "87c93bc458fecc7becd9b38e3af7d9c02ca1ff5f818cf6c9279c07c3d0ed5311"},
      {"manual-bos-kmer", manual_bos_kmer_manifest(),
       "243805d28f920ec85519609d311cde394a1fff19770eb1711c2bfa15488d6b9d"},
  };
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "evo-tokenizers-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const auto *const created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      std::perror("mkdtemp");
      std::abort();
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

bool write_text(const std::filesystem::path &path,
                const std::string_view payload) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  return output.good();
}

bool parse_token_ids(const std::string_view text,
                     std::vector<evo::TokenId> *const output) {
  output->clear();
  std::size_t offset = 0;
  while (offset < text.size()) {
    const auto comma = text.find(',', offset);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    evo::TokenId value = 0;
    const auto parsed =
        std::from_chars(text.data() + offset, text.data() + end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + end)
      return false;
    output->push_back(value);
    if (comma == std::string_view::npos)
      return true;
    offset = comma + 1U;
    if (offset == text.size())
      return false;
  }
  return text.empty();
}

evo::Status load(const std::filesystem::path &root, const Case &fixture,
                 std::unique_ptr<evo::ArtifactTokenizer> *const output) {
  const auto path = root / (fixture.name + ".json");
  if (!write_text(path, fixture.payload))
    return {evo::ErrorCode::kIo, "test could not write tokenizer fixture"};
  return evo::ArtifactTokenizer::Load(
      root.string(),
      {path.filename().string(), fixture.sha256, fixture.payload.size()},
      output);
}

void expect_tokens(evo::ArtifactTokenizer *const tokenizer,
                   const std::string_view input,
                   const evo::TokenizerEncodeOptions &options,
                   const std::vector<evo::TokenId> &expected,
                   const std::string_view description) {
  std::vector<evo::TokenId> tokens;
  const auto status = tokenizer->encode(input, options, &tokens);
  check(status.ok(), std::string{description} + ": " + status.message());
  check(tokens == expected, description);
}

void run_vectors(const std::vector<Case> &fixtures,
                 const std::filesystem::path &root) {
  const auto run = [&](const std::size_t index, const std::string_view input,
                       const evo::TokenizerEncodeOptions &options,
                       const std::vector<evo::TokenId> &expected,
                       const std::string_view description) {
    std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
    const auto status = load(root, fixtures[index], &tokenizer);
    check(status.ok(), std::string{"load "} + fixtures[index].name + ": " +
                           status.message());
    if (status.ok())
      expect_tokens(tokenizer.get(), input, options, expected, description);
  };

  run(0, "ACX", {}, {0, 1, 2}, "character tokenizer vector");
  evo::TokenizerEncodeOptions padded;
  padded.pad_to_length = 9;
  run(1, "acgu?", padded, {7, 0, 1, 2, 3, 5, 8, 6, 6},
      "SN normalization, template, and right padding vector");
  run(2, "AACG", {}, {0, 1}, "fixed non-overlapping k-mer vector");
  run(3, "AACG", {}, {0, 1, 2}, "overlapping k-mer vector");
  run(4, "ACGT", {}, {2, 5, 7, 3}, "WordPiece longest-match vector");
  run(5, " ACNNNAC ", {}, {3, 2, 3},
      "GENA strip, byte-run, isolated-split BPE vector");
  run(5, "AAAA", {}, {6}, "BPE repeated-pair vector");
  run(5, "AAA", {}, {5, 0}, "BPE overlapping-pair vector");
  run(6, "AC", {}, {4}, "OmniNA UTF-8 prepend BPE vector");
  run(7, "AC", {}, {256}, "byte-level BPE vector");
  run(8, "ACGX", {}, {2, 4}, "longest-match trie vector");
  evo::TokenizerEncodeOptions merged_limit;
  merged_limit.token_limit = 1;
  run(9, "AACG", merged_limit, {2},
      "k-mer BPE applies token limit after merging");
  run(13, "AC!? AC", {}, {4, 2, 3, 4},
      "HF Whitespace ASCII word/punctuation boundaries");
  run(13, "[BPE]", {}, {6},
      "appended HF special literal is atomic before pre-tokenization");
  run(14, "<R>AAXA", {}, {0, 2, 1},
      "fixed k-mer keeps an invalid chunk atomic");
  run(14, "<R>AAA", {}, {0, 2, 3},
      "fixed k-mer looks up a short trailing piece");

  std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
  auto status = load(root, fixtures[13], &tokenizer);
  check(status.ok(), "load HF Whitespace tokenizer for non-ASCII gate");
  if (status.ok()) {
    std::vector<evo::TokenId> tokens;
    status = tokenizer->encode("AC\xC2\xA0"
                               "AC",
                               {}, &tokens);
    check(!status.ok() && status.code() == evo::ErrorCode::kUnsupported,
          "HF Whitespace ASCII subset rejects non-ASCII input");
  }
}

void run_large_sn_gate(const Case &fixture, const std::filesystem::path &root) {
  std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
  auto status = load(root, fixture, &tokenizer);
  check(status.ok(), "load SN tokenizer for one-million-base gate");
  if (!status.ok())
    return;
  const std::string sequence(1'000'000, 'a');
  evo::TokenizerEncodeOptions options;
  options.add_special_tokens = false;
  options.raw_byte_limit = sequence.size();
  options.token_limit = sequence.size();
  std::vector<evo::TokenId> tokens;
  status = tokenizer->encode(sequence, options, &tokens);
  check(status.ok(), "one-million-base SN encode completes");
  check(tokens.size() == sequence.size(),
        "one-million-base SN encode emits exactly one token per input byte");
  check(!tokens.empty() && tokens.front() == 0 && tokens.back() == 0,
        "one-million-base SN encode preserves first and final token");
}

void run_manual_cls_sep_gate(const Case &fixture,
                             const std::filesystem::path &root) {
  std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
  auto status = load(root, fixture, &tokenizer);
  check(status.ok(), "load tokenizer for manual CLS-SEP gate");
  if (!status.ok())
    return;
  evo::GenebEmbeddingArtifactSpec spec;
  spec.input_transform.raw_safety_cap = 32U;
  spec.input_transform.case_policy = evo::GenebCasePolicy::kPreserve;
  spec.input_transform.invalid_base_policy =
      evo::GenebInvalidBasePolicy::kTokenizerDefined;
  spec.input_transform.special_token_policy =
      evo::GenebSpecialTokenPolicy::kManualClsThenSepBeforeSequence;
  spec.input_transform.prefix = "[CLS][SEP]";
  spec.input_transform.token_truncation = evo::GenebTokenTruncation::kRight;
  spec.input_transform.context_unit = evo::GenebContextUnit::kTokens;
  spec.input_transform.length_policy =
      evo::GenebLengthPolicy::kTokenizerTruncate;
  spec.input_transform.reference_context_limit = 4U;
  evo::GenebPreparedEmbeddingInput prepared;
  status = evo::prepare_geneb_embedding_input("AC", spec, *tokenizer, &prepared);
  check(status.ok(), "manual CLS-SEP GENEB input prepares");
  check(prepared.tokens == std::vector<evo::TokenId>({7U, 8U, 0U, 1U}) &&
            prepared.attention_mask ==
                std::vector<std::uint8_t>({1U, 1U, 1U, 1U}) &&
            prepared.token_plan.prefix_token_count == 2U &&
            prepared.token_plan.suffix_token_count == 0U,
        "manual CLS-SEP policy places both boundary tokens before payload");
}

void run_manual_bos_gate(const Case &fixture, const Case &no_default_fixture,
                         const std::filesystem::path &root) {
  std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
  auto status = load(root, fixture, &tokenizer);
  check(status.ok(), "load tokenizer for manual BOS gate");
  if (!status.ok())
    return;
  evo::GenebEmbeddingArtifactSpec spec;
  spec.input_transform.raw_safety_cap = 64U;
  spec.input_transform.case_policy = evo::GenebCasePolicy::kPreserve;
  spec.input_transform.invalid_base_policy =
      evo::GenebInvalidBasePolicy::kTokenizerDefined;
  spec.input_transform.special_token_policy =
      evo::GenebSpecialTokenPolicy::kManualBosPlusTokenizerDefault;
  spec.input_transform.prefix = "tokenizer-bos-text";
  spec.input_transform.token_truncation = evo::GenebTokenTruncation::kRight;
  spec.input_transform.context_unit = evo::GenebContextUnit::kTokens;
  spec.input_transform.length_policy =
      evo::GenebLengthPolicy::kTokenizerTruncate;
  spec.input_transform.reference_context_limit = 16U;

  evo::GenebPreparedEmbeddingInput prepared;
  status = evo::prepare_geneb_embedding_input("ACGTNACGTNAC", spec, *tokenizer,
                                              &prepared);
  check(status.ok() &&
            prepared.tokens == std::vector<evo::TokenId>({1U, 1U, 0U, 0U}) &&
            prepared.token_plan.prefix_token_count == 2U &&
            prepared.token_plan.source_offset == 2U,
        "B118 preserves tokenizer-default and manual BOS before unknown kmers");
  status = evo::prepare_geneb_embedding_input("AAAAAATTTTTT", spec, *tokenizer,
                                              &prepared);
  check(status.ok() &&
            prepared.tokens == std::vector<evo::TokenId>({1U, 1U, 2U, 3U}) &&
            prepared.token_plan.prefix_token_count == 2U,
        "B118 preserves tokenizer-default and manual BOS before valid kmers");

  auto corrupted = spec;
  corrupted.input_transform.prefix = "<s>";
  status = evo::prepare_geneb_embedding_input("AAAAAA", corrupted, *tokenizer,
                                              &prepared);
  check(!status.ok() && status.code() == evo::ErrorCode::kInvalidArgument &&
            status.message().find("tokenizer-bos-text") != std::string::npos,
        "B118 rejects a literal in place of the symbolic BOS marker");

  status = load(root, no_default_fixture, &tokenizer);
  check(status.ok(), "load no-default tokenizer for manual BOS corruption gate");
  if (status.ok()) {
    status = evo::prepare_geneb_embedding_input("AAAAAA", spec, *tokenizer,
                                                &prepared);
    check(!status.ok() && status.code() == evo::ErrorCode::kModelFormat &&
              status.message().find("one tokenizer prefix token") !=
                  std::string::npos,
          "B118 rejects manual BOS without one tokenizer-default prefix");
  }
}

void run_failure_gates(const std::vector<Case> &fixtures,
                       const std::filesystem::path &root) {
  std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
  auto status = load(root, fixtures[10], &tokenizer);
  check(!status.ok() && status.code() == evo::ErrorCode::kUnsupported,
        "unknown tokenizer kind fails closed");
  status = load(root, fixtures[11], &tokenizer);
  check(!status.ok() && status.code() == evo::ErrorCode::kModelFormat &&
            status.message().find("duplicate vocab ID") != std::string::npos,
        "duplicate vocab ID is rejected after hash verification");
  status = load(root, fixtures[12], &tokenizer);
  check(!status.ok() && status.code() == evo::ErrorCode::kModelFormat &&
            status.message().find("duplicate vocab piece") != std::string::npos,
        "duplicate vocab piece is rejected after hash verification");

  const auto path = root / "corrupted.json";
  check(write_text(path, fixtures[0].payload), "write corruption fixture");
  evo::TokenizerAssetDescriptor descriptor{
      path.filename().string(), fixtures[0].sha256, fixtures[0].payload.size()};
  check(write_text(path, fixtures[0].payload + " "), "corrupt fixture bytes");
  descriptor.size += 1;
  status = evo::ArtifactTokenizer::Load(root.string(), descriptor, &tokenizer);
  check(!status.ok() &&
            status.message().find("SHA256 differs") != std::string::npos,
        "tokenizer asset corruption is rejected before parse");

  descriptor = {"../escape.json", fixtures[0].sha256,
                fixtures[0].payload.size()};
  status = evo::ArtifactTokenizer::Load(root.string(), descriptor, &tokenizer);
  check(!status.ok(), "parent traversal in tokenizer asset path is rejected");
  descriptor = {"/absolute.json", fixtures[0].sha256,
                fixtures[0].payload.size()};
  status = evo::ArtifactTokenizer::Load(root.string(), descriptor, &tokenizer);
  check(!status.ok(), "absolute tokenizer asset path is rejected");

  const auto nested = root / "nested";
  std::filesystem::create_directory(nested);
  check(write_text(nested / "asset.json", fixtures[0].payload),
        "write nested tokenizer asset");
  std::error_code error;
  std::filesystem::create_directory_symlink(nested, root / "link", error);
  check(!error, "create tokenizer intermediate symlink fixture");
  descriptor = {"link/asset.json", fixtures[0].sha256,
                fixtures[0].payload.size()};
  status = evo::ArtifactTokenizer::Load(root.string(), descriptor, &tokenizer);
  check(!status.ok() &&
            status.message().find("symbolic link") != std::string::npos,
        "intermediate tokenizer asset symlink is rejected");

  status = load(root, fixtures[0], &tokenizer);
  check(status.ok(), "reload character tokenizer for transactional gate");
  if (status.ok()) {
    std::vector<evo::TokenId> tokens{999};
    evo::TokenizerEncodeOptions options;
    options.token_limit = 2;
    status = tokenizer->encode("ACA", options, &tokens);
    check(!status.ok() && tokens == std::vector<evo::TokenId>{999},
          "failed artifact encode leaves caller output unchanged");
  }
  status = load(root, fixtures[1], &tokenizer);
  check(status.ok(), "reload SN tokenizer for padding safety gate");
  if (status.ok()) {
    evo::TokenizerEncodeOptions options;
    options.pad_to_length = (16U << 20U) + 1U;
    std::vector<evo::TokenId> tokens;
    status = tokenizer->encode("A", options, &tokens);
    check(!status.ok() && status.message().find("padding") != std::string::npos,
          "padding has an independent hard allocation bound");
  }
}

} // namespace

int main(const int argc, char **argv) {
  const auto fixtures = cases();
  if (argc == 3 && std::string_view{argv[1]} == "--write-fixtures") {
    const std::filesystem::path output{argv[2]};
    std::filesystem::create_directories(output);
    for (const auto &fixture : fixtures) {
      if (!write_text(output / (fixture.name + ".json"), fixture.payload))
        return 1;
    }
    return 0;
  }
  if (argc == 8 &&
      (std::string_view{argv[1]} == "--verify-asset" ||
       std::string_view{argv[1]} == "--verify-asset-no-special" ||
       std::string_view{argv[1]} == "--verify-manual-bos")) {
    std::uint64_t size = 0;
    const std::string_view size_text{argv[5]};
    const auto parsed_size = std::from_chars(
        size_text.data(), size_text.data() + size_text.size(), size);
    if (parsed_size.ec != std::errc{} ||
        parsed_size.ptr != size_text.data() + size_text.size())
      return 1;
    std::vector<evo::TokenId> expected;
    if (!parse_token_ids(argv[7], &expected))
      return 2;
    std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
    const auto descriptor =
        evo::TokenizerAssetDescriptor{argv[3], argv[4], size};
    auto status = evo::ArtifactTokenizer::Load(argv[2], descriptor, &tokenizer);
    if (!status.ok()) {
      std::cerr << status.message() << '\n';
      return 1;
    }
    std::vector<evo::TokenId> actual;
    if (std::string_view{argv[1]} == "--verify-manual-bos") {
      evo::GenebEmbeddingArtifactSpec spec;
      spec.input_transform.raw_safety_cap = 1U << 20U;
      spec.input_transform.case_policy = evo::GenebCasePolicy::kPreserve;
      spec.input_transform.invalid_base_policy =
          evo::GenebInvalidBasePolicy::kTokenizerDefined;
      spec.input_transform.frame_trim_enabled = true;
      spec.input_transform.frame_multiple = 6U;
      spec.input_transform.frame_trim_side = evo::GenebTrimSide::kRight;
      spec.input_transform.special_token_policy =
          evo::GenebSpecialTokenPolicy::kManualBosPlusTokenizerDefault;
      spec.input_transform.prefix = "tokenizer-bos-text";
      spec.input_transform.token_truncation =
          evo::GenebTokenTruncation::kRight;
      spec.input_transform.context_unit = evo::GenebContextUnit::kTokens;
      spec.input_transform.length_policy =
          evo::GenebLengthPolicy::kTokenizerTruncate;
      spec.input_transform.reference_context_limit = 1024U;
      evo::GenebPreparedEmbeddingInput prepared;
      status = evo::prepare_geneb_embedding_input(argv[6], spec, *tokenizer,
                                                  &prepared);
      if (status.ok() && prepared.token_plan.prefix_token_count != 2U)
        status = {evo::ErrorCode::kInternal,
                  "manual BOS prefix token count differs"};
      actual = std::move(prepared.tokens);
    } else {
      evo::TokenizerEncodeOptions options;
      options.add_special_tokens =
          std::string_view{argv[1]} != "--verify-asset-no-special";
      status = tokenizer->encode(argv[6], options, &actual);
    }
    if (!status.ok() || actual != expected) {
      std::cerr << (status.ok() ? "token vector differs" : status.message())
                << '\n';
      return 1;
    }
    return 0;
  }
  if (argc == 7 &&
      std::string_view{argv[1]} == "--verify-asset-vectors") {
    std::uint64_t size = 0;
    const std::string_view size_text{argv[5]};
    const auto parsed_size = std::from_chars(
        size_text.data(), size_text.data() + size_text.size(), size);
    if (parsed_size.ec != std::errc{} ||
        parsed_size.ptr != size_text.data() + size_text.size())
      return 1;
    std::unique_ptr<evo::ArtifactTokenizer> tokenizer;
    const auto descriptor =
        evo::TokenizerAssetDescriptor{argv[3], argv[4], size};
    auto status = evo::ArtifactTokenizer::Load(argv[2], descriptor, &tokenizer);
    if (!status.ok()) {
      std::cerr << status.message() << '\n';
      return 1;
    }
    std::ifstream vectors(argv[6], std::ios::binary);
    if (!vectors.good()) {
      std::cerr << "cannot open tokenizer vector file\n";
      return 1;
    }
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(vectors, line)) {
      ++line_number;
      const auto separator = line.find('\t');
      if (separator == std::string::npos ||
          line.find('\t', separator + 1U) != std::string::npos) {
        std::cerr << "invalid tokenizer vector line " << line_number << '\n';
        return 2;
      }
      std::vector<evo::TokenId> expected;
      if (!parse_token_ids(std::string_view{line}.substr(separator + 1U),
                           &expected)) {
        std::cerr << "invalid tokenizer IDs on line " << line_number << '\n';
        return 2;
      }
      std::vector<evo::TokenId> actual;
      status = tokenizer->encode(std::string_view{line}.substr(0U, separator),
                                 {}, &actual);
      if (!status.ok() || actual != expected) {
        std::cerr << "tokenizer vector differs on line " << line_number << ": "
                  << (status.ok() ? "token IDs differ" : status.message())
                  << '\n';
        return 1;
      }
    }
    if (!vectors.eof() || line_number == 0U) {
      std::cerr << "cannot read tokenizer vector file\n";
      return 1;
    }
    return 0;
  }
  if (argc != 1) {
    std::cerr << "usage: test_tokenizer_assets [--write-fixtures DIR | "
                 "--verify-asset ROOT PATH SHA256 SIZE INPUT IDS | "
                 "--verify-asset-no-special ROOT PATH SHA256 SIZE INPUT IDS | "
                 "--verify-manual-bos ROOT PATH SHA256 SIZE INPUT IDS | "
                 "--verify-asset-vectors ROOT PATH SHA256 SIZE VECTORS]\n";
    return 2;
  }
  TemporaryDirectory temporary;
  run_vectors(fixtures, temporary.path());
  run_large_sn_gate(fixtures[1], temporary.path());
  run_manual_cls_sep_gate(fixtures[1], temporary.path());
  run_manual_bos_gate(fixtures[15], fixtures[14], temporary.path());
  run_failure_gates(fixtures, temporary.path());
  if (failures != 0) {
    std::cerr << failures << " tokenizer asset test(s) failed\n";
    return 1;
  }
  std::cout << "tokenizer asset tests passed\n";
  return 0;
}
