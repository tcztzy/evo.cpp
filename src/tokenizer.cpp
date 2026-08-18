// SPDX-License-Identifier: Apache-2.0
#include "evo/tokenizer.hpp"

#include "evo/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace evo {

std::vector<TokenId> encode_bytes(const std::string_view bytes) {
  std::vector<TokenId> tokens;
  tokens.reserve(bytes.size());
  for (const char character : bytes) {
    tokens.push_back(
        static_cast<TokenId>(static_cast<unsigned char>(character)));
  }
  return tokens;
}

Status token_to_byte(const TokenId token, std::uint8_t *const byte) {
  if (byte == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "token_to_byte output pointer is null"};
  }
  if (token > std::numeric_limits<std::uint8_t>::max()) {
    return {ErrorCode::kInvalidArgument,
            "token " + std::to_string(token) + " has no byte representation"};
  }
  *byte = static_cast<std::uint8_t>(token);
  return Status::Ok();
}

Status encode_sequence(const ArchitectureTokenizer tokenizer,
                       const std::string_view sequence,
                       std::vector<TokenId> *const tokens) {
  if (tokens == nullptr)
    return {ErrorCode::kInvalidArgument, "token output is null"};
  if (tokenizer == ArchitectureTokenizer::kArtifact) {
    return {ErrorCode::kUnsupported,
            "architecture requires a verified artifact tokenizer"};
  }
  if (tokenizer == ArchitectureTokenizer::kByteIdentity) {
    *tokens = encode_bytes(sequence);
    return Status::Ok();
  }
  if (tokenizer == ArchitectureTokenizer::kEsmcProtein) {
    constexpr std::array<std::string_view, 6> special{
        "<cls>", "<pad>", "<eos>", "<unk>", "|", "<mask>"};
    constexpr std::array<TokenId, 6> special_ids{0, 1, 2, 3, 31, 32};
    constexpr std::string_view alphabet = "LAGVSERTIDPKQNFYMHWCXBUZO.-|";
    tokens->clear();
    tokens->reserve(sequence.size() + 2);
    tokens->push_back(0);
    std::size_t offset = 0;
    while (offset < sequence.size()) {
      bool matched = false;
      for (std::size_t index = 0; index < special.size(); ++index) {
        if (sequence.substr(offset, special[index].size()) == special[index]) {
          tokens->push_back(special_ids[index]);
          offset += special[index].size();
          matched = true;
          break;
        }
      }
      if (matched)
        continue;
      const auto found = alphabet.find(sequence[offset]);
      tokens->push_back(found == std::string_view::npos
                            ? TokenId{3}
                            : static_cast<TokenId>(4 + found));
      ++offset;
    }
    tokens->push_back(2);
    return Status::Ok();
  }
  tokens->clear();
  tokens->reserve(sequence.size());
  for (const char character : sequence) {
    TokenId token = 6;
    switch (character) {
    case 'A':
      token = 7;
      break;
    case 'C':
      token = 8;
      break;
    case 'G':
      token = 9;
      break;
    case 'T':
      token = 10;
      break;
    case 'N':
      token = 11;
      break;
    default:
      break;
    }
    tokens->push_back(token);
  }
  return Status::Ok();
}

Status decode_sequence_token(const ArchitectureTokenizer tokenizer,
                             const TokenId token, std::uint8_t *const byte) {
  if (tokenizer == ArchitectureTokenizer::kArtifact) {
    return {ErrorCode::kUnsupported,
            "artifact tokenizer detokenization is not supported"};
  }
  if (tokenizer == ArchitectureTokenizer::kByteIdentity)
    return token_to_byte(token, byte);
  if (byte == nullptr)
    return {ErrorCode::kInvalidArgument, "byte output is null"};
  if (tokenizer == ArchitectureTokenizer::kEsmcProtein) {
    constexpr std::string_view alphabet = "LAGVSERTIDPKQNFYMHWCXBUZO.-|";
    if (token < 4 || token >= 4 + alphabet.size())
      return {ErrorCode::kModelFormat,
              "ESMC token has no raw protein byte representation"};
    *byte = static_cast<std::uint8_t>(alphabet[token - 4]);
    return Status::Ok();
  }
  constexpr char alphabet[] = "ACGTN";
  if (token < 7 || token > 11)
    return {ErrorCode::kModelFormat,
            "HyenaDNA token has no raw DNA byte representation"};
  *byte = static_cast<std::uint8_t>(alphabet[token - 7]);
  return Status::Ok();
}

namespace {

namespace fs = std::filesystem;

constexpr std::size_t kMaximumTokenizerAssetBytes = 64U << 20U;
constexpr std::size_t kMaximumNormalizationLiteralBytes = 4096U;
constexpr std::size_t kMaximumNormalizedBytes = 1U << 30U;
constexpr std::size_t kMaximumIntermediateTokens = 16U << 20U;

enum class NormalizationKind {
  kAsciiUppercase,
  kAsciiLowercase,
  kUToT,
  kStripAsciiWhitespace,
  kPrependLiteral,
  kReplaceLiteral,
  kReplaceByteRun,
};

struct NormalizationOp final {
  NormalizationKind kind{NormalizationKind::kAsciiUppercase};
  std::string first;
  std::string second;
  unsigned char byte{0};
  std::size_t minimum_count{0};
};

enum class PretokenizerKind {
  kNone,
  kWholeInput,
  kAsciiWhitespace,
  kHfWhitespaceAscii,
  kSplitIsolated,
};

enum class UnknownPolicy { kError, kUnk };
enum class TailPolicy { kDrop, kError, kUnk, kLookup };
enum class PaddingSide { kNone, kLeft, kRight };

struct TokenTrieNode final {
  std::map<unsigned char, std::size_t> children;
  std::optional<TokenId> token;
};

struct BpeMergeRule final {
  std::size_t rank{0};
  TokenId result{0};
};

struct TokenizerData final {
  ArtifactTokenizerKind kind{ArtifactTokenizerKind::kCharacter};
  std::vector<NormalizationOp> normalization;
  PretokenizerKind pretokenizer{PretokenizerKind::kNone};
  std::string split_literal;
  std::map<std::string, TokenId, std::less<>> vocab;
  std::map<TokenId, std::string> reverse_vocab;
  std::array<std::optional<TokenId>, 7> specials{};
  std::vector<TokenId> prefix_ids;
  std::vector<TokenId> suffix_ids;
  std::vector<TokenId> literal_token_ids;
  PaddingSide padding_side{PaddingSide::kNone};
  std::optional<TokenId> pad_id;
  UnknownPolicy unknown_policy{UnknownPolicy::kError};
  bool match_special_literals{false};
  std::size_t k{0};
  std::size_t stride{0};
  TailPolicy tail_policy{TailPolicy::kDrop};
  std::string continuation_prefix;
  std::size_t maximum_word_bytes{0};
  bool add_prefix_space{false};
  std::array<TokenId, 256> byte_encoder{};
  std::map<std::pair<TokenId, TokenId>, BpeMergeRule> merge_rules;
  std::vector<TokenTrieNode> longest_match_trie;
  std::size_t vocabulary_size{0};
};

class Sha256 final {
public:
  void update(const std::uint8_t *data, std::size_t size) noexcept {
    while (size != 0) {
      const auto copied = std::min(size, block_.size() - block_size_);
      std::copy_n(data, copied,
                  block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
      block_size_ += copied;
      data += copied;
      size -= copied;
      if (block_size_ == block_.size()) {
        transform(block_.data());
        bit_count_ += 512U;
        block_size_ = 0;
      }
    }
  }

  [[nodiscard]] std::array<std::uint8_t, 32> finish() noexcept {
    bit_count_ += static_cast<std::uint64_t>(block_size_) * 8U;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56U) {
      std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                block_.end(), 0U);
      transform(block_.data());
      block_size_ = 0;
    }
    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
              block_.begin() + 56, 0U);
    for (std::size_t index = 0; index < 8; ++index) {
      block_[63U - index] =
          static_cast<std::uint8_t>(bit_count_ >> (index * 8U));
    }
    transform(block_.data());
    std::array<std::uint8_t, 32> digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4; ++byte) {
        digest[word * 4U + byte] =
            static_cast<std::uint8_t>(state_[word] >> ((3U - byte) * 8U));
      }
    }
    return digest;
  }

private:
  static std::uint32_t rotate_right(const std::uint32_t value,
                                    const unsigned shift) noexcept {
    return (value >> shift) | (value << (32U - shift));
  }

  void transform(const std::uint8_t *const block) noexcept {
    static constexpr std::array<std::uint32_t, 64> constants{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
        0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
        0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
        0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
        0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
        0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
        0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
        0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const auto offset = index * 4U;
      words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                     (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                     static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const auto s0 = rotate_right(words[index - 15U], 7U) ^
                      rotate_right(words[index - 15U], 18U) ^
                      (words[index - 15U] >> 3U);
      const auto s1 = rotate_right(words[index - 2U], 17U) ^
                      rotate_right(words[index - 2U], 19U) ^
                      (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }
    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const auto sum1 =
          rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
      const auto choose = (e & f) ^ (~e & g);
      const auto first = h + sum1 + choose + constants[index] + words[index];
      const auto sum0 =
          rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto second = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                      0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                      0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_{0};
  std::uint64_t bit_count_{0};
};

std::string hexadecimal(const std::array<std::uint8_t, 32> &digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto byte : digest)
    output << std::setw(2) << static_cast<unsigned>(byte);
  return output.str();
}

bool canonical_sha256(const std::string_view value) noexcept {
  if (value.size() != 64)
    return false;
  return std::all_of(value.begin(), value.end(), [](const char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

Status tokenizer_format_error(const std::string_view message) {
  return {ErrorCode::kModelFormat, "tokenizer asset: " + std::string{message}};
}

Status read_and_hash(const fs::path &path, const std::size_t size,
                     std::string *const bytes, std::string *const digest) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {ErrorCode::kIo,
            "cannot open tokenizer asset '" + path.string() + "'"};
  bytes->clear();
  bytes->reserve(size);
  Sha256 sha256;
  // Artifact loads can occur on server worker threads whose platform stack is
  // substantially smaller than the process main stack. Keep the bounded I/O
  // scratch allocation off-stack.
  std::vector<char> buffer(1U << 20U);
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      const auto amount = static_cast<std::size_t>(count);
      if (bytes->size() > size || amount > size - bytes->size())
        return tokenizer_format_error("file grew after size validation");
      bytes->append(buffer.data(), amount);
      sha256.update(reinterpret_cast<const std::uint8_t *>(buffer.data()),
                    amount);
    }
  }
  if (!input.eof())
    return {ErrorCode::kIo,
            "cannot read tokenizer asset '" + path.string() + "'"};
  if (bytes->size() != size)
    return tokenizer_format_error("file size changed while reading");
  *digest = hexadecimal(sha256.finish());
  return Status::Ok();
}

bool path_is_beneath(const fs::path &root, const fs::path &candidate) {
  auto root_iterator = root.begin();
  auto candidate_iterator = candidate.begin();
  for (; root_iterator != root.end(); ++root_iterator, ++candidate_iterator) {
    if (candidate_iterator == candidate.end() ||
        *root_iterator != *candidate_iterator)
      return false;
  }
  return true;
}

Status resolve_asset_path(const std::string &artifact_root,
                          const TokenizerAssetDescriptor &descriptor,
                          fs::path *const output) {
  if (artifact_root.empty())
    return {ErrorCode::kInvalidArgument, "tokenizer artifact root is empty"};
  if (descriptor.path.empty())
    return tokenizer_format_error("descriptor path is empty");
  const fs::path relative{descriptor.path};
  if (relative.is_absolute() || relative.has_root_name() ||
      relative.has_root_directory())
    return tokenizer_format_error("descriptor path must be relative");
  for (const auto &component : relative) {
    if (component.empty() || component == "." || component == "..")
      return tokenizer_format_error("descriptor path is not canonical");
  }
  if (descriptor.size == 0 || descriptor.size > kMaximumTokenizerAssetBytes)
    return tokenizer_format_error(
        "descriptor size is outside the supported range");
  if (!canonical_sha256(descriptor.sha256))
    return tokenizer_format_error(
        "descriptor SHA256 is not canonical lowercase hex");

  std::error_code error;
  const auto root = fs::weakly_canonical(fs::path{artifact_root}, error);
  if (error)
    return {ErrorCode::kIo,
            "cannot resolve tokenizer artifact root: " + error.message()};
  auto inspected = root;
  for (const auto &component : relative) {
    inspected /= component;
    const auto component_status = fs::symlink_status(inspected, error);
    if (error)
      return {ErrorCode::kIo,
              "cannot inspect tokenizer asset path component '" +
                  inspected.string() + "': " + error.message()};
    if (fs::is_symlink(component_status))
      return tokenizer_format_error(
          "descriptor path contains a symbolic link component");
  }
  const auto candidate = root / relative;
  const auto link_status = fs::symlink_status(candidate, error);
  if (error)
    return {ErrorCode::kIo, "cannot inspect tokenizer asset '" +
                                candidate.string() + "': " + error.message()};
  if (fs::is_symlink(link_status) || !fs::is_regular_file(link_status))
    return tokenizer_format_error(
        "descriptor path must name a regular non-symlink file");
  const auto canonical = fs::canonical(candidate, error);
  if (error)
    return {ErrorCode::kIo, "cannot resolve tokenizer asset '" +
                                candidate.string() + "': " + error.message()};
  if (!path_is_beneath(root, canonical))
    return tokenizer_format_error("descriptor path escapes the artifact root");
  const auto actual_size = fs::file_size(canonical, error);
  if (error)
    return {ErrorCode::kIo, "cannot stat tokenizer asset '" +
                                canonical.string() + "': " + error.message()};
  if (actual_size != descriptor.size)
    return tokenizer_format_error("descriptor file size differs");
  *output = canonical;
  return Status::Ok();
}

Status exact_keys(const JsonValue &object,
                  const std::initializer_list<std::string_view> expected,
                  const std::string_view context) {
  if (object.type != JsonType::kObject)
    return tokenizer_format_error(std::string{context} + " must be an object");
  std::set<std::string, std::less<>> names;
  for (const auto name : expected)
    names.emplace(name);
  if (object.object.size() != names.size())
    return tokenizer_format_error(std::string{context} +
                                  " has missing or unknown fields");
  for (const auto &member : object.object) {
    if (names.find(member.first) == names.end())
      return tokenizer_format_error(
          std::string{context} + " has unknown field '" + member.first + "'");
  }
  return Status::Ok();
}

const JsonValue *required(const JsonValue &object, const std::string_view key) {
  return object.find(key);
}

Status required_string(const JsonValue &object, const std::string_view key,
                       const std::string_view context,
                       std::string *const output) {
  const auto *const value = required(object, key);
  if (value == nullptr || value->type != JsonType::kString)
    return tokenizer_format_error(std::string{context} + "." +
                                  std::string{key} + " must be a string");
  *output = value->string;
  return Status::Ok();
}

Status required_bool(const JsonValue &object, const std::string_view key,
                     const std::string_view context, bool *const output) {
  const auto *const value = required(object, key);
  if (value == nullptr || value->type != JsonType::kBoolean)
    return tokenizer_format_error(std::string{context} + "." +
                                  std::string{key} + " must be a boolean");
  *output = value->boolean;
  return Status::Ok();
}

Status unsigned_integer(const JsonValue &value, const std::uint64_t maximum,
                        const std::string_view context,
                        std::uint64_t *const output) {
  if (value.type != JsonType::kNumber || !std::isfinite(value.number) ||
      value.number < 0.0 || std::floor(value.number) != value.number ||
      value.number > static_cast<double>(maximum))
    return tokenizer_format_error(std::string{context} +
                                  " must be an unsigned integer");
  *output = static_cast<std::uint64_t>(value.number);
  return Status::Ok();
}

Status required_size(const JsonValue &object, const std::string_view key,
                     const std::size_t minimum, const std::size_t maximum,
                     const std::string_view context,
                     std::size_t *const output) {
  const auto *const value = required(object, key);
  std::uint64_t parsed = 0;
  if (value == nullptr)
    return tokenizer_format_error(std::string{context} + "." +
                                  std::string{key} + " is required");
  auto status = unsigned_integer(*value, maximum, context, &parsed);
  if (!status.ok())
    return status;
  if (parsed < minimum)
    return tokenizer_format_error(std::string{context} + "." +
                                  std::string{key} + " is too small");
  *output = static_cast<std::size_t>(parsed);
  return Status::Ok();
}

Status token_id(const JsonValue &value, const std::string_view context,
                TokenId *const output) {
  std::uint64_t parsed = 0;
  auto status = unsigned_integer(value, std::numeric_limits<TokenId>::max(),
                                 context, &parsed);
  if (!status.ok())
    return status;
  *output = static_cast<TokenId>(parsed);
  return Status::Ok();
}

bool is_ascii_space(const unsigned char byte) noexcept {
  return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' ||
         byte == '\f' || byte == '\v';
}

Status parse_kind(const std::string_view name,
                  ArtifactTokenizerKind *const output) {
  if (name == "character")
    *output = ArtifactTokenizerKind::kCharacter;
  else if (name == "single-nucleotide")
    *output = ArtifactTokenizerKind::kSingleNucleotide;
  else if (name == "kmer")
    *output = ArtifactTokenizerKind::kKmer;
  else if (name == "wordpiece")
    *output = ArtifactTokenizerKind::kWordPiece;
  else if (name == "bpe")
    *output = ArtifactTokenizerKind::kBpe;
  else if (name == "byte-bpe")
    *output = ArtifactTokenizerKind::kByteBpe;
  else if (name == "longest-match")
    *output = ArtifactTokenizerKind::kLongestMatch;
  else if (name == "kmer-bpe")
    *output = ArtifactTokenizerKind::kKmerBpe;
  else
    return {ErrorCode::kUnsupported,
            "unsupported tokenizer asset kind '" + std::string{name} + "'"};
  return Status::Ok();
}

Status bounded_literal(const std::string &value, const std::string_view context,
                       const bool allow_empty = true) {
  if ((!allow_empty && value.empty()) ||
      value.size() > kMaximumNormalizationLiteralBytes)
    return tokenizer_format_error(std::string{context} +
                                  " has an invalid literal length");
  return Status::Ok();
}

Status parse_normalization(const JsonValue &value,
                           std::vector<NormalizationOp> *const output) {
  if (value.type != JsonType::kArray)
    return tokenizer_format_error("normalization must be an array");
  if (value.array.size() > 32)
    return tokenizer_format_error("normalization has too many operations");
  output->clear();
  output->reserve(value.array.size());
  for (std::size_t index = 0; index < value.array.size(); ++index) {
    const auto &entry = value.array[index];
    if (entry.type != JsonType::kObject)
      return tokenizer_format_error("normalization entry must be an object");
    const auto *const op_value = entry.find("op");
    if (op_value == nullptr || op_value->type != JsonType::kString)
      return tokenizer_format_error("normalization entry op must be a string");
    const auto &name = op_value->string;
    NormalizationOp op;
    Status status = Status::Ok();
    if (name == "ascii-uppercase" || name == "ascii-lowercase" ||
        name == "u-to-t" || name == "strip-ascii-whitespace") {
      status = exact_keys(entry, {"op"}, "normalization entry");
      if (name == "ascii-uppercase")
        op.kind = NormalizationKind::kAsciiUppercase;
      else if (name == "ascii-lowercase")
        op.kind = NormalizationKind::kAsciiLowercase;
      else if (name == "u-to-t")
        op.kind = NormalizationKind::kUToT;
      else
        op.kind = NormalizationKind::kStripAsciiWhitespace;
    } else if (name == "prepend-literal") {
      status = exact_keys(entry, {"op", "value"}, "normalization entry");
      if (status.ok())
        status =
            required_string(entry, "value", "normalization entry", &op.first);
      if (status.ok())
        status = bounded_literal(op.first, "prepend-literal");
      op.kind = NormalizationKind::kPrependLiteral;
    } else if (name == "replace-literal") {
      status = exact_keys(entry, {"op", "from", "to"}, "normalization entry");
      if (status.ok())
        status =
            required_string(entry, "from", "normalization entry", &op.first);
      if (status.ok())
        status =
            required_string(entry, "to", "normalization entry", &op.second);
      if (status.ok())
        status = bounded_literal(op.first, "replace-literal.from", false);
      if (status.ok())
        status = bounded_literal(op.second, "replace-literal.to");
      op.kind = NormalizationKind::kReplaceLiteral;
    } else if (name == "replace-byte-run") {
      status = exact_keys(entry, {"op", "byte", "min_count", "replacement"},
                          "normalization entry");
      if (status.ok())
        status =
            required_string(entry, "byte", "normalization entry", &op.first);
      if (status.ok())
        status = required_string(entry, "replacement", "normalization entry",
                                 &op.second);
      if (status.ok() && (op.first.size() != 1 ||
                          static_cast<unsigned char>(op.first.front()) > 0x7fU))
        status = tokenizer_format_error(
            "replace-byte-run.byte must be one ASCII byte");
      if (status.ok())
        status = required_size(entry, "min_count", 1,
                               std::numeric_limits<std::uint32_t>::max(),
                               "normalization entry", &op.minimum_count);
      if (status.ok())
        status = bounded_literal(op.second, "replace-byte-run.replacement");
      if (status.ok())
        op.byte = static_cast<unsigned char>(op.first.front());
      op.kind = NormalizationKind::kReplaceByteRun;
    } else {
      return {ErrorCode::kUnsupported,
              "unsupported tokenizer normalization operation '" + name + "'"};
    }
    if (!status.ok())
      return status;
    output->push_back(std::move(op));
  }
  return Status::Ok();
}

Status parse_pretokenizer(const JsonValue &value, TokenizerData *const data) {
  if (value.type != JsonType::kObject)
    return tokenizer_format_error("pre_tokenizer must be an object");
  std::string kind;
  auto status = required_string(value, "kind", "pre_tokenizer", &kind);
  if (!status.ok())
    return status;
  if (kind == "split-isolated") {
    status = exact_keys(value, {"kind", "literal"}, "pre_tokenizer");
    if (!status.ok())
      return status;
    status = required_string(value, "literal", "pre_tokenizer",
                             &data->split_literal);
    if (!status.ok())
      return status;
    status = bounded_literal(data->split_literal,
                             "pre_tokenizer.split-isolated", false);
    if (!status.ok())
      return status;
    data->pretokenizer = PretokenizerKind::kSplitIsolated;
    return Status::Ok();
  }
  status = exact_keys(value, {"kind"}, "pre_tokenizer");
  if (!status.ok())
    return status;
  if (kind == "none")
    data->pretokenizer = PretokenizerKind::kNone;
  else if (kind == "whole-input")
    data->pretokenizer = PretokenizerKind::kWholeInput;
  else if (kind == "ascii-whitespace")
    data->pretokenizer = PretokenizerKind::kAsciiWhitespace;
  else if (kind == "hf-whitespace-ascii")
    data->pretokenizer = PretokenizerKind::kHfWhitespaceAscii;
  else
    return {ErrorCode::kUnsupported,
            "unsupported tokenizer pre_tokenizer kind '" + kind + "'"};
  return Status::Ok();
}

Status parse_vocab(const JsonValue &value, TokenizerData *const data) {
  if (value.type != JsonType::kArray || value.array.empty())
    return tokenizer_format_error("vocab must be a non-empty array");
  data->vocab.clear();
  data->reverse_vocab.clear();
  TokenId largest = 0;
  for (const auto &entry : value.array) {
    auto status = exact_keys(entry, {"id", "piece"}, "vocab entry");
    if (!status.ok())
      return status;
    const auto *const id_value = entry.find("id");
    TokenId id = 0;
    if (id_value == nullptr ||
        !(status = token_id(*id_value, "vocab entry.id", &id)).ok())
      return id_value == nullptr
                 ? tokenizer_format_error("vocab entry.id is required")
                 : status;
    std::string piece;
    status = required_string(entry, "piece", "vocab entry", &piece);
    if (!status.ok())
      return status;
    if (piece.empty())
      return tokenizer_format_error("vocab pieces must be non-empty");
    if (!data->vocab.emplace(piece, id).second)
      return tokenizer_format_error("duplicate vocab piece '" + piece + "'");
    if (!data->reverse_vocab.emplace(id, std::move(piece)).second)
      return tokenizer_format_error("duplicate vocab ID " + std::to_string(id));
    largest = std::max(largest, id);
  }
  const auto size = static_cast<std::uint64_t>(largest) + 1U;
  if (size > std::numeric_limits<std::size_t>::max())
    return tokenizer_format_error(
        "vocab size is not representable on this platform");
  data->vocabulary_size = static_cast<std::size_t>(size);
  return Status::Ok();
}

Status parse_optional_id(const JsonValue &object, const std::string_view name,
                         std::optional<TokenId> *const output) {
  const auto *const value = object.find(name);
  if (value == nullptr)
    return tokenizer_format_error("special_tokens field is missing");
  if (value->type == JsonType::kNull) {
    output->reset();
    return Status::Ok();
  }
  TokenId id = 0;
  auto status = token_id(*value, "special_tokens ID", &id);
  if (!status.ok())
    return status;
  *output = id;
  return Status::Ok();
}

Status validate_known_id(const TokenizerData &data, const TokenId id,
                         const std::string_view context) {
  if (data.reverse_vocab.find(id) == data.reverse_vocab.end())
    return tokenizer_format_error(std::string{context} +
                                  " references an ID absent from vocab");
  return Status::Ok();
}

Status parse_special_tokens(const JsonValue &value, TokenizerData *const data) {
  auto status =
      exact_keys(value, {"unk", "pad", "bos", "eos", "cls", "sep", "mask"},
                 "special_tokens");
  if (!status.ok())
    return status;
  constexpr std::array<std::string_view, 7> names{"unk", "pad", "bos", "eos",
                                                  "cls", "sep", "mask"};
  for (std::size_t index = 0; index < names.size(); ++index) {
    status = parse_optional_id(value, names[index], &data->specials[index]);
    if (!status.ok())
      return status;
    if (data->specials[index].has_value()) {
      status =
          validate_known_id(*data, *data->specials[index], "special_tokens");
      if (!status.ok())
        return status;
    }
  }
  return Status::Ok();
}

Status parse_id_array(const JsonValue &value, const std::string_view context,
                      const TokenizerData &data,
                      std::vector<TokenId> *const output) {
  if (value.type != JsonType::kArray || value.array.size() > 32)
    return tokenizer_format_error(std::string{context} +
                                  " must be an array of at most 32 IDs");
  output->clear();
  output->reserve(value.array.size());
  for (const auto &entry : value.array) {
    TokenId id = 0;
    auto status = token_id(entry, context, &id);
    if (!status.ok())
      return status;
    status = validate_known_id(data, id, context);
    if (!status.ok())
      return status;
    output->push_back(id);
  }
  return Status::Ok();
}

Status parse_post_processor(const JsonValue &value, TokenizerData *const data) {
  auto status = exact_keys(value, {"prefix_ids", "suffix_ids", "padding"},
                           "post_processor");
  if (!status.ok())
    return status;
  const auto *const prefix = value.find("prefix_ids");
  const auto *const suffix = value.find("suffix_ids");
  if (prefix == nullptr || suffix == nullptr)
    return tokenizer_format_error("post_processor ID arrays are required");
  status = parse_id_array(*prefix, "post_processor.prefix_ids", *data,
                          &data->prefix_ids);
  if (!status.ok())
    return status;
  status = parse_id_array(*suffix, "post_processor.suffix_ids", *data,
                          &data->suffix_ids);
  if (!status.ok())
    return status;
  const auto *const padding = value.find("padding");
  if (padding == nullptr)
    return tokenizer_format_error("post_processor.padding is required");
  status = exact_keys(*padding, {"side", "pad_id"}, "post_processor.padding");
  if (!status.ok())
    return status;
  std::string side;
  status = required_string(*padding, "side", "post_processor.padding", &side);
  if (!status.ok())
    return status;
  if (side == "none")
    data->padding_side = PaddingSide::kNone;
  else if (side == "left")
    data->padding_side = PaddingSide::kLeft;
  else if (side == "right")
    data->padding_side = PaddingSide::kRight;
  else
    return {ErrorCode::kUnsupported,
            "unsupported tokenizer padding side '" + side + "'"};
  status = parse_optional_id(*padding, "pad_id", &data->pad_id);
  if (!status.ok())
    return status;
  if (data->padding_side == PaddingSide::kNone && data->pad_id.has_value())
    return tokenizer_format_error("padding side none requires null pad_id");
  if (data->padding_side != PaddingSide::kNone && !data->pad_id.has_value())
    return tokenizer_format_error("enabled padding requires pad_id");
  if (data->pad_id.has_value()) {
    status = validate_known_id(*data, *data->pad_id, "post_processor.padding");
    if (!status.ok())
      return status;
    if (!data->specials[1].has_value() || *data->specials[1] != *data->pad_id)
      return tokenizer_format_error(
          "post_processor pad_id must equal special_tokens.pad");
  }
  return Status::Ok();
}

Status parse_unknown_policy(const JsonValue &object,
                            UnknownPolicy *const output) {
  std::string value;
  auto status = required_string(object, "unknown_policy", "model", &value);
  if (!status.ok())
    return status;
  if (value == "error")
    *output = UnknownPolicy::kError;
  else if (value == "unk")
    *output = UnknownPolicy::kUnk;
  else
    return {ErrorCode::kUnsupported,
            "unsupported tokenizer unknown_policy '" + value + "'"};
  return Status::Ok();
}

Status parse_tail_policy(const JsonValue &object, TailPolicy *const output) {
  std::string value;
  auto status = required_string(object, "tail", "model", &value);
  if (!status.ok())
    return status;
  if (value == "drop")
    *output = TailPolicy::kDrop;
  else if (value == "error")
    *output = TailPolicy::kError;
  else if (value == "unk")
    *output = TailPolicy::kUnk;
  else if (value == "lookup")
    *output = TailPolicy::kLookup;
  else
    return {ErrorCode::kUnsupported,
            "unsupported tokenizer tail policy '" + value + "'"};
  return Status::Ok();
}

Status require_unknown_id(const TokenizerData &data) {
  if (data.unknown_policy == UnknownPolicy::kUnk &&
      !data.specials[0].has_value())
    return tokenizer_format_error(
        "unknown_policy unk requires special_tokens.unk");
  if (data.tail_policy == TailPolicy::kUnk && !data.specials[0].has_value())
    return tokenizer_format_error("tail unk requires special_tokens.unk");
  return Status::Ok();
}

Status parse_merges(const JsonValue &value, TokenizerData *const data) {
  if (value.type != JsonType::kArray)
    return tokenizer_format_error("model.merges must be an array");
  data->merge_rules.clear();
  for (std::size_t rank = 0; rank < value.array.size(); ++rank) {
    const auto &entry = value.array[rank];
    if (entry.type != JsonType::kArray || entry.array.size() != 2 ||
        entry.array[0].type != JsonType::kString ||
        entry.array[1].type != JsonType::kString ||
        entry.array[0].string.empty() || entry.array[1].string.empty())
      return tokenizer_format_error(
          "model.merges entries must be pairs of non-empty strings");
    const std::pair<std::string, std::string> pair{entry.array[0].string,
                                                   entry.array[1].string};
    const auto left = data->vocab.find(pair.first);
    const auto right = data->vocab.find(pair.second);
    const auto result = data->vocab.find(pair.first + pair.second);
    if (left == data->vocab.end() || right == data->vocab.end() ||
        result == data->vocab.end())
      return tokenizer_format_error(
          "model.merges references pieces absent from vocab");
    if (!data->merge_rules
             .emplace(std::pair<TokenId, TokenId>{left->second, right->second},
                      BpeMergeRule{rank, result->second})
             .second)
      return tokenizer_format_error("duplicate model.merges pair");
  }
  return Status::Ok();
}

Status validate_pretokenizer(const TokenizerData &data) {
  bool valid = false;
  switch (data.kind) {
  case ArtifactTokenizerKind::kCharacter:
  case ArtifactTokenizerKind::kSingleNucleotide:
  case ArtifactTokenizerKind::kKmer:
  case ArtifactTokenizerKind::kKmerBpe:
    valid = data.pretokenizer == PretokenizerKind::kNone;
    break;
  case ArtifactTokenizerKind::kWordPiece:
    valid = data.pretokenizer == PretokenizerKind::kAsciiWhitespace ||
            data.pretokenizer == PretokenizerKind::kHfWhitespaceAscii;
    break;
  case ArtifactTokenizerKind::kBpe:
    valid = data.pretokenizer == PretokenizerKind::kWholeInput ||
            data.pretokenizer == PretokenizerKind::kAsciiWhitespace ||
            data.pretokenizer == PretokenizerKind::kHfWhitespaceAscii ||
            data.pretokenizer == PretokenizerKind::kSplitIsolated;
    break;
  case ArtifactTokenizerKind::kByteBpe:
    valid = data.pretokenizer == PretokenizerKind::kWholeInput;
    break;
  case ArtifactTokenizerKind::kLongestMatch:
    valid = data.pretokenizer == PretokenizerKind::kNone ||
            data.pretokenizer == PretokenizerKind::kWholeInput;
    break;
  }
  if (!valid)
    return {ErrorCode::kUnsupported,
            "tokenizer kind and pre_tokenizer combination is unsupported"};
  return Status::Ok();
}

Status parse_model(const JsonValue &value, TokenizerData *const data) {
  Status status = Status::Ok();
  switch (data->kind) {
  case ArtifactTokenizerKind::kCharacter:
  case ArtifactTokenizerKind::kSingleNucleotide:
  case ArtifactTokenizerKind::kLongestMatch:
    status = exact_keys(value, {"unknown_policy", "match_special_literals"},
                        "model");
    if (status.ok())
      status = parse_unknown_policy(value, &data->unknown_policy);
    if (status.ok())
      status = required_bool(value, "match_special_literals", "model",
                             &data->match_special_literals);
    break;
  case ArtifactTokenizerKind::kKmer:
    if (value.find("match_special_literals") == nullptr)
      status =
          exact_keys(value, {"k", "stride", "tail", "unknown_policy"}, "model");
    else
      status = exact_keys(
          value,
          {"k", "stride", "tail", "unknown_policy", "match_special_literals"},
          "model");
    if (status.ok())
      status = required_size(value, "k", 1, 4096, "model", &data->k);
    if (status.ok())
      status =
          required_size(value, "stride", 1, data->k, "model", &data->stride);
    if (status.ok())
      status = parse_tail_policy(value, &data->tail_policy);
    if (status.ok())
      status = parse_unknown_policy(value, &data->unknown_policy);
    if (status.ok() && value.find("match_special_literals") != nullptr)
      status = required_bool(value, "match_special_literals", "model",
                             &data->match_special_literals);
    break;
  case ArtifactTokenizerKind::kWordPiece:
    status = exact_keys(
        value, {"continuation_prefix", "max_input_chars_per_word"}, "model");
    if (status.ok())
      status = required_string(value, "continuation_prefix", "model",
                               &data->continuation_prefix);
    if (status.ok())
      status = bounded_literal(data->continuation_prefix,
                               "model.continuation_prefix");
    if (status.ok())
      status = required_size(value, "max_input_chars_per_word", 1, 1U << 20U,
                             "model", &data->maximum_word_bytes);
    if (status.ok() && !data->specials[0].has_value())
      status = tokenizer_format_error("wordpiece requires special_tokens.unk");
    break;
  case ArtifactTokenizerKind::kBpe: {
    if (value.find("literal_token_ids") == nullptr)
      status = exact_keys(value, {"merges"}, "model");
    else
      status = exact_keys(value, {"merges", "literal_token_ids"}, "model");
    const auto *const merges = value.find("merges");
    if (status.ok() && merges != nullptr)
      status = parse_merges(*merges, data);
    if (status.ok() && value.find("literal_token_ids") != nullptr)
      status = parse_id_array(*value.find("literal_token_ids"),
                              "model.literal_token_ids", *data,
                              &data->literal_token_ids);
    if (status.ok() && value.find("literal_token_ids") != nullptr &&
        data->literal_token_ids.empty())
      status =
          tokenizer_format_error("model.literal_token_ids must not be empty");
    break;
  }
  case ArtifactTokenizerKind::kByteBpe: {
    status = exact_keys(value, {"add_prefix_space", "byte_encoder", "merges"},
                        "model");
    if (status.ok())
      status = required_bool(value, "add_prefix_space", "model",
                             &data->add_prefix_space);
    const auto *const encoder = value.find("byte_encoder");
    if (status.ok() &&
        (encoder == nullptr || encoder->type != JsonType::kArray ||
         encoder->array.size() != data->byte_encoder.size()))
      status = tokenizer_format_error(
          "model.byte_encoder must contain exactly 256 strings");
    std::set<std::string, std::less<>> encoder_pieces;
    if (status.ok()) {
      for (std::size_t index = 0; index < encoder->array.size(); ++index) {
        if (encoder->array[index].type != JsonType::kString ||
            encoder->array[index].string.empty()) {
          status = tokenizer_format_error(
              "model.byte_encoder entries must be non-empty strings");
          break;
        }
        const auto &piece = encoder->array[index].string;
        if (!encoder_pieces.emplace(piece).second) {
          status = tokenizer_format_error(
              "model.byte_encoder entries must be unique");
          break;
        }
        const auto vocab_piece = data->vocab.find(piece);
        if (vocab_piece == data->vocab.end()) {
          status = tokenizer_format_error(
              "model.byte_encoder piece is absent from vocab");
          break;
        }
        data->byte_encoder[index] = vocab_piece->second;
      }
    }
    const auto *const merges = value.find("merges");
    if (status.ok() && merges != nullptr)
      status = parse_merges(*merges, data);
    break;
  }
  case ArtifactTokenizerKind::kKmerBpe: {
    status = exact_keys(
        value, {"k", "stride", "tail", "unknown_policy", "merges"}, "model");
    if (status.ok())
      status = required_size(value, "k", 1, 4096, "model", &data->k);
    if (status.ok())
      status =
          required_size(value, "stride", 1, data->k, "model", &data->stride);
    if (status.ok())
      status = parse_tail_policy(value, &data->tail_policy);
    if (status.ok())
      status = parse_unknown_policy(value, &data->unknown_policy);
    const auto *const merges = value.find("merges");
    if (status.ok() && merges != nullptr)
      status = parse_merges(*merges, data);
    break;
  }
  }
  if (!status.ok())
    return status;
  return require_unknown_id(*data);
}

Status build_longest_match_trie(TokenizerData *const data) {
  data->longest_match_trie.clear();
  data->longest_match_trie.emplace_back();
  std::set<TokenId> excluded;
  if (!data->match_special_literals) {
    for (const auto &special : data->specials) {
      if (special.has_value())
        excluded.emplace(*special);
    }
  }
  for (const auto &entry : data->vocab) {
    if (excluded.find(entry.second) != excluded.end())
      continue;
    std::size_t node = 0;
    for (const char raw_byte : entry.first) {
      const auto byte = static_cast<unsigned char>(raw_byte);
      auto child = data->longest_match_trie[node].children.find(byte);
      if (child == data->longest_match_trie[node].children.end()) {
        const auto next = data->longest_match_trie.size();
        data->longest_match_trie[node].children.emplace(byte, next);
        data->longest_match_trie.emplace_back();
        node = next;
      } else {
        node = child->second;
      }
    }
    data->longest_match_trie[node].token = entry.second;
  }
  if (data->longest_match_trie.front().children.empty())
    return tokenizer_format_error(
        "longest-match vocabulary has no usable pieces");
  return Status::Ok();
}

Status parse_tokenizer_manifest(const std::string_view bytes,
                                TokenizerData *const data) {
  JsonValue root;
  auto status = parse_json(bytes, &root);
  if (!status.ok())
    return tokenizer_format_error("invalid JSON: " + status.message());
  status = exact_keys(root,
                      {"format", "kind", "normalization", "pre_tokenizer",
                       "model", "post_processor", "special_tokens", "vocab"},
                      "root");
  if (!status.ok())
    return status;
  std::string format;
  status = required_string(root, "format", "root", &format);
  if (!status.ok())
    return status;
  if (format != "evo-tokenizer-v1")
    return {ErrorCode::kUnsupported,
            "unsupported tokenizer asset format '" + format + "'"};
  std::string kind;
  status = required_string(root, "kind", "root", &kind);
  if (!status.ok())
    return status;
  status = parse_kind(kind, &data->kind);
  if (!status.ok())
    return status;
  const auto *const normalization = root.find("normalization");
  const auto *const pretokenizer = root.find("pre_tokenizer");
  const auto *const vocab = root.find("vocab");
  const auto *const specials = root.find("special_tokens");
  const auto *const post = root.find("post_processor");
  const auto *const model = root.find("model");
  if (normalization == nullptr || pretokenizer == nullptr || vocab == nullptr ||
      specials == nullptr || post == nullptr || model == nullptr)
    return tokenizer_format_error("root has missing required fields");
  status = parse_normalization(*normalization, &data->normalization);
  if (!status.ok())
    return status;
  status = parse_pretokenizer(*pretokenizer, data);
  if (!status.ok())
    return status;
  status = validate_pretokenizer(*data);
  if (!status.ok())
    return status;
  status = parse_vocab(*vocab, data);
  if (!status.ok())
    return status;
  status = parse_special_tokens(*specials, data);
  if (!status.ok())
    return status;
  status = parse_post_processor(*post, data);
  if (!status.ok())
    return status;
  status = parse_model(*model, data);
  if (!status.ok())
    return status;
  if (!data->literal_token_ids.empty()) {
    std::set<TokenId> unique;
    for (const auto id : data->literal_token_ids) {
      if (!unique.emplace(id).second)
        return tokenizer_format_error("model.literal_token_ids has duplicates");
    }
  }
  if (data->kind == ArtifactTokenizerKind::kLongestMatch) {
    status = build_longest_match_trie(data);
    if (!status.ok())
      return status;
  }
  return Status::Ok();
}

Status checked_string_append(std::string *const output,
                             const std::string_view value) {
  if (value.size() > kMaximumNormalizedBytes ||
      output->size() > kMaximumNormalizedBytes - value.size())
    return {ErrorCode::kInvalidArgument,
            "tokenizer normalization exceeds the byte safety limit"};
  output->append(value);
  return Status::Ok();
}

Status replace_literal(const std::string &input, const std::string &from,
                       const std::string &to, std::string *const output) {
  output->clear();
  std::size_t offset = 0;
  while (offset < input.size()) {
    const auto found = input.find(from, offset);
    if (found == std::string::npos) {
      return checked_string_append(output,
                                   std::string_view{input}.substr(offset));
    }
    auto status = checked_string_append(
        output, std::string_view{input}.substr(offset, found - offset));
    if (!status.ok())
      return status;
    status = checked_string_append(output, to);
    if (!status.ok())
      return status;
    offset = found + from.size();
  }
  return Status::Ok();
}

Status replace_byte_run(const std::string &input,
                        const NormalizationOp &operation,
                        std::string *const output) {
  output->clear();
  std::size_t offset = 0;
  while (offset < input.size()) {
    if (static_cast<unsigned char>(input[offset]) != operation.byte) {
      if (output->size() == kMaximumNormalizedBytes)
        return {ErrorCode::kInvalidArgument,
                "tokenizer normalization exceeds the byte safety limit"};
      output->push_back(input[offset++]);
      continue;
    }
    const auto begin = offset;
    while (offset < input.size() &&
           static_cast<unsigned char>(input[offset]) == operation.byte)
      ++offset;
    const auto count = offset - begin;
    const std::string_view replacement =
        count >= operation.minimum_count
            ? std::string_view{operation.second}
            : std::string_view{input}.substr(begin, count);
    auto status = checked_string_append(output, replacement);
    if (!status.ok())
      return status;
  }
  return Status::Ok();
}

Status validate_raw_input_size(const std::string_view input,
                               const TokenizerEncodeOptions &options) {
  if (options.raw_byte_limit != 0 && input.size() > options.raw_byte_limit)
    return {ErrorCode::kInvalidArgument,
            "tokenizer raw input exceeds configured byte limit"};
  if (input.size() > kMaximumNormalizedBytes)
    return {ErrorCode::kInvalidArgument,
            "tokenizer raw input exceeds the byte safety limit"};
  return Status::Ok();
}

Status normalize(const TokenizerData &data, const std::string_view input,
                 const TokenizerEncodeOptions &options,
                 std::string *const output) {
  const auto raw_status = validate_raw_input_size(input, options);
  if (!raw_status.ok())
    return raw_status;
  output->assign(input);
  std::string replacement;
  for (const auto &operation : data.normalization) {
    switch (operation.kind) {
    case NormalizationKind::kAsciiUppercase:
      for (char &character : *output) {
        if (character >= 'a' && character <= 'z')
          character = static_cast<char>(character - 'a' + 'A');
      }
      break;
    case NormalizationKind::kAsciiLowercase:
      for (char &character : *output) {
        if (character >= 'A' && character <= 'Z')
          character = static_cast<char>(character - 'A' + 'a');
      }
      break;
    case NormalizationKind::kUToT:
      for (char &character : *output) {
        if (character == 'U')
          character = 'T';
        else if (character == 'u')
          character = 't';
      }
      break;
    case NormalizationKind::kStripAsciiWhitespace: {
      std::size_t begin = 0;
      while (begin < output->size() &&
             is_ascii_space(static_cast<unsigned char>((*output)[begin])))
        ++begin;
      std::size_t end = output->size();
      while (end > begin &&
             is_ascii_space(static_cast<unsigned char>((*output)[end - 1U])))
        --end;
      if (begin != 0 || end != output->size())
        *output = output->substr(begin, end - begin);
      break;
    }
    case NormalizationKind::kPrependLiteral:
      replacement.clear();
      {
        auto status = checked_string_append(&replacement, operation.first);
        if (!status.ok())
          return status;
        status = checked_string_append(&replacement, *output);
        if (!status.ok())
          return status;
      }
      output->swap(replacement);
      break;
    case NormalizationKind::kReplaceLiteral: {
      auto status = replace_literal(*output, operation.first, operation.second,
                                    &replacement);
      if (!status.ok())
        return status;
      output->swap(replacement);
      break;
    }
    case NormalizationKind::kReplaceByteRun: {
      auto status = replace_byte_run(*output, operation, &replacement);
      if (!status.ok())
        return status;
      output->swap(replacement);
      break;
    }
    }
  }
  return Status::Ok();
}

Status next_utf8_symbol(const std::string_view input, const std::size_t offset,
                        std::size_t *const width_out) {
  if (offset >= input.size())
    return {ErrorCode::kInvalidArgument,
            "tokenizer UTF-8 offset is outside the input"};
  const auto first = static_cast<unsigned char>(input[offset]);
  std::size_t width = 0;
  std::uint32_t codepoint = 0;
  if (first <= 0x7fU) {
    width = 1;
    codepoint = first;
  } else if (first >= 0xc2U && first <= 0xdfU) {
    width = 2;
    codepoint = first & 0x1fU;
  } else if (first >= 0xe0U && first <= 0xefU) {
    width = 3;
    codepoint = first & 0x0fU;
  } else if (first >= 0xf0U && first <= 0xf4U) {
    width = 4;
    codepoint = first & 0x07U;
  } else {
    return {ErrorCode::kInvalidArgument,
            "tokenizer input is not canonical UTF-8"};
  }
  if (width > input.size() - offset)
    return {ErrorCode::kInvalidArgument,
            "tokenizer input ends inside a UTF-8 codepoint"};
  for (std::size_t index = 1; index < width; ++index) {
    const auto continuation = static_cast<unsigned char>(input[offset + index]);
    if ((continuation & 0xc0U) != 0x80U)
      return {ErrorCode::kInvalidArgument,
              "tokenizer input has an invalid UTF-8 continuation byte"};
    codepoint = (codepoint << 6U) | (continuation & 0x3fU);
  }
  if ((width == 3 && codepoint < 0x800U) ||
      (width == 4 && codepoint < 0x10000U) ||
      (codepoint >= 0xd800U && codepoint <= 0xdfffU) || codepoint > 0x10ffffU)
    return {ErrorCode::kInvalidArgument,
            "tokenizer input contains a non-canonical UTF-8 codepoint"};
  *width_out = width;
  return Status::Ok();
}

Status utf8_symbols(const std::string_view input,
                    std::vector<std::string_view> *const output) {
  output->clear();
  output->reserve(input.size());
  std::size_t offset = 0;
  while (offset < input.size()) {
    std::size_t width = 0;
    auto status = next_utf8_symbol(input, offset, &width);
    if (!status.ok())
      return status;
    output->push_back(input.substr(offset, width));
    offset += width;
  }
  return Status::Ok();
}

Status append_token(const TokenId id, const TokenizerEncodeOptions &options,
                    std::vector<TokenId> *const output) {
  if (output->size() >= kMaximumIntermediateTokens)
    return {ErrorCode::kInvalidArgument,
            "tokenizer output exceeds the token safety limit"};
  if (options.token_limit != 0 && output->size() >= options.token_limit)
    return {ErrorCode::kInvalidArgument,
            "tokenizer output exceeds configured token limit"};
  output->push_back(id);
  return Status::Ok();
}

Status unknown_token(const TokenizerData &data,
                     const TokenizerEncodeOptions &options,
                     std::vector<TokenId> *const output,
                     const std::string_view piece) {
  if (data.unknown_policy == UnknownPolicy::kUnk &&
      data.specials[0].has_value())
    return append_token(*data.specials[0], options, output);
  return {ErrorCode::kInvalidArgument,
          "tokenizer has no vocabulary entry for input piece '" +
              std::string{piece} + "'"};
}

Status append_piece(const TokenizerData &data, const std::string_view piece,
                    const TokenizerEncodeOptions &options,
                    std::vector<TokenId> *const output) {
  const auto iterator = data.vocab.find(piece);
  if (iterator == data.vocab.end())
    return unknown_token(data, options, output, piece);
  return append_token(iterator->second, options, output);
}

Status tokenize_character(const TokenizerData &data,
                          const std::string_view input,
                          const TokenizerEncodeOptions &options,
                          std::vector<TokenId> *const output) {
  std::vector<std::pair<std::string_view, TokenId>> special_literals;
  if (data.match_special_literals) {
    for (const auto &special : data.specials) {
      if (!special.has_value())
        continue;
      const auto piece = data.reverse_vocab.find(*special);
      if (piece != data.reverse_vocab.end())
        special_literals.emplace_back(piece->second, *special);
    }
    std::sort(special_literals.begin(), special_literals.end(),
              [](const auto &left, const auto &right) {
                return left.first.size() > right.first.size();
              });
  }
  std::size_t offset = 0;
  while (offset < input.size()) {
    bool matched = false;
    for (const auto &special : special_literals) {
      if (input.substr(offset, special.first.size()) == special.first) {
        auto status = append_token(special.second, options, output);
        if (!status.ok())
          return status;
        offset += special.first.size();
        matched = true;
        break;
      }
    }
    if (matched)
      continue;
    std::size_t width = 0;
    auto status = next_utf8_symbol(input, offset, &width);
    if (!status.ok())
      return status;
    const auto symbol = input.substr(offset, width);
    if (data.kind == ArtifactTokenizerKind::kSingleNucleotide &&
        symbol.size() != 1)
      return unknown_token(data, options, output, symbol);
    status = append_piece(data, symbol, options, output);
    if (!status.ok())
      return status;
    offset += width;
  }
  return Status::Ok();
}

std::vector<std::pair<std::string_view, TokenId>>
special_literals(const TokenizerData &data) {
  std::vector<std::pair<std::string_view, TokenId>> result;
  if (!data.match_special_literals)
    return result;
  for (const auto &special : data.specials) {
    if (!special.has_value())
      continue;
    const auto piece = data.reverse_vocab.find(*special);
    if (piece != data.reverse_vocab.end())
      result.emplace_back(piece->second, *special);
  }
  std::sort(result.begin(), result.end(),
            [](const auto &left, const auto &right) {
              return left.first.size() > right.first.size();
            });
  return result;
}

std::vector<std::pair<std::string_view, TokenId>>
configured_literal_tokens(const TokenizerData &data) {
  std::vector<std::pair<std::string_view, TokenId>> result;
  result.reserve(data.literal_token_ids.size());
  for (const auto id : data.literal_token_ids) {
    const auto piece = data.reverse_vocab.find(id);
    if (piece != data.reverse_vocab.end())
      result.emplace_back(piece->second, id);
  }
  std::sort(result.begin(), result.end(),
            [](const auto &left, const auto &right) {
              return left.first.size() > right.first.size();
            });
  return result;
}

template <typename EmitSegment>
Status tokenize_literal_aware(const TokenizerData &data,
                              const std::string_view input,
                              const TokenizerEncodeOptions &options,
                              std::vector<TokenId> *const output,
                              EmitSegment emit_segment) {
  const auto literals = configured_literal_tokens(data);
  if (literals.empty())
    return emit_segment(input);
  std::size_t offset = 0;
  while (offset < input.size()) {
    bool matched = false;
    for (const auto &literal : literals) {
      if (input.substr(offset, literal.first.size()) == literal.first) {
        auto status = append_token(literal.second, options, output);
        if (!status.ok())
          return status;
        offset += literal.first.size();
        matched = true;
        break;
      }
    }
    if (matched)
      continue;
    auto end = input.size();
    for (const auto &literal : literals) {
      const auto found = input.find(literal.first, offset);
      if (found != std::string_view::npos)
        end = std::min(end, found);
    }
    auto status = emit_segment(input.substr(offset, end - offset));
    if (!status.ok())
      return status;
    offset = end;
  }
  return Status::Ok();
}

Status tokenize_longest_match(const TokenizerData &data,
                              const std::string_view input,
                              const TokenizerEncodeOptions &options,
                              std::vector<TokenId> *const output) {
  std::size_t offset = 0;
  while (offset < input.size()) {
    std::size_t node = 0;
    std::size_t cursor = offset;
    std::optional<TokenId> matched_token;
    std::size_t matched_end = offset;
    while (cursor < input.size()) {
      const auto child = data.longest_match_trie[node].children.find(
          static_cast<unsigned char>(input[cursor]));
      if (child == data.longest_match_trie[node].children.end())
        break;
      node = child->second;
      ++cursor;
      if (data.longest_match_trie[node].token.has_value()) {
        matched_token = data.longest_match_trie[node].token;
        matched_end = cursor;
      }
    }
    if (matched_token.has_value()) {
      auto status = append_token(*matched_token, options, output);
      if (!status.ok())
        return status;
      offset = matched_end;
      continue;
    }
    std::size_t width = 0;
    auto status = next_utf8_symbol(input, offset, &width);
    if (!status.ok())
      return status;
    const auto symbol = input.substr(offset, width);
    status = unknown_token(data, options, output, symbol);
    if (!status.ok())
      return status;
    offset += width;
  }
  return Status::Ok();
}

Status initial_kmers(const TokenizerData &data, const std::string_view input,
                     const TokenizerEncodeOptions &options,
                     const bool enforce_final_limit,
                     std::vector<TokenId> *const pieces) {
  pieces->clear();
  for (const char raw_byte : input) {
    const auto byte = static_cast<unsigned char>(raw_byte);
    if (byte > 0x7fU)
      return {ErrorCode::kInvalidArgument,
              "k-mer tokenizer input must be ASCII"};
  }
  std::size_t start = 0;
  std::size_t covered = 0;
  while (start <= input.size() && data.k <= input.size() - start) {
    const auto raw_piece = input.substr(start, data.k);
    const auto piece = data.vocab.find(raw_piece);
    TokenId id = 0;
    if (piece == data.vocab.end()) {
      if (data.unknown_policy != UnknownPolicy::kUnk)
        return {ErrorCode::kInvalidArgument,
                "tokenizer has no vocabulary entry for k-mer '" +
                    std::string{raw_piece} + "'"};
      id = *data.specials[0];
    } else {
      id = piece->second;
    }
    if (pieces->size() >= kMaximumIntermediateTokens)
      return {ErrorCode::kInvalidArgument,
              "tokenizer intermediate token count exceeds the safety limit"};
    if (enforce_final_limit && options.token_limit != 0 &&
        pieces->size() >= options.token_limit)
      return {ErrorCode::kInvalidArgument,
              "tokenizer output exceeds configured token limit"};
    pieces->push_back(id);
    covered = start + data.k;
    if (data.stride > input.size() - start)
      break;
    start += data.stride;
  }
  const auto tail = input.size() - covered;
  if (tail != 0) {
    if (data.tail_policy == TailPolicy::kError)
      return {ErrorCode::kInvalidArgument,
              "tokenizer input has an incomplete trailing k-mer"};
    if (data.tail_policy == TailPolicy::kUnk) {
      if (pieces->size() >= kMaximumIntermediateTokens)
        return {ErrorCode::kInvalidArgument,
                "tokenizer intermediate token count exceeds the safety limit"};
      if (enforce_final_limit && options.token_limit != 0 &&
          pieces->size() >= options.token_limit)
        return {ErrorCode::kInvalidArgument,
                "tokenizer output exceeds configured token limit"};
      pieces->push_back(*data.specials[0]);
    } else if (data.tail_policy == TailPolicy::kLookup) {
      const auto raw_piece = input.substr(covered);
      const auto piece = data.vocab.find(raw_piece);
      TokenId id = 0;
      if (piece == data.vocab.end()) {
        if (data.unknown_policy != UnknownPolicy::kUnk)
          return {ErrorCode::kInvalidArgument,
                  "tokenizer has no vocabulary entry for trailing k-mer '" +
                      std::string{raw_piece} + "'"};
        id = *data.specials[0];
      } else {
        id = piece->second;
      }
      if (pieces->size() >= kMaximumIntermediateTokens)
        return {ErrorCode::kInvalidArgument,
                "tokenizer intermediate token count exceeds the safety limit"};
      if (enforce_final_limit && options.token_limit != 0 &&
          pieces->size() >= options.token_limit)
        return {ErrorCode::kInvalidArgument,
                "tokenizer output exceeds configured token limit"};
      pieces->push_back(id);
    }
  }
  return Status::Ok();
}

Status tokenize_kmers(const TokenizerData &data, const std::string_view input,
                      const TokenizerEncodeOptions &options,
                      std::vector<TokenId> *const output) {
  const auto emit_segment = [&](const std::string_view segment) {
    std::vector<TokenId> pieces;
    auto status = initial_kmers(data, segment, options, true, &pieces);
    if (!status.ok())
      return status;
    for (const auto piece : pieces) {
      status = append_token(piece, options, output);
      if (!status.ok())
        return status;
    }
    return Status::Ok();
  };
  const auto literals = special_literals(data);
  if (literals.empty())
    return emit_segment(input);
  std::size_t offset = 0;
  while (offset < input.size()) {
    bool matched = false;
    for (const auto &literal : literals) {
      if (input.substr(offset, literal.first.size()) == literal.first) {
        auto status = append_token(literal.second, options, output);
        if (!status.ok())
          return status;
        offset += literal.first.size();
        matched = true;
        break;
      }
    }
    if (matched)
      continue;
    auto end = input.size();
    for (const auto &literal : literals) {
      const auto found = input.find(literal.first, offset);
      if (found != std::string_view::npos)
        end = std::min(end, found);
    }
    auto status = emit_segment(input.substr(offset, end - offset));
    if (!status.ok())
      return status;
    offset = end;
  }
  return Status::Ok();
}

struct BpeNode final {
  TokenId token{0};
  std::size_t previous{std::numeric_limits<std::size_t>::max()};
  std::size_t next{std::numeric_limits<std::size_t>::max()};
  std::uint64_t generation{0};
  bool active{true};
};

struct BpeCandidate final {
  std::size_t rank{0};
  std::size_t left{0};
  std::size_t right{0};
  std::uint64_t left_generation{0};
  std::uint64_t right_generation{0};
};

struct BpeCandidateLater final {
  bool operator()(const BpeCandidate &left,
                  const BpeCandidate &right) const noexcept {
    if (left.rank != right.rank)
      return left.rank > right.rank;
    return left.left > right.left;
  }
};

Status apply_bpe(const TokenizerData &data,
                 std::vector<TokenId> *const pieces) {
  if (pieces->size() > kMaximumIntermediateTokens)
    return {ErrorCode::kInvalidArgument,
            "BPE intermediate token count exceeds the safety limit"};
  if (pieces->size() < 2)
    return Status::Ok();
  constexpr auto none = std::numeric_limits<std::size_t>::max();
  std::vector<BpeNode> nodes;
  nodes.reserve(pieces->size());
  for (std::size_t index = 0; index < pieces->size(); ++index) {
    nodes.push_back(BpeNode{(*pieces)[index], index == 0 ? none : index - 1U,
                            index + 1U == pieces->size() ? none : index + 1U, 0,
                            true});
  }
  std::priority_queue<BpeCandidate, std::vector<BpeCandidate>,
                      BpeCandidateLater>
      candidates;
  const auto push_candidate = [&](const std::size_t left) {
    if (left == none || !nodes[left].active || nodes[left].next == none)
      return;
    const auto right = nodes[left].next;
    const auto rule =
        data.merge_rules.find({nodes[left].token, nodes[right].token});
    if (rule == data.merge_rules.end())
      return;
    candidates.push(BpeCandidate{rule->second.rank, left, right,
                                 nodes[left].generation,
                                 nodes[right].generation});
  };
  for (std::size_t index = 0; index + 1U < nodes.size(); ++index)
    push_candidate(index);
  while (!candidates.empty()) {
    const auto candidate = candidates.top();
    candidates.pop();
    if (!nodes[candidate.left].active || !nodes[candidate.right].active ||
        nodes[candidate.left].next != candidate.right ||
        nodes[candidate.left].generation != candidate.left_generation ||
        nodes[candidate.right].generation != candidate.right_generation)
      continue;
    const auto rule = data.merge_rules.find(
        {nodes[candidate.left].token, nodes[candidate.right].token});
    if (rule == data.merge_rules.end() || rule->second.rank != candidate.rank)
      continue;
    auto &left = nodes[candidate.left];
    auto &right = nodes[candidate.right];
    left.token = rule->second.result;
    ++left.generation;
    right.active = false;
    ++right.generation;
    left.next = right.next;
    if (right.next != none)
      nodes[right.next].previous = candidate.left;
    push_candidate(left.previous);
    push_candidate(candidate.left);
  }
  pieces->clear();
  for (std::size_t index = 0; index != none; index = nodes[index].next) {
    if (nodes[index].active)
      pieces->push_back(nodes[index].token);
  }
  return Status::Ok();
}

Status emit_bpe_pieces(const std::vector<TokenId> &pieces,
                       const TokenizerEncodeOptions &options,
                       std::vector<TokenId> *const output) {
  for (const auto piece : pieces) {
    auto status = append_token(piece, options, output);
    if (!status.ok())
      return status;
  }
  return Status::Ok();
}

Status tokenize_kmer_bpe(const TokenizerData &data,
                         const std::string_view input,
                         const TokenizerEncodeOptions &options,
                         std::vector<TokenId> *const output) {
  std::vector<TokenId> pieces;
  auto status = initial_kmers(data, input, options, false, &pieces);
  if (!status.ok())
    return status;
  status = apply_bpe(data, &pieces);
  if (!status.ok())
    return status;
  return emit_bpe_pieces(pieces, options, output);
}

bool is_ascii_word(const unsigned char byte) noexcept {
  return (byte >= '0' && byte <= '9') || (byte >= 'A' && byte <= 'Z') ||
         (byte >= 'a' && byte <= 'z') || byte == '_';
}

Status pretokenize(const TokenizerData &data, const std::string_view input,
                   std::vector<std::string_view> *const segments) {
  segments->clear();
  if (data.pretokenizer == PretokenizerKind::kWholeInput ||
      data.pretokenizer == PretokenizerKind::kNone) {
    if (!input.empty())
      segments->push_back(input);
    return Status::Ok();
  }
  if (data.pretokenizer == PretokenizerKind::kAsciiWhitespace) {
    std::size_t offset = 0;
    while (offset < input.size()) {
      while (offset < input.size() &&
             is_ascii_space(static_cast<unsigned char>(input[offset])))
        ++offset;
      const auto begin = offset;
      while (offset < input.size() &&
             !is_ascii_space(static_cast<unsigned char>(input[offset])))
        ++offset;
      if (offset != begin)
        segments->push_back(input.substr(begin, offset - begin));
    }
    return Status::Ok();
  }
  if (data.pretokenizer == PretokenizerKind::kHfWhitespaceAscii) {
    std::size_t offset = 0;
    while (offset < input.size()) {
      const auto byte = static_cast<unsigned char>(input[offset]);
      if (byte > 0x7fU)
        return {ErrorCode::kUnsupported,
                "HF Whitespace pre-tokenizer supports ASCII input only"};
      if (is_ascii_space(byte)) {
        ++offset;
        continue;
      }
      const auto begin = offset;
      const auto word = is_ascii_word(byte);
      do {
        ++offset;
        if (offset == input.size())
          break;
        const auto next = static_cast<unsigned char>(input[offset]);
        if (next > 0x7fU)
          return {ErrorCode::kUnsupported,
                  "HF Whitespace pre-tokenizer supports ASCII input only"};
        if (is_ascii_space(next) || is_ascii_word(next) != word)
          break;
      } while (true);
      segments->push_back(input.substr(begin, offset - begin));
    }
    return Status::Ok();
  }
  std::size_t offset = 0;
  while (offset < input.size()) {
    const auto found = input.find(data.split_literal, offset);
    if (found == std::string_view::npos) {
      if (offset != input.size())
        segments->push_back(input.substr(offset));
      return Status::Ok();
    }
    if (found != offset)
      segments->push_back(input.substr(offset, found - offset));
    segments->push_back(input.substr(found, data.split_literal.size()));
    offset = found + data.split_literal.size();
  }
  return Status::Ok();
}

Status tokenize_bpe_segment(const TokenizerData &data,
                            const std::string_view segment,
                            const TokenizerEncodeOptions &options,
                            std::vector<TokenId> *const output) {
  std::vector<std::string_view> symbols;
  auto status = utf8_symbols(segment, &symbols);
  if (!status.ok())
    return status;
  if (symbols.size() > kMaximumIntermediateTokens)
    return {ErrorCode::kInvalidArgument,
            "BPE intermediate token count exceeds the safety limit"};
  std::vector<TokenId> pieces;
  pieces.reserve(symbols.size());
  for (const auto symbol : symbols) {
    const auto piece = data.vocab.find(symbol);
    if (piece != data.vocab.end())
      pieces.push_back(piece->second);
    else if (data.specials[0].has_value())
      pieces.push_back(*data.specials[0]);
    else
      return {ErrorCode::kInvalidArgument,
              "BPE tokenizer has no vocabulary entry for initial symbol '" +
                  std::string{symbol} + "'"};
  }
  status = apply_bpe(data, &pieces);
  if (!status.ok())
    return status;
  return emit_bpe_pieces(pieces, options, output);
}

Status tokenize_bpe_without_literals(const TokenizerData &data,
                                     const std::string_view input,
                                     const TokenizerEncodeOptions &options,
                                     std::vector<TokenId> *const output) {
  std::vector<std::string_view> segments;
  auto status = pretokenize(data, input, &segments);
  if (!status.ok())
    return status;
  for (const auto segment : segments) {
    status = tokenize_bpe_segment(data, segment, options, output);
    if (!status.ok())
      return status;
  }
  return Status::Ok();
}

Status tokenize_bpe(const TokenizerData &data, const std::string_view input,
                    const TokenizerEncodeOptions &options,
                    std::vector<TokenId> *const output) {
  const auto emit = [&](const std::string_view literal_free_input) {
    return tokenize_bpe_without_literals(data, literal_free_input, options,
                                         output);
  };
  return tokenize_literal_aware(data, input, options, output, emit);
}

Status tokenize_bpe_raw_literals(const TokenizerData &data,
                                 const std::string_view input,
                                 const TokenizerEncodeOptions &options,
                                 std::vector<TokenId> *const output) {
  auto status = validate_raw_input_size(input, options);
  if (!status.ok())
    return status;
  std::size_t normalized_bytes = 0;
  const auto emit = [&](const std::string_view raw_literal_free_input) {
    std::string normalized;
    auto segment_options = options;
    segment_options.raw_byte_limit = 0;
    auto segment_status =
        normalize(data, raw_literal_free_input, segment_options, &normalized);
    if (!segment_status.ok())
      return segment_status;
    if (normalized.size() > kMaximumNormalizedBytes - normalized_bytes)
      return Status{ErrorCode::kInvalidArgument,
                    "tokenizer normalization exceeds the byte safety limit"};
    normalized_bytes += normalized.size();
    return tokenize_bpe_without_literals(data, normalized, options, output);
  };
  return tokenize_literal_aware(data, input, options, output, emit);
}

Status tokenize_byte_bpe(const TokenizerData &data,
                         const std::string_view input,
                         const TokenizerEncodeOptions &options,
                         std::vector<TokenId> *const output) {
  std::string prefixed;
  std::string_view bytes = input;
  if (data.add_prefix_space && (bytes.empty() || bytes.front() != ' ')) {
    if (bytes.size() == kMaximumNormalizedBytes)
      return {ErrorCode::kInvalidArgument,
              "byte BPE prefix exceeds the byte safety limit"};
    prefixed.reserve(bytes.size() + 1U);
    prefixed.push_back(' ');
    prefixed.append(bytes);
    bytes = prefixed;
  }
  if (bytes.size() > kMaximumIntermediateTokens)
    return {ErrorCode::kInvalidArgument,
            "byte BPE intermediate token count exceeds the safety limit"};
  std::vector<TokenId> pieces;
  pieces.reserve(bytes.size());
  for (const char raw_byte : bytes)
    pieces.push_back(data.byte_encoder[static_cast<unsigned char>(raw_byte)]);
  auto status = apply_bpe(data, &pieces);
  if (!status.ok())
    return status;
  return emit_bpe_pieces(pieces, options, output);
}

Status tokenize_wordpiece_word(const TokenizerData &data,
                               const std::string_view word,
                               const TokenizerEncodeOptions &options,
                               std::vector<TokenId> *const output) {
  std::vector<std::string_view> symbols;
  auto status = utf8_symbols(word, &symbols);
  if (!status.ok())
    return status;
  if (symbols.size() > data.maximum_word_bytes)
    return append_token(*data.specials[0], options, output);
  std::vector<std::size_t> offsets;
  offsets.reserve(symbols.size() + 1U);
  std::size_t byte_offset = 0;
  offsets.push_back(0);
  for (const auto symbol : symbols) {
    byte_offset += symbol.size();
    offsets.push_back(byte_offset);
  }
  std::vector<TokenId> word_tokens;
  std::size_t start = 0;
  while (start < symbols.size()) {
    bool matched = false;
    for (std::size_t end = symbols.size(); end > start; --end) {
      std::string candidate;
      if (start != 0)
        candidate = data.continuation_prefix;
      candidate.append(
          word.substr(offsets[start], offsets[end] - offsets[start]));
      const auto piece = data.vocab.find(candidate);
      if (piece == data.vocab.end())
        continue;
      if (word_tokens.size() >= kMaximumIntermediateTokens)
        return {ErrorCode::kInvalidArgument,
                "WordPiece output exceeds the token safety limit"};
      word_tokens.push_back(piece->second);
      start = end;
      matched = true;
      break;
    }
    if (!matched)
      return append_token(*data.specials[0], options, output);
  }
  if (word_tokens.size() >
      kMaximumIntermediateTokens -
          std::min(kMaximumIntermediateTokens, output->size()))
    return {ErrorCode::kInvalidArgument,
            "tokenizer output exceeds the token safety limit"};
  if (options.token_limit != 0 &&
      word_tokens.size() >
          options.token_limit - std::min(options.token_limit, output->size()))
    return {ErrorCode::kInvalidArgument,
            "tokenizer output exceeds configured token limit"};
  output->insert(output->end(), word_tokens.begin(), word_tokens.end());
  return Status::Ok();
}

Status tokenize_wordpiece(const TokenizerData &data,
                          const std::string_view input,
                          const TokenizerEncodeOptions &options,
                          std::vector<TokenId> *const output) {
  std::vector<std::string_view> words;
  auto status = pretokenize(data, input, &words);
  if (!status.ok())
    return status;
  for (const auto word : words) {
    status = tokenize_wordpiece_word(data, word, options, output);
    if (!status.ok())
      return status;
  }
  return Status::Ok();
}

Status apply_post_processor(const TokenizerData &data,
                            const TokenizerEncodeOptions &options,
                            std::vector<TokenId> *const tokens) {
  if (options.add_special_tokens) {
    const auto extra = data.prefix_ids.size() + data.suffix_ids.size();
    if (tokens->size() > kMaximumIntermediateTokens ||
        extra > kMaximumIntermediateTokens - tokens->size())
      return {ErrorCode::kInvalidArgument,
              "tokenizer post-processor exceeds the token safety limit"};
    if (options.token_limit != 0 &&
        (tokens->size() > options.token_limit ||
         extra > options.token_limit - tokens->size()))
      return {ErrorCode::kInvalidArgument,
              "tokenizer post-processor exceeds configured token limit"};
    std::vector<TokenId> decorated;
    decorated.reserve(tokens->size() + extra);
    decorated.insert(decorated.end(), data.prefix_ids.begin(),
                     data.prefix_ids.end());
    decorated.insert(decorated.end(), tokens->begin(), tokens->end());
    decorated.insert(decorated.end(), data.suffix_ids.begin(),
                     data.suffix_ids.end());
    tokens->swap(decorated);
  }
  if (options.pad_to_length == 0)
    return Status::Ok();
  if (options.pad_to_length > kMaximumIntermediateTokens)
    return {ErrorCode::kInvalidArgument,
            "tokenizer padding exceeds the token safety limit"};
  if (options.token_limit != 0 && options.pad_to_length > options.token_limit)
    return {ErrorCode::kInvalidArgument,
            "tokenizer padding exceeds configured token limit"};
  if (tokens->size() > options.pad_to_length)
    return {ErrorCode::kInvalidArgument,
            "tokenizer output is longer than requested padding length"};
  if (tokens->size() == options.pad_to_length)
    return Status::Ok();
  if (data.padding_side == PaddingSide::kNone || !data.pad_id.has_value())
    return {ErrorCode::kUnsupported, "tokenizer asset does not define padding"};
  const auto count = options.pad_to_length - tokens->size();
  if (data.padding_side == PaddingSide::kRight)
    tokens->insert(tokens->end(), count, *data.pad_id);
  else
    tokens->insert(tokens->begin(), count, *data.pad_id);
  return Status::Ok();
}

Status encode_artifact(const TokenizerData &data,
                       const std::string_view sequence,
                       const TokenizerEncodeOptions &options,
                       std::vector<TokenId> *const tokens) {
  if (tokens == nullptr)
    return {ErrorCode::kInvalidArgument,
            "artifact tokenizer output pointer is null"};
  std::vector<TokenId> candidate;
  if (data.kind == ArtifactTokenizerKind::kBpe &&
      !data.literal_token_ids.empty() && !data.normalization.empty()) {
    auto status =
        tokenize_bpe_raw_literals(data, sequence, options, &candidate);
    if (!status.ok())
      return status;
    status = apply_post_processor(data, options, &candidate);
    if (!status.ok())
      return status;
    *tokens = std::move(candidate);
    return Status::Ok();
  }
  std::string normalized;
  auto status = normalize(data, sequence, options, &normalized);
  if (!status.ok())
    return status;
  switch (data.kind) {
  case ArtifactTokenizerKind::kCharacter:
  case ArtifactTokenizerKind::kSingleNucleotide:
    status = tokenize_character(data, normalized, options, &candidate);
    break;
  case ArtifactTokenizerKind::kKmer:
    status = tokenize_kmers(data, normalized, options, &candidate);
    break;
  case ArtifactTokenizerKind::kWordPiece:
    status = tokenize_wordpiece(data, normalized, options, &candidate);
    break;
  case ArtifactTokenizerKind::kBpe:
    status = tokenize_bpe(data, normalized, options, &candidate);
    break;
  case ArtifactTokenizerKind::kByteBpe:
    status = tokenize_byte_bpe(data, normalized, options, &candidate);
    break;
  case ArtifactTokenizerKind::kLongestMatch:
    status = tokenize_longest_match(data, normalized, options, &candidate);
    break;
  case ArtifactTokenizerKind::kKmerBpe:
    status = tokenize_kmer_bpe(data, normalized, options, &candidate);
    break;
  }
  if (!status.ok())
    return status;
  status = apply_post_processor(data, options, &candidate);
  if (!status.ok())
    return status;
  *tokens = std::move(candidate);
  return Status::Ok();
}

} // namespace

struct ArtifactTokenizer::Impl final {
  TokenizerData data;
};

ArtifactTokenizer::ArtifactTokenizer(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ArtifactTokenizer::ArtifactTokenizer(ArtifactTokenizer &&) noexcept = default;
ArtifactTokenizer &
ArtifactTokenizer::operator=(ArtifactTokenizer &&) noexcept = default;
ArtifactTokenizer::~ArtifactTokenizer() = default;

Status
ArtifactTokenizer::Load(const std::string &artifact_root,
                        const TokenizerAssetDescriptor &descriptor,
                        std::unique_ptr<ArtifactTokenizer> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "artifact tokenizer output pointer is null"};
  fs::path path;
  auto status = resolve_asset_path(artifact_root, descriptor, &path);
  if (!status.ok())
    return status;
  std::string bytes;
  std::string digest;
  status = read_and_hash(path, static_cast<std::size_t>(descriptor.size),
                         &bytes, &digest);
  if (!status.ok())
    return status;
  if (digest != descriptor.sha256)
    return tokenizer_format_error("descriptor SHA256 differs");
  auto impl = std::make_unique<Impl>();
  status = parse_tokenizer_manifest(bytes, &impl->data);
  if (!status.ok())
    return status;
  *output = std::unique_ptr<ArtifactTokenizer>(
      new ArtifactTokenizer{std::move(impl)});
  return Status::Ok();
}

Status ArtifactTokenizer::encode(const std::string_view sequence,
                                 const TokenizerEncodeOptions &options,
                                 std::vector<TokenId> *const tokens) const {
  if (!impl_)
    return {ErrorCode::kInternal, "artifact tokenizer is not initialized"};
  return encode_artifact(impl_->data, sequence, options, tokens);
}

ArtifactTokenizerKind ArtifactTokenizer::kind() const noexcept {
  return impl_->data.kind;
}

std::size_t ArtifactTokenizer::vocabulary_size() const noexcept {
  return impl_->data.vocabulary_size;
}

ArtifactTokenizerPaddingSide ArtifactTokenizer::padding_side() const noexcept {
  switch (impl_->data.padding_side) {
  case PaddingSide::kNone:
    return ArtifactTokenizerPaddingSide::kNone;
  case PaddingSide::kLeft:
    return ArtifactTokenizerPaddingSide::kLeft;
  case PaddingSide::kRight:
    return ArtifactTokenizerPaddingSide::kRight;
  }
  return ArtifactTokenizerPaddingSide::kNone;
}

std::optional<TokenId> ArtifactTokenizer::pad_token_id() const noexcept {
  return impl_->data.pad_id.has_value() ? impl_->data.pad_id
                                        : impl_->data.specials[1];
}

std::size_t ArtifactTokenizer::post_processor_prefix_size() const noexcept {
  return impl_->data.prefix_ids.size();
}

std::size_t ArtifactTokenizer::post_processor_suffix_size() const noexcept {
  return impl_->data.suffix_ids.size();
}

} // namespace evo
