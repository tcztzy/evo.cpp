// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "evo/model_registry.hpp"
#include "evo/status.hpp"

namespace evo {

// GENEB includes vocabularies larger than 65,536 entries (for example the
// GPT2-Gene checkpoints), so token IDs must remain lossless across every
// native backend and the uint32 C ABI.
using TokenId = std::uint32_t;
static_assert(sizeof(TokenId) == sizeof(std::uint32_t),
              "the public token type must remain uint32");

[[nodiscard]] inline constexpr bool
token_id_in_vocabulary(const TokenId token,
                       const std::size_t vocab_size) noexcept {
  return static_cast<std::size_t>(token) < vocab_size;
}

[[nodiscard]] inline constexpr bool
token_vocabulary_size_supported(const std::size_t vocab_size) noexcept {
  return vocab_size == 0 || static_cast<std::uint64_t>(vocab_size - 1U) <=
                                std::numeric_limits<TokenId>::max();
}

static_assert(token_vocabulary_size_supported(0));
static_assert(token_vocabulary_size_supported(115000));
static_assert(sizeof(std::size_t) < 8 ||
              token_vocabulary_size_supported(
                  static_cast<std::size_t>(std::uint64_t{1} << 32U)));
static_assert(sizeof(std::size_t) < 8 ||
              !token_vocabulary_size_supported(
                  static_cast<std::size_t>((std::uint64_t{1} << 32U) + 1U)));

inline constexpr std::size_t kEvo2TokenizerVocabSize = 512;
inline constexpr TokenId kEvo2EosToken = 0;
inline constexpr TokenId kEvo2PadToken = 1;
// Source-compatible aliases from the original single-architecture API.
inline constexpr std::size_t kTokenizerVocabSize = kEvo2TokenizerVocabSize;
inline constexpr TokenId kEosToken = kEvo2EosToken;
inline constexpr TokenId kPadToken = kEvo2PadToken;
// Artifact tokenizers may compress raw input (BPE/k-mer) or expand it through
// normalization.  Frontends use this independent pre-tokenization safety cap;
// model context limits are enforced on the resulting token stream.
inline constexpr std::size_t kArtifactTokenizerInputSafetyLimit =
    16U * 1024U * 1024U;

// Evo 2 CharLevelTokenizer maps each UTF-8 byte to the numerically identical
// ID.
[[nodiscard]] std::vector<TokenId> encode_bytes(std::string_view bytes);

// IDs 256..511 have logits but no byte representation in the supported
// tokenizer.
[[nodiscard]] Status token_to_byte(TokenId token, std::uint8_t *byte);

// Architecture-aware raw biological sequence conversion. Causal tokenizers do
// not add boundaries; ESMC follows its official <cls> sequence <eos> template.
[[nodiscard]] Status encode_sequence(ArchitectureTokenizer tokenizer,
                                     std::string_view sequence,
                                     std::vector<TokenId> *tokens);
[[nodiscard]] Status decode_sequence_token(ArchitectureTokenizer tokenizer,
                                           TokenId token, std::uint8_t *byte);

// Canonical, offline tokenizer asset used by non-legacy artifacts. The
// descriptor is stored in the model artifact; the referenced manifest is
// verified before any tokenizer configuration is parsed.
struct TokenizerAssetDescriptor final {
  std::string path;
  std::string sha256;
  std::uint64_t size{0};
};

enum class ArtifactTokenizerKind {
  kCharacter,
  kSingleNucleotide,
  kKmer,
  kWordPiece,
  kBpe,
  kByteBpe,
  kLongestMatch,
  kKmerBpe,
};

enum class ArtifactTokenizerPaddingSide { kNone, kLeft, kRight };

struct TokenizerEncodeOptions final {
  bool add_special_tokens{true};
  // Zero means that no padding is requested.
  std::size_t pad_to_length{0};
  // Zero means that the caller did not impose an additional limit. When
  // present, the raw-byte limit is checked before normalization allocates.
  std::size_t raw_byte_limit{0};
  // Zero means no additional token limit. This limit is checked while
  // tokenizing and before padding allocates.
  std::size_t token_limit{0};
};

class ArtifactTokenizer final {
public:
  ArtifactTokenizer(ArtifactTokenizer &&) noexcept;
  ArtifactTokenizer &operator=(ArtifactTokenizer &&) noexcept;
  ~ArtifactTokenizer();

  ArtifactTokenizer(const ArtifactTokenizer &) = delete;
  ArtifactTokenizer &operator=(const ArtifactTokenizer &) = delete;

  // artifact_root is an already-local artifact directory. descriptor.path
  // must remain beneath it and identify a regular, non-symlink file whose
  // exact size and SHA256 match the descriptor.
  [[nodiscard]] static Status Load(const std::string &artifact_root,
                                   const TokenizerAssetDescriptor &descriptor,
                                   std::unique_ptr<ArtifactTokenizer> *output);

  [[nodiscard]] Status encode(std::string_view sequence,
                              const TokenizerEncodeOptions &options,
                              std::vector<TokenId> *tokens) const;

  [[nodiscard]] ArtifactTokenizerKind kind() const noexcept;
  [[nodiscard]] std::size_t vocabulary_size() const noexcept;
  [[nodiscard]] ArtifactTokenizerPaddingSide padding_side() const noexcept;
  [[nodiscard]] std::optional<TokenId> pad_token_id() const noexcept;
  [[nodiscard]] std::size_t post_processor_prefix_size() const noexcept;
  [[nodiscard]] std::size_t post_processor_suffix_size() const noexcept;

private:
  struct Impl;
  explicit ArtifactTokenizer(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

} // namespace evo
