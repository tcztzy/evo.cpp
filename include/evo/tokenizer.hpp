// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "evo/model_registry.hpp"
#include "evo/status.hpp"

namespace evo {

using TokenId = std::uint16_t;

inline constexpr std::size_t kEvo2TokenizerVocabSize = 512;
inline constexpr TokenId kEvo2EosToken = 0;
inline constexpr TokenId kEvo2PadToken = 1;
// Source-compatible aliases from the original single-architecture API.
inline constexpr std::size_t kTokenizerVocabSize = kEvo2TokenizerVocabSize;
inline constexpr TokenId kEosToken = kEvo2EosToken;
inline constexpr TokenId kPadToken = kEvo2PadToken;

// Evo 2 CharLevelTokenizer maps each UTF-8 byte to the numerically identical ID.
[[nodiscard]] std::vector<TokenId> encode_bytes(std::string_view bytes);

// IDs 256..511 have logits but no byte representation in the supported tokenizer.
[[nodiscard]] Status token_to_byte(TokenId token, std::uint8_t* byte);

// Architecture-aware raw biological sequence conversion. Causal tokenizers do
// not add boundaries; ESMC follows its official <cls> sequence <eos> template.
[[nodiscard]] Status encode_sequence(ArchitectureTokenizer tokenizer,
                                     std::string_view sequence,
                                     std::vector<TokenId> *tokens);
[[nodiscard]] Status decode_sequence_token(ArchitectureTokenizer tokenizer,
                                           TokenId token,
                                           std::uint8_t *byte);

}  // namespace evo
