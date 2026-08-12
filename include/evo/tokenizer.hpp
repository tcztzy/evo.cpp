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

inline constexpr std::size_t kTokenizerVocabSize = 512;
inline constexpr TokenId kEosToken = 0;
inline constexpr TokenId kPadToken = 1;

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
