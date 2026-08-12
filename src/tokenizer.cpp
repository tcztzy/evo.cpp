// SPDX-License-Identifier: Apache-2.0
#include "evo/tokenizer.hpp"

#include <limits>

namespace evo {

std::vector<TokenId> encode_bytes(const std::string_view bytes) {
  std::vector<TokenId> tokens;
  tokens.reserve(bytes.size());
  for (const char character : bytes) {
    tokens.push_back(static_cast<TokenId>(static_cast<unsigned char>(character)));
  }
  return tokens;
}

Status token_to_byte(const TokenId token, std::uint8_t* const byte) {
  if (byte == nullptr) {
    return {ErrorCode::kInvalidArgument, "token_to_byte output pointer is null"};
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
  if (tokenizer == ArchitectureTokenizer::kByteIdentity) {
    *tokens = encode_bytes(sequence);
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
                             const TokenId token,
                             std::uint8_t *const byte) {
  if (tokenizer == ArchitectureTokenizer::kByteIdentity)
    return token_to_byte(token, byte);
  if (byte == nullptr)
    return {ErrorCode::kInvalidArgument, "byte output is null"};
  constexpr char alphabet[] = "ACGTN";
  if (token < 7 || token > 11)
    return {ErrorCode::kModelFormat,
            "HyenaDNA token has no raw DNA byte representation"};
  *byte = static_cast<std::uint8_t>(alphabet[token - 7]);
  return Status::Ok();
}

}  // namespace evo
