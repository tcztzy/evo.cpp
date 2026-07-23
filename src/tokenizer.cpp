// SPDX-License-Identifier: Apache-2.0
#include "evo2c/tokenizer.hpp"

#include <limits>

namespace evo2c {

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

}  // namespace evo2c
