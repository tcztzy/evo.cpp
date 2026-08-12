// SPDX-License-Identifier: Apache-2.0
#include "evo/tokenizer.hpp"

#include <array>
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
                             const TokenId token,
                             std::uint8_t *const byte) {
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

}  // namespace evo
