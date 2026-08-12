// SPDX-License-Identifier: Apache-2.0
#include "evo/json.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

namespace evo {
namespace {

class Parser final {
public:
  Parser(const std::string_view text, const std::size_t maximum_depth)
      : text_(text), maximum_depth_(maximum_depth) {}

  [[nodiscard]] Status parse(JsonValue *const output) {
    if (output == nullptr)
      return {ErrorCode::kInvalidArgument, "JSON output is null"};
    skip_space();
    auto status = value(0, output);
    if (!status.ok())
      return status;
    skip_space();
    if (position_ != text_.size())
      return error("trailing bytes after JSON value");
    return Status::Ok();
  }

private:
  [[nodiscard]] Status error(const std::string_view message) const {
    return {ErrorCode::kInvalidArgument, "JSON byte " +
                                             std::to_string(position_) + ": " +
                                             std::string{message}};
  }

  void skip_space() noexcept {
    while (position_ < text_.size() &&
           (text_[position_] == ' ' || text_[position_] == '\t' ||
            text_[position_] == '\r' || text_[position_] == '\n')) {
      ++position_;
    }
  }

  bool consume(const char expected) noexcept {
    if (position_ >= text_.size() || text_[position_] != expected)
      return false;
    ++position_;
    return true;
  }

  [[nodiscard]] Status literal(const std::string_view expected,
                               const JsonType type, const bool boolean,
                               JsonValue *const output) {
    if (text_.substr(position_, expected.size()) != expected)
      return error("invalid literal");
    position_ += expected.size();
    *output = JsonValue{};
    output->type = type;
    output->boolean = boolean;
    return Status::Ok();
  }

  static int hex(const char value) noexcept {
    if (value >= '0' && value <= '9')
      return value - '0';
    if (value >= 'a' && value <= 'f')
      return value - 'a' + 10;
    if (value >= 'A' && value <= 'F')
      return value - 'A' + 10;
    return -1;
  }

  [[nodiscard]] Status code_unit(std::uint32_t *const output) {
    if (position_ + 4 > text_.size())
      return error("incomplete Unicode escape");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
      const int digit = hex(text_[position_++]);
      if (digit < 0)
        return error("invalid Unicode escape");
      value = value * 16U + static_cast<std::uint32_t>(digit);
    }
    *output = value;
    return Status::Ok();
  }

  static void append_utf8(const std::uint32_t codepoint,
                          std::string *const output) {
    if (codepoint <= 0x7fU) {
      output->push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
      output->push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
      output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else if (codepoint <= 0xffffU) {
      output->push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
      output->push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
      output->push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
      output->push_back(
          static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
      output->push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
  }

  [[nodiscard]] Status string(std::string *const output) {
    if (!consume('"'))
      return error("expected string");
    output->clear();
    while (position_ < text_.size()) {
      const unsigned char byte = static_cast<unsigned char>(text_[position_++]);
      if (byte == '"')
        return Status::Ok();
      if (byte < 0x20U)
        return error("unescaped control byte in string");
      if (byte != '\\') {
        output->push_back(static_cast<char>(byte));
        continue;
      }
      if (position_ >= text_.size())
        return error("incomplete string escape");
      const char escaped = text_[position_++];
      switch (escaped) {
      case '"':
      case '\\':
      case '/':
        output->push_back(escaped);
        break;
      case 'b':
        output->push_back('\b');
        break;
      case 'f':
        output->push_back('\f');
        break;
      case 'n':
        output->push_back('\n');
        break;
      case 'r':
        output->push_back('\r');
        break;
      case 't':
        output->push_back('\t');
        break;
      case 'u': {
        std::uint32_t first = 0;
        auto status = code_unit(&first);
        if (!status.ok())
          return status;
        std::uint32_t codepoint = first;
        if (first >= 0xd800U && first <= 0xdbffU) {
          if (position_ + 2 > text_.size() || text_[position_] != '\\' ||
              text_[position_ + 1] != 'u') {
            return error("high surrogate is not followed by a low surrogate");
          }
          position_ += 2;
          std::uint32_t second = 0;
          status = code_unit(&second);
          if (!status.ok())
            return status;
          if (second < 0xdc00U || second > 0xdfffU)
            return error("invalid low surrogate");
          codepoint =
              0x10000U + ((first - 0xd800U) << 10U) + (second - 0xdc00U);
        } else if (first >= 0xdc00U && first <= 0xdfffU) {
          return error("unexpected low surrogate");
        }
        append_utf8(codepoint, output);
        break;
      }
      default:
        return error("unknown string escape");
      }
    }
    return error("unterminated string");
  }

  [[nodiscard]] Status number(JsonValue *const output) {
    const std::size_t start = position_;
    consume('-');
    if (position_ >= text_.size())
      return error("incomplete number");
    if (text_[position_] == '0') {
      ++position_;
      if (position_ < text_.size() && text_[position_] >= '0' &&
          text_[position_] <= '9') {
        return error("number has a leading zero");
      }
    } else {
      if (text_[position_] < '1' || text_[position_] > '9')
        return error("invalid number integer part");
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9')
        ++position_;
    }
    if (consume('.')) {
      const std::size_t fraction = position_;
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9')
        ++position_;
      if (fraction == position_)
        return error("number fraction is empty");
    }
    if (position_ < text_.size() &&
        (text_[position_] == 'e' || text_[position_] == 'E')) {
      ++position_;
      if (position_ < text_.size() &&
          (text_[position_] == '+' || text_[position_] == '-'))
        ++position_;
      const std::size_t exponent = position_;
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9')
        ++position_;
      if (exponent == position_)
        return error("number exponent is empty");
    }
    const std::string owned{text_.substr(start, position_ - start)};
    char *end = nullptr;
    errno = 0;
    const double parsed = std::strtod(owned.c_str(), &end);
    if (errno == ERANGE || end != owned.c_str() + owned.size() ||
        !std::isfinite(parsed)) {
      return error("number is outside finite F64 range");
    }
    *output = JsonValue{};
    output->type = JsonType::kNumber;
    output->number = parsed;
    return Status::Ok();
  }

  [[nodiscard]] Status value(const std::size_t depth, JsonValue *const output) {
    if (depth > maximum_depth_)
      return error("maximum nesting depth exceeded");
    skip_space();
    if (position_ >= text_.size())
      return error("expected value");
    switch (text_[position_]) {
    case 'n':
      return literal("null", JsonType::kNull, false, output);
    case 't':
      return literal("true", JsonType::kBoolean, true, output);
    case 'f':
      return literal("false", JsonType::kBoolean, false, output);
    case '"':
      *output = JsonValue{};
      output->type = JsonType::kString;
      return string(&output->string);
    case '[':
      return array(depth, output);
    case '{':
      return object(depth, output);
    default:
      return number(output);
    }
  }

  [[nodiscard]] Status array(const std::size_t depth, JsonValue *const output) {
    consume('[');
    *output = JsonValue{};
    output->type = JsonType::kArray;
    skip_space();
    if (consume(']'))
      return Status::Ok();
    while (true) {
      JsonValue element;
      auto status = value(depth + 1, &element);
      if (!status.ok())
        return status;
      output->array.push_back(std::move(element));
      skip_space();
      if (consume(']'))
        return Status::Ok();
      if (!consume(','))
        return error("expected ',' or ']' in array");
      skip_space();
    }
  }

  [[nodiscard]] Status object(const std::size_t depth,
                              JsonValue *const output) {
    consume('{');
    *output = JsonValue{};
    output->type = JsonType::kObject;
    skip_space();
    if (consume('}'))
      return Status::Ok();
    while (true) {
      std::string key;
      auto status = string(&key);
      if (!status.ok())
        return status;
      skip_space();
      if (!consume(':'))
        return error("expected ':' after object key");
      JsonValue member;
      status = value(depth + 1, &member);
      if (!status.ok())
        return status;
      if (!output->object.emplace(std::move(key), std::move(member)).second)
        return error("duplicate object key");
      skip_space();
      if (consume('}'))
        return Status::Ok();
      if (!consume(','))
        return error("expected ',' or '}' in object");
      skip_space();
    }
  }

  std::string_view text_;
  std::size_t maximum_depth_{0};
  std::size_t position_{0};
};

} // namespace

const JsonValue *JsonValue::find(const std::string_view key) const noexcept {
  if (type != JsonType::kObject)
    return nullptr;
  const auto iterator = object.find(key);
  return iterator == object.end() ? nullptr : &iterator->second;
}

Status parse_json(const std::string_view text, JsonValue *const output,
                  const std::size_t maximum_depth) {
  if (maximum_depth == 0)
    return {ErrorCode::kInvalidArgument, "JSON maximum depth must be positive"};
  return Parser{text, maximum_depth}.parse(output);
}

void append_json_string(std::string *const output,
                        const std::string_view value) {
  if (output == nullptr)
    return;
  constexpr char hex[] = "0123456789abcdef";
  output->push_back('"');
  for (const char raw : value) {
    const auto byte = static_cast<unsigned char>(raw);
    switch (byte) {
    case '"':
      output->append("\\\"");
      break;
    case '\\':
      output->append("\\\\");
      break;
    case '\b':
      output->append("\\b");
      break;
    case '\f':
      output->append("\\f");
      break;
    case '\n':
      output->append("\\n");
      break;
    case '\r':
      output->append("\\r");
      break;
    case '\t':
      output->append("\\t");
      break;
    default:
      if (byte < 0x20U || byte >= 0x7fU) {
        output->append("\\u00");
        output->push_back(hex[byte >> 4U]);
        output->push_back(hex[byte & 0x0fU]);
      } else {
        output->push_back(static_cast<char>(byte));
      }
      break;
    }
  }
  output->push_back('"');
}

} // namespace evo
