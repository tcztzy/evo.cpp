// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "evo/status.hpp"

namespace evo {

enum class JsonType { kNull, kBoolean, kNumber, kString, kArray, kObject };

struct JsonValue final {
  JsonType type{JsonType::kNull};
  bool boolean{false};
  double number{0.0};
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue, std::less<>> object;

  [[nodiscard]] const JsonValue *find(std::string_view key) const noexcept;
};

[[nodiscard]] Status parse_json(std::string_view text, JsonValue *output,
                                std::size_t maximum_depth = 32);
void append_json_string(std::string *output, std::string_view value);

} // namespace evo
