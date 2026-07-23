// SPDX-License-Identifier: Apache-2.0
#include "evo2c/crc32.hpp"

#include <array>

namespace evo2c {
namespace {

const std::array<std::uint32_t, 256>& crc_table() noexcept {
  static const std::array<std::uint32_t, 256> table = [] {
    std::array<std::uint32_t, 256> values{};
    for (std::uint32_t index = 0; index < values.size(); ++index) {
      std::uint32_t value = index;
      for (int bit = 0; bit < 8; ++bit) {
        value = (value >> 1U) ^ ((value & 1U) != 0U ? 0xedb88320U : 0U);
      }
      values[index] = value;
    }
    return values;
  }();
  return table;
}

}  // namespace

std::uint32_t crc32(const std::uint8_t* data, const std::size_t size) noexcept {
  std::uint32_t value = 0xffffffffU;
  const auto& table = crc_table();
  for (std::size_t index = 0; index < size; ++index) {
    const auto table_index = static_cast<std::uint8_t>(value ^ data[index]);
    value = (value >> 8U) ^ table[table_index];
  }
  return value ^ 0xffffffffU;
}

}  // namespace evo2c

