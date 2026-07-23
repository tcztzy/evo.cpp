// SPDX-License-Identifier: Apache-2.0
#include "evo2c/crc32.hpp"

#include <array>

namespace evo2c {
namespace {

using CrcTables = std::array<std::array<std::uint32_t, 256>, 8>;

const CrcTables &crc_tables() noexcept {
  static const CrcTables tables = [] {
    CrcTables values{};
    for (std::uint32_t index = 0; index < values[0].size(); ++index) {
      std::uint32_t value = index;
      for (int bit = 0; bit < 8; ++bit) {
        value = (value >> 1U) ^ ((value & 1U) != 0U ? 0xedb88320U : 0U);
      }
      values[0][index] = value;
    }
    for (std::size_t slice = 1; slice < values.size(); ++slice) {
      for (std::size_t index = 0; index < values[slice].size(); ++index) {
        const std::uint32_t previous = values[slice - 1][index];
        values[slice][index] =
            (previous >> 8U) ^ values[0][static_cast<std::uint8_t>(previous)];
      }
    }
    return values;
  }();
  return tables;
}

std::uint32_t read_u32_le(const std::uint8_t *const data) noexcept {
  return static_cast<std::uint32_t>(data[0]) |
         (static_cast<std::uint32_t>(data[1]) << 8U) |
         (static_cast<std::uint32_t>(data[2]) << 16U) |
         (static_cast<std::uint32_t>(data[3]) << 24U);
}

} // namespace

std::uint32_t crc32(const std::uint8_t *data, const std::size_t size) noexcept {
  std::uint32_t value = 0xffffffffU;
  const auto &tables = crc_tables();
  std::size_t index = 0;
  while (size - index >= 8) {
    const std::uint32_t first = value ^ read_u32_le(data + index);
    const std::uint32_t second = read_u32_le(data + index + 4);
    value = tables[7][static_cast<std::uint8_t>(first)] ^
            tables[6][static_cast<std::uint8_t>(first >> 8U)] ^
            tables[5][static_cast<std::uint8_t>(first >> 16U)] ^
            tables[4][static_cast<std::uint8_t>(first >> 24U)] ^
            tables[3][static_cast<std::uint8_t>(second)] ^
            tables[2][static_cast<std::uint8_t>(second >> 8U)] ^
            tables[1][static_cast<std::uint8_t>(second >> 16U)] ^
            tables[0][static_cast<std::uint8_t>(second >> 24U)];
    index += 8;
  }
  for (; index < size; ++index) {
    const auto table_index = static_cast<std::uint8_t>(value ^ data[index]);
    value = (value >> 8U) ^ tables[0][table_index];
  }
  return value ^ 0xffffffffU;
}

} // namespace evo2c
