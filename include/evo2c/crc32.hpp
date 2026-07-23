// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace evo2c {

[[nodiscard]] std::uint32_t crc32(const std::uint8_t* data, std::size_t size) noexcept;

}  // namespace evo2c

