#include "evo/fp8.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace evo::fp8 {
namespace {

[[nodiscard]] float decode_positive(const std::uint8_t magnitude) noexcept {
  const auto exponent = static_cast<unsigned>((magnitude >> 3U) & 0x0fU);
  const auto mantissa = static_cast<unsigned>(magnitude & 0x07U);
  if (exponent == 0U) {
    return std::ldexp(static_cast<float>(mantissa), -9);
  }
  return std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                    static_cast<int>(exponent) - 7);
}

}  // namespace

bool valid_scale(const float scale) noexcept {
  return scale > 0.0F && std::isfinite(scale);
}

std::uint8_t encode_e4m3fn(const float value) noexcept {
  const bool negative = std::signbit(value);
  const auto sign = static_cast<std::uint8_t>(negative ? 0x80U : 0x00U);
  if (std::isnan(value)) {
    return static_cast<std::uint8_t>(sign | 0x7fU);
  }

  const float magnitude = std::fabs(value);
  if (!std::isfinite(magnitude) || magnitude >= kE4m3fnMaximum) {
    return static_cast<std::uint8_t>(sign | 0x7eU);
  }

  std::uint8_t lower = 0U;
  std::uint8_t upper = 0x7eU;
  while (static_cast<unsigned>(upper) - static_cast<unsigned>(lower) > 1U) {
    const auto midpoint = static_cast<std::uint8_t>(
        (static_cast<unsigned>(lower) + static_cast<unsigned>(upper)) / 2U);
    if (decode_positive(midpoint) <= magnitude) {
      lower = midpoint;
    } else {
      upper = midpoint;
    }
  }

  const float lower_distance = magnitude - decode_positive(lower);
  const float upper_distance = decode_positive(upper) - magnitude;
  std::uint8_t rounded = lower;
  if (upper_distance < lower_distance ||
      (upper_distance == lower_distance && (upper & 1U) == 0U)) {
    rounded = upper;
  }
  return static_cast<std::uint8_t>(sign | rounded);
}

float decode_e4m3fn(const std::uint8_t bits) noexcept {
  const auto magnitude = static_cast<std::uint8_t>(bits & 0x7fU);
  if (magnitude == 0x7fU) {
    return std::copysign(std::numeric_limits<float>::quiet_NaN(),
                         (bits & 0x80U) == 0U ? 1.0F : -1.0F);
  }
  return std::copysign(decode_positive(magnitude),
                       (bits & 0x80U) == 0U ? 1.0F : -1.0F);
}

float scaled_e4m3fn_roundtrip(const float value,
                              const float scale) noexcept {
  if (!valid_scale(scale)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float quantized = decode_e4m3fn(encode_e4m3fn(value * scale));
  return quantized * (1.0F / scale);
}

}  // namespace evo::fp8
