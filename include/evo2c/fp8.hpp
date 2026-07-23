#pragma once

#include <cstdint>

namespace evo2c::fp8 {

inline constexpr float kE4m3fnMaximum = 448.0F;

[[nodiscard]] bool valid_scale(float scale) noexcept;

// Round-to-nearest-even E4M3FN with NVIDIA __NV_SATFINITE overflow semantics.
[[nodiscard]] std::uint8_t encode_e4m3fn(float value) noexcept;
[[nodiscard]] float decode_e4m3fn(std::uint8_t bits) noexcept;

// Equivalent to dequantize(E4M3FN(value * scale)) with a float32 reciprocal.
[[nodiscard]] float scaled_e4m3fn_roundtrip(float value, float scale) noexcept;

}  // namespace evo2c::fp8
