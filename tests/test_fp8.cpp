#include "evo2c/fp8.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void require(const bool condition, const char *const message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void test_pytorch_e4m3fn_vectors() {
  const std::vector<std::pair<float, std::uint8_t>> vectors{
      {-448.0F, 0xfe},       {-432.0F, 0xfe},      {-416.0F, 0xfd},
      {-1.25F, 0xba},        {-1.1875F, 0xba},     {-1.125F, 0xb9},
      {-1.0625F, 0xb8},      {-1.0F, 0xb8},        {-0.5F, 0xb0},
      {-0.015625F, 0x88},    {-0.00390625F, 0x82}, {-0.001953125F, 0x81},
      {-0.0009765625F, 0x80}, {-0.0F, 0x80},       {0.0F, 0x00},
      {0.0009765625F, 0x00}, {0.001953125F, 0x01}, {0.0029296875F, 0x02},
      {0.00390625F, 0x02},   {0.015625F, 0x08},    {0.5F, 0x30},
      {1.0F, 0x38},          {1.0625F, 0x38},      {1.125F, 0x39},
      {1.1875F, 0x3a},       {1.25F, 0x3a},        {416.0F, 0x7d},
      {432.0F, 0x7e},        {448.0F, 0x7e},
  };
  for (const auto &[value, expected] : vectors) {
    const auto actual = evo2c::fp8::encode_e4m3fn(value);
    if (actual != expected) {
      std::cerr << "FAIL: E4M3FN(" << value << ") produced "
                << static_cast<unsigned>(actual) << ", expected "
                << static_cast<unsigned>(expected) << '\n';
      ++failures;
    }
  }
}

void test_decode_and_satfinite() {
  require(evo2c::fp8::decode_e4m3fn(0x01) == 0.001953125F,
          "smallest E4M3FN subnormal decodes exactly");
  require(evo2c::fp8::decode_e4m3fn(0x7e) == 448.0F,
          "maximum E4M3FN value decodes exactly");
  require(evo2c::fp8::decode_e4m3fn(0xfe) == -448.0F,
          "negative maximum E4M3FN value decodes exactly");
  require(std::isnan(evo2c::fp8::decode_e4m3fn(0x7f)),
          "E4M3FN NaN code decodes to NaN");
  require(evo2c::fp8::encode_e4m3fn(1000.0F) == 0x7e,
          "positive overflow saturates to finite maximum");
  require(evo2c::fp8::encode_e4m3fn(-1000.0F) == 0xfe,
          "negative overflow saturates to finite maximum");
  require(evo2c::fp8::encode_e4m3fn(
              std::numeric_limits<float>::infinity()) == 0x7e,
          "positive infinity uses NVIDIA satfinite semantics");
  require(evo2c::fp8::encode_e4m3fn(
              -std::numeric_limits<float>::infinity()) == 0xfe,
          "negative infinity uses NVIDIA satfinite semantics");
}

void test_scaled_roundtrip() {
  require(evo2c::fp8::scaled_e4m3fn_roundtrip(1.1F, 2.0F) == 1.125F,
          "scaled roundtrip quantizes in FP32 and descales");
  require(evo2c::fp8::valid_scale(1.0F), "finite positive FP8 scale is valid");
  require(!evo2c::fp8::valid_scale(0.0F), "zero FP8 scale is invalid");
  require(!evo2c::fp8::valid_scale(
              std::numeric_limits<float>::infinity()),
          "infinite FP8 scale is invalid");
}

}  // namespace

int main() {
  test_pytorch_e4m3fn_vectors();
  test_decode_and_satfinite();
  test_scaled_roundtrip();
  if (failures != 0) {
    std::cerr << failures << " FP8 test(s) failed\n";
    return 1;
  }
  std::cout << "FP8 tests passed\n";
  return 0;
}
