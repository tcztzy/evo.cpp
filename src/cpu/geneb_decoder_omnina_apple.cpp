// SPDX-License-Identifier: Apache-2.0
#include "geneb_decoder_omnina_apple.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#endif

namespace evo::cpu::detail {
namespace {

[[maybe_unused]] Status unsupported_host() {
  return {ErrorCode::kUnsupported,
          "decoder.f32_math_kernel=torch-2.7.1-apple-arm64-exact-v1 "
          "requires Apple arm64"};
}

#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))

// The three vector elementary functions below are a minimal derivative of
// SLEEF's AdvSIMD SP u10 implementation at
// shibatch/sleef@56e1f79cb140fb9326d612d0be06b5250565cade.  SLEEF is
// distributed under the Boost Software License 1.0; see NOTICE.  OmniNA's
// closed position domain is [0, 2047], so rempif can only address the first
// four SP reduction constants.  Keeping only that reachable closure avoids
// vendoring SLEEF's unrelated functions and full large-argument tables.
using Vec = float32x4_t;
using IntVec = int32x4_t;
using Mask = uint32x4_t;

struct Float2 final {
  Vec high;
  Vec low;
};

struct FloatInt final {
  Vec value;
  IntVec integer;
};

struct Float2Int final {
  Float2 value;
  IntVec integer;
};

inline Vec splat(const float value) noexcept { return vdupq_n_f32(value); }

inline IntVec splat_int(const int value) noexcept { return vdupq_n_s32(value); }

inline Vec fmadd(const Vec left, const Vec right, const Vec addend) noexcept {
  return vfmaq_f32(addend, left, right);
}

inline Vec fmsub(const Vec left, const Vec right,
                 const Vec subtrahend) noexcept {
  return vfmaq_f32(vnegq_f32(subtrahend), left, right);
}

inline Vec select(const Mask mask, const Vec selected,
                  const Vec fallback) noexcept {
  return vbslq_f32(mask, selected, fallback);
}

inline IntVec select(const Mask mask, const IntVec selected,
                     const IntVec fallback) noexcept {
  return vbslq_s32(mask, selected, fallback);
}

inline Float2 select(const Mask mask, const Float2 selected,
                     const Float2 fallback) noexcept {
  return {select(mask, selected.high, fallback.high),
          select(mask, selected.low, fallback.low)};
}

inline bool all_true(const Mask mask) noexcept {
  const uint32x2_t halves = vand_u32(vget_low_u32(mask), vget_high_u32(mask));
  return vget_lane_u32(vpmin_u32(halves, halves), 0) != 0U;
}

inline Float2 normalize(const Float2 value) noexcept {
  const Vec high = vaddq_f32(value.high, value.low);
  return {high, vaddq_f32(vsubq_f32(value.high, high), value.low)};
}

inline Float2 df_add(const Vec left, const Vec right) noexcept {
  const Vec high = vaddq_f32(left, right);
  return {high, vaddq_f32(vsubq_f32(left, high), right)};
}

inline Float2 df_add_exact(const Vec left, const Vec right) noexcept {
  const Vec high = vaddq_f32(left, right);
  const Vec rounded_right = vsubq_f32(high, left);
  const Vec low = vaddq_f32(vsubq_f32(left, vsubq_f32(high, rounded_right)),
                            vsubq_f32(right, rounded_right));
  return {high, low};
}

inline Float2 df_add(const Float2 left, const Vec right) noexcept {
  const Vec high = vaddq_f32(left.high, right);
  const Vec low =
      vaddq_f32(vaddq_f32(vsubq_f32(left.high, high), right), left.low);
  return {high, low};
}

inline Float2 df_add_exact(const Float2 left, const Vec right) noexcept {
  const Vec high = vaddq_f32(left.high, right);
  const Vec rounded_right = vsubq_f32(high, left.high);
  const Vec correction =
      vaddq_f32(vsubq_f32(left.high, vsubq_f32(high, rounded_right)),
                vsubq_f32(right, rounded_right));
  return {high, vaddq_f32(correction, left.low)};
}

inline Float2 df_add(const Vec left, const Float2 right) noexcept {
  const Vec high = vaddq_f32(left, right.high);
  const Vec low =
      vaddq_f32(vaddq_f32(vsubq_f32(left, high), right.high), right.low);
  return {high, low};
}

inline Float2 df_add_exact(const Float2 left, const Float2 right) noexcept {
  const Vec high = vaddq_f32(left.high, right.high);
  const Vec rounded_right = vsubq_f32(high, left.high);
  const Vec correction =
      vaddq_f32(vsubq_f32(left.high, vsubq_f32(high, rounded_right)),
                vsubq_f32(right.high, rounded_right));
  return {high, vaddq_f32(correction, vaddq_f32(left.low, right.low))};
}

inline Float2 df_multiply(const Vec left, const Vec right) noexcept {
  const Vec high = vmulq_f32(left, right);
  return {high, fmsub(left, right, high)};
}

inline Float2 df_multiply(const Float2 left, const Vec right) noexcept {
  const Vec high = vmulq_f32(left.high, right);
  return {high, fmadd(left.low, right, fmsub(left.high, right, high))};
}

inline Float2 df_multiply(const Float2 left, const Float2 right) noexcept {
  const Vec high = vmulq_f32(left.high, right.high);
  const Vec low =
      fmadd(left.high, right.low,
            fmadd(left.low, right.high, fmsub(left.high, right.high, high)));
  return {high, low};
}

inline Vec df_multiply_result(const Float2 left, const Float2 right) noexcept {
  return fmadd(left.high, right.high,
               fmadd(left.low, right.high, vmulq_f32(left.high, right.low)));
}

inline Float2 df_square(const Float2 value) noexcept {
  const Vec high = vmulq_f32(value.high, value.high);
  return {high, fmadd(vaddq_f32(value.high, value.high), value.low,
                      fmsub(value.high, value.high, high))};
}

inline Vec multiply_sign(const Vec value, const Vec sign_source) noexcept {
  const Mask sign = vandq_u32(vreinterpretq_u32_f32(sign_source),
                              vreinterpretq_u32_f32(splat(-0.0F)));
  return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(value), sign));
}

inline bool is_negative_zero_lane(const float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits == 0x80000000U;
}

inline FloatInt rempi_sub(const Vec value) noexcept {
  const Vec rounded_quarters = vrndnq_f32(vmulq_f32(value, splat(4.0F)));
  const Vec rounded_integer = vrndnq_f32(value);
  const IntVec integer = vcvtq_s32_f32(
      vsubq_f32(rounded_quarters, vmulq_f32(rounded_integer, splat(4.0F))));
  return {vsubq_f32(value, vmulq_f32(rounded_quarters, splat(0.25F))), integer};
}

inline Float2Int rempi_bounded(const Vec value) noexcept {
  // First reachable Sleef_rempitabsp entry for all |value| < 2^25.
  constexpr float kInvTwoPi0 = 0.159154892F;
  constexpr float kInvTwoPi1 = 5.112411827e-08F;
  constexpr float kInvTwoPi2 = 3.626141271e-15F;
  constexpr float kInvTwoPi3 = -2.036222915e-22F;
  Float2 reduced = df_multiply(value, splat(kInvTwoPi0));
  auto sub = rempi_sub(reduced.high);
  IntVec quadrant = sub.integer;
  reduced.high = sub.value;
  reduced = normalize(reduced);
  reduced = df_add_exact(reduced, df_multiply(value, splat(kInvTwoPi1)));
  sub = rempi_sub(reduced.high);
  quadrant = vaddq_s32(quadrant, sub.integer);
  reduced.high = sub.value;
  reduced = normalize(reduced);
  Float2 tail{splat(kInvTwoPi2), splat(kInvTwoPi3)};
  reduced = df_add_exact(reduced, df_multiply(tail, value));
  reduced = normalize(reduced);
  reduced =
      df_multiply(reduced, Float2{splat(3.1415927410125732422F * 2.0F),
                                  splat(-8.7422776573475857731e-08F * 2.0F)});
  const Mask tiny = vcltq_f32(vabsq_f32(value), splat(0.7F));
  reduced = select(tiny, Float2{value, splat(0.0F)}, reduced);
  return {reduced, quadrant};
}

Vec sleef_sin_u10(const Vec input) noexcept {
  IntVec quadrant;
  Float2 reduced;
  if (all_true(vcltq_f32(vabsq_f32(input), splat(125.0F)))) {
    const Vec rounded =
        vrndnq_f32(vmulq_f32(input, splat(0.31830988618379067154F)));
    quadrant = vcvtq_s32_f32(vrndnq_f32(rounded));
    const Vec first = fmadd(rounded, splat(-3.1414794921875F), input);
    reduced = df_add_exact(
        first, vmulq_f32(rounded, splat(-0.00011315941810607910156F)));
    reduced =
        df_add(reduced, vmulq_f32(rounded, splat(-1.9841872589410058936e-09F)));
  } else {
    Float2Int rempi = rempi_bounded(input);
    quadrant = vandq_s32(rempi.integer, splat_int(3));
    const Mask positive = vcgtq_f32(rempi.value.high, splat(0.0F));
    quadrant =
        vshrq_n_s32(vaddq_s32(vaddq_s32(quadrant, quadrant),
                              select(positive, splat_int(2), splat_int(1))),
                    2);
    const Mask odd =
        vceqq_s32(vandq_s32(rempi.integer, splat_int(1)), splat_int(1));
    const Float2 half_pi{
        multiply_sign(splat(3.1415927410125732422F * -0.5F), rempi.value.high),
        multiply_sign(splat(-8.7422776573475857731e-08F * -0.5F),
                      rempi.value.high)};
    const Float2 shifted = df_add_exact(rempi.value, half_pi);
    reduced = normalize(select(odd, shifted, rempi.value));
  }

  const Float2 original = reduced;
  const Float2 squared = df_square(reduced);
  Vec polynomial = splat(2.6083159809786593541503e-06F);
  polynomial =
      fmadd(polynomial, squared.high, splat(-0.0001981069071916863322258F));
  polynomial =
      fmadd(polynomial, squared.high, splat(0.00833307858556509017944336F));
  const Float2 correction =
      df_multiply(df_add(splat(-0.166666597127914428710938F),
                         vmulq_f32(polynomial, squared.high)),
                  squared);
  const Float2 factor = df_add(splat(1.0F), correction);
  Vec result = df_multiply_result(original, factor);
  const Mask negate =
      vceqq_s32(vandq_s32(quadrant, splat_int(1)), splat_int(1));
  result = vreinterpretq_f32_u32(
      veorq_u32(vandq_u32(negate, vreinterpretq_u32_f32(splat(-0.0F))),
                vreinterpretq_u32_f32(result)));

  float lanes[4];
  vst1q_f32(lanes, input);
  Mask negative_zero = vdupq_n_u32(0U);
  negative_zero = vsetq_lane_u32(is_negative_zero_lane(lanes[0]) ? ~0U : 0U,
                                 negative_zero, 0);
  negative_zero = vsetq_lane_u32(is_negative_zero_lane(lanes[1]) ? ~0U : 0U,
                                 negative_zero, 1);
  negative_zero = vsetq_lane_u32(is_negative_zero_lane(lanes[2]) ? ~0U : 0U,
                                 negative_zero, 2);
  negative_zero = vsetq_lane_u32(is_negative_zero_lane(lanes[3]) ? ~0U : 0U,
                                 negative_zero, 3);
  return select(negative_zero, input, result);
}

Vec sleef_cos_u10(const Vec input) noexcept {
  IntVec quadrant;
  Float2 reduced;
  if (all_true(vcltq_f32(vabsq_f32(input), splat(125.0F)))) {
    const Vec rounded = fmadd(
        vrndnq_f32(fmadd(input, splat(0.31830988618379067154F), splat(-0.5F))),
        splat(2.0F), splat(1.0F));
    quadrant = vcvtq_s32_f32(vrndnq_f32(rounded));
    reduced = df_add_exact(input,
                           vmulq_f32(rounded, splat(-3.1414794921875F * 0.5F)));
    reduced = df_add_exact(
        reduced, vmulq_f32(rounded, splat(-0.00011315941810607910156F * 0.5F)));
    reduced = df_add_exact(
        reduced, vmulq_f32(rounded, splat(-1.9841872589410058936e-09F * 0.5F)));
  } else {
    Float2Int rempi = rempi_bounded(input);
    quadrant = vandq_s32(rempi.integer, splat_int(3));
    const Mask positive = vcgtq_f32(rempi.value.high, splat(0.0F));
    quadrant =
        vshrq_n_s32(vaddq_s32(vaddq_s32(quadrant, quadrant),
                              select(positive, splat_int(8), splat_int(7))),
                    1);
    const Mask even =
        vceqq_s32(vandq_s32(rempi.integer, splat_int(1)), splat_int(0));
    const Vec sign_source = select(positive, splat(0.0F), splat(-1.0F));
    const Float2 half_pi{
        multiply_sign(splat(3.1415927410125732422F * -0.5F), sign_source),
        multiply_sign(splat(-8.7422776573475857731e-08F * -0.5F), sign_source)};
    const Float2 shifted = df_add_exact(rempi.value, half_pi);
    reduced = normalize(select(even, shifted, rempi.value));
  }

  const Float2 original = reduced;
  const Float2 squared = df_square(reduced);
  Vec polynomial = splat(2.6083159809786593541503e-06F);
  polynomial =
      fmadd(polynomial, squared.high, splat(-0.0001981069071916863322258F));
  polynomial =
      fmadd(polynomial, squared.high, splat(0.00833307858556509017944336F));
  const Float2 correction =
      df_multiply(df_add(splat(-0.166666597127914428710938F),
                         vmulq_f32(polynomial, squared.high)),
                  squared);
  const Float2 factor = df_add(splat(1.0F), correction);
  Vec result = df_multiply_result(original, factor);
  const Mask preserve_sign =
      vceqq_s32(vandq_s32(quadrant, splat_int(2)), splat_int(0));
  return vreinterpretq_f32_u32(
      veorq_u32(vandq_u32(preserve_sign, vreinterpretq_u32_f32(splat(-0.0F))),
                vreinterpretq_u32_f32(result)));
}

inline Vec power_of_two(const IntVec exponent) noexcept {
  return vreinterpretq_f32_s32(
      vshlq_n_s32(vaddq_s32(exponent, splat_int(0x7f)), 23));
}

Vec sleef_exp_u10(const Vec input) noexcept {
  const IntVec exponent = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(
      input, splat(1.442695040888963407359924681001892137426645954153F))));
  const Vec exponent_float = vcvtq_f32_s32(exponent);
  Vec reduced = fmadd(exponent_float, splat(-0.693145751953125F), input);
  reduced = fmadd(exponent_float, splat(-1.428606765330187045e-06F), reduced);
  Vec polynomial = splat(0.000198527617612853646278381F);
  polynomial = fmadd(polynomial, reduced, splat(0.00139304355252534151077271F));
  polynomial = fmadd(polynomial, reduced, splat(0.00833336077630519866943359F));
  polynomial = fmadd(polynomial, reduced, splat(0.0416664853692054748535156F));
  polynomial = fmadd(polynomial, reduced, splat(0.166666671633720397949219F));
  polynomial = fmadd(polynomial, reduced, splat(0.5F));
  Vec result = vaddq_f32(
      splat(1.0F), fmadd(vmulq_f32(reduced, reduced), polynomial, reduced));
  const IntVec half = vshrq_n_s32(exponent, 1);
  result = vmulq_f32(vmulq_f32(result, power_of_two(half)),
                     power_of_two(vsubq_s32(exponent, half)));
  result = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(result),
                                           vcltq_f32(input, splat(-104.0F))));
  return select(vcgtq_f32(input, splat(100.0F)),
                splat(std::numeric_limits<float>::infinity()), result);
}

float neon_reduce_sum(Vec value) noexcept {
  value = vaddq_f32(value, vextq_f32(value, value, 2));
  value = vaddq_f32(value, vrev64q_f32(value));
  return vgetq_lane_f32(value, 0);
}

float neon_reduce_max(Vec value) noexcept {
  value = vmaxq_f32(value, vextq_f32(value, value, 2));
  value = vmaxq_f32(value, vrev64q_f32(value));
  return vgetq_lane_f32(value, 0);
}

bool finite_values(const std::vector<float> &values) noexcept {
  for (const float value : values) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

float torch_cascade_sum_squares_1024(const float *const values) noexcept {
  constexpr std::size_t kWidth = 1024;
  constexpr std::size_t kVectorWidth = 4;
  constexpr std::size_t kIlpFactor = 4;
  constexpr std::size_t kLevels = 4;
  constexpr std::size_t kLevelStep = 16;
  constexpr std::size_t kLevelMask = kLevelStep - 1U;
  float accumulators[kLevels][kIlpFactor][kVectorWidth] = {};
  constexpr std::size_t kVectorItems = kWidth / kVectorWidth;
  constexpr std::size_t kSizeIlp = kVectorItems / kIlpFactor;
  std::size_t item = 0;
  for (; item + kLevelStep <= kSizeIlp;) {
    for (std::size_t chunk = 0; chunk < kLevelStep; ++chunk, ++item) {
      for (std::size_t group = 0; group < kIlpFactor; ++group) {
        for (std::size_t lane = 0; lane < kVectorWidth; ++lane) {
          const std::size_t index =
              (item * kIlpFactor + group) * kVectorWidth + lane;
          const float squared = values[index] * values[index];
          accumulators[0][group][lane] += squared;
        }
      }
    }
    for (std::size_t level = 1; level < kLevels; ++level) {
      for (std::size_t group = 0; group < kIlpFactor; ++group) {
        for (std::size_t lane = 0; lane < kVectorWidth; ++lane) {
          accumulators[level][group][lane] +=
              accumulators[level - 1U][group][lane];
          accumulators[level - 1U][group][lane] = 0.0F;
        }
      }
      const std::size_t mask = kLevelMask << (level * 4U);
      if ((item & mask) != 0U)
        break;
    }
  }
  for (; item < kSizeIlp; ++item) {
    for (std::size_t group = 0; group < kIlpFactor; ++group) {
      for (std::size_t lane = 0; lane < kVectorWidth; ++lane) {
        const std::size_t index =
            (item * kIlpFactor + group) * kVectorWidth + lane;
        const float squared = values[index] * values[index];
        accumulators[0][group][lane] += squared;
      }
    }
  }
  for (std::size_t level = 1; level < kLevels; ++level) {
    for (std::size_t group = 0; group < kIlpFactor; ++group) {
      for (std::size_t lane = 0; lane < kVectorWidth; ++lane)
        accumulators[0][group][lane] += accumulators[level][group][lane];
    }
  }
  float vector_sum[kVectorWidth];
  for (std::size_t lane = 0; lane < kVectorWidth; ++lane) {
    vector_sum[lane] = accumulators[0][0][lane];
    for (std::size_t group = 1; group < kIlpFactor; ++group)
      vector_sum[lane] += accumulators[0][group][lane];
  }
  float sum = 0.0F;
  for (const float value : vector_sum)
    sum += value;
  return sum;
}

void apply_rope_tensor(std::vector<float> *const tensor,
                       const std::vector<float> &cosine,
                       const std::vector<float> &sine,
                       const std::size_t rows) noexcept {
  constexpr std::size_t kHeads = 16;
  constexpr std::size_t kHeadDimension = 64;
  constexpr std::size_t kPairs = 32;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t head = 0; head < kHeads; ++head) {
      const std::size_t base = (row * kHeads + head) * kHeadDimension;
      for (std::size_t pair = 0; pair < kPairs; ++pair) {
        const float first = (*tensor)[base + pair];
        const float second = (*tensor)[base + pair + kPairs];
        const float cosine_value = cosine[row * kPairs + pair];
        const float sine_value = sine[row * kPairs + pair];
        const float first_cosine = first * cosine_value;
        const float second_sine = second * sine_value;
        const float second_cosine = second * cosine_value;
        const float first_sine = first * sine_value;
        (*tensor)[base + pair] = first_cosine - second_sine;
        (*tensor)[base + pair + kPairs] = second_cosine + first_sine;
      }
    }
  }
}

#endif

} // namespace

bool omnina_apple_f32_kernel_supported() noexcept {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  return true;
#else
  return false;
#endif
}

Status omnina_apple_f32_linear(const std::vector<float> &input,
                               const std::size_t rows,
                               const std::size_t input_width,
                               const GenebDecoderTensorView &weight,
                               const std::size_t output_width,
                               const GenebDecoderTensorView *const bias,
                               std::vector<float> *const output) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  if (output == nullptr || rows == 0 || input_width == 0 || output_width == 0 ||
      bias != nullptr || weight.data == nullptr ||
      weight.dtype != TensorDType::kF32 ||
      weight.shape != std::vector<std::size_t>{output_width, input_width} ||
      weight.bytes != output_width * input_width * sizeof(float) ||
      input.size() != rows * input_width ||
      rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      input_width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      output_width >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return {ErrorCode::kInvalidArgument,
            "OmniNA Apple F32 linear arguments are invalid"};
  }
  output->assign(rows * output_width, 0.0F);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(rows),
              static_cast<int>(output_width), static_cast<int>(input_width),
              1.0F, input.data(), static_cast<int>(input_width),
              reinterpret_cast<const float *>(weight.data),
              static_cast<int>(input_width), 0.0F, output->data(),
              static_cast<int>(output_width));
  return finite_values(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "OmniNA Apple F32 linear produced non-finite output"};
#else
  (void)input;
  (void)rows;
  (void)input_width;
  (void)weight;
  (void)output_width;
  (void)bias;
  (void)output;
  return unsupported_host();
#endif
}

Status omnina_apple_f32_rms_norm(const std::vector<float> &input,
                                 const std::size_t rows,
                                 const GenebDecoderTensorView &scale,
                                 const float epsilon,
                                 std::vector<float> *const output) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  constexpr std::size_t kWidth = 1024;
  if (output == nullptr || rows == 0 || input.size() != rows * kWidth ||
      scale.data == nullptr || scale.dtype != TensorDType::kF32 ||
      scale.shape != std::vector<std::size_t>{kWidth} ||
      scale.bytes != kWidth * sizeof(float) || !std::isfinite(epsilon) ||
      epsilon <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "OmniNA Apple F32 RMSNorm arguments are invalid"};
  }
  const auto *const scale_values = reinterpret_cast<const float *>(scale.data);
  output->resize(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    const float sum_squares =
        torch_cascade_sum_squares_1024(input.data() + row * kWidth);
    const float inverse =
        1.0F / std::sqrt(sum_squares / static_cast<float>(kWidth) + epsilon);
    if (!std::isfinite(inverse) || inverse <= 0.0F) {
      return {ErrorCode::kInvalidArgument,
              "OmniNA Apple F32 RMSNorm inverse is invalid"};
    }
    for (std::size_t column = 0; column < kWidth; ++column) {
      const std::size_t index = row * kWidth + column;
      (*output)[index] = input[index] * inverse * scale_values[column];
    }
  }
  return finite_values(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "OmniNA Apple F32 RMSNorm produced non-finite output"};
#else
  (void)input;
  (void)rows;
  (void)scale;
  (void)epsilon;
  (void)output;
  return unsupported_host();
#endif
}

Status omnina_apple_f32_apply_rope(std::vector<float> *const query,
                                   std::vector<float> *const key,
                                   const std::size_t rows,
                                   const std::size_t position_offset) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  constexpr std::size_t kRowWidth = 16 * 64;
  constexpr std::size_t kPairs = 32;
  constexpr std::size_t kMaximumSequenceLength = 2048;
  if (query == nullptr || key == nullptr || rows == 0 ||
      query->size() != rows * kRowWidth || key->size() != rows * kRowWidth ||
      position_offset > kMaximumSequenceLength ||
      rows > kMaximumSequenceLength - position_offset) {
    return {ErrorCode::kInvalidArgument,
            "OmniNA Apple F32 RoPE arguments are invalid"};
  }
  std::vector<float> cosine(rows * kPairs);
  std::vector<float> sine(rows * kPairs);
  for (std::size_t row = 0; row < rows; ++row) {
    const float position = static_cast<float>(position_offset + row);
    for (std::size_t pair = 0; pair < kPairs; pair += 4U) {
      float angles[4];
      for (std::size_t lane = 0; lane < 4U; ++lane) {
        const float exponent = 2.0F * static_cast<float>(pair + lane) / 64.0F;
        const float inverse_frequency = 1.0F / std::pow(10000.0F, exponent);
        angles[lane] = position * inverse_frequency;
      }
      const Vec angle = vld1q_f32(angles);
      vst1q_f32(cosine.data() + row * kPairs + pair, sleef_cos_u10(angle));
      vst1q_f32(sine.data() + row * kPairs + pair, sleef_sin_u10(angle));
    }
  }
  apply_rope_tensor(query, cosine, sine, rows);
  apply_rope_tensor(key, cosine, sine, rows);
  return finite_values(*query) && finite_values(*key)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "OmniNA Apple F32 RoPE produced non-finite output"};
#else
  (void)query;
  (void)key;
  (void)rows;
  (void)position_offset;
  return unsupported_host();
#endif
}

Status omnina_apple_f32_causal_attention(const std::vector<float> &query,
                                         const std::vector<float> &key,
                                         const std::vector<float> &value,
                                         const std::size_t rows,
                                         std::vector<float> *const output) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  constexpr std::size_t kHeads = 16;
  constexpr std::size_t kHeadDimension = 64;
  constexpr std::size_t kRowWidth = kHeads * kHeadDimension;
  constexpr std::size_t kKvSplit = 512;
  if (output == nullptr || rows == 0 || rows > 2048 ||
      query.size() != rows * kRowWidth || key.size() != query.size() ||
      value.size() != query.size() ||
      rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return {ErrorCode::kInvalidArgument,
            "OmniNA Apple F32 attention arguments are invalid"};
  }
  const std::size_t q_split = rows >= 768 ? 256 : (rows >= 192 ? 64 : 32);
  const std::size_t kv_split = std::min(kKvSplit, rows);
  const int row_stride = static_cast<int>(kRowWidth);
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDimension));
  const Vec scale_vector = splat(scale);
  const Vec negative_infinity = splat(-std::numeric_limits<float>::infinity());
  output->assign(query.size(), 0.0F);
  std::vector<float> scores(q_split * kv_split);
  std::vector<float> maxima(q_split);
  std::vector<float> sums(q_split);
  std::vector<float> destination(q_split * kHeadDimension);

  for (std::size_t head = 0; head < kHeads; ++head) {
    const float *const query_head = query.data() + head * kHeadDimension;
    const float *const key_head = key.data() + head * kHeadDimension;
    const float *const value_head = value.data() + head * kHeadDimension;
    for (std::size_t query_begin = 0; query_begin < rows;
         query_begin += q_split) {
      const std::size_t query_count = std::min(q_split, rows - query_begin);
      std::fill_n(maxima.data(), query_count,
                  -std::numeric_limits<float>::infinity());
      std::fill_n(sums.data(), query_count, 0.0F);
      const std::size_t key_limit = std::min(query_begin + query_count, rows);
      for (std::size_t key_begin = 0; key_begin < key_limit;
           key_begin += kv_split) {
        const std::size_t key_count = std::min(kv_split, rows - key_begin);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(query_count), static_cast<int>(key_count),
                    static_cast<int>(kHeadDimension), 1.0F,
                    query_head + query_begin * kRowWidth, row_stride,
                    key_head + key_begin * kRowWidth, row_stride, 0.0F,
                    scores.data(), static_cast<int>(key_count));

        if (key_limit - key_begin <= kv_split) {
          for (std::size_t row = 0; row < query_count; ++row) {
            const std::size_t last = query_begin + row - key_begin;
            std::fill(scores.begin() + static_cast<std::ptrdiff_t>(
                                           row * key_count + last + 1U),
                      scores.begin() +
                          static_cast<std::ptrdiff_t>((row + 1U) * key_count),
                      -std::numeric_limits<float>::infinity());
          }
        }

        for (std::size_t row = 0; row < query_count; ++row) {
          float *const row_scores = scores.data() + row * key_count;
          Vec maximum_vector = negative_infinity;
          std::size_t index = 0;
          for (; index + 4U <= key_count; index += 4U) {
            const Vec scaled =
                vmulq_f32(vld1q_f32(row_scores + index), scale_vector);
            vst1q_f32(row_scores + index, scaled);
            maximum_vector = vmaxq_f32(maximum_vector, scaled);
          }
          float scalar_maximum = -std::numeric_limits<float>::infinity();
          for (; index < key_count; ++index) {
            row_scores[index] *= scale;
            scalar_maximum = std::max(scalar_maximum, row_scores[index]);
          }
          const float block_maximum =
              std::max(scalar_maximum, neon_reduce_max(maximum_vector));
          const float next_maximum =
              maxima[row] > block_maximum ? maxima[row] : block_maximum;
          Vec sum_vector = splat(0.0F);
          const Vec maximum_broadcast = splat(next_maximum);
          index = 0;
          for (; index + 4U <= key_count; index += 4U) {
            const Vec exponential = sleef_exp_u10(
                vsubq_f32(vld1q_f32(row_scores + index), maximum_broadcast));
            vst1q_f32(row_scores + index, exponential);
            sum_vector = vaddq_f32(sum_vector, exponential);
          }
          float block_sum = neon_reduce_sum(sum_vector);
          for (; index < key_count; ++index) {
            row_scores[index] = std::exp(row_scores[index] - next_maximum);
            block_sum += row_scores[index];
          }
          const float previous_scale = std::exp(maxima[row] - next_maximum);
          sums[row] = block_sum + previous_scale * sums[row];
          maxima[row] = next_maximum;
          if (key_begin > 0) {
            const Vec previous_scale_vector = splat(previous_scale);
            for (std::size_t dimension = 0; dimension < kHeadDimension;
                 dimension += 4U) {
              float *const target =
                  destination.data() + row * kHeadDimension + dimension;
              vst1q_f32(target,
                        vmulq_f32(vld1q_f32(target), previous_scale_vector));
            }
          }
        }

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(query_count), static_cast<int>(kHeadDimension),
            static_cast<int>(key_count), 1.0F, scores.data(),
            static_cast<int>(key_count), value_head + key_begin * kRowWidth,
            row_stride, key_begin == 0 ? 0.0F : 1.0F, destination.data(),
            static_cast<int>(kHeadDimension));
      }

      for (std::size_t row = 0; row < query_count; ++row) {
        if (!std::isfinite(sums[row]) || sums[row] <= 0.0F) {
          return {ErrorCode::kInvalidArgument,
                  "OmniNA Apple F32 attention softmax is invalid"};
        }
        const Vec reciprocal = splat(1.0F / sums[row]);
        for (std::size_t dimension = 0; dimension < kHeadDimension;
             dimension += 4U) {
          const float *const source =
              destination.data() + row * kHeadDimension + dimension;
          float *const target =
              output->data() +
              ((query_begin + row) * kHeads + head) * kHeadDimension +
              dimension;
          vst1q_f32(target, vmulq_f32(vld1q_f32(source), reciprocal));
        }
      }
    }
  }
  return finite_values(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "OmniNA Apple F32 attention produced non-finite output"};
#else
  (void)query;
  (void)key;
  (void)value;
  (void)rows;
  (void)output;
  return unsupported_host();
#endif
}

Status omnina_apple_f32_swiglu(const std::vector<float> &gate,
                               std::vector<float> *const up) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  if (up == nullptr || gate.empty() || gate.size() != up->size() ||
      gate.size() % 4U != 0U) {
    return {ErrorCode::kInvalidArgument,
            "OmniNA Apple F32 SwiGLU arguments are invalid"};
  }
  const Vec one = splat(1.0F);
  for (std::size_t index = 0; index < gate.size(); index += 4U) {
    const Vec gate_values = vld1q_f32(gate.data() + index);
    const Vec exponentials = sleef_exp_u10(vnegq_f32(gate_values));
    const Vec activated = vdivq_f32(gate_values, vaddq_f32(one, exponentials));
    vst1q_f32(up->data() + index,
              vmulq_f32(vld1q_f32(up->data() + index), activated));
  }
  return finite_values(*up)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "OmniNA Apple F32 SwiGLU produced non-finite output"};
#else
  (void)gate;
  (void)up;
  return unsupported_host();
#endif
}

} // namespace evo::cpu::detail
