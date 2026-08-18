// SPDX-License-Identifier: Apache-2.0
#include "fft_convolution.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace evo::cpu::detail {
namespace {

using Complex = std::complex<float>;

constexpr float kPi = 3.14159265358979323846F;

Status invalid(const std::string &message) {
  return {ErrorCode::kInvalidArgument, "FFT convolution: " + message};
}

bool is_power_of_two(const std::size_t value) noexcept {
  return value != 0U && (value & (value - 1U)) == 0U;
}

bool checked_add(const std::size_t left, const std::size_t right,
                 std::size_t *const output) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    return false;
  *output = left + right;
  return true;
}

bool checked_mul(const std::size_t left, const std::size_t right,
                 std::size_t *const output) noexcept {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *output = left * right;
  return true;
}

Status next_power_of_two(const std::size_t minimum, std::size_t *const output) {
  if (minimum == 0U || output == nullptr)
    return invalid("radix-2 length is invalid");
  std::size_t value = 1U;
  while (value < minimum) {
    if (value > std::numeric_limits<std::size_t>::max() / 2U)
      return invalid("radix-2 length overflows size_t");
    value *= 2U;
  }
  *output = value;
  return Status::Ok();
}

void add_butterflies(const std::size_t length,
                     FftWorkStats *const stats) noexcept {
  if (stats == nullptr || length < 2U)
    return;
  std::size_t levels = 0U;
  for (std::size_t value = length; value > 1U; value /= 2U)
    ++levels;
  const auto count = static_cast<std::uint64_t>(length / 2U) *
                     static_cast<std::uint64_t>(levels);
  if (count >
      std::numeric_limits<std::uint64_t>::max() - stats->butterfly_count)
    stats->butterfly_count = std::numeric_limits<std::uint64_t>::max();
  else
    stats->butterfly_count += count;
}

void radix2_fft(std::vector<Complex> *const values, const bool inverse,
                FftWorkStats *const stats) {
  const std::size_t length = values->size();
  for (std::size_t index = 1U, reversed = 0U; index < length; ++index) {
    std::size_t bit = length >> 1U;
    while ((reversed & bit) != 0U) {
      reversed ^= bit;
      bit >>= 1U;
    }
    reversed ^= bit;
    if (index < reversed)
      std::swap((*values)[index], (*values)[reversed]);
  }

  for (std::size_t span = 2U; span <= length; span <<= 1U) {
    const float angle =
        (inverse ? 2.0F : -2.0F) * kPi / static_cast<float>(span);
    const Complex step{std::cos(angle), std::sin(angle)};
    for (std::size_t block = 0U; block < length; block += span) {
      Complex factor{1.0F, 0.0F};
      const std::size_t half = span / 2U;
      for (std::size_t offset = 0U; offset < half; ++offset) {
        const Complex even = (*values)[block + offset];
        const Complex odd = (*values)[block + offset + half] * factor;
        (*values)[block + offset] = even + odd;
        (*values)[block + offset + half] = even - odd;
        factor *= step;
      }
    }
    if (span == length)
      break;
  }
  add_butterflies(length, stats);
  if (inverse) {
    const float scale = 1.0F / static_cast<float>(length);
    for (auto &value : *values)
      value *= scale;
  }
}

Complex chirp(const std::size_t index, const std::size_t length,
              const bool inverse) {
  const std::uint64_t modulus = static_cast<std::uint64_t>(length) * 2U;
  const std::uint64_t raw =
      static_cast<std::uint64_t>(index) * static_cast<std::uint64_t>(index);
  const auto reduced = raw % modulus;
  const float angle = (inverse ? 1.0F : -1.0F) * kPi *
                      static_cast<float>(reduced) / static_cast<float>(length);
  return {std::cos(angle), std::sin(angle)};
}

Status transform(const std::vector<Complex> &input, const bool inverse,
                 std::vector<Complex> *const output,
                 FftWorkStats *const stats) {
  if (input.empty() || output == nullptr)
    return invalid("transform input is empty");
  const std::size_t length = input.size();
  if (is_power_of_two(length)) {
    *output = input;
    radix2_fft(output, inverse, stats);
    if (stats != nullptr) {
      stats->radix2_length = std::max(stats->radix2_length, length);
      stats->peak_complex_values =
          std::max(stats->peak_complex_values, output->size());
    }
    return Status::Ok();
  }

  std::size_t twice = 0U;
  if (!checked_mul(length, 2U, &twice) || twice == 0U)
    return invalid("Bluestein length overflows");
  const std::size_t convolution_minimum = twice - 1U;
  std::size_t radix_length = 0U;
  auto status = next_power_of_two(convolution_minimum, &radix_length);
  if (!status.ok())
    return status;

  std::vector<Complex> left(radix_length, Complex{0.0F, 0.0F});
  std::vector<Complex> right(radix_length, Complex{0.0F, 0.0F});
  for (std::size_t index = 0U; index < length; ++index) {
    const Complex phase = chirp(index, length, inverse);
    left[index] = input[index] * phase;
    const Complex inverse_phase = std::conj(phase);
    right[index] = inverse_phase;
    if (index != 0U)
      right[radix_length - index] = inverse_phase;
  }
  if (stats != nullptr) {
    stats->radix2_length = std::max(stats->radix2_length, radix_length);
    std::size_t peak = 0U;
    if (!checked_add(radix_length, radix_length, &peak) ||
        !checked_add(peak, length, &peak))
      peak = std::numeric_limits<std::size_t>::max();
    stats->peak_complex_values = std::max(stats->peak_complex_values, peak);
  }
  radix2_fft(&left, false, stats);
  radix2_fft(&right, false, stats);
  for (std::size_t index = 0U; index < radix_length; ++index)
    left[index] *= right[index];
  radix2_fft(&left, true, stats);

  output->resize(length);
  const float inverse_scale =
      inverse ? 1.0F / static_cast<float>(length) : 1.0F;
  for (std::size_t index = 0U; index < length; ++index)
    (*output)[index] =
        left[index] * chirp(index, length, inverse) * inverse_scale;
  return Status::Ok();
}

} // namespace

Status causal_fft_convolution(const std::vector<float> &input,
                              const std::vector<float> &kernel,
                              std::vector<float> *const output,
                              FftWorkStats *const stats) {
  if (output == nullptr)
    return invalid("output is null");
  if (input.empty() || kernel.size() != input.size())
    return invalid("input and kernel must have the same nonzero length");
  std::size_t transform_length = 0U;
  if (!checked_mul(input.size(), 2U, &transform_length))
    return invalid("2L transform length overflows size_t");
  if (stats != nullptr) {
    *stats = {};
    stats->signal_length = input.size();
    stats->transform_length = transform_length;
  }

  try {
    std::vector<Complex> input_time(transform_length, Complex{0.0F, 0.0F});
    std::vector<Complex> kernel_time(transform_length, Complex{0.0F, 0.0F});
    for (std::size_t index = 0U; index < input.size(); ++index) {
      input_time[index] = {input[index], 0.0F};
      kernel_time[index] = {kernel[index], 0.0F};
    }
    std::vector<Complex> input_frequency;
    std::vector<Complex> kernel_frequency;
    auto status = transform(input_time, false, &input_frequency, stats);
    if (!status.ok())
      return status;
    status = transform(kernel_time, false, &kernel_frequency, stats);
    if (!status.ok())
      return status;
    for (std::size_t index = 0U; index < transform_length; ++index)
      input_frequency[index] *= kernel_frequency[index];
    std::vector<Complex> convolved;
    status = transform(input_frequency, true, &convolved, stats);
    if (!status.ok())
      return status;
    output->resize(input.size());
    for (std::size_t index = 0U; index < input.size(); ++index) {
      const float value = convolved[index].real();
      if (!std::isfinite(value))
        return invalid("result is non-finite");
      (*output)[index] = value;
    }
    if (stats != nullptr) {
      std::size_t resident = 0U;
      std::size_t bluestein = 0U;
      if (!checked_mul(transform_length, 5U, &resident) ||
          (!is_power_of_two(transform_length) &&
           (!checked_mul(stats->radix2_length, 2U, &bluestein) ||
            !checked_add(resident, bluestein, &resident))))
        resident = std::numeric_limits<std::size_t>::max();
      stats->peak_complex_values =
          std::max(stats->peak_complex_values, resident);
    }
  } catch (const std::bad_alloc &) {
    return {ErrorCode::kInternal, "FFT convolution: scratch allocation failed"};
  }
  return Status::Ok();
}

} // namespace evo::cpu::detail
