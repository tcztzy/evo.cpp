// SPDX-License-Identifier: Apache-2.0
#include "mps/model.hpp"

#include "evo/fp8.hpp"
#include "linear_executor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(const bool condition, const char *const message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

bool close(const std::vector<float> &actual, const std::vector<float> &expected,
           const float tolerance) {
  if (actual.size() != expected.size())
    return false;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (std::fabs(actual[index] - expected[index]) > tolerance)
      return false;
  }
  return true;
}

std::vector<std::uint8_t> bf16(const std::vector<float> &values) {
  std::vector<std::uint8_t> output(values.size() * 2);
  for (std::size_t index = 0; index < values.size(); ++index) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &values[index], sizeof(bits));
    output[index * 2] = static_cast<std::uint8_t>(bits >> 16U);
    output[index * 2 + 1] = static_cast<std::uint8_t>(bits >> 24U);
  }
  return output;
}

} // namespace

int main() {
  std::shared_ptr<evo::detail::LinearExecutor> executor;
  const auto create = evo::mps::create_linear_executor(&executor);
  if (create.code() == evo::ErrorCode::kMps &&
      create.message() == "no Metal device is available") {
    std::cout << "SKIP: this host has no Metal device\n";
    return 77;
  }
  check(create.ok() && executor != nullptr, "create MPS linear executor");
  if (!create.ok() || !executor)
    return 1;
  check(std::string{executor->name()} == "mps-f32-gemm+host-ops",
        "report transparent MPS kernel name");

  const std::vector<float> input{1.0F, 2.0F, -1.0F, -2.0F, 0.5F, 3.0F};
  const std::vector<float> weight{1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 4.0F};
  const std::vector<float> bias{0.25F, -0.5F};
  const std::vector<float> expected{2.25F, -4.5F, 8.25F, 13.75F};
  const evo::detail::LinearTensorView weight_view{
      reinterpret_cast<const std::uint8_t *>(weight.data()),
      evo::TensorDType::kF32, weight.size()};
  const evo::detail::LinearTensorView bias_view{
      reinterpret_cast<const std::uint8_t *>(bias.data()),
      evo::TensorDType::kF32, bias.size()};
  std::vector<float> output;
  auto status =
      executor->linear(input.data(), 2, 3, weight_view, 2, &bias_view, &output);
  check(status.ok() && close(output, expected, 1.0e-5F),
        "F32 MPS matrix multiplication and bias");
  const std::vector<float> deterministic_output = output;

  auto invalid_weight = weight_view;
  --invalid_weight.elements;
  output = {42.0F};
  status = executor->linear(input.data(), 2, 3, invalid_weight, 2, &bias_view,
                            &output);
  check(status.code() == evo::ErrorCode::kInvalidArgument &&
            output == std::vector<float>{42.0F},
        "invalid MPS dimensions fail before mutating output");

  const auto bf16_weight = bf16(weight);
  const evo::detail::LinearTensorView bf16_view{
      bf16_weight.data(), evo::TensorDType::kBF16, weight.size()};
  status =
      executor->linear(input.data(), 2, 3, bf16_view, 2, &bias_view, &output);
  check(status.ok() && close(output, expected, 1.0e-5F),
        "BF16 Metal linear kernel");

  std::vector<std::uint8_t> e4m3_weight;
  e4m3_weight.reserve(weight.size());
  for (const float value : weight)
    e4m3_weight.push_back(evo::fp8::encode_e4m3fn(value));
  const evo::detail::LinearTensorView e4m3_view{
      e4m3_weight.data(), evo::TensorDType::kE4M3Software, weight.size()};
  status =
      executor->linear(input.data(), 2, 3, e4m3_view, 2, &bias_view, &output);
  check(status.ok() && close(output, expected, 1.0e-5F),
        "E4M3 Metal linear kernel");

  std::vector<std::thread> workers;
  std::vector<int> thread_failures(4, 0);
  for (std::size_t worker = 0; worker < thread_failures.size(); ++worker) {
    workers.emplace_back([&, worker] {
      for (int iteration = 0; iteration < 8; ++iteration) {
        std::vector<float> concurrent;
        const auto inner = executor->linear(input.data(), 2, 3, weight_view, 2,
                                            &bias_view, &concurrent);
        if (!inner.ok() || concurrent != deterministic_output)
          ++thread_failures[worker];
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  for (const int count : thread_failures)
    check(count == 0, "shared MPS executor is deterministic across contexts");

  return failures == 0 ? 0 : 1;
}
