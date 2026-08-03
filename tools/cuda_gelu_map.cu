// SPDX-License-Identifier: Apache-2.0
// Diagnostic: emit exact-GELU BF16 output bits for every BF16 input bit
// pattern.
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace {

__global__ void gelu_map_kernel(const __nv_bfloat16 *const input,
                                __nv_bfloat16 *const output) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     static_cast<std::size_t>(threadIdx.x);
  if (index >= 65536)
    return;
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  const float value = __bfloat162float(input[index]);
  output[index] = __float2bfloat16_rn(value * 0.5F *
                                      (1.0F + ::erf(value * kInverseSqrtTwo)));
}

bool checked(const cudaError_t error, const char *const operation) {
  if (error == cudaSuccess)
    return true;
  std::cerr << operation << ": " << cudaGetErrorString(error) << '\n';
  return false;
}

} // namespace

int main(const int argc, char **const argv) {
  if (argc != 2) {
    std::cerr << "usage: cuda_gelu_map OUTPUT.bin\n";
    return 2;
  }
  std::vector<std::uint16_t> input(65536);
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<std::uint16_t>(index);
  std::vector<std::uint16_t> output(input.size());
  void *device_input = nullptr;
  void *device_output = nullptr;
  const std::size_t bytes = input.size() * sizeof(input[0]);
  if (!checked(cudaMalloc(&device_input, bytes), "cudaMalloc input") ||
      !checked(cudaMalloc(&device_output, bytes), "cudaMalloc output") ||
      !checked(
          cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice),
          "cudaMemcpy input")) {
    return 1;
  }
  gelu_map_kernel<<<256, 256>>>(
      static_cast<const __nv_bfloat16 *>(device_input),
      static_cast<__nv_bfloat16 *>(device_output));
  if (!checked(cudaGetLastError(), "gelu_map_kernel") ||
      !checked(cudaMemcpy(output.data(), device_output, bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy output")) {
    return 1;
  }
  std::ofstream stream(argv[1], std::ios::binary);
  stream.write(reinterpret_cast<const char *>(output.data()),
               static_cast<std::streamsize>(bytes));
  if (!stream) {
    std::cerr << "failed to write GELU map\n";
    return 1;
  }
  static_cast<void>(cudaFree(device_output));
  static_cast<void>(cudaFree(device_input));
  return 0;
}
