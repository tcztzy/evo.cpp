// SPDX-License-Identifier: Apache-2.0
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <iostream>

namespace {

int cuda_failure(const char* operation, const cudaError_t error) {
  std::cerr << "evo-cuda-smoke: cuda: " << operation << " failed: "
            << cudaGetErrorString(error) << '\n';
  return 6;
}

int cublas_failure(const char* operation, const cublasStatus_t status) {
  std::cerr << "evo-cuda-smoke: cuda: " << operation
            << " failed with cuBLAS status " << static_cast<int>(status) << '\n';
  return 6;
}

int cufft_failure(const char* operation, const cufftResult status) {
  std::cerr << "evo-cuda-smoke: cuda: " << operation
            << " failed with cuFFT status " << static_cast<int>(status) << '\n';
  return 6;
}

__global__ void write_sentinel(int* output) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *output = 0x2c40;
  }
}

}  // namespace

int main() {
  int device_count = 0;
  if (const auto error = cudaGetDeviceCount(&device_count); error != cudaSuccess) {
    return cuda_failure("cudaGetDeviceCount", error);
  }
  if (device_count < 1) {
    std::cerr << "evo-cuda-smoke: cuda: no CUDA device available\n";
    return 6;
  }
  if (const auto error = cudaSetDevice(0); error != cudaSuccess) {
    return cuda_failure("cudaSetDevice", error);
  }

  cublasLtHandle_t cublas = nullptr;
  if (const auto status = cublasLtCreate(&cublas); status != CUBLAS_STATUS_SUCCESS) {
    return cublas_failure("cublasLtCreate", status);
  }
  if (const auto status = cublasLtDestroy(cublas); status != CUBLAS_STATUS_SUCCESS) {
    return cublas_failure("cublasLtDestroy", status);
  }

  cufftHandle fft = 0;
  if (const auto status = cufftPlan1d(&fft, 8, CUFFT_R2C, 1); status != CUFFT_SUCCESS) {
    return cufft_failure("cufftPlan1d", status);
  }
  if (const auto status = cufftDestroy(fft); status != CUFFT_SUCCESS) {
    return cufft_failure("cufftDestroy", status);
  }

  int* device_value = nullptr;
  if (const auto error = cudaMalloc(&device_value, sizeof(int)); error != cudaSuccess) {
    return cuda_failure("cudaMalloc", error);
  }
  write_sentinel<<<1, 1>>>(device_value);
  if (const auto error = cudaGetLastError(); error != cudaSuccess) {
    cudaFree(device_value);
    return cuda_failure("write_sentinel launch", error);
  }

  int host_value = 0;
  if (const auto error = cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost);
      error != cudaSuccess) {
    cudaFree(device_value);
    return cuda_failure("cudaMemcpy", error);
  }
  if (const auto error = cudaFree(device_value); error != cudaSuccess) {
    return cuda_failure("cudaFree", error);
  }
  if (host_value != 0x2c40) {
    std::cerr << "evo-cuda-smoke: cuda: kernel returned wrong sentinel\n";
    return 6;
  }

  cudaDeviceProp properties{};
  if (const auto error = cudaGetDeviceProperties(&properties, 0); error != cudaSuccess) {
    return cuda_failure("cudaGetDeviceProperties", error);
  }
  std::cout << "CUDA smoke passed: " << properties.name << " sm_" << properties.major
            << properties.minor << '\n';
  return 0;
}

