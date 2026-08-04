// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "evo/status.hpp"

namespace evo::cuda {

[[nodiscard]] Status cuda_status(cudaError_t error, const char *operation);
[[nodiscard]] Status cublas_status(cublasStatus_t error, const char *operation);
[[nodiscard]] Status select_device(int device);
[[nodiscard]] Status synchronize_device();
[[nodiscard]] Status enable_peer_access(int device, int peer_device);

class Stream final {
public:
  Stream() = default;
  ~Stream();

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  Stream(Stream &&other) noexcept;
  Stream &operator=(Stream &&other) noexcept;

  [[nodiscard]] Status create(unsigned int flags = cudaStreamNonBlocking);
  void reset() noexcept;
  [[nodiscard]] Status synchronize() const;
  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
  [[nodiscard]] bool valid() const noexcept { return stream_ != nullptr; }
  [[nodiscard]] int device() const noexcept { return device_; }

private:
  cudaStream_t stream_{nullptr};
  int device_{-1};
};

class DeviceBuffer final {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;

  [[nodiscard]] Status allocate(int device, std::size_t bytes);
  void reset() noexcept;
  [[nodiscard]] Status copy_from_host(const void *source, std::size_t bytes,
                                      const Stream &stream);
  [[nodiscard]] Status copy_to_host(void *destination, std::size_t bytes,
                                    const Stream &stream) const;
  [[nodiscard]] Status copy_from_peer(const DeviceBuffer &source,
                                      std::size_t bytes, const Stream &stream);
  [[nodiscard]] Status zero(const Stream &stream);

  [[nodiscard]] void *data() noexcept { return data_; }
  [[nodiscard]] const void *data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] int device() const noexcept { return device_; }
  [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }

private:
  void *data_{nullptr};
  std::size_t bytes_{0};
  int device_{-1};
};

class Blas final {
public:
  Blas() = default;
  ~Blas();

  Blas(const Blas &) = delete;
  Blas &operator=(const Blas &) = delete;
  Blas(Blas &&other) noexcept;
  Blas &operator=(Blas &&other) noexcept;

  [[nodiscard]] Status create();
  void reset() noexcept;
  [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }
  [[nodiscard]] bool valid() const noexcept { return handle_ != nullptr; }
  [[nodiscard]] int device() const noexcept { return device_; }

private:
  cublasHandle_t handle_{nullptr};
  int device_{-1};
};

class BlasLt final {
public:
  BlasLt() = default;
  ~BlasLt();

  BlasLt(const BlasLt &) = delete;
  BlasLt &operator=(const BlasLt &) = delete;
  BlasLt(BlasLt &&other) noexcept;
  BlasLt &operator=(BlasLt &&other) noexcept;

  [[nodiscard]] Status create();
  void reset() noexcept;
  [[nodiscard]] cublasLtHandle_t get() const noexcept { return handle_; }
  [[nodiscard]] bool valid() const noexcept { return handle_ != nullptr; }

private:
  cublasLtHandle_t handle_{nullptr};
};

} // namespace evo::cuda
