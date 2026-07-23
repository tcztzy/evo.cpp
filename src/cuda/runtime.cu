// SPDX-License-Identifier: Apache-2.0
#include "evo2c/cuda/runtime.hpp"

#include <string>
#include <utility>

namespace evo2c::cuda {

Status cuda_status(const cudaError_t error, const char *const operation) {
  if (error == cudaSuccess) {
    return Status::Ok();
  }
  return {ErrorCode::kCuda, std::string{operation} +
                                " failed: " + cudaGetErrorName(error) + " (" +
                                cudaGetErrorString(error) + ")"};
}

Status cublas_status(const cublasStatus_t error, const char *const operation) {
  if (error == CUBLAS_STATUS_SUCCESS) {
    return Status::Ok();
  }
  const char *message = cublasGetStatusString(error);
  return {ErrorCode::kCuda,
          std::string{operation} + " failed: " +
              (message == nullptr ? std::to_string(static_cast<int>(error))
                                  : message)};
}

Status select_device(const int device) {
  int count = 0;
  auto status = cuda_status(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (!status.ok())
    return status;
  if (device < 0 || device >= count) {
    return {ErrorCode::kInvalidArgument,
            "CUDA device " + std::to_string(device) + " is outside [0, " +
                std::to_string(count) + ")"};
  }
  return cuda_status(cudaSetDevice(device), "cudaSetDevice");
}

Status synchronize_device() {
  return cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

Stream::~Stream() { reset(); }

Stream::Stream(Stream &&other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)),
      device_(std::exchange(other.device_, -1)) {}

Stream &Stream::operator=(Stream &&other) noexcept {
  if (this != &other) {
    reset();
    stream_ = std::exchange(other.stream_, nullptr);
    device_ = std::exchange(other.device_, -1);
  }
  return *this;
}

Status Stream::create(const unsigned int flags) {
  if (stream_ != nullptr) {
    return {ErrorCode::kInvalidArgument, "CUDA stream is already initialized"};
  }
  auto status = cuda_status(cudaGetDevice(&device_), "cudaGetDevice");
  if (!status.ok())
    return status;
  status = cuda_status(cudaStreamCreateWithFlags(&stream_, flags),
                       "cudaStreamCreateWithFlags");
  if (!status.ok())
    device_ = -1;
  return status;
}

void Stream::reset() noexcept {
  if (stream_ != nullptr) {
    int previous = -1;
    static_cast<void>(cudaGetDevice(&previous));
    static_cast<void>(cudaSetDevice(device_));
    static_cast<void>(cudaStreamDestroy(stream_));
    if (previous >= 0 && previous != device_) {
      static_cast<void>(cudaSetDevice(previous));
    }
    stream_ = nullptr;
  }
  device_ = -1;
}

Status Stream::synchronize() const {
  if (stream_ == nullptr) {
    return {ErrorCode::kInvalidArgument, "CUDA stream is not initialized"};
  }
  auto status = select_device(device_);
  if (!status.ok())
    return status;
  return cuda_status(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

DeviceBuffer::~DeviceBuffer() { reset(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)),
      device_(std::exchange(other.device_, -1)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
  if (this != &other) {
    reset();
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    device_ = std::exchange(other.device_, -1);
  }
  return *this;
}

Status DeviceBuffer::allocate(const int device, const std::size_t bytes) {
  if (data_ != nullptr) {
    return {ErrorCode::kInvalidArgument, "device buffer is already allocated"};
  }
  if (bytes == 0) {
    return {ErrorCode::kInvalidArgument,
            "device buffer allocation size must be positive"};
  }
  auto status = select_device(device);
  if (!status.ok())
    return status;
  status = cuda_status(cudaMalloc(&data_, bytes), "cudaMalloc");
  if (!status.ok()) {
    data_ = nullptr;
    return status;
  }
  bytes_ = bytes;
  device_ = device;
  return Status::Ok();
}

void DeviceBuffer::reset() noexcept {
  if (data_ == nullptr)
    return;
  int previous = -1;
  static_cast<void>(cudaGetDevice(&previous));
  static_cast<void>(cudaSetDevice(device_));
  static_cast<void>(cudaFree(data_));
  if (previous >= 0 && previous != device_) {
    static_cast<void>(cudaSetDevice(previous));
  }
  data_ = nullptr;
  bytes_ = 0;
  device_ = -1;
}

Status DeviceBuffer::copy_from_host(const void *const source,
                                    const std::size_t bytes,
                                    const Stream &stream) {
  if (data_ == nullptr || source == nullptr || !stream.valid() ||
      stream.device() != device_) {
    return {ErrorCode::kInvalidArgument,
            "copy_from_host requires an allocated buffer, source, and "
            "same-device stream"};
  }
  if (bytes > bytes_) {
    return {ErrorCode::kInvalidArgument,
            "copy_from_host exceeds device buffer capacity"};
  }
  auto status = select_device(device_);
  if (!status.ok())
    return status;
  return cuda_status(cudaMemcpyAsync(data_, source, bytes,
                                     cudaMemcpyHostToDevice, stream.get()),
                     "cudaMemcpyAsync host-to-device");
}

Status DeviceBuffer::copy_to_host(void *const destination,
                                  const std::size_t bytes,
                                  const Stream &stream) const {
  if (data_ == nullptr || destination == nullptr || !stream.valid() ||
      stream.device() != device_) {
    return {ErrorCode::kInvalidArgument,
            "copy_to_host requires an allocated buffer, destination, and "
            "same-device stream"};
  }
  if (bytes > bytes_) {
    return {ErrorCode::kInvalidArgument,
            "copy_to_host exceeds device buffer capacity"};
  }
  auto status = select_device(device_);
  if (!status.ok())
    return status;
  return cuda_status(cudaMemcpyAsync(destination, data_, bytes,
                                     cudaMemcpyDeviceToHost, stream.get()),
                     "cudaMemcpyAsync device-to-host");
}

Status DeviceBuffer::zero(const Stream &stream) {
  if (data_ == nullptr || !stream.valid() || stream.device() != device_) {
    return {ErrorCode::kInvalidArgument,
            "zero requires an allocated buffer and same-device stream"};
  }
  auto status = select_device(device_);
  if (!status.ok())
    return status;
  return cuda_status(cudaMemsetAsync(data_, 0, bytes_, stream.get()),
                     "cudaMemsetAsync");
}

BlasLt::~BlasLt() { reset(); }

BlasLt::BlasLt(BlasLt &&other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {}

BlasLt &BlasLt::operator=(BlasLt &&other) noexcept {
  if (this != &other) {
    reset();
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

Status BlasLt::create() {
  if (handle_ != nullptr) {
    return {ErrorCode::kInvalidArgument,
            "cuBLASLt handle is already initialized"};
  }
  return cublas_status(cublasLtCreate(&handle_), "cublasLtCreate");
}

void BlasLt::reset() noexcept {
  if (handle_ != nullptr) {
    static_cast<void>(cublasLtDestroy(handle_));
    handle_ = nullptr;
  }
}

} // namespace evo2c::cuda
