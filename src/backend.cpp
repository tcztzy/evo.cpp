// SPDX-License-Identifier: Apache-2.0
#include "evo/backend.hpp"

namespace evo {

const char *execution_backend_name(const ExecutionBackend backend) noexcept {
  switch (backend) {
  case ExecutionBackend::kAuto:
    return "auto";
  case ExecutionBackend::kCpu:
    return "cpu";
  case ExecutionBackend::kCuda:
    return "cuda";
  case ExecutionBackend::kMps:
    return "mps";
  }
  return "unknown";
}

Status parse_execution_backend(const std::string_view text,
                               ExecutionBackend *const backend) {
  if (backend == nullptr)
    return {ErrorCode::kInvalidArgument, "execution backend output is null"};
  if (text == "auto") {
    *backend = ExecutionBackend::kAuto;
    return Status::Ok();
  }
  if (text == "cpu") {
    *backend = ExecutionBackend::kCpu;
    return Status::Ok();
  }
  if (text == "cuda") {
    *backend = ExecutionBackend::kCuda;
    return Status::Ok();
  }
  if (text == "mps") {
    *backend = ExecutionBackend::kMps;
    return Status::Ok();
  }
  return {ErrorCode::kInvalidArgument,
          "backend must be one of auto, cpu, cuda, or mps"};
}

} // namespace evo
