// SPDX-License-Identifier: Apache-2.0
#include "evo2c/status.hpp"

namespace evo2c {

const char* error_code_name(const ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kOk:
      return "ok";
    case ErrorCode::kInvalidArgument:
      return "invalid_argument";
    case ErrorCode::kIo:
      return "io";
    case ErrorCode::kModelFormat:
      return "model_format";
    case ErrorCode::kUnsupported:
      return "unsupported";
    case ErrorCode::kCuda:
      return "cuda";
    case ErrorCode::kInternal:
      return "internal";
  }
  return "unknown";
}

int exit_code(const ErrorCode code) noexcept {
  return static_cast<int>(code);
}

}  // namespace evo2c

