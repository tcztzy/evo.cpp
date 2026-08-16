// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <utility>

namespace evo {

enum class ErrorCode : int {
  kOk = 0,
  kInvalidArgument = 2,
  kIo = 3,
  kModelFormat = 4,
  kUnsupported = 5,
  kCuda = 6,
  kMps = 7,
  kInternal = 70,
};

class Status final {
 public:
  static Status Ok() { return Status(); }

  Status(ErrorCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  [[nodiscard]] bool ok() const noexcept { return code_ == ErrorCode::kOk; }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] const std::string& message() const noexcept { return message_; }

 private:
  Status() = default;

  ErrorCode code_{ErrorCode::kOk};
  std::string message_{};
};

[[nodiscard]] const char* error_code_name(ErrorCode code) noexcept;
[[nodiscard]] int exit_code(ErrorCode code) noexcept;

}  // namespace evo
