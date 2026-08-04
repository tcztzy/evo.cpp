// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "evo/status.hpp"

namespace evo {

class MappedFile final {
 public:
  MappedFile() = default;
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  [[nodiscard]] Status open(const std::string& path);
  void reset() noexcept;

  [[nodiscard]] const std::uint8_t* data() const noexcept {
    return static_cast<const std::uint8_t*>(mapping_);
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool is_open() const noexcept { return mapping_ != nullptr; }

 private:
  int descriptor_{-1};
  void* mapping_{nullptr};
  std::size_t size_{0};
};

}  // namespace evo

