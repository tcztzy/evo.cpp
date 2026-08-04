// SPDX-License-Identifier: Apache-2.0
#include "evo/mapped_file.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace evo {
namespace {

Status io_error(const std::string& path, const char* operation, const int error) {
  std::ostringstream message;
  message << operation << " '" << path << "': " << std::strerror(error);
  return {ErrorCode::kIo, message.str()};
}

}  // namespace

MappedFile::~MappedFile() {
  reset();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : descriptor_(other.descriptor_), mapping_(other.mapping_), size_(other.size_) {
  other.descriptor_ = -1;
  other.mapping_ = nullptr;
  other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    reset();
    descriptor_ = other.descriptor_;
    mapping_ = other.mapping_;
    size_ = other.size_;
    other.descriptor_ = -1;
    other.mapping_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

Status MappedFile::open(const std::string& path) {
  MappedFile candidate;
  candidate.descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (candidate.descriptor_ < 0) {
    return io_error(path, "open", errno);
  }

  struct stat file_status {};
  if (::fstat(candidate.descriptor_, &file_status) != 0) {
    return io_error(path, "fstat", errno);
  }
  if (!S_ISREG(file_status.st_mode)) {
    return {ErrorCode::kIo, "model path is not a regular file: '" + path + "'"};
  }
  if (file_status.st_size <= 0) {
    return {ErrorCode::kIo, "model file is empty: '" + path + "'"};
  }

  const auto unsigned_size = static_cast<std::uintmax_t>(file_status.st_size);
  if (unsigned_size > std::numeric_limits<std::size_t>::max()) {
    return {ErrorCode::kIo, "model file is too large for this process: '" + path + "'"};
  }
  candidate.size_ = static_cast<std::size_t>(unsigned_size);
  candidate.mapping_ =
      ::mmap(nullptr, candidate.size_, PROT_READ, MAP_PRIVATE, candidate.descriptor_, 0);
  if (candidate.mapping_ == MAP_FAILED) {
    candidate.mapping_ = nullptr;
    return io_error(path, "mmap", errno);
  }

  *this = std::move(candidate);
  return Status::Ok();
}

void MappedFile::reset() noexcept {
  if (mapping_ != nullptr) {
    ::munmap(mapping_, size_);
    mapping_ = nullptr;
  }
  size_ = 0;
  if (descriptor_ >= 0) {
    ::close(descriptor_);
    descriptor_ = -1;
  }
}

}  // namespace evo

