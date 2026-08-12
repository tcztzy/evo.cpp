// SPDX-License-Identifier: Apache-2.0
#include "input_stream.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <memory>
#include <streambuf>

#include <unistd.h>

// Stable gzip subset of zlib's C ABI. Minimal runtime containers may provide
// libz.so.1 without the zlib-devel header or unversioned linker name.
extern "C" {
struct gzFile_s;
using gzFile = gzFile_s *;
gzFile gzopen(const char *path, const char *mode);
gzFile gzdopen(int descriptor, const char *mode);
int gzread(gzFile file, void *buffer, unsigned length);
const char *gzerror(gzFile file, int *error_number);
int gzclose(gzFile file);
}

namespace evo::detail {
namespace {

class GzipStreamBuffer final : public std::streambuf {
public:
  explicit GzipStreamBuffer(gzFile file) : file_(file) {
    setg(buffer_.data(), buffer_.data(), buffer_.data());
  }

  [[nodiscard]] const std::string &error() const noexcept { return error_; }

protected:
  int_type underflow() override {
    if (gptr() < egptr())
      return traits_type::to_int_type(*gptr());
    const int count = gzread(file_, buffer_.data(),
                             static_cast<unsigned>(buffer_.size()));
    if (count > 0) {
      setg(buffer_.data(), buffer_.data(), buffer_.data() + count);
      return traits_type::to_int_type(*gptr());
    }
    if (count < 0) {
      int code = 0;
      const char *const message = gzerror(file_, &code);
      error_ = message == nullptr ? "gzip read failed" : message;
    }
    return traits_type::eof();
  }

private:
  gzFile file_{nullptr};
  std::array<char, 64U * 1024U> buffer_{};
  std::string error_;
};

struct GzipCloser final {
  void operator()(gzFile_s *const file) const noexcept {
    if (file != nullptr)
      gzclose(file);
  }
};

} // namespace

Status with_input_stream(const std::string &path,
                         const InputStreamCallback &callback) {
  if (path.empty())
    return {ErrorCode::kInvalidArgument, "input path must not be empty"};
  if (!callback)
    return {ErrorCode::kInvalidArgument, "input callback must not be empty"};
  gzFile opened = nullptr;
  if (path == "-") {
    const int descriptor = ::dup(STDIN_FILENO);
    if (descriptor < 0) {
      return {ErrorCode::kIo,
              "cannot duplicate stdin: " + std::string{std::strerror(errno)}};
    }
    opened = gzdopen(descriptor, "rb");
    if (opened == nullptr) {
      ::close(descriptor);
      return {ErrorCode::kIo, "cannot open stdin through gzip reader"};
    }
  } else {
    opened = gzopen(path.c_str(), "rb");
    if (opened == nullptr)
      return {ErrorCode::kIo, "cannot open input '" + path + "'"};
  }
  std::unique_ptr<gzFile_s, GzipCloser> file(opened);
  GzipStreamBuffer buffer(opened);
  std::istream input(&buffer);
  auto status = callback(input);
  if (status.ok() && !buffer.error().empty()) {
    status = {ErrorCode::kIo,
              "failed while reading input '" + path + "': " + buffer.error()};
  }
  return status;
}

Status read_bounded_line(std::istream &input, const std::size_t max_bytes,
                         std::string *const line, bool *const has_line) {
  if (line == nullptr || has_line == nullptr)
    return {ErrorCode::kInvalidArgument, "line reader output is null"};
  line->clear();
  *has_line = false;
  bool consumed = false;
  char character = 0;
  while (input.get(character)) {
    consumed = true;
    if (character == '\n') {
      *has_line = true;
      return Status::Ok();
    }
    if (line->size() >= max_bytes) {
      return {ErrorCode::kInvalidArgument,
              "input line exceeds parser limit of " +
                  std::to_string(max_bytes) + " bytes"};
    }
    line->push_back(character);
  }
  if (input.bad())
    return {ErrorCode::kIo, "failed while reading input line"};
  *has_line = consumed;
  return Status::Ok();
}

} // namespace evo::detail
