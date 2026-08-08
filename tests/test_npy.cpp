// SPDX-License-Identifier: Apache-2.0
#include "evo/npy.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

class TemporaryPath final {
 public:
  TemporaryPath() {
    const char *temporary_directory = std::getenv("TMPDIR");
    std::string pattern =
        temporary_directory != nullptr && temporary_directory[0] != '\0'
            ? temporary_directory
            : "/tmp";
    if (pattern.back() != '/') {
      pattern.push_back('/');
    }
    pattern += "evo-npy-XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int descriptor = ::mkstemp(writable.data());
    if (descriptor < 0) {
      std::abort();
    }
    ::close(descriptor);
    path_ = writable.data();
  }

  ~TemporaryPath() { static_cast<void>(::unlink(path_.c_str())); }

  TemporaryPath(const TemporaryPath &) = delete;
  TemporaryPath &operator=(const TemporaryPath &) = delete;

  [[nodiscard]] const std::string &path() const noexcept { return path_; }

 private:
  std::string path_;
};

std::vector<char> read_bytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void check_file(const std::string &path, const std::vector<float> &expected,
                const std::string_view expected_shape) {
  const auto bytes = read_bytes(path);
  constexpr std::array<char, 8> magic{
      static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y', 1, 0};
  if (bytes.size() < 10) {
    check(false, "NPY output contains its version-1 prefix");
    return;
  }
  check(std::equal(magic.begin(), magic.end(), bytes.begin()),
        "NPY output has the version-1 magic");
  const auto header_size =
      static_cast<std::size_t>(static_cast<unsigned char>(bytes[8])) |
      (static_cast<std::size_t>(static_cast<unsigned char>(bytes[9])) << 8U);
  const std::size_t payload_offset = 10U + header_size;
  if (payload_offset > bytes.size()) {
    check(false, "NPY output contains its complete header");
    return;
  }
  const std::string header(bytes.data() + 10,
                           bytes.data() + payload_offset);
  check(header.find("'descr': '<f4'") != std::string::npos,
        "NPY output declares little-endian F32");
  check(header.find("'fortran_order': False") != std::string::npos,
        "NPY output declares row-major order");
  check(header.find("'shape': " + std::string{expected_shape}) !=
            std::string::npos,
        "NPY output preserves its matrix shape");
  std::string expected_header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': " +
      std::string{expected_shape} + ", }";
  const std::size_t expected_padding =
      16U - ((10U + expected_header.size() + 1U) % 16U);
  expected_header.append(expected_padding, ' ');
  expected_header.push_back('\n');
  check(header == expected_header,
        "NPY header matches libnpy's native version-1 encoding");
  check(payload_offset % 16U == 0,
        "NPY header satisfies libnpy's version-1 alignment contract");
  if (bytes.size() != payload_offset + expected.size() * sizeof(float)) {
    check(false, "NPY payload length matches its shape");
    return;
  }
  check(std::memcmp(bytes.data() + payload_offset, expected.data(),
                    expected.size() * sizeof(float)) == 0,
        "NPY output preserves every F32 payload bit");
}

void test_write_and_overwrite() {
  TemporaryPath output;
  const std::vector<float> matrix{1.0F, -2.5F, 3.25F,
                                  0.0F, 42.5F, -0.125F};
  auto status = evo::npy::write_f32(output.path(), matrix, 2, 3);
  check(status.ok(), std::string{"2x3 NPY write succeeds: "} +
                         status.message());
  if (status.ok()) {
    check_file(output.path(), matrix, "(2, 3)");
  }

  const std::vector<float> scalar_matrix{7.0F};
  status = evo::npy::write_f32(output.path(), scalar_matrix, 1, 1);
  check(status.ok(), std::string{"shorter NPY overwrite succeeds: "} +
                         status.message());
  if (status.ok()) {
    check_file(output.path(), scalar_matrix, "(1, 1)");
  }
}

void test_validation_precedes_output() {
  TemporaryPath output;
  const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
  const auto original = read_bytes(output.path());

  auto status = evo::npy::write_f32(output.path(), values, 2, 3);
  check(status.code() == evo::ErrorCode::kInvalidArgument,
        "mismatched NPY dimensions are rejected");
  check(read_bytes(output.path()) == original,
        "dimension failure does not touch the output");

  status = evo::npy::write_f32(output.path(), values, 0, 4);
  check(status.code() == evo::ErrorCode::kInvalidArgument,
        "empty NPY dimensions are rejected");
  status = evo::npy::write_f32(output.path(), values,
                               std::numeric_limits<std::size_t>::max(), 2);
  check(status.code() == evo::ErrorCode::kInvalidArgument,
        "overflowing NPY dimensions are rejected");
  check(read_bytes(output.path()) == original,
        "all validation failures leave the output unchanged");
}

void test_io_error() {
  TemporaryPath regular_file;
  const std::string path = regular_file.path() + "/child.npy";
  const auto status = evo::npy::write_f32(path, {1.0F}, 1, 1);
  check(status.code() == evo::ErrorCode::kIo,
        "unopenable NPY output reports an I/O error");
  check(status.message().find(path) != std::string::npos,
        "NPY I/O error identifies its output path");
}

}  // namespace

int main() {
  test_write_and_overwrite();
  test_validation_precedes_output();
  test_io_error();
  if (failures != 0) {
    std::cerr << failures << " NPY test(s) failed\n";
    return 1;
  }
  std::cout << "NPY tests passed\n";
  return 0;
}
