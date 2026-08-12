// SPDX-License-Identifier: Apache-2.0
#include "evo/npy.hpp"

#include <cctype>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>

#include <npy.hpp>

namespace evo::npy {

static_assert(sizeof(float) == 4, "NPY F32 output requires 32-bit float");

F32MatrixWriter::~F32MatrixWriter() { discard(); }

Status F32MatrixWriter::open(const std::string &path, const std::size_t rows,
                             const std::size_t columns) {
  if (output_.is_open() || path.empty() || rows == 0 || columns == 0 ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    return {ErrorCode::kInvalidArgument,
            "streaming NPY path/dimensions are invalid"};
  }
  const std::uint16_t endian_probe = 1;
  if (*reinterpret_cast<const std::uint8_t *>(&endian_probe) != 1) {
    return {ErrorCode::kUnsupported,
            "NPY F32 output requires a little-endian host"};
  }
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       std::to_string(rows) + ", " + std::to_string(columns) +
                       "), }";
  constexpr std::size_t prefix_size = 10;
  const std::size_t unpadded = prefix_size + header.size() + 1;
  const std::size_t padding = (64 - unpadded % 64) % 64;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
    return {ErrorCode::kInvalidArgument, "streaming NPY header is too large"};
  }
  path_ = path;
  expected_elements_ = rows * columns;
  written_elements_ = 0;
  completed_ = false;
  output_.open(path_, std::ios::binary | std::ios::trunc);
  if (!output_)
    return {ErrorCode::kIo, "cannot open NPY output '" + path_ + "'"};
  constexpr char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
  output_.write(magic, static_cast<std::streamsize>(sizeof(magic)));
  const auto header_size = static_cast<std::uint16_t>(header.size());
  const char length[2] = {
      static_cast<char>(header_size & 0xffU),
      static_cast<char>((header_size >> 8U) & 0xffU),
  };
  output_.write(length, static_cast<std::streamsize>(sizeof(length)));
  output_.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (!output_) {
    discard();
    return {ErrorCode::kIo, "failed to write NPY header '" + path + "'"};
  }
  return Status::Ok();
}

Status F32MatrixWriter::append(const float *const values,
                               const std::size_t elements) {
  if (!output_.is_open() || completed_ ||
      (elements != 0 && values == nullptr) ||
      elements > expected_elements_ - written_elements_ ||
      elements > static_cast<std::size_t>(
                     std::numeric_limits<std::streamsize>::max()) /
                     sizeof(float)) {
    return {ErrorCode::kInvalidArgument,
            "streaming NPY append exceeds its declared matrix"};
  }
  output_.write(reinterpret_cast<const char *>(values),
                static_cast<std::streamsize>(elements * sizeof(float)));
  if (!output_)
    return {ErrorCode::kIo, "failed to append NPY output '" + path_ + "'"};
  written_elements_ += elements;
  return Status::Ok();
}

Status F32MatrixWriter::close() {
  if (!output_.is_open() || completed_ ||
      written_elements_ != expected_elements_) {
    return {ErrorCode::kInvalidArgument, "streaming NPY matrix is incomplete"};
  }
  output_.close();
  if (!output_) {
    discard();
    return {ErrorCode::kIo, "failed to close NPY output '" + path_ + "'"};
  }
  completed_ = true;
  return Status::Ok();
}

void F32MatrixWriter::discard() noexcept {
  if (output_.is_open())
    output_.close();
  if (!completed_ && !path_.empty()) {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
}

Status write_f32(const std::string &path, const std::vector<float> &values,
                 const std::size_t rows, const std::size_t columns) {
  if (::npy::host_endian_char != ::npy::little_endian_char) {
    return {ErrorCode::kUnsupported,
            "NPY F32 output requires a little-endian host"};
  }
  if (rows == 0 || columns == 0 ||
      rows > std::numeric_limits<std::size_t>::max() / columns ||
      rows * columns != values.size()) {
    return {ErrorCode::kInvalidArgument,
            "NPY dimensions do not match its F32 payload"};
  }
  const auto maximum_shape_value = static_cast<std::uintmax_t>(
      std::numeric_limits<::npy::ndarray_len_t>::max());
  if (static_cast<std::uintmax_t>(rows) > maximum_shape_value ||
      static_cast<std::uintmax_t>(columns) > maximum_shape_value ||
      static_cast<std::uintmax_t>(values.size()) > maximum_shape_value) {
    return {ErrorCode::kInvalidArgument,
            "NPY dimensions exceed libnpy's shape capacity"};
  }
  if (values.size() >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) /
          sizeof(float)) {
    return {ErrorCode::kInvalidArgument,
            "NPY payload exceeds the platform stream size"};
  }

  const ::npy::shape_t shape{static_cast<::npy::ndarray_len_t>(rows),
                             static_cast<::npy::ndarray_len_t>(columns)};
  const ::npy::npy_data_ptr<float> data{values.data(), shape, false};

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return {ErrorCode::kIo, "cannot open NPY output '" + path + "'"};
  }

  try {
    ::npy::write_npy(output, data);
  } catch (const std::exception &error) {
    return {ErrorCode::kInternal,
            "libnpy failed to encode '" + path + "': " + error.what()};
  }
  output.close();
  if (!output) {
    return {ErrorCode::kIo, "failed to write NPY output '" + path + "'"};
  }
  return Status::Ok();
}

} // namespace evo::npy
