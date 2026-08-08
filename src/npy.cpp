// SPDX-License-Identifier: Apache-2.0
#include "evo/npy.hpp"

#include <cctype>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>

#include <npy.hpp>

namespace evo::npy {

static_assert(sizeof(float) == 4, "NPY F32 output requires 32-bit float");

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

  const ::npy::shape_t shape{
      static_cast<::npy::ndarray_len_t>(rows),
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

}  // namespace evo::npy
