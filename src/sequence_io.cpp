// SPDX-License-Identifier: Apache-2.0
#include "evo/sequence_io.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <istream>
#include <limits>
#include <string_view>

namespace evo {
namespace {

constexpr std::size_t kMaxFastaHeaderBytes = 1U << 20U;

Status read_bounded_line(std::istream &input, const std::size_t max_bytes,
                         std::string *const line, bool *const has_line) {
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
              "FASTA line exceeds parser limit of " +
                  std::to_string(max_bytes) + " bytes"};
    }
    line->push_back(character);
  }
  if (input.bad()) {
    return {ErrorCode::kIo, "failed while reading FASTA input"};
  }
  *has_line = consumed;
  return Status::Ok();
}

Status append_sequence_line(const std::string_view line,
                            const std::size_t line_number,
                            const std::size_t max_record_bytes,
                            SequenceRecord *const record) {
  for (const char character : line) {
    if (character == ' ' || character == '\t' || character == '\v' ||
        character == '\f' || character == '\r') {
      return {ErrorCode::kInvalidArgument,
              "FASTA sequence contains whitespace at line " +
                  std::to_string(line_number)};
    }
  }
  if (line.size() > max_record_bytes - record->bytes.size()) {
    return {ErrorCode::kInvalidArgument,
            "FASTA record '" + record->name + "' exceeds maximum " +
                std::to_string(max_record_bytes) + " bytes"};
  }
  record->bytes.append(line);
  return Status::Ok();
}

Status parse_fasta(std::istream &input, const std::size_t max_record_bytes,
                   const SequenceRecordCallback &callback) {
  SequenceRecord current;
  bool have_record = false;
  std::size_t line_number = 0;
  const std::size_t line_limit =
      std::max(max_record_bytes, kMaxFastaHeaderBytes);
  while (true) {
    std::string line;
    bool has_line = false;
    auto status = read_bounded_line(input, line_limit, &line, &has_line);
    if (!status.ok()) {
      return {status.code(),
              status.message() + " at line " + std::to_string(line_number + 1)};
    }
    if (!has_line) {
      break;
    }
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    if (!line.empty() && line.front() == '>') {
      const std::string_view name{line.data() + 1, line.size() - 1};
      if (name.empty()) {
        return {ErrorCode::kInvalidArgument, "FASTA line " +
                                                 std::to_string(line_number) +
                                                 " has an empty header"};
      }
      if (have_record && current.bytes.empty()) {
        return {ErrorCode::kInvalidArgument,
                "FASTA record '" + current.name + "' has no sequence"};
      }
      if (have_record) {
        status = callback(current);
        if (!status.ok()) {
          return status;
        }
      }
      current = {std::string{name}, {}};
      have_record = true;
    } else if (!line.empty()) {
      if (!have_record) {
        return {ErrorCode::kInvalidArgument,
                "FASTA sequence appears before the first header at line " +
                    std::to_string(line_number)};
      }
      status =
          append_sequence_line(line, line_number, max_record_bytes, &current);
      if (!status.ok()) {
        return status;
      }
    }
  }
  if (!have_record) {
    return {ErrorCode::kInvalidArgument, "FASTA contains no records"};
  }
  if (current.bytes.empty()) {
    return {ErrorCode::kInvalidArgument,
            "FASTA record '" + current.name + "' has no sequence"};
  }
  return callback(current);
}

Status parse_raw(std::istream &input, const std::string &path,
                 const std::size_t max_record_bytes,
                 const SequenceRecordCallback &callback) {
  SequenceRecord record{path, {}};
  std::array<char, 64U * 1024U> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count <= 0) {
      break;
    }
    const auto bytes = static_cast<std::size_t>(count);
    if (bytes > max_record_bytes - record.bytes.size()) {
      return {ErrorCode::kInvalidArgument,
              "raw sequence '" + path + "' exceeds maximum " +
                  std::to_string(max_record_bytes) + " bytes"};
    }
    record.bytes.append(buffer.data(), bytes);
  }
  if (input.bad()) {
    return {ErrorCode::kIo,
            "failed while reading sequence input '" + path + "'"};
  }
  if (record.bytes.empty()) {
    return {ErrorCode::kInvalidArgument,
            "sequence input is empty: '" + path + "'"};
  }
  return callback(record);
}

} // namespace

Status stream_sequence_file(const std::string &path,
                            const std::size_t max_record_bytes,
                            const SequenceRecordCallback &callback) {
  if (path.empty()) {
    return {ErrorCode::kInvalidArgument,
            "sequence input path must not be empty"};
  }
  if (max_record_bytes == 0) {
    return {ErrorCode::kInvalidArgument,
            "sequence record limit must be positive"};
  }
  if (!callback) {
    return {ErrorCode::kInvalidArgument,
            "sequence record callback must not be empty"};
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {ErrorCode::kIo, "cannot open sequence input '" + path + "'"};
  }
  const auto first = input.peek();
  if (first == std::char_traits<char>::eof()) {
    if (input.bad()) {
      return {ErrorCode::kIo,
              "failed while reading sequence input '" + path + "'"};
    }
    return {ErrorCode::kInvalidArgument,
            "sequence input is empty: '" + path + "'"};
  }
  return first == '>' ? parse_fasta(input, max_record_bytes, callback)
                      : parse_raw(input, path, max_record_bytes, callback);
}

Status read_sequence_file(const std::string &path,
                          std::vector<SequenceRecord> *const records) {
  if (records == nullptr) {
    return {ErrorCode::kInvalidArgument, "sequence output pointer is null"};
  }
  records->clear();
  return stream_sequence_file(path, std::numeric_limits<std::size_t>::max(),
                              [records](const SequenceRecord &record) {
                                records->push_back(record);
                                return Status::Ok();
                              });
}

} // namespace evo
