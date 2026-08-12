// SPDX-License-Identifier: Apache-2.0
#include "evo/sequence_io.hpp"

#include "input_stream.hpp"

#include <algorithm>
#include <array>
#include <istream>
#include <limits>
#include <string_view>

namespace evo {
namespace {

constexpr std::size_t kMaxFastaHeaderBytes = 1U << 20U;

std::size_t allow_cr(const std::size_t limit) noexcept {
  return limit == std::numeric_limits<std::size_t>::max() ? limit : limit + 1;
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
  while (true) {
    std::string line;
    bool has_line = false;
    const bool header = input.peek() == '>';
    const std::size_t remaining = have_record
                                      ? max_record_bytes - current.bytes.size()
                                      : max_record_bytes;
    auto status = detail::read_bounded_line(
        input, header ? kMaxFastaHeaderBytes : allow_cr(remaining), &line,
        &has_line);
    if (!status.ok()) {
      if (!header && have_record &&
          status.code() == ErrorCode::kInvalidArgument) {
        return {ErrorCode::kInvalidArgument,
                "FASTA record '" + current.name + "' exceeds maximum " +
                    std::to_string(max_record_bytes) + " bytes"};
      }
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
      current = {std::string{name}, {}, SequenceFormat::kFasta};
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

Status validate_fastq_sequence(const std::string_view line,
                               const std::size_t line_number) {
  if (line.empty())
    return {ErrorCode::kInvalidArgument,
            "FASTQ sequence is empty at line " + std::to_string(line_number)};
  for (const char character : line) {
    if (character == ' ' || character == '\t' || character == '\v' ||
        character == '\f' || character == '\r') {
      return {ErrorCode::kInvalidArgument,
              "FASTQ sequence contains whitespace at line " +
                  std::to_string(line_number)};
    }
  }
  return Status::Ok();
}

Status parse_fastq(std::istream &input, const std::size_t max_record_bytes,
                   const SequenceRecordCallback &callback) {
  std::size_t line_number = 0;
  std::size_t records = 0;
  while (input.peek() != std::char_traits<char>::eof()) {
    std::string header;
    bool has_line = false;
    auto status = detail::read_bounded_line(
        input, kMaxFastaHeaderBytes, &header, &has_line);
    ++line_number;
    if (!status.ok())
      return {status.code(), status.message() + " at line " +
                                 std::to_string(line_number)};
    if (!has_line || header.empty() || header.front() != '@' ||
        header.size() == 1) {
      return {ErrorCode::kInvalidArgument,
              "FASTQ line " + std::to_string(line_number) +
                  " must contain a nonempty @ header"};
    }
    if (header.back() == '\r')
      header.pop_back();
    if (header.size() == 1) {
      return {ErrorCode::kInvalidArgument,
              "FASTQ line " + std::to_string(line_number) +
                  " has an empty header"};
    }
    const std::string name = header.substr(1);

    std::string sequence;
    status = detail::read_bounded_line(input, allow_cr(max_record_bytes), &sequence,
                                       &has_line);
    ++line_number;
    if (!status.ok() || !has_line) {
      return {status.ok() ? ErrorCode::kInvalidArgument : status.code(),
              status.ok() ? "FASTQ record '" + name +
                                "' is truncated before its sequence"
                          : status.message() + " at line " +
                                std::to_string(line_number)};
    }
    if (!sequence.empty() && sequence.back() == '\r')
      sequence.pop_back();
    if (sequence.size() > max_record_bytes) {
      return {ErrorCode::kInvalidArgument,
              "FASTQ record '" + name + "' exceeds maximum " +
                  std::to_string(max_record_bytes) + " bytes"};
    }
    status = validate_fastq_sequence(sequence, line_number);
    if (!status.ok())
      return status;

    std::string plus;
    status = detail::read_bounded_line(input, kMaxFastaHeaderBytes, &plus,
                                       &has_line);
    ++line_number;
    if (!status.ok() || !has_line) {
      return {status.ok() ? ErrorCode::kInvalidArgument : status.code(),
              status.ok() ? "FASTQ record '" + name +
                                "' is truncated before its + line"
                          : status.message() + " at line " +
                                std::to_string(line_number)};
    }
    if (!plus.empty() && plus.back() == '\r')
      plus.pop_back();
    if (plus.empty() || plus.front() != '+' ||
        (plus.size() > 1 && plus.substr(1) != name)) {
      return {ErrorCode::kInvalidArgument,
              "FASTQ + line does not match record '" + name + "' at line " +
                  std::to_string(line_number)};
    }

    std::string quality;
    status = detail::read_bounded_line(input, allow_cr(max_record_bytes), &quality,
                                       &has_line);
    ++line_number;
    if (!status.ok() || !has_line) {
      return {status.ok() ? ErrorCode::kInvalidArgument : status.code(),
              status.ok() ? "FASTQ record '" + name +
                                "' is truncated before its quality"
                          : status.message() + " at line " +
                                std::to_string(line_number)};
    }
    if (!quality.empty() && quality.back() == '\r')
      quality.pop_back();
    if (quality.size() != sequence.size()) {
      return {ErrorCode::kInvalidArgument,
              "FASTQ record '" + name +
                  "' quality length does not match sequence at line " +
                  std::to_string(line_number)};
    }
    for (const char character : quality) {
      const auto value = static_cast<unsigned char>(character);
      if (value < 33U || value > 126U) {
        return {ErrorCode::kInvalidArgument,
                "FASTQ record '" + name +
                    "' has a quality byte outside ASCII 33..126 at line " +
                    std::to_string(line_number)};
      }
    }
    status = callback({name, std::move(sequence), SequenceFormat::kFastq});
    if (!status.ok())
      return status;
    ++records;
  }
  return records == 0
             ? Status{ErrorCode::kInvalidArgument, "FASTQ contains no records"}
             : Status::Ok();
}

Status parse_raw(std::istream &input, const std::string &path,
                 const std::size_t max_record_bytes,
                 const SequenceRecordCallback &callback) {
  SequenceRecord record{path == "-" ? "stdin" : path, {},
                        SequenceFormat::kRaw};
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
              "raw sequence '" + record.name + "' exceeds maximum " +
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
  return detail::with_input_stream(path, [&](std::istream &input) {
    const auto first = input.peek();
    if (first == std::char_traits<char>::eof()) {
      if (input.bad()) {
        return Status{ErrorCode::kIo,
                      "failed while reading sequence input '" + path + "'"};
      }
      return Status{ErrorCode::kInvalidArgument,
                    "sequence input is empty: '" + path + "'"};
    }
    if (first == '>')
      return parse_fasta(input, max_record_bytes, callback);
    if (first == '@')
      return parse_fastq(input, max_record_bytes, callback);
    return parse_raw(input, path, max_record_bytes, callback);
  });
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

const char *sequence_format_name(const SequenceFormat format) noexcept {
  switch (format) {
  case SequenceFormat::kRaw:
    return "raw";
  case SequenceFormat::kFasta:
    return "fasta";
  case SequenceFormat::kFastq:
    return "fastq";
  }
  return "unknown";
}

} // namespace evo
