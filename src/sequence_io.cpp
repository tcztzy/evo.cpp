// SPDX-License-Identifier: Apache-2.0
#include "evo/sequence_io.hpp"

#include <fstream>
#include <iterator>
#include <string_view>
#include <utility>

namespace evo {
namespace {

Status parse_fasta(const std::string_view input, std::vector<SequenceRecord>* const records) {
  std::size_t cursor = 0;
  SequenceRecord* current = nullptr;
  std::size_t line_number = 0;
  while (cursor < input.size()) {
    const auto line_end = input.find('\n', cursor);
    const auto end = line_end == std::string_view::npos ? input.size() : line_end;
    auto line = input.substr(cursor, end - cursor);
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }

    if (!line.empty() && line.front() == '>') {
      const auto name = line.substr(1);
      if (name.empty()) {
        return {ErrorCode::kInvalidArgument,
                "FASTA line " + std::to_string(line_number) + " has an empty header"};
      }
      if (current != nullptr && current->bytes.empty()) {
        return {ErrorCode::kInvalidArgument,
                "FASTA record '" + current->name + "' has no sequence"};
      }
      records->push_back({std::string{name}, {}});
      current = &records->back();
    } else if (!line.empty()) {
      if (current == nullptr) {
        return {ErrorCode::kInvalidArgument,
                "FASTA sequence appears before the first header at line " +
                    std::to_string(line_number)};
      }
      for (const char character : line) {
        if (character == ' ' || character == '\t' || character == '\v' ||
            character == '\f' || character == '\r') {
          return {ErrorCode::kInvalidArgument,
                  "FASTA sequence contains whitespace at line " +
                      std::to_string(line_number)};
        }
      }
      current->bytes.append(line);
    }

    if (line_end == std::string_view::npos) {
      break;
    }
    cursor = line_end + 1;
  }
  if (records->empty()) {
    return {ErrorCode::kInvalidArgument, "FASTA contains no records"};
  }
  if (records->back().bytes.empty()) {
    return {ErrorCode::kInvalidArgument,
            "FASTA record '" + records->back().name + "' has no sequence"};
  }
  return Status::Ok();
}

}  // namespace

Status read_sequence_file(const std::string& path, std::vector<SequenceRecord>* const records) {
  if (records == nullptr) {
    return {ErrorCode::kInvalidArgument, "sequence output pointer is null"};
  }
  records->clear();
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {ErrorCode::kIo, "cannot open sequence input '" + path + "'"};
  }
  std::string bytes{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  if (input.bad()) {
    return {ErrorCode::kIo, "failed while reading sequence input '" + path + "'"};
  }
  if (bytes.empty()) {
    return {ErrorCode::kInvalidArgument, "sequence input is empty: '" + path + "'"};
  }
  if (bytes.front() == '>') {
    return parse_fasta(bytes, records);
  }
  records->push_back({path, std::move(bytes)});
  return Status::Ok();
}

}  // namespace evo
