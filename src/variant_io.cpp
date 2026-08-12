// SPDX-License-Identifier: Apache-2.0
#include "evo/variant_io.hpp"

#include "evo/variant.hpp"
#include "input_stream.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <string_view>
#include <vector>

namespace evo {
namespace {

constexpr std::size_t kMaxVcfLineBytes = 16U << 20U;
constexpr std::size_t kMaxReferenceLineBytes = 16U << 20U;
constexpr std::size_t kMaxReferenceHeaderBytes = 1U << 20U;

std::vector<std::string_view> split(const std::string_view text,
                                    const char separator) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const auto end = text.find(separator, begin);
    result.push_back(text.substr(
        begin, end == std::string_view::npos ? text.size() - begin
                                             : end - begin));
    if (end == std::string_view::npos)
      break;
    begin = end + 1;
  }
  return result;
}

Status parse_positive(const std::string_view text, std::size_t *const value) {
  if (value == nullptr || text.empty())
    return {ErrorCode::kInvalidArgument, "VCF POS must be a positive integer"};
  std::size_t parsed = 0;
  for (const char character : text) {
    if (character < '0' || character > '9')
      return {ErrorCode::kInvalidArgument,
              "VCF POS must be a positive integer"};
    const auto digit = static_cast<std::size_t>(character - '0');
    if (parsed > (std::numeric_limits<std::size_t>::max() - digit) / 10U)
      return {ErrorCode::kInvalidArgument, "VCF POS overflows size_t"};
    parsed = parsed * 10U + digit;
  }
  if (parsed == 0)
    return {ErrorCode::kInvalidArgument, "VCF POS must be a positive integer"};
  *value = parsed;
  return Status::Ok();
}

std::string_view fasta_contig_name(const std::string_view header) {
  const auto end = header.find_first_of(" \t");
  return header.substr(0, end);
}

Status validate_reference_line(const std::string_view line,
                               const std::size_t line_number) {
  for (const char character : line) {
    if (std::isspace(static_cast<unsigned char>(character)) != 0) {
      return {ErrorCode::kInvalidArgument,
              "reference FASTA contains whitespace at line " +
                  std::to_string(line_number)};
    }
  }
  return Status::Ok();
}

struct ReferencePass final {
  std::size_t matches{0};
  std::size_t length{0};
  std::string allele;
  std::string slice;
};

Status scan_reference(const std::string &path, const std::string_view contig,
                      const std::size_t allele_start,
                      const std::size_t allele_end,
                      const std::size_t slice_start,
                      const std::size_t slice_end,
                      ReferencePass *const result) {
  if (result == nullptr)
    return {ErrorCode::kInvalidArgument, "reference pass output is null"};
  *result = ReferencePass{};
  return detail::with_input_stream(path, [&](std::istream &input) -> Status {
    bool have_header = false;
    bool selected = false;
    std::size_t coordinate = 0;
    std::size_t line_number = 0;
    while (true) {
      std::string line;
      bool has_line = false;
      auto status = detail::read_bounded_line(
          input,
          input.peek() == '>' ? kMaxReferenceHeaderBytes
                              : kMaxReferenceLineBytes,
          &line, &has_line);
      if (!status.ok()) {
        return Status{status.code(), status.message() + " at line " +
                                         std::to_string(line_number + 1)};
      }
      if (!has_line)
        break;
      ++line_number;
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      if (!line.empty() && line.front() == '>') {
        const auto name = fasta_contig_name(
            std::string_view{line}.substr(1));
        if (name.empty()) {
          return Status{ErrorCode::kInvalidArgument,
                        "reference FASTA has an empty header at line " +
                            std::to_string(line_number)};
        }
        have_header = true;
        selected = name == contig;
        coordinate = 0;
        if (selected)
          ++result->matches;
        continue;
      }
      if (line.empty())
        continue;
      if (!have_header) {
        return Status{ErrorCode::kInvalidArgument,
                      "reference FASTA sequence appears before a header at "
                      "line " +
                          std::to_string(line_number)};
      }
      if (!selected)
        continue;
      status = validate_reference_line(line, line_number);
      if (!status.ok())
        return status;
      if (coordinate > std::numeric_limits<std::size_t>::max() - line.size()) {
        return Status{ErrorCode::kInvalidArgument,
                      "reference contig length overflows size_t"};
      }
      const std::size_t line_end = coordinate + line.size();
      const auto append_overlap = [&](const std::size_t begin,
                                      const std::size_t end,
                                      std::string *const output) {
        const std::size_t overlap_start = std::max(coordinate, begin);
        const std::size_t overlap_end = std::min(line_end, end);
        if (overlap_start < overlap_end) {
          output->append(line.data() +
                             static_cast<std::ptrdiff_t>(overlap_start -
                                                         coordinate),
                         overlap_end - overlap_start);
        }
      };
      append_overlap(allele_start, allele_end, &result->allele);
      append_overlap(slice_start, slice_end, &result->slice);
      coordinate = line_end;
      result->length = coordinate;
    }
    if (!have_header) {
      return Status{ErrorCode::kInvalidArgument,
                    "reference input is not FASTA"};
    }
    return Status::Ok();
  });
}

} // namespace

Status stream_vcf_file(const std::string &path,
                       const VcfRecordCallback &callback) {
  if (path.empty())
    return {ErrorCode::kInvalidArgument, "VCF path must not be empty"};
  if (!callback)
    return {ErrorCode::kInvalidArgument, "VCF callback must not be empty"};
  return detail::with_input_stream(path, [&](std::istream &input) -> Status {
    bool saw_column_header = false;
    std::size_t record_index = 0;
    std::size_t line_number = 0;
    while (true) {
      std::string line;
      bool has_line = false;
      auto status = detail::read_bounded_line(input, kMaxVcfLineBytes, &line,
                                               &has_line);
      if (!status.ok()) {
        return Status{status.code(), status.message() + " at line " +
                                         std::to_string(line_number + 1)};
      }
      if (!has_line)
        break;
      ++line_number;
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      if (line.empty())
        return Status{ErrorCode::kInvalidArgument,
                      "VCF has an empty line at " +
                          std::to_string(line_number)};
      if (line.rfind("##", 0) == 0)
        continue;
      if (line.front() == '#') {
        const auto header = split(line, '\t');
        const std::vector<std::string_view> required{
            "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"};
        if (header.size() >= required.size() &&
            std::equal(required.begin(), required.end(), header.begin())) {
          if (saw_column_header) {
            return Status{ErrorCode::kInvalidArgument,
                          "VCF repeats the #CHROM header at line " +
                              std::to_string(line_number)};
          }
          saw_column_header = true;
          continue;
        }
        return Status{ErrorCode::kInvalidArgument,
                      "VCF has an unsupported header at line " +
                          std::to_string(line_number)};
      }
      if (!saw_column_header) {
        return Status{ErrorCode::kInvalidArgument,
                      "VCF data appears before the #CHROM header at line " +
                          std::to_string(line_number)};
      }
      const auto fields = split(line, '\t');
      if (fields.size() < 8 || fields[0].empty() || fields[2].empty() ||
          fields[3].empty() || fields[4].empty()) {
        return Status{ErrorCode::kInvalidArgument,
                      "VCF record has fewer than 8 valid fields at line " +
                          std::to_string(line_number)};
      }
      std::size_t position = 0;
      status = parse_positive(fields[1], &position);
      if (!status.ok())
        return Status{status.code(), status.message() + " at line " +
                                         std::to_string(line_number)};
      status = validate_iupac_dna(fields[3], "VCF REF");
      if (!status.ok())
        return Status{status.code(), status.message() + " at line " +
                                         std::to_string(line_number)};
      const auto alternates = split(fields[4], ',');
      for (std::size_t allele_index = 0; allele_index < alternates.size();
           ++allele_index) {
        status = validate_iupac_dna(alternates[allele_index], "VCF ALT");
        if (!status.ok()) {
          return Status{status.code(), status.message() + " at line " +
                                           std::to_string(line_number)};
        }
        VcfRecord record{std::string{fields[0]},
                         position,
                         std::string{fields[2]},
                         std::string{fields[3]},
                         std::string{alternates[allele_index]},
                         record_index,
                         allele_index,
                         line_number};
        status = callback(record);
        if (!status.ok())
          return status;
      }
      ++record_index;
    }
    if (!saw_column_header)
      return Status{ErrorCode::kInvalidArgument, "VCF is missing #CHROM header"};
    if (record_index == 0)
      return Status{ErrorCode::kInvalidArgument, "VCF contains no records"};
    return Status::Ok();
  });
}

Status fetch_reference_slice(const std::string &path,
                             const std::string_view contig,
                             const std::size_t position_1based,
                             const std::string_view reference,
                             const std::string_view alternate,
                             const std::size_t maximum_tokens,
                             ReferenceSlice *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "reference slice output is null"};
  *output = ReferenceSlice{};
  if (path.empty() || path == "-") {
    return {ErrorCode::kInvalidArgument,
            "reference FASTA must be a reusable file path"};
  }
  if (contig.empty())
    return {ErrorCode::kInvalidArgument, "reference contig must not be empty"};
  auto status = validate_iupac_dna(reference, "reference allele");
  if (!status.ok())
    return status;
  status = validate_iupac_dna(alternate, "alternate allele");
  if (!status.ok())
    return status;
  if (position_1based == 0)
    return {ErrorCode::kInvalidArgument, "reference position must be positive"};
  const std::size_t start = position_1based - 1;
  if (reference.size() > std::numeric_limits<std::size_t>::max() - start)
    return {ErrorCode::kInvalidArgument, "reference allele coordinates overflow"};
  const std::size_t allele_end = start + reference.size();
  const std::size_t core = std::max(reference.size(), alternate.size());
  if (maximum_tokens < core || maximum_tokens < 2) {
    return {ErrorCode::kInvalidArgument,
            "variant window must fit both alleles and at least two tokens"};
  }

  ReferencePass first;
  status = scan_reference(path, contig, start, allele_end, 0, 0, &first);
  if (!status.ok())
    return status;
  if (first.matches == 0)
    return {ErrorCode::kInvalidArgument,
            "reference FASTA has no contig '" + std::string{contig} + "'"};
  if (first.matches != 1)
    return {ErrorCode::kInvalidArgument,
            "reference FASTA contains duplicate contig '" +
                std::string{contig} + "'"};
  if (start > first.length || reference.size() > first.length - start) {
    return {ErrorCode::kInvalidArgument,
            "VCF REF extends beyond reference contig '" +
                std::string{contig} + "'"};
  }
  if (first.allele != reference) {
    return {ErrorCode::kInvalidArgument,
            "VCF REF does not match reference contig '" +
                std::string{contig} + "' at 1-based position " +
                std::to_string(position_1based)};
  }

  const std::size_t flank_budget = maximum_tokens - core;
  std::size_t left = std::min(start, flank_budget / 2);
  const std::size_t right_available = first.length - allele_end;
  std::size_t right = std::min(right_available, flank_budget - left);
  std::size_t remaining = flank_budget - left - right;
  const std::size_t extra_left = std::min(start - left, remaining);
  left += extra_left;
  remaining -= extra_left;
  right += std::min(right_available - right, remaining);
  const std::size_t slice_start = start - left;
  const std::size_t slice_end = allele_end + right;

  ReferencePass second;
  status = scan_reference(path, contig, start, allele_end, slice_start,
                          slice_end, &second);
  if (!status.ok())
    return status;
  if (second.matches != 1 || second.slice.size() != slice_end - slice_start) {
    return {ErrorCode::kIo,
            "reference FASTA changed between bounded lookup passes"};
  }
  *output = {std::string{contig}, std::move(second.slice), slice_start,
             slice_end, first.length};
  return Status::Ok();
}

} // namespace evo
