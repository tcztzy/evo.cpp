// SPDX-License-Identifier: Apache-2.0
#include "evo/variant.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace evo {
namespace {

char complement(const char value) noexcept {
  const bool lower = std::islower(static_cast<unsigned char>(value)) != 0;
  const char upper =
      static_cast<char>(std::toupper(static_cast<unsigned char>(value)));
  char result = 0;
  switch (upper) {
  case 'A':
    result = 'T';
    break;
  case 'C':
    result = 'G';
    break;
  case 'G':
    result = 'C';
    break;
  case 'T':
    result = 'A';
    break;
  case 'R':
    result = 'Y';
    break;
  case 'Y':
    result = 'R';
    break;
  case 'S':
    result = 'S';
    break;
  case 'W':
    result = 'W';
    break;
  case 'K':
    result = 'M';
    break;
  case 'M':
    result = 'K';
    break;
  case 'B':
    result = 'V';
    break;
  case 'D':
    result = 'H';
    break;
  case 'H':
    result = 'D';
    break;
  case 'V':
    result = 'B';
    break;
  case 'N':
    result = 'N';
    break;
  default:
    return 0;
  }
  return lower ? static_cast<char>(
                     std::tolower(static_cast<unsigned char>(result)))
               : result;
}

} // namespace

Status validate_iupac_dna(const std::string_view sequence,
                          const std::string_view label) {
  if (sequence.empty()) {
    return {ErrorCode::kInvalidArgument,
            std::string{label} + " must not be empty"};
  }
  for (std::size_t index = 0; index < sequence.size(); ++index) {
    if (complement(sequence[index]) == 0) {
      return {ErrorCode::kInvalidArgument,
              std::string{label} + " contains non-IUPAC DNA byte at offset " +
                  std::to_string(index)};
    }
  }
  return Status::Ok();
}

Status reverse_complement(const std::string_view sequence,
                          std::string *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "reverse-complement output is null"};
  auto status = validate_iupac_dna(sequence, "sequence");
  if (!status.ok())
    return status;
  output->clear();
  output->reserve(sequence.size());
  for (auto iterator = sequence.rbegin(); iterator != sequence.rend();
       ++iterator) {
    output->push_back(complement(*iterator));
  }
  return Status::Ok();
}

Status make_variant_window(const std::string_view sequence,
                           const std::size_t position_1based,
                           const std::string_view reference,
                           const std::string_view alternate,
                           const std::size_t maximum_tokens,
                           VariantWindow *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "variant window output is null"};
  *output = VariantWindow{};
  auto status = validate_iupac_dna(sequence, "reference sequence");
  if (!status.ok())
    return status;
  status = validate_iupac_dna(reference, "reference allele");
  if (!status.ok())
    return status;
  status = validate_iupac_dna(alternate, "alternate allele");
  if (!status.ok())
    return status;
  if (position_1based == 0) {
    return {ErrorCode::kInvalidArgument,
            "variant position is 1-based and must be positive"};
  }
  const std::size_t start = position_1based - 1;
  if (start > sequence.size() || reference.size() > sequence.size() - start) {
    return {ErrorCode::kInvalidArgument,
            "reference allele extends beyond the supplied sequence"};
  }
  if (sequence.substr(start, reference.size()) != reference) {
    return {ErrorCode::kInvalidArgument,
            "reference allele does not match the supplied sequence at "
            "1-based position " +
                std::to_string(position_1based)};
  }
  const std::size_t core = std::max(reference.size(), alternate.size());
  if (maximum_tokens < core || maximum_tokens < 2) {
    return {ErrorCode::kInvalidArgument,
            "variant window must fit both alleles and at least two tokens"};
  }
  const std::size_t end = start + reference.size();
  const std::size_t left_available = start;
  const std::size_t right_available = sequence.size() - end;
  const std::size_t flank_budget = maximum_tokens - core;
  std::size_t left = std::min(left_available, flank_budget / 2);
  std::size_t right = std::min(right_available, flank_budget - left);
  std::size_t remaining = flank_budget - left - right;
  const std::size_t extra_left = std::min(left_available - left, remaining);
  left += extra_left;
  remaining -= extra_left;
  right += std::min(right_available - right, remaining);

  output->reference_start = start - left;
  output->reference_end = end + right;
  output->variant_offset = left;
  const std::string_view left_flank =
      sequence.substr(output->reference_start, left);
  const std::string_view right_flank = sequence.substr(end, right);
  output->reference.reserve(left + reference.size() + right);
  output->reference.append(left_flank.data(), left_flank.size());
  output->reference.append(reference.data(), reference.size());
  output->reference.append(right_flank.data(), right_flank.size());
  output->alternate.reserve(left + alternate.size() + right);
  output->alternate.append(left_flank.data(), left_flank.size());
  output->alternate.append(alternate.data(), alternate.size());
  output->alternate.append(right_flank.data(), right_flank.size());
  if (output->reference.size() < 2 || output->alternate.size() < 2) {
    *output = VariantWindow{};
    return {ErrorCode::kInvalidArgument,
            "reference and alternate windows must each contain two tokens"};
  }
  return Status::Ok();
}

const char *variant_strand_name(const VariantStrand strand) noexcept {
  switch (strand) {
  case VariantStrand::kForward:
    return "+";
  case VariantStrand::kReverse:
    return "-";
  case VariantStrand::kBoth:
    return "both";
  }
  return "unknown";
}

const char *
variant_normalization_name(const VariantNormalization normalization) noexcept {
  switch (normalization) {
  case VariantNormalization::kSum:
    return "sum_log_likelihood";
  case VariantNormalization::kMean:
    return "mean_log_likelihood";
  }
  return "unknown";
}

} // namespace evo
