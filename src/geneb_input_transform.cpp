// SPDX-License-Identifier: Apache-2.0
#include "evo/geneb_input_transform.hpp"

#include <limits>
#include <string>
#include <utility>

namespace evo {
namespace {

Status invalid_metadata(const std::string_view field,
                        const std::string_view value) {
  return {ErrorCode::kInvalidArgument,
          "GENEB input-transform metadata field '" + std::string{field} +
              "' has unsupported value '" + std::string{value} + "'"};
}

bool to_size(const std::uint64_t value, std::size_t *const output) noexcept {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  *output = static_cast<std::size_t>(value);
  return true;
}

bool is_acgt(const char byte) noexcept {
  return byte == 'A' || byte == 'C' || byte == 'G' || byte == 'T' ||
         byte == 'a' || byte == 'c' || byte == 'g' || byte == 't';
}

bool is_n(const char byte) noexcept { return byte == 'N' || byte == 'n'; }

bool is_ascii_whitespace(const char byte) noexcept {
  return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' ||
         byte == '\f' || byte == '\v';
}

char ascii_upper(const char byte) noexcept {
  return byte >= 'a' && byte <= 'z' ? static_cast<char>(byte - ('a' - 'A'))
                                    : byte;
}

void padding_counts(const std::size_t amount, const GenebPaddingSide side,
                    const GenebPaddingBalance balance, std::size_t *const left,
                    std::size_t *const right) noexcept {
  *left = 0;
  *right = 0;
  switch (side) {
  case GenebPaddingSide::kLeft:
    *left = amount;
    return;
  case GenebPaddingSide::kRight:
    *right = amount;
    return;
  case GenebPaddingSide::kBoth:
    if (balance == GenebPaddingBalance::kExtraLeft) {
      *right = amount / 2U;
      *left = amount - *right;
    } else {
      *left = amount / 2U;
      *right = amount - *left;
    }
    return;
  }
}

Status parse_case(const std::string_view value, GenebCasePolicy *const output) {
  if (value == "preserve")
    *output = GenebCasePolicy::kPreserve;
  else if (value == "upper")
    *output = GenebCasePolicy::kUpper;
  else
    return invalid_metadata("case", value);
  return Status::Ok();
}

Status parse_invalid(const std::string_view value,
                     GenebInvalidBasePolicy *const output) {
  if (value == "tokenizer-defined")
    *output = GenebInvalidBasePolicy::kTokenizerDefined;
  else if (value == "biofm-tokenizer-defined")
    *output = GenebInvalidBasePolicy::kBiofmTokenizerDefined;
  else if (value == "non-acgt-to-N")
    *output = GenebInvalidBasePolicy::kNonAcgtToN;
  else if (value == "unknown-to-N")
    *output = GenebInvalidBasePolicy::kUnknownToN;
  else if (value == "zero-vector")
    *output = GenebInvalidBasePolicy::kZeroVector;
  else if (value == "reject")
    *output = GenebInvalidBasePolicy::kReject;
  else
    return invalid_metadata("invalid", value);
  return Status::Ok();
}

Status parse_trim_side(const std::string_view value,
                       GenebTrimSide *const output) {
  if (value == "left")
    *output = GenebTrimSide::kLeft;
  else if (value == "right")
    *output = GenebTrimSide::kRight;
  else
    return invalid_metadata("frame_trim.remove_from", value);
  return Status::Ok();
}

Status parse_crop_keep(const std::string_view value,
                       GenebCropKeep *const output) {
  if (value == "prefix" || value == "left")
    *output = GenebCropKeep::kLeft;
  else if (value == "suffix" || value == "right")
    *output = GenebCropKeep::kRight;
  else if (value == "center")
    *output = GenebCropKeep::kCenter;
  else
    return invalid_metadata("raw_crop.keep", value);
  return Status::Ok();
}

Status parse_padding_side(const std::string_view value,
                          GenebPaddingSide *const output) {
  if (value == "left")
    *output = GenebPaddingSide::kLeft;
  else if (value == "right")
    *output = GenebPaddingSide::kRight;
  else if (value == "both")
    *output = GenebPaddingSide::kBoth;
  else
    return invalid_metadata("fixed_pad.side", value);
  return Status::Ok();
}

Status parse_padding_value(const std::string_view value,
                           GenebPaddingValue *const output) {
  if (value == "N")
    *output = GenebPaddingValue::kN;
  else if (value == "tokenizer-pad")
    *output = GenebPaddingValue::kTokenizerPad;
  else
    return invalid_metadata("fixed_pad.value", value);
  return Status::Ok();
}

Status parse_padding_balance(const std::string_view value,
                             GenebPaddingBalance *const output) {
  if (value.empty())
    *output = GenebPaddingBalance::kNotApplicable;
  else if (value == "extra-left")
    *output = GenebPaddingBalance::kExtraLeft;
  else if (value == "extra-right")
    *output = GenebPaddingBalance::kExtraRight;
  else
    return invalid_metadata("fixed_pad.balance", value);
  return Status::Ok();
}

Status parse_special_tokens(const std::string_view value,
                            GenebSpecialTokenPolicy *const output) {
  if (value == "none")
    *output = GenebSpecialTokenPolicy::kNone;
  else if (value == "tokenizer-default")
    *output = GenebSpecialTokenPolicy::kTokenizerDefault;
  else if (value == "tokenizer-default-then-one-hot-float")
    *output = GenebSpecialTokenPolicy::kTokenizerDefaultThenOneHotFloat;
  else if (value == "tokenizer-default-with-eos-as-pad")
    *output = GenebSpecialTokenPolicy::kTokenizerDefaultWithEosAsPad;
  else if (value == "prefix-only")
    *output = GenebSpecialTokenPolicy::kPrefixOnly;
  else if (value == "manual-bos-plus-tokenizer-default")
    *output = GenebSpecialTokenPolicy::kManualBosPlusTokenizerDefault;
  else if (value == "manual-cls-then-sep-before-sequence")
    *output = GenebSpecialTokenPolicy::kManualClsThenSepBeforeSequence;
  else if (value == "gene-modality-default")
    *output = GenebSpecialTokenPolicy::kGeneModalityDefault;
  else
    return invalid_metadata("special_tokens", value);
  return Status::Ok();
}

Status parse_token_truncation(const std::string_view value,
                              GenebTokenTruncation *const output) {
  if (value == "none")
    *output = GenebTokenTruncation::kNone;
  else if (value == "left")
    *output = GenebTokenTruncation::kLeft;
  else if (value == "right")
    *output = GenebTokenTruncation::kRight;
  else if (value == "model-preset")
    *output = GenebTokenTruncation::kModelPreset;
  else
    return invalid_metadata("token_truncation", value);
  return Status::Ok();
}

Status parse_context_unit(const std::string_view value,
                          GenebContextUnit *const output) {
  if (value == "bases")
    *output = GenebContextUnit::kBases;
  else if (value == "tokens")
    *output = GenebContextUnit::kTokens;
  else
    return invalid_metadata("context.unit", value);
  return Status::Ok();
}

Status parse_length_policy(const std::string_view value,
                           GenebLengthPolicy *const output) {
  if (value == "reject")
    *output = GenebLengthPolicy::kReject;
  else if (value == "tokenizer-truncate")
    *output = GenebLengthPolicy::kTokenizerTruncate;
  else if (value == "model-preset")
    *output = GenebLengthPolicy::kModelPreset;
  else
    return invalid_metadata("context.length_policy", value);
  return Status::Ok();
}

} // namespace

Status check_geneb_raw_sequence_append(const std::size_t materialized_bytes,
                                       const std::size_t incoming_bytes,
                                       const std::size_t raw_safety_cap) {
  if (raw_safety_cap == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB raw sequence safety cap must be positive"};
  }
  if (materialized_bytes > raw_safety_cap ||
      incoming_bytes > raw_safety_cap - materialized_bytes) {
    return {ErrorCode::kInvalidArgument,
            "GENEB raw sequence exceeds safety cap of " +
                std::to_string(raw_safety_cap) + " bytes before append"};
  }
  return Status::Ok();
}

Status GenebBoundedSequenceRecord::append(const std::string_view chunk) {
  auto status = check_geneb_raw_sequence_append(bytes_.size(), chunk.size(),
                                                raw_safety_cap_);
  if (!status.ok())
    return status;
  if (!chunk.empty())
    bytes_.append(chunk.data(), chunk.size());
  return Status::Ok();
}

std::string GenebBoundedSequenceRecord::take() noexcept {
  std::string result = std::move(bytes_);
  bytes_.clear();
  return result;
}

Status
compile_geneb_input_transform(const GenebInputTransformMetadata &metadata,
                              GenebInputTransformSpec *const output) {
  if (output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "GENEB input-transform output is null"};
  }
  GenebInputTransformSpec compiled;
  if (!to_size(metadata.raw_safety_cap, &compiled.raw_safety_cap) ||
      compiled.raw_safety_cap == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB raw_safety_cap must be a positive size_t"};
  }
  compiled.strip_ascii_whitespace = metadata.strip_ascii_whitespace;
  auto status = parse_case(metadata.case_policy, &compiled.case_policy);
  if (!status.ok())
    return status;
  compiled.u_to_t = metadata.u_to_t;
  status = parse_invalid(metadata.invalid_base_policy,
                         &compiled.invalid_base_policy);
  if (!status.ok())
    return status;

  compiled.frame_trim_enabled = metadata.frame_trim.present;
  if (metadata.frame_trim.present) {
    if (!to_size(metadata.frame_trim.multiple, &compiled.frame_multiple) ||
        compiled.frame_multiple == 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB frame_trim.multiple must be a positive size_t"};
    }
    status = parse_trim_side(metadata.frame_trim.remove_from,
                             &compiled.frame_trim_side);
    if (!status.ok())
      return status;
  } else if (metadata.frame_trim.multiple != 0 ||
             !metadata.frame_trim.remove_from.empty()) {
    return {ErrorCode::kInvalidArgument,
            "GENEB absent frame_trim must not carry values"};
  }

  compiled.raw_crop_enabled = metadata.raw_crop.present;
  if (metadata.raw_crop.present) {
    if (!to_size(metadata.raw_crop.length, &compiled.raw_crop_length) ||
        compiled.raw_crop_length == 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB raw_crop.length must be a positive size_t"};
    }
    status = parse_crop_keep(metadata.raw_crop.keep, &compiled.raw_crop_keep);
    if (!status.ok())
      return status;
  } else if (metadata.raw_crop.length != 0 || !metadata.raw_crop.keep.empty()) {
    return {ErrorCode::kInvalidArgument,
            "GENEB absent raw_crop must not carry values"};
  }

  compiled.fixed_padding_enabled = metadata.fixed_padding.present;
  if (metadata.fixed_padding.present) {
    if (!to_size(metadata.fixed_padding.length,
                 &compiled.fixed_padding_length) ||
        compiled.fixed_padding_length == 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB fixed_pad.length must be a positive size_t"};
    }
    status = parse_padding_side(metadata.fixed_padding.side,
                                &compiled.fixed_padding_side);
    if (!status.ok())
      return status;
    status = parse_padding_value(metadata.fixed_padding.value,
                                 &compiled.fixed_padding_value);
    if (!status.ok())
      return status;
    status = parse_padding_balance(metadata.fixed_padding.balance,
                                   &compiled.fixed_padding_balance);
    if (!status.ok())
      return status;
  } else if (metadata.fixed_padding.length != 0 ||
             !metadata.fixed_padding.side.empty() ||
             !metadata.fixed_padding.value.empty() ||
             !metadata.fixed_padding.balance.empty()) {
    return {ErrorCode::kInvalidArgument,
            "GENEB absent fixed_pad must not carry values"};
  }

  compiled.prefix = metadata.prefix;
  status = parse_special_tokens(metadata.special_token_policy,
                                &compiled.special_token_policy);
  if (!status.ok())
    return status;
  status = parse_token_truncation(metadata.token_truncation,
                                  &compiled.token_truncation);
  if (!status.ok())
    return status;
  status = parse_context_unit(metadata.context_unit, &compiled.context_unit);
  if (!status.ok())
    return status;
  status = parse_length_policy(metadata.length_policy, &compiled.length_policy);
  if (!status.ok())
    return status;
  if (!to_size(metadata.reference_context_limit,
               &compiled.reference_context_limit)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB reference context limit exceeds size_t"};
  }

  status = validate_geneb_input_transform_spec(compiled);
  if (!status.ok())
    return status;
  *output = std::move(compiled);
  return Status::Ok();
}

Status
validate_geneb_input_transform_spec(const GenebInputTransformSpec &spec) {
  if (spec.raw_safety_cap == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB raw sequence safety cap must be positive"};
  }
  if (spec.frame_trim_enabled && spec.frame_multiple == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB frame trim multiple must be positive"};
  }
  if (spec.raw_crop_enabled && spec.raw_crop_length == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB raw crop length must be positive"};
  }
  if (spec.fixed_padding_enabled && spec.fixed_padding_length == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB fixed padding length must be positive"};
  }
  if (spec.raw_crop_enabled && spec.raw_crop_length > spec.raw_safety_cap) {
    return {ErrorCode::kInvalidArgument,
            "GENEB raw crop length exceeds raw safety cap"};
  }
  if (spec.fixed_padding_enabled &&
      spec.fixed_padding_value == GenebPaddingValue::kN &&
      spec.fixed_padding_length > spec.raw_safety_cap) {
    return {ErrorCode::kInvalidArgument,
            "GENEB raw fixed padding length exceeds raw safety cap"};
  }
  if (spec.fixed_padding_side == GenebPaddingSide::kBoth) {
    if (spec.fixed_padding_balance == GenebPaddingBalance::kNotApplicable) {
      return {ErrorCode::kInvalidArgument,
              "GENEB two-sided padding requires an extra-left/right balance"};
    }
  } else if (spec.fixed_padding_balance !=
             GenebPaddingBalance::kNotApplicable) {
    return {ErrorCode::kInvalidArgument,
            "GENEB one-sided padding must not declare a balance"};
  }
  if (spec.context_unit == GenebContextUnit::kBases &&
      spec.fixed_padding_enabled &&
      spec.fixed_padding_value == GenebPaddingValue::kTokenizerPad) {
    return {ErrorCode::kInvalidArgument,
            "GENEB base-unit context cannot use tokenizer padding"};
  }
  if (spec.length_policy == GenebLengthPolicy::kReject &&
      spec.reference_context_limit == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB reject length policy requires a known context limit"};
  }
  if (spec.length_policy == GenebLengthPolicy::kTokenizerTruncate &&
      spec.token_truncation == GenebTokenTruncation::kNone) {
    return {ErrorCode::kInvalidArgument,
            "GENEB tokenizer-truncate length policy requires an explicit "
            "left/right or model-preset token truncation"};
  }
  const bool prefix_policy =
      spec.special_token_policy == GenebSpecialTokenPolicy::kPrefixOnly ||
      spec.special_token_policy ==
          GenebSpecialTokenPolicy::kManualBosPlusTokenizerDefault ||
      spec.special_token_policy ==
          GenebSpecialTokenPolicy::kManualClsThenSepBeforeSequence;
  if (prefix_policy != !spec.prefix.empty()) {
    return {ErrorCode::kInvalidArgument,
            prefix_policy
                ? "GENEB special-token policy requires a prefix handoff"
                : "GENEB prefix requires a prefix-aware special-token policy"};
  }
  if (spec.special_token_policy ==
          GenebSpecialTokenPolicy::kManualBosPlusTokenizerDefault &&
      spec.prefix != "tokenizer-bos-text") {
    return {ErrorCode::kInvalidArgument,
            "GENEB manual BOS policy requires the tokenizer-bos-text marker"};
  }
  return Status::Ok();
}

Status transform_geneb_input(const std::string_view raw_sequence,
                             const GenebInputTransformSpec &spec,
                             GenebInputTransformResult *const output) {
  if (output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "GENEB input-transform result is null"};
  }
  auto status = validate_geneb_input_transform_spec(spec);
  if (!status.ok())
    return status;
  // This check deliberately precedes the first output string allocation.
  status = check_geneb_raw_sequence_append(0, raw_sequence.size(),
                                           spec.raw_safety_cap);
  if (!status.ok())
    return status;

  GenebInputTransformResult transformed;
  transformed.original_length = raw_sequence.size();
  transformed.prefix = spec.prefix;
  transformed.special_token_policy = spec.special_token_policy;
  if (!raw_sequence.empty())
    transformed.sequence.assign(raw_sequence.data(), raw_sequence.size());

  if (spec.strip_ascii_whitespace) {
    std::size_t begin = 0;
    while (begin < transformed.sequence.size() &&
           is_ascii_whitespace(transformed.sequence[begin]))
      ++begin;
    std::size_t end = transformed.sequence.size();
    while (end > begin && is_ascii_whitespace(transformed.sequence[end - 1U]))
      --end;
    transformed.trim_left = begin;
    transformed.trim_right = transformed.sequence.size() - end;
    if (begin != 0 || end != transformed.sequence.size())
      transformed.sequence = transformed.sequence.substr(begin, end - begin);
  }

  if (spec.case_policy == GenebCasePolicy::kUpper) {
    for (char &byte : transformed.sequence)
      byte = ascii_upper(byte);
  }
  if (spec.u_to_t) {
    for (char &byte : transformed.sequence) {
      if (byte == 'U')
        byte = 'T';
      else if (byte == 'u')
        byte = 't';
    }
  }

  switch (spec.invalid_base_policy) {
  case GenebInvalidBasePolicy::kTokenizerDefined:
  case GenebInvalidBasePolicy::kBiofmTokenizerDefined:
    break;
  case GenebInvalidBasePolicy::kNonAcgtToN:
  case GenebInvalidBasePolicy::kZeroVector:
    for (char &byte : transformed.sequence) {
      if (!is_acgt(byte)) {
        ++transformed.invalid_base_count;
        byte = 'N';
      }
    }
    break;
  case GenebInvalidBasePolicy::kUnknownToN:
    for (char &byte : transformed.sequence) {
      if (!is_acgt(byte) && !is_n(byte)) {
        ++transformed.invalid_base_count;
        byte = 'N';
      }
    }
    break;
  case GenebInvalidBasePolicy::kReject:
    for (std::size_t index = 0; index < transformed.sequence.size(); ++index) {
      if (!is_acgt(transformed.sequence[index]) &&
          !is_n(transformed.sequence[index])) {
        return {ErrorCode::kInvalidArgument,
                "GENEB input contains an invalid base at byte " +
                    std::to_string(index)};
      }
    }
    break;
  }

  if (spec.frame_trim_enabled) {
    const std::size_t amount =
        transformed.sequence.size() % spec.frame_multiple;
    if (spec.frame_trim_side == GenebTrimSide::kLeft) {
      transformed.trim_left += amount;
      transformed.sequence.erase(0, amount);
    } else {
      transformed.trim_right += amount;
      transformed.sequence.resize(transformed.sequence.size() - amount);
    }
  }

  if (spec.raw_crop_enabled &&
      transformed.sequence.size() > spec.raw_crop_length) {
    const std::size_t removed =
        transformed.sequence.size() - spec.raw_crop_length;
    switch (spec.raw_crop_keep) {
    case GenebCropKeep::kLeft:
      transformed.crop_right = removed;
      break;
    case GenebCropKeep::kRight:
      transformed.crop_left = removed;
      break;
    case GenebCropKeep::kCenter:
      transformed.crop_left = removed / 2U;
      transformed.crop_right = removed - transformed.crop_left;
      break;
    }
    transformed.crop_source_offset = transformed.crop_left;
    transformed.sequence = transformed.sequence.substr(transformed.crop_left,
                                                       spec.raw_crop_length);
  }

  if (spec.fixed_padding_enabled &&
      spec.fixed_padding_value == GenebPaddingValue::kN &&
      transformed.sequence.size() < spec.fixed_padding_length) {
    const std::size_t amount =
        spec.fixed_padding_length - transformed.sequence.size();
    padding_counts(amount, spec.fixed_padding_side, spec.fixed_padding_balance,
                   &transformed.pad_left, &transformed.pad_right);
    if (transformed.pad_left != 0) {
      transformed.sequence.insert(0, transformed.pad_left, 'N');
    }
    if (transformed.pad_right != 0) {
      transformed.sequence.append(transformed.pad_right, 'N');
    }
  }
  transformed.effective_length = transformed.sequence.size();

  if (spec.context_unit == GenebContextUnit::kBases &&
      spec.reference_context_limit != 0 &&
      transformed.effective_length > spec.reference_context_limit &&
      spec.length_policy == GenebLengthPolicy::kReject) {
    return {ErrorCode::kInvalidArgument,
            "GENEB transformed base length exceeds reject-policy context " +
                std::to_string(spec.reference_context_limit)};
  }
  *output = std::move(transformed);
  return Status::Ok();
}

Status plan_geneb_token_length(const std::size_t token_count,
                               const GenebInputTransformSpec &spec,
                               GenebTokenLengthPlan *const output) {
  return plan_geneb_token_length(token_count, 0, 0, spec, output);
}

Status plan_geneb_token_length(const std::size_t token_count,
                               const std::size_t prefix_token_count,
                               const std::size_t suffix_token_count,
                               const GenebInputTransformSpec &spec,
                               GenebTokenLengthPlan *const output) {
  if (output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "GENEB token-length plan output is null"};
  }
  auto status = validate_geneb_input_transform_spec(spec);
  if (!status.ok())
    return status;
  if (prefix_token_count > token_count ||
      suffix_token_count > token_count - prefix_token_count) {
    return {ErrorCode::kInvalidArgument,
            "GENEB tokenizer boundary count exceeds token count"};
  }

  GenebTokenLengthPlan plan;
  plan.original_token_count = token_count;
  plan.prefix_token_count = prefix_token_count;
  plan.suffix_token_count = suffix_token_count;
  plan.source_offset = prefix_token_count;
  plan.retained_payload_token_count =
      token_count - prefix_token_count - suffix_token_count;
  plan.retained_token_count = token_count;
  plan.prefix = spec.prefix;
  plan.special_token_policy = spec.special_token_policy;

  const std::size_t limit = spec.context_unit == GenebContextUnit::kTokens
                                ? spec.reference_context_limit
                                : 0;
  if (limit != 0 && token_count > limit) {
    plan.exceeds_context = true;
    if (spec.length_policy == GenebLengthPolicy::kReject) {
      return {ErrorCode::kInvalidArgument,
              "GENEB token count exceeds reject-policy context " +
                  std::to_string(limit)};
    }
    if (limit < prefix_token_count + suffix_token_count) {
      return {ErrorCode::kInvalidArgument,
              "GENEB context cannot retain tokenizer boundary tokens"};
    }
    if (spec.length_policy == GenebLengthPolicy::kModelPreset ||
        spec.token_truncation == GenebTokenTruncation::kModelPreset) {
      plan.deferred_to_model_preset = true;
    } else if (spec.length_policy == GenebLengthPolicy::kTokenizerTruncate) {
      const std::size_t removed = token_count - limit;
      const std::size_t payload_capacity =
          limit - prefix_token_count - suffix_token_count;
      if (spec.token_truncation == GenebTokenTruncation::kLeft) {
        plan.source_offset = prefix_token_count + removed;
        plan.truncated_left = removed;
        plan.retained_payload_token_count = payload_capacity;
        plan.retained_token_count = limit;
      } else if (spec.token_truncation == GenebTokenTruncation::kRight) {
        plan.truncated_right = removed;
        plan.retained_payload_token_count = payload_capacity;
        plan.retained_token_count = limit;
      }
      // kNone is rejected by spec validation before this branch.
    }
  }

  if (spec.fixed_padding_enabled &&
      spec.fixed_padding_value == GenebPaddingValue::kTokenizerPad &&
      plan.retained_token_count < spec.fixed_padding_length) {
    const std::size_t amount =
        spec.fixed_padding_length - plan.retained_token_count;
    padding_counts(amount, spec.fixed_padding_side, spec.fixed_padding_balance,
                   &plan.pad_left, &plan.pad_right);
  }
  plan.effective_token_count =
      plan.retained_token_count + plan.pad_left + plan.pad_right;
  *output = std::move(plan);
  return Status::Ok();
}

} // namespace evo
