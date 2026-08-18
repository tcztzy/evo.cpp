// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "evo/status.hpp"

namespace evo {

enum class GenebCasePolicy : std::uint8_t { kPreserve, kUpper };

enum class GenebInvalidBasePolicy : std::uint8_t {
  kTokenizerDefined,
  kBiofmTokenizerDefined,
  kNonAcgtToN,
  kUnknownToN,
  kZeroVector,
  kReject,
};

enum class GenebTrimSide : std::uint8_t { kLeft, kRight };
enum class GenebCropKeep : std::uint8_t { kLeft, kRight, kCenter };
enum class GenebPaddingSide : std::uint8_t { kLeft, kRight, kBoth };
enum class GenebPaddingValue : std::uint8_t { kN, kTokenizerPad };
enum class GenebPaddingBalance : std::uint8_t {
  kNotApplicable,
  kExtraLeft,
  kExtraRight,
};

enum class GenebSpecialTokenPolicy : std::uint8_t {
  kNone,
  kTokenizerDefault,
  kTokenizerDefaultThenOneHotFloat,
  kTokenizerDefaultWithEosAsPad,
  kPrefixOnly,
  kManualBosPlusTokenizerDefault,
  kManualClsThenSepBeforeSequence,
  kGeneModalityDefault,
};

// The side names the side removed when the token count exceeds the limit.
enum class GenebTokenTruncation : std::uint8_t {
  kNone,
  kLeft,
  kRight,
  kModelPreset,
};

enum class GenebContextUnit : std::uint8_t { kBases, kTokens };
enum class GenebLengthPolicy : std::uint8_t {
  kReject,
  kTokenizerTruncate,
  kModelPreset,
};

struct GenebInputTransformSpec final {
  // A transport safety limit, deliberately independent from model context.
  // Readers must enforce it before appending a chunk to a record buffer.
  std::size_t raw_safety_cap{0};
  bool strip_ascii_whitespace{false};
  GenebCasePolicy case_policy{GenebCasePolicy::kPreserve};
  bool u_to_t{false};
  GenebInvalidBasePolicy invalid_base_policy{
      GenebInvalidBasePolicy::kTokenizerDefined};

  bool frame_trim_enabled{false};
  std::size_t frame_multiple{0};
  GenebTrimSide frame_trim_side{GenebTrimSide::kRight};

  bool raw_crop_enabled{false};
  std::size_t raw_crop_length{0};
  GenebCropKeep raw_crop_keep{GenebCropKeep::kLeft};

  bool fixed_padding_enabled{false};
  std::size_t fixed_padding_length{0};
  GenebPaddingSide fixed_padding_side{GenebPaddingSide::kRight};
  GenebPaddingValue fixed_padding_value{GenebPaddingValue::kN};
  GenebPaddingBalance fixed_padding_balance{
      GenebPaddingBalance::kNotApplicable};

  std::string prefix;
  GenebSpecialTokenPolicy special_token_policy{GenebSpecialTokenPolicy::kNone};
  GenebTokenTruncation token_truncation{GenebTokenTruncation::kNone};

  GenebContextUnit context_unit{GenebContextUnit::kTokens};
  GenebLengthPolicy length_policy{GenebLengthPolicy::kModelPreset};
  // Zero denotes an explicitly unknown limit in the pinned manifest.
  std::size_t reference_context_limit{0};
};

// Typed converter input. Strings retain the canonical catalog spelling and
// are compiled once when an artifact is loaded; inference uses only enums.
struct GenebFrameTrimMetadata final {
  bool present{false};
  std::uint64_t multiple{0};
  std::string remove_from;
};

struct GenebRawCropMetadata final {
  bool present{false};
  std::uint64_t length{0};
  std::string keep;
};

struct GenebFixedPaddingMetadata final {
  bool present{false};
  std::uint64_t length{0};
  std::string side;
  std::string value;
  // Empty is the typed representation of catalog null.
  std::string balance;
};

struct GenebInputTransformMetadata final {
  std::uint64_t raw_safety_cap{0};
  bool strip_ascii_whitespace{false};
  std::string case_policy;
  bool u_to_t{false};
  std::string invalid_base_policy;
  GenebFrameTrimMetadata frame_trim;
  GenebRawCropMetadata raw_crop;
  GenebFixedPaddingMetadata fixed_padding;
  // Empty is the typed representation of catalog null.
  std::string prefix;
  std::string special_token_policy;
  std::string token_truncation;
  std::string context_unit;
  std::string length_policy;
  // Zero is permitted only for a catalog field explicitly recorded unknown.
  std::uint64_t reference_context_limit{0};
};

struct GenebInputTransformResult final {
  std::string sequence;
  std::size_t original_length{0};
  std::size_t effective_length{0};
  std::size_t invalid_base_count{0};
  std::size_t trim_left{0};
  std::size_t trim_right{0};
  std::size_t crop_source_offset{0};
  std::size_t crop_left{0};
  std::size_t crop_right{0};
  std::size_t pad_left{0};
  std::size_t pad_right{0};
  std::string prefix;
  GenebSpecialTokenPolicy special_token_policy{GenebSpecialTokenPolicy::kNone};
};

struct GenebTokenLengthPlan final {
  std::size_t original_token_count{0};
  std::size_t prefix_token_count{0};
  std::size_t suffix_token_count{0};
  // Offset of the retained payload within the original decorated token vector.
  std::size_t source_offset{0};
  std::size_t retained_payload_token_count{0};
  std::size_t retained_token_count{0};
  std::size_t truncated_left{0};
  std::size_t truncated_right{0};
  std::size_t pad_left{0};
  std::size_t pad_right{0};
  std::size_t effective_token_count{0};
  bool exceeds_context{false};
  bool deferred_to_model_preset{false};
  std::string prefix;
  GenebSpecialTokenPolicy special_token_policy{GenebSpecialTokenPolicy::kNone};
};

// One-record streaming buffer. Each append is checked before std::string is
// allowed to materialize the incoming chunk. reset() starts the next record.
class GenebBoundedSequenceRecord final {
public:
  explicit GenebBoundedSequenceRecord(std::size_t raw_safety_cap) noexcept
      : raw_safety_cap_(raw_safety_cap) {}

  [[nodiscard]] Status append(std::string_view chunk);
  void reset() noexcept { bytes_.clear(); }
  [[nodiscard]] std::string_view bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::size_t size() const noexcept { return bytes_.size(); }
  [[nodiscard]] std::string take() noexcept;

private:
  std::size_t raw_safety_cap_{0};
  std::string bytes_;
};

[[nodiscard]] Status
check_geneb_raw_sequence_append(std::size_t materialized_bytes,
                                std::size_t incoming_bytes,
                                std::size_t raw_safety_cap);

[[nodiscard]] Status
compile_geneb_input_transform(const GenebInputTransformMetadata &metadata,
                              GenebInputTransformSpec *output);

[[nodiscard]] Status
validate_geneb_input_transform_spec(const GenebInputTransformSpec &spec);

[[nodiscard]] Status transform_geneb_input(std::string_view raw_sequence,
                                           const GenebInputTransformSpec &spec,
                                           GenebInputTransformResult *output);

// Called after tokenization/special-token insertion. The plan deliberately
// does not depend on a tokenizer token-id width or pad-id representation.
[[nodiscard]] Status
plan_geneb_token_length(std::size_t token_count,
                        const GenebInputTransformSpec &spec,
                        GenebTokenLengthPlan *output);

// The boundary counts describe tokenizer post-processor IDs already present in
// token_count. Truncation applies only to the payload between those boundaries.
[[nodiscard]] Status plan_geneb_token_length(
    std::size_t token_count, std::size_t prefix_token_count,
    std::size_t suffix_token_count, const GenebInputTransformSpec &spec,
    GenebTokenLengthPlan *output);

} // namespace evo
