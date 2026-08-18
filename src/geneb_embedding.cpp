// SPDX-License-Identifier: Apache-2.0
#include "evo/geneb_embedding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <utility>

namespace evo {
namespace {

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "GENEB artifact: " + message};
}

std::uint64_t read_u64(const MetadataEntry &entry) noexcept {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < entry.value.size(); ++index)
    value |= static_cast<std::uint64_t>(entry.value[index]) << (index * 8U);
  return value;
}

Status string_value(const ModelFile &artifact, const std::string_view key,
                    std::string *const output, const bool allow_empty = false) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kString)
    return format_error("missing/wrong-typed string '" + std::string{key} + "'");
  std::string value{entry->value.begin(), entry->value.end()};
  if (!allow_empty && value.empty())
    return format_error("empty string '" + std::string{key} + "'");
  *output = std::move(value);
  return Status::Ok();
}

Status bool_value(const ModelFile &artifact, const std::string_view key,
                  bool *const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kBool ||
      entry->value.size() != 1U || entry->value[0] > 1U)
    return format_error("missing/wrong-typed bool '" + std::string{key} + "'");
  *output = entry->value[0] != 0U;
  return Status::Ok();
}

Status u64_value(const ModelFile &artifact, const std::string_view key,
                 std::uint64_t *const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr || entry->type != MetadataType::kU64 ||
      entry->value.size() != sizeof(std::uint64_t))
    return format_error("missing/wrong-typed u64 '" + std::string{key} + "'");
  *output = read_u64(*entry);
  return Status::Ok();
}

bool canonical_sha256(const std::string_view value) noexcept {
  if (value.size() != 64U)
    return false;
  return std::all_of(value.begin(), value.end(), [](const char byte) {
    return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
  });
}

Status optional_transform(const ModelFile &artifact,
                          const std::string_view prefix,
                          GenebFrameTrimMetadata *const output) {
  bool enabled = false;
  auto status = bool_value(artifact, std::string{prefix} + ".enabled", &enabled);
  if (!status.ok())
    return status;
  output->present = enabled;
  if (!enabled)
    return Status::Ok();
  status = u64_value(artifact, std::string{prefix} + ".multiple",
                     &output->multiple);
  if (!status.ok())
    return status;
  return string_value(artifact, std::string{prefix} + ".remove_from",
                      &output->remove_from);
}

Status optional_transform(const ModelFile &artifact,
                          const std::string_view prefix,
                          GenebRawCropMetadata *const output) {
  bool enabled = false;
  auto status = bool_value(artifact, std::string{prefix} + ".enabled", &enabled);
  if (!status.ok())
    return status;
  output->present = enabled;
  if (!enabled)
    return Status::Ok();
  status = u64_value(artifact, std::string{prefix} + ".length", &output->length);
  if (!status.ok())
    return status;
  return string_value(artifact, std::string{prefix} + ".keep", &output->keep);
}

Status optional_transform(const ModelFile &artifact,
                          const std::string_view prefix,
                          GenebFixedPaddingMetadata *const output) {
  bool enabled = false;
  auto status = bool_value(artifact, std::string{prefix} + ".enabled", &enabled);
  if (!status.ok())
    return status;
  output->present = enabled;
  if (!enabled)
    return Status::Ok();
  status = u64_value(artifact, std::string{prefix} + ".length", &output->length);
  if (!status.ok())
    return status;
  status = string_value(artifact, std::string{prefix} + ".side", &output->side);
  if (!status.ok())
    return status;
  status = string_value(artifact, std::string{prefix} + ".value", &output->value);
  if (!status.ok())
    return status;
  return string_value(artifact, std::string{prefix} + ".balance",
                      &output->balance, true);
}

Status preset_value(const ModelFile &artifact, const std::string_view name,
                    GenebEmbeddingPresetSpec *const output) {
  output->name = "geneb-v4-" + std::string{name};
  const std::string prefix = "geneb.preset." + std::string{name} + ".";
  auto status = string_value(artifact, prefix + "hidden_tap", &output->hidden_tap);
  if (!status.ok())
    return status;
  status = string_value(artifact, prefix + "pooling", &output->pooling);
  if (!status.ok())
    return status;
  status = string_value(artifact, prefix + "special_tokens",
                        &output->special_tokens);
  if (!status.ok())
    return status;
  status = string_value(artifact, prefix + "mask_domain", &output->mask_domain);
  if (!status.ok())
    return status;
  std::uint64_t width = 0;
  status = u64_value(artifact, prefix + "output_width", &width);
  if (!status.ok())
    return status;
  if (width == 0 || width > std::numeric_limits<std::size_t>::max())
    return format_error("invalid preset output width");
  output->output_width = static_cast<std::size_t>(width);
  return Status::Ok();
}

Status known_geneb_keys(const ModelFile &artifact) {
  static const std::set<std::string_view> fixed{
      "geneb.schema_version",
      "geneb.suite",
      "geneb.catalog_contract_sha256",
      "geneb.runtime_id",
      "geneb.model_id",
      "geneb.paper_name",
      "geneb.source.kind",
      "geneb.source.immutable",
      "geneb.source.revision",
      "geneb.raw_safety_cap_bytes",
      "geneb.context.unit",
      "geneb.context.length_policy",
      "geneb.context.declared_max_tokens.known",
      "geneb.context.declared_max_tokens",
      "geneb.context.reference_max_tokens.known",
      "geneb.context.reference_max_tokens",
      "geneb.input.case",
      "geneb.input.strip_ascii_whitespace",
      "geneb.input.u_to_t",
      "geneb.input.invalid",
      "geneb.input.prefix",
      "geneb.input.special_tokens",
      "geneb.input.token_truncation",
      "geneb.input.frame_trim.enabled",
      "geneb.input.frame_trim.multiple",
      "geneb.input.frame_trim.remove_from",
      "geneb.input.raw_crop.enabled",
      "geneb.input.raw_crop.length",
      "geneb.input.raw_crop.keep",
      "geneb.input.fixed_pad.enabled",
      "geneb.input.fixed_pad.length",
      "geneb.input.fixed_pad.side",
      "geneb.input.fixed_pad.value",
      "geneb.input.fixed_pad.balance",
      "geneb.provenance.extractor_commit",
      "geneb.provenance.reference_patch_sha256",
      "geneb.provenance.normalization_patch_sha256",
      "geneb.preset.reference.hidden_tap",
      "geneb.preset.reference.pooling",
      "geneb.preset.reference.special_tokens",
      "geneb.preset.reference.mask_domain",
      "geneb.preset.reference.output_width",
      "geneb.preset.normalized.hidden_tap",
      "geneb.preset.normalized.pooling",
      "geneb.preset.normalized.special_tokens",
      "geneb.preset.normalized.mask_domain",
      "geneb.preset.normalized.output_width",
  };
  for (const auto &entry : artifact.metadata()) {
    if (entry.key.rfind("geneb.", 0) == 0 && fixed.count(entry.key) == 0U)
      return format_error("unknown metadata key '" + entry.key + "'");
  }
  return Status::Ok();
}

bool prefix_is_literal(const GenebSpecialTokenPolicy policy) noexcept {
  return policy == GenebSpecialTokenPolicy::kPrefixOnly;
}

bool add_special_tokens(const GenebSpecialTokenPolicy policy) noexcept {
  switch (policy) {
  case GenebSpecialTokenPolicy::kNone:
  case GenebSpecialTokenPolicy::kPrefixOnly:
    return false;
  case GenebSpecialTokenPolicy::kTokenizerDefault:
  case GenebSpecialTokenPolicy::kTokenizerDefaultThenOneHotFloat:
  case GenebSpecialTokenPolicy::kTokenizerDefaultWithEosAsPad:
  case GenebSpecialTokenPolicy::kManualBosPlusTokenizerDefault:
  case GenebSpecialTokenPolicy::kManualClsThenSepBeforeSequence:
  case GenebSpecialTokenPolicy::kGeneModalityDefault:
    return true;
  }
  return false;
}

} // namespace

Status geneb_embedding_spec_from_artifact(
    const ModelFile &artifact, GenebEmbeddingArtifactSpec *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB embedding spec output is null"};
  auto status = known_geneb_keys(artifact);
  if (!status.ok())
    return status;
  std::uint64_t schema = 0;
  status = u64_value(artifact, "geneb.schema_version", &schema);
  if (!status.ok() || schema != 1U)
    return status.ok() ? format_error("unsupported schema version") : status;
  GenebEmbeddingArtifactSpec compiled;
  status = string_value(artifact, "geneb.suite", &compiled.suite);
  if (!status.ok() || compiled.suite != "geneb-v4")
    return status.ok() ? format_error("suite must be geneb-v4") : status;
  status = string_value(artifact, "geneb.runtime_id", &compiled.runtime_id);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.model_id", &compiled.geneb_model_id);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.paper_name", &compiled.paper_name);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.catalog_contract_sha256",
                        &compiled.catalog_contract_sha256);
  if (!status.ok() || !canonical_sha256(compiled.catalog_contract_sha256))
    return status.ok() ? format_error("catalog contract digest is not canonical")
                       : status;

  // Source/provenance evidence is required even though execution does not
  // consume its values. This prevents a runnable artifact from dropping the
  // promotion identity that justifies its GENEB claim.
  std::string evidence;
  bool immutable = false;
  for (const auto key : {"geneb.source.kind", "geneb.source.revision",
                         "geneb.provenance.extractor_commit",
                         "geneb.provenance.reference_patch_sha256",
                         "geneb.provenance.normalization_patch_sha256"}) {
    status = string_value(artifact, key, &evidence,
                          key == std::string_view{"geneb.source.revision"});
    if (!status.ok())
      return status;
    if (std::string_view{key}.find("sha256") != std::string_view::npos &&
        !canonical_sha256(evidence))
      return format_error("noncanonical evidence digest '" +
                          std::string{key} + "'");
  }
  status = bool_value(artifact, "geneb.source.immutable", &immutable);
  if (!status.ok())
    return status;

  GenebInputTransformMetadata metadata;
  status = u64_value(artifact, "geneb.raw_safety_cap_bytes",
                     &metadata.raw_safety_cap);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.input.case", &metadata.case_policy);
  if (!status.ok())
    return status;
  status = bool_value(artifact, "geneb.input.strip_ascii_whitespace",
                      &metadata.strip_ascii_whitespace);
  if (!status.ok())
    return status;
  status = bool_value(artifact, "geneb.input.u_to_t", &metadata.u_to_t);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.input.invalid",
                        &metadata.invalid_base_policy);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.input.prefix", &metadata.prefix, true);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.input.special_tokens",
                        &metadata.special_token_policy);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.input.token_truncation",
                        &metadata.token_truncation);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.context.unit", &metadata.context_unit);
  if (!status.ok())
    return status;
  status = string_value(artifact, "geneb.context.length_policy",
                        &metadata.length_policy);
  if (!status.ok())
    return status;
  bool reference_known = false;
  status = bool_value(artifact, "geneb.context.reference_max_tokens.known",
                      &reference_known);
  if (!status.ok())
    return status;
  status = u64_value(artifact, "geneb.context.reference_max_tokens",
                     &metadata.reference_context_limit);
  if (!status.ok())
    return status;
  if (!reference_known && metadata.reference_context_limit != 0U)
    return format_error("unknown reference limit must be encoded as zero");
  status = optional_transform(artifact, "geneb.input.frame_trim",
                              &metadata.frame_trim);
  if (!status.ok())
    return status;
  status = optional_transform(artifact, "geneb.input.raw_crop",
                              &metadata.raw_crop);
  if (!status.ok())
    return status;
  status = optional_transform(artifact, "geneb.input.fixed_pad",
                              &metadata.fixed_padding);
  if (!status.ok())
    return status;
  status = compile_geneb_input_transform(metadata, &compiled.input_transform);
  if (!status.ok())
    return {ErrorCode::kModelFormat, status.message()};
  status = preset_value(artifact, "reference", &compiled.reference);
  if (!status.ok())
    return status;
  status = preset_value(artifact, "normalized", &compiled.normalized);
  if (!status.ok())
    return status;
  *output = std::move(compiled);
  return Status::Ok();
}

const GenebEmbeddingPresetSpec &geneb_embedding_preset(
    const GenebEmbeddingArtifactSpec &spec,
    const GenebEmbeddingPresetKind preset) noexcept {
  return preset == GenebEmbeddingPresetKind::kReference ? spec.reference
                                                        : spec.normalized;
}

Status prepare_geneb_embedding_input(
    const std::string_view raw_sequence,
    const GenebEmbeddingArtifactSpec &spec,
    const ArtifactTokenizer &tokenizer,
    GenebPreparedEmbeddingInput *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB prepared input output is null"};
  GenebPreparedEmbeddingInput prepared;
  auto status = transform_geneb_input(raw_sequence, spec.input_transform,
                                      &prepared.transform);
  if (!status.ok())
    return status;
  std::string tokenizer_input;
  if (prefix_is_literal(prepared.transform.special_token_policy)) {
    if (prepared.transform.prefix.size() >
        std::numeric_limits<std::size_t>::max() -
            prepared.transform.sequence.size()) {
      return {ErrorCode::kInvalidArgument, "GENEB prefix length overflows"};
    }
    tokenizer_input = prepared.transform.prefix;
    tokenizer_input.append(prepared.transform.sequence);
  }
  const std::string_view input = tokenizer_input.empty()
                                     ? std::string_view{prepared.transform.sequence}
                                     : std::string_view{tokenizer_input};
  TokenizerEncodeOptions options;
  options.add_special_tokens =
      add_special_tokens(prepared.transform.special_token_policy);
  options.raw_byte_limit = spec.input_transform.raw_safety_cap;
  status = tokenizer.encode(input, options, &prepared.tokens);
  if (!status.ok())
    return status;
  std::size_t prefix_token_count =
      options.add_special_tokens ? tokenizer.post_processor_prefix_size() : 0U;
  std::size_t suffix_token_count =
      options.add_special_tokens ? tokenizer.post_processor_suffix_size() : 0U;
  if (prepared.transform.special_token_policy ==
      GenebSpecialTokenPolicy::kManualBosPlusTokenizerDefault) {
    if (prefix_token_count != 1U || suffix_token_count != 0U ||
        prepared.tokens.empty()) {
      return {ErrorCode::kModelFormat,
              "GENEB manual BOS policy requires one tokenizer prefix token "
              "and no suffix token"};
    }
    prepared.tokens.insert(prepared.tokens.begin() + 1,
                           prepared.tokens.front());
    prefix_token_count = 2U;
  }
  if (prepared.transform.special_token_policy ==
      GenebSpecialTokenPolicy::kManualClsThenSepBeforeSequence) {
    if (prefix_token_count != 1U || suffix_token_count != 1U ||
        prepared.tokens.size() < 2U) {
      return {ErrorCode::kModelFormat,
              "GENEB manual CLS-SEP policy requires one tokenizer prefix "
              "and one tokenizer suffix token"};
    }
    const TokenId separator = prepared.tokens.back();
    prepared.tokens.pop_back();
    prepared.tokens.insert(prepared.tokens.begin() + 1, separator);
    prefix_token_count = 2U;
    suffix_token_count = 0U;
  }
  status = plan_geneb_token_length(
      prepared.tokens.size(), prefix_token_count, suffix_token_count,
      spec.input_transform, &prepared.token_plan);
  if (!status.ok())
    return status;
  if (prepared.token_plan.deferred_to_model_preset)
    return {ErrorCode::kUnsupported,
            "GENEB token length requires an unimplemented model preset"};
  std::vector<TokenId> retained;
  retained.reserve(prepared.token_plan.effective_token_count);
  const auto prefix_end = prepared.tokens.begin() + static_cast<std::ptrdiff_t>(
                                                    prefix_token_count);
  retained.insert(retained.end(), prepared.tokens.begin(), prefix_end);
  const auto payload_begin =
      prepared.tokens.begin() + static_cast<std::ptrdiff_t>(
                                    prepared.token_plan.source_offset);
  const auto payload_end = payload_begin + static_cast<std::ptrdiff_t>(
                                             prepared.token_plan
                                                 .retained_payload_token_count);
  retained.insert(retained.end(), payload_begin, payload_end);
  const auto suffix_begin =
      prepared.tokens.end() - static_cast<std::ptrdiff_t>(suffix_token_count);
  retained.insert(retained.end(), suffix_begin, prepared.tokens.end());
  prepared.attention_mask.assign(prepared.token_plan.effective_token_count, 1U);
  if (prepared.token_plan.pad_left != 0 || prepared.token_plan.pad_right != 0) {
    const auto pad = tokenizer.pad_token_id();
    if (!pad.has_value())
      return {ErrorCode::kModelFormat,
              "GENEB tokenizer padding policy has no pad token ID"};
    retained.insert(retained.begin(), prepared.token_plan.pad_left, *pad);
    retained.insert(retained.end(), prepared.token_plan.pad_right, *pad);
    std::fill_n(prepared.attention_mask.begin(), prepared.token_plan.pad_left,
                0U);
    std::fill_n(prepared.attention_mask.end() -
                    static_cast<std::ptrdiff_t>(prepared.token_plan.pad_right),
                prepared.token_plan.pad_right, 0U);
  }
  prepared.tokens = std::move(retained);
  if (prepared.tokens.empty())
    return {ErrorCode::kInvalidArgument,
            "GENEB preprocessing produced an empty token sequence"};
  *output = std::move(prepared);
  return Status::Ok();
}

Status pool_geneb_embedding(
    const std::vector<float> &hidden, const std::size_t rows,
    const std::size_t columns,
    const std::vector<std::uint8_t> &attention_mask,
    const std::string_view pooling, std::vector<float> *const output) {
  if (output == nullptr || rows == 0 || columns == 0 ||
      hidden.size() != rows * columns || attention_mask.size() != rows)
    return {ErrorCode::kInvalidArgument, "GENEB pooling shape is invalid"};
  if (pooling == "model-defined-sequence-pooling") {
    if (rows != 1U)
      return {ErrorCode::kUnsupported,
              "GENEB model-defined pooling requires a runtime-pooled row"};
    *output = hidden;
    return Status::Ok();
  }
  if (pooling == "cls-token") {
    if (attention_mask.front() == 0U)
      return {ErrorCode::kInvalidArgument, "GENEB CLS row is masked"};
    output->assign(hidden.begin(), hidden.begin() +
                                      static_cast<std::ptrdiff_t>(columns));
    return Status::Ok();
  }
  if (pooling == "penultimate-valid-token") {
    std::size_t last = rows;
    std::size_t penultimate = rows;
    for (std::size_t row = 0; row < rows; ++row) {
      if (attention_mask[row] == 0U)
        continue;
      penultimate = last;
      last = row;
    }
    if (penultimate == rows) {
      return {ErrorCode::kInvalidArgument,
              "GENEB penultimate-token pooling requires two visible rows"};
    }
    const auto begin = hidden.begin() +
                       static_cast<std::ptrdiff_t>(penultimate * columns);
    output->assign(begin,
                   begin + static_cast<std::ptrdiff_t>(columns));
    return Status::Ok();
  }
  const bool masked = pooling == "attention-mask-mean";
  const bool all_rows =
      masked || pooling == "unmasked-mean-all-token-rows" ||
      pooling == "mean-first-record-only" || pooling == "per-record-mean" ||
      pooling == "spatial-mean";
  if (!all_rows)
    return {ErrorCode::kUnsupported,
            "unsupported GENEB pooling contract '" + std::string{pooling} + "'"};
  output->assign(columns, 0.0F);
  std::size_t count = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    if (masked && attention_mask[row] == 0U)
      continue;
    ++count;
    for (std::size_t column = 0; column < columns; ++column)
      (*output)[column] += hidden[row * columns + column];
  }
  if (count == 0)
    return {ErrorCode::kInvalidArgument, "GENEB pooling mask is empty"};
  for (float &value : *output)
    value /= static_cast<float>(count);
  return Status::Ok();
}

} // namespace evo
