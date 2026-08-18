// SPDX-License-Identifier: Apache-2.0
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

#include "evo/geneb_embedding.hpp"
#include "evo/geneb_input_transform.hpp"
#include "evo/json.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

const evo::JsonValue *required(const evo::JsonValue &object,
                               const std::string_view key,
                               const evo::JsonType type) {
  const auto *value = object.find(key);
  check(value != nullptr,
        std::string{"catalog field exists: "} + std::string{key});
  if (value != nullptr)
    check(value->type == type,
          std::string{"catalog field type: "} + std::string{key});
  return value != nullptr && value->type == type ? value : nullptr;
}

std::uint64_t positive_integer(const evo::JsonValue &object,
                               const std::string_view key) {
  const auto *value = required(object, key, evo::JsonType::kNumber);
  if (value == nullptr || value->number <= 0.0)
    return 0;
  const auto converted = static_cast<std::uint64_t>(value->number);
  check(static_cast<double>(converted) == value->number,
        std::string{"catalog integer is exact: "} + std::string{key});
  return converted;
}

std::string nullable_string(const evo::JsonValue &object,
                            const std::string_view key) {
  const auto *value = object.find(key);
  check(value != nullptr,
        std::string{"catalog field exists: "} + std::string{key});
  if (value == nullptr || value->type == evo::JsonType::kNull)
    return {};
  check(value->type == evo::JsonType::kString,
        std::string{"catalog nullable string type: "} + std::string{key});
  return value->type == evo::JsonType::kString ? value->string : std::string{};
}

evo::GenebInputTransformMetadata
catalog_metadata(const evo::JsonValue &model,
                 const std::uint64_t raw_safety_cap) {
  evo::GenebInputTransformMetadata metadata;
  metadata.raw_safety_cap = raw_safety_cap;
  const auto *context = required(model, "context", evo::JsonType::kObject);
  const auto *transform =
      required(model, "input_transform", evo::JsonType::kObject);
  if (context == nullptr || transform == nullptr)
    return metadata;

  const auto *case_policy =
      required(*transform, "case", evo::JsonType::kString);
  const auto *strip_ascii_whitespace =
      required(*transform, "strip_ascii_whitespace", evo::JsonType::kBoolean);
  const auto *u_to_t = required(*transform, "u_to_t", evo::JsonType::kBoolean);
  const auto *invalid = required(*transform, "invalid", evo::JsonType::kString);
  const auto *special_tokens =
      required(*transform, "special_tokens", evo::JsonType::kString);
  const auto *token_truncation =
      required(*transform, "token_truncation", evo::JsonType::kString);
  const auto *unit = required(*context, "unit", evo::JsonType::kString);
  const auto *length_policy =
      required(*context, "length_policy", evo::JsonType::kString);
  const auto *reference_limit = context->find("reference_max_tokens");
  check(reference_limit != nullptr, "catalog reference context field exists");
  if (case_policy == nullptr || strip_ascii_whitespace == nullptr ||
      u_to_t == nullptr || invalid == nullptr ||
      special_tokens == nullptr || token_truncation == nullptr ||
      unit == nullptr || length_policy == nullptr || reference_limit == nullptr)
    return metadata;

  metadata.case_policy = case_policy->string;
  metadata.strip_ascii_whitespace = strip_ascii_whitespace->boolean;
  metadata.u_to_t = u_to_t->boolean;
  metadata.invalid_base_policy = invalid->string;
  metadata.prefix = nullable_string(*transform, "prefix");
  metadata.special_token_policy = special_tokens->string;
  metadata.token_truncation = token_truncation->string;
  metadata.context_unit = unit->string;
  metadata.length_policy = length_policy->string;
  if (reference_limit->type == evo::JsonType::kNumber) {
    metadata.reference_context_limit =
        positive_integer(*context, "reference_max_tokens");
  } else {
    check(reference_limit->type == evo::JsonType::kNull,
          "unknown reference context is null");
  }

  const auto *frame = transform->find("frame_trim");
  check(frame != nullptr, "catalog frame_trim field exists");
  if (frame != nullptr && frame->type == evo::JsonType::kObject) {
    metadata.frame_trim.present = true;
    metadata.frame_trim.multiple = positive_integer(*frame, "multiple");
    const auto *side = required(*frame, "remove_from", evo::JsonType::kString);
    if (side != nullptr)
      metadata.frame_trim.remove_from = side->string;
  } else if (frame != nullptr) {
    check(frame->type == evo::JsonType::kNull,
          "catalog frame_trim is object/null");
  }

  const auto *crop = transform->find("raw_crop");
  check(crop != nullptr, "catalog raw_crop field exists");
  if (crop != nullptr && crop->type == evo::JsonType::kObject) {
    metadata.raw_crop.present = true;
    metadata.raw_crop.length = positive_integer(*crop, "length");
    const auto *keep = required(*crop, "keep", evo::JsonType::kString);
    if (keep != nullptr)
      metadata.raw_crop.keep = keep->string;
  } else if (crop != nullptr) {
    check(crop->type == evo::JsonType::kNull,
          "catalog raw_crop is object/null");
  }

  const auto *pad = transform->find("fixed_pad");
  check(pad != nullptr, "catalog fixed_pad field exists");
  if (pad != nullptr && pad->type == evo::JsonType::kObject) {
    metadata.fixed_padding.present = true;
    metadata.fixed_padding.length = positive_integer(*pad, "length");
    const auto *side = required(*pad, "side", evo::JsonType::kString);
    const auto *value = required(*pad, "value", evo::JsonType::kString);
    if (side != nullptr)
      metadata.fixed_padding.side = side->string;
    if (value != nullptr)
      metadata.fixed_padding.value = value->string;
    metadata.fixed_padding.balance = nullable_string(*pad, "balance");
  } else if (pad != nullptr) {
    check(pad->type == evo::JsonType::kNull,
          "catalog fixed_pad is object/null");
  }
  return metadata;
}

const evo::JsonValue *find_model(const evo::JsonValue &catalog,
                                 const std::string_view paper_name) {
  const auto *models = required(catalog, "models", evo::JsonType::kArray);
  if (models == nullptr)
    return nullptr;
  for (const auto &model : models->array) {
    const auto *name = model.find("paper_name");
    if (name != nullptr && name->type == evo::JsonType::kString &&
        name->string == paper_name) {
      return &model;
    }
  }
  check(false, std::string{"catalog model exists: "} + std::string{paper_name});
  return nullptr;
}

evo::GenebInputTransformSpec
compile_catalog_model(const evo::JsonValue &catalog,
                      const std::string_view paper_name,
                      const std::uint64_t raw_safety_cap) {
  evo::GenebInputTransformSpec spec;
  const auto *model = find_model(catalog, paper_name);
  if (model == nullptr)
    return spec;
  const auto status = evo::compile_geneb_input_transform(
      catalog_metadata(*model, raw_safety_cap), &spec);
  check(status.ok(), std::string{"compile catalog transform: "} +
                         std::string{paper_name} + ": " + status.message());
  return spec;
}

void test_all_catalog_transforms_compile(const evo::JsonValue &catalog) {
  const auto *models = required(catalog, "models", evo::JsonType::kArray);
  if (models == nullptr)
    return;
  check(models->array.size() == 40,
        "catalog-derived transform table contains exactly 40 models");
  for (const auto &model : models->array) {
    const auto *name = required(model, "paper_name", evo::JsonType::kString);
    evo::GenebInputTransformSpec spec;
    const auto status = evo::compile_geneb_input_transform(
        catalog_metadata(model, 2000000), &spec);
    check(status.ok(),
          std::string{"compile complete catalog transform table: "} +
              (name == nullptr ? "<unknown>" : name->string) + ": " +
              status.message());
  }
}

void test_bounded_record() {
  evo::GenebBoundedSequenceRecord record{6};
  check(record.append("ACG").ok() && record.append("TNN").ok() &&
            record.bytes() == "ACGTNN",
        "streaming record accepts chunks through the exact safety cap");
  const auto overflow = record.append("A");
  check(!overflow.ok() && overflow.code() == evo::ErrorCode::kInvalidArgument &&
            record.bytes() == "ACGTNN",
        "streaming record rejects before materializing an overflow chunk");
  check(record.take() == "ACGTNN" && record.size() == 0,
        "take releases exactly one record");
  check(record.append("TG").ok() && record.bytes() == "TG",
        "record buffer can be reset for the next streaming record");
  check(record.append({}).ok() && record.bytes() == "TG",
        "empty stream chunks are accepted without dereferencing null data");
}

void test_space(const evo::JsonValue &catalog) {
  const auto spec = compile_catalog_model(catalog, "SPACE", 200000);
  std::string raw(131075, 'A');
  raw[0] = 'u';
  raw[1] = '?';
  raw[131074] = 'c';
  evo::GenebInputTransformResult result;
  const auto status = evo::transform_geneb_input(raw, spec, &result);
  check(status.ok(), "SPACE catalog transform succeeds");
  check(result.original_length == 131075 && result.effective_length == 131072 &&
            result.crop_left == 1 && result.crop_right == 2 &&
            result.crop_source_offset == 1,
        "SPACE uses deterministic center crop with odd excess on the right");
  check(result.sequence.front() == 'N' && result.sequence.back() == 'A' &&
            result.invalid_base_count == 1,
        "SPACE applies upper/U-to-T/zero-vector normalization before crop");

  evo::GenebInputTransformResult padded;
  check(evo::transform_geneb_input("acgtu", spec, &padded).ok() &&
            padded.pad_left == 65533 && padded.pad_right == 65534 &&
            padded.sequence.substr(padded.pad_left, 5) == "ACGTT",
        "SPACE center padding assigns an odd extra base to the right");
}

void test_enformer(const evo::JsonValue &catalog) {
  const auto spec = compile_catalog_model(catalog, "Enformer", 300000);
  std::string raw(196610, 'C');
  raw[0] = 'a';
  raw[1] = '?';
  raw[196609] = 'T';
  evo::GenebInputTransformResult cropped;
  check(evo::transform_geneb_input(raw, spec, &cropped).ok() &&
            cropped.sequence.size() == 196608 &&
            cropped.sequence.substr(0, 3) == "ANC" && cropped.crop_left == 0 &&
            cropped.crop_right == 2,
        "Enformer keeps the prefix and normalizes unknown bases to N");

  evo::GenebInputTransformResult padded;
  check(evo::transform_geneb_input("AC", spec, &padded).ok() &&
            padded.sequence.size() == 196608 && padded.pad_left == 0 &&
            padded.pad_right == 196606 && padded.sequence.substr(0, 3) == "ACN",
        "Enformer applies deterministic right N padding");
}

void test_generator(const evo::JsonValue &catalog) {
  const auto spec =
      compile_catalog_model(catalog, "GENERator-Eukaryote-3B", 100000);
  evo::GenebInputTransformResult result;
  check(
      evo::transform_geneb_input("acgtACGT", spec, &result).ok() &&
          result.sequence == "acgtAC" && result.trim_left == 0 &&
          result.trim_right == 2 && result.prefix == "tokenizer-bos-text" &&
          result.special_token_policy ==
              evo::GenebSpecialTokenPolicy::kManualBosPlusTokenizerDefault,
      "GENERator trims the right edge to a multiple of six and hands off BOS");
}

void test_omni_dna(const evo::JsonValue &catalog) {
  const auto spec = compile_catalog_model(catalog, "Omni-DNA-1B", 100000);
  evo::GenebTokenLengthPlan plan;
  check(evo::plan_geneb_token_length(253, spec, &plan).ok() &&
            plan.original_token_count == 253 &&
            plan.retained_token_count == 250 && plan.truncated_left == 0 &&
            plan.truncated_right == 3 && plan.source_offset == 0 &&
            plan.effective_token_count == 250,
        "Omni-DNA applies its catalog 250-token right truncation cap");
}

void test_explicit_reject_only() {
  evo::GenebInputTransformMetadata metadata;
  metadata.raw_safety_cap = 32;
  metadata.case_policy = "preserve";
  metadata.invalid_base_policy = "tokenizer-defined";
  metadata.special_token_policy = "none";
  metadata.token_truncation = "none";
  metadata.context_unit = "tokens";
  metadata.length_policy = "model-preset";
  metadata.reference_context_limit = 4;
  evo::GenebInputTransformSpec spec;
  check(evo::compile_geneb_input_transform(metadata, &spec).ok(),
        "model-preset metadata compiles");
  evo::GenebTokenLengthPlan deferred;
  check(evo::plan_geneb_token_length(5, spec, &deferred).ok() &&
            deferred.exceeds_context && deferred.deferred_to_model_preset,
        "non-reject model preset reports overflow without rejecting");
  spec.length_policy = evo::GenebLengthPolicy::kReject;
  const auto rejected = evo::plan_geneb_token_length(5, spec, &deferred);
  check(!rejected.ok() && rejected.code() == evo::ErrorCode::kInvalidArgument,
        "explicit length reject policy rejects token overflow");

  metadata.case_policy = "locale-upper";
  const auto invalid = evo::compile_geneb_input_transform(metadata, &spec);
  check(!invalid.ok() &&
            invalid.message().find("unsupported value") != std::string::npos,
        "typed manifest compiler fails closed on unknown policy spelling");

  metadata.case_policy = "preserve";
  metadata.length_policy = "tokenizer-truncate";
  const auto inconsistent = evo::compile_geneb_input_transform(metadata, &spec);
  check(!inconsistent.ok() && inconsistent.message().find(
                                  "explicit left/right") != std::string::npos,
        "tokenizer-truncate with no truncation side fails closed at compile");
}

void test_fixed_operation_order() {
  evo::GenebInputTransformMetadata metadata;
  metadata.raw_safety_cap = 16;
  metadata.case_policy = "upper";
  metadata.u_to_t = true;
  metadata.invalid_base_policy = "unknown-to-N";
  metadata.frame_trim = {true, 3, "left"};
  metadata.raw_crop = {true, 4, "right"};
  metadata.fixed_padding = {true, 6, "left", "N", ""};
  metadata.special_token_policy = "none";
  metadata.token_truncation = "left";
  metadata.context_unit = "tokens";
  metadata.length_policy = "tokenizer-truncate";
  metadata.reference_context_limit = 4;
  evo::GenebInputTransformSpec spec;
  check(evo::compile_geneb_input_transform(metadata, &spec).ok(),
        "synthetic ordered transform compiles");
  evo::GenebInputTransformResult result;
  check(evo::transform_geneb_input("u?acgtt", spec, &result).ok() &&
            result.sequence == "NNCGTT" && result.invalid_base_count == 1 &&
            result.trim_left == 1 && result.crop_left == 2 &&
            result.crop_source_offset == 2 && result.pad_left == 2,
        "raw operations execute in case/U/invalid/trim/crop/pad order");
  evo::GenebTokenLengthPlan plan;
  check(evo::plan_geneb_token_length(7, spec, &plan).ok() &&
            plan.source_offset == 3 && plan.truncated_left == 3 &&
            plan.retained_token_count == 4,
        "post-token helper deterministically removes the configured left side");

  evo::GenebInputTransformResult untouched;
  untouched.sequence = "sentinel";
  const auto oversize =
      evo::transform_geneb_input(std::string(17, 'A'), spec, &untouched);
  check(!oversize.ok() && untouched.sequence == "sentinel",
        "raw safety overflow fails before allocating or mutating result state");
}

void test_v62_outer_strip_and_boundary_plan() {
  evo::GenebInputTransformMetadata metadata;
  metadata.raw_safety_cap = 32;
  metadata.strip_ascii_whitespace = true;
  metadata.case_policy = "upper";
  metadata.invalid_base_policy = "non-acgt-to-N";
  metadata.special_token_policy = "tokenizer-default";
  metadata.token_truncation = "right";
  metadata.context_unit = "tokens";
  metadata.length_policy = "tokenizer-truncate";
  metadata.reference_context_limit = 4;
  evo::GenebInputTransformSpec spec;
  check(evo::compile_geneb_input_transform(metadata, &spec).ok(),
        "V62 outer-strip metadata compiles");
  evo::GenebInputTransformResult transformed;
  check(evo::transform_geneb_input(" \tacg?\r\n", spec, &transformed).ok() &&
            transformed.sequence == "ACGN" && transformed.trim_left == 2 &&
            transformed.trim_right == 2 &&
            transformed.invalid_base_count == 1,
        "V62 strips ASCII edges before case and invalid-base replacement");

  evo::GenebTokenLengthPlan plan;
  check(evo::plan_geneb_token_length(6, 1, 1, spec, &plan).ok() &&
            plan.prefix_token_count == 1 && plan.suffix_token_count == 1 &&
            plan.source_offset == 1 && plan.retained_payload_token_count == 2 &&
            plan.truncated_right == 2 && plan.retained_token_count == 4,
        "V62 truncates payload while reserving tokenizer prefix/suffix slots");
}

void test_penultimate_valid_token_pooling() {
  const std::vector<float> hidden{
      0.0F, 1.0F, 10.0F, 11.0F, 20.0F,
      21.0F, 30.0F, 31.0F, 40.0F, 41.0F,
  };
  const std::vector<std::uint8_t> mask{0U, 1U, 1U, 1U, 0U};
  std::vector<float> pooled;
  check(evo::pool_geneb_embedding(hidden, 5U, 2U, mask,
                                  "penultimate-valid-token", &pooled)
                .ok() &&
            pooled == std::vector<float>({20.0F, 21.0F}),
        "B96 selects the visible payload row immediately before SEP");
  const std::vector<std::uint8_t> one_visible{0U, 0U, 1U, 0U, 0U};
  check(!evo::pool_geneb_embedding(hidden, 5U, 2U, one_visible,
                                   "penultimate-valid-token", &pooled)
             .ok(),
        "B96 rejects penultimate-token pooling with fewer than two rows");
}

} // namespace

int main(const int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: test_geneb_input_transform CATALOG\n";
    return 2;
  }
  std::ifstream input{argv[1], std::ios::binary};
  check(static_cast<bool>(input), "GENEB catalog opens");
  const std::string text{std::istreambuf_iterator<char>{input},
                         std::istreambuf_iterator<char>{}};
  evo::JsonValue catalog;
  const auto status = evo::parse_json(text, &catalog);
  check(status.ok() && catalog.type == evo::JsonType::kObject,
        "GENEB catalog parses as an object");
  if (status.ok() && catalog.type == evo::JsonType::kObject) {
    test_all_catalog_transforms_compile(catalog);
    test_space(catalog);
    test_enformer(catalog);
    test_generator(catalog);
    test_omni_dna(catalog);
  }
  test_bounded_record();
  test_explicit_reject_only();
  test_fixed_operation_order();
  test_v62_outer_strip_and_boundary_plan();
  test_penultimate_valid_token_pooling();
  if (failures != 0) {
    std::cerr << failures << " GENEB input-transform test(s) failed\n";
    return 1;
  }
  std::cout << "GENEB input-transform tests passed\n";
  return 0;
}
