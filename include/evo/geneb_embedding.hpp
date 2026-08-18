// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "evo/geneb_input_transform.hpp"
#include "evo/model_format.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo {

enum class GenebEmbeddingPresetKind : std::uint8_t {
  kReference,
  kNormalized,
};

struct GenebEmbeddingPresetSpec final {
  std::string name;
  std::string hidden_tap;
  std::string pooling;
  std::string special_tokens;
  std::string mask_domain;
  std::size_t output_width{0};
};

struct GenebEmbeddingArtifactSpec final {
  std::string suite;
  std::string runtime_id;
  std::string geneb_model_id;
  std::string paper_name;
  std::string catalog_contract_sha256;
  GenebInputTransformSpec input_transform;
  GenebEmbeddingPresetSpec reference;
  GenebEmbeddingPresetSpec normalized;
};

struct GenebPreparedEmbeddingInput final {
  std::vector<TokenId> tokens;
  std::vector<std::uint8_t> attention_mask;
  GenebInputTransformResult transform;
  GenebTokenLengthPlan token_plan;
};

[[nodiscard]] Status geneb_embedding_spec_from_artifact(
    const ModelFile &artifact, GenebEmbeddingArtifactSpec *output);

[[nodiscard]] const GenebEmbeddingPresetSpec &geneb_embedding_preset(
    const GenebEmbeddingArtifactSpec &spec,
    GenebEmbeddingPresetKind preset) noexcept;

[[nodiscard]] Status prepare_geneb_embedding_input(
    std::string_view raw_sequence, const GenebEmbeddingArtifactSpec &spec,
    const ArtifactTokenizer &tokenizer, GenebPreparedEmbeddingInput *output);

// Applies a pinned vector pooling contract to a row-major hidden-state matrix.
// The output is exactly one row; model-defined pooling is accepted only when
// the runtime has already reduced the result to one row.
[[nodiscard]] Status pool_geneb_embedding(
    const std::vector<float> &hidden, std::size_t rows, std::size_t columns,
    const std::vector<std::uint8_t> &attention_mask,
    std::string_view pooling, std::vector<float> *output);

} // namespace evo
