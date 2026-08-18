// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "evo/model_format.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo::detail {
class LinearExecutor;
}

namespace evo::cpu {

inline constexpr std::string_view kGenebBertArtifactProfile =
    "geneb-bert-runtime-v1";
inline constexpr std::string_view kGenebBertRuntimeAbi =
    "geneb-bert-safetensors-v1";
inline constexpr std::string_view kGenebBertArchitecture = "GenebBertEncoder";

enum class GenebBertPositionEncoding : std::uint8_t {
  kAbsolute,
  // MosaicBERT's bidirectional ALiBi:
  // bias[h,q,k] = -slope[h] * abs(q-k).
  kAlibi,
  // Full-head split-half RoPE as used by pinned MutBERT.
  kRope,
};

enum class GenebBertNormPlacement : std::uint8_t {
  kPre,
  kPost,
};

enum class GenebBertMlpKind : std::uint8_t {
  kGelu,
  // GELU(first half) * second half, followed by the output projection.
  kGatedGelu,
};

enum class GenebBertQkvLayout : std::uint8_t {
  kSeparate,
  // Row-major [Q;K;V] fused projection.
  kFused,
};

enum class GenebBertInputKind : std::uint8_t {
  kTokenIds,
  // Each input row is multiplied by the complete vocabulary embedding table.
  kSoftVocabulary,
};

enum class GenebBertPooling : std::uint8_t {
  kAttentionMaskMean,
  // Raw row zero from last_hidden_state, not the learned BERT pooler.
  kClsToken,
};

struct GenebBertTopology final {
  std::size_t vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t attention_heads{0};
  std::size_t head_dimension{0};
  std::size_t inner_width{0};
  std::size_t maximum_sequence_length{0};
  std::size_t token_type_vocabulary_size{0};
  float layer_norm_epsilon{0.0F};
  float rope_base{0.0F};
  GenebBertPositionEncoding position_encoding{
      GenebBertPositionEncoding::kAbsolute};
  GenebBertNormPlacement norm_placement{GenebBertNormPlacement::kPost};
  GenebBertMlpKind mlp_kind{GenebBertMlpKind::kGelu};
  GenebBertQkvLayout qkv_layout{GenebBertQkvLayout::kSeparate};
  GenebBertInputKind input_kind{GenebBertInputKind::kTokenIds};
  GenebBertPooling pooling{GenebBertPooling::kAttentionMaskMean};
  bool final_layer_norm{false};
  // Pinned MosaicBERT removes masked query rows before every block and pads
  // zeros back after the last layer. Standard BERT leaves masked query rows
  // materialized while excluding them as attention keys.
  bool unpad_masked_tokens{false};
  bool attention_bias{true};
  bool mlp_input_bias{true};
  bool mlp_output_bias{true};
  TensorDType embedding_dtype{TensorDType::kF32};
  TensorDType projection_dtype{TensorDType::kF32};
  TensorDType norm_dtype{TensorDType::kF32};
  TensorDType activation_dtype{TensorDType::kF32};
};

struct GenebBertTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebBertNamedTensorView final {
  std::string name;
  GenebBertTensorView tensor;
};

struct GenebBertTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebBertHiddenCapture final {
  // 0 is the normalized embedding. Public indices 1..layers are successive
  // blocks; when final_layer_norm is true, index layers is post-final-LN.
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebBertForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebBertHiddenCapture> captures;
  std::vector<float> final_hidden;
};

[[nodiscard]] Status
validate_geneb_bert_topology(const GenebBertTopology &topology);

[[nodiscard]] Status
geneb_bert_topology_from_artifact(const ModelFile &artifact,
                                  GenebBertTopology *output);

[[nodiscard]] Status
canonical_geneb_bert_tensors(const GenebBertTopology &topology,
                             std::vector<GenebBertTensorRequirement> *output);

[[nodiscard]] Status
geneb_bert_pool(const GenebBertForwardResult &forward,
                const std::vector<std::uint8_t> &attention_mask,
                GenebBertPooling pooling, std::vector<float> *output);

// Non-owning weights: the tensor byte spans and ModelFile mappings must outlive
// the model and every forward call.
class GenebBertModel final {
public:
  GenebBertModel();
  ~GenebBertModel();
  GenebBertModel(const GenebBertModel &) = delete;
  GenebBertModel &operator=(const GenebBertModel &) = delete;
  GenebBertModel(GenebBertModel &&) noexcept;
  GenebBertModel &operator=(GenebBertModel &&) noexcept;

  [[nodiscard]] Status
  load(const GenebBertTopology &topology,
       const std::vector<GenebBertNamedTensorView> &tensors,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status
  load(const ModelFile &artifact,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status load_artifact(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] const GenebBertTopology *topology() const noexcept;
  [[nodiscard]] const char *linear_executor_name() const noexcept;

  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebBertForwardResult *output) const;

  // soft_vocabulary is row-major [rows,vocabulary_size]. For masked rows the
  // pinned MutBERT transform zeros the row before multiplying by embeddings.
  [[nodiscard]] Status
  forward_soft(const std::vector<float> &soft_vocabulary, std::size_t rows,
               const std::vector<std::uint8_t> &attention_mask,
               const std::vector<std::size_t> &capture_layers,
               GenebBertForwardResult *output) const;

  [[nodiscard]] Status pool(const GenebBertForwardResult &forward,
                            const std::vector<std::uint8_t> &attention_mask,
                            std::vector<float> *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
