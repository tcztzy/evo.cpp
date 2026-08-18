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

inline constexpr std::string_view kGenebCustomEncoderArtifactProfile =
    "geneb-custom-encoder-runtime-v1";
inline constexpr std::string_view kGenebCustomEncoderRuntimeAbi =
    "geneb-custom-encoder-safetensors-v1";
inline constexpr std::string_view kGenebCustomEncoderArchitecture =
    "GenebCustomEncoder";

enum class GenebCustomEncoderVariant : std::uint8_t {
  kLucaOne,
  kGenomicsFm,
};

enum class GenebCustomPositionEncoding : std::uint8_t {
  kAbsolute,
  kRopeSplitHalf,
};

enum class GenebCustomNormPlacement : std::uint8_t {
  kPre,
  kPost,
};

enum class GenebCustomQkvLayout : std::uint8_t {
  kSeparate,
  kFused,
};

enum class GenebCustomMlpKind : std::uint8_t {
  kGelu,
  kGatedGelu,
};

enum class GenebCustomPooling : std::uint8_t {
  kAttentionMaskMean,
  kClsToken,
};

struct GenebCustomEncoderTopology final {
  GenebCustomEncoderVariant variant{GenebCustomEncoderVariant::kLucaOne};
  std::size_t vocabulary_size{0};
  std::size_t tokenizer_vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t attention_heads{0};
  std::size_t head_dimension{0};
  std::size_t inner_width{0};
  std::size_t maximum_sequence_length{0};
  std::size_t token_type_vocabulary_size{0};
  std::size_t pad_token_id{0};
  std::size_t cls_token_id{0};
  std::size_t sep_token_id{0};
  float layer_norm_epsilon{0.0F};
  float rope_base{0.0F};
  GenebCustomPositionEncoding position_encoding{
      GenebCustomPositionEncoding::kRopeSplitHalf};
  GenebCustomNormPlacement norm_placement{GenebCustomNormPlacement::kPre};
  GenebCustomQkvLayout qkv_layout{GenebCustomQkvLayout::kSeparate};
  GenebCustomMlpKind mlp_kind{GenebCustomMlpKind::kGelu};
  GenebCustomPooling pooling{GenebCustomPooling::kAttentionMaskMean};
  bool attention_bias{true};
  bool mlp_input_bias{true};
  bool mlp_output_bias{true};
  bool embedding_layer_norm{false};
  bool final_layer_norm{true};
  bool unpad_masked_tokens{false};
  bool token_type_embeddings{true};
  TensorDType weight_dtype{TensorDType::kF32};
};

struct GenebCustomEncoderTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebCustomEncoderNamedTensorView final {
  std::string name;
  GenebCustomEncoderTensorView tensor;
};

struct GenebCustomEncoderTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebCustomEncoderHiddenCapture final {
  // 0 is the embedding output. 1..layers are successive block outputs;
  // LucaOne's layer==layers capture includes its official final LayerNorm.
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebCustomEncoderForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebCustomEncoderHiddenCapture> captures;
  std::vector<float> final_hidden;
  std::vector<float> pooled;
};

[[nodiscard]] Status validate_geneb_custom_encoder_topology(
    const GenebCustomEncoderTopology &topology);

[[nodiscard]] Status
geneb_custom_encoder_topology_from_artifact(const ModelFile &artifact,
                                            GenebCustomEncoderTopology *output);

[[nodiscard]] Status canonical_geneb_custom_encoder_tensors(
    const GenebCustomEncoderTopology &topology,
    std::vector<GenebCustomEncoderTensorRequirement> *output);

[[nodiscard]] Status
geneb_custom_encoder_pool(const GenebCustomEncoderForwardResult &forward,
                          const std::vector<std::uint8_t> &attention_mask,
                          GenebCustomPooling pooling,
                          std::vector<float> *output);

// Non-owning weights. Tensor byte spans and their ModelFile/storage owner must
// outlive the model and every forward call. Forward state is call-local.
class GenebCustomEncoderModel final {
public:
  GenebCustomEncoderModel();
  ~GenebCustomEncoderModel();
  GenebCustomEncoderModel(const GenebCustomEncoderModel &) = delete;
  GenebCustomEncoderModel &operator=(const GenebCustomEncoderModel &) = delete;
  GenebCustomEncoderModel(GenebCustomEncoderModel &&) noexcept;
  GenebCustomEncoderModel &operator=(GenebCustomEncoderModel &&) noexcept;

  [[nodiscard]] Status
  load(const GenebCustomEncoderTopology &topology,
       const std::vector<GenebCustomEncoderNamedTensorView> &tensors,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status
  load(const ModelFile &artifact,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status load_artifact(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] const GenebCustomEncoderTopology *topology() const noexcept;
  [[nodiscard]] const char *linear_executor_name() const noexcept;

  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebCustomEncoderForwardResult *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
