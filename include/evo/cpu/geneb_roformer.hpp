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

inline constexpr std::string_view kGenebRoformerArtifactProfile =
    "geneb-roformer-runtime-v1";
inline constexpr std::string_view kGenebRoformerRuntimeAbi =
    "geneb-roformer-pytorch-v1";
inline constexpr std::string_view kGenebRoformerArchitecture =
    "GenebRoformerEncoder";

struct GenebRoformerTopology final {
  std::size_t vocabulary_size{0};
  std::size_t tokenizer_vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t attention_heads{0};
  std::size_t head_dimension{0};
  std::size_t inner_width{0};
  std::size_t maximum_sequence_length{0};
  std::size_t token_type_vocabulary_size{0};
  TokenId pad_token_id{0};
  TokenId cls_token_id{0};
  TokenId sep_token_id{0};
  float layer_norm_epsilon{0.0F};
  float rope_base{0.0F};
  bool rotary_value{false};
  TensorDType weight_dtype{TensorDType::kF32};
};

struct GenebRoformerTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebRoformerNamedTensorView final {
  std::string name;
  GenebRoformerTensorView tensor;
};

struct GenebRoformerTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebRoformerHiddenCapture final {
  // 0 is embedding LayerNorm; 1..layers are successive post-FFN LayerNorms.
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebRoformerForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebRoformerHiddenCapture> captures;
  std::vector<float> final_hidden;
};

[[nodiscard]] Status
validate_geneb_roformer_topology(const GenebRoformerTopology &topology);

[[nodiscard]] Status
geneb_roformer_topology_from_artifact(const ModelFile &artifact,
                                      GenebRoformerTopology *output);

[[nodiscard]] Status canonical_geneb_roformer_tensors(
    const GenebRoformerTopology &topology,
    std::vector<GenebRoformerTensorRequirement> *output);

[[nodiscard]] Status geneb_roformer_pool(
    const GenebRoformerForwardResult &forward,
    const std::vector<std::uint8_t> &attention_mask,
    std::vector<float> *output);

// Non-owning weights: tensor spans and the ModelFile mapping must outlive the
// model and all forward calls.
class GenebRoformerModel final {
public:
  GenebRoformerModel();
  ~GenebRoformerModel();
  GenebRoformerModel(const GenebRoformerModel &) = delete;
  GenebRoformerModel &operator=(const GenebRoformerModel &) = delete;
  GenebRoformerModel(GenebRoformerModel &&) noexcept;
  GenebRoformerModel &operator=(GenebRoformerModel &&) noexcept;

  [[nodiscard]] Status load(
      const GenebRoformerTopology &topology,
      const std::vector<GenebRoformerNamedTensorView> &tensors,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});
  [[nodiscard]] Status load(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});
  [[nodiscard]] Status load_artifact(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] const GenebRoformerTopology *topology() const noexcept;
  [[nodiscard]] const char *linear_executor_name() const noexcept;

  [[nodiscard]] Status forward(
      const std::vector<TokenId> &tokens,
      const std::vector<std::uint8_t> &attention_mask,
      const std::vector<std::size_t> &capture_layers,
      GenebRoformerForwardResult *output) const;

  // GENEB DeepGene constructs [CLS],[SEP],payload (SEP precedes payload).
  [[nodiscard]] Status forward_payload(
      const std::vector<TokenId> &payload,
      const std::vector<std::size_t> &capture_layers,
      GenebRoformerForwardResult *output) const;

  [[nodiscard]] Status pool(const GenebRoformerForwardResult &forward,
                            const std::vector<std::uint8_t> &attention_mask,
                            std::vector<float> *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
