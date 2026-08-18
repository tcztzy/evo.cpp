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

namespace evo::detail {
class LinearExecutor;
}

namespace evo::cpu {

inline constexpr std::string_view kGenebSequenceCnnArtifactProfile =
    "geneb-sequence-cnn-runtime-v1";
inline constexpr std::string_view kGenebSequenceCnnRuntimeAbi =
    "geneb-sequence-cnn-pytorch-v1";
inline constexpr std::string_view kGenebSequenceCnnArchitecture =
    "GenebSequenceCnnEncoder";

enum class GenebSequenceCnnVariant : std::uint8_t {
  kEnformer,
  kSpace,
};

struct GenebSequenceCnnTopology final {
  GenebSequenceCnnVariant variant{GenebSequenceCnnVariant::kEnformer};
  std::size_t input_length{0};
  std::size_t stem_width{0};
  // One entry per post-stem tower block. Production Enformer/SPACE have six.
  std::vector<std::size_t> tower_widths;
  std::size_t width{0};
  std::size_t output_width{0};
  std::size_t layers{0};
  std::size_t attention_heads{0};
  std::size_t key_dimension{0};
  std::size_t value_dimension{0};
  std::size_t relative_feature_width{0};
  std::size_t target_length{0};
  float batch_norm_epsilon{0.0F};
  float gelu_sigmoid_scale{0.0F};
  bool use_tf_gamma{false};
  std::size_t species_num_experts{0};
  std::size_t top_k{0};
  float gate_negative_slope{0.0F};
  std::string species;
  TensorDType weight_dtype{TensorDType::kF32};
};

struct GenebSequenceCnnTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebSequenceCnnNamedTensorView final {
  std::string name;
  GenebSequenceCnnTensorView tensor;
};

struct GenebSequenceCnnTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebSequenceCnnForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<float> final_hidden;
};

[[nodiscard]] Status validate_geneb_sequence_cnn_topology(
    const GenebSequenceCnnTopology &topology);

[[nodiscard]] Status geneb_sequence_cnn_topology_from_artifact(
    const ModelFile &artifact, GenebSequenceCnnTopology *output);

[[nodiscard]] Status canonical_geneb_sequence_cnn_tensors(
    const GenebSequenceCnnTopology &topology,
    std::vector<GenebSequenceCnnTensorRequirement> *output);

[[nodiscard]] Status geneb_sequence_cnn_pool(
    const GenebSequenceCnnForwardResult &forward, std::vector<float> *output);

// Non-owning weights: tensor spans and the ModelFile mapping must outlive the
// model and all forward calls.
class GenebSequenceCnnModel final {
public:
  GenebSequenceCnnModel();
  ~GenebSequenceCnnModel();
  GenebSequenceCnnModel(const GenebSequenceCnnModel &) = delete;
  GenebSequenceCnnModel &operator=(const GenebSequenceCnnModel &) = delete;
  GenebSequenceCnnModel(GenebSequenceCnnModel &&) noexcept;
  GenebSequenceCnnModel &operator=(GenebSequenceCnnModel &&) noexcept;

  [[nodiscard]] Status load(
      const GenebSequenceCnnTopology &topology,
      const std::vector<GenebSequenceCnnNamedTensorView> &tensors,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});
  [[nodiscard]] Status load(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});
  [[nodiscard]] Status load_artifact(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] const GenebSequenceCnnTopology *topology() const noexcept;
  [[nodiscard]] const char *linear_executor_name() const noexcept;

  // Applies the frozen uppercase/crop/N-pad/one-hot input transform.
  [[nodiscard]] Status forward(std::string_view sequence,
                               GenebSequenceCnnForwardResult *output) const;
  [[nodiscard]] Status pool(const GenebSequenceCnnForwardResult &forward,
                            std::vector<float> *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
