// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "evo/cpu/mamba_primitives.hpp"
#include "evo/model_format.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo::detail {
class LinearExecutor;
}

namespace evo::cpu {

inline constexpr std::string_view kGenebJanusDnaArtifactProfile =
    "geneb-janusdna-runtime-v1";
inline constexpr std::string_view kGenebJanusDnaRuntimeAbi =
    "geneb-janusdna-lightning-v1";
inline constexpr std::string_view kGenebJanusDnaArchitecture =
    "GenebJanusDnaEncoder";

enum class GenebJanusDnaVariant : std::uint8_t {
  kWithMiddleAttention,
  kWithoutMiddleAttention,
};

// The public topology is deliberately closed to the two production variants
// and one tiny oracle geometry. Operation choices which are constants in the
// pinned source (pre-RMSNorm, biasless projections, top-2 softmax routing,
// separate directions, final FlexAttention/fusion) are not configurable.
struct GenebJanusDnaTopology final {
  GenebJanusDnaVariant variant{GenebJanusDnaVariant::kWithMiddleAttention};
  std::size_t vocabulary_size{0};
  std::size_t tokenizer_vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t attention_heads{0};
  std::size_t head_dimension{0};
  std::size_t flex_attention_head_dimension{0};
  std::size_t inner_width{0};
  std::size_t state_width{0};
  std::size_t convolution_width{0};
  std::size_t time_step_rank{0};
  std::size_t mlp_width{0};
  std::size_t experts{0};
  std::size_t experts_per_token{0};
  std::size_t maximum_sequence_length{0};
  std::size_t middle_attention_layer{0};
  std::size_t pad_token_id{0};
  float norm_epsilon{1.0e-6F};
  TensorDType weight_dtype{TensorDType::kF32};
};

struct GenebJanusDnaNamedTensorView final {
  std::string name;
  MambaTensorView tensor;
};

struct GenebJanusDnaTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebJanusDnaHiddenCapture final {
  // The pinned wrapper exposes no intermediate hidden states. Capture 0 is the
  // original (not doubled) token embedding and capture==layers is the exact
  // twice-post-final-RMSNorm hidden tap.
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebJanusDnaForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebJanusDnaHiddenCapture> captures;
  std::vector<float> final_hidden;
  std::vector<float> pooled;
};

[[nodiscard]] Status
validate_geneb_janusdna_topology(const GenebJanusDnaTopology &topology);

[[nodiscard]] Status
geneb_janusdna_topology_from_artifact(const ModelFile &artifact,
                                      GenebJanusDnaTopology *output);

[[nodiscard]] Status canonical_geneb_janusdna_tensors(
    const GenebJanusDnaTopology &topology,
    std::vector<GenebJanusDnaTensorRequirement> *output);

[[nodiscard]] Status
geneb_janusdna_pool(const GenebJanusDnaForwardResult &forward,
                    const std::vector<std::uint8_t> &attention_mask,
                    std::vector<float> *output);

// Non-owning weights. The named tensor byte spans and their ModelFile/storage
// owner must outlive this model and every forward call. Mutable scan/attention
// state is local to each call.
class GenebJanusDnaModel final {
public:
  GenebJanusDnaModel();
  ~GenebJanusDnaModel();
  GenebJanusDnaModel(const GenebJanusDnaModel &) = delete;
  GenebJanusDnaModel &operator=(const GenebJanusDnaModel &) = delete;
  GenebJanusDnaModel(GenebJanusDnaModel &&) noexcept;
  GenebJanusDnaModel &operator=(GenebJanusDnaModel &&) noexcept;

  [[nodiscard]] Status
  load(const GenebJanusDnaTopology &topology,
       const std::vector<GenebJanusDnaNamedTensorView> &tensors,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status
  load(const ModelFile &artifact,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status load_artifact(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] const GenebJanusDnaTopology *topology() const noexcept;
  [[nodiscard]] const char *linear_executor_name() const noexcept;

  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebJanusDnaForwardResult *output) const;

  [[nodiscard]] Status pool(const GenebJanusDnaForwardResult &forward,
                            const std::vector<std::uint8_t> &attention_mask,
                            std::vector<float> *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
