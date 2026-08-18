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

inline constexpr std::string_view kGenebOlmoArtifactProfile =
    "geneb-olmo-runtime-v1";
inline constexpr std::string_view kGenebOlmoRuntimeAbi =
    "geneb-olmo-safetensors-v1";
inline constexpr std::string_view kGenebOlmoArchitecture = "GenebOlmoDecoder";
inline constexpr std::string_view kGenebOlmoTorch212AppleArm64LayerNormKernel =
    "torch-2.1.2-apple-arm64-exact-v1";

// The two pinned Omni-DNA checkpoints intentionally use different norm
// topologies.  Keep them as closed variants instead of treating OLMo as a
// Llama-shaped decoder.
enum class GenebOlmoNormType : std::uint8_t {
  kLayerNormNoAffine,
  kRmsNormAffine,
};

enum class GenebOlmoLayerNormKernel : std::uint8_t {
  // Existing portable scalar two-pass mean/variance implementation.
  kPortableTwoPass,
  // Exact F32 reduction/output order of pinned Torch 2.1.2 on Apple arm64.
  kTorch212AppleArm64ExactV1,
};

struct GenebOlmoTopology final {
  std::size_t vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t heads{0};
  // Width of the fused ff_proj output before OLMo's x/gate split.  SwiGLU
  // produces fused_mlp_width/2 features for ff_out.
  std::size_t fused_mlp_width{0};
  std::size_t maximum_sequence_length{0};
  float norm_epsilon{0.0F};
  float rope_theta{10000.0F};
  GenebOlmoNormType norm_type{GenebOlmoNormType::kLayerNormNoAffine};
  GenebOlmoLayerNormKernel layer_norm_kernel{
      GenebOlmoLayerNormKernel::kPortableTwoPass};
};

struct GenebOlmoTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebOlmoNamedTensorView final {
  std::string name;
  GenebOlmoTensorView tensor;
};

struct GenebOlmoTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebOlmoHiddenCapture final {
  // 0 is token embedding; i in [1,layers) is block i-1 output; layers is the
  // official post-final-norm representation returned as hidden_states[-1].
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebOlmoForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebOlmoHiddenCapture> captures;
  std::vector<float> final_hidden;
  // Attention-mask mean of final_hidden, matching GENEB OmniDNAExtractor.
  std::vector<float> pooled;
};

[[nodiscard]] Status
validate_geneb_olmo_topology(const GenebOlmoTopology &topology);

// Compile and cross-check the strict common config.*, runtime.*, and OLMo ABI
// metadata from a canonical runtime artifact.
[[nodiscard]] Status
geneb_olmo_topology_from_artifact(const ModelFile &artifact,
                                  GenebOlmoTopology *output);

// Exact canonical tensor set.  The non-affine LayerNorm topology has no norm
// tensors; the affine RMSNorm topology requires two scales per block and one
// final scale.
[[nodiscard]] Status
canonical_geneb_olmo_tensors(const GenebOlmoTopology &topology,
                             std::vector<GenebOlmoTensorRequirement> *output);

// Parameterless LayerNorm primitive used by the OLMo forward path.  The exact
// kernel is admitted only for width=2048, epsilon=1e-5 and on Apple arm64;
// other hosts return kUnsupported without executing it.
[[nodiscard]] Status geneb_olmo_layer_norm_no_affine(
    const std::vector<float> &input, std::size_t rows, std::size_t width,
    float epsilon, GenebOlmoLayerNormKernel kernel, std::vector<float> *output);

// Non-owning weights: tensor spans and the ModelFile backing them must outlive
// this model and all forward calls.  Forward state is local to each call.
class GenebOlmoModel final {
public:
  GenebOlmoModel();
  ~GenebOlmoModel();
  GenebOlmoModel(const GenebOlmoModel &) = delete;
  GenebOlmoModel &operator=(const GenebOlmoModel &) = delete;
  GenebOlmoModel(GenebOlmoModel &&) noexcept;
  GenebOlmoModel &operator=(GenebOlmoModel &&) noexcept;

  [[nodiscard]] Status
  load(const GenebOlmoTopology &topology,
       const std::vector<GenebOlmoNamedTensorView> &tensors,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status
  load(const ModelFile &artifact,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] const GenebOlmoTopology *topology() const noexcept;
  [[nodiscard]] const char *linear_executor_name() const noexcept;

  // attention_mask must be a nonempty right-padded binary mask with at least
  // one valid token. capture_layers preserves caller order and rejects
  // duplicates. The public final layer index is topology.layers.
  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebOlmoForwardResult *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
