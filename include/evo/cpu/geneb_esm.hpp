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

inline constexpr std::string_view kGenebEsmArtifactProfile =
    "geneb-esm-runtime-v1";
inline constexpr std::string_view kGenebEsmRuntimeAbi =
    "geneb-esm-safetensors-v1";
inline constexpr std::string_view kGenebEsmArchitecture = "GenebEsmEncoder";
inline constexpr std::string_view kGenebEsmTorch212AppleArm64LayerNormKernel =
    "torch-2.1.2-apple-arm64-exact-v1";

enum class GenebEsmPositionType : std::uint8_t {
  kAbsolute,
  kRotary,
};

enum class GenebEsmMlpActivation : std::uint8_t {
  // Original ESM exact-erf GELU.
  kGelu,
  // NT-v2 fused [2*inner,width] projection, split in half, then
  // SiLU(first) * second.
  kSwiGlu,
};

enum class GenebEsmLayerNormKernel : std::uint8_t {
  // Existing portable scalar two-pass mean/variance implementation.
  kPortableTwoPass,
  // Exact affine F32 reduction/output order of pinned Torch 2.1.2 on Apple
  // arm64. The artifact loader admits this only for Agro-NT-1B.
  kTorch212AppleArm64ExactV1,
};

struct GenebEsmTopology final {
  std::size_t vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t heads{0};
  std::size_t head_dimension{0};
  std::size_t inner_width{0};
  // Final token tensor length accepted by the pinned tokenizer/extractor.
  std::size_t maximum_sequence_length{0};
  // Source config.max_position_embeddings. Absolute ESM positions use
  // padding-cumsum IDs beginning at pad_token_id+1.
  std::size_t position_embedding_count{0};
  float layer_norm_epsilon{0.0F};
  float rope_base{10000.0F};
  GenebEsmPositionType position_type{GenebEsmPositionType::kAbsolute};
  GenebEsmMlpActivation mlp_activation{GenebEsmMlpActivation::kGelu};
  GenebEsmLayerNormKernel layer_norm_kernel{
      GenebEsmLayerNormKernel::kPortableTwoPass};
  bool attention_bias{true};
  bool feed_forward_bias{true};
  bool token_dropout{false};
  std::size_t pad_token_id{1};
  std::size_t mask_token_id{2};
  std::size_t cls_token_id{3};
};

struct GenebEsmTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebEsmNamedTensorView final {
  std::string name;
  GenebEsmTensorView tensor;
};

struct GenebEsmTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebEsmHiddenCapture final {
  // 0 is the masked token/position embedding output; i in [1,layers) is
  // block i-1 output; layers is hidden_states[-1] after final LayerNorm.
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebEsmForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebEsmHiddenCapture> captures;
  std::vector<float> final_hidden;
  // Attention-mask mean of final_hidden. Pinned tokenizers add CLS only, so
  // CLS participates and right-padding rows do not.
  std::vector<float> pooled;
};

[[nodiscard]] Status
validate_geneb_esm_topology(const GenebEsmTopology &topology);

// Strictly compiles common config.*, runtime.*, and esm.* metadata. Unknown
// esm.* keys, missing/wrong-typed fields, and common/topology disagreement are
// rejected before any tensor becomes visible.
[[nodiscard]] Status geneb_esm_topology_from_artifact(const ModelFile &artifact,
                                                      GenebEsmTopology *output);

// Exact runtime tensor set. NT-v2 source position_embeddings and per-layer
// rotary inv_freq buffers are converter-validated but deliberately omitted
// because HF rotary forward does not read the learned position table.
[[nodiscard]] Status
canonical_geneb_esm_tensors(const GenebEsmTopology &topology,
                            std::vector<GenebEsmTensorRequirement> *output);

[[nodiscard]] Status geneb_esm_layer_norm(
    const std::vector<float> &input, std::size_t rows, std::size_t width,
    const GenebEsmTensorView &scale, const GenebEsmTensorView &bias,
    float epsilon, std::vector<float> *output,
    GenebEsmLayerNormKernel kernel = GenebEsmLayerNormKernel::kPortableTwoPass);

// In-place HF ESM split-half RoPE. Query has already been scaled by
// head_dimension^-0.5, matching the pinned implementation.
[[nodiscard]] Status geneb_esm_apply_rotary(std::vector<float> *query,
                                            std::vector<float> *key,
                                            std::size_t rows, std::size_t heads,
                                            std::size_t head_dimension,
                                            float rope_base);

// Full bidirectional MHA with a key-padding mask. Query rows, including pad
// rows, are evaluated; only masked key/value source rows are excluded.
[[nodiscard]] Status geneb_esm_bidirectional_attention(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value,
    const std::vector<std::uint8_t> &attention_mask, std::size_t rows,
    std::size_t heads, std::size_t head_dimension, std::vector<float> *output);

// Non-owning weights. Tensor payloads and ModelFile mappings must outlive the
// model and all calls. Forward state and returned captures are call-local.
class GenebEsmModel final {
public:
  GenebEsmModel();
  ~GenebEsmModel();
  GenebEsmModel(const GenebEsmModel &) = delete;
  GenebEsmModel &operator=(const GenebEsmModel &) = delete;
  GenebEsmModel(GenebEsmModel &&) noexcept;
  GenebEsmModel &operator=(GenebEsmModel &&) noexcept;

  [[nodiscard]] Status
  load(const GenebEsmTopology &topology,
       const std::vector<GenebEsmNamedTensorView> &tensors,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status
  load(const ModelFile &artifact,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status load_artifact(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] const GenebEsmTopology *topology() const noexcept;
  [[nodiscard]] const char *linear_executor_name() const noexcept;

  // attention_mask must be nonempty, binary, right padded, and consistent
  // with pad_token_id. At least one valid token is required. Capture order is
  // caller order and duplicate/out-of-range layers are rejected.
  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebEsmForwardResult *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
