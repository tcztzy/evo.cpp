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

inline constexpr std::string_view kGenebDecoderArtifactProfile =
    "geneb-decoder-runtime-v1";
inline constexpr std::string_view kGenebDecoderRuntimeAbi =
    "geneb-decoder-safetensors-v1";
inline constexpr std::string_view kGenebDecoderArchitecture =
    "GenebTransformerDecoder";

enum class GenebDecoderRmsEpsilonPlacement : std::uint8_t {
  kInsideSqrt,
  kAfterSqrt,
};

enum class GenebDecoderRopeLayout : std::uint8_t {
  // Half pairs: (0,d/2), (1,d/2+1), ... . HF Llama/Mistral rotate_half
  // uses this layout.
  kSplitHalf,
  // Adjacent pairs: (0,1), (2,3), ... . Kept as an explicit artifact choice.
  kAdjacentPairs,
};

enum class GenebDecoderMlpActivation : std::uint8_t {
  kSwiGlu,
  kGelu,
};

enum class GenebDecoderAttentionKernel : std::uint8_t {
  // Explicit eager tensor-operation boundaries used by the original seven
  // Llama/Mistral decoder profiles.
  kEager,
  // Portable forward contract of PyTorch's CPU BF16 Flash-SDPA: F32 QK,
  // scale/max/sum, BF16 exponential buffer, F32 value accumulation, then a
  // BF16 output cast. Backend-specific GEMM/exp reduction order is not part of
  // this portable contract.
  kTorchCpuFlashBf16Portable,
};

enum class GenebDecoderF32MathKernel : std::uint8_t {
  // Existing portable scalar/reference operator boundaries.
  kPortable,
  // Exact Apple-arm64 F32 operator order used by Torch 2.7.1 for OmniNA.
  kTorch271AppleArm64ExactV1,
};

struct GenebDecoderTopology final {
  std::size_t vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t query_heads{0};
  std::size_t key_value_heads{0};
  // Explicit: BioFM uses width=640, query_heads=32, head_dimension=128.
  // Therefore query_heads*head_dimension (4096) need not equal width.
  std::size_t head_dimension{0};
  std::size_t rotary_dimension{0};
  std::size_t inner_width{0};
  std::size_t maximum_sequence_length{0};
  // Zero selects full causal attention. A positive value includes the current
  // token and at most window-1 preceding tokens.
  std::size_t sliding_window{0};
  float rms_epsilon{0.0F};
  GenebDecoderRmsEpsilonPlacement rms_epsilon_placement{
      GenebDecoderRmsEpsilonPlacement::kInsideSqrt};
  float rope_base{10000.0F};
  float rope_position_scale{1.0F};
  GenebDecoderRopeLayout rope_layout{GenebDecoderRopeLayout::kSplitHalf};
  GenebDecoderMlpActivation mlp_activation{GenebDecoderMlpActivation::kSwiGlu};
  GenebDecoderAttentionKernel attention_kernel{
      GenebDecoderAttentionKernel::kEager};
  GenebDecoderF32MathKernel f32_math_kernel{
      GenebDecoderF32MathKernel::kPortable};
  bool attention_bias{false};
  bool mlp_bias{false};
  // These three declarations make mixed F32/BF16 artifacts strict rather
  // than accepting either dtype opportunistically.
  TensorDType embedding_dtype{TensorDType::kF32};
  TensorDType projection_dtype{TensorDType::kF32};
  TensorDType norm_dtype{TensorDType::kF32};
  // Activations use F32 storage in the portable reference runtime. BF16
  // selects explicit round-to-nearest-even casts at the documented eager
  // operation boundaries so the stored values reproduce BF16 semantics.
  TensorDType activation_dtype{TensorDType::kF32};
};

struct GenebDecoderTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebDecoderNamedTensorView final {
  std::string name;
  GenebDecoderTensorView tensor;
};

struct GenebDecoderTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebDecoderHiddenCapture final {
  // 0 is token embedding; i in [1,layers) is block i-1 output; layers is the
  // official post-final-RMSNorm representation.
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebDecoderForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebDecoderHiddenCapture> captures;
  // Official post-final-RMSNorm representation.
  std::vector<float> final_hidden;
};

[[nodiscard]] Status
validate_geneb_decoder_topology(const GenebDecoderTopology &topology);

// Compile the strict runtime topology metadata stored in a canonical artifact.
[[nodiscard]] Status
geneb_decoder_topology_from_artifact(const ModelFile &artifact,
                                     GenebDecoderTopology *output);

// Canonical converter target names and exact shapes/dtypes. The runtime loader
// requires this set exactly: missing, duplicate, extra, wrong-shape, and
// wrong-dtype tensors are rejected before weights become visible. The frozen
// embedding artifact deliberately omits lm_head.weight after converter-side
// source validation, irrespective of source tie_word_embeddings.
[[nodiscard]] Status canonical_geneb_decoder_tensors(
    const GenebDecoderTopology &topology,
    std::vector<GenebDecoderTensorRequirement> *output);

[[nodiscard]] Status
geneb_decoder_rms_norm(const std::vector<float> &input, std::size_t rows,
                       std::size_t width, const GenebDecoderTensorView &scale,
                       float epsilon, GenebDecoderRmsEpsilonPlacement placement,
                       TensorDType activation_dtype,
                       std::vector<float> *output);

// q=[rows,query_heads,head_dimension], k=[rows,key_value_heads,head_dimension].
[[nodiscard]] Status geneb_decoder_apply_rope(
    std::vector<float> *query, std::size_t query_heads, std::vector<float> *key,
    std::size_t key_value_heads, std::size_t rows, std::size_t head_dimension,
    std::size_t rotary_dimension, std::size_t position_offset, float rope_base,
    float position_scale, GenebDecoderRopeLayout layout,
    TensorDType activation_dtype);

// GQA and MHA share this implementation. query_heads must be divisible by
// key_value_heads. A zero sliding_window selects full causal attention.
[[nodiscard]] Status geneb_decoder_causal_attention(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value, std::size_t rows, std::size_t query_heads,
    std::size_t key_value_heads, std::size_t head_dimension,
    std::size_t sliding_window, TensorDType activation_dtype,
    std::vector<float> *output);

// Non-owning weights: every tensor byte span must outlive the model and all
// forward calls. Activations and captured hidden states are owned by results.
class GenebDecoderModel final {
public:
  GenebDecoderModel();
  ~GenebDecoderModel();
  GenebDecoderModel(const GenebDecoderModel &) = delete;
  GenebDecoderModel &operator=(const GenebDecoderModel &) = delete;
  GenebDecoderModel(GenebDecoderModel &&) noexcept;
  GenebDecoderModel &operator=(GenebDecoderModel &&) noexcept;

  [[nodiscard]] Status
  load(const GenebDecoderTopology &topology,
       const std::vector<GenebDecoderNamedTensorView> &tensors,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  // ModelFile and its mappings must outlive this non-owning model. This
  // overload validates profile/ABI/architecture, typed decoder.* metadata,
  // and the same exact canonical tensor set as the typed-view loader.
  [[nodiscard]] Status
  load(const ModelFile &artifact,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  // Source-compatible descriptive alias for the ModelFile overload.
  [[nodiscard]] Status load_artifact(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] const GenebDecoderTopology *topology() const noexcept;
  [[nodiscard]] const char *linear_executor_name() const noexcept;

  // capture_layers preserves caller order and rejects duplicates. The raw last
  // block output is internal; layer topology.layers is post-final-RMSNorm.
  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               std::size_t position_offset,
                               const std::vector<std::size_t> &capture_layers,
                               GenebDecoderForwardResult *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
