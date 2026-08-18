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

namespace evo::cpu {

inline constexpr std::string_view kGenebGpt2ArtifactProfile =
    "geneb-gpt2-runtime-v1";
inline constexpr std::string_view kGenebGpt2RuntimeAbi =
    "geneb-gpt2-safetensors-v1";
inline constexpr std::string_view kGenebGpt2Architecture = "GenebGpt2Decoder";

struct GenebGpt2Topology final {
  std::size_t vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t heads{0};
  std::size_t inner_width{0};
  std::size_t maximum_sequence_length{0};
  float norm_epsilon{0.0F};
};

struct GenebGpt2TensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebGpt2NamedTensorView final {
  std::string name;
  GenebGpt2TensorView tensor;
};

struct GenebGpt2TensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebGpt2HiddenCapture final {
  // 0 is the post token+absolute-position embedding. Intermediate indices are
  // post-block outputs; topology.layers is the official post-final-LayerNorm
  // hidden_states[-1] consumed by the pinned GENEB extractor.
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebGpt2ForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebGpt2HiddenCapture> captures;
  std::vector<float> final_hidden;
  // Mean over attention-mask rows, including no implicit special tokens.
  std::vector<float> pooled;
};

[[nodiscard]] Status
validate_geneb_gpt2_topology(const GenebGpt2Topology &topology);

[[nodiscard]] Status
geneb_gpt2_topology_from_artifact(const ModelFile &artifact,
                                  GenebGpt2Topology *output);

// Exact canonical set. GPT-2 Conv1D matrices retain their official [in,out]
// source orientation; all affine biases and LayerNorm biases are required.
[[nodiscard]] Status
canonical_geneb_gpt2_tensors(const GenebGpt2Topology &topology,
                             std::vector<GenebGpt2TensorRequirement> *output);

// Non-owning tensor spans. The ModelFile or typed fixture backing the spans
// must outlive this model and every forward call.
class GenebGpt2Model final {
public:
  GenebGpt2Model();
  ~GenebGpt2Model();
  GenebGpt2Model(const GenebGpt2Model &) = delete;
  GenebGpt2Model &operator=(const GenebGpt2Model &) = delete;
  GenebGpt2Model(GenebGpt2Model &&) noexcept;
  GenebGpt2Model &operator=(GenebGpt2Model &&) noexcept;

  [[nodiscard]] Status
  load(const GenebGpt2Topology &topology,
       const std::vector<GenebGpt2NamedTensorView> &tensors);
  [[nodiscard]] Status load(const ModelFile &artifact);

  [[nodiscard]] const GenebGpt2Topology *topology() const noexcept;
  [[nodiscard]] const char *kernel_name() const noexcept;

  // The mask must be nonempty, binary, and right padded. GPT-2 applies it to
  // attention keys and the GENEB extractor reuses it for pooling.
  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebGpt2ForwardResult *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
