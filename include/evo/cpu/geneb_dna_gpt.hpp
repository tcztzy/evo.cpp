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

inline constexpr std::string_view kGenebDnaGptArtifactProfile =
    "geneb-dna-gpt-runtime-v1";
inline constexpr std::string_view kGenebDnaGptRuntimeAbi =
    "geneb-dna-gpt-torch-pth-v1";
inline constexpr std::string_view kGenebDnaGptArchitecture =
    "GenebDnaGptDecoder";

struct GenebDnaGptTopology final {
  std::size_t vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t heads{0};
  std::size_t inner_width{0};
  std::size_t maximum_sequence_length{0};
  float norm_epsilon{0.0F};
};

struct GenebDnaGptTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebDnaGptNamedTensorView final {
  std::string name;
  GenebDnaGptTensorView tensor;
};

struct GenebDnaGptTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebDnaGptHiddenCapture final {
  // 0 is post token+absolute-position embedding. topology.layers is the
  // vendored DNAGPT forward's post-final-LayerNorm `x` tensor.
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebDnaGptForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebDnaGptHiddenCapture> captures;
  std::vector<float> final_hidden;
  // Mean over non-pad rows. The mandatory <R> prefix row is included.
  std::vector<float> pooled;
};

[[nodiscard]] Status
validate_geneb_dna_gpt_topology(const GenebDnaGptTopology &topology);

[[nodiscard]] Status
geneb_dna_gpt_topology_from_artifact(const ModelFile &artifact,
                                     GenebDnaGptTopology *output);

// Exact canonical backbone set. Official nn.Linear matrices use [out,in], all
// linears are bias-free, and LayerNorm is affine weight-only. Source task
// heads are converter-validated and deliberately omitted from this artifact.
[[nodiscard]] Status canonical_geneb_dna_gpt_tensors(
    const GenebDnaGptTopology &topology,
    std::vector<GenebDnaGptTensorRequirement> *output);

class GenebDnaGptModel final {
public:
  GenebDnaGptModel();
  ~GenebDnaGptModel();
  GenebDnaGptModel(const GenebDnaGptModel &) = delete;
  GenebDnaGptModel &operator=(const GenebDnaGptModel &) = delete;
  GenebDnaGptModel(GenebDnaGptModel &&) noexcept;
  GenebDnaGptModel &operator=(GenebDnaGptModel &&) noexcept;

  [[nodiscard]] Status
  load(const GenebDnaGptTopology &topology,
       const std::vector<GenebDnaGptNamedTensorView> &tensors);
  [[nodiscard]] Status load(const ModelFile &artifact);

  [[nodiscard]] const GenebDnaGptTopology *topology() const noexcept;
  [[nodiscard]] const char *kernel_name() const noexcept;

  // The pinned model does not receive a padding attention mask. It is causal,
  // so right-padding cannot alter valid rows; the mask is applied only to the
  // final GENEB mean pool and must be binary/right-padded.
  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &nonpad_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebDnaGptForwardResult *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
