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

inline constexpr std::string_view kGenebHyenaDnaArtifactProfile =
    "geneb-hyenadna-runtime-v1";
inline constexpr std::string_view kGenebHyenaDnaRuntimeAbi =
    "geneb-hyenadna-safetensors-v1";
inline constexpr std::string_view kGenebHyenaDnaArchitecture =
    "GenebHyenaDnaDecoder";

struct GenebLongContextStats final {
  std::size_t maximum_fft_transform_length{0};
  std::size_t maximum_fft_radix2_length{0};
  std::uint64_t fft_butterfly_count{0};
  std::size_t peak_fft_complex_values{0};
  // Evo attention is architecturally quadratic in compute, but this counter
  // and peak buffer make the absence of an L^2 score allocation testable.
  std::uint64_t attention_score_pairs{0};
  std::size_t peak_attention_logits{0};
};

struct GenebHyenaDnaTopology final {
  std::size_t vocabulary_size{0};
  std::size_t embedding_rows{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t inner_width{0};
  std::size_t filter_width{0};
  std::size_t positional_width{0};
  std::size_t short_filter_width{0};
  std::size_t maximum_sequence_length{0};
  float norm_epsilon{0.0F};
};

struct GenebHyenaDnaTensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebHyenaDnaNamedTensorView final {
  std::string name;
  GenebHyenaDnaTensorView tensor;
};

struct GenebHyenaDnaTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebHyenaDnaHiddenCapture final {
  // 0=token embedding, 1..layers-1=post-block output, layers=official
  // post-final-LayerNorm hidden_states[-1].
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebHyenaDnaForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebHyenaDnaHiddenCapture> captures;
  std::vector<float> final_hidden;
  std::vector<float> pooled;
  GenebLongContextStats work;
};

[[nodiscard]] Status
validate_geneb_hyenadna_topology(const GenebHyenaDnaTopology &topology);
[[nodiscard]] Status
geneb_hyenadna_topology_from_artifact(const ModelFile &artifact,
                                      GenebHyenaDnaTopology *output);
[[nodiscard]] Status canonical_geneb_hyenadna_tensors(
    const GenebHyenaDnaTopology &topology,
    std::vector<GenebHyenaDnaTensorRequirement> *output);

// Directly exercises the exact-2L FFT primitive used by both long-Hyena
// families.  It exists so the one-million-token complexity gate does not need
// a synthetic 55M-parameter fixture and does not rely on timing.
[[nodiscard]] Status geneb_long_causal_convolution(
    const std::vector<float> &input, const std::vector<float> &kernel,
    std::vector<float> *output, GenebLongContextStats *stats = nullptr);

class GenebHyenaDnaModel final {
public:
  GenebHyenaDnaModel();
  ~GenebHyenaDnaModel();
  GenebHyenaDnaModel(const GenebHyenaDnaModel &) = delete;
  GenebHyenaDnaModel &operator=(const GenebHyenaDnaModel &) = delete;
  GenebHyenaDnaModel(GenebHyenaDnaModel &&) noexcept;
  GenebHyenaDnaModel &operator=(GenebHyenaDnaModel &&) noexcept;

  [[nodiscard]] Status
  load(const GenebHyenaDnaTopology &topology,
       const std::vector<GenebHyenaDnaNamedTensorView> &tensors);
  [[nodiscard]] Status load(const ModelFile &artifact);
  [[nodiscard]] const GenebHyenaDnaTopology *topology() const noexcept;
  [[nodiscard]] const char *kernel_name() const noexcept;

  // The pinned tokenizer uses left padding.  The model deliberately receives
  // no attention mask; the mask is consumed only for GENEB mean pooling.
  [[nodiscard]] Status forward(
      const std::vector<TokenId> &tokens,
      const std::vector<std::uint8_t> &attention_mask,
      const std::vector<std::size_t> &capture_layers,
      GenebHyenaDnaForwardResult *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
