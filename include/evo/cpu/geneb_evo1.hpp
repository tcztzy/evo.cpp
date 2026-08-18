// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "evo/cpu/geneb_hyenadna.hpp"
#include "evo/model_format.hpp"
#include "evo/status.hpp"

namespace evo::cpu {

inline constexpr std::string_view kGenebEvo1ArtifactProfile =
    "geneb-evo1-runtime-v1";
inline constexpr std::string_view kGenebEvo1RuntimeAbi =
    "geneb-evo1-safetensors-v1";
inline constexpr std::string_view kGenebEvo1Architecture =
    "GenebStripedHyenaV1";

struct GenebEvo1Topology final {
  std::size_t vocabulary_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t heads{0};
  std::size_t inner_width{0};
  std::size_t state_width{0};
  std::size_t short_filter_width{0};
  std::size_t maximum_sequence_length{0};
  float norm_epsilon{0.0F};
  float rope_theta{10000.0F};
  float rope_scaling_factor{1.0F};
  std::vector<std::size_t> attention_layers;
};

struct GenebEvo1TensorView final {
  const std::uint8_t *data{nullptr};
  std::size_t bytes{0};
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebEvo1NamedTensorView final {
  std::string name;
  GenebEvo1TensorView tensor;
};

struct GenebEvo1TensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebEvo1HiddenCapture final {
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebEvo1ForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebEvo1HiddenCapture> captures;
  std::vector<float> final_hidden;
  // Pinned GENEB semantics: unmasked all-row mean of outputs[-1].
  std::vector<float> pooled;
  GenebLongContextStats work;
};

[[nodiscard]] Status
validate_geneb_evo1_topology(const GenebEvo1Topology &topology);
[[nodiscard]] Status geneb_evo1_topology_from_artifact(
    const ModelFile &artifact, GenebEvo1Topology *output);
[[nodiscard]] Status canonical_geneb_evo1_tensors(
    const GenebEvo1Topology &topology,
    std::vector<GenebEvo1TensorRequirement> *output);

class GenebEvo1Model final {
public:
  GenebEvo1Model();
  ~GenebEvo1Model();
  GenebEvo1Model(const GenebEvo1Model &) = delete;
  GenebEvo1Model &operator=(const GenebEvo1Model &) = delete;
  GenebEvo1Model(GenebEvo1Model &&) noexcept;
  GenebEvo1Model &operator=(GenebEvo1Model &&) noexcept;

  [[nodiscard]] Status
  load(const GenebEvo1Topology &topology,
       const std::vector<GenebEvo1NamedTensorView> &tensors);
  [[nodiscard]] Status load(const ModelFile &artifact);
  [[nodiscard]] const GenebEvo1Topology *topology() const noexcept;
  [[nodiscard]] const char *kernel_name() const noexcept;

  [[nodiscard]] Status forward(
      const std::vector<TokenId> &tokens,
      const std::vector<std::size_t> &capture_layers,
      GenebEvo1ForwardResult *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
