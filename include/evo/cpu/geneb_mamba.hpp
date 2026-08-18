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

inline constexpr std::string_view kGenebMambaArtifactProfile =
    "geneb-mamba-runtime-v1";
inline constexpr std::string_view kGenebMambaRuntimeAbi =
    "geneb-mamba-safetensors-v1";
inline constexpr std::string_view kGenebMambaArchitecture = "GenebMambaEncoder";

enum class GenebMambaVariant : std::uint8_t {
  kCaduceusMamba1,
  kEccDnaMamba2,
};

struct GenebMambaTopology final {
  GenebMambaVariant variant{GenebMambaVariant::kCaduceusMamba1};
  std::size_t vocabulary_size{0};
  std::size_t tokenizer_vocabulary_size{0};
  std::size_t width{0};
  std::size_t output_width{0};
  std::size_t layers{0};
  // Zero means that upstream declares no finite model limit. Frontends still
  // enforce the artifact raw safety cap and their explicit token context.
  std::size_t maximum_sequence_length{0};
  // Informational training/evaluation window, never an inference rejection or
  // truncation gate. PlantCaduceus advertises 512 while its hard max is
  // unknown.
  std::size_t advertised_training_sequence_length{0};
  std::size_t inner_width{0};
  std::size_t state_width{0};
  std::size_t convolution_width{0};
  std::size_t time_step_rank{0};
  std::size_t mlp_width{0};
  std::size_t head_width{0};
  std::size_t heads{0};
  std::size_t groups{0};
  float norm_epsilon{1.0e-5F};
  bool reverse_complement_parameter_sharing{false};
  std::vector<TokenId> complement_map;
};

struct GenebMambaNamedTensorView final {
  std::string name;
  MambaTensorView tensor;
};

struct GenebMambaTensorRequirement final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct GenebMambaHiddenCapture final {
  // 0 is token embedding; layers is the pinned post-final-norm/projection tap.
  std::size_t layer{0};
  std::vector<float> values;
};

struct GenebMambaForwardResult final {
  std::size_t rows{0};
  std::size_t width{0};
  std::vector<GenebMambaHiddenCapture> captures;
  std::vector<float> final_hidden;
};

[[nodiscard]] Status
validate_geneb_mamba_topology(const GenebMambaTopology &topology);

[[nodiscard]] Status
geneb_mamba_topology_from_artifact(const ModelFile &artifact,
                                   GenebMambaTopology *output);

[[nodiscard]] Status
canonical_geneb_mamba_tensors(const GenebMambaTopology &topology,
                              std::vector<GenebMambaTensorRequirement> *output);

[[nodiscard]] Status
geneb_mamba_pool(const GenebMambaForwardResult &forward,
                 const std::vector<std::uint8_t> &attention_mask,
                 std::vector<float> *output);

// Non-owning model. Artifact mappings and tensor byte spans must outlive it.
class GenebMambaModel final {
public:
  GenebMambaModel();
  ~GenebMambaModel();
  GenebMambaModel(const GenebMambaModel &) = delete;
  GenebMambaModel &operator=(const GenebMambaModel &) = delete;
  GenebMambaModel(GenebMambaModel &&) noexcept;
  GenebMambaModel &operator=(GenebMambaModel &&) noexcept;

  [[nodiscard]] Status
  load(const GenebMambaTopology &topology,
       const std::vector<GenebMambaNamedTensorView> &tensors,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status
  load(const ModelFile &artifact,
       std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] Status load_artifact(
      const ModelFile &artifact,
      std::shared_ptr<evo::detail::LinearExecutor> linear_executor = {});

  [[nodiscard]] const GenebMambaTopology *topology() const noexcept;
  [[nodiscard]] const char *linear_executor_name() const noexcept;

  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const std::vector<std::uint8_t> &attention_mask,
                               const std::vector<std::size_t> &capture_layers,
                               GenebMambaForwardResult *output) const;

  [[nodiscard]] Status pool(const GenebMambaForwardResult &forward,
                            const std::vector<std::uint8_t> &attention_mask,
                            std::vector<float> *output) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu
