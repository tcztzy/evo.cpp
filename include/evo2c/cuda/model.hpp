// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "evo2c/model_format.hpp"
#include "evo2c/status.hpp"
#include "evo2c/tokenizer.hpp"

namespace evo2c::cuda {

enum class MixerType { kHcs, kHcm, kHcl, kAttention };

enum class LayerDumpPoint {
  kBlockOutput,
  kPreNorm,
  kMixerOutput,
  kMixerResidual,
  kPostNorm,
  kMlpOutput,
};

struct RuntimeModelConfig final {
  std::size_t vocab_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t heads{0};
  std::size_t state_size{0};
  std::size_t inner_width{0};
  std::size_t short_filter_length{0};
  std::size_t hcs_filter_length{0};
  std::size_t hcm_filter_length{0};
  std::size_t hcs_filter_groups{0};
  std::size_t hcm_filter_groups{0};
  float epsilon{0.0F};
  float rope_scale{0.0F};
  bool qkv_head_major{false};
  bool use_fp8_input_projections{false};
  bool test_fixture{false};
  std::vector<MixerType> mixer_types;

  [[nodiscard]] std::size_t head_dim() const noexcept {
    return heads == 0 ? 0 : width / heads;
  }
};

struct LayerDump final {
  std::size_t layer{0};
  std::string path;
  LayerDumpPoint point{LayerDumpPoint::kBlockOutput};
};

struct StageAssignment final {
  int device{-1};
  std::size_t layer_begin{0};
  std::size_t layer_end{0};
  std::size_t weight_bytes{0};
  std::size_t cache_bytes{0};
  std::size_t arena_bytes{0};
};

[[nodiscard]] Status read_runtime_model_config(const ModelFile &model,
                                               bool allow_test_fixture,
                                               RuntimeModelConfig *config);

class SingleGpuModel final {
public:
  SingleGpuModel();
  ~SingleGpuModel();

  SingleGpuModel(const SingleGpuModel &) = delete;
  SingleGpuModel &operator=(const SingleGpuModel &) = delete;
  SingleGpuModel(SingleGpuModel &&) noexcept;
  SingleGpuModel &operator=(SingleGpuModel &&) noexcept;

  [[nodiscard]] Status load(const ModelFile &model, int device,
                            std::size_t context_capacity,
                            bool allow_test_fixture = false);

  [[nodiscard]] Status
  prefill(const std::vector<TokenId> &tokens, std::vector<float> *logits,
          const std::optional<LayerDump> &dump = std::nullopt);
  [[nodiscard]] Status prefill_with_dumps(const std::vector<TokenId> &tokens,
                                          std::vector<float> *logits,
                                          const std::vector<LayerDump> &dumps);
  [[nodiscard]] Status prefill_chunk(const std::vector<TokenId> &tokens,
                                     std::vector<float> *logits);
  [[nodiscard]] Status decode(TokenId token, std::vector<float> *logits);

  [[nodiscard]] const RuntimeModelConfig &config() const noexcept;
  [[nodiscard]] std::size_t position() const noexcept;
  [[nodiscard]] std::size_t activation_capacity() const noexcept;
  [[nodiscard]] bool uses_q8_kv_cache() const noexcept;
  [[nodiscard]] int device() const noexcept;

private:
  friend class PipelineModel;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class PipelineModel final {
public:
  PipelineModel();
  ~PipelineModel();

  PipelineModel(const PipelineModel &) = delete;
  PipelineModel &operator=(const PipelineModel &) = delete;
  PipelineModel(PipelineModel &&) noexcept;
  PipelineModel &operator=(PipelineModel &&) noexcept;

  [[nodiscard]] Status load(const ModelFile &model,
                            const std::vector<int> &devices,
                            std::size_t context_capacity,
                            bool allow_test_fixture = false);

  [[nodiscard]] Status
  prefill(const std::vector<TokenId> &tokens, std::vector<float> *logits,
          const std::optional<LayerDump> &dump = std::nullopt);
  [[nodiscard]] Status prefill_with_dumps(const std::vector<TokenId> &tokens,
                                          std::vector<float> *logits,
                                          const std::vector<LayerDump> &dumps);
  [[nodiscard]] Status prefill_chunk(const std::vector<TokenId> &tokens,
                                     std::vector<float> *logits);
  [[nodiscard]] Status decode(TokenId token, std::vector<float> *logits);

  [[nodiscard]] const RuntimeModelConfig &config() const noexcept;
  [[nodiscard]] const std::vector<StageAssignment> &stages() const noexcept;
  [[nodiscard]] std::size_t position() const noexcept;
  [[nodiscard]] std::size_t activation_capacity() const noexcept;
  [[nodiscard]] bool uses_q8_kv_cache() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo2c::cuda
