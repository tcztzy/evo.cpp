// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "evo/model_format.hpp"
#include "evo/profile.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo::cuda {

enum class MixerType { kHcs, kHcm, kHcl, kAttention };
enum class HyenaProjectionDType { kBF16, kE4M3Sw };
enum class HcmFilterDType { kBF16, kF32 };

enum class LayerDumpPoint {
  kBlockOutput,
  kPreNorm,
  kMixerInputProjection,
  kMixerShortFilter,
  kMixerShortState,
  kMixerInnerState,
  kMixerX2,
  kMixerX1,
  kMixerValue,
  kMixerPregate,
  kMixerState,
  kMixerFilter,
  kMixerConvolution,
  kMixerOutput,
  kMixerProjection,
  kMixerResidual,
  kPostNorm,
  kMlpL1,
  kMlpL2,
  kMlpActivation,
  kMlpGated,
  kMlpOutput,
};

struct RuntimeModelConfig final {
  std::string model_id;
  std::string architecture;
  std::string artifact_profile;
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
  std::size_t hcl_filter_groups{0};
  std::size_t max_seqlen{0};
  float epsilon{0.0F};
  float rope_scale{0.0F};
  float rope_base{0.0F};
  bool interpolated_rope{false};
  bool qkv_head_major{false};
  bool qkv_source_head_major{false};
  HyenaProjectionDType hyena_projection_dtype{HyenaProjectionDType::kBF16};
  HcmFilterDType hcm_filter_dtype{HcmFilterDType::kBF16};
  bool test_fixture{false};
  InferenceProfile inference_profile{InferenceProfile::kExact};
  std::vector<MixerType> mixer_types;

  [[nodiscard]] std::size_t head_dim() const noexcept {
    return heads == 0 ? 0 : width / heads;
  }
};

[[nodiscard]] std::size_t
backend_warmup_tokens(const RuntimeModelConfig &config,
                      std::size_t arena_capacity) noexcept;
[[nodiscard]] Status
validate_runtime_exact_support(const RuntimeModelConfig &config,
                               InferenceProfile profile);

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

  [[nodiscard]] Status
  load(const ModelFile &model, int device, std::size_t context_capacity,
       bool allow_test_fixture = false,
       InferenceProfile profile = InferenceProfile::kExact);

  [[nodiscard]] Status
  prefill(const std::vector<TokenId> &tokens, std::vector<float> *logits,
          const std::optional<LayerDump> &dump = std::nullopt);
  [[nodiscard]] Status prefill_with_dumps(const std::vector<TokenId> &tokens,
                                          std::vector<float> *logits,
                                          const std::vector<LayerDump> &dumps);
  [[nodiscard]] Status prefill_stateless(const std::vector<TokenId> &tokens,
                                         std::vector<float> *logits);
  [[nodiscard]] Status prefill_cached(const std::vector<TokenId> &tokens,
                                      std::vector<float> *logits);
  [[nodiscard]] Status
  prefill_cached_with_dumps(const std::vector<TokenId> &tokens,
                            std::vector<float> *logits,
                            const std::vector<LayerDump> &dumps);
  [[nodiscard]] Status prefill_chunk(const std::vector<TokenId> &tokens,
                                     std::vector<float> *logits);
  [[nodiscard]] Status prefill_embedding(const std::vector<TokenId> &tokens,
                                         std::size_t layer,
                                         std::vector<float> *embedding);
  [[nodiscard]] Status
  prefill_chunk_embedding(const std::vector<TokenId> &tokens, std::size_t layer,
                          std::vector<float> *embedding);
  [[nodiscard]] Status decode(TokenId token, std::vector<float> *logits);
  [[nodiscard]] Status decode_with_dumps(TokenId token,
                                         std::vector<float> *logits,
                                         const std::vector<LayerDump> &dumps);

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
                            bool allow_test_fixture = false,
                            InferenceProfile profile = InferenceProfile::kExact,
                            std::size_t layer_limit = 0);
  // Creates independent mutable cache/arena/stream state while retaining the
  // source model's immutable device-weight allocations.
  [[nodiscard]] Status
  initialize_shared(const PipelineModel &source, std::size_t context_capacity,
                    InferenceProfile profile = InferenceProfile::kExact);

  [[nodiscard]] Status
  prefill(const std::vector<TokenId> &tokens, std::vector<float> *logits,
          const std::optional<LayerDump> &dump = std::nullopt);
  [[nodiscard]] Status prefill_with_dumps(const std::vector<TokenId> &tokens,
                                          std::vector<float> *logits,
                                          const std::vector<LayerDump> &dumps);
  [[nodiscard]] Status prefill_stateless(const std::vector<TokenId> &tokens,
                                         std::vector<float> *logits);
  [[nodiscard]] Status prefill_cached(const std::vector<TokenId> &tokens,
                                      std::vector<float> *logits);
  [[nodiscard]] Status
  prefill_cached_with_dumps(const std::vector<TokenId> &tokens,
                            std::vector<float> *logits,
                            const std::vector<LayerDump> &dumps);
  [[nodiscard]] Status prefill_chunk(const std::vector<TokenId> &tokens,
                                     std::vector<float> *logits);
  [[nodiscard]] Status prefill_embedding(const std::vector<TokenId> &tokens,
                                         std::size_t layer,
                                         std::vector<float> *embedding);
  [[nodiscard]] Status
  prefill_chunk_embedding(const std::vector<TokenId> &tokens, std::size_t layer,
                          std::vector<float> *embedding);
  [[nodiscard]] Status decode(TokenId token, std::vector<float> *logits);
  [[nodiscard]] Status decode_with_dumps(TokenId token,
                                         std::vector<float> *logits,
                                         const std::vector<LayerDump> &dumps);
  [[nodiscard]] Status prefill_prefix(const std::vector<TokenId> &tokens,
                                      std::vector<float> *hidden);
  [[nodiscard]] Status prefill_chunk_prefix(const std::vector<TokenId> &tokens,
                                            std::vector<float> *hidden);
  [[nodiscard]] Status decode_prefix(TokenId token, std::vector<float> *hidden);

  [[nodiscard]] const RuntimeModelConfig &config() const noexcept;
  [[nodiscard]] const std::vector<StageAssignment> &stages() const noexcept;
  void refresh_cache_bytes() noexcept;
  [[nodiscard]] std::size_t position() const noexcept;
  [[nodiscard]] std::size_t activation_capacity() const noexcept;
  [[nodiscard]] bool uses_q8_kv_cache() const noexcept;
  [[nodiscard]] bool
  shares_weights_with(const PipelineModel &other) const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cuda
