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
  bool test_fixture{false};
  std::vector<MixerType> mixer_types;

  [[nodiscard]] std::size_t head_dim() const noexcept {
    return heads == 0 ? 0 : width / heads;
  }
};

struct LayerDump final {
  std::size_t layer{0};
  std::string path;
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
  [[nodiscard]] Status decode(TokenId token, std::vector<float> *logits);

  [[nodiscard]] const RuntimeModelConfig &config() const noexcept;
  [[nodiscard]] std::size_t position() const noexcept;
  [[nodiscard]] int device() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo2c::cuda
