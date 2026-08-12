// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "evo/model_format.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo::cuda {

struct EsmcConfig final {
  std::string model_id;
  std::string architecture;
  std::string artifact_profile;
  std::size_t vocab_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t heads{0};
  std::size_t inner_width{0};
  std::size_t max_seqlen{0};
  float epsilon{0.0F};
  float rope_base{0.0F};
  float residue_scale{0.0F};

  [[nodiscard]] std::size_t head_width() const noexcept {
    return heads == 0 ? 0 : width / heads;
  }
};

class EsmcModel final {
public:
  EsmcModel();
  ~EsmcModel();
  EsmcModel(const EsmcModel &) = delete;
  EsmcModel &operator=(const EsmcModel &) = delete;
  EsmcModel(EsmcModel &&) noexcept;
  EsmcModel &operator=(EsmcModel &&) noexcept;

  [[nodiscard]] Status load(const ModelFile &artifact,
                            const std::vector<int> &devices,
                            bool allow_test_fixture = false);
  [[nodiscard]] const EsmcConfig &config() const noexcept;
  [[nodiscard]] int device() const noexcept;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class EsmcContext;
};

class EsmcContext final {
public:
  EsmcContext();
  ~EsmcContext();
  EsmcContext(const EsmcContext &) = delete;
  EsmcContext &operator=(const EsmcContext &) = delete;
  EsmcContext(EsmcContext &&) noexcept;
  EsmcContext &operator=(EsmcContext &&) noexcept;

  [[nodiscard]] Status initialize_shared(const EsmcModel &model,
                                         std::size_t context_capacity);
  [[nodiscard]] Status prefill(const std::vector<TokenId> &tokens,
                               std::vector<float> *logits);
  [[nodiscard]] Status prefill_embedding(const std::vector<TokenId> &tokens,
                                         std::size_t layer,
                                         std::vector<float> *embedding);
  [[nodiscard]] Status prefill_chunk(const std::vector<TokenId> &tokens,
                                     std::vector<float> *logits);
  [[nodiscard]] Status decode(TokenId token, std::vector<float> *logits);
  [[nodiscard]] std::size_t position() const noexcept;
  [[nodiscard]] std::size_t activation_capacity() const noexcept;
  [[nodiscard]] const EsmcConfig &config() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cuda
