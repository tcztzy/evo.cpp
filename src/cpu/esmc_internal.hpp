// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "evo/cpu/model.hpp"
#include "evo/model_format.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo::cpu::detail {

class EsmcModel final {
public:
  EsmcModel();
  ~EsmcModel();
  EsmcModel(const EsmcModel &) = delete;
  EsmcModel &operator=(const EsmcModel &) = delete;

  [[nodiscard]] Status load(const ModelFile &artifact, bool allow_test_fixture);
  [[nodiscard]] const ModelConfig &config() const noexcept;
  [[nodiscard]] const char *kernel_name() const noexcept;

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

  [[nodiscard]] Status initialize_shared(const EsmcModel &model,
                                         std::size_t capacity);
  [[nodiscard]] Status prefill(const std::vector<TokenId> &tokens,
                               std::vector<float> *logits);
  [[nodiscard]] Status prefill_embedding(const std::vector<TokenId> &tokens,
                                         std::size_t layer,
                                         std::vector<float> *embedding);
  [[nodiscard]] std::size_t position() const noexcept;
  [[nodiscard]] std::size_t activation_capacity() const noexcept;
  [[nodiscard]] const ModelConfig &config() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu::detail
