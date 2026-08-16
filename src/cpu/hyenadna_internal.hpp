// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "evo/cpu/model.hpp"
#include "evo/model_format.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo::cpu::detail {

class HyenaDnaModel final {
public:
  HyenaDnaModel();
  ~HyenaDnaModel();
  HyenaDnaModel(const HyenaDnaModel &) = delete;
  HyenaDnaModel &operator=(const HyenaDnaModel &) = delete;

  [[nodiscard]] Status
  load(const ModelFile &artifact, bool allow_test_fixture,
       std::shared_ptr<evo::detail::LinearExecutor> executor = {});
  [[nodiscard]] const ModelConfig &config() const noexcept;
  [[nodiscard]] const char *kernel_name() const noexcept;
  [[nodiscard]] Status encode(std::string_view sequence,
                              std::vector<TokenId> *tokens) const;
  [[nodiscard]] Status decode_token(TokenId token, std::uint8_t *byte) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class HyenaDnaContext;
};

class HyenaDnaContext final {
public:
  HyenaDnaContext();
  ~HyenaDnaContext();
  HyenaDnaContext(const HyenaDnaContext &) = delete;
  HyenaDnaContext &operator=(const HyenaDnaContext &) = delete;

  [[nodiscard]] Status initialize_shared(const HyenaDnaModel &model,
                                         std::size_t capacity);
  [[nodiscard]] Status prefill(const std::vector<TokenId> &tokens,
                               std::vector<float> *logits);
  [[nodiscard]] Status prefill_chunk(const std::vector<TokenId> &tokens,
                                     std::vector<float> *logits);
  [[nodiscard]] Status decode(TokenId token, std::vector<float> *logits);
  [[nodiscard]] Status prefill_embedding(const std::vector<TokenId> &tokens,
                                         std::size_t layer,
                                         std::vector<float> *embedding);
  [[nodiscard]] Status
  prefill_chunk_embedding(const std::vector<TokenId> &tokens, std::size_t layer,
                          std::vector<float> *embedding);
  [[nodiscard]] std::size_t position() const noexcept;
  [[nodiscard]] std::size_t activation_capacity() const noexcept;
  [[nodiscard]] const ModelConfig &config() const noexcept;
  [[nodiscard]] const char *kernel_name() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace evo::cpu::detail
