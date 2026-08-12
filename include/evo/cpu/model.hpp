// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "evo/model_format.hpp"
#include "evo/model_registry.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo::cpu {

namespace detail {
class HyenaDnaModel;
class HyenaDnaContext;
} // namespace detail

struct ModelConfig final {
  std::string model_id;
  std::string architecture;
  std::size_t vocab_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t max_seqlen{0};
  bool test_fixture{false};
  ArchitectureTokenizer tokenizer{ArchitectureTokenizer::kByteIdentity};
};

class Model final {
public:
  Model();
  ~Model();
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  Model(Model &&) noexcept;
  Model &operator=(Model &&) noexcept;

  [[nodiscard]] Status load(const ModelFile &model,
                            bool allow_test_fixture = false);
  [[nodiscard]] const ModelConfig &config() const noexcept;
  [[nodiscard]] const char *kernel_name() const noexcept;
  [[nodiscard]] Status encode(std::string_view sequence,
                              std::vector<TokenId> *tokens) const;
  [[nodiscard]] Status decode_token(TokenId token, std::uint8_t *byte) const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  std::shared_ptr<detail::HyenaDnaModel> hyena_;
  friend class Context;
};

class Context final {
public:
  Context();
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) noexcept;
  Context &operator=(Context &&) noexcept;

  [[nodiscard]] Status initialize_shared(const Model &model,
                                         std::size_t context_capacity,
                                         std::size_t layer_begin = 0);
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
  [[nodiscard]] Status prefill_from_hidden(const std::vector<float> &hidden,
                                           std::size_t rows,
                                           std::vector<float> *logits);
  [[nodiscard]] Status
  prefill_chunk_from_hidden(const std::vector<float> &hidden, std::size_t rows,
                            std::vector<float> *logits);

  [[nodiscard]] std::size_t position() const noexcept;
  [[nodiscard]] std::size_t activation_capacity() const noexcept;
  [[nodiscard]] const ModelConfig &config() const noexcept;
  [[nodiscard]] const char *kernel_name() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<detail::HyenaDnaContext> hyena_;
};

} // namespace evo::cpu
