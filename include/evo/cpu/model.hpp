// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "evo/geneb_embedding.hpp"
#include "evo/model_format.hpp"
#include "evo/model_registry.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo::detail {
class LinearExecutor;
}

namespace evo::mps {
class ModelLoader;
}

namespace evo::cpu {

namespace detail {
class HyenaDnaModel;
class HyenaDnaContext;
class EsmcModel;
class EsmcContext;
class GenebDecoderModelAdapter;
class GenebDecoderContext;
class GenebOlmoModelAdapter;
class GenebOlmoContext;
class GenebEsmModelAdapter;
class GenebEsmContext;
class GenebBertModelAdapter;
class GenebBertContext;
class GenebGpt2ModelAdapter;
class GenebGpt2Context;
class GenebDnaGptModelAdapter;
class GenebDnaGptContext;
class GenebCustomEncoderModelAdapter;
class GenebCustomEncoderContext;
class GenebMambaModelAdapter;
class GenebMambaContext;
class GenebHyenaDnaModelAdapter;
class GenebHyenaDnaContext;
class GenebEvo1ModelAdapter;
class GenebEvo1Context;
class GenebJanusDnaModelAdapter;
class GenebJanusDnaContext;
class GenebSequenceCnnModelAdapter;
class GenebSequenceCnnContext;
class GenebRoformerModelAdapter;
class GenebRoformerContext;
} // namespace detail

struct ModelConfig final {
  std::string model_id;
  std::string architecture;
  std::string artifact_profile;
  std::size_t vocab_size{0};
  std::size_t width{0};
  std::size_t layers{0};
  std::size_t max_seqlen{0};
  bool test_fixture{false};
  ArchitectureImplementation implementation{
      ArchitectureImplementation::kUnknown};
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
  [[nodiscard]] const GenebEmbeddingArtifactSpec *
  geneb_embedding_spec() const noexcept;
  [[nodiscard]] Status
  prepare_geneb_embedding_input(std::string_view sequence,
                                GenebPreparedEmbeddingInput *output) const;

private:
  [[nodiscard]] Status
  load_with_executor(const ModelFile &model,
                     std::shared_ptr<evo::detail::LinearExecutor> executor,
                     bool allow_test_fixture);

  struct Impl;
  std::shared_ptr<Impl> impl_;
  std::shared_ptr<detail::HyenaDnaModel> hyena_;
  std::shared_ptr<detail::EsmcModel> esmc_;
  std::shared_ptr<detail::GenebDecoderModelAdapter> geneb_decoder_;
  std::shared_ptr<detail::GenebOlmoModelAdapter> geneb_olmo_;
  std::shared_ptr<detail::GenebEsmModelAdapter> geneb_esm_;
  std::shared_ptr<detail::GenebBertModelAdapter> geneb_bert_;
  std::shared_ptr<detail::GenebGpt2ModelAdapter> geneb_gpt2_;
  std::shared_ptr<detail::GenebDnaGptModelAdapter> geneb_dna_gpt_;
  std::shared_ptr<detail::GenebCustomEncoderModelAdapter> geneb_custom_;
  std::shared_ptr<detail::GenebMambaModelAdapter> geneb_mamba_;
  std::shared_ptr<detail::GenebHyenaDnaModelAdapter> geneb_hyenadna_;
  std::shared_ptr<detail::GenebEvo1ModelAdapter> geneb_evo1_;
  std::shared_ptr<detail::GenebJanusDnaModelAdapter> geneb_janusdna_;
  std::shared_ptr<detail::GenebSequenceCnnModelAdapter> geneb_sequence_cnn_;
  std::shared_ptr<detail::GenebRoformerModelAdapter> geneb_roformer_;
  std::shared_ptr<ArtifactTokenizer> artifact_tokenizer_;
  std::optional<GenebEmbeddingArtifactSpec> geneb_embedding_spec_;
  const ArchitectureBackendFactorySpec *factory_{nullptr};
  friend class Context;
  friend class evo::mps::ModelLoader;
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
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           std::size_t layer, std::vector<float> *embedding);
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
  std::unique_ptr<detail::EsmcContext> esmc_;
  std::unique_ptr<detail::GenebDecoderContext> geneb_decoder_;
  std::unique_ptr<detail::GenebOlmoContext> geneb_olmo_;
  std::unique_ptr<detail::GenebEsmContext> geneb_esm_;
  std::unique_ptr<detail::GenebBertContext> geneb_bert_;
  std::unique_ptr<detail::GenebGpt2Context> geneb_gpt2_;
  std::unique_ptr<detail::GenebDnaGptContext> geneb_dna_gpt_;
  std::unique_ptr<detail::GenebCustomEncoderContext> geneb_custom_;
  std::unique_ptr<detail::GenebMambaContext> geneb_mamba_;
  std::unique_ptr<detail::GenebHyenaDnaContext> geneb_hyenadna_;
  std::unique_ptr<detail::GenebEvo1Context> geneb_evo1_;
  std::unique_ptr<detail::GenebJanusDnaContext> geneb_janusdna_;
  std::unique_ptr<detail::GenebSequenceCnnContext> geneb_sequence_cnn_;
  std::unique_ptr<detail::GenebRoformerContext> geneb_roformer_;
  const ArchitectureBackendFactorySpec *factory_{nullptr};
};

} // namespace evo::cpu
