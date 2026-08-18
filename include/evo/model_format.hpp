// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "evo/mapped_file.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace evo {

inline constexpr std::string_view kEvo2ModelProfile = "evo2-runtime-v1";
inline constexpr std::string_view kHyenaDnaModelProfile = "hyenadna-runtime-v1";
inline constexpr std::string_view kEsmcModelProfile = "esmc-runtime-v1";
inline constexpr std::string_view kGenebGpt2ModelProfile =
    "geneb-gpt2-runtime-v1";
inline constexpr std::string_view kGenebDnaGptModelProfile =
    "geneb-dna-gpt-runtime-v1";
inline constexpr std::string_view kGenebCustomEncoderModelProfile =
    "geneb-custom-encoder-runtime-v1";
inline constexpr std::string_view kGenebMambaModelProfile =
    "geneb-mamba-runtime-v1";
inline constexpr std::string_view kGenebHyenaDnaModelProfile =
    "geneb-hyenadna-runtime-v1";
inline constexpr std::string_view kGenebEvo1ModelProfile =
    "geneb-evo1-runtime-v1";
inline constexpr std::string_view kGenebJanusDnaModelProfile =
    "geneb-janusdna-runtime-v1";
inline constexpr std::string_view kGenebSequenceCnnModelProfile =
    "geneb-sequence-cnn-runtime-v1";
inline constexpr std::string_view kGenebRoformerModelProfile =
    "geneb-roformer-runtime-v1";
// Source-compatible alias for the original Evo 2-only public constant.
inline constexpr std::string_view kModelProfile = kEvo2ModelProfile;
inline constexpr std::size_t kMaximumSafetensorsHeaderSize =
    16U * 1024U * 1024U;
inline constexpr std::size_t kTensorNameCapacity = 96;
inline constexpr std::size_t kTensorMaxRank = 8;

enum class MetadataType : std::uint8_t {
  kString = 1,
  kU64 = 2,
  kF64 = 3,
  kBool = 4,
  kU64List = 5,
  kBytes = 6,
};

enum class TensorDType : std::uint8_t {
  kF32 = 1,
  kBF16 = 2,
  kE4M3Software = 3,
};

struct MetadataEntry final {
  std::string key;
  MetadataType type{MetadataType::kBytes};
  std::vector<std::uint8_t> value;
};

struct TensorInfo final {
  std::string name;
  TensorDType dtype{TensorDType::kF32};
  std::uint8_t rank{0};
  std::array<std::uint64_t, kTensorMaxRank> dimensions{};
  std::size_t shard_index{0};
  std::uint64_t data_offset{0};
  std::uint64_t data_size{0};
  std::uint64_t element_count{0};
};

class ModelFile final {
public:
  ModelFile() = default;
  ~ModelFile() = default;

  ModelFile(const ModelFile &) = delete;
  ModelFile &operator=(const ModelFile &) = delete;
  ModelFile(ModelFile &&) noexcept = default;
  ModelFile &operator=(ModelFile &&) noexcept = default;

  // Opens a registered strict Safetensors runtime profile. Payload authenticity
  // is established by the external artifact SHA256, not by a startup rescan.
  [[nodiscard]] Status open(const std::string &path);

  [[nodiscard]] std::string_view format_name() const noexcept {
    return "SAFETENSORS";
  }
  [[nodiscard]] std::string_view profile() const noexcept { return profile_; }
  [[nodiscard]] std::size_t file_size() const noexcept { return file_size_; }
  [[nodiscard]] std::size_t shard_count() const noexcept {
    return mappings_.size();
  }
  [[nodiscard]] std::string_view artifact_root() const noexcept {
    return artifact_root_;
  }
  [[nodiscard]] const std::optional<TokenizerAssetDescriptor> &
  tokenizer_asset_descriptor() const noexcept {
    return tokenizer_asset_descriptor_;
  }
  [[nodiscard]] const std::vector<MetadataEntry> &metadata() const noexcept {
    return metadata_;
  }
  [[nodiscard]] const std::vector<TensorInfo> &tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] const MetadataEntry *
  find_metadata(std::string_view key) const noexcept;
  [[nodiscard]] const TensorInfo *
  find_tensor(std::string_view name) const noexcept;
  [[nodiscard]] const std::uint8_t *
  tensor_data(const TensorInfo &tensor) const noexcept;
  [[nodiscard]] Status read_tensor(const TensorInfo &tensor,
                                   std::uint64_t offset, void *destination,
                                   std::size_t bytes) const;

private:
  [[nodiscard]] Status open_index(const std::string &path);
  [[nodiscard]] Status parse_shard(std::size_t shard_index);
  [[nodiscard]] Status parse_converter_profile_contract();
  [[nodiscard]] Status parse_tokenizer_asset_descriptor();

  std::vector<MappedFile> mappings_;
  std::vector<MetadataEntry> metadata_;
  std::vector<TensorInfo> tensors_;
  std::string profile_;
  std::string artifact_root_;
  std::optional<TokenizerAssetDescriptor> tokenizer_asset_descriptor_;
  std::size_t file_size_{0};
};

[[nodiscard]] const char *metadata_type_name(MetadataType type) noexcept;
[[nodiscard]] const char *tensor_dtype_name(TensorDType dtype) noexcept;
[[nodiscard]] std::string metadata_value_text(const MetadataEntry &entry);

} // namespace evo
