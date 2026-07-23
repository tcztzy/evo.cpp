// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "evo2c/mapped_file.hpp"
#include "evo2c/status.hpp"

namespace evo2c {

inline constexpr std::uint32_t kModelFormatVersion = 1;
inline constexpr std::size_t kModelHeaderSize = 128;
inline constexpr std::size_t kTensorDescriptorSize = 256;
inline constexpr std::size_t kTensorNameCapacity = 96;
inline constexpr std::size_t kTensorMaxRank = 8;
inline constexpr std::uint32_t kModelAlignment = 64;

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
  kQ8_0 = 3,
  kE4M3Software = 4,
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
  std::uint64_t data_offset{0};
  std::uint64_t data_size{0};
  std::uint64_t element_count{0};
  std::uint32_t data_crc32{0};
};

class ModelFile final {
 public:
  ModelFile() = default;
  ~ModelFile() = default;

  ModelFile(const ModelFile&) = delete;
  ModelFile& operator=(const ModelFile&) = delete;
  ModelFile(ModelFile&&) noexcept = default;
  ModelFile& operator=(ModelFile&&) noexcept = default;

  // Opens and validates the complete file, including every tensor payload CRC.
  [[nodiscard]] Status open(const std::string& path);

  [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
  [[nodiscard]] std::size_t file_size() const noexcept { return mapping_.size(); }
  [[nodiscard]] const std::vector<MetadataEntry>& metadata() const noexcept { return metadata_; }
  [[nodiscard]] const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
  [[nodiscard]] const MetadataEntry* find_metadata(std::string_view key) const noexcept;
  [[nodiscard]] const TensorInfo* find_tensor(std::string_view name) const noexcept;
  [[nodiscard]] const std::uint8_t* tensor_data(const TensorInfo& tensor) const noexcept;

 private:
  [[nodiscard]] Status parse();
  [[nodiscard]] Status parse_metadata(std::uint64_t offset, std::uint64_t size);
  [[nodiscard]] Status parse_tensors(std::uint64_t table_offset,
                                     std::uint64_t tensor_count,
                                     std::uint64_t data_offset);

  MappedFile mapping_;
  std::uint32_t version_{0};
  std::vector<MetadataEntry> metadata_;
  std::vector<TensorInfo> tensors_;
};

[[nodiscard]] const char* metadata_type_name(MetadataType type) noexcept;
[[nodiscard]] const char* tensor_dtype_name(TensorDType dtype) noexcept;
[[nodiscard]] std::string metadata_value_text(const MetadataEntry& entry);

}  // namespace evo2c

