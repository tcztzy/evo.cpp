// SPDX-License-Identifier: Apache-2.0
#include "evo2c/model_format.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

#include "evo2c/crc32.hpp"

namespace evo2c {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'E', 'V', 'O', '2', 'C', 0, 0, 0};
constexpr std::array<std::uint8_t, 4> kMetadataMagic{'M', 'E', 'T', 'A'};
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint16_t kMetadataVersion = 1;
constexpr std::uint64_t kMaxMetadataSize = 16U * 1024U * 1024U;
constexpr std::uint32_t kMaxMetadataEntries = 4096;
constexpr std::uint64_t kMaxTensorCount = 1'000'000;
constexpr std::size_t kHeaderCrcOffset = 80;
constexpr std::size_t kDescriptorCrcOffset = 196;

Status format_error(const std::string& message) {
  return {ErrorCode::kModelFormat, message};
}

std::uint16_t read_u16(const std::uint8_t* data) noexcept {
  return static_cast<std::uint16_t>(static_cast<unsigned int>(data[0]) |
                                    (static_cast<unsigned int>(data[1]) << 8U));
}

std::uint32_t read_u32(const std::uint8_t* data) noexcept {
  return static_cast<std::uint32_t>(data[0]) |
         (static_cast<std::uint32_t>(data[1]) << 8U) |
         (static_cast<std::uint32_t>(data[2]) << 16U) |
         (static_cast<std::uint32_t>(data[3]) << 24U);
}

std::uint64_t read_u64(const std::uint8_t* data) noexcept {
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    value |= static_cast<std::uint64_t>(data[byte]) << (8U * byte);
  }
  return value;
}

bool checked_add(const std::uint64_t left,
                 const std::uint64_t right,
                 std::uint64_t* output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  *output = left + right;
  return true;
}

bool checked_mul(const std::uint64_t left,
                 const std::uint64_t right,
                 std::uint64_t* output) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *output = left * right;
  return true;
}

bool align_up(const std::uint64_t value,
              const std::uint64_t alignment,
              std::uint64_t* output) noexcept {
  const auto remainder = value % alignment;
  if (remainder == 0) {
    *output = value;
    return true;
  }
  return checked_add(value, alignment - remainder, output);
}

bool zero_bytes(const std::uint8_t* data, const std::size_t size) noexcept {
  return std::all_of(data, data + size, [](const std::uint8_t value) { return value == 0; });
}

bool valid_key_character(const char character) noexcept {
  const auto value = static_cast<unsigned char>(character);
  return std::isalnum(value) != 0 || character == '.' || character == '_' || character == '-';
}

bool valid_tensor_dtype(const std::uint8_t raw) noexcept {
  return raw >= static_cast<std::uint8_t>(TensorDType::kF32) &&
         raw <= static_cast<std::uint8_t>(TensorDType::kE4M3Software);
}

bool valid_metadata_type(const std::uint8_t raw) noexcept {
  return raw >= static_cast<std::uint8_t>(MetadataType::kString) &&
         raw <= static_cast<std::uint8_t>(MetadataType::kBytes);
}

bool expected_tensor_size(const TensorDType dtype,
                          const std::uint64_t elements,
                          std::uint64_t* bytes) noexcept {
  switch (dtype) {
    case TensorDType::kF32:
      return checked_mul(elements, 4, bytes);
    case TensorDType::kBF16:
      return checked_mul(elements, 2, bytes);
    case TensorDType::kE4M3Software:
      *bytes = elements;
      return true;
    case TensorDType::kQ8_0:
      if (elements % 32 != 0) {
        return false;
      }
      return checked_mul(elements / 32, 34, bytes);
  }
  return false;
}

std::uint32_t crc_with_zeroed_field(const std::uint8_t* data,
                                    const std::size_t size,
                                    const std::size_t field_offset) {
  std::array<std::uint8_t, kTensorDescriptorSize> scratch{};
  if (size > scratch.size() || field_offset + sizeof(std::uint32_t) > size) {
    return 0;
  }
  std::copy(data, data + size, scratch.begin());
  std::fill(scratch.begin() + static_cast<std::ptrdiff_t>(field_offset),
            scratch.begin() + static_cast<std::ptrdiff_t>(field_offset + sizeof(std::uint32_t)),
            0);
  return crc32(scratch.data(), size);
}

std::string tensor_context(const std::uint64_t index, const std::string& detail) {
  std::ostringstream message;
  message << "tensor descriptor " << index << ": " << detail;
  return message.str();
}

}  // namespace

Status ModelFile::open(const std::string& path) {
  ModelFile candidate;
  auto status = candidate.mapping_.open(path);
  if (!status.ok()) {
    return status;
  }
  status = candidate.parse();
  if (!status.ok()) {
    return status;
  }
  *this = std::move(candidate);
  return Status::Ok();
}

Status ModelFile::parse() {
  if (mapping_.size() < kModelHeaderSize) {
    return format_error("file shorter than 128-byte EVO2C header");
  }
  const auto* header = mapping_.data();
  if (!std::equal(kMagic.begin(), kMagic.end(), header)) {
    return format_error("bad magic; expected EVO2C v1 file");
  }

  version_ = read_u32(header + 8);
  if (version_ != kModelFormatVersion) {
    return format_error("unsupported model format version " + std::to_string(version_));
  }
  if (read_u32(header + 12) != kEndianMarker) {
    return format_error("bad endian marker; only canonical little-endian files are supported");
  }
  if (read_u32(header + 16) != kModelHeaderSize) {
    return format_error("header_size must be 128");
  }
  if (read_u32(header + 20) != 0) {
    return format_error("unsupported nonzero header flags");
  }
  if (read_u64(header + 24) != mapping_.size()) {
    return format_error("header file_size does not match mapped file size");
  }
  if (!zero_bytes(header + 84, 44)) {
    return format_error("header reserved bytes must be zero");
  }
  const auto stored_header_crc = read_u32(header + kHeaderCrcOffset);
  const auto computed_header_crc =
      crc_with_zeroed_field(header, kModelHeaderSize, kHeaderCrcOffset);
  if (stored_header_crc != computed_header_crc) {
    return format_error("header CRC32 mismatch");
  }

  const auto metadata_offset = read_u64(header + 32);
  const auto metadata_size = read_u64(header + 40);
  const auto tensor_table_offset = read_u64(header + 48);
  const auto tensor_count = read_u64(header + 56);
  const auto descriptor_size = read_u32(header + 64);
  const auto alignment = read_u32(header + 68);
  const auto data_offset = read_u64(header + 72);

  if (metadata_offset != kModelHeaderSize) {
    return format_error("metadata_offset must immediately follow header at byte 128");
  }
  if (metadata_size < 16 || metadata_size > kMaxMetadataSize) {
    return format_error("metadata_size must be in [16, 16777216]");
  }
  if (tensor_count == 0 || tensor_count > kMaxTensorCount) {
    return format_error("tensor_count must be in [1, 1000000]");
  }
  if (descriptor_size != kTensorDescriptorSize) {
    return format_error("tensor_descriptor_size must be 256");
  }
  if (alignment != kModelAlignment) {
    return format_error("alignment must be 64");
  }

  std::uint64_t metadata_end = 0;
  std::uint64_t expected_table_offset = 0;
  if (!checked_add(metadata_offset, metadata_size, &metadata_end) ||
      !align_up(metadata_end, kModelAlignment, &expected_table_offset)) {
    return format_error("metadata range overflows uint64");
  }
  if (tensor_table_offset != expected_table_offset) {
    return format_error("tensor_table_offset is not canonical 64-byte alignment after metadata");
  }

  std::uint64_t table_size = 0;
  std::uint64_t table_end = 0;
  std::uint64_t expected_data_offset = 0;
  if (!checked_mul(tensor_count, kTensorDescriptorSize, &table_size) ||
      !checked_add(tensor_table_offset, table_size, &table_end) ||
      !align_up(table_end, kModelAlignment, &expected_data_offset)) {
    return format_error("tensor table range overflows uint64");
  }
  if (data_offset != expected_data_offset) {
    return format_error("data_offset is not canonical 64-byte alignment after tensor table");
  }
  if (data_offset > mapping_.size()) {
    return format_error("data_offset lies beyond end of file");
  }
  if (!zero_bytes(mapping_.data() + metadata_end,
                  static_cast<std::size_t>(tensor_table_offset - metadata_end)) ||
      !zero_bytes(mapping_.data() + table_end,
                  static_cast<std::size_t>(data_offset - table_end))) {
    return format_error("section alignment padding must be zero");
  }

  auto status = parse_metadata(metadata_offset, metadata_size);
  if (!status.ok()) {
    return status;
  }
  return parse_tensors(tensor_table_offset, tensor_count, data_offset);
}

Status ModelFile::parse_metadata(const std::uint64_t offset, const std::uint64_t size) {
  if (offset > mapping_.size() || size > mapping_.size() - offset) {
    return format_error("metadata range lies beyond end of file");
  }
  const auto* section = mapping_.data() + offset;
  const auto section_size = static_cast<std::size_t>(size);
  if (!std::equal(kMetadataMagic.begin(), kMetadataMagic.end(), section)) {
    return format_error("metadata bad magic");
  }
  if (read_u16(section + 4) != kMetadataVersion) {
    return format_error("unsupported metadata version");
  }
  if (read_u16(section + 6) != 0) {
    return format_error("metadata reserved field must be zero");
  }
  const auto entry_count = read_u32(section + 8);
  if (entry_count > kMaxMetadataEntries) {
    return format_error("metadata entry_count exceeds 4096");
  }
  const auto stored_crc = read_u32(section + 12);
  const auto computed_crc = crc32(section + 16, section_size - 16);
  if (stored_crc != computed_crc) {
    return format_error("metadata CRC32 mismatch");
  }

  std::set<std::string> keys;
  std::size_t cursor = 16;
  metadata_.reserve(entry_count);
  for (std::uint32_t index = 0; index < entry_count; ++index) {
    if (cursor > section_size || section_size - cursor < 8) {
      return format_error("metadata entry " + std::to_string(index) + " header is truncated");
    }
    const auto key_length = read_u16(section + cursor);
    const auto raw_type = section[cursor + 2];
    const auto reserved = section[cursor + 3];
    const auto value_length = read_u32(section + cursor + 4);
    if (key_length == 0 || key_length > 255) {
      return format_error("metadata entry " + std::to_string(index) + " key length is invalid");
    }
    if (!valid_metadata_type(raw_type)) {
      return format_error("metadata entry " + std::to_string(index) + " type is invalid");
    }
    if (reserved != 0) {
      return format_error("metadata entry " + std::to_string(index) + " reserved byte must be zero");
    }

    std::uint64_t content_end_u64 = 0;
    const auto content_start = static_cast<std::uint64_t>(cursor) + 8U;
    if (!checked_add(content_start, key_length, &content_end_u64) ||
        !checked_add(content_end_u64, value_length, &content_end_u64) ||
        content_end_u64 > section_size) {
      return format_error("metadata entry " + std::to_string(index) + " content is truncated");
    }
    const auto key_start = cursor + 8;
    const auto value_start = key_start + key_length;
    std::string key(reinterpret_cast<const char*>(section + key_start), key_length);
    if (!std::all_of(key.begin(), key.end(), valid_key_character)) {
      return format_error("metadata entry " + std::to_string(index) + " key has invalid characters");
    }
    if (!keys.insert(key).second) {
      return format_error("duplicate metadata key '" + key + "'");
    }

    const auto type = static_cast<MetadataType>(raw_type);
    const bool valid_length =
        (type == MetadataType::kString || type == MetadataType::kBytes) ||
        (type == MetadataType::kU64 && value_length == 8) ||
        (type == MetadataType::kF64 && value_length == 8) ||
        (type == MetadataType::kBool && value_length == 1) ||
        (type == MetadataType::kU64List && value_length % 8 == 0);
    if (!valid_length) {
      return format_error("metadata entry '" + key + "' has invalid value length for its type");
    }
    if (type == MetadataType::kBool && section[value_start] > 1) {
      return format_error("metadata bool entry '" + key + "' must be 0 or 1");
    }

    MetadataEntry entry;
    entry.key = std::move(key);
    entry.type = type;
    entry.value.assign(section + value_start, section + value_start + value_length);
    metadata_.push_back(std::move(entry));

    std::uint64_t aligned_end_u64 = 0;
    if (!align_up(content_end_u64, 8, &aligned_end_u64) || aligned_end_u64 > section_size) {
      return format_error("metadata entry " + std::to_string(index) + " padding is truncated");
    }
    const auto content_end = static_cast<std::size_t>(content_end_u64);
    const auto aligned_end = static_cast<std::size_t>(aligned_end_u64);
    if (!zero_bytes(section + content_end, aligned_end - content_end)) {
      return format_error("metadata entry " + std::to_string(index) + " padding must be zero");
    }
    cursor = aligned_end;
  }
  if (cursor != section_size) {
    return format_error("metadata section has unparsed trailing bytes");
  }
  return Status::Ok();
}

Status ModelFile::parse_tensors(const std::uint64_t table_offset,
                                const std::uint64_t tensor_count,
                                const std::uint64_t data_offset) {
  std::set<std::string> names;
  tensors_.reserve(static_cast<std::size_t>(tensor_count));

  for (std::uint64_t index = 0; index < tensor_count; ++index) {
    const auto descriptor_offset = table_offset + index * kTensorDescriptorSize;
    const auto* descriptor = mapping_.data() + descriptor_offset;
    const auto stored_descriptor_crc = read_u32(descriptor + kDescriptorCrcOffset);
    const auto computed_descriptor_crc =
        crc_with_zeroed_field(descriptor, kTensorDescriptorSize, kDescriptorCrcOffset);
    if (stored_descriptor_crc != computed_descriptor_crc) {
      return format_error(tensor_context(index, "descriptor CRC32 mismatch"));
    }
    if (read_u16(descriptor + 98) != 0 || read_u32(descriptor + 100) != 0 ||
        !zero_bytes(descriptor + 200, 56)) {
      return format_error(tensor_context(index, "flags and reserved bytes must be zero"));
    }

    const auto* name_end = std::find(descriptor, descriptor + kTensorNameCapacity, 0);
    if (name_end == descriptor || name_end == descriptor + kTensorNameCapacity) {
      return format_error(tensor_context(index, "name must be nonempty and NUL-terminated"));
    }
    if (!zero_bytes(name_end, static_cast<std::size_t>(descriptor + kTensorNameCapacity - name_end))) {
      return format_error(tensor_context(index, "bytes after name terminator must be zero"));
    }
    std::string name(reinterpret_cast<const char*>(descriptor),
                     static_cast<std::size_t>(name_end - descriptor));
    if (!std::all_of(name.begin(), name.end(), valid_key_character)) {
      return format_error(tensor_context(index, "name has invalid characters"));
    }
    if (!names.insert(name).second) {
      return format_error("duplicate tensor name '" + name + "'");
    }

    const auto raw_dtype = descriptor[96];
    const auto rank = descriptor[97];
    if (!valid_tensor_dtype(raw_dtype)) {
      return format_error(tensor_context(index, "dtype is invalid"));
    }
    if (rank > kTensorMaxRank) {
      return format_error(tensor_context(index, "rank exceeds 8"));
    }

    TensorInfo tensor;
    tensor.name = std::move(name);
    tensor.dtype = static_cast<TensorDType>(raw_dtype);
    tensor.rank = rank;
    std::uint64_t computed_elements = 1;
    for (std::size_t dimension = 0; dimension < kTensorMaxRank; ++dimension) {
      tensor.dimensions[dimension] = read_u64(descriptor + 104 + dimension * 8);
      if (dimension < rank) {
        if (tensor.dimensions[dimension] == 0 ||
            !checked_mul(computed_elements, tensor.dimensions[dimension], &computed_elements)) {
          return format_error(tensor_context(index, "dimensions are zero or overflow uint64"));
        }
      } else if (tensor.dimensions[dimension] != 0) {
        return format_error(tensor_context(index, "unused dimensions must be zero"));
      }
    }

    tensor.data_offset = read_u64(descriptor + 168);
    tensor.data_size = read_u64(descriptor + 176);
    tensor.element_count = read_u64(descriptor + 184);
    tensor.data_crc32 = read_u32(descriptor + 192);
    if (tensor.element_count != computed_elements) {
      return format_error(tensor_context(index, "element_count does not match shape product"));
    }
    std::uint64_t expected_size = 0;
    if (!expected_tensor_size(tensor.dtype, tensor.element_count, &expected_size) ||
        tensor.data_size != expected_size) {
      return format_error(tensor_context(index, "data_size does not match dtype and element_count"));
    }
    if (tensor.data_size == 0 || tensor.data_offset < data_offset ||
        tensor.data_offset % kModelAlignment != 0) {
      return format_error(tensor_context(index, "payload offset/size/alignment is invalid"));
    }
    if (tensor.data_offset > mapping_.size() || tensor.data_size > mapping_.size() - tensor.data_offset) {
      return format_error(tensor_context(index, "payload range lies beyond end of file"));
    }
    tensors_.push_back(std::move(tensor));
  }

  std::vector<const TensorInfo*> by_offset;
  by_offset.reserve(tensors_.size());
  for (const auto& tensor : tensors_) {
    by_offset.push_back(&tensor);
  }
  std::sort(by_offset.begin(), by_offset.end(), [](const TensorInfo* left, const TensorInfo* right) {
    return left->data_offset < right->data_offset;
  });
  for (std::size_t index = 1; index < by_offset.size(); ++index) {
    const auto previous_end = by_offset[index - 1]->data_offset + by_offset[index - 1]->data_size;
    if (previous_end > by_offset[index]->data_offset) {
      return format_error("tensor payloads overlap: '" + by_offset[index - 1]->name + "' and '" +
                          by_offset[index]->name + "'");
    }
  }

  for (const auto& tensor : tensors_) {
    const auto payload_size = static_cast<std::size_t>(tensor.data_size);
    const auto computed_crc = crc32(mapping_.data() + tensor.data_offset, payload_size);
    if (computed_crc != tensor.data_crc32) {
      return format_error("tensor payload CRC32 mismatch for '" + tensor.name + "'");
    }
  }
  return Status::Ok();
}

const MetadataEntry* ModelFile::find_metadata(const std::string_view key) const noexcept {
  const auto found = std::find_if(metadata_.begin(), metadata_.end(), [key](const MetadataEntry& entry) {
    return entry.key == key;
  });
  return found == metadata_.end() ? nullptr : &*found;
}

const TensorInfo* ModelFile::find_tensor(const std::string_view name) const noexcept {
  const auto found = std::find_if(tensors_.begin(), tensors_.end(), [name](const TensorInfo& tensor) {
    return tensor.name == name;
  });
  return found == tensors_.end() ? nullptr : &*found;
}

const std::uint8_t* ModelFile::tensor_data(const TensorInfo& tensor) const noexcept {
  if (tensor.data_offset > mapping_.size() || tensor.data_size > mapping_.size() - tensor.data_offset) {
    return nullptr;
  }
  return mapping_.data() + tensor.data_offset;
}

const char* metadata_type_name(const MetadataType type) noexcept {
  switch (type) {
    case MetadataType::kString:
      return "string";
    case MetadataType::kU64:
      return "u64";
    case MetadataType::kF64:
      return "f64";
    case MetadataType::kBool:
      return "bool";
    case MetadataType::kU64List:
      return "u64[]";
    case MetadataType::kBytes:
      return "bytes";
  }
  return "unknown";
}

const char* tensor_dtype_name(const TensorDType dtype) noexcept {
  switch (dtype) {
    case TensorDType::kF32:
      return "F32";
    case TensorDType::kBF16:
      return "BF16";
    case TensorDType::kQ8_0:
      return "Q8_0";
    case TensorDType::kE4M3Software:
      return "E4M3_SW";
  }
  return "UNKNOWN";
}

std::string metadata_value_text(const MetadataEntry& entry) {
  std::ostringstream output;
  switch (entry.type) {
    case MetadataType::kString:
      return std::string(entry.value.begin(), entry.value.end());
    case MetadataType::kU64:
      return std::to_string(read_u64(entry.value.data()));
    case MetadataType::kF64: {
      const auto bits = read_u64(entry.value.data());
      double value = 0.0;
      std::memcpy(&value, &bits, sizeof(value));
      output << std::setprecision(17) << value;
      return output.str();
    }
    case MetadataType::kBool:
      return entry.value[0] == 0 ? "false" : "true";
    case MetadataType::kU64List:
      output << '[';
      for (std::size_t offset = 0; offset < entry.value.size(); offset += 8) {
        if (offset != 0) {
          output << ',';
        }
        output << read_u64(entry.value.data() + offset);
      }
      output << ']';
      return output.str();
    case MetadataType::kBytes:
      output << '<' << entry.value.size() << " bytes>";
      return output.str();
  }
  return "<invalid>";
}

}  // namespace evo2c
