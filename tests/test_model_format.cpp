// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "evo2c/crc32.hpp"
#include "evo2c/model_format.hpp"

namespace {

constexpr std::size_t kHeaderCrcOffset = 80;
constexpr std::size_t kDescriptorCrcOffset = 196;

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

std::size_t align_up(const std::size_t value, const std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

void write_u16(std::vector<std::uint8_t>& bytes,
               const std::size_t offset,
               const std::uint16_t value) {
  bytes.at(offset) = static_cast<std::uint8_t>(value);
  bytes.at(offset + 1) = static_cast<std::uint8_t>(value >> 8U);
}

void write_u32(std::vector<std::uint8_t>& bytes,
               const std::size_t offset,
               const std::uint32_t value) {
  for (std::size_t byte = 0; byte < 4; ++byte) {
    bytes.at(offset + byte) = static_cast<std::uint8_t>(value >> (8U * byte));
  }
}

void write_u64(std::vector<std::uint8_t>& bytes,
               const std::size_t offset,
               const std::uint64_t value) {
  for (std::size_t byte = 0; byte < 8; ++byte) {
    bytes.at(offset + byte) = static_cast<std::uint8_t>(value >> (8U * byte));
  }
}

std::vector<std::uint8_t> u64_value(const std::initializer_list<std::uint64_t> values) {
  std::vector<std::uint8_t> bytes(values.size() * 8, 0);
  std::size_t offset = 0;
  for (const auto value : values) {
    write_u64(bytes, offset, value);
    offset += 8;
  }
  return bytes;
}

void append_metadata_entry(std::vector<std::uint8_t>& metadata,
                           const std::string_view key,
                           const evo2c::MetadataType type,
                           const std::vector<std::uint8_t>& value) {
  const auto start = metadata.size();
  metadata.resize(start + 8 + key.size() + value.size(), 0);
  write_u16(metadata, start, static_cast<std::uint16_t>(key.size()));
  metadata[start + 2] = static_cast<std::uint8_t>(type);
  write_u32(metadata, start + 4, static_cast<std::uint32_t>(value.size()));
  std::copy(key.begin(), key.end(), metadata.begin() + static_cast<std::ptrdiff_t>(start + 8));
  std::copy(value.begin(), value.end(),
            metadata.begin() + static_cast<std::ptrdiff_t>(start + 8 + key.size()));
  metadata.resize(align_up(metadata.size(), 8), 0);
}

std::vector<std::uint8_t> build_metadata() {
  std::vector<std::uint8_t> metadata(16, 0);
  std::copy_n("META", 4, metadata.begin());
  write_u16(metadata, 4, 1);

  const std::string model_name = "tiny-evo2";
  append_metadata_entry(metadata, "model.name", evo2c::MetadataType::kString,
                        {model_name.begin(), model_name.end()});
  append_metadata_entry(metadata, "config.hidden_size", evo2c::MetadataType::kU64,
                        u64_value({4}));
  append_metadata_entry(metadata, "config.tie_embeddings", evo2c::MetadataType::kBool,
                        {1});
  append_metadata_entry(metadata, "config.layers", evo2c::MetadataType::kU64List,
                        u64_value({0, 3}));
  write_u32(metadata, 8, 4);
  write_u32(metadata, 12, evo2c::crc32(metadata.data() + 16, metadata.size() - 16));
  return metadata;
}

void set_descriptor_crc(std::vector<std::uint8_t>& bytes, const std::size_t offset) {
  write_u32(bytes, offset + kDescriptorCrcOffset, 0);
  write_u32(bytes, offset + kDescriptorCrcOffset,
            evo2c::crc32(bytes.data() + offset, evo2c::kTensorDescriptorSize));
}

void set_header_crc(std::vector<std::uint8_t>& bytes) {
  write_u32(bytes, kHeaderCrcOffset, 0);
  write_u32(bytes, kHeaderCrcOffset, evo2c::crc32(bytes.data(), evo2c::kModelHeaderSize));
}

struct Fixture final {
  std::vector<std::uint8_t> bytes;
  std::size_t metadata_offset{evo2c::kModelHeaderSize};
  std::size_t table_offset{0};
  std::size_t first_payload_offset{0};
  std::size_t second_payload_offset{0};
};

void write_descriptor(std::vector<std::uint8_t>& bytes,
                      const std::size_t descriptor_offset,
                      const std::string_view name,
                      const evo2c::TensorDType dtype,
                      const std::initializer_list<std::uint64_t> dimensions,
                      const std::size_t payload_offset,
                      const std::vector<std::uint8_t>& payload) {
  std::copy(name.begin(), name.end(),
            bytes.begin() + static_cast<std::ptrdiff_t>(descriptor_offset));
  bytes[descriptor_offset + 96] = static_cast<std::uint8_t>(dtype);
  bytes[descriptor_offset + 97] = static_cast<std::uint8_t>(dimensions.size());
  std::uint64_t elements = 1;
  std::size_t dimension_index = 0;
  for (const auto dimension : dimensions) {
    write_u64(bytes, descriptor_offset + 104 + dimension_index * 8, dimension);
    elements *= dimension;
    ++dimension_index;
  }
  write_u64(bytes, descriptor_offset + 168, payload_offset);
  write_u64(bytes, descriptor_offset + 176, payload.size());
  write_u64(bytes, descriptor_offset + 184, elements);
  write_u32(bytes, descriptor_offset + 192, evo2c::crc32(payload.data(), payload.size()));
  set_descriptor_crc(bytes, descriptor_offset);
}

Fixture build_fixture() {
  const auto metadata = build_metadata();
  Fixture fixture;
  fixture.table_offset = align_up(fixture.metadata_offset + metadata.size(), evo2c::kModelAlignment);
  fixture.first_payload_offset = align_up(
      fixture.table_offset + 2 * evo2c::kTensorDescriptorSize, evo2c::kModelAlignment);
  fixture.second_payload_offset = align_up(fixture.first_payload_offset + 8, evo2c::kModelAlignment);
  fixture.bytes.resize(fixture.second_payload_offset + 8, 0);

  std::copy(metadata.begin(), metadata.end(),
            fixture.bytes.begin() + static_cast<std::ptrdiff_t>(fixture.metadata_offset));
  const std::vector<std::uint8_t> first_payload{0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40};
  const std::vector<std::uint8_t> second_payload{0x00, 0x00, 0x80, 0x3f,
                                                 0x00, 0x00, 0x00, 0x40};
  std::copy(first_payload.begin(), first_payload.end(),
            fixture.bytes.begin() + static_cast<std::ptrdiff_t>(fixture.first_payload_offset));
  std::copy(second_payload.begin(), second_payload.end(),
            fixture.bytes.begin() + static_cast<std::ptrdiff_t>(fixture.second_payload_offset));

  write_descriptor(fixture.bytes, fixture.table_offset, "embed.weight",
                   evo2c::TensorDType::kBF16, {2, 2}, fixture.first_payload_offset,
                   first_payload);
  write_descriptor(fixture.bytes, fixture.table_offset + evo2c::kTensorDescriptorSize,
                   "blocks.0.scale", evo2c::TensorDType::kF32, {2},
                   fixture.second_payload_offset, second_payload);

  std::copy_n("EVO2C", 5, fixture.bytes.begin());
  write_u32(fixture.bytes, 8, evo2c::kModelFormatVersion);
  write_u32(fixture.bytes, 12, 0x01020304U);
  write_u32(fixture.bytes, 16, evo2c::kModelHeaderSize);
  write_u64(fixture.bytes, 24, fixture.bytes.size());
  write_u64(fixture.bytes, 32, fixture.metadata_offset);
  write_u64(fixture.bytes, 40, metadata.size());
  write_u64(fixture.bytes, 48, fixture.table_offset);
  write_u64(fixture.bytes, 56, 2);
  write_u32(fixture.bytes, 64, evo2c::kTensorDescriptorSize);
  write_u32(fixture.bytes, 68, evo2c::kModelAlignment);
  write_u64(fixture.bytes, 72, fixture.first_payload_offset);
  set_header_crc(fixture.bytes);
  return fixture;
}

class TemporaryFile final {
 public:
  TemporaryFile() {
    auto pattern = (std::filesystem::temp_directory_path() / "evo2c-format-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int descriptor = ::mkstemp(writable.data());
    if (descriptor < 0) {
      std::perror("mkstemp");
      std::abort();
    }
    ::close(descriptor);
    path_ = writable.data();
  }

  ~TemporaryFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  TemporaryFile(const TemporaryFile&) = delete;
  TemporaryFile& operator=(const TemporaryFile&) = delete;

  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  std::string path_;
};

bool write_file(const std::string& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  return output.good();
}

void expect_failure(const std::vector<std::uint8_t>& bytes,
                    const std::string_view expected_message) {
  TemporaryFile file;
  check(write_file(file.path(), bytes), "write corrupted fixture");
  evo2c::ModelFile model;
  const auto status = model.open(file.path());
  check(!status.ok(), "corrupted model is rejected");
  check(status.code() == evo2c::ErrorCode::kModelFormat,
        "corrupted model returns model_format error");
  check(status.message().find(expected_message) != std::string::npos,
        std::string{"corruption error explains "} + std::string{expected_message});
}

void test_valid_fixture(const Fixture& fixture) {
  TemporaryFile file;
  check(write_file(file.path(), fixture.bytes), "write valid fixture");
  evo2c::ModelFile model;
  const auto status = model.open(file.path());
  check(status.ok(), std::string{"valid fixture loads: "} + status.message());
  if (!status.ok()) {
    return;
  }
  check(model.version() == 1, "format version parsed");
  check(model.file_size() == fixture.bytes.size(), "file size parsed");
  check(model.metadata().size() == 4, "all metadata entries parsed");
  check(model.tensors().size() == 2, "all tensor descriptors parsed");

  const auto* model_name = model.find_metadata("model.name");
  check(model_name != nullptr, "metadata lookup finds model.name");
  if (model_name != nullptr) {
    check(evo2c::metadata_value_text(*model_name) == "tiny-evo2",
          "string metadata value parsed");
  }
  const auto* tensor = model.find_tensor("embed.weight");
  check(tensor != nullptr, "tensor lookup finds embed.weight");
  if (tensor != nullptr) {
    check(tensor->dtype == evo2c::TensorDType::kBF16, "tensor dtype parsed");
    check(tensor->rank == 2 && tensor->dimensions[0] == 2 && tensor->dimensions[1] == 2,
          "tensor shape parsed");
    check(model.tensor_data(*tensor) != nullptr, "verified tensor payload is accessible");
  }
}

void test_corruption(const Fixture& fixture) {
  auto bytes = fixture.bytes;
  bytes[0] ^= 0xffU;
  expect_failure(bytes, "bad magic");

  bytes = fixture.bytes;
  bytes[kHeaderCrcOffset] ^= 0x01U;
  expect_failure(bytes, "header CRC32 mismatch");

  bytes = fixture.bytes;
  bytes.pop_back();
  expect_failure(bytes, "file_size does not match");

  bytes = fixture.bytes;
  bytes[fixture.metadata_offset + 24] ^= 0x01U;
  expect_failure(bytes, "metadata CRC32 mismatch");

  bytes = fixture.bytes;
  bytes[fixture.table_offset + 8] ^= 0x01U;
  expect_failure(bytes, "descriptor CRC32 mismatch");

  bytes = fixture.bytes;
  const auto second_descriptor = fixture.table_offset + evo2c::kTensorDescriptorSize;
  std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(second_descriptor),
            bytes.begin() + static_cast<std::ptrdiff_t>(second_descriptor + evo2c::kTensorNameCapacity),
            0);
  std::copy_n("embed.weight", 12,
              bytes.begin() + static_cast<std::ptrdiff_t>(second_descriptor));
  set_descriptor_crc(bytes, second_descriptor);
  expect_failure(bytes, "duplicate tensor name");

  bytes = fixture.bytes;
  write_u64(bytes, second_descriptor + 168, fixture.first_payload_offset);
  set_descriptor_crc(bytes, second_descriptor);
  expect_failure(bytes, "payloads overlap");

  bytes = fixture.bytes;
  write_u64(bytes, second_descriptor + 168, fixture.second_payload_offset + 1);
  set_descriptor_crc(bytes, second_descriptor);
  expect_failure(bytes, "offset/size/alignment is invalid");

  bytes = fixture.bytes;
  write_u64(bytes, second_descriptor + 104, 3);
  set_descriptor_crc(bytes, second_descriptor);
  expect_failure(bytes, "element_count does not match");

  bytes = fixture.bytes;
  bytes[second_descriptor + 97] = 2;
  write_u64(bytes, second_descriptor + 104, UINT64_MAX);
  write_u64(bytes, second_descriptor + 112, 2);
  set_descriptor_crc(bytes, second_descriptor);
  expect_failure(bytes, "dimensions are zero or overflow uint64");

  bytes = fixture.bytes;
  bytes[second_descriptor + 96] = 99;
  set_descriptor_crc(bytes, second_descriptor);
  expect_failure(bytes, "dtype is invalid");

  bytes = fixture.bytes;
  bytes[fixture.first_payload_offset] ^= 0x01U;
  expect_failure(bytes, "payload CRC32 mismatch");
}

}  // namespace

int main(const int argc, char** argv) {
  const auto fixture = build_fixture();
  if (argc == 3 && std::string_view{argv[1]} == "--write-fixture") {
    if (!write_file(argv[2], fixture.bytes)) {
      std::cerr << "failed to write fixture: " << argv[2] << '\n';
      return 1;
    }
    return 0;
  }
  if (argc != 1) {
    std::cerr << "usage: test_model_format [--write-fixture PATH]\n";
    return 2;
  }

  const std::array<std::uint8_t, 9> crc_vector{'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  check(evo2c::crc32(crc_vector.data(), crc_vector.size()) == 0xcbf43926U,
        "CRC32 matches IEEE reference vector");
  test_valid_fixture(fixture);
  test_corruption(fixture);

  if (failures != 0) {
    std::cerr << failures << " model format test(s) failed\n";
    return 1;
  }
  std::cout << "model format tests passed\n";
  return 0;
}
