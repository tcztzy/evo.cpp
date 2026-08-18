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
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "evo/model_format.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

void append_u64(std::vector<std::uint8_t> *const bytes,
                const std::uint64_t value) {
  for (std::size_t byte = 0; byte < 8; ++byte) {
    bytes->push_back(static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU));
  }
}

struct Fixture final {
  std::vector<std::uint8_t> bytes;
  std::size_t data_offset{0};
};

Fixture make_fixture(std::string header,
                     const std::vector<std::uint8_t> &payload) {
  while (header.size() % 8 != 0)
    header.push_back(' ');
  Fixture fixture;
  fixture.bytes.reserve(8 + header.size() + payload.size());
  append_u64(&fixture.bytes, header.size());
  fixture.bytes.insert(fixture.bytes.end(), header.begin(), header.end());
  fixture.data_offset = fixture.bytes.size();
  fixture.bytes.insert(fixture.bytes.end(), payload.begin(), payload.end());
  return fixture;
}

std::string metadata_json(const std::string_view profile = "s:evo2-runtime-v1",
                          const std::string_view extra = {}) {
  return "\"__metadata__\":{"
         "\"evo2.profile\":\"" +
         std::string{profile} +
         "\","
         "\"model.name\":\"s:tiny-evo2\","
         "\"model.architecture\":\"s:StripedHyena2\","
         "\"runtime.abi\":\"s:evo2-safetensors-v1\","
         "\"config.hidden_size\":\"u:4\","
         "\"config.tie_embeddings\":\"b:1\","
         "\"config.layers\":\"l:0,3\","
         "\"config.eps\":\"f:3eb0c6f7a0b5ed8d\","
         "\"fixture.opaque\":\"x:00ff\"" +
         std::string{extra} + "}";
}

Fixture valid_fixture() {
  const std::string header =
      "{" + metadata_json() +
      ",\"embed.weight\":{\"dtype\":\"BF16\",\"shape\":[2,2],"
      "\"data_offsets\":[0,8]},"
      "\"blocks.0.scale\":{\"dtype\":\"F32\",\"shape\":[2],"
      "\"data_offsets\":[8,16]}}";
  return make_fixture(header, {0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40,
                               0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40});
}

Fixture esmc_fixture() {
  const std::string header =
      "{\"__metadata__\":{"
      "\"runtime.profile\":\"s:esmc-runtime-v1\","
      "\"runtime.abi\":\"s:esmc-safetensors-v1\","
      "\"model.id\":\"s:esmc_300m\","
      "\"model.architecture\":\"s:ESMC\"},"
      "\"embedding_layer.weight\":{"
      "\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
  return make_fixture(header, {0, 0, 0, 0});
}

Fixture tokenizer_descriptor_fixture(const std::string_view extra) {
  const std::string header =
      "{" + metadata_json("s:evo2-runtime-v1", extra) +
      ",\"embed.weight\":{\"dtype\":\"F32\",\"shape\":[1],"
      "\"data_offsets\":[0,4]}}";
  return make_fixture(header, {0, 0, 0, 0});
}

constexpr std::string_view kTokenizerDescriptor =
    ",\"tokenizer.profile\":\"s:evo-tokenizer-v1\""
    ",\"tokenizer.path\":\"s:assets/tokenizer.evo.json\""
    ",\"tokenizer.sha256\":\"s:"
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\""
    ",\"tokenizer.size\":\"u:123\"";

class TemporaryFile final {
public:
  TemporaryFile() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "evo-safetensors-XXXXXX")
            .string();
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

  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;

  [[nodiscard]] const std::string &path() const noexcept { return path_; }

private:
  std::string path_;
};

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    auto pattern =
        (std::filesystem::temp_directory_path() / "evo-model-index-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const auto *const created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      std::perror("mkdtemp");
      std::abort();
    }
    path_ = created;
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

bool write_file(const std::string &path,
                const std::vector<std::uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  return output.good();
}

bool write_text(const std::filesystem::path &path,
                const std::string_view text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  return output.good();
}

void expect_failure(const Fixture &fixture,
                    const std::string_view expected_message) {
  TemporaryFile file;
  check(write_file(file.path(), fixture.bytes), "write invalid fixture");
  evo::ModelFile model;
  const auto status = model.open(file.path());
  check(!status.ok(), "invalid Safetensors file is rejected");
  check(status.code() == evo::ErrorCode::kModelFormat,
        "invalid Safetensors file returns model_format");
  check(status.message().find(expected_message) != std::string::npos,
        std::string{"failure explains "} + std::string{expected_message});
}

void test_valid_fixture(const Fixture &fixture) {
  TemporaryFile file;
  check(write_file(file.path(), fixture.bytes), "write valid fixture");
  evo::ModelFile model;
  const auto status = model.open(file.path());
  check(status.ok(), std::string{"valid fixture loads: "} + status.message());
  if (!status.ok())
    return;

  check(model.format_name() == "SAFETENSORS", "format is Safetensors");
  check(model.profile() == "evo2-runtime-v1", "runtime profile is exposed");
  check(model.file_size() == fixture.bytes.size(), "file size is exposed");
  check(model.metadata().size() == 9, "typed metadata entries are decoded");
  check(model.tensors().size() == 2, "tensor descriptors are parsed");

  const auto *const model_name = model.find_metadata("model.name");
  check(model_name != nullptr &&
            evo::metadata_value_text(*model_name) == "tiny-evo2",
        "string metadata is decoded");
  const auto *const layers = model.find_metadata("config.layers");
  check(layers != nullptr && evo::metadata_value_text(*layers) == "[0,3]",
        "list metadata is decoded");

  const auto *const tensor = model.find_tensor("embed.weight");
  check(tensor != nullptr, "tensor lookup succeeds");
  if (tensor == nullptr)
    return;
  check(tensor->dtype == evo::TensorDType::kBF16, "tensor dtype is decoded");
  check(tensor->rank == 2 && tensor->dimensions[0] == 2 &&
            tensor->dimensions[1] == 2,
        "tensor shape is decoded");
  check(tensor->data_offset == fixture.data_offset,
        "relative Safetensors offset becomes a mapped file offset");
  check(model.tensor_data(*tensor) != nullptr,
        "mapped tensor payload is directly addressable");
  std::array<std::uint8_t, 3> range{};
  auto read_status = model.read_tensor(*tensor, 2, range.data(), range.size());
  check(read_status.ok(), "bounded tensor reads succeed");
  check(std::equal(range.begin(), range.end(),
                   fixture.bytes.begin() +
                       static_cast<std::ptrdiff_t>(tensor->data_offset + 2)),
        "bounded tensor reads are byte exact");
  read_status = model.read_tensor(*tensor, tensor->data_size, range.data(), 1);
  check(!read_status.ok(), "out-of-range tensor reads are rejected");
}

void test_tokenizer_descriptor() {
  TemporaryFile file;
  const auto fixture = tokenizer_descriptor_fixture(kTokenizerDescriptor);
  check(write_file(file.path(), fixture.bytes),
        "write tokenizer descriptor fixture");
  evo::ModelFile model;
  const auto status = model.open(file.path());
  check(status.ok(),
        std::string{"tokenizer descriptor fixture loads: "} + status.message());
  if (!status.ok())
    return;
  const auto expected_root =
      std::filesystem::path{file.path()}.parent_path().string();
  check(model.artifact_root() == expected_root,
        "ModelFile exposes the entry path parent as artifact root");
  const auto &descriptor = model.tokenizer_asset_descriptor();
  check(descriptor.has_value(),
        "ModelFile exposes an optional tokenizer descriptor");
  if (descriptor.has_value()) {
    check(descriptor->path == "assets/tokenizer.evo.json" &&
              descriptor->size == 123 &&
              descriptor->sha256 == "0123456789abcdef0123456789abcdef0123456789"
                                    "abcdef0123456789abcdef",
          "ModelFile decodes all strict tokenizer descriptor fields");
  }
}

void test_index_artifact_root() {
  TemporaryDirectory temporary;
  const auto shard = tokenizer_descriptor_fixture(kTokenizerDescriptor);
  const auto shard_path = temporary.path() / "weights.safetensors";
  check(write_file(shard_path.string(), shard.bytes),
        "write indexed tokenizer descriptor shard");
  const auto index_path = temporary.path() / "model.safetensors.index.json";
  check(
      write_text(
          index_path,
          R"({"metadata":{"total_size":4},"weight_map":{"embed.weight":"weights.safetensors"}})"),
      "write tokenizer descriptor index");
  evo::ModelFile model;
  const auto status = model.open(index_path.string());
  check(status.ok(),
        std::string{"indexed tokenizer fixture loads: "} + status.message());
  if (!status.ok())
    return;
  check(model.artifact_root() == temporary.path().string(),
        "indexed ModelFile exposes the index path parent as artifact root");
  check(model.tokenizer_asset_descriptor().has_value(),
        "indexed ModelFile preserves tokenizer descriptor metadata");
}

void test_converter_profile_contract() {
  constexpr std::string_view digest =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  TemporaryFile file;
  const auto valid = tokenizer_descriptor_fixture(
      std::string{",\"source.converter_profile_contract_sha256\":\"s:"} +
      std::string{digest} + "\"");
  check(write_file(file.path(), valid.bytes),
        "write converter profile contract fixture");
  evo::ModelFile model;
  const auto status = model.open(file.path());
  check(status.ok(), std::string{"converter profile contract loads: "} +
                         status.message());

  expect_failure(
      tokenizer_descriptor_fixture(
          std::string{",\"source.converter_manifest_sha256\":\"s:"} +
          std::string{digest} + "\""),
      "legacy whole-profile converter digest");
  expect_failure(
      tokenizer_descriptor_fixture(
          ",\"source.converter_profile_contract_sha256\":\"s:ABC\""),
      "canonical lowercase hex");
  expect_failure(
      tokenizer_descriptor_fixture(
          ",\"source.converter_profile_contract_sha256\":\"u:1\""),
      "canonical lowercase hex");
}

void test_invalid_files() {
  auto fixture = valid_fixture();
  fixture.bytes.resize(7);
  expect_failure(fixture, "file is too short");

  fixture = valid_fixture();
  fixture.bytes[0] = 7;
  std::fill(fixture.bytes.begin() + 1, fixture.bytes.begin() + 8, 0);
  expect_failure(fixture, "header size");

  expect_failure(make_fixture("{", {0}), "invalid root object entry");

  expect_failure(make_fixture("{\"__metadata__\":{\"model.name\":\"s:tiny\"},"
                              "\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
                              "\"data_offsets\":[0,4]}}",
                              {0, 0, 0, 0}),
                 "exactly one registered artifact profile");

  expect_failure(make_fixture("{" + metadata_json() +
                                  ",\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
                                  "\"data_offsets\":[0,4]},"
                                  "\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
                                  "\"data_offsets\":[4,8]}}",
                              std::vector<std::uint8_t>(8)),
                 "duplicate root key");

  expect_failure(make_fixture("{" + metadata_json() +
                                  ",\"x\":{\"dtype\":\"F16\",\"shape\":[1],"
                                  "\"data_offsets\":[0,2]}}",
                              std::vector<std::uint8_t>(2)),
                 "unsupported dtype");

  expect_failure(make_fixture("{" + metadata_json() +
                                  ",\"x\":{\"dtype\":\"F32\",\"shape\":[2],"
                                  "\"data_offsets\":[0,4]}}",
                              std::vector<std::uint8_t>(4)),
                 "dtype/shape size mismatch");

  expect_failure(make_fixture("{" + metadata_json() +
                                  ",\"a\":{\"dtype\":\"F32\",\"shape\":[1],"
                                  "\"data_offsets\":[0,4]},"
                                  "\"b\":{\"dtype\":\"F32\",\"shape\":[1],"
                                  "\"data_offsets\":[8,12]}}",
                              std::vector<std::uint8_t>(12)),
                 "hole or overlap");

  expect_failure(make_fixture("{" + metadata_json() +
                                  ",\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
                                  "\"data_offsets\":[0,4]}}",
                              std::vector<std::uint8_t>(5)),
                 "does not cover the complete file");

  expect_failure(make_fixture("{" + metadata_json("plain-text") +
                                  ",\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
                                  "\"data_offsets\":[0,4]}}",
                              std::vector<std::uint8_t>(4)),
                 "not typed");

  expect_failure(tokenizer_descriptor_fixture(
                     ",\"tokenizer.profile\":\"s:evo-tokenizer-v1\""),
                 "incomplete tokenizer descriptor");
  expect_failure(
      tokenizer_descriptor_fixture(
          ",\"tokenizer.path\":\"s:tokenizer.json\""),
      "incomplete tokenizer descriptor");
  expect_failure(
      tokenizer_descriptor_fixture(
          ",\"tokenizer.profile\":\"s:evo-tokenizer-v1\""
          ",\"tokenizer.path\":\"s:tokenizer.json\""
          ",\"tokenizer.sha256\":\"s:"
          "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\""
          ",\"tokenizer.size\":\"u:1\""
          ",\"tokenizer.unknown\":\"s:value\""),
      "unknown tokenizer descriptor");
  expect_failure(
      tokenizer_descriptor_fixture(
          ",\"tokenizer.profile\":\"s:evo-tokenizer-v1\""
          ",\"tokenizer.path\":\"s:../tokenizer.json\""
          ",\"tokenizer.sha256\":\"s:"
          "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\""
          ",\"tokenizer.size\":\"u:1\""),
      "canonical relative path");
  expect_failure(
      tokenizer_descriptor_fixture(
          ",\"tokenizer.profile\":\"s:unknown-tokenizer\""
          ",\"tokenizer.path\":\"s:tokenizer.json\""
          ",\"tokenizer.sha256\":\"s:"
          "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\""
          ",\"tokenizer.size\":\"u:1\""),
      "tokenizer.profile");
}

} // namespace

int main(const int argc, char **argv) {
  const auto fixture = valid_fixture();
  if (argc == 3 && std::string_view{argv[1]} == "--write-fixture") {
    if (!write_file(argv[2], fixture.bytes)) {
      std::cerr << "failed to write fixture: " << argv[2] << '\n';
      return 1;
    }
    return 0;
  }
  if (argc == 3 && std::string_view{argv[1]} == "--write-esmc-fixture") {
    if (!write_file(argv[2], esmc_fixture().bytes)) {
      std::cerr << "failed to write ESMC fixture: " << argv[2] << '\n';
      return 1;
    }
    return 0;
  }
  if (argc != 1) {
    std::cerr << "usage: test_model_format "
                 "[--write-fixture|--write-esmc-fixture PATH]\n";
    return 2;
  }

  test_valid_fixture(fixture);
  test_tokenizer_descriptor();
  test_index_artifact_root();
  test_converter_profile_contract();
  test_invalid_files();
  if (failures != 0) {
    std::cerr << failures << " model format test(s) failed\n";
    return 1;
  }
  std::cout << "model format tests passed\n";
  return 0;
}
