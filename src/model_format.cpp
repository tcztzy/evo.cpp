// SPDX-License-Identifier: Apache-2.0
#include "evo/model_format.hpp"

#include "evo/model_registry.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace evo {
namespace {

constexpr std::size_t kSafetensorsPrefixSize = 8;
constexpr std::size_t kMaximumMetadataEntries = 4096;
constexpr std::size_t kMaximumTensorCount = 1'000'000;
constexpr std::size_t kMaximumMetadataValueSize = 16U * 1024U * 1024U;
constexpr std::size_t kMaximumIndexSize = 64U * 1024U * 1024U;

struct RawTensor final {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "Safetensors: " + message};
}

Status index_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "Safetensors index: " + message};
}

bool has_suffix(const std::string_view value,
                const std::string_view suffix) noexcept {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

std::uint64_t read_u64(const std::uint8_t *const data) noexcept {
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    value |= static_cast<std::uint64_t>(data[byte]) << (byte * 8U);
  }
  return value;
}

void append_u64(std::vector<std::uint8_t> *const output,
                const std::uint64_t value) {
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    output->push_back(
        static_cast<std::uint8_t>((value >> (byte * 8U)) & 0xffU));
  }
}

bool checked_add(const std::uint64_t left, const std::uint64_t right,
                 std::uint64_t *const output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  *output = left + right;
  return true;
}

bool checked_mul(const std::uint64_t left, const std::uint64_t right,
                 std::uint64_t *const output) noexcept {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *output = left * right;
  return true;
}

bool valid_key_character(const char character) noexcept {
  const auto value = static_cast<unsigned char>(character);
  return std::isalnum(value) != 0 || character == '.' || character == '_' ||
         character == '-';
}

bool valid_identifier(const std::string_view value,
                      const std::size_t maximum) noexcept {
  return !value.empty() && value.size() <= maximum &&
         std::all_of(value.begin(), value.end(), valid_key_character);
}

bool parse_decimal_u64(const std::string_view text,
                       std::uint64_t *const output) noexcept {
  if (text.empty() || (text.size() > 1 && text.front() == '0')) {
    return false;
  }
  std::uint64_t value = 0;
  for (const char character : text) {
    if (character < '0' || character > '9') {
      return false;
    }
    const auto digit = static_cast<std::uint64_t>(character - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
      return false;
    }
    value = value * 10U + digit;
  }
  *output = value;
  return true;
}

int hex_digit(const char character) noexcept {
  if (character >= '0' && character <= '9') {
    return character - '0';
  }
  if (character >= 'a' && character <= 'f') {
    return 10 + character - 'a';
  }
  return -1;
}

bool parse_hex_u64(const std::string_view text,
                   std::uint64_t *const output) noexcept {
  if (text.size() != 16) {
    return false;
  }
  std::uint64_t value = 0;
  for (const char character : text) {
    const int digit = hex_digit(character);
    if (digit < 0) {
      return false;
    }
    value = (value << 4U) | static_cast<std::uint64_t>(digit);
  }
  *output = value;
  return true;
}

Status decode_metadata(const std::string &key, const std::string &encoded,
                       MetadataEntry *const output) {
  if (!valid_identifier(key, 255)) {
    return format_error("invalid metadata key '" + key + "'");
  }
  if (encoded.size() < 2 || encoded[1] != ':') {
    return format_error("metadata '" + key +
                        "' is not typed by a registered runtime profile");
  }
  MetadataEntry entry;
  entry.key = key;
  const std::string_view value{encoded.data() + 2, encoded.size() - 2};
  switch (encoded[0]) {
  case 's':
    entry.type = MetadataType::kString;
    entry.value.assign(value.begin(), value.end());
    break;
  case 'u': {
    std::uint64_t parsed = 0;
    if (!parse_decimal_u64(value, &parsed)) {
      return format_error("metadata '" + key + "' has invalid u64");
    }
    entry.type = MetadataType::kU64;
    append_u64(&entry.value, parsed);
    break;
  }
  case 'f': {
    std::uint64_t bits = 0;
    if (!parse_hex_u64(value, &bits)) {
      return format_error("metadata '" + key + "' has invalid f64 bits");
    }
    entry.type = MetadataType::kF64;
    append_u64(&entry.value, bits);
    break;
  }
  case 'b':
    if (value != "0" && value != "1") {
      return format_error("metadata '" + key + "' has invalid bool");
    }
    entry.type = MetadataType::kBool;
    entry.value.push_back(static_cast<std::uint8_t>(value == "1"));
    break;
  case 'l': {
    entry.type = MetadataType::kU64List;
    std::size_t begin = 0;
    while (begin < value.size()) {
      const auto comma = value.find(',', begin);
      const auto end = comma == std::string_view::npos ? value.size() : comma;
      std::uint64_t item = 0;
      if (!parse_decimal_u64(value.substr(begin, end - begin), &item)) {
        return format_error("metadata '" + key + "' has invalid u64 list");
      }
      append_u64(&entry.value, item);
      if (comma == std::string_view::npos) {
        break;
      }
      begin = comma + 1;
      if (begin == value.size()) {
        return format_error("metadata '" + key + "' has invalid u64 list");
      }
    }
    break;
  }
  case 'x':
    if (value.size() % 2 != 0) {
      return format_error("metadata '" + key + "' has invalid bytes");
    }
    entry.type = MetadataType::kBytes;
    entry.value.reserve(value.size() / 2);
    for (std::size_t offset = 0; offset < value.size(); offset += 2) {
      const int high = hex_digit(value[offset]);
      const int low = hex_digit(value[offset + 1]);
      if (high < 0 || low < 0) {
        return format_error("metadata '" + key + "' has invalid bytes");
      }
      entry.value.push_back(static_cast<std::uint8_t>((high << 4) | low));
    }
    break;
  default:
    return format_error("metadata '" + key + "' has unknown type prefix");
  }
  if (entry.value.size() > kMaximumMetadataValueSize) {
    return format_error("metadata '" + key + "' exceeds 16 MiB");
  }
  *output = std::move(entry);
  return Status::Ok();
}

class HeaderParser final {
public:
  HeaderParser(const std::uint8_t *const data, const std::size_t size)
      : begin_(reinterpret_cast<const char *>(data)), cursor_(begin_),
        end_(begin_ + size) {}

  [[nodiscard]] Status
  parse(std::vector<std::pair<std::string, std::string>> *const metadata,
        std::vector<RawTensor> *const tensors) {
    skip_whitespace();
    if (!take('{')) {
      return error("header must begin with a JSON object");
    }
    std::set<std::string> root_keys;
    bool has_metadata = false;
    skip_whitespace();
    if (take('}')) {
      return error("root object must not be empty");
    }
    while (true) {
      std::string name;
      if (!parse_string(&name) || !take_after_whitespace(':')) {
        return error("invalid root object entry");
      }
      if (!root_keys.insert(name).second) {
        return error("duplicate root key '" + name + "'");
      }
      if (name == "__metadata__") {
        if (!parse_metadata(metadata)) {
          return error("invalid __metadata__ object");
        }
        has_metadata = true;
      } else {
        RawTensor tensor;
        tensor.name = std::move(name);
        if (!parse_tensor(&tensor)) {
          return error("invalid tensor descriptor");
        }
        tensors->push_back(std::move(tensor));
      }
      skip_whitespace();
      if (take('}')) {
        break;
      }
      if (!take(',')) {
        return error("expected ',' or '}' in root object");
      }
    }
    skip_whitespace();
    if (cursor_ != end_) {
      return error("header has non-whitespace trailing bytes");
    }
    if (!has_metadata) {
      return error("missing __metadata__ object");
    }
    return Status::Ok();
  }

private:
  void skip_whitespace() noexcept {
    while (cursor_ != end_ && (*cursor_ == ' ' || *cursor_ == '\t' ||
                               *cursor_ == '\n' || *cursor_ == '\r')) {
      ++cursor_;
    }
  }

  bool take(const char expected) noexcept {
    if (cursor_ == end_ || *cursor_ != expected) {
      return false;
    }
    ++cursor_;
    return true;
  }

  bool take_after_whitespace(const char expected) noexcept {
    skip_whitespace();
    const bool found = take(expected);
    skip_whitespace();
    return found;
  }

  bool parse_string(std::string *const output) {
    skip_whitespace();
    if (!take('"')) {
      return false;
    }
    output->clear();
    while (cursor_ != end_) {
      const auto character = static_cast<unsigned char>(*cursor_++);
      if (character == '"') {
        skip_whitespace();
        return true;
      }
      if (character < 0x20U) {
        return false;
      }
      if (character != '\\') {
        output->push_back(static_cast<char>(character));
        continue;
      }
      if (cursor_ == end_) {
        return false;
      }
      const char escaped = *cursor_++;
      switch (escaped) {
      case '"':
      case '\\':
      case '/':
        output->push_back(escaped);
        break;
      case 'b':
        output->push_back('\b');
        break;
      case 'f':
        output->push_back('\f');
        break;
      case 'n':
        output->push_back('\n');
        break;
      case 'r':
        output->push_back('\r');
        break;
      case 't':
        output->push_back('\t');
        break;
      default:
        return false;
      }
    }
    return false;
  }

  bool parse_u64(std::uint64_t *const output) {
    skip_whitespace();
    const char *const start = cursor_;
    while (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9') {
      ++cursor_;
    }
    if (cursor_ == start ||
        !parse_decimal_u64({start, static_cast<std::size_t>(cursor_ - start)},
                           output)) {
      return false;
    }
    skip_whitespace();
    return true;
  }

  bool parse_u64_array(std::vector<std::uint64_t> *const values) {
    if (!take_after_whitespace('[')) {
      return false;
    }
    values->clear();
    if (take(']')) {
      skip_whitespace();
      return true;
    }
    while (true) {
      std::uint64_t value = 0;
      if (!parse_u64(&value)) {
        return false;
      }
      values->push_back(value);
      if (take(']')) {
        skip_whitespace();
        return true;
      }
      if (!take_after_whitespace(',')) {
        return false;
      }
    }
  }

  bool parse_metadata(
      std::vector<std::pair<std::string, std::string>> *const metadata) {
    if (!take_after_whitespace('{')) {
      return false;
    }
    std::set<std::string> keys;
    if (take('}')) {
      skip_whitespace();
      return true;
    }
    while (true) {
      std::string key;
      std::string value;
      if (!parse_string(&key) || !take_after_whitespace(':') ||
          !parse_string(&value) || !keys.insert(key).second) {
        return false;
      }
      metadata->emplace_back(std::move(key), std::move(value));
      if (take('}')) {
        skip_whitespace();
        return true;
      }
      if (!take_after_whitespace(',')) {
        return false;
      }
    }
  }

  bool parse_tensor(RawTensor *const tensor) {
    if (!take_after_whitespace('{')) {
      return false;
    }
    bool has_dtype = false;
    bool has_shape = false;
    bool has_offsets = false;
    if (take('}')) {
      return false;
    }
    while (true) {
      std::string field;
      if (!parse_string(&field) || !take_after_whitespace(':')) {
        return false;
      }
      if (field == "dtype" && !has_dtype) {
        if (!parse_string(&tensor->dtype)) {
          return false;
        }
        has_dtype = true;
      } else if (field == "shape" && !has_shape) {
        if (!parse_u64_array(&tensor->shape)) {
          return false;
        }
        has_shape = true;
      } else if (field == "data_offsets" && !has_offsets) {
        std::vector<std::uint64_t> offsets;
        if (!parse_u64_array(&offsets) || offsets.size() != 2) {
          return false;
        }
        tensor->begin = offsets[0];
        tensor->end = offsets[1];
        has_offsets = true;
      } else {
        return false;
      }
      if (take('}')) {
        skip_whitespace();
        return has_dtype && has_shape && has_offsets;
      }
      if (!take_after_whitespace(',')) {
        return false;
      }
    }
  }

  Status error(const std::string &detail) const {
    const auto offset = static_cast<std::size_t>(cursor_ - begin_);
    return format_error(detail + " at header byte " + std::to_string(offset));
  }

  const char *begin_;
  const char *cursor_;
  const char *end_;
};

class IndexParser final {
public:
  IndexParser(const std::uint8_t *const data, const std::size_t size)
      : begin_(reinterpret_cast<const char *>(data)), cursor_(begin_),
        end_(begin_ + size) {}

  [[nodiscard]] Status
  parse(std::uint64_t *const total_size,
        std::vector<std::pair<std::string, std::string>> *const weight_map) {
    skip_whitespace();
    if (!take('{')) {
      return error("root must be a JSON object");
    }
    std::set<std::string> root_keys;
    bool has_metadata = false;
    bool has_weight_map = false;
    skip_whitespace();
    if (take('}')) {
      return error("root object must not be empty");
    }
    while (true) {
      std::string field;
      if (!parse_string(&field) || !take_after_whitespace(':') ||
          !root_keys.insert(field).second) {
        return error("invalid or duplicate root field");
      }
      if (field == "metadata") {
        if (has_metadata || !parse_metadata(total_size)) {
          return error("invalid metadata object");
        }
        has_metadata = true;
      } else if (field == "weight_map") {
        if (has_weight_map || !parse_weight_map(weight_map)) {
          return error("invalid weight_map object");
        }
        has_weight_map = true;
      } else {
        return error("unknown root field '" + field + "'");
      }
      if (take('}')) {
        break;
      }
      if (!take_after_whitespace(',')) {
        return error("expected ',' or '}' in root object");
      }
    }
    skip_whitespace();
    if (cursor_ != end_) {
      return error("non-whitespace trailing bytes");
    }
    if (!has_metadata || !has_weight_map) {
      return error("metadata and weight_map are required");
    }
    return Status::Ok();
  }

private:
  void skip_whitespace() noexcept {
    while (cursor_ != end_ && (*cursor_ == ' ' || *cursor_ == '\t' ||
                               *cursor_ == '\n' || *cursor_ == '\r')) {
      ++cursor_;
    }
  }

  bool take(const char expected) noexcept {
    if (cursor_ == end_ || *cursor_ != expected) {
      return false;
    }
    ++cursor_;
    return true;
  }

  bool take_after_whitespace(const char expected) noexcept {
    skip_whitespace();
    const bool found = take(expected);
    skip_whitespace();
    return found;
  }

  bool parse_string(std::string *const output) {
    skip_whitespace();
    if (!take('"')) {
      return false;
    }
    output->clear();
    while (cursor_ != end_) {
      const auto character = static_cast<unsigned char>(*cursor_++);
      if (character == '"') {
        skip_whitespace();
        return true;
      }
      if (character < 0x20U) {
        return false;
      }
      if (character != '\\') {
        output->push_back(static_cast<char>(character));
        continue;
      }
      if (cursor_ == end_) {
        return false;
      }
      const char escaped = *cursor_++;
      if (escaped == '"' || escaped == '\\' || escaped == '/') {
        output->push_back(escaped);
      } else {
        return false;
      }
    }
    return false;
  }

  bool parse_u64(std::uint64_t *const output) {
    skip_whitespace();
    const char *const start = cursor_;
    while (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9') {
      ++cursor_;
    }
    if (cursor_ == start ||
        !parse_decimal_u64({start, static_cast<std::size_t>(cursor_ - start)},
                           output)) {
      return false;
    }
    skip_whitespace();
    return true;
  }

  bool parse_metadata(std::uint64_t *const total_size) {
    if (!take_after_whitespace('{')) {
      return false;
    }
    std::string field;
    if (!parse_string(&field) || field != "total_size" ||
        !take_after_whitespace(':') || !parse_u64(total_size) || !take('}')) {
      return false;
    }
    skip_whitespace();
    return true;
  }

  bool parse_weight_map(
      std::vector<std::pair<std::string, std::string>> *const weight_map) {
    if (!take_after_whitespace('{')) {
      return false;
    }
    std::set<std::string> names;
    if (take('}')) {
      skip_whitespace();
      return true;
    }
    while (true) {
      std::string name;
      std::string shard;
      if (!parse_string(&name) || !take_after_whitespace(':') ||
          !parse_string(&shard) || !names.insert(name).second) {
        return false;
      }
      weight_map->emplace_back(std::move(name), std::move(shard));
      if (take('}')) {
        skip_whitespace();
        return true;
      }
      if (!take_after_whitespace(',')) {
        return false;
      }
    }
  }

  Status error(const std::string &detail) const {
    const auto offset = static_cast<std::size_t>(cursor_ - begin_);
    return index_error(detail + " at byte " + std::to_string(offset));
  }

  const char *begin_;
  const char *cursor_;
  const char *end_;
};

bool tensor_dtype(const std::string_view name, TensorDType *const dtype,
                  std::uint64_t *const width) noexcept {
  if (name == "F32") {
    *dtype = TensorDType::kF32;
    *width = 4;
    return true;
  }
  if (name == "BF16") {
    *dtype = TensorDType::kBF16;
    *width = 2;
    return true;
  }
  if (name == "F8_E4M3") {
    *dtype = TensorDType::kE4M3Software;
    *width = 1;
    return true;
  }
  return false;
}

bool metadata_equal(const std::vector<MetadataEntry> &left,
                    const std::vector<MetadataEntry> &right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (left[index].key != right[index].key ||
        left[index].type != right[index].type ||
        left[index].value != right[index].value) {
      return false;
    }
  }
  return true;
}

bool valid_shard_filename(const std::string_view name) {
  if (name.empty() || !has_suffix(name, ".safetensors") ||
      name.find('/') != std::string_view::npos ||
      name.find('\\') != std::string_view::npos || name == "." ||
      name == "..") {
    return false;
  }
  return true;
}

std::string sibling_path(const std::string &path,
                         const std::string_view filename) {
  const auto separator = path.find_last_of("/\\");
  if (separator == std::string::npos) {
    return std::string{filename};
  }
  return path.substr(0, separator + 1) + std::string{filename};
}

bool canonical_sha256(const std::string_view value) noexcept {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool valid_tokenizer_asset_path(const std::string_view value) {
  if (value.empty() || value.find('\\') != std::string_view::npos)
    return false;
  const std::filesystem::path path{value};
  if (path.is_absolute() || path.has_root_name() || path.has_root_directory())
    return false;
  for (const auto &component : path) {
    if (component.empty() || component == "." || component == "..")
      return false;
  }
  return path.generic_string() == value;
}

std::string metadata_string(const MetadataEntry &entry) {
  if (entry.value.empty())
    return {};
  return {reinterpret_cast<const char *>(entry.value.data()),
          entry.value.size()};
}

} // namespace

Status ModelFile::open(const std::string &path) {
  ModelFile candidate;
  const auto parent = std::filesystem::path{path}.parent_path();
  candidate.artifact_root_ = parent.empty() ? "." : parent.string();
  if (has_suffix(path, ".index.json")) {
    auto status = candidate.open_index(path);
    if (!status.ok()) {
      return status;
    }
    status = candidate.parse_tokenizer_asset_descriptor();
    if (!status.ok())
      return status;
    status = candidate.parse_converter_profile_contract();
    if (!status.ok())
      return status;
    *this = std::move(candidate);
    return Status::Ok();
  }

  candidate.mappings_.emplace_back();
  auto status = candidate.mappings_.back().open(path);
  if (!status.ok()) {
    return status;
  }
  candidate.file_size_ = candidate.mappings_.back().size();
  status = candidate.parse_shard(0);
  if (!status.ok()) {
    return status;
  }
  status = candidate.parse_tokenizer_asset_descriptor();
  if (!status.ok())
    return status;
  status = candidate.parse_converter_profile_contract();
  if (!status.ok())
    return status;
  *this = std::move(candidate);
  return Status::Ok();
}

Status ModelFile::open_index(const std::string &path) {
  MappedFile index;
  auto status = index.open(path);
  if (!status.ok()) {
    return status;
  }
  if (index.size() > kMaximumIndexSize) {
    return index_error("file exceeds 64 MiB");
  }

  std::uint64_t expected_payload_size = 0;
  std::vector<std::pair<std::string, std::string>> weight_map;
  IndexParser parser(index.data(), index.size());
  status = parser.parse(&expected_payload_size, &weight_map);
  if (!status.ok()) {
    return status;
  }
  if (expected_payload_size == 0 || weight_map.empty() ||
      weight_map.size() > kMaximumTensorCount) {
    return index_error("weight_map count and total_size must be positive");
  }

  std::vector<std::string> shard_names;
  std::map<std::string, std::size_t> shard_indices;
  for (const auto &[tensor_name, shard_name] : weight_map) {
    if (!valid_identifier(tensor_name, kTensorNameCapacity - 1)) {
      return index_error("invalid tensor name '" + tensor_name + "'");
    }
    if (!valid_shard_filename(shard_name)) {
      return index_error("invalid shard filename '" + shard_name + "'");
    }
    if (shard_indices.find(shard_name) == shard_indices.end()) {
      const auto shard_index = shard_names.size();
      shard_indices.emplace(shard_name, shard_index);
      shard_names.push_back(shard_name);
    }
  }

  mappings_.reserve(shard_names.size());
  if (index.size() > std::numeric_limits<std::size_t>::max() - file_size_) {
    return index_error("artifact size overflows this process");
  }
  file_size_ += index.size();
  for (const auto &shard_name : shard_names) {
    mappings_.emplace_back();
    status = mappings_.back().open(sibling_path(path, shard_name));
    if (!status.ok()) {
      return {status.code(), "open shard '" + shard_name +
                                 "' from index: " + status.message()};
    }
    if (mappings_.back().size() >
        std::numeric_limits<std::size_t>::max() - file_size_) {
      return index_error("artifact size overflows this process");
    }
    file_size_ += mappings_.back().size();
    status = parse_shard(mappings_.size() - 1);
    if (!status.ok()) {
      return {status.code(), "shard '" + shard_name + "': " + status.message()};
    }
  }

  std::map<std::string, std::string> expected_shards;
  for (const auto &[tensor_name, shard_name] : weight_map) {
    if (!expected_shards.emplace(tensor_name, shard_name).second) {
      return index_error("duplicate tensor name '" + tensor_name + "'");
    }
  }
  if (expected_shards.size() != tensors_.size()) {
    return index_error("weight_map does not match shard tensor count");
  }
  std::uint64_t actual_payload_size = 0;
  for (const auto &tensor : tensors_) {
    const auto expected = expected_shards.find(tensor.name);
    if (expected == expected_shards.end() ||
        expected->second != shard_names[tensor.shard_index]) {
      return index_error("weight_map points tensor '" + tensor.name +
                         "' to the wrong shard");
    }
    if (!checked_add(actual_payload_size, tensor.data_size,
                     &actual_payload_size)) {
      return index_error("tensor payload size overflows uint64");
    }
  }
  if (actual_payload_size != expected_payload_size) {
    return index_error("metadata.total_size does not match tensor payloads");
  }
  return Status::Ok();
}

Status ModelFile::parse_shard(const std::size_t shard_index) {
  if (shard_index >= mappings_.size()) {
    return format_error("internal shard index is out of range");
  }
  const auto &mapping = mappings_[shard_index];
  if (mapping.size() < kSafetensorsPrefixSize + 2) {
    return format_error("file is too short");
  }
  const std::uint64_t header_size = read_u64(mapping.data());
  if (header_size == 0 || header_size > kMaximumSafetensorsHeaderSize ||
      header_size % 8 != 0) {
    return format_error("header size must be a nonzero 8-byte multiple at "
                        "most 16 MiB");
  }
  std::uint64_t data_offset = 0;
  if (!checked_add(kSafetensorsPrefixSize, header_size, &data_offset) ||
      data_offset > mapping.size()) {
    return format_error("header range lies beyond the file");
  }

  std::vector<std::pair<std::string, std::string>> raw_metadata;
  std::vector<RawTensor> raw_tensors;
  HeaderParser parser(mapping.data() + kSafetensorsPrefixSize,
                      static_cast<std::size_t>(header_size));
  auto status = parser.parse(&raw_metadata, &raw_tensors);
  if (!status.ok()) {
    return status;
  }
  if (raw_metadata.size() > kMaximumMetadataEntries) {
    return format_error("metadata entry count exceeds 4096");
  }
  if (raw_tensors.empty() || raw_tensors.size() > kMaximumTensorCount) {
    return format_error("tensor count must be in [1, 1000000]");
  }

  std::vector<MetadataEntry> shard_metadata;
  shard_metadata.reserve(raw_metadata.size());
  for (const auto &[key, value] : raw_metadata) {
    MetadataEntry entry;
    status = decode_metadata(key, value, &entry);
    if (!status.ok()) {
      return status;
    }
    shard_metadata.push_back(std::move(entry));
  }
  std::sort(shard_metadata.begin(), shard_metadata.end(),
            [](const MetadataEntry &left, const MetadataEntry &right) {
              return left.key < right.key;
            });
  const auto find_profile = [&](const std::string_view key) {
    return std::find_if(
        shard_metadata.begin(), shard_metadata.end(),
        [key](const MetadataEntry &entry) { return entry.key == key; });
  };
  const auto evo_profile = find_profile("evo2.profile");
  const auto runtime_profile = find_profile("runtime.profile");
  if ((evo_profile == shard_metadata.end()) ==
      (runtime_profile == shard_metadata.end())) {
    return format_error(
        "exactly one registered artifact profile metadata key is required");
  }
  const auto profile =
      evo_profile != shard_metadata.end() ? evo_profile : runtime_profile;
  if (profile->type != MetadataType::kString) {
    return format_error("artifact profile metadata must be a string");
  }
  const std::string_view profile_value{
      reinterpret_cast<const char *>(profile->value.data()),
      profile->value.size()};
  if (find_artifact_profile(profile_value) == nullptr) {
    return format_error("unsupported artifact profile '" +
                        std::string{profile_value} + "'");
  }
  if (!profile_.empty() && profile_ != profile_value) {
    return format_error("artifact profile differs between shards");
  }
  if (metadata_.empty()) {
    profile_ = profile_value;
    metadata_ = std::move(shard_metadata);
  } else if (!metadata_equal(metadata_, shard_metadata)) {
    return format_error("metadata differs between shards");
  }

  const auto data_size =
      static_cast<std::uint64_t>(mapping.size()) - data_offset;
  std::set<std::string> names;
  for (const auto &tensor : tensors_) {
    names.insert(tensor.name);
  }
  std::vector<TensorInfo> shard_tensors;
  shard_tensors.reserve(raw_tensors.size());
  for (const auto &raw : raw_tensors) {
    if (!valid_identifier(raw.name, kTensorNameCapacity - 1) ||
        !names.insert(raw.name).second) {
      return format_error("invalid or duplicate tensor name '" + raw.name +
                          "'");
    }
    if (raw.shape.size() > kTensorMaxRank || raw.end <= raw.begin ||
        raw.end > data_size) {
      return format_error("invalid shape or data range for '" + raw.name + "'");
    }
    TensorInfo tensor;
    tensor.name = raw.name;
    tensor.rank = static_cast<std::uint8_t>(raw.shape.size());
    tensor.shard_index = shard_index;
    std::uint64_t width = 0;
    if (!tensor_dtype(raw.dtype, &tensor.dtype, &width)) {
      return format_error("unsupported dtype '" + raw.dtype + "' for '" +
                          raw.name + "'");
    }
    tensor.element_count = 1;
    for (std::size_t index = 0; index < raw.shape.size(); ++index) {
      const auto dimension = raw.shape[index];
      if (dimension == 0 || !checked_mul(tensor.element_count, dimension,
                                         &tensor.element_count)) {
        return format_error("invalid dimensions for '" + raw.name + "'");
      }
      tensor.dimensions[index] = dimension;
    }
    std::uint64_t expected_size = 0;
    if (!checked_mul(tensor.element_count, width, &expected_size) ||
        expected_size != raw.end - raw.begin ||
        !checked_add(data_offset, raw.begin, &tensor.data_offset)) {
      return format_error("dtype/shape size mismatch for '" + raw.name + "'");
    }
    tensor.data_size = expected_size;
    shard_tensors.push_back(std::move(tensor));
  }

  std::sort(shard_tensors.begin(), shard_tensors.end(),
            [](const TensorInfo &left, const TensorInfo &right) {
              return left.data_offset < right.data_offset;
            });
  std::uint64_t cursor = data_offset;
  for (const auto &tensor : shard_tensors) {
    if (tensor.data_offset != cursor ||
        !checked_add(cursor, tensor.data_size, &cursor)) {
      return format_error("tensor data buffer contains a hole or overlap near "
                          "'" +
                          tensor.name + "'");
    }
  }
  if (cursor != mapping.size()) {
    return format_error("tensor data does not cover the complete file");
  }
  tensors_.insert(tensors_.end(),
                  std::make_move_iterator(shard_tensors.begin()),
                  std::make_move_iterator(shard_tensors.end()));
  return Status::Ok();
}

Status ModelFile::parse_tokenizer_asset_descriptor() {
  constexpr std::array<std::string_view, 4> keys{
      "tokenizer.profile", "tokenizer.path", "tokenizer.sha256",
      "tokenizer.size"};
  const bool present = find_metadata(keys[0]) != nullptr;
  // Existing architecture-local tokenizer metadata (for example
  // tokenizer.kind and tokenizer.pad_token_id) predates the sidecar asset
  // descriptor.  It remains valid when none of the four descriptor keys is
  // present; once a descriptor starts, its namespace is deliberately closed.
  if (!present) {
    // `tokenizer.sha256` is also an established architecture-local evidence
    // field in legacy ESMC artifacts.  A path or size, unlike a digest alone,
    // unambiguously starts the sidecar descriptor contract.
    if (find_metadata(keys[1]) != nullptr || find_metadata(keys[3]) != nullptr)
      return format_error("incomplete tokenizer descriptor metadata: missing "
                          "'tokenizer.profile'");
    tokenizer_asset_descriptor_.reset();
    return Status::Ok();
  }
  for (const auto &entry : metadata_) {
    if (entry.key.rfind("tokenizer.", 0) == 0 &&
        std::find(keys.begin(), keys.end(), entry.key) == keys.end())
      return format_error("unknown tokenizer descriptor metadata '" +
                          entry.key + "'");
  }
  std::array<const MetadataEntry *, keys.size()> entries{};
  for (std::size_t index = 0; index < keys.size(); ++index) {
    entries[index] = find_metadata(keys[index]);
    if (entries[index] == nullptr)
      return format_error(
          "incomplete tokenizer descriptor metadata: missing '" +
          std::string{keys[index]} + "'");
  }
  if (entries[0]->type != MetadataType::kString ||
      metadata_string(*entries[0]) != "evo-tokenizer-v1")
    return format_error("tokenizer.profile must be evo-tokenizer-v1");
  if (entries[1]->type != MetadataType::kString ||
      !valid_tokenizer_asset_path(metadata_string(*entries[1])))
    return format_error("tokenizer.path must be a canonical relative path");
  if (entries[2]->type != MetadataType::kString ||
      !canonical_sha256(metadata_string(*entries[2])))
    return format_error("tokenizer.sha256 must be canonical lowercase hex");
  if (entries[3]->type != MetadataType::kU64 ||
      entries[3]->value.size() != sizeof(std::uint64_t))
    return format_error("tokenizer.size must be a u64");
  const auto size = read_u64(entries[3]->value.data());
  if (size == 0 || size > (64U << 20U))
    return format_error("tokenizer.size is outside the supported range");
  TokenizerAssetDescriptor descriptor;
  descriptor.path = metadata_string(*entries[1]);
  descriptor.sha256 = metadata_string(*entries[2]);
  descriptor.size = size;
  tokenizer_asset_descriptor_ = std::move(descriptor);
  return Status::Ok();
}

Status ModelFile::parse_converter_profile_contract() {
  constexpr std::string_view legacy_key{"source.converter_manifest_sha256"};
  constexpr std::string_view contract_key{
      "source.converter_profile_contract_sha256"};
  if (find_metadata(legacy_key) != nullptr)
    return format_error("legacy whole-profile converter digest metadata is "
                        "unsupported");
  const auto *const entry = find_metadata(contract_key);
  if (entry == nullptr)
    return Status::Ok();
  if (entry->type != MetadataType::kString ||
      !canonical_sha256(metadata_string(*entry)))
    return format_error(
        "source.converter_profile_contract_sha256 must be canonical "
        "lowercase hex");
  return Status::Ok();
}

const MetadataEntry *
ModelFile::find_metadata(const std::string_view key) const noexcept {
  const auto found = std::find_if(
      metadata_.begin(), metadata_.end(),
      [key](const MetadataEntry &entry) { return entry.key == key; });
  return found == metadata_.end() ? nullptr : &*found;
}

const TensorInfo *
ModelFile::find_tensor(const std::string_view name) const noexcept {
  const auto found = std::find_if(
      tensors_.begin(), tensors_.end(),
      [name](const TensorInfo &tensor) { return tensor.name == name; });
  return found == tensors_.end() ? nullptr : &*found;
}

const std::uint8_t *
ModelFile::tensor_data(const TensorInfo &tensor) const noexcept {
  if (tensor.shard_index >= mappings_.size()) {
    return nullptr;
  }
  const auto &mapping = mappings_[tensor.shard_index];
  if (tensor.data_offset > mapping.size() ||
      tensor.data_size > mapping.size() - tensor.data_offset) {
    return nullptr;
  }
  return mapping.data() + tensor.data_offset;
}

Status ModelFile::read_tensor(const TensorInfo &tensor,
                              const std::uint64_t offset,
                              void *const destination,
                              const std::size_t bytes) const {
  const auto *const stored = find_tensor(tensor.name);
  if (stored == nullptr || stored->dtype != tensor.dtype ||
      stored->data_size != tensor.data_size ||
      stored->shard_index != tensor.shard_index) {
    return {ErrorCode::kInvalidArgument,
            "tensor descriptor does not belong to this model: " + tensor.name};
  }
  if (destination == nullptr && bytes != 0) {
    return {ErrorCode::kInvalidArgument, "tensor read destination is null"};
  }
  if (offset > stored->data_size || bytes > stored->data_size - offset) {
    return {ErrorCode::kInvalidArgument,
            "tensor read exceeds payload: " + tensor.name};
  }
  const auto *const data = tensor_data(*stored);
  if (data == nullptr) {
    return {ErrorCode::kModelFormat,
            "tensor payload is unavailable: " + tensor.name};
  }
  if (bytes != 0) {
    std::memcpy(destination, data + offset, bytes);
  }
  return Status::Ok();
}

const char *metadata_type_name(const MetadataType type) noexcept {
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

const char *tensor_dtype_name(const TensorDType dtype) noexcept {
  switch (dtype) {
  case TensorDType::kF32:
    return "F32";
  case TensorDType::kBF16:
    return "BF16";
  case TensorDType::kE4M3Software:
    return "F8_E4M3";
  }
  return "UNKNOWN";
}

std::string metadata_value_text(const MetadataEntry &entry) {
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

} // namespace evo
