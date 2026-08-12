// SPDX-License-Identifier: Apache-2.0
#include "artifact_cache.hpp"

#include "evo/json.hpp"
#include "evo/model_registry.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace evo {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kMaximumManifestBytes = 8U << 20U;
constexpr std::size_t kMaximumReceiptBytes = 16U << 20U;
constexpr std::size_t kMaximumArtifactFiles = 10000;

class Sha256 final {
public:
  void update(const std::uint8_t *data, std::size_t size) noexcept {
    while (size != 0) {
      const auto copied = std::min(size, block_.size() - block_size_);
      std::copy_n(data, copied, block_.begin() +
                                    static_cast<std::ptrdiff_t>(block_size_));
      block_size_ += copied;
      data += copied;
      size -= copied;
      if (block_size_ == block_.size()) {
        transform(block_.data());
        bit_count_ += 512U;
        block_size_ = 0;
      }
    }
  }

  [[nodiscard]] std::array<std::uint8_t, 32> finish() noexcept {
    bit_count_ += static_cast<std::uint64_t>(block_size_) * 8U;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56U) {
      std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                block_.end(), 0U);
      transform(block_.data());
      block_size_ = 0;
    }
    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
              block_.begin() + 56, 0U);
    for (std::size_t index = 0; index < 8; ++index) {
      block_[63U - index] =
          static_cast<std::uint8_t>(bit_count_ >> (index * 8U));
    }
    transform(block_.data());
    std::array<std::uint8_t, 32> digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4; ++byte) {
        digest[word * 4U + byte] = static_cast<std::uint8_t>(
            state_[word] >> ((3U - byte) * 8U));
      }
    }
    return digest;
  }

private:
  static std::uint32_t rotate_right(const std::uint32_t value,
                                    const unsigned shift) noexcept {
    return (value >> shift) | (value << (32U - shift));
  }

  void transform(const std::uint8_t *const block) noexcept {
    static constexpr std::array<std::uint32_t, 64> constants{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const auto offset = index * 4U;
      words[index] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                     (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                     static_cast<std::uint32_t>(block[offset + 3U]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const auto s0 = rotate_right(words[index - 15U], 7U) ^
                      rotate_right(words[index - 15U], 18U) ^
                      (words[index - 15U] >> 3U);
      const auto s1 = rotate_right(words[index - 2U], 17U) ^
                      rotate_right(words[index - 2U], 19U) ^
                      (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }
    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const auto sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                        rotate_right(e, 25U);
      const auto choose = (e & f) ^ (~e & g);
      const auto first = h + sum1 + choose + constants[index] + words[index];
      const auto sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                        rotate_right(a, 22U);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto second = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_{0};
  std::uint64_t bit_count_{0};
};

Status read_limited_file(const fs::path &path, const std::size_t limit,
                         std::string *const output) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return {ErrorCode::kIo, "cannot open cached file '" + path.string() + "'"};
  const auto end = input.tellg();
  if (end < 0 || static_cast<std::uintmax_t>(end) > limit)
    return {ErrorCode::kModelFormat,
            "cached JSON file is empty, unreadable, or too large: '" +
                path.string() + "'"};
  output->resize(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!output->empty())
    input.read(output->data(), static_cast<std::streamsize>(output->size()));
  if (!input)
    return {ErrorCode::kIo, "cannot read cached file '" + path.string() + "'"};
  return Status::Ok();
}

std::string hexadecimal(const std::array<std::uint8_t, 32> &digest) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto byte : digest)
    output << std::setw(2) << static_cast<unsigned>(byte);
  return output.str();
}

Status sha256_file(const fs::path &path, std::string *const output) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {ErrorCode::kIo, "cannot open cached artifact file '" +
                                path.string() + "'"};
  Sha256 digest;
  std::array<char, 1U << 20U> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      digest.update(reinterpret_cast<const std::uint8_t *>(buffer.data()),
                    static_cast<std::size_t>(count));
    }
  }
  if (!input.eof())
    return {ErrorCode::kIo, "cannot read cached artifact file '" +
                                path.string() + "'"};
  *output = hexadecimal(digest.finish());
  return Status::Ok();
}

bool is_hex(const std::string_view value, const std::size_t size) noexcept {
  if (value.size() != size)
    return false;
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f') ||
          (character >= 'A' && character <= 'F')))
      return false;
  }
  return true;
}

std::string lower_hex(std::string value) {
  for (char &character : value) {
    if (character >= 'A' && character <= 'F')
      character = static_cast<char>(character - 'A' + 'a');
  }
  return value;
}

bool valid_repo_component(const std::string_view value) noexcept {
  if (value.empty() || !std::isalnum(static_cast<unsigned char>(value.front())))
    return false;
  for (const char character : value) {
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '.' && character != '_' && character != '-')
      return false;
  }
  return true;
}

bool normalized_relative(const std::string_view value) {
  if (value.empty() || value.front() == '/' ||
      value.find('\\') != std::string_view::npos)
    return false;
  const fs::path path{std::string{value}};
  if (path.is_absolute() || path.has_root_path())
    return false;
  for (const auto &component : path) {
    if (component.empty() || component == "." || component == "..")
      return false;
  }
  return path.generic_string() == value;
}

Status parse_repository(const std::string_view specification, std::string *repo,
                        std::string *revision) {
  const auto at = specification.rfind('@');
  *repo = std::string{at == std::string_view::npos
                          ? specification
                          : specification.substr(0, at)};
  *revision = at == std::string_view::npos
                  ? "main"
                  : std::string{specification.substr(at + 1U)};
  const auto slash = repo->find('/');
  if (slash == std::string::npos || slash != repo->rfind('/') ||
      !valid_repo_component(std::string_view{*repo}.substr(0, slash)) ||
      !valid_repo_component(std::string_view{*repo}.substr(slash + 1U))) {
    return {ErrorCode::kInvalidArgument,
            "Hugging Face repository must be OWNER/NAME with optional @REV"};
  }
  if (!normalized_relative(*revision)) {
    return {ErrorCode::kInvalidArgument,
            "Hugging Face revision must be nonempty, normalized, and contain "
            "no path traversal"};
  }
  return Status::Ok();
}

Status cache_root(fs::path *const output) {
  const auto environment = [](const char *const name) -> const char * {
    const char *const value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? value : nullptr;
  };
  if (const char *const evo_cache = environment("EVO_CACHE_HOME")) {
    *output = fs::path{evo_cache} / "huggingface" / "hub";
  } else if (const char *const hf_cache = environment("HF_HOME")) {
    *output = fs::path{hf_cache} / "hub";
  } else if (const char *const xdg_cache = environment("XDG_CACHE_HOME")) {
    *output = fs::path{xdg_cache} / "huggingface" / "hub";
  } else if (const char *const user_home = environment("HOME")) {
    *output = fs::path{user_home} / ".cache" / "huggingface" / "hub";
  } else {
    return {ErrorCode::kIo,
            "cannot locate the Hugging Face cache: HOME and cache overrides "
            "are unset"};
  }
  return Status::Ok();
}

Status json_document(const fs::path &path, const std::size_t limit,
                     JsonValue *const output, std::string *const bytes) {
  auto status = read_limited_file(path, limit, bytes);
  if (!status.ok())
    return status;
  status = parse_json(*bytes, output);
  if (!status.ok())
    return {ErrorCode::kModelFormat,
            "invalid cached JSON '" + path.string() + "': " +
                status.message()};
  if (output->type != JsonType::kObject)
    return {ErrorCode::kModelFormat,
            "cached JSON root must be an object: '" + path.string() + "'"};
  return Status::Ok();
}

const JsonValue *member(const JsonValue &object, const std::string_view key,
                        const JsonType type) noexcept {
  const auto *const value = object.find(key);
  return value != nullptr && value->type == type ? value : nullptr;
}

bool integer_value(const JsonValue *const value, std::uintmax_t *const output) {
  if (value == nullptr || value->type != JsonType::kNumber ||
      !std::isfinite(value->number) || value->number < 0.0 ||
      value->number > 9007199254740991.0)
    return false;
  double integral = 0.0;
  if (std::modf(value->number, &integral) != 0.0)
    return false;
  *output = static_cast<std::uintmax_t>(integral);
  return true;
}

bool path_is_within(const fs::path &child, const fs::path &parent) {
  auto child_part = child.begin();
  for (auto parent_part = parent.begin(); parent_part != parent.end();
       ++parent_part, ++child_part) {
    if (child_part == child.end() || *child_part != *parent_part)
      return false;
  }
  return true;
}

Status canonical_artifact_path(const fs::path &repo_root,
                               const fs::path &snapshot_root,
                               const std::string_view relative,
                               fs::path *const output) {
  std::error_code error;
  const auto canonical_repo = fs::canonical(repo_root, error);
  if (error)
    return {ErrorCode::kIo, "cannot resolve cached repository '" +
                                repo_root.string() + "': " + error.message()};
  const auto candidate = snapshot_root / fs::path{std::string{relative}};
  const auto canonical_file = fs::canonical(candidate, error);
  if (error)
    return {ErrorCode::kIo,
            "cached artifact file is unavailable: '" + candidate.string() +
                "'; run evo-fetch runtime first"};
  if (!path_is_within(canonical_file, canonical_repo))
    return {ErrorCode::kModelFormat,
            "cached artifact symlink escapes its repository: '" +
                candidate.string() + "'"};
  if (!fs::is_regular_file(canonical_file, error) || error)
    return {ErrorCode::kModelFormat,
            "cached artifact entry is not a regular file: '" +
                candidate.string() + "'"};
  *output = canonical_file;
  return Status::Ok();
}

Status require_receipt(const fs::path &path, const std::string &repo,
                       const std::string &revision,
                       const std::string &manifest_sha256,
                       const std::string &model_id,
                       const std::string_view artifact_profile) {
  JsonValue receipt;
  std::string bytes;
  auto status = json_document(path, kMaximumReceiptBytes, &receipt, &bytes);
  if (!status.ok())
    return {status.code(), status.message() +
                               "; run evo-fetch runtime REPO[@REV] first"};
  std::uintmax_t schema = 0;
  const auto *const kind = member(receipt, "kind", JsonType::kString);
  const auto *const receipt_repo = member(receipt, "repo", JsonType::kString);
  const auto *const resolved =
      member(receipt, "resolved_revision", JsonType::kString);
  const auto *const profile =
      member(receipt, "artifact_profile", JsonType::kString);
  const auto *const receipt_model =
      member(receipt, "model_id", JsonType::kString);
  const auto *const digest =
      member(receipt, "manifest_sha256", JsonType::kString);
  if (!integer_value(receipt.find("schema_version"), &schema) || schema != 1U ||
      kind == nullptr || kind->string != "runtime-artifact" ||
      receipt_repo == nullptr || receipt_repo->string != repo ||
      resolved == nullptr || lower_hex(resolved->string) != revision ||
      profile == nullptr ||
      std::string_view{profile->string} != artifact_profile ||
      receipt_model == nullptr || receipt_model->string != model_id ||
      digest == nullptr || !is_hex(digest->string, 64U) ||
      lower_hex(digest->string) != manifest_sha256) {
    return {ErrorCode::kModelFormat,
            "cached runtime receipt does not match repository, revision, "
            "profile, model, and manifest SHA256"};
  }
  return Status::Ok();
}

} // namespace

Status resolve_cached_hf_artifact(const std::string_view repository,
                                  std::string *const model_path) {
  if (model_path == nullptr)
    return {ErrorCode::kInvalidArgument, "model path output is null"};
  model_path->clear();
  std::string repo;
  std::string requested_revision;
  auto status = parse_repository(repository, &repo, &requested_revision);
  if (!status.ok())
    return status;

  fs::path cache;
  status = cache_root(&cache);
  if (!status.ok())
    return status;
  std::string cache_name = repo;
  cache_name.replace(cache_name.find('/'), 1U, "--");
  const fs::path repo_root = cache / ("models--" + cache_name);

  std::string revision = requested_revision;
  if (!is_hex(revision, 40U)) {
    std::string reference;
    status = read_limited_file(repo_root / "refs" / requested_revision, 4096U,
                               &reference);
    if (!status.ok())
      return {ErrorCode::kIo,
              "no cached immutable revision for " + repo + "@" +
                  requested_revision + "; run evo-fetch runtime first"};
    while (!reference.empty() &&
           (reference.back() == '\n' || reference.back() == '\r' ||
            reference.back() == ' ' || reference.back() == '\t'))
      reference.pop_back();
    if (!is_hex(reference, 40U))
      return {ErrorCode::kModelFormat,
              "cached Hugging Face ref is not an immutable 40-hex commit"};
    revision = reference;
  }
  revision = lower_hex(revision);
  const fs::path snapshot = repo_root / "snapshots" / revision;
  fs::path manifest_path;
  status = canonical_artifact_path(repo_root, snapshot, "evo-artifact.json",
                                   &manifest_path);
  if (!status.ok())
    return status;

  JsonValue manifest;
  std::string manifest_bytes;
  status = json_document(manifest_path, kMaximumManifestBytes, &manifest,
                         &manifest_bytes);
  if (!status.ok())
    return status;
  std::uintmax_t schema = 0;
  const auto *const profile =
      member(manifest, "artifact_profile", JsonType::kString);
  const auto *const model_id = member(manifest, "model_id", JsonType::kString);
  const auto *const load_path =
      member(manifest, "load_path", JsonType::kString);
  const auto *const files = member(manifest, "files", JsonType::kArray);
  if (!integer_value(manifest.find("schema_version"), &schema) || schema != 1U ||
      profile == nullptr ||
      find_artifact_profile(profile->string) == nullptr ||
      model_id == nullptr || model_id->string.empty() || load_path == nullptr ||
      !normalized_relative(load_path->string) || files == nullptr ||
      files->array.empty() || files->array.size() > kMaximumArtifactFiles) {
    return {ErrorCode::kModelFormat,
            "cached runtime manifest has an invalid schema, profile, model_id, "
            "load_path, or files array"};
  }
  if (load_path->string.size() < 12U ||
      (load_path->string.compare(load_path->string.size() - 12U, 12U,
                                 ".safetensors") != 0 &&
       (load_path->string.size() < 23U ||
        load_path->string.compare(load_path->string.size() - 23U, 23U,
                                  ".safetensors.index.json") != 0))) {
    return {ErrorCode::kModelFormat,
            "cached runtime load_path must be Safetensors or its index"};
  }

  Sha256 manifest_digest;
  manifest_digest.update(
      reinterpret_cast<const std::uint8_t *>(manifest_bytes.data()),
      manifest_bytes.size());
  const auto manifest_sha256 = hexadecimal(manifest_digest.finish());
  const fs::path receipt = cache / "evo-receipts" / cache_name / revision /
                           "runtime-artifact.json";
  status = require_receipt(receipt, repo, revision, manifest_sha256,
                           model_id->string, profile->string);
  if (!status.ok())
    return status;

  std::set<std::string> names;
  fs::path resolved_load_path;
  for (std::size_t index = 0; index < files->array.size(); ++index) {
    const auto &entry = files->array[index];
    const auto *const path = member(entry, "path", JsonType::kString);
    const auto *const expected_digest =
        member(entry, "sha256", JsonType::kString);
    std::uintmax_t expected_size = 0;
    if (entry.type != JsonType::kObject || path == nullptr ||
        !normalized_relative(path->string) || expected_digest == nullptr ||
        !is_hex(expected_digest->string, 64U) ||
        !integer_value(entry.find("size"), &expected_size) ||
        !names.insert(path->string).second) {
      return {ErrorCode::kModelFormat,
              "cached runtime manifest file entry " + std::to_string(index) +
                  " is invalid or duplicated"};
    }
    fs::path resolved;
    status = canonical_artifact_path(repo_root, snapshot, path->string, &resolved);
    if (!status.ok())
      return status;
    std::error_code error;
    const auto actual_size = fs::file_size(resolved, error);
    if (error)
      return {ErrorCode::kIo, "cannot stat cached artifact file '" +
                                  resolved.string() + "': " + error.message()};
    if (actual_size != expected_size)
      return {ErrorCode::kModelFormat,
              "cached artifact size mismatch for '" + path->string + "'"};
    std::string actual_digest;
    status = sha256_file(resolved, &actual_digest);
    if (!status.ok())
      return status;
    if (actual_digest != lower_hex(expected_digest->string))
      return {ErrorCode::kModelFormat,
              "cached artifact SHA256 mismatch for '" + path->string + "'"};
    if (path->string == load_path->string)
      resolved_load_path = resolved;
  }
  if (resolved_load_path.empty())
    return {ErrorCode::kModelFormat,
            "cached runtime load_path does not name a listed file"};
  *model_path = resolved_load_path.string();
  return Status::Ok();
}

} // namespace evo
