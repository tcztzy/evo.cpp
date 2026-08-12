// SPDX-License-Identifier: Apache-2.0
#include "evo/profile.hpp"

namespace evo {

const char *inference_profile_name(const InferenceProfile profile) noexcept {
  switch (profile) {
  case InferenceProfile::kExact:
    return "exact";
  case InferenceProfile::kFastQ8Kv:
    return "fast-q8-kv";
  case InferenceProfile::kCpuF32:
    return "cpu-f32";
  }
  return "unknown";
}

Status parse_inference_profile(const std::string_view text,
                               InferenceProfile *const profile) {
  if (profile == nullptr)
    return {ErrorCode::kInvalidArgument, "inference profile output is null"};
  if (text == "exact") {
    *profile = InferenceProfile::kExact;
    return Status::Ok();
  }
  if (text == "fast-q8-kv") {
    *profile = InferenceProfile::kFastQ8Kv;
    return Status::Ok();
  }
  if (text == "cpu-f32") {
    *profile = InferenceProfile::kCpuF32;
    return Status::Ok();
  }
  return {ErrorCode::kInvalidArgument,
          "profile must be one of exact, fast-q8-kv, or cpu-f32"};
}

bool inference_profile_is_exact(const InferenceProfile profile) noexcept {
  return profile == InferenceProfile::kExact;
}

} // namespace evo
