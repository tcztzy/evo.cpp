// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

#include "evo2c/cuda/model.hpp"
#include "evo2c/cuda/runtime.hpp"
#include "evo2c/model_format.hpp"
#include "evo2c/status.hpp"
#include "evo2c/tokenizer.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

void require(const evo2c::Status &status, const std::string_view operation) {
  if (!status.ok())
    throw std::runtime_error(std::string{operation} + ": " + status.message());
}

std::vector<char> read_bytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open expected file: " + path);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::vector<float> read_f32(const std::string &path) {
  const auto bytes = read_bytes(path);
  if (bytes.size() % sizeof(float) != 0)
    throw std::runtime_error("expected F32 file has invalid length: " + path);
  std::vector<float> output(bytes.size() / sizeof(float));
  std::memcpy(output.data(), bytes.data(), bytes.size());
  return output;
}

std::vector<float> read_npy_f32(const std::string &path,
                                const std::size_t expected_elements) {
  const auto bytes = read_bytes(path);
  if (bytes.size() < 10 || static_cast<unsigned char>(bytes[0]) != 0x93U ||
      std::string(bytes.data() + 1, bytes.data() + 6) != "NUMPY" ||
      bytes[6] != 1 || bytes[7] != 0) {
    throw std::runtime_error("layer dump is not NPY v1: " + path);
  }
  const auto header_size =
      static_cast<std::size_t>(static_cast<unsigned char>(bytes[8])) |
      (static_cast<std::size_t>(static_cast<unsigned char>(bytes[9])) << 8U);
  const std::size_t payload = 10 + header_size;
  if (payload > bytes.size() ||
      bytes.size() - payload != expected_elements * sizeof(float)) {
    throw std::runtime_error("layer dump NPY payload has wrong size: " + path);
  }
  std::vector<float> output(expected_elements);
  std::memcpy(output.data(),
              bytes.data() + static_cast<std::ptrdiff_t>(payload),
              output.size() * sizeof(float));
  return output;
}

bool all_close(const std::vector<float> &actual,
               const std::vector<float> &expected, const float absolute,
               const float relative) {
  if (actual.size() != expected.size()) {
    std::cerr << "size mismatch: " << actual.size() << " vs " << expected.size()
              << '\n';
    return false;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float tolerance = absolute + relative * std::abs(expected[index]);
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance) {
      std::cerr << "mismatch at " << index << ": actual=" << actual[index]
                << " expected=" << expected[index] << " tolerance=" << tolerance
                << '\n';
      return false;
    }
  }
  return true;
}

float cosine(const std::vector<float> &left, const std::vector<float> &right) {
  double dot = 0.0;
  double left_norm = 0.0;
  double right_norm = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    dot += static_cast<double>(left[index]) * right[index];
    left_norm += static_cast<double>(left[index]) * left[index];
    right_norm += static_cast<double>(right[index]) * right[index];
  }
  return static_cast<float>(dot / std::sqrt(left_norm * right_norm));
}

int requested_device() {
  const char *environment = std::getenv("EVO2C_TEST_DEVICE");
  return environment == nullptr ? 0 : std::stoi(environment);
}

} // namespace

int main(const int argc, char **argv) {
  try {
    if (argc != 10) {
      std::cerr << "usage: test_cuda_model MODEL EXPECTED_LOGITS "
                   "EXPECTED_DECODE EXPECTED_LAYER DUMP_PATH "
                   "EXPECTED_CHUNKED INVALID_PROJECTION_MISMATCH "
                   "INVALID_PROJECTION_DTYPE INVALID_FP8_RESIDUE\n";
      return 2;
    }
    const int device = requested_device();
    require(evo2c::cuda::select_device(device), "select CUDA device");
    cudaDeviceProp properties{};
    require(
        evo2c::cuda::cuda_status(cudaGetDeviceProperties(&properties, device),
                                 "cudaGetDeviceProperties"),
        "query CUDA device");
    std::cout << "GPU=" << properties.name << " sm=" << properties.major
              << properties.minor << '\n';

    evo2c::cuda::RuntimeModelConfig warmup_config;
    warmup_config.model_id = "evo2_7b";
    check(evo2c::cuda::backend_warmup_tokens(warmup_config, 128) == 128,
          "7B backend warmup policy selects 128 tokens");
    warmup_config.model_id = "evo2_7b_262k";
    check(evo2c::cuda::backend_warmup_tokens(warmup_config, 8192) == 128,
          "7B variants share the backend warmup policy");
    check(evo2c::cuda::backend_warmup_tokens(warmup_config, 127) == 0,
          "undersized activation arena disables backend warmup");
    warmup_config.test_fixture = true;
    check(evo2c::cuda::backend_warmup_tokens(warmup_config, 8192) == 0,
          "synthetic fixtures never run production backend warmup");
    warmup_config.test_fixture = false;
    warmup_config.model_id = "evo2_40b";
    check(evo2c::cuda::backend_warmup_tokens(warmup_config, 8192) == 0,
          "non-7B profiles do not inherit the 7B warmup policy");

    evo2c::ModelFile file;
    require(file.open(argv[1]), "open synthetic model");
    evo2c::cuda::RuntimeModelConfig config;
    check(!evo2c::cuda::read_runtime_model_config(file, false, &config).ok(),
          "synthetic model is rejected without explicit permission");
    require(evo2c::cuda::read_runtime_model_config(file, true, &config),
            "read synthetic config");
    check(config.layers == 50, "synthetic topology contains 50 blocks");
    check(config.mixer_types.size() == 50,
          "every synthetic block has a mixer type");
    check(std::count(config.mixer_types.begin(), config.mixer_types.end(),
                     evo2c::cuda::MixerType::kAttention) == 8,
          "synthetic topology contains eight attention blocks");
    check(config.hyena_projection_dtype ==
              evo2c::cuda::HyenaProjectionDType::kBF16,
          "synthetic model selects native BF16 Hyena projections");
    check(config.hcm_filter_dtype == evo2c::cuda::HcmFilterDType::kF32,
          "synthetic model selects BioNeMo F32 medium-Hyena filters");

    const auto check_rejected_config = [](const char *const path,
                                          const std::string_view expected) {
      evo2c::ModelFile invalid;
      require(invalid.open(path), "open invalid synthetic model");
      evo2c::cuda::RuntimeModelConfig ignored;
      const auto rejected =
          evo2c::cuda::read_runtime_model_config(invalid, true, &ignored);
      check(!rejected.ok() &&
                rejected.message().find(expected) != std::string::npos,
            "invalid precision metadata is rejected before CUDA load");
    };
    check_rejected_config(argv[7], "disagrees");
    check_rejected_config(argv[8], "unsupported Hyena projection dtype");
    check_rejected_config(argv[9], "unknown tensor");

    evo2c::cuda::SingleGpuModel model;
    require(model.load(file, device, 12, true), "load single-GPU model");
    check(model.activation_capacity() == 8,
          "test model exposes its fixed eight-token activation arena");
    std::vector<float> one_token_logits;
    require(model.prefill({1}, &one_token_logits), "one-token model prefill");
    check(
        one_token_logits.size() == config.vocab_size &&
            std::all_of(one_token_logits.begin(), one_token_logits.end(),
                        [](const float value) { return std::isfinite(value); }),
        "one-token prefill produces finite vocabulary logits");
    check(model.position() == 1, "one-token prefill records its position");

    const std::vector<evo2c::TokenId> prompt{2, 5, 7, 3};
    std::vector<float> stateless_logits;
    require(model.prefill_stateless(prompt, &stateless_logits),
            "stateless model prefill");
    check(model.position() == 0,
          "stateless prefill does not publish a continuation position");
    std::vector<float> invalid_decode_logits;
    const auto invalid_stateless_decode =
        model.decode(9, &invalid_decode_logits);
    check(!invalid_stateless_decode.ok(),
          "stateless prefill cannot be followed by cached decode");
    std::vector<float> logits;
    const evo2c::cuda::LayerDump dump{17, argv[5]};
    require(model.prefill(prompt, &logits, dump), "50-layer model prefill");
    const auto expected_logits = read_f32(argv[2]);
    check(all_close(logits, expected_logits, 0.08F, 0.06F),
          "synthetic prefill logits match independent oracle");
    check(stateless_logits == logits,
          "stateless and stateful prefill logits are bit-identical");
    check(cosine(logits, expected_logits) >= 0.999F,
          "synthetic prefill logits cosine is at least 0.999");
    check(model.position() == prompt.size(),
          "prefill records the prompt position");

    const auto expected_layer = read_f32(argv[4]);
    const auto dumped_layer = read_npy_f32(argv[5], expected_layer.size());
    check(all_close(dumped_layer, expected_layer, 0.06F, 0.05F),
          "layer 17 NPY dump matches independent oracle");

    std::vector<float> decoded;
    require(model.decode(9, &decoded), "50-layer cached decode");
    const auto expected_decode = read_f32(argv[3]);
    check(all_close(decoded, expected_decode, 0.08F, 0.06F),
          "synthetic cached decode logits match independent oracle");
    check(cosine(decoded, expected_decode) >= 0.999F,
          "synthetic decode logits cosine is at least 0.999");
    check(model.position() == prompt.size() + 1,
          "decode advances the model position");

    const std::vector<evo2c::TokenId> long_prompt{2,  5,  7,  3, 9,
                                                  11, 13, 17, 19};
    const std::vector<evo2c::TokenId> initial_chunk(long_prompt.begin(),
                                                    long_prompt.begin() + 8);
    std::vector<float> first_chunk;
    std::vector<float> final_chunk;
    require(model.prefill(initial_chunk, &first_chunk),
            "eight-token initial model chunk");
    require(model.prefill_chunk({long_prompt.back()}, &final_chunk),
            "one-token continued model chunk");
    const auto expected_chunked = read_f32(argv[6]);
    const std::vector<float> expected_last(
        expected_chunked.end() - static_cast<std::ptrdiff_t>(config.vocab_size),
        expected_chunked.end());
    check(all_close(final_chunk, expected_last, 0.08F, 0.06F),
          "chunked model final logits match full Python causal oracle");
    check(cosine(final_chunk, expected_last) >= 0.999F,
          "chunked model final-logit cosine is at least 0.999");
    check(model.position() == long_prompt.size(),
          "chunked model records the full logical position");

    std::vector<float> cached_logits;
    require(model.prefill_cached({2, 5}, &cached_logits),
            "exact cached model prefill");
    const auto invalid_cached_chunk = model.prefill_chunk({7}, &cached_logits);
    check(!invalid_cached_chunk.ok() &&
              invalid_cached_chunk.message().find("exact token decode") !=
                  std::string::npos,
          "cached prefill rejects a numerically different chunk continuation");
    require(model.decode(7, &cached_logits),
            "cached prefill continues through exact token decode");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " CUDA model test(s) failed\n";
    return 1;
  }
  std::cout << "CUDA model tests passed\n";
  return 0;
}
