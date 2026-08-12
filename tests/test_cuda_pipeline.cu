// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime_api.h>

#include "evo/cuda/model.hpp"
#include "evo/cuda/runtime.hpp"
#include "evo/model_format.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

void require(const evo::Status &status, const std::string_view operation) {
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
      std::string(bytes.data() + 1, bytes.data() + 6) != "NUMPY") {
    throw std::runtime_error("pipeline layer dump is not NPY: " + path);
  }
  const auto header_size =
      static_cast<std::size_t>(static_cast<unsigned char>(bytes[8])) |
      (static_cast<std::size_t>(static_cast<unsigned char>(bytes[9])) << 8U);
  const std::size_t payload = 10 + header_size;
  if (payload > bytes.size() ||
      bytes.size() - payload != expected_elements * sizeof(float)) {
    throw std::runtime_error("pipeline NPY payload has wrong size: " + path);
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
  if (actual.size() != expected.size())
    return false;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float tolerance = absolute + relative * std::abs(expected[index]);
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance) {
      std::cerr << "mismatch at " << index << ": actual=" << actual[index]
                << " expected=" << expected[index] << '\n';
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

} // namespace

int main(const int argc, char **argv) {
  try {
    if (argc != 7) {
      std::cerr << "usage: test_cuda_pipeline MODEL EXPECTED_LOGITS "
                   "EXPECTED_DECODE EXPECTED_LAYER DUMP_PATH "
                   "EXPECTED_CHUNKED\n";
      return 2;
    }
    int device_count = 0;
    require(evo::cuda::cuda_status(cudaGetDeviceCount(&device_count),
                                   "cudaGetDeviceCount"),
            "count CUDA devices");
    if (device_count < 1) {
      std::cout << "SKIP: one CUDA device is required\n";
      return 77;
    }

    evo::ModelFile file;
    require(file.open(argv[1]), "open synthetic pipeline model");
    {
      evo::cuda::PipelineModel one_gpu_model;
      require(one_gpu_model.load(file, {0}, 12, true),
              "load one-GPU pipeline model");
      const auto &one_gpu_stages = one_gpu_model.stages();
      check(one_gpu_stages.size() == 1 && one_gpu_stages[0].layer_begin == 0 &&
                one_gpu_stages[0].layer_end == 50,
            "one-GPU pipeline owns every layer without P2P");
      std::vector<float> one_gpu_logits;
      require(one_gpu_model.prefill({2, 5, 7, 3}, &one_gpu_logits),
              "one-GPU 50-layer pipeline prefill");
      const auto expected_logits = read_f32(argv[2]);
      check(all_close(one_gpu_logits, expected_logits, 0.08F, 0.06F) &&
                cosine(one_gpu_logits, expected_logits) >= 0.999F,
            "one-GPU pipeline logits match independent oracle");
      evo::cuda::PipelineModel shared_model;
      require(shared_model.initialize_shared(one_gpu_model, 12),
              "initialize shared-weight pipeline context");
      check(shared_model.shares_weights_with(one_gpu_model),
            "shared context aliases every immutable device weight");
      check(shared_model.stages()[0].weight_bytes ==
                    one_gpu_model.stages()[0].weight_bytes &&
                shared_model.stages()[0].cache_bytes ==
                    one_gpu_model.stages()[0].cache_bytes &&
                shared_model.stages()[0].arena_bytes ==
                    one_gpu_model.stages()[0].arena_bytes,
            "shared context retains independent cache/arena accounting");
      std::vector<float> shared_logits;
      require(shared_model.prefill({2, 5, 7, 3}, &shared_logits),
              "shared-weight pipeline prefill");
      check(shared_logits == one_gpu_logits,
            "shared-weight context preserves byte-exact logits");
    }
    if (device_count < 2) {
      std::cout << "SKIP: two-GPU checks require two CUDA devices\n";
      return failures == 0 ? 0 : 1;
    }
    {
      evo::cuda::PipelineModel two_gpu_model;
      require(two_gpu_model.load(file, {0, 1}, 12, true),
              "load two-GPU pipeline model");
      const auto &two_gpu_stages = two_gpu_model.stages();
      check(two_gpu_stages.size() == 2 && two_gpu_stages[0].layer_begin == 0 &&
                two_gpu_stages[0].layer_end == 25 &&
                two_gpu_stages[1].layer_begin == 25 &&
                two_gpu_stages[1].layer_end == 50,
            "two-GPU pipeline splits all 50 layers deterministically");
      std::vector<float> two_gpu_logits;
      require(two_gpu_model.prefill({2, 5, 7, 3}, &two_gpu_logits),
              "two-GPU 50-layer prefill");
      const auto expected_logits = read_f32(argv[2]);
      check(all_close(two_gpu_logits, expected_logits, 0.08F, 0.06F) &&
                cosine(two_gpu_logits, expected_logits) >= 0.999F,
            "two-GPU pipeline logits match independent oracle");
    }

    if (device_count < 4) {
      std::cout << "SKIP: four-GPU checks require four CUDA devices\n";
      return failures == 0 ? 0 : 1;
    }
    evo::cuda::PipelineModel model;
    check(!model.load(file, {0, 0, 1, 2}, 8, true).ok(),
          "pipeline rejects duplicate CUDA devices");
    require(model.load(file, {0, 1, 2, 3}, 12, true),
            "load four-GPU pipeline model");
    check(model.activation_capacity() == 8,
          "pipeline exposes its fixed eight-token activation arena");

    const auto &stages = model.stages();
    const std::vector<evo::cuda::StageAssignment> expected_stages{
        {0, 0, 13, 0, 0, 0},
        {1, 13, 26, 0, 0, 0},
        {2, 26, 38, 0, 0, 0},
        {3, 38, 50, 0, 0, 0}};
    check(stages.size() == expected_stages.size(),
          "pipeline has exactly four stages");
    for (std::size_t index = 0;
         index < stages.size() && index < expected_stages.size(); ++index) {
      check(stages[index].device == expected_stages[index].device &&
                stages[index].layer_begin ==
                    expected_stages[index].layer_begin &&
                stages[index].layer_end == expected_stages[index].layer_end,
            "pipeline layer ownership is deterministic and contiguous");
      check(stages[index].weight_bytes > 0 && stages[index].cache_bytes > 0 &&
                stages[index].arena_bytes > 0,
            "each pipeline stage owns weights, cache, and an activation arena");
    }
    std::uint64_t file_weight_bytes = 0;
    for (const auto &tensor : file.tensors())
      file_weight_bytes += tensor.data_size;
    std::uint64_t assigned_weight_bytes = 0;
    for (const auto &stage : stages)
      assigned_weight_bytes += stage.weight_bytes;
    const auto &config = model.config();
    const auto hcm_layers = static_cast<std::uint64_t>(
        std::count(config.mixer_types.begin(), config.mixer_types.end(),
                   evo::cuda::MixerType::kHcm));
    const std::uint64_t hcm_element_bytes =
        config.hcm_filter_dtype == evo::cuda::HcmFilterDType::kF32 ? 4 : 2;
    const std::uint64_t grouped_filter_expansion =
        hcm_layers * (config.width - config.hcm_filter_groups) *
        config.hcm_filter_length * hcm_element_bytes;
    check(assigned_weight_bytes == file_weight_bytes + grouped_filter_expansion,
          "pipeline counts each checkpoint tensor plus the exact expanded HCM "
          "filter storage once");

    std::vector<float> rejected_logits;
    check(!model
               .prefill_with_dumps(
                   {1}, &rejected_logits,
                   {{17, std::string{argv[5]} + ".invalid.npy",
                     static_cast<evo::cuda::LayerDumpPoint>(999)}})
               .ok(),
          "pipeline rejects an invalid intermediate dump point");

    const std::vector<evo::TokenId> prompt{2, 5, 7, 3};
    std::vector<float> logits;
    const std::string copy_path = std::string{argv[5]} + ".copy.npy";
    const std::string norm_path = std::string{argv[5]} + ".pre_norm.npy";
    const std::vector<evo::cuda::LayerDump> dumps{
        {17, argv[5]},
        {17, copy_path},
        {17, norm_path, evo::cuda::LayerDumpPoint::kPreNorm}};
    std::vector<float> stateless_logits;
    require(model.prefill_stateless(prompt, &stateless_logits),
            "stateless pipeline prefill");
    check(model.position() == 0,
          "stateless pipeline prefill has no continuation position");
    require(model.prefill_with_dumps(prompt, &logits, dumps),
            "four-GPU 50-layer prefill with multiple dumps");
    const auto expected_logits = read_f32(argv[2]);
    check(all_close(logits, expected_logits, 0.08F, 0.06F),
          "pipeline prefill logits match independent oracle");
    check(stateless_logits == logits,
          "stateless and stateful pipeline logits are bit-identical");
    check(cosine(logits, expected_logits) >= 0.999F,
          "pipeline prefill cosine is at least 0.999");
    check(model.position() == prompt.size(),
          "pipeline prefill records prompt position");

    const auto expected_layer = read_f32(argv[4]);
    const auto dumped_layer = read_npy_f32(argv[5], expected_layer.size());
    const auto copied_layer = read_npy_f32(copy_path, expected_layer.size());
    const auto dumped_norm = read_npy_f32(norm_path, expected_layer.size());
    check(all_close(dumped_layer, expected_layer, 0.06F, 0.05F),
          "pipeline stage-local layer dump matches oracle");
    check(dumped_layer == copied_layer,
          "multiple dumps from one layer are bit-identical");
    check(
        dumped_norm != dumped_layer &&
            std::all_of(dumped_norm.begin(), dumped_norm.end(),
                        [](const float value) { return std::isfinite(value); }),
        "named intermediate dump captures finite pre-norm values");

    std::vector<float> decoded;
    require(model.decode(9, &decoded), "four-GPU cached decode");
    const auto expected_decode = read_f32(argv[3]);
    check(all_close(decoded, expected_decode, 0.08F, 0.06F),
          "pipeline decode logits match independent oracle");
    check(cosine(decoded, expected_decode) >= 0.999F,
          "pipeline decode cosine is at least 0.999");
    check(model.position() == prompt.size() + 1,
          "pipeline decode advances position");

    const std::vector<evo::TokenId> long_prompt{2, 5, 7, 3, 9, 11, 13, 17, 19};
    const std::vector<evo::TokenId> initial_chunk(long_prompt.begin(),
                                                  long_prompt.begin() + 8);
    std::vector<float> first_chunk;
    std::vector<float> final_chunk;
    require(model.prefill(initial_chunk, &first_chunk),
            "eight-token initial pipeline chunk");
    require(model.prefill_chunk({long_prompt.back()}, &final_chunk),
            "one-token continued pipeline chunk");
    const auto expected_chunked = read_f32(argv[6]);
    const std::vector<float> expected_last(
        expected_chunked.end() -
            static_cast<std::ptrdiff_t>(model.config().vocab_size),
        expected_chunked.end());
    check(all_close(final_chunk, expected_last, 0.08F, 0.06F),
          "chunked pipeline final logits match full Python causal oracle");
    check(cosine(final_chunk, expected_last) >= 0.999F,
          "chunked pipeline final-logit cosine is at least 0.999");
    check(model.position() == long_prompt.size(),
          "chunked pipeline records the full logical position");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " CUDA pipeline test(s) failed\n";
    return 1;
  }
  std::cout << "CUDA pipeline tests passed\n";
  return 0;
}
