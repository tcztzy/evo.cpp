// SPDX-License-Identifier: Apache-2.0
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
    if (argc != 6) {
      std::cerr << "usage: test_cuda_pipeline MODEL EXPECTED_LOGITS "
                   "EXPECTED_DECODE EXPECTED_LAYER DUMP_PATH\n";
      return 2;
    }
    int device_count = 0;
    require(evo2c::cuda::cuda_status(cudaGetDeviceCount(&device_count),
                                     "cudaGetDeviceCount"),
            "count CUDA devices");
    if (device_count < 4) {
      std::cout << "SKIP: four CUDA devices are required\n";
      return 77;
    }

    evo2c::ModelFile file;
    require(file.open(argv[1]), "open synthetic pipeline model");
    evo2c::cuda::PipelineModel model;
    check(!model.load(file, {0, 0, 1, 2}, 8, true).ok(),
          "pipeline rejects duplicate CUDA devices");
    require(model.load(file, {0, 1, 2, 3}, 8, true),
            "load four-GPU pipeline model");

    const auto &stages = model.stages();
    const std::vector<evo2c::cuda::StageAssignment> expected_stages{
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
    check(assigned_weight_bytes == file_weight_bytes,
          "every model tensor is placed on exactly one pipeline stage");

    const std::vector<evo2c::TokenId> prompt{2, 5, 7, 3};
    std::vector<float> logits;
    require(model.prefill(prompt, &logits, evo2c::cuda::LayerDump{17, argv[5]}),
            "four-GPU 50-layer prefill");
    const auto expected_logits = read_f32(argv[2]);
    check(all_close(logits, expected_logits, 0.08F, 0.06F),
          "pipeline prefill logits match independent oracle");
    check(cosine(logits, expected_logits) >= 0.999F,
          "pipeline prefill cosine is at least 0.999");
    check(model.position() == prompt.size(),
          "pipeline prefill records prompt position");

    const auto expected_layer = read_f32(argv[4]);
    const auto dumped_layer = read_npy_f32(argv[5], expected_layer.size());
    check(all_close(dumped_layer, expected_layer, 0.06F, 0.05F),
          "pipeline stage-local layer dump matches oracle");

    std::vector<float> decoded;
    require(model.decode(9, &decoded), "four-GPU cached decode");
    const auto expected_decode = read_f32(argv[3]);
    check(all_close(decoded, expected_decode, 0.08F, 0.06F),
          "pipeline decode logits match independent oracle");
    check(cosine(decoded, expected_decode) >= 0.999F,
          "pipeline decode cosine is at least 0.999");
    check(model.position() == prompt.size() + 1,
          "pipeline decode advances position");
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
