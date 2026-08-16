// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "evo/cpu/model.hpp"
#include "evo/model_format.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string &description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

std::vector<float> read_f32(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return {};
  const auto bytes = input.tellg();
  if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
    return {};
  input.seekg(0);
  std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
  input.read(reinterpret_cast<char *>(values.data()), bytes);
  return input ? values : std::vector<float>{};
}

float maximum_error(const std::vector<float> &left,
                    const std::vector<float> &right) {
  if (left.size() != right.size() || left.empty())
    return std::numeric_limits<float>::infinity();
  float maximum = 0.0F;
  for (std::size_t index = 0; index < left.size(); ++index)
    maximum = std::max(maximum, std::fabs(left[index] - right[index]));
  return maximum;
}

} // namespace

int main(const int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: test_cpu_model MODEL LOGITS DECODE LAYER\n";
    return 2;
  }
  evo::ModelFile artifact;
  auto status = artifact.open(argv[1]);
  check(status.ok(), "CPU fixture artifact opens: " + status.message());
  if (!status.ok())
    return 1;

  evo::cpu::Model model;
  status = model.load(artifact, true);
  check(status.ok(), "CPU model loads: " + status.message());
  check(model.config().layers == 50 && model.config().width == 8 &&
            model.config().vocab_size == 512 &&
            model.config().implementation ==
                evo::ArchitectureImplementation::kStripedHyena2,
        "CPU model exposes validated configuration");
  const std::string kernel = model.kernel_name();
  check(kernel == "neon-f32" || kernel == "avx2-fma-f32" ||
            kernel == "scalar-f32",
        "CPU model reports its selected vector kernel");
  if (!status.ok())
    return 1;

  evo::cpu::Context context;
  status = context.initialize_shared(model, 12);
  check(status.ok(), "CPU context shares immutable weights");
  std::vector<float> logits;
  status = context.prefill({2, 5, 7, 3}, &logits);
  check(status.ok() && logits.size() == 4 * 512,
        "CPU prefill returns complete logits");
  const auto expected_logits = read_f32(argv[2]);
  check(maximum_error(logits, expected_logits) < 0.08F,
        "portable CPU logits stay within the declared F32 profile envelope");

  std::vector<float> decode;
  status = context.decode(9, &decode);
  check(status.ok() && decode.size() == 512 && context.position() == 5,
        "CPU decode advances recurrent and attention caches");
  const auto expected_decode = read_f32(argv[3]);
  check(maximum_error(decode, expected_decode) < 0.08F,
        "portable CPU decode stays within the declared F32 profile envelope");

  evo::cpu::Context embedding_context;
  status = embedding_context.initialize_shared(model, 12);
  std::vector<float> embedding;
  if (status.ok())
    status = embedding_context.prefill_embedding({2, 5, 7, 3}, 17, &embedding);
  check(status.ok() && embedding.size() == 4 * 8,
        "CPU intermediate embeddings use the shared context backend");
  const auto expected_layer = read_f32(argv[4]);
  check(maximum_error(embedding, expected_layer) < 0.08F,
        "CPU embeddings stay within the declared F32 profile envelope");

  evo::cpu::Context repeat;
  status = repeat.initialize_shared(model, 12);
  std::vector<float> repeated;
  if (status.ok())
    status = repeat.prefill({2, 5, 7, 3}, &repeated);
  check(status.ok() && repeated == logits,
        "same CPU build, artifact, profile, and input are deterministic");

  if (failures != 0) {
    std::cerr << failures << " CPU model test(s) failed\n";
    return 1;
  }
  std::cout << "CPU model tests passed with kernel " << kernel << '\n';
  return 0;
}
