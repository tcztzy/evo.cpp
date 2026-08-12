// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include "evo/cuda/esmc.hpp"
#include "evo/model_format.hpp"
#include "evo/tokenizer.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

std::vector<float> read_floats(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return {};
  const auto end = input.tellg();
  if (end <= 0 || end % static_cast<std::streamoff>(sizeof(float)) != 0)
    return {};
  std::vector<float> values(static_cast<std::size_t>(end) / sizeof(float));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(values.data()), end);
  return input ? values : std::vector<float>{};
}

void check_close(const std::vector<float> &actual,
                 const std::vector<float> &expected,
                 const std::string_view description) {
  if (actual.size() != expected.size()) {
    check(false, description);
    return;
  }
  float maximum = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index)
    maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
  if (maximum > 1.0e-3F) {
    std::cerr << "FAIL: " << description << " max_abs=" << maximum << '\n';
    ++failures;
  }
}

} // namespace

int main(const int argc, char **const argv) {
  if (argc != 3) {
    std::cerr << "usage: test_cuda_esmc ARTIFACT ORACLE_DIR\n";
    return 2;
  }
  evo::ModelFile artifact;
  auto status = artifact.open(argv[1]);
  check(status.ok(), "CUDA ESMC fixture opens through the strict loader");
  if (!status.ok())
    return 1;

  evo::cuda::EsmcModel multi_device;
  status = multi_device.load(artifact, {0, 1}, true);
  check(!status.ok() && status.code() == evo::ErrorCode::kUnsupported,
        "CUDA ESMC rejects multi-device execution before allocation");

  evo::cuda::EsmcModel model;
  status = model.load(artifact, {0}, true);
  check(status.ok(), "CUDA ESMC fixture uploads to one device");
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 1;
  }
  check(model.config().width == 4 && model.config().layers == 2 &&
            model.config().vocab_size == 64,
        "CUDA ESMC configuration preserves fixture topology");

  std::vector<evo::TokenId> tokens;
  status = evo::encode_sequence(evo::ArchitectureTokenizer::kEsmcProtein,
                                "LAG<mask>|Z?", &tokens);
  check(status.ok() && tokens.size() == 9,
        "CUDA ESMC test uses the official protein tokenizer");
  const std::filesystem::path oracle_dir{argv[2]};

  evo::cuda::EsmcContext logits_context;
  status = logits_context.initialize_shared(model, tokens.size());
  std::vector<float> logits;
  if (status.ok())
    status = logits_context.prefill(tokens, &logits);
  check(status.ok(), "CUDA ESMC full-sequence logits execute");
  check_close(logits, read_floats(oracle_dir / "logits.f32"),
              "CUDA ESMC logits match the scalar F32 oracle");
  std::vector<float> rejected;
  status = logits_context.prefill_chunk({4}, &rejected);
  check(!status.ok() && status.code() == evo::ErrorCode::kUnsupported,
        "CUDA ESMC continuation is typed unsupported");
  status = logits_context.decode(4, &rejected);
  check(!status.ok() && status.code() == evo::ErrorCode::kUnsupported,
        "CUDA ESMC autoregressive decode is typed unsupported");

  for (std::size_t layer = 0; layer <= model.config().layers; ++layer) {
    evo::cuda::EsmcContext context;
    status = context.initialize_shared(model, tokens.size());
    std::vector<float> embedding;
    if (status.ok())
      status = context.prefill_embedding(tokens, layer, &embedding);
    check(status.ok(), "CUDA ESMC indexed embedding executes");
    check_close(
        embedding,
        read_floats(oracle_dir / ("layer-" + std::to_string(layer) + ".f32")),
        "CUDA ESMC hidden index matches pinned Transformers semantics");
  }

  if (failures != 0) {
    std::cerr << failures << " CUDA ESMC test(s) failed\n";
    return 1;
  }
  std::cout << "CUDA ESMC backend tests passed\n";
  return 0;
}
