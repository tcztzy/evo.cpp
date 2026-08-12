// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "evo/cpu/model.hpp"
#include "evo/model_format.hpp"

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
  if (maximum > 3.0e-4F) {
    std::cerr << "FAIL: " << description << " max_abs=" << maximum << '\n';
    ++failures;
  }
}

} // namespace

int main(const int argc, char **const argv) {
  if (argc != 3) {
    std::cerr << "usage: test_esmc_cpu ARTIFACT ORACLE_DIR\n";
    return 2;
  }
  evo::ModelFile artifact;
  auto status = artifact.open(argv[1]);
  check(status.ok(), "ESMC fixture opens through the strict native loader");
  if (!status.ok())
    return 1;

  evo::cpu::Model denied;
  status = denied.load(artifact, false);
  check(!status.ok() && status.code() == evo::ErrorCode::kUnsupported,
        "synthetic ESMC fixture requires explicit test permission");

  evo::cpu::Model model;
  status = model.load(artifact, true);
  check(status.ok(), "ESMC fixture loads in the CPU backend");
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 1;
  }
  check(model.config().architecture == "ESMCTest" &&
            model.config().vocab_size == 64 && model.config().width == 4 &&
            model.config().layers == 2,
        "ESMC CPU configuration preserves fixture topology");
  check(std::string_view{model.kernel_name()} == "scalar-f32-bidirectional",
        "ESMC CPU backend reports its bidirectional F32 kernel");

  constexpr std::string_view sequence = "LAG<mask>|Z<pad>";
  std::vector<evo::TokenId> tokens;
  status = model.encode(sequence, &tokens);
  check(status.ok() &&
            tokens == std::vector<evo::TokenId>{0, 4, 5, 6, 32, 31, 27, 1, 2},
        "ESMC CPU model uses the official protein tokenizer");

  const std::filesystem::path oracle_dir{argv[2]};
  evo::cpu::Context logits_context;
  status = logits_context.initialize_shared(model, tokens.size());
  check(status.ok(), "ESMC logits context initializes at encoded capacity");
  std::vector<float> logits;
  if (status.ok())
    status = logits_context.prefill(tokens, &logits);
  check(status.ok(), "ESMC CPU full-sequence logits execute");
  check_close(logits, read_floats(oracle_dir / "logits.f32"),
              "ESMC CPU logits match the independent scalar oracle");
  std::vector<float> rejected;
  status = logits_context.prefill_chunk({4}, &rejected);
  check(!status.ok() && status.code() == evo::ErrorCode::kUnsupported,
        "ESMC CPU continuation is typed unsupported");
  status = logits_context.decode(4, &rejected);
  check(!status.ok() && status.code() == evo::ErrorCode::kUnsupported,
        "ESMC CPU autoregressive decode is typed unsupported");

  for (std::size_t layer = 0; layer <= model.config().layers; ++layer) {
    evo::cpu::Context context;
    status = context.initialize_shared(model, tokens.size());
    std::vector<float> embedding;
    if (status.ok())
      status = context.prefill_embedding(tokens, layer, &embedding);
    check(status.ok(), "ESMC indexed embedding executes");
    check_close(
        embedding,
        read_floats(oracle_dir / ("layer-" + std::to_string(layer) + ".f32")),
        "ESMC hidden-state index matches pinned Transformers semantics");
  }

  if (failures != 0) {
    std::cerr << failures << " ESMC CPU test(s) failed\n";
    return 1;
  }
  std::cout << "ESMC CPU backend tests passed\n";
  return 0;
}
