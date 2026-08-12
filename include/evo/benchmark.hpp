// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "evo/sequence_io.hpp"
#include "evo/status.hpp"

namespace evo {

struct BenchmarkReport final {
  std::string model_path;
  std::string model_id;
  std::string architecture;
  std::string artifact_profile;
  std::string execution_profile;
  std::string backend;
  std::string input_path;
  std::string input_name;
  SequenceFormat input_format{SequenceFormat::kRaw};
  std::uint64_t input_fnv1a64{0};
  std::size_t context_size{0};
  std::size_t warmup{0};
  std::size_t repetitions{0};
  std::size_t tokens{0};
  std::string timing_scope{"prefill"};
  std::vector<double> samples_seconds;
};

[[nodiscard]] std::uint64_t fnv1a64(std::string_view bytes) noexcept;
[[nodiscard]] Status write_benchmark_report(std::ostream &output,
                                            const BenchmarkReport &report);

} // namespace evo
