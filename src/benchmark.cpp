// SPDX-License-Identifier: Apache-2.0
#include "evo/benchmark.hpp"

#include "evo/json.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace evo {
namespace {

void write_string(std::ostream &output, const std::string_view value) {
  std::string encoded;
  append_json_string(&encoded, value);
  output << encoded;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  return values.size() % 2 == 0
             ? (values[middle - 1] + values[middle]) / 2.0
             : values[middle];
}

} // namespace

std::uint64_t fnv1a64(const std::string_view bytes) noexcept {
  std::uint64_t value = UINT64_C(14695981039346656037);
  for (const char character : bytes) {
    value ^= static_cast<unsigned char>(character);
    value *= UINT64_C(1099511628211);
  }
  return value;
}

Status write_benchmark_report(std::ostream &output,
                              const BenchmarkReport &report) {
  if (report.model_path.empty() || report.architecture.empty() ||
      report.artifact_profile.empty() ||
      report.execution_profile.empty() || report.backend.empty() ||
      report.input_path.empty() || report.input_name.empty() ||
      report.context_size == 0 || report.repetitions == 0 || report.tokens == 0 ||
      report.samples_seconds.size() != report.repetitions) {
    return {ErrorCode::kInvalidArgument,
            "benchmark report identity or sample count is incomplete"};
  }
  for (const double sample : report.samples_seconds) {
    if (!std::isfinite(sample) || sample <= 0.0)
      return {ErrorCode::kInternal,
              "benchmark produced a nonpositive or nonfinite duration"};
  }
  const double median_seconds = median(report.samples_seconds);
  const double tokens_per_second =
      static_cast<double>(report.tokens) / median_seconds;
  std::ostringstream hash;
  hash << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
       << report.input_fnv1a64;

  output << "{\"schema_version\":1,\"command\":\"bench\",\"model_path\":";
  write_string(output, report.model_path);
  output << ",\"model_id\":";
  if (report.model_id.empty())
    output << "null";
  else
    write_string(output, report.model_id);
  output << ",\"architecture\":";
  write_string(output, report.architecture);
  output << ",\"artifact_profile\":";
  write_string(output, report.artifact_profile);
  output << ",\"execution_profile\":";
  write_string(output, report.execution_profile);
  output << ",\"backend\":";
  write_string(output, report.backend);
  output << ",\"input\":{\"path\":";
  write_string(output, report.input_path);
  output << ",\"name\":";
  write_string(output, report.input_name);
  output << ",\"format\":\"" << sequence_format_name(report.input_format)
         << "\",\"identity\":";
  write_string(output, hash.str());
  output << "},\"context_size\":" << report.context_size
         << ",\"warmup\":" << report.warmup
         << ",\"repetitions\":" << report.repetitions
         << ",\"tokens\":" << report.tokens << ",\"timing_scope\":";
  write_string(output, report.timing_scope);
  output << ",\"samples_seconds\":[" << std::setprecision(17);
  for (std::size_t index = 0; index < report.samples_seconds.size(); ++index) {
    if (index != 0)
      output.put(',');
    output << report.samples_seconds[index];
  }
  output << "],\"statistic\":\"median\",\"median_seconds\":"
         << median_seconds << ",\"median_tokens_per_second\":"
         << tokens_per_second << "}\n";
  return output ? Status::Ok()
                : Status{ErrorCode::kIo,
                         "failed writing benchmark JSONL to stdout"};
}

} // namespace evo
