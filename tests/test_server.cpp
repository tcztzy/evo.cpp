// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "evo/json.hpp"
#include "evo/server.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

void test_json() {
  evo::JsonValue value;
  auto status = evo::parse_json(
      R"({"sequence":"ACGT","enabled":true,"values":[1,-2.5e1,null],"emoji":"\ud83e\uddec"})",
      &value);
  check(status.ok() && value.type == evo::JsonType::kObject,
        "strict JSON object parses");
  const auto *sequence = value.find("sequence");
  const auto *emoji = value.find("emoji");
  check(sequence != nullptr && sequence->type == evo::JsonType::kString &&
            sequence->string == "ACGT",
        "JSON string member is retained");
  check(emoji != nullptr && emoji->string == "\xF0\x9F\xA7\xAC",
        "JSON surrogate pair decodes to UTF-8");

  check(!evo::parse_json(R"({"x":1,"x":2})", &value).ok(),
        "duplicate JSON keys fail closed");
  check(!evo::parse_json("[01]", &value).ok(),
        "leading-zero JSON number fails closed");
  check(!evo::parse_json("[[[0]]]", &value, 2).ok(),
        "JSON nesting limit is enforced");
  check(!evo::parse_json("NaN", &value).ok(),
        "non-standard numeric literals fail closed");

  std::string encoded;
  evo::append_json_string(&encoded, std::string{"A\n\0\xFF", 4});
  check(encoded == R"("A\n\u0000\u00ff")",
        "JSON serializer escapes controls and arbitrary bytes");
}

void test_scheduler() {
  using namespace std::chrono_literals;
  {
    evo::DynamicScheduler scheduler{8, 2, 5s};
    std::atomic<int> entered{0};
    auto task = [&entered](const evo::CancellationToken &) {
      ++entered;
      std::this_thread::sleep_for(30ms);
      return evo::ServerResponse{200, "application/json", "{}"};
    };
    auto first = scheduler.submit(task);
    auto second = scheduler.submit(task);
    check(first.has_value() && second.has_value(),
          "scheduler accepts work below its queue limit");
    if (first && second) {
      check(first->future().get().http_status == 200 &&
                second->future().get().http_status == 200,
            "scheduled tasks complete independently");
    }
    const auto metrics = scheduler.metrics();
    check(entered == 2 && metrics.submitted == 2 && metrics.batches == 1 &&
              metrics.batch_items == 2 && metrics.active_peak == 2 &&
              metrics.completed == 2,
          "batch window coalesces requests while retaining isolated tasks");
  }

  evo::DynamicScheduler scheduler{8, 1, 0ms};

  auto cancelled = scheduler.submit([](const evo::CancellationToken &) {
    return evo::ServerResponse{200, "application/json", "{}"};
  });
  check(cancelled.has_value(), "cancellable work is accepted");
  if (cancelled) {
    cancelled->cancel();
    check(cancelled->future().get().http_status == 499,
          "cancelled queued work does not execute");
  }
  check(scheduler.metrics().cancelled == 1,
        "scheduler exposes cancellation metrics");

  auto throwing = scheduler.submit(
      [](const evo::CancellationToken &) -> evo::ServerResponse {
        throw std::runtime_error("sensitive implementation detail");
      });
  if (throwing) {
    const auto response = throwing->future().get();
    check(response.http_status == 500 &&
              response.body.find("sensitive") == std::string::npos,
          "task exceptions become sanitized server errors");
  }
}

} // namespace

int main() {
  test_json();
  test_scheduler();
  if (failures != 0) {
    std::cerr << failures << " server-core test(s) failed\n";
    return 1;
  }
  std::cout << "server-core tests passed\n";
  return 0;
}
