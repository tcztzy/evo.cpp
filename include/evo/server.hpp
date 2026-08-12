// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>

#include "evo/status.hpp"

namespace evo {

struct CliOptions;

struct ServerResponse final {
  int http_status{500};
  std::string content_type{"application/json"};
  std::string body;
};

class CancellationToken final {
public:
  CancellationToken();
  [[nodiscard]] bool cancelled() const noexcept;
  void cancel() noexcept;

private:
  struct State;
  std::shared_ptr<State> state_;
  friend class DynamicScheduler;
};

class ScheduledRequest final {
public:
  ScheduledRequest(ScheduledRequest &&) noexcept;
  ScheduledRequest &operator=(ScheduledRequest &&) noexcept;
  ~ScheduledRequest();

  ScheduledRequest(const ScheduledRequest &) = delete;
  ScheduledRequest &operator=(const ScheduledRequest &) = delete;

  void cancel() noexcept;
  [[nodiscard]] bool cancelled() const noexcept;
  [[nodiscard]] std::future<ServerResponse> &future() noexcept;

private:
  ScheduledRequest(CancellationToken token, std::future<ServerResponse> future);
  CancellationToken token_;
  std::future<ServerResponse> future_;
  friend class DynamicScheduler;
};

struct SchedulerMetrics final {
  std::uint64_t submitted{0};
  std::uint64_t rejected{0};
  std::uint64_t batches{0};
  std::uint64_t batch_items{0};
  std::uint64_t completed{0};
  std::uint64_t failed{0};
  std::uint64_t cancelled{0};
  std::size_t queued{0};
  std::size_t queue_peak{0};
  std::size_t active{0};
  std::size_t active_peak{0};
};

class DynamicScheduler final {
public:
  using Task = std::function<ServerResponse(const CancellationToken &)>;

  DynamicScheduler(std::size_t maximum_queue, std::size_t maximum_batch,
                   std::chrono::milliseconds batch_window);
  ~DynamicScheduler();

  DynamicScheduler(const DynamicScheduler &) = delete;
  DynamicScheduler &operator=(const DynamicScheduler &) = delete;

  [[nodiscard]] std::optional<ScheduledRequest> submit(Task task);
  [[nodiscard]] SchedulerMetrics metrics() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] Status run_server(const CliOptions &options,
                                bool allow_test_fixture);

} // namespace evo
