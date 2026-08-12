// SPDX-License-Identifier: Apache-2.0
#include "evo/server.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace evo {

struct CancellationToken::State final {
  std::atomic<bool> cancelled{false};
};

CancellationToken::CancellationToken() : state_(std::make_shared<State>()) {}

bool CancellationToken::cancelled() const noexcept {
  return state_ && state_->cancelled.load(std::memory_order_relaxed);
}

void CancellationToken::cancel() noexcept {
  if (state_)
    state_->cancelled.store(true, std::memory_order_relaxed);
}

ScheduledRequest::ScheduledRequest(CancellationToken token,
                                   std::future<ServerResponse> future)
    : token_(std::move(token)), future_(std::move(future)) {}

ScheduledRequest::ScheduledRequest(ScheduledRequest &&) noexcept = default;
ScheduledRequest &
ScheduledRequest::operator=(ScheduledRequest &&) noexcept = default;
ScheduledRequest::~ScheduledRequest() = default;

void ScheduledRequest::cancel() noexcept { token_.cancel(); }

bool ScheduledRequest::cancelled() const noexcept { return token_.cancelled(); }

std::future<ServerResponse> &ScheduledRequest::future() noexcept {
  return future_;
}

struct DynamicScheduler::Impl final {
  struct Work final {
    Task task;
    CancellationToken token;
    std::promise<ServerResponse> promise;
  };

  Impl(const std::size_t queue_limit, const std::size_t batch_limit,
       const std::chrono::milliseconds window)
      : maximum_queue(queue_limit), maximum_batch(batch_limit),
        batch_window(window), dispatcher([this] { dispatch(); }) {}

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      for (auto &work : pending)
        work->token.cancel();
    }
    condition.notify_all();
    if (dispatcher.joinable())
      dispatcher.join();
  }

  void execute(const std::shared_ptr<Work> &work) noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex);
      ++snapshot.active;
      snapshot.active_peak = std::max(snapshot.active_peak, snapshot.active);
    }
    ServerResponse response;
    try {
      if (work->token.cancelled()) {
        response = {499, "application/json",
                    "{\"error\":{\"code\":\"cancelled\",\"message\":"
                    "\"request was cancelled before execution\"}}"};
      } else {
        response = work->task(work->token);
      }
    } catch (const std::exception &) {
      response = {500, "application/json",
                  "{\"error\":{\"code\":\"internal\",\"message\":"
                  "\"scheduler task threw\"}}"};
    } catch (...) {
      response = {500, "application/json",
                  "{\"error\":{\"code\":\"internal\",\"message\":"
                  "\"scheduler task threw\"}}"};
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      --snapshot.active;
      if (work->token.cancelled() || response.http_status == 499) {
        ++snapshot.cancelled;
      } else if (response.http_status >= 400) {
        ++snapshot.failed;
      } else {
        ++snapshot.completed;
      }
    }
    work->promise.set_value(std::move(response));
  }

  void dispatch() noexcept {
    while (true) {
      std::vector<std::shared_ptr<Work>> batch;
      {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return stopping || !pending.empty(); });
        if (stopping && pending.empty())
          return;
        const auto deadline = std::chrono::steady_clock::now() + batch_window;
        condition.wait_until(lock, deadline, [&] {
          return stopping || pending.size() >= maximum_batch;
        });
        const std::size_t count = std::min(maximum_batch, pending.size());
        batch.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
          batch.push_back(std::move(pending.front()));
          pending.pop_front();
        }
        snapshot.queued = pending.size();
        ++snapshot.batches;
        snapshot.batch_items += count;
      }
      std::vector<std::thread> workers;
      std::size_t spawned = 0;
      try {
        workers.reserve(batch.size());
        for (; spawned < batch.size(); ++spawned) {
          const auto work = batch[spawned];
          workers.emplace_back([this, work] { execute(work); });
        }
      } catch (...) {
        // Resource exhaustion must not terminate the dispatcher. Preserve
        // progress and request isolation by executing unspawned work here.
        for (; spawned < batch.size(); ++spawned)
          execute(batch[spawned]);
      }
      for (auto &worker : workers)
        worker.join();
    }
  }

  const std::size_t maximum_queue;
  const std::size_t maximum_batch;
  const std::chrono::milliseconds batch_window;
  mutable std::mutex mutex;
  std::condition_variable condition;
  std::deque<std::shared_ptr<Work>> pending;
  bool stopping{false};
  SchedulerMetrics snapshot;
  std::thread dispatcher;
};

DynamicScheduler::DynamicScheduler(
    const std::size_t maximum_queue, const std::size_t maximum_batch,
    const std::chrono::milliseconds batch_window) {
  if (maximum_queue == 0 || maximum_batch == 0 ||
      maximum_batch > maximum_queue || batch_window.count() < 0) {
    throw std::invalid_argument("invalid dynamic scheduler limits");
  }
  impl_ = std::make_unique<Impl>(maximum_queue, maximum_batch, batch_window);
}

DynamicScheduler::~DynamicScheduler() = default;

std::optional<ScheduledRequest> DynamicScheduler::submit(Task task) {
  if (!task)
    return std::nullopt;
  auto work = std::make_shared<Impl::Work>();
  work->task = std::move(task);
  auto future = work->promise.get_future();
  const CancellationToken token = work->token;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping || impl_->pending.size() >= impl_->maximum_queue) {
      ++impl_->snapshot.rejected;
      return std::nullopt;
    }
    impl_->pending.push_back(work);
    ++impl_->snapshot.submitted;
    impl_->snapshot.queued = impl_->pending.size();
    impl_->snapshot.queue_peak =
        std::max(impl_->snapshot.queue_peak, impl_->snapshot.queued);
  }
  impl_->condition.notify_one();
  return ScheduledRequest{token, std::move(future)};
}

SchedulerMetrics DynamicScheduler::metrics() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->snapshot;
}

} // namespace evo
