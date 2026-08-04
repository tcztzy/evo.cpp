// SPDX-License-Identifier: Apache-2.0
#include "evo/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace evo {

Status validate_sampling_config(const SamplingConfig& config) {
  if (!std::isfinite(config.temperature) || config.temperature <= 0.0F) {
    return {ErrorCode::kInvalidArgument, "temperature must be finite and greater than zero"};
  }
  if (config.top_k > kTokenizerVocabSize) {
    return {ErrorCode::kInvalidArgument, "top-k must be in [0, 512]"};
  }
  if (!std::isfinite(config.top_p) || config.top_p <= 0.0F || config.top_p > 1.0F) {
    return {ErrorCode::kInvalidArgument, "top-p must be finite and in (0, 1]"};
  }
  return Status::Ok();
}

Sampler::Sampler(SamplingConfig config) : config_(config), random_(config.seed) {}

Status Sampler::sample(const std::vector<float>& logits, TokenId* const token) {
  if (token == nullptr) {
    return {ErrorCode::kInvalidArgument, "sample output pointer is null"};
  }
  auto status = validate_sampling_config(config_);
  if (!status.ok()) {
    return status;
  }
  if (logits.size() != kTokenizerVocabSize) {
    return {ErrorCode::kInvalidArgument,
            "sampler requires exactly 512 logits, got " + std::to_string(logits.size())};
  }
  for (std::size_t index = 0; index < logits.size(); ++index) {
    if (!std::isfinite(logits[index])) {
      return {ErrorCode::kInvalidArgument,
              "logit " + std::to_string(index) + " is not finite"};
    }
  }

  std::vector<std::size_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&logits](const std::size_t left,
                                                         const std::size_t right) {
    if (logits[left] == logits[right]) {
      return left < right;
    }
    return logits[left] > logits[right];
  });

  if (config_.top_k == 1) {
    *token = static_cast<TokenId>(order.front());
    return Status::Ok();
  }
  if (config_.top_k != 0 && order.size() > config_.top_k) {
    order.resize(config_.top_k);
  }

  std::vector<double> weights;
  weights.reserve(order.size());
  const double scaled_max = static_cast<double>(logits[order.front()]) /
                            static_cast<double>(config_.temperature);
  double total = 0.0;
  for (const auto index : order) {
    const double scaled = static_cast<double>(logits[index]) /
                          static_cast<double>(config_.temperature);
    const double weight = std::exp(scaled - scaled_max);
    weights.push_back(weight);
    total += weight;
  }

  if (config_.top_p < 1.0F) {
    double cumulative = 0.0;
    std::size_t keep = 0;
    while (keep < weights.size() && cumulative / total < static_cast<double>(config_.top_p)) {
      cumulative += weights[keep];
      ++keep;
    }
    order.resize(keep);
    weights.resize(keep);
    total = cumulative;
  }

  constexpr double kUnit53 = 1.0 / 9007199254740992.0;
  const double draw = static_cast<double>(random_() >> 11U) * kUnit53 * total;
  double cumulative = 0.0;
  for (std::size_t index = 0; index < order.size(); ++index) {
    cumulative += weights[index];
    if (draw < cumulative || index + 1 == order.size()) {
      *token = static_cast<TokenId>(order[index]);
      return Status::Ok();
    }
  }
  return {ErrorCode::kInternal, "sampler failed to select a token"};
}

}  // namespace evo
