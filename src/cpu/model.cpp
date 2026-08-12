// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/model.hpp"

#include "hyenadna_internal.hpp"
#include "evo/model_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#elif (defined(__x86_64__) || defined(_M_X64)) &&                              \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace evo::cpu {
namespace {

constexpr std::size_t kMaximumArenaTokens = 256;

enum class MixerType { kHcs, kHcm, kHcl, kAttention };

std::uint64_t read_u64(const std::uint8_t *const data) noexcept {
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<std::uint64_t>(data[byte]) << (byte * 8U);
  return value;
}

Status metadata_entry(const ModelFile &model, const std::string_view key,
                      const MetadataType type,
                      const MetadataEntry **const output) {
  const auto *const entry = model.find_metadata(key);
  if (entry == nullptr)
    return {ErrorCode::kModelFormat,
            "required metadata is missing: " + std::string{key}};
  if (entry->type != type)
    return {ErrorCode::kModelFormat,
            "metadata has wrong type: " + std::string{key}};
  *output = entry;
  return Status::Ok();
}

Status metadata_size(const ModelFile &model, const std::string_view key,
                     std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  const auto value = read_u64(entry->value.data());
  if (value > std::numeric_limits<std::size_t>::max())
    return {ErrorCode::kModelFormat,
            "metadata exceeds size_t: " + std::string{key}};
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

Status metadata_float(const ModelFile &model, const std::string_view key,
                      float *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kF64, &entry);
  if (!status.ok())
    return status;
  const auto bits = read_u64(entry->value.data());
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  if (!std::isfinite(value) || value > std::numeric_limits<float>::max() ||
      value < -std::numeric_limits<float>::max()) {
    return {ErrorCode::kModelFormat,
            "metadata is not finite F32: " + std::string{key}};
  }
  *output = static_cast<float>(value);
  return Status::Ok();
}

Status metadata_bool(const ModelFile &model, const std::string_view key,
                     bool *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kBool, &entry);
  if (!status.ok())
    return status;
  *output = entry->value[0] != 0;
  return Status::Ok();
}

Status metadata_string(const ModelFile &model, const std::string_view key,
                       std::string *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kString, &entry);
  if (!status.ok())
    return status;
  output->assign(entry->value.begin(), entry->value.end());
  return Status::Ok();
}

Status metadata_list(const ModelFile &model, const std::string_view key,
                     std::vector<std::size_t> *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kU64List, &entry);
  if (!status.ok())
    return status;
  output->clear();
  output->reserve(entry->value.size() / sizeof(std::uint64_t));
  for (std::size_t offset = 0; offset < entry->value.size();
       offset += sizeof(std::uint64_t)) {
    const auto value = read_u64(entry->value.data() + offset);
    if (value > std::numeric_limits<std::size_t>::max())
      return {ErrorCode::kModelFormat,
              "metadata list exceeds size_t: " + std::string{key}};
    output->push_back(static_cast<std::size_t>(value));
  }
  return Status::Ok();
}

float bf16_value(const std::uint8_t *const data) noexcept {
  std::uint32_t bits = static_cast<std::uint32_t>(data[0]) << 16U;
  bits |= static_cast<std::uint32_t>(data[1]) << 24U;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float fp8_value(const std::uint8_t bits) noexcept {
  const auto magnitude = static_cast<std::uint8_t>(bits & 0x7fU);
  const auto exponent = static_cast<unsigned>((magnitude >> 3U) & 0x0fU);
  const auto mantissa = static_cast<unsigned>(magnitude & 0x07U);
  if (magnitude == 0x7fU)
    return std::numeric_limits<float>::quiet_NaN();
  const float value =
      exponent == 0U ? std::ldexp(static_cast<float>(mantissa), -9)
                     : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                                  static_cast<int>(exponent) - 7);
  return (bits & 0x80U) == 0U ? value : -value;
}

struct TensorView final {
  const std::uint8_t *data{nullptr};
  TensorDType dtype{TensorDType::kF32};
  std::size_t elements{0};

  [[nodiscard]] float at(const std::size_t index) const noexcept {
    if (dtype == TensorDType::kBF16)
      return bf16_value(data + index * 2);
    if (dtype == TensorDType::kE4M3Software)
      return fp8_value(data[index]);
    float value = 0.0F;
    std::memcpy(&value, data + index * sizeof(float), sizeof(value));
    return value;
  }
};

Status tensor_view(const ModelFile &model, const std::string &name,
                   const std::vector<std::size_t> &shape,
                   TensorView *const output,
                   const TensorDType expected = TensorDType::kBF16,
                   const bool accept_f32 = false,
                   const bool accept_fp8 = false) {
  const auto *const tensor = model.find_tensor(name);
  if (tensor == nullptr)
    return {ErrorCode::kModelFormat, "required tensor is missing: " + name};
  if (tensor->rank != shape.size())
    return {ErrorCode::kModelFormat, "tensor rank mismatch: " + name};
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (tensor->dimensions[index] != shape[index])
      return {ErrorCode::kModelFormat, "tensor shape mismatch: " + name};
  }
  if (tensor->dtype != expected &&
      !(accept_f32 && tensor->dtype == TensorDType::kF32) &&
      !(accept_fp8 && tensor->dtype == TensorDType::kE4M3Software)) {
    return {ErrorCode::kModelFormat, "tensor dtype mismatch: " + name};
  }
  const auto *const data = model.tensor_data(*tensor);
  if (data == nullptr ||
      tensor->element_count > std::numeric_limits<std::size_t>::max()) {
    return {ErrorCode::kModelFormat, "tensor payload is unavailable: " + name};
  }
  *output = {data, tensor->dtype,
             static_cast<std::size_t>(tensor->element_count)};
  return Status::Ok();
}

#if (defined(__x86_64__) || defined(_M_X64)) &&                                \
    (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma"))) float
dot_bf16_avx2(const float *const input, const std::uint8_t *const weight,
              const std::size_t size) noexcept {
  __m256 total = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 8 <= size; index += 8) {
    const auto packed =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight + index * 2));
    const auto wide = _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16);
    const auto values = _mm256_castsi256_ps(wide);
    total = _mm256_fmadd_ps(_mm256_loadu_ps(input + index), values, total);
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, total);
  float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
                 lanes[5] + lanes[6] + lanes[7];
  for (; index < size; ++index)
    result += input[index] * bf16_value(weight + index * 2);
  return result;
}

bool has_avx2_fma() noexcept {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

float dot_bf16(const float *const input, const std::uint8_t *const weight,
               const std::size_t size) noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
  {
    float32x4_t total = vdupq_n_f32(0.0F);
    std::size_t index = 0;
    for (; index + 4 <= size; index += 4) {
      std::uint16_t packed[4];
      std::memcpy(packed, weight + index * 2, sizeof(packed));
      const auto wide = vshll_n_u16(vld1_u16(packed), 16);
      total = vfmaq_f32(total, vld1q_f32(input + index),
                        vreinterpretq_f32_u32(wide));
    }
    float result = vaddvq_f32(total);
    for (; index < size; ++index)
      result += input[index] * bf16_value(weight + index * 2);
    return result;
  }
#elif (defined(__x86_64__) || defined(_M_X64)) &&                              \
    (defined(__GNUC__) || defined(__clang__))
  if (has_avx2_fma())
    return dot_bf16_avx2(input, weight, size);
#endif
  float result = 0.0F;
  for (std::size_t index = 0; index < size; ++index)
    result += input[index] * bf16_value(weight + index * 2);
  return result;
}

const char *selected_kernel_name() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
  return "neon-f32";
#elif (defined(__x86_64__) || defined(_M_X64)) &&                              \
    (defined(__GNUC__) || defined(__clang__))
  return has_avx2_fma() ? "avx2-fma-f32" : "scalar-f32";
#else
  return "scalar-f32";
#endif
}

float dot(const float *const input, const TensorView &weight,
          const std::size_t offset, const std::size_t size) noexcept {
  if (weight.dtype == TensorDType::kBF16)
    return dot_bf16(input, weight.data + offset * 2, size);
  float total = 0.0F;
  for (std::size_t index = 0; index < size; ++index)
    total += input[index] * weight.at(offset + index);
  return total;
}

Status linear(const std::vector<float> &input, const std::size_t rows,
              const std::size_t input_width, const TensorView &weight,
              const std::size_t output_width, const TensorView *const bias,
              std::vector<float> *const output) {
  if (input.size() != rows * input_width ||
      weight.elements != output_width * input_width || output == nullptr)
    return {ErrorCode::kInternal, "CPU linear dimensions are inconsistent"};
  output->resize(rows * output_width);
  for (std::size_t row = 0; row < rows; ++row) {
    const float *const source = input.data() + row * input_width;
    for (std::size_t target = 0; target < output_width; ++target) {
      float value = dot(source, weight, target * input_width, input_width);
      if (bias != nullptr)
        value += bias->at(target);
      (*output)[row * output_width + target] = value;
    }
  }
  return Status::Ok();
}

Status rms_norm(const std::vector<float> &input, const std::size_t rows,
                const std::size_t width, const TensorView &scale,
                const float epsilon, std::vector<float> *const output) {
  output->resize(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      const float value = input[row * width + column];
      sum += value * value;
    }
    const float denominator =
        std::sqrt(sum / static_cast<float>(width)) + epsilon;
    for (std::size_t column = 0; column < width; ++column) {
      (*output)[row * width + column] =
          input[row * width + column] / denominator * scale.at(column);
    }
  }
  return Status::Ok();
}

void add_inplace(std::vector<float> *const left,
                 const std::vector<float> &right) {
  for (std::size_t index = 0; index < left->size(); ++index)
    (*left)[index] += right[index];
}

void update_fir_state(std::vector<float> *const state,
                      const std::size_t history, const std::size_t channel,
                      const float value) {
  if (history == 0)
    return;
  const auto begin =
      state->begin() + static_cast<std::ptrdiff_t>(channel * history);
  std::move(begin + 1, begin + static_cast<std::ptrdiff_t>(history), begin);
  (*state)[channel * history + history - 1] = value;
}

} // namespace

struct Model::Impl final {
  struct Layer final {
    MixerType type{MixerType::kHcs};
    TensorView pre_norm;
    TensorView post_norm;
    TensorView projection;
    TensorView output_weight;
    TensorView output_bias;
    TensorView short_filter;
    TensorView inner_filter;
    TensorView direct;
    TensorView log_poles;
    TensorView residues;
    TensorView inverse_frequency;
    TensorView l1;
    TensorView l2;
    TensorView l3;
  };

  ModelConfig public_config;
  const ModelFile *file{nullptr};
  std::size_t heads{0};
  std::size_t state_size{0};
  std::size_t inner_width{0};
  std::size_t short_filter_length{0};
  std::size_t hcs_filter_length{0};
  std::size_t hcm_filter_length{0};
  std::size_t hcs_filter_groups{0};
  std::size_t hcm_filter_groups{0};
  float epsilon{0.0F};
  float rope_scale{0.0F};
  bool qkv_source_head_major{false};
  TensorView embedding;
  TensorView final_norm;
  std::vector<Layer> layer;

  [[nodiscard]] std::size_t head_dim() const noexcept {
    return heads == 0 ? 0 : public_config.width / heads;
  }
};

struct Context::Impl final {
  struct LayerState final {
    std::vector<float> short_state;
    std::vector<float> inner_state;
    std::vector<float> iir_state;
    std::vector<float> key;
    std::vector<float> value;
  };

  std::shared_ptr<Model::Impl> weights;
  std::size_t capacity{0};
  std::size_t arena_capacity{0};
  std::size_t layer_begin{0};
  std::size_t position{0};
  bool valid{false};
  std::vector<LayerState> state;

  void reset() {
    position = 0;
    valid = false;
    for (auto &item : state) {
      std::fill(item.short_state.begin(), item.short_state.end(), 0.0F);
      std::fill(item.inner_state.begin(), item.inner_state.end(), 0.0F);
      std::fill(item.iir_state.begin(), item.iir_state.end(), 0.0F);
      item.key.clear();
      item.value.clear();
    }
  }

  Status fir(const std::vector<float> &input, const std::size_t rows,
             const std::size_t channels, const TensorView &filter,
             const std::size_t groups, const std::size_t kernel,
             const bool causal, const TensorView *const direct,
             std::vector<float> *const cache,
             std::vector<float> *const output) {
    const std::size_t history = kernel - 1;
    output->resize(input.size());
    const std::size_t channels_per_group = channels / groups;
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t channel = 0; channel < channels; ++channel) {
        const std::size_t group = channel / channels_per_group;
        float total = 0.0F;
        for (std::size_t old = 0; old < history; ++old) {
          const std::size_t tap = causal ? kernel - 1 - old : old;
          total += (*cache)[channel * history + old] *
                   filter.at(group * kernel + tap);
        }
        const float current = input[row * channels + channel];
        total += current * filter.at(group * kernel + (causal ? 0 : history));
        if (direct != nullptr)
          total += direct->at(channel) * current;
        (*output)[row * channels + channel] = total;
        update_fir_state(cache, history, channel, current);
      }
    }
    return Status::Ok();
  }

  Status attention(const std::vector<float> &input, const std::size_t rows,
                   const Model::Impl::Layer &layer, LayerState *const cache,
                   std::vector<float> *const output) {
    const auto width = weights->public_config.width;
    const auto head_dim = weights->head_dim();
    std::vector<float> projected;
    auto status = linear(input, rows, width, layer.projection, width * 3,
                         nullptr, &projected);
    if (!status.ok())
      return status;
    std::vector<float> query(rows * width);
    std::vector<float> key(rows * width);
    std::vector<float> value(rows * width);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t component = 0; component < 3; ++component) {
        for (std::size_t head = 0; head < weights->heads; ++head) {
          for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
            const std::size_t destination =
                row * width + head * head_dim + dimension;
            const std::size_t source_output =
                weights->qkv_source_head_major
                    ? head * 3 * head_dim + component * head_dim + dimension
                    : component * width + head * head_dim + dimension;
            const float item = projected[row * width * 3 + source_output];
            (component == 0   ? query
             : component == 1 ? key
                              : value)[destination] = item;
          }
        }
      }
    }
    const std::size_t prefix = cache->key.size() / width;
    const std::size_t half = head_dim / 2;
    for (std::size_t row = 0; row < rows; ++row) {
      const float position_value =
          static_cast<float>(prefix + row) / weights->rope_scale;
      for (std::size_t head = 0; head < weights->heads; ++head) {
        const std::size_t base = row * width + head * head_dim;
        for (std::size_t pair = 0; pair < half; ++pair) {
          const float angle = position_value * layer.inverse_frequency.at(pair);
          const float cosine = std::cos(angle);
          const float sine = std::sin(angle);
          const auto rotate = [&](std::vector<float> *const tensor) {
            const float first = (*tensor)[base + pair];
            const float second = (*tensor)[base + half + pair];
            (*tensor)[base + pair] = first * cosine - second * sine;
            (*tensor)[base + half + pair] = second * cosine + first * sine;
          };
          rotate(&query);
          rotate(&key);
        }
      }
    }
    cache->key.insert(cache->key.end(), key.begin(), key.end());
    cache->value.insert(cache->value.end(), value.begin(), value.end());
    output->assign(rows * width, 0.0F);
    const float score_scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores;
    for (std::size_t row = 0; row < rows; ++row) {
      const std::size_t sources = prefix + row + 1;
      scores.resize(sources);
      for (std::size_t head = 0; head < weights->heads; ++head) {
        const std::size_t query_base = row * width + head * head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t source = 0; source < sources; ++source) {
          float score = 0.0F;
          const std::size_t key_base = source * width + head * head_dim;
          for (std::size_t dimension = 0; dimension < head_dim; ++dimension)
            score += query[query_base + dimension] *
                     cache->key[key_base + dimension];
          scores[source] = score * score_scale;
          maximum = std::max(maximum, scores[source]);
        }
        float denominator = 0.0F;
        for (auto &score : scores) {
          score = std::exp(score - maximum);
          denominator += score;
        }
        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
          float total = 0.0F;
          for (std::size_t source = 0; source < sources; ++source) {
            total += scores[source] / denominator *
                     cache->value[source * width + head * head_dim + dimension];
          }
          (*output)[query_base + dimension] = total;
        }
      }
    }
    return Status::Ok();
  }

  Status mixer(const std::vector<float> &input, const std::size_t rows,
               const Model::Impl::Layer &layer, LayerState *const cache,
               std::vector<float> *const output) {
    const auto width = weights->public_config.width;
    if (layer.type == MixerType::kAttention)
      return attention(input, rows, layer, cache, output);
    std::vector<float> projection;
    auto status = linear(input, rows, width, layer.projection, width * 3,
                         nullptr, &projection);
    if (!status.ok())
      return status;
    std::vector<float> filtered;
    status = fir(projection, rows, width * 3, layer.short_filter, width * 3,
                 weights->short_filter_length, false, nullptr,
                 &cache->short_state, &filtered);
    if (!status.ok())
      return status;
    std::vector<float> x2(rows * width);
    std::vector<float> gated(rows * width);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t channel = 0; channel < width; ++channel) {
        const auto source = row * width * 3 + channel * 3;
        x2[row * width + channel] = filtered[source];
        gated[row * width + channel] =
            filtered[source + 1] * filtered[source + 2];
      }
    }
    std::vector<float> mixed;
    if (layer.type == MixerType::kHcs) {
      status = fir(gated, rows, width, layer.inner_filter,
                   weights->hcs_filter_groups, weights->hcs_filter_length,
                   false, nullptr, &cache->inner_state, &mixed);
    } else if (layer.type == MixerType::kHcm) {
      status = fir(gated, rows, width, layer.inner_filter,
                   weights->hcm_filter_groups, weights->hcm_filter_length, true,
                   &layer.direct, &cache->inner_state, &mixed);
    } else {
      mixed.resize(rows * width);
      for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t channel = 0; channel < width; ++channel) {
          float modal = 0.0F;
          for (std::size_t mode = 0; mode < weights->state_size; ++mode) {
            const std::size_t index = channel * weights->state_size + mode;
            cache->iir_state[index] =
                std::exp(layer.log_poles.at(index)) * cache->iir_state[index] +
                gated[row * width + channel];
            modal += layer.residues.at(index) * cache->iir_state[index];
          }
          mixed[row * width + channel] =
              modal + layer.direct.at(channel) * gated[row * width + channel];
        }
      }
    }
    if (!status.ok())
      return status;
    output->resize(rows * width);
    for (std::size_t index = 0; index < output->size(); ++index)
      (*output)[index] = x2[index] * mixed[index];
    return Status::Ok();
  }

  Status forward_hidden(std::vector<float> hidden, const std::size_t rows,
                        const bool initial, std::vector<float> *const logits,
                        const std::size_t capture_layer,
                        std::vector<float> *const capture) {
    if (!weights || rows == 0 || rows > arena_capacity ||
        hidden.size() != rows * weights->public_config.width ||
        (logits == nullptr && capture == nullptr)) {
      return {ErrorCode::kInvalidArgument,
              "CPU hidden forward arguments are invalid"};
    }
    if (initial) {
      reset();
    } else if (!valid || rows > capacity - position) {
      return {ErrorCode::kInvalidArgument,
              "CPU continuation requires a valid prefill and free capacity"};
    }
    const auto width = weights->public_config.width;
    std::vector<float> normalized;
    std::vector<float> mixed;
    std::vector<float> residual;
    std::vector<float> first;
    std::vector<float> second;
    std::vector<float> gated;
    for (std::size_t index = layer_begin; index < weights->layer.size();
         ++index) {
      const auto &layer = weights->layer[index];
      auto status = rms_norm(hidden, rows, width, layer.pre_norm,
                             weights->epsilon, &normalized);
      if (!status.ok())
        return status;
      status = mixer(normalized, rows, layer, &state[index], &mixed);
      if (!status.ok())
        return {status.code(), "CPU mixer block " + std::to_string(index) +
                                   ": " + status.message()};
      status = linear(mixed, rows, width, layer.output_weight, width,
                      &layer.output_bias, &residual);
      if (!status.ok())
        return status;
      add_inplace(&residual, hidden);
      status = rms_norm(residual, rows, width, layer.post_norm,
                        weights->epsilon, &normalized);
      if (!status.ok())
        return status;
      status = linear(normalized, rows, width, layer.l1, weights->inner_width,
                      nullptr, &first);
      if (!status.ok())
        return status;
      status = linear(normalized, rows, width, layer.l2, weights->inner_width,
                      nullptr, &second);
      if (!status.ok())
        return status;
      gated.resize(first.size());
      for (std::size_t item = 0; item < gated.size(); ++item) {
        const float activated =
            index == 0
                ? 0.5F * first[item] *
                      (1.0F + std::erf(first[item] * 0.7071067811865475244F))
                : first[item];
        gated[item] = activated * second[item];
      }
      status = linear(gated, rows, weights->inner_width, layer.l3, width,
                      nullptr, &hidden);
      if (!status.ok())
        return status;
      add_inplace(&hidden, residual);
      if (capture != nullptr && capture_layer == index)
        *capture = hidden;
    }
    if (capture != nullptr && capture_layer >= weights->layer.size())
      return {ErrorCode::kInvalidArgument, "CPU embedding layer is invalid"};
    if (logits != nullptr) {
      auto status = rms_norm(hidden, rows, width, weights->final_norm,
                             weights->epsilon, &normalized);
      if (!status.ok())
        return status;
      status = linear(normalized, rows, width, weights->embedding,
                      weights->public_config.vocab_size, nullptr, logits);
      if (!status.ok())
        return status;
    }
    position = initial ? rows : position + rows;
    valid = true;
    return Status::Ok();
  }

  Status forward(const std::vector<TokenId> &tokens, const bool initial,
                 std::vector<float> *const logits,
                 const std::size_t capture_layer,
                 std::vector<float> *const capture) {
    if (!weights || tokens.empty() || tokens.size() > arena_capacity)
      return {ErrorCode::kInvalidArgument, "CPU forward arguments are invalid"};
    const auto rows = tokens.size();
    const auto width = weights->public_config.width;
    std::vector<float> hidden(rows * width);
    for (std::size_t row = 0; row < rows; ++row) {
      if (tokens[row] >= weights->public_config.vocab_size)
        return {ErrorCode::kInvalidArgument,
                "token exceeds CPU model vocabulary"};
      for (std::size_t column = 0; column < width; ++column) {
        hidden[row * width + column] = weights->embedding.at(
            static_cast<std::size_t>(tokens[row]) * width + column);
      }
    }
    return forward_hidden(std::move(hidden), rows, initial, logits,
                          capture_layer, capture);
  }
};

Model::Model() : impl_(std::make_shared<Impl>()) {}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model &Model::operator=(Model &&) noexcept = default;

Status Model::load(const ModelFile &model, const bool allow_test_fixture) {
  if ((!impl_ && !hyena_) || (impl_ && impl_->file != nullptr) || hyena_)
    return {ErrorCode::kInvalidArgument, "CPU model is already loaded"};
  std::string runtime_abi;
  std::string architecture;
  auto status = metadata_string(model, "runtime.abi", &runtime_abi);
  if (!status.ok())
    return status;
  status = metadata_string(model, "model.architecture", &architecture);
  if (!status.ok())
    return status;
  const auto *const registered = find_architecture(architecture);
  if (registered == nullptr || registered->artifact_profile != model.profile() ||
      registered->runtime_abi != runtime_abi ||
      (registered->backends & kArchitectureBackendCpu) == 0) {
    return {ErrorCode::kUnsupported,
            "CPU backend does not support this runtime architecture"};
  }
  if (registered->tokenizer == ArchitectureTokenizer::kHyenaDnaCharacter) {
    auto candidate = std::make_shared<detail::HyenaDnaModel>();
    status = candidate->load(model, allow_test_fixture);
    if (!status.ok())
      return status;
    impl_.reset();
    hyena_ = std::move(candidate);
    return Status::Ok();
  }
  auto candidate = std::make_shared<Impl>();
  candidate->public_config.architecture = architecture;
  candidate->public_config.tokenizer = registered->tokenizer;
  if (architecture == "StripedHyena2Test") {
    status = metadata_bool(model, "fixture.synthetic",
                           &candidate->public_config.test_fixture);
    if (!status.ok())
      return status;
    if (!candidate->public_config.test_fixture || !allow_test_fixture)
      return {ErrorCode::kUnsupported,
              "synthetic model fixtures require explicit test permission"};
  }
  if (model.find_metadata("model.id") != nullptr) {
    status =
        metadata_string(model, "model.id", &candidate->public_config.model_id);
  } else {
    status = metadata_string(model, "model.name",
                             &candidate->public_config.model_id);
  }
  if (!status.ok())
    return status;
  for (const auto &item :
       {std::pair{"config.vocab_size", &candidate->public_config.vocab_size},
        std::pair{"config.hidden_size", &candidate->public_config.width},
        std::pair{"config.num_layers", &candidate->public_config.layers},
        std::pair{"config.num_attention_heads", &candidate->heads},
        std::pair{"config.state_size", &candidate->state_size},
        std::pair{"config.inner_mlp_size", &candidate->inner_width},
        std::pair{"config.short_filter_length",
                  &candidate->short_filter_length},
        std::pair{"config.hcs_filter_length", &candidate->hcs_filter_length},
        std::pair{"config.hcm_filter_length", &candidate->hcm_filter_length},
        std::pair{"config.hcs_filter_groups", &candidate->hcs_filter_groups},
        std::pair{"config.hcm_filter_groups", &candidate->hcm_filter_groups}}) {
    status = metadata_size(model, item.first, item.second);
    if (!status.ok())
      return status;
  }
  if (candidate->public_config.test_fixture) {
    candidate->public_config.max_seqlen =
        std::numeric_limits<std::size_t>::max();
  } else {
    status = metadata_size(model, "config.max_seqlen",
                           &candidate->public_config.max_seqlen);
    if (!status.ok())
      return status;
  }
  status = metadata_float(model, "config.eps", &candidate->epsilon);
  if (!status.ok())
    return status;
  status = metadata_float(model, "config.rotary_emb_scaling_factor",
                          &candidate->rope_scale);
  if (!status.ok())
    return status;
  if (model.find_metadata("config.column_split") != nullptr) {
    status = metadata_bool(model, "config.column_split",
                           &candidate->qkv_source_head_major);
    if (!status.ok())
      return status;
  }
  if (candidate->public_config.vocab_size == 0 ||
      candidate->public_config.width == 0 ||
      candidate->public_config.layers == 0 || candidate->heads == 0 ||
      candidate->public_config.width % candidate->heads != 0 ||
      candidate->head_dim() % 2 != 0 || candidate->state_size == 0 ||
      candidate->inner_width == 0 || candidate->short_filter_length < 2 ||
      candidate->hcs_filter_length < 2 || candidate->hcm_filter_length < 2 ||
      candidate->hcs_filter_groups == 0 || candidate->hcm_filter_groups == 0 ||
      candidate->public_config.width % candidate->hcs_filter_groups != 0 ||
      candidate->public_config.width % candidate->hcm_filter_groups != 0 ||
      !std::isfinite(candidate->epsilon) || candidate->epsilon <= 0.0F ||
      !std::isfinite(candidate->rope_scale) || candidate->rope_scale <= 0.0F) {
    return {ErrorCode::kUnsupported, "CPU model dimensions are unsupported"};
  }
  std::vector<std::size_t> hcs;
  std::vector<std::size_t> hcm;
  std::vector<std::size_t> hcl;
  std::vector<std::size_t> attention;
  for (auto item : {std::pair{"config.hcs_layer_idxs", &hcs},
                    std::pair{"config.hcm_layer_idxs", &hcm},
                    std::pair{"config.hcl_layer_idxs", &hcl},
                    std::pair{"config.attn_layer_idxs", &attention}}) {
    status = metadata_list(model, item.first, item.second);
    if (!status.ok())
      return status;
  }
  std::vector<MixerType> types(candidate->public_config.layers,
                               MixerType::kHcs);
  std::set<std::size_t> classified;
  for (const auto index : hcs) {
    if (index >= types.size() || !classified.insert(index).second)
      return {ErrorCode::kModelFormat, "invalid HCS layer list"};
  }
  for (const auto index : hcm) {
    if (index >= types.size() || !classified.insert(index).second)
      return {ErrorCode::kModelFormat, "invalid HCM layer list"};
    types[index] = MixerType::kHcm;
  }
  for (const auto index : hcl) {
    if (index >= types.size() || !classified.insert(index).second)
      return {ErrorCode::kModelFormat, "invalid HCL layer list"};
    types[index] = MixerType::kHcl;
  }
  for (const auto index : attention) {
    if (index >= types.size() || !classified.insert(index).second)
      return {ErrorCode::kModelFormat, "invalid attention layer list"};
    types[index] = MixerType::kAttention;
  }
  if (classified.size() != types.size())
    return {ErrorCode::kModelFormat, "mixer lists do not cover all CPU blocks"};

  const auto width = candidate->public_config.width;
  status = tensor_view(model, "embedding_layer.weight",
                       {candidate->public_config.vocab_size, width},
                       &candidate->embedding);
  if (!status.ok())
    return status;
  status = tensor_view(model, "norm.scale", {width}, &candidate->final_norm,
                       TensorDType::kF32);
  if (!status.ok())
    return status;
  candidate->layer.resize(types.size());
  for (std::size_t index = 0; index < types.size(); ++index) {
    auto &layer = candidate->layer[index];
    layer.type = types[index];
    const std::string prefix = "blocks." + std::to_string(index);
    for (auto item :
         {std::pair{prefix + ".pre_norm.scale", &layer.pre_norm},
          std::pair{prefix + ".post_norm.scale", &layer.post_norm}}) {
      status = tensor_view(model, item.first, {width}, item.second,
                           TensorDType::kF32);
      if (!status.ok())
        return status;
    }
    status = tensor_view(model, prefix + ".mlp.l1.weight",
                         {candidate->inner_width, width}, &layer.l1);
    if (!status.ok())
      return status;
    status = tensor_view(model, prefix + ".mlp.l2.weight",
                         {candidate->inner_width, width}, &layer.l2);
    if (!status.ok())
      return status;
    status = tensor_view(model, prefix + ".mlp.l3.weight",
                         {width, candidate->inner_width}, &layer.l3);
    if (!status.ok())
      return status;
    if (layer.type == MixerType::kAttention) {
      status = tensor_view(model, prefix + ".inner_mha_cls.Wqkv.weight",
                           {width * 3, width}, &layer.projection);
      if (!status.ok())
        return status;
      status = tensor_view(model, prefix + ".inner_mha_cls.out_proj.weight",
                           {width, width}, &layer.output_weight);
      if (!status.ok())
        return status;
      status = tensor_view(model, prefix + ".inner_mha_cls.out_proj.bias",
                           {width}, &layer.output_bias);
      if (!status.ok())
        return status;
      status = tensor_view(model, prefix + ".inner_mha_cls.rotary_emb.inv_freq",
                           {candidate->head_dim() / 2},
                           &layer.inverse_frequency, TensorDType::kF32);
      if (!status.ok())
        return status;
      continue;
    }
    status =
        tensor_view(model, prefix + ".projections.weight", {width * 3, width},
                    &layer.projection, TensorDType::kBF16, false, true);
    if (!status.ok())
      return status;
    status = tensor_view(model, prefix + ".out_filter_dense.weight",
                         {width, width}, &layer.output_weight);
    if (!status.ok())
      return status;
    status = tensor_view(model, prefix + ".out_filter_dense.bias", {width},
                         &layer.output_bias);
    if (!status.ok())
      return status;
    status = tensor_view(model, prefix + ".filter.short_filter_weight",
                         {width * 3, 1, candidate->short_filter_length},
                         &layer.short_filter);
    if (!status.ok())
      return status;
    if (layer.type == MixerType::kHcs) {
      status = tensor_view(
          model, prefix + ".filter.h",
          {candidate->hcs_filter_groups, 1, candidate->hcs_filter_length},
          &layer.inner_filter);
    } else {
      status = tensor_view(model, prefix + ".filter.D", {width}, &layer.direct);
      if (status.ok() && layer.type == MixerType::kHcm) {
        status = tensor_view(
            model, prefix + ".filter.h",
            {candidate->hcm_filter_groups, 1, candidate->hcm_filter_length},
            &layer.inner_filter, TensorDType::kBF16, true);
      } else if (status.ok()) {
        status = tensor_view(model, prefix + ".filter.log_poles",
                             {width, candidate->state_size, 1},
                             &layer.log_poles, TensorDType::kF32);
        if (status.ok())
          status = tensor_view(model, prefix + ".filter.residues",
                               {width, candidate->state_size}, &layer.residues,
                               TensorDType::kF32);
      }
    }
    if (!status.ok())
      return status;
  }
  candidate->file = &model;
  impl_ = std::move(candidate);
  return Status::Ok();
}

const ModelConfig &Model::config() const noexcept {
  return hyena_ ? hyena_->config() : impl_->public_config;
}
const char *Model::kernel_name() const noexcept {
  return hyena_ ? hyena_->kernel_name() : selected_kernel_name();
}

Status Model::encode(const std::string_view sequence,
                     std::vector<TokenId> *const tokens) const {
  return encode_sequence(config().tokenizer, sequence, tokens);
}

Status Model::decode_token(const TokenId token,
                           std::uint8_t *const byte) const {
  return decode_sequence_token(config().tokenizer, token, byte);
}

Context::Context() : impl_(std::make_unique<Impl>()) {}
Context::~Context() = default;
Context::Context(Context &&) noexcept = default;
Context &Context::operator=(Context &&) noexcept = default;

Status Context::initialize_shared(const Model &model,
                                  const std::size_t context_capacity,
                                  const std::size_t layer_begin) {
  if (model.hyena_) {
    if (!impl_ || hyena_ || layer_begin != 0)
      return {ErrorCode::kInvalidArgument,
              "HyenaDNA contexts do not support suffix-layer placement"};
    auto candidate = std::make_unique<detail::HyenaDnaContext>();
    const auto status =
        candidate->initialize_shared(*model.hyena_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    hyena_ = std::move(candidate);
    return Status::Ok();
  }
  if (!impl_ || impl_->weights || !model.impl_ ||
      model.impl_->file == nullptr || context_capacity == 0 ||
      context_capacity > model.impl_->public_config.max_seqlen ||
      layer_begin > model.impl_->public_config.layers) {
    return {ErrorCode::kInvalidArgument,
            "CPU context requires a loaded model and valid capacity"};
  }
  impl_->weights = model.impl_;
  impl_->capacity = context_capacity;
  impl_->arena_capacity = std::min(context_capacity, kMaximumArenaTokens);
  impl_->layer_begin = layer_begin;
  impl_->state.resize(model.impl_->layer.size());
  const auto width = model.impl_->public_config.width;
  for (std::size_t index = 0; index < impl_->state.size(); ++index) {
    const auto type = model.impl_->layer[index].type;
    if (type == MixerType::kAttention)
      continue;
    impl_->state[index].short_state.assign(
        width * 3 * (model.impl_->short_filter_length - 1), 0.0F);
    if (type == MixerType::kHcs) {
      impl_->state[index].inner_state.assign(
          width * (model.impl_->hcs_filter_length - 1), 0.0F);
    } else if (type == MixerType::kHcm) {
      impl_->state[index].inner_state.assign(
          width * (model.impl_->hcm_filter_length - 1), 0.0F);
    } else {
      impl_->state[index].iir_state.assign(width * model.impl_->state_size,
                                           0.0F);
    }
  }
  return Status::Ok();
}

Status Context::prefill(const std::vector<TokenId> &tokens,
                        std::vector<float> *const logits) {
  if (hyena_)
    return hyena_->prefill(tokens, logits);
  return impl_->forward(tokens, true, logits,
                        std::numeric_limits<std::size_t>::max(), nullptr);
}

Status Context::prefill_chunk(const std::vector<TokenId> &tokens,
                              std::vector<float> *const logits) {
  if (hyena_)
    return hyena_->prefill_chunk(tokens, logits);
  return impl_->forward(tokens, false, logits,
                        std::numeric_limits<std::size_t>::max(), nullptr);
}

Status Context::decode(const TokenId token, std::vector<float> *const logits) {
  if (hyena_)
    return hyena_->decode(token, logits);
  return impl_->forward({token}, false, logits,
                        std::numeric_limits<std::size_t>::max(), nullptr);
}

Status Context::prefill_embedding(const std::vector<TokenId> &tokens,
                                  const std::size_t layer,
                                  std::vector<float> *const embedding) {
  if (hyena_)
    return hyena_->prefill_embedding(tokens, layer, embedding);
  return impl_->forward(tokens, true, nullptr, layer, embedding);
}

Status Context::prefill_chunk_embedding(const std::vector<TokenId> &tokens,
                                        const std::size_t layer,
                                        std::vector<float> *const embedding) {
  if (hyena_)
    return hyena_->prefill_chunk_embedding(tokens, layer, embedding);
  return impl_->forward(tokens, false, nullptr, layer, embedding);
}

Status Context::prefill_from_hidden(const std::vector<float> &hidden,
                                    const std::size_t rows,
                                    std::vector<float> *const logits) {
  if (hyena_)
    return {ErrorCode::kUnsupported,
            "HyenaDNA does not support CUDA-prefix hidden input"};
  return impl_->forward_hidden(hidden, rows, true, logits,
                               std::numeric_limits<std::size_t>::max(),
                               nullptr);
}

Status Context::prefill_chunk_from_hidden(const std::vector<float> &hidden,
                                          const std::size_t rows,
                                          std::vector<float> *const logits) {
  if (hyena_)
    return {ErrorCode::kUnsupported,
            "HyenaDNA does not support CUDA-prefix hidden input"};
  return impl_->forward_hidden(hidden, rows, false, logits,
                               std::numeric_limits<std::size_t>::max(),
                               nullptr);
}

std::size_t Context::position() const noexcept {
  return hyena_ ? hyena_->position() : impl_->position;
}
std::size_t Context::activation_capacity() const noexcept {
  return hyena_ ? hyena_->activation_capacity() : impl_->arena_capacity;
}
const ModelConfig &Context::config() const noexcept {
  return hyena_ ? hyena_->config() : impl_->weights->public_config;
}
const char *Context::kernel_name() const noexcept {
  return hyena_ ? "direct-convolution-f32" : selected_kernel_name();
}

} // namespace evo::cpu
