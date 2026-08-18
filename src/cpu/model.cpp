// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/model.hpp"
#include "evo/cpu/geneb_bert.hpp"
#include "evo/cpu/geneb_custom_encoder.hpp"
#include "evo/cpu/geneb_decoder.hpp"
#include "evo/cpu/geneb_dna_gpt.hpp"
#include "evo/cpu/geneb_esm.hpp"
#include "evo/cpu/geneb_evo1.hpp"
#include "evo/cpu/geneb_gpt2.hpp"
#include "evo/cpu/geneb_hyenadna.hpp"
#include "evo/cpu/geneb_janusdna.hpp"
#include "evo/cpu/geneb_mamba.hpp"
#include "evo/cpu/geneb_olmo.hpp"
#include "evo/cpu/geneb_roformer.hpp"
#include "evo/cpu/geneb_sequence_cnn.hpp"

#include "../linear_executor.hpp"
#include "esmc_internal.hpp"
#include "evo/model_registry.hpp"
#include "hyenadna_internal.hpp"

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
              evo::detail::LinearExecutor *const executor,
              std::vector<float> *const output) {
  if (input.size() != rows * input_width ||
      weight.elements != output_width * input_width || output == nullptr)
    return {ErrorCode::kInternal, "host linear dimensions are inconsistent"};
  if (executor != nullptr) {
    const evo::detail::LinearTensorView weight_view{weight.data, weight.dtype,
                                                    weight.elements};
    const evo::detail::LinearTensorView bias_view =
        bias == nullptr ? evo::detail::LinearTensorView{}
                        : evo::detail::LinearTensorView{bias->data, bias->dtype,
                                                        bias->elements};
    return executor->linear(input.data(), rows, input_width, weight_view,
                            output_width,
                            bias == nullptr ? nullptr : &bias_view, output);
  }
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
  std::shared_ptr<evo::detail::LinearExecutor> executor;

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
                         nullptr, weights->executor.get(), &projected);
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
                         nullptr, weights->executor.get(), &projection);
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
              "host hidden forward arguments are invalid"};
    }
    if (initial) {
      reset();
    } else if (!valid || rows > capacity - position) {
      return {ErrorCode::kInvalidArgument,
              "host continuation requires a valid prefill and free capacity"};
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
        return {status.code(), "host mixer block " + std::to_string(index) +
                                   ": " + status.message()};
      status = linear(mixed, rows, width, layer.output_weight, width,
                      &layer.output_bias, weights->executor.get(), &residual);
      if (!status.ok())
        return status;
      add_inplace(&residual, hidden);
      status = rms_norm(residual, rows, width, layer.post_norm,
                        weights->epsilon, &normalized);
      if (!status.ok())
        return status;
      status = linear(normalized, rows, width, layer.l1, weights->inner_width,
                      nullptr, weights->executor.get(), &first);
      if (!status.ok())
        return status;
      status = linear(normalized, rows, width, layer.l2, weights->inner_width,
                      nullptr, weights->executor.get(), &second);
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
                      nullptr, weights->executor.get(), &hidden);
      if (!status.ok())
        return status;
      add_inplace(&hidden, residual);
      if (capture != nullptr && capture_layer == index)
        *capture = hidden;
    }
    if (capture != nullptr && capture_layer >= weights->layer.size())
      return {ErrorCode::kInvalidArgument, "host embedding layer is invalid"};
    if (logits != nullptr) {
      auto status = rms_norm(hidden, rows, width, weights->final_norm,
                             weights->epsilon, &normalized);
      if (!status.ok())
        return status;
      status = linear(normalized, rows, width, weights->embedding,
                      weights->public_config.vocab_size, nullptr,
                      weights->executor.get(), logits);
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
      return {ErrorCode::kInvalidArgument,
              "host forward arguments are invalid"};
    const auto rows = tokens.size();
    const auto width = weights->public_config.width;
    std::vector<float> hidden(rows * width);
    for (std::size_t row = 0; row < rows; ++row) {
      if (!token_id_in_vocabulary(tokens[row],
                                  weights->public_config.vocab_size))
        return {ErrorCode::kInvalidArgument, "token exceeds model vocabulary"};
      for (std::size_t column = 0; column < width; ++column) {
        hidden[row * width + column] = weights->embedding.at(
            static_cast<std::size_t>(tokens[row]) * width + column);
      }
    }
    return forward_hidden(std::move(hidden), rows, initial, logits,
                          capture_layer, capture);
  }
};

namespace detail {

class GenebDecoderModelAdapter final {
public:
  Status load(const ModelFile &artifact,
              std::shared_ptr<evo::detail::LinearExecutor> executor,
              const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load_artifact(artifact, std::move(executor));
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr)
      return {ErrorCode::kInternal, "GENEB decoder topology was not retained"};
    public_config.architecture = std::string{kGenebDecoderArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebTransformerDecoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    if (artifact.find_metadata("model.id") != nullptr) {
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    } else {
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    }
    return status;
  }

  GenebDecoderModel runtime;
  ModelConfig public_config;
};

class GenebDecoderContext final {
public:
  Status
  initialize_shared(const std::shared_ptr<GenebDecoderModelAdapter> &model,
                    const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB decoder context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB decoder embedding requires a fresh full-sequence "
              "context and valid layer"};
    }
    bool padding_started = false;
    for (const auto value : attention_mask) {
      if (value > 1U || (padding_started && value != 0U))
        return {ErrorCode::kInvalidArgument,
                "GENEB decoder attention mask must be right padded"};
      padding_started = padding_started || value == 0U;
    }
    GenebDecoderForwardResult result;
    auto status = model_->runtime.forward(tokens, 0, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB decoder returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.linear_executor_name();
  }

private:
  std::shared_ptr<GenebDecoderModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebOlmoModelAdapter final {
public:
  Status load(const ModelFile &artifact,
              std::shared_ptr<evo::detail::LinearExecutor> executor,
              const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load(artifact, std::move(executor));
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr)
      return {ErrorCode::kInternal, "GENEB OLMo topology was not retained"};
    public_config.architecture = std::string{kGenebOlmoArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebOlmoDecoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebOlmoModel runtime;
  ModelConfig public_config;
};

class GenebOlmoContext final {
public:
  Status initialize_shared(const std::shared_ptr<GenebOlmoModelAdapter> &model,
                           const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB OLMo context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB OLMo embedding requires a fresh full-sequence context "
              "and valid layer"};
    }
    GenebOlmoForwardResult result;
    auto status =
        model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB OLMo returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.linear_executor_name();
  }

private:
  std::shared_ptr<GenebOlmoModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebEsmModelAdapter final {
public:
  Status load(const ModelFile &artifact,
              std::shared_ptr<evo::detail::LinearExecutor> executor,
              const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load_artifact(artifact, std::move(executor));
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr)
      return {ErrorCode::kInternal, "GENEB ESM topology was not retained"};
    public_config.architecture = std::string{kGenebEsmArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation = ArchitectureImplementation::kGenebEsmEncoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebEsmModel runtime;
  ModelConfig public_config;
};

class GenebEsmContext final {
public:
  Status initialize_shared(const std::shared_ptr<GenebEsmModelAdapter> &model,
                           const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB ESM context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    std::vector<std::uint8_t> attention_mask(tokens.size(), 1U);
    const auto *const topology = model_ ? model_->runtime.topology() : nullptr;
    if (topology != nullptr) {
      bool padding_started = false;
      for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index] == topology->pad_token_id)
          padding_started = true;
        if (padding_started)
          attention_mask[index] = 0U;
      }
    }
    return prefill_embedding_masked(tokens, attention_mask, layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB ESM embedding requires a fresh full-sequence context "
              "and valid layer"};
    }
    GenebEsmForwardResult result;
    auto status =
        model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB ESM returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.linear_executor_name();
  }

private:
  std::shared_ptr<GenebEsmModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebBertModelAdapter final {
public:
  Status load(const ModelFile &artifact,
              std::shared_ptr<evo::detail::LinearExecutor> executor,
              const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load_artifact(artifact, std::move(executor));
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr)
      return {ErrorCode::kInternal, "GENEB BERT topology was not retained"};
    public_config.architecture = std::string{kGenebBertArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebBertEncoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebBertModel runtime;
  ModelConfig public_config;
};

class GenebBertContext final {
public:
  Status initialize_shared(const std::shared_ptr<GenebBertModelAdapter> &model,
                           const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT embedding requires a fresh full-sequence context "
              "and valid layer"};
    }
    const auto *const topology = model_->runtime.topology();
    if (topology == nullptr)
      return {ErrorCode::kInternal, "GENEB BERT topology is unavailable"};
    GenebBertForwardResult result;
    Status status = Status::Ok();
    if (topology->input_kind == GenebBertInputKind::kSoftVocabulary) {
      if (topology->vocabulary_size != model_->public_config.vocab_size ||
          tokens.size() > std::numeric_limits<std::size_t>::max() /
                              topology->vocabulary_size) {
        return {ErrorCode::kInvalidArgument,
                "GENEB BERT soft-vocabulary dimensions overflow"};
      }
      std::vector<float> one_hot(tokens.size() * topology->vocabulary_size,
                                 0.0F);
      for (std::size_t row = 0; row < tokens.size(); ++row) {
        if (!token_id_in_vocabulary(tokens[row], topology->vocabulary_size)) {
          return {ErrorCode::kInvalidArgument,
                  "GENEB BERT token exceeds soft vocabulary"};
        }
        if (attention_mask[row] != 0U)
          one_hot[row * topology->vocabulary_size + tokens[row]] = 1.0F;
      }
      status = model_->runtime.forward_soft(one_hot, tokens.size(),
                                            attention_mask, {layer}, &result);
    } else {
      status =
          model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    }
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB BERT returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.linear_executor_name();
  }

private:
  std::shared_ptr<GenebBertModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebGpt2ModelAdapter final {
public:
  Status load(const ModelFile &artifact, const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load(artifact);
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr)
      return {ErrorCode::kInternal, "GENEB GPT-2 topology was not retained"};
    public_config.architecture = std::string{kGenebGpt2Architecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebGpt2Decoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebGpt2Model runtime;
  ModelConfig public_config;
};

class GenebGpt2Context final {
public:
  Status initialize_shared(const std::shared_ptr<GenebGpt2ModelAdapter> &model,
                           const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB GPT-2 context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB GPT-2 embedding requires a fresh full-sequence context "
              "and valid layer"};
    }
    GenebGpt2ForwardResult result;
    auto status =
        model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB GPT-2 returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.kernel_name();
  }

private:
  std::shared_ptr<GenebGpt2ModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebDnaGptModelAdapter final {
public:
  Status load(const ModelFile &artifact, const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load(artifact);
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr)
      return {ErrorCode::kInternal, "GENEB DNA-GPT topology was not retained"};
    public_config.architecture = std::string{kGenebDnaGptArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebDnaGptDecoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebDnaGptModel runtime;
  ModelConfig public_config;
};

class GenebDnaGptContext final {
public:
  Status
  initialize_shared(const std::shared_ptr<GenebDnaGptModelAdapter> &model,
                    const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB DNA-GPT context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB DNA-GPT embedding requires a fresh full-sequence context "
              "and valid layer"};
    }
    GenebDnaGptForwardResult result;
    auto status =
        model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB DNA-GPT returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.kernel_name();
  }

private:
  std::shared_ptr<GenebDnaGptModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebCustomEncoderModelAdapter final {
public:
  Status load(const ModelFile &artifact,
              std::shared_ptr<evo::detail::LinearExecutor> executor,
              const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load_artifact(artifact, std::move(executor));
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr) {
      return {ErrorCode::kInternal,
              "GENEB custom encoder topology was not retained"};
    }
    public_config.architecture = std::string{kGenebCustomEncoderArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebCustomEncoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    tokenizer_vocabulary_size = topology->tokenizer_vocabulary_size;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebCustomEncoderModel runtime;
  ModelConfig public_config;
  std::size_t tokenizer_vocabulary_size{0};
};

class GenebCustomEncoderContext final {
public:
  Status initialize_shared(
      const std::shared_ptr<GenebCustomEncoderModelAdapter> &model,
      const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB custom encoder context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    std::vector<std::uint8_t> attention_mask(tokens.size(), 1U);
    const auto *const topology = model_ ? model_->runtime.topology() : nullptr;
    if (topology != nullptr) {
      bool padding_started = false;
      for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index] == topology->pad_token_id)
          padding_started = true;
        if (padding_started)
          attention_mask[index] = 0U;
      }
    }
    return prefill_embedding_masked(tokens, attention_mask, layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB custom encoder embedding requires a fresh "
              "full-sequence context and valid layer"};
    }
    GenebCustomEncoderForwardResult result;
    auto status =
        model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB custom encoder returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.linear_executor_name();
  }

private:
  std::shared_ptr<GenebCustomEncoderModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebMambaModelAdapter final {
public:
  Status load(const ModelFile &artifact,
              std::shared_ptr<evo::detail::LinearExecutor> executor,
              const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load_artifact(artifact, std::move(executor));
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr) {
      return {ErrorCode::kInternal, "GENEB Mamba topology was not retained"};
    }
    public_config.architecture = std::string{kGenebMambaArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebMambaEncoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->output_width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    tokenizer_vocabulary_size = topology->tokenizer_vocabulary_size;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebMambaModel runtime;
  ModelConfig public_config;
  std::size_t tokenizer_vocabulary_size{0};
};

class GenebMambaContext final {
public:
  Status initialize_shared(const std::shared_ptr<GenebMambaModelAdapter> &model,
                           const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        (model->public_config.max_seqlen != 0 &&
         context_capacity > model->public_config.max_seqlen)) {
      return {ErrorCode::kInvalidArgument,
              "GENEB Mamba context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB Mamba embedding requires a fresh full-sequence context "
              "and valid layer"};
    }
    GenebMambaForwardResult result;
    auto status =
        model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB Mamba returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.linear_executor_name();
  }

private:
  std::shared_ptr<GenebMambaModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebHyenaDnaModelAdapter final {
public:
  Status load(const ModelFile &artifact, const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load(artifact);
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr) {
      return {ErrorCode::kInternal, "GENEB HyenaDNA topology was not retained"};
    }
    public_config.architecture = std::string{kGenebHyenaDnaArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebHyenaDnaDecoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebHyenaDnaModel runtime;
  ModelConfig public_config;
};

class GenebHyenaDnaContext final {
public:
  Status
  initialize_shared(const std::shared_ptr<GenebHyenaDnaModelAdapter> &model,
                    const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB HyenaDNA context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB HyenaDNA embedding requires a fresh full-sequence "
              "context and valid layer"};
    }
    GenebHyenaDnaForwardResult result;
    auto status =
        model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB HyenaDNA returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.kernel_name();
  }

private:
  std::shared_ptr<GenebHyenaDnaModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebEvo1ModelAdapter final {
public:
  Status load(const ModelFile &artifact, const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load(artifact);
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr) {
      return {ErrorCode::kInternal, "GENEB Evo-1 topology was not retained"};
    }
    public_config.architecture = std::string{kGenebEvo1Architecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebStripedHyenaV1;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebEvo1Model runtime;
  ModelConfig public_config;
};

class GenebEvo1Context final {
public:
  Status initialize_shared(const std::shared_ptr<GenebEvo1ModelAdapter> &model,
                           const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB Evo-1 context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB Evo-1 embedding requires a fresh full-sequence context "
              "and valid layer"};
    }
    GenebEvo1ForwardResult result;
    auto status = model_->runtime.forward(tokens, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB Evo-1 returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (attention_mask.size() != tokens.size() ||
        std::any_of(attention_mask.begin(), attention_mask.end(),
                    [](const std::uint8_t value) { return value != 1U; })) {
      return {ErrorCode::kUnsupported,
              "GENEB Evo-1 reference embedding does not accept padding"};
    }
    return prefill_embedding(tokens, layer, embedding);
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.kernel_name();
  }

private:
  std::shared_ptr<GenebEvo1ModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebJanusDnaModelAdapter final {
public:
  Status load(const ModelFile &artifact,
              std::shared_ptr<evo::detail::LinearExecutor> executor,
              const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load_artifact(artifact, std::move(executor));
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr) {
      return {ErrorCode::kInternal, "GENEB JanusDNA topology was not retained"};
    }
    public_config.architecture = std::string{kGenebJanusDnaArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebJanusDnaEncoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    tokenizer_vocabulary_size = topology->tokenizer_vocabulary_size;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebJanusDnaModel runtime;
  ModelConfig public_config;
  std::size_t tokenizer_vocabulary_size{0};
};

class GenebJanusDnaContext final {
public:
  Status
  initialize_shared(const std::shared_ptr<GenebJanusDnaModelAdapter> &model,
                    const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB JanusDNA context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer, embedding);
  }

  Status
  prefill_embedding_masked(const std::vector<TokenId> &tokens,
                           const std::vector<std::uint8_t> &attention_mask,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB JanusDNA embedding requires a fresh full-sequence "
              "context and valid layer"};
    }
    GenebJanusDnaForwardResult result;
    auto status =
        model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1 ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB JanusDNA returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.linear_executor_name();
  }

private:
  std::shared_ptr<GenebJanusDnaModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebSequenceCnnModelAdapter final {
public:
  Status load(const ModelFile &artifact,
              std::shared_ptr<evo::detail::LinearExecutor> executor,
              const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load_artifact(artifact, std::move(executor));
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr) {
      return {ErrorCode::kInternal,
              "GENEB sequence CNN topology was not retained"};
    }
    public_config.architecture = std::string{kGenebSequenceCnnArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebSequenceCnnEncoder;
    // The public transport is raw bytes. The runtime converts them to the
    // frozen four-channel one-hot input internally.
    public_config.tokenizer = ArchitectureTokenizer::kByteIdentity;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = 256U;
    public_config.width = topology->output_width;
    // Public layer 0 is the returned final sequence embedding. Internal
    // transformer depth is deliberately not exposed as hidden-state taps.
    public_config.layers = 0U;
    public_config.max_seqlen = topology->input_length;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebSequenceCnnModel runtime;
  ModelConfig public_config;
};

class GenebSequenceCnnContext final {
public:
  Status initialize_shared(
      const std::shared_ptr<GenebSequenceCnnModelAdapter> &model,
      const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB sequence CNN context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer,
        embedding);
  }

  Status prefill_embedding_masked(
      const std::vector<TokenId> &tokens,
      const std::vector<std::uint8_t> &attention_mask,
      const std::size_t layer, std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer != 0U ||
        std::any_of(attention_mask.begin(), attention_mask.end(),
                    [](const std::uint8_t value) { return value != 1U; })) {
      return {ErrorCode::kInvalidArgument,
              "GENEB sequence CNN embedding requires one fresh unmasked "
              "byte sequence at public layer zero"};
    }
    std::string sequence;
    sequence.reserve(tokens.size());
    for (const TokenId token : tokens) {
      if (token > 255U) {
        return {ErrorCode::kInvalidArgument,
                "GENEB sequence CNN transport token is not a byte"};
      }
      sequence.push_back(static_cast<char>(token));
    }
    GenebSequenceCnnForwardResult result;
    auto status = model_->runtime.forward(sequence, &result);
    if (!status.ok())
      return status;
    if (result.rows == 0 || result.width != model_->public_config.width ||
        result.rows > std::numeric_limits<std::size_t>::max() / result.width ||
        result.final_hidden.size() != result.rows * result.width) {
      return {ErrorCode::kInternal,
              "GENEB sequence CNN returned an incomplete final embedding"};
    }
    *embedding = std::move(result.final_hidden);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.linear_executor_name();
  }

private:
  std::shared_ptr<GenebSequenceCnnModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

class GenebRoformerModelAdapter final {
public:
  Status load(const ModelFile &artifact,
              std::shared_ptr<evo::detail::LinearExecutor> executor,
              const bool allow_test_fixture) {
    bool synthetic = false;
    if (artifact.find_metadata("fixture.synthetic") != nullptr) {
      auto status = metadata_bool(artifact, "fixture.synthetic", &synthetic);
      if (!status.ok())
        return status;
      if (synthetic && !allow_test_fixture) {
        return {ErrorCode::kUnsupported,
                "synthetic model fixtures require explicit test permission"};
      }
    }
    auto status = runtime.load_artifact(artifact, std::move(executor));
    if (!status.ok())
      return status;
    const auto *const topology = runtime.topology();
    if (topology == nullptr) {
      return {ErrorCode::kInternal,
              "GENEB RoFormer topology was not retained"};
    }
    public_config.architecture = std::string{kGenebRoformerArchitecture};
    public_config.artifact_profile = std::string{artifact.profile()};
    public_config.implementation =
        ArchitectureImplementation::kGenebRoformerEncoder;
    public_config.tokenizer = ArchitectureTokenizer::kArtifact;
    public_config.test_fixture = synthetic;
    public_config.vocab_size = topology->vocabulary_size;
    public_config.width = topology->width;
    public_config.layers = topology->layers;
    public_config.max_seqlen = topology->maximum_sequence_length;
    tokenizer_vocabulary_size = topology->tokenizer_vocabulary_size;
    if (artifact.find_metadata("model.id") != nullptr)
      status = metadata_string(artifact, "model.id", &public_config.model_id);
    else
      status = metadata_string(artifact, "model.name", &public_config.model_id);
    return status;
  }

  GenebRoformerModel runtime;
  ModelConfig public_config;
  std::size_t tokenizer_vocabulary_size{0};
};

class GenebRoformerContext final {
public:
  Status initialize_shared(
      const std::shared_ptr<GenebRoformerModelAdapter> &model,
      const std::size_t context_capacity) {
    if (!model || model_ || context_capacity == 0 ||
        context_capacity > model->public_config.max_seqlen) {
      return {ErrorCode::kInvalidArgument,
              "GENEB RoFormer context capacity/model is invalid"};
    }
    model_ = model;
    capacity_ = context_capacity;
    return Status::Ok();
  }

  Status prefill_embedding(const std::vector<TokenId> &tokens,
                           const std::size_t layer,
                           std::vector<float> *const embedding) {
    return prefill_embedding_masked(
        tokens, std::vector<std::uint8_t>(tokens.size(), 1U), layer,
        embedding);
  }

  Status prefill_embedding_masked(
      const std::vector<TokenId> &tokens,
      const std::vector<std::uint8_t> &attention_mask,
      const std::size_t layer, std::vector<float> *const embedding) {
    if (!model_ || embedding == nullptr || position_ != 0 || tokens.empty() ||
        tokens.size() > capacity_ || attention_mask.size() != tokens.size() ||
        layer > model_->public_config.layers) {
      return {ErrorCode::kInvalidArgument,
              "GENEB RoFormer embedding requires a fresh full-sequence "
              "context and valid layer"};
    }
    GenebRoformerForwardResult result;
    auto status =
        model_->runtime.forward(tokens, attention_mask, {layer}, &result);
    if (!status.ok())
      return status;
    if (result.captures.size() != 1U ||
        result.captures.front().layer != layer ||
        result.captures.front().values.size() !=
            tokens.size() * model_->public_config.width) {
      return {ErrorCode::kInternal,
              "GENEB RoFormer returned an incomplete hidden state"};
    }
    *embedding = std::move(result.captures.front().values);
    position_ = tokens.size();
    return Status::Ok();
  }

  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  [[nodiscard]] std::size_t activation_capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] const ModelConfig &config() const noexcept {
    return model_->public_config;
  }
  [[nodiscard]] const char *kernel_name() const noexcept {
    return model_->runtime.linear_executor_name();
  }

private:
  std::shared_ptr<GenebRoformerModelAdapter> model_;
  std::size_t capacity_{0};
  std::size_t position_{0};
};

} // namespace detail

Model::Model() : impl_(std::make_shared<Impl>()) {}
Model::~Model() = default;
Model::Model(Model &&other) noexcept
    : impl_(std::move(other.impl_)), hyena_(std::move(other.hyena_)),
      esmc_(std::move(other.esmc_)),
      geneb_decoder_(std::move(other.geneb_decoder_)),
      geneb_olmo_(std::move(other.geneb_olmo_)),
      geneb_esm_(std::move(other.geneb_esm_)),
      geneb_bert_(std::move(other.geneb_bert_)),
      geneb_gpt2_(std::move(other.geneb_gpt2_)),
      geneb_dna_gpt_(std::move(other.geneb_dna_gpt_)),
      geneb_custom_(std::move(other.geneb_custom_)),
      geneb_mamba_(std::move(other.geneb_mamba_)),
      geneb_hyenadna_(std::move(other.geneb_hyenadna_)),
      geneb_evo1_(std::move(other.geneb_evo1_)),
      geneb_janusdna_(std::move(other.geneb_janusdna_)),
      geneb_sequence_cnn_(std::move(other.geneb_sequence_cnn_)),
      geneb_roformer_(std::move(other.geneb_roformer_)),
      artifact_tokenizer_(std::move(other.artifact_tokenizer_)),
      geneb_embedding_spec_(std::move(other.geneb_embedding_spec_)),
      factory_(std::exchange(other.factory_, nullptr)) {}
Model &Model::operator=(Model &&other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
    hyena_ = std::move(other.hyena_);
    esmc_ = std::move(other.esmc_);
    geneb_decoder_ = std::move(other.geneb_decoder_);
    geneb_olmo_ = std::move(other.geneb_olmo_);
    geneb_esm_ = std::move(other.geneb_esm_);
    geneb_bert_ = std::move(other.geneb_bert_);
    geneb_gpt2_ = std::move(other.geneb_gpt2_);
    geneb_dna_gpt_ = std::move(other.geneb_dna_gpt_);
    geneb_custom_ = std::move(other.geneb_custom_);
    geneb_mamba_ = std::move(other.geneb_mamba_);
    geneb_hyenadna_ = std::move(other.geneb_hyenadna_);
    geneb_evo1_ = std::move(other.geneb_evo1_);
    geneb_janusdna_ = std::move(other.geneb_janusdna_);
    geneb_sequence_cnn_ = std::move(other.geneb_sequence_cnn_);
    geneb_roformer_ = std::move(other.geneb_roformer_);
    artifact_tokenizer_ = std::move(other.artifact_tokenizer_);
    geneb_embedding_spec_ = std::move(other.geneb_embedding_spec_);
    factory_ = std::exchange(other.factory_, nullptr);
  }
  return *this;
}

Status Model::load(const ModelFile &model, const bool allow_test_fixture) {
  return load_with_executor(model, {}, allow_test_fixture);
}

Status
Model::load_with_executor(const ModelFile &model,
                          std::shared_ptr<evo::detail::LinearExecutor> executor,
                          const bool allow_test_fixture) {
  if (factory_ != nullptr ||
      (!impl_ && !hyena_ && !esmc_ && !geneb_decoder_ && !geneb_olmo_ &&
       !geneb_custom_ && !geneb_hyenadna_ && !geneb_evo1_ &&
       !geneb_janusdna_ && !geneb_sequence_cnn_ && !geneb_roformer_))
    return {ErrorCode::kInvalidArgument, "host model is already loaded"};
  std::shared_ptr<ArtifactTokenizer> artifact_tokenizer;
  if (model.tokenizer_asset_descriptor().has_value()) {
    std::unique_ptr<ArtifactTokenizer> loaded;
    auto tokenizer_status =
        ArtifactTokenizer::Load(std::string{model.artifact_root()},
                                *model.tokenizer_asset_descriptor(), &loaded);
    if (!tokenizer_status.ok())
      return tokenizer_status;
    artifact_tokenizer = std::move(loaded);
  }
  std::string runtime_abi;
  std::string architecture;
  auto status = metadata_string(model, "runtime.abi", &runtime_abi);
  if (!status.ok())
    return status;
  status = metadata_string(model, "model.architecture", &architecture);
  if (!status.ok())
    return status;
  const auto *const registered = find_architecture(architecture);
  if (registered == nullptr) {
    return {ErrorCode::kUnsupported,
            "unregistered runtime architecture: " + architecture};
  }
  if (registered->artifact_profile != model.profile() ||
      registered->runtime_abi != runtime_abi) {
    return {ErrorCode::kUnsupported,
            "artifact profile/runtime ABI does not match architecture '" +
                architecture + "'"};
  }
  const auto backend =
      executor ? kArchitectureBackendMps : kArchitectureBackendCpu;
  const auto *const factory =
      find_architecture_backend_factory(*registered, backend);
  if (factory == nullptr) {
    return {ErrorCode::kUnsupported,
            "no " + std::string{executor ? "MPS" : "CPU"} +
                " backend factory is registered for architecture '" +
                architecture + "'"};
  }
  std::optional<GenebEmbeddingArtifactSpec> geneb_embedding_spec;
  switch (factory->implementation) {
  case ArchitectureImplementation::kGenebTransformerDecoder:
  case ArchitectureImplementation::kGenebOlmoDecoder:
  case ArchitectureImplementation::kGenebEsmEncoder:
  case ArchitectureImplementation::kGenebBertEncoder:
  case ArchitectureImplementation::kGenebGpt2Decoder:
  case ArchitectureImplementation::kGenebDnaGptDecoder:
  case ArchitectureImplementation::kGenebCustomEncoder:
  case ArchitectureImplementation::kGenebMambaEncoder:
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
  case ArchitectureImplementation::kGenebStripedHyenaV1:
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
  case ArchitectureImplementation::kGenebRoformerEncoder: {
    GenebEmbeddingArtifactSpec compiled;
    status = geneb_embedding_spec_from_artifact(model, &compiled);
    if (!status.ok())
      return status;
    geneb_embedding_spec = std::move(compiled);
    break;
  }
  case ArchitectureImplementation::kUnknown:
  case ArchitectureImplementation::kStripedHyena2:
  case ArchitectureImplementation::kHyenaDna:
  case ArchitectureImplementation::kEsmc:
    break;
  }
  switch (factory->implementation) {
  case ArchitectureImplementation::kUnknown:
    return {ErrorCode::kUnsupported,
            "architecture backend factory has unknown implementation"};
  case ArchitectureImplementation::kHyenaDna: {
    auto candidate = std::make_shared<detail::HyenaDnaModel>();
    status = candidate->load(model, allow_test_fixture, std::move(executor));
    if (!status.ok())
      return status;
    if (artifact_tokenizer &&
        artifact_tokenizer->vocabulary_size() != candidate->config().vocab_size)
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from model vocabulary"};
    impl_.reset();
    hyena_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kEsmc: {
    auto candidate = std::make_shared<detail::EsmcModel>();
    status = candidate->load(model, allow_test_fixture, std::move(executor));
    if (!status.ok())
      return status;
    if (artifact_tokenizer &&
        artifact_tokenizer->vocabulary_size() != candidate->config().vocab_size)
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from model vocabulary"};
    impl_.reset();
    esmc_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebTransformerDecoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB decoder artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebDecoderModelAdapter>();
    status = candidate->load(model, std::move(executor), allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->public_config.vocab_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from model vocabulary"};
    }
    impl_.reset();
    geneb_decoder_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebOlmoDecoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB OLMo artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebOlmoModelAdapter>();
    status = candidate->load(model, std::move(executor), allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->public_config.vocab_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from model vocabulary"};
    }
    impl_.reset();
    geneb_olmo_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebEsmEncoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB ESM artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebEsmModelAdapter>();
    status = candidate->load(model, std::move(executor), allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->public_config.vocab_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from model vocabulary"};
    }
    impl_.reset();
    geneb_esm_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebBertEncoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB BERT artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebBertModelAdapter>();
    status = candidate->load(model, std::move(executor), allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->public_config.vocab_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from model vocabulary"};
    }
    impl_.reset();
    geneb_bert_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebGpt2Decoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB GPT-2 artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebGpt2ModelAdapter>();
    status = candidate->load(model, allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->public_config.vocab_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from model vocabulary"};
    }
    impl_.reset();
    geneb_gpt2_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebDnaGptDecoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB DNA-GPT artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebDnaGptModelAdapter>();
    status = candidate->load(model, allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->public_config.vocab_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from model vocabulary"};
    }
    impl_.reset();
    geneb_dna_gpt_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebCustomEncoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB custom encoder artifact is missing its tokenizer "
              "descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebCustomEncoderModelAdapter>();
    status = candidate->load(model, std::move(executor), allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->tokenizer_vocabulary_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from custom encoder "
              "tokenizer vocabulary"};
    }
    impl_.reset();
    geneb_custom_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebMambaEncoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB Mamba artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebMambaModelAdapter>();
    status = candidate->load(model, std::move(executor), allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->tokenizer_vocabulary_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from GENEB Mamba "
              "tokenizer vocabulary"};
    }
    impl_.reset();
    geneb_mamba_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebHyenaDnaDecoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB HyenaDNA artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebHyenaDnaModelAdapter>();
    status = candidate->load(model, allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->public_config.vocab_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from GENEB HyenaDNA "
              "vocabulary"};
    }
    impl_.reset();
    geneb_hyenadna_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebStripedHyenaV1: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB Evo-1 artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebEvo1ModelAdapter>();
    status = candidate->load(model, allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->public_config.vocab_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from GENEB Evo-1 "
              "vocabulary"};
    }
    impl_.reset();
    geneb_evo1_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebJanusDnaEncoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB JanusDNA artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebJanusDnaModelAdapter>();
    status = candidate->load(model, std::move(executor), allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->tokenizer_vocabulary_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from GENEB JanusDNA "
              "tokenizer vocabulary"};
    }
    impl_.reset();
    geneb_janusdna_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebSequenceCnnEncoder: {
    if (artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB sequence CNN must use the registered byte transport "
              "instead of an artifact tokenizer"};
    }
    auto candidate =
        std::make_shared<detail::GenebSequenceCnnModelAdapter>();
    status = candidate->load(model, std::move(executor), allow_test_fixture);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_sequence_cnn_ = std::move(candidate);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebRoformerEncoder: {
    if (!artifact_tokenizer) {
      return {ErrorCode::kModelFormat,
              "GENEB RoFormer artifact is missing its tokenizer descriptor"};
    }
    auto candidate = std::make_shared<detail::GenebRoformerModelAdapter>();
    status = candidate->load(model, std::move(executor), allow_test_fixture);
    if (!status.ok())
      return status;
    if (artifact_tokenizer->vocabulary_size() !=
        candidate->tokenizer_vocabulary_size) {
      return {ErrorCode::kModelFormat,
              "tokenizer vocabulary size differs from GENEB RoFormer "
              "tokenizer vocabulary"};
    }
    impl_.reset();
    geneb_roformer_ = std::move(candidate);
    artifact_tokenizer_ = std::move(artifact_tokenizer);
    geneb_embedding_spec_ = std::move(geneb_embedding_spec);
    factory_ = factory;
    return Status::Ok();
  }
  case ArchitectureImplementation::kStripedHyena2:
    break;
  }
  auto candidate = std::make_shared<Impl>();
  candidate->executor = std::move(executor);
  candidate->public_config.architecture = architecture;
  candidate->public_config.artifact_profile = std::string{model.profile()};
  candidate->public_config.implementation = registered->implementation;
  candidate->public_config.tokenizer = registered->tokenizer;
  candidate->public_config.test_fixture = registered->synthetic_fixture;
  if (candidate->public_config.test_fixture) {
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
    return {ErrorCode::kUnsupported, "host model dimensions are unsupported"};
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
    return {ErrorCode::kModelFormat,
            "mixer lists do not cover all host blocks"};

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
  if (artifact_tokenizer && artifact_tokenizer->vocabulary_size() !=
                                candidate->public_config.vocab_size)
    return {ErrorCode::kModelFormat,
            "tokenizer vocabulary size differs from model vocabulary"};
  impl_ = std::move(candidate);
  artifact_tokenizer_ = std::move(artifact_tokenizer);
  factory_ = factory;
  return Status::Ok();
}

const ModelConfig &Model::config() const noexcept {
  if (factory_ == nullptr) {
    if (impl_)
      return impl_->public_config;
    static const ModelConfig unloaded;
    return unloaded;
  }
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    break;
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->public_config;
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->config();
  case ArchitectureImplementation::kEsmc:
    return esmc_->config();
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return geneb_decoder_->public_config;
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return geneb_olmo_->public_config;
  case ArchitectureImplementation::kGenebEsmEncoder:
    return geneb_esm_->public_config;
  case ArchitectureImplementation::kGenebBertEncoder:
    return geneb_bert_->public_config;
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return geneb_gpt2_->public_config;
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return geneb_dna_gpt_->public_config;
  case ArchitectureImplementation::kGenebCustomEncoder:
    return geneb_custom_->public_config;
  case ArchitectureImplementation::kGenebMambaEncoder:
    return geneb_mamba_->public_config;
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return geneb_hyenadna_->public_config;
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return geneb_evo1_->public_config;
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return geneb_janusdna_->public_config;
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return geneb_sequence_cnn_->public_config;
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return geneb_roformer_->public_config;
  }
  static const ModelConfig unsupported;
  return unsupported;
}
const char *Model::kernel_name() const noexcept {
  if (factory_ == nullptr)
    return "unloaded";
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    return "unsupported-architecture";
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->executor ? impl_->executor->name() : selected_kernel_name();
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->kernel_name();
  case ArchitectureImplementation::kEsmc:
    return esmc_->kernel_name();
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return geneb_decoder_->runtime.linear_executor_name();
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return geneb_olmo_->runtime.linear_executor_name();
  case ArchitectureImplementation::kGenebEsmEncoder:
    return geneb_esm_->runtime.linear_executor_name();
  case ArchitectureImplementation::kGenebBertEncoder:
    return geneb_bert_->runtime.linear_executor_name();
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return geneb_gpt2_->runtime.kernel_name();
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return geneb_dna_gpt_->runtime.kernel_name();
  case ArchitectureImplementation::kGenebCustomEncoder:
    return geneb_custom_->runtime.linear_executor_name();
  case ArchitectureImplementation::kGenebMambaEncoder:
    return geneb_mamba_->runtime.linear_executor_name();
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return geneb_hyenadna_->runtime.kernel_name();
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return geneb_evo1_->runtime.kernel_name();
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return geneb_janusdna_->runtime.linear_executor_name();
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return geneb_sequence_cnn_->runtime.linear_executor_name();
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return geneb_roformer_->runtime.linear_executor_name();
  }
  return "unsupported-architecture";
}

Status Model::encode(const std::string_view sequence,
                     std::vector<TokenId> *const tokens) const {
  if (artifact_tokenizer_)
    return artifact_tokenizer_->encode(sequence, {}, tokens);
  return encode_sequence(config().tokenizer, sequence, tokens);
}

Status Model::decode_token(const TokenId token,
                           std::uint8_t *const byte) const {
  if (artifact_tokenizer_)
    return {ErrorCode::kUnsupported,
            "artifact tokenizer detokenization is not supported"};
  return decode_sequence_token(config().tokenizer, token, byte);
}

const GenebEmbeddingArtifactSpec *Model::geneb_embedding_spec() const noexcept {
  return geneb_embedding_spec_ ? &*geneb_embedding_spec_ : nullptr;
}

Status Model::prepare_geneb_embedding_input(
    const std::string_view sequence,
    GenebPreparedEmbeddingInput *const output) const {
  if (!geneb_embedding_spec_) {
    return {ErrorCode::kUnsupported,
            "GENEB preset requires verified artifact metadata"};
  }
  if (artifact_tokenizer_) {
    return evo::prepare_geneb_embedding_input(sequence, *geneb_embedding_spec_,
                                              *artifact_tokenizer_, output);
  }
  if (config().implementation !=
          ArchitectureImplementation::kGenebSequenceCnnEncoder ||
      config().tokenizer != ArchitectureTokenizer::kByteIdentity ||
      output == nullptr) {
    return {ErrorCode::kUnsupported,
            "GENEB preset requires a verified artifact tokenizer"};
  }
  GenebPreparedEmbeddingInput prepared;
  auto status = transform_geneb_input(sequence,
                                      geneb_embedding_spec_->input_transform,
                                      &prepared.transform);
  if (!status.ok())
    return status;
  if (prepared.transform.special_token_policy !=
          GenebSpecialTokenPolicy::kNone ||
      !prepared.transform.prefix.empty()) {
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN byte transport cannot add special tokens"};
  }
  status = encode_sequence(ArchitectureTokenizer::kByteIdentity,
                           prepared.transform.sequence, &prepared.tokens);
  if (!status.ok())
    return status;
  status = plan_geneb_token_length(prepared.tokens.size(),
                                   geneb_embedding_spec_->input_transform,
                                   &prepared.token_plan);
  if (!status.ok())
    return status;
  if (prepared.token_plan.deferred_to_model_preset ||
      prepared.token_plan.retained_token_count != prepared.tokens.size() ||
      prepared.token_plan.pad_left != 0U ||
      prepared.token_plan.pad_right != 0U) {
    return {ErrorCode::kModelFormat,
            "GENEB sequence CNN base transform unexpectedly requires a token "
            "length policy"};
  }
  prepared.attention_mask.assign(prepared.tokens.size(), 1U);
  if (prepared.tokens.empty()) {
    return {ErrorCode::kInvalidArgument,
            "GENEB preprocessing produced an empty sequence"};
  }
  *output = std::move(prepared);
  return Status::Ok();
}

Context::Context() : impl_(std::make_unique<Impl>()) {}
Context::~Context() = default;
Context::Context(Context &&other) noexcept
    : impl_(std::move(other.impl_)), hyena_(std::move(other.hyena_)),
      esmc_(std::move(other.esmc_)),
      geneb_decoder_(std::move(other.geneb_decoder_)),
      geneb_olmo_(std::move(other.geneb_olmo_)),
      geneb_esm_(std::move(other.geneb_esm_)),
      geneb_bert_(std::move(other.geneb_bert_)),
      geneb_gpt2_(std::move(other.geneb_gpt2_)),
      geneb_dna_gpt_(std::move(other.geneb_dna_gpt_)),
      geneb_custom_(std::move(other.geneb_custom_)),
      geneb_mamba_(std::move(other.geneb_mamba_)),
      geneb_hyenadna_(std::move(other.geneb_hyenadna_)),
      geneb_evo1_(std::move(other.geneb_evo1_)),
      geneb_janusdna_(std::move(other.geneb_janusdna_)),
      geneb_sequence_cnn_(std::move(other.geneb_sequence_cnn_)),
      geneb_roformer_(std::move(other.geneb_roformer_)),
      factory_(std::exchange(other.factory_, nullptr)) {}
Context &Context::operator=(Context &&other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
    hyena_ = std::move(other.hyena_);
    esmc_ = std::move(other.esmc_);
    geneb_decoder_ = std::move(other.geneb_decoder_);
    geneb_olmo_ = std::move(other.geneb_olmo_);
    geneb_esm_ = std::move(other.geneb_esm_);
    geneb_bert_ = std::move(other.geneb_bert_);
    geneb_gpt2_ = std::move(other.geneb_gpt2_);
    geneb_dna_gpt_ = std::move(other.geneb_dna_gpt_);
    geneb_custom_ = std::move(other.geneb_custom_);
    geneb_mamba_ = std::move(other.geneb_mamba_);
    geneb_hyenadna_ = std::move(other.geneb_hyenadna_);
    geneb_evo1_ = std::move(other.geneb_evo1_);
    geneb_janusdna_ = std::move(other.geneb_janusdna_);
    geneb_sequence_cnn_ = std::move(other.geneb_sequence_cnn_);
    geneb_roformer_ = std::move(other.geneb_roformer_);
    factory_ = std::exchange(other.factory_, nullptr);
  }
  return *this;
}

Status Context::initialize_shared(const Model &model,
                                  const std::size_t context_capacity,
                                  const std::size_t layer_begin) {
  if (factory_ != nullptr)
    return {ErrorCode::kInvalidArgument, "host context is already initialized"};
  if (model.factory_ == nullptr)
    return {ErrorCode::kInvalidArgument,
            "host context requires a loaded model"};
  switch (model.factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    return {ErrorCode::kUnsupported,
            "host model implementation is not registered"};
  case ArchitectureImplementation::kHyenaDna: {
    if (!impl_ || !model.hyena_ || layer_begin != 0)
      return {ErrorCode::kInvalidArgument,
              "HyenaDNA contexts do not support suffix-layer placement"};
    auto candidate = std::make_unique<detail::HyenaDnaContext>();
    const auto status =
        candidate->initialize_shared(*model.hyena_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    hyena_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebSequenceCnnEncoder: {
    if (!impl_ || !model.geneb_sequence_cnn_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB sequence CNN contexts do not support suffix-layer "
              "placement"};
    }
    auto candidate = std::make_unique<detail::GenebSequenceCnnContext>();
    const auto status = candidate->initialize_shared(
        model.geneb_sequence_cnn_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_sequence_cnn_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebRoformerEncoder: {
    if (!impl_ || !model.geneb_roformer_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB RoFormer contexts do not support suffix-layer "
              "placement"};
    }
    auto candidate = std::make_unique<detail::GenebRoformerContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_roformer_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_roformer_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kEsmc: {
    if (!impl_ || !model.esmc_ || layer_begin != 0)
      return {ErrorCode::kInvalidArgument,
              "ESMC contexts do not support suffix-layer placement"};
    auto candidate = std::make_unique<detail::EsmcContext>();
    const auto status =
        candidate->initialize_shared(*model.esmc_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    esmc_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebTransformerDecoder: {
    if (!impl_ || !model.geneb_decoder_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB decoder contexts do not support suffix-layer placement"};
    }
    auto candidate = std::make_unique<detail::GenebDecoderContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_decoder_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_decoder_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebOlmoDecoder: {
    if (!impl_ || !model.geneb_olmo_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB OLMo contexts do not support suffix-layer placement"};
    }
    auto candidate = std::make_unique<detail::GenebOlmoContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_olmo_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_olmo_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebEsmEncoder: {
    if (!impl_ || !model.geneb_esm_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB ESM contexts do not support suffix-layer placement"};
    }
    auto candidate = std::make_unique<detail::GenebEsmContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_esm_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_esm_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebBertEncoder: {
    if (!impl_ || !model.geneb_bert_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB BERT contexts do not support suffix-layer placement"};
    }
    auto candidate = std::make_unique<detail::GenebBertContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_bert_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_bert_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebGpt2Decoder: {
    if (!impl_ || !model.geneb_gpt2_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB GPT-2 contexts do not support suffix-layer placement"};
    }
    auto candidate = std::make_unique<detail::GenebGpt2Context>();
    const auto status =
        candidate->initialize_shared(model.geneb_gpt2_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_gpt2_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebDnaGptDecoder: {
    if (!impl_ || !model.geneb_dna_gpt_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB DNA-GPT contexts do not support suffix-layer placement"};
    }
    auto candidate = std::make_unique<detail::GenebDnaGptContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_dna_gpt_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_dna_gpt_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebCustomEncoder: {
    if (!impl_ || !model.geneb_custom_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB custom encoder contexts do not support suffix-layer "
              "placement"};
    }
    auto candidate = std::make_unique<detail::GenebCustomEncoderContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_custom_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_custom_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebMambaEncoder: {
    if (!impl_ || !model.geneb_mamba_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB Mamba contexts do not support suffix-layer placement"};
    }
    auto candidate = std::make_unique<detail::GenebMambaContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_mamba_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_mamba_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebHyenaDnaDecoder: {
    if (!impl_ || !model.geneb_hyenadna_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB HyenaDNA contexts do not support suffix-layer "
              "placement"};
    }
    auto candidate = std::make_unique<detail::GenebHyenaDnaContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_hyenadna_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_hyenadna_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebStripedHyenaV1: {
    if (!impl_ || !model.geneb_evo1_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB Evo-1 contexts do not support suffix-layer placement"};
    }
    auto candidate = std::make_unique<detail::GenebEvo1Context>();
    const auto status =
        candidate->initialize_shared(model.geneb_evo1_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_evo1_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kGenebJanusDnaEncoder: {
    if (!impl_ || !model.geneb_janusdna_ || layer_begin != 0) {
      return {ErrorCode::kInvalidArgument,
              "GENEB JanusDNA contexts do not support suffix-layer "
              "placement"};
    }
    auto candidate = std::make_unique<detail::GenebJanusDnaContext>();
    const auto status =
        candidate->initialize_shared(model.geneb_janusdna_, context_capacity);
    if (!status.ok())
      return status;
    impl_.reset();
    geneb_janusdna_ = std::move(candidate);
    factory_ = model.factory_;
    return Status::Ok();
  }
  case ArchitectureImplementation::kStripedHyena2:
    break;
  }
  if (!impl_ || impl_->weights || !model.impl_ ||
      model.impl_->file == nullptr || context_capacity == 0 ||
      context_capacity > model.impl_->public_config.max_seqlen ||
      layer_begin > model.impl_->public_config.layers) {
    return {ErrorCode::kInvalidArgument,
            "host context requires a loaded model and valid capacity"};
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
  factory_ = model.factory_;
  return Status::Ok();
}

Status Context::prefill(const std::vector<TokenId> &tokens,
                        std::vector<float> *const logits) {
  if (factory_ == nullptr)
    return {ErrorCode::kInvalidArgument, "host context is not initialized"};
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    break;
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->forward(tokens, true, logits,
                          std::numeric_limits<std::size_t>::max(), nullptr);
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->prefill(tokens, logits);
  case ArchitectureImplementation::kEsmc:
    return esmc_->prefill(tokens, logits);
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB decoder embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB OLMo embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebEsmEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB ESM embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebBertEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB BERT embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return {ErrorCode::kUnsupported,
            "GENEB GPT-2 embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB DNA-GPT embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebCustomEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB custom encoder embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebMambaEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB Mamba embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB HyenaDNA embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return {ErrorCode::kUnsupported,
            "GENEB Evo-1 embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB JanusDNA embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB sequence CNN embedding profile does not expose logits"};
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB RoFormer embedding profile does not expose logits"};
  }
  return {ErrorCode::kUnsupported,
          "host context implementation is not registered"};
}

Status Context::prefill_chunk(const std::vector<TokenId> &tokens,
                              std::vector<float> *const logits) {
  if (factory_ == nullptr)
    return {ErrorCode::kInvalidArgument, "host context is not initialized"};
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    break;
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->forward(tokens, false, logits,
                          std::numeric_limits<std::size_t>::max(), nullptr);
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->prefill_chunk(tokens, logits);
  case ArchitectureImplementation::kEsmc:
    return {ErrorCode::kUnsupported,
            "ESMC requires one full-sequence bidirectional prefill"};
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB decoder requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB OLMo requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebEsmEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB ESM requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebBertEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB BERT requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return {ErrorCode::kUnsupported,
            "GENEB GPT-2 requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB DNA-GPT requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebCustomEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB custom encoder requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebMambaEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB Mamba requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB HyenaDNA requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return {ErrorCode::kUnsupported,
            "GENEB Evo-1 requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB JanusDNA requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB sequence CNN requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB RoFormer requires one full-sequence embedding"};
  }
  return {ErrorCode::kUnsupported,
          "host context implementation is not registered"};
}

Status Context::decode(const TokenId token, std::vector<float> *const logits) {
  if (factory_ == nullptr)
    return {ErrorCode::kInvalidArgument, "host context is not initialized"};
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    break;
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->forward({token}, false, logits,
                          std::numeric_limits<std::size_t>::max(), nullptr);
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->decode(token, logits);
  case ArchitectureImplementation::kEsmc:
    return {ErrorCode::kUnsupported,
            "ESMC does not support autoregressive decode"};
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB decoder embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB OLMo embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebEsmEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB ESM embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebBertEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB BERT embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return {ErrorCode::kUnsupported,
            "GENEB GPT-2 embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB DNA-GPT embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebCustomEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB custom encoder embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebMambaEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB Mamba embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB HyenaDNA embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return {ErrorCode::kUnsupported,
            "GENEB Evo-1 embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB JanusDNA embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB sequence CNN embedding profile does not support decode"};
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB RoFormer embedding profile does not support decode"};
  }
  return {ErrorCode::kUnsupported,
          "host context implementation is not registered"};
}

Status Context::prefill_embedding(const std::vector<TokenId> &tokens,
                                  const std::size_t layer,
                                  std::vector<float> *const embedding) {
  if (factory_ == nullptr)
    return {ErrorCode::kInvalidArgument, "host context is not initialized"};
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    break;
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->forward(tokens, true, nullptr, layer, embedding);
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kEsmc:
    return esmc_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return geneb_decoder_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return geneb_olmo_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebEsmEncoder:
    return geneb_esm_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebBertEncoder:
    return geneb_bert_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return geneb_gpt2_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return geneb_dna_gpt_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebCustomEncoder:
    return geneb_custom_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebMambaEncoder:
    return geneb_mamba_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return geneb_hyenadna_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return geneb_evo1_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return geneb_janusdna_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return geneb_sequence_cnn_->prefill_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return geneb_roformer_->prefill_embedding(tokens, layer, embedding);
  }
  return {ErrorCode::kUnsupported,
          "host context implementation is not registered"};
}

Status Context::prefill_embedding_masked(
    const std::vector<TokenId> &tokens,
    const std::vector<std::uint8_t> &attention_mask, const std::size_t layer,
    std::vector<float> *const embedding) {
  if (factory_ == nullptr)
    return {ErrorCode::kInvalidArgument, "host context is not initialized"};
  if (attention_mask.size() != tokens.size())
    return {ErrorCode::kInvalidArgument,
            "embedding attention mask size differs from token count"};
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    break;
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return geneb_decoder_->prefill_embedding_masked(tokens, attention_mask,
                                                    layer, embedding);
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return geneb_olmo_->prefill_embedding_masked(tokens, attention_mask, layer,
                                                 embedding);
  case ArchitectureImplementation::kGenebEsmEncoder:
    return geneb_esm_->prefill_embedding_masked(tokens, attention_mask, layer,
                                                embedding);
  case ArchitectureImplementation::kGenebBertEncoder:
    return geneb_bert_->prefill_embedding_masked(tokens, attention_mask, layer,
                                                 embedding);
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return geneb_gpt2_->prefill_embedding_masked(tokens, attention_mask, layer,
                                                 embedding);
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return geneb_dna_gpt_->prefill_embedding_masked(tokens, attention_mask,
                                                    layer, embedding);
  case ArchitectureImplementation::kGenebCustomEncoder:
    return geneb_custom_->prefill_embedding_masked(tokens, attention_mask,
                                                   layer, embedding);
  case ArchitectureImplementation::kGenebMambaEncoder:
    return geneb_mamba_->prefill_embedding_masked(tokens, attention_mask, layer,
                                                  embedding);
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return geneb_hyenadna_->prefill_embedding_masked(tokens, attention_mask,
                                                     layer, embedding);
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return geneb_evo1_->prefill_embedding_masked(tokens, attention_mask, layer,
                                                 embedding);
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return geneb_janusdna_->prefill_embedding_masked(tokens, attention_mask,
                                                     layer, embedding);
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return geneb_sequence_cnn_->prefill_embedding_masked(
        tokens, attention_mask, layer, embedding);
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return geneb_roformer_->prefill_embedding_masked(tokens, attention_mask,
                                                     layer, embedding);
  case ArchitectureImplementation::kStripedHyena2:
  case ArchitectureImplementation::kHyenaDna:
  case ArchitectureImplementation::kEsmc:
    if (std::any_of(attention_mask.begin(), attention_mask.end(),
                    [](const std::uint8_t value) { return value != 1U; })) {
      return {ErrorCode::kUnsupported,
              "legacy architecture does not accept an embedding mask"};
    }
    return prefill_embedding(tokens, layer, embedding);
  }
  return {ErrorCode::kUnsupported,
          "host context implementation is not registered"};
}

Status Context::prefill_chunk_embedding(const std::vector<TokenId> &tokens,
                                        const std::size_t layer,
                                        std::vector<float> *const embedding) {
  if (factory_ == nullptr)
    return {ErrorCode::kInvalidArgument, "host context is not initialized"};
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    break;
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->forward(tokens, false, nullptr, layer, embedding);
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->prefill_chunk_embedding(tokens, layer, embedding);
  case ArchitectureImplementation::kEsmc:
    return {ErrorCode::kUnsupported,
            "ESMC requires one full-sequence bidirectional embedding"};
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB decoder requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB OLMo requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebEsmEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB ESM requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebBertEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB BERT requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return {ErrorCode::kUnsupported,
            "GENEB GPT-2 requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB DNA-GPT requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebCustomEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB custom encoder requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebMambaEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB Mamba requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return {ErrorCode::kUnsupported,
            "GENEB HyenaDNA requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return {ErrorCode::kUnsupported,
            "GENEB Evo-1 requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB JanusDNA requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB sequence CNN requires one full-sequence embedding"};
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return {ErrorCode::kUnsupported,
            "GENEB RoFormer requires one full-sequence embedding"};
  }
  return {ErrorCode::kUnsupported,
          "host context implementation is not registered"};
}

Status Context::prefill_from_hidden(const std::vector<float> &hidden,
                                    const std::size_t rows,
                                    std::vector<float> *const logits) {
  if (factory_ == nullptr)
    return {ErrorCode::kInvalidArgument, "host context is not initialized"};
  if (factory_->implementation != ArchitectureImplementation::kStripedHyena2) {
    return {ErrorCode::kUnsupported,
            "this architecture does not support CUDA-prefix hidden input"};
  }
  return impl_->forward_hidden(hidden, rows, true, logits,
                               std::numeric_limits<std::size_t>::max(),
                               nullptr);
}

Status Context::prefill_chunk_from_hidden(const std::vector<float> &hidden,
                                          const std::size_t rows,
                                          std::vector<float> *const logits) {
  if (factory_ == nullptr)
    return {ErrorCode::kInvalidArgument, "host context is not initialized"};
  if (factory_->implementation != ArchitectureImplementation::kStripedHyena2) {
    return {ErrorCode::kUnsupported,
            "this architecture does not support CUDA-prefix hidden input"};
  }
  return impl_->forward_hidden(hidden, rows, false, logits,
                               std::numeric_limits<std::size_t>::max(),
                               nullptr);
}

std::size_t Context::position() const noexcept {
  if (factory_ == nullptr)
    return 0;
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    return 0;
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->position;
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->position();
  case ArchitectureImplementation::kEsmc:
    return esmc_->position();
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return geneb_decoder_->position();
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return geneb_olmo_->position();
  case ArchitectureImplementation::kGenebEsmEncoder:
    return geneb_esm_->position();
  case ArchitectureImplementation::kGenebBertEncoder:
    return geneb_bert_->position();
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return geneb_gpt2_->position();
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return geneb_dna_gpt_->position();
  case ArchitectureImplementation::kGenebCustomEncoder:
    return geneb_custom_->position();
  case ArchitectureImplementation::kGenebMambaEncoder:
    return geneb_mamba_->position();
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return geneb_hyenadna_->position();
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return geneb_evo1_->position();
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return geneb_janusdna_->position();
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return geneb_sequence_cnn_->position();
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return geneb_roformer_->position();
  }
  return 0;
}
std::size_t Context::activation_capacity() const noexcept {
  if (factory_ == nullptr)
    return 0;
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    return 0;
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->arena_capacity;
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->activation_capacity();
  case ArchitectureImplementation::kEsmc:
    return esmc_->activation_capacity();
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return geneb_decoder_->activation_capacity();
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return geneb_olmo_->activation_capacity();
  case ArchitectureImplementation::kGenebEsmEncoder:
    return geneb_esm_->activation_capacity();
  case ArchitectureImplementation::kGenebBertEncoder:
    return geneb_bert_->activation_capacity();
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return geneb_gpt2_->activation_capacity();
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return geneb_dna_gpt_->activation_capacity();
  case ArchitectureImplementation::kGenebCustomEncoder:
    return geneb_custom_->activation_capacity();
  case ArchitectureImplementation::kGenebMambaEncoder:
    return geneb_mamba_->activation_capacity();
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return geneb_hyenadna_->activation_capacity();
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return geneb_evo1_->activation_capacity();
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return geneb_janusdna_->activation_capacity();
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return geneb_sequence_cnn_->activation_capacity();
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return geneb_roformer_->activation_capacity();
  }
  return 0;
}
const ModelConfig &Context::config() const noexcept {
  if (factory_ == nullptr) {
    static const ModelConfig unloaded;
    return unloaded;
  }
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    break;
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->weights->public_config;
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->config();
  case ArchitectureImplementation::kEsmc:
    return esmc_->config();
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return geneb_decoder_->config();
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return geneb_olmo_->config();
  case ArchitectureImplementation::kGenebEsmEncoder:
    return geneb_esm_->config();
  case ArchitectureImplementation::kGenebBertEncoder:
    return geneb_bert_->config();
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return geneb_gpt2_->config();
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return geneb_dna_gpt_->config();
  case ArchitectureImplementation::kGenebCustomEncoder:
    return geneb_custom_->config();
  case ArchitectureImplementation::kGenebMambaEncoder:
    return geneb_mamba_->config();
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return geneb_hyenadna_->config();
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return geneb_evo1_->config();
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return geneb_janusdna_->config();
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return geneb_sequence_cnn_->config();
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return geneb_roformer_->config();
  }
  static const ModelConfig unsupported;
  return unsupported;
}
const char *Context::kernel_name() const noexcept {
  if (factory_ == nullptr)
    return "unloaded";
  switch (factory_->implementation) {
  case ArchitectureImplementation::kUnknown:
    return "unsupported-architecture";
  case ArchitectureImplementation::kStripedHyena2:
    return impl_->weights->executor ? impl_->weights->executor->name()
                                    : selected_kernel_name();
  case ArchitectureImplementation::kHyenaDna:
    return hyena_->kernel_name();
  case ArchitectureImplementation::kEsmc:
    return esmc_->kernel_name();
  case ArchitectureImplementation::kGenebTransformerDecoder:
    return geneb_decoder_->kernel_name();
  case ArchitectureImplementation::kGenebOlmoDecoder:
    return geneb_olmo_->kernel_name();
  case ArchitectureImplementation::kGenebEsmEncoder:
    return geneb_esm_->kernel_name();
  case ArchitectureImplementation::kGenebBertEncoder:
    return geneb_bert_->kernel_name();
  case ArchitectureImplementation::kGenebGpt2Decoder:
    return geneb_gpt2_->kernel_name();
  case ArchitectureImplementation::kGenebDnaGptDecoder:
    return geneb_dna_gpt_->kernel_name();
  case ArchitectureImplementation::kGenebCustomEncoder:
    return geneb_custom_->kernel_name();
  case ArchitectureImplementation::kGenebMambaEncoder:
    return geneb_mamba_->kernel_name();
  case ArchitectureImplementation::kGenebHyenaDnaDecoder:
    return geneb_hyenadna_->kernel_name();
  case ArchitectureImplementation::kGenebStripedHyenaV1:
    return geneb_evo1_->kernel_name();
  case ArchitectureImplementation::kGenebJanusDnaEncoder:
    return geneb_janusdna_->kernel_name();
  case ArchitectureImplementation::kGenebSequenceCnnEncoder:
    return geneb_sequence_cnn_->kernel_name();
  case ArchitectureImplementation::kGenebRoformerEncoder:
    return geneb_roformer_->kernel_name();
  }
  return "unsupported-architecture";
}

} // namespace evo::cpu
