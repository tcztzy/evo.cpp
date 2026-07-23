// SPDX-License-Identifier: Apache-2.0
#include "evo2c/cpu_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace evo2c::cpu {
namespace {

bool product(const std::size_t left, const std::size_t right, std::size_t* const result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

Status matrix_size(const std::size_t rows,
                   const std::size_t columns,
                   const std::size_t actual,
                   const std::string& name) {
  std::size_t expected = 0;
  if (rows == 0 || columns == 0 || !product(rows, columns, &expected)) {
    return {ErrorCode::kInvalidArgument, name + " dimensions are zero or overflow"};
  }
  if (actual != expected) {
    return {ErrorCode::kInvalidArgument,
            name + " has " + std::to_string(actual) + " elements; expected " +
                std::to_string(expected)};
  }
  return Status::Ok();
}

Status output_pointer(const void* const output, const std::string& name) {
  if (output == nullptr) {
    return {ErrorCode::kInvalidArgument, name + " output pointer is null"};
  }
  return Status::Ok();
}

Status finite_values(const std::vector<float>& values, const std::string& name) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      return {ErrorCode::kInvalidArgument,
              name + " contains a non-finite value at index " + std::to_string(index)};
    }
  }
  return Status::Ok();
}

float gelu(const float value) {
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  return 0.5F * value * (1.0F + std::erf(value * kInverseSqrtTwo));
}

}  // namespace

Status rms_norm(const std::vector<float>& input,
                const std::size_t rows,
                const std::size_t width,
                const std::vector<float>& scale,
                const float epsilon,
                std::vector<float>* const output) {
  auto status = output_pointer(output, "rms_norm");
  if (!status.ok()) return status;
  status = matrix_size(rows, width, input.size(), "rms_norm input");
  if (!status.ok()) return status;
  status = matrix_size(1, width, scale.size(), "rms_norm scale");
  if (!status.ok()) return status;
  status = finite_values(input, "rms_norm input");
  if (!status.ok()) return status;
  status = finite_values(scale, "rms_norm scale");
  if (!status.ok()) return status;
  if (!std::isfinite(epsilon) || epsilon < 0.0F) {
    return {ErrorCode::kInvalidArgument, "rms_norm epsilon must be finite and nonnegative"};
  }

  output->resize(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    float sum_squares = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      const float value = input[row * width + column];
      sum_squares += value * value;
    }
    const float denominator = std::sqrt(sum_squares / static_cast<float>(width)) + epsilon;
    if (!std::isfinite(denominator) || denominator <= 0.0F) {
      return {ErrorCode::kInvalidArgument, "rms_norm encountered a nonpositive denominator"};
    }
    for (std::size_t column = 0; column < width; ++column) {
      (*output)[row * width + column] =
          input[row * width + column] / denominator * scale[column];
    }
  }
  return Status::Ok();
}

Status linear(const std::vector<float>& input,
              const std::size_t rows,
              const std::size_t in_features,
              const std::vector<float>& weight,
              const std::size_t out_features,
              const std::vector<float>* const bias,
              std::vector<float>* const output) {
  auto status = output_pointer(output, "linear");
  if (!status.ok()) return status;
  status = matrix_size(rows, in_features, input.size(), "linear input");
  if (!status.ok()) return status;
  status = matrix_size(out_features, in_features, weight.size(), "linear weight");
  if (!status.ok()) return status;
  status = finite_values(input, "linear input");
  if (!status.ok()) return status;
  status = finite_values(weight, "linear weight");
  if (!status.ok()) return status;
  if (bias != nullptr) {
    status = matrix_size(1, out_features, bias->size(), "linear bias");
    if (!status.ok()) return status;
    status = finite_values(*bias, "linear bias");
    if (!status.ok()) return status;
  }
  std::size_t output_size = 0;
  if (!product(rows, out_features, &output_size)) {
    return {ErrorCode::kInvalidArgument, "linear output dimensions overflow"};
  }
  output->assign(output_size, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < out_features; ++out) {
      float total = bias == nullptr ? 0.0F : (*bias)[out];
      for (std::size_t in = 0; in < in_features; ++in) {
        total += input[row * in_features + in] * weight[out * in_features + in];
      }
      if (!std::isfinite(total)) {
        return {ErrorCode::kInvalidArgument, "linear accumulation became non-finite"};
      }
      (*output)[row * out_features + out] = total;
    }
  }
  return Status::Ok();
}

Status gated_mlp(const std::vector<float>& input,
                 const std::size_t rows,
                 const std::size_t width,
                 const std::size_t inner_width,
                 const std::vector<float>& l1_weight,
                 const std::vector<float>& l2_weight,
                 const std::vector<float>& l3_weight,
                 const MlpActivation activation,
                 std::vector<float>* const output) {
  auto status = output_pointer(output, "gated_mlp");
  if (!status.ok()) return status;
  if (activation != MlpActivation::kGelu && activation != MlpActivation::kIdentity) {
    return {ErrorCode::kInvalidArgument, "gated_mlp activation is invalid"};
  }
  std::vector<float> first;
  std::vector<float> second;
  status = linear(input, rows, width, l1_weight, inner_width, nullptr, &first);
  if (!status.ok()) return status;
  status = linear(input, rows, width, l2_weight, inner_width, nullptr, &second);
  if (!status.ok()) return status;
  for (std::size_t index = 0; index < first.size(); ++index) {
    const float activated = activation == MlpActivation::kGelu ? gelu(first[index]) : first[index];
    first[index] = activated * second[index];
  }
  return linear(first, rows, inner_width, l3_weight, width, nullptr, output);
}

Status causal_depthwise_fir(const std::vector<float>& input,
                            const std::size_t length,
                            const std::size_t channels,
                            const std::vector<float>& weight,
                            const std::size_t kernel_size,
                            const std::vector<float>* const bias,
                            std::vector<float>* const output,
                            const FirOrientation orientation,
                            const FirBiasMode bias_mode) {
  auto status = output_pointer(output, "causal_depthwise_fir");
  if (!status.ok()) return status;
  if (orientation != FirOrientation::kCrossCorrelation &&
      orientation != FirOrientation::kCausalConvolution) {
    return {ErrorCode::kInvalidArgument, "causal_depthwise_fir orientation is invalid"};
  }
  if (bias_mode != FirBiasMode::kAdd && bias_mode != FirBiasMode::kMultiplyInput) {
    return {ErrorCode::kInvalidArgument, "causal_depthwise_fir bias mode is invalid"};
  }
  status = matrix_size(length, channels, input.size(), "causal_depthwise_fir input");
  if (!status.ok()) return status;
  status = matrix_size(channels, kernel_size, weight.size(), "causal_depthwise_fir weight");
  if (!status.ok()) return status;
  status = finite_values(input, "causal_depthwise_fir input");
  if (!status.ok()) return status;
  status = finite_values(weight, "causal_depthwise_fir weight");
  if (!status.ok()) return status;
  if (bias != nullptr) {
    status = matrix_size(1, channels, bias->size(), "causal_depthwise_fir bias");
    if (!status.ok()) return status;
    status = finite_values(*bias, "causal_depthwise_fir bias");
    if (!status.ok()) return status;
  }
  output->assign(input.size(), 0.0F);
  for (std::size_t time = 0; time < length; ++time) {
    for (std::size_t channel = 0; channel < channels; ++channel) {
      float total = bias == nullptr || bias_mode == FirBiasMode::kMultiplyInput
                        ? 0.0F
                        : (*bias)[channel];
      for (std::size_t tap = 0; tap < kernel_size; ++tap) {
        const std::size_t delay = orientation == FirOrientation::kCrossCorrelation
                                      ? kernel_size - 1 - tap
                                      : tap;
        if (time >= delay) {
          total += input[(time - delay) * channels + channel] *
                   weight[channel * kernel_size + tap];
        }
      }
      if (bias != nullptr && bias_mode == FirBiasMode::kMultiplyInput) {
        total += (*bias)[channel] * input[time * channels + channel];
      }
      if (!std::isfinite(total)) {
        return {ErrorCode::kInvalidArgument,
                "causal_depthwise_fir accumulation became non-finite"};
      }
      (*output)[time * channels + channel] = total;
    }
  }
  return Status::Ok();
}

Status split_hyena_projection(const std::vector<float>& projection,
                              const std::size_t length,
                              const std::size_t width,
                              std::vector<float>* const x2,
                              std::vector<float>* const x1,
                              std::vector<float>* const value) {
  auto status = output_pointer(x2, "split_hyena_projection x2");
  if (!status.ok()) return status;
  status = output_pointer(x1, "split_hyena_projection x1");
  if (!status.ok()) return status;
  status = output_pointer(value, "split_hyena_projection value");
  if (!status.ok()) return status;
  std::size_t triple_width = 0;
  if (!product(width, 3, &triple_width)) {
    return {ErrorCode::kInvalidArgument, "split_hyena_projection width overflows"};
  }
  status = matrix_size(length, triple_width, projection.size(), "split_hyena_projection input");
  if (!status.ok()) return status;
  status = finite_values(projection, "split_hyena_projection input");
  if (!status.ok()) return status;
  std::size_t output_size = 0;
  if (!product(length, width, &output_size)) {
    return {ErrorCode::kInvalidArgument, "split_hyena_projection output dimensions overflow"};
  }
  x2->resize(output_size);
  x1->resize(output_size);
  value->resize(output_size);
  for (std::size_t time = 0; time < length; ++time) {
    for (std::size_t channel = 0; channel < width; ++channel) {
      const auto source = time * triple_width + channel * 3;
      const auto destination = time * width + channel;
      (*x2)[destination] = projection[source];
      (*x1)[destination] = projection[source + 1];
      (*value)[destination] = projection[source + 2];
    }
  }
  return Status::Ok();
}

Status hcl_recurrence(const std::vector<float>& x2,
                      const std::vector<float>& x1,
                      const std::vector<float>& value,
                      const std::size_t length,
                      const std::size_t width,
                      const std::vector<float>& direct,
                      const std::vector<float>& log_poles,
                      const std::vector<float>& residues,
                      const std::size_t state_size,
                      std::vector<float>* const state,
                      std::vector<float>* const output) {
  auto status = output_pointer(state, "hcl_recurrence state");
  if (!status.ok()) return status;
  status = output_pointer(output, "hcl_recurrence");
  if (!status.ok()) return status;
  status = matrix_size(length, width, x2.size(), "hcl_recurrence x2");
  if (!status.ok()) return status;
  status = matrix_size(length, width, x1.size(), "hcl_recurrence x1");
  if (!status.ok()) return status;
  status = matrix_size(length, width, value.size(), "hcl_recurrence value");
  if (!status.ok()) return status;
  status = matrix_size(1, width, direct.size(), "hcl_recurrence direct");
  if (!status.ok()) return status;
  status = matrix_size(width, state_size, log_poles.size(), "hcl_recurrence log_poles");
  if (!status.ok()) return status;
  status = matrix_size(width, state_size, residues.size(), "hcl_recurrence residues");
  if (!status.ok()) return status;
  status = matrix_size(width, state_size, state->size(), "hcl_recurrence state");
  if (!status.ok()) return status;
  const std::array<const std::vector<float>*, 7> inputs{
      &x2, &x1, &value, &direct, &log_poles, &residues, state};
  for (const auto* tensor : inputs) {
    status = finite_values(*tensor, "hcl_recurrence input");
    if (!status.ok()) return status;
  }

  output->assign(x2.size(), 0.0F);
  for (std::size_t time = 0; time < length; ++time) {
    for (std::size_t channel = 0; channel < width; ++channel) {
      const auto data_index = time * width + channel;
      const float x1v = x1[data_index] * value[data_index];
      float modal = 0.0F;
      for (std::size_t state_index = 0; state_index < state_size; ++state_index) {
        const auto index = channel * state_size + state_index;
        const float pole = std::exp(log_poles[index]);
        (*state)[index] = pole * (*state)[index] + x1v;
        modal += residues[index] * (*state)[index];
      }
      (*output)[data_index] = x2[data_index] * (modal + direct[channel] * x1v);
      if (!std::isfinite((*output)[data_index])) {
        return {ErrorCode::kInvalidArgument, "hcl_recurrence produced a non-finite value"};
      }
    }
  }
  return Status::Ok();
}

Status apply_rope(std::vector<float>* const query,
                  std::vector<float>* const key,
                  const std::size_t length,
                  const std::size_t heads,
                  const std::size_t head_dim,
                  const std::vector<float>& inverse_frequency,
                  const std::size_t position_offset,
                  const float position_scale) {
  auto status = output_pointer(query, "apply_rope query");
  if (!status.ok()) return status;
  status = output_pointer(key, "apply_rope key");
  if (!status.ok()) return status;
  if (head_dim == 0 || head_dim % 2 != 0) {
    return {ErrorCode::kInvalidArgument, "apply_rope head_dim must be positive and even"};
  }
  if (!std::isfinite(position_scale) || position_scale <= 0.0F) {
    return {ErrorCode::kInvalidArgument, "apply_rope position_scale must be finite and positive"};
  }
  std::size_t rows = 0;
  if (!product(length, heads, &rows)) {
    return {ErrorCode::kInvalidArgument, "apply_rope dimensions overflow"};
  }
  status = matrix_size(rows, head_dim, query->size(), "apply_rope query");
  if (!status.ok()) return status;
  status = matrix_size(rows, head_dim, key->size(), "apply_rope key");
  if (!status.ok()) return status;
  status = matrix_size(1, head_dim / 2, inverse_frequency.size(), "apply_rope inverse_frequency");
  if (!status.ok()) return status;
  status = finite_values(*query, "apply_rope query");
  if (!status.ok()) return status;
  status = finite_values(*key, "apply_rope key");
  if (!status.ok()) return status;
  status = finite_values(inverse_frequency, "apply_rope inverse_frequency");
  if (!status.ok()) return status;
  if (length != 0 && position_offset > std::numeric_limits<std::size_t>::max() - (length - 1)) {
    return {ErrorCode::kInvalidArgument, "apply_rope position range overflows"};
  }

  const auto rotate = [&](std::vector<float>* const tensor) {
    const auto half = head_dim / 2;
    for (std::size_t time = 0; time < length; ++time) {
      const float position = static_cast<float>(position_offset + time) / position_scale;
      for (std::size_t head = 0; head < heads; ++head) {
        const auto base = (time * heads + head) * head_dim;
        for (std::size_t dimension = 0; dimension < half; ++dimension) {
          const float angle = position * inverse_frequency[dimension];
          if (!std::isfinite(angle)) {
            return false;
          }
          const float cosine = std::cos(angle);
          const float sine = std::sin(angle);
          const float first = (*tensor)[base + dimension];
          const float second = (*tensor)[base + half + dimension];
          (*tensor)[base + dimension] = first * cosine - second * sine;
          (*tensor)[base + half + dimension] = second * cosine + first * sine;
        }
      }
    }
    return true;
  };
  if (!rotate(query) || !rotate(key)) {
    return {ErrorCode::kInvalidArgument, "apply_rope angle became non-finite"};
  }
  return Status::Ok();
}

Status causal_attention(const std::vector<float>& query,
                        const std::vector<float>& key,
                        const std::vector<float>& value,
                        const std::size_t length,
                        const std::size_t heads,
                        const std::size_t head_dim,
                        std::vector<float>* const output) {
  auto status = output_pointer(output, "causal_attention");
  if (!status.ok()) return status;
  std::size_t rows = 0;
  if (!product(length, heads, &rows)) {
    return {ErrorCode::kInvalidArgument, "causal_attention dimensions overflow"};
  }
  status = matrix_size(rows, head_dim, query.size(), "causal_attention query");
  if (!status.ok()) return status;
  status = matrix_size(rows, head_dim, key.size(), "causal_attention key");
  if (!status.ok()) return status;
  status = matrix_size(rows, head_dim, value.size(), "causal_attention value");
  if (!status.ok()) return status;
  status = finite_values(query, "causal_attention query");
  if (!status.ok()) return status;
  status = finite_values(key, "causal_attention key");
  if (!status.ok()) return status;
  status = finite_values(value, "causal_attention value");
  if (!status.ok()) return status;
  output->assign(query.size(), 0.0F);
  std::vector<float> scores(length, 0.0F);
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  for (std::size_t time = 0; time < length; ++time) {
    for (std::size_t head = 0; head < heads; ++head) {
      float maximum = -std::numeric_limits<float>::infinity();
      const auto query_base = (time * heads + head) * head_dim;
      for (std::size_t source = 0; source <= time; ++source) {
        const auto key_base = (source * heads + head) * head_dim;
        float dot = 0.0F;
        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
          dot += query[query_base + dimension] * key[key_base + dimension];
        }
        scores[source] = dot * scale;
        if (!std::isfinite(scores[source])) {
          return {ErrorCode::kInvalidArgument, "causal_attention score became non-finite"};
        }
        maximum = std::max(maximum, scores[source]);
      }
      float denominator = 0.0F;
      for (std::size_t source = 0; source <= time; ++source) {
        scores[source] = std::exp(scores[source] - maximum);
        denominator += scores[source];
      }
      if (!std::isfinite(denominator) || denominator <= 0.0F) {
        return {ErrorCode::kInvalidArgument, "causal_attention softmax denominator is invalid"};
      }
      for (std::size_t source = 0; source <= time; ++source) {
        const float probability = scores[source] / denominator;
        const auto value_base = (source * heads + head) * head_dim;
        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
          (*output)[query_base + dimension] += probability * value[value_base + dimension];
        }
      }
    }
  }
  return Status::Ok();
}

}  // namespace evo2c::cpu
