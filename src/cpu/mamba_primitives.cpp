// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/mamba_primitives.hpp"

#include "../linear_executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace evo::cpu {
namespace {

bool checked_multiply(const std::size_t left, const std::size_t right,
                      std::size_t *const output) noexcept {
  if (output == nullptr ||
      (left != 0U && right > std::numeric_limits<std::size_t>::max() / left))
    return false;
  *output = left * right;
  return true;
}

Status invalid(const std::string &message) {
  return {ErrorCode::kInvalidArgument, "Mamba primitive: " + message};
}

Status format_error(const std::string &message) {
  return {ErrorCode::kModelFormat, "Mamba primitive: " + message};
}

float tensor_value(const MambaTensorView &tensor,
                   const std::size_t index) noexcept {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return value;
}

Status require_tensor(const MambaTensorView &tensor,
                      const std::vector<std::size_t> &shape,
                      const std::string &name) {
  if (tensor.dtype != TensorDType::kF32)
    return format_error(name + " must use F32");
  if (tensor.shape != shape)
    return format_error(name + " shape differs from the frozen contract");
  std::size_t elements = 1U;
  for (const auto dimension : shape) {
    if (dimension == 0U || !checked_multiply(elements, dimension, &elements))
      return format_error(name + " shape is zero or overflows");
  }
  std::size_t bytes = 0U;
  if (!checked_multiply(elements, sizeof(float), &bytes) ||
      tensor.data == nullptr || tensor.bytes != bytes)
    return format_error(name + " payload extent differs");
  return Status::Ok();
}

Status require_matrix(const std::vector<float> &values, const std::size_t rows,
                      const std::size_t width, const std::string &name) {
  std::size_t expected = 0U;
  if (rows == 0U || width == 0U || !checked_multiply(rows, width, &expected) ||
      values.size() != expected)
    return invalid(name + " dimensions differ");
  return Status::Ok();
}

bool finite_values(const std::vector<float> &values) noexcept {
  return std::all_of(values.begin(), values.end(),
                     [](const float value) { return std::isfinite(value); });
}

float silu(const float value) noexcept {
  if (value >= 0.0F)
    return value / (1.0F + std::exp(-value));
  const float exponential = std::exp(value);
  return value * exponential / (1.0F + exponential);
}

float softplus(const float value) noexcept {
  if (value > 20.0F)
    return value;
  if (value < -20.0F)
    return std::exp(value);
  return std::log1p(std::exp(value));
}

Status linear_impl(const std::vector<float> &input, const std::size_t rows,
                   const std::size_t input_width, const MambaTensorView &weight,
                   const std::size_t output_width,
                   evo::detail::LinearExecutor *const executor,
                   std::vector<float> *const output) {
  auto status = require_matrix(input, rows, input_width, "linear input");
  if (!status.ok())
    return status;
  status = require_tensor(weight, {output_width, input_width}, "linear weight");
  if (!status.ok())
    return status;
  if (output == nullptr)
    return invalid("linear output is null");
  if (executor != nullptr) {
    const evo::detail::LinearTensorView view{weight.data, weight.dtype,
                                             output_width * input_width};
    status = executor->linear(input.data(), rows, input_width, view,
                              output_width, nullptr, output);
    if (!status.ok())
      return status;
    if (output->size() != rows * output_width || !finite_values(*output))
      return invalid("linear executor returned invalid values");
    return Status::Ok();
  }
  output->assign(rows * output_width, 0.0F);
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t target = 0U; target < output_width; ++target) {
      float total = 0.0F;
      for (std::size_t source = 0U; source < input_width; ++source) {
        total += input[row * input_width + source] *
                 tensor_value(weight, target * input_width + source);
      }
      (*output)[row * output_width + target] = total;
    }
  }
  return finite_values(*output) ? Status::Ok()
                                : invalid("linear result became non-finite");
}

Status validate_mamba1_config(const Mamba1Config &config) {
  if (config.model_width == 0U || config.inner_width == 0U ||
      config.state_width == 0U || config.convolution_width == 0U ||
      config.time_step_rank == 0U ||
      !std::isfinite(config.parameter_projection_norm_epsilon) ||
      config.parameter_projection_norm_epsilon <= 0.0F)
    return invalid("Mamba1 geometry contains zero");
  return Status::Ok();
}

bool empty_tensor(const MambaTensorView &tensor) noexcept {
  return tensor.data == nullptr && tensor.bytes == 0U && tensor.shape.empty();
}

Status validate_mamba2_config(const Mamba2Config &config) {
  if (config.model_width == 0U || config.inner_width == 0U ||
      config.state_width == 0U || config.convolution_width == 0U ||
      config.head_width == 0U || config.heads == 0U || config.groups == 0U ||
      config.inner_width != config.head_width * config.heads ||
      config.heads % config.groups != 0U ||
      !std::isfinite(config.norm_epsilon) || config.norm_epsilon <= 0.0F)
    return invalid("Mamba2 geometry differs from the frozen contract");
  return Status::Ok();
}

} // namespace

Status mamba_linear_f32(const std::vector<float> &input, const std::size_t rows,
                        const std::size_t input_width,
                        const MambaTensorView &weight,
                        const std::size_t output_width,
                        evo::detail::LinearExecutor *const linear_executor,
                        std::vector<float> *const output) {
  return linear_impl(input, rows, input_width, weight, output_width,
                     linear_executor, output);
}

Status mamba_causal_depthwise_conv_silu(const std::vector<float> &input,
                                        const std::size_t rows,
                                        const std::size_t channels,
                                        const std::size_t kernel_width,
                                        const MambaTensorView &weight,
                                        const MambaTensorView *const bias,
                                        std::vector<float> *const output) {
  auto status = require_matrix(input, rows, channels, "convolution input");
  if (!status.ok())
    return status;
  status = require_tensor(weight, {channels, 1U, kernel_width},
                          "convolution weight");
  if (!status.ok())
    return status;
  if (bias != nullptr) {
    status = require_tensor(*bias, {channels}, "convolution bias");
    if (!status.ok())
      return status;
  }
  if (output == nullptr)
    return invalid("convolution output is null");
  output->assign(rows * channels, 0.0F);
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t channel = 0U; channel < channels; ++channel) {
      float total = bias == nullptr ? 0.0F : tensor_value(*bias, channel);
      for (std::size_t tap = 0U; tap < kernel_width; ++tap) {
        const std::size_t lag = kernel_width - 1U - tap;
        if (row >= lag) {
          total += input[(row - lag) * channels + channel] *
                   tensor_value(weight, channel * kernel_width + tap);
        }
      }
      (*output)[row * channels + channel] = silu(total);
    }
  }
  return finite_values(*output) ? Status::Ok()
                                : invalid("convolution became non-finite");
}

Status mamba_rms_norm(const std::vector<float> &input, const std::size_t rows,
                      const std::size_t width, const MambaTensorView &scale,
                      const float epsilon, std::vector<float> *const output) {
  auto status = require_matrix(input, rows, width, "RMSNorm input");
  if (!status.ok())
    return status;
  status = require_tensor(scale, {width}, "RMSNorm scale");
  if (!status.ok())
    return status;
  if (output == nullptr || !std::isfinite(epsilon) || epsilon <= 0.0F)
    return invalid("RMSNorm output/epsilon is invalid");
  output->resize(input.size());
  for (std::size_t row = 0U; row < rows; ++row) {
    float square_sum = 0.0F;
    for (std::size_t column = 0U; column < width; ++column) {
      const float value = input[row * width + column];
      square_sum += value * value;
    }
    const float inverse =
        1.0F / std::sqrt(square_sum / static_cast<float>(width) + epsilon);
    for (std::size_t column = 0U; column < width; ++column) {
      (*output)[row * width + column] =
          input[row * width + column] * inverse * tensor_value(scale, column);
    }
  }
  return finite_values(*output) ? Status::Ok()
                                : invalid("RMSNorm became non-finite");
}

Status mamba_group_rms_norm_gated(
    const std::vector<float> &input, const std::vector<float> &gate,
    const std::size_t rows, const std::size_t width,
    const std::size_t group_width, const MambaTensorView &scale,
    const float epsilon, const bool norm_before_gate,
    std::vector<float> *const output) {
  auto status = require_matrix(input, rows, width, "gated RMSNorm input");
  if (!status.ok())
    return status;
  status = require_matrix(gate, rows, width, "gated RMSNorm gate");
  if (!status.ok())
    return status;
  status = require_tensor(scale, {width}, "gated RMSNorm scale");
  if (!status.ok())
    return status;
  if (output == nullptr || group_width == 0U || width % group_width != 0U ||
      !std::isfinite(epsilon) || epsilon <= 0.0F)
    return invalid("gated RMSNorm geometry/epsilon is invalid");
  output->resize(input.size());
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t begin = 0U; begin < width; begin += group_width) {
      float square_sum = 0.0F;
      for (std::size_t offset = 0U; offset < group_width; ++offset) {
        const std::size_t index = row * width + begin + offset;
        float value = input[index];
        if (!norm_before_gate)
          value *= silu(gate[index]);
        square_sum += value * value;
      }
      const float inverse =
          1.0F /
          std::sqrt(square_sum / static_cast<float>(group_width) + epsilon);
      for (std::size_t offset = 0U; offset < group_width; ++offset) {
        const std::size_t column = begin + offset;
        const std::size_t index = row * width + column;
        float value = input[index];
        if (!norm_before_gate)
          value *= silu(gate[index]);
        value *= inverse * tensor_value(scale, column);
        if (norm_before_gate)
          value *= silu(gate[index]);
        (*output)[index] = value;
      }
    }
  }
  return finite_values(*output) ? Status::Ok()
                                : invalid("gated RMSNorm became non-finite");
}

Status mamba1_selective_scan_f32(
    const std::vector<float> &x, const std::vector<float> &raw_time_step,
    const std::vector<float> &b, const std::vector<float> &c,
    const std::vector<float> &gate, const std::size_t rows,
    const Mamba1Config &config, const MambaTensorView &time_step_bias,
    const MambaTensorView &a_log, const MambaTensorView &skip,
    std::vector<float> *const output) {
  auto status = validate_mamba1_config(config);
  if (!status.ok())
    return status;
  for (const auto *const matrix : {&x, &raw_time_step, &gate}) {
    status = require_matrix(*matrix, rows, config.inner_width,
                            "Mamba1 scan channel input");
    if (!status.ok())
      return status;
  }
  status = require_matrix(b, rows, config.state_width, "Mamba1 scan B");
  if (!status.ok())
    return status;
  status = require_matrix(c, rows, config.state_width, "Mamba1 scan C");
  if (!status.ok())
    return status;
  status = require_tensor(time_step_bias, {config.inner_width},
                          "Mamba1 time-step bias");
  if (!status.ok())
    return status;
  status = require_tensor(a_log, {config.inner_width, config.state_width},
                          "Mamba1 A_log");
  if (!status.ok())
    return status;
  status = require_tensor(skip, {config.inner_width}, "Mamba1 skip");
  if (!status.ok())
    return status;
  if (output == nullptr)
    return invalid("Mamba1 scan output is null");
  std::vector<float> state(config.inner_width * config.state_width, 0.0F);
  output->assign(rows * config.inner_width, 0.0F);
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t channel = 0U; channel < config.inner_width; ++channel) {
      const float delta =
          softplus(raw_time_step[row * config.inner_width + channel] +
                   tensor_value(time_step_bias, channel));
      const float input_value = x[row * config.inner_width + channel];
      float total = 0.0F;
      for (std::size_t state_index = 0U; state_index < config.state_width;
           ++state_index) {
        const std::size_t parameter =
            channel * config.state_width + state_index;
        const float a = -std::exp(tensor_value(a_log, parameter));
        float &cell = state[parameter];
        cell = std::exp(delta * a) * cell +
               delta * b[row * config.state_width + state_index] * input_value;
        total += cell * c[row * config.state_width + state_index];
      }
      total += tensor_value(skip, channel) * input_value;
      (*output)[row * config.inner_width + channel] =
          total * silu(gate[row * config.inner_width + channel]);
    }
  }
  return finite_values(*output) ? Status::Ok()
                                : invalid("Mamba1 scan became non-finite");
}

Status mamba1_mixer_f32(const std::vector<float> &input, const std::size_t rows,
                        const Mamba1Config &config,
                        const Mamba1Weights &weights,
                        evo::detail::LinearExecutor *const linear_executor,
                        std::vector<float> *const output) {
  auto status = validate_mamba1_config(config);
  if (!status.ok())
    return status;
  status =
      require_matrix(input, rows, config.model_width, "Mamba1 mixer input");
  if (!status.ok())
    return status;
  std::vector<float> projected;
  status =
      linear_impl(input, rows, config.model_width, weights.input_projection,
                  config.inner_width * 2U, linear_executor, &projected);
  if (!status.ok())
    return status;
  std::vector<float> x(rows * config.inner_width);
  std::vector<float> gate(rows * config.inner_width);
  for (std::size_t row = 0U; row < rows; ++row) {
    std::copy_n(projected.begin() +
                    static_cast<std::ptrdiff_t>(row * config.inner_width * 2U),
                config.inner_width,
                x.begin() +
                    static_cast<std::ptrdiff_t>(row * config.inner_width));
    std::copy_n(projected.begin() +
                    static_cast<std::ptrdiff_t>(row * config.inner_width * 2U +
                                                config.inner_width),
                config.inner_width,
                gate.begin() +
                    static_cast<std::ptrdiff_t>(row * config.inner_width));
  }
  std::vector<float> convolved;
  status = mamba_causal_depthwise_conv_silu(
      x, rows, config.inner_width, config.convolution_width,
      weights.convolution_weight, &weights.convolution_bias, &convolved);
  if (!status.ok())
    return status;
  std::vector<float> parameters;
  status =
      linear_impl(convolved, rows, config.inner_width, weights.x_projection,
                  config.time_step_rank + config.state_width * 2U,
                  linear_executor, &parameters);
  if (!status.ok())
    return status;
  std::vector<float> rank_time_step(rows * config.time_step_rank);
  std::vector<float> b(rows * config.state_width);
  std::vector<float> c(rows * config.state_width);
  const std::size_t parameter_width =
      config.time_step_rank + config.state_width * 2U;
  for (std::size_t row = 0U; row < rows; ++row) {
    const auto source =
        parameters.begin() + static_cast<std::ptrdiff_t>(row * parameter_width);
    std::copy_n(source, config.time_step_rank,
                rank_time_step.begin() +
                    static_cast<std::ptrdiff_t>(row * config.time_step_rank));
    std::copy_n(source + static_cast<std::ptrdiff_t>(config.time_step_rank),
                config.state_width,
                b.begin() +
                    static_cast<std::ptrdiff_t>(row * config.state_width));
    std::copy_n(source + static_cast<std::ptrdiff_t>(config.time_step_rank +
                                                     config.state_width),
                config.state_width,
                c.begin() +
                    static_cast<std::ptrdiff_t>(row * config.state_width));
  }
  if (config.parameter_projection_rms_norm) {
    std::vector<float> normalized;
    status =
        mamba_rms_norm(rank_time_step, rows, config.time_step_rank,
                       weights.projected_time_step_norm_scale,
                       config.parameter_projection_norm_epsilon, &normalized);
    if (!status.ok())
      return status;
    rank_time_step = std::move(normalized);
    status = mamba_rms_norm(
        b, rows, config.state_width, weights.projected_b_norm_scale,
        config.parameter_projection_norm_epsilon, &normalized);
    if (!status.ok())
      return status;
    b = std::move(normalized);
    status = mamba_rms_norm(
        c, rows, config.state_width, weights.projected_c_norm_scale,
        config.parameter_projection_norm_epsilon, &normalized);
    if (!status.ok())
      return status;
    c = std::move(normalized);
  } else if (!empty_tensor(weights.projected_time_step_norm_scale) ||
             !empty_tensor(weights.projected_b_norm_scale) ||
             !empty_tensor(weights.projected_c_norm_scale)) {
    return format_error(
        "Mamba1 projected-parameter norm scales require the config flag");
  }
  std::vector<float> raw_time_step;
  status = linear_impl(rank_time_step, rows, config.time_step_rank,
                       weights.time_step_projection, config.inner_width,
                       linear_executor, &raw_time_step);
  if (!status.ok())
    return status;
  std::vector<float> scanned;
  status = mamba1_selective_scan_f32(convolved, raw_time_step, b, c, gate, rows,
                                     config, weights.time_step_bias,
                                     weights.a_log, weights.skip, &scanned);
  if (!status.ok())
    return status;
  return linear_impl(scanned, rows, config.inner_width,
                     weights.output_projection, config.model_width,
                     linear_executor, output);
}

Status mamba2_selective_scan_f32(
    const std::vector<float> &x, const std::vector<float> &b,
    const std::vector<float> &c, const std::vector<float> &raw_time_step,
    const std::vector<float> &gate, const std::size_t rows,
    const Mamba2Config &config, const MambaTensorView &time_step_bias,
    const MambaTensorView &a_log, const MambaTensorView &skip,
    const MambaTensorView *const norm_scale, std::vector<float> *const output) {
  auto status = validate_mamba2_config(config);
  if (!status.ok())
    return status;
  status = require_matrix(x, rows, config.inner_width, "Mamba2 scan x");
  if (!status.ok())
    return status;
  status = require_matrix(gate, rows, config.inner_width, "Mamba2 scan gate");
  if (!status.ok())
    return status;
  status = require_matrix(b, rows, config.groups * config.state_width,
                          "Mamba2 scan B");
  if (!status.ok())
    return status;
  status = require_matrix(c, rows, config.groups * config.state_width,
                          "Mamba2 scan C");
  if (!status.ok())
    return status;
  status =
      require_matrix(raw_time_step, rows, config.heads, "Mamba2 raw time-step");
  if (!status.ok())
    return status;
  status =
      require_tensor(time_step_bias, {config.heads}, "Mamba2 time-step bias");
  if (!status.ok())
    return status;
  status = require_tensor(a_log, {config.heads}, "Mamba2 A_log");
  if (!status.ok())
    return status;
  status = require_tensor(
      skip,
      {config.skip_has_inner_dimension ? config.inner_width : config.heads},
      "Mamba2 skip");
  if (!status.ok())
    return status;
  if (config.rms_norm) {
    if (norm_scale == nullptr)
      return format_error("Mamba2 gated norm scale is missing");
    status = require_tensor(*norm_scale, {config.inner_width},
                            "Mamba2 gated norm scale");
    if (!status.ok())
      return status;
  }
  if (output == nullptr)
    return invalid("Mamba2 scan output is null");
  std::vector<float> state(config.inner_width * config.state_width, 0.0F);
  std::vector<float> scanned(rows * config.inner_width, 0.0F);
  const std::size_t heads_per_group = config.heads / config.groups;
  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t head = 0U; head < config.heads; ++head) {
      const float delta = softplus(raw_time_step[row * config.heads + head] +
                                   tensor_value(time_step_bias, head));
      const float a = -std::exp(tensor_value(a_log, head));
      const float decay = std::exp(delta * a);
      const std::size_t group = head / heads_per_group;
      for (std::size_t lane = 0U; lane < config.head_width; ++lane) {
        const std::size_t channel = head * config.head_width + lane;
        const float input_value = x[row * config.inner_width + channel];
        float total = 0.0F;
        for (std::size_t state_index = 0U; state_index < config.state_width;
             ++state_index) {
          const std::size_t cell_index =
              channel * config.state_width + state_index;
          float &cell = state[cell_index];
          cell =
              decay * cell + delta *
                                 b[row * config.groups * config.state_width +
                                   group * config.state_width + state_index] *
                                 input_value;
          total += cell * c[row * config.groups * config.state_width +
                            group * config.state_width + state_index];
        }
        const std::size_t skip_index =
            config.skip_has_inner_dimension ? channel : head;
        scanned[row * config.inner_width + channel] =
            total + tensor_value(skip, skip_index) * input_value;
      }
    }
  }
  if (!finite_values(scanned))
    return invalid("Mamba2 scan became non-finite");
  if (config.rms_norm) {
    return mamba_group_rms_norm_gated(scanned, gate, rows, config.inner_width,
                                      config.inner_width / config.groups,
                                      *norm_scale, config.norm_epsilon,
                                      config.norm_before_gate, output);
  }
  output->resize(scanned.size());
  for (std::size_t index = 0U; index < scanned.size(); ++index)
    (*output)[index] = scanned[index] * silu(gate[index]);
  return finite_values(*output) ? Status::Ok()
                                : invalid("Mamba2 gate became non-finite");
}

Status mamba2_mixer_f32(const std::vector<float> &input, const std::size_t rows,
                        const Mamba2Config &config,
                        const Mamba2Weights &weights,
                        evo::detail::LinearExecutor *const linear_executor,
                        std::vector<float> *const output) {
  auto status = validate_mamba2_config(config);
  if (!status.ok())
    return status;
  status =
      require_matrix(input, rows, config.model_width, "Mamba2 mixer input");
  if (!status.ok())
    return status;
  const std::size_t grouped_state = config.groups * config.state_width;
  const std::size_t projection_width =
      config.inner_width * 2U + grouped_state * 2U + config.heads;
  std::vector<float> projected;
  status =
      linear_impl(input, rows, config.model_width, weights.input_projection,
                  projection_width, linear_executor, &projected);
  if (!status.ok())
    return status;
  std::vector<float> gate(rows * config.inner_width);
  std::vector<float> convolution_input(
      rows * (config.inner_width + grouped_state * 2U));
  std::vector<float> raw_time_step(rows * config.heads);
  const std::size_t convolution_channels =
      config.inner_width + grouped_state * 2U;
  for (std::size_t row = 0U; row < rows; ++row) {
    const auto source =
        projected.begin() + static_cast<std::ptrdiff_t>(row * projection_width);
    std::copy_n(source, config.inner_width,
                gate.begin() +
                    static_cast<std::ptrdiff_t>(row * config.inner_width));
    std::copy_n(source + static_cast<std::ptrdiff_t>(config.inner_width),
                convolution_channels,
                convolution_input.begin() +
                    static_cast<std::ptrdiff_t>(row * convolution_channels));
    std::copy_n(source + static_cast<std::ptrdiff_t>(config.inner_width +
                                                     convolution_channels),
                config.heads,
                raw_time_step.begin() +
                    static_cast<std::ptrdiff_t>(row * config.heads));
  }
  std::vector<float> convolved;
  status = mamba_causal_depthwise_conv_silu(
      convolution_input, rows, convolution_channels, config.convolution_width,
      weights.convolution_weight, &weights.convolution_bias, &convolved);
  if (!status.ok())
    return status;
  std::vector<float> x(rows * config.inner_width);
  std::vector<float> b(rows * grouped_state);
  std::vector<float> c(rows * grouped_state);
  for (std::size_t row = 0U; row < rows; ++row) {
    const auto source = convolved.begin() +
                        static_cast<std::ptrdiff_t>(row * convolution_channels);
    std::copy_n(source, config.inner_width,
                x.begin() +
                    static_cast<std::ptrdiff_t>(row * config.inner_width));
    std::copy_n(source + static_cast<std::ptrdiff_t>(config.inner_width),
                grouped_state,
                b.begin() + static_cast<std::ptrdiff_t>(row * grouped_state));
    std::copy_n(source + static_cast<std::ptrdiff_t>(config.inner_width +
                                                     grouped_state),
                grouped_state,
                c.begin() + static_cast<std::ptrdiff_t>(row * grouped_state));
  }
  std::vector<float> scanned;
  status = mamba2_selective_scan_f32(
      x, b, c, raw_time_step, gate, rows, config, weights.time_step_bias,
      weights.a_log, weights.skip,
      config.rms_norm ? &weights.norm_scale : nullptr, &scanned);
  if (!status.ok())
    return status;
  return linear_impl(scanned, rows, config.inner_width,
                     weights.output_projection, config.model_width,
                     linear_executor, output);
}

} // namespace evo::cpu
