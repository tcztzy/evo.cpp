// SPDX-License-Identifier: Apache-2.0
#include "evo/cuda/esmc.hpp"

#include "evo/cuda/runtime.hpp"
#include "evo/model_registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace evo::cuda {
namespace {

constexpr int kThreads = 256;
constexpr std::size_t kWeightAlignment = 256;

std::uint64_t read_u64(const std::uint8_t *const data) noexcept {
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<std::uint64_t>(data[byte]) << (byte * 8U);
  return value;
}

Status metadata_entry(const ModelFile &artifact, const std::string_view key,
                      const MetadataType type,
                      const MetadataEntry **const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr)
    return {ErrorCode::kModelFormat,
            "required ESMC metadata is missing: " + std::string{key}};
  if (entry->type != type)
    return {ErrorCode::kModelFormat,
            "ESMC metadata has wrong type: " + std::string{key}};
  *output = entry;
  return Status::Ok();
}

Status metadata_size(const ModelFile &artifact, const std::string_view key,
                     std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  const std::uint64_t value = read_u64(entry->value.data());
  if (value > std::numeric_limits<std::size_t>::max())
    return {ErrorCode::kModelFormat,
            "ESMC metadata exceeds size_t: " + std::string{key}};
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

Status metadata_float(const ModelFile &artifact, const std::string_view key,
                      float *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kF64, &entry);
  if (!status.ok())
    return status;
  const std::uint64_t bits = read_u64(entry->value.data());
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  if (!std::isfinite(value) || value > std::numeric_limits<float>::max() ||
      value < -std::numeric_limits<float>::max())
    return {ErrorCode::kModelFormat,
            "ESMC metadata is not finite F32: " + std::string{key}};
  *output = static_cast<float>(value);
  return Status::Ok();
}

Status metadata_string(const ModelFile &artifact, const std::string_view key,
                       std::string *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kString, &entry);
  if (!status.ok())
    return status;
  output->assign(entry->value.begin(), entry->value.end());
  return Status::Ok();
}

Status metadata_bool(const ModelFile &artifact, const std::string_view key,
                     bool *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kBool, &entry);
  if (!status.ok())
    return status;
  *output = entry->value[0] != 0;
  return Status::Ok();
}

bool multiply(const std::size_t left, const std::size_t right,
              std::size_t *const output) noexcept {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *output = left * right;
  return true;
}

Status element_bytes(const std::size_t elements, std::size_t *const bytes) {
  if (!multiply(elements, sizeof(float), bytes))
    return {ErrorCode::kModelFormat, "ESMC F32 byte count overflows size_t"};
  return Status::Ok();
}

Status allocate_floats(DeviceBuffer *const buffer, const int device,
                       const std::size_t elements) {
  std::size_t bytes = 0;
  auto status = element_bytes(elements, &bytes);
  return status.ok() ? buffer->allocate(device, bytes) : status;
}

unsigned blocks_for(const std::size_t elements) noexcept {
  return static_cast<unsigned>(
      (elements + static_cast<std::size_t>(kThreads) - 1U) /
      static_cast<std::size_t>(kThreads));
}

__global__ void embedding_kernel(const TokenId *const tokens,
                                 const float *const weights,
                                 const std::size_t rows,
                                 const std::size_t width, float *const output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < rows * width) {
    const std::size_t row = index / width;
    const std::size_t column = index - row * width;
    output[index] =
        weights[static_cast<std::size_t>(tokens[row]) * width + column];
  }
}

__global__ void layer_norm_kernel(const float *const input,
                                  const float *const weight,
                                  const float *const bias,
                                  const std::size_t width, const float epsilon,
                                  float *const output) {
  __shared__ float sum[kThreads];
  const std::size_t row = blockIdx.x;
  float local_sum = 0.0F;
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    local_sum += input[row * width + column];
  }
  sum[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride != 0U; stride /= 2U) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  const float mean = sum[0] / static_cast<float>(width);
  float local_variance = 0.0F;
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    const float centered = input[row * width + column] - mean;
    local_variance += centered * centered;
  }
  sum[threadIdx.x] = local_variance;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride != 0U; stride /= 2U) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  const float variance = sum[0] / static_cast<float>(width);
  const float inverse = rsqrtf(variance + epsilon);
  for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
    float value =
        (input[row * width + column] - mean) * inverse * weight[column];
    if (bias != nullptr)
      value += bias[column];
    output[row * width + column] = value;
  }
}

__global__ void add_bias_kernel(float *const output, const float *const bias,
                                const std::size_t elements,
                                const std::size_t width) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements)
    output[index] += bias[index % width];
}

__global__ void split_qkv_kernel(const float *const qkv, const std::size_t rows,
                                 const std::size_t width, float *const query,
                                 float *const key, float *const value) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < rows * width) {
    const std::size_t row = index / width;
    const std::size_t column = index - row * width;
    const std::size_t base = row * width * 3U + column;
    query[index] = qkv[base];
    key[index] = qkv[base + width];
    value[index] = qkv[base + width * 2U];
  }
}

__global__ void rope_kernel(float *const query, float *const key,
                            const std::size_t rows, const std::size_t width,
                            const std::size_t heads,
                            const std::size_t head_width,
                            const float rope_base) {
  const std::size_t half = head_width / 2U;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * heads * half)
    return;
  const std::size_t pair = index % half;
  const std::size_t head_row = index / half;
  const std::size_t head = head_row % heads;
  const std::size_t row = head_row / heads;
  const std::size_t base = row * width + head * head_width;
  const float exponent =
      static_cast<float>(pair * 2U) / static_cast<float>(head_width);
  const float angle = static_cast<float>(row) / powf(rope_base, exponent);
  float sine = 0.0F;
  float cosine = 0.0F;
  sincosf(angle, &sine, &cosine);
  const auto rotate = [&](float *const tensor) {
    const float first = tensor[base + pair];
    const float second = tensor[base + half + pair];
    tensor[base + pair] = first * cosine - second * sine;
    tensor[base + half + pair] = second * cosine + first * sine;
  };
  rotate(query);
  rotate(key);
}

__global__ void softmax_kernel(float *const scores, const TokenId *const tokens,
                               const std::size_t rows) {
  __shared__ float scratch[kThreads];
  const std::size_t target = blockIdx.x;
  const bool target_padding = tokens[target] == 1;
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t source = threadIdx.x; source < rows; source += blockDim.x) {
    const bool visible = target_padding == (tokens[source] == 1);
    maximum = fmaxf(maximum, visible ? scores[target * rows + source]
                                     : -std::numeric_limits<float>::infinity());
  }
  scratch[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride != 0U; stride /= 2U) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] =
          fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = scratch[0];
  float denominator = 0.0F;
  for (std::size_t source = threadIdx.x; source < rows; source += blockDim.x) {
    const bool visible = target_padding == (tokens[source] == 1);
    const float probability =
        visible ? expf(scores[target * rows + source] - maximum) : 0.0F;
    scores[target * rows + source] = probability;
    denominator += probability;
  }
  scratch[threadIdx.x] = denominator;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2U; stride != 0U; stride /= 2U) {
    if (threadIdx.x < stride)
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  denominator = scratch[0];
  for (std::size_t source = threadIdx.x; source < rows; source += blockDim.x)
    scores[target * rows + source] /= denominator;
}

__global__ void residual_kernel(float *const residual,
                                const float *const update,
                                const std::size_t elements, const float scale) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements)
    residual[index] += update[index] / scale;
}

__global__ void swiglu_kernel(const float *const projected,
                              const std::size_t rows,
                              const std::size_t inner_width,
                              float *const gated) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < rows * inner_width) {
    const std::size_t row = index / inner_width;
    const std::size_t column = index - row * inner_width;
    const float first = projected[row * inner_width * 2U + column];
    const float second =
        projected[row * inner_width * 2U + inner_width + column];
    gated[index] = first / (1.0F + expf(-first)) * second;
  }
}

__global__ void gelu_kernel(float *const values, const std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = values[index];
    values[index] =
        0.5F * value * (1.0F + erff(value * 0.7071067811865475244F));
  }
}

Status kernel_status(const char *const operation) {
  return cuda_status(cudaPeekAtLastError(), operation);
}

struct DeviceTensor final {
  const float *data{nullptr};
  std::size_t elements{0};
};

struct PendingTensor final {
  const void *host{nullptr};
  std::size_t bytes{0};
  std::size_t offset{0};
  DeviceTensor *target{nullptr};
};

Status schedule_tensor(const ModelFile &artifact, const std::string &name,
                       const std::vector<std::size_t> &shape,
                       std::vector<PendingTensor> *const pending,
                       std::size_t *const cursor, DeviceTensor *const target) {
  const auto *const tensor = artifact.find_tensor(name);
  if (tensor == nullptr)
    return {ErrorCode::kModelFormat,
            "required ESMC tensor is missing: " + name};
  if (tensor->dtype != TensorDType::kF32 || tensor->rank != shape.size())
    return {ErrorCode::kModelFormat,
            "ESMC tensor dtype/rank mismatch: " + name};
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (tensor->dimensions[index] != shape[index])
      return {ErrorCode::kModelFormat, "ESMC tensor shape mismatch: " + name};
  }
  const auto *const host = artifact.tensor_data(*tensor);
  if (host == nullptr ||
      tensor->element_count > std::numeric_limits<std::size_t>::max())
    return {ErrorCode::kModelFormat,
            "ESMC tensor payload is unavailable: " + name};
  if (*cursor >
      std::numeric_limits<std::size_t>::max() - (kWeightAlignment - 1U))
    return {ErrorCode::kModelFormat, "ESMC weight arena alignment overflows"};
  *cursor = (*cursor + kWeightAlignment - 1U) & ~(kWeightAlignment - 1U);
  if (tensor->data_size > std::numeric_limits<std::size_t>::max() - *cursor)
    return {ErrorCode::kModelFormat, "ESMC weight arena size overflows"};
  pending->push_back(
      {host, static_cast<std::size_t>(tensor->data_size), *cursor, target});
  *cursor += static_cast<std::size_t>(tensor->data_size);
  return Status::Ok();
}

} // namespace

struct EsmcModel::Impl final {
  struct Layer final {
    DeviceTensor qkv_norm_weight;
    DeviceTensor qkv_norm_bias;
    DeviceTensor qkv_weight;
    DeviceTensor q_norm_weight;
    DeviceTensor k_norm_weight;
    DeviceTensor attention_output_weight;
    DeviceTensor ffn_norm_weight;
    DeviceTensor ffn_norm_bias;
    DeviceTensor ffn_input_weight;
    DeviceTensor ffn_output_weight;
  };

  EsmcConfig public_config;
  int device{-1};
  bool test_fixture{false};
  DeviceBuffer weight_storage;
  DeviceTensor embedding;
  DeviceTensor final_norm;
  DeviceTensor head_input_weight;
  DeviceTensor head_input_bias;
  DeviceTensor head_norm_weight;
  DeviceTensor head_norm_bias;
  DeviceTensor head_output_weight;
  DeviceTensor head_output_bias;
  std::vector<Layer> layer;
};

struct EsmcContext::Impl final {
  std::shared_ptr<EsmcModel::Impl> weights;
  std::size_t capacity{0};
  std::size_t position{0};
  Stream stream;
  Blas blas;
  DeviceBuffer tokens;
  DeviceBuffer hidden_a;
  DeviceBuffer hidden_b;
  DeviceBuffer normalized;
  DeviceBuffer qkv;
  DeviceBuffer query;
  DeviceBuffer key;
  DeviceBuffer value;
  DeviceBuffer context;
  DeviceBuffer scores;
  DeviceBuffer projected;
  DeviceBuffer gated;

  [[nodiscard]] float *floats(DeviceBuffer &buffer) const noexcept {
    return static_cast<float *>(buffer.data());
  }
  [[nodiscard]] const float *floats(const DeviceBuffer &buffer) const noexcept {
    return static_cast<const float *>(buffer.data());
  }

  Status linear(const float *const input, const std::size_t rows,
                const std::size_t input_width, const DeviceTensor &weight,
                const std::size_t output_width, const DeviceTensor *const bias,
                float *const output) {
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        input_width >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        output_width >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        weight.elements != input_width * output_width) {
      return {ErrorCode::kInvalidArgument,
              "ESMC CUDA GEMM dimensions are unsupported"};
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    auto status = cublas_status(
        cublasSgemm(blas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(output_width), static_cast<int>(rows),
                    static_cast<int>(input_width), &alpha, weight.data,
                    static_cast<int>(input_width), input,
                    static_cast<int>(input_width), &beta, output,
                    static_cast<int>(output_width)),
        "cublasSgemm ESMC linear");
    if (!status.ok())
      return status;
    if (bias != nullptr) {
      const std::size_t elements = rows * output_width;
      add_bias_kernel<<<blocks_for(elements), kThreads, 0, stream.get()>>>(
          output, bias->data, elements, output_width);
      status = kernel_status("ESMC linear bias kernel");
    }
    return status;
  }

  Status normalize(const float *const input, const DeviceTensor &weight,
                   const DeviceTensor *const bias, const std::size_t rows,
                   float *const output) {
    layer_norm_kernel<<<static_cast<unsigned>(rows), kThreads, 0,
                        stream.get()>>>(
        input, weight.data, bias == nullptr ? nullptr : bias->data,
        weights->public_config.width, weights->public_config.epsilon, output);
    return kernel_status("ESMC layer-norm kernel");
  }

  Status copy_output(const float *const source, const std::size_t elements,
                     std::vector<float> *const output) {
    if (output == nullptr)
      return {ErrorCode::kInvalidArgument, "ESMC host output is null"};
    output->resize(elements);
    std::size_t bytes = 0;
    auto status = element_bytes(elements, &bytes);
    if (!status.ok())
      return status;
    status = cuda_status(cudaMemcpyAsync(output->data(), source, bytes,
                                         cudaMemcpyDeviceToHost, stream.get()),
                         "cudaMemcpyAsync ESMC output");
    return status.ok() ? stream.synchronize() : status;
  }

  Status attention(const std::size_t rows,
                   const EsmcModel::Impl::Layer &layer) {
    const auto &config = weights->public_config;
    float *const hidden = floats(hidden_a);
    float *const norm = floats(normalized);
    float *const qkv_output = floats(qkv);
    float *const q = floats(query);
    float *const k = floats(key);
    float *const v = floats(value);
    auto status = normalize(hidden, layer.qkv_norm_weight, &layer.qkv_norm_bias,
                            rows, norm);
    if (status.ok())
      status = linear(norm, rows, config.width, layer.qkv_weight,
                      config.width * 3U, nullptr, qkv_output);
    if (!status.ok())
      return status;
    const std::size_t hidden_elements = rows * config.width;
    split_qkv_kernel<<<blocks_for(hidden_elements), kThreads, 0,
                       stream.get()>>>(qkv_output, rows, config.width, q, k, v);
    status = kernel_status("ESMC split-QKV kernel");
    if (status.ok())
      status = normalize(q, layer.q_norm_weight, nullptr, rows, norm);
    if (status.ok()) {
      status =
          cuda_status(cudaMemcpyAsync(q, norm, hidden_elements * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream.get()),
                      "cudaMemcpyAsync ESMC normalized query");
    }
    if (status.ok())
      status = normalize(k, layer.k_norm_weight, nullptr, rows, norm);
    if (status.ok()) {
      status =
          cuda_status(cudaMemcpyAsync(k, norm, hidden_elements * sizeof(float),
                                      cudaMemcpyDeviceToDevice, stream.get()),
                      "cudaMemcpyAsync ESMC normalized key");
    }
    if (!status.ok())
      return status;
    const std::size_t rope_elements =
        rows * config.heads * (config.head_width() / 2U);
    rope_kernel<<<blocks_for(rope_elements), kThreads, 0, stream.get()>>>(
        q, k, rows, config.width, config.heads, config.head_width(),
        config.rope_base);
    status = kernel_status("ESMC RoPE kernel");
    if (!status.ok())
      return status;

    const float score_alpha =
        1.0F / std::sqrt(static_cast<float>(config.head_width()));
    const float beta = 0.0F;
    float *const score_data = floats(scores);
    float *const context_data = floats(context);
    for (std::size_t head = 0; head < config.heads; ++head) {
      float *const head_scores = score_data + head * rows * rows;
      const std::size_t head_offset = head * config.head_width();
      status = cublas_status(
          cublasSgemm(blas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                      static_cast<int>(rows), static_cast<int>(rows),
                      static_cast<int>(config.head_width()), &score_alpha,
                      k + head_offset, static_cast<int>(config.width),
                      q + head_offset, static_cast<int>(config.width), &beta,
                      head_scores, static_cast<int>(rows)),
          "cublasSgemm ESMC QK");
      if (!status.ok())
        return status;
      softmax_kernel<<<static_cast<unsigned>(rows), kThreads, 0,
                       stream.get()>>>(
          head_scores, static_cast<const TokenId *>(tokens.data()), rows);
      status = kernel_status("ESMC attention softmax kernel");
      if (!status.ok())
        return status;
      const float context_alpha = 1.0F;
      status = cublas_status(
          cublasSgemm(blas.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                      static_cast<int>(config.head_width()),
                      static_cast<int>(rows), static_cast<int>(rows),
                      &context_alpha, v + head_offset,
                      static_cast<int>(config.width), head_scores,
                      static_cast<int>(rows), &beta, context_data + head_offset,
                      static_cast<int>(config.width)),
          "cublasSgemm ESMC attention value");
      if (!status.ok())
        return status;
    }
    return linear(context_data, rows, config.width,
                  layer.attention_output_weight, config.width, nullptr,
                  floats(hidden_b));
  }

  Status forward(const std::vector<TokenId> &host_tokens,
                 std::vector<float> *const logits,
                 const std::size_t capture_layer,
                 std::vector<float> *const embedding_output) {
    if (!weights || host_tokens.empty() || host_tokens.size() > capacity ||
        position != 0 || (logits == nullptr && embedding_output == nullptr) ||
        (embedding_output != nullptr &&
         capture_layer > weights->public_config.layers)) {
      return {ErrorCode::kInvalidArgument,
              "ESMC CUDA forward requires one fresh full-sequence context"};
    }
    for (const TokenId token : host_tokens) {
      if (token >= weights->public_config.vocab_size)
        return {ErrorCode::kInvalidArgument, "token exceeds ESMC vocabulary"};
    }
    auto status = select_device(weights->device);
    if (!status.ok())
      return status;
    const std::size_t rows = host_tokens.size();
    const auto &config = weights->public_config;
    status = tokens.copy_from_host(host_tokens.data(), rows * sizeof(TokenId),
                                   stream);
    if (!status.ok())
      return status;
    const std::size_t hidden_elements = rows * config.width;
    embedding_kernel<<<blocks_for(hidden_elements), kThreads, 0,
                       stream.get()>>>(
        static_cast<const TokenId *>(tokens.data()), weights->embedding.data,
        rows, config.width, floats(hidden_a));
    status = kernel_status("ESMC embedding kernel");
    if (!status.ok())
      return status;
    if (embedding_output != nullptr && capture_layer == 0) {
      status = copy_output(floats(hidden_a), hidden_elements, embedding_output);
      if (status.ok())
        position = rows;
      return status;
    }
    for (std::size_t index = 0; index < weights->layer.size(); ++index) {
      const auto &layer = weights->layer[index];
      status = attention(rows, layer);
      if (!status.ok())
        return {status.code(), "ESMC CUDA attention block " +
                                   std::to_string(index) + ": " +
                                   status.message()};
      residual_kernel<<<blocks_for(hidden_elements), kThreads, 0,
                        stream.get()>>>(floats(hidden_a), floats(hidden_b),
                                        hidden_elements, config.residue_scale);
      status = kernel_status("ESMC attention residual kernel");
      if (status.ok())
        status = normalize(floats(hidden_a), layer.ffn_norm_weight,
                           &layer.ffn_norm_bias, rows, floats(normalized));
      if (status.ok())
        status = linear(floats(normalized), rows, config.width,
                        layer.ffn_input_weight, config.inner_width * 2U,
                        nullptr, floats(projected));
      if (!status.ok())
        return status;
      const std::size_t gated_elements = rows * config.inner_width;
      swiglu_kernel<<<blocks_for(gated_elements), kThreads, 0, stream.get()>>>(
          floats(projected), rows, config.inner_width, floats(gated));
      status = kernel_status("ESMC SwiGLU kernel");
      if (status.ok())
        status = linear(floats(gated), rows, config.inner_width,
                        layer.ffn_output_weight, config.width, nullptr,
                        floats(hidden_b));
      if (!status.ok())
        return status;
      residual_kernel<<<blocks_for(hidden_elements), kThreads, 0,
                        stream.get()>>>(floats(hidden_a), floats(hidden_b),
                                        hidden_elements, config.residue_scale);
      status = kernel_status("ESMC FFN residual kernel");
      if (!status.ok())
        return status;
      if (embedding_output != nullptr && capture_layer == index + 1U &&
          capture_layer < config.layers) {
        status =
            copy_output(floats(hidden_a), hidden_elements, embedding_output);
        if (status.ok())
          position = rows;
        return status;
      }
    }
    status = normalize(floats(hidden_a), weights->final_norm, nullptr, rows,
                       floats(normalized));
    if (!status.ok())
      return status;
    if (embedding_output != nullptr && capture_layer == config.layers) {
      status =
          copy_output(floats(normalized), hidden_elements, embedding_output);
      if (status.ok())
        position = rows;
      return status;
    }
    status = linear(floats(normalized), rows, config.width,
                    weights->head_input_weight, config.width,
                    &weights->head_input_bias, floats(hidden_b));
    if (!status.ok())
      return status;
    gelu_kernel<<<blocks_for(hidden_elements), kThreads, 0, stream.get()>>>(
        floats(hidden_b), hidden_elements);
    status = kernel_status("ESMC exact GELU kernel");
    if (status.ok())
      status = normalize(floats(hidden_b), weights->head_norm_weight,
                         &weights->head_norm_bias, rows, floats(normalized));
    if (status.ok())
      status = linear(floats(normalized), rows, config.width,
                      weights->head_output_weight, config.vocab_size,
                      &weights->head_output_bias, floats(hidden_a));
    if (!status.ok())
      return status;
    status = copy_output(floats(hidden_a), rows * config.vocab_size, logits);
    if (status.ok())
      position = rows;
    return status;
  }
};

EsmcModel::EsmcModel() : impl_(std::make_shared<Impl>()) {}
EsmcModel::~EsmcModel() = default;
EsmcModel::EsmcModel(EsmcModel &&) noexcept = default;
EsmcModel &EsmcModel::operator=(EsmcModel &&) noexcept = default;

Status EsmcModel::load(const ModelFile &artifact,
                       const std::vector<int> &devices,
                       const bool allow_test_fixture) {
  if (!impl_ || impl_->weight_storage.valid())
    return {ErrorCode::kInvalidArgument, "ESMC CUDA model is already loaded"};
  if (devices.size() != 1)
    return {ErrorCode::kUnsupported,
            "ESMC CUDA v1 requires exactly one device"};
  auto candidate = std::make_shared<Impl>();
  candidate->device = devices.front();
  auto status = select_device(candidate->device);
  if (!status.ok())
    return status;
  std::string runtime_abi;
  status = metadata_string(artifact, "model.architecture",
                           &candidate->public_config.architecture);
  if (status.ok())
    status = metadata_string(artifact, "runtime.abi", &runtime_abi);
  if (!status.ok())
    return status;
  const auto *const registered =
      find_architecture(candidate->public_config.architecture);
  if (registered == nullptr ||
      registered->tokenizer != ArchitectureTokenizer::kEsmcProtein ||
      registered->artifact_profile != artifact.profile() ||
      registered->runtime_abi != runtime_abi ||
      (registered->backends & kArchitectureBackendCuda) == 0) {
    return {ErrorCode::kUnsupported,
            "artifact is not a registered ESMC CUDA architecture"};
  }
  candidate->public_config.artifact_profile = std::string{artifact.profile()};
  candidate->test_fixture = registered->synthetic_fixture;
  if (candidate->test_fixture) {
    bool marked = false;
    status = metadata_bool(artifact, "fixture.synthetic", &marked);
    if (!status.ok())
      return status;
    if (!marked || !allow_test_fixture)
      return {ErrorCode::kUnsupported,
              "synthetic ESMC fixtures require explicit test permission"};
  }
  status =
      metadata_string(artifact, "model.id", &candidate->public_config.model_id);
  if (!status.ok())
    return status;
  for (const auto &item :
       {std::pair{"config.vocab_size", &candidate->public_config.vocab_size},
        std::pair{"config.hidden_size", &candidate->public_config.width},
        std::pair{"config.num_layers", &candidate->public_config.layers},
        std::pair{"config.max_seqlen", &candidate->public_config.max_seqlen},
        std::pair{"config.num_attention_heads",
                  &candidate->public_config.heads},
        std::pair{"config.inner_mlp_size",
                  &candidate->public_config.inner_width}}) {
    status = metadata_size(artifact, item.first, item.second);
    if (!status.ok())
      return status;
  }
  status = metadata_float(artifact, "config.layer_norm_epsilon",
                          &candidate->public_config.epsilon);
  if (status.ok())
    status = metadata_float(artifact, "config.rope_base",
                            &candidate->public_config.rope_base);
  if (status.ok())
    status = metadata_float(artifact, "config.residue_scaling_factor",
                            &candidate->public_config.residue_scale);
  if (!status.ok())
    return status;
  std::size_t embedding_layers = 0;
  status = metadata_size(artifact, "runtime.embedding_layer_count",
                         &embedding_layers);
  if (!status.ok())
    return status;
  const auto &config = candidate->public_config;
  if (config.vocab_size != 64 || config.width == 0 || config.layers == 0 ||
      config.heads == 0 || config.width % config.heads != 0 ||
      config.head_width() % 2 != 0 || config.inner_width == 0 ||
      config.max_seqlen == 0 || config.max_seqlen > 2048 ||
      embedding_layers != config.layers + 1 || !std::isfinite(config.epsilon) ||
      config.epsilon <= 0.0F || !std::isfinite(config.rope_base) ||
      config.rope_base <= 1.0F || !std::isfinite(config.residue_scale) ||
      config.residue_scale <= 0.0F) {
    return {ErrorCode::kUnsupported, "ESMC CUDA dimensions are unsupported"};
  }
  if (!candidate->test_fixture) {
    const auto *const official = find_official_esmc_model(config.model_id);
    std::string source_repo;
    std::string source_revision;
    status = metadata_string(artifact, "source.repo", &source_repo);
    if (status.ok())
      status = metadata_string(artifact, "source.revision", &source_revision);
    if (!status.ok())
      return status;
    if (official == nullptr || official->huggingface_repo != source_repo ||
        official->huggingface_revision != source_revision ||
        official->vocab_size != config.vocab_size ||
        official->hidden_size != config.width ||
        official->layers != config.layers || official->heads != config.heads ||
        official->inner_width != config.inner_width ||
        official->max_seqlen != config.max_seqlen) {
      return {ErrorCode::kUnsupported,
              "ESMC artifact does not match a pinned official topology"};
    }
  }
  candidate->layer.resize(config.layers);
  std::vector<PendingTensor> pending;
  pending.reserve(config.layers * 10U + 8U);
  std::size_t cursor = 0;
  status = schedule_tensor(artifact, "esmc.embed.weight",
                           {config.vocab_size, config.width}, &pending, &cursor,
                           &candidate->embedding);
  if (status.ok())
    status = schedule_tensor(artifact, "esmc.transformer.norm.weight",
                             {config.width}, &pending, &cursor,
                             &candidate->final_norm);
  if (status.ok())
    status = schedule_tensor(artifact, "lm_head.0.weight",
                             {config.width, config.width}, &pending, &cursor,
                             &candidate->head_input_weight);
  if (status.ok())
    status = schedule_tensor(artifact, "lm_head.0.bias", {config.width},
                             &pending, &cursor, &candidate->head_input_bias);
  if (status.ok())
    status = schedule_tensor(artifact, "lm_head.2.weight", {config.width},
                             &pending, &cursor, &candidate->head_norm_weight);
  if (status.ok())
    status = schedule_tensor(artifact, "lm_head.2.bias", {config.width},
                             &pending, &cursor, &candidate->head_norm_bias);
  if (status.ok())
    status = schedule_tensor(artifact, "lm_head.3.weight",
                             {config.vocab_size, config.width}, &pending,
                             &cursor, &candidate->head_output_weight);
  if (status.ok())
    status = schedule_tensor(artifact, "lm_head.3.bias", {config.vocab_size},
                             &pending, &cursor, &candidate->head_output_bias);
  if (!status.ok())
    return status;
  for (std::size_t index = 0; index < candidate->layer.size(); ++index) {
    auto &layer = candidate->layer[index];
    const std::string block =
        "esmc.transformer.blocks." + std::to_string(index);
    status = schedule_tensor(
        artifact, block + ".attn.layernorm_qkv.layer_norm_weight",
        {config.width}, &pending, &cursor, &layer.qkv_norm_weight);
    if (status.ok())
      status = schedule_tensor(
          artifact, block + ".attn.layernorm_qkv.layer_norm_bias",
          {config.width}, &pending, &cursor, &layer.qkv_norm_bias);
    if (status.ok())
      status = schedule_tensor(artifact, block + ".attn.layernorm_qkv.weight",
                               {config.width * 3U, config.width}, &pending,
                               &cursor, &layer.qkv_weight);
    if (status.ok())
      status =
          schedule_tensor(artifact, block + ".attn.q_ln.weight", {config.width},
                          &pending, &cursor, &layer.q_norm_weight);
    if (status.ok())
      status =
          schedule_tensor(artifact, block + ".attn.k_ln.weight", {config.width},
                          &pending, &cursor, &layer.k_norm_weight);
    if (status.ok())
      status = schedule_tensor(artifact, block + ".attn.out_proj.weight",
                               {config.width, config.width}, &pending, &cursor,
                               &layer.attention_output_weight);
    if (status.ok())
      status = schedule_tensor(artifact, block + ".ffn.layer_norm_weight",
                               {config.width}, &pending, &cursor,
                               &layer.ffn_norm_weight);
    if (status.ok())
      status = schedule_tensor(artifact, block + ".ffn.layer_norm_bias",
                               {config.width}, &pending, &cursor,
                               &layer.ffn_norm_bias);
    if (status.ok())
      status = schedule_tensor(artifact, block + ".ffn.fc1_weight",
                               {config.inner_width * 2U, config.width},
                               &pending, &cursor, &layer.ffn_input_weight);
    if (status.ok())
      status = schedule_tensor(artifact, block + ".ffn.fc2_weight",
                               {config.width, config.inner_width}, &pending,
                               &cursor, &layer.ffn_output_weight);
    if (!status.ok())
      return status;
  }
  Stream upload_stream;
  status = upload_stream.create();
  if (status.ok())
    status = candidate->weight_storage.allocate(candidate->device, cursor);
  if (!status.ok())
    return status;
  auto *const base =
      static_cast<std::uint8_t *>(candidate->weight_storage.data());
  for (const auto &item : pending) {
    item.target->data = reinterpret_cast<const float *>(base + item.offset);
    item.target->elements = item.bytes / sizeof(float);
    status = cuda_status(cudaMemcpyAsync(base + item.offset, item.host,
                                         item.bytes, cudaMemcpyHostToDevice,
                                         upload_stream.get()),
                         "cudaMemcpyAsync ESMC weight");
    if (!status.ok())
      return status;
  }
  status = upload_stream.synchronize();
  if (!status.ok())
    return status;
  impl_ = std::move(candidate);
  return Status::Ok();
}

const EsmcConfig &EsmcModel::config() const noexcept {
  return impl_->public_config;
}

int EsmcModel::device() const noexcept { return impl_->device; }

EsmcContext::EsmcContext() : impl_(std::make_unique<Impl>()) {}
EsmcContext::~EsmcContext() = default;
EsmcContext::EsmcContext(EsmcContext &&) noexcept = default;
EsmcContext &EsmcContext::operator=(EsmcContext &&) noexcept = default;

Status EsmcContext::initialize_shared(const EsmcModel &model,
                                      const std::size_t context_capacity) {
  if (!impl_ || impl_->weights || !model.impl_ ||
      !model.impl_->weight_storage.valid() || context_capacity == 0 ||
      context_capacity > model.impl_->public_config.max_seqlen) {
    return {ErrorCode::kInvalidArgument,
            "ESMC CUDA context requires a loaded model and valid capacity"};
  }
  auto status = select_device(model.impl_->device);
  if (!status.ok())
    return status;
  impl_->weights = model.impl_;
  impl_->capacity = context_capacity;
  status = impl_->stream.create();
  if (status.ok())
    status = impl_->blas.create();
  if (status.ok())
    status =
        cublas_status(cublasSetStream(impl_->blas.get(), impl_->stream.get()),
                      "cublasSetStream ESMC");
  if (status.ok())
    status = cublas_status(
        cublasSetMathMode(impl_->blas.get(), CUBLAS_PEDANTIC_MATH),
        "cublasSetMathMode ESMC");
  if (status.ok())
    status = cublas_status(
        cublasSetAtomicsMode(impl_->blas.get(), CUBLAS_ATOMICS_NOT_ALLOWED),
        "cublasSetAtomicsMode ESMC");
  if (!status.ok())
    return status;
  const int device = model.impl_->device;
  const auto &config = model.impl_->public_config;
  std::size_t hidden_elements = 0;
  std::size_t score_elements = 0;
  if (!multiply(context_capacity, config.width, &hidden_elements) ||
      !multiply(context_capacity, context_capacity, &score_elements) ||
      !multiply(score_elements, config.heads, &score_elements)) {
    return {ErrorCode::kInvalidArgument, "ESMC CUDA workspace size overflows"};
  }
  status = impl_->tokens.allocate(device, context_capacity * sizeof(TokenId));
  if (status.ok())
    status = allocate_floats(&impl_->hidden_a, device,
                             context_capacity *
                                 std::max(config.width, config.vocab_size));
  for (DeviceBuffer *const buffer :
       {&impl_->hidden_b, &impl_->normalized, &impl_->query, &impl_->key,
        &impl_->value, &impl_->context}) {
    if (status.ok())
      status = allocate_floats(buffer, device, hidden_elements);
  }
  if (status.ok())
    status = allocate_floats(&impl_->qkv, device, hidden_elements * 3U);
  if (status.ok())
    status = allocate_floats(&impl_->scores, device, score_elements);
  if (status.ok())
    status = allocate_floats(&impl_->projected, device,
                             context_capacity * config.inner_width * 2U);
  if (status.ok())
    status = allocate_floats(&impl_->gated, device,
                             context_capacity * config.inner_width);
  return status;
}

Status EsmcContext::prefill(const std::vector<TokenId> &tokens,
                            std::vector<float> *const logits) {
  return impl_->forward(tokens, logits, std::numeric_limits<std::size_t>::max(),
                        nullptr);
}

Status EsmcContext::prefill_embedding(const std::vector<TokenId> &tokens,
                                      const std::size_t layer,
                                      std::vector<float> *const embedding) {
  return impl_->forward(tokens, nullptr, layer, embedding);
}

Status EsmcContext::prefill_chunk(const std::vector<TokenId> &,
                                  std::vector<float> *) {
  return {ErrorCode::kUnsupported,
          "ESMC requires one full-sequence bidirectional prefill"};
}

Status EsmcContext::decode(const TokenId, std::vector<float> *) {
  return {ErrorCode::kUnsupported,
          "ESMC does not support autoregressive decode"};
}

std::size_t EsmcContext::position() const noexcept { return impl_->position; }

std::size_t EsmcContext::activation_capacity() const noexcept {
  return impl_->capacity;
}

const EsmcConfig &EsmcContext::config() const noexcept {
  return impl_->weights->public_config;
}

} // namespace evo::cuda
