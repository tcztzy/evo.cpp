// SPDX-License-Identifier: Apache-2.0
#include "evo/cuda/model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <future>
#include <initializer_list>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_bf16.h>

#include "evo/cuda/attention.hpp"
#include "evo/cuda/hyena.hpp"
#include "evo/cuda/ops.hpp"
#include "evo/cuda/runtime.hpp"
#include "evo/model_registry.hpp"

namespace evo::cuda {
namespace {

constexpr int kThreads = 256;
constexpr std::size_t kMaximumArenaTokens = 8192;
constexpr std::size_t kTestFixtureArenaTokens = 8;
constexpr std::size_t kQ8KvContextThreshold = 131072;
constexpr std::size_t kQ8KvPageTokens = 16384;
constexpr std::size_t kBackendWarmupTokens = 128;
enum class ForwardMode {
  kPrefill,
  kStatelessPrefill,
  kCachedPrefill,
  kContinue,
  kDecode,
  kCachedDecode
};

constexpr bool is_initial_prefill(const ForwardMode mode) noexcept {
  return mode == ForwardMode::kPrefill ||
         mode == ForwardMode::kStatelessPrefill ||
         mode == ForwardMode::kCachedPrefill;
}

constexpr bool is_stateless_prefill(const ForwardMode mode) noexcept {
  return mode == ForwardMode::kStatelessPrefill;
}

constexpr bool is_decode(const ForwardMode mode) noexcept {
  return mode == ForwardMode::kDecode || mode == ForwardMode::kCachedDecode;
}

constexpr bool uses_cached_attention(const ForwardMode mode) noexcept {
  return mode == ForwardMode::kCachedPrefill ||
         mode == ForwardMode::kCachedDecode;
}

bool multiply(const std::size_t left, const std::size_t right,
              std::size_t *const result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *result = left * right;
  return true;
}

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

Status metadata_u64(const ModelFile &model, const std::string_view key,
                    std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  const std::uint64_t value = read_u64(entry->value.data());
  if (value > std::numeric_limits<std::size_t>::max())
    return {ErrorCode::kModelFormat,
            "metadata exceeds size_t: " + std::string{key}};
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

Status metadata_f32(const ModelFile &model, const std::string_view key,
                    float *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(model, key, MetadataType::kF64, &entry);
  if (!status.ok())
    return status;
  const std::uint64_t bits = read_u64(entry->value.data());
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  if (!std::isfinite(value) || value > std::numeric_limits<float>::max() ||
      value < -std::numeric_limits<float>::max()) {
    return {ErrorCode::kModelFormat,
            "metadata is not a finite F32 value: " + std::string{key}};
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
  output->reserve(entry->value.size() / 8);
  for (std::size_t offset = 0; offset < entry->value.size(); offset += 8) {
    const std::uint64_t value = read_u64(entry->value.data() + offset);
    if (value > std::numeric_limits<std::size_t>::max()) {
      return {ErrorCode::kModelFormat,
              "metadata list exceeds size_t: " + std::string{key}};
    }
    output->push_back(static_cast<std::size_t>(value));
  }
  return Status::Ok();
}

bool valid_dump_point(const LayerDumpPoint point) noexcept {
  return point == LayerDumpPoint::kBlockOutput ||
         point == LayerDumpPoint::kPreNorm ||
         point == LayerDumpPoint::kMixerInputProjection ||
         point == LayerDumpPoint::kMixerShortFilter ||
         point == LayerDumpPoint::kMixerShortState ||
         point == LayerDumpPoint::kMixerInnerState ||
         point == LayerDumpPoint::kMixerX2 ||
         point == LayerDumpPoint::kMixerX1 ||
         point == LayerDumpPoint::kMixerValue ||
         point == LayerDumpPoint::kMixerPregate ||
         point == LayerDumpPoint::kMixerState ||
         point == LayerDumpPoint::kMixerFilter ||
         point == LayerDumpPoint::kMixerConvolution ||
         point == LayerDumpPoint::kMixerOutput ||
         point == LayerDumpPoint::kMixerProjection ||
         point == LayerDumpPoint::kMixerResidual ||
         point == LayerDumpPoint::kPostNorm ||
         point == LayerDumpPoint::kMlpL1 || point == LayerDumpPoint::kMlpL2 ||
         point == LayerDumpPoint::kMlpActivation ||
         point == LayerDumpPoint::kMlpGated ||
         point == LayerDumpPoint::kMlpOutput;
}

Status checked_tensor(const ModelFile &model, const std::string &name,
                      const TensorDType dtype,
                      const std::vector<std::size_t> &shape,
                      const TensorInfo **const output) {
  const auto *const tensor = model.find_tensor(name);
  if (tensor == nullptr)
    return {ErrorCode::kModelFormat, "required tensor is missing: " + name};
  if (tensor->dtype != dtype || tensor->rank != shape.size())
    return {ErrorCode::kModelFormat, "tensor dtype/rank mismatch: " + name};
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (tensor->dimensions[index] != shape[index])
      return {ErrorCode::kModelFormat, "tensor shape mismatch: " + name};
  }
  if (model.tensor_data(*tensor) == nullptr)
    return {ErrorCode::kModelFormat, "tensor payload is unavailable: " + name};
  *output = tensor;
  return Status::Ok();
}

Status upload_tensor(const ModelFile &model, const std::string &name,
                     const TensorDType dtype,
                     const std::vector<std::size_t> &shape, const int device,
                     const Stream &stream, DeviceBuffer *const output) {
  const TensorInfo *tensor = nullptr;
  auto status = checked_tensor(model, name, dtype, shape, &tensor);
  if (!status.ok())
    return status;
  if (tensor->data_size > std::numeric_limits<std::size_t>::max())
    return {ErrorCode::kModelFormat, "tensor is too large for host: " + name};
  const auto bytes = static_cast<std::size_t>(tensor->data_size);
  status = output->allocate(device, bytes);
  if (!status.ok())
    return {status.code(), "allocate '" + name + "': " + status.message()};
  status = output->copy_from_host(model.tensor_data(*tensor), bytes, stream);
  if (!status.ok())
    return {status.code(), "upload '" + name + "': " + status.message()};
  return Status::Ok();
}

Status projection_fp8_scales(const ModelFile &model, const std::string &prefix,
                             float *const input_scale,
                             float *const output_scale) {
  const TensorInfo *scale_tensor = nullptr;
  auto status = checked_tensor(model, prefix + ".fp8_runtime_scales",
                               TensorDType::kF32, {2}, &scale_tensor);
  if (!status.ok())
    return status;

  std::array<float, 2> scales{};
  status = model.read_tensor(*scale_tensor, 0, scales.data(), sizeof(scales));
  if (!status.ok())
    return status;
  if (std::any_of(scales.begin(), scales.end(), [](const float value) {
        return !std::isfinite(value) || value <= 0.0F;
      })) {
    return {ErrorCode::kModelFormat,
            prefix + " contains invalid software-FP8 runtime scales"};
  }
  *input_scale = scales[0];
  *output_scale = scales[1];
  return Status::Ok();
}

Status allocate_bf16(DeviceBuffer *const buffer, const int device,
                     const std::size_t elements, const char *const name) {
  std::size_t bytes = 0;
  if (elements == 0 || !multiply(elements, sizeof(__nv_bfloat16), &bytes))
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " allocation size overflows"};
  auto status = buffer->allocate(device, bytes);
  if (!status.ok())
    return {status.code(), std::string{name} + ": " + status.message()};
  return Status::Ok();
}

unsigned int grid_for(const std::size_t elements) {
  const auto blocks = (elements + static_cast<std::size_t>(kThreads) - 1) /
                      static_cast<std::size_t>(kThreads);
  return static_cast<unsigned int>(
      blocks < 65535 ? blocks : static_cast<std::size_t>(65535));
}

__global__ void embedding_kernel(const TokenId *const tokens,
                                 const __nv_bfloat16 *const embedding,
                                 __nv_bfloat16 *const output,
                                 const std::size_t token_count,
                                 const std::size_t width) {
  const std::size_t elements = token_count * width;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t token = index / width;
    const std::size_t column = index % width;
    output[index] =
        embedding[static_cast<std::size_t>(tokens[token]) * width + column];
  }
}

__global__ void qkv_weight_to_input_major_kernel(
    const __nv_bfloat16 *const source, __nv_bfloat16 *const target,
    const std::size_t width, const std::size_t head_dim,
    const std::size_t elements) {
  const std::size_t output_width = width * 3;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t source_output = index / width;
    const std::size_t input = index % width;
    const std::size_t head = source_output / (3 * head_dim);
    const std::size_t within_head = source_output % (3 * head_dim);
    const std::size_t component = within_head / head_dim;
    const std::size_t dimension = within_head % head_dim;
    const std::size_t projection_output =
        component * width + head * head_dim + dimension;
    target[input * output_width + projection_output] = source[index];
  }
}

template <typename Element>
__global__ void
expand_grouped_filter_kernel(const Element *const source, Element *const target,
                             const std::size_t channels_per_group,
                             const std::size_t kernel_size,
                             const std::size_t elements) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t channel = index / kernel_size;
    const std::size_t tap = index % kernel_size;
    const std::size_t group = channel / channels_per_group;
    target[index] = source[group * kernel_size + tap];
  }
}

Status canonicalize_qkv_weight(const int device, const std::size_t width,
                               const std::size_t head_dim, const Stream &stream,
                               DeviceBuffer *const weight) {
  std::size_t elements = 0;
  if (weight == nullptr || head_dim == 0 || width % head_dim != 0 ||
      !multiply(width, width, &elements) || !multiply(elements, 3, &elements)) {
    return {ErrorCode::kInvalidArgument,
            "QKV weight canonicalization dimensions overflow"};
  }
  DeviceBuffer source = std::move(*weight);
  auto status =
      allocate_bf16(weight, device, elements, "input-major QKV weight");
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  qkv_weight_to_input_major_kernel<<<grid_for(elements), kThreads, 0,
                                     stream.get()>>>(
      static_cast<const __nv_bfloat16 *>(source.data()),
      static_cast<__nv_bfloat16 *>(weight->data()), width, head_dim, elements);
  status = cuda_status(cudaPeekAtLastError(),
                       "input-major QKV weight canonicalization kernel");
  if (!status.ok())
    return status;
  // The temporary source owns the checkpoint-order allocation. Complete the
  // queued upload and reorder before that allocation is released.
  return stream.synchronize();
}

Status expand_grouped_filter(const int device, const std::size_t channels,
                             const std::size_t groups,
                             const std::size_t kernel_size,
                             const HcmFilterDType dtype, const Stream &stream,
                             DeviceBuffer *const weight) {
  std::size_t elements = 0;
  const std::size_t element_size =
      dtype == HcmFilterDType::kF32 ? sizeof(float) : sizeof(__nv_bfloat16);
  std::size_t bytes = 0;
  if (weight == nullptr || groups == 0 || channels % groups != 0 ||
      !multiply(channels, kernel_size, &elements) ||
      !multiply(elements, element_size, &bytes)) {
    return {ErrorCode::kInvalidArgument,
            "grouped filter expansion dimensions overflow"};
  }
  DeviceBuffer source = std::move(*weight);
  auto status = weight->allocate(device, bytes);
  if (!status.ok())
    return status;
  status = select_device(device);
  if (!status.ok())
    return status;
  const std::size_t channels_per_group = channels / groups;
  if (dtype == HcmFilterDType::kF32) {
    expand_grouped_filter_kernel<float>
        <<<grid_for(elements), kThreads, 0, stream.get()>>>(
            static_cast<const float *>(source.data()),
            static_cast<float *>(weight->data()), channels_per_group,
            kernel_size, elements);
  } else {
    expand_grouped_filter_kernel<__nv_bfloat16>
        <<<grid_for(elements), kThreads, 0, stream.get()>>>(
            static_cast<const __nv_bfloat16 *>(source.data()),
            static_cast<__nv_bfloat16 *>(weight->data()), channels_per_group,
            kernel_size, elements);
  }
  status =
      cuda_status(cudaPeekAtLastError(), "grouped filter expansion kernel");
  if (!status.ok())
    return status;
  // The upload and expansion are ordered on this stream. Synchronize before
  // the checkpoint-order source allocation is released.
  return stream.synchronize();
}

Status write_npy(const std::string &path, const std::vector<float> &values,
                 const std::size_t rows, const std::size_t columns) {
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       std::to_string(rows) + ", " + std::to_string(columns) +
                       "), }";
  const std::size_t prefix = 10;
  const std::size_t padding = (64 - ((prefix + header.size() + 1) % 64)) % 64;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<std::uint16_t>::max())
    return {ErrorCode::kInternal, "NPY header is too large"};
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    return {ErrorCode::kIo, "cannot open layer dump: " + path};
  const std::array<char, 8> magic{
      static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y', 1, 0};
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  const auto header_size = static_cast<std::uint16_t>(header.size());
  const std::array<char, 2> encoded_size{
      static_cast<char>(header_size & 0xffU),
      static_cast<char>((header_size >> 8U) & 0xffU)};
  output.write(encoded_size.data(),
               static_cast<std::streamsize>(encoded_size.size()));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!output)
    return {ErrorCode::kIo, "failed to write layer dump: " + path};
  return Status::Ok();
}

struct Layer final {
  MixerType type{MixerType::kHcs};
  bool software_fp8_projection{false};
  float projection_input_scale{1.0F};
  float projection_output_scale{1.0F};
  DeviceBuffer pre_norm;
  DeviceBuffer post_norm;
  DeviceBuffer projection;
  DeviceBuffer output_weight;
  DeviceBuffer output_bias;
  DeviceBuffer short_filter;
  DeviceBuffer inner_filter;
  DeviceBuffer direct;
  DeviceBuffer log_poles;
  DeviceBuffer residues;
  DeviceBuffer inverse_frequency;
  DeviceBuffer l1;
  DeviceBuffer l2;
  DeviceBuffer l3;
  FirCache short_cache;
  FirCache inner_cache;
  IirCache iir_cache;
  KvCache kv_cache;
};

struct Arena final {
  DeviceBuffer token_ids;
  DeviceBuffer hidden;
  DeviceBuffer residual;
  DeviceBuffer normalized;
  DeviceBuffer projection;
  DeviceBuffer short_filtered;
  DeviceBuffer x2;
  DeviceBuffer x1;
  DeviceBuffer value;
  DeviceBuffer mixer_scratch;
  DeviceBuffer mixer_output;
  DeviceBuffer attention_lse_accum;
  DeviceBuffer attention_output_accum;
  DeviceBuffer attention_scaled_key;
  DeviceBuffer attention_scores;
  DeviceBuffer attention_probabilities;
  DeviceBuffer logits;
  DeviceBuffer blas_workspace;
  Bf16LinearPlan projection_plan;
  Bf16LinearPlan output_plan;
  Bf16LinearPlan final_plan;
  MlpWorkspace mlp;
};

std::size_t total_bytes(
    const std::initializer_list<const DeviceBuffer *> buffers) noexcept {
  std::size_t total = 0;
  for (const auto *const buffer : buffers)
    total += buffer->bytes();
  return total;
}

std::size_t layer_weight_bytes(const Layer &layer) noexcept {
  return total_bytes(
      {&layer.pre_norm, &layer.post_norm, &layer.projection,
       &layer.output_weight, &layer.output_bias, &layer.short_filter,
       &layer.inner_filter, &layer.direct, &layer.log_poles, &layer.residues,
       &layer.inverse_frequency, &layer.l1, &layer.l2, &layer.l3});
}

std::size_t layer_cache_bytes(const Layer &layer) noexcept {
  return total_bytes({&layer.short_cache.state, &layer.inner_cache.state,
                      &layer.iir_cache.state}) +
         layer.kv_cache.allocated_bytes();
}

std::size_t arena_bytes(const Arena &arena) noexcept {
  return total_bytes({&arena.token_ids,
                      &arena.hidden,
                      &arena.residual,
                      &arena.normalized,
                      &arena.projection,
                      &arena.short_filtered,
                      &arena.x2,
                      &arena.x1,
                      &arena.value,
                      &arena.mixer_scratch,
                      &arena.mixer_output,
                      &arena.attention_lse_accum,
                      &arena.attention_output_accum,
                      &arena.attention_scaled_key,
                      &arena.attention_scores,
                      &arena.attention_probabilities,
                      &arena.logits,
                      &arena.blas_workspace,
                      &arena.mlp.first,
                      &arena.mlp.second,
                      &arena.mlp.activated,
                      &arena.mlp.gated,
                      &arena.mlp.blas});
}

} // namespace

std::size_t backend_warmup_tokens(const RuntimeModelConfig &config,
                                  const std::size_t arena_capacity) noexcept {
  constexpr std::string_view family = "evo2_7b";
  // Populate lazy CUDA modules plus the common short-prefill FFT/BLAS plans
  // while model loading is already an explicitly synchronized operation.
  return !config.test_fixture && arena_capacity >= kBackendWarmupTokens &&
                 config.model_id.compare(0, family.size(), family) == 0
             ? kBackendWarmupTokens
             : 0;
}

Status read_runtime_model_config(const ModelFile &model,
                                 const bool allow_test_fixture,
                                 RuntimeModelConfig *const config) {
  if (config == nullptr)
    return {ErrorCode::kInvalidArgument, "model config output is null"};
  RuntimeModelConfig candidate;
  std::string runtime_abi;
  auto status = metadata_string(model, "runtime.abi", &runtime_abi);
  if (!status.ok())
    return status;
  if (runtime_abi != "evo2-safetensors-v1") {
    return {ErrorCode::kUnsupported, "unsupported runtime ABI: " + runtime_abi};
  }
  std::string architecture;
  status = metadata_string(model, "model.architecture", &architecture);
  if (!status.ok())
    return status;
  if (architecture != "StripedHyena2" && architecture != "StripedHyena2Test") {
    return {ErrorCode::kUnsupported,
            "unsupported model architecture: " + architecture};
  }
  if (architecture == "StripedHyena2Test") {
    status = metadata_bool(model, "fixture.synthetic", &candidate.test_fixture);
    if (!status.ok())
      return status;
    if (!candidate.test_fixture || !allow_test_fixture)
      return {ErrorCode::kUnsupported,
              "synthetic model fixtures require explicit test permission"};
  }
  const bool has_model_id = model.find_metadata("model.id") != nullptr;
  for (const auto &item :
       {std::pair{"config.vocab_size", &candidate.vocab_size},
        std::pair{"config.hidden_size", &candidate.width},
        std::pair{"config.num_layers", &candidate.layers},
        std::pair{"config.num_attention_heads", &candidate.heads},
        std::pair{"config.state_size", &candidate.state_size},
        std::pair{"config.inner_mlp_size", &candidate.inner_width},
        std::pair{"config.short_filter_length", &candidate.short_filter_length},
        std::pair{"config.hcs_filter_length", &candidate.hcs_filter_length},
        std::pair{"config.hcm_filter_length", &candidate.hcm_filter_length},
        std::pair{"config.hcs_filter_groups", &candidate.hcs_filter_groups},
        std::pair{"config.hcm_filter_groups", &candidate.hcm_filter_groups}}) {
    status = metadata_u64(model, item.first, item.second);
    if (!status.ok())
      return status;
  }
  status = metadata_f32(model, "config.eps", &candidate.epsilon);
  if (!status.ok())
    return status;
  status = metadata_f32(model, "config.rotary_emb_scaling_factor",
                        &candidate.rope_scale);
  if (!status.ok())
    return status;
  if (!candidate.test_fixture) {
    std::size_t num_filters = 0;
    std::size_t max_batch_size = 0;
    std::size_t inner_size_multiple = 0;
    std::size_t projection_groups = 0;
    for (const auto &item :
         {std::pair{"config.num_filters", &num_filters},
          std::pair{"config.max_batch_size", &max_batch_size},
          std::pair{"config.inner_size_multiple_of", &inner_size_multiple},
          std::pair{"config.proj_groups", &projection_groups}}) {
      status = metadata_u64(model, item.first, item.second);
      if (!status.ok())
        return status;
    }
    if (num_filters != candidate.width || max_batch_size != 1 ||
        inner_size_multiple != 128 || projection_groups != 1) {
      return {ErrorCode::kUnsupported,
              "model filter/batch/projection metadata is not an official "
              "Evo 2 profile"};
    }
    status = metadata_u64(model, "config.hcl_filter_groups",
                          &candidate.hcl_filter_groups);
    if (!status.ok())
      return status;
    status = metadata_u64(model, "config.max_seqlen", &candidate.max_seqlen);
    if (!status.ok())
      return status;
    status =
        metadata_f32(model, "config.rotary_emb_base", &candidate.rope_base);
    if (!status.ok())
      return status;
    status = metadata_bool(model, "config.use_interpolated_rotary_pos_emb",
                           &candidate.interpolated_rope);
    if (!status.ok())
      return status;
  } else {
    candidate.hcl_filter_groups = candidate.width;
    candidate.max_seqlen = std::numeric_limits<std::size_t>::max();
    candidate.rope_base = 10000.0F;
    candidate.interpolated_rope = true;
  }

  bool tie_embeddings = false;
  bool column_split = false;
  bool column_split_hyena = true;
  bool interleave = false;
  bool flip = true;
  bool use_fp8_input_projections = false;
  status = metadata_bool(model, "config.tie_embeddings", &tie_embeddings);
  if (!status.ok())
    return status;
  if (model.find_metadata("config.column_split") != nullptr) {
    status = metadata_bool(model, "config.column_split", &column_split);
    if (!status.ok())
      return status;
  } else if (!candidate.test_fixture) {
    return {ErrorCode::kModelFormat,
            "required metadata is missing: config.column_split"};
  }
  status =
      metadata_bool(model, "config.column_split_hyena", &column_split_hyena);
  if (!status.ok())
    return status;
  status = metadata_bool(model, "config.interleave", &interleave);
  if (!status.ok())
    return status;
  status = metadata_bool(model, "config.hyena_flip_x1x2", &flip);
  if (!status.ok())
    return status;
  if (model.find_metadata("config.use_fp8_input_projections") != nullptr) {
    status = metadata_bool(model, "config.use_fp8_input_projections",
                           &use_fp8_input_projections);
    if (!status.ok())
      return status;
  } else if (!candidate.test_fixture) {
    return {ErrorCode::kModelFormat,
            "required metadata is missing: config.use_fp8_input_projections"};
  }
  if (!tie_embeddings || (!candidate.test_fixture && !column_split) ||
      column_split_hyena || !interleave || flip)
    return {ErrorCode::kUnsupported,
            "model semantic flags do not match supported Evo 2"};
  candidate.qkv_source_head_major = column_split;
  candidate.qkv_head_major = false;
  if (!candidate.test_fixture) {
    bool short_filter_bias = true;
    bool evo2_activations = false;
    bool final_norm = false;
    bool qkv_bias = true;
    bool mha_bias = false;
    bool hyena_bias = false;
    for (const auto &item :
         {std::pair{"config.short_filter_bias", &short_filter_bias},
          std::pair{"config.evo2_style_activations", &evo2_activations},
          std::pair{"config.final_norm", &final_norm},
          std::pair{"config.qkv_proj_bias", &qkv_bias},
          std::pair{"config.mha_out_proj_bias", &mha_bias},
          std::pair{"config.hyena_out_proj_bias", &hyena_bias}}) {
      status = metadata_bool(model, item.first, item.second);
      if (!status.ok())
        return status;
    }
    std::string tokenizer;
    std::string prefill;
    std::string activation;
    for (const auto &item : {std::pair{"config.tokenizer_type", &tokenizer},
                             std::pair{"config.prefill_style", &prefill},
                             std::pair{"config.mlp_activation", &activation}}) {
      status = metadata_string(model, item.first, item.second);
      if (!status.ok())
        return status;
    }
    if (short_filter_bias || !evo2_activations || !final_norm || qkv_bias ||
        !mha_bias || !hyena_bias || tokenizer != "CharLevelTokenizer" ||
        prefill != "fft" || activation != "gelu") {
      return {ErrorCode::kUnsupported,
              "model semantic metadata is not an official Evo 2 profile"};
    }
  }

  std::string projection_dtype;
  if (model.find_metadata("hyena_projection_dtype") != nullptr) {
    status =
        metadata_string(model, "hyena_projection_dtype", &projection_dtype);
    if (!status.ok())
      return status;
  } else if (candidate.test_fixture || use_fp8_input_projections) {
    projection_dtype = use_fp8_input_projections ? "E4M3_SW" : "BF16";
  } else {
    return {ErrorCode::kModelFormat,
            "required metadata is missing: hyena_projection_dtype"};
  }
  if (projection_dtype == "BF16") {
    candidate.hyena_projection_dtype = HyenaProjectionDType::kBF16;
  } else if (projection_dtype == "E4M3_SW") {
    candidate.hyena_projection_dtype = HyenaProjectionDType::kE4M3Sw;
  } else {
    return {ErrorCode::kUnsupported,
            "unsupported Hyena projection dtype: " + projection_dtype};
  }
  const bool software_fp8 =
      candidate.hyena_projection_dtype == HyenaProjectionDType::kE4M3Sw;
  if (software_fp8 != use_fp8_input_projections) {
    return {ErrorCode::kModelFormat, "hyena_projection_dtype disagrees with "
                                     "config.use_fp8_input_projections"};
  }

  std::string hcm_filter_dtype = "BF16";
  if (model.find_metadata("hcm_filter_dtype") != nullptr) {
    status = metadata_string(model, "hcm_filter_dtype", &hcm_filter_dtype);
    if (!status.ok())
      return status;
  } else if (has_model_id)
    return {ErrorCode::kModelFormat,
            "registered models require hcm_filter_dtype metadata"};
  if (hcm_filter_dtype == "BF16") {
    candidate.hcm_filter_dtype = HcmFilterDType::kBF16;
  } else if (hcm_filter_dtype == "F32") {
    candidate.hcm_filter_dtype = HcmFilterDType::kF32;
  } else {
    return {ErrorCode::kUnsupported,
            "unsupported medium-Hyena filter dtype: " + hcm_filter_dtype};
  }

  if (!candidate.test_fixture && software_fp8) {
    std::string fp8_reference;
    status = metadata_string(model, "fp8.reference", &fp8_reference);
    if (!status.ok())
      return status;
    if (fp8_reference != "TransformerEngine-2.3-HYBRID") {
      return {ErrorCode::kUnsupported,
              "unsupported software-FP8 checkpoint reference: " +
                  fp8_reference};
    }
  } else if (!software_fp8 && model.find_metadata("fp8.reference") != nullptr) {
    return {ErrorCode::kModelFormat,
            "BF16 Hyena projections must not declare fp8.reference"};
  }

  std::vector<std::size_t> hcs;
  std::vector<std::size_t> hcm;
  std::vector<std::size_t> hcl;
  std::vector<std::size_t> attention;
  status = metadata_list(model, "config.hcs_layer_idxs", &hcs);
  if (!status.ok())
    return status;
  status = metadata_list(model, "config.hcm_layer_idxs", &hcm);
  if (!status.ok())
    return status;
  status = metadata_list(model, "config.hcl_layer_idxs", &hcl);
  if (!status.ok())
    return status;
  status = metadata_list(model, "config.attn_layer_idxs", &attention);
  if (!status.ok())
    return status;
  if (candidate.vocab_size == 0 || candidate.width == 0 ||
      candidate.heads == 0 || candidate.width % candidate.heads != 0 ||
      candidate.head_dim() % 2 != 0 || candidate.head_dim() > 256 ||
      candidate.state_size == 0 || candidate.inner_width == 0 ||
      candidate.short_filter_length < 2 || candidate.hcs_filter_length < 2 ||
      candidate.hcm_filter_length < 2 || candidate.hcs_filter_groups == 0 ||
      candidate.width % candidate.hcs_filter_groups != 0 ||
      candidate.hcm_filter_groups == 0 ||
      candidate.width % candidate.hcm_filter_groups != 0 ||
      candidate.hcl_filter_groups == 0 ||
      candidate.width % candidate.hcl_filter_groups != 0 ||
      !std::isfinite(candidate.epsilon) || candidate.epsilon <= 0.0F ||
      !std::isfinite(candidate.rope_scale) || candidate.rope_scale <= 0.0F ||
      !std::isfinite(candidate.rope_base) || candidate.rope_base <= 0.0F) {
    return {ErrorCode::kUnsupported, "model dimensions are unsupported"};
  }

  if (!candidate.test_fixture) {
    const auto matches = [&](const OfficialModelSpec &spec) {
      const bool precision =
          (candidate.hyena_projection_dtype == HyenaProjectionDType::kBF16) ==
          (spec.projection_precision == OfficialProjectionPrecision::kBF16);
      const bool hcm_dtype =
          (candidate.hcm_filter_dtype == HcmFilterDType::kBF16) ==
          (spec.hcm_filter_dtype == OfficialHcmFilterDType::kBF16);
      return candidate.vocab_size == spec.vocab_size &&
             candidate.width == spec.hidden_size &&
             candidate.layers == spec.layers && candidate.heads == spec.heads &&
             candidate.state_size == 16 &&
             candidate.inner_width == spec.inner_width &&
             candidate.short_filter_length == 3 &&
             candidate.hcs_filter_length == 7 &&
             candidate.hcm_filter_length == 128 &&
             candidate.hcs_filter_groups == spec.hcs_groups &&
             candidate.hcm_filter_groups == spec.hcm_groups &&
             candidate.hcl_filter_groups == spec.hcl_groups &&
             candidate.max_seqlen == spec.max_seqlen &&
             candidate.epsilon == 1.0e-6F &&
             candidate.rope_base == static_cast<float>(spec.rope_base) &&
             candidate.rope_scale == static_cast<float>(spec.rope_scale) &&
             candidate.interpolated_rope == spec.interpolated_rope &&
             precision && hcm_dtype && hcs == spec.hcs && hcm == spec.hcm &&
             hcl == spec.hcl && attention == spec.attention;
    };
    const OfficialModelSpec *official = nullptr;
    if (has_model_id) {
      status = metadata_string(model, "model.id", &candidate.model_id);
      if (!status.ok())
        return status;
      official = find_official_model(candidate.model_id);
      if (official == nullptr) {
        return {ErrorCode::kUnsupported,
                "unsupported model.id: " + candidate.model_id};
      }
      if (!matches(*official)) {
        return {ErrorCode::kUnsupported,
                "model metadata does not match official registry profile '" +
                    candidate.model_id + "'"};
      }
    } else {
      for (const auto &spec : official_model_specs()) {
        if (matches(spec)) {
          if (official != nullptr) {
            return {ErrorCode::kUnsupported,
                    "legacy model metadata is ambiguous; reconvert with "
                    "model.id"};
          }
          official = &spec;
        }
      }
      if (official == nullptr) {
        return {ErrorCode::kUnsupported,
                "legacy model metadata does not match a supported official "
                "Evo 2 profile; reconvert with the matching config"};
      }
      candidate.model_id = std::string{official->id};
    }
  }

  std::vector<std::size_t> classified = hcs;
  classified.insert(classified.end(), hcm.begin(), hcm.end());
  classified.insert(classified.end(), hcl.begin(), hcl.end());
  classified.insert(classified.end(), attention.begin(), attention.end());
  std::sort(classified.begin(), classified.end());
  std::vector<std::size_t> expected_layers(candidate.layers);
  for (std::size_t index = 0; index < expected_layers.size(); ++index)
    expected_layers[index] = index;
  if (classified != expected_layers) {
    return {ErrorCode::kModelFormat,
            "mixer layer lists must be disjoint and cover every block"};
  }

  candidate.mixer_types.assign(candidate.layers, MixerType::kHcs);
  for (const auto index : hcm)
    candidate.mixer_types[index] = MixerType::kHcm;
  for (const auto index : hcl)
    candidate.mixer_types[index] = MixerType::kHcl;
  for (const auto index : attention)
    candidate.mixer_types[index] = MixerType::kAttention;

  std::set<std::string> expected_tensors;
  const auto require_tensor =
      [&](const std::string &name, const TensorDType dtype,
          const std::vector<std::size_t> &shape) -> Status {
    if (!expected_tensors.insert(name).second)
      return {ErrorCode::kInternal, "duplicate runtime tensor rule: " + name};
    const TensorInfo *tensor = nullptr;
    return checked_tensor(model, name, dtype, shape, &tensor);
  };
  status = require_tensor("embedding_layer.weight", TensorDType::kBF16,
                          {candidate.vocab_size, candidate.width});
  if (!status.ok())
    return status;
  status = require_tensor("unembed.weight", TensorDType::kBF16,
                          {candidate.vocab_size, candidate.width});
  if (!status.ok())
    return status;
  status = require_tensor("norm.scale", TensorDType::kF32, {candidate.width});
  if (!status.ok())
    return status;

  for (std::size_t index = 0; index < candidate.layers; ++index) {
    const auto type = candidate.mixer_types[index];
    const std::string block = "blocks." + std::to_string(index);
    for (const auto &name :
         {block + ".pre_norm.scale", block + ".post_norm.scale"}) {
      status = require_tensor(name, TensorDType::kF32, {candidate.width});
      if (!status.ok())
        return status;
    }
    status = require_tensor(block + ".mlp.l1.weight", TensorDType::kBF16,
                            {candidate.inner_width, candidate.width});
    if (!status.ok())
      return status;
    status = require_tensor(block + ".mlp.l2.weight", TensorDType::kBF16,
                            {candidate.inner_width, candidate.width});
    if (!status.ok())
      return status;
    status = require_tensor(block + ".mlp.l3.weight", TensorDType::kBF16,
                            {candidate.width, candidate.inner_width});
    if (!status.ok())
      return status;
    if (type == MixerType::kAttention) {
      status = require_tensor(block + ".inner_mha_cls.Wqkv.weight",
                              TensorDType::kBF16,
                              {candidate.width * 3, candidate.width});
      if (!status.ok())
        return status;
      status = require_tensor(block + ".inner_mha_cls.out_proj.weight",
                              TensorDType::kBF16,
                              {candidate.width, candidate.width});
      if (!status.ok())
        return status;
      status = require_tensor(block + ".inner_mha_cls.out_proj.bias",
                              TensorDType::kBF16, {candidate.width});
      if (!status.ok())
        return status;
      status = require_tensor(block + ".inner_mha_cls.rotary_emb.inv_freq",
                              TensorDType::kF32, {candidate.head_dim() / 2});
      if (!status.ok())
        return status;
      continue;
    }

    const std::string projection = block + ".projections";
    status = require_tensor(projection + ".weight",
                            software_fp8 ? TensorDType::kE4M3Software
                                         : TensorDType::kBF16,
                            {candidate.width * 3, candidate.width});
    if (!status.ok())
      return status;
    status =
        require_tensor(block + ".out_filter_dense.weight", TensorDType::kBF16,
                       {candidate.width, candidate.width});
    if (!status.ok())
      return status;
    status = require_tensor(block + ".out_filter_dense.bias",
                            TensorDType::kBF16, {candidate.width});
    if (!status.ok())
      return status;
    status = require_tensor(
        block + ".filter.short_filter_weight", TensorDType::kBF16,
        {candidate.width * 3, 1, candidate.short_filter_length});
    if (!status.ok())
      return status;
    if (type == MixerType::kHcs) {
      status = require_tensor(
          block + ".filter.h", TensorDType::kBF16,
          {candidate.hcs_filter_groups, 1, candidate.hcs_filter_length});
    } else {
      status = require_tensor(block + ".filter.D", TensorDType::kBF16,
                              {candidate.width});
      if (status.ok() && type == MixerType::kHcm) {
        status = require_tensor(
            block + ".filter.h",
            candidate.hcm_filter_dtype == HcmFilterDType::kF32
                ? TensorDType::kF32
                : TensorDType::kBF16,
            {candidate.hcm_filter_groups, 1, candidate.hcm_filter_length});
      } else if (status.ok()) {
        status = require_tensor(block + ".filter.log_poles", TensorDType::kF32,
                                {candidate.width, candidate.state_size, 1});
        if (status.ok()) {
          status = require_tensor(block + ".filter.residues", TensorDType::kF32,
                                  {candidate.width, candidate.state_size});
        }
      }
    }
    if (!status.ok())
      return status;
    if (software_fp8) {
      status = require_tensor(projection + ".fp8_runtime_scales",
                              TensorDType::kF32, {2});
      if (!status.ok())
        return status;
    }
  }
  if (expected_tensors.size() != model.tensors().size()) {
    for (const auto &tensor : model.tensors()) {
      if (expected_tensors.count(tensor.name) == 0) {
        return {ErrorCode::kModelFormat, "unknown tensor for model profile '" +
                                             candidate.model_id +
                                             "': " + tensor.name};
      }
    }
    return {ErrorCode::kModelFormat,
            "model tensor table contains duplicate runtime rules"};
  }
  *config = std::move(candidate);
  return Status::Ok();
}

struct SingleGpuModel::Impl final {
  RuntimeModelConfig config;
  int device{-1};
  std::size_t context_capacity{0};
  std::size_t arena_capacity{0};
  std::size_t layer_offset{0};
  std::size_t position{0};
  bool q8_kv_cache{false};
  bool loaded{false};
  bool state_valid{false};
  bool cached_attention{false};
  Stream stream;
  Blas cached_attention_blas;
  BlasLt blas;
  DeviceBuffer embedding;
  DeviceBuffer unembed;
  DeviceBuffer final_norm;
  std::vector<Layer> layers;
  Arena arena;
  FftWorkspace hcm_fft;
  FftWorkspace hcl_fft;

  [[nodiscard]] Status allocate_arena() {
    std::size_t sequence = 0;
    std::size_t triple_sequence = 0;
    std::size_t logits_elements = 0;
    if (!multiply(arena_capacity, config.width, &sequence) ||
        !multiply(sequence, 3, &triple_sequence) ||
        !multiply(arena_capacity, config.vocab_size, &logits_elements)) {
      return {ErrorCode::kInvalidArgument,
              "activation arena dimensions overflow"};
    }
    std::size_t token_bytes = 0;
    if (!multiply(arena_capacity, sizeof(TokenId), &token_bytes))
      return {ErrorCode::kInvalidArgument, "token arena dimensions overflow"};
    auto status = arena.token_ids.allocate(device, token_bytes);
    if (!status.ok())
      return status;
    for (auto *const buffer :
         {&arena.hidden, &arena.residual, &arena.normalized, &arena.x2,
          &arena.x1, &arena.value, &arena.mixer_scratch, &arena.mixer_output}) {
      status = allocate_bf16(buffer, device, sequence, "activation arena");
      if (!status.ok())
        return status;
    }
    status = allocate_bf16(&arena.projection, device, triple_sequence,
                           "projection arena");
    if (!status.ok())
      return status;
    status = allocate_bf16(&arena.short_filtered, device, triple_sequence,
                           "short-filter arena");
    if (!status.ok())
      return status;
    status =
        allocate_bf16(&arena.logits, device, logits_elements, "logits arena");
    if (!status.ok())
      return status;
    status = arena.blas_workspace.allocate(device, kDefaultBlasWorkspaceBytes);
    if (!status.ok())
      return status;
    return arena.mlp.allocate(device, arena_capacity, config.inner_width);
  }

  [[nodiscard]] Status load_common_layer(const ModelFile &model,
                                         const std::size_t index,
                                         Layer *const layer) {
    const std::string prefix = "blocks." + std::to_string(index);
    auto status =
        upload_tensor(model, prefix + ".pre_norm.scale", TensorDType::kF32,
                      {config.width}, device, stream, &layer->pre_norm);
    if (!status.ok())
      return status;
    status =
        upload_tensor(model, prefix + ".post_norm.scale", TensorDType::kF32,
                      {config.width}, device, stream, &layer->post_norm);
    if (!status.ok())
      return status;
    status = upload_tensor(model, prefix + ".mlp.l1.weight", TensorDType::kBF16,
                           {config.inner_width, config.width}, device, stream,
                           &layer->l1);
    if (!status.ok())
      return status;
    status = upload_tensor(model, prefix + ".mlp.l2.weight", TensorDType::kBF16,
                           {config.inner_width, config.width}, device, stream,
                           &layer->l2);
    if (!status.ok())
      return status;
    return upload_tensor(model, prefix + ".mlp.l3.weight", TensorDType::kBF16,
                         {config.width, config.inner_width}, device, stream,
                         &layer->l3);
  }

  [[nodiscard]] Status load_layer(const ModelFile &model,
                                  const std::size_t index, Layer *const layer) {
    layer->type = config.mixer_types[index];
    auto status = load_common_layer(model, index, layer);
    if (!status.ok())
      return status;
    const std::string prefix = "blocks." + std::to_string(index);
    if (layer->type == MixerType::kAttention) {
      status = upload_tensor(
          model, prefix + ".inner_mha_cls.Wqkv.weight", TensorDType::kBF16,
          {config.width * 3, config.width}, device, stream, &layer->projection);
      if (!status.ok())
        return status;
      if (config.qkv_source_head_major) {
        status =
            canonicalize_qkv_weight(device, config.width, config.head_dim(),
                                    stream, &layer->projection);
        if (!status.ok())
          return status;
      }
      status = upload_tensor(model, prefix + ".inner_mha_cls.out_proj.weight",
                             TensorDType::kBF16, {config.width, config.width},
                             device, stream, &layer->output_weight);
      if (!status.ok())
        return status;
      status = upload_tensor(model, prefix + ".inner_mha_cls.out_proj.bias",
                             TensorDType::kBF16, {config.width}, device, stream,
                             &layer->output_bias);
      if (!status.ok())
        return status;
      status =
          upload_tensor(model, prefix + ".inner_mha_cls.rotary_emb.inv_freq",
                        TensorDType::kF32, {config.head_dim() / 2}, device,
                        stream, &layer->inverse_frequency);
      if (!status.ok())
        return status;
      return q8_kv_cache
                 ? layer->kv_cache.allocate_q8_paged(
                       device, context_capacity, config.heads,
                       config.head_dim(), kQ8KvPageTokens, stream)
                 : layer->kv_cache.allocate(device, context_capacity,
                                            config.heads, config.head_dim());
    }

    if (config.hyena_projection_dtype == HyenaProjectionDType::kE4M3Sw) {
      status = projection_fp8_scales(model, prefix + ".projections",
                                     &layer->projection_input_scale,
                                     &layer->projection_output_scale);
      if (!status.ok())
        return status;
      status = upload_tensor(
          model, prefix + ".projections.weight", TensorDType::kE4M3Software,
          {config.width * 3, config.width}, device, stream, &layer->projection);
      if (!status.ok())
        return status;
      layer->software_fp8_projection = true;
    } else {
      status = upload_tensor(
          model, prefix + ".projections.weight", TensorDType::kBF16,
          {config.width * 3, config.width}, device, stream, &layer->projection);
      if (!status.ok())
        return status;
    }
    status = upload_tensor(model, prefix + ".out_filter_dense.weight",
                           TensorDType::kBF16, {config.width, config.width},
                           device, stream, &layer->output_weight);
    if (!status.ok())
      return status;
    status = upload_tensor(model, prefix + ".out_filter_dense.bias",
                           TensorDType::kBF16, {config.width}, device, stream,
                           &layer->output_bias);
    if (!status.ok())
      return status;
    status = upload_tensor(model, prefix + ".filter.short_filter_weight",
                           TensorDType::kBF16,
                           {config.width * 3, 1, config.short_filter_length},
                           device, stream, &layer->short_filter);
    if (!status.ok())
      return status;
    status = layer->short_cache.allocate(device, config.width * 3,
                                         config.short_filter_length, stream);
    if (!status.ok())
      return status;

    if (layer->type == MixerType::kHcs) {
      status =
          upload_tensor(model, prefix + ".filter.h", TensorDType::kBF16,
                        {config.hcs_filter_groups, 1, config.hcs_filter_length},
                        device, stream, &layer->inner_filter);
      if (!status.ok())
        return status;
      return layer->inner_cache.allocate(device, config.width,
                                         config.hcs_filter_length, stream);
    }
    if (layer->type == MixerType::kHcm) {
      status = upload_tensor(
          model, prefix + ".filter.h",
          config.hcm_filter_dtype == HcmFilterDType::kF32 ? TensorDType::kF32
                                                          : TensorDType::kBF16,
          {config.hcm_filter_groups, 1, config.hcm_filter_length}, device,
          stream, &layer->inner_filter);
      if (!status.ok())
        return status;
      status = expand_grouped_filter(
          device, config.width, config.hcm_filter_groups,
          config.hcm_filter_length, config.hcm_filter_dtype, stream,
          &layer->inner_filter);
      if (!status.ok())
        return status;
      status = upload_tensor(model, prefix + ".filter.D", TensorDType::kBF16,
                             {config.width}, device, stream, &layer->direct);
      if (!status.ok())
        return status;
      return layer->inner_cache.allocate(device, config.width,
                                         config.hcm_filter_length, stream);
    }
    status = upload_tensor(model, prefix + ".filter.D", TensorDType::kBF16,
                           {config.width}, device, stream, &layer->direct);
    if (!status.ok())
      return status;
    status =
        upload_tensor(model, prefix + ".filter.log_poles", TensorDType::kF32,
                      {config.width, config.state_size, 1}, device, stream,
                      &layer->log_poles);
    if (!status.ok())
      return status;
    status = upload_tensor(model, prefix + ".filter.residues",
                           TensorDType::kF32, {config.width, config.state_size},
                           device, stream, &layer->residues);
    if (!status.ok())
      return status;
    return layer->iir_cache.allocate(device, config.width, config.state_size,
                                     stream);
  }

  [[nodiscard]] Status prepare_prefill(const std::size_t rows) {
    position = 0;
    state_valid = false;
    for (auto &layer : layers) {
      if (layer.type == MixerType::kAttention)
        layer.kv_cache.reset_length();
    }
    const bool hcm_uses_fft = hcm_prefill_uses_fft(rows);
    const std::size_t hcm_size = hcm_uses_fft ? hcm_fft_size(rows) : 0;
    const std::size_t hcl_size = hcl_fft_size(rows);
    if ((hcm_uses_fft && hcm_size == 0) || hcl_size == 0)
      return {ErrorCode::kInvalidArgument, "prefill FFT dimensions overflow"};
    if (hcm_uses_fft &&
        !hcm_fft.matches(device, config.width, config.width, hcm_size)) {
      hcm_fft.reset();
      auto status =
          hcm_fft.allocate(device, config.width, config.width, hcm_size);
      if (!status.ok())
        return status;
    }
    if (hcl_fft.matches(device, config.width, config.width, hcl_size,
                        FftInputMode::kRealFullSpectrum))
      return Status::Ok();
    hcl_fft.reset();
    return hcl_fft.allocate(device, config.width, config.width, hcl_size,
                            FftInputMode::kRealFullSpectrum);
  }

  [[nodiscard]] Status embed(const std::vector<TokenId> &tokens) {
    for (const TokenId token : tokens) {
      if (static_cast<std::size_t>(token) >= config.vocab_size)
        return {ErrorCode::kInvalidArgument,
                "token ID exceeds model vocabulary"};
    }
    const std::size_t token_bytes = tokens.size() * sizeof(TokenId);
    auto status =
        arena.token_ids.copy_from_host(tokens.data(), token_bytes, stream);
    if (!status.ok())
      return status;
    std::size_t elements = 0;
    if (!multiply(tokens.size(), config.width, &elements))
      return {ErrorCode::kInvalidArgument, "embedding dimensions overflow"};
    status = select_device(device);
    if (!status.ok())
      return status;
    embedding_kernel<<<grid_for(elements), kThreads, 0, stream.get()>>>(
        static_cast<const TokenId *>(arena.token_ids.data()),
        static_cast<const __nv_bfloat16 *>(embedding.data()),
        static_cast<__nv_bfloat16 *>(arena.hidden.data()), tokens.size(),
        config.width);
    return cuda_status(cudaPeekAtLastError(), "embedding lookup kernel");
  }

  [[nodiscard]] Status run_mixer(Layer *const layer, const std::size_t rows,
                                 const ForwardMode mode,
                                 const std::size_t index,
                                 const std::vector<LayerDump> &dumps) {
    auto status =
        layer->software_fp8_projection
            ? software_e4m3_h100_linear(
                  arena.normalized, layer->projection, rows, config.width,
                  config.width * 3, layer->projection_input_scale,
                  layer->projection_output_scale, &arena.mixer_scratch,
                  &arena.projection, stream)
            : bf16_linear(blas, arena.normalized, layer->projection, nullptr,
                          rows, config.width, config.width * 3,
                          &arena.projection, &arena.blas_workspace, stream,
                          &arena.projection_plan,
                          layer->type == MixerType::kAttention &&
                                  config.qkv_source_head_major
                              ? LinearWeightLayout::kInputMajor
                              : LinearWeightLayout::kOutputMajor);
    if (!status.ok())
      return status;
    status = dump_matching(dumps, index, LayerDumpPoint::kMixerInputProjection,
                           arena.projection, rows, config.width * 3);
    if (!status.ok())
      return status;
    if (layer->type == MixerType::kAttention) {
      status = bf16_split_qkv(
          arena.projection, rows, config.heads, config.head_dim(), &arena.x2,
          &arena.x1, &arena.value, stream,
          config.qkv_head_major ? QkvLayout::kHeadMajor
                                : QkvLayout::kProjectionMajor);
      if (!status.ok())
        return status;
      const std::size_t prefix = layer->kv_cache.length;
      status = bf16_apply_rope(&arena.x2, &arena.x1, layer->inverse_frequency,
                               rows, config.heads, config.head_dim(), prefix,
                               config.rope_scale, stream);
      if (!status.ok())
        return status;
      for (const auto &[point, buffer] :
           {std::pair{LayerDumpPoint::kMixerX2, &arena.x2},
            std::pair{LayerDumpPoint::kMixerX1, &arena.x1},
            std::pair{LayerDumpPoint::kMixerValue, &arena.value}}) {
        status =
            dump_matching(dumps, index, point, *buffer, rows, config.width);
        if (!status.ok())
          return status;
      }
      status =
          bf16_kv_append(arena.x1, arena.value, rows, &layer->kv_cache, stream);
      if (!status.ok())
        return status;
      if (uses_cached_attention(mode)) {
        if (layer->kv_cache.type != KvCacheType::kBF16) {
          return {ErrorCode::kUnsupported,
                  "exact Vortex cached attention requires a BF16 KV cache"};
        }
        return bf16_cached_cross_attention(
            cached_attention_blas, arena.x2, layer->kv_cache.key,
            layer->kv_cache.value, rows, layer->kv_cache.length, config.heads,
            config.head_dim(), &arena.attention_scaled_key,
            &arena.attention_scores, &arena.attention_probabilities,
            &arena.mixer_output, stream);
      }
      return layer->kv_cache.type == KvCacheType::kBF16 &&
                     config.head_dim() == 128
                 ? bf16_flash_causal_attention(
                       arena.x2, layer->kv_cache.key, layer->kv_cache.value,
                       rows, layer->kv_cache.length, config.heads,
                       config.head_dim(), &arena.mixer_scratch,
                       &arena.attention_lse_accum,
                       &arena.attention_output_accum, &arena.mixer_output,
                       stream)
                 : bf16_online_causal_attention(arena.x2, rows, prefix,
                                                layer->kv_cache,
                                                &arena.mixer_output, stream);
    }

    if (is_initial_prefill(mode)) {
      status = bf16_fir_prefill_direct(
          arena.projection, layer->short_filter, nullptr, rows,
          config.width * 3, config.width * 3, config.short_filter_length,
          FirOrientation::kCrossCorrelation, FirBiasMode::kAdd,
          &arena.short_filtered, &layer->short_cache, stream);
    } else if (mode == ForwardMode::kContinue) {
      status = bf16_fir_continue_direct(
          arena.projection, layer->short_filter, nullptr, rows,
          config.width * 3, config.width * 3, config.short_filter_length,
          FirOrientation::kCrossCorrelation, FirBiasMode::kAdd,
          &arena.short_filtered, &layer->short_cache, stream);
    } else {
      status = bf16_fir_decode(
          arena.projection, layer->short_filter, nullptr, config.width * 3,
          config.width * 3, config.short_filter_length,
          FirOrientation::kCrossCorrelation, FirBiasMode::kAdd,
          &layer->short_cache, &arena.short_filtered, stream);
    }
    if (!status.ok())
      return status;
    status = dump_matching(dumps, index, LayerDumpPoint::kMixerShortFilter,
                           arena.short_filtered, rows, config.width * 3);
    if (!status.ok())
      return status;
    if (!is_stateless_prefill(mode)) {
      status = dump_f32_tail_matching(
          dumps, index, LayerDumpPoint::kMixerShortState,
          layer->short_cache.state, config.width * 3,
          config.short_filter_length - 1, layer->short_cache.length);
      if (!status.ok())
        return status;
    }
    status =
        bf16_split_hyena_projection(arena.short_filtered, rows, config.width,
                                    &arena.x2, &arena.x1, &arena.value, stream);
    if (!status.ok())
      return status;
    for (const auto &[point, buffer] :
         {std::pair{LayerDumpPoint::kMixerX2, &arena.x2},
          std::pair{LayerDumpPoint::kMixerX1, &arena.x1},
          std::pair{LayerDumpPoint::kMixerValue, &arena.value}}) {
      status = dump_matching(dumps, index, point, *buffer, rows, config.width);
      if (!status.ok())
        return status;
    }
    if (layer->type == MixerType::kHcs) {
      if (is_initial_prefill(mode)) {
        status = bf16_hcs_prefill(
            arena.x2, arena.x1, arena.value, layer->inner_filter, rows,
            config.width, config.hcs_filter_groups, config.hcs_filter_length,
            &layer->inner_cache, &arena.mixer_scratch, &arena.mixer_output,
            stream);
      } else if (mode == ForwardMode::kContinue) {
        status = bf16_hcs_continue(
            arena.x2, arena.x1, arena.value, layer->inner_filter, rows,
            config.width, config.hcs_filter_groups, config.hcs_filter_length,
            &layer->inner_cache, &arena.mixer_scratch, &arena.mixer_output,
            stream);
      } else {
        status = bf16_hcs_decode(
            arena.x2, arena.x1, arena.value, layer->inner_filter, config.width,
            config.hcs_filter_groups, config.hcs_filter_length,
            &layer->inner_cache, &arena.mixer_scratch, &arena.mixer_output,
            stream);
      }
    } else if (layer->type == MixerType::kHcm) {
      const auto weight_type = config.hcm_filter_dtype == HcmFilterDType::kF32
                                   ? FirWeightType::kF32
                                   : FirWeightType::kBF16;
      if (is_initial_prefill(mode)) {
        if (!hcm_prefill_uses_fft(rows)) {
          status = bf16_hcm_prefill_direct(
              arena.x2, arena.x1, arena.value, layer->inner_filter,
              layer->direct, rows, config.width, config.width,
              config.hcm_filter_length, &layer->inner_cache,
              &arena.mixer_scratch, &arena.mixer_output, stream, weight_type);
        } else {
          status = bf16_hcm_prefill(
              arena.x2, arena.x1, arena.value, layer->inner_filter,
              layer->direct, rows, config.width, config.width,
              config.hcm_filter_length, &layer->inner_cache,
              &arena.mixer_scratch, &arena.mixer_output, &hcm_fft, stream,
              weight_type);
        }
      } else if (mode == ForwardMode::kContinue) {
        status = bf16_hcm_continue(
            arena.x2, arena.x1, arena.value, layer->inner_filter, layer->direct,
            rows, config.width, config.width, config.hcm_filter_length,
            &layer->inner_cache, &arena.mixer_scratch, &arena.mixer_output,
            stream, weight_type);
      } else {
        status = bf16_hcm_decode(
            arena.x2, arena.x1, arena.value, layer->inner_filter, layer->direct,
            config.width, config.width, config.hcm_filter_length,
            &layer->inner_cache, &arena.mixer_scratch, &arena.mixer_output,
            stream, weight_type);
      }
    } else if (mode == ForwardMode::kContinue) {
      status = bf16_hcl_prefill(
          arena.x2, arena.x1, arena.value, layer->direct, layer->log_poles,
          layer->residues, rows, config.width, config.state_size,
          HclPrefillMode::kRecurrenceContinue, &layer->iir_cache,
          &arena.mixer_scratch, &arena.mixer_output, nullptr, stream);
    } else if (is_initial_prefill(mode)) {
      status = bf16_hcl_prefill(
          arena.x2, arena.x1, arena.value, layer->direct, layer->log_poles,
          layer->residues, rows, config.width, config.state_size,
          is_stateless_prefill(mode) ? HclPrefillMode::kFftStateless
                                     : HclPrefillMode::kFft,
          &layer->iir_cache, &arena.mixer_scratch, &arena.mixer_output,
          &hcl_fft, stream);
    } else {
      status = bf16_hcl_decode(
          arena.x2, arena.x1, arena.value, layer->direct, layer->log_poles,
          layer->residues, config.width, config.state_size, &layer->iir_cache,
          &arena.mixer_scratch, &arena.mixer_output, stream);
    }
    if (!status.ok())
      return status;
    if ((layer->type == MixerType::kHcs || layer->type == MixerType::kHcm) &&
        !is_stateless_prefill(mode)) {
      status = dump_f32_tail_matching(
          dumps, index, LayerDumpPoint::kMixerInnerState,
          layer->inner_cache.state, config.width,
          layer->inner_cache.kernel_size - 1, layer->inner_cache.length);
      if (!status.ok())
        return status;
    }
    if (layer->type == MixerType::kHcl && !is_stateless_prefill(mode)) {
      status = dump_f32_matching(dumps, index, LayerDumpPoint::kMixerState,
                                 layer->iir_cache.state, config.width,
                                 config.state_size);
      if (!status.ok())
        return status;
    }
    status = dump_matching(dumps, index, LayerDumpPoint::kMixerPregate,
                           arena.mixer_scratch, rows, config.width);
    if (!status.ok())
      return status;
    if (layer->type == MixerType::kHcl && is_initial_prefill(mode)) {
      status = dump_channel_time_f32_matching(
          dumps, index, LayerDumpPoint::kMixerFilter, hcl_fft.filter_time(),
          config.width, rows, hcl_fft.fft_size());
      if (!status.ok())
        return status;
      return dump_channel_time_f32_matching(
          dumps, index, LayerDumpPoint::kMixerConvolution,
          hcl_fft.output_time(), config.width, rows, hcl_fft.fft_size());
    }
    return Status::Ok();
  }

  [[nodiscard]] Status dump_buffer(const DeviceBuffer &buffer,
                                   const LayerDump &dump,
                                   const std::size_t rows,
                                   const std::size_t columns) {
    std::vector<__nv_bfloat16> raw(rows * columns);
    auto status =
        buffer.copy_to_host(raw.data(), raw.size() * sizeof(raw[0]), stream);
    if (!status.ok())
      return status;
    status = stream.synchronize();
    if (!status.ok())
      return status;
    std::vector<float> values;
    values.reserve(raw.size());
    for (const auto value : raw)
      values.push_back(__bfloat162float(value));
    return write_npy(dump.path, values, rows, columns);
  }

  [[nodiscard]] Status dump_f32(const DeviceBuffer &buffer,
                                const LayerDump &dump, const std::size_t rows,
                                const std::size_t columns) {
    std::vector<float> values(rows * columns);
    auto status = buffer.copy_to_host(
        values.data(), values.size() * sizeof(values[0]), stream);
    if (!status.ok())
      return status;
    status = stream.synchronize();
    if (!status.ok())
      return status;
    return write_npy(dump.path, values, rows, columns);
  }

  [[nodiscard]] Status dump_f32_matching(const std::vector<LayerDump> &dumps,
                                         const std::size_t layer,
                                         const LayerDumpPoint point,
                                         const DeviceBuffer &buffer,
                                         const std::size_t rows,
                                         const std::size_t columns) {
    for (const auto &dump : dumps) {
      if (dump.layer == layer && dump.point == point) {
        auto status = dump_f32(buffer, dump, rows, columns);
        if (!status.ok())
          return status;
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] Status
  dump_f32_tail_matching(const std::vector<LayerDump> &dumps,
                         const std::size_t layer, const LayerDumpPoint point,
                         const DeviceBuffer &buffer, const std::size_t rows,
                         const std::size_t columns,
                         const std::size_t active_columns) {
    if (active_columns > columns) {
      return {ErrorCode::kInternal,
              "active FIR cache columns exceed cache capacity"};
    }
    for (const auto &dump : dumps) {
      if (dump.layer != layer || dump.point != point)
        continue;
      std::vector<float> padded(rows * columns);
      auto status = buffer.copy_to_host(
          padded.data(), padded.size() * sizeof(padded[0]), stream);
      if (!status.ok())
        return status;
      status = stream.synchronize();
      if (!status.ok())
        return status;
      std::vector<float> active(rows * active_columns);
      for (std::size_t row = 0; row < rows; ++row) {
        const auto source =
            padded.begin() + static_cast<std::ptrdiff_t>(
                                 row * columns + columns - active_columns);
        std::copy_n(source, active_columns,
                    active.begin() +
                        static_cast<std::ptrdiff_t>(row * active_columns));
      }
      status = write_npy(dump.path, active, rows, active_columns);
      if (!status.ok())
        return status;
    }
    return Status::Ok();
  }

  [[nodiscard]] Status dump_channel_time_f32(const DeviceBuffer &buffer,
                                             const LayerDump &dump,
                                             const std::size_t channels,
                                             const std::size_t length,
                                             const std::size_t stride) {
    std::vector<float> channel_major(channels * stride);
    auto status = buffer.copy_to_host(
        channel_major.data(), channel_major.size() * sizeof(float), stream);
    if (!status.ok())
      return status;
    status = stream.synchronize();
    if (!status.ok())
      return status;
    std::vector<float> time_major(length * channels);
    for (std::size_t time = 0; time < length; ++time) {
      for (std::size_t channel = 0; channel < channels; ++channel) {
        time_major[time * channels + channel] =
            channel_major[channel * stride + time];
      }
    }
    return write_npy(dump.path, time_major, length, channels);
  }

  [[nodiscard]] Status dump_channel_time_f32_matching(
      const std::vector<LayerDump> &dumps, const std::size_t layer,
      const LayerDumpPoint point, const DeviceBuffer &buffer,
      const std::size_t channels, const std::size_t length,
      const std::size_t stride) {
    for (const auto &dump : dumps) {
      if (dump.layer == layer && dump.point == point) {
        auto status =
            dump_channel_time_f32(buffer, dump, channels, length, stride);
        if (!status.ok())
          return status;
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] Status
  dump_matching(const std::vector<LayerDump> &dumps, const std::size_t layer,
                const LayerDumpPoint point, const DeviceBuffer &buffer,
                const std::size_t rows, const std::size_t columns) {
    for (const auto &dump : dumps) {
      if (dump.layer == layer && dump.point == point) {
        auto status = dump_buffer(buffer, dump, rows, columns);
        if (!status.ok())
          return status;
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] Status run_blocks(const std::size_t rows,
                                  const ForwardMode mode,
                                  const std::vector<LayerDump> &dumps = {}) {
    auto status = Status::Ok();
    for (std::size_t local_index = 0; local_index < layers.size();
         ++local_index) {
      const std::size_t index = layer_offset + local_index;
      Layer &layer = layers[local_index];
      status = bf16_rms_norm(arena.hidden, layer.pre_norm, rows, config.width,
                             config.epsilon, &arena.normalized, stream);
      if (!status.ok())
        break;
      status = dump_matching(dumps, index, LayerDumpPoint::kPreNorm,
                             arena.normalized, rows, config.width);
      if (!status.ok())
        break;
      status = run_mixer(&layer, rows, mode, index, dumps);
      if (!status.ok())
        break;
      status = dump_matching(dumps, index, LayerDumpPoint::kMixerOutput,
                             arena.mixer_output, rows, config.width);
      if (!status.ok())
        break;
      const bool hyena_single_row =
          layer.type != MixerType::kAttention && rows == 1;
      const DeviceBuffer *const fused_output_bias =
          layer.type == MixerType::kAttention || hyena_single_row
              ? &layer.output_bias
              : nullptr;
      const bool hyena_noncontiguous =
          layer.type != MixerType::kAttention && rows > 1;
      const DeviceBuffer *projection_input = &arena.mixer_output;
      auto projection_input_layout = LinearInputLayout::kRowMajor;
      if (hyena_noncontiguous) {
        status =
            bf16_row_to_column_major(arena.mixer_output, rows, config.width,
                                     &arena.mixer_scratch, stream);
        if (!status.ok())
          break;
        projection_input = &arena.mixer_scratch;
        projection_input_layout = LinearInputLayout::kColumnMajor;
      }
      status = bf16_linear(blas, *projection_input, layer.output_weight,
                           fused_output_bias, rows, config.width, config.width,
                           &arena.residual, &arena.blas_workspace, stream,
                           &arena.output_plan, LinearWeightLayout::kOutputMajor,
                           projection_input_layout);
      if (!status.ok())
        break;
      if (fused_output_bias == nullptr) {
        status = bf16_add_bias_inplace(&arena.residual, layer.output_bias, rows,
                                       config.width, stream);
        if (!status.ok())
          break;
      }
      status = dump_matching(dumps, index, LayerDumpPoint::kMixerProjection,
                             arena.residual, rows, config.width);
      if (!status.ok())
        break;
      status = bf16_add_inplace(&arena.residual, arena.hidden,
                                rows * config.width, stream);
      if (!status.ok())
        break;
      status = dump_matching(dumps, index, LayerDumpPoint::kMixerResidual,
                             arena.residual, rows, config.width);
      if (!status.ok())
        break;
      status =
          bf16_rms_norm(arena.residual, layer.post_norm, rows, config.width,
                        config.epsilon, &arena.normalized, stream);
      if (!status.ok())
        break;
      status = dump_matching(dumps, index, LayerDumpPoint::kPostNorm,
                             arena.normalized, rows, config.width);
      if (!status.ok())
        break;
      const auto activation =
          index == 0 ? GatedActivation::kGelu : GatedActivation::kIdentity;
      status = bf16_mlp(blas, arena.normalized, layer.l1, layer.l2, layer.l3,
                        rows, config.width, config.inner_width, activation,
                        &arena.mlp, &arena.hidden, stream);
      if (!status.ok())
        break;
      status = dump_matching(dumps, index, LayerDumpPoint::kMlpL1,
                             arena.mlp.first, rows, config.inner_width);
      if (!status.ok())
        break;
      status = dump_matching(dumps, index, LayerDumpPoint::kMlpL2,
                             arena.mlp.second, rows, config.inner_width);
      if (!status.ok())
        break;
      if (activation == GatedActivation::kGelu) {
        status = dump_matching(dumps, index, LayerDumpPoint::kMlpActivation,
                               arena.mlp.activated, rows, config.inner_width);
        if (!status.ok())
          break;
      }
      status = dump_matching(dumps, index, LayerDumpPoint::kMlpGated,
                             arena.mlp.gated, rows, config.inner_width);
      if (!status.ok())
        break;
      status = dump_matching(dumps, index, LayerDumpPoint::kMlpOutput,
                             arena.hidden, rows, config.width);
      if (!status.ok())
        break;
      status = bf16_add_inplace(&arena.hidden, arena.residual,
                                rows * config.width, stream);
      if (!status.ok())
        break;
      status = dump_matching(dumps, index, LayerDumpPoint::kBlockOutput,
                             arena.hidden, rows, config.width);
      if (!status.ok())
        break;
    }
    if (!status.ok())
      return {status.code(), "model block forward: " + status.message()};
    return Status::Ok();
  }

  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const ForwardMode mode,
                               std::vector<float> *const logits,
                               const std::vector<LayerDump> &dumps) {
    if (!loaded || logits == nullptr || tokens.empty() ||
        tokens.size() > arena_capacity) {
      return {ErrorCode::kInvalidArgument,
              "model forward arguments are invalid"};
    }
    if (is_initial_prefill(mode)) {
      auto status = prepare_prefill(tokens.size());
      if (!status.ok())
        return status;
    } else if (!state_valid || position > context_capacity ||
               tokens.size() > context_capacity - position ||
               (is_decode(mode) && tokens.size() != 1)) {
      return {ErrorCode::kInvalidArgument,
              "continuation requires a valid prefill and free context "
              "capacity"};
    }
    for (const auto &dump : dumps) {
      if (dump.layer >= config.layers || dump.path.empty() ||
          !valid_dump_point(dump.point))
        return {ErrorCode::kInvalidArgument,
                "layer dump index or path is invalid"};
    }
    auto status = embed(tokens);
    if (!status.ok()) {
      state_valid = false;
      return status;
    }
    const std::size_t rows = tokens.size();
    status = run_blocks(rows, mode, dumps);
    if (!status.ok()) {
      state_valid = false;
      return status;
    }
    status = bf16_rms_norm(arena.hidden, final_norm, rows, config.width,
                           config.epsilon, &arena.normalized, stream);
    if (status.ok()) {
      status = bf16_linear(blas, arena.normalized, embedding, nullptr, rows,
                           config.width, config.vocab_size, &arena.logits,
                           &arena.blas_workspace, stream, &arena.final_plan);
    }
    if (!status.ok()) {
      state_valid = false;
      return {status.code(), "final projection: " + status.message()};
    }
    std::vector<__nv_bfloat16> raw(rows * config.vocab_size);
    status = arena.logits.copy_to_host(raw.data(), raw.size() * sizeof(raw[0]),
                                       stream);
    if (status.ok())
      status = stream.synchronize();
    if (!status.ok()) {
      state_valid = false;
      return status;
    }
    logits->clear();
    logits->reserve(raw.size());
    for (const auto value : raw)
      logits->push_back(__bfloat162float(value));
    position = is_stateless_prefill(mode)
                   ? 0
                   : (is_initial_prefill(mode) ? rows : position + rows);
    state_valid = !is_stateless_prefill(mode);
    if (is_initial_prefill(mode))
      cached_attention = uses_cached_attention(mode);
    return Status::Ok();
  }
};

SingleGpuModel::SingleGpuModel() : impl_(std::make_unique<Impl>()) {}
SingleGpuModel::~SingleGpuModel() = default;
SingleGpuModel::SingleGpuModel(SingleGpuModel &&) noexcept = default;
SingleGpuModel &SingleGpuModel::operator=(SingleGpuModel &&) noexcept = default;

Status SingleGpuModel::load(const ModelFile &model, const int device,
                            const std::size_t context_capacity,
                            const bool allow_test_fixture) {
  if (impl_->loaded || context_capacity == 0)
    return {ErrorCode::kInvalidArgument,
            "model is already loaded or context is zero"};
  auto status =
      read_runtime_model_config(model, allow_test_fixture, &impl_->config);
  if (!status.ok())
    return status;
  if (!impl_->config.test_fixture &&
      context_capacity > impl_->config.max_seqlen) {
    return {ErrorCode::kInvalidArgument,
            "requested context exceeds model maximum of " +
                std::to_string(impl_->config.max_seqlen)};
  }
  impl_->device = device;
  impl_->context_capacity = context_capacity;
  impl_->arena_capacity = std::min(
      context_capacity, impl_->config.test_fixture ? kTestFixtureArenaTokens
                                                   : kMaximumArenaTokens);
  impl_->q8_kv_cache = context_capacity >= kQ8KvContextThreshold;
  status = select_device(device);
  if (!status.ok())
    return status;
  status = impl_->stream.create();
  if (!status.ok())
    return status;
  status = impl_->cached_attention_blas.create();
  if (!status.ok())
    return status;
  status = impl_->blas.create();
  if (!status.ok())
    return status;
  status = upload_tensor(model, "embedding_layer.weight", TensorDType::kBF16,
                         {impl_->config.vocab_size, impl_->config.width},
                         device, impl_->stream, &impl_->embedding);
  if (!status.ok())
    return status;
  status = upload_tensor(model, "unembed.weight", TensorDType::kBF16,
                         {impl_->config.vocab_size, impl_->config.width},
                         device, impl_->stream, &impl_->unembed);
  if (!status.ok())
    return status;
  status = upload_tensor(model, "norm.scale", TensorDType::kF32,
                         {impl_->config.width}, device, impl_->stream,
                         &impl_->final_norm);
  if (!status.ok())
    return status;
  impl_->layers.reserve(impl_->config.layers);
  for (std::size_t index = 0; index < impl_->config.layers; ++index) {
    impl_->layers.emplace_back();
    status = impl_->load_layer(model, index, &impl_->layers.back());
    if (!status.ok())
      return {status.code(),
              "load block " + std::to_string(index) + ": " + status.message()};
  }
  status = impl_->allocate_arena();
  if (!status.ok())
    return status;
  status = impl_->stream.synchronize();
  if (!status.ok())
    return status;
  impl_->loaded = true;
  const auto warmup_tokens =
      backend_warmup_tokens(impl_->config, impl_->arena_capacity);
  if (warmup_tokens != 0) {
    const std::vector<TokenId> tokens(warmup_tokens, static_cast<TokenId>('A'));
    std::vector<float> logits;
    status = impl_->forward(tokens, ForwardMode::kPrefill, &logits, {});
    if (!status.ok()) {
      impl_->loaded = false;
      return {status.code(), "warm up CUDA backend: " + status.message()};
    }
    impl_->position = 0;
    impl_->state_valid = false;
    impl_->cached_attention = false;
  }
  return Status::Ok();
}

Status SingleGpuModel::prefill(const std::vector<TokenId> &tokens,
                               std::vector<float> *const logits,
                               const std::optional<LayerDump> &dump) {
  std::vector<LayerDump> dumps;
  if (dump.has_value())
    dumps.push_back(*dump);
  return impl_->forward(tokens, ForwardMode::kPrefill, logits, dumps);
}

Status SingleGpuModel::prefill_with_dumps(const std::vector<TokenId> &tokens,
                                          std::vector<float> *const logits,
                                          const std::vector<LayerDump> &dumps) {
  return impl_->forward(tokens, ForwardMode::kPrefill, logits, dumps);
}

Status SingleGpuModel::prefill_stateless(const std::vector<TokenId> &tokens,
                                         std::vector<float> *const logits) {
  return impl_->forward(tokens, ForwardMode::kStatelessPrefill, logits, {});
}

Status SingleGpuModel::prefill_cached(const std::vector<TokenId> &tokens,
                                      std::vector<float> *const logits) {
  return impl_->forward(tokens, ForwardMode::kCachedPrefill, logits, {});
}

Status
SingleGpuModel::prefill_cached_with_dumps(const std::vector<TokenId> &tokens,
                                          std::vector<float> *const logits,
                                          const std::vector<LayerDump> &dumps) {
  return impl_->forward(tokens, ForwardMode::kCachedPrefill, logits, dumps);
}

Status SingleGpuModel::prefill_chunk(const std::vector<TokenId> &tokens,
                                     std::vector<float> *const logits) {
  if (impl_->cached_attention) {
    return {ErrorCode::kInvalidArgument,
            "cached prefill can only be continued with exact token decode"};
  }
  return impl_->forward(tokens, ForwardMode::kContinue, logits, {});
}

Status SingleGpuModel::decode(const TokenId token,
                              std::vector<float> *const logits) {
  const auto mode = impl_->cached_attention ? ForwardMode::kCachedDecode
                                            : ForwardMode::kDecode;
  return impl_->forward(std::vector<TokenId>{token}, mode, logits, {});
}

Status SingleGpuModel::decode_with_dumps(const TokenId token,
                                         std::vector<float> *const logits,
                                         const std::vector<LayerDump> &dumps) {
  const auto mode = impl_->cached_attention ? ForwardMode::kCachedDecode
                                            : ForwardMode::kDecode;
  return impl_->forward(std::vector<TokenId>{token}, mode, logits, dumps);
}

const RuntimeModelConfig &SingleGpuModel::config() const noexcept {
  return impl_->config;
}

std::size_t SingleGpuModel::position() const noexcept {
  return impl_->position;
}
std::size_t SingleGpuModel::activation_capacity() const noexcept {
  return impl_->arena_capacity;
}
bool SingleGpuModel::uses_q8_kv_cache() const noexcept {
  return impl_->q8_kv_cache;
}
int SingleGpuModel::device() const noexcept { return impl_->device; }

struct PipelineModel::Impl final {
  RuntimeModelConfig config;
  std::size_t context_capacity{0};
  std::size_t arena_capacity{0};
  std::size_t position{0};
  bool q8_kv_cache{false};
  bool loaded{false};
  bool state_valid{false};
  bool cached_attention{false};
  std::vector<StageAssignment> assignments;
  std::vector<std::unique_ptr<SingleGpuModel::Impl>> stages;

  [[nodiscard]] Status enable_stage_peers() const {
    if (assignments.size() == 1)
      return Status::Ok();
    std::set<std::pair<int, int>> enabled;
    for (std::size_t index = 0; index < assignments.size(); ++index) {
      const int source = assignments[index].device;
      const int destination =
          assignments[(index + 1) % assignments.size()].device;
      for (const auto &pair :
           {std::pair{source, destination}, std::pair{destination, source}}) {
        if (!enabled.insert(pair).second)
          continue;
        auto status = enable_peer_access(pair.first, pair.second);
        if (!status.ok())
          return status;
      }
    }
    return Status::Ok();
  }

  [[nodiscard]] Status transfer_hidden(SingleGpuModel::Impl *const source,
                                       SingleGpuModel::Impl *const destination,
                                       const std::size_t rows) const {
    std::size_t elements = 0;
    std::size_t bytes = 0;
    if (!multiply(rows, config.width, &elements) ||
        !multiply(elements, sizeof(__nv_bfloat16), &bytes)) {
      return {ErrorCode::kInvalidArgument,
              "stage-boundary activation dimensions overflow"};
    }
    auto status = source->stream.synchronize();
    if (!status.ok())
      return {status.code(),
              "synchronize source pipeline stage: " + status.message()};
    status = destination->arena.hidden.copy_from_peer(
        source->arena.hidden, bytes, destination->stream);
    if (!status.ok()) {
      return {status.code(), "P2P activation transfer from CUDA device " +
                                 std::to_string(source->device) + " to " +
                                 std::to_string(destination->device) + ": " +
                                 status.message()};
    }
    return Status::Ok();
  }

  [[nodiscard]] Status forward(const std::vector<TokenId> &tokens,
                               const ForwardMode mode,
                               std::vector<float> *const logits,
                               const std::vector<LayerDump> &dumps) {
    if (!loaded || logits == nullptr || tokens.empty() ||
        tokens.size() > arena_capacity) {
      return {ErrorCode::kInvalidArgument,
              "pipeline model forward arguments are invalid"};
    }
    if (is_initial_prefill(mode)) {
      position = 0;
      state_valid = false;
      for (auto &stage : stages) {
        auto status = stage->prepare_prefill(tokens.size());
        if (!status.ok())
          return {status.code(), "prepare pipeline stage on CUDA device " +
                                     std::to_string(stage->device) + ": " +
                                     status.message()};
      }
    } else if (!state_valid || position > context_capacity ||
               tokens.size() > context_capacity - position ||
               (is_decode(mode) && tokens.size() != 1)) {
      return {ErrorCode::kInvalidArgument,
              "pipeline continuation requires a valid prefill and free "
              "context capacity"};
    }
    for (const auto &dump : dumps) {
      if (dump.layer >= config.layers || dump.path.empty() ||
          !valid_dump_point(dump.point))
        return {ErrorCode::kInvalidArgument,
                "layer dump index or path is invalid"};
    }

    auto status = stages.front()->embed(tokens);
    if (!status.ok()) {
      state_valid = false;
      return status;
    }
    const std::size_t rows = tokens.size();
    for (std::size_t stage_index = 0; stage_index < stages.size();
         ++stage_index) {
      if (stage_index != 0) {
        status = transfer_hidden(stages[stage_index - 1].get(),
                                 stages[stage_index].get(), rows);
        if (!status.ok())
          break;
      }
      status = stages[stage_index]->run_blocks(rows, mode, dumps);
      if (!status.ok()) {
        status = {status.code(),
                  "pipeline stage " + std::to_string(stage_index) + " [" +
                      std::to_string(assignments[stage_index].layer_begin) +
                      "," + std::to_string(assignments[stage_index].layer_end) +
                      "): " + status.message()};
        break;
      }
    }
    if (status.ok() && stages.size() > 1) {
      status = transfer_hidden(stages.back().get(), stages.front().get(), rows);
    }
    if (!status.ok()) {
      state_valid = false;
      return status;
    }

    auto &head = *stages.front();
    status =
        bf16_rms_norm(head.arena.hidden, head.final_norm, rows, config.width,
                      config.epsilon, &head.arena.normalized, head.stream);
    if (status.ok()) {
      status = bf16_linear(head.blas, head.arena.normalized, head.embedding,
                           nullptr, rows, config.width, config.vocab_size,
                           &head.arena.logits, &head.arena.blas_workspace,
                           head.stream, &head.arena.final_plan);
    }
    if (!status.ok()) {
      state_valid = false;
      return {status.code(), "pipeline final projection: " + status.message()};
    }
    std::vector<__nv_bfloat16> raw(rows * config.vocab_size);
    status = head.arena.logits.copy_to_host(
        raw.data(), raw.size() * sizeof(raw[0]), head.stream);
    if (status.ok())
      status = head.stream.synchronize();
    if (!status.ok()) {
      state_valid = false;
      return status;
    }
    logits->clear();
    logits->reserve(raw.size());
    for (const auto value : raw)
      logits->push_back(__bfloat162float(value));
    position = is_stateless_prefill(mode)
                   ? 0
                   : (is_initial_prefill(mode) ? rows : position + rows);
    state_valid = !is_stateless_prefill(mode);
    if (is_initial_prefill(mode))
      cached_attention = uses_cached_attention(mode);
    return Status::Ok();
  }
};

PipelineModel::PipelineModel() : impl_(std::make_unique<Impl>()) {}
PipelineModel::~PipelineModel() = default;
PipelineModel::PipelineModel(PipelineModel &&) noexcept = default;
PipelineModel &PipelineModel::operator=(PipelineModel &&) noexcept = default;

Status PipelineModel::load(const ModelFile &model,
                           const std::vector<int> &devices,
                           const std::size_t context_capacity,
                           const bool allow_test_fixture) {
  if (impl_->loaded || context_capacity == 0 || devices.empty() ||
      devices.size() > 4) {
    return {ErrorCode::kInvalidArgument,
            "pipeline load requires one to four devices, nonzero context, "
            "and an unloaded model"};
  }
  const std::set<int> unique_devices(devices.begin(), devices.end());
  if (unique_devices.size() != devices.size()) {
    return {ErrorCode::kInvalidArgument,
            "pipeline CUDA device list contains duplicates"};
  }

  auto candidate = std::make_unique<Impl>();
  auto status =
      read_runtime_model_config(model, allow_test_fixture, &candidate->config);
  if (!status.ok())
    return status;
  if (!candidate->config.test_fixture &&
      context_capacity > candidate->config.max_seqlen) {
    return {ErrorCode::kInvalidArgument,
            "requested context exceeds model maximum of " +
                std::to_string(candidate->config.max_seqlen)};
  }
  candidate->context_capacity = context_capacity;
  candidate->arena_capacity = std::min(
      context_capacity, candidate->config.test_fixture ? kTestFixtureArenaTokens
                                                       : kMaximumArenaTokens);
  candidate->q8_kv_cache = context_capacity >= kQ8KvContextThreshold;
  const std::size_t stage_count = devices.size();
  std::vector<std::size_t> layer_ends;
  layer_ends.reserve(stage_count);
  if (candidate->config.test_fixture) {
    const std::size_t base_layers = candidate->config.layers / stage_count;
    const std::size_t extra_layers = candidate->config.layers % stage_count;
    std::size_t end = 0;
    for (std::size_t index = 0; index < stage_count; ++index) {
      end += base_layers + (index < extra_layers ? 1 : 0);
      layer_ends.push_back(end);
    }
  } else {
    std::vector<std::size_t> layer_bytes(candidate->config.layers, 0);
    for (std::size_t layer = 0; layer < candidate->config.layers; ++layer) {
      const std::string prefix = "blocks." + std::to_string(layer) + ".";
      for (const auto &tensor : model.tensors()) {
        if (tensor.name.compare(0, prefix.size(), prefix) == 0) {
          if (tensor.data_size >
              std::numeric_limits<std::size_t>::max() - layer_bytes[layer]) {
            return {ErrorCode::kModelFormat,
                    "layer tensor bytes overflow size_t"};
          }
          layer_bytes[layer] += static_cast<std::size_t>(tensor.data_size);
        }
      }
    }
    std::size_t remaining_bytes = 0;
    for (const auto bytes : layer_bytes)
      remaining_bytes += bytes;
    std::size_t begin = 0;
    for (std::size_t stage = 0; stage < stage_count; ++stage) {
      const std::size_t remaining_stages = stage_count - stage;
      if (remaining_stages == 1) {
        layer_ends.push_back(candidate->config.layers);
        break;
      }
      const std::size_t target =
          (remaining_bytes + remaining_stages - 1) / remaining_stages;
      const std::size_t latest_end =
          candidate->config.layers - (remaining_stages - 1);
      std::size_t end = begin;
      std::size_t assigned = 0;
      while (end < latest_end && (end == begin || assigned < target)) {
        assigned += layer_bytes[end];
        ++end;
      }
      layer_ends.push_back(end);
      remaining_bytes -= assigned;
      begin = end;
    }
  }
  std::size_t layer_begin = 0;
  candidate->assignments.reserve(stage_count);
  candidate->stages.reserve(stage_count);
  for (std::size_t index = 0; index < stage_count; ++index) {
    const std::size_t layer_end = layer_ends[index];
    const std::size_t layer_count = layer_end - layer_begin;
    candidate->assignments.push_back(
        {devices[index], layer_begin, layer_end, 0, 0, 0});
    auto stage = std::make_unique<SingleGpuModel::Impl>();
    stage->config = candidate->config;
    stage->device = devices[index];
    stage->context_capacity = context_capacity;
    stage->arena_capacity = candidate->arena_capacity;
    stage->q8_kv_cache = candidate->q8_kv_cache;
    stage->layer_offset = layer_begin;
    status = select_device(stage->device);
    if (!status.ok())
      return status;
    status = stage->stream.create();
    if (!status.ok())
      return status;
    status = stage->cached_attention_blas.create();
    if (!status.ok())
      return status;
    status = stage->blas.create();
    if (!status.ok())
      return status;
    stage->layers.reserve(layer_count);
    candidate->stages.push_back(std::move(stage));
    layer_begin = layer_end;
  }
  status = candidate->enable_stage_peers();
  if (!status.ok())
    return status;

  auto &head = *candidate->stages.front();
  status =
      upload_tensor(model, "embedding_layer.weight", TensorDType::kBF16,
                    {candidate->config.vocab_size, candidate->config.width},
                    head.device, head.stream, &head.embedding);
  if (!status.ok())
    return status;
  status =
      upload_tensor(model, "unembed.weight", TensorDType::kBF16,
                    {candidate->config.vocab_size, candidate->config.width},
                    head.device, head.stream, &head.unembed);
  if (!status.ok())
    return status;
  status = upload_tensor(model, "norm.scale", TensorDType::kF32,
                         {candidate->config.width}, head.device, head.stream,
                         &head.final_norm);
  if (!status.ok())
    return status;

  auto load_stage = [&model, pipeline = candidate.get()](
                        const std::size_t stage_index) -> Status {
    auto &stage = *pipeline->stages[stage_index];
    const auto &assignment = pipeline->assignments[stage_index];
    for (std::size_t index = assignment.layer_begin;
         index < assignment.layer_end; ++index) {
      stage.layers.emplace_back();
      auto stage_status = stage.load_layer(model, index, &stage.layers.back());
      if (!stage_status.ok()) {
        return {stage_status.code(), "load block " + std::to_string(index) +
                                         " on CUDA device " +
                                         std::to_string(stage.device) + ": " +
                                         stage_status.message()};
      }
    }
    auto stage_status = stage.allocate_arena();
    if (!stage_status.ok()) {
      return {stage_status.code(), "allocate arena on CUDA device " +
                                       std::to_string(stage.device) + ": " +
                                       stage_status.message()};
    }
    auto &memory = pipeline->assignments[stage_index];
    for (const auto &layer : stage.layers) {
      memory.weight_bytes += layer_weight_bytes(layer);
      memory.cache_bytes += layer_cache_bytes(layer);
    }
    memory.arena_bytes = arena_bytes(stage.arena);
    return stage.stream.synchronize();
  };
  std::vector<std::future<Status>> stage_loads;
  stage_loads.reserve(stage_count);
  try {
    for (std::size_t stage_index = 0; stage_index < stage_count;
         ++stage_index) {
      stage_loads.push_back(
          std::async(std::launch::async, load_stage, stage_index));
    }
    for (std::size_t stage_index = 0; stage_index < stage_count;
         ++stage_index) {
      status = stage_loads[stage_index].get();
      if (!status.ok()) {
        return {status.code(), "load pipeline stage " +
                                   std::to_string(stage_index) + ": " +
                                   status.message()};
      }
    }
  } catch (const std::exception &error) {
    return {ErrorCode::kInternal,
            "parallel pipeline load failed: " + std::string{error.what()}};
  }
  candidate->assignments.front().weight_bytes +=
      total_bytes({&head.embedding, &head.unembed, &head.final_norm});
  candidate->loaded = true;
  const auto warmup_tokens =
      backend_warmup_tokens(candidate->config, candidate->arena_capacity);
  if (warmup_tokens != 0) {
    const std::vector<TokenId> tokens(warmup_tokens, static_cast<TokenId>('A'));
    std::vector<float> logits;
    status = candidate->forward(tokens, ForwardMode::kPrefill, &logits, {});
    if (!status.ok()) {
      return {status.code(), "warm up CUDA pipeline: " + status.message()};
    }
    candidate->position = 0;
    candidate->state_valid = false;
    candidate->cached_attention = false;
  }
  impl_ = std::move(candidate);
  return Status::Ok();
}

Status PipelineModel::prefill(const std::vector<TokenId> &tokens,
                              std::vector<float> *const logits,
                              const std::optional<LayerDump> &dump) {
  std::vector<LayerDump> dumps;
  if (dump.has_value())
    dumps.push_back(*dump);
  return impl_->forward(tokens, ForwardMode::kPrefill, logits, dumps);
}

Status PipelineModel::prefill_with_dumps(const std::vector<TokenId> &tokens,
                                         std::vector<float> *const logits,
                                         const std::vector<LayerDump> &dumps) {
  return impl_->forward(tokens, ForwardMode::kPrefill, logits, dumps);
}

Status PipelineModel::prefill_stateless(const std::vector<TokenId> &tokens,
                                        std::vector<float> *const logits) {
  return impl_->forward(tokens, ForwardMode::kStatelessPrefill, logits, {});
}

Status PipelineModel::prefill_cached(const std::vector<TokenId> &tokens,
                                     std::vector<float> *const logits) {
  return impl_->forward(tokens, ForwardMode::kCachedPrefill, logits, {});
}

Status
PipelineModel::prefill_cached_with_dumps(const std::vector<TokenId> &tokens,
                                         std::vector<float> *const logits,
                                         const std::vector<LayerDump> &dumps) {
  return impl_->forward(tokens, ForwardMode::kCachedPrefill, logits, dumps);
}

Status PipelineModel::prefill_chunk(const std::vector<TokenId> &tokens,
                                    std::vector<float> *const logits) {
  if (impl_->cached_attention) {
    return {ErrorCode::kInvalidArgument,
            "cached prefill can only be continued with exact token decode"};
  }
  return impl_->forward(tokens, ForwardMode::kContinue, logits, {});
}

Status PipelineModel::decode(const TokenId token,
                             std::vector<float> *const logits) {
  const auto mode = impl_->cached_attention ? ForwardMode::kCachedDecode
                                            : ForwardMode::kDecode;
  return impl_->forward(std::vector<TokenId>{token}, mode, logits, {});
}

Status PipelineModel::decode_with_dumps(const TokenId token,
                                        std::vector<float> *const logits,
                                        const std::vector<LayerDump> &dumps) {
  const auto mode = impl_->cached_attention ? ForwardMode::kCachedDecode
                                            : ForwardMode::kDecode;
  return impl_->forward(std::vector<TokenId>{token}, mode, logits, dumps);
}

const RuntimeModelConfig &PipelineModel::config() const noexcept {
  return impl_->config;
}

const std::vector<StageAssignment> &PipelineModel::stages() const noexcept {
  return impl_->assignments;
}

void PipelineModel::refresh_cache_bytes() noexcept {
  for (std::size_t index = 0; index < impl_->stages.size(); ++index) {
    auto &assignment = impl_->assignments[index];
    assignment.cache_bytes = 0;
    for (const auto &layer : impl_->stages[index]->layers)
      assignment.cache_bytes += layer_cache_bytes(layer);
  }
}

std::size_t PipelineModel::position() const noexcept { return impl_->position; }

std::size_t PipelineModel::activation_capacity() const noexcept {
  return impl_->arena_capacity;
}

bool PipelineModel::uses_q8_kv_cache() const noexcept {
  return impl_->q8_kv_cache;
}

} // namespace evo::cuda
