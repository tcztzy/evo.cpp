// SPDX-License-Identifier: Apache-2.0
#include "evo2c/cuda/model.hpp"

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
#include <utility>

#include <cuda_bf16.h>

#include "evo2c/cuda/attention.hpp"
#include "evo2c/cuda/hyena.hpp"
#include "evo2c/cuda/ops.hpp"
#include "evo2c/cuda/runtime.hpp"

namespace evo2c::cuda {
namespace {

constexpr int kThreads = 256;
constexpr std::array<std::size_t, 14> kHcsLayers{0,  4,  7,  11, 14, 18, 21,
                                                 25, 28, 32, 36, 39, 43, 46};
constexpr std::array<std::size_t, 14> kHcmLayers{1,  5,  8,  12, 15, 19, 22,
                                                 26, 29, 33, 37, 40, 44, 47};
constexpr std::array<std::size_t, 14> kHclLayers{2,  6,  9,  13, 16, 20, 23,
                                                 27, 30, 34, 38, 41, 45, 48};
constexpr std::array<std::size_t, 8> kAttentionLayers{3,  10, 17, 24,
                                                      31, 35, 42, 49};

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

template <std::size_t Size>
bool same_indices(const std::vector<std::size_t> &actual,
                  const std::array<std::size_t, Size> &expected) {
  return actual.size() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

bool valid_dump_point(const LayerDumpPoint point) noexcept {
  return point == LayerDumpPoint::kBlockOutput ||
         point == LayerDumpPoint::kPreNorm ||
         point == LayerDumpPoint::kMixerOutput ||
         point == LayerDumpPoint::kMixerResidual ||
         point == LayerDumpPoint::kPostNorm ||
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

Status projection_fp8_scales(const ModelFile &model,
                             const std::string &prefix,
                             float *const input_scale,
                             float *const weight_scale,
                             float *const output_scale) {
  const TensorInfo *scale_tensor = nullptr;
  const TensorInfo *inverse_tensor = nullptr;
  const TensorInfo *history_tensor = nullptr;
  auto status = checked_tensor(
      model, prefix + ".fp8_scale_fwd", TensorDType::kF32, {2},
      &scale_tensor);
  if (!status.ok())
    return status;
  status = checked_tensor(
      model, prefix + ".fp8_scale_inv_fwd", TensorDType::kF32, {2},
      &inverse_tensor);
  if (!status.ok())
    return status;
  status = checked_tensor(
      model, prefix + ".fp8_amax_history_fwd", TensorDType::kF32, {16, 2},
      &history_tensor);
  if (!status.ok())
    return status;

  std::array<float, 2> scales{};
  std::array<float, 2> inverse{};
  std::array<float, 32> history{};
  std::memcpy(scales.data(), model.tensor_data(*scale_tensor),
              sizeof(scales));
  std::memcpy(inverse.data(), model.tensor_data(*inverse_tensor),
              sizeof(inverse));
  std::memcpy(history.data(), model.tensor_data(*history_tensor),
              sizeof(history));
  if (std::any_of(scales.begin(), scales.end(), [](const float value) {
        return !std::isfinite(value) || value <= 0.0F;
      }) ||
      std::any_of(inverse.begin(), inverse.end(), [](const float value) {
        return !std::isfinite(value) || value <= 0.0F;
      }) ||
      std::any_of(history.begin(), history.end(), [](const float value) {
        return !std::isfinite(value) || value < 0.0F;
      })) {
    return {ErrorCode::kModelFormat,
            prefix + " contains invalid software-FP8 scale state"};
  }
  if (inverse[0] != 1.0F / scales[0] ||
      inverse[1] != 1.0F / scales[1]) {
    return {ErrorCode::kModelFormat,
            prefix + " inverse scales do not match forward scales"};
  }
  const float combined_inverse = inverse[0] * inverse[1];
  if (!std::isfinite(combined_inverse) || combined_inverse <= 0.0F) {
    return {ErrorCode::kModelFormat,
            prefix + " combined inverse scale is invalid"};
  }
  *input_scale = scales[0];
  *weight_scale = scales[1];
  *output_scale = combined_inverse;
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
  DeviceBuffer logits;
  DeviceBuffer blas_workspace;
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
                      &layer.iir_cache.state, &layer.kv_cache.key,
                      &layer.kv_cache.value});
}

std::size_t arena_bytes(const Arena &arena) noexcept {
  return total_bytes({&arena.token_ids, &arena.hidden, &arena.residual,
                      &arena.normalized, &arena.projection,
                      &arena.short_filtered, &arena.x2, &arena.x1, &arena.value,
                      &arena.mixer_scratch, &arena.mixer_output, &arena.logits,
                      &arena.blas_workspace, &arena.mlp.first,
                      &arena.mlp.second, &arena.mlp.gated, &arena.mlp.blas});
}

} // namespace

Status read_runtime_model_config(const ModelFile &model,
                                 const bool allow_test_fixture,
                                 RuntimeModelConfig *const config) {
  if (config == nullptr)
    return {ErrorCode::kInvalidArgument, "model config output is null"};
  RuntimeModelConfig candidate;
  std::string architecture;
  auto status = metadata_string(model, "model.architecture", &architecture);
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
  for (const auto item :
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
      column_split_hyena || !interleave || flip ||
      (!candidate.test_fixture && !use_fp8_input_projections))
    return {ErrorCode::kUnsupported,
            "model semantic flags do not match supported Evo 2"};
  candidate.qkv_head_major = column_split;
  candidate.use_fp8_input_projections = use_fp8_input_projections;

  if (!candidate.test_fixture) {
    std::string fp8_reference;
    status = metadata_string(model, "fp8.reference", &fp8_reference);
    if (!status.ok())
      return status;
    if (fp8_reference != "TransformerEngine-2.3-HYBRID") {
      return {ErrorCode::kUnsupported,
              "unsupported software-FP8 checkpoint reference: " +
                  fp8_reference};
    }
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
  if (candidate.layers != 50 || !same_indices(hcs, kHcsLayers) ||
      !same_indices(hcm, kHcmLayers) || !same_indices(hcl, kHclLayers) ||
      !same_indices(attention, kAttentionLayers)) {
    return {ErrorCode::kUnsupported,
            "model must use the official Evo 2 50-layer stripe topology"};
  }

  if (candidate.vocab_size == 0 || candidate.width == 0 ||
      candidate.heads == 0 || candidate.width % candidate.heads != 0 ||
      candidate.head_dim() % 2 != 0 || candidate.head_dim() > 256 ||
      candidate.state_size == 0 || candidate.inner_width == 0 ||
      candidate.short_filter_length < 2 || candidate.hcs_filter_length < 2 ||
      candidate.hcm_filter_length < 2 || candidate.hcs_filter_groups == 0 ||
      candidate.width % candidate.hcs_filter_groups != 0 ||
      candidate.hcm_filter_groups == 0 ||
      candidate.width % candidate.hcm_filter_groups != 0 ||
      !std::isfinite(candidate.epsilon) || candidate.epsilon <= 0.0F ||
      !std::isfinite(candidate.rope_scale) || candidate.rope_scale <= 0.0F) {
    return {ErrorCode::kUnsupported, "model dimensions are unsupported"};
  }
  if (!candidate.test_fixture &&
      (candidate.vocab_size != 512 || candidate.width != 8192 ||
       candidate.heads != 64 || candidate.state_size != 16 ||
       candidate.inner_width != 22528 || candidate.short_filter_length != 3 ||
       candidate.hcs_filter_length != 7 || candidate.hcm_filter_length != 128 ||
       candidate.hcs_filter_groups != 512 ||
       candidate.hcm_filter_groups != 512)) {
    return {ErrorCode::kUnsupported,
            "production model dimensions are not Evo 2 40B"};
  }

  candidate.mixer_types.assign(candidate.layers, MixerType::kHcs);
  for (const auto index : hcm)
    candidate.mixer_types[index] = MixerType::kHcm;
  for (const auto index : hcl)
    candidate.mixer_types[index] = MixerType::kHcl;
  for (const auto index : attention)
    candidate.mixer_types[index] = MixerType::kAttention;
  *config = std::move(candidate);
  return Status::Ok();
}

struct SingleGpuModel::Impl final {
  RuntimeModelConfig config;
  int device{-1};
  std::size_t context_capacity{0};
  std::size_t layer_offset{0};
  std::size_t position{0};
  bool loaded{false};
  bool state_valid{false};
  Stream stream;
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
    if (!multiply(context_capacity, config.width, &sequence) ||
        !multiply(sequence, 3, &triple_sequence) ||
        !multiply(context_capacity, config.vocab_size, &logits_elements)) {
      return {ErrorCode::kInvalidArgument,
              "activation arena dimensions overflow"};
    }
    std::size_t token_bytes = 0;
    if (!multiply(context_capacity, sizeof(TokenId), &token_bytes))
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
    return arena.mlp.allocate(device, context_capacity, config.inner_width);
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
      return layer->kv_cache.allocate(device, context_capacity, config.heads,
                                      config.head_dim());
    }

    if (config.use_fp8_input_projections && !config.test_fixture) {
      float weight_scale = 0.0F;
      status = projection_fp8_scales(
          model, prefix + ".projections", &layer->projection_input_scale,
          &weight_scale, &layer->projection_output_scale);
      if (!status.ok())
        return status;
      DeviceBuffer original_projection;
      status = upload_tensor(
          model, prefix + ".projections.weight", TensorDType::kBF16,
          {config.width * 3, config.width}, device, stream,
          &original_projection);
      if (!status.ok())
        return status;
      std::size_t projection_elements = 0;
      if (!multiply(config.width * 3, config.width,
                    &projection_elements)) {
        return {ErrorCode::kInvalidArgument,
                "projection weight dimensions overflow"};
      }
      status = layer->projection.allocate(device, projection_elements);
      if (!status.ok())
        return status;
      status = software_e4m3_quantize_bf16_codes(
          original_projection, projection_elements, weight_scale,
          &layer->projection, stream);
      if (!status.ok())
        return status;
      status = stream.synchronize();
      if (!status.ok())
        return {status.code(),
                "software-FP8 projection quantization: " +
                    status.message()};
      layer->software_fp8_projection = true;
    } else {
      status = upload_tensor(
          model, prefix + ".projections.weight", TensorDType::kBF16,
          {config.width * 3, config.width}, device, stream,
          &layer->projection);
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
      status =
          upload_tensor(model, prefix + ".filter.h", TensorDType::kBF16,
                        {config.hcm_filter_groups, 1, config.hcm_filter_length},
                        device, stream, &layer->inner_filter);
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
    const std::size_t hcm_size = fir_fft_size(rows, config.hcm_filter_length);
    const std::size_t hcl_size = rows == 1 ? 0 : fir_fft_size(rows, rows);
    if (hcm_size == 0 || (rows > 1 && hcl_size == 0))
      return {ErrorCode::kInvalidArgument, "prefill FFT dimensions overflow"};
    hcm_fft.reset();
    hcl_fft.reset();
    auto status = hcm_fft.allocate(device, config.width,
                                   config.hcm_filter_groups, hcm_size);
    if (!status.ok())
      return status;
    if (rows == 1)
      return Status::Ok();
    return hcl_fft.allocate(device, config.width, config.width, hcl_size);
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
                                 const bool prefill) {
    auto status =
        layer->software_fp8_projection
            ? software_e4m3_h100_linear(
                  arena.normalized, layer->projection, rows, config.width,
                  config.width * 3, layer->projection_input_scale,
                  layer->projection_output_scale, &arena.mixer_scratch,
                  &arena.projection, stream)
            : bf16_linear(blas, arena.normalized, layer->projection, nullptr,
                          rows, config.width, config.width * 3,
                          &arena.projection, &arena.blas_workspace, stream);
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
      status =
          bf16_kv_append(arena.x1, arena.value, rows, &layer->kv_cache, stream);
      if (!status.ok())
        return status;
      return bf16_online_causal_attention(
          arena.x2, rows, prefix, layer->kv_cache, &arena.mixer_output, stream);
    }

    if (prefill) {
      status = bf16_fir_prefill_direct(
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
    status =
        bf16_split_hyena_projection(arena.short_filtered, rows, config.width,
                                    &arena.x2, &arena.x1, &arena.value, stream);
    if (!status.ok())
      return status;
    if (layer->type == MixerType::kHcs) {
      return prefill ? bf16_hcs_prefill(
                           arena.x2, arena.x1, arena.value, layer->inner_filter,
                           rows, config.width, config.hcs_filter_groups,
                           config.hcs_filter_length, &layer->inner_cache,
                           &arena.mixer_scratch, &arena.mixer_output, stream)
                     : bf16_hcs_decode(
                           arena.x2, arena.x1, arena.value, layer->inner_filter,
                           config.width, config.hcs_filter_groups,
                           config.hcs_filter_length, &layer->inner_cache,
                           &arena.mixer_scratch, &arena.mixer_output, stream);
    }
    if (layer->type == MixerType::kHcm) {
      return prefill
                 ? bf16_hcm_prefill(arena.x2, arena.x1, arena.value,
                                    layer->inner_filter, layer->direct, rows,
                                    config.width, config.hcm_filter_groups,
                                    config.hcm_filter_length,
                                    &layer->inner_cache, &arena.mixer_scratch,
                                    &arena.mixer_output, &hcm_fft, stream)
                 : bf16_hcm_decode(
                       arena.x2, arena.x1, arena.value, layer->inner_filter,
                       layer->direct, config.width, config.hcm_filter_groups,
                       config.hcm_filter_length, &layer->inner_cache,
                       &arena.mixer_scratch, &arena.mixer_output, stream);
    }
    const auto hcl_mode =
        rows == 1 ? HclPrefillMode::kRecurrence : HclPrefillMode::kFft;
    return prefill ? bf16_hcl_prefill(
                         arena.x2, arena.x1, arena.value, layer->direct,
                         layer->log_poles, layer->residues, rows, config.width,
                         config.state_size, hcl_mode, &layer->iir_cache,
                         &arena.mixer_scratch, &arena.mixer_output,
                         rows == 1 ? nullptr : &hcl_fft, stream)
                   : bf16_hcl_decode(arena.x2, arena.x1, arena.value,
                                     layer->direct, layer->log_poles,
                                     layer->residues, config.width,
                                     config.state_size, &layer->iir_cache,
                                     &arena.mixer_output, stream);
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

  [[nodiscard]] Status run_blocks(const std::size_t rows, const bool prefill,
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
      status = run_mixer(&layer, rows, prefill);
      if (!status.ok())
        break;
      status = dump_matching(dumps, index, LayerDumpPoint::kMixerOutput,
                             arena.mixer_output, rows, config.width);
      if (!status.ok())
        break;
      status = bf16_linear(blas, arena.mixer_output, layer.output_weight,
                           &layer.output_bias, rows, config.width, config.width,
                           &arena.residual, &arena.blas_workspace, stream);
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
                               const bool prefill,
                               std::vector<float> *const logits,
                               const std::vector<LayerDump> &dumps) {
    if (!loaded || logits == nullptr || tokens.empty() ||
        tokens.size() > context_capacity) {
      return {ErrorCode::kInvalidArgument,
              "model forward arguments are invalid"};
    }
    if (prefill) {
      auto status = prepare_prefill(tokens.size());
      if (!status.ok())
        return status;
    } else if (!state_valid || tokens.size() != 1 ||
               position >= context_capacity) {
      return {ErrorCode::kInvalidArgument,
              "decode requires a valid prefill and free context capacity"};
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
    status = run_blocks(rows, prefill, dumps);
    if (!status.ok()) {
      state_valid = false;
      return status;
    }
    status = bf16_rms_norm(arena.hidden, final_norm, rows, config.width,
                           config.epsilon, &arena.normalized, stream);
    if (status.ok()) {
      status = bf16_linear(blas, arena.normalized, embedding, nullptr, rows,
                           config.width, config.vocab_size, &arena.logits,
                           &arena.blas_workspace, stream);
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
    position = prefill ? rows : position + 1;
    state_valid = true;
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
  impl_->device = device;
  impl_->context_capacity = context_capacity;
  status = select_device(device);
  if (!status.ok())
    return status;
  status = impl_->stream.create();
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
  return Status::Ok();
}

Status SingleGpuModel::prefill(const std::vector<TokenId> &tokens,
                               std::vector<float> *const logits,
                               const std::optional<LayerDump> &dump) {
  std::vector<LayerDump> dumps;
  if (dump.has_value())
    dumps.push_back(*dump);
  return impl_->forward(tokens, true, logits, dumps);
}

Status SingleGpuModel::prefill_with_dumps(const std::vector<TokenId> &tokens,
                                          std::vector<float> *const logits,
                                          const std::vector<LayerDump> &dumps) {
  return impl_->forward(tokens, true, logits, dumps);
}

Status SingleGpuModel::decode(const TokenId token,
                              std::vector<float> *const logits) {
  return impl_->forward(std::vector<TokenId>{token}, false, logits, {});
}

const RuntimeModelConfig &SingleGpuModel::config() const noexcept {
  return impl_->config;
}

std::size_t SingleGpuModel::position() const noexcept {
  return impl_->position;
}
int SingleGpuModel::device() const noexcept { return impl_->device; }

struct PipelineModel::Impl final {
  RuntimeModelConfig config;
  std::size_t context_capacity{0};
  std::size_t position{0};
  bool loaded{false};
  bool state_valid{false};
  std::vector<StageAssignment> assignments;
  std::vector<std::unique_ptr<SingleGpuModel::Impl>> stages;

  [[nodiscard]] Status enable_stage_peers() const {
    std::set<std::pair<int, int>> enabled;
    for (std::size_t index = 0; index < assignments.size(); ++index) {
      const int source = assignments[index].device;
      const int destination =
          assignments[(index + 1) % assignments.size()].device;
      for (const auto pair :
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
                               const bool prefill,
                               std::vector<float> *const logits,
                               const std::vector<LayerDump> &dumps) {
    if (!loaded || logits == nullptr || tokens.empty() ||
        tokens.size() > context_capacity) {
      return {ErrorCode::kInvalidArgument,
              "pipeline model forward arguments are invalid"};
    }
    if (prefill) {
      position = 0;
      state_valid = false;
      for (auto &stage : stages) {
        auto status = stage->prepare_prefill(tokens.size());
        if (!status.ok())
          return {status.code(), "prepare pipeline stage on CUDA device " +
                                     std::to_string(stage->device) + ": " +
                                     status.message()};
      }
    } else if (!state_valid || tokens.size() != 1 ||
               position >= context_capacity) {
      return {ErrorCode::kInvalidArgument,
              "pipeline decode requires a valid prefill and free context "
              "capacity"};
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
      status = stages[stage_index]->run_blocks(rows, prefill, dumps);
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
      status =
          bf16_linear(head.blas, head.arena.normalized, head.embedding, nullptr,
                      rows, config.width, config.vocab_size, &head.arena.logits,
                      &head.arena.blas_workspace, head.stream);
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
    position = prefill ? rows : position + 1;
    state_valid = true;
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
  if (impl_->loaded || context_capacity == 0 || devices.size() != 4) {
    return {ErrorCode::kInvalidArgument,
            "pipeline load requires exactly four devices, nonzero context, "
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
  candidate->context_capacity = context_capacity;
  const std::size_t stage_count = devices.size();
  const std::size_t base_layers = candidate->config.layers / stage_count;
  const std::size_t extra_layers = candidate->config.layers % stage_count;
  std::size_t layer_begin = 0;
  candidate->assignments.reserve(stage_count);
  candidate->stages.reserve(stage_count);
  for (std::size_t index = 0; index < stage_count; ++index) {
    const std::size_t layer_count =
        base_layers + (index < extra_layers ? 1 : 0);
    const std::size_t layer_end = layer_begin + layer_count;
    candidate->assignments.push_back(
        {devices[index], layer_begin, layer_end, 0, 0, 0});
    auto stage = std::make_unique<SingleGpuModel::Impl>();
    stage->config = candidate->config;
    stage->device = devices[index];
    stage->context_capacity = context_capacity;
    stage->layer_offset = layer_begin;
    status = select_device(stage->device);
    if (!status.ok())
      return status;
    status = stage->stream.create();
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
        return {stage_status.code(),
                "load block " + std::to_string(index) + " on CUDA device " +
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
      stage_loads.push_back(std::async(std::launch::async, load_stage,
                                       stage_index));
    }
    for (std::size_t stage_index = 0; stage_index < stage_count;
         ++stage_index) {
      status = stage_loads[stage_index].get();
      if (!status.ok()) {
        return {status.code(),
                "load pipeline stage " + std::to_string(stage_index) + ": " +
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
  impl_ = std::move(candidate);
  return Status::Ok();
}

Status PipelineModel::prefill(const std::vector<TokenId> &tokens,
                              std::vector<float> *const logits,
                              const std::optional<LayerDump> &dump) {
  std::vector<LayerDump> dumps;
  if (dump.has_value())
    dumps.push_back(*dump);
  return impl_->forward(tokens, true, logits, dumps);
}

Status PipelineModel::prefill_with_dumps(const std::vector<TokenId> &tokens,
                                         std::vector<float> *const logits,
                                         const std::vector<LayerDump> &dumps) {
  return impl_->forward(tokens, true, logits, dumps);
}

Status PipelineModel::decode(const TokenId token,
                             std::vector<float> *const logits) {
  return impl_->forward(std::vector<TokenId>{token}, false, logits, {});
}

const RuntimeModelConfig &PipelineModel::config() const noexcept {
  return impl_->config;
}

const std::vector<StageAssignment> &PipelineModel::stages() const noexcept {
  return impl_->assignments;
}

std::size_t PipelineModel::position() const noexcept { return impl_->position; }

} // namespace evo2c::cuda
