// SPDX-License-Identifier: Apache-2.0
#include "evo/cpu/geneb_decoder.hpp"

#include "../linear_executor.hpp"
#include "geneb_decoder_omnina_apple.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace evo::cpu {
namespace {

constexpr std::size_t kMaximumTopologyLayers = 1024;
constexpr std::size_t kMaximumTopologyDimension = 1U << 24U;
constexpr char kTorch271AppleArm64ExactF32Kernel[] =
    "torch-2.7.1-apple-arm64-exact-v1";

Status metadata_entry(const ModelFile &artifact, const std::string_view key,
                      const MetadataType type,
                      const MetadataEntry **const output) {
  const auto *const entry = artifact.find_metadata(key);
  if (entry == nullptr) {
    return {ErrorCode::kModelFormat,
            "required GENEB decoder metadata is missing: " + std::string{key}};
  }
  if (entry->type != type) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder metadata has wrong type: " + std::string{key}};
  }
  *output = entry;
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

Status metadata_size(const ModelFile &artifact, const std::string_view key,
                     std::size_t *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kU64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(std::uint64_t)) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder u64 metadata is malformed: " + std::string{key}};
  }
  std::uint64_t value = 0;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (value > std::numeric_limits<std::size_t>::max()) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder metadata exceeds size_t: " + std::string{key}};
  }
  *output = static_cast<std::size_t>(value);
  return Status::Ok();
}

Status metadata_float(const ModelFile &artifact, const std::string_view key,
                      float *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kF64, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != sizeof(double)) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder f64 metadata is malformed: " + std::string{key}};
  }
  double value = 0.0;
  std::memcpy(&value, entry->value.data(), sizeof(value));
  if (!std::isfinite(value) ||
      value > static_cast<double>(std::numeric_limits<float>::max()) ||
      value < -static_cast<double>(std::numeric_limits<float>::max())) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder metadata is not finite F32: " + std::string{key}};
  }
  *output = static_cast<float>(value);
  return Status::Ok();
}

Status metadata_bool(const ModelFile &artifact, const std::string_view key,
                     bool *const output) {
  const MetadataEntry *entry = nullptr;
  auto status = metadata_entry(artifact, key, MetadataType::kBool, &entry);
  if (!status.ok())
    return status;
  if (entry->value.size() != 1) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder bool metadata is malformed: " + std::string{key}};
  }
  *output = entry->value[0] != 0;
  return Status::Ok();
}

Status metadata_literal(const ModelFile &artifact, const std::string_view key,
                        const std::string_view expected) {
  std::string actual;
  auto status = metadata_string(artifact, key, &actual);
  if (!status.ok())
    return status;
  if (actual != expected) {
    return {ErrorCode::kModelFormat, "GENEB decoder metadata mismatch for '" +
                                         std::string{key} + "': expected '" +
                                         std::string{expected} + "'"};
  }
  return Status::Ok();
}

Status metadata_dtype(const ModelFile &artifact, const std::string_view key,
                      TensorDType *const output) {
  std::string value;
  auto status = metadata_string(artifact, key, &value);
  if (!status.ok())
    return status;
  if (value == "F32")
    *output = TensorDType::kF32;
  else if (value == "BF16")
    *output = TensorDType::kBF16;
  else
    return {ErrorCode::kModelFormat,
            "GENEB decoder dtype metadata must be F32 or BF16: " +
                std::string{key}};
  return Status::Ok();
}

bool checked_product(const std::size_t left, const std::size_t right,
                     std::size_t *const output) noexcept {
  if (output == nullptr ||
      (left != 0 && right > std::numeric_limits<std::size_t>::max() / left))
    return false;
  *output = left * right;
  return true;
}

std::size_t dtype_size(const TensorDType dtype) noexcept {
  switch (dtype) {
  case TensorDType::kF32:
    return sizeof(float);
  case TensorDType::kBF16:
    return sizeof(std::uint16_t);
  case TensorDType::kE4M3Software:
    return 0;
  }
  return 0;
}

float tensor_value(const GenebDecoderTensorView &tensor,
                   const std::size_t index) noexcept {
  if (tensor.dtype == TensorDType::kBF16) {
    const auto *const source = tensor.data + index * sizeof(std::uint16_t);
    const std::uint32_t bits = static_cast<std::uint32_t>(source[0]) << 16U |
                               static_cast<std::uint32_t>(source[1]) << 24U;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(float), sizeof(value));
  return value;
}

float round_to_bf16(const float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  // Keep infinities and NaNs in their source class. Finite values use the
  // standard round-to-nearest-even bias before the low 16 bits are cleared.
  if ((bits & 0x7f800000U) != 0x7f800000U)
    bits += 0x7fffU + ((bits >> 16U) & 1U);
  bits &= 0xffff0000U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float activation_value(const float value, const TensorDType dtype) noexcept {
  return dtype == TensorDType::kBF16 ? round_to_bf16(value) : value;
}

void round_activations(std::vector<float> *const values,
                       const TensorDType dtype) noexcept {
  if (dtype != TensorDType::kBF16)
    return;
  for (float &value : *values)
    value = round_to_bf16(value);
}

Status matrix_elements(const std::size_t rows, const std::size_t columns,
                       const std::size_t actual, const std::string_view name) {
  std::size_t expected = 0;
  if (rows == 0 || columns == 0 || !checked_product(rows, columns, &expected)) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " dimensions are zero or overflow"};
  }
  if (expected != actual) {
    return {ErrorCode::kInvalidArgument,
            std::string{name} + " has " + std::to_string(actual) +
                " elements; expected " + std::to_string(expected)};
  }
  return Status::Ok();
}

bool finite_vector(const std::vector<float> &values) noexcept {
  for (const float value : values) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

Status tensor_requirement(const GenebDecoderTensorView &tensor,
                          const GenebDecoderTensorRequirement &requirement) {
  if (tensor.dtype != requirement.dtype) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder tensor dtype mismatch: " + requirement.name};
  }
  if (tensor.shape != requirement.shape) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder tensor shape mismatch: " + requirement.name};
  }
  std::size_t elements = 1;
  for (const std::size_t dimension : tensor.shape) {
    if (dimension == 0 || !checked_product(elements, dimension, &elements)) {
      return {ErrorCode::kModelFormat,
              "GENEB decoder tensor dimensions overflow: " + requirement.name};
    }
  }
  std::size_t expected_bytes = 0;
  const std::size_t element_bytes = dtype_size(tensor.dtype);
  if (element_bytes == 0 ||
      !checked_product(elements, element_bytes, &expected_bytes) ||
      tensor.bytes != expected_bytes || tensor.data == nullptr) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder tensor payload mismatch: " + requirement.name};
  }
  return Status::Ok();
}

std::string layer_prefix(const std::size_t layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

void add_requirement(std::vector<GenebDecoderTensorRequirement> *const output,
                     std::string name, const TensorDType dtype,
                     std::vector<std::size_t> shape) {
  output->push_back({std::move(name), dtype, std::move(shape)});
}

Status runtime_linear(const std::vector<float> &input, const std::size_t rows,
                      const std::size_t input_width,
                      const GenebDecoderTensorView &weight,
                      const std::size_t output_width,
                      const GenebDecoderTensorView *const bias,
                      evo::detail::LinearExecutor *const executor,
                      const TensorDType activation_dtype,
                      const GenebDecoderF32MathKernel f32_math_kernel,
                      std::vector<float> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB decoder linear output is null"};
  auto status =
      matrix_elements(rows, input_width, input.size(), "decoder linear input");
  if (!status.ok())
    return status;
  std::size_t weight_elements = 0;
  if (!checked_product(output_width, input_width, &weight_elements)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder linear weight dimensions overflow"};
  }
  std::size_t weight_bytes = 0;
  std::size_t bias_bytes = 0;
  std::size_t output_elements = 0;
  if (!checked_product(weight_elements, dtype_size(weight.dtype),
                       &weight_bytes) ||
      !checked_product(output_width,
                       bias == nullptr ? 0 : dtype_size(bias->dtype),
                       &bias_bytes) ||
      !checked_product(rows, output_width, &output_elements) ||
      weight.bytes != weight_bytes || weight.data == nullptr ||
      (bias != nullptr &&
       (bias->data == nullptr || bias->bytes != bias_bytes))) {
    return {ErrorCode::kInternal,
            "GENEB decoder linear tensor view is inconsistent"};
  }
  if (f32_math_kernel ==
      GenebDecoderF32MathKernel::kTorch271AppleArm64ExactV1) {
    return detail::omnina_apple_f32_linear(input, rows, input_width, weight,
                                           output_width, bias, output);
  }
  if (executor != nullptr) {
    const evo::detail::LinearTensorView weight_view{weight.data, weight.dtype,
                                                    weight_elements};
    evo::detail::LinearTensorView bias_view;
    const evo::detail::LinearTensorView *bias_pointer = nullptr;
    if (bias != nullptr) {
      bias_view = {bias->data, bias->dtype, output_width};
      bias_pointer = &bias_view;
    }
    status = executor->linear(input.data(), rows, input_width, weight_view,
                              output_width, bias_pointer, output);
    if (!status.ok())
      return status;
    if (output->size() != output_elements || !finite_vector(*output)) {
      return {ErrorCode::kInternal,
              "GENEB decoder linear executor returned invalid output"};
    }
    round_activations(output, activation_dtype);
    return finite_vector(*output)
               ? Status::Ok()
               : Status{ErrorCode::kInvalidArgument,
                        "GENEB decoder linear output overflowed activation "
                        "dtype"};
  }

  output->assign(output_elements, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t target = 0; target < output_width; ++target) {
      float total = bias == nullptr ? 0.0F : tensor_value(*bias, target);
      for (std::size_t source = 0; source < input_width; ++source) {
        total += input[row * input_width + source] *
                 tensor_value(weight, target * input_width + source);
      }
      if (!std::isfinite(total)) {
        return {ErrorCode::kInvalidArgument,
                "GENEB decoder linear accumulation became non-finite"};
      }
      (*output)[row * output_width + target] =
          activation_value(total, activation_dtype);
    }
  }
  return finite_vector(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB decoder linear output overflowed activation "
                      "dtype"};
}

void add_in_place(std::vector<float> *const residual,
                  const std::vector<float> &update,
                  const TensorDType activation_dtype) noexcept {
  for (std::size_t index = 0; index < residual->size(); ++index)
    (*residual)[index] =
        activation_value((*residual)[index] + update[index], activation_dtype);
}

float silu(const float value) noexcept {
  if (value >= 0.0F)
    return value / (1.0F + std::exp(-value));
  const float exponential = std::exp(value);
  return value * exponential / (1.0F + exponential);
}

float gelu(const float value) noexcept {
  constexpr float kInverseSqrtTwo = 0.70710678118654752440F;
  return 0.5F * value * (1.0F + std::erf(value * kInverseSqrtTwo));
}

Status apply_rope_tensor(std::vector<float> *const tensor,
                         const std::size_t heads, const std::size_t rows,
                         const std::size_t head_dimension,
                         const std::size_t rotary_dimension,
                         const std::size_t position_offset,
                         const float rope_base, const float position_scale,
                         const GenebDecoderRopeLayout layout,
                         const TensorDType activation_dtype) {
  std::size_t expected = 0;
  std::size_t row_width = 0;
  if (tensor == nullptr ||
      !checked_product(heads, head_dimension, &row_width) ||
      !checked_product(rows, row_width, &expected) ||
      tensor->size() != expected) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder RoPE tensor shape is inconsistent"};
  }
  const std::size_t pairs = rotary_dimension / 2U;
  for (std::size_t row = 0; row < rows; ++row) {
    const float position =
        static_cast<float>(position_offset + row) / position_scale;
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t base = (row * heads + head) * head_dimension;
      for (std::size_t pair = 0; pair < pairs; ++pair) {
        const float exponent = 2.0F * static_cast<float>(pair) /
                               static_cast<float>(rotary_dimension);
        const float angle = position / std::pow(rope_base, exponent);
        // HF casts the RoPE cache to q/k dtype. Each multiply and the final
        // add/subtract are separate BF16 eager operations.
        const float cosine =
            activation_value(std::cos(angle), activation_dtype);
        const float sine = activation_value(std::sin(angle), activation_dtype);
        const std::size_t first =
            layout == GenebDecoderRopeLayout::kAdjacentPairs ? pair * 2U : pair;
        const std::size_t second =
            layout == GenebDecoderRopeLayout::kAdjacentPairs ? first + 1U
                                                             : pair + pairs;
        const float first_value = (*tensor)[base + first];
        const float second_value = (*tensor)[base + second];
        const float first_cosine =
            activation_value(first_value * cosine, activation_dtype);
        const float second_sine =
            activation_value(second_value * sine, activation_dtype);
        const float second_cosine =
            activation_value(second_value * cosine, activation_dtype);
        const float first_sine =
            activation_value(first_value * sine, activation_dtype);
        (*tensor)[base + first] =
            activation_value(first_cosine - second_sine, activation_dtype);
        (*tensor)[base + second] =
            activation_value(second_cosine + first_sine, activation_dtype);
      }
    }
  }
  return Status::Ok();
}

} // namespace

Status validate_geneb_decoder_topology(const GenebDecoderTopology &topology) {
  if (topology.vocabulary_size == 0 || topology.width == 0 ||
      topology.layers == 0 || topology.query_heads == 0 ||
      topology.key_value_heads == 0 || topology.head_dimension == 0 ||
      topology.inner_width == 0 || topology.maximum_sequence_length == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder topology dimensions must be positive"};
  }
  if (topology.layers > kMaximumTopologyLayers ||
      topology.vocabulary_size > kMaximumTopologyDimension ||
      topology.width > kMaximumTopologyDimension ||
      topology.inner_width > kMaximumTopologyDimension ||
      topology.maximum_sequence_length > kMaximumTopologyDimension) {
    return {ErrorCode::kUnsupported,
            "GENEB decoder topology exceeds reference-runtime limits"};
  }
  std::size_t projected_width = 0;
  std::size_t key_value_width = 0;
  if (!checked_product(topology.query_heads, topology.head_dimension,
                       &projected_width) ||
      !checked_product(topology.key_value_heads, topology.head_dimension,
                       &key_value_width) ||
      projected_width == 0 || key_value_width == 0 ||
      projected_width > kMaximumTopologyDimension ||
      key_value_width > kMaximumTopologyDimension) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder projected attention dimensions overflow or exceed "
            "reference-runtime limits"};
  }
  if (topology.query_heads % topology.key_value_heads != 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder query heads must be divisible by KV heads"};
  }
  if (topology.rotary_dimension == 0 ||
      topology.rotary_dimension > topology.head_dimension ||
      topology.rotary_dimension % 2U != 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder rotary dimension must be positive, even, and no "
            "larger than head dimension"};
  }
  if (topology.sliding_window > topology.maximum_sequence_length) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder sliding window exceeds maximum sequence length"};
  }
  if (!std::isfinite(topology.rms_epsilon) || topology.rms_epsilon <= 0.0F ||
      !std::isfinite(topology.rope_base) || topology.rope_base <= 1.0F ||
      !std::isfinite(topology.rope_position_scale) ||
      topology.rope_position_scale <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder epsilon/RoPE scalars are invalid"};
  }
  if (dtype_size(topology.embedding_dtype) == 0 ||
      dtype_size(topology.projection_dtype) == 0 ||
      dtype_size(topology.norm_dtype) == 0 ||
      dtype_size(topology.activation_dtype) == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder supports only declared F32/BF16 tensor and "
            "activation dtypes"};
  }
  if ((topology.rms_epsilon_placement !=
           GenebDecoderRmsEpsilonPlacement::kInsideSqrt &&
       topology.rms_epsilon_placement !=
           GenebDecoderRmsEpsilonPlacement::kAfterSqrt) ||
      (topology.rope_layout != GenebDecoderRopeLayout::kSplitHalf &&
       topology.rope_layout != GenebDecoderRopeLayout::kAdjacentPairs) ||
      (topology.mlp_activation != GenebDecoderMlpActivation::kSwiGlu &&
       topology.mlp_activation != GenebDecoderMlpActivation::kGelu) ||
      (topology.attention_kernel != GenebDecoderAttentionKernel::kEager &&
       topology.attention_kernel !=
           GenebDecoderAttentionKernel::kTorchCpuFlashBf16Portable) ||
      (topology.f32_math_kernel != GenebDecoderF32MathKernel::kPortable &&
       topology.f32_math_kernel !=
           GenebDecoderF32MathKernel::kTorch271AppleArm64ExactV1)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder topology contains an unknown operation enum"};
  }
  if (topology.attention_kernel ==
          GenebDecoderAttentionKernel::kTorchCpuFlashBf16Portable &&
      (topology.activation_dtype != TensorDType::kBF16 ||
       topology.sliding_window != 0)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder portable CPU Flash attention requires BF16 "
            "activations and full causal attention"};
  }
  if (topology.f32_math_kernel ==
          GenebDecoderF32MathKernel::kTorch271AppleArm64ExactV1 &&
      (topology.vocabulary_size != 32001 || topology.width != 1024 ||
       topology.layers != 16 || topology.query_heads != 16 ||
       topology.key_value_heads != 16 || topology.head_dimension != 64 ||
       topology.rotary_dimension != 64 || topology.inner_width != 4096 ||
       topology.maximum_sequence_length != 2048 ||
       topology.sliding_window != 0 || topology.rms_epsilon != 1.0e-6F ||
       topology.rms_epsilon_placement !=
           GenebDecoderRmsEpsilonPlacement::kInsideSqrt ||
       topology.rope_base != 10000.0F || topology.rope_position_scale != 1.0F ||
       topology.rope_layout != GenebDecoderRopeLayout::kSplitHalf ||
       topology.mlp_activation != GenebDecoderMlpActivation::kSwiGlu ||
       topology.attention_kernel != GenebDecoderAttentionKernel::kEager ||
       topology.attention_bias || topology.mlp_bias ||
       topology.embedding_dtype != TensorDType::kF32 ||
       topology.projection_dtype != TensorDType::kF32 ||
       topology.norm_dtype != TensorDType::kF32 ||
       topology.activation_dtype != TensorDType::kF32)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder Apple exact F32 math kernel requires the closed "
            "OmniNA-220M topology"};
  }
  return Status::Ok();
}

Status
geneb_decoder_topology_from_artifact(const ModelFile &artifact,
                                     GenebDecoderTopology *const output) {
  if (output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder topology output is null"};
  }
  if (artifact.profile() != kGenebDecoderArtifactProfile) {
    return {ErrorCode::kModelFormat,
            "GENEB decoder artifact profile mismatch: expected '" +
                std::string{kGenebDecoderArtifactProfile} + "'"};
  }
  auto status =
      metadata_literal(artifact, "runtime.abi", kGenebDecoderRuntimeAbi);
  if (!status.ok())
    return status;
  status = metadata_literal(artifact, "model.architecture",
                            kGenebDecoderArchitecture);
  if (!status.ok())
    return status;

  const std::set<std::string_view> decoder_metadata{
      "decoder.vocab_size",          "decoder.hidden_size",
      "decoder.num_layers",          "decoder.num_attention_heads",
      "decoder.num_key_value_heads", "decoder.head_dim",
      "decoder.rotary_dim",          "decoder.inner_mlp_size",
      "decoder.max_seqlen",          "decoder.sliding_window",
      "decoder.rms_norm_epsilon",    "decoder.rms_epsilon_placement",
      "decoder.rope_base",           "decoder.rope_position_scale",
      "decoder.rope_layout",         "decoder.mlp_activation",
      "decoder.attention_kernel",    "decoder.f32_math_kernel",
      "decoder.attention_bias",      "decoder.mlp_bias",
      "decoder.embedding_dtype",     "decoder.projection_dtype",
      "decoder.norm_dtype",          "decoder.activation_dtype",
  };
  for (const auto &entry : artifact.metadata()) {
    if (entry.key.compare(0, 8, "decoder.") == 0 &&
        decoder_metadata.find(std::string_view{entry.key}) ==
            decoder_metadata.end()) {
      return {ErrorCode::kModelFormat,
              "unexpected GENEB decoder topology metadata: " + entry.key};
    }
  }

  GenebDecoderTopology topology;
  status =
      metadata_size(artifact, "decoder.vocab_size", &topology.vocabulary_size);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "decoder.hidden_size", &topology.width);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "decoder.num_layers", &topology.layers);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "decoder.num_attention_heads",
                         &topology.query_heads);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "decoder.num_key_value_heads",
                         &topology.key_value_heads);
  if (!status.ok())
    return status;
  status =
      metadata_size(artifact, "decoder.head_dim", &topology.head_dimension);
  if (!status.ok())
    return status;
  status =
      metadata_size(artifact, "decoder.rotary_dim", &topology.rotary_dimension);
  if (!status.ok())
    return status;
  status =
      metadata_size(artifact, "decoder.inner_mlp_size", &topology.inner_width);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "decoder.max_seqlen",
                         &topology.maximum_sequence_length);
  if (!status.ok())
    return status;
  status = metadata_size(artifact, "decoder.sliding_window",
                         &topology.sliding_window);
  if (!status.ok())
    return status;
  status = metadata_float(artifact, "decoder.rms_norm_epsilon",
                          &topology.rms_epsilon);
  if (!status.ok())
    return status;
  status = metadata_float(artifact, "decoder.rope_base", &topology.rope_base);
  if (!status.ok())
    return status;
  status = metadata_float(artifact, "decoder.rope_position_scale",
                          &topology.rope_position_scale);
  if (!status.ok())
    return status;
  status = metadata_bool(artifact, "decoder.attention_bias",
                         &topology.attention_bias);
  if (!status.ok())
    return status;
  status = metadata_bool(artifact, "decoder.mlp_bias", &topology.mlp_bias);
  if (!status.ok())
    return status;
  status = metadata_dtype(artifact, "decoder.embedding_dtype",
                          &topology.embedding_dtype);
  if (!status.ok())
    return status;
  status = metadata_dtype(artifact, "decoder.projection_dtype",
                          &topology.projection_dtype);
  if (!status.ok())
    return status;
  status = metadata_dtype(artifact, "decoder.norm_dtype", &topology.norm_dtype);
  if (!status.ok())
    return status;
  status = metadata_dtype(artifact, "decoder.activation_dtype",
                          &topology.activation_dtype);
  if (!status.ok())
    return status;

  std::string value;
  status = metadata_string(artifact, "decoder.rms_epsilon_placement", &value);
  if (!status.ok())
    return status;
  if (value == "inside-sqrt") {
    topology.rms_epsilon_placement =
        GenebDecoderRmsEpsilonPlacement::kInsideSqrt;
  } else if (value == "after-sqrt") {
    topology.rms_epsilon_placement =
        GenebDecoderRmsEpsilonPlacement::kAfterSqrt;
  } else {
    return {ErrorCode::kModelFormat,
            "GENEB decoder rms_epsilon_placement metadata is invalid"};
  }
  status = metadata_string(artifact, "decoder.rope_layout", &value);
  if (!status.ok())
    return status;
  if (value == "split-half")
    topology.rope_layout = GenebDecoderRopeLayout::kSplitHalf;
  else if (value == "adjacent-pairs")
    topology.rope_layout = GenebDecoderRopeLayout::kAdjacentPairs;
  else
    return {ErrorCode::kModelFormat,
            "GENEB decoder rope_layout metadata is invalid"};
  status = metadata_string(artifact, "decoder.mlp_activation", &value);
  if (!status.ok())
    return status;
  if (value == "swiglu")
    topology.mlp_activation = GenebDecoderMlpActivation::kSwiGlu;
  else if (value == "gelu")
    topology.mlp_activation = GenebDecoderMlpActivation::kGelu;
  else
    return {ErrorCode::kModelFormat,
            "GENEB decoder mlp_activation metadata is invalid"};

  if (artifact.find_metadata("decoder.attention_kernel") != nullptr) {
    status = metadata_string(artifact, "decoder.attention_kernel", &value);
    if (!status.ok())
      return status;
    if (value == "eager") {
      topology.attention_kernel = GenebDecoderAttentionKernel::kEager;
    } else if (value == "torch-cpu-flash-bf16-portable") {
      topology.attention_kernel =
          GenebDecoderAttentionKernel::kTorchCpuFlashBf16Portable;
    } else {
      return {ErrorCode::kModelFormat,
              "GENEB decoder attention_kernel metadata is invalid"};
    }
  }

  if (artifact.find_metadata("decoder.f32_math_kernel") != nullptr) {
    status = metadata_string(artifact, "decoder.f32_math_kernel", &value);
    if (!status.ok())
      return status;
    if (value == "torch-2.7.1-apple-arm64-exact-v1") {
      topology.f32_math_kernel =
          GenebDecoderF32MathKernel::kTorch271AppleArm64ExactV1;
    } else {
      return {ErrorCode::kModelFormat,
              "GENEB decoder f32_math_kernel metadata is invalid"};
    }
  }

  status = validate_geneb_decoder_topology(topology);
  if (!status.ok())
    return {ErrorCode::kModelFormat,
            "GENEB decoder artifact topology is invalid: " + status.message()};

  const auto matching_size = [&](const std::string_view key,
                                 const std::size_t expected) {
    std::size_t actual = 0;
    auto match_status = metadata_size(artifact, key, &actual);
    if (!match_status.ok())
      return match_status;
    if (actual != expected) {
      return Status{ErrorCode::kModelFormat,
                    "GENEB decoder common metadata disagrees with decoder "
                    "topology: " +
                        std::string{key}};
    }
    return Status::Ok();
  };
  status = matching_size("config.vocab_size", topology.vocabulary_size);
  if (!status.ok())
    return status;
  status = matching_size("config.hidden_size", topology.width);
  if (!status.ok())
    return status;
  status = matching_size("config.num_layers", topology.layers);
  if (!status.ok())
    return status;
  status = matching_size("config.max_seqlen", topology.maximum_sequence_length);
  if (!status.ok())
    return status;
  status = matching_size("runtime.embedding_layer_count", topology.layers + 1U);
  if (!status.ok())
    return status;
  *output = topology;
  return Status::Ok();
}

Status canonical_geneb_decoder_tensors(
    const GenebDecoderTopology &topology,
    std::vector<GenebDecoderTensorRequirement> *const output) {
  if (output == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder tensor-requirement output is null"};
  }
  auto status = validate_geneb_decoder_topology(topology);
  if (!status.ok())
    return status;
  std::vector<GenebDecoderTensorRequirement> result;
  const std::size_t query_width =
      topology.query_heads * topology.head_dimension;
  const std::size_t kv_width =
      topology.key_value_heads * topology.head_dimension;
  add_requirement(&result, "model.embed_tokens.weight",
                  topology.embedding_dtype,
                  {topology.vocabulary_size, topology.width});
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    const std::string prefix = layer_prefix(layer);
    add_requirement(&result, prefix + "input_layernorm.weight",
                    topology.norm_dtype, {topology.width});
    add_requirement(&result, prefix + "self_attn.q_proj.weight",
                    topology.projection_dtype, {query_width, topology.width});
    add_requirement(&result, prefix + "self_attn.k_proj.weight",
                    topology.projection_dtype, {kv_width, topology.width});
    add_requirement(&result, prefix + "self_attn.v_proj.weight",
                    topology.projection_dtype, {kv_width, topology.width});
    add_requirement(&result, prefix + "self_attn.o_proj.weight",
                    topology.projection_dtype, {topology.width, query_width});
    if (topology.attention_bias) {
      add_requirement(&result, prefix + "self_attn.q_proj.bias",
                      topology.projection_dtype, {query_width});
      add_requirement(&result, prefix + "self_attn.k_proj.bias",
                      topology.projection_dtype, {kv_width});
      add_requirement(&result, prefix + "self_attn.v_proj.bias",
                      topology.projection_dtype, {kv_width});
      add_requirement(&result, prefix + "self_attn.o_proj.bias",
                      topology.projection_dtype, {topology.width});
    }
    add_requirement(&result, prefix + "post_attention_layernorm.weight",
                    topology.norm_dtype, {topology.width});
    if (topology.mlp_activation == GenebDecoderMlpActivation::kSwiGlu) {
      add_requirement(&result, prefix + "mlp.gate_proj.weight",
                      topology.projection_dtype,
                      {topology.inner_width, topology.width});
    }
    add_requirement(&result, prefix + "mlp.up_proj.weight",
                    topology.projection_dtype,
                    {topology.inner_width, topology.width});
    add_requirement(&result, prefix + "mlp.down_proj.weight",
                    topology.projection_dtype,
                    {topology.width, topology.inner_width});
    if (topology.mlp_bias) {
      if (topology.mlp_activation == GenebDecoderMlpActivation::kSwiGlu) {
        add_requirement(&result, prefix + "mlp.gate_proj.bias",
                        topology.projection_dtype, {topology.inner_width});
      }
      add_requirement(&result, prefix + "mlp.up_proj.bias",
                      topology.projection_dtype, {topology.inner_width});
      add_requirement(&result, prefix + "mlp.down_proj.bias",
                      topology.projection_dtype, {topology.width});
    }
  }
  add_requirement(&result, "model.norm.weight", topology.norm_dtype,
                  {topology.width});
  *output = std::move(result);
  return Status::Ok();
}

Status geneb_decoder_rms_norm(const std::vector<float> &input,
                              const std::size_t rows, const std::size_t width,
                              const GenebDecoderTensorView &scale,
                              const float epsilon,
                              const GenebDecoderRmsEpsilonPlacement placement,
                              const TensorDType activation_dtype,
                              std::vector<float> *const output) {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder RMSNorm output is null"};
  auto status =
      matrix_elements(rows, width, input.size(), "decoder RMSNorm input");
  if (!status.ok())
    return status;
  if (scale.shape != std::vector<std::size_t>{width} ||
      dtype_size(scale.dtype) == 0 ||
      scale.bytes != width * dtype_size(scale.dtype) || scale.data == nullptr) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder RMSNorm scale is inconsistent"};
  }
  if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder RMSNorm epsilon must be finite and positive"};
  }
  if (placement != GenebDecoderRmsEpsilonPlacement::kInsideSqrt &&
      placement != GenebDecoderRmsEpsilonPlacement::kAfterSqrt) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder RMSNorm epsilon placement is invalid"};
  }
  if (dtype_size(activation_dtype) == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder RMSNorm activation dtype must be F32 or BF16"};
  }
  output->resize(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    float sum_squares = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      const float value = input[row * width + column];
      sum_squares += value * value;
    }
    const float mean_square = sum_squares / static_cast<float>(width);
    const float denominator =
        placement == GenebDecoderRmsEpsilonPlacement::kInsideSqrt
            ? std::sqrt(mean_square + epsilon)
            : std::sqrt(mean_square) + epsilon;
    if (!std::isfinite(denominator) || denominator <= 0.0F) {
      return {ErrorCode::kInvalidArgument,
              "GENEB decoder RMSNorm denominator is invalid"};
    }
    for (std::size_t column = 0; column < width; ++column) {
      // HF LlamaRMSNorm normalizes in F32, casts the normalized tensor back to
      // the activation dtype, and only then applies the learned scale.
      const float normalized = activation_value(
          input[row * width + column] / denominator, activation_dtype);
      (*output)[row * width + column] = activation_value(
          normalized * tensor_value(scale, column), activation_dtype);
    }
  }
  return finite_vector(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB decoder RMSNorm produced non-finite output"};
}

Status geneb_decoder_apply_rope(
    std::vector<float> *const query, const std::size_t query_heads,
    std::vector<float> *const key, const std::size_t key_value_heads,
    const std::size_t rows, const std::size_t head_dimension,
    const std::size_t rotary_dimension, const std::size_t position_offset,
    const float rope_base, const float position_scale,
    const GenebDecoderRopeLayout layout, const TensorDType activation_dtype) {
  if (query == nullptr || key == nullptr || query_heads == 0 ||
      key_value_heads == 0 || rows == 0 || head_dimension == 0 ||
      rotary_dimension == 0 || rotary_dimension > head_dimension ||
      rotary_dimension % 2U != 0 || !std::isfinite(rope_base) ||
      rope_base <= 1.0F || !std::isfinite(position_scale) ||
      position_scale <= 0.0F ||
      (layout != GenebDecoderRopeLayout::kSplitHalf &&
       layout != GenebDecoderRopeLayout::kAdjacentPairs) ||
      dtype_size(activation_dtype) == 0 ||
      position_offset > std::numeric_limits<std::size_t>::max() - rows) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder RoPE arguments are invalid"};
  }
  auto status = apply_rope_tensor(query, query_heads, rows, head_dimension,
                                  rotary_dimension, position_offset, rope_base,
                                  position_scale, layout, activation_dtype);
  if (!status.ok())
    return status;
  return apply_rope_tensor(key, key_value_heads, rows, head_dimension,
                           rotary_dimension, position_offset, rope_base,
                           position_scale, layout, activation_dtype);
}

namespace {

Status geneb_decoder_causal_attention_flash_bf16_portable(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value, const std::size_t rows,
    const std::size_t query_heads, const std::size_t key_value_heads,
    const std::size_t head_dimension, std::vector<float> *const output) {
  constexpr std::size_t kKeyBlock = 512;
  std::size_t query_width = 0;
  std::size_t kv_width = 0;
  std::size_t query_elements = 0;
  std::size_t kv_elements = 0;
  if (output == nullptr || rows == 0 || query_heads == 0 ||
      key_value_heads == 0 || head_dimension == 0 ||
      query_heads % key_value_heads != 0 ||
      !checked_product(query_heads, head_dimension, &query_width) ||
      !checked_product(key_value_heads, head_dimension, &kv_width) ||
      !checked_product(rows, query_width, &query_elements) ||
      !checked_product(rows, kv_width, &kv_elements) ||
      query.size() != query_elements || key.size() != kv_elements ||
      value.size() != kv_elements || !finite_vector(query) ||
      !finite_vector(key) || !finite_vector(value)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder portable CPU Flash attention arguments are "
            "invalid"};
  }

  output->assign(query_elements, 0.0F);
  const std::size_t heads_per_kv = query_heads / key_value_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
  std::vector<float> scores;
  std::vector<float> accumulated(head_dimension);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
      const std::size_t kv_head = query_head / heads_per_kv;
      const std::size_t query_base =
          (row * query_heads + query_head) * head_dimension;
      std::fill(accumulated.begin(), accumulated.end(), 0.0F);
      float running_max = -std::numeric_limits<float>::infinity();
      float running_sum = 0.0F;
      for (std::size_t first = 0; first <= row; first += kKeyBlock) {
        const std::size_t count = std::min(kKeyBlock, row + 1U - first);
        scores.resize(count);
        float block_max = -std::numeric_limits<float>::infinity();
        for (std::size_t offset = 0; offset < count; ++offset) {
          const std::size_t source = first + offset;
          const std::size_t key_base =
              (source * key_value_heads + kv_head) * head_dimension;
          float score = 0.0F;
          for (std::size_t dimension = 0; dimension < head_dimension;
               ++dimension) {
            score += query[query_base + dimension] * key[key_base + dimension];
          }
          score *= scale;
          scores[offset] = score;
          block_max = std::max(block_max, score);
        }

        const float next_max = std::max(running_max, block_max);
        const float previous_scale =
            std::isinf(running_max) ? 0.0F : std::exp(running_max - next_max);
        running_sum *= previous_scale;
        for (float &item : accumulated)
          item *= previous_scale;
        for (std::size_t offset = 0; offset < count; ++offset) {
          const float exponential = std::exp(scores[offset] - next_max);
          running_sum += exponential;
          // PyTorch's reduced-type CPU Flash path stores the unnormalized
          // exponential in a BF16 scratch buffer before the value GEMM, while
          // retaining the F32 value for the denominator.
          scores[offset] = activation_value(exponential, TensorDType::kBF16);
          const std::size_t source = first + offset;
          const std::size_t value_base =
              (source * key_value_heads + kv_head) * head_dimension;
          for (std::size_t dimension = 0; dimension < head_dimension;
               ++dimension) {
            accumulated[dimension] +=
                scores[offset] * value[value_base + dimension];
          }
        }
        running_max = next_max;
      }
      if (!std::isfinite(running_sum) || running_sum <= 0.0F) {
        return {ErrorCode::kInvalidArgument,
                "GENEB decoder portable CPU Flash softmax is invalid"};
      }
      for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        (*output)[query_base + dimension] = activation_value(
            accumulated[dimension] / running_sum, TensorDType::kBF16);
      }
    }
  }
  return finite_vector(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB decoder portable CPU Flash attention produced "
                      "non-finite output"};
}

} // namespace

Status geneb_decoder_causal_attention(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value, const std::size_t rows,
    const std::size_t query_heads, const std::size_t key_value_heads,
    const std::size_t head_dimension, const std::size_t sliding_window,
    const TensorDType activation_dtype, std::vector<float> *const output) {
  if (output == nullptr || rows == 0 || query_heads == 0 ||
      key_value_heads == 0 || head_dimension == 0 ||
      query_heads % key_value_heads != 0 || dtype_size(activation_dtype) == 0) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder attention arguments are invalid"};
  }
  std::size_t query_width = 0;
  std::size_t kv_width = 0;
  std::size_t query_elements = 0;
  std::size_t kv_elements = 0;
  if (!checked_product(query_heads, head_dimension, &query_width) ||
      !checked_product(key_value_heads, head_dimension, &kv_width) ||
      !checked_product(rows, query_width, &query_elements) ||
      !checked_product(rows, kv_width, &kv_elements) ||
      query.size() != query_elements || key.size() != kv_elements ||
      value.size() != kv_elements || !finite_vector(query) ||
      !finite_vector(key) || !finite_vector(value)) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder attention tensor shapes/values are invalid"};
  }
  output->assign(query_elements, 0.0F);
  std::vector<float> scores;
  const std::size_t heads_per_kv = query_heads / key_value_heads;
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t first = sliding_window == 0 || row + 1U <= sliding_window
                                  ? 0
                                  : row + 1U - sliding_window;
    const std::size_t attended = row + 1U - first;
    scores.resize(attended);
    for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
      const std::size_t kv_head = query_head / heads_per_kv;
      const std::size_t query_base =
          (row * query_heads + query_head) * head_dimension;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t source = first; source <= row; ++source) {
        const std::size_t key_base =
            (source * key_value_heads + kv_head) * head_dimension;
        float score = 0.0F;
        for (std::size_t dimension = 0; dimension < head_dimension;
             ++dimension) {
          score += query[query_base + dimension] * key[key_base + dimension];
        }
        // A BF16 eager matmul materializes its result before the subsequent
        // scalar scale, so both tensor-operation boundaries are rounded.
        score = activation_value(score, activation_dtype);
        score = activation_value(score * scale, activation_dtype);
        scores[source - first] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0F;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      if (!std::isfinite(denominator) || denominator <= 0.0F) {
        return {ErrorCode::kInvalidArgument,
                "GENEB decoder attention softmax is invalid"};
      }
      for (float &score : scores)
        score = activation_value(score / denominator, activation_dtype);
      for (std::size_t source = first; source <= row; ++source) {
        const float probability = scores[source - first];
        const std::size_t value_base =
            (source * key_value_heads + kv_head) * head_dimension;
        for (std::size_t dimension = 0; dimension < head_dimension;
             ++dimension) {
          (*output)[query_base + dimension] +=
              probability * value[value_base + dimension];
        }
      }
      for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
        const std::size_t index = query_base + dimension;
        (*output)[index] = activation_value((*output)[index], activation_dtype);
      }
    }
  }
  return finite_vector(*output)
             ? Status::Ok()
             : Status{ErrorCode::kInvalidArgument,
                      "GENEB decoder attention produced non-finite output"};
}

struct GenebDecoderModel::Impl final {
  struct Layer final {
    GenebDecoderTensorView attention_norm;
    GenebDecoderTensorView query_weight;
    GenebDecoderTensorView key_weight;
    GenebDecoderTensorView value_weight;
    GenebDecoderTensorView output_weight;
    GenebDecoderTensorView query_bias;
    GenebDecoderTensorView key_bias;
    GenebDecoderTensorView value_bias;
    GenebDecoderTensorView output_bias;
    GenebDecoderTensorView ffn_norm;
    GenebDecoderTensorView gate_weight;
    GenebDecoderTensorView up_weight;
    GenebDecoderTensorView down_weight;
    GenebDecoderTensorView gate_bias;
    GenebDecoderTensorView up_bias;
    GenebDecoderTensorView down_bias;
  };

  GenebDecoderTopology topology;
  GenebDecoderTensorView embedding;
  std::vector<Layer> layers;
  GenebDecoderTensorView final_norm;
  std::shared_ptr<evo::detail::LinearExecutor> executor;
};

GenebDecoderModel::GenebDecoderModel() = default;
GenebDecoderModel::~GenebDecoderModel() = default;
GenebDecoderModel::GenebDecoderModel(GenebDecoderModel &&) noexcept = default;
GenebDecoderModel &
GenebDecoderModel::operator=(GenebDecoderModel &&) noexcept = default;

Status GenebDecoderModel::load(
    const GenebDecoderTopology &topology,
    const std::vector<GenebDecoderNamedTensorView> &tensors,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  std::vector<GenebDecoderTensorRequirement> requirements;
  auto status = canonical_geneb_decoder_tensors(topology, &requirements);
  if (!status.ok())
    return status;
  std::map<std::string, const GenebDecoderTensorView *, std::less<>> provided;
  for (const auto &named : tensors) {
    if (!provided.emplace(named.name, &named.tensor).second) {
      return {ErrorCode::kModelFormat,
              "duplicate GENEB decoder tensor: " + named.name};
    }
  }
  std::set<std::string, std::less<>> expected;
  for (const auto &requirement : requirements)
    expected.insert(requirement.name);
  for (const auto &named : tensors) {
    if (expected.find(named.name) == expected.end()) {
      return {ErrorCode::kModelFormat,
              "unexpected GENEB decoder tensor: " + named.name};
    }
  }
  for (const auto &requirement : requirements) {
    const auto found = provided.find(requirement.name);
    if (found == provided.end()) {
      return {ErrorCode::kModelFormat,
              "required GENEB decoder tensor is missing: " + requirement.name};
    }
    status = tensor_requirement(*found->second, requirement);
    if (!status.ok())
      return status;
  }

  auto candidate = std::make_unique<Impl>();
  candidate->topology = topology;
  candidate->executor = std::move(linear_executor);
  const auto view = [&](const std::string &name) {
    return *provided.find(name)->second;
  };
  candidate->embedding = view("model.embed_tokens.weight");
  candidate->layers.resize(topology.layers);
  for (std::size_t layer = 0; layer < topology.layers; ++layer) {
    const std::string prefix = layer_prefix(layer);
    auto &target = candidate->layers[layer];
    target.attention_norm = view(prefix + "input_layernorm.weight");
    target.query_weight = view(prefix + "self_attn.q_proj.weight");
    target.key_weight = view(prefix + "self_attn.k_proj.weight");
    target.value_weight = view(prefix + "self_attn.v_proj.weight");
    target.output_weight = view(prefix + "self_attn.o_proj.weight");
    if (topology.attention_bias) {
      target.query_bias = view(prefix + "self_attn.q_proj.bias");
      target.key_bias = view(prefix + "self_attn.k_proj.bias");
      target.value_bias = view(prefix + "self_attn.v_proj.bias");
      target.output_bias = view(prefix + "self_attn.o_proj.bias");
    }
    target.ffn_norm = view(prefix + "post_attention_layernorm.weight");
    if (topology.mlp_activation == GenebDecoderMlpActivation::kSwiGlu) {
      target.gate_weight = view(prefix + "mlp.gate_proj.weight");
    }
    target.up_weight = view(prefix + "mlp.up_proj.weight");
    target.down_weight = view(prefix + "mlp.down_proj.weight");
    if (topology.mlp_bias) {
      if (topology.mlp_activation == GenebDecoderMlpActivation::kSwiGlu) {
        target.gate_bias = view(prefix + "mlp.gate_proj.bias");
      }
      target.up_bias = view(prefix + "mlp.up_proj.bias");
      target.down_bias = view(prefix + "mlp.down_proj.bias");
    }
  }
  candidate->final_norm = view("model.norm.weight");
  impl_ = std::move(candidate);
  return Status::Ok();
}

Status GenebDecoderModel::load(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  GenebDecoderTopology topology;
  auto status = geneb_decoder_topology_from_artifact(artifact, &topology);
  if (!status.ok())
    return status;
  std::vector<GenebDecoderNamedTensorView> views;
  views.reserve(artifact.tensors().size());
  for (const auto &tensor : artifact.tensors()) {
    if (tensor.rank == 0 ||
        tensor.data_size > std::numeric_limits<std::size_t>::max()) {
      return {ErrorCode::kModelFormat,
              "GENEB decoder artifact tensor rank/size is invalid: " +
                  tensor.name};
    }
    std::vector<std::size_t> shape;
    shape.reserve(tensor.rank);
    for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
      if (tensor.dimensions[dimension] == 0 ||
          tensor.dimensions[dimension] >
              std::numeric_limits<std::size_t>::max()) {
        return {ErrorCode::kModelFormat,
                "GENEB decoder artifact tensor dimension is invalid: " +
                    tensor.name};
      }
      shape.push_back(static_cast<std::size_t>(tensor.dimensions[dimension]));
    }
    const auto *const data = artifact.tensor_data(tensor);
    if (data == nullptr) {
      return {ErrorCode::kModelFormat,
              "GENEB decoder artifact tensor payload is unavailable: " +
                  tensor.name};
    }
    views.push_back({tensor.name,
                     {data, static_cast<std::size_t>(tensor.data_size),
                      tensor.dtype, std::move(shape)}});
  }
  return load(topology, views, std::move(linear_executor));
}

Status GenebDecoderModel::load_artifact(
    const ModelFile &artifact,
    std::shared_ptr<evo::detail::LinearExecutor> linear_executor) {
  return load(artifact, std::move(linear_executor));
}

const GenebDecoderTopology *GenebDecoderModel::topology() const noexcept {
  return impl_ == nullptr ? nullptr : &impl_->topology;
}

const char *GenebDecoderModel::linear_executor_name() const noexcept {
  if (impl_ == nullptr)
    return "unloaded";
  if (impl_->topology.f32_math_kernel ==
      GenebDecoderF32MathKernel::kTorch271AppleArm64ExactV1) {
    return kTorch271AppleArm64ExactF32Kernel;
  }
  return impl_->executor == nullptr ? "cpu-reference" : impl_->executor->name();
}

Status
GenebDecoderModel::forward(const std::vector<TokenId> &tokens,
                           const std::size_t position_offset,
                           const std::vector<std::size_t> &capture_layers,
                           GenebDecoderForwardResult *const output) const {
  if (output == nullptr)
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder forward output is null"};
  if (impl_ == nullptr)
    return {ErrorCode::kInvalidArgument, "GENEB decoder model is not loaded"};
  const auto &config = impl_->topology;
  const bool omnina_apple_exact =
      config.f32_math_kernel ==
      GenebDecoderF32MathKernel::kTorch271AppleArm64ExactV1;
  if (omnina_apple_exact && !detail::omnina_apple_f32_kernel_supported()) {
    return {ErrorCode::kUnsupported,
            "decoder.f32_math_kernel=torch-2.7.1-apple-arm64-exact-v1 "
            "requires Apple arm64"};
  }
  if (tokens.empty())
    return {ErrorCode::kInvalidArgument, "GENEB decoder token input is empty"};
  if (position_offset > config.maximum_sequence_length ||
      tokens.size() > config.maximum_sequence_length - position_offset) {
    return {ErrorCode::kInvalidArgument,
            "GENEB decoder tokens exceed maximum sequence length"};
  }
  std::set<std::size_t> captures;
  for (const std::size_t layer : capture_layers) {
    if (layer > config.layers || !captures.insert(layer).second) {
      return {ErrorCode::kInvalidArgument,
              "GENEB decoder capture layer is duplicate or out of range"};
    }
  }
  for (const TokenId token : tokens) {
    if (token >= config.vocabulary_size) {
      return {ErrorCode::kInvalidArgument,
              "GENEB decoder token id exceeds vocabulary"};
    }
  }

  GenebDecoderForwardResult result;
  result.rows = tokens.size();
  result.width = config.width;
  result.captures.reserve(capture_layers.size());
  for (const std::size_t layer : capture_layers)
    result.captures.push_back({layer, {}});
  const auto capture = [&](const std::size_t layer,
                           const std::vector<float> &hidden) {
    for (auto &target : result.captures) {
      if (target.layer == layer) {
        target.values = hidden;
        return;
      }
    }
  };

  std::vector<float> hidden(tokens.size() * config.width);
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    const std::size_t source =
        static_cast<std::size_t>(tokens[row]) * config.width;
    for (std::size_t column = 0; column < config.width; ++column)
      hidden[row * config.width + column] =
          activation_value(tensor_value(impl_->embedding, source + column),
                           config.activation_dtype);
  }
  capture(0, hidden);

  const std::size_t kv_width = config.key_value_heads * config.head_dimension;
  const std::size_t query_width = config.query_heads * config.head_dimension;
  for (std::size_t layer_index = 0; layer_index < config.layers;
       ++layer_index) {
    const auto &layer = impl_->layers[layer_index];
    std::vector<float> normalized;
    auto status =
        omnina_apple_exact
            ? detail::omnina_apple_f32_rms_norm(hidden, tokens.size(),
                                                layer.attention_norm,
                                                config.rms_epsilon, &normalized)
            : geneb_decoder_rms_norm(hidden, tokens.size(), config.width,
                                     layer.attention_norm, config.rms_epsilon,
                                     config.rms_epsilon_placement,
                                     config.activation_dtype, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    const GenebDecoderTensorView *query_bias =
        config.attention_bias ? &layer.query_bias : nullptr;
    const GenebDecoderTensorView *key_bias =
        config.attention_bias ? &layer.key_bias : nullptr;
    const GenebDecoderTensorView *value_bias =
        config.attention_bias ? &layer.value_bias : nullptr;
    status = runtime_linear(normalized, tokens.size(), config.width,
                            layer.query_weight, query_width, query_bias,
                            impl_->executor.get(), config.activation_dtype,
                            config.f32_math_kernel, &query);
    if (!status.ok())
      return status;
    status = runtime_linear(normalized, tokens.size(), config.width,
                            layer.key_weight, kv_width, key_bias,
                            impl_->executor.get(), config.activation_dtype,
                            config.f32_math_kernel, &key);
    if (!status.ok())
      return status;
    status = runtime_linear(normalized, tokens.size(), config.width,
                            layer.value_weight, kv_width, value_bias,
                            impl_->executor.get(), config.activation_dtype,
                            config.f32_math_kernel, &value);
    if (!status.ok())
      return status;
    status =
        omnina_apple_exact
            ? detail::omnina_apple_f32_apply_rope(&query, &key, tokens.size(),
                                                  position_offset)
            : geneb_decoder_apply_rope(
                  &query, config.query_heads, &key, config.key_value_heads,
                  tokens.size(), config.head_dimension, config.rotary_dimension,
                  position_offset, config.rope_base, config.rope_position_scale,
                  config.rope_layout, config.activation_dtype);
    if (!status.ok())
      return status;
    std::vector<float> attended;
    if (omnina_apple_exact) {
      status = detail::omnina_apple_f32_causal_attention(
          query, key, value, tokens.size(), &attended);
    } else if (config.attention_kernel ==
               GenebDecoderAttentionKernel::kTorchCpuFlashBf16Portable) {
      status = geneb_decoder_causal_attention_flash_bf16_portable(
          query, key, value, tokens.size(), config.query_heads,
          config.key_value_heads, config.head_dimension, &attended);
    } else {
      status = geneb_decoder_causal_attention(
          query, key, value, tokens.size(), config.query_heads,
          config.key_value_heads, config.head_dimension, config.sliding_window,
          config.activation_dtype, &attended);
    }
    if (!status.ok())
      return status;
    std::vector<float> projected;
    const GenebDecoderTensorView *output_bias =
        config.attention_bias ? &layer.output_bias : nullptr;
    status = runtime_linear(attended, tokens.size(), query_width,
                            layer.output_weight, config.width, output_bias,
                            impl_->executor.get(), config.activation_dtype,
                            config.f32_math_kernel, &projected);
    if (!status.ok())
      return status;
    add_in_place(&hidden, projected, config.activation_dtype);

    status = omnina_apple_exact
                 ? detail::omnina_apple_f32_rms_norm(
                       hidden, tokens.size(), layer.ffn_norm,
                       config.rms_epsilon, &normalized)
                 : geneb_decoder_rms_norm(hidden, tokens.size(), config.width,
                                          layer.ffn_norm, config.rms_epsilon,
                                          config.rms_epsilon_placement,
                                          config.activation_dtype, &normalized);
    if (!status.ok())
      return status;
    std::vector<float> up;
    const GenebDecoderTensorView *up_bias =
        config.mlp_bias ? &layer.up_bias : nullptr;
    status =
        runtime_linear(normalized, tokens.size(), config.width, layer.up_weight,
                       config.inner_width, up_bias, impl_->executor.get(),
                       config.activation_dtype, config.f32_math_kernel, &up);
    if (!status.ok())
      return status;
    if (config.mlp_activation == GenebDecoderMlpActivation::kSwiGlu) {
      std::vector<float> gate;
      const GenebDecoderTensorView *gate_bias =
          config.mlp_bias ? &layer.gate_bias : nullptr;
      status = runtime_linear(normalized, tokens.size(), config.width,
                              layer.gate_weight, config.inner_width, gate_bias,
                              impl_->executor.get(), config.activation_dtype,
                              config.f32_math_kernel, &gate);
      if (!status.ok())
        return status;
      if (omnina_apple_exact) {
        status = detail::omnina_apple_f32_swiglu(gate, &up);
        if (!status.ok())
          return status;
      } else {
        for (std::size_t index = 0; index < up.size(); ++index) {
          const float activated =
              activation_value(silu(gate[index]), config.activation_dtype);
          up[index] =
              activation_value(up[index] * activated, config.activation_dtype);
        }
      }
    } else {
      for (float &value_item : up)
        value_item =
            activation_value(gelu(value_item), config.activation_dtype);
    }
    std::vector<float> mlp_output;
    const GenebDecoderTensorView *down_bias =
        config.mlp_bias ? &layer.down_bias : nullptr;
    status = runtime_linear(up, tokens.size(), config.inner_width,
                            layer.down_weight, config.width, down_bias,
                            impl_->executor.get(), config.activation_dtype,
                            config.f32_math_kernel, &mlp_output);
    if (!status.ok())
      return status;
    add_in_place(&hidden, mlp_output, config.activation_dtype);
    if (!finite_vector(hidden)) {
      return {ErrorCode::kInvalidArgument,
              "GENEB decoder block produced non-finite hidden state"};
    }
    if (layer_index + 1U < config.layers)
      capture(layer_index + 1U, hidden);
  }
  auto status =
      omnina_apple_exact
          ? detail::omnina_apple_f32_rms_norm(
                hidden, tokens.size(), impl_->final_norm, config.rms_epsilon,
                &result.final_hidden)
          : geneb_decoder_rms_norm(
                hidden, tokens.size(), config.width, impl_->final_norm,
                config.rms_epsilon, config.rms_epsilon_placement,
                config.activation_dtype, &result.final_hidden);
  if (!status.ok())
    return status;
  capture(config.layers, result.final_hidden);
  *output = std::move(result);
  return Status::Ok();
}

} // namespace evo::cpu
