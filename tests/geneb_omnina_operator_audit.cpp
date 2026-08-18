// SPDX-License-Identifier: Apache-2.0
#include "../src/cpu/geneb_decoder_omnina_apple.hpp"

#include "evo/cpu/geneb_decoder.hpp"
#include "evo/model_format.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kWidth = 1024;
constexpr std::size_t kInnerWidth = 4096;
constexpr std::size_t kRows = 12;
constexpr std::string_view kExactF32Kernel = "torch-2.7.1-apple-arm64-exact-v1";
constexpr evo::TokenId kTokens[kRows] = {
    1,     2724,  30864, 30864, 30864, 30864,
    30864, 30864, 30864, 30864, 30864, 30864,
};

evo::Status tensor_view(const evo::ModelFile &artifact,
                        const std::string_view name,
                        evo::cpu::GenebDecoderTensorView *const output) {
  if (output == nullptr)
    return {evo::ErrorCode::kInvalidArgument, "tensor output is null"};
  const auto *const tensor = artifact.find_tensor(name);
  if (tensor == nullptr || tensor->rank == 0 ||
      tensor->data_size > std::numeric_limits<std::size_t>::max()) {
    return {evo::ErrorCode::kModelFormat,
            "missing or invalid OmniNA audit tensor: " + std::string{name}};
  }
  std::vector<std::size_t> shape;
  shape.reserve(tensor->rank);
  for (std::size_t index = 0; index < tensor->rank; ++index) {
    if (tensor->dimensions[index] == 0 ||
        tensor->dimensions[index] > std::numeric_limits<std::size_t>::max()) {
      return {evo::ErrorCode::kModelFormat,
              "invalid OmniNA audit tensor shape: " + std::string{name}};
    }
    shape.push_back(static_cast<std::size_t>(tensor->dimensions[index]));
  }
  const auto *const data = artifact.tensor_data(*tensor);
  if (data == nullptr) {
    return {evo::ErrorCode::kModelFormat,
            "unavailable OmniNA audit tensor data: " + std::string{name}};
  }
  *output = {data, static_cast<std::size_t>(tensor->data_size), tensor->dtype,
             std::move(shape)};
  return evo::Status::Ok();
}

evo::Status write_vector(const std::filesystem::path &directory,
                         const std::string_view name,
                         const std::vector<float> &values) {
  const auto path = directory / (std::string{name} + ".f32");
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    return {evo::ErrorCode::kIo,
            "cannot open operator output: " + path.string()};
  stream.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!stream)
    return {evo::ErrorCode::kIo,
            "cannot write operator output: " + path.string()};
  return evo::Status::Ok();
}

evo::Status run(const std::string &artifact_path,
                const std::filesystem::path &output_directory) {
  if (!evo::cpu::detail::omnina_apple_f32_kernel_supported()) {
    return {evo::ErrorCode::kUnsupported,
            "OmniNA operator audit requires Apple arm64"};
  }
  if (!std::filesystem::is_directory(output_directory) ||
      !std::filesystem::is_empty(output_directory)) {
    return {evo::ErrorCode::kInvalidArgument,
            "OmniNA operator output must be an existing empty directory"};
  }

  evo::ModelFile artifact;
  auto status = artifact.open(artifact_path);
  if (!status.ok())
    return status;
  evo::cpu::GenebDecoderModel model;
  status = model.load(artifact);
  if (!status.ok())
    return status;
  if (std::string_view{model.linear_executor_name()} != kExactF32Kernel) {
    return {evo::ErrorCode::kModelFormat,
            "OmniNA exact F32 kernel reports the wrong executor name"};
  }
  const auto *const topology = model.topology();
  if (topology == nullptr ||
      topology->f32_math_kernel !=
          evo::cpu::GenebDecoderF32MathKernel::kTorch271AppleArm64ExactV1) {
    return {evo::ErrorCode::kModelFormat,
            "artifact does not select the OmniNA exact F32 kernel"};
  }

  const auto view = [&](const std::string_view name,
                        evo::cpu::GenebDecoderTensorView *const output) {
    return tensor_view(artifact, name, output);
  };
  evo::cpu::GenebDecoderTensorView embedding;
  evo::cpu::GenebDecoderTensorView attention_norm_scale;
  evo::cpu::GenebDecoderTensorView query_weight;
  evo::cpu::GenebDecoderTensorView key_weight;
  evo::cpu::GenebDecoderTensorView value_weight;
  evo::cpu::GenebDecoderTensorView attention_output_weight;
  evo::cpu::GenebDecoderTensorView ffn_norm_scale;
  evo::cpu::GenebDecoderTensorView gate_weight;
  evo::cpu::GenebDecoderTensorView up_weight;
  evo::cpu::GenebDecoderTensorView down_weight;
  for (const auto &entry : {
           std::pair<std::string_view, evo::cpu::GenebDecoderTensorView *>{
               "model.embed_tokens.weight", &embedding},
           {"model.layers.0.input_layernorm.weight", &attention_norm_scale},
           {"model.layers.0.self_attn.q_proj.weight", &query_weight},
           {"model.layers.0.self_attn.k_proj.weight", &key_weight},
           {"model.layers.0.self_attn.v_proj.weight", &value_weight},
           {"model.layers.0.self_attn.o_proj.weight", &attention_output_weight},
           {"model.layers.0.post_attention_layernorm.weight", &ffn_norm_scale},
           {"model.layers.0.mlp.gate_proj.weight", &gate_weight},
           {"model.layers.0.mlp.up_proj.weight", &up_weight},
           {"model.layers.0.mlp.down_proj.weight", &down_weight},
       }) {
    status = view(entry.first, entry.second);
    if (!status.ok())
      return status;
  }
  if (embedding.dtype != evo::TensorDType::kF32 ||
      embedding.shape != std::vector<std::size_t>{32001, kWidth}) {
    return {evo::ErrorCode::kModelFormat,
            "OmniNA audit embedding tensor differs"};
  }
  const auto *const embedding_values =
      reinterpret_cast<const float *>(embedding.data);
  std::vector<float> hidden(kRows * kWidth);
  for (std::size_t row = 0; row < kRows; ++row) {
    const std::size_t source = static_cast<std::size_t>(kTokens[row]) * kWidth;
    std::copy_n(embedding_values + source, kWidth,
                hidden.data() + row * kWidth);
  }

  std::vector<float> attention_norm;
  status = evo::cpu::detail::omnina_apple_f32_rms_norm(
      hidden, kRows, attention_norm_scale, 1.0e-6F, &attention_norm);
  if (!status.ok())
    return status;
  if (!(status =
            write_vector(output_directory, "attention_norm", attention_norm))
           .ok())
    return status;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  status = evo::cpu::detail::omnina_apple_f32_linear(
      attention_norm, kRows, kWidth, query_weight, kWidth, nullptr, &query);
  if (!status.ok())
    return status;
  status = evo::cpu::detail::omnina_apple_f32_linear(
      attention_norm, kRows, kWidth, key_weight, kWidth, nullptr, &key);
  if (!status.ok())
    return status;
  status = evo::cpu::detail::omnina_apple_f32_linear(
      attention_norm, kRows, kWidth, value_weight, kWidth, nullptr, &value);
  if (!status.ok())
    return status;
  for (const auto &entry : {
           std::pair<std::string_view, const std::vector<float> *>{
               "query_linear", &query},
           {"key_linear", &key},
           {"value_linear", &value},
       }) {
    status = write_vector(output_directory, entry.first, *entry.second);
    if (!status.ok())
      return status;
  }

  status =
      evo::cpu::detail::omnina_apple_f32_apply_rope(&query, &key, kRows, 0);
  if (!status.ok())
    return status;
  for (const auto &entry : {
           std::pair<std::string_view, const std::vector<float> *>{"query_rope",
                                                                   &query},
           {"key_rope", &key},
       }) {
    status = write_vector(output_directory, entry.first, *entry.second);
    if (!status.ok())
      return status;
  }

  std::vector<float> attended;
  status = evo::cpu::detail::omnina_apple_f32_causal_attention(
      query, key, value, kRows, &attended);
  if (!status.ok())
    return status;
  status = write_vector(output_directory, "attended", attended);
  if (!status.ok())
    return status;
  std::vector<float> attention_projected;
  status = evo::cpu::detail::omnina_apple_f32_linear(
      attended, kRows, kWidth, attention_output_weight, kWidth, nullptr,
      &attention_projected);
  if (!status.ok())
    return status;
  status = write_vector(output_directory, "attention_projected",
                        attention_projected);
  if (!status.ok())
    return status;
  for (std::size_t index = 0; index < hidden.size(); ++index)
    hidden[index] += attention_projected[index];
  status = write_vector(output_directory, "attention_residual", hidden);
  if (!status.ok())
    return status;

  std::vector<float> ffn_norm;
  status = evo::cpu::detail::omnina_apple_f32_rms_norm(
      hidden, kRows, ffn_norm_scale, 1.0e-6F, &ffn_norm);
  if (!status.ok())
    return status;
  status = write_vector(output_directory, "ffn_norm", ffn_norm);
  if (!status.ok())
    return status;
  std::vector<float> up;
  std::vector<float> gate;
  status = evo::cpu::detail::omnina_apple_f32_linear(
      ffn_norm, kRows, kWidth, up_weight, kInnerWidth, nullptr, &up);
  if (!status.ok())
    return status;
  status = evo::cpu::detail::omnina_apple_f32_linear(
      ffn_norm, kRows, kWidth, gate_weight, kInnerWidth, nullptr, &gate);
  if (!status.ok())
    return status;
  status = write_vector(output_directory, "up_linear", up);
  if (!status.ok())
    return status;
  status = write_vector(output_directory, "gate_linear", gate);
  if (!status.ok())
    return status;
  status = evo::cpu::detail::omnina_apple_f32_swiglu(gate, &up);
  if (!status.ok())
    return status;
  status = write_vector(output_directory, "swiglu", up);
  if (!status.ok())
    return status;
  std::vector<float> mlp_projected;
  status = evo::cpu::detail::omnina_apple_f32_linear(
      up, kRows, kInnerWidth, down_weight, kWidth, nullptr, &mlp_projected);
  if (!status.ok())
    return status;
  status = write_vector(output_directory, "mlp_projected", mlp_projected);
  if (!status.ok())
    return status;
  for (std::size_t index = 0; index < hidden.size(); ++index)
    hidden[index] += mlp_projected[index];
  return write_vector(output_directory, "block_output", hidden);
}

} // namespace

int main(const int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: evo-geneb-omnina-operator-audit ARTIFACT OUTPUT_DIR\n";
    return 2;
  }
  const auto status = run(argv[1], std::filesystem::path{argv[2]});
  if (!status.ok()) {
    std::cerr << "GENEB OmniNA operator audit failed: " << status.message()
              << '\n';
    return status.code() == evo::ErrorCode::kUnsupported ? 77 : 1;
  }
  std::cout << "GENEB OmniNA block0 operator audit passed\n";
  return 0;
}
