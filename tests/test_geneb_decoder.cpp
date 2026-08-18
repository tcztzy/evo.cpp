// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../src/cpu/geneb_decoder_omnina_apple.hpp"
#include "../src/linear_executor.hpp"
#include "evo/cpu/geneb_decoder.hpp"
#include "evo/cpu/model.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

bool close(const float left, const float right,
           const float tolerance = 1.0e-5F) {
  return std::abs(left - right) <= tolerance;
}

bool finite(const std::vector<float> &values) {
  for (const float value : values) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

std::size_t elements(const std::vector<std::size_t> &shape) {
  std::size_t result = 1;
  for (const std::size_t dimension : shape)
    result *= dimension;
  return result;
}

std::uint16_t bf16(const float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

float bf16_float(const float value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(bf16(value)) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::vector<float> bf16_values(const std::initializer_list<float> values) {
  std::vector<float> result;
  result.reserve(values.size());
  for (const float value : values)
    result.push_back(bf16_float(value));
  return result;
}

float fixture_value(const std::string &name, const std::size_t tensor_index,
                    const std::size_t element_index) {
  const auto integer =
      static_cast<int>(
          ((tensor_index + 1U) * 17U + (element_index + 3U) * 13U) % 37U) -
      18;
  float value = static_cast<float>(integer) / 29.0F;
  if (name.find("layernorm.weight") != std::string::npos ||
      name == "model.norm.weight") {
    value = 1.0F + value * 0.05F;
  } else if (name.size() >= 5 && name.substr(name.size() - 5) == ".bias") {
    value *= 0.1F;
  } else {
    value *= 0.2F;
  }
  return value;
}

evo::cpu::GenebDecoderTopology fixture_topology() {
  evo::cpu::GenebDecoderTopology topology;
  topology.vocabulary_size = 11;
  topology.width = 8;
  topology.layers = 2;
  topology.query_heads = 2;
  topology.key_value_heads = 1;
  topology.head_dimension = 4;
  topology.rotary_dimension = 4;
  topology.inner_width = 12;
  topology.maximum_sequence_length = 16;
  topology.sliding_window = 3;
  topology.rms_epsilon = 1.0e-5F;
  topology.rms_epsilon_placement =
      evo::cpu::GenebDecoderRmsEpsilonPlacement::kInsideSqrt;
  topology.rope_base = 500000.0F;
  topology.rope_position_scale = 1.0F;
  topology.rope_layout = evo::cpu::GenebDecoderRopeLayout::kSplitHalf;
  topology.mlp_activation = evo::cpu::GenebDecoderMlpActivation::kSwiGlu;
  topology.attention_bias = true;
  topology.mlp_bias = true;
  topology.activation_dtype = evo::TensorDType::kF32;
  return topology;
}

evo::cpu::GenebDecoderTopology omnina_topology() {
  evo::cpu::GenebDecoderTopology topology;
  topology.vocabulary_size = 32001;
  topology.width = 1024;
  topology.layers = 16;
  topology.query_heads = 16;
  topology.key_value_heads = 16;
  topology.head_dimension = 64;
  topology.rotary_dimension = 64;
  topology.inner_width = 4096;
  topology.maximum_sequence_length = 2048;
  topology.sliding_window = 0;
  topology.rms_epsilon = 1.0e-6F;
  topology.rms_epsilon_placement =
      evo::cpu::GenebDecoderRmsEpsilonPlacement::kInsideSqrt;
  topology.rope_base = 10000.0F;
  topology.rope_position_scale = 1.0F;
  topology.rope_layout = evo::cpu::GenebDecoderRopeLayout::kSplitHalf;
  topology.mlp_activation = evo::cpu::GenebDecoderMlpActivation::kSwiGlu;
  topology.attention_kernel = evo::cpu::GenebDecoderAttentionKernel::kEager;
  topology.f32_math_kernel =
      evo::cpu::GenebDecoderF32MathKernel::kTorch271AppleArm64ExactV1;
  topology.attention_bias = false;
  topology.mlp_bias = false;
  topology.embedding_dtype = evo::TensorDType::kF32;
  topology.projection_dtype = evo::TensorDType::kF32;
  topology.norm_dtype = evo::TensorDType::kF32;
  topology.activation_dtype = evo::TensorDType::kF32;
  return topology;
}

struct FixtureTensor final {
  std::string name;
  std::vector<std::uint8_t> bytes;
  evo::TensorDType dtype{evo::TensorDType::kF32};
  std::vector<std::size_t> shape;
};

struct Fixture final {
  evo::cpu::GenebDecoderTopology topology;
  std::vector<FixtureTensor> storage;
  std::vector<evo::cpu::GenebDecoderNamedTensorView> views;
};

void build_fixture(const evo::cpu::GenebDecoderTopology &topology,
                   Fixture *const output) {
  output->topology = topology;
  std::vector<evo::cpu::GenebDecoderTensorRequirement> requirements;
  const auto status =
      evo::cpu::canonical_geneb_decoder_tensors(topology, &requirements);
  if (!status.ok()) {
    std::cerr << "fixture topology failed: " << status.message() << '\n';
    std::abort();
  }
  output->storage.clear();
  output->storage.resize(requirements.size());
  for (std::size_t tensor_index = 0; tensor_index < requirements.size();
       ++tensor_index) {
    const auto &requirement = requirements[tensor_index];
    auto &tensor = output->storage[tensor_index];
    tensor.name = requirement.name;
    tensor.dtype = requirement.dtype;
    tensor.shape = requirement.shape;
    const std::size_t count = elements(tensor.shape);
    const std::size_t item_size = tensor.dtype == evo::TensorDType::kF32
                                      ? sizeof(float)
                                      : sizeof(std::uint16_t);
    tensor.bytes.resize(count * item_size);
    for (std::size_t index = 0; index < count; ++index) {
      const float value = fixture_value(tensor.name, tensor_index, index);
      if (tensor.dtype == evo::TensorDType::kF32) {
        std::memcpy(tensor.bytes.data() + index * sizeof(float), &value,
                    sizeof(value));
      } else {
        const std::uint16_t encoded = bf16(value);
        std::memcpy(tensor.bytes.data() + index * sizeof(encoded), &encoded,
                    sizeof(encoded));
      }
    }
  }
  output->views.clear();
  output->views.reserve(output->storage.size());
  for (const auto &tensor : output->storage) {
    output->views.push_back({tensor.name,
                             {tensor.bytes.data(), tensor.bytes.size(),
                              tensor.dtype, tensor.shape}});
  }
}

float linear_value(const evo::detail::LinearTensorView tensor,
                   const std::size_t index) {
  if (tensor.dtype == evo::TensorDType::kBF16) {
    std::uint16_t encoded = 0;
    std::memcpy(&encoded, tensor.data + index * sizeof(encoded),
                sizeof(encoded));
    const std::uint32_t bits = static_cast<std::uint32_t>(encoded) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return value;
}

class CountingExecutor final : public evo::detail::LinearExecutor {
public:
  [[nodiscard]] const char *name() const noexcept override {
    return "counting-reference";
  }

  [[nodiscard]] evo::Status
  linear(const float *const input, const std::size_t rows,
         const std::size_t input_width,
         const evo::detail::LinearTensorView weight,
         const std::size_t output_width,
         const evo::detail::LinearTensorView *const bias,
         std::vector<float> *const output) override {
    ++calls;
    if (input == nullptr || output == nullptr)
      return {evo::ErrorCode::kInvalidArgument, "counting executor null input"};
    output->assign(rows * output_width, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t target = 0; target < output_width; ++target) {
        float total = bias == nullptr ? 0.0F : linear_value(*bias, target);
        for (std::size_t source = 0; source < input_width; ++source) {
          total += input[row * input_width + source] *
                   linear_value(weight, target * input_width + source);
        }
        (*output)[row * output_width + target] = total;
      }
    }
    return evo::Status::Ok();
  }

  std::size_t calls{0};
};

evo::cpu::GenebDecoderForwardResult
run_or_abort(const Fixture &fixture,
             std::shared_ptr<evo::detail::LinearExecutor> executor = {}) {
  evo::cpu::GenebDecoderModel model;
  auto status =
      model.load(fixture.topology, fixture.views, std::move(executor));
  if (!status.ok()) {
    std::cerr << "fixture load failed: " << status.message() << '\n';
    std::abort();
  }
  evo::cpu::GenebDecoderForwardResult result;
  status = model.forward({1, 4, 2, 7}, 3, {0, 1, 2}, &result);
  if (!status.ok()) {
    std::cerr << "fixture forward failed: " << status.message() << '\n';
    std::abort();
  }
  return result;
}

void dump_vector(const std::vector<float> &values) {
  std::cout << '[' << std::setprecision(9);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

void dump_named_vector(const std::string_view name,
                       const std::vector<float> &values, bool *const first) {
  if (!*first)
    std::cout << ',';
  *first = false;
  std::cout << '\"' << name << "\":";
  dump_vector(values);
}

void dump_result_json(const evo::cpu::GenebDecoderForwardResult &result,
                      const std::string_view prefix, bool *const first) {
  for (std::size_t index = 0; index < result.captures.size(); ++index) {
    dump_named_vector(std::string{prefix} + ".capture_" +
                          std::to_string(result.captures[index].layer),
                      result.captures[index].values, first);
  }
  dump_named_vector(std::string{prefix} + ".final_hidden", result.final_hidden,
                    first);
}

void dump_fixture_json() {
  Fixture f32_fixture;
  build_fixture(fixture_topology(), &f32_fixture);
  const auto f32_result = run_or_abort(f32_fixture);

  auto bf16_topology = fixture_topology();
  bf16_topology.embedding_dtype = evo::TensorDType::kBF16;
  bf16_topology.projection_dtype = evo::TensorDType::kBF16;
  bf16_topology.norm_dtype = evo::TensorDType::kBF16;
  bf16_topology.activation_dtype = evo::TensorDType::kBF16;
  Fixture bf16_fixture;
  build_fixture(bf16_topology, &bf16_fixture);
  const auto bf16_result = run_or_abort(bf16_fixture);

  std::cout << "{\"vectors\":{";
  bool first = true;
  dump_result_json(f32_result, "f32", &first);
  dump_result_json(bf16_result, "bf16", &first);
  const auto probe_input = bf16_values({0.1001F, -0.3003F, 0.7007F, -1.1009F});
  const auto probe_scale = bf16_values({1.003F, 0.997F, 1.011F, 0.989F});
  std::vector<std::uint16_t> probe_scale_storage;
  probe_scale_storage.reserve(probe_scale.size());
  for (const float value : probe_scale)
    probe_scale_storage.push_back(bf16(value));
  const evo::cpu::GenebDecoderTensorView scale_view{
      reinterpret_cast<const std::uint8_t *>(probe_scale_storage.data()),
      probe_scale_storage.size() * sizeof(std::uint16_t),
      evo::TensorDType::kBF16,
      {probe_scale_storage.size()}};
  std::vector<float> rms_probe;
  auto status = evo::cpu::geneb_decoder_rms_norm(
      probe_input, 1, probe_input.size(), scale_view, 1.0e-5F,
      evo::cpu::GenebDecoderRmsEpsilonPlacement::kInsideSqrt,
      evo::TensorDType::kBF16, &rms_probe);
  if (!status.ok()) {
    std::cerr << "BF16 RMSNorm probe failed: " << status.message() << '\n';
    std::abort();
  }
  dump_named_vector("bf16.rmsnorm_probe", rms_probe, &first);
  auto rope_query = probe_input;
  auto rope_key = probe_input;
  status = evo::cpu::geneb_decoder_apply_rope(
      &rope_query, 1, &rope_key, 1, 1, 4, 4, 7, 10000.0F, 1.0F,
      evo::cpu::GenebDecoderRopeLayout::kSplitHalf, evo::TensorDType::kBF16);
  if (!status.ok()) {
    std::cerr << "BF16 RoPE probe failed: " << status.message() << '\n';
    std::abort();
  }
  dump_named_vector("bf16.rope_split_probe", rope_query, &first);
  std::cout << "}}\n";
}

void test_topology_and_names() {
  auto topology = fixture_topology();
  check(evo::cpu::validate_geneb_decoder_topology(topology).ok(),
        "GQA topology with explicit head dimension validates");
  std::vector<evo::cpu::GenebDecoderTensorRequirement> requirements;
  check(
      evo::cpu::canonical_geneb_decoder_tensors(topology, &requirements).ok() &&
          requirements.size() == 34,
      "bias-enabled two-layer SwiGLU canonical tensor set is exact");

  topology.width = 640;
  topology.query_heads = 32;
  topology.key_value_heads = 8;
  topology.head_dimension = 128;
  topology.rotary_dimension = 128;
  topology.inner_width = 2048;
  topology.vocabulary_size = 512;
  topology.attention_bias = false;
  topology.mlp_bias = false;
  check(evo::cpu::canonical_geneb_decoder_tensors(topology, &requirements).ok(),
        "BioFM nonstandard attention width topology validates");
  const auto find = [&](const std::string_view name) {
    for (const auto &requirement : requirements) {
      if (requirement.name == name)
        return requirement.shape;
    }
    return std::vector<std::size_t>{};
  };
  check(find("model.layers.0.self_attn.q_proj.weight") ==
                std::vector<std::size_t>({4096, 640}) &&
            find("model.layers.0.self_attn.k_proj.weight") ==
                std::vector<std::size_t>({1024, 640}) &&
            find("model.layers.0.self_attn.o_proj.weight") ==
                std::vector<std::size_t>({640, 4096}),
        "canonical shapes preserve BioFM explicit Q/KV attention widths");

  topology.key_value_heads = 7;
  check(!evo::cpu::validate_geneb_decoder_topology(topology).ok(),
        "topology rejects non-divisible GQA heads");
  topology = fixture_topology();
  topology.embedding_dtype = evo::TensorDType::kE4M3Software;
  check(!evo::cpu::validate_geneb_decoder_topology(topology).ok(),
        "topology rejects undeclared FP8 reference weights");
  topology = fixture_topology();
  topology.activation_dtype = evo::TensorDType::kE4M3Software;
  check(!evo::cpu::validate_geneb_decoder_topology(topology).ok(),
        "topology rejects undeclared FP8 activation semantics");

  topology = omnina_topology();
  check(evo::cpu::validate_geneb_decoder_topology(topology).ok(),
        "OmniNA exact Apple-arm64 F32 kernel accepts only its closed topology");
  topology.width = 1025;
  check(!evo::cpu::validate_geneb_decoder_topology(topology).ok(),
        "OmniNA exact F32 kernel rejects topology drift");
  topology = omnina_topology();
  topology.attention_kernel =
      evo::cpu::GenebDecoderAttentionKernel::kTorchCpuFlashBf16Portable;
  check(!evo::cpu::validate_geneb_decoder_topology(topology).ok(),
        "OmniNA exact F32 kernel cannot compose with another attention kernel");
  topology = fixture_topology();
  topology.f32_math_kernel =
      static_cast<evo::cpu::GenebDecoderF32MathKernel>(255);
  check(!evo::cpu::validate_geneb_decoder_topology(topology).ok(),
        "topology rejects an unknown F32 math kernel enum");
}

void test_omnina_host_gate() {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  check(evo::cpu::detail::omnina_apple_f32_kernel_supported(),
        "OmniNA exact F32 kernel reports Apple-arm64 support");
#else
  check(!evo::cpu::detail::omnina_apple_f32_kernel_supported(),
        "OmniNA exact F32 kernel reports unsupported on non-Apple-arm64");
  std::vector<float> output;
  const evo::cpu::GenebDecoderTensorView weight{};
  const auto status = evo::cpu::detail::omnina_apple_f32_linear(
      {}, 0, 0, weight, 0, nullptr, &output);
  check(!status.ok() && status.code() == evo::ErrorCode::kUnsupported,
        "non-Apple-arm64 OmniNA exact operator returns typed unsupported");
#endif
}

void test_strict_loader() {
  Fixture fixture;
  build_fixture(fixture_topology(), &fixture);
  evo::cpu::GenebDecoderModel model;
  check(model.load(fixture.topology, fixture.views).ok() &&
            model.topology() != nullptr &&
            std::string_view{model.linear_executor_name()} == "cpu-reference",
        "strict canonical F32 model loads");

  auto corrupted = fixture.views;
  corrupted.pop_back();
  auto status = model.load(fixture.topology, corrupted);
  check(!status.ok() && status.code() == evo::ErrorCode::kModelFormat &&
            status.message().find("missing") != std::string::npos,
        "loader rejects a missing canonical tensor");
  corrupted = fixture.views;
  corrupted.push_back(corrupted.front());
  status = model.load(fixture.topology, corrupted);
  check(!status.ok() && status.message().find("duplicate") != std::string::npos,
        "loader rejects duplicate tensor names");
  corrupted = fixture.views;
  corrupted.push_back({"model.extra.weight", corrupted.front().tensor});
  status = model.load(fixture.topology, corrupted);
  check(!status.ok() &&
            status.message().find("unexpected") != std::string::npos,
        "loader rejects extra tensor names");
  corrupted = fixture.views;
  corrupted.front().tensor.shape[0] += 1;
  status = model.load(fixture.topology, corrupted);
  check(!status.ok() && status.message().find("shape") != std::string::npos,
        "loader rejects a wrong tensor shape");
  corrupted = fixture.views;
  corrupted.front().tensor.dtype = evo::TensorDType::kBF16;
  status = model.load(fixture.topology, corrupted);
  check(!status.ok() && status.message().find("dtype") != std::string::npos,
        "loader rejects a wrong tensor dtype");
  corrupted = fixture.views;
  --corrupted.front().tensor.bytes;
  status = model.load(fixture.topology, corrupted);
  check(!status.ok() && status.message().find("payload") != std::string::npos,
        "loader rejects a truncated tensor payload");
  check(model.topology() != nullptr,
        "failed reload leaves the prior valid model intact");

  evo::ModelFile unopened_artifact;
  status = model.load(unopened_artifact);
  check(!status.ok() && status.code() == evo::ErrorCode::kModelFormat &&
            status.message().find("profile") != std::string::npos &&
            model.topology() != nullptr,
        "ModelFile overload rejects an unopened artifact without replacing "
        "the prior valid model");
}

void test_rms_rope_and_attention() {
  const std::vector<float> scale_values{1.0F, 2.0F};
  const evo::cpu::GenebDecoderTensorView scale{
      reinterpret_cast<const std::uint8_t *>(scale_values.data()),
      scale_values.size() * sizeof(float),
      evo::TensorDType::kF32,
      {2}};
  const std::vector<float> input{3.0F, 4.0F};
  std::vector<float> inside;
  std::vector<float> after;
  check(evo::cpu::geneb_decoder_rms_norm(
            input, 1, 2, scale, 0.25F,
            evo::cpu::GenebDecoderRmsEpsilonPlacement::kInsideSqrt,
            evo::TensorDType::kF32, &inside)
                .ok() &&
            evo::cpu::geneb_decoder_rms_norm(
                input, 1, 2, scale, 0.25F,
                evo::cpu::GenebDecoderRmsEpsilonPlacement::kAfterSqrt,
                evo::TensorDType::kF32, &after)
                .ok() &&
            close(inside[0], 3.0F / std::sqrt(12.75F)) &&
            close(after[0], 3.0F / (std::sqrt(12.5F) + 0.25F)) &&
            !close(inside[0], after[0]),
        "RMSNorm makes epsilon placement explicit");

  std::vector<float> split_query{1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> split_key = split_query;
  auto adjacent_query = split_query;
  auto adjacent_key = split_key;
  check(evo::cpu::geneb_decoder_apply_rope(
            &split_query, 1, &split_key, 1, 1, 4, 4, 1, 10000.0F, 1.0F,
            evo::cpu::GenebDecoderRopeLayout::kSplitHalf,
            evo::TensorDType::kF32)
                .ok() &&
            evo::cpu::geneb_decoder_apply_rope(
                &adjacent_query, 1, &adjacent_key, 1, 1, 4, 4, 1, 10000.0F,
                1.0F, evo::cpu::GenebDecoderRopeLayout::kAdjacentPairs,
                evo::TensorDType::kF32)
                .ok() &&
            split_query != adjacent_query,
        "split-half and adjacent-pair RoPE layouts are distinct");

  const std::vector<float> query{1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
  const std::vector<float> key{1.0F, 2.0F, 3.0F};
  const std::vector<float> value{10.0F, 20.0F, 30.0F};
  std::vector<float> windowed;
  std::vector<float> full;
  check(evo::cpu::geneb_decoder_causal_attention(
            query, key, value, 3, 2, 1, 1, 1, evo::TensorDType::kF32, &windowed)
                .ok() &&
            windowed == std::vector<float>(
                            {10.0F, 10.0F, 20.0F, 20.0F, 30.0F, 30.0F}) &&
            evo::cpu::geneb_decoder_causal_attention(
                query, key, value, 3, 2, 1, 1, 0, evo::TensorDType::kF32, &full)
                .ok() &&
            full.back() < 30.0F,
        "GQA maps query heads to shared KV heads and honors sliding window");

  const std::vector<float> mha_query{1.0F, 1.0F};
  const std::vector<float> mha_key{1.0F, 1.0F};
  const std::vector<float> mha_value{7.0F, 11.0F};
  check(evo::cpu::geneb_decoder_causal_attention(mha_query, mha_key, mha_value,
                                                 1, 2, 2, 1, 0,
                                                 evo::TensorDType::kF32, &full)
                .ok() &&
            full == mha_value,
        "same attention kernel covers ordinary MHA");
}

void test_forward_and_executor() {
  Fixture fixture;
  build_fixture(fixture_topology(), &fixture);
  evo::cpu::GenebDecoderModel model;
  check(model.load(fixture.topology, fixture.views).ok(),
        "fixture model loads for forward");
  evo::cpu::GenebDecoderForwardResult result;
  auto status = model.forward({1, 4, 2, 7}, 3, {2, 0, 1}, &result);
  check(status.ok() && result.rows == 4 && result.width == 8 &&
            result.captures.size() == 3 && result.captures[0].layer == 2 &&
            result.captures[1].layer == 0 && result.captures[2].layer == 1 &&
            result.final_hidden.size() == 32 && finite(result.final_hidden),
        "forward preserves requested per-layer capture order and final norm");
  check(result.captures[0].values == result.final_hidden,
        "highest exposed layer is the post-final-RMSNorm representation");
  check(!model.forward({11}, 0, {}, &result).ok(),
        "forward rejects an out-of-vocabulary token");
  check(!model.forward({1}, 0, {1, 1}, &result).ok(),
        "forward rejects duplicate hidden captures");
  check(!model.forward({1, 2}, 15, {}, &result).ok(),
        "forward rejects context overflow before activation allocation");

  auto executor = std::make_shared<CountingExecutor>();
  evo::cpu::GenebDecoderModel delegated;
  check(delegated.load(fixture.topology, fixture.views, executor).ok() &&
            std::string_view{delegated.linear_executor_name()} ==
                "counting-reference" &&
            delegated.forward({1, 4, 2, 7}, 3, {}, &result).ok() &&
            executor->calls == 14,
        "all seven per-layer GEMMs delegate through LinearExecutor");

  auto bf16_topology = fixture_topology();
  bf16_topology.embedding_dtype = evo::TensorDType::kBF16;
  bf16_topology.projection_dtype = evo::TensorDType::kBF16;
  bf16_topology.norm_dtype = evo::TensorDType::kBF16;
  bf16_topology.activation_dtype = evo::TensorDType::kBF16;
  bf16_topology.sliding_window = 0;
  Fixture bf16_fixture;
  build_fixture(bf16_topology, &bf16_fixture);
  evo::cpu::GenebDecoderModel bf16_model;
  check(bf16_model.load(bf16_fixture.topology, bf16_fixture.views).ok() &&
            bf16_model.forward({1, 4, 2, 7}, 3, {}, &result).ok() &&
            finite(result.final_hidden) &&
            std::all_of(
                result.final_hidden.begin(), result.final_hidden.end(),
                [](const float value) { return value == bf16_float(value); }),
        "BF16 reference activations are RNE-rounded at the final eager "
        "boundary");

  const auto eager_bf16_hidden = result.final_hidden;
  auto flash_topology = bf16_topology;
  flash_topology.attention_kernel =
      evo::cpu::GenebDecoderAttentionKernel::kTorchCpuFlashBf16Portable;
  Fixture flash_fixture;
  build_fixture(flash_topology, &flash_fixture);
  evo::cpu::GenebDecoderModel flash_model;
  check(flash_model.load(flash_fixture.topology, flash_fixture.views).ok() &&
            flash_model.forward({1, 4, 2, 7}, 3, {}, &result).ok() &&
            finite(result.final_hidden) &&
            result.final_hidden != eager_bf16_hidden &&
            std::all_of(
                result.final_hidden.begin(), result.final_hidden.end(),
                [](const float value) { return value == bf16_float(value); }),
        "portable CPU Flash-BF16 attention is explicit and preserves BF16 "
        "output boundaries");
  auto invalid_flash_topology = flash_topology;
  invalid_flash_topology.activation_dtype = evo::TensorDType::kF32;
  check(!flash_model.load(invalid_flash_topology, flash_fixture.views).ok(),
        "portable CPU Flash attention rejects non-BF16 activation profiles");

  auto gelu_topology = fixture_topology();
  gelu_topology.mlp_activation = evo::cpu::GenebDecoderMlpActivation::kGelu;
  gelu_topology.attention_bias = false;
  gelu_topology.mlp_bias = false;
  Fixture gelu_fixture;
  build_fixture(gelu_topology, &gelu_fixture);
  evo::cpu::GenebDecoderModel gelu_model;
  check(gelu_model.load(gelu_fixture.topology, gelu_fixture.views).ok() &&
            gelu_model.forward({1, 4}, 0, {}, &result).ok() &&
            finite(result.final_hidden),
        "standard exact-GELU MLP topology executes without a gate tensor");
}

int verify_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << "GENEB decoder artifact open failed: " << status.message()
              << '\n';
    return 1;
  }
  evo::cpu::GenebDecoderModel model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << "GENEB decoder artifact load failed: " << status.message()
              << '\n';
    return 1;
  }
  std::cout << "GENEB decoder artifact verified\n";
  return 0;
}

int verify_cpu_adapter(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << "GENEB decoder adapter artifact open failed: "
              << status.message() << '\n';
    return 1;
  }
  evo::cpu::GenebDecoderModel direct;
  status = direct.load(artifact);
  if (!status.ok()) {
    std::cerr << "GENEB decoder direct load failed: " << status.message()
              << '\n';
    return 1;
  }
  evo::cpu::Model model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << "GENEB decoder CPU Model load failed: " << status.message()
              << '\n';
    return 1;
  }
  std::vector<evo::TokenId> tokens;
  status = model.encode("ACGT", &tokens);
  const auto &config = model.config();
  const bool tiny_tokens = config.model_id != "geneb-decoder-tiny" ||
                           tokens == std::vector<evo::TokenId>{0, 1, 2, 3};
  const bool genome_ocean_tokens =
      (config.model_id != "geneb-genome-ocean-4b" &&
       config.model_id != "geneb-genome-ocean-500m") ||
      tokens == std::vector<evo::TokenId>{1, 29, 9, 2};
  const bool tokens_in_range =
      std::all_of(tokens.begin(), tokens.end(),
                  [&](const auto token) { return token < config.vocab_size; });
  if (!status.ok() || tokens.empty() || !tokens_in_range || !tiny_tokens ||
      !genome_ocean_tokens) {
    std::cerr << "GENEB decoder artifact tokenizer binding failed for "
              << model.config().model_id << ":";
    for (const auto token : tokens)
      std::cerr << ' ' << token;
    std::cerr << '\n';
    return 1;
  }
  evo::cpu::Context context;
  status = context.initialize_shared(model, 16);
  if (!status.ok()) {
    std::cerr << "GENEB decoder CPU context failed: " << status.message()
              << '\n';
    return 1;
  }
  std::vector<float> adapter_hidden;
  status = context.prefill_embedding(tokens, 1, &adapter_hidden);
  if (!status.ok()) {
    std::cerr << "GENEB decoder CPU adapter forward failed: "
              << status.message() << '\n';
    return 1;
  }
  evo::cpu::GenebDecoderForwardResult direct_result;
  status = direct.forward(tokens, 0, {1}, &direct_result);
  if (!status.ok() || direct_result.captures.size() != 1 ||
      adapter_hidden != direct_result.captures[0].values ||
      context.position() != tokens.size() ||
      model.config().implementation !=
          evo::ArchitectureImplementation::kGenebTransformerDecoder) {
    std::cerr << "GENEB decoder CPU adapter differs from typed runtime\n";
    return 1;
  }
  std::cout << "GENEB decoder CPU adapter verified\n";
  return 0;
}

} // namespace

int main(const int argc, char **argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--dump-json") {
    dump_fixture_json();
    return 0;
  }
  if (argc == 3 && std::string_view{argv[1]} == "--verify-artifact")
    return verify_artifact(argv[2]);
  if (argc == 3 && std::string_view{argv[1]} == "--verify-cpu-adapter")
    return verify_cpu_adapter(argv[2]);
  if (argc != 1) {
    std::cerr << "usage: evo-geneb-decoder-tests "
                 "[--dump-json|--verify-artifact PATH|"
                 "--verify-cpu-adapter PATH]\n";
    return 2;
  }
  test_topology_and_names();
  test_omnina_host_gate();
  test_strict_loader();
  test_rms_rope_and_attention();
  test_forward_and_executor();
  if (failures != 0) {
    std::cerr << failures << " GENEB decoder test(s) failed\n";
    return 1;
  }
  std::cout << "GENEB decoder tests passed\n";
  return 0;
}
