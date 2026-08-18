// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../src/linear_executor.hpp"
#include "evo/cpu/geneb_esm.hpp"
#include "evo/cpu/model.hpp"
#include "evo/model_format.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

bool close(const float left, const float right,
           const float tolerance = 2.0e-5F) {
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
  for (const auto dimension : shape)
    result *= dimension;
  return result;
}

float fixture_value(const std::string &name, const std::size_t tensor_index,
                    const std::size_t element_index) {
  const int integer =
      static_cast<int>(
          ((tensor_index + 3U) * 19U + (element_index + 5U) * 11U) % 43U) -
      21;
  float value = static_cast<float>(integer) / 31.0F;
  if (name.find("LayerNorm.weight") != std::string::npos ||
      name.find("layer_norm_after.weight") != std::string::npos)
    value = 1.0F + value * 0.04F;
  else if (name.size() >= 5U && name.substr(name.size() - 5U) == ".bias")
    value *= 0.04F;
  else if (name.find("embeddings.") != std::string::npos)
    value *= 0.12F;
  else
    value *= 0.16F;
  return value;
}

evo::cpu::GenebEsmTopology absolute_topology() {
  evo::cpu::GenebEsmTopology topology;
  topology.vocabulary_size = 8;
  topology.width = 6;
  topology.layers = 2;
  topology.heads = 2;
  topology.head_dimension = 3;
  topology.inner_width = 8;
  topology.maximum_sequence_length = 6;
  topology.position_embedding_count = 8;
  topology.layer_norm_epsilon = 1.0e-5F;
  topology.rope_base = 10000.0F;
  topology.position_type = evo::cpu::GenebEsmPositionType::kAbsolute;
  topology.mlp_activation = evo::cpu::GenebEsmMlpActivation::kGelu;
  topology.attention_bias = true;
  topology.feed_forward_bias = true;
  topology.token_dropout = true;
  topology.pad_token_id = 1;
  topology.mask_token_id = 2;
  topology.cls_token_id = 3;
  return topology;
}

evo::cpu::GenebEsmTopology rotary_topology() {
  auto topology = absolute_topology();
  topology.width = 8;
  topology.head_dimension = 4;
  topology.inner_width = 7;
  topology.position_type = evo::cpu::GenebEsmPositionType::kRotary;
  topology.mlp_activation = evo::cpu::GenebEsmMlpActivation::kSwiGlu;
  topology.feed_forward_bias = false;
  topology.token_dropout = false;
  return topology;
}

struct FixtureTensor final {
  std::string name;
  std::vector<std::uint8_t> bytes;
  std::vector<std::size_t> shape;
};

struct Fixture final {
  evo::cpu::GenebEsmTopology topology;
  std::vector<FixtureTensor> storage;
  std::vector<evo::cpu::GenebEsmNamedTensorView> views;
};

void rebuild_views(Fixture *const fixture) {
  fixture->views.clear();
  fixture->views.reserve(fixture->storage.size());
  for (const auto &tensor : fixture->storage) {
    fixture->views.push_back({tensor.name,
                              {tensor.bytes.data(), tensor.bytes.size(),
                               evo::TensorDType::kF32, tensor.shape}});
  }
}

void build_fixture(const evo::cpu::GenebEsmTopology &topology,
                   Fixture *const output) {
  output->topology = topology;
  std::vector<evo::cpu::GenebEsmTensorRequirement> requirements;
  const auto status =
      evo::cpu::canonical_geneb_esm_tensors(topology, &requirements);
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
    tensor.shape = requirement.shape;
    const std::size_t count = elements(tensor.shape);
    tensor.bytes.resize(count * sizeof(float));
    for (std::size_t index = 0; index < count; ++index) {
      const float value = fixture_value(tensor.name, tensor_index, index);
      std::memcpy(tensor.bytes.data() + index * sizeof(float), &value,
                  sizeof(value));
    }
  }
  rebuild_views(output);
}

float stored_value(const Fixture &fixture, const std::string_view name,
                   const std::size_t index) {
  const auto found = std::find_if(
      fixture.storage.begin(), fixture.storage.end(),
      [&](const FixtureTensor &tensor) { return tensor.name == name; });
  if (found == fixture.storage.end())
    std::abort();
  float value = 0.0F;
  std::memcpy(&value, found->bytes.data() + index * sizeof(value),
              sizeof(value));
  return value;
}

float linear_value(const evo::detail::LinearTensorView tensor,
                   const std::size_t index) {
  float value = 0.0F;
  std::memcpy(&value, tensor.data + index * sizeof(value), sizeof(value));
  return value;
}

class CountingExecutor final : public evo::detail::LinearExecutor {
public:
  [[nodiscard]] const char *name() const noexcept override {
    return "geneb-esm-counting";
  }

  [[nodiscard]] evo::Status
  linear(const float *const input, const std::size_t rows,
         const std::size_t input_width,
         const evo::detail::LinearTensorView weight,
         const std::size_t output_width,
         const evo::detail::LinearTensorView *const bias,
         std::vector<float> *const output) override {
    ++calls;
    if (input == nullptr || output == nullptr || weight.data == nullptr)
      return {evo::ErrorCode::kInvalidArgument, "counting executor null data"};
    output->assign(rows * output_width, 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t target = 0; target < output_width; ++target) {
        float total = bias == nullptr ? 0.0F : linear_value(*bias, target);
        for (std::size_t source = 0; source < input_width; ++source)
          total += input[row * input_width + source] *
                   linear_value(weight, target * input_width + source);
        (*output)[row * output_width + target] = total;
      }
    }
    return evo::Status::Ok();
  }

  std::size_t calls{0};
};

evo::cpu::GenebEsmForwardResult
run_or_abort(const Fixture &fixture,
             std::shared_ptr<evo::detail::LinearExecutor> executor = {}) {
  evo::cpu::GenebEsmModel model;
  auto status =
      model.load(fixture.topology, fixture.views, std::move(executor));
  if (!status.ok()) {
    std::cerr << "fixture load failed: " << status.message() << '\n';
    std::abort();
  }
  const std::vector<evo::TokenId> tokens =
      fixture.topology.token_dropout
          ? std::vector<evo::TokenId>{3, 4, 2, 5, 1, 1}
          : std::vector<evo::TokenId>{3, 4, 6, 5, 1, 1};
  evo::cpu::GenebEsmForwardResult result;
  status = model.forward(tokens, {1, 1, 1, 1, 0, 0}, {0, 1, 2}, &result);
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

void dump_result(const std::string_view prefix,
                 const evo::cpu::GenebEsmForwardResult &result,
                 bool *const first) {
  for (const auto &capture : result.captures)
    dump_named_vector(std::string{prefix} + ".capture_" +
                          std::to_string(capture.layer),
                      capture.values, first);
  dump_named_vector(std::string{prefix} + ".final_hidden", result.final_hidden,
                    first);
  dump_named_vector(std::string{prefix} + ".pooled", result.pooled, first);
}

void dump_fixture_json() {
  Fixture absolute;
  Fixture rotary;
  build_fixture(absolute_topology(), &absolute);
  build_fixture(rotary_topology(), &rotary);
  const auto absolute_result = run_or_abort(absolute);
  const auto rotary_result = run_or_abort(rotary);
  std::cout << "{\"vectors\":{";
  bool first = true;
  dump_result("absolute", absolute_result, &first);
  dump_result("rotary", rotary_result, &first);
  std::cout << "}}\n";
}

int dump_exact_layer_norm_bits() {
  constexpr std::size_t kRows = 2U;
  constexpr std::size_t kWidth = 1500U;
  std::vector<float> input(kRows * kWidth, 0.0F);
  for (std::size_t index = 0; index < input.size(); ++index) {
    const int integer = static_cast<int>(((index + 3U) * 37U) % 257U) - 128;
    input[index] = static_cast<float>(integer) / 31.0F;
  }
  std::vector<float> scale(kWidth, 0.0F);
  std::vector<float> bias(kWidth, 0.0F);
  for (std::size_t index = 0; index < kWidth; ++index) {
    const int scale_integer = static_cast<int>(((index + 5U) * 19U) % 67U) - 33;
    const int bias_integer = static_cast<int>(((index + 7U) * 23U) % 59U) - 29;
    scale[index] = 1.0F + static_cast<float>(scale_integer) / 211.0F;
    bias[index] = static_cast<float>(bias_integer) / 307.0F;
  }
  const evo::cpu::GenebEsmTensorView scale_view{
      reinterpret_cast<const std::uint8_t *>(scale.data()),
      scale.size() * sizeof(float),
      evo::TensorDType::kF32,
      {kWidth}};
  const evo::cpu::GenebEsmTensorView bias_view{
      reinterpret_cast<const std::uint8_t *>(bias.data()),
      bias.size() * sizeof(float),
      evo::TensorDType::kF32,
      {kWidth}};
  std::vector<float> output;
  const auto status = evo::cpu::geneb_esm_layer_norm(
      input, kRows, kWidth, scale_view, bias_view, 1.0e-12F, &output,
      evo::cpu::GenebEsmLayerNormKernel::kTorch212AppleArm64ExactV1);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return status.code() == evo::ErrorCode::kUnsupported ? 77 : 2;
  }
  std::cout << "{\"bits\":[";
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &output[index], sizeof(bits));
    std::cout << bits;
  }
  std::cout << "]}\n";
  return 0;
}

void test_topology_and_manifest() {
  auto absolute = absolute_topology();
  check(evo::cpu::validate_geneb_esm_topology(absolute).ok(),
        "absolute topology permits odd Agro-like head dimensions");
  auto invalid_rotary = absolute;
  invalid_rotary.position_type = evo::cpu::GenebEsmPositionType::kRotary;
  check(!evo::cpu::validate_geneb_esm_topology(invalid_rotary).ok(),
        "rotary topology rejects odd head dimensions");
  auto invalid_positions = absolute;
  invalid_positions.position_embedding_count = 7;
  check(!evo::cpu::validate_geneb_esm_topology(invalid_positions).ok(),
        "position cumsum maximum is checked");
  auto exact = absolute;
  exact.width = 1500U;
  exact.heads = 20U;
  exact.head_dimension = 75U;
  exact.layer_norm_epsilon = 1.0e-12F;
  exact.layer_norm_kernel =
      evo::cpu::GenebEsmLayerNormKernel::kTorch212AppleArm64ExactV1;
  check(evo::cpu::validate_geneb_esm_topology(exact).ok(),
        "Agro exact affine LayerNorm topology is admitted");
  auto wrong_exact_width = exact;
  wrong_exact_width.width = 1499U;
  check(!evo::cpu::validate_geneb_esm_topology(wrong_exact_width).ok(),
        "exact LayerNorm rejects non-Agro width");
  auto wrong_exact_epsilon = exact;
  wrong_exact_epsilon.layer_norm_epsilon = 1.0e-5F;
  check(!evo::cpu::validate_geneb_esm_topology(wrong_exact_epsilon).ok(),
        "exact LayerNorm rejects non-Agro epsilon");

  std::vector<evo::cpu::GenebEsmTensorRequirement> absolute_tensors;
  std::vector<evo::cpu::GenebEsmTensorRequirement> rotary_tensors;
  check(evo::cpu::canonical_geneb_esm_tensors(absolute, &absolute_tensors).ok(),
        "absolute manifest builds");
  check(
      evo::cpu::canonical_geneb_esm_tensors(rotary_topology(), &rotary_tensors)
          .ok(),
      "rotary manifest builds");
  const auto has_name = [](const auto &requirements,
                           const std::string_view name) {
    return std::any_of(requirements.begin(), requirements.end(),
                       [&](const auto &item) { return item.name == name; });
  };
  check(has_name(absolute_tensors, "esm.embeddings.position_embeddings.weight"),
        "absolute runtime retains learned position table");
  check(!has_name(rotary_tensors, "esm.embeddings.position_embeddings.weight"),
        "rotary runtime omits unused learned position table");
  check(
      !has_name(rotary_tensors, "esm.encoder.layer.0.intermediate.dense.bias"),
      "v2 manifest omits feed-forward bias");
  const auto fused = std::find_if(
      rotary_tensors.begin(), rotary_tensors.end(), [](const auto &item) {
        return item.name == "esm.encoder.layer.0.intermediate.dense.weight";
      });
  check(fused != rotary_tensors.end() &&
            fused->shape == std::vector<std::size_t>({14, 8}),
        "v2 intermediate projection has 2*inner rows");
}

void test_embedding_and_pooling() {
  Fixture fixture;
  build_fixture(absolute_topology(), &fixture);
  const auto result = run_or_abort(fixture);
  check(result.rows == 6 && result.width == 6 && result.captures.size() == 3 &&
            result.final_hidden.size() == 36 && result.pooled.size() == 6 &&
            finite(result.final_hidden),
        "absolute forward returns finite hidden/captures/pool");
  const auto &embedding = result.captures[0].values;
  const float token_scale = 0.88F / 0.75F;
  const float expected_cls =
      stored_value(fixture, "esm.embeddings.word_embeddings.weight", 3U * 6U) *
          token_scale +
      stored_value(fixture, "esm.embeddings.position_embeddings.weight",
                   2U * 6U);
  check(close(embedding[0], expected_cls),
        "absolute CLS uses cumsum position 2 and token-dropout scale");
  const float expected_mask = stored_value(
      fixture, "esm.embeddings.position_embeddings.weight", 4U * 6U);
  check(close(embedding[2U * 6U], expected_mask),
        "mask token word embedding is zero before absolute position add");
  check(std::all_of(embedding.begin() + 4 * 6, embedding.end(),
                    [](const float value) { return value == 0.0F; }),
        "embedding output zeros right-padding rows");
  for (std::size_t column = 0; column < result.width; ++column) {
    float expected = 0.0F;
    for (std::size_t row = 0; row < 4; ++row)
      expected += result.final_hidden[row * result.width + column];
    expected /= 4.0F;
    check(result.pooled[column] == expected,
          "pool uses direct F32 division and includes CLS");
  }

  auto no_dropout_topology = absolute_topology();
  no_dropout_topology.token_dropout = false;
  Fixture no_dropout;
  build_fixture(no_dropout_topology, &no_dropout);
  const auto no_dropout_result = run_or_abort(no_dropout);
  check(!close(result.captures[0].values[0],
               no_dropout_result.captures[0].values[0]),
        "token-dropout flag changes non-mask word embedding scale");
}

void test_helpers() {
  std::vector<float> query{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  auto key = query;
  auto status =
      evo::cpu::geneb_esm_apply_rotary(&query, &key, 2, 1, 4, 10000.0F);
  check(status.ok(), "split-half RoPE helper accepts valid tensors");
  check(query[0] == 1.0F && query[1] == 2.0F && query[2] == 3.0F &&
            query[3] == 4.0F,
        "RoPE position zero is identity");
  const float cosine = std::cos(1.0F);
  const float sine = std::sin(1.0F);
  check(close(query[4], 5.0F * cosine - 7.0F * sine),
        "RoPE uses split-half first pair");
  check(close(query[6], 7.0F * cosine + 5.0F * sine),
        "RoPE uses split-half second half");

  const std::vector<float> zeros(6, 0.0F);
  const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F, 100.0F, 100.0F};
  std::vector<float> attended;
  status = evo::cpu::geneb_esm_bidirectional_attention(
      zeros, zeros, values, {1, 1, 0}, 3, 1, 2, &attended);
  check(status.ok(), "bidirectional attention helper accepts key mask");
  for (std::size_t row = 0; row < 3; ++row) {
    check(close(attended[row * 2U], 2.0F) &&
              close(attended[row * 2U + 1U], 3.0F),
          "bidirectional attention sees future valid key and excludes pad key");
  }
}

void test_loader_and_errors() {
  Fixture fixture;
  build_fixture(rotary_topology(), &fixture);
  evo::cpu::GenebEsmModel model;
  auto status = model.load(fixture.topology, fixture.views);
  check(status.ok() &&
            std::string_view{model.linear_executor_name()} == "cpu-reference",
        "typed loader accepts exact rotary tensor set");

  auto missing = fixture.views;
  missing.pop_back();
  check(!model.load(fixture.topology, missing).ok(),
        "typed loader rejects missing tensor");
  auto extra = fixture.views;
  extra.push_back(extra.front());
  extra.back().name = "esm.extra.weight";
  check(!model.load(fixture.topology, extra).ok(),
        "typed loader rejects extra tensor");
  auto duplicate = fixture.views;
  duplicate.push_back(duplicate.front());
  check(!model.load(fixture.topology, duplicate).ok(),
        "typed loader rejects duplicate tensor");
  auto wrong_shape = fixture.views;
  wrong_shape.front().tensor.shape[0] += 1U;
  check(!model.load(fixture.topology, wrong_shape).ok(),
        "typed loader rejects wrong tensor shape");
  auto wrong_dtype = fixture.views;
  wrong_dtype.front().tensor.dtype = evo::TensorDType::kBF16;
  check(!model.load(fixture.topology, wrong_dtype).ok(),
        "typed loader rejects wrong tensor dtype");

  status = model.load(fixture.topology, fixture.views);
  check(status.ok(), "model reloads after negative gates");
  evo::cpu::GenebEsmForwardResult output;
  check(!model.forward({3, 4, 1}, {1, 0, 1}, {}, &output).ok(),
        "forward rejects non-right-padded mask");
  check(!model.forward({3, 4, 5}, {1, 1, 0}, {}, &output).ok(),
        "forward requires masked token to be pad ID");
  check(!model.forward({1}, {1}, {}, &output).ok(),
        "forward rejects valid pad token");
  check(!model.forward({1}, {0}, {}, &output).ok(),
        "forward rejects all-padding input");
  check(!model.forward({99}, {1}, {}, &output).ok(),
        "forward rejects out-of-vocabulary token");
  check(!model.forward({3}, {1}, {1, 1}, &output).ok(),
        "forward rejects duplicate captures");

  auto executor = std::make_shared<CountingExecutor>();
  status = model.load(fixture.topology, fixture.views, executor);
  check(status.ok() && std::string_view{model.linear_executor_name()} ==
                           "geneb-esm-counting",
        "custom LinearExecutor is exposed");
  status = model.forward({3, 4, 6, 1}, {1, 1, 1, 0}, {}, &output);
  check(status.ok() && executor->calls == fixture.topology.layers * 6U,
        "all six per-layer projections route through LinearExecutor");
  bool distinguishes_direct_division = false;
  for (std::size_t column = 0; column < output.width; ++column) {
    float sum = 0.0F;
    for (std::size_t row = 0; row < 3U; ++row)
      sum += output.final_hidden[row * output.width + column];
    const float expected = sum / 3.0F;
    distinguishes_direct_division =
        distinguishes_direct_division || expected != sum * (1.0F / 3.0F);
    check(output.pooled[column] == expected,
          "ESM private pool uses exact direct F32 division");
  }
  check(distinguishes_direct_division,
        "ESM fixture distinguishes division from reciprocal multiply");
}

int verify_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebEsmModel model;
  status = model.load_artifact(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 3;
  }
  const auto *const topology = model.topology();
  if (topology == nullptr) {
    std::cerr << "loaded artifact has no topology\n";
    return 4;
  }
  evo::cpu::GenebEsmForwardResult output;
  status = model.forward({static_cast<evo::TokenId>(topology->cls_token_id)},
                         {1}, {topology->layers}, &output);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 5;
  }
  return 0;
}

int verify_cpu_adapter(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << "GENEB ESM adapter artifact open failed: " << status.message()
              << '\n';
    return 2;
  }
  evo::cpu::GenebEsmModel direct;
  status = direct.load_artifact(artifact);
  if (!status.ok()) {
    std::cerr << "GENEB ESM direct load failed: " << status.message() << '\n';
    return 3;
  }
  evo::cpu::Model model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << "GENEB ESM CPU Model load failed: " << status.message()
              << '\n';
    return 4;
  }
  std::vector<evo::TokenId> tokens;
  status = model.encode("AAAAAA", &tokens);
  const auto *const topology = direct.topology();
  if (!status.ok() || topology == nullptr || tokens.size() < 2U ||
      tokens.front() != static_cast<evo::TokenId>(topology->cls_token_id) ||
      std::any_of(tokens.begin(), tokens.end(), [&](const evo::TokenId token) {
        return token >= topology->vocabulary_size;
      })) {
    std::cerr << "GENEB ESM artifact tokenizer binding failed\n";
    return 5;
  }
  evo::cpu::Context context;
  status = context.initialize_shared(model, model.config().max_seqlen);
  if (!status.ok()) {
    std::cerr << "GENEB ESM CPU context failed: " << status.message() << '\n';
    return 6;
  }
  std::vector<float> adapter_hidden;
  status =
      context.prefill_embedding(tokens, model.config().layers, &adapter_hidden);
  if (!status.ok()) {
    std::cerr << "GENEB ESM CPU adapter forward failed: " << status.message()
              << '\n';
    return 7;
  }
  evo::cpu::GenebEsmForwardResult direct_result;
  status = direct.forward(tokens, std::vector<std::uint8_t>(tokens.size(), 1U),
                          {model.config().layers}, &direct_result);
  if (!status.ok() || direct_result.captures.size() != 1 ||
      adapter_hidden != direct_result.captures.front().values ||
      context.position() != tokens.size() ||
      model.config().implementation !=
          evo::ArchitectureImplementation::kGenebEsmEncoder) {
    std::cerr << "GENEB ESM CPU adapter differs from typed runtime\n";
    return 8;
  }
  return 0;
}

} // namespace

int main(const int argc, char **const argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--dump-json") {
    dump_fixture_json();
    return 0;
  }
  if (argc == 2 && std::string_view{argv[1]} == "--dump-exact-layer-norm-bits")
    return dump_exact_layer_norm_bits();
  if (argc == 3 && std::string_view{argv[1]} == "--verify-artifact")
    return verify_artifact(argv[2]);
  if (argc == 3 && std::string_view{argv[1]} == "--verify-cpu-adapter")
    return verify_cpu_adapter(argv[2]);
  if (argc != 1) {
    std::cerr << "usage: test_geneb_esm [--dump-json|"
                 "--dump-exact-layer-norm-bits|--verify-artifact PATH|"
                 "--verify-cpu-adapter PATH]\n";
    return 64;
  }
  test_topology_and_manifest();
  test_embedding_and_pooling();
  test_helpers();
  test_loader_and_errors();
  if (failures != 0)
    std::cerr << failures << " GENEB ESM test(s) failed\n";
  return failures == 0 ? 0 : 1;
}
