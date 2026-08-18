// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "evo/cpu/geneb_mamba.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

std::size_t elements(const std::vector<std::size_t> &shape) {
  std::size_t result = 1U;
  for (const auto dimension : shape)
    result *= dimension;
  return result;
}

evo::cpu::GenebMambaTopology caduceus_topology(const bool rcps) {
  evo::cpu::GenebMambaTopology topology;
  topology.variant = evo::cpu::GenebMambaVariant::kCaduceusMamba1;
  topology.vocabulary_size = 6U;
  topology.tokenizer_vocabulary_size = 6U;
  topology.width = 2U;
  topology.output_width = rcps ? 4U : 2U;
  topology.layers = 1U;
  topology.maximum_sequence_length = 131072U;
  topology.inner_width = 4U;
  topology.state_width = 2U;
  topology.convolution_width = 2U;
  topology.time_step_rank = 1U;
  topology.norm_epsilon = 1.0e-5F;
  topology.reverse_complement_parameter_sharing = rcps;
  if (rcps)
    topology.complement_map = {0U, 1U, 2U, 4U, 3U, 5U};
  return topology;
}

evo::cpu::GenebMambaTopology ecc_topology() {
  evo::cpu::GenebMambaTopology topology;
  topology.variant = evo::cpu::GenebMambaVariant::kEccDnaMamba2;
  topology.vocabulary_size = 6U;
  topology.tokenizer_vocabulary_size = 6U;
  topology.width = 2U;
  topology.output_width = 2U;
  topology.layers = 1U;
  topology.maximum_sequence_length = 0U;
  topology.inner_width = 4U;
  topology.state_width = 2U;
  topology.convolution_width = 2U;
  topology.mlp_width = 3U;
  topology.head_width = 2U;
  topology.heads = 2U;
  topology.groups = 1U;
  topology.norm_epsilon = 1.0e-5F;
  return topology;
}

float fixture_value(const std::string &name, const std::size_t tensor_index,
                    const std::size_t element_index) {
  const auto integer =
      static_cast<int>(
          ((tensor_index + 3U) * 17U + (element_index + 7U) * 13U) % 41U) -
      20;
  float value = static_cast<float>(integer) / 37.0F;
  if (name.find("norm.weight") != std::string::npos ||
      name == "final_norm.weight")
    return 1.0F + value * 0.035F;
  if (name.find("A_log") != std::string::npos)
    return -0.7F + value * 0.05F;
  if (name.find("dt_proj.bias") != std::string::npos ||
      name.find("dt_bias") != std::string::npos)
    return -1.2F + value * 0.08F;
  if (name.size() >= 2U && name.substr(name.size() - 2U) == ".D")
    return 0.6F + value * 0.04F;
  if (name.find("embedding.weight") != std::string::npos)
    return value * 0.22F;
  return value * 0.11F;
}

struct StoredTensor final {
  std::string name;
  std::vector<std::uint8_t> bytes;
  std::vector<std::size_t> shape;
};

struct Fixture final {
  evo::cpu::GenebMambaTopology topology;
  std::vector<StoredTensor> storage;
  std::vector<evo::cpu::GenebMambaNamedTensorView> views;
};

void build_fixture(const evo::cpu::GenebMambaTopology &topology,
                   Fixture *const output) {
  output->topology = topology;
  std::vector<evo::cpu::GenebMambaTensorRequirement> requirements;
  const auto status =
      evo::cpu::canonical_geneb_mamba_tensors(topology, &requirements);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  output->storage.resize(requirements.size());
  for (std::size_t tensor_index = 0U; tensor_index < requirements.size();
       ++tensor_index) {
    auto &stored = output->storage[tensor_index];
    stored.name = requirements[tensor_index].name;
    stored.shape = requirements[tensor_index].shape;
    stored.bytes.resize(elements(stored.shape) * sizeof(float));
    for (std::size_t index = 0U; index < elements(stored.shape); ++index) {
      const float value = fixture_value(stored.name, tensor_index, index);
      std::memcpy(stored.bytes.data() + index * sizeof(float), &value,
                  sizeof(value));
    }
  }
  output->views.clear();
  output->views.reserve(output->storage.size());
  for (const auto &stored : output->storage) {
    output->views.push_back({stored.name,
                             {stored.bytes.data(), stored.bytes.size(),
                              evo::TensorDType::kF32, stored.shape}});
  }
}

struct RunResult final {
  evo::cpu::GenebMambaForwardResult forward;
  std::vector<float> pooled;
};

evo::Status run_inner_norm_primitive(const int mode,
                                     std::vector<float> *const output) {
  const auto values = [](const std::size_t count, const float scale,
                         const int shift) {
    std::vector<float> result(count);
    for (std::size_t index = 0U; index < count; ++index) {
      const int value = static_cast<int>((index * 11U + 5U) % 17U) - shift;
      result[index] = static_cast<float>(value) * scale;
    }
    return result;
  };
  const std::vector<float> input{0.2F, -0.3F, 0.7F, 0.1F, -0.4F, 0.5F};
  const auto input_projection = values(16U, 0.035F, 8);
  const auto convolution_weight = values(8U, 0.021F, 7);
  const auto convolution_bias = values(4U, 0.013F, 6);
  const auto x_projection = values(24U, 0.027F, 8);
  const auto time_step_projection = values(8U, 0.019F, 7);
  const auto time_step_bias = values(4U, 0.041F, 10);
  const auto a_log = values(8U, 0.017F, 5);
  const auto skip = values(4U, 0.023F, 4);
  const auto output_projection = values(8U, 0.031F, 8);
  const std::vector<float> dt_scale{0.8F, 1.1F};
  const std::vector<float> b_scale{1.2F, 0.9F};
  const std::vector<float> c_scale{0.7F, 1.3F};
  const auto view = [](const std::vector<float> &storage,
                       std::vector<std::size_t> shape) {
    return evo::cpu::MambaTensorView{
        reinterpret_cast<const std::uint8_t *>(storage.data()),
        storage.size() * sizeof(float), evo::TensorDType::kF32,
        std::move(shape)};
  };
  evo::cpu::Mamba1Weights weights;
  weights.input_projection = view(input_projection, {8U, 2U});
  weights.convolution_weight = view(convolution_weight, {4U, 1U, 2U});
  weights.convolution_bias = view(convolution_bias, {4U});
  weights.x_projection = view(x_projection, {6U, 4U});
  weights.time_step_projection = view(time_step_projection, {4U, 2U});
  weights.time_step_bias = view(time_step_bias, {4U});
  weights.a_log = view(a_log, {4U, 2U});
  weights.skip = view(skip, {4U});
  weights.output_projection = view(output_projection, {2U, 4U});
  if (mode != 0) {
    weights.projected_time_step_norm_scale = view(dt_scale, {2U});
    if (mode != 2)
      weights.projected_b_norm_scale = view(b_scale, {2U});
    weights.projected_c_norm_scale = view(c_scale, {2U});
  }
  const evo::cpu::Mamba1Config config{2U, 4U, 2U, 2U, 2U, mode != 0, 1.0e-6F};
  if (mode == 3) {
    auto disabled = config;
    disabled.parameter_projection_rms_norm = false;
    return evo::cpu::mamba1_mixer_f32(input, 3U, disabled, weights, nullptr,
                                      output);
  }
  return evo::cpu::mamba1_mixer_f32(input, 3U, config, weights, nullptr,
                                    output);
}

RunResult run_fixture(const Fixture &fixture) {
  evo::cpu::GenebMambaModel model;
  auto status = model.load(fixture.topology, fixture.views);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  RunResult result;
  status = model.forward({3U, 4U, 5U}, {1U, 1U, 0U},
                         {0U, fixture.topology.layers}, &result.forward);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  status = model.pool(result.forward, {1U, 1U, 0U}, &result.pooled);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    std::abort();
  }
  return result;
}

void dump_vector(const std::vector<float> &values) {
  std::cout << '[' << std::setprecision(9);
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U)
      std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}

void dump_run(const std::string_view name, const RunResult &result,
              bool *const first) {
  const auto emit = [&](const std::string &suffix,
                        const std::vector<float> &values) {
    if (!*first)
      std::cout << ',';
    *first = false;
    std::cout << '"' << name << '.' << suffix << "\":";
    dump_vector(values);
  };
  for (const auto &capture : result.forward.captures)
    emit("capture" + std::to_string(capture.layer), capture.values);
  emit("final", result.forward.final_hidden);
  emit("pooled", result.pooled);
}

int dump_fixture_json() {
  Fixture plain;
  Fixture rcps;
  Fixture ecc;
  build_fixture(caduceus_topology(false), &plain);
  build_fixture(caduceus_topology(true), &rcps);
  build_fixture(ecc_topology(), &ecc);
  const auto plain_result = run_fixture(plain);
  const auto rcps_result = run_fixture(rcps);
  const auto ecc_result = run_fixture(ecc);
  std::vector<float> inner_norm;
  const auto primitive_status = run_inner_norm_primitive(1, &inner_norm);
  if (!primitive_status.ok()) {
    std::cerr << primitive_status.message() << '\n';
    return 2;
  }
  bool first = true;
  std::cout << "{\"vectors\":{";
  dump_run("caduceus", plain_result, &first);
  dump_run("rcps", rcps_result, &first);
  dump_run("ecc", ecc_result, &first);
  if (!first)
    std::cout << ',';
  std::cout << "\"primitive.inner_norm\":";
  dump_vector(inner_norm);
  std::cout << "}}\n";
  return 0;
}

int verify_artifact(const std::string &path) {
  evo::ModelFile artifact;
  auto status = artifact.open(path);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebMambaModel model;
  status = model.load(artifact);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  evo::cpu::GenebMambaForwardResult result;
  status = model.forward({3U, 4U, 5U}, {1U, 1U, 0U}, {model.topology()->layers},
                         &result);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::vector<float> pooled;
  status = model.pool(result, {1U, 1U, 0U}, &pooled);
  if (!status.ok()) {
    std::cerr << status.message() << '\n';
    return 2;
  }
  std::cout << "{\"rows\":" << result.rows << ",\"width\":" << result.width
            << ",\"pooled\":";
  dump_vector(pooled);
  std::cout << "}\n";
  return 0;
}

void test_topology_and_manifest() {
  auto plain = caduceus_topology(false);
  auto rcps = caduceus_topology(true);
  auto ecc = ecc_topology();
  check(evo::cpu::validate_geneb_mamba_topology(plain).ok(),
        "Caduceus topology is accepted");
  check(evo::cpu::validate_geneb_mamba_topology(rcps).ok(),
        "RCPS topology is accepted");
  check(evo::cpu::validate_geneb_mamba_topology(ecc).ok(),
        "ecc Mamba2 topology is accepted");
  auto mismatched_vocab = plain;
  mismatched_vocab.tokenizer_vocabulary_size = 7U;
  check(!evo::cpu::validate_geneb_mamba_topology(mismatched_vocab).ok(),
        "tokenizer vocabulary cannot exceed physical embedding rows");
  check(plain.maximum_sequence_length == 131072U,
        "long Caduceus context is not reduced");
  check(ecc.maximum_sequence_length == 0U,
        "unknown ecc context remains explicitly unbounded");
  auto advertised = ecc;
  advertised.advertised_training_sequence_length = 512U;
  check(evo::cpu::validate_geneb_mamba_topology(advertised).ok() &&
            advertised.maximum_sequence_length == 0U,
        "advertised training length does not become an inference limit");
  rcps.complement_map[3] = 3U;
  check(!evo::cpu::validate_geneb_mamba_topology(rcps).ok(),
        "non-involutive RCPS complement map is rejected");
  std::vector<evo::cpu::GenebMambaTensorRequirement> requirements;
  check(evo::cpu::canonical_geneb_mamba_tensors(plain, &requirements).ok() &&
            std::any_of(requirements.begin(), requirements.end(),
                        [](const auto &item) {
                          return item.name == "layers.0.reverse.A_log";
                        }) &&
            std::none_of(requirements.begin(), requirements.end(),
                         [](const auto &item) {
                           return item.name ==
                                  "layers.0.reverse.in_proj.weight";
                         }),
        "Caduceus manifest preserves tied reverse projections");
}

void test_load_corruption_and_forward() {
  evo::cpu::GenebMambaForwardResult division_probe;
  division_probe.rows = 3U;
  division_probe.width = 1U;
  division_probe.final_hidden = {0.1F, 0.1F, 0.1F};
  std::vector<float> division_probe_pool;
  auto probe_status = evo::cpu::geneb_mamba_pool(division_probe, {1U, 1U, 1U},
                                                 &division_probe_pool);
  const float probe_sum = (0.1F + 0.1F) + 0.1F;
  check(probe_status.ok() && division_probe_pool.size() == 1U &&
            division_probe_pool[0] == probe_sum / 3.0F &&
            division_probe_pool[0] != probe_sum * (1.0F / 3.0F),
        "Mamba pool fixture distinguishes direct division from reciprocal "
        "multiply");
  for (const auto &topology :
       {caduceus_topology(false), caduceus_topology(true), ecc_topology()}) {
    Fixture fixture;
    build_fixture(topology, &fixture);
    evo::cpu::GenebMambaModel model;
    auto status = model.load(fixture.topology, fixture.views);
    check(status.ok(), "tiny Mamba fixture loads");
    if (!status.ok())
      continue;
    evo::cpu::GenebMambaForwardResult result;
    status = model.forward({3U, 4U, 5U}, {1U, 1U, 0U}, {0U, topology.layers},
                           &result);
    check(
        status.ok() && result.rows == 3U &&
            result.width == topology.output_width &&
            result.final_hidden.size() == 3U * topology.output_width &&
            std::all_of(result.final_hidden.begin(), result.final_hidden.end(),
                        [](const float value) { return std::isfinite(value); }),
        "tiny Mamba forward has finite pinned shape");
    std::vector<float> pooled;
    status = model.pool(result, {1U, 1U, 0U}, &pooled);
    check(status.ok() && pooled.size() == topology.output_width,
          "Mamba attention-mask mean pooling succeeds");
    evo::cpu::GenebMambaForwardResult three_visible;
    status = model.forward({3U, 4U, 5U}, {1U, 1U, 1U}, {}, &three_visible);
    check(status.ok(), "three-visible-token Mamba fixture runs");
    status = model.pool(three_visible, {1U, 1U, 1U}, &pooled);
    for (std::size_t column = 0; status.ok() && column < pooled.size();
         ++column) {
      float sum = 0.0F;
      for (std::size_t row = 0; row < 3U; ++row)
        sum += three_visible.final_hidden[row * three_visible.width + column];
      const float expected = sum / 3.0F;
      check(pooled[column] == expected,
            "Mamba private pool uses exact direct F32 division");
    }
    check(status.ok(), "Mamba three-visible-token pooling succeeds");
    status = model.forward({3U}, {2U}, {topology.layers}, &result);
    check(!status.ok(), "invalid attention mask fails closed");

    auto corrupted = fixture.views;
    corrupted.front().tensor.shape.front() += 1U;
    evo::cpu::GenebMambaModel rejected;
    check(!rejected.load(fixture.topology, corrupted).ok(),
          "Mamba tensor shape corruption is rejected");
    auto missing = fixture.views;
    missing.pop_back();
    check(!rejected.load(fixture.topology, missing).ok(),
          "Mamba missing tensor is rejected");
  }
}

void test_long_context_contract() {
  auto long_topology = caduceus_topology(false);
  long_topology.maximum_sequence_length = 131072U;
  Fixture long_fixture;
  build_fixture(long_topology, &long_fixture);
  evo::cpu::GenebMambaModel long_model;
  auto status = long_model.load(long_fixture.topology, long_fixture.views);
  check(status.ok(), "long-context Caduceus fixture loads");
  std::vector<evo::TokenId> tokens(4097U, 3U);
  std::vector<std::uint8_t> mask(tokens.size(), 1U);
  evo::cpu::GenebMambaForwardResult result;
  status = long_model.forward(tokens, mask, {long_topology.layers}, &result);
  check(status.ok() && result.rows == tokens.size() &&
            result.final_hidden.size() == tokens.size() * long_topology.width,
        "Caduceus executes beyond 4096 without crop or hidden cap");

  auto plant_topology = caduceus_topology(false);
  plant_topology.maximum_sequence_length = 0U;
  plant_topology.advertised_training_sequence_length = 512U;
  Fixture plant_fixture;
  build_fixture(plant_topology, &plant_fixture);
  evo::cpu::GenebMambaModel plant_model;
  status = plant_model.load(plant_fixture.topology, plant_fixture.views);
  check(status.ok(), "advertised-only Plant context fixture loads");
  tokens.assign(513U, 3U);
  mask.assign(tokens.size(), 1U);
  status = plant_model.forward(tokens, mask, {plant_topology.layers}, &result);
  check(status.ok() && result.rows == 513U,
        "Plant advertised training length is not an inference gate");
}

void test_primitives() {
  const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<float> kernel{0.5F, 1.0F, -0.5F, 0.25F};
  const std::vector<float> bias{0.1F, -0.2F};
  const evo::cpu::MambaTensorView kernel_view{
      reinterpret_cast<const std::uint8_t *>(kernel.data()),
      kernel.size() * sizeof(float),
      evo::TensorDType::kF32,
      {2U, 1U, 2U}};
  const evo::cpu::MambaTensorView bias_view{
      reinterpret_cast<const std::uint8_t *>(bias.data()),
      bias.size() * sizeof(float),
      evo::TensorDType::kF32,
      {2U}};
  std::vector<float> convolved;
  const auto status = evo::cpu::mamba_causal_depthwise_conv_silu(
      input, 3U, 2U, 2U, kernel_view, &bias_view, &convolved);
  check(status.ok() && convolved.size() == input.size() &&
            std::all_of(convolved.begin(), convolved.end(),
                        [](const float value) { return std::isfinite(value); }),
        "causal convolution primitive is finite and shape-preserving");
  std::vector<float> inner_norm;
  check(run_inner_norm_primitive(1, &inner_norm).ok() &&
            inner_norm.size() == 6U &&
            std::all_of(inner_norm.begin(), inner_norm.end(),
                        [](const float value) { return std::isfinite(value); }),
        "Mamba1 projected dt/B/C RMSNorm path is finite");
  check(!run_inner_norm_primitive(2, &inner_norm).ok(),
        "Mamba1 projected RMSNorm rejects a missing B scale");
  check(!run_inner_norm_primitive(3, &inner_norm).ok(),
        "Mamba1 projected RMSNorm scales reject a disabled config flag");
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--dump-fixture")
    return dump_fixture_json();
  if (argc == 3 && std::string_view{argv[1]} == "--artifact")
    return verify_artifact(argv[2]);
  test_topology_and_manifest();
  test_load_corruption_and_forward();
  test_long_context_contract();
  test_primitives();
  if (failures != 0) {
    std::cerr << failures << " GENEB Mamba test(s) failed\n";
    return 1;
  }
  std::cout << "GENEB Mamba tests passed\n";
  return 0;
}
