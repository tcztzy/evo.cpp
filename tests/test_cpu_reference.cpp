// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "evo/cpu_reference.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

bool close(const float left, const float right, const float tolerance = 1.0e-5F) {
  return std::abs(left - right) <= tolerance;
}

std::vector<float> values(const std::size_t count,
                          const std::size_t offset,
                          const float divisor) {
  std::vector<float> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto raw = static_cast<int>(((index + offset) * 37 + 11) % 29) - 14;
    result.push_back(static_cast<float>(raw) / divisor);
  }
  return result;
}

void require(const evo::Status& status, const std::string_view operation) {
  if (!status.ok()) {
    std::cerr << "fixture " << operation << " failed: " << status.message() << '\n';
    std::abort();
  }
}

using NamedVector = std::pair<std::string, std::vector<float>>;

std::vector<NamedVector> build_fixture() {
  std::vector<NamedVector> vectors;

  const auto rms_input = values(8, 0, 7.0F);
  auto rms_scale_source = values(4, 31, 19.0F);
  std::vector<float> rms_scale;
  rms_scale.reserve(rms_scale_source.size());
  for (const float item : rms_scale_source) {
    rms_scale.push_back(1.0F + item * 0.1F);
  }
  std::vector<float> result;
  require(evo::cpu::rms_norm(rms_input, 2, 4, rms_scale, 1.0e-6F, &result), "rmsnorm");
  vectors.emplace_back("rmsnorm", result);

  const auto linear_input = values(8, 7, 9.0F);
  const auto linear_weight = values(12, 19, 13.0F);
  const auto linear_bias = values(3, 43, 17.0F);
  require(evo::cpu::linear(linear_input, 2, 4, linear_weight, 3, &linear_bias, &result),
          "linear");
  vectors.emplace_back("linear", result);

  const auto mlp_input = values(6, 3, 11.0F);
  const auto l1 = values(15, 17, 23.0F);
  const auto l2 = values(15, 41, 29.0F);
  const auto l3 = values(15, 67, 31.0F);
  require(evo::cpu::gated_mlp(mlp_input, 2, 3, 5, l1, l2, l3,
                                evo::cpu::MlpActivation::kGelu, &result),
          "mlp_gelu");
  vectors.emplace_back("mlp_gelu", result);
  require(evo::cpu::gated_mlp(mlp_input, 2, 3, 5, l1, l2, l3,
                                evo::cpu::MlpActivation::kIdentity, &result),
          "mlp_identity");
  vectors.emplace_back("mlp_identity", result);

  const auto fir_input = values(15, 5, 13.0F);
  const auto fir_weight = values(9, 29, 17.0F);
  const auto fir_bias = values(3, 53, 19.0F);
  require(evo::cpu::causal_depthwise_fir(fir_input, 5, 3, fir_weight, 3, &fir_bias, &result),
          "fir");
  vectors.emplace_back("fir", result);
  require(evo::cpu::causal_depthwise_fir(
              fir_input, 5, 3, fir_weight, 3, &fir_bias, &result,
              evo::cpu::FirOrientation::kCausalConvolution,
              evo::cpu::FirBiasMode::kMultiplyInput),
          "fir_causal_gated");
  vectors.emplace_back("fir_causal_gated", result);

  const auto projection = values(24, 11, 15.0F);
  std::vector<float> x2;
  std::vector<float> x1;
  std::vector<float> value;
  require(evo::cpu::split_hyena_projection(projection, 2, 4, &x2, &x1, &value),
          "hyena_split");
  vectors.emplace_back("hyena_x2", x2);
  vectors.emplace_back("hyena_x1", x1);
  vectors.emplace_back("hyena_value", value);

  const auto hcl_x2 = values(12, 2, 9.0F);
  const auto hcl_x1 = values(12, 37, 11.0F);
  const auto hcl_value = values(12, 71, 13.0F);
  const auto direct = values(3, 13, 17.0F);
  std::vector<float> log_poles;
  for (std::size_t index = 0; index < 6; ++index) {
    log_poles.push_back(-0.03F * static_cast<float>(index + 1));
  }
  const auto residues = values(6, 23, 19.0F);
  auto state = values(6, 47, 31.0F);
  require(evo::cpu::hcl_recurrence(hcl_x2, hcl_x1, hcl_value, 4, 3, direct, log_poles,
                                     residues, 2, &state, &result),
          "hcl");
  vectors.emplace_back("hcl", result);
  vectors.emplace_back("hcl_state", state);

  auto query = values(24, 59, 13.0F);
  auto key = values(24, 83, 17.0F);
  const auto attention_value = values(24, 107, 19.0F);
  const std::vector<float> inverse_frequency{1.0F, 0.01F};
  require(evo::cpu::apply_rope(&query, &key, 3, 2, 4, inverse_frequency, 5, 128.0F),
          "rope");
  vectors.emplace_back("rope_query", query);
  vectors.emplace_back("rope_key", key);
  require(evo::cpu::causal_attention(query, key, attention_value, 3, 2, 4, &result),
          "attention");
  vectors.emplace_back("attention", result);
  return vectors;
}

void dump_json() {
  const auto vectors = build_fixture();
  std::cout << "{\"vectors\":{";
  for (std::size_t vector_index = 0; vector_index < vectors.size(); ++vector_index) {
    if (vector_index != 0) std::cout << ',';
    std::cout << '\"' << vectors[vector_index].first << "\":[" << std::setprecision(9);
    const auto& data = vectors[vector_index].second;
    for (std::size_t index = 0; index < data.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << data[index];
    }
    std::cout << ']';
  }
  std::cout << "}}\n";
}

void test_rms_norm() {
  const std::vector<float> input{3.0F, 4.0F};
  const std::vector<float> scale{1.0F, 2.0F};
  std::vector<float> output;
  auto status = evo::cpu::rms_norm(input, 1, 2, scale, 0.25F, &output);
  const float denominator = std::sqrt(12.5F) + 0.25F;
  check(status.ok() && output.size() == 2, "RMSNorm accepts a valid row");
  check(close(output[0], 3.0F / denominator) && close(output[1], 8.0F / denominator),
        "RMSNorm adds epsilon after square root");
  check(!evo::cpu::rms_norm(input, 1, 3, scale, 1.0e-6F, &output).ok(),
        "RMSNorm rejects inconsistent shapes");
  const std::vector<float> non_finite{std::numeric_limits<float>::quiet_NaN(), 1.0F};
  check(!evo::cpu::rms_norm(non_finite, 1, 2, scale, 1.0e-6F, &output).ok(),
        "RMSNorm rejects non-finite inputs");
}

void test_linear_and_mlp() {
  const std::vector<float> input{2.0F, -1.0F};
  const std::vector<float> weight{1.0F, 3.0F, -2.0F, 4.0F};
  const std::vector<float> bias{0.5F, -0.5F};
  std::vector<float> output;
  auto status = evo::cpu::linear(input, 1, 2, weight, 2, &bias, &output);
  check(status.ok() && close(output[0], -0.5F) && close(output[1], -8.5F),
        "linear uses PyTorch [out,in] weights and bias");

  const std::vector<float> one{1.0F};
  status = evo::cpu::gated_mlp(one, 1, 1, 1, one, one, one,
                                 evo::cpu::MlpActivation::kIdentity, &output);
  check(status.ok() && close(output[0], 1.0F), "identity gated MLP multiplies l1 and l2");
  status = evo::cpu::gated_mlp(one, 1, 1, 1, one, one, one,
                                 evo::cpu::MlpActivation::kGelu, &output);
  const float expected_gelu = 0.5F * (1.0F + std::erf(1.0F / std::sqrt(2.0F)));
  check(status.ok() && close(output[0], expected_gelu), "layer-zero MLP uses exact GELU");
}

void test_fir_and_split() {
  const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> weight{1.0F, 2.0F, 3.0F};
  std::vector<float> output;
  auto status = evo::cpu::causal_depthwise_fir(input, 4, 1, weight, 3, nullptr, &output);
  check(status.ok() && output == std::vector<float>({3.0F, 8.0F, 14.0F, 20.0F}),
        "FIR matches conv1d cross-correlation with causal left padding");
  const std::vector<float> direct{0.5F};
  status = evo::cpu::causal_depthwise_fir(
      input, 4, 1, weight, 3, &direct, &output,
      evo::cpu::FirOrientation::kCausalConvolution,
      evo::cpu::FirBiasMode::kMultiplyInput);
  check(status.ok() && output == std::vector<float>({1.5F, 5.0F, 11.5F, 18.0F}),
        "HCM FIR uses causal filter order and D times the gated input");

  const std::vector<float> projection{20.0F, 10.0F, 30.0F, 21.0F, 11.0F, 31.0F};
  std::vector<float> x2;
  std::vector<float> x1;
  std::vector<float> value;
  status = evo::cpu::split_hyena_projection(projection, 1, 2, &x2, &x1, &value);
  check(status.ok() && x2 == std::vector<float>({20.0F, 21.0F}) &&
            x1 == std::vector<float>({10.0F, 11.0F}) &&
            value == std::vector<float>({30.0F, 31.0F}),
        "Hyena triples split in x2,x1,v order");
}

void test_hcl() {
  const std::vector<float> x2{2.0F, 3.0F};
  const std::vector<float> x1{1.0F, 2.0F};
  const std::vector<float> value{4.0F, 5.0F};
  const std::vector<float> direct{0.5F};
  const std::vector<float> log_poles{std::log(0.5F)};
  const std::vector<float> residues{2.0F};
  std::vector<float> state{1.0F};
  std::vector<float> output;
  const auto status = evo::cpu::hcl_recurrence(x2, x1, value, 2, 1, direct, log_poles,
                                                  residues, 1, &state, &output);
  check(status.ok() && close(output[0], 22.0F) && close(output[1], 88.5F),
        "HCL applies modal update, residue, direct term, and outer gate");
  check(close(state[0], 12.25F), "HCL returns final F32 recurrence state");
}

void test_rope_and_attention() {
  std::vector<float> query{1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> key = query;
  const std::vector<float> frequencies{1.0F, 0.5F};
  auto status = evo::cpu::apply_rope(&query, &key, 1, 1, 4, frequencies, 0, 128.0F);
  check(status.ok() && query == std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F}),
        "RoPE position zero is identity");

  const std::vector<float> q{1.0F, 0.0F, 1.0F, 0.0F};
  const std::vector<float> k{1.0F, 0.0F, 0.0F, 1.0F};
  const std::vector<float> v{2.0F, 3.0F, 5.0F, 7.0F};
  status = evo::cpu::causal_attention(q, k, v, 2, 1, 2, &query);
  const float first_weight = std::exp(1.0F / std::sqrt(2.0F));
  const float denominator = first_weight + 1.0F;
  check(status.ok() && close(query[0], 2.0F) && close(query[1], 3.0F),
        "causal attention first token only sees itself");
  check(close(query[2], (first_weight * 2.0F + 5.0F) / denominator) &&
            close(query[3], (first_weight * 3.0F + 7.0F) / denominator),
        "causal attention uses stable scaled softmax over the prefix");
}

}  // namespace

int main(const int argc, char** argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--dump-json") {
    dump_json();
    return 0;
  }
  if (argc != 1) {
    std::cerr << "usage: evo-cpu-reference-tests [--dump-json]\n";
    return 2;
  }
  test_rms_norm();
  test_linear_and_mlp();
  test_fir_and_split();
  test_hcl();
  test_rope_and_attention();
  if (failures != 0) {
    std::cerr << failures << " CPU reference test(s) failed\n";
    return 1;
  }
  std::cout << "CPU reference tests passed\n";
  return 0;
}
