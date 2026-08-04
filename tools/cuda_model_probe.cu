// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "evo/cuda/model.hpp"
#include "evo/model_format.hpp"
#include "evo/status.hpp"
#include "evo/tokenizer.hpp"

namespace {

void require(const evo::Status &status, const std::string &operation) {
  if (!status.ok()) {
    throw std::runtime_error(operation + ": " +
                             evo::error_code_name(status.code()) + ": " +
                             status.message());
  }
}

void write_npy(const std::filesystem::path &path,
               const std::vector<float> &values, const std::size_t rows,
               const std::size_t columns) {
  if (rows == 0 || columns == 0 ||
      rows > std::numeric_limits<std::size_t>::max() / columns ||
      rows * columns != values.size()) {
    throw std::runtime_error("invalid NPY dimensions");
  }
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       std::to_string(rows) + ", " + std::to_string(columns) +
                       "), }";
  constexpr std::size_t prefix = 10;
  const std::size_t padding = (64 - ((prefix + header.size() + 1) % 64)) % 64;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<std::uint16_t>::max())
    throw std::runtime_error("NPY header is too large");

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    throw std::runtime_error("cannot open NPY output: " + path.string());
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
    throw std::runtime_error("failed to write NPY output: " + path.string());
}

std::string layer_filename(const std::size_t layer) {
  std::ostringstream name;
  name << "layer_" << std::setfill('0') << std::setw(2) << layer << ".npy";
  return name.str();
}

std::vector<int> parse_devices(const std::string &value) {
  std::vector<int> devices;
  std::size_t begin = 0;
  while (begin < value.size()) {
    const std::size_t comma = value.find(',', begin);
    const std::string component = value.substr(begin, comma - begin);
    std::size_t consumed = 0;
    const auto parsed = std::stoull(component, &consumed);
    if (component.empty() || consumed != component.size() ||
        parsed >
            static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("GPU_LIST must contain numeric device IDs");
    }
    const int device = static_cast<int>(parsed);
    if (std::find(devices.begin(), devices.end(), device) != devices.end())
      throw std::runtime_error("GPU_LIST must not contain duplicate devices");
    devices.push_back(device);
    if (comma == std::string::npos)
      break;
    begin = comma + 1;
  }
  if (devices.empty() || devices.size() > 4)
    throw std::runtime_error("GPU_LIST requires between one and four devices");
  return devices;
}

} // namespace

int main(const int argc, char **argv) {
  if (argc < 4 || argc > 7) {
    std::cerr << "usage: evo-cuda-probe MODEL PROMPT OUTPUT_DIR "
                 "[DEBUG_LAYER|- [GPU_LIST [GENERATE_TOKENS]]]\n";
    return 2;
  }
  try {
    std::optional<std::size_t> debug_layer;
    if (argc >= 5 && std::string{argv[4]} != "-") {
      std::size_t consumed = 0;
      const auto parsed = std::stoull(argv[4], &consumed);
      if (consumed != std::string{argv[4]}.size() ||
          parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("DEBUG_LAYER must be a nonnegative integer");
      }
      debug_layer = static_cast<std::size_t>(parsed);
    }
    const std::vector<int> devices =
        argc >= 6 ? parse_devices(argv[5]) : std::vector<int>{0, 1, 2, 3};
    std::size_t generate_tokens = 0;
    if (argc == 7) {
      std::size_t consumed = 0;
      const auto parsed = std::stoull(argv[6], &consumed);
      if (consumed != std::string{argv[6]}.size() || parsed == 0 ||
          parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("GENERATE_TOKENS must be a positive integer");
      }
      generate_tokens = static_cast<std::size_t>(parsed);
    }
    const std::string prompt{argv[2]};
    const auto tokens = evo::encode_bytes(prompt);
    if (tokens.empty())
      throw std::runtime_error("prompt must not be empty");
    const std::filesystem::path output_dir{argv[3]};
    std::filesystem::create_directories(output_dir);

    evo::ModelFile file;
    require(file.open(argv[1]), "open model");
    evo::cuda::PipelineModel model;
    if (generate_tokens >
        std::numeric_limits<std::size_t>::max() - tokens.size()) {
      throw std::runtime_error("generation context capacity overflows");
    }
    require(model.load(file, devices, tokens.size() + generate_tokens),
            "load pipeline");
    if (debug_layer.has_value() && *debug_layer >= model.config().layers) {
      throw std::runtime_error("DEBUG_LAYER exceeds the model layer count");
    }

    std::vector<evo::cuda::LayerDump> dumps;
    dumps.reserve(model.config().layers + (debug_layer.has_value() ? 10 : 0));
    for (std::size_t layer = 0; layer < model.config().layers; ++layer) {
      dumps.push_back({layer, (output_dir / layer_filename(layer)).string()});
    }
    if (debug_layer.has_value()) {
      const std::array<std::pair<evo::cuda::LayerDumpPoint, const char *>, 19>
          points{{
              {evo::cuda::LayerDumpPoint::kPreNorm, "pre_norm"},
              {evo::cuda::LayerDumpPoint::kMixerInputProjection,
               "mixer_input_projection"},
              {evo::cuda::LayerDumpPoint::kMixerShortFilter,
               "mixer_short_filter"},
              {evo::cuda::LayerDumpPoint::kMixerX2, "mixer_x2"},
              {evo::cuda::LayerDumpPoint::kMixerX1, "mixer_x1"},
              {evo::cuda::LayerDumpPoint::kMixerValue, "mixer_value"},
              {evo::cuda::LayerDumpPoint::kMixerPregate, "mixer_pregate"},
              {evo::cuda::LayerDumpPoint::kMixerState, "mixer_state"},
              {evo::cuda::LayerDumpPoint::kMixerFilter, "mixer_filter"},
              {evo::cuda::LayerDumpPoint::kMixerConvolution,
               "mixer_convolution"},
              {evo::cuda::LayerDumpPoint::kMixerOutput, "mixer_output"},
              {evo::cuda::LayerDumpPoint::kMixerProjection,
               "mixer_projection"},
              {evo::cuda::LayerDumpPoint::kMixerResidual, "mixer_residual"},
              {evo::cuda::LayerDumpPoint::kPostNorm, "post_norm"},
              {evo::cuda::LayerDumpPoint::kMlpL1, "mlp_l1"},
              {evo::cuda::LayerDumpPoint::kMlpL2, "mlp_l2"},
              {evo::cuda::LayerDumpPoint::kMlpActivation, "mlp_activation"},
              {evo::cuda::LayerDumpPoint::kMlpGated, "mlp_gated"},
              {evo::cuda::LayerDumpPoint::kMlpOutput, "mlp_output"},
          }};
      for (const auto &[point, suffix] : points) {
        const std::string filename =
            "layer_" + std::to_string(*debug_layer) + "_" + suffix + ".npy";
        dumps.push_back(
            {*debug_layer, (output_dir / filename).string(), point});
      }
    }
    std::vector<float> logits;
    require(generate_tokens == 0
                ? model.prefill_with_dumps(tokens, &logits, dumps)
                : model.prefill_cached_with_dumps(tokens, &logits, dumps),
            generate_tokens == 0 ? "prefill and dump layers"
                                 : "cached prefill and dump layers");
    if (generate_tokens == 0) {
      write_npy(output_dir / "logits.npy", logits, tokens.size(),
                model.config().vocab_size);
    } else {
      write_npy(output_dir / "prefill_logits.npy", logits, tokens.size(),
                model.config().vocab_size);
      std::vector<float> generation_logits;
      generation_logits.reserve(generate_tokens * model.config().vocab_size);
      std::vector<evo::TokenId> generated;
      generated.reserve(generate_tokens);
      std::vector<float> current(
          logits.end() - static_cast<std::ptrdiff_t>(model.config().vocab_size),
          logits.end());
      for (std::size_t step = 0; step < generate_tokens; ++step) {
        generation_logits.insert(generation_logits.end(), current.begin(),
                                 current.end());
        const auto maximum = std::max_element(current.begin(), current.end());
        const auto token =
            static_cast<evo::TokenId>(maximum - current.begin());
        if (static_cast<std::size_t>(token) > 255)
          throw std::runtime_error(
              "generated token has no byte representation");
        generated.push_back(token);
        if (step + 1 < generate_tokens)
          require(model.decode_with_dumps(token, &current, dumps),
                  "cached decode and dump layers");
      }
      write_npy(output_dir / "logits.npy", generation_logits, generate_tokens,
                model.config().vocab_size);
      std::ofstream generated_output(output_dir / "generated.bin",
                                     std::ios::binary | std::ios::trunc);
      if (!generated_output)
        throw std::runtime_error("cannot open generated-token output");
      for (const evo::TokenId token : generated)
        generated_output.put(static_cast<char>(token));
      if (!generated_output)
        throw std::runtime_error("failed to write generated-token output");
    }

    std::ofstream token_output(output_dir / "tokens.txt", std::ios::trunc);
    if (!token_output)
      throw std::runtime_error("cannot open token output");
    for (std::size_t index = 0; index < tokens.size(); ++index) {
      if (index != 0)
        token_output << ',';
      token_output << tokens[index];
    }
    token_output << '\n';
    if (!token_output)
      throw std::runtime_error("failed to write token output");

    for (const auto &stage : model.stages()) {
      std::cout << "stage device=" << stage.device << " layers=["
                << stage.layer_begin << ',' << stage.layer_end
                << ") weights=" << stage.weight_bytes
                << " cache=" << stage.cache_bytes
                << " arena=" << stage.arena_bytes << '\n';
    }
    std::vector<std::size_t> order(model.config().vocab_size);
    std::iota(order.begin(), order.end(), 0);
    const std::size_t last_offset =
        (tokens.size() - 1) * model.config().vocab_size;
    std::partial_sort(
        order.begin(), order.begin() + std::min<std::size_t>(10, order.size()),
        order.end(), [&](const std::size_t left, const std::size_t right) {
          return logits[last_offset + left] > logits[last_offset + right];
        });
    std::cout << "last_top10";
    for (std::size_t rank = 0; rank < std::min<std::size_t>(10, order.size());
         ++rank) {
      const std::size_t token = order[rank];
      std::cout << ' ' << token << ':' << logits[last_offset + token];
    }
    std::cout << '\n';
  } catch (const std::exception &error) {
    std::cerr << "evo-cuda-probe: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
