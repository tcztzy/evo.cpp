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

#include "evo2c/cuda/model.hpp"
#include "evo2c/model_format.hpp"
#include "evo2c/status.hpp"
#include "evo2c/tokenizer.hpp"

namespace {

void require(const evo2c::Status &status, const std::string &operation) {
  if (!status.ok()) {
    throw std::runtime_error(operation + ": " +
                             evo2c::error_code_name(status.code()) + ": " +
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

} // namespace

int main(const int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    std::cerr
        << "usage: evo2c-cuda-probe MODEL PROMPT OUTPUT_DIR [DEBUG_LAYER]\n";
    return 2;
  }
  try {
    std::optional<std::size_t> debug_layer;
    if (argc == 5) {
      std::size_t consumed = 0;
      const auto parsed = std::stoull(argv[4], &consumed);
      if (consumed != std::string{argv[4]}.size() ||
          parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("DEBUG_LAYER must be a nonnegative integer");
      }
      debug_layer = static_cast<std::size_t>(parsed);
    }
    const std::string prompt{argv[2]};
    const auto tokens = evo2c::encode_bytes(prompt);
    if (tokens.empty())
      throw std::runtime_error("prompt must not be empty");
    const std::filesystem::path output_dir{argv[3]};
    std::filesystem::create_directories(output_dir);

    evo2c::ModelFile file;
    require(file.open(argv[1]), "open model");
    evo2c::cuda::PipelineModel model;
    require(model.load(file, {0, 1, 2, 3}, tokens.size()), "load pipeline");
    if (debug_layer.has_value() && *debug_layer >= model.config().layers) {
      throw std::runtime_error("DEBUG_LAYER exceeds the model layer count");
    }

    std::vector<evo2c::cuda::LayerDump> dumps;
    dumps.reserve(model.config().layers + (debug_layer.has_value() ? 5 : 0));
    for (std::size_t layer = 0; layer < model.config().layers; ++layer) {
      dumps.push_back({layer, (output_dir / layer_filename(layer)).string()});
    }
    if (debug_layer.has_value()) {
      const std::array<std::pair<evo2c::cuda::LayerDumpPoint, const char *>, 5>
          points{{
              {evo2c::cuda::LayerDumpPoint::kPreNorm, "pre_norm"},
              {evo2c::cuda::LayerDumpPoint::kMixerOutput, "mixer_output"},
              {evo2c::cuda::LayerDumpPoint::kMixerResidual, "mixer_residual"},
              {evo2c::cuda::LayerDumpPoint::kPostNorm, "post_norm"},
              {evo2c::cuda::LayerDumpPoint::kMlpOutput, "mlp_output"},
          }};
      for (const auto &[point, suffix] : points) {
        const std::string filename =
            "layer_" + std::to_string(*debug_layer) + "_" + suffix + ".npy";
        dumps.push_back(
            {*debug_layer, (output_dir / filename).string(), point});
      }
    }
    std::vector<float> logits;
    require(model.prefill_with_dumps(tokens, &logits, dumps),
            "prefill and dump layers");
    write_npy(output_dir / "logits.npy", logits, tokens.size(),
              model.config().vocab_size);

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
    std::cerr << "evo2c-cuda-probe: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
