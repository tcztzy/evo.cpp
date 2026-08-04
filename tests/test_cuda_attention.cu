// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "evo/cpu_reference.hpp"
#include "evo/cuda/attention.hpp"
#include "evo/cuda/runtime.hpp"

namespace {

int failures = 0;

void check(const bool condition, const std::string_view description) {
  if (!condition) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

void require(const evo::Status &status, const std::string_view operation) {
  if (!status.ok())
    throw std::runtime_error(std::string{operation} + ": " + status.message());
}

std::vector<float> values(const std::size_t count, const std::size_t offset,
                          const float divisor) {
  std::vector<float> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto raw = static_cast<int>(((index + offset) * 37 + 11) % 29) - 14;
    result.push_back(static_cast<float>(raw) / divisor);
  }
  return result;
}

std::vector<__nv_bfloat16> to_bf16(const std::vector<float> &input) {
  std::vector<__nv_bfloat16> output;
  output.reserve(input.size());
  for (const float value : input)
    output.push_back(__float2bfloat16(value));
  return output;
}

std::vector<float> to_float(const std::vector<__nv_bfloat16> &input) {
  std::vector<float> output;
  output.reserve(input.size());
  for (const auto value : input)
    output.push_back(__bfloat162float(value));
  return output;
}

std::vector<float> quantize_bf16(const std::vector<float> &input) {
  return to_float(to_bf16(input));
}

evo::cuda::DeviceBuffer upload(const int device, const void *const source,
                                 const std::size_t bytes,
                                 const evo::cuda::Stream &stream) {
  evo::cuda::DeviceBuffer buffer;
  require(buffer.allocate(device, bytes), "allocate upload buffer");
  require(buffer.copy_from_host(source, bytes, stream), "upload buffer");
  return buffer;
}

evo::cuda::DeviceBuffer upload_bf16(const int device,
                                      const std::vector<float> &input,
                                      const evo::cuda::Stream &stream) {
  const auto converted = to_bf16(input);
  return upload(device, converted.data(),
                converted.size() * sizeof(converted[0]), stream);
}

evo::cuda::DeviceBuffer upload_f32(const int device,
                                     const std::vector<float> &input,
                                     const evo::cuda::Stream &stream) {
  return upload(device, input.data(), input.size() * sizeof(input[0]), stream);
}

std::vector<float> download_bf16(const evo::cuda::DeviceBuffer &buffer,
                                 const std::size_t elements,
                                 const evo::cuda::Stream &stream) {
  std::vector<__nv_bfloat16> raw(elements);
  require(buffer.copy_to_host(raw.data(), raw.size() * sizeof(raw[0]), stream),
          "download BF16");
  require(stream.synchronize(), "synchronize BF16 download");
  return to_float(raw);
}

std::vector<__nv_bfloat16>
download_bf16_raw(const evo::cuda::DeviceBuffer &buffer,
                  const std::size_t elements,
                  const evo::cuda::Stream &stream) {
  std::vector<__nv_bfloat16> output(elements);
  require(buffer.copy_to_host(output.data(), output.size() * sizeof(output[0]),
                              stream),
          "download raw BF16");
  require(stream.synchronize(), "synchronize raw BF16 download");
  return output;
}

std::uint64_t fnv1a(const std::vector<__nv_bfloat16> &values) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  const auto *const bytes =
      reinterpret_cast<const unsigned char *>(values.data());
  for (std::size_t index = 0; index < values.size() * sizeof(values[0]);
       ++index) {
    hash ^= bytes[index];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

bool all_close(const std::vector<float> &actual,
               const std::vector<float> &expected, const float absolute,
               const float relative) {
  if (actual.size() != expected.size())
    return false;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float tolerance = absolute + relative * std::abs(expected[index]);
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance) {
      std::cerr << "mismatch at " << index << ": actual=" << actual[index]
                << " expected=" << expected[index] << " tolerance=" << tolerance
                << '\n';
      return false;
    }
  }
  return true;
}

void test_head_major_split(const int device,
                           const evo::cuda::Stream &stream) {
  constexpr std::size_t tokens = 2;
  constexpr std::size_t heads = 2;
  constexpr std::size_t head_dim = 3;
  constexpr std::size_t width = heads * head_dim;
  std::vector<float> head_major;
  std::vector<float> expected_query;
  std::vector<float> expected_key;
  std::vector<float> expected_value;
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t head = 0; head < heads; ++head) {
      for (std::size_t projection = 0; projection < 3; ++projection) {
        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
          const float value = static_cast<float>(token * 100 + projection * 20 +
                                                 head * head_dim + dimension);
          head_major.push_back(value);
          auto *const expected = projection == 0   ? &expected_query
                                 : projection == 1 ? &expected_key
                                                   : &expected_value;
          expected->push_back(value);
        }
      }
    }
  }
  auto input = upload_bf16(device, head_major, stream);
  evo::cuda::DeviceBuffer query;
  evo::cuda::DeviceBuffer key;
  evo::cuda::DeviceBuffer value;
  for (auto *const output : {&query, &key, &value}) {
    require(output->allocate(device, tokens * width * sizeof(__nv_bfloat16)),
            "allocate head-major split output");
  }
  require(evo::cuda::bf16_split_qkv(input, tokens, heads, head_dim, &query,
                                      &key, &value, stream,
                                      evo::cuda::QkvLayout::kHeadMajor),
          "split head-major QKV");
  check(download_bf16(query, tokens * width, stream) == expected_query,
        "head-major split extracts Q");
  check(download_bf16(key, tokens * width, stream) == expected_key,
        "head-major split extracts K");
  check(download_bf16(value, tokens * width, stream) == expected_value,
        "head-major split extracts V");
}

float cosine(const std::vector<float> &left, const std::vector<float> &right) {
  double dot = 0.0;
  double left_norm = 0.0;
  double right_norm = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    dot += static_cast<double>(left[index]) * right[index];
    left_norm += static_cast<double>(left[index]) * left[index];
    right_norm += static_cast<double>(right[index]) * right[index];
  }
  return static_cast<float>(dot / std::sqrt(left_norm * right_norm));
}

void split_qkv_host(const std::vector<float> &qkv, const std::size_t tokens,
                    const std::size_t width, std::vector<float> *const query,
                    std::vector<float> *const key,
                    std::vector<float> *const value) {
  query->resize(tokens * width);
  key->resize(tokens * width);
  value->resize(tokens * width);
  for (std::size_t token = 0; token < tokens; ++token) {
    const std::size_t source = token * width * 3;
    const std::size_t destination = token * width;
    std::copy_n(qkv.begin() + static_cast<std::ptrdiff_t>(source), width,
                query->begin() + static_cast<std::ptrdiff_t>(destination));
    std::copy_n(qkv.begin() + static_cast<std::ptrdiff_t>(source + width),
                width, key->begin() + static_cast<std::ptrdiff_t>(destination));
    std::copy_n(qkv.begin() + static_cast<std::ptrdiff_t>(source + width * 2),
                width,
                value->begin() + static_cast<std::ptrdiff_t>(destination));
  }
}

void rope_reference(std::vector<float> *const query,
                    std::vector<float> *const key, const std::size_t tokens,
                    const std::size_t heads, const std::size_t head_dim,
                    const std::vector<float> &inverse_frequency,
                    const std::size_t offset, const float position_scale) {
  require(evo::cpu::apply_rope(query, key, tokens, heads, head_dim,
                                 inverse_frequency, offset, position_scale),
          "CPU RoPE reference");
  *query = quantize_bf16(*query);
  *key = quantize_bf16(*key);
}

std::vector<float>
attention_last(const std::vector<float> &query, const std::vector<float> &key,
               const std::vector<float> &value, const std::size_t sources,
               const std::size_t heads, const std::size_t head_dim) {
  std::vector<float> output(heads * head_dim, 0.0F);
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  for (std::size_t head = 0; head < heads; ++head) {
    std::vector<float> scores(sources);
    float maximum = -INFINITY;
    for (std::size_t source = 0; source < sources; ++source) {
      float score = 0.0F;
      for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
        score += query[head * head_dim + dimension] *
                 key[(source * heads + head) * head_dim + dimension];
      }
      scores[source] = score * scale;
      maximum = std::max(maximum, scores[source]);
    }
    float denominator = 0.0F;
    for (float &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (std::size_t source = 0; source < sources; ++source) {
      const float probability = scores[source] / denominator;
      for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
        output[head * head_dim + dimension] +=
            probability * value[(source * heads + head) * head_dim + dimension];
      }
    }
  }
  return output;
}

void test_context(const int device, const evo::cuda::Stream &stream,
                  const std::size_t tokens) {
  constexpr std::size_t heads = 3;
  constexpr std::size_t head_dim = 6;
  constexpr float position_scale = 128.0F;
  constexpr std::size_t width = heads * head_dim;
  const auto qkv_f32 =
      quantize_bf16(values(tokens * width * 3, tokens + 7, 29.0F));
  const std::vector<float> inverse_frequency{1.0F, 0.1F, 0.01F};
  auto qkv = upload_bf16(device, qkv_f32, stream);
  auto frequency = upload_f32(device, inverse_frequency, stream);
  evo::cuda::KvCache cache;
  require(cache.allocate(device, tokens + 1, heads, head_dim),
          "allocate context KV cache");
  evo::cuda::AttentionWorkspace workspace;
  require(workspace.allocate(device, tokens, heads, head_dim),
          "allocate context attention workspace");
  evo::cuda::DeviceBuffer output;
  require(output.allocate(device, tokens * width * sizeof(__nv_bfloat16)),
          "allocate context attention output");
  require(evo::cuda::bf16_mha_prefill(qkv, frequency, tokens, heads, head_dim,
                                        position_scale, &cache, &workspace,
                                        &output, stream),
          "MHA prefill");

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  split_qkv_host(qkv_f32, tokens, width, &query, &key, &value);
  rope_reference(&query, &key, tokens, heads, head_dim, inverse_frequency, 0,
                 position_scale);
  std::vector<float> expected;
  require(evo::cpu::causal_attention(query, key, value, tokens, heads,
                                       head_dim, &expected),
          "CPU causal attention reference");
  const auto actual = download_bf16(output, tokens * width, stream);
  check(all_close(actual, expected, 0.02F, 0.02F),
        "online prefill matches full causal reference");
  check(cosine(actual, expected) >= 0.999F,
        "online prefill cosine is at least 0.999");
  check(cache.length == tokens, "prefill advances KV cache length");
  check(all_close(download_bf16(cache.key, tokens * width, stream), key, 0.004F,
                  0.0F),
        "KV cache stores RoPE-rotated BF16 keys");
  check(download_bf16(cache.value, tokens * width, stream) == value,
        "KV cache stores BF16 values bit exactly");

  const auto next_qkv_f32 =
      quantize_bf16(values(width * 3, tokens + 101, 31.0F));
  auto next_qkv = upload_bf16(device, next_qkv_f32, stream);
  evo::cuda::AttentionWorkspace decode_workspace;
  require(decode_workspace.allocate(device, 1, heads, head_dim),
          "allocate decode workspace");
  evo::cuda::DeviceBuffer decoded;
  require(decoded.allocate(device, width * sizeof(__nv_bfloat16)),
          "allocate decode output");
  require(evo::cuda::bf16_mha_decode(next_qkv, frequency, heads, head_dim,
                                       position_scale, &cache,
                                       &decode_workspace, &decoded, stream),
          "MHA cached decode");
  std::vector<float> next_query;
  std::vector<float> next_key;
  std::vector<float> next_value;
  split_qkv_host(next_qkv_f32, 1, width, &next_query, &next_key, &next_value);
  rope_reference(&next_query, &next_key, 1, heads, head_dim, inverse_frequency,
                 tokens, position_scale);
  key.insert(key.end(), next_key.begin(), next_key.end());
  value.insert(value.end(), next_value.begin(), next_value.end());
  const auto expected_decode =
      attention_last(next_query, key, value, tokens + 1, heads, head_dim);
  check(all_close(download_bf16(decoded, width, stream), expected_decode, 0.02F,
                  0.02F),
        "cached decode equals full causal last-token reference");
  check(cache.length == tokens + 1, "decode advances KV cache length");
}

void test_cached_cross_attention_exact(const int device,
                                       const evo::cuda::Stream &stream) {
  constexpr std::size_t query_tokens = 1;
  constexpr std::size_t heads = 2;
  constexpr std::size_t head_dim = 8;
  constexpr std::size_t width = heads * head_dim;
  struct Case final {
    std::size_t key_tokens;
    std::uint64_t probability_hash;
    std::uint64_t output_hash;
  };
  evo::cuda::Blas blas;
  require(blas.create(), "create cached-attention cuBLAS handle");
  // Oracles are raw BF16 bytes from Vortex CrossAttention on the pinned
  // PyTorch 2.13.0a0+8145d630e8.nv26.06 / CUDA 13.3 stack.  The cases cross
  // the persistent-warp, register, shared-memory, and general ILP kernels;
  // 10241 also gives the second head an unaligned row start.
  constexpr Case cases[]{
      {2048, 0x52e601854383b2cfULL, 0xff9c638caf1e02e2ULL},
      {2049, 0xe053c0d159dccce2ULL, 0x9320cc6a2915f573ULL},
      {10240, 0x2b87d3e6e1df2d52ULL, 0x86b60a698be200aaULL},
      {10241, 0xdb2b377d8f10d570ULL, 0x8793bdb099cc4911ULL},
      {30000, 0xc4b04a98e602a7a7ULL, 0xe81c04dde0a94bffULL},
  };
  const auto query_values = values(query_tokens * width, 17, 31.0F);
  auto query = upload_bf16(device, query_values, stream);
  for (const auto &test_case : cases) {
    auto key = upload_bf16(
        device, values(test_case.key_tokens * width, 43, 37.0F), stream);
    auto value = upload_bf16(
        device, values(test_case.key_tokens * width, 79, 41.0F), stream);
    evo::cuda::DeviceBuffer scaled_key;
    evo::cuda::DeviceBuffer scores;
    evo::cuda::DeviceBuffer probabilities;
    evo::cuda::DeviceBuffer output;
    require(
        output.allocate(device, query_tokens * width * sizeof(__nv_bfloat16)),
        "allocate exact cached-attention output");
    require(evo::cuda::bf16_cached_cross_attention(
                blas, query, key, value, query_tokens, test_case.key_tokens,
                heads, head_dim, &scaled_key, &scores, &probabilities, &output,
                stream),
            "exact cached cross-attention");
    const auto probability_values = download_bf16_raw(
        probabilities, heads * query_tokens * test_case.key_tokens, stream);
    const auto output_values =
        download_bf16_raw(output, query_tokens * width, stream);
    check(fnv1a(probability_values) == test_case.probability_hash,
          "cached softmax probabilities are bit exact to PyTorch");
    check(fnv1a(output_values) == test_case.output_hash,
          "cached cross-attention output is bit exact to PyTorch");
  }
}

void test_tiled_context_boundary(const int device,
                                 const evo::cuda::Stream &stream,
                                 const std::size_t tokens) {
  constexpr std::size_t heads = 2;
  constexpr std::size_t head_dim = 128;
  constexpr std::size_t width = heads * head_dim;
  constexpr float position_scale = 128.0F;
  const auto qkv_f32 =
      quantize_bf16(values(tokens * width * 3, tokens + 701, 83.0F));
  std::vector<float> inverse_frequency(head_dim / 2);
  for (std::size_t index = 0; index < inverse_frequency.size(); ++index) {
    inverse_frequency[index] =
        std::pow(1.0e6F, -2.0F * static_cast<float>(index) /
                             static_cast<float>(head_dim));
  }
  auto qkv = upload_bf16(device, qkv_f32, stream);
  auto frequency = upload_f32(device, inverse_frequency, stream);
  evo::cuda::KvCache cache;
  evo::cuda::AttentionWorkspace workspace;
  evo::cuda::DeviceBuffer output;
  require(cache.allocate(device, tokens, heads, head_dim),
          "allocate tiled-boundary KV cache");
  require(workspace.allocate(device, tokens, heads, head_dim),
          "allocate tiled-boundary attention workspace");
  require(output.allocate(device, tokens * width * sizeof(__nv_bfloat16)),
          "allocate tiled-boundary attention output");
  require(evo::cuda::bf16_mha_prefill(qkv, frequency, tokens, heads, head_dim,
                                        position_scale, &cache, &workspace,
                                        &output, stream),
          "tiled-boundary MHA prefill");

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  split_qkv_host(qkv_f32, tokens, width, &query, &key, &value);
  rope_reference(&query, &key, tokens, heads, head_dim, inverse_frequency, 0,
                 position_scale);
  std::vector<float> expected;
  require(evo::cpu::causal_attention(query, key, value, tokens, heads,
                                       head_dim, &expected),
          "CPU tiled-boundary attention reference");
  const auto actual = download_bf16(output, tokens * width, stream);
  const auto suffix = " at token boundary " + std::to_string(tokens);
  check(all_close(actual, expected, 0.03F, 0.03F),
        "tiled BF16 attention matches CPU reference" + suffix);
  check(cosine(actual, expected) >= 0.999F,
        "tiled BF16 attention cosine is at least 0.999" + suffix);
  check(cache.length == tokens,
        "tiled BF16 attention advances cache length" + suffix);
}

void test_chunked_prefill(const int device, const evo::cuda::Stream &stream) {
  constexpr std::size_t tokens = 7;
  constexpr std::size_t first = 3;
  constexpr std::size_t second = tokens - first;
  constexpr std::size_t heads = 2;
  constexpr std::size_t head_dim = 128;
  constexpr std::size_t width = heads * head_dim;
  constexpr float position_scale = 128.0F;
  const auto qkv_f32 = quantize_bf16(values(tokens * width * 3, 211, 37.0F));
  std::vector<float> inverse_frequency(head_dim / 2);
  for (std::size_t index = 0; index < inverse_frequency.size(); ++index) {
    inverse_frequency[index] =
        std::pow(1.0e6F, -2.0F * static_cast<float>(index) /
                             static_cast<float>(head_dim));
  }
  const std::vector<float> first_qkv(
      qkv_f32.begin(),
      qkv_f32.begin() + static_cast<std::ptrdiff_t>(first * width * 3));
  const std::vector<float> second_qkv(
      qkv_f32.begin() + static_cast<std::ptrdiff_t>(first * width * 3),
      qkv_f32.end());
  auto first_device = upload_bf16(device, first_qkv, stream);
  auto second_device = upload_bf16(device, second_qkv, stream);
  auto frequency = upload_f32(device, inverse_frequency, stream);
  evo::cuda::KvCache cache;
  require(cache.allocate(device, tokens, heads, head_dim),
          "allocate chunked KV cache");
  evo::cuda::AttentionWorkspace first_workspace;
  evo::cuda::AttentionWorkspace second_workspace;
  require(first_workspace.allocate(device, first, heads, head_dim),
          "allocate first chunk workspace");
  require(second_workspace.allocate(device, second, heads, head_dim),
          "allocate second chunk workspace");
  evo::cuda::DeviceBuffer first_output;
  evo::cuda::DeviceBuffer second_output;
  require(first_output.allocate(device, first * width * sizeof(__nv_bfloat16)),
          "allocate first chunk output");
  require(
      second_output.allocate(device, second * width * sizeof(__nv_bfloat16)),
      "allocate second chunk output");
  require(evo::cuda::bf16_mha_prefill(
              first_device, frequency, first, heads, head_dim, position_scale,
              &cache, &first_workspace, &first_output, stream),
          "first chunk prefill");
  require(evo::cuda::bf16_mha_prefill(
              second_device, frequency, second, heads, head_dim, position_scale,
              &cache, &second_workspace, &second_output, stream),
          "second chunk prefill");

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  split_qkv_host(qkv_f32, tokens, width, &query, &key, &value);
  rope_reference(&query, &key, tokens, heads, head_dim, inverse_frequency, 0,
                 position_scale);
  std::vector<float> expected;
  require(evo::cpu::causal_attention(query, key, value, tokens, heads,
                                       head_dim, &expected),
          "CPU chunked attention reference");
  const std::vector<float> expected_second(
      expected.begin() + static_cast<std::ptrdiff_t>(first * width),
      expected.end());
  check(all_close(download_bf16(second_output, second * width, stream),
                  expected_second, 0.02F, 0.02F),
        "chunked prefill respects cached causal prefix");
}

void test_rope_position_1m(const int device,
                           const evo::cuda::Stream &stream) {
  constexpr std::size_t tokens = 1;
  constexpr std::size_t heads = 2;
  constexpr std::size_t head_dim = 32;
  constexpr std::size_t width = heads * head_dim;
  constexpr std::size_t position = 1048575;
  constexpr float position_scale = 128.0F;
  auto query = quantize_bf16(values(width, 227, 41.0F));
  auto key = quantize_bf16(values(width, 233, 43.0F));
  std::vector<float> inverse_frequency(head_dim / 2);
  for (std::size_t index = 0; index < inverse_frequency.size(); ++index) {
    inverse_frequency[index] =
        std::pow(1.0e6F, -2.0F * static_cast<float>(index) /
                             static_cast<float>(head_dim));
  }
  auto query_device = upload_bf16(device, query, stream);
  auto key_device = upload_bf16(device, key, stream);
  auto frequency_device = upload_f32(device, inverse_frequency, stream);
  require(evo::cuda::bf16_apply_rope(
              &query_device, &key_device, frequency_device, tokens, heads,
              head_dim, position, position_scale, stream),
          "apply RoPE at final 1M context position");
  rope_reference(&query, &key, tokens, heads, head_dim, inverse_frequency,
                 position, position_scale);
  check(all_close(download_bf16(query_device, width, stream), query, 0.002F,
                  0.002F) &&
            all_close(download_bf16(key_device, width, stream), key, 0.002F,
                      0.002F),
        "RoPE position 1048575 matches the CPU reference");
}

void test_q8_paged_cache(const int device, const evo::cuda::Stream &stream) {
  constexpr std::size_t tokens = 6;
  constexpr std::size_t capacity = tokens + 1;
  constexpr std::size_t heads = 2;
  constexpr std::size_t head_dim = 32;
  constexpr std::size_t width = heads * head_dim;
  constexpr std::size_t page_tokens = 3;
  constexpr float position_scale = 128.0F;
  const auto qkv_f32 = quantize_bf16(values(tokens * width * 3, 271, 43.0F));
  std::vector<float> inverse_frequency(head_dim / 2);
  for (std::size_t index = 0; index < inverse_frequency.size(); ++index) {
    inverse_frequency[index] =
        std::pow(1.0e6F, -2.0F * static_cast<float>(index) /
                             static_cast<float>(head_dim));
  }
  auto qkv = upload_bf16(device, qkv_f32, stream);
  auto frequency = upload_f32(device, inverse_frequency, stream);
  evo::cuda::KvCache cache;
  require(cache.allocate_q8_paged(device, capacity, heads, head_dim,
                                  page_tokens, stream),
          "allocate paged Q8 KV cache");
  check(cache.quantized() && cache.device_id == device &&
            cache.q8_pages.size() == 3 &&
            std::none_of(cache.q8_pages.begin(), cache.q8_pages.end(),
                         [](const auto &page) { return page.key.valid(); }),
        "Q8 KV cache records three logical pages without eager payloads");
  const std::size_t empty_bytes = cache.allocated_bytes();
  const std::size_t bf16_bytes = capacity * width * sizeof(__nv_bfloat16) * 2;
  check(empty_bytes < bf16_bytes,
        "empty paged Q8 table is smaller than equivalent BF16 storage");

  evo::cuda::AttentionWorkspace workspace;
  evo::cuda::DeviceBuffer output;
  require(workspace.allocate(device, tokens, heads, head_dim),
          "allocate paged Q8 prefill workspace");
  require(output.allocate(device, tokens * width * sizeof(__nv_bfloat16)),
          "allocate paged Q8 prefill output");
  require(evo::cuda::bf16_mha_prefill(qkv, frequency, tokens, heads, head_dim,
                                        position_scale, &cache, &workspace,
                                        &output, stream),
          "paged Q8 MHA prefill");
  check(cache.q8_pages[0].key.valid() && cache.q8_pages[1].key.valid() &&
            !cache.q8_pages[2].key.valid() &&
            cache.allocated_bytes() > empty_bytes &&
            cache.allocated_bytes() < bf16_bytes,
        "Q8 prefill allocates only the two touched physical pages");

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  split_qkv_host(qkv_f32, tokens, width, &query, &key, &value);
  rope_reference(&query, &key, tokens, heads, head_dim, inverse_frequency, 0,
                 position_scale);
  std::vector<float> expected;
  require(evo::cpu::causal_attention(query, key, value, tokens, heads,
                                       head_dim, &expected),
          "CPU paged Q8 attention reference");
  const auto actual = download_bf16(output, tokens * width, stream);
  check(all_close(actual, expected, 0.04F, 0.04F),
        "paged Q8 prefill stays within its quantization tolerance");
  check(cosine(actual, expected) >= 0.999F,
        "paged Q8 prefill cosine is at least 0.999");

  const auto next_qkv_f32 = quantize_bf16(values(width * 3, 331, 47.0F));
  auto next_qkv = upload_bf16(device, next_qkv_f32, stream);
  evo::cuda::AttentionWorkspace decode_workspace;
  evo::cuda::DeviceBuffer decode_output;
  require(decode_workspace.allocate(device, 1, heads, head_dim),
          "allocate paged Q8 decode workspace");
  require(decode_output.allocate(device, width * sizeof(__nv_bfloat16)),
          "allocate paged Q8 decode output");
  require(evo::cuda::bf16_mha_decode(
              next_qkv, frequency, heads, head_dim, position_scale, &cache,
              &decode_workspace, &decode_output, stream),
          "paged Q8 MHA decode across a page boundary");
  check(cache.q8_pages[2].key.valid() && cache.allocated_bytes() < bf16_bytes,
        "paged Q8 decode allocates the newly touched third page");
  std::vector<float> next_query;
  std::vector<float> next_key;
  std::vector<float> next_value;
  split_qkv_host(next_qkv_f32, 1, width, &next_query, &next_key, &next_value);
  rope_reference(&next_query, &next_key, 1, heads, head_dim, inverse_frequency,
                 tokens, position_scale);
  key.insert(key.end(), next_key.begin(), next_key.end());
  value.insert(value.end(), next_value.begin(), next_value.end());
  const auto expected_decode =
      attention_last(next_query, key, value, capacity, heads, head_dim);
  const auto actual_decode = download_bf16(decode_output, width, stream);
  check(all_close(actual_decode, expected_decode, 0.04F, 0.04F),
        "paged Q8 decode stays within its quantization tolerance");
  check(cosine(actual_decode, expected_decode) >= 0.999F,
        "paged Q8 decode cosine is at least 0.999");
  check(cache.length == capacity,
        "paged Q8 decode fills the logical cache capacity");
  check(!evo::cuda::bf16_mha_decode(next_qkv, frequency, heads, head_dim,
                                      position_scale, &cache, &decode_workspace,
                                      &decode_output, stream)
             .ok(),
        "paged Q8 cache rejects decode beyond logical capacity");
}

void test_actual_head_shape(const int device,
                            const evo::cuda::Stream &stream) {
  constexpr std::size_t tokens = 2;
  constexpr std::size_t heads = 64;
  constexpr std::size_t head_dim = 128;
  constexpr std::size_t width = heads * head_dim;
  const auto qkv_f32 = quantize_bf16(values(tokens * width * 3, 307, 101.0F));
  std::vector<float> inverse_frequency(head_dim / 2);
  for (std::size_t index = 0; index < inverse_frequency.size(); ++index)
    inverse_frequency[index] =
        std::pow(1.0e11F, -2.0F * static_cast<float>(index) /
                              static_cast<float>(head_dim));
  auto qkv = upload_bf16(device, qkv_f32, stream);
  auto frequency = upload_f32(device, inverse_frequency, stream);
  evo::cuda::KvCache cache;
  evo::cuda::AttentionWorkspace workspace;
  evo::cuda::DeviceBuffer output;
  require(cache.allocate(device, tokens, heads, head_dim),
          "allocate actual-shape cache");
  require(workspace.allocate(device, tokens, heads, head_dim),
          "allocate actual-shape workspace");
  require(output.allocate(device, tokens * width * sizeof(__nv_bfloat16)),
          "allocate actual-shape output");
  require(evo::cuda::bf16_mha_prefill(qkv, frequency, tokens, heads, head_dim,
                                        128.0F, &cache, &workspace, &output,
                                        stream),
          "actual Evo2 head-shape prefill");
  const auto actual = download_bf16(output, tokens * width, stream);
  check(std::all_of(actual.begin(), actual.end(),
                    [](const float value) { return std::isfinite(value); }),
        "64x128 attention head shape produces finite output");
}

void test_context_8192(const int device, const evo::cuda::Stream &stream) {
  constexpr std::size_t prefix = 8192;
  constexpr std::size_t heads = 1;
  constexpr std::size_t head_dim = 8;
  constexpr std::size_t width = heads * head_dim;
  constexpr float position_scale = 128.0F;
  const auto prefix_key = quantize_bf16(values(prefix * width, 401, 47.0F));
  const auto prefix_value = quantize_bf16(values(prefix * width, 503, 53.0F));
  auto key_device = upload_bf16(device, prefix_key, stream);
  auto value_device = upload_bf16(device, prefix_value, stream);
  evo::cuda::KvCache cache;
  require(cache.allocate(device, prefix + 1, heads, head_dim),
          "allocate 8192-context cache");
  require(evo::cuda::bf16_kv_append(key_device, value_device, prefix, &cache,
                                      stream),
          "append 8192-token prefix");

  const auto qkv_f32 = quantize_bf16(values(width * 3, 607, 59.0F));
  const std::vector<float> inverse_frequency{1.0F, 0.1F, 0.01F, 0.001F};
  auto qkv = upload_bf16(device, qkv_f32, stream);
  auto frequency = upload_f32(device, inverse_frequency, stream);
  evo::cuda::AttentionWorkspace workspace;
  evo::cuda::DeviceBuffer output;
  require(workspace.allocate(device, 1, heads, head_dim),
          "allocate 8192-context decode workspace");
  require(output.allocate(device, width * sizeof(__nv_bfloat16)),
          "allocate 8192-context decode output");
  require(evo::cuda::bf16_mha_decode(qkv, frequency, heads, head_dim,
                                       position_scale, &cache, &workspace,
                                       &output, stream),
          "8192-context cached decode");

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  split_qkv_host(qkv_f32, 1, width, &query, &key, &value);
  rope_reference(&query, &key, 1, heads, head_dim, inverse_frequency, prefix,
                 position_scale);
  std::vector<float> all_key = prefix_key;
  std::vector<float> all_value = prefix_value;
  all_key.insert(all_key.end(), key.begin(), key.end());
  all_value.insert(all_value.end(), value.begin(), value.end());
  const auto expected =
      attention_last(query, all_key, all_value, prefix + 1, heads, head_dim);
  check(all_close(download_bf16(output, width, stream), expected, 0.02F, 0.02F),
        "8192-context cached decode matches full causal last token");
  check(cache.length == prefix + 1,
        "8192-context decode fills cache to capacity");
  const std::size_t full_length = cache.length;
  check(!evo::cuda::bf16_mha_decode(qkv, frequency, heads, head_dim,
                                      position_scale, &cache, &workspace,
                                      &output, stream)
             .ok(),
        "KV cache rejects decode beyond capacity");
  check(cache.length == full_length,
        "failed KV append leaves cache length unchanged");
}

int requested_device() {
  const char *environment = std::getenv("EVO_TEST_DEVICE");
  return environment == nullptr ? 0 : std::stoi(environment);
}

} // namespace

int main() {
  try {
    const int device = requested_device();
    require(evo::cuda::select_device(device), "select CUDA device");
    cudaDeviceProp properties{};
    require(
        evo::cuda::cuda_status(cudaGetDeviceProperties(&properties, device),
                                 "cudaGetDeviceProperties"),
        "query CUDA device");
    std::cout << "GPU=" << properties.name << " sm=" << properties.major
              << properties.minor << '\n';
    evo::cuda::Stream stream;
    require(stream.create(), "create CUDA stream");
    test_head_major_split(device, stream);
    for (const std::size_t tokens : {1U, 2U, 7U, 128U})
      test_context(device, stream, tokens);
    test_cached_cross_attention_exact(device, stream);
    for (const std::size_t tokens : {3U, 4U, 7U, 8U, 9U, 127U, 128U, 129U})
      test_tiled_context_boundary(device, stream, tokens);
    test_chunked_prefill(device, stream);
    test_rope_position_1m(device, stream);
    test_q8_paged_cache(device, stream);
    test_actual_head_shape(device, stream);
    test_context_8192(device, stream);
    require(stream.synchronize(), "final stream synchronize");
    require(evo::cuda::synchronize_device(), "final device synchronize");
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " CUDA attention test(s) failed\n";
    return 1;
  }
  std::cout << "CUDA attention tests passed\n";
  return 0;
}
