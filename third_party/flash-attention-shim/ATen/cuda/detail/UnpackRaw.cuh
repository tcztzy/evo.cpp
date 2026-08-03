// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <tuple>

namespace at {

struct PhiloxCudaState final {};

namespace cuda::philox {

__host__ __device__ __forceinline__ std::tuple<std::uint64_t, std::uint64_t>
unpack(const PhiloxCudaState) {
  return std::make_tuple(std::uint64_t{0}, std::uint64_t{0});
}

} // namespace cuda::philox
} // namespace at
