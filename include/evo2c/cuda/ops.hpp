// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include "evo2c/cuda/runtime.hpp"
#include "evo2c/status.hpp"

namespace evo2c::cuda {

inline constexpr std::size_t kDefaultBlasWorkspaceBytes = 32U * 1024U * 1024U;

enum class GatedActivation { kGelu, kIdentity };

struct MlpWorkspace final {
  DeviceBuffer first;
  DeviceBuffer second;
  DeviceBuffer gated;
  DeviceBuffer blas;

  [[nodiscard]] Status
  allocate(int device, std::size_t rows, std::size_t inner_width,
           std::size_t blas_bytes = kDefaultBlasWorkspaceBytes);
};

// BF16 row-major input[rows,in] times weight[out,in]^T, F32 accumulation, BF16
// output.
[[nodiscard]] Status bf16_linear(const BlasLt &handle,
                                 const DeviceBuffer &input,
                                 const DeviceBuffer &weight,
                                 const DeviceBuffer *bias, std::size_t rows,
                                 std::size_t in_features,
                                 std::size_t out_features, DeviceBuffer *output,
                                 DeviceBuffer *workspace, const Stream &stream);

// Input/output BF16 [rows,width], scale F32 [width].
[[nodiscard]] Status bf16_rms_norm(const DeviceBuffer &input,
                                   const DeviceBuffer &scale, std::size_t rows,
                                   std::size_t width, float epsilon,
                                   DeviceBuffer *output, const Stream &stream);

[[nodiscard]] Status
bf16_gated_elementwise(const DeviceBuffer &first, const DeviceBuffer &second,
                       std::size_t elements, GatedActivation activation,
                       DeviceBuffer *output, const Stream &stream);

[[nodiscard]] Status bf16_add_inplace(DeviceBuffer *output,
                                      const DeviceBuffer &residual,
                                      std::size_t elements,
                                      const Stream &stream);

[[nodiscard]] Status
bf16_mlp(const BlasLt &handle, const DeviceBuffer &input,
         const DeviceBuffer &l1_weight, const DeviceBuffer &l2_weight,
         const DeviceBuffer &l3_weight, std::size_t rows, std::size_t width,
         std::size_t inner_width, GatedActivation activation,
         MlpWorkspace *workspace, DeviceBuffer *output, const Stream &stream);

} // namespace evo2c::cuda
