// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include <cufft.h>

#include "evo2c/cuda/runtime.hpp"
#include "evo2c/status.hpp"

namespace evo2c::cuda {

enum class FirOrientation { kCrossCorrelation, kCausalConvolution };
enum class FirBiasMode { kAdd, kMultiplyInput };
enum class HclPrefillMode { kRecurrence, kFft };

// F32 chronological state [channels,kernel_size-1]. Short sequences are
// left-zero-padded, so the newest value is always in the final column.
struct FirCache final {
  DeviceBuffer state;
  std::size_t channels{0};
  std::size_t kernel_size{0};

  [[nodiscard]] Status allocate(int device, std::size_t channel_count,
                                std::size_t filter_length,
                                const Stream &stream);
};

// F32 modal state [channels,state_size].
struct IirCache final {
  DeviceBuffer state;
  std::size_t channels{0};
  std::size_t state_size{0};

  [[nodiscard]] Status allocate(int device, std::size_t channel_count,
                                std::size_t mode_count, const Stream &stream);
};

// Reusable channel-major cuFFT storage. Plans are dimension-exact and perform
// their internal allocations during construction, never during execution.
class FftWorkspace final {
public:
  FftWorkspace() = default;
  ~FftWorkspace();

  FftWorkspace(const FftWorkspace &) = delete;
  FftWorkspace &operator=(const FftWorkspace &) = delete;
  FftWorkspace(FftWorkspace &&other) noexcept;
  FftWorkspace &operator=(FftWorkspace &&other) noexcept;

  [[nodiscard]] Status allocate(int device, std::size_t channels,
                                std::size_t filter_groups,
                                std::size_t fft_size);
  void reset() noexcept;

  [[nodiscard]] bool matches(int device, std::size_t channels,
                             std::size_t filter_groups,
                             std::size_t fft_size) const noexcept;
  [[nodiscard]] int device() const noexcept { return device_; }
  [[nodiscard]] std::size_t channels() const noexcept { return channels_; }
  [[nodiscard]] std::size_t filter_groups() const noexcept {
    return filter_groups_;
  }
  [[nodiscard]] std::size_t fft_size() const noexcept { return fft_size_; }

private:
  friend Status bf16_fir_prefill_fft(const DeviceBuffer &, const DeviceBuffer &,
                                     const DeviceBuffer *, std::size_t,
                                     std::size_t, std::size_t, std::size_t,
                                     FirOrientation, FirBiasMode,
                                     DeviceBuffer *, FirCache *, FftWorkspace *,
                                     const Stream &);
  friend Status bf16_hcl_prefill(const DeviceBuffer &, const DeviceBuffer &,
                                 const DeviceBuffer &, const DeviceBuffer &,
                                 const DeviceBuffer &, const DeviceBuffer &,
                                 std::size_t, std::size_t, std::size_t,
                                 HclPrefillMode, IirCache *, DeviceBuffer *,
                                 DeviceBuffer *, FftWorkspace *,
                                 const Stream &);

  [[nodiscard]] Status execute(const Stream &stream);

  DeviceBuffer input_time_;
  DeviceBuffer filter_time_;
  DeviceBuffer output_time_;
  DeviceBuffer input_frequency_;
  DeviceBuffer filter_frequency_;
  cufftHandle input_forward_{0};
  cufftHandle filter_forward_{0};
  cufftHandle inverse_{0};
  int device_{-1};
  std::size_t channels_{0};
  std::size_t filter_groups_{0};
  std::size_t fft_size_{0};
};

[[nodiscard]] std::size_t fir_fft_size(std::size_t length,
                                       std::size_t kernel_size) noexcept;

// Projection is BF16 [length,3*width] with interleaved triples [x2,x1,v].
[[nodiscard]] Status
bf16_split_hyena_projection(const DeviceBuffer &projection, std::size_t length,
                            std::size_t width, DeviceBuffer *x2,
                            DeviceBuffer *x1, DeviceBuffer *value,
                            const Stream &stream);

// BF16 sequence/input [length,channels], grouped weight
// [filter_groups,kernel_size], optional per-channel BF16 bias, BF16 output.
// filter_groups must divide channels and follows repeat_interleave semantics.
[[nodiscard]] Status
bf16_fir_prefill_direct(const DeviceBuffer &input, const DeviceBuffer &weight,
                        const DeviceBuffer *bias, std::size_t length,
                        std::size_t channels, std::size_t filter_groups,
                        std::size_t kernel_size, FirOrientation orientation,
                        FirBiasMode bias_mode, DeviceBuffer *output,
                        FirCache *cache, const Stream &stream);

[[nodiscard]] Status bf16_fir_prefill_fft(
    const DeviceBuffer &input, const DeviceBuffer &weight,
    const DeviceBuffer *bias, std::size_t length, std::size_t channels,
    std::size_t filter_groups, std::size_t kernel_size,
    FirOrientation orientation, FirBiasMode bias_mode, DeviceBuffer *output,
    FirCache *cache, FftWorkspace *workspace, const Stream &stream);

// One BF16 token [channels] in, one BF16 token out; updates F32 cache.
[[nodiscard]] Status
bf16_fir_decode(const DeviceBuffer &input, const DeviceBuffer &weight,
                const DeviceBuffer *bias, std::size_t channels,
                std::size_t filter_groups, std::size_t kernel_size,
                FirOrientation orientation, FirBiasMode bias_mode,
                FirCache *cache, DeviceBuffer *output, const Stream &stream);

// HCS/HCM compose the inner gate x1*v, grouped FIR, and outer x2 gate. Scratch
// is BF16 [length,width] for prefill and BF16 [width] for decode.
[[nodiscard]] Status bf16_hcs_prefill(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, std::size_t length, std::size_t width,
    std::size_t filter_groups, std::size_t kernel_size, FirCache *cache,
    DeviceBuffer *scratch, DeviceBuffer *output, const Stream &stream);

[[nodiscard]] Status
bf16_hcs_decode(const DeviceBuffer &x2, const DeviceBuffer &x1,
                const DeviceBuffer &value, const DeviceBuffer &weight,
                std::size_t width, std::size_t filter_groups,
                std::size_t kernel_size, FirCache *cache, DeviceBuffer *scratch,
                DeviceBuffer *output, const Stream &stream);

[[nodiscard]] Status bf16_hcm_prefill(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, const DeviceBuffer &direct, std::size_t length,
    std::size_t width, std::size_t filter_groups, std::size_t kernel_size,
    FirCache *cache, DeviceBuffer *scratch, DeviceBuffer *output,
    FftWorkspace *workspace, const Stream &stream);

[[nodiscard]] Status bf16_hcm_decode(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, const DeviceBuffer &direct, std::size_t width,
    std::size_t filter_groups, std::size_t kernel_size, FirCache *cache,
    DeviceBuffer *scratch, DeviceBuffer *output, const Stream &stream);

// HCL tensors: x2/x1/value and direct are BF16; log_poles/residues and cache
// are F32. scratch is BF16 [length,width]. FFT mode requires an exact matching
// workspace with fft_size=fir_fft_size(length,length).
[[nodiscard]] Status
bf16_hcl_prefill(const DeviceBuffer &x2, const DeviceBuffer &x1,
                 const DeviceBuffer &value, const DeviceBuffer &direct,
                 const DeviceBuffer &log_poles, const DeviceBuffer &residues,
                 std::size_t length, std::size_t width, std::size_t state_size,
                 HclPrefillMode mode, IirCache *cache, DeviceBuffer *scratch,
                 DeviceBuffer *output, FftWorkspace *workspace,
                 const Stream &stream);

[[nodiscard]] Status
bf16_hcl_decode(const DeviceBuffer &x2, const DeviceBuffer &x1,
                const DeviceBuffer &value, const DeviceBuffer &direct,
                const DeviceBuffer &log_poles, const DeviceBuffer &residues,
                std::size_t width, std::size_t state_size, IirCache *cache,
                DeviceBuffer *output, const Stream &stream);

} // namespace evo2c::cuda
