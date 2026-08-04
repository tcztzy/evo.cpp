// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include <cufft.h>

#include "evo/cuda/runtime.hpp"
#include "evo/status.hpp"

namespace evo::cuda {

enum class FirOrientation { kCrossCorrelation, kCausalConvolution };
enum class FirBiasMode { kAdd, kMultiplyInput };
enum class FirWeightType { kBF16, kF32 };
enum class HclPrefillMode {
  kRecurrence,
  kFft,
  kFftStateless,
  kRecurrenceContinue
};
enum class FftInputMode { kRealCompact, kRealFullSpectrum };

[[nodiscard]] inline constexpr bool
hcm_prefill_uses_fft(const std::size_t tokens) noexcept {
  return tokens != 0;
}

// Vortex's dense HCM reference always uses fft_size=2*sequence_length and
// truncates its 128-tap filter to the sequence length before transforming it.
[[nodiscard]] inline constexpr std::size_t
hcm_fft_size(const std::size_t tokens) noexcept {
  return tokens <= static_cast<std::size_t>(-1) / 2 ? tokens * 2 : 0;
}

// Vortex's HCL reference also fixes n=2*sequence_length for the filter R2C,
// real-input full-spectrum FFT, and inverse C2R operations. It does not round
// the transform length up to a power of two.
[[nodiscard]] inline constexpr std::size_t
hcl_fft_size(const std::size_t tokens) noexcept {
  return tokens <= static_cast<std::size_t>(-1) / 2 ? tokens * 2 : 0;
}

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

  [[nodiscard]] Status
  allocate(int device, std::size_t channels, std::size_t filter_groups,
           std::size_t fft_size,
           FftInputMode input_mode = FftInputMode::kRealCompact);
  void reset() noexcept;

  [[nodiscard]] bool
  matches(int device, std::size_t channels, std::size_t filter_groups,
          std::size_t fft_size,
          FftInputMode input_mode = FftInputMode::kRealCompact) const noexcept;
  [[nodiscard]] int device() const noexcept { return device_; }
  [[nodiscard]] std::size_t channels() const noexcept { return channels_; }
  [[nodiscard]] std::size_t filter_groups() const noexcept {
    return filter_groups_;
  }
  [[nodiscard]] std::size_t fft_size() const noexcept { return fft_size_; }
  [[nodiscard]] std::size_t generation() const noexcept { return generation_; }
  [[nodiscard]] const DeviceBuffer &filter_time() const noexcept {
    return filter_time_;
  }
  [[nodiscard]] const DeviceBuffer &output_time() const noexcept {
    return output_time_;
  }

private:
  friend Status bf16_fir_prefill_fft(const DeviceBuffer &, const DeviceBuffer &,
                                     const DeviceBuffer *, std::size_t,
                                     std::size_t, std::size_t, std::size_t,
                                     FirOrientation, FirBiasMode,
                                     DeviceBuffer *, FirCache *, FftWorkspace *,
                                     const Stream &, FirWeightType);
  friend Status bf16_hcl_prefill(const DeviceBuffer &, const DeviceBuffer &,
                                 const DeviceBuffer &, const DeviceBuffer &,
                                 const DeviceBuffer &, const DeviceBuffer &,
                                 std::size_t, std::size_t, std::size_t,
                                 HclPrefillMode, IirCache *, DeviceBuffer *,
                                 DeviceBuffer *, FftWorkspace *,
                                 const Stream &);

  [[nodiscard]] Status execute(const Stream &stream);
  [[nodiscard]] Status ensure_hcl_state(std::size_t state_size);
  [[nodiscard]] Status execute_hcl_state(const DeviceBuffer &gated,
                                         const DeviceBuffer &log_poles,
                                         std::size_t length,
                                         std::size_t state_size,
                                         IirCache *cache, const Stream &stream);
  void reset_hcl_state() noexcept;

  DeviceBuffer input_time_;
  DeviceBuffer filter_time_;
  DeviceBuffer output_time_;
  DeviceBuffer input_frequency_;
  DeviceBuffer filter_frequency_;
  DeviceBuffer work_area_;
  DeviceBuffer state_input_;
  DeviceBuffer state_modes_;
  DeviceBuffer state_output_;
  DeviceBuffer state_work_area_;
  cufftHandle input_forward_{0};
  cufftHandle filter_forward_{0};
  cufftHandle inverse_{0};
  cufftHandle state_input_forward_{0};
  cufftHandle state_transform_{0};
  int device_{-1};
  std::size_t channels_{0};
  std::size_t filter_groups_{0};
  std::size_t fft_size_{0};
  std::size_t state_size_{0};
  FftInputMode input_mode_{FftInputMode::kRealCompact};
  std::size_t generation_{0};
};

[[nodiscard]] std::size_t fir_fft_size(std::size_t length,
                                       std::size_t kernel_size) noexcept;

// Projection is BF16 [length,3*width] with interleaved triples [x2,x1,v].
[[nodiscard]] Status
bf16_split_hyena_projection(const DeviceBuffer &projection, std::size_t length,
                            std::size_t width, DeviceBuffer *x2,
                            DeviceBuffer *x1, DeviceBuffer *value,
                            const Stream &stream);

// BF16 sequence/input [length,channels], explicitly selected BF16/F32 grouped
// weight [filter_groups,kernel_size], optional per-channel BF16 bias, BF16
// output. filter_groups divides channels and follows repeat_interleave.
[[nodiscard]] Status
bf16_fir_prefill_direct(const DeviceBuffer &input, const DeviceBuffer &weight,
                        const DeviceBuffer *bias, std::size_t length,
                        std::size_t channels, std::size_t filter_groups,
                        std::size_t kernel_size, FirOrientation orientation,
                        FirBiasMode bias_mode, DeviceBuffer *output,
                        FirCache *cache, const Stream &stream,
                        FirWeightType weight_type = FirWeightType::kBF16);

// Continues a sequence from an initialized F32 FIR cache and updates that
// cache with the final kernel_size-1 inputs.
[[nodiscard]] Status
bf16_fir_continue_direct(const DeviceBuffer &input, const DeviceBuffer &weight,
                         const DeviceBuffer *bias, std::size_t length,
                         std::size_t channels, std::size_t filter_groups,
                         std::size_t kernel_size, FirOrientation orientation,
                         FirBiasMode bias_mode, DeviceBuffer *output,
                         FirCache *cache, const Stream &stream,
                         FirWeightType weight_type = FirWeightType::kBF16);

[[nodiscard]] Status bf16_fir_prefill_fft(
    const DeviceBuffer &input, const DeviceBuffer &weight,
    const DeviceBuffer *bias, std::size_t length, std::size_t channels,
    std::size_t filter_groups, std::size_t kernel_size,
    FirOrientation orientation, FirBiasMode bias_mode, DeviceBuffer *output,
    FirCache *cache, FftWorkspace *workspace, const Stream &stream,
    FirWeightType weight_type = FirWeightType::kBF16);

// One BF16 token [channels] in, one BF16 token out; updates F32 cache.
[[nodiscard]] Status
bf16_fir_decode(const DeviceBuffer &input, const DeviceBuffer &weight,
                const DeviceBuffer *bias, std::size_t channels,
                std::size_t filter_groups, std::size_t kernel_size,
                FirOrientation orientation, FirBiasMode bias_mode,
                FirCache *cache, DeviceBuffer *output, const Stream &stream,
                FirWeightType weight_type = FirWeightType::kBF16);

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

[[nodiscard]] Status bf16_hcs_continue(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, std::size_t length, std::size_t width,
    std::size_t filter_groups, std::size_t kernel_size, FirCache *cache,
    DeviceBuffer *scratch, DeviceBuffer *output, const Stream &stream);

[[nodiscard]] Status bf16_hcm_prefill(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, const DeviceBuffer &direct, std::size_t length,
    std::size_t width, std::size_t filter_groups, std::size_t kernel_size,
    FirCache *cache, DeviceBuffer *scratch, DeviceBuffer *output,
    FftWorkspace *workspace, const Stream &stream,
    FirWeightType weight_type = FirWeightType::kBF16);

[[nodiscard]] Status bf16_hcm_prefill_direct(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, const DeviceBuffer &direct, std::size_t length,
    std::size_t width, std::size_t filter_groups, std::size_t kernel_size,
    FirCache *cache, DeviceBuffer *scratch, DeviceBuffer *output,
    const Stream &stream, FirWeightType weight_type = FirWeightType::kBF16);

[[nodiscard]] Status bf16_hcm_decode(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, const DeviceBuffer &direct, std::size_t width,
    std::size_t filter_groups, std::size_t kernel_size, FirCache *cache,
    DeviceBuffer *scratch, DeviceBuffer *output, const Stream &stream,
    FirWeightType weight_type = FirWeightType::kBF16);

[[nodiscard]] Status bf16_hcm_continue(
    const DeviceBuffer &x2, const DeviceBuffer &x1, const DeviceBuffer &value,
    const DeviceBuffer &weight, const DeviceBuffer &direct, std::size_t length,
    std::size_t width, std::size_t filter_groups, std::size_t kernel_size,
    FirCache *cache, DeviceBuffer *scratch, DeviceBuffer *output,
    const Stream &stream, FirWeightType weight_type = FirWeightType::kBF16);

// HCL tensors: x2/x1/value and direct are BF16; log_poles/residues and cache
// are F32. scratch is BF16 [length,width]. FFT modes require an exact matching
// workspace with fft_size=2*length. kFftStateless preserves the exact parallel
// output path but leaves the modal cache untouched for terminal scoring calls.
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
                DeviceBuffer *gated, DeviceBuffer *output,
                const Stream &stream);

} // namespace evo::cuda
