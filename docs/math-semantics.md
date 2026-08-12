# Evo 2 reference math semantics

The CPU F32 reference in `src/cpu_reference.cpp` is the portable mathematical
specification for later CUDA kernels. The stronger BF16 7B requirement is the
bit-exact PyTorch/Vortex contract documented in
[`vortex-7b-bit-exactness.md`](vortex-7b-bit-exactness.md). A production kernel
may change layout or fuse operations only when it preserves the reference's
dtype boundaries and reduction order.

- RMSNorm uses `x / (sqrt(mean(x²)) + eps) * scale`. The epsilon is outside the
  square root.
- Linear weights follow PyTorch `nn.Linear`: `[out_features, in_features]`.
- The gated MLP is `l3(activation(l1(x)) * l2(x))`. Evo 2 uses exact GELU in
  block 0 and identity activation in blocks 1–49.
- The projection output consists of interleaved triples `[x2_i, x1_i, v_i]`.
- The input short filter and HCS use PyTorch depthwise `conv1d`
  cross-correlation order, so the last stored coefficient multiplies the
  current input. HCM uses causal convolution order, so coefficient zero
  multiplies the current input, and its `D` term multiplies `x1*v`.
- HCL keeps an F32 state per `(channel, mode)`:
  `state = exp(log_pole) * state + x1*v`, followed by
  `y = x2 * (sum(residue*state) + D*x1*v)`. The two F32 state operations are
  separately rounded. The contiguous decode sum uses a zero-padded 32-lane
  shuffle-down tree; the non-contiguous parallel-filter sum uses four
  independent accumulators.
- RoPE uses non-interleaved GPT-NeoX half pairs. Positions are F32 and divided
  by the configured linear scaling factor (`128` for the 40B 1M model) before
  multiplication by the checkpoint `inv_freq` tensor.
- MHA packs Q, K, and V as `[time, 3, heads, head_dim]`, applies RoPE only to Q
  and K, scales scores by `1/sqrt(head_dim)`, and uses causal softmax.

The CUDA Hyena path keeps sequence tensors channel-last (`[time, width]`).
Grouped HCS/HCM filters use the checkpoint's `repeat_interleave` mapping: each
stored filter owns a contiguous group of channels. FIR decode caches are F32
`[width, kernel-1]` arrays in chronological order, while HCL modal caches are
F32 `[width, state_size]`. HCM and HCL use `fft_size=2*length`, including
length one. Exact HCL cached prefill constructs the state with Vortex's
real-input full-spectrum FFT, ordinary complex multiplication, complex inverse
FFT, and real extraction at `length-1`; a mathematically equivalent recurrence
does not produce the same F32 bits. A continuation chunk applies direct FIR
against the retained chronological prefix and then advances that cache.

CUDA exact attention stores RoPE-transformed keys and raw values in contiguous
BF16 caches at every supported context length. Exact 7B cached generation reproduces Vortex
`CrossAttention`: BF16 key scaling, BF16 QK batched GEMM, BF16 causal mask,
PyTorch's dtype-preserving softmax, and BF16 probability/value batched GEMM.
The softmax uses the persistent warp kernel through 2048 keys and the same
register/shared/general reduction dispatch as PyTorch above that boundary.

The explicitly selected `fast-q8-kv` execution profile independently quantizes each K and V
`(token, head)` vector as
`scale=max(abs(vector))/127`, `q=clamp(round(vector/scale),-127,127)`. The int8
payload and F32 scale are stored in fixed 16384-token pages. The attention
kernel selects a page for each source, dequantizes its elements into F32, and
retains the same F32 dot-product, online-softmax, and value-accumulator
semantics as the previous scalable path. Q8 is not bit-equivalent to Vortex;
no context length silently selects it. The device page
tables cover the full logical context at load time, while physical payload and
scale pages are allocated on first append.

The activation arena is capped separately from context capacity. Sequence
scoring may use bounded continuation chunks. Exact generation performs one
cached prefill (the default is `min(3000, activation_capacity)`) and then
teacher-forces the remaining prompt through single-token cached decode. The
exact CrossAttention path intentionally materializes BF16 score and probability
workspaces because that is the pinned PyTorch implementation being reproduced;
the scalable online/Q8 path remains a separate, non-bit-exact execution
profile. See [`execution-profiles.md`](execution-profiles.md) for selection and
acceptance gates.

The Arc 1B, 20B, and 40B Hyena input projections use fixed Transformer Engine
2.3 E4M3FN (21, 21, and 42 projections respectively)
inference scales and H100 QGMMA K=32 global-alignment semantics. Their Ampere
software path is specified separately in `software-fp8.md`; replacing it with
an ordinary BF16 GEMM is not numerically equivalent.

The CPU attention implementation materializes a score vector for clarity.
Native non-cached BF16 prefill follows PyTorch SDPA's pinned FlashAttention
standard/split-KV dispatch, while exact cached generation follows the explicit
CrossAttention implementation described above.
