# Evo 2 reference math semantics

The CPU F32 reference in `src/cpu_reference.cpp` is the executable numerical
specification for later CUDA kernels. It was checked against Vortex commit
`8b00afebeac745d1f31e7e2788f0e0e39fa47637`. Production kernels may change
layout or fuse operations, but must match these results within the tolerances in
`tests/vectors/cpu_ops.json`.

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
  `y = x2 * (sum(residue*state) + D*x1*v)`.
- RoPE uses non-interleaved GPT-NeoX half pairs. Positions are F32 and divided
  by the configured linear scaling factor (`128` for the 40B 1M model) before
  multiplication by the checkpoint `inv_freq` tensor.
- MHA packs Q, K, and V as `[time, 3, heads, head_dim]`, applies RoPE only to Q
  and K, scales scores by `1/sqrt(head_dim)`, and uses causal softmax.

The CUDA Hyena path keeps sequence tensors channel-last (`[time, width]`).
Grouped HCS/HCM filters use the checkpoint's `repeat_interleave` mapping: each
stored filter owns a contiguous group of channels. FIR decode caches are F32
`[width, kernel-1]` arrays in chronological order, while HCL modal caches are
F32 `[width, state_size]`. HCM prefill and the parallel HCL prefill path use
zero-padded cuFFT convolution; both populate the same caches consumed by the
single-token decode kernels. HCL also has a low-memory direct-recurrence prefill
path with identical state semantics.

CUDA attention stores RoPE-transformed keys and raw values in contiguous BF16
`[capacity, heads, head_dim]` caches. Chunked prefill uses the existing cache
length as both its causal prefix and RoPE position offset. Each query/head block
updates an F32 running maximum, normalizer, and value accumulator, so no
sequence-by-sequence score matrix is allocated. Single-token decode calls the
same online-softmax kernel against the populated cache.

The CPU attention implementation intentionally materializes a score vector for
clarity. CUDA and long-context paths must use online/chunked softmax and may not
materialize a full sequence-by-sequence score matrix.
