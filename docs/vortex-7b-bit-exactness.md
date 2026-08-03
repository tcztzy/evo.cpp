# Vortex 7B bit-exact inference audit

This document records the white-box numerical audit of evo2c's BF16
`evo2_7b` path. “Exact” here means equality of the stored BF16/F32 bit
patterns. Cosine similarity, top-1 agreement, and tolerance checks are not
accepted as evidence of equality.

## Fixed reference

- Vortex commit: `8b00afebeac745d1f31e7e2788f0e0e39fa47637`
- PyTorch: `2.13.0a0+8145d630e8.nv26.06`
- Reference CUDA runtime: 13.3
- Checkpoint SHA256:
  `c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4`
- Hardware used for the audit: NVIDIA A800 80 GB (`sm_80`)
- Vortex optional fused depthwise, RMSNorm, HCS/HCM/HCL, FP8, and external
  FlashAttention flags were disabled. PyTorch SDPA remains active for ordinary
  parallel attention and dispatches its own FlashAttention backend; cached
  generation uses Vortex's explicit `CrossAttention` instead.

The native runtime does not load PyTorch, Vortex, libtorch, or Python. They are
used only to produce independent reference tensors.

## First-divergence results

The audit compared every block output and then instrumented the first
different block until the first different primitive was identified. The
differences were implementation differences, not a checkpoint-conversion
error.

| Area | Native behavior before the audit | Pinned PyTorch/Vortex behavior now reproduced |
|---|---|---|
| Hyena output projection | Treated the projection as an ordinary row-major linear | Uses the non-contiguous/channel-major input layout and the same bias epilogue |
| GELU | Combined GELU and the following gate, with toolkit-dependent `erf` | Materializes BF16 GELU before the BF16 gate; seven CUDA 12.8/13.3 `erf` boundary cases are canonicalized |
| HCM/HCL convolution | Used mathematically equivalent FFT sizes, normalization, and reordered arithmetic | Uses `n=2L`, PyTorch's exact R2C/C2R or real-full-spectrum FFT semantics, scaling points, and complex operation order |
| HCL filter reduction | Sequential mode sum | Uses the four independent accumulators selected by ATen for the non-contiguous `[width,state,length]` tensor |
| HCL length one | Used a recurrence shortcut | Still executes the official FFT path |
| QKV and RoPE | Used the converted layout directly and raw F32 sine/cosine | Canonicalizes checkpoint QKV rows and rounds the RoPE cache to BF16 before rotation |
| Parallel attention | Independent online attention | Uses the pinned FlashAttention forward kernel and its exact split-KV heuristic/combine kernel |
| Cached attention | Independent online softmax | Reproduces `CrossAttention`: BF16 K scaling, BF16 strided batched QK GEMM, BF16 mask, PyTorch softmax, and BF16 probability/value GEMM |
| Cached softmax | Persistent warp implementation only | Uses PyTorch's persistent kernel through 2048 keys, then its register, shared-memory, or general ILP block reduction path |
| HCL prefill state | Mathematically equivalent real recurrence | Reproduces Vortex's full modal FFT, complex inverse FFT, and real extraction at `L-1` |
| HCL decode state | Fused multiply-add | Uses separate F32 multiply and add, matching the two eager TensorIterator launches |
| HCL decode residue sum | Reused the filter's four-accumulator tree | Uses the contiguous 16-value, zero-padded 32-lane shuffle-down tree used by `torch.sum` |

The HCL distinction is important: two `torch.sum` calls over the same 16
mathematical modes do not use the same tree because one input is contiguous
and the other is not. Reusing one reduction implementation for both was enough
to leave a single BF16 mixer element wrong and to produce four wrong logits
after three decode updates.

## White-box code map

The comparison was performed by dumping every block output, finding the first
block with unequal raw storage, and then dumping progressively earlier tensors
inside that block. A change was accepted only when the first unequal primitive
became equal and every downstream block and logit remained equal. The relevant
source paths are:

| Contract | Pinned Vortex/PyTorch implementation | evo2c implementation |
|---|---|---|
| Exact GELU and the following BF16 gate | `vortex/model/layers.py::ParallelGatedMLP.forward`; PyTorch `aten/src/ATen/native/cuda/ActivationGeluKernel.cu::GeluCUDAKernelImpl` | [`pytorch_gelu_bf16`, `gated_kernel`, and `gelu_kernel`](../src/cuda/ops.cu) |
| Hyena short/medium/long convolution and state | `vortex/model/engine.py::fftconv_func`, `parallel_iir`, `prefill_via_modal_fft`, `step_iir`; `vortex/model/model.py::HyenaCascade.compute_filter` | [`hcl_filter_kernel`, modal-state FFT kernels, `hcl_decode_kernel`, and the `bf16_hc*` dispatchers](../src/cuda/hyena.cu) |
| Hyena output projection | `vortex/model/model.py::ParallelGatedConvBlock.forward`; `out_filter_dense` consumes the channel-major/non-contiguous mixer view and applies its bias | [`run_blocks`](../src/cuda/model.cu) and [`bf16_row_to_column_major`](../src/cuda/ops.cu), with the matching cuBLAS input layout and bias boundary |
| QKV checkpoint layout | `vortex/model/model.py::custom_load_state_dict`, which concatenates per-head Q/K/V rows before assigning `Wqkv` | [`canonicalize_qkv_weight`](../src/cuda/model.cu) and [`bf16_split_qkv`](../src/cuda/attention.cu) |
| RoPE dtype boundary | `vortex/model/rotary.py::RotaryEmbedding`, whose cosine/sine cache has the QKV dtype | [`rope_kernel`](../src/cuda/attention.cu), which rounds cosine and sine to BF16 before rotation |
| Parallel attention | `vortex/model/attention.py::SelfAttention.forward` calling `torch.nn.functional.scaled_dot_product_attention`; pinned PyTorch SDPA FlashAttention dispatch | [`bf16_flash_causal_attention`](../src/cuda/flash_attention.cu), built from FlashAttention commit `628452c73a4fab560189a7caa8702642c6a38235` with the pinned standard/split heuristic |
| Cached attention | `vortex/model/attention.py::CrossAttention.forward`: scaled K, two `einsum` calls, BF16 `masked_fill`, and dtype-preserving `torch.softmax` | [`bf16_cached_cross_attention` and all four cached-softmax kernels](../src/cuda/attention.cu); softmax follows pinned PyTorch `SoftMax.cu` and `block_reduce.cuh` |
| Generation state machine | `vortex/model/generation.py::Generator.generate`, including the default `force_prompt_threshold=3000` | [`run_generate` and `prefill_in_chunks`](../src/cuda/inference_cli.cpp), backed by [`prefill_cached` and exact token decode](../src/cuda/model.cu) |

This map also explains why several mathematically valid replacements failed.
An F32 recurrence is not the same program as Vortex's modal FFT; an online
softmax is not the same program as materializing BF16 scores and probabilities;
and a sequential sum is not the same program as ATen's layout-dependent CUDA
reduction. The exact path preserves those implementation choices rather than
matching only the real-number formula.

## Exact performance recovery

The post-audit optimization pass did not substitute approximately equivalent
math. It removed work only when that work cannot feed the requested result:

- Ordinary one-chunk `--score` calls use a stateless prefill mode. HCL still
  executes the exact PyTorch-compatible FFT convolution and produces the same
  BF16 mixer output, but skips the separate modal-state FFT because score does
  not continue decoding. Layer-dump and chunked/stateful calls retain the full
  state path.
- Cached attention owns one cuBLAS handle per pipeline stage instead of
  creating and destroying a handle in every layer and token. This removed 636
  process-level `cudaDeviceSynchronize` calls from the profiled 32-token
  generation case. Workspace capacity grows geometrically, but every executed
  scale, GEMM, softmax, and output GEMM is unchanged.
- Hyena's row-major to channel-major materialization is a 32x32 shared-memory
  tiled transpose. It copies `__nv_bfloat16` storage directly and never
  converts through F32. A partial-tile regression fills the input with arbitrary
  `uint16` patterns, including non-finite BF16 encodings, and requires exact
  output words. In the 128x20 profile its total kernel time fell from about
  9.02 ms to 2.05 ms.
- A one-token cached query skips the causal-mask launch because the existing
  predicate `key > query + key_tokens - query_tokens` is false for every key
  when `query_tokens == 1`. The score buffer is not modified in either case.

The final CUDA 12.8 binary used for the performance run has SHA256
`c770bfc9f9a51b6a13581001df7ff36b3373208128d4efe36ef394a57ddbf36b`.
Its 128-token score logits NPY file is byte-identical to the fixed reference.
For four cached generation steps, all 51 common intermediate NPY arrays have
zero unequal raw elements, and the complete logits NPY plus generated bytes are
byte-identical. These checks were rerun after the last performance change.

## Bit-exact evidence

All counts below compare raw values, not a tolerance:

| Case | Compared data | Result |
|---|---|---|
| Full prefill lengths 1, 7, 64, 128, and 129 | All 32 block outputs and all logits | Zero unequal elements |
| `ACGTGATTACAACGTT`, 16 generated tokens | 16×512 step logits, generated bytes, and final-decode outputs of all 32 blocks | Zero unequal elements |
| `ACGTGATTACAACGTT`, 4 generated tokens, CUDA 12.8 native versus CUDA 13.3 reference | 51 common NPY tensors (all 32 blocks plus layer-2 primitive/state dumps), 4×512 logits, and generated bytes | Zero unequal elements |
| `TTGACCATGNNRYACGT`, 8 generated tokens | 8×512 step logits, generated bytes, and final-decode outputs of all 32 blocks | Zero unequal elements |
| 2049-byte prompt, 2 generated tokens | 2×512 step logits across the 2048-key softmax dispatch boundary | Zero unequal elements |
| 3002-byte prompt with the official 3000-token force threshold | Cached prefill, two teacher-forced tokens, and 2×512 generation logits | Zero unequal elements |
| Cached softmax sizes 2048, 2049, 10240, 10241, and 30000 | Complete BF16 probability tensors and attention outputs | Reference BF16 byte hashes match |
| CUDA 12.8 build, length-129 prefill | All 32 block outputs and all logits against the CUDA 13.3 PyTorch reference | Zero unequal elements |
| CUDA 12.8 build, four cached generation steps | All 32 final-decode block outputs, HCL state, and 4×512 logits | Zero unequal elements |

The 10240/10241 softmax pair exercises the shared-memory/general-kernel
boundary and an unaligned second-head row. The 30000 case exercises the
general ILP path because the row no longer fits the shared-memory cache.

The model comparisons reinterpret BF16 outputs as `uint16` and F32 state or
logits as `uint32`; success requires `count_nonzero(left_bits != right_bits) ==
0`. The CLI regression compares complete NPY files byte-for-byte. It contains
no cosine, tolerance, or top-1 fallback.

The final CUDA 12.8 `sm_80` tree compiled every target with project warnings
treated as errors. Its single-GPU `cuda_smoke`, `cuda_ops`, `cuda_hyena`,
`cuda_attention`, `cuda_model`, and `cuda_cli` CTest entries all passed on the
same idle A800. Local Release and ASan/UBSan each completed all 22 configured
CTest entries without a failure; four unavailable external integrations were
declared skips in each run.

The audit also checks the probes themselves. An earlier cached-decode hook
recomputed a length-one parallel Hyena convolution, even though Vortex had
actually taken its FIR/IIR state path, while the native probe exposed an
unwritten gate scratch buffer. Those files looked like a model divergence but
did not feed inference. `generate_vortex_vectors.py` now records the actual
`step_iir` inputs and leaves parallel-only filter/convolution dumps at the
prefill call; `hcl_decode_kernel` writes the executed BF16 gate explicitly.
The fresh 51-tensor comparison above is after both probe fixes. Thus a future
diagnostic mismatch cannot be dismissed merely because final logits happen to
agree.

## CUDA 12.8 versus CUDA 13.3 GELU

`tools/cuda_gelu_map.cu` evaluates exact GELU for every BF16 input bit pattern.
The two toolkits differ for only seven finite inputs:

| BF16 input | CUDA 12.8 output bits | Pinned PyTorch/CUDA 13.3 bits |
|---:|---:|---:|
| -3.140625 | `0xbb2e` | `0xbb2d` |
| -4.03125 | `0xb8ea` | `0xb8eb` |
| -4.28125 | `0xb826` | `0xb827` |
| -4.5625 | `0xb740` | `0xb742` |
| -5.0 | `0xb5b4` | `0xb5c8` |
| -5.25 | `0xb4a8` | `0xb4fc` |
| -5.34375 | `0xb42b` | `0xb4ab` |

`pytorch_gelu_bf16` in `src/cuda/ops.cu` preserves the PyTorch expression and
canonicalizes precisely these seven outputs. The exhaustive map and the full
model checks show that this removes the toolchain-dependent divergence without
using a lookup table for ordinary inputs.

## Exactness boundary

This claim is deliberately narrow:

- It covers the official BF16 7B checkpoint and the fixed reference above.
- Exact cached generation requires a contiguous BF16 KV cache. The public
  exact path rejects paged Q8 KV rather than silently returning an
  approximation.
- The CLI performs one cached prefill and then advances the remaining prompt
  with exact single-token decode. Its default prefill threshold is the smaller
  of the official 3000 and the activation capacity; an explicitly larger
  threshold is rejected rather than split through a numerically different
  continuation path.
- Software-FP8 Arc models, the BioNeMo 40B model, Q8 long-context attention,
  sampling distributions, and other GPU architectures have separate quality
  claims. They are not made bit-exact by this audit.

The source-level regression tests cover the GELU exceptions, both HCL
reduction trees, cached CrossAttention, all four softmax implementations, and
the existing FlashAttention split boundary. Any future optimization must keep
those bit-pattern checks passing or explicitly declare a different numerical
mode.

The concrete regressions live in
[`tests/test_cuda_ops.cu`](../tests/test_cuda_ops.cu),
[`tests/test_cuda_hyena.cu`](../tests/test_cuda_hyena.cu),
[`tests/test_cuda_attention.cu`](../tests/test_cuda_attention.cu),
[`tests/test_cuda_model.cu`](../tests/test_cuda_model.cu), and
[`tests/test_cuda_cli.py`](../tests/test_cuda_cli.py). The softmax tests use
complete-tensor BF16 hashes produced by the fixed PyTorch build, not values
computed by the native implementation itself.
