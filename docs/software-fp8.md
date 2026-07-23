# Software E4M3 inference on Ampere

Evo 2 40B is not a plain BF16 checkpoint. Its 42 Hyena input projections were
trained and evaluated through Transformer Engine 2.3 `HYBRID` FP8. Running
those projections as BF16 changes the continuation distribution enough to
collapse greedy generation. The eight attention input projections remain
BF16.

The native `sm_80` runtime reproduces the required forward path without an FP8
instruction:

1. Read the fixed inference input and weight scales from each projection's
   checkpoint `._extra_state`.
2. Quantize the static BF16 weight once to one-byte E4M3FN codes while loading
   the model.
3. Quantize each normalized projection input to E4M3FN codes.
4. For every K=32 group, include the incoming accumulator when selecting the
   shared product exponent, truncate all aligned terms to the H100 QGMMA
   `(2 integer, 13 fractional)` accumulator, sum once, and truncate the
   normalized result.
5. Apply the two stored FP32 inverse scales after raw accumulation and round
   the projection output to BF16.

This ordering matters. Dequantizing both operands before an ordinary BF16
GEMM, or truncating only an already-summed FP32 dot product, does not reproduce
the model.

The kernel caches weights as compact E4M3 codes and uses shared-memory output
tiles. Codes are decoded to a signed four-bit significand and exponent once
per tile load; a padded K-major weight tile avoids shared-memory bank
conflicts. The implementation contains no native E4M3/E5M2 SASS instruction,
which is enforced by `tests/fp8_sass_contract.cmake`.

Numerical validation has four layers:

- CPU E4M3FN encoding vectors are bit-exact against PyTorch
  `float8_e4m3fn`, including subnormals, ties, and saturation.
- CUDA projection tests are bit-exact against an independent host
  global-alignment oracle for multi-group prefill and one-row decode.
- The official Vortex software oracle generates the first 500 bases of
  official prompt 0 at 99.2% target identity.
- The native four-A800 engine and the oracle both generate
  `TACACTCTCC` for the first ten bases; every step has the same greedy top-1
  and logit cosine above 0.999998.

The accumulator behavior is based on:

- Faizan A. Khattak and Mantas Mikaitis,
  [Accurate Models of NVIDIA Tensor Cores](https://arxiv.org/abs/2512.07004).
- Their BSD-2-Clause
  [MATLAB-tensor-core models](https://github.com/north-numerical-computing/MATLAB-tensor-core),
  version `0.5` / commit
  `bbcf00a273868172494eaacaa8d6128ab0fb8704`.
