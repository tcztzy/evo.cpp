# evo2c

**English** | [简体中文](README.zh_CN.md)

`evo2c` is an independent C++17/CUDA inference runtime for every official
Evo 2 size: 1B, 7B, 20B, and 40B, including the supported base and
long-context variants. It runs batch-1 inference on one to four CUDA GPUs.
The inference process does not depend on PyTorch, Vortex, Transformer Engine,
or hardware FP8 instructions.

The project follows the narrow-runtime philosophy of `llama.cpp` and `ds4.c`:
one checked model container, one model architecture, purpose-built native
kernels, offline weight conversion, and reproducible numerical tests. It is
not a new biological model, and it does not retrain or modify a checkpoint. It
solves the deployment problem of running Evo 2 on hardware without FP8.

## Support and validation matrix

The topology, source tensor manifest, converter, native registry, and
corruption checks are locally tested for every row. “GPU validated” means a
real converted checkpoint was actually compared or smoke-tested on CUDA; a
Python/Vortex reference vector alone is not counted as native validation.

| Model ID | Config | Projection semantics | Suggested starting hardware at ctx 8K | Real-checkpoint native status |
|---|---|---|---|---|
| `evo2_1b_base` | `evo2-1b-8k.yml` | fixed TE 2.3 software E4M3; BF16 source weight | 1×16 GB | Not yet run (checkpoint/GPU unavailable locally) |
| `evo2_7b` | `evo2-7b-1m.yml` | BF16 | 1×24 GB | Not yet run; existing 7B vectors are reference evidence only |
| `evo2_7b_base` | `evo2-7b-8k.yml` | BF16 | 1×24 GB | Not yet run |
| `evo2_7b_262k` | `evo2-7b-262k.yml` | BF16 | 1×24 GB | Not yet run |
| `evo2_20b` | `evo2-20b-1m.yml` | fixed TE 2.3 software E4M3; F32 source weight | 2×40–48 GB | Not yet run |
| `evo2_40b` | `evo2-40b-1m.yml` | fixed TE 2.3 software E4M3 | 2×80 GB | Validated on 4×A800 80 GB |
| `evo2_40b_base` | `evo2-40b-8k.yml` | fixed TE 2.3 software E4M3 | 2×80 GB | Not yet run |
| `evo2_40b_bionemo_bf16` | `evo2-40b-1m-bionemo-bf16.yml` | BF16 | 2×80 GB | Validated/aligned on 2×A800 80 GB |

These are conservative starting points, not hard-coded requirements. Pipeline
boundaries are chosen from layer payload bytes; context length, KV format,
driver overhead, and other GPU processes change the actual requirement.
Allocation failures remain explicit errors.

## Conclusions for researchers

- **BioNeMo alignment:** with the same NVIDIA
  `evo2/40b-1m-fp8-bf16:1.0` checkpoint, every value in the 16×512 logit
  matrix is finite, top-1 agrees on all 16 rows, and the 8-token greedy
  continuation is byte-identical.
- **The numerical differences are small and explainable:** the minimum
  row-wise cosine similarity is `0.999998748`; 2,723 of the 8,192 logits are
  exactly equal, and the mean absolute difference is `0.112738` while the
  largest absolute logit is `23.625`. Across the 15 actual target tokens, the
  mean absolute log-probability difference is only
  `0.008972 nat/token`.
- **Lower observed latency and memory use on gpu02:** in an 8-token greedy
  test on the same 2×A800 hardware and the same BF16 checkpoint, BioNeMo 2.4
  took `8.71 s` and evo2c took `1.288 s`; BioNeMo therefore took `6.76×` as
  long. The recorded per-GPU peak decreased from `55.546 GB` to approximately
  `45.4 GB`.
- **The original Arc checkpoint is independently validated:** four official
  long prompts generated 500 bases each, with identities of `99.2%`, `97.2%`,
  `78.2%`, and `99.2%`, for a mean of `93.45%`. Prompt 0 is byte-identical to
  an independent Vortex software-H100-QGMMA oracle.
- **The boundary is explicit:** `--ctx 8192` is the quality-gated production
  path. The 131K and 1M results are KV-capacity and short-decode numerical
  smokes, not claims that the current implementation can efficiently prefill a
  complete 1M-token prompt.

These results support using the validated configurations for sequence scoring
and generation. They do not mean that every downstream biological task has
already been proven equivalent. A serious study should still run a small
cross-validation on its own task, sequence distribution, and statistical
metric before scaling up.

## How evo2c relates to Arc/Vortex and BioNeMo

| Project | What it provides | 40B weights | Typical software stack | Support without Hopper |
|---|---|---|---|---|
| [Arc Institute Evo 2](https://github.com/arcinstitute/evo2) | Original model, checkpoints, paper, and research interfaces | Original Arc 40B/40B-base | Python + Vortex | The official documentation states that numerical accuracy for 40B depends on Hopper FP8 |
| [Vortex](https://github.com/Zymrael/vortex) | Arc's StripedHyena 2 inference and numerical implementation | Loads the original Arc weights | PyTorch + Transformer Engine, optionally FlashAttention | 7B can run in pure BF16; 40B requires Transformer Engine/FP8 |
| [NVIDIA BioNeMo](https://docs.nvidia.com/bionemo-framework/latest/models/evo2/index.html) | Evo 2 training, fine-tuning, prediction, and inference framework | Original weights and NVIDIA fine-tuned variants | PyTorch + NeMo/Megatron + Transformer Engine | NVIDIA's fine-tuned 40B checkpoint natively supports BF16 on Ampere+ |
| **evo2c** | Lightweight native inference runtime for official 1B/7B/20B/40B profiles | Supports Arc variants and the supported BioNeMo BF16 40B weights | C++17 + CUDA + cuBLASLt/cuFFT | Software E4M3 for 1B/20B/40B Arc; BF16 for 7B and BioNeMo |

The [Arc documentation](https://github.com/arcinstitute/evo2#requirements)
states that the original 1B, 20B, and 40B checkpoints are sensitive to the
Vortex-style FP8 numerical semantics used during training, so simply switching
them to BF16 may damage biological accuracy; 7B is the exception.
[NVIDIA documents the same issue](https://docs.nvidia.com/bionemo-recipes/latest/main/examples/bionemo-evo2/examples/fine-tuning-tutorial/index.html#fp8-and-hardware-compatibility)
and provides `evo2/40b-1m-fp8-bf16:1.0`, fine-tuned for FP8/BF16 on Ampere and
newer GPUs. NVIDIA also notes that this 40B variant has a slight accuracy
regression relative to the original model on Hopper FP8.

The two supported paths must therefore remain distinct:

1. **For stable BF16 40B inference on gpu02/A800:** use BioNeMo
   `evo2/40b-1m-fp8-bf16:1.0`. This is the recommended path and the checkpoint
   used in the BioNeMo performance and logit comparisons below.
2. **To reproduce the original Arc 40B weights and Vortex FP8 semantics:** use
   `arcinstitute/evo2_40b`. On Ampere, evo2c uses software E4M3 to emulate the
   checkpoint's fixed Transformer Engine 2.3 scales and H100 QGMMA
   accumulation semantics.
3. **Do not directly compare scores from the two checkpoints:** BioNeMo 40B is
   a separately fine-tuned set of NVIDIA weights, not another file format for
   the original Arc weights. A cross-checkpoint score difference includes a
   model-weight difference and cannot be attributed to the runtime.

## Same-checkpoint comparison with BioNeMo

### Test conditions

| Item | BioNeMo oracle | evo2c |
|---|---|---|
| Checkpoint | `evo2/40b-1m-fp8-bf16:1.0` | The same checkpoint converted to a read-only `.evo2` file |
| GPU | 2×A800 80GB | 2×A800 80GB on the same gpu02 server |
| Numerical path | BF16 | BF16 |
| Parallelism | Megatron tensor parallelism, TP=2 | 25+25 layer pipeline, PP=2 |
| Batch/context | batch 1, `ctx=8192` | batch 1, `ctx=8192` |
| Greedy prompt | `ACGTACGTACGTACGT` | Same |
| Output tokens | 8 | 8 |

The BioNeMo oracle used BioNeMo Recipes 2.4, Megatron Bridge 0.4.1, Megatron
Core 0.17.0rc0, PyTorch `2.13.0a0+8145d630e8.nv26.06`, Transformer Engine
`2.16.0+4220403e`, and CUDA 13.3. evo2c used CUDA 12.8.93. The complete
environment and artifact hashes are recorded in
[`docs/gpu02-environment.md`](docs/gpu02-environment.md).

### Performance

| Metric | BioNeMo 2.4 | evo2c | Observed difference |
|---|---:|---:|---:|
| 8-token generation time, excluding model load | 8.710 s | 1.288 s | BioNeMo/evo2c time ratio: `6.76×` |
| End-to-end output rate | Reported 0.9 tok/s | 6.21 tok/s | Approximately `6.9×` |
| Steady-state cached decode | Not reported separately | 8.248 tok/s | — |
| 16-token prefill | Not reported separately | 0.440 s / 36.394 tok/s | — |
| Recorded per-GPU peak | 55.546 GB | 45.393 / 45.376 GB | Approximately `18.3%` lower |
| Model load | Not reported separately | 54.878 s | — |

The evo2c generation time is `0.439633 s` of prefill plus `0.848684 s` for
seven measured decode steps. Dividing eight output tokens by that total gives
`6.21 tok/s`. BioNeMo's `0.9 tok/s` is the value reported by its log, while
`6.76×` is computed directly from `8.710 / 1.288`.

This is an **observed same-server, same-GPU-model, same-checkpoint
comparison**, not a controlled vendor benchmark. Other workloads on the
server were not identical between the runs, and the parallelization
strategies differ. BioNeMo's memory number is the framework-reported per-rank
peak, whereas evo2c reports the process's CUDA allocation delta. The result is
useful for understanding the deployment's practical scale, but it should not
be extrapolated as a fixed `6.76×` advantage for every sequence length, GPU,
or batch size.

## Logit difference tables

### Aggregate differences

The scoring input is `ACGTGATTACAACGTT`. Both implementations emit an F32 NPY
array with shape `[16, 512]`. In the tables below, signed delta means
`evo2c - BioNeMo`.

| Metric | Value |
|---|---:|
| Total elements | 8192 |
| Exactly equal | 2723 (33.24%) |
| Global mean absolute difference | 0.112738370 |
| Global RMS difference | 0.164874028 |
| Maximum absolute difference | 0.5 |
| Largest absolute logit across both outputs | 23.625 |
| Minimum row-wise cosine similarity | 0.999998748028 |
| Top-1 agreement | 16/16 |
| All values finite | Yes |

The most common element-wise signed deltas are:

| `evo2c - BioNeMo` | Elements | Share of 8192 |
|---:|---:|---:|
| `-0.125` | 2946 | 35.96% |
| `0` | 2723 | 33.24% |
| `+0.125` | 1459 | 17.81% |
| `-0.25` | 479 | 5.85% |
| `-0.5` | 476 | 5.81% |
| `-0.375` | 28 | 0.34% |
| `+0.25` | 10 | 0.12% |
| Other small BF16-grid deltas | 71 | 0.87% |

These discrete multiples of `0.125` are characteristic BF16 quantization
intervals at this numerical scale. They indicate finite-precision differences
accumulating through the layers, rather than random memory corruption or NaNs.

### Per-position differences

`target` is the next ground-truth token used for scoring. The last row has no
next target, but its logits are retained for a complete comparison. `exact`
is the number of numerically equal values among the 512 logits in that row.

| row | input | target | BioNeMo top-1 | evo2c top-1 | cosine | max abs | mean abs | exact / 512 |
|---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| 0 | A | C | T | T | 0.999999052520 | 0.25 | 0.240627 | 1 |
| 1 | C | G | T | T | 0.999998748028 | 0.125 | 0.115118 | 38 |
| 2 | G | T | A | A | 0.999999388966 | 0.5 | 0.487663 | 0 |
| 3 | T | G | T | T | 0.999999602599 | 0.25 | 0.125381 | 6 |
| 4 | G | A | T | T | 0.999999762142 | 0.125 | 0.001856 | 501 |
| 5 | A | T | A | A | 0.999999212200 | 0.125 | 0.116225 | 33 |
| 6 | T | T | A | A | 0.999999460517 | 0.125 | 0.004314 | 490 |
| 7 | T | A | T | T | 0.999999872079 | 0.125 | 0.001130 | 504 |
| 8 | A | C | T | T | 0.999999650671 | 0.125 | 0.120406 | 14 |
| 9 | C | A | A | A | 0.999999423607 | 0.125 | 0.117972 | 26 |
| 10 | A | A | T | T | 0.999999138835 | 0.125 | 0.115395 | 35 |
| 11 | A | C | A | A | 0.999999991730 | 0.0625 | 0.000132 | 508 |
| 12 | C | G | A | A | 0.999999855896 | 0.125 | 0.001003 | 504 |
| 13 | G | T | A | A | 0.999999210265 | 0.125 | 0.117914 | 25 |
| 14 | T | T | G | G | 0.999999491825 | 0.125 | 0.119436 | 19 |
| 15 | T | — | G | G | 0.999999469196 | 0.125 | 0.119240 | 19 |

Row 2 appears to have the largest difference, but its mean signed delta is
`-0.487434`, meaning that nearly all logits in the row are shifted together by
approximately `-0.5`. Softmax is exactly invariant to adding the same constant
to every value in a row, so `max abs = 0.5` must not be interpreted as a large
probability error. The actual probability impact is better measured after
log-softmax than by the largest raw-logit difference.

### Impact on sequence score

Applying stable log-softmax to the same 15 target tokens gives:

| Metric | BioNeMo | evo2c | Difference |
|---|---:|---:|---:|
| Total log-likelihood | -20.7778942673 | -20.7006628617 | +0.0772314056 |
| Mean log-likelihood | -1.3851929512 | -1.3800441908 | +0.0051487604 |
| Perplexity | 3.99560 | 3.97508 | -0.02052 |

The mean absolute log-probability difference over the 15 tokens is
`0.0089722188 nat/token`, and the maximum is `0.0607365483 nat`. These values
describe the real scoring impact better than the maximum raw-logit difference
of `0.5`. This short sequence is a numerical regression vector, not a
biological benchmark. Research conclusions should still be based on a
sufficiently large task dataset and confidence intervals.

An independent generation check uses the prompt `ACGTACGTACGTACGT`.
BioNeMo and evo2c both greedily generate `ACGTACGT`, with all eight bytes
identical.

## Why the cosine similarity is not exactly 1

Cosine similarity equals 1 only when two vectors have exactly the same
direction. The minimum observed value, `0.999998748028`, has
`1 - cosine ≈ 1.25×10⁻⁶`. The two 512-dimensional vectors therefore point in
almost exactly the same direction, but they are not element-wise identical.

The following facts have been established:

1. **The checkpoint is not different.** Both outputs use the same NGC
   archive. The converter copies ordinary BF16 weight payloads bit-exactly and
   strictly validates the name, shape, and dtype mapping from 506 source
   tensors to 537 output tensors.
2. **This is not a final, simple cast difference.** BioNeMo exports the final
   logits in an F32 file, but every exported value is already exactly
   representable in BF16; evo2c's logits are also on the BF16 grid. Rounding
   the BioNeMo output to BF16 one more time does not produce the evo2c output.
3. **The differences do not change the discrete decisions in this test.**
   Top-1 agrees on all 16 rows, the greedy continuation is byte-identical, and
   every logit is finite.

The most likely sources of the difference, in execution order, are:

1. **Different parallel decompositions.** The BioNeMo oracle uses TP=2,
   shards GEMMs, and performs collective reductions. evo2c uses PP=2 and
   computes each layer's complete GEMMs on the GPU that owns that layer.
   Floating-point addition is not associative, so sharding and reduction order
   change the least significant bits.
2. **Different GEMM tiling and epilogues.** BioNeMo goes through fused
   PyTorch/Transformer Engine/Megatron kernels. evo2c uses cuBLASLt with BF16
   inputs, FP32 accumulation, and BF16 layer outputs. The reduction tree, tile
   size, and bias/epilogue fusion points differ.
3. **Different normalization reductions.** evo2c implements BF16 RMSNorm with
   a custom warp reduction, while Transformer Engine uses its own fused
   implementation. Small changes in the order of the sum of squares affect
   the normalization scale.
4. **Different Hyena and attention kernels.** FIR/IIR filters, FFT, RoPE,
   online softmax, and cache access are independent implementations. A few ULPs
   introduced per layer can accumulate across 50 StripedHyena 2 blocks.
5. **BF16 writeback at every layer boundary.** Even when a GEMM accumulates in
   FP32, its layer output is rounded on writeback to BF16. A least-significant
   difference from one operation can place the next layer on an adjacent BF16
   value.

The evidence strictly establishes that a difference is present in the final
logits, but it **does not yet establish the exact first layer or operation
where the two executions diverge**. The official oracle exported only final
logits, not internal activations from all 50 layers. Locating the first
divergence would require read-only BioNeMo layer hooks and a comparison against
evo2c `--dump-layer` outputs for `pre_norm`, mixer output, residual,
post-norm, MLP, and block output. This README deliberately records that as
finer-grained work not yet performed, rather than presenting the likely causes
as a proven unique cause.

## Quality gate for the original Arc/Vortex checkpoint

The results below use the original `arcinstitute/evo2_40b` weights and the
evo2c software-E4M3 path. They are **not the same weights** as the BioNeMo BF16
checkpoint above. For every prompt, the first 3,000 bytes are prefetched in
parallel, the rest of the prompt is teacher-forced through the cache, and 500
bases are then generated greedily.

| Official prompt | Prompt bytes | Generated bases | Identity to official target |
|---:|---:|---:|---:|
| 0 | 3268 | 500 | 99.2% |
| 1 | 3528 | 500 | 97.2% |
| 2 | 3080 | 500 | 78.2% |
| 3 | 3808 | 500 | 99.2% |
| **Mean** | — | 2000 | **93.45%** |

The fixed gate is `91.15 ± 3` percentage points, and the current result passes.
Prompt 0's 500-base continuation is byte-identical to an independent Vortex
software-H100-QGMMA oracle. A separate repeated short generation from the same
binary is also byte-identical, establishing deterministic greedy output.

On an idle 4×A800 system, the original Arc path produced the following
observed prompt-0 performance:

| Phase | Work | Time | Throughput |
|---|---:|---:|---:|
| Model load | 40B | 60.290 s | — |
| Parallel prefill | 3000 tokens | 135.261 s | 22.179 tok/s |
| Teacher force | 268 tokens | 99.037 s | 2.706 tok/s |
| Cached decode | 9 tokens | 3.484 s | 2.583 tok/s |

This path uses four GPUs and compressed software-E4M3 projection weights. Its
speed and memory use must not be compared directly with the two-GPU BioNeMo
BF16 table above.

## Quick start on gpu02

The validated gpu02 environment is Rocky Linux 8.10, 4×A800 80GB PCIe, driver
580.126.20, and CUDA 12.8.93 inside Apptainer. Every pair of GPUs supports
bidirectional P2P.

### Build and test

Run from this repository on the client:

```sh
scripts/gpu02_build.sh
scripts/gpu02_test.sh
```

### Prepare BioNeMo BF16 40B (recommended)

```sh
scripts/gpu02_prepare_bionemo_40b.sh
```

The NGC archive uses a separate cache:

```text
/build/grp_icg/users/tang/.cache/bionemo
```

The script creates and validates:

```text
$HOME/evo2c-models/evo2-40b-bionemo-bf16.evo2
```

The file is `82,254,509,184` bytes, with SHA256
`3fb2ec7ed2c89c4f88dcb9c4c6f675e46c2b37722ee82778ce0ff84794dfa5c8`.

Reproduce the BioNeMo alignment gate on any two idle GPUs:

```sh
EVO2C_BIONEMO_GPU_LIST=1,2 scripts/gpu02_validate_bionemo_40b.sh
```

### Prepare the original Arc 40B checkpoint

The script defaults to the requested Hugging Face cache and mirror:

```sh
export HF_HOME=/build/grp_icg/users/tang/.cache
export HF_ENDPOINT=https://hf-mirror.com
scripts/gpu02_prepare_40b.sh
```

It pins the official revision, resumes interrupted transfers, validates both
parts and the merged file, and then creates:

```text
$HOME/evo2c-models/evo2-40b-e4m3sw.evo2
```

The file is `82,252,717,056` bytes, with SHA256
`d1619e3b2eef0fba7c5838bb61982e891cf63d55385ced865af06693222d6687`.
`HF_ENDPOINT` applies only to the Arc/Hugging Face path and is not used for the
NGC download.

### Manual offline conversion

The gpu02 preparation scripts already wrap download, resume, hashing, and
conversion. If the checkpoint is available locally, the converters can also
be called directly. PyTorch is required only for this offline step:

```sh
python3 -m venv .venv-convert
. .venv-convert/bin/activate
python3 -m pip install -r requirements-convert.txt

scripts/convert_arc_checkpoint.sh \
  evo2_7b evo2_7b.pt evo2-7b.evo2

python3 tools/convert_checkpoint.py \
  --input evo2_40b.pt \
  --config configs/evo2-40b-1m.yml \
  --output evo2-40b-e4m3sw.evo2 \
  --dtype bf16

python3 tools/convert_bionemo_checkpoint.py \
  --input /path/to/nemo2/weights \
  --config configs/evo2-40b-1m-bionemo-bf16.yml \
  --output evo2-40b-bionemo-bf16.evo2 \
  --source-sha256 544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680
```

The wrapper accepts every Arc model ID in the support table. Exact source
revisions, hashes, manifests, precision rules, and per-size smoke commands are
in [`docs/checkpoint-conversion.md`](docs/checkpoint-conversion.md).

### Run generation

On gpu02:

```sh
image="$HOME/evo2c-cuda12.8-rocky8.sif"
nix_root="$HOME/.local/share/nix-root"
binary="$HOME/evo2c/build-gpu/evo2c"
model="$HOME/evo2c-models/evo2-40b-bionemo-bf16.evo2"

apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$model" -p ACGTACGTACGTACGT -n 8 --ctx 8192 \
  --gpu 1,2 --top-k 1 --seed 1
```

Generated sequence bytes are written to stdout. Diagnostics and the final
`evo2c_metrics` JSON record are written to stderr.

### Run FASTA scoring

```sh
apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$model" --score sequence.fa --ctx 8192 \
  --gpu 1,2 >scores.jsonl 2>metrics.log
```

One JSON object is emitted for each FASTA record, including token count,
total/mean log-likelihood, perplexity, and per-token log-likelihood. Inputs
longer than 8,192 tokens use state-preserving chunked prefill, and scoring
still includes logits from every position.

### Original Arc long-prompt gate

```sh
scripts/gpu02_quality.sh
```

The task survives SSH reconnects. Its consolidated report is written to
`$HOME/evo2c-artifacts/t13-native-quality-4x500.json`.

## Long-context support and limitations

The following 32K, 131K, and 1M results come from the original Arc `.evo2`
checkpoint on 4×A800. The same long-context capacity gate has not yet been
extended to the two-GPU BioNeMo BF16 path.

- The quality-gated production target is batch 1 with `--ctx 8192`.
- A real 8,193-token prompt has crossed the 8,192-token activation arena under
  `--ctx 32768`, validating state-preserving chunking.
- `--ctx 131072` and above automatically select Q8 KV with 16,384-token pages.
  A physical page is allocated only when a token first writes to it.
- In the 131K smoke, the 2×512 logits retain a minimum cosine similarity of
  `0.9999995896` to the BF16-cache baseline, with exact top-1 and output bytes.
- The same binary has passed a real-model capacity smoke with
  `--ctx 1048576`.
- A fully populated 1M Q8 KV cache is projected to use approximately 33 GiB
  per GPU. Including weights, state, and activations, the four stages are
  projected at 52.37–53.74 GiB each and fit on otherwise idle A800 80GB cards.
- **A complete 1M-token prefill has not been performance-validated.** The
  current attention prefill remains quadratic and does not reuse K/V tiles
  across queries. The 1M capacity is useful for sequences that grow
  incrementally; it does not make a complete 1M-token prefix practical.

Example:

```sh
arc_model="$HOME/evo2c-models/evo2-40b-e4m3sw.evo2"

apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$arc_model" -p ACGTACGTACGTACGT -n 2 --ctx 131072 \
  --gpu 0,1,2,3 --top-k 1 --seed 1 --dump-logits q8-logits.npy
```

## Implementation scope

- Registry-selected 24-, 25-, 32-, or 50-layer StripedHyena 2 with HCS,
  HCM, HCL, MHA/RoPE, and MLP.
- KV, FIR, and IIR caches with cached autoregressive decode.
- Byte tokenizer, text/FASTA scoring, greedy sampling, and top-k/top-p
  sampling.
- One-to-four-GPU, payload-balanced contiguous layer pipeline at batch 1.
- Native BioNeMo BF16 Hyena projection path.
- E4M3FN one-byte weight cache and software-H100-QGMMA accumulation for the
  original Arc checkpoint.
- Fixed 8,192-token activation arena with state-preserving chunked prefill.
- Fixed-page Q8 KV for 131K+ contexts with F32 dequantization inside online
  softmax.
- `EVO2C` v1 mmap container validates header, tensors, shapes, dtypes, offsets,
  checksums, and model metadata before any CUDA allocation.
- `--dump-tokens`, `--dump-logits`, and `--dump-layer` support external
  numerical auditing.

The runtime does not support general model loading, training, LoRA, batch
sizes greater than one, a serving framework, or a non-CUDA backend. The
validated production hardware is A800 `sm_80`; older architectures, other CUDA
architectures, and smaller-memory devices are not claimed without testing.

Multi-chunk scoring and generation are supported. If one prefill spans
multiple activation chunks, `--dump-layer` fails with an explicit diagnostic
because one NPY file cannot represent multiple independent stage-local
invocations.

## Reproducibility and artifacts

| Artifact | Size (bytes) | SHA256 |
|---|---:|---|
| Merged Arc checkpoint | 82,253,491,694 | `dd299612b1c1cdded0dfdcaf4d16f98fc97458261d80f4d662429f0ccb316bc3` |
| Arc `.evo2` | 82,252,717,056 | `d1619e3b2eef0fba7c5838bb61982e891cf63d55385ced865af06693222d6687` |
| BioNeMo NGC archive | 63,680,606,710 | `544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680` |
| BioNeMo BF16 `.evo2` | 82,254,509,184 | `3fb2ec7ed2c89c4f88dcb9c4c6f675e46c2b37722ee82778ce0ff84794dfa5c8` |
| Official BioNeMo logits NPY | 32,896 | `9e16b0de532e57350b0b0ffdb9c48728b339c584925070ede75ea38d308d51d6` |
| evo2c logits NPY | 32,896 | `99c2c6de5291a7b9e525921f1c4fa9a089b94b96eab39320a4d87a738cda2244` |
| Logit comparison report | 3,110 | `864f8f64ffaf18de005c770412c9ab31a1775c98556dd1da67ff49e0b984e44c` |

The Arc checkpoint is pinned to Hugging Face revision
`d529aa57c30771814217ad89baaeaf6e2315c7d7`. Every preparation script verifies
the fixed size and SHA256 before publishing its final file. An existing
artifact with a mismatched hash is never silently reused.

The BioNeMo DCP manifest contains 506 BF16 data tensors and 210 metadata
entries, which are converted to 537 runtime tensors. Ordinary BF16 payloads
are bit-exact. A missing, duplicate, or unknown tensor, or an inconsistent
shape or dtype, fails conversion rather than being silently ignored.

## Local build and tests

CPU-only validation requires CMake, a C++17 compiler, and Python 3:

```sh
cmake -S . -B build-cpu -DEVO2C_CUDA=OFF \
  -DEVO2C_WARNINGS_AS_ERRORS=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

A CUDA build requires CUDA 12.x, cuBLASLt, cuFFT, and an `sm_80` target:

```sh
cmake -S . -B build-gpu -DEVO2C_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DEVO2C_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
```

Canonical local entrypoints:

```sh
scripts/local_test.sh build-release
EVO2C_SANITIZE=ON scripts/local_test.sh build-sanitize
```

Current status:

- Local Release: 21/21 passed.
- ASan/UBSan: 21/21 passed.
- Last pre-multi-size gpu02 baseline: 28/28 passed, including nine CUDA tests
  and two multi-GPU tests. This multi-size change has not been rebuilt or run
  on gpu02 because the current host has no CUDA compiler/GPU.
- Real PyTorch DCP converter integration: 5/5 passed.

PyTorch is used only for offline checkpoint conversion and optional oracle
tests. The production binary does not link PyTorch, Vortex, or Transformer
Engine.

## Design and audit documentation

- [`SPEC.md`](SPEC.md): executable invariants, validation gates, and task
  ledger.
- [`docs/model-format.md`](docs/model-format.md): checked `.evo2` container.
- [`docs/checkpoint-conversion.md`](docs/checkpoint-conversion.md): strict Arc
  and BioNeMo checkpoint conversion rules.
- [`docs/math-semantics.md`](docs/math-semantics.md): Vortex-compatible layer
  semantics.
- [`docs/software-fp8.md`](docs/software-fp8.md): why the original
  1B/20B/40B checkpoints cannot simply switch to BF16, and how Ampere
  emulates the required FP8 semantics.
- [`docs/gpu02-environment.md`](docs/gpu02-environment.md): gpu02 environment,
  benchmarks, artifact paths, and SHA256 values.

## Upstream sources and license

- [Evo 2 paper](https://www.nature.com/articles/s41586-026-10176-5)
- [Arc Institute Evo 2 repository](https://github.com/arcinstitute/evo2)
- [Arc `evo2_40b` checkpoint](https://huggingface.co/arcinstitute/evo2_40b)
- [Vortex inference repository](https://github.com/Zymrael/vortex)
- [BioNeMo Evo 2 documentation](https://docs.nvidia.com/bionemo-framework/latest/models/evo2/index.html)
- [BioNeMo FP8/BF16 hardware compatibility](https://docs.nvidia.com/bionemo-recipes/latest/main/examples/bionemo-evo2/examples/fine-tuning-tutorial/index.html#fp8-and-hardware-compatibility)

This project is an independent Apache-2.0 implementation. See
[`NOTICE`](NOTICE) for upstream research, code, and checkpoint attribution.
Users must also comply with the terms of the corresponding checkpoint source.
