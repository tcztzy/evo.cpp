# gpu02 reproducible environment

Validated on 2026-07-23:

- Rocky Linux 8.10, x86-64
- four NVIDIA A800 80GB PCIe GPUs
- NVIDIA driver 580.126.20
- bidirectional CUDA P2P read/write available between every GPU pair
- Apptainer 1.4.5
- CUDA 12.8.93 inside `$HOME/evo.cpp-cuda12.8-rocky8.sif`
- image: 4,843,704,320 bytes, SHA256
  `d5f2682c742a4bd1af0f2d70c7c8d5e63bdbc6ca76c41df0c941b31849fe5667`
- native CUDA target: `sm_80`
- aria2 1.37.0 from the user's read-only Nix profile
- conversion Python 3.12 and CPU PyTorch 2.10.0 from the user's read-only
  Nix profile; the script bootstraps `$HOME/.venv-evo.cpp-convert`

The checkpoint cache is remote-only. Set
`HF_HOME=/build/grp_icg/users/tang/.cache/huggingface`; Hugging Face stores repository
objects below its `hub/` child. The official Hugging Face
revision is `d529aa57c30771814217ad89baaeaf6e2315c7d7`; gpu02 uses
`https://hf-mirror.com` (equivalent to `HF_ENDPOINT=https://hf-mirror.com`)
because the canonical endpoint is not reachable from the server.

Official checkpoint parts:

| File | Bytes | SHA256 |
|---|---:|---|
| `evo2_40b.pt.part0` | 41,126,745,847 | `3b74fa4e6158d49265e3e270ba8869390d064358f8bf3d2af0b3e1772728f485` |
| `evo2_40b.pt.part1` | 41,126,745,847 | `bdc4a76e0f23f8295e7061c2f0deff24f723bd916dc4cdc4d9216cac9c2d49d5` |

Historical generated artifacts from the pre-Safetensors container:

| File | Bytes | SHA256 |
|---|---:|---|
| `$HOME/evo.cpp-models/evo2_40b.pt` | 82,253,491,694 | `dd299612b1c1cdded0dfdcaf4d16f98fc97458261d80f4d662429f0ccb316bc3` |
| `$HOME/evo.cpp-models/evo2-40b-e4m3sw.evo2` | 82,252,717,056 | `d1619e3b2eef0fba7c5838bb61982e891cf63d55385ced865af06693222d6687` |
| `$HOME/evo.cpp-models/evo2-40b-bionemo-bf16.evo2` | 82,254,509,184 | `3fb2ec7ed2c89c4f88dcb9c4c6f675e46c2b37722ee82778ce0ff84794dfa5c8` |

## BioNeMo 40B BF16 provenance and oracle

The native BF16 model comes from NVIDIA's NGC resource
`evo2/40b-1m-fp8-bf16:1.0`. The downloaded archive is 63,680,606,710 bytes
with SHA256
`544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680`.
Its NeMo2 DCP manifest contains 506 BF16 data tensors and 210 byte metadata
entries. The evo.cpp converter maps those data tensors to 537 output tensors and
uses the same BF16-to-F32 filter materialization as NVIDIA's exporter.

As an independent loader check, Megatron Bridge 0.4.1 converted the same DCP
to its native checkpoint at
`$HOME/evo.cpp-models/evo2-40b-bionemo-mbridge-bf16`. The resulting 515 files
total 82,241,616,477 bytes; `.metadata` has SHA256
`bbd16963098f00812b400d8488e6588918ba515d6006d27efcfd35c34411dace`.

The numerical oracle uses BioNeMo Recipes 2.4 from source commit
`b35c2556209282bd9389fba24f5931f6701e50c5`, Megatron Bridge 0.4.1,
Megatron Core 0.17.0rc0, PyTorch
`2.13.0a0+8145d630e8.nv26.06`, Transformer Engine
`2.16.0+4220403e`, CUDA 13.3, and the
`nvcr.io/nvidia/pytorch:26.06-py3` base image. The local Apptainer image is
9,530,638,336 bytes with SHA256
`1df753c08c3169357e5ecfb9497873250f78efbf1080494550b59d9ce5fe0943`.
Because gpu02 cannot reach GitHub, all source dependencies are pinned to local
mirrors, including libnpy commit
`890ea4fcda302a580e633c624c6a63e2a5d422f6`; no Evo2 runtime code is patched.
The libnpy commit archive has SHA256
`c7b275c6cb8e46df43a20271e65010bdf63945831f2c0931ea6f2eda6a842acd`.
Triton is pointed at the driver mounted by Apptainer through
`TRITON_LIBCUDA_PATH=/.singularity.d/libs`.

NPY SHA256 values below identify the archived run artifacts. The current
libnpy writer uses its native standards-compliant header padding, so
regenerating an evo.cpp NPY can change the container length and SHA256 without
changing its shape, row-major F32 type, or raw payload bits. The validation
tools use the declared NPY header length and compare parsed array data; hashes
for external oracle files and generated raw byte outputs retain their stated
meaning.

The official TP=2 score oracle used `tests/vectors/t13_short.fasta` and emitted
a finite 16×512 F32 logit matrix. Its NPY SHA256 is
`9e16b0de532e57350b0b0ffdb9c48728b339c584925070ede75ea38d308d51d6`.
The extraction report has SHA256
`75221ee337d35b61c3a9008f7f661dff9bafc31329978211d588efaa65861b7f`.
For prompt `ACGTACGTACGTACGT`, the official TP=2 greedy oracle generated the
8-byte continuation `ACGTACGT`; its raw SHA256 is
`b28b7e7e6b70661dfee15d5290c4bca097ca145f721c4fbc4de73ad1d1660b8b`.
The JSONL record has SHA256
`acd9ffbd04aab93f446dbedabce493c330b0a422b7f4177f5599967a27594721`.
Generation took 8.71 seconds (0.9 tok/s) after model loading, with a reported
55.546 GB peak per rank.

The matching native gate ran on physical GPUs 1 and 2 while unrelated jobs
occupied about 8.5 GiB on each card. Runtime binary SHA256 was
`73b3c2ac5cdfd362ff9a4c4c973b81a4682437ddd1d08e92f60bb99e1f84c123`.
evo.cpp split the 50 layers evenly and reported peak allocation deltas of
45,392,855,040 and 45,376,077,824 bytes, so both processes remained well
below the 80 GiB physical limit even under contention.

| Native phase | Work | Seconds | Throughput |
|---|---:|---:|---:|
| Score load | 40B model | 55.172 | — |
| Score prefill | 16 tokens | 0.454 | 35.253 tok/s |
| Greedy load | 40B model | 54.878 | — |
| Greedy prefill | 16 tokens | 0.440 | 36.394 tok/s |
| Greedy decode | 7 measured steps | 0.849 | 8.248 tok/s |

The native score logits have SHA256
`99c2c6de5291a7b9e525921f1c4fa9a089b94b96eab39320a4d87a738cda2244`.
All values are finite, all 16 top-1 token IDs exactly match BioNeMo, and the
minimum row cosine is 0.999998748027925. The native greedy output is
byte-identical to the official `ACGTACGT`. The combined comparison report is
3,110 bytes with SHA256
`864f8f64ffaf18de005c770412c9ab31a1775c98556dd1da67ff49e0b984e44c`.
All artifacts are under
`$HOME/evo.cpp-artifacts/t22-bionemo-bf16-3fb2ec7e-gpu12`; its hash manifest has
SHA256
`fa68a8c1a061c992ba4f1fb3647309138ffaf15ca1fb29c02a446acbfd77f5b2`.

From the local repository, build and test with:

```sh
scripts/gpu02_build.sh
scripts/gpu02_test.sh
scripts/gpu02_smoke.sh
scripts/gpu02_prepare_40b.sh
scripts/gpu02_quality.sh
```

### Remote build roots and path overrides

The build, test, and smoke entrypoints share one path contract. If
`EVO_REMOTE_ROOT` is unset, the remote shell's real `$HOME` is the root; the
scripts never replace or export `HOME`. Setting, for example,

```sh
EVO_REMOTE_ROOT=/build/grp_icg/users/tang scripts/gpu02_build.sh
```

changes all root-derived defaults together. A leaf override always wins over
`EVO_REMOTE_ROOT`:

| Variable | Default |
|---|---|
| `EVO_REMOTE_SOURCE_DIR` | `<root>/evo.cpp` |
| `EVO_REMOTE_BUILD_DIR` | `<source>/build-gpu` |
| `EVO_REMOTE_DEPS_DIR` | `<root>/evo.cpp-deps` |
| `EVO_REMOTE_CONTAINER_PATH` | `<root>/evo.cpp-cuda12.8-rocky8.sif` |
| `EVO_REMOTE_NIX_ROOT` | `<root>/.local/share/nix-root` |
| `EVO_REMOTE_CACHE_DIR` | `<root>/.cache/evo.cpp` |

Thus an isolated validation can reuse the read-only dependencies and container
without touching the canonical source or build directory:

```sh
export EVO_REMOTE_ROOT=/build/grp_icg/users/tang
export EVO_REMOTE_SOURCE_DIR=/build/grp_icg/users/tang/evo.cpp-remote-check
export EVO_REMOTE_BUILD_DIR=/build/grp_icg/users/tang/evo.cpp-build-remote-check
scripts/gpu02_build.sh
```

Every supplied remote path must be absolute and may not contain whitespace or
`.`/`..` components. Source synchronization uses delayed updates and exclusions
for build trees, caches, environments, and model artifacts. It deliberately
does not use `rsync --delete`, so an existing remote-only artifact is not
removed. The image build likewise publishes a process-unique partial file by
atomic rename rather than deleting an older partial or image.

`EVO_APPTAINER` can select the remote Apptainer executable. `EVO_CMAKE_BIN` can
select an absolute CMake path inside the container. Without that override, each
candidate in `<nix-root>/store` is executed inside the container, the script
requires CMake >=3.25, and the highest verified version is selected.
CTest must exist beside that exact CMake binary and report the same version; the
scripts never select an unrelated CTest from another Nix store hash.
`EVO_PYTHON_BIN` can select Python inside the container; otherwise the highest
verified Nix Python >=3.9 is chosen, preferring a canonical `python3-<version>`
package over an environment wrapper at the same version.
`EVO_BUILD_JOBS` controls compile parallelism and defaults to 4; reduce it for a
shared host even though compilation does not occupy a GPU.

`gpu02_build.sh` constructs the configured container from
`containers/evo.cpp-cuda12.8-rocky8.def` when the image is absent, then reuses it
on later runs. Compilation does not query the GPU count and does not require four
idle devices.

### Visible GPUs and multi-GPU tests

`gpu02_test.sh` and `gpu02_smoke.sh` expose only physical GPU 0 by default.
Choose a different visible physical-device list explicitly:

```sh
EVO_CUDA_VISIBLE_DEVICES=2 scripts/gpu02_test.sh
```

The test binaries see those devices renumbered from logical device 0. Ordinary
and single-GPU tests do not require four cards. To claim the four-GPU branches,
request them explicitly and narrow the CTest selection to the tests that use
them:

```sh
EVO_CUDA_VISIBLE_DEVICES=0,1,2,3 \
EVO_CTEST_REQUIRED_GPUS=4 \
EVO_CTEST_REGEX='^(cuda_pipeline|cuda_cli)$' \
scripts/gpu02_test.sh
```

When `EVO_CTEST_REQUIRED_GPUS` is greater than one, the wrapper verifies that
the requested physical IDs exist and have no reported compute process before it
starts. This is a safety preflight, not a scheduler reservation: another process
can claim a device after the check. Inspect LSF and `nvidia-smi` first, and do not
run the multi-GPU command while another user's work is present.

The preparation script resumes both official checkpoint parts into the shared
Hugging Face cache with ordinary HTTP Range requests against the mirror. This
avoids depending on FP8 packages or the Xet reconstruction client. It verifies
exact sizes and SHA256 values before publishing cache blobs, merges them in
numeric order, records the merged SHA in evo.cpp metadata, performs the streaming
BF16 data conversion, extracts the 42 Transformer Engine projection scale
states, and validates the resulting 663-tensor file with `evo-inspect`.
Model files remain under `$HOME/evo.cpp-models` on gpu02.

The long-running preparation phase executes as a detached remote worker. Its
PID, log, and atomically published exit status are kept under
`$HOME/evo.cpp-models/.prepare-40b-jobs`; the local wrapper reconnects and resumes
polling after SSH or jump-host interruptions.

The official four-prompt generation gate uses the same detached-worker
contract through `gpu02_quality.sh`. It stores job control below
`$HOME/evo.cpp-artifacts/.quality-jobs` and writes its consolidated quality
report to `$HOME/evo.cpp-artifacts/t13-native-quality-4x500.json`.

## Evo 2 40B production validation

The native `ctx=8192` quality run used binary SHA256
`a2ce240351dddf4d2ec3af4622ff41cbfeb3e5f627f39e256f33c8447f79c7ec`
and the model SHA above. Its report is 3,777 bytes with SHA256
`314f50939dade079ef69494a57f954488082c361172547930c49ca4a02ef3a40`.

| Official prompt | Prompt bytes | Generated bytes | Identity | Continuation SHA256 |
|---:|---:|---:|---:|---|
| 0 | 3,268 | 500 | 99.2% | `8d9edb7e7a1e694b1d4e624516e252b3277cb4c55746535cb23af043b429c9fe` |
| 1 | 3,528 | 500 | 97.2% | `45c586cfe3fbe0f6eeba8795a3f8bea97f076f8b2e68f2977354d8fb597c6cc3` |
| 2 | 3,080 | 500 | 78.2% | `cba3393c44fe320b50996dbbaa97f24cb4115a8ef6804fb0405e5de862b2f09a` |
| 3 | 3,808 | 500 | 99.2% | `164196dd6b4fa01fdc160c658598cf98f5bd4866a335210ef179c83a35d95bfd` |

Mean identity is 93.45%, which passes the fixed 91.15±3 percentage-point
gate. Prompt 0 is byte-identical to the independent Vortex
software-H100-QGMMA oracle. Its first ten bytes also match a separate native
process exactly (`d5c1771dd559cf6b0e17a4b0bbc354ac212e98331c0e2dbbff238535e56ce213`),
establishing real-model greedy determinism for the same binary.

The uncontended packed-kernel prompt-0 benchmark recorded:

| Phase | Work | Seconds | Throughput |
|---|---:|---:|---:|
| Model load | 40B model | 60.290 | — |
| Parallel prefill | 3,000 tokens | 135.261 | 22.179 tok/s |
| Teacher force | 268 tokens | 99.037 | 2.706 tok/s |
| Cached decode | 9 tokens | 3.484 | 2.583 tok/s |

Observed peak deltas were 23.922, 23.904, 22.561, and 22.559 GiB on GPUs
0–3. The tiled/padded software-QGMMA path improved prefill from 8.35 to
22.18 tok/s (2.66×) without changing the continuation.

The exact-log-softmax score run used binary SHA256
`e6e356d24d5a425db43bef8dfad02060b3e0516d64700433572bd06bc7b413f7`.
For `tests/vectors/t13_short.fasta`, it scored 15 targets with total
log-likelihood -21.115582024538305, mean -1.4077054683025536, and perplexity
4.0865678786711825. The canonical artifact directory is
`$HOME/evo.cpp-artifacts/t13-native-score-short-e6e356d2`; `score.jsonl` has
SHA256 `84c5463f62bd0c0244518a1a9b19c7d453bd84de6486569618c9bc599ceb8906`.

The final gpu02 build passed all 25 CTest entries; four optional
PyTorch/Triton/Vortex reference tests reported the declared skip status.
Local Release and ASan/UBSan builds each passed all 17 entries, with their
three dependency-gated oracle tests skipped.

## T15 loading optimization

Binary
`f0a09d4af1cc2d73c665157b8e7e5b9b14abb014c7f81971c895565bf30776ff`
loads and statically quantizes the four independent pipeline stages
concurrently. The observed real-model initialization comparisons were:

| Workload | Sequential load | Parallel load | Change |
|---|---:|---:|---:|
| 16-token score under contention | 66.814 s | 56.186 s | -15.9% |
| Official prompt-0 10-token generation | 60.290 s | 55.801 s | -7.4% |

The score JSON remained byte-identical with SHA256
`84c5463f62bd0c0244518a1a9b19c7d453bd84de6486569618c9bc599ceb8906`.
The long-prompt continuation also remained byte-identical with SHA256
`d5c1771dd559cf6b0e17a4b0bbc354ac212e98331c0e2dbbff238535e56ce213`.
Artifacts are in
`$HOME/evo.cpp-artifacts/t15-parallel-load-score-f0a09d4a` and
`$HOME/evo.cpp-artifacts/t15-parallel-load-p0-10-f0a09d4a`.

The long-prompt run overlapped unrelated jobs using 60–70 GiB on GPUs 1–3,
so its prefill/decode throughput is not used as a kernel comparison. The
previous uncontended packed-kernel throughput table remains the reproducible
compute baseline. The parallel-loading build again passed all 25 gpu02 CTest
entries and both 17-entry local Release/Sanitizer suites.

## T16 chunked 32K context

Binary
`0129dc143986a3295dd5a1120e344c2f51f134746e215f90a1ee0dee6b4a80c1`
ran the real 40B model with `--ctx 32768`, an 8193-byte `A` prompt, one greedy
output token, and no teacher forcing. The runtime processed an 8192-token
initial activation chunk followed by a state-preserving one-token continuation.
It exited zero and emitted byte `0x41` (`A`), whose SHA256 is
`559aead08264d5795d3909718cdd05abd49572e84fe55590eef31a88a08fdffd`.

The artifact directory is
`$HOME/evo.cpp-artifacts/t16-ctx32768-chunk8193-0129dc14`. Its `metrics.log` is
1,169 bytes with SHA256
`3f00ec66c89ece53a848def79985892c7c9a666273dad03054c065d977c67225`.
Model loading took 56.614 seconds. Prefill took 1,064.976 seconds, or
7.693 tokens/s, while unrelated jobs saturated the same GPUs; that rate is a
contention observation, not a replacement for the uncontended T13 baseline.

The process-level allocation snapshot recorded 30,488, 30,470, 29,094, and
29,086 MiB on GPUs 0–3. The 630-byte snapshot has SHA256
`19fa6b8dd46e57473f4c75e935074fe119a4f2294706d5109223ba3fd61d07e9`.
Unlike the global peak deltas in `metrics.log`, these per-process values exclude
the other jobs sharing gpu02.

The final implementation build passed all 25 gpu02 CTest entries. Its
single-GPU and four-GPU model tests compare an 8+1 activation split against an
independent one-shot nine-token Python causal oracle; the CLI test also checks
automatic generation chunks, score-logit concatenation, teacher-force
separation, and the explicit multi-chunk layer-dump limitation.

## T17 paged Q8 KV and 131K capacity

Binary
`6f803ea0a5692da6a3c11190b40e15bfbeef5fb4bed8b6edd00976d52814cd41`
ran the real 40B model with `--ctx 131072`, prompt `ACGTACGTACGTACGT`,
two greedy output bytes, and a logits dump. It selected `q8_paged`, exited
zero, and emitted `AC`. The output SHA256 is
`472e73d796e20aa8ff9059e6316f218e0322548f661ec4dc267507ed66317404`.

The artifact directory is
`$HOME/evo.cpp-artifacts/t17-ctx131072-q8-short-6f803ea0`. Its `metrics.log`
has SHA256
`6782b4af158317c011f549a0b986839c5fd038bcb87050da136738844c3874c5`.
Loading took 58.650 seconds, 16-token prefill took 2.992 seconds, and one
measured decode step took 1.104 seconds. The allocated cache bytes were
4,450,353,664, 4,446,716,416, 4,449,960,448, and 4,446,323,200 on pipeline
stages 0–3. Each stage owns two attention layers; the Q8 payload plus one F32
scale per token/head uses 2.0625 GiB per layer at 131072 tokens.

A second run of the same binary, model, prompt, and seed used
`--ctx 32768`, selecting the contiguous BF16 cache. The dependency-free
`tools/compare_logits.py` comparator checked the two 2×512 F32 NPY files.
The Q8 rows had cosine similarities 0.9999995896399383 and
0.9999998089002153, exact top-1 token ids 65 and 67, maximum absolute
difference 0.125, finite values, and byte-identical generated output. The
comparison report has SHA256
`1006667defa9a5c2f4cd61dd183aa3b920936387112536806555639bfcaa94af`.

This run validates allocation, quantize/dequantize execution, cached decode,
and the real-model numerical gate at 131K capacity. The prompt itself is
short; it does not claim the quadratic work of pre-filling 131072 attention
positions. The final source build passed all 26 gpu02 CTest entries, including
the paged-cache boundary test and comparator contract. Four optional external
PyTorch/Triton/Vortex tests reported their declared skip status. Local Release
and ASan/UBSan builds each passed all 18 entries, with three dependency-gated
oracles skipped.

## T18 1M context capacity and compute boundary

Binary
`f9924b96ba784f312158b986f73a02fb5718a0aadcb1704d08779f0df2cceb67`
ran the real 40B model with `--ctx 1048576`, the same 16-byte prompt and
two-byte greedy generation used by T17. It selected `q8_paged`, exited zero,
and emitted `AC`. Its output and 2×512 logits are byte-identical to the 131K
Q8 run, with SHA256
`472e73d796e20aa8ff9059e6316f218e0322548f661ec4dc267507ed66317404`
and
`b087bfa8823f088d8de6bd05663afed2e23b636240ac66614a5af5e0a8ae70ff`.
The direct BF16 comparison again passed with minimum cosine
0.9999995896399383, exact top-1 and output bytes. A separate CUDA operator
test checks RoPE at position 1,048,575 against the CPU reference.

The canonical artifact directory is
`$HOME/evo.cpp-artifacts/t18-ctx1048576-q8-short-f9924b96`. Its 1,184-byte
`metrics.log` has SHA256
`19d82de81977896539bb00f77154eb0d3d34b8d7b6cf1f54f8267cf4c7f71614`;
the 926-byte BF16 comparison has SHA256
`f44700e37e057366707d68be15fda5c047b1b1fdec33d558f8bf4a802ac69a4c`.
Model loading took 55.571 seconds, 16-token prefill took 2.049 seconds
(7.807 tokens/s), and one measured decode took 0.605 seconds
(1.652 tokens/s). These rates were observed while unrelated jobs kept several
GPUs busy and are not replacements for the uncontended T13 benchmark.

The 1M page tables contain 64 logical pages per attention layer, but only the
first page was physically needed by this short run. Final cache accounting was
574,820,352, 571,183,104, 574,427,136, and 570,789,888 bytes on stages 0–3.
This includes one 276,824,064-byte Q8 page for each of the two attention layers
on a stage plus its Hyena state. The declared owned allocations
weights+cache+arena were 22,819,570,176, 22,800,040,448, 21,359,289,856, and
21,356,570,112 bytes. Global `cudaMemGetInfo` deltas are not process-isolated
when other jobs allocate or exit, so these component totals are the reliable
per-process accounting under contention.

For a completely populated 1M cache:

| Stage | Full Q8 cache | Weights + cache + arena |
|---:|---:|---:|
| 0 | 35,454,652,416 B | 57,699,402,240 B (53.737 GiB) |
| 1 | 35,451,015,168 B | 57,679,872,512 B (53.719 GiB) |
| 2 | 35,454,259,200 B | 56,239,121,920 B (52.377 GiB) |
| 3 | 35,450,621,952 B | 56,236,402,176 B (52.374 GiB) |

Each stage owns two of the eight MHA layers. One 16384-token page for one layer
uses `2*16384*64*128` int8 payload bytes plus `2*16384*64*4` scale bytes,
or 264 MiB. Sixty-four pages therefore use 16.5 GiB/layer and 33 GiB/stage.
The equivalent BF16 totals would be 83.37–84.74 GiB/stage including weights
and arena, beyond an 80GB A800; Q8 is required. The Q8 totals fit when gpu02 is
otherwise idle, while lazy allocation also allowed this smoke to run when GPU2
had only about 31 GiB free.

Full-prefix compute is the remaining boundary. At `N=1048576`, one attention
layer has `N*(N+1)/2 = 549,756,338,176` causal query/source pairs. Across
64 heads, the current scalar online kernel directly reads 16,896 Q8/scale bytes
and performs about 32,768 multiply/add FLOPs per pair. Across all eight MHA
layers this is about 74.3 PB of direct reads and 144.1 PFLOP. Even an idealized
2 TB/s bound is over ten hours when the four batch-1 pipeline stages execute
sequentially, before exponentials, launch overhead, Hyena layers, and
projections. Consequently T18 validates 1M addressability and incremental
decode, not a full 1M prefill. Host offload is not selected: the resident Q8
cache fits idle hardware, while PCIe replay of the quadratic read volume would
be slower. A useful full 1M prefill requires query-tiled, shared-K/V
FlashAttention-style kernels.

## T19 Evo 2 7B prefill optimization

The final 2026-07-31 comparison used otherwise-idle GPU3, one A800 80GB, and
the official `evo2_7b` checkpoint at Hugging Face revision
`bda0089f92582d5baabf0f22d9fc85f3588f6b58`. The source checkpoint SHA256 is
`c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4`.
The official run used its documented BF16 fallback because Transformer Engine
FP8 is not valid for this 7B profile on A800.

The native implementation now:

- shares an eight-source K/V tile across eight BF16 attention queries;
- caches cuBLASLt descriptors and selected algorithms;
- retains matching cuFFT buffers and plans across prefills;
- uses direct grouped HCM convolution through 128 tokens; and
- performs one 128-token backend warmup during 7B model loading.

The warmup is included in `model_load_seconds`, which was 2.07–2.12 seconds in
the final runs. It does not appear in `prefill_seconds`. “First scored” below
is therefore the first user record after model load. “Steady” is the mean of
the remaining records in the same process after excluding its first scored
record. The historical native baseline had no load-time warmup, so its first
score also included lazy backend initialization.

| Tokens | Historical native first score | Optimized first scored | Optimized steady | Official warmed median |
|---:|---:|---:|---:|---:|
| 16 | 0.257103 s / 62.2 tok/s | 0.036869 s / 434.0 tok/s | 0.011381 s / 1,405.9 tok/s | 0.026869 s / 595.5 tok/s |
| 128 | 0.242981 s / 526.8 tok/s | 0.016732 s / 7,649.9 tok/s | 0.016745 s / 7,643.9 tok/s | 0.026803 s / 4,775.7 tok/s |
| 1,024 | 0.444887 s / 2,301.7 tok/s | 0.124504 s / 8,224.6 tok/s | 0.102759 s / 9,965.1 tok/s | 0.108803 s / 9,411.5 tok/s |

Thus the optimized steady native path is 2.36×, 1.60×, and 1.06× the official
throughput at 16, 128, and 1,024 tokens. A previously unseen sequence length
can still create a new cuFFT plan, explaining why first-scored 16 and 1,024
latency is lower than the old native path but slower than the corresponding
steady result.

The final logits remained aligned with the official Python/Vortex output:

| Case | Minimum row cosine | Mean absolute error | Maximum absolute error | Top-1 agreement |
|---|---:|---:|---:|---:|
| Prefill 16 | 0.999999736 | 0.038712 | 0.250 | 93.75% |
| Prefill 128 | 0.999998178 | 0.061423 | 0.375 | 98.44% |
| Prefill 1,024 | 0.999992595 | 0.062613 | 0.500 | 99.90% |
| Generation 32 | 0.999998682 | 0.053591 | 0.250 | 100% |

The 32 generated bytes are exactly
`ACGTGCAATGCCGTTAACGTGCAATGCCGTTA`. Native cached decode measured
79.29 tok/s over 31 timed steps, compared with the official median of
42.89 tok/s.

The benchmarked T19 binary SHA256 is
`6283156c025d1ef9fd6195d8567540b4b505066a6292ee679152e4e1fd28dccd`;
the Safetensors index SHA256 is
`ee1757566fdf8616f706ebcce3ae65487beab6fab3246cff1762c799d7951e1e`.
Artifacts are under
`/data/grp_icg/users/tang/evo.cpp_7b_1m_runtime_compare/native_prefill_final`.
The current gpu02 build completed all 30 CTest entries with zero failures,
including nine CUDA tests and the one-/two-GPU pipeline paths. Five optional
external oracle tests reported their declared skip status.

A fresh revalidation of the current worktree ran on idle GPU1 on 2026-08-03.
The 7B Safetensors load, official-logit comparisons, byte-exact greedy output,
and performance gate all passed. Repeated prefill reached 1,265.7, 7,610.8,
and 9,727.5 tok/s for 16, 128, and 1,024 tokens respectively; model load was
2.07--2.14 seconds. The current binary SHA256 is
`805e81208609e035d23776695d416bd6354b7255d6a039439322bce2b76114eb`, and the
artifact directory is
`$HOME/evo.cpp-artifacts/t19-evo2-7b-prefill-805e81208609-gpu1`.
