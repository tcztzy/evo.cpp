# evo2c

`evo2c` is a standalone C++17/CUDA inference runtime for Evo 2 40B and
40B-base. It runs the model on four Ampere `sm_80` GPUs without PyTorch,
Vortex, Transformer Engine, Hopper, or hardware FP8 instructions.

The implementation is intentionally narrow, in the style of `llama.cpp` and
`ds4.c`: one checked model container, one model architecture, native kernels,
an offline converter, and reproducible reference vectors.

## What works

- Four-GPU layer pipeline for the 50-layer StripedHyena 2 model.
- Byte tokenizer, FASTA/text scoring, greedy or sampled generation.
- HCS, HCM, HCL, MHA/RoPE, MLP, KV/FIR/IIR caches, and cached decode.
- Exact Transformer Engine 2.3 E4M3FN projection scales extracted from the
  checkpoint.
- Software H100-QGMMA accumulation on A800, with one-byte cached E4M3 weights.
- Parallel per-GPU stage loading and static E4M3 weight preparation.
- State-preserving chunked prefill with an 8192-token activation arena
  independent of the logical context capacity.
- Fixed-page Q8 KV at context capacities of 131072 tokens and above, with
  on-read F32 dequantization inside online-softmax attention.
- `EVO2C` v1 mmap model validation before any CUDA allocation.
- CPU, CUDA, SASS, converter, four-GPU CLI, and official continuation gates.

The quality-gated production target is batch 1 with `--ctx 8192`. A real 40B
8193-token prompt has additionally crossed the activation boundary under
`--ctx 32768`. A real 40B `--ctx 131072` smoke has also allocated and exercised
the paged Q8 KV path. Its two rows of logits retained at least
0.999999589 cosine similarity to the BF16-cache baseline with exact top-1 and
output bytes. The same binary path has passed a real `--ctx 1048576` capacity
smoke. These are capacity and numerical smokes, not claims that full 131072- or
1048576-token prefixes have been prefetched.

## gpu02 quick start

The checked gpu02 environment is Rocky Linux 8.10 with four A800 80GB PCIe
GPUs, driver 580.126.20, CUDA 12.8.93 in Apptainer, and bidirectional P2P
between every pair.

From this repository on the client:

```sh
scripts/gpu02_build.sh
scripts/gpu02_test.sh
scripts/gpu02_prepare_40b.sh
scripts/gpu02_quality.sh
```

The preparation script uses the required cache and mirror defaults:

```sh
export HF_HOME=/build/grp_icg/users/tang/.cache
export HF_ENDPOINT=https://hf-mirror.com
```

It resumes and verifies the official checkpoint, merges its two parts, and
creates:

```text
$HOME/evo2c-models/evo2-40b-e4m3sw.evo2
```

The validated file is 82,252,717,056 bytes with SHA256
`d1619e3b2eef0fba7c5838bb61982e891cf63d55385ced865af06693222d6687`.

On gpu02, define the container paths once:

```sh
image="$HOME/evo2c-cuda12.8-rocky8.sif"
nix_root="$HOME/.local/share/nix-root"
binary="$HOME/evo2c/build-gpu/evo2c"
model="$HOME/evo2c-models/evo2-40b-e4m3sw.evo2"
```

Generate ten greedy bases:

```sh
apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$model" -p ACGTACGT -n 10 --ctx 8192 \
  --gpu 0,1,2,3 --top-k 1 --seed 1
```

Long-context cache selection is automatic. `--ctx` below 131072 uses
contiguous BF16 KV; `--ctx 131072` or greater uses 16384-token Q8 pages.
Logical page tables cover the requested capacity, while physical pages are
allocated only when appended tokens first touch them:

```sh
apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$model" -p ACGTACGTACGTACGT -n 2 --ctx 131072 \
  --gpu 0,1,2,3 --top-k 1 --seed 1 \
  --dump-logits q8-logits.npy
```

The final `evo2c_metrics` record names the selected `kv_cache` format and
reports the currently allocated cache bytes per pipeline stage. Set
`--ctx 1048576` on the same command for the verified 1M-capacity path.

A completely populated 1M Q8 cache is projected at 33 GiB per GPU because each
pipeline stage owns two attention layers. With weights, non-attention state,
and the fixed activation arena, the four stages require 52.37–53.74 GiB each
before CUDA runtime overhead, so they fit on otherwise idle A800 80GB cards.
The corresponding BF16 stage totals would require 83.37–84.74 GiB and do not
fit.

Capacity is not prefill throughput. The current online attention kernel does
quadratic work and does not tile multiple queries over a shared K/V tile. A
full 1M causal prefill across the eight attention layers contains
549,756,338,176 causal query/source pairs per layer, about 144.1 PFLOP and
74.3 PB of direct Q8 K/V reads in the current kernel. Host offload would worsen
that repeated-read bottleneck. Use 1M capacity for sequences that grow
incrementally; a practical full 1M prefill still needs a tiled FlashAttention-
style Q8 kernel.

For the official long-prompt semantics, prefill 3000 bytes in parallel and
teacher-force the remainder through the caches:

```sh
apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$model" -p "$prompt" -n 500 --ctx 8192 \
  --gpu 0,1,2,3 --top-k 1 --seed 1 \
  --force-prompt-threshold 3000 \
  --dump-logits continuation-logits.npy \
  >continuation.bin 2>metrics.log
```

Score a text or FASTA file:

```sh
apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$model" --score sequence.fa --ctx 8192 \
  --gpu 0,1,2,3 >scores.jsonl 2>metrics.log
```

Generated bytes go to stdout. Diagnostics and a final `evo2c_metrics` JSON
record go to stderr. Score mode emits one JSON object per sequence.
Prompts and score records longer than the activation arena are split
automatically, and all chunk logits are retained for exact scoring.
`--force-prompt-threshold` preserves its meaning: the selected prefix is
chunk-prefilled and only the remainder is teacher-forced token by token.
Because one NPY tensor cannot represent stage-local activations from separate
invocations, `--dump-layer` currently rejects a prefill spanning multiple
chunks with an actionable error.

The official four-prompt, 500-base gate runs as a detached gpu02 worker:

```sh
scripts/gpu02_quality.sh
```

It can survive SSH reconnects and publishes the consolidated report at
`$HOME/evo2c-artifacts/t13-native-quality-4x500.json`, with prompt, target,
continuation, logits, and per-prompt records below the matching
`-artifacts` directory. To resume after an interrupted prompt, set
`EVO2C_QUALITY_START_INDEX` to its zero-based index.

## Build outside gpu02

CPU-only validation needs CMake, a C++17 compiler, and Python 3:

```sh
cmake -S . -B build-cpu -DEVO2C_CUDA=OFF \
  -DEVO2C_WARNINGS_AS_ERRORS=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

The production CUDA build additionally needs CUDA 12.x, cuBLASLt, cuFFT, four
visible GPUs for integration tests, and an `sm_80` target:

```sh
cmake -S . -B build-gpu -DEVO2C_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DEVO2C_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
```

PyTorch is conversion-only:

```sh
python3 -m venv .venv-convert
. .venv-convert/bin/activate
python3 -m pip install -r requirements-convert.txt
python3 tools/convert_checkpoint.py \
  --input evo2_40b.pt \
  --config configs/evo2-40b-1m.yml \
  --output evo2-40b-e4m3sw.evo2 \
  --dtype bf16
```

## Validation commands

The canonical local entrypoint always configures before it builds:

```sh
scripts/local_test.sh build-release
EVO2C_SANITIZE=ON scripts/local_test.sh build-sanitize
```

Both suites currently pass all 18 entries; three dependency-gated
PyTorch/Triton/Vortex oracles report their declared skip status when those
packages are unavailable. The production target is validated on gpu02 with:

```sh
scripts/gpu02_build.sh
scripts/gpu02_test.sh
```

The gpu02 suite passes all 26 entries: the CPU/converter contracts plus nine
CUDA tests, including two four-GPU tests. Four optional external-reference
tests are skipped in the pinned runtime. Build warnings are errors in all
canonical entrypoints.

## Limits

- The runtime is deliberately limited to Evo 2 40B/40B-base, batch 1, and
  exactly four CUDA devices. It is not a general model framework or server.
- The checked production environment is four Ampere `sm_80` A800 80GB cards.
  Other CUDA architectures and smaller cards are not claimed.
- The official continuation quality gate is `--ctx 8192`. A real 8193-token
  prompt validates chunking at `--ctx 32768`; 131K and 1M runs validate logical
  capacity, real-model Q8 numerics, and short cached decode.
- Full 1M Q8 KV residency fits idle A800 80GB hardware, but a full 1M prompt is
  not currently practical because the attention prefill kernel is quadratic
  and not query-tiled. No host-offload fallback is enabled.
- Multi-chunk scoring and generation are supported. `--dump-layer` is rejected
  when one prefill spans multiple activation chunks because a single NPY file
  cannot represent separate stage-local invocations.
- PyTorch is required only for offline checkpoint conversion and optional
  oracle tests. The production binary does not link PyTorch, Vortex,
  Transformer Engine, or hardware FP8 support.

## Design and validation

- [`SPEC.md`](SPEC.md) is the executable task and invariant ledger.
- [`docs/model-format.md`](docs/model-format.md) defines the checked container.
- [`docs/checkpoint-conversion.md`](docs/checkpoint-conversion.md) documents
  strict checkpoint handling.
- [`docs/math-semantics.md`](docs/math-semantics.md) fixes Vortex-compatible
  layer semantics.
- [`docs/software-fp8.md`](docs/software-fp8.md) explains why BF16 alone fails
  and how Ampere emulates the required FP8 accumulation.
- [`docs/gpu02-environment.md`](docs/gpu02-environment.md) records the remote
  environment and artifact hashes.

This is an independent Apache-2.0 implementation. See `NOTICE` for upstream
research and license attribution.
