# evo.cpp

**Evo 2 inference without Python — and, for 7B, without a single bit of
numerical drift.**

**English** | [简体中文](README.zh_CN.md)

`evo.cpp` is a focused C++17/CUDA runtime for batch-1 Evo 2 inference. It runs
the official 1B, 7B, 20B, and 40B model families on one to four NVIDIA GPUs
from standard Safetensors checkpoints, without PyTorch, Vortex, Transformer
Engine, Python, or hardware FP8 at inference time.

It exists for one reason: make Evo 2 easier to deploy **without treating
numerical similarity as correctness**.

## Why evo.cpp?

- **Bit-exact 7B inference.** Every audited BF16/F32 bit matches a pinned
  Vortex/PyTorch reference: all logits, all 32 block outputs, cached
  generation, the 2,048-key softmax boundary, and the official 3,000-token
  prompt-forcing transition. There is no cosine, tolerance, or top-1 fallback.
- **A small native runtime.** The inference process is purpose-built C++/CUDA,
  not a Python framework distribution. It loads bounded, strictly validated
  Safetensors and exposes scoring, generation, and white-box tensor dumps.
- **Exactness does not require giving up throughput.** On one A800, the final
  exact 7B path reaches 1,071.5 / 7,619.2 / 11,369.3 tok/s for 16 / 128 /
  1,024-token prefill — 1.80× / 1.60× / 1.21× the warmed official runtime in
  the same comparison.
- **Original Evo 2 semantics on Ampere.** The Arc 1B/20B/40B checkpoints rely
  on Transformer Engine FP8 behavior. `evo.cpp` reproduces their fixed E4M3 and
  H100-QGMMA accumulation semantics in software, so deployment is not gated
  on Hopper FP8 instructions.

The design is deliberately narrow: one architecture, explicit numerical
contracts, native kernels, and failures instead of silent approximations. It
is closer in spirit to `llama.cpp` and `ds4.c` than to a general training
framework.

## Evidence, not promises

| Claim | Measured result |
|---|---|
| BF16 7B exactness | Zero unequal raw elements across every audited prefill block and logit, 16-step cached generation, long softmax dispatch boundaries, and CUDA 12.8 vs. 13.3 checks |
| Exact 7B prefill, 1×A800 | 1,071.5 / 7,619.2 / 11,369.3 tok/s at 16 / 128 / 1,024 tokens; warmed official Vortex: 595 / 4,776 / 9,412 tok/s |
| Exact 7B cached decode, 1×A800 | 72.62 tok/s median |
| Original Arc 40B quality gate, 4×A800 | 500-base identity: 99.2%, 97.2%, 78.2%, 99.2%; 93.45% mean; prompt 0 byte-identical to an independent oracle |
| BioNeMo BF16 40B, 2×A800 | Historical same-checkpoint run: 1.288 s vs. 8.710 s for 8-token generation, with ~45.4 GB vs. 55.546 GB recorded peak per GPU |

These are batch-1 results from the documented A800 environment, not universal
vendor benchmarks. The 40B measurements predate the current Safetensors-only
runtime and are marked accordingly. Exact commands, hashes, first-divergence
analysis, and test scope are recorded in the
[7B bit-exact audit](docs/vortex-7b-bit-exactness.md) and
[GPU validation record](docs/gpu02-environment.md).

## Quick start

PyTorch is needed only in an isolated conversion environment. The native
runtime never loads it.

```sh
python3 -m venv .venv-convert
. .venv-convert/bin/activate
python3 -m pip install -r requirements-convert.txt

scripts/convert_arc_checkpoint.sh \
  evo2_7b /models/evo2_7b.pt /models/evo2-7b.safetensors
```

Build the CUDA runtime:

```sh
cmake -S . -B build \
  -DEVO_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

Inspect, generate, or score:

```sh
MODEL=/models/evo2-7b.safetensors.index.json

build/evo-inspect "$MODEL"
build/evo -m "$MODEL" -p ACGT -n 32 --ctx 8192 --gpu 0
build/evo -m "$MODEL" --score sequences.fa --ctx 8192 --gpu 0
```

The converter uses safe CPU mapping, validates every registered tensor's
name, shape, and dtype, and writes standard size-based Safetensors shards.
Missing, duplicated, or unknown tensors are hard errors. See the
[conversion guide](docs/checkpoint-conversion.md) and
[format contract](docs/model-format.md).

## Supported paths

| Model path | Arithmetic contract | Suggested starting hardware at ctx 8K | Validation status |
|---|---|---:|---|
| Official 7B | BF16, bit-exact pinned Vortex/PyTorch semantics | 1×24 GB | Real checkpoint converted, benchmarked, and bit-exact audited on A800 |
| Arc 1B / 20B | Software E4M3 for trained FP8 projections | 1×16 GB / 2×40–48 GB | Registry, conversion, topology, and kernel contracts tested; real-checkpoint GPU run pending |
| Original Arc 40B | Software E4M3 plus H100-QGMMA accumulation emulation | 2×80 GB | Legacy runtime validated on 4×A800; Safetensors rerun pending |
| BioNeMo BF16 40B | BF16 | 2×80 GB | Legacy runtime aligned on 2×A800; Safetensors rerun pending |

The registry includes the official base and long-context variants. Hardware
figures are conservative starting points, not hard requirements; context
length, KV format, driver overhead, and other GPU processes affect memory use.

## What the guarantees mean

- **“Exact” is intentionally narrow.** It covers the official BF16 7B
  checkpoint and the pinned reference documented in the audit. Software-FP8,
  BioNeMo 40B, Q8 KV, sampling, and other GPU architectures have separate
  validation claims.
- **8K is the production quality gate.** Chunked prefill and paged Q8 KV make
  much larger contexts possible; 131K and 1M results are capacity and
  short-decode smokes, not claims of efficient full-context prefill.
- **This is an inference runtime.** Training, fine-tuning, LoRA, distributed
  serving, speculative decoding, and a stable library ABI are out of scope.
- **Unsupported cases fail explicitly.** The runtime does not quietly switch
  an exact path to approximately equivalent math.

## Read the implementation

- [Why 7B is bit-exact, down to the first divergent primitive](docs/vortex-7b-bit-exactness.md)
- [How software E4M3 preserves the Arc checkpoints on Ampere](docs/software-fp8.md)
- [The model's numerical contracts](docs/math-semantics.md)
- [Safetensors format and conversion](docs/model-format.md)
- [Reproducible GPU environment and validation artifacts](docs/gpu02-environment.md)
- [Architecture and acceptance criteria](SPEC.md)

`evo.cpp` is an independent runtime, not an Arc Institute or NVIDIA project. The
model architecture and checkpoints come from the
[official Arc Institute Evo 2 project](https://github.com/arcinstitute/evo2).
Code in this repository is licensed under Apache-2.0; model weights retain
their upstream licenses.
