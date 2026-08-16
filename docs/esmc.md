# Native Biohub ESMC inference

`evo.cpp` supports the three public Biohub ESMC masked protein language models
as a separate `esmc-runtime-v1` artifact family. Conversion and oracle creation
may use Python tooling, but the product inference path is native C++17/CUDA and
does not load Python, PyTorch, Transformers, `esm`, or libtorch.

## Verified upstream identities

The registry pins complete 40-hex Hugging Face revisions and hashes every
required source file. The old lowercase `*-2024-12` Hugging Face repositories
are legacy identities; their names remain the Biohub hosted aliases, but new
source acquisition uses the current uppercase repositories below.

| Registry ID | Canonical Hugging Face repository and revision | Biohub hosted alias | Width / layers / heads / MLP width | Source weights |
|---|---|---|---:|---|
| `esmc_300m` | [`biohub/ESMC-300M@a59b831785f907e96e6a246b1d142bfb76df31ee`](https://huggingface.co/biohub/ESMC-300M/tree/a59b831785f907e96e6a246b1d142bfb76df31ee) | `esmc-300m-2024-12` | 960 / 30 / 15 / 2560 | one F32 Safetensors file |
| `esmc_600m` | [`biohub/ESMC-600M@a7e82012c83126b9eedb055fea9fa84b6c02f094`](https://huggingface.co/biohub/ESMC-600M/tree/a7e82012c83126b9eedb055fea9fa84b6c02f094) | `esmc-600m-2024-12` | 1152 / 36 / 18 / 3072 | one F32 Safetensors file |
| `esmc_6b` | [`biohub/ESMC-6B@45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`](https://huggingface.co/biohub/ESMC-6B/tree/45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a) | `esmc-6b-2024-12` | 2560 / 80 / 40 / 6912 | six indexed F32 Safetensors shards |

All three have vocabulary size 64 and a maximum encoded length of 2048,
including the automatically added CLS and EOS tokens. The exact source file
sizes and SHA256 values live in
[`configs/model-registry.json`](../configs/model-registry.json); they are part
of the acquisition and conversion gate, not documentation-only checks.

The numerical reference is Biohub's Transformers fork at commit
[`3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf`](https://github.com/Biohub/transformers/tree/3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf/src/transformers/models/esmc).
The forward contract is the pinned
[`modeling_esmc.py`](https://github.com/Biohub/transformers/blob/3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf/src/transformers/models/esmc/modeling_esmc.py),
and tokenizer behavior is the pinned
[`tokenization_esmc.py`](https://github.com/Biohub/transformers/blob/3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf/src/transformers/models/esmc/tokenization_esmc.py).
The ESMC model weights and Biohub `esm` code are MIT-licensed and use is also
subject to Biohub's Acceptable Use Policy; review the pinned
[`esm` license notice](https://github.com/Biohub/esm/blob/26b0bc2b771e3e419ea74f445a5f35cc094a1509/LICENSE.md)
and the policy linked by the model cards. Biohub's Transformers fork is
Apache-2.0. These upstream terms are separate from `evo.cpp`'s Apache-2.0
license.

## Architecture and tokenizer contract

The implementation follows the official bidirectional Pre-LN transformer:
fused layer-norm/QKV projection, per-head Q/K normalization, non-interleaved
RoPE, unmasked scaled dot-product attention, output residual scaled by
`sqrt(num_layers / 36)`, layer-norm/SwiGLU feed-forward, a second scaled
residual, and final layer normalization. The masked-LM head is
linear-plus-bias, exact GELU, layer norm, then linear-plus-bias. Source weights
remain F32 on the native CPU, MPS, and CUDA paths.

The tokenizer adds token 0 (CLS) and token 2 (EOS). Its allocated tokens are:

```text
0 <cls>  1 <pad>  2 <eos>  3 <unk>
4 L  5 A  6 G  7 V  8 S  9 E  10 R  11 T  12 I  13 D  14 P
15 K  16 Q  17 N  18 F  19 Y  20 M  21 H  22 W  23 C
24 X  25 B  26 U  27 Z  28 O  29 .  30 -  31 |  32 <mask>
```

Tokens 33 through 63 are reserved padding vocabulary. The multi-byte literal
`<mask>` is recognized as token 32, `|` is token 31, and an otherwise unknown
input byte becomes token 3. For example, `LAG<mask>|Z?` encodes to
`[0,4,5,6,32,31,27,3,2]`. Input may contain explicit registered special-token
literals; the runtime still owns the outer CLS/EOS pair. Since those two tokens
count toward the 2048 limit, a plain residue sequence is limited to 2046 tokens.

Padding is not inserted for the CLI's batch-one sequence records. The C API
also accepts one sequence per exact context. A full encoded sequence is run in
one bidirectional forward pass; ESMC inference is never split into recurrent or
KV-cache chunks.

## Offline acquisition and conversion

Install the fetcher's Python dependency in a conversion-only environment, then
fetch by registered ID. The command verifies the pinned revision, every file's
size and SHA256, and writes an atomic receipt:

```sh
python3 -m venv .venv-esmc-convert
. .venv-esmc-convert/bin/activate
python3 -m pip install -r requirements-fetch.txt

python3 tools/evo_fetch.py source esmc_300m > fetch.json
RECEIPT="$(python3 -c 'import json; print(json.load(open("fetch.json"))["receipt"])')"

PYTHONPATH=tools python3 tools/convert_esmc_checkpoint.py \
  --receipt "$RECEIPT" \
  --output /models/esmc-300m.safetensors
```

Repeat with `esmc_600m` or `esmc_6b`. The converter is deliberately
torch-free. It reads Safetensors headers and payload ranges directly, checks
the pinned config and tokenizer, requires the complete official tensor
name/shape/F32 manifest, accepts only the expected zero-byte U8 `_extra_state`
entries, and writes atomically. The 6B result is sharded automatically; use the
printed `.safetensors.index.json` load path.

For a disconnected conversion host, populate the cache once and repeat the
source command with `--local-files-only`. Corrupt or incomplete cached bytes
fail rather than being trusted.

On gpu02, keep Hugging Face objects in the high-capacity shared cache. The
maintained acceptance runner exports this default and passes its `hub/` child
to the fetcher explicitly:

```sh
export HF_HOME=/build/grp_icg/users/tang/.cache/huggingface
python3 tools/evo_fetch.py --cache-dir "$HF_HOME/hub" \
  --local-files-only source esmc_6b
```

`EVO_ESMC_HF_HOME` and `EVO_ESMC_CACHE_DIR` are explicit runner overrides;
ordinary gpu02 validation must not reconstruct checkpoints below `$HOME`.

Converted runtime weights can also be distributed in a Hugging Face repository
with the standard `evo-artifact.json` manifest described in
[artifact acquisition](artifact-distribution.md). Use
`artifact_profile: "esmc-runtime-v1"`, list the index and every shard with exact
size/SHA256, and pin the runtime repository commit. After
`evo-fetch runtime OWNER/REPO@COMMIT`, `-hf OWNER/REPO@COMMIT` performs no
network access: native C++ revalidates the cached ref, receipt, manifest,
profile, paths, sizes, and SHA256 values before opening weights.

## CLI outputs

ESMC exposes masked-LM logits and hidden states, not causal generation or
next-token scoring:

```sh
MODEL=/models/esmc-300m.safetensors

build/evo logits -m "$MODEL" --input proteins.fa --output logits \
  --ctx 2048 --gpu 0

build/evo embed -m "$MODEL" --input proteins.fa --output embeddings \
  --layer 30 --pooling none --ctx 2048 --gpu 0
```

Use `--backend cpu --profile cpu-f32` for the portable scalar F32 reference
path. On Apple silicon, `--backend mps --profile mps-f32` accelerates ESMC
linear layers while retaining host F32 attention and nonlinear operations.
Both are approximate portability profiles intended for correctness checks and
small inputs; production-sized models are computationally expensive on these
paths. CUDA ESMC v1 accepts exactly one GPU and `--profile exact`. Multi-GPU
placement, hybrid placement, `fast-q8-kv`, causal score, variant score,
generation, incremental decode, and server startup return typed `unsupported`
errors before CUDA weight upload.

`evo logits` streams FASTA, FASTQ, raw text, gzip, or stdin records. It creates
an empty output directory containing `000000.npy`, `000001.npy`, and so on,
plus `logits.jsonl`. Each row-major F32 matrix has shape
`[encoded_tokens, 64]`, including CLS and EOS rows.

`evo embed` writes the same deterministic file numbering and
`embeddings.jsonl`. For an `n`-block ESMC model, valid official hidden-state
indices are:

- `0`: token embedding before block 0;
- `i` in `[1, n-1]`: output of block `i-1` before the following block;
- `n`: output after the final layer norm.

The manifest records `point=official_hidden_state`. `pooling=none` preserves
all CLS/residue/EOS rows; `mean` averages all of those rows, and `last` selects
the EOS row. This matches the tensor rows seen by the official model and does
not silently strip special tokens.

The C ABI exposes the same tokenizer, logits callback, and hidden-state
indices through `evo_model_encode()`, `evo_context_prefill()`, and
`evo_context_embed()`. Consequently `evo_model_layer_count()` reports `n+1`
for ESMC while preserving the existing meaning for Evo 2 and HyenaDNA.

## Numerical acceptance

The deterministic tiny fixture gates tokenizer behavior, conversion failure
cases, every hidden-state index, logits, CPU/MPS/CUDA agreement, CLI, and C ABI.
Production promotion additionally runs every canonical converted artifact on
gpu02 against outputs generated by the pinned Biohub Transformers reference.
For logits and final hidden states the required thresholds are
`max_abs <= 5e-3`, `mean_abs <= 5e-4`, and cosine similarity `>= 0.99999`.
Per-model evidence and current promotion status are recorded in
[model-size validation](model-size-validation.md).

All three canonical sizes passed that gate on gpu02. For 300M, 600M, and 6B,
the worst logits maximum absolute errors were respectively `5.34e-5`,
`4.77e-5`, and `5.53e-5`; the worst final-hidden maximum absolute errors were
`8.94e-7`, `3.28e-7`, and `1.67e-5`. Every logits and hidden cosine exceeded
`0.999999999998`. The archived gate also records the converted artifact,
official oracle, native NPY outputs, comparison JSON, GPU state, binary hash,
and recursive artifact hashes.

## A800 performance boundary

The same pinned checkpoints were measured on one idle A800 80GB. Entries are
native/official median milliseconds; the parenthesized value is native
speedup. Both sides use batch 1, F32 with TF32 disabled, identical token IDs,
two warmups, five retained samples, CUDA synchronization, and
`forward_with_host_logits`. Model load is timed separately.

| Model | Load | 128 tokens | 512 tokens | 2048 tokens |
|---|---:|---:|---:|---:|
| ESMC-300M | 241.7 / 884.9 (3.66×) | 19.23 / 28.35 (1.47×) | 37.32 / 28.29 (0.76×) | 142.01 / 52.88 (0.37×) |
| ESMC-600M | 444.6 / 1054.5 (2.37×) | 26.26 / 34.66 (1.32×) | 56.58 / 41.91 (0.74×) | 221.87 / 77.90 (0.35×) |
| ESMC-6B | 4716.4 / 5757.1 (1.22×) | 162.47 / 85.64 (0.53×) | 483.63 / 104.43 (0.22×) | 1891.57 / 490.21 (0.26×) |

Thus the native path is a latency win for short 300M/600M requests and for
loading every size, while the pinned official runtime remains the performance
choice for longer sequences and 6B forward. The native deployment advantage
does not depend on that crossover: product inference has no Python, PyTorch,
Transformers, Transformer Engine, or libtorch runtime dependency.

The maintained reproduction command is:

```sh
export HF_HOME=/build/grp_icg/users/tang/.cache/huggingface
scripts/gpu02_benchmark_esmc.sh
```

This run used source commit `6c16115`, source fingerprint
`8c08607794d452840d435a869ad9e1b8702051d3cd30ec43a98244420594dd01`,
the three revisions in the identity table, and Biohub Transformers commit
`3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf`. The native Release/CUDA 12.8
binary SHA256 was
`7ef2c8a48e566c5db84dba0a44cfb53b4e0170e41b42bfb6dc2ab8660db4b9d4`.
The official container used Python 3.12.3, PyTorch
`2.13.0a0+8145d630e8.nv26.06`, Transformers 5.1.0, CUDA runtime 13.3,
`config._attn_implementation=eager`, available Transformer Engine, and the
official PyTorch RoPE fallback. The GPU was idle at 41 MiB before and after;
driver version was 580.126.20. The complete report is archived on gpu02 at
`$HOME/evo.cpp-artifacts/esmc-performance-8c08607794d4-gpu1`; its
`artifact-sha256.txt` has SHA256
`ac7c7dc883924184f721c3cf77226fb71de8bde3da0f1f066327f82a16bbac67`.
