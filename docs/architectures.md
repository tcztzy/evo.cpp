# Architecture registry

Model families are selected from typed artifact metadata, not from filenames
or tensor-name guesses. Each registered architecture binds an artifact
profile, runtime ABI, tokenizer, supported backends, and capabilities. An
unknown or contradictory tuple fails before a context is created.

| Architecture | Artifact profile / ABI | Tokenizer | Backends | Surfaces |
|---|---|---|---|---|
| `StripedHyena2` | `evo2-runtime-v1` / `evo2-safetensors-v1` | byte identity, 512 logits | CPU, CUDA | generate, score, embed, variant, serve |
| `HyenaDNA` | `hyenadna-runtime-v1` / `hyenadna-safetensors-v1` | single-nucleotide IDs, 16 padded logits | CPU | generate, score, embed, variant, serve |
| `ESMC` | `esmc-runtime-v1` / `esmc-safetensors-v1` | Biohub protein vocabulary, 64 logits | CPU, one CUDA GPU | logits, embed |

`StripedHyena2Test`, `HyenaDNATest`, and `ESMCTest` mirror these entries for generated
fixtures and require the explicit test-only model flag. They are never
accepted by production callers accidentally.

The ESMC adapter covers Biohub's registered 300M, 600M, and 6B F32
bidirectional masked language models. It does not inherit causal generation,
score, variant, KV-cache, server, or multi-GPU semantics from Evo 2. Its exact
source identities, tokenizer, conversion, hidden-state indexing, and numerical
gates are documented in [native ESMC inference](esmc.md).

## HyenaDNA support boundary

The HyenaDNA adapter targets the official Hugging Face
`HyenaDNAForCausalLM` F32 layout: character embeddings, post-residual
LayerNorm blocks, order-2 Hyena operators, two-layer GELU MLPs, final
LayerNorm, and an untied LM head. It is based on the official HyenaDNA code and
single-nucleotide causal-model contract. The project does not vendor or load
remote Python model code at inference time.

The current CPU kernel uses a deterministic direct causal convolution and
precomputes the implicit filters. It accepts contexts up to 4096 tokens and
reports `cpu-f32`; larger official configurations fail as unsupported rather
than entering an unbounded quadratic path. CUDA and hybrid placement are not
registered for HyenaDNA yet. This is a functional portability profile, not an
exactness claim against PyTorch FFT arithmetic.

Raw sequence mapping is:

```text
A=7 C=8 G=9 T=10 N=11 other-byte=[UNK]=6
```

IDs 0–5 and 12–15 are special/padded logits. Raw CLI, server, and
`evo_model_encode()` input does not append `[SEP]`; causal boundaries stay
caller-controlled. Generation returns nucleotide bytes for IDs 7–11 and
fails explicitly if sampling selects a special/padded ID. The integer token
array remains authoritative in the server response.

## Acceptance evidence

The checked-in independent full-sequence fixture covers converter manifest
validation, artifact profile dispatch, tokenization, direct convolution,
logits, cached decode, embeddings, variant scoring, the C ABI, and native
serving. Its maximum F32 logit error against the NumPy oracle is
`2.3841858e-7`.

The converter was also exercised against
`LongSafari/hyenadna-tiny-1k-seqlen-hf` revision
`e8c1effa8673814e257e627d2e1eda9ea5a373f6`, whose `model.safetensors` SHA256
is `5ce2146c21e9c4baa6bddc4998fd3d029903ae84a563bf80218644082194a12d`.
That real-checkpoint run proves manifest compatibility and native execution;
it is not presented as a PyTorch numerical-equivalence gate.

Primary references: the [official HyenaDNA repository](https://github.com/HazyResearch/hyena-dna),
the [HyenaDNA paper](https://arxiv.org/abs/2306.15794), and the
[official tiny-1k Hugging Face artifact](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf).
