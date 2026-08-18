# Architecture registry

Model families are selected from typed artifact metadata, not from filenames
or tensor-name guesses. Each registered architecture binds an artifact
profile, runtime ABI, tokenizer, supported backends, and capabilities. An
unknown or contradictory tuple fails before a context is created.

| Architecture | Artifact profile / ABI | Tokenizer | Backends | Surfaces |
|---|---|---|---|---|
| `StripedHyena2` | `evo2-runtime-v1` / `evo2-safetensors-v1` | byte identity, 512 logits | CPU, MPS, CUDA | generate, score, embed, variant, serve, logits |
| `HyenaDNA` | `hyenadna-runtime-v1` / `hyenadna-safetensors-v1` | single-nucleotide IDs, 16 padded logits | CPU, MPS | generate, score, embed, variant, serve, logits |
| `ESMC` | `esmc-runtime-v1` / `esmc-safetensors-v1` | Biohub protein vocabulary, 64 logits | CPU, MPS, one CUDA GPU | logits, embed |
| `GenebTransformerDecoder` | `geneb-decoder-runtime-v1` / `geneb-decoder-safetensors-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebOlmoDecoder` | `geneb-olmo-runtime-v1` / `geneb-olmo-safetensors-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebEsmEncoder` | `geneb-esm-runtime-v1` / `geneb-esm-safetensors-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebBertEncoder` | `geneb-bert-runtime-v1` / `geneb-bert-safetensors-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebGpt2Decoder` | `geneb-gpt2-runtime-v1` / `geneb-gpt2-safetensors-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebDnaGptDecoder` | `geneb-dna-gpt-runtime-v1` / `geneb-dna-gpt-torch-pth-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebCustomEncoder` | `geneb-custom-encoder-runtime-v1` / `geneb-custom-encoder-safetensors-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebMambaEncoder` | `geneb-mamba-runtime-v1` / `geneb-mamba-safetensors-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebHyenaDnaDecoder` | `geneb-hyenadna-runtime-v1` / `geneb-hyenadna-safetensors-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebStripedHyenaV1` | `geneb-evo1-runtime-v1` / `geneb-evo1-safetensors-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebJanusDnaEncoder` | `geneb-janusdna-runtime-v1` / `geneb-janusdna-lightning-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |
| `GenebSequenceCnnEncoder` | `geneb-sequence-cnn-runtime-v1` / `geneb-sequence-cnn-pytorch-v1` | fixed A/C/G/T/N one-hot input | CPU | embed, serve |
| `GenebRoformerEncoder` | `geneb-roformer-runtime-v1` / `geneb-roformer-pytorch-v1` | verified `evo-tokenizer-v1` asset | CPU | embed, serve |

The checked-in model registry and release metadata carry this backend and
capability matrix in machine-readable form. A backend absent from an
architecture row has no registered factory: explicit selection fails with a
typed unsupported result before tokenizer, tensor, context, CLI embedding, or
server execution. In particular, none of the registered GENEB architectures
currently claims a CUDA or MPS kernel.

`StripedHyena2Test`, `HyenaDNATest`, and `ESMCTest` mirror these entries for generated
fixtures and require the explicit test-only model flag. They are never
accepted by production callers accidentally.

`GenebTransformerDecoder` is the shared typed execution graph for the seven
GENEB Llama/Mistral-derived decoder checkpoints. Registration proves artifact
admission and CPU embedding dispatch; all 40 GENEB v4 checkpoints have passed
their pinned real-checkpoint CPU oracle gates.
The decoder artifact may select a closed operation kernel only when the
profile requires it: BioFM alone uses
`torch-cpu-flash-bf16-portable` attention, while absent metadata means the
existing eager kernel. The Flash profile preserves F32 score reductions,
BF16 exponential scratch, F32 value accumulation, direct division, and BF16
output; it is rejected for non-BF16 or sliding-window topologies.
`GenebOlmoDecoder` is separate because Omni-DNA uses OLMo fused QKV and
x/gate SwiGLU ordering plus checkpoint-specific LayerNorm/RMSNorm topology.
`GenebEsmEncoder` is the shared typed bidirectional graph for the five
Nucleotide Transformer checkpoints and Agro-NT; absolute/rotary positions,
token dropout, GELU/SwiGLU, and tokenizer limits remain artifact-validated.
`GenebBertEncoder` retains absolute, symmetric-ALiBi, and rotary position
variants, pre/post normalization, CLS/mean pooling, and MutBERT soft-vocabulary
input as explicit artifact topology rather than inferring them from names.
`GenebGpt2Decoder` keeps Hugging Face GPT-2 Conv1D orientation, learned
absolute positions, fused QKV, and affine pre-LayerNorm semantics.
`GenebDnaGptDecoder` separately keeps the official bias-free projection,
weight-only LayerNorm, static/dynamic six-mer tokenizer, and source BF16
conversion rules. Both expose only their validated embedding surface.
`GenebCustomEncoder` has closed `lucaone` and `genomics_fm` variants; their
RoPE/absolute positions, pre/post normalization, mean/CLS pooling, and distinct
model/tokenizer vocabulary sizes are artifact-validated rather than inferred.
`GenebMambaEncoder` has closed eccDNAMamba, PlantCaduceus, and Caduceus PS/PH
variants; bidirectional scans, reverse-complement parameter sharing, and
physical versus emitted tokenizer vocabulary sizes are explicit topology.
`GenebHyenaDnaDecoder` uses FFT long convolution for the pinned 160k and 1M
checkpoints while retaining their left-padding reference semantics.
`GenebStripedHyenaV1` is a separate Evo-1 graph with its pinned Hyena and
attention layer schedule; it never aliases the Evo 2 implementation.
`GenebJanusDnaEncoder` fixes the two 72-wide JanusDNA Mamba+MoE variants,
including the optional middle attention block and doubled final hidden tap.
`GenebSequenceCnnEncoder` keeps Enformer and SPACE in one closed sequence-CNN
family while validating their different input lengths, attention/MoE
topologies, crop windows, and common 3072-wide spatial-mean output.
`GenebRoformerEncoder` is the pinned DeepGene RoFormer-only embedding path;
the separate graph stage is not admitted by this architecture/profile.

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

The CPU and MPS paths use a deterministic direct causal convolution and
precompute the implicit filters. MPS accelerates runtime linear layers while
the convolution and nonlinear/state updates stay on the host. Both accept
contexts up to 4096 tokens and report their distinct `cpu-f32` or `mps-f32`
profile; larger official configurations fail as unsupported rather than
entering an unbounded quadratic path. CUDA and hybrid placement are not
registered for HyenaDNA yet. These are functional portability profiles, not
exactness claims against PyTorch FFT arithmetic.

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
serving. Its CPU maximum F32 logit error against the NumPy oracle is
`2.3841858e-7`; the macOS arm64 MPS gate additionally compares all three model
families with CPU; the maximum absolute error gate is `1e-5` for HyenaDNA and
`1e-4` for StripedHyena2 and ESMC.

The converter was also exercised against
`LongSafari/hyenadna-tiny-1k-seqlen-hf` revision
`e8c1effa8673814e257e627d2e1eda9ea5a373f6`, whose `model.safetensors` SHA256
is `5ce2146c21e9c4baa6bddc4998fd3d029903ae84a563bf80218644082194a12d`.
That real-checkpoint run proves manifest compatibility and native execution;
it is not presented as a PyTorch numerical-equivalence gate.

Primary references: the [official HyenaDNA repository](https://github.com/HazyResearch/hyena-dna),
the [HyenaDNA paper](https://arxiv.org/abs/2306.15794), and the
[official tiny-1k Hugging Face artifact](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf).
