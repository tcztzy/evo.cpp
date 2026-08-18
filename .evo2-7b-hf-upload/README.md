---
license: apache-2.0
base_model: arcinstitute/evo2_7b
datasets:
- arcinstitute/opengenome2
tags:
- evo2
- genomics
- dna
- safetensors
- evo.cpp
---

# Evo 2 7B — Safetensors for evo.cpp

This repository is a format-converted copy of the official
[Arc Institute Evo 2 7B checkpoint](https://huggingface.co/arcinstitute/evo2_7b).
The official PyTorch `.pt` checkpoint was converted to sharded Safetensors for
direct loading by [evo.cpp](https://github.com/tcztzy/evo.cpp).

evo.cpp is an independent C++17 biological-sequence inference runtime with
registered StripedHyena2 (Evo 2), HyenaDNA, ESMC, and
GenebTransformerDecoder, GenebOlmoDecoder, GenebEsmEncoder, GenebBertEncoder,
GenebGpt2Decoder, GenebDnaGptDecoder, GenebCustomEncoder, and
GenebMambaEncoder, GenebHyenaDnaDecoder, and GenebStripedHyenaV1
plus GenebJanusDnaEncoder, GenebSequenceCnnEncoder, and
GenebRoformerEncoder architectures. This model
card is scoped only to its Evo 2 artifact, which runs on CUDA devices without a
Hopper FP8 execution path. Model files are memory-mapped and loaded into the
required runtime representation without embedding or invoking Python. The 7B
checkpoint in this repository uses BF16 weights and has been validated on
NVIDIA A800 hardware.

No training or fine-tuning was performed. The BF16 model weights retain their
source values. Small normalization and RoPE tensors are exactly widened from
BF16 to F32, tensor names are normalized to the evo.cpp runtime layout, and
checkpoint-only Transformer Engine state is validated but not copied.

## Compatibility

The files use the standard Safetensors container and standard Hugging Face
shard index:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

The tensor schema and typed metadata implement the
`evo2-runtime-v1` / `evo2-safetensors-v1` profile. This is an evo.cpp runtime
checkpoint, not a Transformers checkpoint and not a drop-in replacement for
the original Vortex `.pt` file.

Safetensors-aware tools can inspect individual tensors without executing
pickle code. Inference with evo.cpp uses the native C++/CUDA runtime and does
not require Python, PyTorch, libtorch, Vortex, or Transformer Engine.

## Provenance

| Field | Value |
|---|---|
| Source repository | `arcinstitute/evo2_7b` |
| Source revision | `bda0089f92582d5baabf0f22d9fc85f3588f6b58` |
| Source file | `evo2_7b.pt` |
| Source file size | `13,766,621,200` bytes |
| Source SHA-256 | `c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4` |
| Runtime profile | `evo2-runtime-v1` |
| Runtime ABI | `evo2-safetensors-v1` |
| Tensor count | `345` |
| Tensor payload size | `13,170,533,120` bytes |
| Sharding policy | Greedy physical order, maximum 4,096 MiB tensor payload per shard |

`conversion.json` records the same provenance in machine-readable form.
`SHA256SUMS` covers every model shard and the index.

## Download and run

Download the complete repository:

```bash
hf download tcztzy/evo2-7b --local-dir evo2-7b
```

Inspect and run it with a build of evo.cpp that supports
`evo2-safetensors-v1`:

```bash
build/evo-inspect evo2-7b/model.safetensors.index.json

build/evo run \
  -m evo2-7b/model.safetensors.index.json \
  -p ACGTGCAATGCCGTTA \
  -n 128 \
  --ctx 8192 \
  --gpu 0
```

The index and all four shard files must remain in the same directory.

## Reproduce the conversion

After installing the conversion-only dependencies from evo.cpp:

```bash
scripts/convert_arc_checkpoint.sh \
  evo2_7b \
  evo2_7b.pt \
  model.safetensors
```

The converter uses bounded streaming reads and writes, validates the complete
registered source tensor manifest, and emits deterministic size-only shards.
The inference runtime reads the result directly and never loads the source
PyTorch checkpoint.

## Validation

The published artifact passes the strict evo.cpp reader checks:

- all four Safetensors files have valid, bounded, non-overlapping layouts;
- metadata is identical across shards;
- the index maps exactly 345 unique tensors and reports the exact payload size;
- the embedded source revision and source SHA-256 match the official
  checkpoint; and
- CUDA inference was compared with the official Evo 2 implementation on the
  same A800 and source checkpoint. The minimum row-wise cosine similarity was
  `0.999992595034` for a 1,024-token prefill and `0.999998681863` for
  generation; greedy generation was byte-identical.

## License and attribution

The upstream checkpoint is published under the Apache License 2.0. This
repository preserves that license and the original model and dataset
attribution. Please also read the
[official model card](https://huggingface.co/arcinstitute/evo2_7b) and the
[Evo 2 repository](https://github.com/ArcInstitute/evo2).
