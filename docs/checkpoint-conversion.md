# Evo 2 checkpoint conversion

The native runtime never imports PyTorch. Conversion is a one-time offline
step that memory-maps an official checkpoint, validates its complete tensor
and extra-state manifest, and atomically streams an `EVO2C` v1 file.

## Pinned authoritative sources

Architecture values come from
`arcinstitute/evo2@53f195997257c56c00e5ef8d33a54f5baad143a6`.
Tensor names and Transformer Engine payload handling come from
`evo-design/vortex@8b00afebeac745d1f31e7e2788f0e0e39fa47637`
and the corresponding `vtx==1.1.0` layout. The machine-readable source of
truth is [`configs/model-registry.json`](../configs/model-registry.json).

| Model ID | Hugging Face repository | Pinned revision |
|---|---|---|
| `evo2_1b_base` | `arcinstitute/evo2_1b_base` | `2279e1df422c991037470302360edd40d0d2ea1e` |
| `evo2_7b` | `arcinstitute/evo2_7b` | `bda0089f92582d5baabf0f22d9fc85f3588f6b58` |
| `evo2_7b_base` | `arcinstitute/evo2_7b_base` | `074097e9dc788e8bfe045d6495b9f6153a7c6bfc` |
| `evo2_7b_262k` | `arcinstitute/evo2_7b_262k` | `27c4f055e8b729a27ab5c3e6c8b9c35367a76fba` |
| `evo2_20b` | `arcinstitute/evo2_20b` | `8b0f0a9a70c66367ed181a17d049b95699a28fed` |
| `evo2_40b` | `arcinstitute/evo2_40b` | `d529aa57c30771814217ad89baaeaf6e2315c7d7` |
| `evo2_40b_base` | `arcinstitute/evo2_40b_base` | `a91f87a93194a23d553186b11e11c55d66456d55` |

The registry also pins every checkpoint filename, byte size, published SHA256,
extra-state count, source dtype, output dtype count, and payload byte count.
Converters do not infer a topology from a filename.

## Precision and manifest rules

| Family | Native projection | Checkpoint projection weight | Norm / RoPE source |
|---|---|---|---|
| 1B | software E4M3 with fixed TE 2.3 scales | BF16 | BF16 / BF16 |
| all 7B | BF16; no FP8 metadata is applied | BF16 | BF16 / BF16 |
| 20B | software E4M3 with fixed TE 2.3 scales | F32 | BF16 / F32 |
| Arc 40B | software E4M3 with fixed TE 2.3 scales | BF16 | F32 / F32 |
| BioNeMo 40B | BF16 | BF16 | exporter-normalized F32 |

Small BF16 norm and `inv_freq` vectors are explicitly and exactly widened to
F32 because the native kernels consume F32 scale/frequency vectors. Large
weights remain in checkpoint-native BF16 or F32. In particular, the 20B F32
projection weight is stored as F32 and quantized directly to E4M3 while loading;
it is never silently downcast to BF16.

The official 7B 1M and 262K checkpoints contain one deterministic F32
`blocks.<HCL>.mixer.mixer.filter.t` grid per HCL layer. Conversion validates
its exact name, dtype, and `(1,1,max_seqlen)` shape, then omits it because the
runtime reconstructs the grid. Every other unknown data or non-tensor entry
fails conversion.

## Arc conversion

Install the conversion-only dependency in a separate environment:

```sh
python3 -m venv .venv-convert
. .venv-convert/bin/activate
python3 -m pip install -r requirements-convert.txt
```

Use the registry-aware wrapper:

```sh
scripts/convert_arc_checkpoint.sh evo2_1b_base evo2_1b_base.pt evo2-1b.evo2
scripts/convert_arc_checkpoint.sh evo2_7b evo2_7b.pt evo2-7b.evo2
scripts/convert_arc_checkpoint.sh evo2_20b evo2_20b.pt evo2-20b.evo2
scripts/convert_arc_checkpoint.sh evo2_40b evo2_40b.pt evo2-40b.evo2
```

Base/262K IDs work identically. Set `EVO2C_SOURCE_SHA256` to record a
separately verified merged checkpoint hash, or `EVO2C_DRY_RUN=1` to validate
without writing the output. The direct equivalent is:

```sh
PYTHONPATH=tools python3 tools/convert_checkpoint.py \
  --input CHECKPOINT.pt \
  --config configs/REGISTERED_CONFIG.yml \
  --output MODEL.evo2 \
  --source-sha256 CHECKPOINT_SHA256
```

Split 40B checkpoints must be merged in numeric `.part0`, `.part1` order
before conversion. Passing a part directly is rejected. The existing
reconnectable 40B preparation entrypoint remains:

```sh
scripts/gpu02_prepare_40b.sh
```

## BioNeMo BF16 40B

NVIDIA's `evo2/40b-1m-fp8-bf16:1.0` is a separately fine-tuned NeMo2
distributed checkpoint, not the Arc checkpoint with FP8 state deleted.
Its mapping is intentionally limited to
`evo2_40b_bionemo_bf16`; other model IDs fail before DCP payload reads.

```sh
scripts/gpu02_prepare_bionemo_40b.sh

PYTHONPATH=tools python3 tools/convert_bionemo_checkpoint.py \
  --input /path/to/checkpoint/weights \
  --config configs/evo2-40b-1m-bionemo-bf16.yml \
  --output evo2-40b-bionemo-bf16.evo2 \
  --source-sha256 544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680
```

The converter validates the DCP mapping, splits combined FC1, ties embedding
and unembedding, and derives the Vortex-compatible HCM/HCL F32 tensors.

## Reproducible native smoke/alignment commands

After conversion, use the same command for every size; choose a GPU list that
fits the model. The loader checks `model.id`, the exact official signature,
precision metadata, tensor dtypes, and maximum context before allocation.

```sh
scripts/validate_model.sh evo2-1b.evo2 0 8192
scripts/validate_model.sh evo2-7b.evo2 0 8192
scripts/validate_model.sh evo2-20b.evo2 0,1 8192
scripts/validate_model.sh evo2-40b.evo2 0,1 8192
```

For a reference logit matrix, add
`EVO2C_EXPECTED_LOGITS=/path/reference.npy`; the script invokes the repository
comparator. Record the checkpoint/container SHA, GPU, driver, CUDA version,
peak memory, and comparator output. As of 2026-07-26, real native GPU
validation exists for Arc 40B 1M and BioNeMo 40B only. Other rows are
structurally tested but remain unverified until their checkpoints and suitable
CUDA hardware are available.
