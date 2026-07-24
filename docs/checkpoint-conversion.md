# Evo 2 40B checkpoint conversion

The native runtime never imports PyTorch. Conversion is a one-time offline step
that memory-maps the official PyTorch zip checkpoint, validates its complete
state dictionary, and streams tensor bytes into an `EVO2C` v1 file. The writer
uses bounded chunks and publishes the completed file atomically.

The supported 40B manifest contains exactly 537 model tensors: 400 BF16 tensors
and 137 F32 tensors, totaling 82,252,533,760 payload bytes. The official
checkpoint also contains 258 `._extra_state` `io.BytesIO` values created by
Transformer Engine. Those are the only non-tensor entries accepted. For each
of the 42 Hyena input projections, conversion strictly extracts the forward
input/weight scale, inverse scale, and 16-row amax history into another 126 F32
tensors. The resulting native container has 663 tensors. Unknown, missing,
wrong-shape, wrong-dtype, or inconsistent FP8 state aborts conversion.

Install the conversion-only dependency in a separate environment:

```sh
python3 -m venv .venv-convert
. .venv-convert/bin/activate
python3 -m pip install -r requirements-convert.txt
```

The Hugging Face checkpoint is distributed in numbered byte parts. On gpu02,
the reproducible preparation script retrieves both parts through the configured
mirror, verifies them, merges them in numeric order, and runs the converter:

```sh
scripts/gpu02_prepare_40b.sh
```

For an already merged local checkpoint, conversion itself is:

```sh
python3 tools/convert_checkpoint.py \
  --input evo2_40b.pt \
  --config configs/evo2-40b-1m.yml \
  --output evo2-40b-e4m3sw.evo2 \
  --dtype bf16
build/evo2c-inspect evo2-40b-e4m3sw.evo2 \
  --tensor blocks.0.projections.fp8_scale_inv_fwd
```

Use `--dry-run` to validate without writing 82 GB. If a checkpoint SHA256 was
verified separately, pass it through `--source-sha256`; it is recorded as model
metadata. Existing output files are never replaced unless `--force` is given.

## BioNeMo BF16 checkpoint

NVIDIA's fine-tuned NGC resource `evo2/40b-1m-fp8-bf16:1.0` is a distinct
NeMo2 distributed checkpoint intended to retain quality when its Hyena input
projections execute in BF16 on Ampere. It is not the original ARC checkpoint
with FP8 state removed.

The BioNeMo manifest maps 506 source tensors to 537 output tensors: 386 BF16
and 151 F32 tensors, totaling 82,254,368,768 payload bytes. The converter
streams one DCP tensor at a time, splits the combined MLP FC1 weight, ties the
embedding and unembedding weights, materializes the Vortex-compatible medium
and long Hyena filters in F32, and records:

```text
hyena_projection_dtype=BF16
hcm_filter_dtype=F32
config.use_fp8_input_projections=false
```

It rejects missing, duplicate, unknown, wrong-shape, or wrong-dtype data
tensors. NeMo byte entries are ignored because they are checkpoint metadata;
Transformer Engine FP8 scale tensors are neither required nor emitted.

On gpu02, the detached and reconnectable preparation entrypoint is:

```sh
scripts/gpu02_prepare_bionemo_40b.sh
```

It uses `/build/grp_icg/users/tang/.cache/bionemo`, independently of
`HF_HOME` and `HF_ENDPOINT`. The official 63,680,606,710-byte archive is
accepted only with SHA256
`544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680`.
For an already extracted checkpoint:

```sh
python3 tools/convert_bionemo_checkpoint.py \
  --input /path/to/checkpoint/weights \
  --config configs/evo2-40b-1m-bionemo-bf16.yml \
  --output evo2-40b-bionemo-bf16.evo2 \
  --source-sha256 544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680
build/evo2c-inspect evo2-40b-bionemo-bf16.evo2 \
  --tensor blocks.0.projections.weight
```

Use `--dry-run` to validate all DCP metadata and mappings without reading the
82 GB tensor payload or writing an output file.
