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
