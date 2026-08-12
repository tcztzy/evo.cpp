# Execution profiles and acceptance gates

An artifact profile describes the model file and its registered tensor
contract. An execution profile describes the runtime arithmetic and cache
layout selected for a context. They are independent: loading the same artifact
with a different execution profile does not change its weights or identity.

The CLI, native server, C API, and metrics expose the selected execution
profile. Context length never selects one implicitly.

## `exact` (default)

```sh
evo -m /models/evo2-7b.safetensors.index.json \
  --profile exact --ctx 8192 -p ACGT -n 32 --gpu 0
```

`exact` preserves the checkpoint's pinned BF16 or software-E4M3 arithmetic and
uses contiguous BF16 K/V storage at every supported context length. If the
requested exact context does not fit, creation fails with a typed error; the
runtime does not substitute a quantized cache. Omitting `--profile` selects
`exact`, preserving the raw-bit behavior of commands written before execution
profiles were introduced.

## `fast-q8-kv` (experimental, approximate)

```sh
evo -m /models/evo2-7b.safetensors.index.json \
  --profile fast-q8-kv --ctx 1048576 -p ACGT -n 32 --gpu 0
```

`fast-q8-kv` uses the same model artifact and weights, but stores K and V in
fixed 16384-token pages. Each `(token, head)` vector is independently
quantized as `scale=max(abs(vector))/127` and
`q=clamp(round(vector/scale),-127,127)`. Attention dequantizes into F32 and
uses an online softmax. This profile is not bit-equivalent to the pinned
Vortex/PyTorch execution and is always reported as `fast-q8-kv`.

The checked-in synthetic CUDA fixture must pass these plumbing gates against
`exact`:

- minimum per-row logit cosine: `0.9999`
- maximum mean absolute logit error: `0.01`
- top-1 agreement: `1.0`
- within-task biological rank agreement: `1.0`
- variant-effect sign agreement: `1.0`

The current fixture records minimum cosine `0.9999901512`, logit MAE
`0.001802208`, maximum absolute error `0.0078125`, and full top-1, rank, and
variant-sign agreement. These values verify the comparison pipeline and catch
runtime regressions; a synthetic fixture is not an accuracy claim for an
official model release. Release claims require the real-checkpoint evidence
described in the model-size validation records.

## Acceptance tool

`evo-profile-gate` (or `tools/evo_profile_gate.py` in a source checkout)
compares rank-2 little-endian F32 NPY logits and a small biological benchmark:

```sh
evo-profile-gate \
  --exact-logits exact.npy --candidate-logits fast.npy \
  --exact-bio exact-bio.json --candidate-bio fast-bio.json \
  --report profile-report.json \
  --min-cosine 0.9999 --max-logit-mae 0.01 \
  --min-top1-agreement 1 --min-rank-agreement 1 \
  --min-variant-sign-agreement 1
```

Biological input uses schema version 1 with a declared `profile` and records
containing unique `id`, `task`, and finite numeric `score` fields. Exact and
candidate files must contain the same records. The tool checks record counts,
within-task pairwise ranking, and the sign of records whose task is `variant`,
writes a machine-readable JSON report, and exits nonzero on malformed input or
a failed threshold.
