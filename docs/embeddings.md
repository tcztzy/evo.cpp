# Intermediate embeddings

The embedding command captures the F32 block output of a zero-based Evo 2
layer for each FASTA or raw sequence record:

```sh
evo embed -m evo2.safetensors.index.json \
  --input sequences.fa \
  --output embeddings \
  --layer 24 \
  --pooling mean \
  --ctx 8192 \
  --gpu 0
```

The output directory must be new or empty. It receives one deterministic file
per input record (`000000.npy`, `000001.npy`, ...) and an
`embeddings.jsonl` manifest. The same complete metadata lines are written to
stdout. Record names never become paths.

Each manifest row records:

- input `record_index`, `name`, and `source_tokens`;
- NPY `file`, two-dimensional `shape`, and `dtype=float32`;
- zero-based `layer` and `point=block_output`;
- explicit `pooling`, `backend`, model `profile`, and `model_id` when present.

Pooling has exact, simple semantics:

- `none`: shape is `[source_tokens, hidden_width]`;
- `mean`: arithmetic mean of all token rows, shape `[1, hidden_width]`;
- `last`: final token row, shape `[1, hidden_width]`.

Token-level output is appended directly to a row-major little-endian NPY v1
file as activation chunks finish; it does not retain the full record matrix in
host memory. `mean` uses an F64 accumulator of `O(hidden_width)` and writes an
F32 result, while `last` retains one F32 row. Long records continue through the
same exact recurrent/KV state as scoring and generation.

If a later FASTA record or inference operation fails, the command exits
nonzero. Previously completed NPY files and manifest lines remain valid;
incomplete NPY files are removed. Existing nonempty output directories are
rejected to prevent accidental overwrites.

For embedding applications, use `evo_context_embed()` from the C ABI described
in [c-api.md](c-api.md). Its callback receives chunk rows, hidden width, and
the record-relative token offset without imposing a pooling policy.
