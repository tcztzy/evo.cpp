# Intermediate embeddings

The embedding command captures the F32 block output of a zero-based model
layer for each FASTA, FASTQ, or raw sequence record. Plain files, gzip files,
and stdin (`--input -`) share the same bounded stream:

```sh
evo embed -m evo2.safetensors.index.json \
  --input sequences.fa \
  --output embeddings \
  --layer 24 \
  --pooling mean \
  --ctx 8192 \
  --gpu 0
```

For ESMC, `layer` follows Biohub Transformers hidden-state indexing instead of
the recurrent-model block-output convention: `0` is the token embedding,
`1..n-1` are outputs of blocks `0..n-2`, and `n` is the final layer-normalized
representation. Its manifest therefore records `point=official_hidden_state`.
ESMC token rows include the automatically added CLS and EOS. `mean` includes
both; `last` selects EOS. See [native ESMC inference](esmc.md).

The output directory must be new or empty. It receives one deterministic file
per input record (`000000.npy`, `000001.npy`, ...) and an
`embeddings.jsonl` manifest. The same complete metadata lines are written to
stdout. Record names never become paths.

Each manifest row records:

- input `record_index`, `name`, `input_format`, and `source_tokens`;
- NPY `file`, two-dimensional `shape`, and `dtype=float32`;
- zero-based `layer` and architecture-specific `point`;
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

That chunking behavior applies to causal Evo 2/HyenaDNA paths. ESMC is
bidirectional and requires the entire encoded sequence in one forward pass; it
rejects a record that cannot fit one context rather than chunking it.

If a later FASTA record or inference operation fails, the command exits
nonzero. Previously completed NPY files and manifest lines remain valid;
incomplete NPY files are removed. Existing nonempty output directories are
rejected to prevent accidental overwrites.

FASTQ quality is validated but is not part of the embedding. Full format,
gzip, stdin, and late-error semantics are documented in
[sequence and variant input](sequence-inputs.md).

For embedding applications, use `evo_context_embed()` from the C ABI described
in [c-api.md](c-api.md). Its callback receives chunk rows, hidden width, and
the record-relative token offset without imposing a pooling policy.
