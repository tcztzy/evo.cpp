# GENEB v4 support contract

`evo.cpp` targets the 40 evaluated checkpoints in Table 4 of
[GENEB: Why Genomic Models Are Hard to Compare](https://arxiv.org/abs/2606.04525v4).
The 13 models listed as excluded in Table 5 are outside this support matrix.
In particular, Evo 2 is not one of the 40 evaluated GENEB checkpoints.

The machine-readable source of truth is
[`configs/geneb-models.json`](../configs/geneb-models.json). It pins the paper,
the official benchmark metadata, the reference extractor revision, the dataset
revision, model acquisition identities, and the embedding decisions needed to
turn each checkpoint into one vector per input record. The catalog contains
metadata and does not contain model weights.

## Identity and support states

Every catalog row has three distinct, unique identifiers:

- `paper_name` is the spelling printed in GENEB v4 Table 4;
- `geneb_model_id` is the key used by the pinned official `model_meta.json`;
- `runtime_id` is evo.cpp's stable command-line and artifact identity.

Aliases live outside the 40-row model array and never count as additional
models. A catalog row is not, by itself, a supported runtime. The status fields
are deliberately independent:

- `runtime_support.status` describes whether native conversion, loading,
  tokenization, and embedding have passed a canonical real-checkpoint oracle;
- `benchmark_provenance.reference_status` describes whether the official
  extractor can produce a reference-eligible result in the locked environment;
- `benchmark_provenance.normalized_status` describes compatibility with the
  normalized GENEB probing protocol; and
- each backend is promoted only after its own implementation and evidence gate.

At the catalog milestone, entries remain `cataloged` or `manual-source`, and
CPU, CUDA, and MPS remain `not-promoted`. Those values must not be interpreted
as native model support. Promotion to `supported` is permitted only after a
pinned real checkpoint passes conversion, load, token, hidden-state, pooling,
and independent numerical-oracle checks.
All reference provenance is initially `blocked`: pinning an official
submission path and SHA256 proves file identity, not that a clean locked
reference extractor reproduced it.

## Pinned benchmark inputs

The catalog pins these upstream identities:

- paper: `arXiv:2606.04525v4`, Table 4;
- benchmark metadata: `darlednik/GENEB` commit
  `b54d018903e7f6b874ee45b74e275936deff4cd3`;
- reference extractors: `darlednik/GENEB` commit
  `b465d2d6a11efbbc9a22c105e34832725ce50e05`;
- dataset: `darlednik/geneb-tasks` revision
  `4edd705be573e48c585c2cf79dc320f9f43c7b04` (100 tasks in 13
  categories, declared `apache-2.0` on its official dataset card); and
- probing seeds: `13`, `17`, `42`, `123`, and `997`.

The upstream Enformer extractor refers to an undefined sequence-length name.
The reference preset applies only the hashed unblock patch in
[`configs/geneb-reference-patches/enformer-seq-length.patch`](../configs/geneb-reference-patches/enformer-seq-length.patch).
No-patch rows use the SHA256 of an empty byte string, so an absent patch remains
an explicit, verifiable decision rather than an omitted field.

## Embedding presets

`geneb-v4-reference` preserves executable upstream extractor behavior after
the cataloged unblock patch. Reference batching is input ordered, uses batches
of eight, flushes at train/test boundaries, does not fabricate rows in the
final batch, and pads only to the longest input in that batch using the
model-specific padding side.

`geneb-v4-normalized` processes every record independently for embedding
semantics. It removes known padding and batch-position contamination, including
extractors that returned only the first row of a batch. Normalized output is
protocol-compatible and must not be reported as a reproduction of an official
GENEB submission. The short alias `geneb` resolves to this normalized preset,
and output metadata records the expanded name.

Both presets explicitly record the input transform, tokenization, context and
length policy, hidden-state tap, pooling operation, special-token inclusion,
mask domain, and output width for every checkpoint. Missing oracle environment,
input, tolerance, or evidence values remain JSON `null`; the catalog validator
rejects a `supported` claim while those gates are absent.

Normalized changes are stored in `provenance.normalization_decisions`. Their
`normalization_patch_sha256` is the SHA256 of the UTF-8 encoding of
`json.dumps(decisions, sort_keys=True, separators=(",", ":"),
ensure_ascii=False)`. The validator recomputes this digest, including for an
empty decision array. The summary carries the extractor commit, reference
patch hash, normalization hash, oracle environment, and oracle input digest so
the evidence fields required by the validation interface remain auditable.

## Sources, licenses, and redistribution

Sources are immutable Hugging Face revisions or identified Google Drive and
Harvard Dataverse records. Manual acquisition stays `manual-source` and must
provide instructions instead of pretending a downloader succeeded. Each model
separately records the weight, code, tokenizer, and dataset license state, any
acceptable-use-policy requirement, and redistribution permissions.

Third-party weights are never committed, bundled in releases, or mirrored by
this project. Unknown or non-commercial licenses remain visible as such. Only
Apache-2.0 project code, license-compatible metadata, conversion logic, and
small provenance fixtures are distributed here.

## Validate the catalog

The catalog contract uses only the Python standard library and remains
compatible with Python 3.8:

```sh
python3 tools/validate_geneb_models.py \
  --catalog configs/geneb-models.json \
  --source-dir .
```

Success writes a deterministic JSON summary to standard output. Validation
checks both exact 40-element identity sets, parameter counts, immutable
upstream revisions, aliases, reference batching, source and license records,
patch file hashes, embedding decisions, status/evidence consistency, and
backend promotion claims. A schema or provenance error exits nonzero without
performing network access.

The eventual full validation command will append canonical checkpoint and
oracle evidence to the same summary. Only rows that pass that real-checkpoint
gate may move from catalog status to native runtime support.
