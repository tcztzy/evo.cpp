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

All 40 GENEB v4 rows are now `runtime-supported` with promoted CPU backends:
every pinned checkpoint passed conversion, load, tokenization, hidden-state,
pooling, and an independent numerical-oracle gate. CUDA and MPS
remain `unsupported` because no registered GENEB architecture has either
backend factory or a verified kernel. Explicit CUDA/MPS selection therefore
returns a typed unsupported result across the C ABI, CLI embedding, and server
startup; it never falls back to CPU. Reference provenance is likewise
per-model: pinning an official submission path and SHA256 proves file identity,
not that a clean locked reference extractor reproduced it.

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

## Tokenizer assets and input transforms

GENEB artifacts do not depend on a tokenizer installed on the host. A model
artifact may reference one canonical `evo-tokenizer-v1` JSON asset through the
four typed metadata fields `tokenizer.profile`, `tokenizer.path`,
`tokenizer.sha256`, and `tokenizer.size`. The runtime requires a canonical
relative path below the artifact directory, rejects symbolic-link components,
and verifies the exact size and SHA-256 before parsing any tokenizer fields.
There is no network lookup or legacy-tokenizer fallback after a descriptor is
present.

The closed runtime schema covers character/single-nucleotide vocabularies,
fixed and overlapping k-mers, WordPiece, ordinary and byte-level BPE,
longest-match mixed/BioToken vocabularies, and k-mer+BPE composition. It stores
the ordered normalization operations, pre-tokenizer, vocabulary, merge ranks,
special IDs, post-processing template, and padding policy. Token IDs are
unsigned 32-bit values so the 115,000-entry GPT2-Gene vocabularies remain
lossless across the C++, C, CPU, CUDA, and MPS boundaries.

Compile a verified upstream tokenizer offline with:

```sh
python3 tools/convert_tokenizer_asset.py \
  --manifest tokenizer-compiler-manifest.json \
  --receipt tokenizer-source-receipt.json \
  --output tokenizer.evo.json \
  --descriptor tokenizer-descriptor.json \
  --asset-path assets/tokenizer.evo.json
```

The source receipt must carry exact size and SHA-256 values for every input.
Unsupported normalizers, pre-tokenizers, added-token flags, byte-level regex
behavior, duplicate IDs/pieces/merges, or non-representable compositions fail
conversion instead of being approximated.

Raw input processing has an equally closed order: enforce the per-record safety
cap (16 MiB, pinned by `suite.raw_safety_cap_bytes`) before append/allocation,
then case normalization, U-to-T mapping, invalid
base policy, frame trim, raw crop, and fixed padding; token truncation and token
padding happen only after tokenization. Metadata records original/effective
length, trim/crop offsets, padding, and token counts. In particular, this
preserves SPACE's center crop, Enformer's prefix crop/right padding,
GENERator's six-base frame trim, and Omni-DNA's token cap without turning those
model presets into a global rejection rule.

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

List the frozen matrix from either a source checkout or an installed data
directory:

```sh
build/Release/evo models --suite geneb
build/Release/evo models --suite geneb --json
```

Acquire a Hugging Face checkpoint at the cataloged immutable commit and write a
per-file size/SHA-256 receipt with:

```sh
python3 tools/evo_fetch.py source geneb-metagene-1 \
  --catalog configs/geneb-models.json
```

The same command returns a typed `manual-source` diagnostic, provider URL,
instructions, and required file identities for Drive/Dataverse checkpoints;
it never invents an immutable revision for those providers.

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

Canonical checkpoint evidence is generated without hand-editing catalog
status fields. The runner verifies every source-receipt file, any offline
tokenizer descriptor and asset, family-converter output metadata, and native
embedding files before writing a deterministic evidence record:

```sh
python tools/run_geneb_checkpoint_evidence.py run \
  --catalog configs/geneb-models.json \
  --model geneb-hyenadna-medium-160k \
  --source-receipt /verified/source-checkpoint.json \
  --source-endpoint https://huggingface.co \
  --tokenizer-descriptor /work/tokenizer-descriptor.json \
  --tokenizer-root /work \
  --converter tools/convert_geneb_hyenadna_checkpoint.py \
  --profiles configs/geneb-hyenadna-models.json \
  --artifact /work/model.safetensors \
  --native build/evo \
  --input /work/canonical.fa \
  --embedding-dir /work/embedding \
  --evidence /work/evidence.json \
  --backend cpu
```

The record keeps source snapshot, artifact, input, output, executable, and
environment digests and gives `cpu`, `cuda`, and `mps` separate typed results.
An omitted independent oracle is recorded as `not-run`; native success alone
cannot promote a catalog row. `catalog-update` requires both a passed CPU gate
and an actually compared independent oracle, while `catalog-validate` checks
the resulting evidence path and SHA-256 binding. Hugging Face runs require an
explicit source endpoint so a shell-level `HF_ENDPOINT` mirror cannot silently
change acquisition provenance.

Models whose catalog declares `tokenizer.asset_source=input-transform` (the
Enformer/SPACE one-hot paths) omit `--tokenizer-descriptor` and
`--tokenizer-root`. Their evidence instead binds a canonical digest of the
catalog tokenizer plus input-transform contract and requires the artifact to
contain no tokenizer-asset metadata. Supplying both representations, or
omitting a descriptor for any asset-backed model, fails closed.

The checked BioFM evidence locks the pinned AnnotationTokenizer, the GENEB
Embedder source, official Mistral CPU-BF16 SDPA, and the post-final-RMSNorm
row selected immediately before SEP. Because the clean batch path left-pads,
drops the model attention mask, and indexes a valid-count as an absolute row,
the normalized oracle runs one record at a time. Reproduce either frozen input
offline with:

```sh
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python tools/generate_geneb_biofm_upstream_oracle.py \
  --snapshot /verified/BioFM-265M-52218bbd \
  --receipt /verified/source-checkpoint.json \
  --geneb-root /verified/GENEB-b465d2d6 \
  --input configs/evidence/oracles/geneb-biofm-265m/canonical-input.fa \
  --output-dir /work/biofm-oracle
```

The runtime selects the BioFM-only portable CPU Flash-BF16 attention contract
rather than the generic eager kernel. Two independent exact-SDPA upstream
vectors freeze the model-specific gate at `max_abs=0.125`,
`mean_abs=0.005`, and `cosine>=0.99997`; these BF16 tolerances do not apply to
F32 models or other decoder profiles.

The checked SPACE evidence uses the pinned 131072-base human-species path and
the returned 896-by-3072 sequence embedding before output heads or track MoE,
followed by a spatial mean. Reproduce its independent vectors offline with:

```sh
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python tools/generate_geneb_space_upstream_oracle.py \
  --config /verified/space/config.json \
  --checkpoint /verified/space/pytorch_model.bin \
  --official-source /verified/SPACE-4cdba18 \
  --vendored-source /verified/GENEB/embedding_pipeline/utility_modules/SPACE \
  --extractor /verified/GENEB/embedding_pipeline/extractors/space.py \
  --output-dir /work/space-oracle
```

The generator locks the official numerical modules, the GENEB extractor and
vendored modules, both target-index tables, the checkpoint, and the complete
CPU package environment. It proves that the vendored module change only makes
the target-table path repository-relative. It also binds the pinned source
control flow and a final-pointwise forward hook before using
`return_only_embeddings=True`; for the extractor's batched input this is the
same `x` returned by `return_embeddings=True`, while avoiding unused heads and
track computations. Its two inputs independently cover odd symmetric padding
and odd center cropping, U-to-T conversion, and invalid-base zero vectors.
The canonical CPU gate is fixed at `max_abs=2e-5`, `mean_abs=1e-6`, and
`cosine>=0.99999999`.

The checked eccDNAMamba evidence uses the pinned 1M checkpoint, the exact
GENEB `MaskedLMOutput.hidden_states` tap, its leading `[CLS]` token, and an
attention-mask mean over the 768-wide projected bidirectional Mamba2 state.
Reproduce its two independent CPU-F32 vectors offline with:

```sh
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python tools/generate_geneb_eccdna_mamba_upstream_oracle.py \
  --snapshot /verified/eccDNAMamba-36d95a3 \
  --source-receipt /verified/source-checkpoint.json \
  --geneb-repo /verified/GENEB-b465d2d6 \
  --extractor /verified/GENEB-b465d2d6/embedding_pipeline/extractors/eccdnamamba.py \
  --model-source /verified/GENEB-b465d2d6/embedding_pipeline/utility_modules/GenAI_Lab_project/BiMambaForMaskedLM.py \
  --mamba-source /verified/mamba-ssm-2.2.4 \
  --output-dir /work/eccdna-mamba-oracle
```

The generator locks all 584 source tensors, the GENEB wrapper/extractor, and
the pinned mamba-ssm 2.2.4 implementation. For CPU it disables only the CUDA
memory-efficient path and executes the official non-fused Mamba2 path with
official PyTorch reference scan and RMSNorm math. The separate tokenizer gate
compiles the pinned manifest in two absolute roots, requires byte-identical
assets/descriptors, and checks official/native parity including literal
`[PAD]`, `[CLS]`, `[UNK]`, and `[MASK]` IDs. Its portable audit is stored at
`configs/evidence/tokenizers/geneb-eccdna-mamba-parity.json`. The native gate
uses the same fixed tolerances: `max_abs=2e-5`, `mean_abs=1e-6`, and
`cosine>=0.99999999`.

The checked DNA-GPT-0.1B-H evidence executes the exact pinned GENEB-vendored
DNAGPT model on CPU in F32. It validates the Google Drive checkpoint before a
safe weights-only load, preserves the literal `<R>` prefix, static
nonoverlapping 6-mer tokenizer (including unknown short tails), post-final-LN
hidden tap, and direct-F32 attention-mask mean. Reproduce its independent
vector offline with:

```sh
python tools/generate_geneb_dna_gpt_upstream_oracle.py \
  --source-root /verified/GENEB/embedding_pipeline/utility_modules/DNAGPT_project \
  --extractor /verified/GENEB/embedding_pipeline/extractors/dnagpt.py \
  --checkpoint /verified/dna_gpt0.1b_h.pth \
  --input configs/evidence/oracles/geneb-dna-gpt-0-1b-h/canonical-input.fa \
  --output-dir /work/dna-gpt-oracle
```

The generator locks the GENEB and upstream code revisions, every imported
vendored source file, the extractor, the complete 84-entry checkpoint state,
and the deterministic CPU environment. It imports neither the evo.cpp runtime
nor a converted artifact. The checked native comparison uses fixed tolerances
`max_abs=2e-5`, `mean_abs=1e-6`, and `cosine>=0.99999999`.

The checked NT-v2-50M-MS, NT-v2-50M-3mer-MS, and NT-v2-100M-MS evidence uses
the official pinned remote models and GENEB extractor semantics. Reproduce the
independent vectors in an isolated
Python 3.11 environment containing PyTorch 2.0.1, Transformers 4.32.0,
Tokenizers 0.13.3, Safetensors 0.3.3, and NumPy 1.24.4:

```sh
python tools/generate_geneb_nt_upstream_oracle.py \
  --snapshot /verified/81b29e5786726d891dbf929404ef20adca5b36f1 \
  --receipt /verified/source-checkpoint.json \
  --input configs/evidence/oracles/geneb-nt-v2-50m-ms/canonical-input.fa \
  --output-dir /work/nt50-oracle

python tools/generate_geneb_nt_3mer_upstream_oracle.py \
  --snapshot /verified/ff82eaf931e483feeb6bf7ecf03f1febe6b2fe76 \
  --receipt /verified/nt3-source-checkpoint.json \
  --input configs/evidence/oracles/geneb-nt-v2-50m-3mer-ms/canonical-input.fa \
  --output-dir /work/nt50-3mer-oracle

python tools/generate_geneb_nt_100m_upstream_oracle.py \
  --snapshot /verified/f34324c6fde36a4f635f0f1f06cac5d25acd6798 \
  --receipt /verified/nt100-source-checkpoint.json \
  --input configs/evidence/oracles/geneb-nt-v2-100m-ms/canonical-input.fa \
  --output-dir /work/nt100-oracle
```

The 3-mer profile pins vocab size 75, LayerNorm epsilon `1e-5`, the duplicated
equal source-config key, and exact full-storage column-major PyTorch matrices.
Its converter materializes only those audited matrices in logical row-major
order; all other non-contiguous layouts fail closed. The portable vectors,
environment/source audits, and final native comparisons are stored below the
matching `configs/evidence/oracles/geneb-nt-v2-*/` directories and
`configs/evidence/geneb-nt-v2-*.json`. They contain no checkpoint weights or
machine-local paths; converted runtime artifacts remain external.

The checked METAGENE-1 evidence loads the pinned 291-tensor F32 Llama
checkpoint with the official Transformers loader on an exact CPU-F32 host,
preserves its BPE tokenizer and EOS-only post-processor, and compares the
normalized attention-mask mean against native
`geneb-decoder-runtime-v1` output at `max_abs=1e-4`, `mean_abs=1e-5`, and
`cosine>=0.999999`.

The checked Evo-1-131k evidence uses the pinned GPU-free normalized oracle.
Its portable MHA patch replaces only the three CUDA attention layers; the 29
StripedHyena layers execute a double-accumulation BF16-rounded arithmetic
contract shared by the independent Python oracle and the native
`geneb-evo1-runtime-v1` path. The canonical gate is `max_abs=0.01`,
`mean_abs=0.0005`, and `cosine>=0.999999`, reflecting one-to-two BF16 ulps in
the final 4096-wide hidden state after 32 layers; clean-reference provenance
remains blocked because the upstream model imports CUDA-only `flash_attn`.

## Locked 100-task probe

`configs/geneb-benchmark-spec.json` is the byte-identical Apache-2.0 benchmark
specification from the pinned GENEB commit. `configs/geneb-probe-lock.json`
adds the exact Python package versions, all `LogisticRegression` arguments,
five seeds, and thread/locale environment that the upstream open-ended
requirements did not freeze. Install that environment separately from model
conversion:

```sh
python3.11 -m venv .venv-geneb
. .venv-geneb/bin/activate
python -m pip install -r requirements-geneb.txt
```

After native embedding extraction has produced a hash-receipted
`geneb-embedding-set` manifest for every train/test split, run:

```sh
python tools/run_geneb.py \
  --model geneb-metagene-1 \
  --preset geneb-v4-normalized \
  --data-dir /data/geneb-tasks \
  --embeddings /results/metagene/embedding-manifest.json \
  --output /results/normalized/METAGENE-1.json
```

The runner refuses an unpromoted model, an incomplete 100-task manifest,
changed CSV/NPY bytes, a non-F32 matrix, a package/thread lock mismatch, or a
reference preset whose catalog row is not reference-eligible. Its output keeps
`reference` and `normalized` claims in distinct evidence namespaces and records
the catalog, embedding manifest, benchmark spec, dataset revision, environment
lock, full probe arguments, and seeds. It validates all 100×3×3 task/regime/
metric cells before atomically publishing the submission.
