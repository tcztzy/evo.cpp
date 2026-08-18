# Tokenizer assets

`evo-tokenizer-v1` is the offline, architecture-independent tokenizer format
used by native runtime artifacts. Model metadata contains exactly four typed
descriptor entries:

| key | type | contract |
|---|---|---|
| `tokenizer.profile` | string | exactly `evo-tokenizer-v1` |
| `tokenizer.path` | string | canonical relative path below the model entry path's parent |
| `tokenizer.sha256` | string | lowercase 64-character SHA256 |
| `tokenizer.size` | u64 | nonzero, at most 64 MiB |

The runtime rejects partial descriptors, unknown `tokenizer.*` keys, absolute
paths, `.`/`..`, every symlink path component, size drift, and hash drift. It
checks the size and SHA256 before parsing JSON. Fetching and Python tokenizer
calls are never part of model loading.

The conversion receipt is a separate file. In addition to the four descriptor
keys it records `converter.schema=evo-tokenizer-conversion-receipt`,
`converter.version=1`, `compiler_manifest_sha256`, and
`source_receipt_contract_sha256`. The source-receipt contract digest covers a
canonical projection of `schema_version`, `kind`, and each verified file's
role/name/size/content SHA; machine-local `path` locators are excluded. Those
provenance keys are not model metadata.

Hugging Face compiler manifests normally require source `truncation` to be
null and accept only batch-longest source padding. The optional
`ignore_hf_backend_truncation_padding=true` manifest flag is a narrow audited
exception for wrappers such as pinned DNABERT-S that explicitly override both
settings on every encode call. With the flag, the compiler requires exact
right/longest-first/zero-stride truncation and matching fixed right-padding
length/token metadata before discarding that backend-only state. Missing
opt-in, unequal lengths, or any unknown shape still fails closed.

The separate
`ignore_hf_base_vocab_added_special_literals=true` option is a narrow audited
escape hatch for the pinned PlantCaduceus tokenizer: its `normalized=false`
special AddedToken aliases coexist with a `Lowercase` normalizer. That pinned
DNA-domain profile intentionally preserves its pre-literal-matching asset
contract rather than making bracketed control strings part of its accepted
input domain. The option accepts only exact aliases already in the base BPE
vocabulary, requires a nonempty supported normalizer, and rejects appended
tokens, collisions, unsupported AddedToken flags, explicit false, and
empty/no-op use. All other audited BPE special literals are protected by the
raw-input longest-match path described below.

## Canonical JSON schema

The asset is one strict JSON object with exactly these keys:

```json
{
  "format": "evo-tokenizer-v1",
  "kind": "kmer",
  "normalization": [{"op": "ascii-uppercase"}],
  "pre_tokenizer": {"kind": "none"},
  "model": {
    "k": 6,
    "stride": 1,
    "tail": "drop",
    "unknown_policy": "unk"
  },
  "post_processor": {
    "prefix_ids": [],
    "suffix_ids": [],
    "padding": {"side": "right", "pad_id": 1}
  },
  "special_tokens": {
    "unk": 0,
    "pad": 1,
    "bos": null,
    "eos": null,
    "cls": null,
    "sep": null,
    "mask": null
  },
  "vocab": [
    {"id": 0, "piece": "[UNK]"},
    {"id": 1, "piece": "[PAD]"},
    {"id": 2, "piece": "AAAAAA"}
  ]
}
```

Vocabulary IDs and pieces must both be unique uint32/string pairs. Every
special, template, padding, byte-map, and merge reference must resolve to the
vocabulary. Unknown fields and combinations fail closed.

Supported runtime kinds and model fields are:

- `character`, `single-nucleotide`, and `longest-match`:
  `unknown_policy`, `match_special_literals`.
- `kmer`: `k`, `stride`, `tail`, `unknown_policy`, with optional
  `match_special_literals`. `stride=k` is fixed non-overlapping tokenization
  and `stride=1` is overlapping tokenization. Tail policy `lookup` performs
  one whole-tail vocabulary lookup and then follows `unknown_policy`.
- `wordpiece`: `continuation_prefix`, `max_input_chars_per_word`.
- `bpe`: ordered `merges` over UTF-8 codepoint symbols, with optional
  `literal_token_ids` for every strictly validated HF special AddedToken,
  including exact `(piece,id)` aliases already present in the base vocabulary
  and tokens appended contiguously after it. Those literals are matched by
  longest raw-input match before pre-tokenization. With a nonempty supported
  normalizer, each nonliteral gap is normalized and pre-tokenized separately;
  literal bytes are never normalized. Unknown AddedToken flags, collisions,
  and unsupported normalizers still fail closed.
- `byte-bpe`: `add_prefix_space`, an exact 256-entry `byte_encoder`, and
  ordered `merges`.
- `kmer-bpe`: the k-mer fields plus ordered `merges` over the k-mer token
  stream.

`mixed` and `biotoken` sources compile to the deterministic
`longest-match` runtime kind. BPE uses a linked token list and rank heap, so
merging is bounded by `O((symbols + merges) log symbols)` rather than repeated
full-vector rescans.

Normalization is an ordered closed list. Supported operations are
`ascii-uppercase`, `ascii-lowercase`, `u-to-t`,
`strip-ascii-whitespace`, `prepend-literal`, `replace-literal`, and
`replace-byte-run`. Supported pre-tokenizers are the kind-compatible subset of
`none`, `whole-input`, `ascii-whitespace`, `hf-whitespace-ascii`, and
`split-isolated` with a literal. `hf-whitespace-ascii` preserves Hugging Face
`Whitespace` word/punctuation boundaries for every ASCII input and rejects
non-ASCII input explicitly; it never substitutes byte-wise whitespace
splitting for Unicode regex behavior.
General regex, locale-dependent Unicode normalization, ByteLevel regex mode,
tokenizer dropout, decoder-only behavior, and added-token strip/word flags are
typed unsupported at conversion instead of approximated.

Encoding is transactional. Callers can set raw-byte and token limits; the
runtime also enforces independent hard normalization, intermediate-token, and
padding bounds. Prefix/suffix templates and left/right padding are applied in a
deterministic final pass.

`vocab-text` compiler manifests support `kmer`, `longest-match`, and
`wordpiece`. The longest-match form is used for official ESM vocabularies
whose tokenizer trie greedily prefers 6-mers while retaining single-base and
literal-token rows.

An optional `audited_vocab_boundary` may be used only when a pinned full
`vocab-text` source contains a known source-only trailing suffix beyond the
model embedding table. It records the exact `source_size`, `compiled_size`,
and every contiguous `excluded_suffix` item as `(id, piece,
input_policy="reject")`. The compiler first verifies the complete source file
through its manifest size/SHA, then requires the declared suffix to cover the
entire remainder exactly before publishing the prefix. Missing rows, gaps,
reordering, changed pieces, non-reject policy, and empty/no-op boundaries fail
closed. Repository manifests currently restrict this exception to the pinned
Agro-NT-1B vocabulary, whose source-only `<eos>`/`<bos>` rows exceed the model
vocabulary and are outside the accepted input domain.
