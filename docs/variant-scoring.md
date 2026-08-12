# Variant scoring

`evo variant-score` compares one inline DNA alternate allele with its stated
reference allele under the loaded Evo 2 model. It is intended as a small,
auditable primitive for biological tools; VCF and indexed reference-genome
batch input belong to the broader sequence-I/O work.

```sh
evo variant-score -m MODEL \
  --sequence AACCGGTT --position 3 --ref C --alt T \
  --window 6 --strand both --normalization sum \
  --ctx 8192 --gpu 0
```

## Coordinates and windows

- `--position` is a positive, 1-based coordinate into `--sequence`.
- `--ref` must exactly match `--sequence` at that position. A mismatch is an
  error and no score is emitted.
- Output window `start` and `end` use 0-based, half-open coordinates on the
  supplied reference sequence.
- The window is centered around the allele where boundaries permit. Its flank
  budget is based on the longer of the reference and alternate alleles, so
  both derived sequences fit `--window`. At contig boundaries, unused flank
  budget is moved to the other side.
- `--window` defaults to `--ctx` and cannot exceed it. Each resulting sequence
  must contain at least two byte tokens because the first token has no
  left-context likelihood.

Only IUPAC DNA symbols `ACGTRYSWKMBDHVN` (in either case) are accepted. Reverse
strand scoring uses the complete IUPAC reverse complement of each derived
window.

## Score definition

For a byte-tokenized sequence `x` of length `L`, the sum score is:

```text
LL(x) = sum(i=1..L-1) log p(x[i] | x[0..i))
delta = LL(alternate) - LL(reference)
```

`--normalization sum` reports that sum. `--normalization mean` first divides
each allele likelihood by its own `L - 1` scored-token count, then subtracts.
The distinction matters for indels, so the selected policy and both token
counts are always written to JSON.

`--strand forward` scores `+`; `reverse` scores the reverse complements as
`-`; and `both` reports both deltas and sets the top-level `score` to their
arithmetic mean. No strand is silently substituted or omitted.

The single JSON object includes the alleles, coordinate systems, derived
window sequences and lengths, normalization, strand-level likelihoods and
deltas, aggregation policy, backend, execution profile, and model identifier.
`profile: "exact"` means the registered exact runtime; approximate profiles
must use distinct names and acceptance gates.

## Current scope

This command accepts one variant and an inline reference sequence. It does not
yet parse VCF, fetch a contig from reference FASTA, normalize multiallelic
records, or left-align indels. Those operations will be added with the planned
VCF/reference I/O layer; callers must perform them explicitly for now.
