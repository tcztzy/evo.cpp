# Sequence and variant input

Scoring and embedding share one bounded record stream. The input path may be
an ordinary file, a gzip-compressed file, or `-` for stdin. Compression is
detected from the bytes by zlib, not from the filename suffix.

```sh
gzip -c sequences.fastq |
  evo -m MODEL --backend cpu --ctx 8192 --score -

evo embed -m MODEL --input sequences.fa.gz --output embeddings \
  --layer 24 --pooling mean --ctx 8192 --gpu 0
```

The first decompressed byte selects the format:

| First byte | Format | Contract |
|---|---|---|
| `>` | FASTA | wrapped sequence lines, strict nonempty headers and records |
| `@` | FASTQ | strict four-line records with a matching optional `+` name |
| other | raw | one byte-exact record; stdin is named `stdin` |

FASTA and FASTQ reject whitespace inside sequence lines. CRLF is normalized.
FASTQ quality bytes must be ASCII 33–126 and exactly match the sequence
length; qualities are validated but are not model input. Wrapped FASTQ is not
accepted. Each scoring JSONL or embedding manifest row includes
`input_format` (`raw`, `fasta`, or `fastq`).

The `--ctx` limit is enforced as bytes are appended, before a record callback
or inference context is created. Only one sequence record is materialized at a
time. A parse or inference error after earlier records leaves only their
already-complete JSONL rows and output files, then exits nonzero.

## VCF plus reference FASTA

Batch variant scoring accepts plain or gzip VCF and reference FASTA:

```sh
evo variant-score -m MODEL \
  --vcf cohort.vcf.gz --reference GRCh38.fa.gz \
  --window 8192 --strand both --normalization sum \
  --ctx 8192 --gpu 0
```

`--vcf -` reads VCF from stdin; `--reference` must remain a reusable file
because each allele requires bounded reference lookups. The parser requires a
VCF `#CHROM` header and at least the standard eight columns. It emits one event
per ALT in source record/allele order. Canonical and ambiguous IUPAC DNA
alleles are accepted; symbolic alleles, breakends, `*`, and empty alleles fail
explicitly. FILTER and genotype columns are preserved in the source file but
do not suppress scoring.

Reference contigs match the first whitespace-delimited token in each FASTA
header. For every allele, the runtime performs two streaming passes: the first
determines contig length and verifies REF, and the second extracts only the
boundary-adjusted scoring slice. This keeps memory at `O(window)` for an
unindexed plain or gzip reference. It is intentionally a portable correctness
path; large cohorts should use a future `.fai`/BGZF random-access adapter to
avoid repeated scans.

Every output line records:

- `record_index`, `allele_index`, VCF `id`, and source `vcf_line`;
- `contig`, `position`, and `position_coordinate_system: "VCF-1-based"`;
- a contig-relative `window` with `0-based-half-open` start/end;
- REF/ALT-derived window sequences, normalization, strand likelihoods,
  backend, execution profile, and model identity.

A REF mismatch, absent/duplicate contig, malformed later VCF line, or
unsupported allele produces a typed nonzero failure. Complete JSONL rows for
prior alleles remain valid; no partial row is emitted.
