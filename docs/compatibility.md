# Compatibility policy

Current project version: **0.1.0**. Current C ABI: **1.3.0** with shared-library
SONAME major **1**.

The project is pre-1.0, so architecture coverage and experimental profiles can
grow quickly. The following contracts are nevertheless deliberate: changes
must fail visibly or receive a versioned name rather than silently reinterpret
an existing command, artifact, or result.

## Compatibility surfaces

| Surface | Stable contract | How incompatible change is introduced |
|---|---|---|
| C ABI (`include/evo/evo.h`) | C99 header, opaque handles, ownership/free pairs, status values, existing symbols and struct prefixes | Increment ABI major and library SONAME; keep a migration note |
| Legacy CLI | `evo -m ... -p ...` and `evo -m ... --score ...`, documented short options, nonzero typed failures | New command/option first; removal or semantic change only in a project major release |
| JSONL and manifests | Existing field meaning, coordinate systems, record order, complete-line-on-error behavior | Add fields additively; rename a schema/profile for incompatible meaning |
| Artifact profiles | Tensor identity, dtype/shape rules, tokenizer and numerical interpretation of a named profile | Register a new immutable artifact profile/runtime ABI; never guess or upgrade in place |
| Execution profiles | Arithmetic/cache semantics and visible profile name | Add a new profile with independent numeric and biological gates |
| Model registry | Pinned model IDs, revisions, sizes, hashes and expected configuration | Add a new ID or explicitly version a corrected identity; cached bytes are always reverified |
| CMake package | `find_package(evo CONFIG)` and `evo::evo` within the same project major | Project major release for removal; additive targets are allowed |

C++ implementation headers and static internal targets are not an ABI surface.
Only installed public headers, the shared C library, installed commands, and
documented file/protocol formats are covered.

## C ABI versioning

`evo_abi_version()` encodes `major:8 | minor:8 | patch:16` and the header
publishes the same values as `EVO_ABI_VERSION_*`. ABI major 1 follows these
rules:

- an old binary linked against an earlier 1.x library must keep working with a
  newer 1.x library;
- a minor increment may add symbols, enum values, or fields appended after a
  versioned `struct_size`, but may not reorder fields or change ownership;
- a patch increment fixes implementation behavior without adding a required
  public surface;
- clients using a newly added symbol require at least the minor version that
  introduced it; a major mismatch is always incompatible; and
- exported symbols remain limited by the checked symbol allowlist. Exceptions
  and implementation types never cross the boundary.

The current ABI history is: 1.0 core model/context/batch/sampler ownership,
1.2 explicit execution profiles, and 1.3 architecture-aware token encoding and
decoding. ABI documentation and contract tests must change in the same commit
as the header.

## CLI and data compatibility

Existing legacy generation and scoring invocations remain valid. New command
families (`embed`, `variant-score`, and `serve`) and new flags are additive.
Defaults may not silently choose a fast/quantized profile, a different backend,
or a different GPU placement policy. Unsupported combinations return a typed
nonzero error.

Scoring and variant outputs are JSONL. Readers should ignore unknown fields,
but may rely on existing field types and meanings. Coordinates always carry an
explicit system: VCF positions are 1-based and scoring windows are 0-based,
half-open. A failed later record preserves only complete earlier rows. FASTA,
FASTQ, gzip, stdin, and reference semantics are specified in
[sequence and variant input](sequence-inputs.md).

Server routes under `/v1` retain request/response meanings within project major
0 unless an incompatible route is given a new version prefix. Metrics may gain
new labeled series; existing label meanings are not repurposed.

## Artifact and model compatibility

`evo2-runtime-v1` and `hyenadna-runtime-v1` are immutable contracts. Metadata
selects the architecture; filenames and tensor-name heuristics do not. A loader
must reject an unknown architecture/profile/ABI, conflicting metadata,
unsupported dtype or shape, and hash mismatch. Converters may evolve, but a
changed tensor or arithmetic interpretation requires a new profile name.

Exact claims apply only to the registered artifact, profile, backend, hardware
boundary, and oracle revision in the acceptance record. `cpu-f32`, hybrid, and
`fast-q8-kv` remain visibly approximate even when their regression gates pass.

## Supported build and release matrix

| Environment | Source/CI status | Prebuilt release status |
|---|---|---|
| Linux x86-64, C++17 CPU | CI contract | Included in the Linux CUDA archive |
| macOS arm64/x86-64, C++17 CPU | CI contract | Build from source; no archive yet |
| Linux x86-64, CUDA 12.8, sm80 | gpu02 exact/functional gates | `linux-x86_64-cuda12.8` archive |
| Other CUDA architectures or versions | Unsupported until separately gated | None |
| Windows, HIP, Metal, Vulkan, SYCL | Not implemented | None |

CPU compilation does not imply numerical equivalence to CUDA exact. See
[the benchmark matrix](benchmark-matrix.md) and
[release artifacts](artifact-distribution.md) for the evidence attached to
each claim.

Deprecations must be documented in release notes and remain accepted with an
actionable diagnostic for at least the next project minor release. Security or
correctness issues may require immediate rejection of unsafe input/artifacts;
such changes must identify the affected versions and migration path.
