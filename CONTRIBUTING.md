# Contributing to evo.cpp

`evo.cpp` is a biological-sequence inference runtime, not an Evo 2-only
wrapper. Contributions are welcome when they keep model identity, execution
semantics, and unsupported paths explicit. Start with [the compatibility
policy](docs/compatibility.md), [the architecture registry](docs/architectures.md),
and `SPEC.md`.

## Build and test

The baseline developer gate is a strict CPU-only C++17 build. It needs CMake
3.25 or newer, a C/C++ compiler, Python 3, threads, and a zlib runtime:

```sh
cmake -S . -B build \
  -DEVO_CUDA=OFF -DEVO_NPY=OFF \
  -DEVO_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run the sanitizer gate for changes to parsers, ownership, memory mapping,
tensor bounds, the scheduler, or the public API:

```sh
cmake -S . -B build-asan \
  -DEVO_CUDA=OFF -DEVO_NPY=OFF \
  -DEVO_SANITIZE=ON -DEVO_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-asan --parallel
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 \
  ctest --test-dir build-asan --output-on-failure
```

CUDA changes must additionally pass the relevant gpu01/gpu02 build and CTest
entrypoints documented in [the GPU environment record](docs/gpu02-environment.md).
Never treat a skipped hardware or external-checkpoint test as evidence that
the corresponding path passed.

## Change gates

| Change | Evidence required before merge |
|---|---|
| CLI, JSONL, FASTA/FASTQ/VCF, or server surface | Existing compatibility contracts plus success, malformed-input, and late-failure tests |
| C ABI | Old-client contract, symbol allowlist, ownership test, version-policy update, and install-consumer test |
| Exact Evo 2 arithmetic or conversion | Full CPU suite, relevant CUDA suite, and the affected real-checkpoint raw-bit oracle |
| CPU, hybrid, quantized, or fast profile | Independent numerical and biological acceptance report; the profile name must remain visible |
| New architecture | Registry, artifact ABI, tokenizer, converter, runtime, CLI/C API/server coverage, independent oracle, real official checkpoint, and support-boundary docs |
| Release or dependency graph | Release metadata test, installed-tree test, runtime dependency audit, checksums, and platform metadata |

Synthetic fixtures are excellent regression gates but are not evidence for a
real-checkpoint numerical or biological claim. Performance changes must follow
[the benchmark matrix](docs/benchmark-matrix.md): record source commit, binary
hash, artifact hash/revision, complete hardware/software environment, warmup,
sample count, statistic, inputs, profile, and raw report files.

## Adding a model family

Do not branch on filenames or reuse another family's tensor assumptions. A
complete adapter should:

1. assign immutable architecture, artifact-profile, and runtime-ABI names;
2. validate tensor names, shapes, dtypes, tokenizer, context limit, and
   supported backends before allocating a context;
3. implement backend-neutral model/context/batch behavior and architecture-
   aware encode/decode through the existing public surfaces;
4. add conversion from a pinned official revision with file SHA256 and no
   remote-code dependency at inference time;
5. cover generation, scoring, embeddings, variants, server isolation, and C
   ABI behavior, or reject each unsupported capability with a typed error;
6. publish both an independent small oracle and a real official-checkpoint
   acceptance record without overstating their scope; and
7. update architecture, compatibility, benchmark, artifact, and release docs.

Large checkpoints, converted artifacts, datasets, benchmark outputs, generated
Python caches, and credentials do not belong in Git. Keep only small,
license-compatible fixtures whose provenance is documented.

## Pull requests and commits

Keep commits scoped and describe the user-visible contract, not just the
implementation. Update `SPEC.md` when a change adds an invariant, completes a
task, or fixes a bug with a reusable lesson. A pull request should state the
supported and unsupported cases, commands actually run, skipped gates and why,
and links or hashes for external evidence. Do not weaken a gate to make a new
backend pass; introduce a correctly named profile and its own acceptance bound.

By contributing, you agree that your contribution is licensed under the
repository's Apache-2.0 license. Model weights and upstream code retain their
own licenses.
