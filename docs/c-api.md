# C API and embedding contract

`include/evo/evo.h` is the stable, backend-neutral embedding surface. It is a
pure C99 header and is also usable from C++, Rust FFI, Python extensions, R,
and other languages with a C foreign-function interface.

## ABI policy

- `evo_abi_version()` must equal `EVO_ABI_VERSION_CURRENT` for major-version
  compatibility. The encoded layout is `major:8 | minor:8 | patch:16`.
- All public objects are opaque handles. Every owning create/load call has a
  matching null-safe `*_free` function.
- Parameter structs begin with `struct_size`. Callers should initialize them
  with the corresponding `*_default_params()` function. New fields may only be
  appended within ABI major 1.
- Calls return `evo_status`; `evo_last_error()` provides a thread-local,
  human-readable diagnostic. Exceptions never cross the C boundary.
- ABI 1.2 adds explicit execution profiles. A zero-initialized context remains
  exact; new callers can opt into the approximate Q8 cache without changing
  model loading or artifact identity.
- ABI 1.3 adds architecture-aware `evo_model_encode()` and
  `evo_model_decode_token()`. Existing batch ownership/layout is unchanged;
  raw sequences are tokenized by the context's registered architecture.
- A CUDA model handle owns one uploaded copy of the read-only weights plus the
  mapped artifact. Contexts share those immutable device allocations while
  owning independent streams, activation arenas, and recurrent/KV caches. A
  context retains the model state, so `evo_model_free()` may be called after
  context creation. Each context itself is mutable and single-threaded.
- A CPU model handle shares the mmap-backed artifact directly. CPU contexts
  allocate only mutable caches and report `cpu-f32`; AVX2/FMA, NEON, and scalar
  kernels have the same public callback and ownership contract.

## Minimal use

```c
#include <evo/evo.h>

evo_model_params mp = evo_model_default_params();
evo_model *model = NULL;
if (evo_model_load("evo2.safetensors", &mp, &model) != EVO_STATUS_OK) {
    fprintf(stderr, "%s\n", evo_last_error());
    return 1;
}

evo_context_params cp = evo_context_default_params();
cp.context_size = 8192;
evo_context *ctx = NULL;
if (evo_context_create(model, &cp, &ctx) != EVO_STATUS_OK) {
    fprintf(stderr, "%s\n", evo_last_error());
}
evo_model_free(model);
evo_context_free(ctx);
```

`cp.flags = EVO_CONTEXT_FLAG_FAST_Q8_KV` explicitly selects the experimental
`fast-q8-kv` execution profile. With zero flags, every context is `exact`
regardless of `context_size`; insufficient memory is an error rather than an
implicit profile change. `evo_context_profile()` returns the selected execution
profile. A model loaded with `EVO_BACKEND_CPU` accepts zero context flags and
reports `cpu-f32`. By contrast, `evo_model_profile()` returns artifact
metadata.

Call `evo_model_encode()` first with a null output buffer to query token count,
then with caller-owned `uint32_t` storage. `evo_model_decode_token()` converts
one sampled token back to a raw sequence byte or returns a typed error for a
special token. These calls remove the former assumption that every model uses
Evo 2's byte-identity vocabulary.

`evo_context_prefill()` accepts an opaque batch and emits logits through a
chunk callback. The callback view is borrowed and valid only during that call;
this keeps host memory bounded by the backend activation chunk. Returning a
non-OK status aborts inference and invalidates the mutable context.

`evo_context_embed()` uses the same bounded callback model for the F32 block
output of a zero-based intermediate layer. Query
`evo_model_embedding_width()` and `evo_model_layer_count()` before creating
downstream tensor storage. Pooling is deliberately caller-controlled in the C
ABI; the CLI supplies exact `none`, `mean`, and `last` policies.

For ESMC, prefill returns one `[encoded_tokens, 64]` masked-LM logit matrix in
a single callback and embedding returns one full bidirectional hidden matrix.
`evo_model_layer_count()` is `num_layers + 1`: index 0 is the token embedding,
indices `1..n-1` are preceding block outputs, and index `n` is the final
layer-normalized representation. Incremental `evo_context_decode()` is typed
unsupported. Context capacity includes ESMC's automatically added CLS/EOS.

Current CPU and CUDA backends accept one sequence per batch. Larger batch
objects are ABI-valid but return `EVO_STATUS_UNSUPPORTED` at execution time,
rather than silently changing semantics. CPU inference is available in
CPU-only builds and shares mmap-backed weights across isolated contexts.
CPU+GPU layer placement is a CLI policy; the stable C ABI requires
the caller to choose one backend per model handle.

## Installation

```sh
cmake -S . -B build -DEVO_CUDA=OFF -DEVO_NPY=OFF
cmake --build build
cmake --install build --prefix /your/prefix
```

Consumers can then use `find_package(evo CONFIG REQUIRED)` and link
`evo::evo`. The install tree includes `libevo`, the public headers, `evo`,
`evo-inspect`, and versioned CMake package metadata.
