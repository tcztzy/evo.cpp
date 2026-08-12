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
- A CUDA model handle owns one uploaded copy of the read-only weights plus the
  mapped artifact. Contexts share those immutable device allocations while
  owning independent streams, activation arenas, and recurrent/KV caches. A
  context retains the model state, so `evo_model_free()` may be called after
  context creation. Each context itself is mutable and single-threaded.

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

`evo_context_prefill()` accepts an opaque batch and emits logits through a
chunk callback. The callback view is borrowed and valid only during that call;
this keeps host memory bounded by the backend activation chunk. Returning a
non-OK status aborts inference and invalidates the mutable context.

The exact CUDA backend currently accepts one sequence per batch. Larger batch
objects are ABI-valid but return `EVO_STATUS_UNSUPPORTED` at execution time,
rather than silently changing semantics. CPU artifact loading and metadata are
available in CPU-only builds; CPU inference remains an explicit unsupported
operation until the native CPU backend lands.

## Installation

```sh
cmake -S . -B build -DEVO_CUDA=OFF -DEVO_NPY=OFF
cmake --build build
cmake --install build --prefix /your/prefix
```

Consumers can then use `find_package(evo CONFIG REQUIRED)` and link
`evo::evo`. The install tree includes `libevo`, the public headers, `evo`,
`evo-inspect`, and versioned CMake package metadata.
