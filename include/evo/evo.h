// SPDX-License-Identifier: Apache-2.0
#ifndef EVO_EVO_H
#define EVO_EVO_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(EVO_SHARED)
#if defined(EVO_BUILDING_LIBRARY)
#define EVO_API __declspec(dllexport)
#else
#define EVO_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define EVO_API __attribute__((visibility("default")))
#else
#define EVO_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define EVO_ABI_VERSION_MAJOR 1u
#define EVO_ABI_VERSION_MINOR 2u
#define EVO_ABI_VERSION_PATCH 0u
#define EVO_ABI_VERSION_ENCODE(major, minor, patch)                            \
  ((((uint32_t)(major) & 0xffu) << 24u) |                                      \
   (((uint32_t)(minor) & 0xffu) << 16u) | ((uint32_t)(patch) & 0xffffu))
#define EVO_ABI_VERSION_CURRENT                                                \
  EVO_ABI_VERSION_ENCODE(EVO_ABI_VERSION_MAJOR, EVO_ABI_VERSION_MINOR,         \
                         EVO_ABI_VERSION_PATCH)

typedef enum evo_status {
  EVO_STATUS_OK = 0,
  EVO_STATUS_INVALID_ARGUMENT = 2,
  EVO_STATUS_IO = 3,
  EVO_STATUS_MODEL_FORMAT = 4,
  EVO_STATUS_UNSUPPORTED = 5,
  EVO_STATUS_CUDA = 6,
  EVO_STATUS_INTERNAL = 70
} evo_status;

typedef enum evo_backend {
  EVO_BACKEND_AUTO = 0,
  EVO_BACKEND_CUDA = 1,
  EVO_BACKEND_CPU = 2
} evo_backend;

enum {
  // Synthetic StripedHyena2 fixtures are never accepted unless this explicit
  // test-only flag is set.
  EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC = 1u << 0u
};

enum {
  // Selects the explicitly approximate paged-Q8 KV profile. Zero retains the
  // BF16 exact profile regardless of context length.
  EVO_CONTEXT_FLAG_FAST_Q8_KV = 1u << 0u
};

typedef struct evo_model evo_model;
typedef struct evo_context evo_context;
typedef struct evo_batch evo_batch;
typedef struct evo_sampler evo_sampler;

typedef struct evo_model_params {
  size_t struct_size;
  evo_backend backend;
  const int32_t *devices;
  size_t device_count;
  uint32_t flags;
} evo_model_params;

typedef struct evo_context_params {
  size_t struct_size;
  size_t context_size;
  uint32_t flags;
} evo_context_params;

typedef struct evo_sampler_params {
  size_t struct_size;
  float temperature;
  size_t top_k;
  float top_p;
  uint64_t seed;
} evo_sampler_params;

// The view remains valid only for the duration of the callback. Returning a
// non-OK status aborts the operation and permanently invalidates that context.
typedef evo_status (*evo_logits_callback)(const float *logits, size_t rows,
                                          size_t columns, size_t token_offset,
                                          void *user_data);
typedef evo_status (*evo_embedding_callback)(const float *embedding,
                                             size_t rows, size_t columns,
                                             size_t token_offset,
                                             void *user_data);

EVO_API uint32_t evo_abi_version(void);
EVO_API const char *evo_version_string(void);
EVO_API const char *evo_status_name(evo_status status);
EVO_API const char *evo_backend_name(evo_backend backend);

// Returns a thread-local message owned by the library. It remains valid until
// the next evo API call on the same thread.
EVO_API const char *evo_last_error(void);

EVO_API evo_model_params evo_model_default_params(void);
EVO_API evo_context_params evo_context_default_params(void);
EVO_API evo_sampler_params evo_sampler_default_params(void);

EVO_API evo_status evo_model_load(const char *path,
                                  const evo_model_params *params,
                                  evo_model **model_out);
EVO_API void evo_model_free(evo_model *model);
EVO_API evo_backend evo_model_backend(const evo_model *model);
EVO_API const char *evo_model_id(const evo_model *model);
EVO_API const char *evo_model_architecture(const evo_model *model);
EVO_API const char *evo_model_profile(const evo_model *model);
EVO_API size_t evo_model_vocab_size(const evo_model *model);
EVO_API size_t evo_model_max_context(const evo_model *model);
EVO_API size_t evo_model_embedding_width(const evo_model *model);
EVO_API size_t evo_model_layer_count(const evo_model *model);

// Contexts retain the immutable model artifact, so the model handle may be
// freed after context creation. A context itself is single-threaded.
EVO_API evo_status evo_context_create(const evo_model *model,
                                      const evo_context_params *params,
                                      evo_context **context_out);
EVO_API void evo_context_free(evo_context *context);
EVO_API size_t evo_context_position(const evo_context *context);
EVO_API size_t evo_context_capacity(const evo_context *context);
EVO_API const char *evo_context_profile(const evo_context *context);
EVO_API evo_status evo_context_prefill(evo_context *context,
                                       const evo_batch *batch,
                                       evo_logits_callback callback,
                                       void *user_data);
EVO_API evo_status evo_context_decode(evo_context *context, uint32_t token,
                                      evo_logits_callback callback,
                                      void *user_data);
EVO_API evo_status evo_context_embed(evo_context *context,
                                     const evo_batch *batch, size_t layer,
                                     evo_embedding_callback callback,
                                     void *user_data);

EVO_API evo_status evo_batch_create(size_t sequence_capacity,
                                    evo_batch **batch_out);
EVO_API void evo_batch_free(evo_batch *batch);
EVO_API void evo_batch_clear(evo_batch *batch);
EVO_API evo_status evo_batch_add_sequence(evo_batch *batch,
                                          const uint8_t *sequence,
                                          size_t sequence_length);
EVO_API size_t evo_batch_sequence_count(const evo_batch *batch);
EVO_API size_t evo_batch_token_count(const evo_batch *batch);

EVO_API evo_status evo_sampler_create(const evo_sampler_params *params,
                                      evo_sampler **sampler_out);
EVO_API void evo_sampler_free(evo_sampler *sampler);
EVO_API evo_status evo_sampler_sample(evo_sampler *sampler, const float *logits,
                                      size_t logits_count, uint32_t *token_out);

#ifdef __cplusplus
}
#endif

#endif
