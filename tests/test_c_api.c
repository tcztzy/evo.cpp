// SPDX-License-Identifier: Apache-2.0
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "evo/evo.h"

static int failures = 0;

static void check(int condition, const char *description) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s (%s)\n", description, evo_last_error());
    ++failures;
  }
}

typedef struct geneb_embedding_capture {
  float values[16];
  size_t value_count;
  size_t rows;
  size_t columns;
  int saw_info;
} geneb_embedding_capture;

static evo_status capture_geneb_legacy(const float *values, size_t rows,
                                       size_t columns, size_t offset,
                                       void *user_data) {
  geneb_embedding_capture *state = (geneb_embedding_capture *)user_data;
  const size_t count = rows * columns;
  if (values == NULL || offset != state->rows || columns != 4 ||
      state->value_count + count > 16)
    return EVO_STATUS_INTERNAL;
  memcpy(state->values + state->value_count, values, count * sizeof(float));
  state->value_count += count;
  state->rows += rows;
  state->columns = columns;
  return EVO_STATUS_OK;
}

static evo_status capture_geneb_ex(
    const float *values, size_t rows, size_t columns, size_t offset,
    const evo_embedding_result_info *info, void *user_data) {
  geneb_embedding_capture *state = (geneb_embedding_capture *)user_data;
  if (values == NULL || rows != 1 || columns != 4 || offset != 0 ||
      info == NULL || info->struct_size != sizeof(*info) ||
      strcmp(info->resolved_preset, "geneb-v4-normalized") != 0 ||
      strcmp(info->hidden_tap, "last-hidden-state") != 0 ||
      strcmp(info->pooling, "attention-mask-mean") != 0 || info->layer != 1 ||
      info->original_length != 6 || info->effective_length != 6 ||
      info->crop_left != 0 || info->crop_right != 0 || info->pad_left != 0 ||
      info->pad_right != 2 || info->token_count != 4 || info->rows != 1 ||
      info->columns != 4)
    return EVO_STATUS_INTERNAL;
  memcpy(state->values, values, columns * sizeof(float));
  state->value_count = columns;
  state->rows = rows;
  state->columns = columns;
  state->saw_info = 1;
  return EVO_STATUS_OK;
}

static void test_geneb_embedding_fixture(const char *path) {
  evo_model_params params = evo_model_default_params();
  params.backend = EVO_BACKEND_CPU;
  evo_model *model = NULL;
  check(evo_model_load(path, &params, &model) == EVO_STATUS_OK && model != NULL,
        "GENEB ESM artifact loads through C API");
  if (model == NULL)
    return;
  evo_context_params context_params = evo_context_default_params();
  context_params.context_size = 4;
  evo_batch *batch = NULL;
  check(evo_batch_create(1, &batch) == EVO_STATUS_OK && batch != NULL,
        "GENEB embedding batch is created");
  static const uint8_t sequence[] = "AAAAAA";
  check(batch != NULL &&
            evo_batch_add_sequence(batch, sequence, sizeof(sequence) - 1) ==
                EVO_STATUS_OK,
        "GENEB embedding sequence is added");

  evo_context *legacy_context = NULL;
  geneb_embedding_capture legacy = {{0}, 0, 0, 0, 0};
  check(evo_context_create(model, &context_params, &legacy_context) ==
                EVO_STATUS_OK &&
            legacy_context != NULL &&
            evo_context_embed(legacy_context, batch, 1, capture_geneb_legacy,
                              &legacy) == EVO_STATUS_OK &&
            legacy.rows == 2 && legacy.columns == 4,
        "legacy C embedding captures the two non-pad rows");
  evo_context_free(legacy_context);

  evo_context *preset_context = NULL;
  geneb_embedding_capture preset = {{0}, 0, 0, 0, 0};
  evo_embedding_options options = evo_embedding_default_options();
  options.preset = "geneb";
  check(evo_context_create(model, &context_params, &preset_context) ==
                EVO_STATUS_OK &&
            preset_context != NULL &&
            evo_context_embed_ex(preset_context, batch, &options,
                                 capture_geneb_ex, &preset) == EVO_STATUS_OK &&
            preset.saw_info,
        "GENEB alias resolves normalized preset and result metadata");
  if (legacy.rows == 2 && preset.value_count == 4) {
    for (size_t column = 0; column < 4; ++column) {
      const float mean =
          (legacy.values[column] + legacy.values[4 + column]) * 0.5F;
      check(memcmp(&mean, &preset.values[column], sizeof(float)) == 0,
            "masked preset pooling matches the unpadded batch-one mean");
    }
  }
  evo_context_free(preset_context);

  preset_context = NULL;
  check(evo_context_create(model, &context_params, &preset_context) ==
                EVO_STATUS_OK &&
            preset_context != NULL,
        "fresh GENEB validation context is created");
  options.layer = 1;
  check(evo_context_embed_ex(preset_context, batch, &options,
                             capture_geneb_ex, &preset) ==
            EVO_STATUS_INVALID_ARGUMENT,
        "GENEB preset is mutually exclusive with explicit layer");
  evo_context_free(preset_context);
  evo_batch_free(batch);
  evo_model_free(model);
}

static void test_geneb_unsupported_backend(const char *backend,
                                           const char *path) {
  evo_model_params params = evo_model_default_params();
  evo_model *model = NULL;
  char description[160] = {0};
  if (strcmp(backend, "cuda") == 0)
    params.backend = EVO_BACKEND_CUDA;
  else if (strcmp(backend, "mps") == 0)
    params.backend = EVO_BACKEND_MPS;
  else {
    check(0, "GENEB unsupported-backend fixture names cuda or mps");
    return;
  }
  (void)snprintf(description, sizeof(description),
                 "GENEB %s load returns typed unsupported", backend);
  check(evo_model_load(path, &params, &model) == EVO_STATUS_UNSUPPORTED,
        description);
  (void)snprintf(description, sizeof(description),
                 "GENEB %s rejection leaves the model handle null", backend);
  check(model == NULL, description);
  (void)snprintf(description, sizeof(description),
                 "GENEB %s rejection identifies the missing factory", backend);
  check(strstr(evo_last_error(), "architecture/backend factory") != NULL,
        description);
  evo_model_free(model);
}

static void test_model_fixture(const char *path) {
  evo_model_params params = evo_model_default_params();
  evo_model *model = NULL;
  evo_context *context = NULL;
  evo_context_params context_params = evo_context_default_params();
#if !defined(EVO_TEST_HAS_MPS)
  {
    evo_model_params mps_params = evo_model_default_params();
    evo_model *mps_model = NULL;
    mps_params.backend = EVO_BACKEND_MPS;
    check(evo_model_load(path, &mps_params, &mps_model) ==
              EVO_STATUS_UNSUPPORTED,
          "MPS-disabled C library rejects the explicit MPS backend");
    check(mps_model == NULL && strstr(evo_last_error(), "without MPS") != NULL,
          "MPS-disabled C library returns an actionable diagnostic");
  }
#endif
  params.backend = EVO_BACKEND_CPU;
  check(evo_model_load(path, &params, &model) == EVO_STATUS_OK,
        "strict model artifact loads through C API");
  check(model != NULL, "successful model load returns a handle");
  if (model == NULL)
    return;
  check(evo_model_backend(model) == EVO_BACKEND_CPU,
        "model exposes its selected backend");
  check(strcmp(evo_model_id(model), "tiny-evo2") == 0,
        "model exposes a stable artifact identifier");
  check(strcmp(evo_model_profile(model), "evo2-runtime-v1") == 0,
        "model exposes the strict artifact profile");
  context_params.context_size = 8;
  check(evo_context_create(model, &context_params, &context) ==
            EVO_STATUS_MODEL_FORMAT,
        "metadata-only fixture cannot create an inference context");
  check(context == NULL, "failed context creation leaves output null");
  evo_model_free(model);
}

static void test_artifact_tokenizer_fixture(const char *path) {
  evo_model_params params = evo_model_default_params();
  evo_model *model = NULL;
  const uint8_t sequence[] = {'A', 'C', 'X'};
  uint32_t tokens[3] = {0, 0, 0};
  size_t count = 0;
  uint8_t byte = 0;
  params.backend = EVO_BACKEND_CPU;
  check(evo_model_load(path, &params, &model) == EVO_STATUS_OK && model != NULL,
        "artifact-tokenizer model loads through C API");
  if (model == NULL)
    return;
  check(evo_model_encode(model, sequence, sizeof(sequence), NULL, 0, &count) ==
                EVO_STATUS_OK &&
            count == 3,
        "artifact tokenizer C API size query returns exact token count");
  check(evo_model_encode(model, sequence, sizeof(sequence), tokens, 3,
                         &count) == EVO_STATUS_OK &&
            count == 3 && tokens[0] == UINT32_C(400) &&
            tokens[1] == UINT32_C(401) && tokens[2] == UINT32_C(402),
        "C API encodes with the verified artifact tokenizer");
  check(evo_model_decode_token(model, UINT32_C(400), &byte) ==
            EVO_STATUS_UNSUPPORTED,
        "C API does not fall back to legacy detokenization for an artifact "
        "tokenizer");
  evo_model_free(model);
}

int main(int argc, char **argv) {
  evo_model_params model_params = evo_model_default_params();
  evo_context_params context_params = evo_context_default_params();
  evo_sampler_params sampler_params = evo_sampler_default_params();
  evo_model *model = NULL;
  evo_context *context = NULL;
  evo_batch *batch = NULL;
  evo_sampler *sampler = NULL;
  uint8_t sequence[] = {'A', 'C', 'G', 'T'};
  float logits[512] = {0.0F};
  uint32_t token = 0;

  check(evo_abi_version() == EVO_ABI_VERSION_CURRENT,
        "runtime and header ABI versions match");
  check(strcmp(evo_version_string(), "0.1.0") == 0,
        "C API exposes the runtime version");
  check(strcmp(evo_status_name(EVO_STATUS_MODEL_FORMAT), "model_format") == 0,
        "C status names are stable");
  check(strcmp(evo_status_name(EVO_STATUS_MPS), "mps") == 0,
        "C MPS status name is stable");
  check(strcmp(evo_backend_name(EVO_BACKEND_CUDA), "cuda") == 0,
        "C backend names are stable");
  check(strcmp(evo_backend_name(EVO_BACKEND_MPS), "mps") == 0,
        "C MPS backend name is stable");
  check(EVO_ABI_VERSION_MINOR == 5u,
        "GENEB embedding options increment the ABI feature version");
  {
    const evo_embedding_options embedding = evo_embedding_default_options();
    check(embedding.struct_size == sizeof(embedding) &&
              embedding.preset == NULL && embedding.layer == SIZE_MAX &&
              embedding.pooling == NULL,
          "embedding default options select no implicit preset");
  }
  check(model_params.struct_size == sizeof(model_params) &&
            context_params.struct_size == sizeof(context_params) &&
            sampler_params.struct_size == sizeof(sampler_params),
        "default parameter structures carry their ABI sizes");

  model_params.struct_size = 0;
  check(evo_model_load("unused", &model_params, &model) ==
            EVO_STATUS_INVALID_ARGUMENT,
        "undersized model params are rejected");
  check(model == NULL, "failed model load clears output handle");
  model_params = evo_model_default_params();
  check(evo_model_load("evo-c-api-file-that-does-not-exist", &model_params,
                       &model) == EVO_STATUS_IO,
        "missing model returns typed IO status");
  check(strstr(evo_last_error(), "io:") != NULL,
        "C API preserves an actionable thread-local error");

  check(evo_context_create(NULL, &context_params, &context) ==
            EVO_STATUS_INVALID_ARGUMENT,
        "null model cannot create a context");
  check(context == NULL, "invalid context creation clears output handle");
  check(strcmp(evo_context_profile(NULL), "") == 0,
        "null context has no execution profile");

  check(evo_batch_create(1, &batch) == EVO_STATUS_OK && batch != NULL,
        "batch allocation succeeds");
  check(evo_batch_add_sequence(batch, sequence, sizeof(sequence)) ==
            EVO_STATUS_OK,
        "byte sequence is added to batch");
  check(evo_batch_sequence_count(batch) == 1 &&
            evo_batch_token_count(batch) == sizeof(sequence),
        "batch reports exact sequence/token counts");
  check(evo_batch_add_sequence(batch, sequence, sizeof(sequence)) ==
            EVO_STATUS_INVALID_ARGUMENT,
        "batch sequence capacity is enforced");
  evo_batch_clear(batch);
  check(evo_batch_sequence_count(batch) == 0 &&
            evo_batch_token_count(batch) == 0,
        "batch clear resets reusable storage");
  evo_batch_free(batch);

  check(evo_sampler_create(&sampler_params, &sampler) == EVO_STATUS_OK &&
            sampler != NULL,
        "sampler allocation succeeds");
  logits[37] = 2.0F;
  check(evo_sampler_sample(sampler, logits, 512, &token) == EVO_STATUS_OK &&
            token == 37,
        "greedy C sampler selects the exact maximum");
  check(evo_sampler_sample(sampler, logits, 511, &token) == EVO_STATUS_OK &&
            token == 37,
        "sampler accepts a model-defined vocabulary width");
  {
    const size_t large_vocabulary = 115000;
    float *large_logits =
        (float *)calloc(large_vocabulary, sizeof(*large_logits));
    check(large_logits != NULL, "large-vocabulary C sampler fixture allocates");
    if (large_logits != NULL) {
      large_logits[114999] = 3.0F;
      check(evo_sampler_sample(sampler, large_logits, large_vocabulary,
                               &token) == EVO_STATUS_OK &&
                token == UINT32_C(114999),
            "uint32 C ABI preserves token 114999 for vocabulary 115000");
      free(large_logits);
    }
  }
  check(evo_sampler_sample(sampler, logits, 0, &token) ==
            EVO_STATUS_INVALID_ARGUMENT,
        "sampler rejects an empty vocabulary");
  evo_sampler_free(sampler);

  evo_model_free(NULL);
  evo_context_free(NULL);
  evo_batch_free(NULL);
  evo_sampler_free(NULL);

  if (argc == 3 && strcmp(argv[1], "--tokenizer-vector") == 0)
    test_artifact_tokenizer_fixture(argv[2]);
  else if (argc == 3 && strcmp(argv[1], "--geneb-embed") == 0)
    test_geneb_embedding_fixture(argv[2]);
  else if (argc == 4 && strcmp(argv[1], "--geneb-unsupported") == 0)
    test_geneb_unsupported_backend(argv[2], argv[3]);
  else if (argc == 2)
    test_model_fixture(argv[1]);
  else
    check(argc == 1,
          "usage: test_c_api [MODEL | --tokenizer-vector MODEL | "
          "--geneb-embed MODEL | --geneb-unsupported BACKEND MODEL]");

  if (failures != 0) {
    fprintf(stderr, "%d C API test(s) failed\n", failures);
    return 1;
  }
  puts("C API tests passed");
  return 0;
}
