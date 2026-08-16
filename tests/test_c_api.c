// SPDX-License-Identifier: Apache-2.0
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "evo/evo.h"

static int failures = 0;

static void check(int condition, const char *description) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", description);
    ++failures;
  }
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
  check(EVO_ABI_VERSION_MINOR == 4u,
        "MPS enum extension increments the ABI feature version");
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
  check(evo_sampler_sample(sampler, logits, 0, &token) ==
            EVO_STATUS_INVALID_ARGUMENT,
        "sampler rejects an empty vocabulary");
  evo_sampler_free(sampler);

  evo_model_free(NULL);
  evo_context_free(NULL);
  evo_batch_free(NULL);
  evo_sampler_free(NULL);

  if (argc == 2)
    test_model_fixture(argv[1]);
  else
    check(argc == 1, "usage: test_c_api [MODEL]");

  if (failures != 0) {
    fprintf(stderr, "%d C API test(s) failed\n", failures);
    return 1;
  }
  puts("C API tests passed");
  return 0;
}
