// SPDX-License-Identifier: Apache-2.0
#include "evo/evo.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;
static size_t seen_rows = 0;
static size_t seen_columns = 0;
static uint64_t seen_hash = UINT64_C(1469598103934665603);

static void check(int condition, const char *description) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s (%s)\n", description, evo_last_error());
    ++failures;
  }
}

static evo_status capture(const float *values, size_t rows, size_t columns,
                          size_t offset, void *user_data) {
  size_t index = 0;
  (void)offset;
  (void)user_data;
  if (values == NULL || rows == 0 || columns == 0)
    return EVO_STATUS_INTERNAL;
  for (index = 0; index < rows * columns; ++index) {
    uint32_t bits = 0;
    size_t byte = 0;
    if (!isfinite(values[index]))
      return EVO_STATUS_INTERNAL;
    memcpy(&bits, &values[index], sizeof(bits));
    for (byte = 0; byte < sizeof(bits); ++byte) {
      seen_hash ^= (bits >> (byte * 8U)) & 0xffU;
      seen_hash *= UINT64_C(1099511628211);
    }
  }
  seen_rows += rows;
  seen_columns = columns;
  return EVO_STATUS_OK;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s MODEL\n", argv[0]);
    return 2;
  }

  {
    int32_t device = 0;
    evo_model *invalid = NULL;
    evo_model_params invalid_params = evo_model_default_params();
    invalid_params.backend = EVO_BACKEND_MPS;
    invalid_params.devices = &device;
    invalid_params.device_count = 1;
    check(evo_model_load(argv[1], &invalid_params, &invalid) ==
                  EVO_STATUS_INVALID_ARGUMENT &&
              invalid == NULL,
          "MPS C ABI rejects CUDA device lists");
  }
  {
    evo_model *automatic = NULL;
    evo_model_params automatic_params = evo_model_default_params();
    automatic_params.flags = EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC;
    check(evo_model_load(argv[1], &automatic_params, &automatic) ==
                  EVO_STATUS_OK &&
              automatic != NULL &&
              evo_model_backend(automatic) == EVO_BACKEND_CPU,
          "automatic C ABI selection remains CPU when CUDA is unavailable");
    evo_model_free(automatic);
  }

  evo_model_params params = evo_model_default_params();
  params.backend = EVO_BACKEND_MPS;
  params.flags = EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC;
  evo_model *model = NULL;
  check(evo_model_load(argv[1], &params, &model) == EVO_STATUS_OK &&
            model != NULL,
        "MPS model loads through the stable C ABI");
  if (model == NULL)
    return 1;
  check(evo_model_backend(model) == EVO_BACKEND_MPS,
        "C ABI reports the explicit MPS backend");

  evo_context_params context_params = evo_context_default_params();
  context_params.context_size = 12;
  evo_context *first = NULL;
  evo_context *second = NULL;
  evo_context *third = NULL;
  check(evo_context_create(model, &context_params, &first) == EVO_STATUS_OK &&
            first != NULL,
        "first MPS context is created");
  check(evo_context_create(model, &context_params, &second) == EVO_STATUS_OK &&
            second != NULL,
        "second MPS context shares immutable weights");
  check(evo_context_create(model, &context_params, &third) == EVO_STATUS_OK &&
            third != NULL,
        "third MPS context shares immutable weights");
  check(first != NULL && strcmp(evo_context_profile(first), "mps-f32") == 0,
        "MPS C context reports its execution profile");

  static const uint8_t sequence[] = {2, 5, 7, 3};
  evo_batch *batch = NULL;
  check(evo_batch_create(1, &batch) == EVO_STATUS_OK && batch != NULL,
        "MPS C batch is created");
  if (batch != NULL)
    check(evo_batch_add_sequence(batch, sequence, sizeof(sequence)) ==
              EVO_STATUS_OK,
          "MPS C batch accepts a sequence");
  if (first != NULL && batch != NULL) {
    uint64_t first_hash = 0;
    seen_rows = 0;
    seen_hash = UINT64_C(1469598103934665603);
    check(evo_context_prefill(first, batch, capture, NULL) == EVO_STATUS_OK &&
              seen_rows == 4 && seen_columns == 512,
          "MPS C prefill returns complete finite logits");
    first_hash = seen_hash;
    if (second != NULL) {
      seen_rows = 0;
      seen_hash = UINT64_C(1469598103934665603);
      check(
          evo_context_prefill(second, batch, capture, NULL) == EVO_STATUS_OK &&
              seen_rows == 4 && seen_columns == 512 && seen_hash == first_hash,
          "shared MPS contexts produce bit-deterministic prefill logits");
    }
    seen_rows = 0;
    check(evo_context_decode(first, 9, capture, NULL) == EVO_STATUS_OK &&
              seen_rows == 1 && seen_columns == 512,
          "MPS C decode advances the shared context");
  }
  if (third != NULL && batch != NULL) {
    seen_rows = 0;
    check(evo_context_embed(third, batch, 17, capture, NULL) == EVO_STATUS_OK &&
              seen_rows == 4 && seen_columns == 8,
          "MPS C embedding returns intermediate rows");
  }

  evo_batch_free(batch);
  evo_context_free(first);
  evo_context_free(second);
  evo_context_free(third);
  evo_model_free(model);
  if (failures != 0) {
    fprintf(stderr, "%d MPS C API test(s) failed\n", failures);
    return 1;
  }
  puts("MPS C API tests passed");
  return 0;
}
