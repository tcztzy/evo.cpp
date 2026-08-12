// SPDX-License-Identifier: Apache-2.0
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "evo/evo.h"

static int failures = 0;
static size_t callback_rows = 0;
static size_t callback_columns = 0;
static size_t callback_offset = 0;

static void check(int condition, const char *description) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s (%s)\n", description, evo_last_error());
    ++failures;
  }
}

static evo_status capture(const float *values, size_t rows, size_t columns,
                          size_t offset, void *user_data) {
  size_t index = 0;
  (void)user_data;
  if (values == NULL || rows == 0 || columns == 0)
    return EVO_STATUS_INTERNAL;
  for (index = 0; index < rows * columns; ++index) {
    if (!isfinite(values[index]))
      return EVO_STATUS_INTERNAL;
  }
  callback_rows += rows;
  callback_columns = columns;
  callback_offset = offset;
  return EVO_STATUS_OK;
}

int main(int argc, char **argv) {
  evo_model_params model_params = evo_model_default_params();
  evo_context_params context_params = evo_context_default_params();
  evo_model *model = NULL;
  evo_context *context = NULL;
  evo_context *embedding = NULL;
  evo_batch *batch = NULL;
  const uint8_t sequence[] = {2, 5, 7, 3};
  if (argc != 2) {
    fprintf(stderr, "usage: test_cpu_c_api MODEL\n");
    return 2;
  }
  model_params.backend = EVO_BACKEND_CPU;
  model_params.flags = EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC;
  check(evo_model_load(argv[1], &model_params, &model) == EVO_STATUS_OK,
        "CPU model loads through stable C ABI");
  context_params.context_size = 12;
  if (model != NULL) {
    check(evo_context_create(model, &context_params, &context) == EVO_STATUS_OK,
          "CPU context is created");
    check(evo_context_create(model, &context_params, &embedding) ==
              EVO_STATUS_OK,
          "second CPU context shares immutable weights");
  }
  evo_model_free(model);
  check(context != NULL && strcmp(evo_context_profile(context), "cpu-f32") == 0,
        "CPU C context reports its non-exact execution profile");
  check(evo_batch_create(1, &batch) == EVO_STATUS_OK, "CPU C batch is created");
  if (batch != NULL)
    check(evo_batch_add_sequence(batch, sequence, sizeof(sequence)) ==
              EVO_STATUS_OK,
          "CPU C batch accepts byte tokens");
  if (context != NULL && batch != NULL) {
    callback_rows = 0;
    check(evo_context_prefill(context, batch, capture, NULL) == EVO_STATUS_OK &&
              callback_rows == 4 && callback_columns == 512 &&
              callback_offset == 0,
          "CPU C prefill streams complete logits");
    callback_rows = 0;
    check(evo_context_decode(context, 9, capture, NULL) == EVO_STATUS_OK &&
              callback_rows == 1 && callback_columns == 512 &&
              callback_offset == 4,
          "CPU C decode advances position");
  }
  if (embedding != NULL && batch != NULL) {
    callback_rows = 0;
    check(evo_context_embed(embedding, batch, 17, capture, NULL) ==
                  EVO_STATUS_OK &&
              callback_rows == 4 && callback_columns == 8,
          "CPU C embedding exposes intermediate rows");
  }
  evo_batch_free(batch);
  evo_context_free(context);
  evo_context_free(embedding);
  if (failures != 0) {
    fprintf(stderr, "%d CPU C API test(s) failed\n", failures);
    return 1;
  }
  puts("CPU C API tests passed");
  return 0;
}
