// SPDX-License-Identifier: Apache-2.0
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "evo/evo.h"

typedef struct capture {
  float *values;
  size_t capacity;
  size_t count;
  size_t rows;
  size_t columns;
  size_t expected_offset;
} capture;

static int failures = 0;

static void check(int condition, const char *description) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", description);
    ++failures;
  }
}

static evo_status capture_logits(const float *logits, size_t rows,
                                 size_t columns, size_t token_offset,
                                 void *user_data) {
  capture *output = (capture *)user_data;
  const size_t elements = rows * columns;
  if (logits == NULL || output == NULL ||
      token_offset != output->expected_offset + output->rows ||
      elements > output->capacity - output->count ||
      (output->columns != 0 && output->columns != columns)) {
    return EVO_STATUS_INVALID_ARGUMENT;
  }
  memcpy(output->values + output->count, logits, elements * sizeof(float));
  output->count += elements;
  output->rows += rows;
  output->columns = columns;
  return EVO_STATUS_OK;
}

static evo_status abort_logits(const float *logits, size_t rows, size_t columns,
                               size_t token_offset, void *user_data) {
  (void)logits;
  (void)rows;
  (void)columns;
  (void)token_offset;
  (void)user_data;
  return EVO_STATUS_IO;
}

static void reset_capture(capture *output, size_t expected_offset) {
  output->count = 0;
  output->rows = 0;
  output->columns = 0;
  output->expected_offset = expected_offset;
}

int main(int argc, char **argv) {
  const uint8_t sequence[] = {2, 5, 7, 3, 9, 11, 13, 17, 19};
  evo_model_params model_params = evo_model_default_params();
  evo_context_params context_params = evo_context_default_params();
  evo_model *model = NULL;
  evo_context *first = NULL;
  evo_context *second = NULL;
  evo_context *embedding_context = NULL;
  evo_batch *batch = NULL;
  capture first_capture = {0};
  capture second_capture = {0};
  capture embedding_capture = {0};
  const size_t capture_elements = 9 * 512;
  const size_t embedding_elements = 9 * 8;

  if (argc != 2) {
    fprintf(stderr, "usage: test_cuda_c_api MODEL\n");
    return 2;
  }
  first_capture.values = (float *)malloc(capture_elements * sizeof(float));
  second_capture.values = (float *)malloc(capture_elements * sizeof(float));
  embedding_capture.values =
      (float *)malloc(embedding_elements * sizeof(float));
  first_capture.capacity = capture_elements;
  second_capture.capacity = capture_elements;
  embedding_capture.capacity = embedding_elements;
  if (first_capture.values == NULL || second_capture.values == NULL ||
      embedding_capture.values == NULL) {
    fprintf(stderr, "capture allocation failed\n");
    free(first_capture.values);
    free(second_capture.values);
    free(embedding_capture.values);
    return 1;
  }

  model_params.backend = EVO_BACKEND_CUDA;
  model_params.flags = EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC;
  check(evo_model_load(argv[1], &model_params, &model) == EVO_STATUS_OK,
        "CUDA C API loads strict tiny model");
  check(model != NULL && evo_model_backend(model) == EVO_BACKEND_CUDA,
        "CUDA model handle records backend");
  check(model != NULL && evo_model_vocab_size(model) == 512,
        "CUDA model exposes vocabulary width");
  check(model != NULL && evo_model_embedding_width(model) == 8 &&
            evo_model_layer_count(model) == 50,
        "CUDA model exposes embedding topology");
  check(model != NULL &&
            strcmp(evo_model_architecture(model), "StripedHyena2Test") == 0,
        "CUDA model exposes architecture metadata");

  context_params.context_size = 12;
  if (model != NULL) {
    check(evo_context_create(model, &context_params, &first) == EVO_STATUS_OK,
          "first context loads from shared model artifact");
    check(evo_context_create(model, &context_params, &second) == EVO_STATUS_OK,
          "second context loads from shared model artifact");
    check(evo_context_create(model, &context_params, &embedding_context) ==
              EVO_STATUS_OK,
          "embedding context loads from shared model artifact");
  }
  evo_model_free(model);
  model = NULL;
  check(first != NULL && second != NULL && embedding_context != NULL,
        "three mutable contexts coexist after model handle release");

  check(evo_batch_create(1, &batch) == EVO_STATUS_OK,
        "CUDA test batch allocation succeeds");
  check(evo_batch_add_sequence(batch, sequence, sizeof(sequence)) ==
            EVO_STATUS_OK,
        "CUDA test sequence is added");
  if (first != NULL && second != NULL && batch != NULL) {
    check(evo_context_embed(embedding_context, batch, 50, capture_logits,
                            &embedding_capture) == EVO_STATUS_INVALID_ARGUMENT,
          "embedding rejects a layer outside the model");
    check(evo_context_embed(embedding_context, batch, 17, capture_logits,
                            &embedding_capture) == EVO_STATUS_OK,
          "embedding streams an intermediate layer through the C ABI");
    check(embedding_capture.rows == 9 && embedding_capture.columns == 8 &&
              embedding_capture.count == embedding_elements,
          "embedding C ABI preserves chunked row and width metadata");
    for (size_t index = 0; index < embedding_capture.count; ++index) {
      if (!isfinite(embedding_capture.values[index])) {
        check(0, "embedding C ABI returns only finite F32 values");
        break;
      }
    }
    check(evo_context_prefill(first, batch, capture_logits, &first_capture) ==
              EVO_STATUS_OK,
          "first context prefill succeeds");
    check(evo_context_prefill(second, batch, capture_logits, &second_capture) ==
              EVO_STATUS_OK,
          "second context prefill succeeds");
    check(first_capture.rows == 9 && first_capture.columns == 512 &&
              first_capture.count == capture_elements,
          "prefill streams every logit row across activation chunks");
    check(first_capture.count == second_capture.count &&
              memcmp(first_capture.values, second_capture.values,
                     first_capture.count * sizeof(float)) == 0,
          "contexts sharing a model artifact are byte deterministic");
    check(evo_context_position(first) == 9 && evo_context_position(second) == 9,
          "context positions advance independently");

    reset_capture(&first_capture, 9);
    reset_capture(&second_capture, 9);
    check(evo_context_decode(first, 9, capture_logits, &first_capture) ==
              EVO_STATUS_OK,
          "first context decode succeeds");
    check(evo_context_decode(second, 9, capture_logits, &second_capture) ==
              EVO_STATUS_OK,
          "second context decode succeeds");
    check(first_capture.rows == 1 && first_capture.columns == 512 &&
              first_capture.count == 512 &&
              memcmp(first_capture.values, second_capture.values,
                     512 * sizeof(float)) == 0,
          "independent context decode is byte deterministic");
    check(evo_context_decode(first, 9, abort_logits, NULL) == EVO_STATUS_IO,
          "callback status aborts decode with the same typed status");
    check(evo_context_decode(first, 9, capture_logits, &first_capture) ==
              EVO_STATUS_INVALID_ARGUMENT,
          "callback abort permanently invalidates the advanced context");
  }

  evo_batch_free(batch);
  evo_context_free(first);
  evo_context_free(second);
  evo_context_free(embedding_context);
  free(first_capture.values);
  free(second_capture.values);
  free(embedding_capture.values);
  if (failures != 0) {
    fprintf(stderr, "%d CUDA C API test(s) failed\n", failures);
    return 1;
  }
  puts("CUDA C API tests passed");
  return 0;
}
