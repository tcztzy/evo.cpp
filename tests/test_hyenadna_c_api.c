// SPDX-License-Identifier: Apache-2.0
#include "evo/evo.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s (%s)\n", message, evo_last_error());
    ++failures;
  }
}

static evo_status capture(const float *values, size_t rows, size_t columns,
                          size_t offset, void *user_data) {
  size_t *seen = (size_t *)user_data;
  if (values == NULL || rows == 0 || columns == 0 || offset != *seen)
    return EVO_STATUS_INTERNAL;
  *seen += rows;
  return EVO_STATUS_OK;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s MODEL\n", argv[0]);
    return 2;
  }
  evo_model *model = NULL;
  evo_model_params params = evo_model_default_params();
  check(evo_model_load(argv[1], &params, &model) == EVO_STATUS_OK &&
            model != NULL,
        "HyenaDNA model loads through auto backend");
  if (model == NULL)
    return 1;
  check(evo_model_backend(model) == EVO_BACKEND_CPU,
        "auto selects the registered CPU backend");
  check(strcmp(evo_model_architecture(model), "HyenaDNA") == 0 &&
            strcmp(evo_model_profile(model), "hyenadna-runtime-v1") == 0,
        "C API exposes HyenaDNA architecture and artifact profile");
  check(evo_model_vocab_size(model) == 16 &&
            evo_model_embedding_width(model) == 4 &&
            evo_model_layer_count(model) == 2,
        "C API exposes architecture-specific dimensions");

  {
    static const uint8_t sequence[] = {'A', 'C', 'G', 'T', 'N', '?'};
    static const uint32_t expected[] = {7, 8, 9, 10, 11, 6};
    uint32_t tokens[6] = {0};
    size_t count = 0;
    check(evo_model_encode(model, sequence, sizeof(sequence), NULL, 0,
                           &count) == EVO_STATUS_OK &&
              count == 6,
          "tokenizer query returns required capacity");
    check(evo_model_encode(model, sequence, sizeof(sequence), tokens, 6,
                           &count) == EVO_STATUS_OK &&
              memcmp(tokens, expected, sizeof(expected)) == 0,
          "tokenizer implements the registered HyenaDNA vocabulary");
    uint8_t byte = 0;
    check(evo_model_decode_token(model, 10, &byte) == EVO_STATUS_OK &&
              byte == (uint8_t)'T',
          "detokenizer returns raw DNA bytes");
    check(evo_model_decode_token(model, 1, &byte) == EVO_STATUS_MODEL_FORMAT,
          "special token detokenization fails explicitly");
  }

  evo_context_params context_params = evo_context_default_params();
  context_params.context_size = 16;
  evo_context *context = NULL;
  check(evo_context_create(model, &context_params, &context) == EVO_STATUS_OK &&
            context != NULL,
        "HyenaDNA context shares the loaded model");
  evo_batch *batch = NULL;
  check(evo_batch_create(1, &batch) == EVO_STATUS_OK && batch != NULL,
        "HyenaDNA batch is created");
  if (context != NULL && batch != NULL) {
    static const uint8_t sequence[] = {'A', 'C', 'G', 'T', 'N', 'A'};
    check(evo_batch_add_sequence(batch, sequence, sizeof(sequence)) ==
              EVO_STATUS_OK,
          "raw sequence is added without architecture coupling");
    size_t seen = 0;
    check(evo_context_prefill(context, batch, capture, &seen) ==
                  EVO_STATUS_OK &&
              seen == sizeof(sequence),
          "C ABI prefill tokenizes and executes HyenaDNA");
  }
  evo_batch_free(batch);
  evo_context_free(context);
  evo_model_free(model);
  return failures == 0 ? 0 : 1;
}
