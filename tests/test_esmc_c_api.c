// SPDX-License-Identifier: Apache-2.0
#include "evo/evo.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s (%s)\n", message, evo_last_error());
    ++failures;
  }
}

typedef struct capture_state {
  size_t expected_rows;
  size_t expected_columns;
  size_t calls;
} capture_state;

static evo_status capture(const float *values, size_t rows, size_t columns,
                          size_t offset, void *user_data) {
  capture_state *state = (capture_state *)user_data;
  if (values == NULL || rows != state->expected_rows ||
      columns != state->expected_columns || offset != 0 || !isfinite(values[0]))
    return EVO_STATUS_INTERNAL;
  ++state->calls;
  return EVO_STATUS_OK;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    fprintf(stderr, "usage: %s MODEL [cuda]\n", argv[0]);
    return 2;
  }
  evo_model_params params = evo_model_default_params();
  const int use_cuda = argc == 3 && strcmp(argv[2], "cuda") == 0;
  int32_t device = 0;
  params.backend = use_cuda ? EVO_BACKEND_CUDA : EVO_BACKEND_CPU;
  if (use_cuda) {
    params.devices = &device;
    params.device_count = 1;
  }
  params.flags = EVO_MODEL_FLAG_TEST_ONLY_ALLOW_SYNTHETIC;
  evo_model *model = NULL;
  check(evo_model_load(argv[1], &params, &model) == EVO_STATUS_OK &&
            model != NULL,
        "synthetic ESMC model loads with explicit test permission");
  if (model == NULL)
    return 1;
  check(strcmp(evo_model_architecture(model), "ESMCTest") == 0 &&
            evo_model_vocab_size(model) == 64 &&
            evo_model_embedding_width(model) == 4 &&
            evo_model_layer_count(model) == 3,
        "C API exposes ESMC dimensions and n+1 hidden-state indices");

  {
    static const uint8_t sequence[] = "LAG<mask>|Z?";
    static const uint32_t expected[] = {0, 4, 5, 6, 32, 31, 27, 3, 2};
    uint32_t tokens[9] = {0};
    size_t count = 0;
    check(evo_model_encode(model, sequence, sizeof(sequence) - 1, tokens, 9,
                           &count) == EVO_STATUS_OK &&
              count == 9 && memcmp(tokens, expected, sizeof(expected)) == 0,
          "C API implements exact ESMC special-token vocabulary");
    uint8_t byte = 0;
    check(evo_model_decode_token(model, 4, &byte) == EVO_STATUS_OK &&
              byte == (uint8_t)'L',
          "C API detokenizes ESMC residues");
    check(evo_model_decode_token(model, 0, &byte) == EVO_STATUS_MODEL_FORMAT,
          "C API rejects special-token detokenization");
  }

  evo_context_params context_params = evo_context_default_params();
  context_params.context_size = 16;
  evo_context *context = NULL;
  check(evo_context_create(model, &context_params, &context) == EVO_STATUS_OK &&
            context != NULL,
        "ESMC context is created");
  evo_batch *batch = NULL;
  check(evo_batch_create(1, &batch) == EVO_STATUS_OK && batch != NULL,
        "ESMC batch is created");
  if (context != NULL && batch != NULL) {
    static const uint8_t sequence[] = "LAGV";
    check(evo_batch_add_sequence(batch, sequence, sizeof(sequence) - 1) ==
              EVO_STATUS_OK,
          "protein sequence is added");
    capture_state logits = {6, 64, 0};
    check(evo_context_prefill(context, batch, capture, &logits) ==
                  EVO_STATUS_OK &&
              logits.calls == 1 && evo_context_position(context) == 6,
          "C ABI returns full ESMC logits including CLS/EOS rows");
    check(evo_context_decode(context, 4, capture, &logits) ==
              EVO_STATUS_UNSUPPORTED,
          "incremental ESMC decode is explicitly unsupported");
  }
  evo_context_free(context);

  context = NULL;
  check(evo_context_create(model, &context_params, &context) == EVO_STATUS_OK &&
            context != NULL,
        "fresh ESMC embedding context is created");
  if (context != NULL && batch != NULL) {
    capture_state embedding = {6, 4, 0};
    check(evo_context_embed(context, batch, 2, capture, &embedding) ==
                  EVO_STATUS_OK &&
              embedding.calls == 1,
          "C ABI returns official final-normalized hidden state index n");
  }
  evo_context_free(context);

  evo_batch_clear(batch);
  {
    static const uint8_t masked[] = "<mask>";
    check(evo_batch_add_sequence(batch, masked, sizeof(masked) - 1) ==
              EVO_STATUS_OK,
          "multi-byte mask token is added as raw input");
  }
  context_params.context_size = 3;
  context = NULL;
  check(evo_context_create(model, &context_params, &context) == EVO_STATUS_OK &&
            context != NULL,
        "minimum masked-token context is created");
  if (context != NULL && batch != NULL) {
    capture_state masked_logits = {3, 64, 0};
    check(evo_context_prefill(context, batch, capture, &masked_logits) ==
                  EVO_STATUS_OK &&
              masked_logits.calls == 1,
          "capacity validation uses encoded ESMC length, not source bytes");
  }
  evo_context_free(context);
  evo_batch_free(batch);
  evo_model_free(model);
  return failures == 0 ? 0 : 1;
}
