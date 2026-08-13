# Native bio-sequence server

`evo serve` loads one validated causal-model artifact (`StripedHyena2` or
`HyenaDNA`) and exposes generation, likelihood scoring, intermediate
embeddings, and strand-aware variant scoring over HTTP/1.1. ESMC v1 is a
logits/embedding CLI and C API path; server startup returns typed unsupported.
The production process is C++17; Python is not in the runtime dependency graph.
Requests never switch architecture, backend, or profile implicitly.

## Start the server

```sh
evo serve -m /models/evo2-7b.safetensors.index.json \
  --profile exact --ctx 8192 --gpu 0 \
  --host 127.0.0.1 --port 8080 \
  --max-queue 64 --max-batch 4 --batch-window-ms 2 \
  --max-request-bytes 1048576 \
  --max-sequence-bytes 8192 \
  --max-embedding-values 1048576
```

For CPU serving, select the backend explicitly and omit GPU-only flags:

```sh
evo serve -m /models/evo2-7b.safetensors.index.json \
  --backend cpu --ctx 8192 \
  --host 127.0.0.1 --port 8080 \
  --max-queue 64 --max-batch 4
```

The default bind address is loopback. Binding `0.0.0.0` is an explicit
operator choice; the native server does not provide TLS or authentication, so
put it behind an authenticated reverse proxy before exposing it outside a
trusted host. Only numeric IPv4 bind addresses are accepted.

The CUDA backend uses `exact` by default. `fast-q8-kv` explicitly selects the
experimental approximate paged-Q8 cache; context length never changes the
profile. The CPU backend always reports `cpu-f32`. Every inference response
reports its execution `profile`, and `/health` reports both `backend` and
`execution_profile`. See [execution profiles](execution-profiles.md) for the
numerical and biological acceptance gates. Hybrid `--gpu-layers` serving is
currently rejected because request-level GPU-prefix/CPU-suffix isolation is
not yet part of the server contract.

`--max-queue` bounds pending inference work. `--max-batch` is the maximum
number of isolated request contexts launched together after the
`--batch-window-ms` coalescing window. The exact Evo 2 kernels retain their
verified batch-one semantics: requests share immutable model weights, while
each request has a separate recurrent/convolution/attention cache and sampler.
The scheduler never fuses or reuses mutable session state across requests.

## Routes

All inference routes are `POST` requests with a JSON object and a decimal
`Content-Length`. Unknown JSON fields, duplicate keys, non-finite numbers,
chunked transfer encoding, and HTTP pipelining fail closed.

### Generate

```sh
curl -sS http://127.0.0.1:8080/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ACGT","max_tokens":32,"top_k":1,"seed":7}'
```

Accepted optional sampling fields are `temperature`, `top_k`, `top_p`, and
`seed`. The response contains both `generated` and integer `tokens`. Generated
model output can contain arbitrary bytes, so `tokens` is the authoritative
byte-exact representation (`0..255`). Prompt bytes plus `max_tokens` must fit
`--ctx`.

### Score

```sh
curl -sS http://127.0.0.1:8080/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"ACGTACGT"}'
```

The response reports total and mean log likelihood, perplexity, and one
log-likelihood value for every next-byte target. A score sequence must contain
at least two bytes.

### Embeddings

```sh
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"ACGT","layer":24,"pooling":"mean"}'
```

`pooling` is `none`, `mean`, or `last`. The response records model, layer,
pooling, shape, and a row-major flat F64 JSON array sourced from native F32
activations. `--max-embedding-values` rejects responses before inference when
their value count would exceed the configured bound.

### Variants

```sh
curl -sS http://127.0.0.1:8080/v1/variants \
  -H 'Content-Type: application/json' \
  -d '{
    "sequence":"AACCGGTT", "position":3, "ref":"C", "alt":"T",
    "window":6, "strand":"both", "normalization":"mean"
  }'
```

`position` is 1-based. Output window intervals are 0-based, half-open. Strand
is `forward`, `reverse`, or `both`; normalization is `sum` or `mean`. A
reference mismatch is a typed client error before inference.

### Health and metrics

`GET /health` returns loaded model/profile information and active resource
limits. `GET /metrics` returns Prometheus text metrics for connections, queue
rejection, batch count/items, active/peak contexts, completions, failures, and
cancellations.

## Cancellation and limits

Closing a client connection cancels its scheduled request. A client can also
send `X-Evo-Timeout-Ms: N` (`0..3600000`); expiry returns HTTP 408. Cancellation
is checked before execution and at every prefill chunk/decode/embedding
callback boundary. An in-flight CUDA kernel completes before the next boundary
can observe cancellation.

The HTTP body, sequence, context, embedding output, pending queue, simultaneous
connection, and JSON nesting sizes are independently bounded. Over-limit input
uses HTTP 413, a full queue uses 503, malformed requests use 400, and backend
runtime failures use 503. Error responses have the stable shape:

```json
{"error":{"code":"invalid_argument","message":"..."}}
```
