# Sprint 288 - TP/EP Diagnostic Completions Endpoint

Date: 2026-05-23

## Goal

Add a serving-shaped, OpenAI-compatible diagnostic completions endpoint to the
TP/EP HTTP harness while preserving the coalesced and bucketed admission policy
from Sprints 286-287.

This remains TP/EP-only. The frozen PP/layer-split path is not modified.

## Context

Sprint 287 made mixed concurrent selected-token requests admissible by holding
different `max_tokens` buckets and running one resident decode batch per
bucket. The next operational gap is API shape: clients and load tools expect a
completion-style endpoint, but the TP/EP path still does not have prompt
prefill, tokenizer/output text, or output-head token selection wired.

The correct next increment is therefore explicit: expose a diagnostic
`/v1/completions` surface that exercises the real TP/EP resident decode path,
but marks itself as diagnostic and nests TP/EP metrology under `ds4_v100`
instead of pretending to be full model serving.

## Implementation

- Extend the TP/EP HTTP generation classifier to accept:
  - `POST /v1/completions`
  - `POST /v100/diagnostic-completions`
- Reuse the same coalesced and bucketed decode path as
  `POST /v100/selected-token`.
- Return OpenAI-style response fields:
  - `id`
  - `object: text_completion`
  - `created`
  - `model: ds4-v100-tp-ep-diagnostic`
  - `choices`
  - `usage`
- Preserve the complete TP/EP timing and admission metadata under `ds4_v100`.
- Mark the endpoint with `diagnostic: true` and a note that prompt prefill and
  output head are not yet wired in this TP/EP endpoint.
- Extend `tools/ds4-v100-tp-ep-http-bench.sh` with
  `--endpoint selected-token|completions`.
- Teach the bench aggregator to read metrics from either the original
  selected-token response shape or the nested `ds4_v100` completion response.

## Definition of Done

- [x] `tools/ds4-v100-tp-ep-http-bench.sh` passes shell syntax validation.
- [x] `tools/ds4-v100-tp-ep-full-layer-smoke.cu` builds on the V100 pod.
- [x] `/v1/completions` returns an OpenAI-style diagnostic response.
- [x] Completion requests use the same coalesced/bucketed TP/EP resident decode
  path as selected-token requests.
- [x] A V100 completion-shaped mixed run with 32 concurrent requests and
  `32,64` requested tokens returns `32/32` token match.
- [x] The existing selected-token endpoint still passes a 32-request sanity run.
- [x] Sprint status and vision are updated with the result.

## Validation

Local:

```text
bash -n tools/ds4-v100-tp-ep-http-bench.sh
git diff --check
```

V100 pod:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke

tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir /workspace/logs/sprint288-tp-ep-diagnostic-completions \
  --tokens-cases 32 \
  --request-token-pattern 32,64 \
  --requests 32 \
  --endpoint completions \
  --port-base 28800

tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir /workspace/logs/sprint288-tp-ep-selected-token-sanity \
  --tokens-cases 32 \
  --requests 32 \
  --endpoint selected-token \
  --port-base 28820
```

Completion-shaped mixed run:

| Endpoint | Pattern | Requests | Batches | Max batch | Generated tokens | Wall generated tok/s | Decode generated tok/s | Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| completions | 32,64 | 32 | 2 | 16 | 1536 | 384.581100 | 505.797315 | 32/32 |

Selected-token sanity:

| Endpoint | Pattern | Requests | Batches | Max batch | Generated tokens | Wall generated tok/s | Decode generated tok/s | Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selected-token | 32 | 32 | 1 | 32 | 1024 | 726.823991 | 944.195924 | 32/32 |

Evidence:

```text
logs/from-cluster/sprint288-tp-ep-diagnostic-completions/cluster/
logs/from-cluster/sprint288-tp-ep-selected-token-sanity/cluster/
```

## Decision

Promote `/v1/completions` as the TP/EP diagnostic serving-shaped endpoint for
load testing and client-harness integration. It is not yet real text serving:
prompt prefill, tokenizer output, logits/output-head token selection, and stop
handling remain to be wired.

The selected-token endpoint remains available for lower-level correctness and
throughput diagnostics.

## Risks

- The response shape is OpenAI-compatible enough for load tools, but text is
  intentionally empty because this endpoint does not yet emit decoded model
  text.
- Metrics still count admitted client tokens while partial buckets run the
  configured 32-slot decode shape internally.
- The next sprint should move from diagnostic completions to real model output,
  starting with output-head/top-token integration in the TP/EP path.
