# Sprint 287 - TP/EP Bucketed Admission

Date: 2026-05-23

## Goal

Make the TP/EP HTTP serving path handle mixed concurrent generation lengths by
queueing requests into token-count buckets instead of rejecting mismatches
during coalescing.

This continues the TP/EP-only serving direction. The frozen PP/layer-split path
is not modified.

## Context

Sprint 286 made 32 independent HTTP requests coalesce into one 32-slot resident
decode batch, but it still rejected mixed `max_tokens` values inside the
coalescing window. Real serving traffic will naturally contain mixed decode
lengths. Until the runtime has full per-slot early stop and dynamic slot refill,
the conservative serving policy is bucketed admission: group requests with the
same requested decode length and run one resident batch per bucket.

## Implementation

- Keep a pending generation queue inside the TP/EP HTTP server.
- When a request with a different `max_tokens` arrives during coalescing, keep
  its socket open and queue it instead of returning `409`.
- Before accepting new sockets for a batch, drain queued requests that match
  the current token bucket.
- Continue the server loop while pending generation requests exist, even if the
  accepted socket count has reached `--max-requests`.
- Expose bucketed and pending request counters in `/v100/status` and
  `/metrics`.
- Extend the TP/EP HTTP bench with `--request-token-pattern CSV` so mixed
  concurrent traffic can be replayed and measured.

## Definition of Done

- [x] `tools/ds4-v100-tp-ep-full-layer-smoke.cu` builds on the V100 pod.
- [x] Mixed concurrent requests no longer return `409`.
- [x] A V100 bench with `--requests 32 --request-token-pattern 32,64` returns
  `32/32` token match.
- [x] The mixed run reports multiple coalesced batches and a nonzero
  bucketed-request counter.
- [x] Uniform concurrent request performance remains covered by Sprint 286
  semantics.
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

./tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir /workspace/logs/sprint287-tp-ep-bucketed-admission \
  --tokens-cases 32 \
  --request-token-pattern 32,64 \
  --requests 32 \
  --port-base 18480

./tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir /workspace/logs/sprint287-tp-ep-uniform-sanity \
  --tokens-cases 32 \
  --requests 32 \
  --port-base 18490
```

Mixed result:

| Pattern | Requests | Batches | Max batch | Generated tokens | Wall generated tok/s | Decode generated tok/s | Match | Bucketed requests | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,64 | 32 | 2 | 16 | 1536 | 387.877251 | 510.747848 | 32/32 | 16 | 0 |

Uniform sanity:

| Pattern | Requests | Batches | Max batch | Generated tokens | Wall generated tok/s | Decode generated tok/s | Match | Bucketed requests | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 1 | 32 | 1024 | 759.490446 | 991.405750 | 32/32 | 0 | 0 |

Evidence:

```text
logs/from-cluster/sprint287-tp-ep-bucketed-admission/cluster/
logs/from-cluster/sprint287-tp-ep-uniform-sanity/cluster/
```

## Decision

Promote bucketed admission for the TP/EP selected-token harness. Mixed decode
lengths are now held and served in later same-length batches instead of being
rejected.

The implementation intentionally keeps the underlying resident decode shape at
the configured `32` slots even when a bucket admits fewer clients. This avoids
the compact route-compose peer-copy sizing bug that appears at dynamic partial
bucket sizes. Serving metrics count only admitted client tokens, while response
metadata also reports `decode_slots` and decode token counts for transparency.

Next work should move from selected-token serving semantics to a real
prompt/token-compatible TP/EP endpoint, still using the coalesced and bucketed
admission policy built in Sprints 286-287.

## Risks

- This is still bucketed by requested decode length, not true per-slot early
  stop. It is intentionally conservative so all active slots in one resident
  decode batch have the same loop length.
- Open sockets for queued requests remain held until their bucket executes.
  That is acceptable for the harness, but the production server should grow an
  explicit worker/admission queue.
- Mixed queues may produce smaller batches and lower aggregate tok/s. That is
  expected and gives more honest serving metrology.
- Partial buckets currently run the configured full-slot decode shape with
  unused synthetic slots. A later runtime sprint should add true dynamic-slot
  compact route-compose or per-slot refill.
