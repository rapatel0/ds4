# Sprint 286 - TP/EP HTTP Request Coalescing

Date: 2026-05-23

## Goal

Make the TP/EP HTTP serving path admit concurrent selected-token requests into
one resident decode batch instead of treating every HTTP request as an
independent synthetic 32-slot run.

This sprint stays inside the TP/EP-only codepath. The frozen PP/layer-split
path is not modified.

## Context

Sprint 285 established the promoted launcher topline at `32` slots / `256K`,
but the harness still measured repeated synthetic requests where each request
filled all slots by itself. That is useful for kernel metrology, but it is not
serving semantics. Practical serving needs independent client requests to
coalesce into the active microbatch so the 32-slot shape represents 32 users,
not one request replicated across 32 slots.

## Implementation

- Add `--microbatch-wait-us` to the TP/EP HTTP server.
- When the server receives `POST /v100/selected-token`, briefly accept more
  pending generation requests up to `slots`.
- Reject mixed `max_tokens` inside one coalesced group for now.
- Run one resident TP/EP decode with `req_opt.slots = coalesced_batch_size`.
- Return one HTTP response per client with:
  - per-client token counts,
  - shared batch timing,
  - `coalesced_batch_id`,
  - `coalesced_batch_size`,
  - `coalesced_slot_index`,
  - batch token counts.
- Expose coalescing counters in `/v100/status` and `/metrics`.
- Update the TP/EP HTTP bench to send concurrent requests by default and
  aggregate timing once per coalesced batch.

## Definition of Done

- [x] `tools/ds4-v100-run-appliance.sh` passes the resolved
  `DS4_V100_MICROBATCH_WAIT_US` value to the TP/EP server.
- [x] `tools/ds4-v100-tp-ep-full-layer-smoke.cu` builds on the V100 pod.
- [x] `tools/ds4-v100-tp-ep-http-bench.sh` supports concurrent and sequential
  request modes.
- [x] A V100 HTTP bench with `--requests 32` demonstrates at least one
  `coalesced_batch_size=32` response at `32` slots / `256K`.
- [x] The bench report includes coalesced batch count and max batch size.
- [x] Correctness remains `token_match == generation_requests`.
- [x] Sprint status and vision are updated with the result.

## Validation

Local:

```text
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-tp-ep-http-bench.sh
git diff --check
```

V100 pod:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
./tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir /workspace/logs/sprint286-tp-ep-coalescing-topline \
  --tokens-cases 32,64 \
  --requests 32 \
  --port-base 18440
```

Results:

| Tokens/request | Concurrent requests | Coalesced batches | Max batch | Generated tokens | Wall generated tok/s | Decode generated tok/s | Match | Avg GPU util | Max GPU util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 1 | 32 | 1024 | 721.446441 | 950.363316 | 32/32 | 9.270833 | 30.000000 |
| 64 | 32 | 1 | 32 | 2048 | 787.316214 | 1030.972573 | 32/32 | 11.050000 | 32.000000 |

Evidence:

```text
logs/from-cluster/sprint286-tp-ep-coalescing-topline/cluster/
```

## Decision

Promote request coalescing as the default TP/EP HTTP benchmark semantics.
The current practical-serving topline is now measured as 32 independent
concurrent selected-token requests coalesced into one 32-slot resident decode
batch. This is slightly lower than Sprint 285's synthetic slot-filled
topline, but it is the correct serving-shaped measurement.

Next work should build the real prompt/token API and bucketed admission queues
on top of this coalescing path. MTP remains deferred until TP/EP serving is
operational beyond the selected-token harness.

## Risks

- The current HTTP reader is intentionally small and still not a production
  HTTP stack; this sprint only proves admission/coalescing semantics for the
  current harness.
- Smaller coalesced batches can lower tok/s versus old synthetic slot-filled
  metrics. That is expected and more honest.
- Mixed token counts are rejected rather than scheduled into separate queues.
  A later serving sprint should add bucketed queues.
