# Sprint 160 - Async Slot Chunking And Routed-FFN Density Gate

Date: 2026-05-21

## Objective

Verify whether the current HTTP serving path can expose dense routed-FFN shapes
without losing the stage overlap that makes the 16-slot / 256K appliance fast.

## Rationale

Sprint 158 showed that the fixed96 routed executor selects correctly when the
scheduler presents:

```text
n_slots = 16
routes_per_token = 6
total_routes = 96
```

but normal HTTP serving had previously exposed `total_routes=6`. The question
for Sprint 160 was whether this is a batching bug we can fix cheaply, or
whether dense routed-FFN shapes require a different execution topology.

## Findings

The HTTP request layer does coalesce requests. The split happens below it:

- HTTP groups requests in `process_pending_generation_batch()`.
- `ds4_v100_replay_generate_batch()` calls `replay_feed_token_batch_selected()`.
- With appliance defaults, `auto` async resolves to `per-step`.
- `per-step` uses `replay_async_slot_chunk()`.
- The default chunk is `1`, so each stage calls the scheduler with one slot.
- A one-slot scheduler call reaches the single-slot layer path and the routed
  FFN sees `total_routes = 1 * 6 = 6`.

Setting `DS4_V100_ASYNC_SLOT_CHUNK=16` is the zero-code proof: it makes the
stage worker call the slot-span scheduler with 16 slots, which reaches
`execute_ffn_delta_batch()` and presents `total_routes=96`.

## Implementation

Added `DS4_V100_ASYNC_SLOT_CHUNK` to the launcher config contract so future
chunk experiments are visible in:

- `--check` output
- `runtime/startup.env`
- the deployment env example

The default remains empty, which preserves the measured per-slot stage pipeline.

## Validation

Launcher validation:

```text
bash -n tools/ds4-v100-run-appliance.sh
DS4_V100_ASYNC_SLOT_CHUNK=4 ./tools/ds4-v100-run-appliance.sh --check --allow-missing
```

passed and reported:

```text
async_slot_chunk=4
```

V100 fixed96 dense-shape proof:

```text
DS4_V100_ASYNC_SLOT_CHUNK=16
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed96
DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE=1
ctx=262144
slots=16
tokens=64
requests=16
```

Server log proved selection:

```text
ds4: TurboMind routed executor fixed96 shape total_routes=96 active_experts=6 max_routes_per_expert=16
ds4: TurboMind routed executor selected fixed gate_up total_routes=96
```

Correctness passed (`16/16`), but throughput collapsed:

```text
generated tok/s:     20.666415
continuation tok/s:  20.343502
```

Async slot chunk sweep at the same 16-slot / 256K / 64-token / 16-request
shape:

| Chunk | Generated tok/s | Continuation tok/s | Correctness |
|---:|---:|---:|---:|
| default / 1 | `71.391103` | `70.275617` | `16/16` |
| 2 | `61.673632` | `60.709982` | `16/16` |
| 4 | `29.031399` | `28.577783` | `16/16` |
| 8 | `27.031623` | `26.609254` | `16/16` |
| 16 + fixed96 | `20.666415` | `20.343502` | `16/16` |

## Decision

Do **not** default `DS4_V100_ASYNC_SLOT_CHUNK` above 1.

Dense routed-FFN shapes are reachable, but global slot chunking destroys the
stage overlap that dominates end-to-end decode throughput on the current
layer-split appliance. The result validates the diagnosis but rejects the
simple fix.

## Next Step

The next material serving implementation should avoid global slot chunking.
There are two credible paths:

- a layer-local batching redesign that batches routed FFN work across slots
  while keeping inter-stage pipeline overlap; or
- a TP/EP prototype that creates denser HMMA-heavy kernels inside each layer
  without serializing 16 slots through each stage as a single chunk.

Given prior TP probes were positive on NV2 pairs at the 768-route shape and
slot-widening has plateaued, the next sprint should focus on a bounded TP/EP
prototype for the 256K practical-serving target.

## Artifacts

- `logs/from-cluster/sprint160-chunk-control-16slot-64tok-16req/`
- `logs/from-cluster/sprint160-chunk-2-16slot-64tok-16req/`
- `logs/from-cluster/sprint160-chunk-4-16slot-64tok-16req/`
- `logs/from-cluster/sprint160-chunk-8-16slot-64tok-16req/`
- `logs/from-cluster/sprint160-fixed96-chunk16-16slot-64tok-16req/`

