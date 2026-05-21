# Sprint 119 - Event-Ordered Stage Handoff

Date: 2026-05-21

## Objective

Promote a no-math scheduling improvement for the served multi-slot path before
starting another kernel rewrite. The current production path uses per-step
stage workers, but the default handoff path synchronizes each stage/slot before
marking it ready. This sprint validates CUDA event-ordered handoff so the next
stage waits on the producer event and copies HC asynchronously.

## Implementation

1. Add `DS4_V100_ASYNC_EVENT_HANDOFF=auto` to the launcher.
2. Resolve `auto` to enabled only when the runtime is using the measured
   multi-slot `per-step` async pipeline.
3. Keep one-slot latency and non-per-step modes on the old behavior.
4. Add explicit event-handoff plumbing to the soak harness and summary JSON.
5. Update the appliance example environment to use the promoted `auto` mode.

The model math is unchanged. The path still uses the same decode kernels,
same HC tensors, same final all-device fence, and same output-token selection.

## Results

Same-binary served A/B on the 8x V100 node:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Control, event handoff off | 262,144 | 8 | `33.379839` | `31.293599` | 8/8 token match |
| Event handoff on | 262,144 | 8 | `34.433252` | `32.281173` | 8/8 token match |
| Control, event handoff off | 1,048,576 | 4 | `21.566859` | `20.218931` | 4/4 token match |
| Event handoff on | 1,048,576 | 4 | `21.771077` | `20.410385` | 4/4 token match |

Artifacts:

- `logs/from-cluster/sprint119-event/control/summary.json`
- `logs/from-cluster/sprint119-event/event/summary.json`
- `logs/from-cluster/sprint119-event/control-1m/summary.json`
- `logs/from-cluster/sprint119-event/event-1m/summary.json`
- `logs/from-cluster/sprint119-auto/auto/summary.json`

The final auto-mode verification run used four requests against an eight-slot
configuration, so it is not used as the topline throughput number. It verifies
that the promoted launcher/harness default resolves to `"async_event_handoff":
true` in the real served process and preserves `4/4` token matches.

## Decision

Ship `DS4_V100_ASYNC_EVENT_HANDOFF=auto` as the production appliance default.
It is a small but material improvement, and it does not change model numerics.

This does not close the practical throughput gap. The next larger optimization
target remains a real SM70 software-pipelined F8 path, starting with the hot
single-token `2048 x 4096` shared gate/up plus SwiGLU shape, or deeper
TurboMind grouped expert scheduling.
