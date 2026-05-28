# Sprint 462: TP/EP Graph-Order Event Slot Isolation

## Objective

Stop reusing the same CUDA event objects for every graph-order barrier inside a
decode step.

## Rationale

Sprint 461 fixed one real missing router dependency, but graph event-order still
failed parity and kept HC-current gather badly regressed. The remaining likely
mechanism is shared event reuse: graph mode records and waits on `stream_done`
and `dense_done` repeatedly across many barrier sites without host
synchronization between stages.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Add a per-rank graph-order event ring.
- Use fresh event slots for graph-order control/rank barrier helpers.
- Start with HC-current and shared helper barriers.
- Re-run the same 8-slot / 256K / 3-token no-replay graph A/B.

## Definition of Done

- CUDA binary rebuilds.
- HTTP A/B completes on a clean node.
- Result records parity, first token, HC-current gather timing, and decision.

## Implementation

- Added a per-rank graph-order event ring with `1024` event slots for stream
  and dense-stream ordering.
- Switched graph-order helper barriers away from repeatedly recording/waiting
  on the same `stream_done` / `dense_done` event handles.
- Covered the shared HC-current/final-HC local barriers, the global rank/control
  helper barriers, dense-stream helper barriers, cross-GPU barrier helpers, and
  the two explicit indexer fanout graph waits.

## Validation

Remote build:

```text
node: gpu-01
workspace: /localpool/ds4/workspace/ds4-sprint181
command: make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
result: PASS
```

HTTP A/B:

```text
artifact: /localpool/ds4/workspace/logs/s462-event-ring-graph-gate-s8-t3
shape: 8 requests, 8 slots, 256K context, position 262000, 3 decode tokens
control: current TP/EP baseline with HC-current NCCL, router+FFN rank-major
candidate: same baseline plus decode CUDA graph capture, no replay
```

| Metric | Control | Candidate |
|---|---:|---:|
| readiness | pass | fail |
| response parity | pass | 0/8 |
| HTTP 200 responses | 8 | 8 |
| output-head first token | 52762 | 57097 |
| server generated decode tok/s | 20.322165 | 9.328611 |
| server continuation decode tok/s | 20.491394 | 9.322448 |
| client generated tok/s | 2.054682 | 0.839028 |
| HC-current gather ms | 4.310811 | 157.328098 |
| HC-current input ms | 220.716319 | 331.008149 |
| graph captures | 0/0 | 43/43 |
| graph replays | 0/0 | 0/0 |
| graph event barrier calls | 0 | 215 |
| graph blocker | n/a | none |
| min free VRAM | 5092 MiB | 5086 MiB |

## Decision

Do not promote. The event-ring isolation did not change the graph no-replay
failure signature: the candidate still emits token `57097`, fails parity, and
regresses decode throughput. CUDA event handle reuse was a plausible bug, but it
is not the current root cause.

## Next

Graph capture is now correctly reaching `capture_succeeded=43/43` with no
reported blocker, so the remaining failure is semantic ordering or captured
dynamic-state semantics inside the graph path. The next sprint should compare
non-graph event-order versus graph-captured event-order at a smaller direct
shape and add per-stage checksums around HC-current, attention projection,
router logits, and output head to locate the first divergence before more
serving-scale graph runs.
