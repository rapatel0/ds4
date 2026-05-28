# TEMP Status Report 462

## Focus

TP/EP-only CUDA graph event-order investigation. No PP/layer-split work.

## What Changed

- Added graph-order event rings to `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Replaced repeated `stream_done` / `dense_done` graph barriers with fresh event
  slots across HC-current, final-HC, global rank/control helpers, dense-stream
  helpers, cross-GPU barriers, and indexer fanout waits.

## Build

```text
node: gpu-01
workspace: /localpool/ds4/workspace/ds4-sprint181
build: make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
result: PASS
```

## Experiment

```text
artifact: /localpool/ds4/workspace/logs/s462-event-ring-graph-gate-s8-t3
shape: 8 requests, 8 slots, 256K context, position 262000, 3 decode tokens
control: HC-current NCCL + router/FFN rank-major baseline
candidate: control + decode CUDA graph capture, no replay
```

## Topline

| Metric | Control | Candidate |
|---|---:|---:|
| readiness | pass | fail |
| response parity | pass | 0/8 |
| first token | 52762 | 57097 |
| server generated decode tok/s | 20.322165 | 9.328611 |
| server continuation decode tok/s | 20.491394 | 9.322448 |
| client generated tok/s | 2.054682 | 0.839028 |
| HC-current gather ms | 4.310811 | 157.328098 |
| HC-current input ms | 220.716319 | 331.008149 |
| graph capture | 0/0 | 43/43 |
| graph replay | 0/0 | 0/0 |
| graph blocker | n/a | none |
| min free VRAM | 5092 MiB | 5086 MiB |

## Conclusion

The event-ring patch did not fix graph semantic correctness. The result has the
same wrong first token as the previous graph no-replay candidate, so simple CUDA
event reuse is not the root cause. The next useful work is first-divergence
instrumentation around graph-captured event ordering at a smaller direct shape,
then returning to serving once the first bad stage is known.
