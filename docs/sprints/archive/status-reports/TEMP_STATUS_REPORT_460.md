# TEMP Status Report 460

## Topline

Graph event-order without replay is also not correctness-safe. It changes the
first token and fails response parity, so the immediate graph work is now
event-order correctness, not persistent replay optimization.

Clean artifact:

```text
/localpool/ds4/workspace/logs/s460-graph-gate-s8-t3-r2
```

## Result

Shape:

```text
8 requests / 8 slots / 256K / 3 tokens
```

| Metric | Control | Graph event-order, no replay |
|---|---:|---:|
| readiness | pass | fail |
| parity | - | 0/8 |
| HTTP 200 | 8 | 8 |
| server generated decode tok/s | 20.009325 | 9.388085 |
| server continuation decode tok/s | 19.967099 | 9.350279 |
| client generated tok/s | 2.057285 | 0.839831 |
| output-head first token | 52762 | 57097 |
| graph capture | 0/0 | 43/43 |
| graph replay | 0/0 | 0/0 |
| graph blocker | n/a | none |
| HC-current gather ms | 4.487008 | 157.184537 |
| HC-current input ms | 228.028590 | 331.324808 |
| min free VRAM MiB | 5092 | 5086 |

## Interpretation

Sprint 459 showed persistent graph replay is not getting cache hits and still
fails parity. Sprint 460 shows even no-replay graph mode fails parity and
regresses decode. That moves the root blocker earlier:

```text
graph-safe event ordering / cross-stream dependencies
```

The large HC-current gather regression is the strongest clue. The graph path's
replacement for stream synchronization is not semantically/performance
equivalent to the eager baseline.

## Next Action

Instrument and fix event-order synchronization before any more persistent graph
reuse work:

1. Compare eager vs graph-event ordering around HC-current NCCL allgather.
2. Add per-rank event wait/record counters and dependency IDs for HC-current
   gather, split, router, and route upload.
3. Build a one-layer reduced probe that runs eager and graph-event ordering in
   the same process and compares `d_current_full`, rank-major buffers, route
   totals, and output-head token.
4. Only return to persistent graph replay after no-replay graph mode matches
   parity.
