# TEMP Status Report 461

## Topline

I fixed one real graph event-order dependency hole, but it did not repair graph
mode. No-replay graph event-order still fails parity and remains much slower.

Artifact:

```text
/localpool/ds4/workspace/logs/s461-router-wait-graph-gate-s8-t3
```

## Patch

`run_model_router_rank_major_logits` now waits on rank streams after NCCL
allgather before the control stream consumes rank-major router logits.

This was a real bug: eager mode had stream synchronizes there, graph mode did
not.

## Validation Result

Shape:

```text
8 requests / 8 slots / 256K / 3 tokens
```

| Metric | Control | Graph event-order + router wait |
|---|---:|---:|
| readiness | pass | fail |
| parity | - | 0/8 |
| HTTP 200 | 8 | 8 |
| server generated decode tok/s | 20.087551 | 9.441529 |
| server continuation decode tok/s | 20.014039 | 9.440907 |
| client generated tok/s | 2.015089 | 0.831814 |
| output-head first token | 52762 | 42549 |
| graph capture | 0/0 | 43/43 |
| graph replay | 0/0 | 0/0 |
| HC-current gather ms | 4.457466 | 158.533187 |
| HC-current input ms | 224.170869 | 326.915807 |

## Interpretation

The graph-order blocker is more systemic than the router allgather edge. The
most likely issue is the event-barrier implementation itself: it reuses shared
CUDA events like `stream_done` and `dense_done` across multiple barrier sites in
one decode step. Once host synchronizes are removed, those reused events can
make dependencies ambiguous or force bad ordering.

## Next Action

Stop treating persistent replay as the immediate lever. First make no-replay
graph event-order match eager:

1. Introduce per-stage event slots for HC-current graph-order barriers.
2. Start with HC-current gather/split/router/fill-pack, since gather is the
   measured regression.
3. Re-run the same 8-slot/256K/3-token A/B after each event-order fix.
4. Return to persistent replay only after no-replay graph mode passes parity.
