# Sprint 469: TP/EP Persistent Graph Replay Recheck

## Objective

Turn the Sprint 468 graph correctness fix into a performance path by testing
and repairing persistent CUDA graph replay for TP/EP serving.

## Rationale

Sprint 468 proved non-persistent graph-event-order serving is correctness-clean
without diagnostic host sync, but it captures `43` graphs and replays `0`, so it
is slower than eager. The next bottleneck is not typed-history correctness; it
is that serving is still paying capture cost instead of replaying stable graphs.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- First run the existing persistent graph serving mode with the Sprint 468
  typed-history boundary.
- If persistent replay is still absent, fix cache eligibility/keying so replay
  is attempted for repeated shape execution.
- If persistent replay occurs but parity fails, use the Sprint 466 stage
  checksum gates to localize the first replay-only divergence.
- Validate at `8` requests / `8` slots / `256K` / `3` tokens before scaling.

## Definition of Done

- Existing persistent graph candidate is tested on V100 at the Sprint 468
  shape.
- Results record graph captures, cache hits/misses, replay attempts, parity,
  and decode tok/s.
- If code changes are needed, they build on the V100 node and are retested.
- The sprint has an explicit promote/reject/continue decision.

## Experiments

### Baseline Persistent Replay Recheck

Artifacts:

`/localpool/ds4/workspace/logs/s469-persistent-recheck-s8-t3`

Shape: `8` requests / `8` slots / `256K` / `3` tokens.

| Metric | Eager control | Persistent graph |
|---|---:|---:|
| Response parity | - | `0/8` |
| Server generated decode tok/s | `19.857286` | `39.548197` |
| Server continuation tok/s | `19.812029` | `39.569093` |
| Replay attempts/successes | `0/0` | `43/43` |
| Persistent cache hits | `0` | `0` |
| Persistent cache misses | `0` | `43` |
| Position invalidations | `0` | `43` |

This confirmed the speed signal is real when replay runs, but correctness still
fails. Per-layer checksum comparison showed the first divergence is already at
`step=0 layer=0`, so the issue is not a late MoE layer, long-context layer, or
MTP/output-head artifact.

### Single-Token Persistent Replay

Artifacts:

`/localpool/ds4/workspace/logs/s469-persistent-recheck-s8-t1`

Shape: `8` requests / `8` slots / `256K` / `1` token.

| Metric | Eager control | Persistent graph |
|---|---:|---:|
| Response parity | - | `0/8` |
| Server generated decode tok/s | `20.369642` | `42.148757` |
| Replay attempts/successes | `0/0` | `43/43` |

Single-token replay also failed, which rules out token-to-token position reuse
as the sole cause.

### Suffix-Only Persistent Replay

Changed persistent replay to run the dynamic prefix normally and capture only
the post-attention suffix:

- HC-current input
- attention projection / compressed KV / state update
- typed KV history
- raw read
- attention output
- post-attention FFN input and route preparation

The graph suffix then starts at routed FFN / dense overlap / compose / final HC.

Artifacts:

`/localpool/ds4/workspace/logs/s469-suffix-persistent-s8-t1-clean`

| Metric | Eager control | Suffix persistent graph |
|---|---:|---:|
| Response parity | - | `0/8` |
| Server generated decode tok/s | `20.318784` | `27.320756` |
| Replay attempts/successes | `0/0` | `43/43` |
| Persistent cache hits | `0` | `43` |
| Position invalidations | `0` | `0` |

This removed position invalidation and narrowed the graph to the compute suffix,
but parity still failed from layer 0.

### Suffix Replay With Prefix Completion Barrier

Added a correctness-first host barrier after the dynamic prefix and before
suffix capture/replay to rule out in-flight prefix work racing the graph suffix.

Artifacts:

`/localpool/ds4/workspace/logs/s469-suffix-prefix-sync-s8-t1`

| Metric | Eager control | Suffix + prefix sync |
|---|---:|---:|
| Response parity | - | `0/8` |
| Server generated decode tok/s | `19.941094` | `29.635109` |
| Replay attempts/successes | `0/0` | `43/43` |
| Persistent cache hits | `0` | `43` |
| Position invalidations | `0` | `0` |

The barrier did not restore parity. The remaining replay bug is inside the
captured suffix itself, or in data consumed by the suffix that is not represented
as replay-safe graph input.

## Decision

Do not promote persistent graph replay.

Keep the suffix-only replay work default-off as diagnostic scaffolding. It gives
a cleaner next debugging surface than full-layer capture because dynamic
attention/KV/router prep is outside the graph, but it is still not correct.

## Next

The next sprint should isolate the suffix by stage:

- capture/replay routed FFN only;
- capture/replay dense overlap only;
- capture/replay compose/final-HC only;
- compare per-layer checksums against eager at layer 0 before running full HTTP
  A/Bs.

If routed FFN replay is the first failing stage, inspect TurboMind graph-capture
compatibility and any host/device descriptor mutation. If compose/final-HC is
the first failing stage, inspect captured route-plan pointers and graph replay
inputs.
