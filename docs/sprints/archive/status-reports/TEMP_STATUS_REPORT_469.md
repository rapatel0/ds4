# TEMP Status Report 469

Date: 2026-05-27

## Focus

Persistent CUDA graph replay for TP/EP serving.

## Summary

Persistent replay is fast but not correct yet.

The existing full-layer persistent graph path reaches roughly `2x` server
decode throughput at the small 8-slot/256K shape, but response parity fails.
The failure is present even for a single generated token and first appears at
layer 0, so this is not a late-layer, MTP, output-head, or multi-token position
reuse issue.

## Tests Run

| Test | Shape | Parity | Server decode tok/s | Replay |
|---|---|---:|---:|---:|
| Full persistent | 8 req / 8 slots / 256K / 3 tok | `0/8` | `19.857286 -> 39.548197` | `43/43` |
| Full persistent | 8 req / 8 slots / 256K / 1 tok | `0/8` | `20.369642 -> 42.148757` | `43/43` |
| Suffix-only persistent | 8 req / 8 slots / 256K / 1 tok | `0/8` | `20.318784 -> 27.320756` | `43/43` |
| Suffix + prefix sync | 8 req / 8 slots / 256K / 1 tok | `0/8` | `19.941094 -> 29.635109` | `43/43` |

Artifacts:

- `/localpool/ds4/workspace/logs/s469-persistent-recheck-s8-t3`
- `/localpool/ds4/workspace/logs/s469-persistent-recheck-s8-t1`
- `/localpool/ds4/workspace/logs/s469-suffix-persistent-s8-t1-clean`
- `/localpool/ds4/workspace/logs/s469-suffix-prefix-sync-s8-t1`

## Code Change

Added default-off diagnostic behavior to persistent graph replay:

- run dynamic prefix normally;
- capture/replay only the post-attention suffix;
- remove position from the suffix graph invalidation condition;
- add a correctness-first prefix stream synchronization barrier before suffix
  replay.

This is not production-ready and remains behind the existing persistent graph
flag.

## Interpretation

Full-layer graph replay is unsafe because dynamic route/KV/attention work is
inside the captured graph. Suffix-only replay narrows the problem and removes
position invalidation, but parity still fails from layer 0. Prefix completion is
not enough, so the next bug is inside the captured suffix or in suffix inputs
that are not replay-stable.

## Next Step

Do not run more full HTTP graph A/Bs until the suffix is isolated by stage.

Next sprint should add a layer-0 direct harness for:

- routed FFN graph replay only;
- dense overlap graph replay only;
- compose/final-HC graph replay only.

The first stage that changes the layer-0 checksum is the next target.
