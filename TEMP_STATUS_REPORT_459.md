# TEMP Status Report 459

## Topline

Sprint 459 added persistent CUDA graph cache telemetry and ran a focused V100
HTTP A/B. Persistent graph replay is still not promotable.

Artifact:

```text
/localpool/ds4/workspace/logs/s459-graph-cache-s8-t3
```

## Current Best Production Path

Current usable TP/EP serving remains the graph-off rank-major/NCCL baseline.
The latest reduced-shape control in this sprint served:

```text
8 requests / 8 slots / 256K / 3 tokens
server generated decode:      20.032612 tok/s
server continuation decode:   20.024299 tok/s
request-window avg GPU util:  11.325%
min free VRAM:                5092 MiB
readiness/parity:             pass
```

The target-shape baseline from Sprint 458 remains:

```text
32 requests / 32 slots / 256K / 4 tokens
server generated decode:      35.616755 tok/s
server continuation decode:   35.499112 tok/s
avg GPU util:                 12.067073%
min free VRAM:                1734 MiB
readiness:                    pass
```

## What Changed

Added graph cache telemetry to the TP/EP harness:

- cache hits
- cache misses
- invalidations
- invalidation reasons: layer, slots, position, root device, root stream
- parser fields in `summary.json`
- A/B fields in `ab-summary.json` and markdown

Validation:

```text
local py_compile: pass
remote py_compile: pass
remote CUDA rebuild: pass
```

## Persistent Graph Result

Focused shape:

```text
8 requests / 8 slots / 256K / 3 tokens
```

| Metric | Control | Persistent graph |
|---|---:|---:|
| readiness | pass | fail |
| parity | - | 0/8 |
| HTTP 200 | 8 | 8 |
| server generated decode tok/s | 20.032612 | 39.823731 |
| server continuation decode tok/s | 20.024299 | 39.737946 |
| client generated tok/s | 1.941361 | 0.572085 |
| output-head first token | 52762 | 123327 |
| graph capture | 0/0 | 43/43 |
| graph replay | 0/0 | 43/43 |
| persistent cache hits | 0 | 0 |
| persistent cache misses | 0 | 43 |
| persistent invalidations | 0 | 43 |
| position invalidations | 0 | 43 |
| instantiate ms | 0 | 268.522412 |
| replay ms | 0 | 201.336832 |

## Interpretation

Persistent graph is not actually persistent across token positions in HTTP
serving today. Every layer invalidates on position, so the candidate recaptures
and reinstantiates every step. The wrong first token with zero cache hits means
the issue is not stale persistent-cache reuse; graph replay itself is unsafe in
the current serving step.

The server-side decode speedup is real but invalid because parity fails and
client throughput regresses.

## Next Focus

1. Re-run graph-safe event ordering without replay when the node is clean.
2. Add a safe capture/no-replay diagnostic mode if needed.
3. Start device-resident dynamic metadata for graph replay:
   position, raw KV row, compressed-row counters, route totals/offsets.
4. Do not promote graph replay until response parity and readiness pass.
