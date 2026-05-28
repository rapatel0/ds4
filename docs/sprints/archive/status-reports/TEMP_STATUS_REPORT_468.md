# TEMP Status Report 468

Date: 2026-05-27

## Focus

Sprint 468 tested whether the graph correctness issue isolated in Sprint 467
was the missing final ordering boundary after typed KV history's ratio-4
indexer top-k copy/broadcast.

## Code Change

- Added a final graph-safe `sync_typed_kv_boundary(opt, ranks)` after typed KV
  history completes.
- Removed the failed typed KV store-side `__threadfence_system()` experiment.
- Kept Sprint 467's default-off diagnostic stage sync gates for future
  bisection.

## Cluster Validation

Remote workspace:

`/localpool/ds4/workspace/ds4-sprint181`

Build:

`make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke`

Result: build succeeded on the 8x V100 node with only expected unused-kernel
warnings.

## A/B Results

### 8 requests / 8 slots / 256K / 1 token

Artifacts:

`/localpool/ds4/workspace/logs/s468-typed-history-final-barrier-s8-t1`

| Metric | Eager control | Graph candidate |
|---|---:|---:|
| Response parity | - | `8/8` |
| Server generated decode tok/s | `20.331259` | `8.478366` |
| Client generated tok/s | `0.831886` | `0.303515` |
| Request-window GPU util avg | `11.35%` | `4.59%` |
| Graph captures | `0` | `43` |
| Graph replays | `0` | `0` |

### 8 requests / 8 slots / 256K / 3 tokens

Artifacts:

`/localpool/ds4/workspace/logs/s468-typed-history-final-barrier-s8-t3`

| Metric | Eager control | Graph candidate |
|---|---:|---:|
| Response parity | - | `8/8` |
| Server generated decode tok/s | `20.333332` | `7.522808` |
| Server continuation tok/s | `20.240114` | `7.551176` |
| Client generated tok/s | `2.073092` | `0.710383` |
| Request-window GPU util avg | `10.32%` | `3.56%` |
| Graph captures | `0` | `43` |
| Graph replays | `0` | `0` |

## Interpretation

Correctness improved materially: the graph-event-order candidate now matches
eager without the diagnostic `typed_history` host sync.

Performance did not improve because the active serving graph mode is still
capture-only for this path. It captures all 43 layer graphs and never replays
them, so it adds graph overhead instead of removing launch overhead.

## Current Topline

- Best current validated serving baseline remains the eager TP/EP path with
  router+FFN rank-major and scratch 1280.
- Sprint 468 validated graph correctness at 8 slots / 256K, but not graph
  performance.
- The next useful lever is persistent graph replay with dynamic decode state
  updated on-device.

## Next Step

Create the next sprint around persistent graph replay:

- enable the persistent graph serving flag at the same 8-slot / 256K shape;
- verify whether the typed-history final boundary fixes the previous parity
  failure there;
- if replay still does not occur, fix the persistent cache eligibility/keying;
- if replay occurs but parity fails, use the stage checksum gates from Sprint
  466 to localize the first replay-only divergence.
