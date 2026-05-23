# Sprint 274 - TP/EP Resident Serving Loop

Date: 2026-05-23
Status: Complete

## Overview

Sprint 274 turns the serving metric bridge into a resident TP/EP serving loop.
Sprint 273 showed good decode-only rates but terrible wall throughput because
token-major serving still called the heavy per-layer `run_layer()` scaffold for
every token/layer invocation.

## Implementation

`--serving-bench` now uses a direct resident decode loop when the required
state is available:

- per-layer contract rows are parsed once
- shared TP runtime is reused
- shared rank buffers are reused
- resident expert bindings are reused
- shared dense cache is reused
- optional shared dense ops are reused
- per-token/per-layer `run_layer()` setup is bypassed

Serving-bench mode also skips validation checksum readback by default. Strict
checksum validation remains available outside serving-bench mode.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint274-resident-serving-loop/cluster/resident-serving-32req-16tok.log`
- `logs/from-cluster/sprint274-resident-serving-loop/cluster/resident-serving-skip-checksum-32req-16tok.log`
- `logs/from-cluster/sprint274-resident-serving-loop/cluster/resident-serving-shared-dense-ops-32req-16tok.log`
- `logs/from-cluster/sprint274-resident-serving-loop/cluster/resident-serving-shared-dense-ops-32req-32tok.log`

Key results at `32` slots / `256K`:

| Mode | Generated/request | Generated tok/s wall | Continuation tok/s wall | Generated tok/s decode | Continuation tok/s decode |
|---|---:|---:|---:|---:|---:|
| Direct resident loop, local dense ops | 16 | 14.423455 | 14.444535 | 888.254567 | 952.278658 |
| Direct resident loop, checksum skipped | 16 | 14.514365 | 14.528157 | 879.985153 | 946.196617 |
| Direct resident loop, shared dense ops | 16 | 712.984100 | 763.961099 | 912.435839 | 990.053318 |
| Direct resident loop, shared dense ops | 32 | 669.222644 | 690.469286 | 876.524260 | 910.270244 |

Best current TP/EP serving-shaped result:

```text
32 requests / 32 slots
256K context
32 generated tokens/request
1024 generated tokens total
992 continuation tokens total

wall generated tok/s:       669.222644
wall continuation tok/s:    690.469286
decode generated tok/s:     876.524260
decode continuation tok/s:  910.270244
```

## Decision

The full resident serving loop is now operational enough for useful TP/EP
metrology. Shared dense ops are required for wall-time serving because local
dense-op prepare/free dominates otherwise. This reverses the earlier
token-major scaffold decision: shared dense ops remain bad for isolated proxy
timing, but they are required for operational serving-loop wall throughput.

Next step: wrap this resident serving loop in the HTTP sustained-decode
harness so generated and continuation tok/s can be measured with the same
load client and operational reports as the frozen PP appliance.
