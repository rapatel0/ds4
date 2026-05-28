# TEMP Status Report 068 - Sprint 356

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 356 made the compressed-fusion gates usable from the serving
launcher and tested combined fused input-fill plus fused pool+norm in the direct
emitted-row profile shape.

## Implemented

- Added launcher env vars:
  - `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL`
  - `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND`
  - `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM`
- Added launcher validation and command propagation.
- Added HTTP profile harness propagation for the same variables.
- Added `DS4_V100_TP_EP_POSITION` propagation in HTTP profile mode.
- Updated `deploy/v100/ds4-v100-appliance.env.example`.

## Validation

Local:

- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`: pass
- `bash -n tools/ds4-v100-run-appliance.sh`: pass
- `git diff --check`: pass

V100:

- launcher `--print-command` includes fused input-fill and fused pool-norm
  flags
- `sm_70` build passes with only known unused-kernel warnings

Direct emitted-row A/B:

| Variant | Fused input layers | Fused pool layers | Decode tok/s | Decode ms | Pre-EP compressed-KV ms | First token |
|---|---:|---:|---:|---:|---:|---:|
| control | `0` | `0` | `80.511365` | `397.459414` | `130.812593` | `54639` |
| input-fill + pool-norm | `21` | `41` | `81.311102` | `393.550195` | `128.988052` | `54639` |

## Decision

Keep the gates opt-in. The combined candidate is positive but small. The next
step should be either repeated direct A/B or an emitted-row HTTP/profile mode
that avoids prompt-prefill position ambiguity.

## Artifacts

```text
logs/from-cluster/sprint356-combined-compressed-fusions/cluster/
```
