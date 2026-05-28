# TEMP Status Report 397

Date: 2026-05-26

## Current Focus

TP/EP only. No PP/layer-split work.

This pass focused on NCCL integration for the serving harness, specifically the
EP compose boundary.

## Implemented

- Added `--nccl-reduce-scatter-compose-gate`.
- Added launcher env:
  `DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE`.
- Added profile harness flag:
  `--nccl-reduce-scatter-compose`.
- Linked `tools/ds4-v100-tp-ep-full-layer-smoke` with `-lnccl`.
- NCCL only initializes for non-compact FP32 compose. It stays inactive for the
  current production compact route compose path.

## Results

Build on V100: PASS.

Single-layer non-compact compose, layer 2, 32 slots, position 262080:

| Mode | Compose ms | Checksum | Result |
|---|---:|---:|---|
| Peer-copy fused compose | 2.521989 | 1908166124 | PASS |
| NCCL reduce-scatter | 6.401091 | 1908166124 | PASS |

Compact route compose with the NCCL gate requested:

| Mode | NCCL active | Compose ms | Result |
|---|---:|---:|---|
| Compact route | 0 | 2.617325 | PASS |

Non-compact all-layer control, 32 slots, 256K position, 2 generated tokens:

```text
aggregate_generated_tok_s_decode 246.390241
aggregate_continuation_tok_s_decode 293.155203
sum_compose_ms 39.927698
sum_compose_copy_ms 20.109601
PASS
```

## Decision

Do not promote NCCL compose. Keep it default-off as a diagnostic backend.

The current production compose path is compact, route-indexed EP traffic. A
dense reduce-scatter is not the same operation. On the compatible non-compact
path, NCCL is correct but slower at the tested serving shape.

## Next

Continue TP/EP performance work, but not by forcing NCCL into compact EP
compose. Better next targets:

- HC-current orchestration and fill/pack cost.
- compact route compose fusion.
- true TP hidden/expert collective boundaries when dense/expert TP work resumes.
