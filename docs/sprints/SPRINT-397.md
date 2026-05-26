# Sprint 397: NCCL Serving Compose Gate

Date: 2026-05-26

## Goal

Bring the Sprint 396 NCCL collective evidence into the TP/EP serving harness
without touching PP/layer-split variants. The target was the EP compose boundary
where each rank produces destination shards and the destination rank needs the
sum over source ranks.

## Implementation

- Added `--nccl-reduce-scatter-compose-gate` to
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Added `DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE` to
  `tools/ds4-v100-run-appliance.sh`.
- Added `--nccl-reduce-scatter-compose` to
  `tools/ds4-v100-tp-ep-profile.py`.
- Linked `tools/ds4-v100-tp-ep-full-layer-smoke` with `-lnccl`.
- The gate only initializes NCCL for the compatible non-compact FP32 compose
  path. It stays inactive for the production compact route compose path because
  compact compose is route-indexed and is not equivalent to a plain
  reduce-scatter.

## V100 Validation

Build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: build passed on the V100 pod.

Single-layer non-compact compose A/B, layer 2, slots 32, position 262080:

| Mode | NCCL active | Compose ms | Checksum | Result |
|---|---:|---:|---:|---|
| Peer-copy sum8 | 0 | 2.521989 | 1908166124 | PASS |
| NCCL reduce-scatter | 1 | 6.401091 | 1908166124 | PASS |

Compact route compatibility smoke:

| Mode | Compact compose | NCCL active | Compose ms | Result |
|---|---:|---:|---:|---|
| Gate requested | 1 | 0 | 2.617325 | PASS |

Longer non-compact all-layer control, 32 slots, 256K position, 2 generated
tokens:

```text
aggregate_generated_tok_s_decode 246.390241
aggregate_continuation_tok_s_decode 293.155203
sum_compose_ms 39.927698
sum_compose_copy_ms 20.109601
PASS
```

The matching all-layer NCCL run was stopped after the single-layer test showed
the backend was slower for this boundary and after confirming compact serving
would not use it.

## Decision

Do not promote NCCL compose. Keep it as a default-off diagnostic backend.

Sprint 396 showed NCCL is clearly better for dense collective workbench
patterns, but the current production serving compose path is compact
route-indexed EP traffic rather than a dense all-reduce/reduce-scatter. For the
compatible non-compact boundary, NCCL preserved correctness but was slower than
the existing peer-copy plus fused compose kernel at the tested shape.

## Next

Keep NCCL available for future true TP hidden-state or expert-parallel
collectives, but do not spend more time forcing it into compact EP compose. The
next performance work should stay on TP/EP and target a boundary that is both
serving-visible and currently hot: HC-current orchestration, compact route
compose fusion, or the true TP/EP all-reduce boundary once dense/expert TP is
introduced.
