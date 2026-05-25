# TEMP Status Report 075 - Sprint 363 Fused Emit Boundary

Date: 2026-05-25

## Current Focus

TP/EP compressed-KV emitted-row fusion. Sprint 363 tested whether combining
pooling, RMSNorm, compressed-row RoPE, F16 rounding, and the final row write
would outperform the promoted pool+norm default.

## What Shipped

New opt-in diagnostic path:

- `--true-ds4-compressed-kv-fused-pool-norm-rope-round-gate`
- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM_ROPE_ROUND=1`
- `tools/ds4-v100-tp-ep-profile.py --fused-compressed-pool-norm-rope-round`

The production launcher default remains unchanged: pool+norm is still on,
while pool+norm+RoPE+round is off.

## Main A/B

Shape:

```text
run mode: direct-token-major
slots: 32
context: 256K
position: 262112
decode steps: 32
```

| Variant | First token | Finite bad | Decode tok/s | Wall tok/s | Compressed-KV sum | Fused rows |
|---|---:|---:|---:|---:|---:|---:|
| pool+norm default | 98751 | 0 | 95.908399 | 75.035176 | 3460.932833 ms | 188 |
| pool+norm+RoPE+round | 98751 | 0 | 95.463298 | 74.707227 | 3470.682826 ms | 188 |

## Profiler Window

One-token emitted-row `nvprof-window-gpu-trace` at `position=262143`:

| Variant | Decode tok/s | Wall tok/s | Compressed-KV sum |
|---|---:|---:|---:|
| pool+norm default | 64.213984 | 19.203528 | 142.456129 ms |
| pool+norm+RoPE+round | 64.735060 | 19.430130 | 140.699321 ms |

The profiler window shows the fused path can reduce a narrow emitted-row stage,
but the full 32-step serving-shaped run regresses. That full run is the
decision gate.

## Decision

Do not promote pool+norm+RoPE+round. Keep it diagnostic-only.

Next optimization should move upstream into compressed/indexer dense projection
or current/gather staging. Wider scalar emitted-row fusion is not the next
best lever.

Artifacts:

```text
logs/from-cluster/sprint363-fused-pool-norm-rope-round/
logs/from-cluster/sprint363-fused-pool-norm-rope-round-prof/
```
