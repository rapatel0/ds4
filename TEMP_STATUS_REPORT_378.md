# TEMP Status Report 378: Compact MoE Decode Gate

Date: 2026-05-25

## Current Focus

Sprint 378 implements `--compact-moe-decode-gate`, the next Vision gate after
Sprint 377.

The concrete first target is model-router compatibility with compact EP
compose. Current `--compact-route-compose` only supports one route index per
source-rank/slot. Real model-router top-k can select multiple experts on the
same rank for the same slot, so the current route uploader rejects that shape
and forces model-router runs to use the larger all-destination contribution
path.

## Planned Gate

```text
--compact-moe-decode-gate
DS4_V100_TP_EP_COMPACT_MOE_DECODE=1
tools/ds4-v100-tp-ep-profile.py --compact-moe-decode
```

## Initial Implementation Target

- Add bounded per-source-rank, per-slot route index lists:

```text
route_indices_by_slot[src_rank][slot][k]
route_count_by_slot[src_rank][slot]
```

- Use those lists in a multi-route compact compose kernel.
- Allow `--model-router-routes --compact-route-compose` only when the new gate
  is enabled.

## Validation Target

Direct V100 A/B first:

```text
control:   --model-router-routes, compact compose off
candidate: --model-router-routes --compact-route-compose --compact-moe-decode-gate
```

Then HTTP serving A/B at `32` active requests / `32` slots / `256K` /
`position=262080` / `32` generated tokens/request with GPU sampling.

## Current Status

Implemented, built, and validated on the V100 pod.

Cluster artifact root:

```text
/workspace/logs/sprint378-compact-moe
```

Build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS on gpu-01 after two compact-path fixes:

- shared-rank cleanup double-freed `d_route_index_by_slot`;
- compact model-router layers with zero-route source ranks attempted zero-byte
  peer copies, and remote compact buffers were initially sized for `slots`
  instead of `slots * top_k`.

## Direct A/B

Shape:

```text
direct-token-major
32 slots
256K context
position 262080
1 generated token
model-router routes
```

| Mode | Return | First token | Direct decode tok/s | Compose ms | EP ms | Checksum |
|---|---:|---:|---:|---:|---:|---:|
| control: model-router, compact compose off | 0 | 54639 | 62.617354 | 26.025358 | 20.965788 | 6840320333 |
| candidate: compact-MoE compose | 0 | 54639 | 66.481242 | 20.993585 | 17.952991 | 6840320333 |

Direct verdict: candidate is correct for this gate and improves the decode
proxy by `+6.17%`, compose by `-19.34%`, and EP by `-14.37%`.

## HTTP Serving A/B

Shape:

```text
/v1/chat/completions
32 concurrent requests
32 configured slots
256K context
position 262080
32 generated tokens/request
GPU sampling: 1000 ms
model-router routes
```

| Mode | HTTP 200 | Response token stream | Client tok/s | Server decode tok/s | Avg GPU util | Compose ms | EP ms |
|---|---:|---|---:|---:|---:|---:|---:|
| control: compact compose off | 32/32 | matched | 37.394075 | 80.812914 | 8.385417% | 19.167728 | 11.236307 |
| candidate: compact-MoE compose | 32/32 | matched | 39.034685 | 81.313535 | 8.559783% | 14.703119 | 11.221664 |

HTTP verdict: candidate preserves the generated response token stream and
improves client throughput by `+4.39%`, server decode by `+0.62%`, average GPU
utilization by `+0.17 percentage points`, and compose time by `-23.29%`.

Route-shape evidence from the candidate:

```text
duplicate_slots=64
max_same_rank_routes=2
all_dest_bytes=4194304
compact_bytes=3145728
```

## Decision

PROMOTE for the real model-router compact-compose path.

The global default remains unchanged because model-router routing itself is
still explicitly selected. When model-router routes are enabled, this gate is
the production-compatible compact EP return path and should be used for the
next serving/performance work.

## Next

Proceed to S-E `--fused-gated-silu-gate`, focused on removing the routed-FFN
clamp/SwiGLU launch and intermediate while preserving the same model-router
serving parity checks.
