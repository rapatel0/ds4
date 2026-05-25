---
sprint: 363
title: TP/EP Fused Compressed Pool Norm RoPE Round
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 363 - TP/EP Fused Compressed Pool Norm RoPE Round

## Overview

Sprint 359 promoted fused compressed pool+norm because direct long-context
decode showed a real win. The default emitted-row path still writes the pooled
normalized row, then launches separate RoPE and F16-round kernels. Sprint 354
showed RoPE+round alone was too narrow, but combining it with pool+norm removes
a larger state/emit boundary:

```text
pool state -> norm -> global row write -> rope row read/write -> round row read/write
```

becomes:

```text
pool state -> norm -> rope -> F16 round -> one global row write
```

This is TP/EP-only. No PP/layer-split work. No MTP.

## Implementation

1. Add a new opt-in TP/EP smoke gate:
   `--true-ds4-compressed-kv-fused-pool-norm-rope-round-gate`.
2. Add one CUDA kernel that fuses:
   - compressor state pooling,
   - RMSNorm with the compressor norm weight,
   - DS4 compressed-row RoPE on the rotary tail,
   - F16-saturating round,
   - final emitted-row write.
3. Apply the gate to both attention compressed rows and ratio-4 indexer rows.
4. Expose the gate through:
   - `tools/ds4-v100-run-appliance.sh`,
   - `deploy/v100/ds4-v100-appliance.env.example`,
   - `tools/ds4-v100-tp-ep-profile.py`.
5. Keep the existing pool+norm default unchanged. The new fused boundary is
   opt-in until direct and launcher-level evidence justify promotion.

## Verification

- Local syntax checks pass.
- V100 build passes:
  `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Direct emitted-row A/B at `32` slots / `256K` / `position=262112` /
  `32` decode steps:
  - control: launcher-default pool+norm,
  - candidate: pool+norm+RoPE+round fused.
- Both variants preserve finite output head and first selected token.
- Compare generated decode tok/s, wall tok/s, compressed-KV sum, and fused row
  counts.

## Definition of Done

- [x] The new fused kernel compiles for `sm_70`.
- [x] The gate is reachable from the direct smoke, launcher, and profile
      harness.
- [x] V100 direct A/B passes correctness invariants.
- [x] Results are summarized in this sprint doc and `docs/sprints/STATUS.md`.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

Implemented the fused emitted-row boundary:

- new CUDA kernel:
  `compressor_pool_norm_rope_round_emit_slots_kernel`,
- new direct/smoke gate:
  `--true-ds4-compressed-kv-fused-pool-norm-rope-round-gate`,
- new launcher env:
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM_ROPE_ROUND`,
- new profile harness flag:
  `--fused-compressed-pool-norm-rope-round`.

V100 build passed with:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Direct 32-step A/B at `32` slots / `256K` / `position=262112`:

| Variant | First token | Finite bad | Decode tok/s | Wall tok/s | Compressed-KV sum | Fused rows |
|---|---:|---:|---:|---:|---:|---:|
| pool+norm default | 98751 | 0 | 95.908399 | 75.035176 | 3460.932833 ms | 188 pool+norm |
| pool+norm+RoPE+round | 98751 | 0 | 95.463298 | 74.707227 | 3470.682826 ms | 188 fused |

One-token `nvprof-window-gpu-trace` at `position=262143`:

| Variant | First token | Decode tok/s | Wall tok/s | Compressed-KV sum | Top fused/kernel evidence |
|---|---:|---:|---:|---:|---|
| pool+norm default | 54639 | 64.213984 | 19.203528 | 142.456129 ms | `compressor_pool_norm_emit_slots_kernel`: 62 calls, 7.155846 ms |
| pool+norm+RoPE+round | 54639 | 64.735060 | 19.430130 | 140.699321 ms | `compressor_pool_norm_rope_round_emit_slots_kernel`: 62 calls, 8.286669 ms |

The one-token profiler window moves in the right direction at the stage level,
but the production-relevant 32-step direct run regresses slightly. This likely
means the fused kernel's extra math/register/shared-memory pressure offsets
the removed RoPE/round launches once the whole decode loop is amortized.

## Decision

Do not promote the new fused pool+norm+RoPE+round path. Keep it as an opt-in
diagnostic gate for future kernel-structure experiments.

The next implementation target should not be a wider emitted-row scalar fusion.
The larger remaining costs are dense projection and compressor/indexer dense
fragmentation. Future work should either:

1. attack the compressed attention/indexer dense projection path with a
   shape-specific HMMA/TurboMind-style kernel, or
2. reduce cross-rank gather/current-full staging before these dense calls.

Artifacts:

```text
logs/from-cluster/sprint363-fused-pool-norm-rope-round/
logs/from-cluster/sprint363-fused-pool-norm-rope-round-prof/
```

## Risks

- The fused kernel may increase register/shared-memory pressure enough to lose
  occupancy. If so, keep it diagnostic-only and use the profiling evidence to
  decide the next boundary.
- Ratio-128 attention rows and ratio-4 indexer rows have different head dims;
  the kernel must keep the head-dim and rotary-dim guards explicit.
