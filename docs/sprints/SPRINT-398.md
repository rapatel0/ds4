# Sprint 398: HC-Current Fill/Pack Fusion Gate

Date: 2026-05-26

## Overview

Stay on the TP/EP serving path and target a serving-visible hot boundary after
Sprint 397 rejected NCCL compose promotion. The next measured bottleneck is
HC-current orchestration, especially the fill/pack tail that prepares per-rank
dense inputs and routed FFN inputs after the model-router route plan is known.

This sprint adds a default-off fused fill/pack gate. It is intentionally narrow:
one kernel per rank replaces the current per-rank sequence of current-full copy,
dense input fills, half input fills, optional routed-normalized current copy, and
route packing.

## Constraints

- TP/EP only. No PP/layer-split work.
- No generic scheduler abstraction.
- Default off until V100 same-binary A/B proves it.
- Preserve DS4 quality gates: first token, response parity, checksum/readiness.
- Keep `32` slots and `256K` context as the target serving shape.
- Use the V100 pod for build and benchmark.

## Implementation

Files:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-tp-ep-profile.py`

Planned changes:

1. Add `--tp-hc-current-input-fused-fill-pack-gate`.
2. Add launcher/profile wiring:
   `DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK`.
3. Add a fused kernel for the production unscaled path:
   - writes `r.d_current_full`
   - fills `attn_op.d_x` / `shared_op.d_x` when present
   - fills `attn_op.d_x_half` / `shared_op.d_x_half` when present
   - packs `r.d_a` from route slots
   - preserves `routed_ffn_norm_input_gate` semantics by using
     `hc->d_ffn_normed` for route packing and final `r.d_current_full` when
     that gate is active
4. Fall back to the existing path for reference-HC scaled route packing and any
   shape the fused kernel does not cover.

## Validation

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Direct same-binary A/B:

- `32` slots
- `256K` context / `position=262080`
- model-router compact-MoE
- route-plan async upload enabled
- current promoted defaults

HTTP same-binary A/B if the direct run is correct and non-regressing:

- `32` requests
- `32` slots
- `256K`
- `32` generated tokens
- response parity checker
- readiness checker
- GPU utilization sampling

## Definition of Done

- Gate exists in binary, launcher, and profile harness.
- V100 build passes.
- Direct A/B records:
  - decode tok/s
  - HC-current fill/pack ms
  - checksum or first-token evidence
  - pass/fail
- If direct A/B is positive, HTTP A/B records:
  - server decode tok/s
  - client generated tok/s
  - response parity
  - readiness
  - GPU utilization
  - VRAM admission
- Sprint doc, status report, and vision/status updates record an explicit
  PROMOTE or REJECT decision.
- Commit all kept artifacts explicitly.

## Risks

- Direct peer/UVA reads from GPU0 buffers may be slower than explicit peer copy
  plus local reads.
- Fusing can accidentally change `routed_ffn_norm_input_gate` semantics by
  leaving `r.d_current_full` with the wrong source.
- The current production hot path may be dominated by synchronization around the
  boundary rather than launch count inside it; in that case the gate will be
  correct but flat.

## Outcome

Implemented the gate in the binary, launcher, and profile harness:

- `--tp-hc-current-input-fused-fill-pack-gate`
- `DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK`
- `tools/ds4-v100-tp-ep-profile.py --hc-current-fused-fill-pack`

The fused kernel preserves current dense-input semantics and routed-input
semantics separately: dense fills read `hc->d_current_full`, while routed pack
reads `hc->d_ffn_normed` only when `routed_ffn_norm_input_gate` is active.
The reference scaled route-pack path, graph-event path, and peer-gather path
fall back to the existing implementation.

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Direct same-binary A/B at `32` slots / `256K` / `position=262080` /
model-router compact-MoE / route-plan async upload:

| Mode | First token | Generated decode tok/s | Continuation decode tok/s | HC fill/pack ms | HC-current ms | Total decode ms | VRAM failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 54639 | 87.759480 | 97.163622 | 28.140415 | 599.389235 | 729.265944 | 0 |
| fused fill/pack | 54639 | 64.310075 | 67.874579 | 320.439853 | 869.514336 | 995.178434 | 0 |

Artifacts:

- `logs/from-cluster/sprint398-hc-fused-fill-pack/direct-control/`
- `logs/from-cluster/sprint398-hc-fused-fill-pack/direct-candidate/`

## Decision

REJECT as a default.

The gate is correct on first-token evidence and VRAM admission, but the
direct peer/UVA remote-load design is much slower than explicit peer copy plus
local reads. HC-current fill/pack increased by `292.299438` ms across the
2-token all-layer run, and generated decode regressed by `26.7%`.

Keep the gate as an opt-in diagnostic only. Future work should not fuse this
boundary by replacing local staging with direct remote loads. If this area is
revisited, the likely useful direction is local-staging-preserving fusion or
fusion with the downstream dense/expert consumers so the staged data is reused
without adding remote load pressure.
