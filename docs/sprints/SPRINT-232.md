# Sprint 232 - One-Layer TP/EP Correctness Gate

Date: 2026-05-23
Status: Planned

## Overview

Sprint 230 proved sharded TP KV row ownership. Sprint 231 proved bounded EP8
routed expert execution with real TurboMind MXFP4 kernels. Sprint 232 combines
those primitives into the first one-layer TP/EP fixture gate.

This sprint should still avoid the frozen PP scheduler. The purpose is to make
the per-layer TP/EP execution boundary concrete enough that the following
sprint can scale from a fixture layer to DS4 layer descriptors and then to all
43 layers.

## Goals

- Add a new TP/EP-only one-layer smoke tool.
- Own all eight V100s in one process.
- Allocate/use the separate TP runtime at the target `32` slots / `256K` /
  F8-KV shape.
- Execute a ratio-4 dense/KV slice for one layer.
- Execute an EP8 routed expert slice for the same active-slot shape:
  - `32` slots;
  - `top_k=6`;
  - `192` aggregate routes;
  - `256` global experts;
  - `32` local experts per GPU.
- Produce a deterministic next-hidden fixture per rank.
- Validate:
  - finite output;
  - deterministic repeat;
  - KV row update correctness;
  - EP route distribution accounting;
  - per-rank latency visibility.
- Report the one-layer timing envelope:
  - dense/KV slice time;
  - EP gate/up time;
  - EP down time;
  - EP worst-rank time;
  - modeled dispatch/return bytes.

## Non-Goals

- No PP scheduler changes.
- No generic PP/TP scheduler abstraction.
- No full DS4 layer descriptor loading.
- No logits comparison yet.
- No serving integration.
- No MTP.
- No production throughput claim.

## Implementation

1. Add `tools/ds4-v100-tp-ep-layer-smoke.cu`.
2. Link it with `ds4_v100_tp_runtime.cu`.
3. Reuse the public TurboMind C ABI for MXFP4 gated-SiLU/down.
4. Reuse the Sprint 230 TP runtime dense/KV API for the ratio-4 layer slice.
5. Reuse/adapt the Sprint 231 EP route plan and kernel invocation.
6. Add a Makefile target and local CUDA guard.
7. Build on the V100 pod with `CUDA_ARCH=sm_70`.
8. Run at `32` slots / `256K` / `top_k=6`.
9. Copy evidence to
   `logs/from-cluster/sprint232-tp-ep-layer-smoke/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-layer-smoke.cu` | one-layer TP/EP fixture gate |
| `ds4_v100_tp_runtime.h` | only if additional TP runtime reporting is needed |
| `ds4_v100_tp_runtime.cu` | only if additional TP runtime reporting is needed |
| `Makefile` | build target |
| `docs/sprints/SPRINT-232.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint232-tp-ep-layer-smoke/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] New smoke tool is TP/EP-only and does not modify PP scheduler files.
- [ ] Tool opens the separate TP runtime at `32` slots / `256K` / F8 KV.
- [ ] Tool updates/verifies a ratio-4 dense/KV row through the TP runtime.
- [ ] Tool runs real TurboMind MXFP4 grouped gated-SiLU and down kernels on all
      eight GPUs.
- [ ] Tool reports route distribution, dispatch bytes, return bytes, and
      per-rank latency.
- [ ] Tool validates finite deterministic repeat output.
- [ ] V100 build passes with `CUDA_ARCH=sm_70`.
- [ ] V100 smoke passes at `32` slots / `256K` / `top_k=6`.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint232-tp-ep-layer-smoke/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Risks

- This is still a fixture layer, not a DS4 descriptor-driven layer. It proves
  the execution boundary and lifecycle, not model equivalence.
- If rank-7 skew persists from Sprint 231, it may dominate the one-layer timing
  envelope. That should be recorded rather than averaged away.
- If linking the TP runtime and TurboMind ABI in one tool exposes lifecycle
  issues, the sprint should record the exact failure and fix the lifecycle
  boundary before adding more DS4 math.

## Decision

Pending.
