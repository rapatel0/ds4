---
sprint: 033
title: V100 Resident MTP Q8 Projection Probe
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
vision: VISION.md
verdict: SHIP
---

# SPRINT-033: V100 Resident MTP Q8 Projection Probe

## Overview

Sprint 033 starts the Level 3 MTP-assisted correctness rung with a real
resident compute primitive. Sprint 031 proved that the MTP sidecar can be
uploaded into a compact gpu7 arena. This sprint proves that Q8_0 tensors inside
that arena can be used directly by V100 CUDA kernels and match the existing
GGUF-backed Q8_0 path.

The target is the MTP prefix projection surface: `mtp.0.enorm.weight`,
`mtp.0.e_proj.weight`, `mtp.0.hnorm.weight`, and `mtp.0.h_proj.weight` are the
first tensors consumed by the native MTP draft path before the dense MTP block.
The shipped gate should validate resident Q8_0 projection parity for those
projection tensors without enabling speculative serving.

## Outcome Contract

- `SHIP`: the repo has a resident arena Q8_0 matmul API, MTP sidecar Q8_0 view
  helpers, a V100 smoke that compares resident sidecar projection output against
  the existing mapped-model Q8_0 CUDA path, and the full gate reports the new
  MTP projection gate passing while readiness remains blocked on full
  `mtp_forward`.
- `EXTEND`: helpers and build targets exist, but cluster parity evidence is
  incomplete or the gate is not wired.
- `STOP`: resident Q8_0 parity fails on V100, memory use exceeds the gpu7
  reserve, or the new path regresses the existing base appliance gate.

## Non-Goals

- No speculative decode acceptance, rollback, or serving enablement.
- No full MTP block attention/FFN/output-head implementation.
- No MTP Q4_K routed expert execution in this sprint.
- No multi-slot MTP scheduling.
- No production API change.

## Implementation

### Phase 1: Resident Q8_0 Tensor Views

- [x] Expose safe MTP sidecar map/size accessors for parity testing.
- [x] Add a helper that converts a named sidecar Q8_0 tensor into a
  `ds4_gpu_source_row_view`.
- [x] Validate Q8_0 shape, row stride, arena bounds, and source bounds.

### Phase 2: Arena Q8_0 Matmul Kernel Adapter

- [x] Add `ds4_gpu_arena_q8_0_matmul_f32` to the public GPU arena API.
- [x] Implement the CUDA path by reusing the existing V100 Q8_0 prequantized
  DP4A kernels against an arena pointer instead of a mapped GGUF range.
- [x] Add a stub implementation that fails closed on non-CUDA local builds.
- [x] Support `n_tok >= 1` so the same primitive covers `e_proj` and HC
  projection rows.

### Phase 3: MTP Projection Parity Smoke

- [x] Add `tools/ds4-v100-mtp-prefix-smoke`.
- [x] Upload the real MTP sidecar to gpu7, build Q8_0 views for
  `mtp.0.e_proj.weight` and `mtp.0.h_proj.weight`, and allocate deterministic
  F32 activation tensors.
- [x] Compare resident-arena Q8_0 output against the existing
  `ds4_gpu_matmul_q8_0_tensor` path using the same sidecar tensor bytes.
- [x] Emit dimensions, byte counts, max error, and timing/evidence report.

### Phase 4: Gate And Vision

- [x] Add the new smoke to `tools/ds4-v100-gate.sh` when `--mtp-model` is
  supplied.
- [x] Keep readiness honest: if sidecar and residency pass but prefix parity
  does not, report `missing=mtp_prefix`; if prefix passes, continue to report
  `missing=mtp_forward`.
- [x] Add a Sprint 033 report and update the vision readiness ladder/current
  state.
- [x] Commit implementation, docs, and V100 evidence.

## Parallel Work

- One agent may continue read-only native MTP graph mapping.
- One agent may continue read-only kernel-surface inventory for Q8_0/Q4_K/F32
  sidecar execution.
- Mainline implementation owns the arena Q8_0 API, smoke, gate, and cluster
  validation.

## Definition Of Done

- Local syntax/build checks pass.
- CUDA build target `tools/ds4-v100-mtp-prefix-smoke` builds on the V100 pod
  for `CUDA_ARCH=sm_70`.
- The prefix smoke passes on gpu7 with the real MTP sidecar and preserves the
  configured 4096 MiB reserve.
- The full V100 gate passes with zero failures and lists only full
  `mtp_forward` as the MTP readiness blocker.
- Sprint 033 report, vision update, and git commit are complete.
