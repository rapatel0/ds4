---
sprint: 034
title: V100 Resident MTP Prefix Composition Probe
status: completed
date: 2026-05-19
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
vision: VISION.md
verdict: SHIP
---

# SPRINT-034: V100 Resident MTP Prefix Composition Probe

## Overview

Sprint 034 extends the Sprint 033 resident MTP projection probe into the full
native MTP prefix composition chain on gpu7. Sprint 033 proved that
`mtp.0.e_proj.weight` and `mtp.0.h_proj.weight` can execute from the compact
sidecar arena with Q8_0 parity. This sprint adds the F32 norm-weight views and
the HC composition steps needed to produce `mtp_input_hc`.

The target sequence mirrors the native graph in `ds4.c`:

1. normalize an embedding-like row with `mtp.0.enorm.weight`;
2. project it through `mtp.0.e_proj.weight`;
3. repeat the projection over the 4 HC rows;
4. normalize a previous `[4 x 4096]` HC state with `mtp.0.hnorm.weight`;
5. project those rows through `mtp.0.h_proj.weight`;
6. add the two HC tensors into the resident MTP prefix input.

## Outcome Contract

- `SHIP`: the repo has arena-resident F32 norm-weight execution, MTP F32 view
  helpers, an extended prefix smoke that validates the complete resident prefix
  chain against the existing GGUF-backed CUDA path on the V100 pod, and the
  full gate keeps passing while readiness remains blocked on full
  `mtp_forward`.
- `EXTEND`: helpers and smoke compile, but V100 cluster parity evidence or gate
  wiring is incomplete.
- `STOP`: resident prefix composition fails parity, overfills gpu7 reserve, or
  regresses the existing base/MTP residency gates.

## Non-Goals

- No dense MTP block attention or FFN execution.
- No MTP Q4_K routed expert execution.
- No MTP output-head logits/top-k parity.
- No speculative draft acceptance, verifier rollback, serving enablement, or
  multi-slot MTP scheduling.
- No production API change.

## Implementation

### Phase 1: Resident F32 MTP Views

- [x] Add a sidecar helper that converts a named 1D F32 MTP tensor into a
  `ds4_gpu_source_row_view`.
- [x] Validate dtype, dimensionality, byte length, source range, resident
  range, and 4-byte alignment.
- [x] Cover `mtp.0.enorm.weight` and `mtp.0.hnorm.weight` in the prefix smoke.

### Phase 2: Arena F32 RMSNorm Adapter

- [x] Add `ds4_gpu_arena_f32_rms_norm_f32` to the public arena API.
- [x] Implement the CUDA path by reusing the existing F32 weighted RMSNorm
  kernel against arena-resident weight bytes.
- [x] Add a fail-closed non-CUDA stub.
- [x] Preserve the existing arena API convention: `0` means success, `1` means
  failure.

### Phase 3: Full Prefix Chain Smoke

- [x] Extend `tools/ds4-v100-mtp-prefix-smoke` to run both standalone Q8_0
  projection parity and the full resident prefix chain.
- [x] Use deterministic F32 activation inputs so the smoke is stable and does
  not require the base model embedding tensor.
- [x] Compare resident Q8_0 projections against the existing CUDA model-map
  projection path, and compare the full chain against an independent CPU
  F32/Q8_0 reference.
- [x] Report intermediate and final max-abs/max-rel deltas, dimensions,
  timings, and gpu7 reserve evidence.

### Phase 4: Gate, Vision, And Evidence

- [x] Keep `tools/ds4-v100-gate.sh` wired through the existing `mtp_prefix`
  gate, now backed by full prefix-chain validation.
- [x] Run local syntax/build checks.
- [x] Build and run the focused prefix smoke on the V100 pod with
  `CUDA_ARCH=sm_70`.
- [x] Run the full 8-GPU V100 gate and preserve `failures=0` with
  `missing=mtp_forward`.
- [x] Write a Sprint 034 report/follow-ups, update the vision readiness ladder,
  and commit implementation plus evidence.

## Parallel Work

- Explorer 1: read-only native MTP graph mapping, especially prefix order,
  tensor names, dimensions, and eps values.
- Explorer 2: read-only GPU kernel surface inventory for RMSNorm, repeat, add,
  Q8_0, and upcoming Q4_K work.
- Mainline implementation owns the arena F32 API, smoke extension, gate
  validation, docs, and commit.

## Definition Of Done

- Local syntax/build checks pass.
- CUDA build target `tools/ds4-v100-mtp-prefix-smoke` builds on the V100 pod
  for `CUDA_ARCH=sm_70`.
- The prefix smoke passes on gpu7 with the real MTP sidecar, validates the full
  prefix chain, and preserves the configured 4096 MiB reserve.
- The full V100 gate passes with zero failures and still lists only full
  `mtp_forward` as the readiness blocker.
- Sprint 034 report, follow-ups, vision update, and git commit are complete.
