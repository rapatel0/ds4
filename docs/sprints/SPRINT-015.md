---
sprint: 015
title: V100 Descriptor-Bound FFN Compute Gate
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-015-INTENT.md
deferred: SPRINT-015-DEFERRED.md
verdict: SHIP
---

# SPRINT-015: V100 Descriptor-Bound FFN Compute Gate

## Overview

Sprint 014 proved that the real pack index contains the layer-2 descriptors
needed by the appliance path. Sprint 015 turns those descriptors into runtime
bindings and uses them to execute a bounded FFN slice from real source-model
bytes at real pack offsets.

This sprint should not unlock serving. It should remove the current gap between
"descriptor exists" and "descriptor can launch compute."

## Outcome Contract

- `SHIP`: runtime descriptor bindings exist, local binding validation passes,
  a CUDA smoke reads real model bytes through layer-2 pack descriptors,
  executes routed MXFP4 and shared F8 FFN paths on V100, compares against CPU
  source-format references, and the appliance gate runs the new smoke.
- `EXTEND`: descriptor bindings ship and are gate-integrated, but real-byte
  FFN compute needs a documented follow-up due cluster IO/runtime limits.
- `STOP`: real model bytes cannot be read safely from the source GGUF, arena
  offsets cannot be allocated within the V100 reserve, or descriptor shape
  semantics conflict with the existing source-format kernels.

## Non-Goals

- No public source-layout server unlock.
- No full layer with attention/residual/norm.
- No real router-selected expert scheduling.
- No output-head logits in the descriptor-bound FFN smoke.
- No MTP/speculative decoding.
- No throughput benchmark or production grouped expert kernel.

## Implementation

### Phase 1: Runtime Descriptor Binding API

**Files:**
- `ds4_v100_context.h`
- `ds4_v100_context.c`

**Tasks:**
- [x] Add a public `ds4_v100_tensor_binding` struct.
- [x] Add lookup helpers for semantic tensor id, layer tensor suffix, and
      output head.
- [x] Parse source shape dimensions into stable binding metadata.
- [x] Fail closed on missing tensors, bad layer ids, bad suffixes, or shapes
      with too many dimensions.

### Phase 2: Local Binding Smoke

**Files:**
- `tests/v100_layer_binding_smoke.c`
- `Makefile`

**Tasks:**
- [x] Open the real pack-index fixture through `ds4_v100_context`.
- [x] Materialize layer-2 routed expert, shared expert, router, HC, and
      output-head bindings.
- [x] Verify expected dtypes, owners, dimensions, offsets, and byte lengths.
- [x] Add a negative missing-tensor or bad-layer case.

### Phase 3: Descriptor-Bound CUDA FFN Smoke

**Files:**
- `tests/cuda_v100_descriptor_bound_ffn_smoke.c`
- `Makefile`

**Tasks:**
- [x] Load layer-2 routed MXFP4 gate/up/down expert bytes from the source GGUF
      using pack-index source offsets.
- [x] Load layer-2 shared F8 gate/up/down expert bytes from the source GGUF.
- [x] Upload bytes into a V100 arena at the real shard offsets from the pack
      descriptors.
- [x] Run routed MXFP4 gate/up/down plus SwiGLU for one fixed expert.
- [x] Run shared F8 gate/up/down plus SwiGLU.
- [x] Sum routed and shared FFN outputs and compare with CPU source-format
      references.

### Phase 4: Appliance Gate Integration

**Files:**
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Build the new local/CUDA smoke targets when `--build` is used.
- [x] Run local binding validation when `--pack-index` is supplied.
- [x] Run descriptor-bound FFN smoke when both `--pack-index` and `--model`
      are available.
- [x] Keep existing behavior unchanged when no pack index is supplied.

### Phase 5: Validation And Closeout

**Files:**
- `docs/sprints/drafts/SPRINT-015-*.log`
- `docs/sprints/SPRINT-015-REPORT.md`
- `docs/sprints/SPRINT-015-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [x] Run local validation and `git diff --check`.
- [x] Run cluster validation with `--model` and `--pack-index`.
- [x] Archive logs.
- [x] Write report/follow-ups and update vision.

## Definition Of Done

- Runtime descriptor binding API is committed and covered by a local smoke.
- The CUDA descriptor-bound FFN smoke passes on the V100 pod with the real
  source GGUF and real pack index.
- The appliance gate builds/runs the new binding and FFN checks when
  `--pack-index` is supplied.
- Full V100 gate passes implemented checks and still reports not-ready for
  serving.
- Report, logs, follow-ups/deferred notes, and vision update are committed.

## Risks

- The real FFN tensors are large enough that arena allocation must use the real
  stage arena size and preserve reserve headroom.
- The smoke validates one fixed expert and synthetic activation input; it does
  not prove router scheduling or end-to-end model quality.
- Source GGUF offsets must remain identical to pack-index source offsets. If
  the cluster model differs from the fixture, the smoke should fail closed.
