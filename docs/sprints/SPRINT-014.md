---
sprint: 014
title: V100 Real Pack-Index Layer Descriptor Gate
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-014-INTENT.md
deferred: SPRINT-014-DEFERRED.md
verdict: SHIP
---

# SPRINT-014: V100 Real Pack-Index Layer Descriptor Gate

## Overview

Sprint 013 proved bounded synthetic MXFP4 MoE and selected-token composition on
V100. Sprint 014 moves from synthetic fixtures toward real model execution by
validating the exact pack-index descriptors needed for a real source-layout
layer.

This sprint should not unlock serving or run a full real layer. It should
produce a strict descriptor contract that later compute code can consume.

## Outcome Contract

- `SHIP`: a descriptor gate validates the real pack index for layer 2, covers
  attention, compressor/indexer, router, routed/shared experts, HC controls, and
  output head, rejects incomplete indexes, and is wired into the appliance gate
  behind `--pack-index`.
- `EXTEND`: the standalone descriptor gate ships, but appliance-gate wiring or
  cluster validation needs a documented follow-up.
- `STOP`: implementation requires public serving unlock, full real-layer
  compute, large weight upload, MTP, throughput tuning, or ambiguous descriptor
  naming that cannot be resolved from the pack index.

## Non-Goals

- No normal source-layout generation/server unlock.
- No real layer compute execution.
- No throughput benchmark.
- No MTP/speculative decoding.
- No production kernel selection.

## Implementation

### Phase 1: Descriptor Gate Tool

**Files:**
- `tools/ds4-v100-layer-descriptor-gate.c`
- `Makefile`

**Tasks:**
- [x] Parse a `pack-index.tsv` using `ds4_pack_open`.
- [x] Validate required descriptors for layer 2.
- [x] Check source dtype, runtime layout, kernel family, layer id, owning GPU,
      shard file, shard offset, and byte length.
- [x] Print stable descriptor and summary rows.

### Phase 2: Negative Coverage

**Files:**
- descriptor gate tool or shell smoke
- sprint logs

**Tasks:**
- [x] Run against the committed real pack-index fixture.
- [x] Run against an intentionally incomplete synthetic index and verify
      fail-closed behavior.

### Phase 3: Appliance Gate Integration

**Files:**
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Add `--pack-index FILE`.
- [x] Build/run the descriptor gate when `--pack-index` is supplied.
- [x] Preserve current gate behavior when no pack index is supplied.

### Phase 4: Validation And Closeout

**Files:**
- `docs/sprints/drafts/SPRINT-014-*.log`
- `docs/sprints/SPRINT-014-REPORT.md`
- `docs/sprints/SPRINT-014-FOLLOWUPS.md` if needed
- `docs/sprints/VISION.md`

**Tasks:**
- [x] Run local validation and `git diff --check`.
- [x] Run cluster validation with `--pack-index`.
- [x] Archive logs.
- [x] Write report/follow-ups and update vision.

## Definition Of Done

- Descriptor gate passes on the real pack-index fixture for layer 2.
- Descriptor gate fails on an incomplete synthetic index.
- Appliance gate accepts `--pack-index` and runs descriptor validation.
- Full V100 gate passes implemented checks and still reports not-ready for real
  serving.
- Report, logs, follow-ups/deferred notes, and vision update are committed.

## Risks

- Layer 2 descriptor strictness may miss ratio-128 or final-layer differences.
  That should be a follow-up, not a reason to avoid the first real descriptor
  contract.
- A descriptor gate does not prove math correctness. It proves the real model
  binding needed before math can safely consume the pack.
