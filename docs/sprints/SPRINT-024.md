---
sprint: 024
title: Full 8-Stage Scheduler Chain
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-024-REPORT.md
followups: SPRINT-024-FOLLOWUPS.md
verdict: SHIP
---

# SPRINT-024: Full 8-Stage Scheduler Chain

## Overview

Sprint 024 extends the resident scheduler from the first two stages to the
full 8-GPU layer chain. The target is a single-slot decode body walk:
stage 0 seeds HC from token embedding, each stage executes its owned layers,
HC is handed off to the next GPU, and stage 7 produces a final finite HC state
after layer 42.

This sprint intentionally stops before output-head selected-token comparison.
The result is the first full-model body traversal on the V100 appliance path.

## Outcome Contract

- `SHIP`: all 8 stages open resident arenas, execute all 43 layers, peer-copy
  HC across every stage boundary, and pass in the full V100 appliance gate.
- `EXTEND`: partial stage chaining works but one or more later stages expose a
  CUDA/cache/topology issue requiring targeted follow-up.
- `STOP`: full-chain scheduling cannot fit in 32 GB V100 VRAM or cannot run
  safely under the current device/cache model.

## Non-Goals

- No output-head selected-token claim.
- No public server.
- No MTP.
- No throughput benchmark.
- No multi-slot wavefront.
- No relay-stream optimization.

## Workstreams

| Lane | Responsibility | Write Scope | Validation |
|---|---|---|---|
| A: full-chain smoke | Add an 8-stage scheduler smoke over all layer owners. | `tests/cuda_v100_full_scheduler_smoke.c` | standalone V100 run |
| B: build/gate | Add the full-chain smoke to CUDA build and the appliance gate. | `Makefile`, `tools/ds4-v100-gate.sh` | full V100 gate |
| C: readiness policy | Remove `full_43_layer_scheduler` only when the full-chain gate passes. | `tools/ds4-v100-gate.sh`, docs | gate summary |

## Implementation

### Phase 1: Full-Chain Smoke

**Files:**
- `tests/cuda_v100_full_scheduler_smoke.c`

**Tasks:**
- [x] Open stages 0-7 with resident arenas.
- [x] Decode stage 0 from token embedding.
- [x] Handoff HC stage-to-stage through gpu7.
- [x] Decode stages 1-7 from incoming HC.
- [x] Verify every stage executed its assigned layer range.
- [x] Verify 43 total layers executed.
- [x] Read final stage HC and validate finite nonzero output.

### Phase 2: Build And Gate Integration

**Files:**
- `Makefile`
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Add CUDA and Darwin-stub build rules.
- [x] Add clean target coverage.
- [x] Add `full_scheduler` to the V100 appliance gate.
- [x] Gate readiness dynamically keeps or removes
  `full_43_layer_scheduler` based on the full-chain result.

## Outcome

`SHIP`.

The V100 pod now executes a single-slot decode body through all 43 layers
across all 8 V100s with resident stage arenas and peer HC handoffs. The full
V100 appliance gate passes and readiness now lists only the remaining product
milestones: real-model selected token, public serving, MTP, and throughput.

## Definition Of Done

- Stage-0 scheduler regression still passes.
- Two-stage scheduler regression still passes.
- Full scheduler executes 43 layers across 8 V100s.
- Full scheduler keeps all observed resident footprints below 32 GB per GPU.
- Full V100 gate passes.
- Readiness no longer lists `full_43_layer_scheduler`.
