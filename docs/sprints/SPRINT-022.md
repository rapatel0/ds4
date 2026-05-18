---
sprint: 022
title: Bias Router And Resident Stage Scheduler
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-022-REPORT.md
followups: SPRINT-022-FOLLOWUPS.md
verdict: SHIP
---

# SPRINT-022: Bias Router And Resident Stage Scheduler

## Overview

Sprint 022 moves from a single representative layer toward appliance
scheduling. The sprint removes the hash-router-only executor limit, validates a
real bias-router layer, and adds a reusable resident stage scheduler that owns
one GPU shard, per-layer decode caches, HC ping-pong tensors, and a stage-local
multi-layer decode walk.

The target stage is gpu0 because it contains token embedding plus layers 0-5:
two SWA-only hash-router layers, two ratio-4 layers, and two ratio-128
bias-router layers.

## Outcome Contract

- `SHIP`: bias-router layers execute through the common layer executor; the V100
  gate validates layer 2 hash-router, layer 3 bias-router, and a resident
  stage-0 HC walk over layers 0-5 from a token embedding seed.
- `EXTEND`: bias-router support ships, but resident multi-layer scheduling is
  blocked by a concrete VRAM, arena, or cache ownership issue.
- `STOP`: bias-router or resident stage scheduling cannot be reconciled with
  the current source pack layout without replacing the layer executor contract.

## Non-Goals

- No 43-layer cross-GPU selected-token claim.
- No output-head selected-token gate.
- No public server.
- No MTP.
- No throughput benchmark.
- No tensor-parallel output-head or expert sharding.

## Parallel Workstreams

| Lane | Responsibility | Write Scope | Validation |
|---|---|---|---|
| A: router unblock | Generalize executor routing from hash-only to hash or bias metadata. | `ds4_v100_layer_execute.c` | layer 2 and layer 3 integrated smokes |
| B: CPU smoke oracle | Add bias-router CPU reference selection and weights. | `tests/cuda_v100_integrated_layer_smoke.c` | selected experts and route weights |
| C: resident scheduler | Add stage scheduler with full stage arena upload and per-layer caches. | `ds4_v100_scheduler.*`, stage smoke | stage 0 layers 0-5 on V100 |
| D: gate evidence | Add bias and stage scheduler checks to the V100 gate. | `tools/ds4-v100-gate.sh`, docs | full gate with `failures=0` |

## Implementation

### Phase 1: Bias Router Execution

**Files:**
- `ds4_v100_layer_execute.c`
- `tests/cuda_v100_integrated_layer_smoke.c`

**Tasks:**
- [x] Replace the hash-router-only executor guard with hash/bias mode
      selection from `ds4_v100_layer_state`.
- [x] Pass the correct bias or hash source offsets into
      `ds4_gpu_router_select_tensor`.
- [x] Add CPU bias-router reference logic using
      `sqrt(softplus(logit)) + exp_probs_b` for top-6 selection and raw
      probability normalization for route weights.
- [x] Validate a real ratio-128 bias-router layer on V100.

### Phase 2: Resident Stage Scheduler

**Files:**
- `ds4_v100_scheduler.h`
- `ds4_v100_scheduler.c`
- `tests/cuda_v100_stage_scheduler_smoke.c`

**Tasks:**
- [x] Add a reusable stage scheduler object for the token-embedding stage.
- [x] Open the real V100 context and pack index.
- [x] Allocate one full resident gpu0 arena and upload all gpu0 pack entries.
- [x] Initialize layer states for every stage-local layer.
- [x] Allocate executor-owned raw/compressed/indexer decode-cache tensors for
      each local layer.
- [x] Seed HC from `token_embd.weight` and run layers 0-5 through
      `ds4_v100_layer_execute_hc_decode`.
- [x] Read back final HC and validate finite nonzero output.

### Phase 3: Gate

**Files:**
- `tools/ds4-v100-gate.sh`
- `docs/sprints/SPRINT-022-REPORT.md`
- `docs/sprints/SPRINT-022-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [x] Add `integrated_layer_bias` to the appliance gate.
- [x] Add `stage_scheduler` to the appliance gate.
- [x] Run local compile checks.
- [x] Run full V100 gate with `--build`.
- [x] Record sprint evidence and remaining blockers.

## Outcome

`SHIP`.

The executor now supports both router families needed for the model, and the
runtime has a first resident multi-layer scheduler path. Stage 0 loads
22,524,130,064 bytes from 173 source tensors into the gpu0 arena and executes
layers 0-5 from a token embedding seed on a V100.

The full V100 gate passes with `ready=false` because cross-GPU 43-layer
scheduling, real selected-token decode, public serving, MTP, and throughput
benchmarking remain intentionally outside this sprint.

## Definition Of Done

- Layer 2 hash-router integrated smoke still passes.
- Layer 3 bias-router integrated smoke passes.
- Stage scheduler executes layers 0-5 from resident gpu0 pack bytes.
- Full V100 gate passes and includes the new bias-router and stage-scheduler
  checks.
