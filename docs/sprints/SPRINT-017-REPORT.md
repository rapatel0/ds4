---
sprint: 017
title: V100 Scheduler-Owned Layer State Gate
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-017 Report: V100 Scheduler-Owned Layer State Gate

## Verdict

`SHIP`

Sprint 017 moved the descriptor-bound router/FFN ownership out of the CUDA
smoke and into a reusable V100 layer-state API. The state binds real layer
descriptors once, validates router and FFN dimensions, carries layer/stage/KV
metadata, exposes source row views, constructs selected routed expert matrices,
and computes the FFN arena span used by the real-router FFN path.

The appliance still is not ready for serving. The next functional gap is full
layer output: attention, residual/norm, HC transforms, and then real-model
selected-token decode.

## What Shipped

- Added `ds4_v100_layer_state.h` and `ds4_v100_layer_state.c`.
- Added public helpers for bound matrices, source row views, selected expert
  route matrices, router kind, and FFN arena-span sizing.
- Added `tests/v100_layer_state_smoke.c`.
- Refactored `tests/cuda_v100_descriptor_bound_ffn_smoke.c` to use
  `ds4_v100_layer_state` for router/shared/routed views and arena sizing.
- Added `layer_state` to `tools/ds4-v100-gate.sh` when `--pack-index` is
  supplied.

## Evidence

Local validation:

- `make tests/v100_layer_state_smoke`
- `./tests/v100_layer_state_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --layer 2`
- `make tests/cuda_v100_descriptor_bound_ffn_smoke.o`
- `bash -n tools/ds4-v100-gate.sh`
- `git diff --check`

Cluster validation:

- `docs/sprints/drafts/SPRINT-017-LAYER-STATE-CLUSTER.log`
  - `v100_layer_state_smoke: layer=2 stage=0 gpu=0 class=ratio_4 router=hash hidden=4096 mid=2048 experts=256 span=11784434944 ok`
- `docs/sprints/drafts/SPRINT-017-ROUTER-FFN-CLUSTER.log`
  - `cuda_v100_descriptor_bound_ffn_smoke: layer=2 token=16 expert0=84 gpu=0 arena_bytes=11784434944 hidden=4096 mid=2048 ... ok`
- `docs/sprints/drafts/SPRINT-017-GATE-CLUSTER/gate-summary.log`
  - New `layer_state` gate passed.
  - Full gate summary: `PASS`, `failures=0`, `ready=false`.
  - Remaining readiness list:
    `full_layer_scheduler,attention_residual_norm,real_model_selected_token,public_serving,mtp,throughput_benchmark`.

## Deviations

- The state API owns descriptor metadata and arena-span planning, but not
  production arena allocation or persistent shard residency reuse.
- The gate still validates hash-router layer 2. Representative bias-router
  coverage remains a follow-up.
- The layer state carries KV metadata but does not execute attention or update
  the layer hidden state.

## Handoff

Sprint 018 should use `ds4_v100_layer_state` as the surface for a
descriptor-bound attention/residual/norm slice, or a narrow full-layer-output
slice if that can be kept bounded.
