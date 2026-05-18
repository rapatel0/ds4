---
sprint: 016
title: V100 Descriptor-Bound Router FFN Gate
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-016 Report: V100 Descriptor-Bound Router FFN Gate

## Verdict

`SHIP`

Sprint 016 upgraded descriptor-bound FFN execution from a fixed expert to
model-selected routed experts. The V100 smoke now computes router logits from
real `ffn_gate_inp.weight` bytes in the GPU arena, selects experts through the
real layer-2 `ffn_gate_tid2eid` hash table, executes all six selected routed
MXFP4 experts plus the shared F8 expert, and compares the summed output against
CPU source-format references.

The appliance still is not ready for serving. Remaining gaps are full layer
scheduling, attention/residual/norm integration, selected-token real-model
decode, public serving, MTP, and throughput.

## What Shipped

- Added `ds4_gpu_arena_f32_matmul_f32` to the public GPU arena API.
- Added a V100 CUDA source-F32 arena matmul kernel with range/shape checks.
- Added fail-closed non-CUDA stub coverage for source-F32 arena matmul.
- Extended `tests/cuda_v100_descriptor_bound_ffn_smoke.c` to:
  - bind and upload `ffn_gate_inp.weight`;
  - read the real `ffn_gate_tid2eid` hash-router table;
  - compute CPU and GPU router logits;
  - compare selected experts and route weights;
  - upload all six selected routed expert spans;
  - execute all selected routed experts plus shared expert.
- Updated the appliance gate to run the router-enabled smoke by default.

## Evidence

Local validation:

- `make tests/cuda_v100_descriptor_bound_ffn_smoke.o`
- `bash -n tools/ds4-v100-gate.sh`
- `git diff --check`

Cluster validation:

- `docs/sprints/drafts/SPRINT-016-ROUTER-FFN-CLUSTER.log`
  - `cuda_v100_descriptor_bound_ffn_smoke: layer=2 token=16 expert0=84 gpu=0 arena_bytes=11784434944 hidden=4096 mid=2048 ... ok`
- `docs/sprints/drafts/SPRINT-016-GATE-CLUSTER/gate-summary.log`
  - Existing source, KV, projection/attention, logits, synthetic MXFP4 MoE,
    descriptor, binding, and router-enabled descriptor-bound FFN checks passed.
  - Gate summary: `PASS`, `failures=0`, `ready=false`.
  - Remaining readiness list:
    `full_layer_scheduler,attention_residual_norm,real_model_selected_token,public_serving,mtp,throughput_benchmark`.

## Deviations

- The router-enabled FFN smoke still uses deterministic synthetic hidden input.
- The smoke covers layer-2 hash routing. Bias-router layers should get their
  own representative gate before serving.
- The smoke still allocates a partial arena spanning the real layer-2 offsets;
  production should reuse the resident full stage arena.

## Handoff

Sprint 017 should introduce scheduler-owned layer state and extend the
descriptor-bound execution slice toward attention/residual/norm or selected
logits. The next important change is making this more than a standalone smoke.
