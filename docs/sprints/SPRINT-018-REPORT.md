---
sprint: 018
title: V100 Descriptor-Bound Attention Projection Residual Norm Gate
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-018 Report: V100 Descriptor-Bound Attention Projection Residual Norm Gate

## Verdict

`SHIP`

Sprint 018 extended `ds4_v100_layer_state` with real attention/control
descriptor ownership and added a CUDA smoke that runs layer-2 attention
projection, residual, and norm surfaces from real source bytes on V100.

This is still not full layer output. It does not prove attention softmax,
compressed-KV visibility, or selected-token decode.

## What Shipped

- Added attention metadata to `ds4_v100_layer_state`:
  - `attn_q_a`, `attn_q_b`, `attn_kv_latent`, `attn_output_a`,
    `attn_output_b`;
  - `attn_norm`, `attn_q_a_norm`, `attn_kv_a_norm`, `attn_sinks`;
  - HC attention controls.
- Added `ds4_v100_layer_state_attention_arena_span`.
- Extended `tests/v100_layer_state_smoke.c` to validate attention dimensions
  and arena span.
- Added `tests/cuda_v100_descriptor_bound_attention_smoke.c`.
- Added `descriptor_bound_attention` to the appliance gate.

## Evidence

Local validation:

- `make tests/v100_layer_state_smoke`
- `./tests/v100_layer_state_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --layer 2`
- `make tests/cuda_v100_descriptor_bound_attention_smoke.o`
- `bash -n tools/ds4-v100-gate.sh`
- `git diff --check`

Cluster validation:

- `docs/sprints/drafts/SPRINT-018-LAYER-STATE-CLUSTER.log`
  - `v100_layer_state_smoke: layer=2 ... q=32768 kv=512 ... attn_span=8329207040 ok`
- `docs/sprints/drafts/SPRINT-018-ATTENTION-CLUSTER.log`
  - `cuda_v100_descriptor_bound_attention_smoke: layer=2 gpu=0 arena_bytes=8329207040 hidden=4096 q=32768 kv=512 out_rank=8192 ok`
- `docs/sprints/drafts/SPRINT-018-GATE-CLUSTER/gate-summary.log`
  - New `descriptor_bound_attention` gate passed.
  - Full gate summary: `PASS`, `failures=0`, `ready=false`.

## Deviations

- The smoke feeds the output projection path from the first 4096 values of the
  real q projection as a bounded attention-summary proxy. It does not claim
  semantic attention output correctness.
- Production resident arena reuse remains deferred; the smoke still allocates a
  partial arena spanning the real attention descriptor offsets.

## Handoff

Sprint 019 should either implement the full attention softmax/compressed-KV
layer-output slice using these real descriptor-bound projections, or drive the
combined attention+FFN state far enough to produce bounded real-model logits.
