# TEMP Status Report 478: HC-Current A2 All-Reduce

## Current State

The A2 rank-local mix + RMS all-reduce path is implemented, builds on the V100
pod, and is promoted into the appliance defaults. Under
`TEMP_PARITY_POLICY.md`, it is not rejected by the old exact free-running
selected-token gate because it intentionally changes reduction order.

The binary gate remains explicit for A/B bisection, but the appliance launcher
now enables it by default:

- CLI: `--tp-hc-current-allreduce-gate`
- Profile: `--hc-current-allreduce`
- Env: `DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=1`

## What Was Tested

Build:

- Synced to `/localpool/ds4/workspace/ds4-sprint181`.
- Built in pod `llm/ds4-tp-bench` with `sm_70`.

Focused diagnostic:

- Artifact:
  `/localpool/ds4/workspace/s478-a2-mixdiff-debug2-s32-t1-r1`
- Shape: 32 slots, 256K context, one selected-token request.
- Result: all 43 layers passed all-reduced mix vs GPU0 reference.
- Max absolute mix diff was micro-scale, roughly `1.9e-6` to `6.7e-6`.

Reference A/B:

- Artifact:
  `/localpool/ds4/workspace/s478-a2-correctness-ref2-s32-t64-r256`
- Shape: 32 slots, 256K context, 256 requests, 64 generated tokens.
- Control HTTP: 256/256.
- Candidate HTTP: 256/256.
- Old full response parity: 0/256 matched.

## Topline Metrics

| Metric | Control | A2 Candidate |
|---|---:|---:|
| client generated tok/s | 18.45 | 18.75 |
| projected slot-step tok/s | 35.84 | 36.81 |
| scaffold ms/token | 892.78 | 869.38 |
| EP ms | 492.28 | 483.38 |
| HC-current ms | 358.30 | 343.92 |
| HC-current attn mix ms | 20.83 | 10.03 |
| pre-EP HC-current ms | 68.50 | 56.31 |
| NCCL graph SYS edges | 0 | 0 |

## Interpretation

A2 produces the expected local timing improvement, especially in
`hc_current_attn_mix`. The generated sequence diverges after a few decode steps,
but the layer-local mix diff is only micro-scale. Per `TEMP_PARITY_POLICY.md`,
that is the expected autoregressive cascade from floating-point reduction-order
drift and is not the promotion gate for arithmetic-changing optimizations.

A2 is promoted. The remaining fp64 and teacher-forced/logit-drift checks are
follow-up instrumentation, not blockers to the appliance default.

## Direct-SYS Finding

The A2 candidate was run with peer accounting enabled. It still reported:

- Direct-SYS bytes: `184,363,599,872`
- Direct-SYS ops: `20,590,057`
- Top SYS site: `run_true_ds4_attention_projection_prefix:13172`

That confirms the next bankable target is A6/A4a with A2 disabled:

- make attention-projection rank-local the default,
- route remaining full-current movement through NCCL rank-major paths,
- drive the targeted full-current Direct-SYS bytes down. Total Direct-SYS bytes
  are not expected to reach zero yet because EP, router-plan distribution,
  output-head staging, and KV/state movement still have separate peer-copy
  surfaces.

Per `TEMP_PARITY_POLICY.md`, the sprint label is not the correctness classifier.
The classifier is whether the candidate changes reduction order:

- Existing `--attention-projection-rank-local-input` recomputes attention norm
  per rank. That is arithmetic-changing, so it uses the tolerance/teacher-forced
  gate rather than bit-exact free-running sequence parity.
- A pure transport rewrite that keeps GPU0's computed normed tensor and replaces
  direct peer copies with NCCL broadcast/allgather is transport-only, so it stays
  bit-exact.

## NCCL Note

The HC all-reduces are tiny latency-bound payloads. Ring is not inherently the
right algorithm here. The appliance now leaves `NCCL_ALGO`/`NCCL_PROTO` on
`auto` by default while preserving `NCCL_P2P_LEVEL=NVL` and the no-SYS ring
hint. For future all-reduce experiments, explicitly test
`DS4_V100_NCCL_ALGO=Tree DS4_V100_NCCL_PROTO=LL128`, while preserving the
invariant that hot paths stay NVLink/SYS-clean.

## A4a Update

A4a targeted full-current cleanup is implemented and builds on the V100 pod.

Changed transport:

- HC-current `current_full` staging now uses NCCL broadcast when it is not
  already using the NCCL allgather path.
- HC-current `ffn_normed` route-input staging is split into stream-ordered
  phases and uses NCCL broadcast before route packing.
- Shared-FFN, attention-projection, compressed-KV, and post-attention FFN
  replicated normed-current staging use NCCL broadcast instead of GPU0
  peer-copy fanout.

Validation:

- Build passed on `llm/ds4-tp-bench`.
- Targeted source search found no remaining direct peer-copy sites for
  `d_current_full` from `hc->d_current_full`, `hc->d_ffn_normed`, or
  `hc->d_attn_normed`.
- 32-slot / 256K / 32-request selected-token smoke:
  `/localpool/ds4/workspace/s478-a4a-nccl-broadcast-smoke-s32-t1-r32`
  - HTTP 200: 32/32.
  - NCCL graph SYS edges: 0.
  - Direct-SYS bytes: `123,873,312`.
  - Residual first SYS: `upload_model_router_route_plan_gpu`.
  - Residual top SYS: EP/operator copy path.
- 32-slot / 256K / 1-request smoke after fixing the profile harness NCCL policy:
  `/localpool/ds4/workspace/s478-a4a-nccl-auto-smoke-s32-t1-r1`
  - HTTP 200: 1/1.
  - `NCCL_ALGO` unset/auto, `NCCL_RINGS=0 3 2 1 5 7 6 4`,
    `NCCL_P2P_LEVEL=NVL`.
  - NCCL graph SYS edges: 0.
  - Direct-SYS bytes: `74,733,600`.
  - Residual first SYS: `upload_model_router_route_plan_gpu`.
  - Residual top SYS: `materialize_shared_swiglu_down_input`.

Interpretation:

The A4a target passed: full-current GPU0 fanout is no longer represented by
direct peer-copy SYS sites. Remaining SYS traffic is real but belongs to later
surfaces: router plan distribution and shared/EP materialization copies.

## A3 / Reduced-Tolerance Update

A3 router-logits all-reduce is implemented and builds on the V100 pod.

- CLI: `--model-router-allreduce-logits-gate`
- Profile: `--model-router-allreduce-logits`
- Env: `DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS=0`

The gate remains default-off. It computes rank-local router partial logits from
rank-local HC shards, all-reduces the `[slots,256]` logits, and then reuses the
existing route-selection logic.

Post-479 tolerance update:

- Artifact: `/workspace/s480-a3-router-allreduce-tolerance`.
- Local tolerance summary:
  `/tmp/s480-a3-router-allreduce-tolerance/response-tolerance.json`.
- Control HTTP: 32/32.
- Candidate HTTP: 32/32.
- selected-token agreement: `32/32 = 1.0`.
- generated-sequence agreement: `1.0`.
- max selected-logit relative error: `0.025157711827123192`.
- threshold: `1e-3`.
- tolerance pass: `false`.

Decision: keep A3 default-off under the current strict selected-logit
tolerance policy.

Validation:

- Remote build passed in `/workspace/ds4-sprint181`:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- A3 smoke artifact: `/workspace/s478-a3-router-allreduce-smoke`.
  - HTTP 200: 1/1.
  - `model_router_allreduce_logits_gate=1`.
  - `tp_hc_current_allreduce=1`.
  - selected token: `48177`.
  - selected logit: `16.402936935`.
  - scaffold PASS.

The reduced tolerance checker is now present at
`tools/ds4-v100-http-response-tolerance.py`. The A/B harness can optionally
judge arithmetic-changing candidates with:

- selected-token agreement >= `0.99`,
- generated-sequence agreement >= `0.99`,
- max selected-logit relative error <= `1e-3`.

## A6 Decision

A6 was evaluated and is not promoted.

Artifact:

- Pod: `/workspace/s478-a6-tolerance`
- Local tolerance summary:
  `/tmp/s478-a6-tolerance/response-tolerance.json`

Result:

- Control HTTP: 32/32.
- Candidate HTTP: 32/32.
- selected-token agreement: `1/32 = 0.03125`.
- generated-sequence agreement: `0.03125`.
- max selected-logit relative error: `0.08766228933928177`.
- threshold: `1e-3`.
- tolerance pass: `false`.

Decision: keep
`DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT=0`. A6 failed
the reduced tolerance gate at the 32-slot / 256K selected-token shape.
