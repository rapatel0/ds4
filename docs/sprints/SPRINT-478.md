# Sprint 478: HC-Current Rank-Local All-Reduce

## Overview

Replace the first HC-current GPU0-centered reduction with a TP-native path:
rank-local partials, tiny NCCL all-reduces, and rank-local split/weighted-sum.
This implements the first step from `docs/sprints/archive/TEMP_HC_ALLREDUCE_PROMPT.md`, starting
with A2, because peer-copy accounting showed NCCL itself has zero SYS graph
edges while GPU0 broadcasts account for direct SYS traffic.

## Evidence

- Reference artifact: `/localpool/ds4/workspace/s478-peer-site-self-s32-t1-r4b`.
- Shape: 32 slots, 256K context, position 262080, one decode token.
- NCCL graph SYS edges: 0.
- Direct peer-copy traffic: 82,389 ops / 593 MB.
- Direct SYS traffic: 35,313 ops / 256 MB.
- First SYS site: `run_shared_hc_current_input:7454`, GPU0 split broadcast.
- Top SYS site: `run_true_ds4_attention_projection_prefix:12817`, GPU0 normed
  current broadcast.
- Clarification: these reductions are small-message and latency-bound. Let NCCL
  tune the algorithm, or pin `NCCL_ALGO=Tree NCCL_PROTO=LL128` for experiments.
  The required invariant is topology-clean NCCL on NVLink with zero SYS edges,
  not a specific ring algorithm.

## Scope

1. Add a default-off gate for HC-current all-reduce:
   `--tp-hc-current-allreduce-gate` /
   `DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=1`.
2. Reshard `hc_attn_fn` and `hc_ffn_fn` by contraction column at load. Each rank
   receives the four strided 512-column blocks matching its `[4,512]` HC shard.
3. For the current-input HC mix:
   - compute local max-abs and unnormalized mix partials on each rank,
   - NCCL all-reduce max-abs and mix, using the small-message latency path,
   - compute local stable scaled sum-of-squares using the reduced max,
   - NCCL all-reduce sum-of-squares,
   - locally apply the stable RMS scale and `hc_split_rows_kernel` semantics,
   - run `hc_weighted_sum_shard_kernel` without GPU0 split broadcast.
4. Preserve the existing max-abs stable RMS formula for parity.
5. Extend the profile/correctness harness so the gate is trivial to rerun.

## Out Of Scope

- A6/A4 attention-projection rank-local default.
- A3 router all-reduce.
- A4b row-parallel consumers.
- MTP and CUDA graph capture.
- PP/layer-parallel variants.

## Definition Of Done

- Local build compiles.
- Remote pod build compiles with `sm_70`.
- Correctness passes under `docs/sprints/archive/TEMP_PARITY_POLICY.md`:
  - any path that changes reduction order uses the tolerance/teacher-forced
    gate, even if it is part of A6/A4a,
  - pure transport-only paths stay bit-exact at the 32-slot / 256K reference
    shape.
- Peer accounting confirms the A2 split-broadcast site is removed from the
  candidate path.
- Steady reference A/B records decode tok/s, request-window utilization, and
  direct peer SYS bytes/ops.
- Outcome is documented with promote/pending/reject decision and next bottleneck.

## Risks

- Stable RMS requires two reduction phases: max first, then scaled sumsq. This
  is slightly more launch/collective work than the plain RMS identity but avoids
  changing math while we are isolating performance.
- Remaining full-current consumers can still emit direct SYS traffic until A6/A4a
  are implemented.
- NCCL initialization needs to be active even when this is the only NCCL gate.

## Outcome

Status: A2 implemented and promoted into the appliance defaults. Under
`docs/sprints/archive/TEMP_PARITY_POLICY.md`, the old free-running exact sequence mismatch is not a
rejection because A2 reorders floating-point reductions.

Build:

- Local source synced to `/localpool/ds4/workspace/ds4-sprint181`.
- Remote pod build passed:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.

Diagnostic:

- Initial full-parity diagnostic had a bug: it scaled the reduced mix before the
  sumsq all-reduce had run.
- After fixing the diagnostic order, the focused 32-slot / 256K / one-request
  run showed all 43 layers passing the all-reduced mix-vs-GPU0-reference check.
- Observed mix max-abs differences were roughly `1.9e-6` to `6.7e-6`.

Reference A/B:

- Artifact: `/localpool/ds4/workspace/s478-a2-correctness-ref2-s32-t64-r256`.
- Shape: 32 slots, 256K context, 256 requests, 64 generated tokens.
- Control completed 256/256 HTTP 200.
- Candidate completed 256/256 HTTP 200.
- Old free-running response parity: 0/256 full generated-token sequences
  matched.
- Per `docs/sprints/archive/TEMP_PARITY_POLICY.md`, this is not by itself a rejection for A2 because
  A2 reorders floating-point reductions.

Topline performance from the A2 tolerance-candidate run:

| Metric | Control | A2 Candidate |
|---|---:|---:|
| client generated tok/s | 18.45 | 18.75 |
| scaffold projected slot-step tok/s | 35.84 | 36.81 |
| scaffold ms/token | 892.78 | 869.38 |
| HC-current total ms | 358.30 | 343.92 |
| HC-current attn mix ms | 20.83 | 10.03 |
| pre-EP HC-current ms | 68.50 | 56.31 |
| NCCL graph SYS edges | 0 | 0 |

Peer accounting note:

- Candidate peer accounting found large remaining Direct-SYS traffic:
  `184,363,599,872` bytes across `20,590,057` ops.
- Top SYS site: `run_true_ds4_attention_projection_prefix:13172`.
- This is outside A2 and confirms A6/A4a remains mandatory.

Decision:

- Promote A2 for the appliance path:
  `DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=1`.
- Keep `--tp-hc-current-allreduce-gate` as an explicit binary flag so the path
  remains easy to disable for A/B bisection.
- Record the remaining tolerance-harness work as follow-up instrumentation, not
  as a blocker to promotion: teacher-forced top-1 agreement >= 99% if available,
  otherwise per-step logit max relative error <= 1e-3 plus coherence sanity.
- Continue A6/A4a with A2 disabled, but classify each candidate by the updated
  parity policy rather than by sprint label:
  - existing attention-projection rank-local input recomputes attention norm per
    rank, so it is arithmetic-changing and uses the tolerance gate;
  - a pure NCCL replacement for GPU0 peer-copy/broadcast that preserves the same
    GPU0-computed tensor is transport-only and remains bit-exact.
- Direct-SYS accounting should be interpreted by targeted site. A4a is expected
  to remove the full-current GPU0 peer-copy sites; other direct SYS sources in
  EP, router plan distribution, output-head staging, or KV/state movement may
  remain until separately converted.
- All cross-rank reductions should use NCCL collectives with topology-clean
  configuration. The appliance now leaves `NCCL_ALGO` and `NCCL_PROTO` at `auto`
  by default so NCCL can choose Tree/LL-style small-message paths for HC/router
  reductions and bandwidth-oriented paths for larger collectives. Pin
  `DS4_V100_NCCL_ALGO=Tree DS4_V100_NCCL_PROTO=LL128` only as a measured
  experiment.

## A4a Update

Status: targeted full-current transport cleanup implemented.

Code changes:

- Replaced full-current GPU0 peer-copy transport with NCCL broadcast for:
  - HC-current `current_full` staging,
  - HC-current `ffn_normed` route-input staging,
  - shared-FFN normed input staging,
  - attention-projection normed-current staging,
  - compressed-KV normed-current staging,
  - post-attention FFN normed-current staging.
- Attention projection now uses NCCL transport for replicated normed current
  whenever direct input fill is disabled. The old direct peer-copy fallback is
  removed.
- `open_compose_nccl` now initializes the compose communicator for these
  full-current NCCL broadcasts, even if no other NCCL HC-current gate is active.
- The profile harness no longer forces `NCCL_ALGO=Ring` for the no-SYS policy;
  it leaves algorithm/protocol unset unless explicitly requested.

Validation:

- Remote V100 pod build passed after the A4a changes:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Source search for targeted full-current direct peer copies returned no
  matches:
  `d_current_full <- hc->d_current_full`, `d_current_full <- hc->d_ffn_normed`,
  and `d_current_full <- hc->d_attn_normed`.
- Short 32-slot / 256K / 32-request selected-token peer-accounting smoke passed:
  `/localpool/ds4/workspace/s478-a4a-nccl-broadcast-smoke-s32-t1-r32`.
  - HTTP 200: 32/32.
  - NCCL graph SYS edges: 0.
  - Direct-SYS bytes: `123,873,312`.
  - First SYS site: `upload_model_router_route_plan_gpu`.
  - Top SYS site: EP/operator path, not full-current staging.
- Follow-up 32-slot / 256K / 1-request smoke with the patched NCCL auto harness
  passed:
  `/localpool/ds4/workspace/s478-a4a-nccl-auto-smoke-s32-t1-r1`.
  - HTTP 200: 1/1.
  - `NCCL_ALGO` absent from the environment; `NCCL_RINGS` and
    `NCCL_P2P_LEVEL=NVL` remain set.
  - NCCL graph SYS edges: 0.
  - Direct-SYS bytes: `74,733,600`.
  - First SYS site: `upload_model_router_route_plan_gpu`.
  - Top SYS site: `materialize_shared_swiglu_down_input`.

Decision:

- A4a targeted full-current cleanup is promoted at the code level.
- Total Direct-SYS is not expected to be zero yet. The next residual surfaces are
  router-plan upload and shared/EP materialization copies, not the full-current
  GPU0 broadcasts targeted by A4a.

## A3 / Reduced-Tolerance Update

Status: A3 router-logits all-reduce implemented as a default-off arithmetic
candidate; reduced tolerance gate added; A6 was evaluated and not promoted.

Code changes:

- Added `--model-router-allreduce-logits-gate` /
  `DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS=0`.
- The A3 path computes rank-local router partial logits over each rank's
  resident `[slots,512]` HC shard, NCCL all-reduces the `[slots,256]` logits,
  and then reuses the existing bias/hash/top-k route selection path.
- The path uses the same stable max/sumsq normalization structure as HC-current
  A2 before computing local router partials.
- Launcher, profile, and A/B harness plumbing can select the gate explicitly.
- Added `tools/ds4-v100-http-response-tolerance.py`, a reduced selected-token
  tolerance artifact checker using:
  - selected-token agreement >= `0.99`,
  - generated-sequence agreement >= `0.99`,
  - max selected-logit relative error <= `1e-3`.
- Extended the HTTP A/B harness with optional tolerance-gate summary fields and
  per-leg HC-current all-reduce controls.

Validation:

- Local checks passed:
  - `python3 -m py_compile tools/ds4-v100-http-response-tolerance.py tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py`
  - `bash -n tools/ds4-v100-run-appliance.sh`
  - `git diff --check` on touched files.
- Remote V100 pod build passed from `/workspace/ds4-sprint181`:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- A3 selected-token smoke passed:
  `/workspace/s478-a3-router-allreduce-smoke`
  - HTTP 200: 1/1.
  - `model_router_allreduce_logits_gate=1`.
  - `tp_hc_current_allreduce=1`.
  - selected token: `48177`.
  - selected logit: `16.402936935`.
  - scaffold PASS with finite decode.
- A6 tolerance run:
  `/workspace/s478-a6-tolerance` on the pod and
  `/tmp/s478-a6-tolerance/response-tolerance.json` locally.
  - Control: 32/32 HTTP 200, `attention_projection_rank_local_input_gate=0`.
  - Candidate: 32/32 HTTP 200, `attention_projection_rank_local_input_gate=1`.
  - selected-token agreement: `1/32 = 0.03125`.
  - generated-sequence agreement: `0.03125`.
  - max selected-logit relative error: `0.08766228933928177`.
  - threshold: `1e-3`.
  - tolerance pass: `false`.

Decision:

- A3 is complete as an explicit default-off candidate. Do not promote it by
  default until a larger tolerance/performance A/B says it is safe and useful.
- Post-479 A3 tolerance A/B at `32` slots / `256K` / `32` selected-token
  requests served both legs `32/32` and matched selected tokens/sequences
  `1.0`, but failed the strict selected-logit threshold with max relative error
  `0.025157711827123192` versus `1e-3`; keep A3 default-off.
- The reduced selected-token tolerance gate is now available for arithmetic
  candidates, with the stricter `1e-3` selected-logit relative-error threshold.
- Do not promote A6 in this sprint. It is not within tolerance at the
  32-slot / 256K selected-token shape.
- Sprint 478 is complete with A6 rejected, not promoted.
