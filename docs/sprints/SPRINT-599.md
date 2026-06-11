# Sprint 599 - Post-C1 Layer Budget: Prefix, Overlap, Barriers

Date: 2026-06-11
Status: planned

## Goal

Close the gap between the Sprint 598 result (162.06 tok/s decode-domain,
4.25 ms layer replay) and the cycle's honest stretch target: **3x the
re-anchored baseline ≈ 220 tok/s decode-domain ≈ ~3.13 ms layer replay**.
This is the loop's confirmation sprint: it either demonstrates ≥ ~220
decode-domain at the reference shape, or it produces a measured floor
analysis showing what bounds the remaining gap and whether the target is
reachable within B2-scope structural changes.

The s597 cycle target as written (≥ 80 tok/s aggregate decode) is already
exceeded; this sprint adjudicates the stretch target.

## Where the 4.25 ms goes (s598 post-C1 profile, ms/layer)

| Component | ms | Notes |
|---|---:|---|
| pre-EP prefix (HC-current + attention + final_hc) | ~1.78 | not yet decomposed in-graph; HC-current measured 5.55 ms in eager (s597 follow-up #1) |
| shared swiglu_down | 0.78 | candidate to overlap with the now-cheap EP return |
| ep_return_nccl | 0.611 | s598 floor; NCCL ring LL |
| route_plan_pack | 0.53 | route-plan kernels + routed-input pack |
| barriers (954/978/1144/1373 et al.) | 0.30 | B2-D per-pair events candidate |
| expert GEMMs | 0.20 | leave alone |
| compose + contrib pack | ~0.05 | leave alone |

Budget math: reaching ~3.13 ms needs ~1.1 ms removed. Candidate pool sums
to >2 ms across {overlap swiglu_down (≤0.78), prefix cuts (≤~1.0 of 1.78),
route_plan_pack (≤~0.3), barriers (≤~0.2)} — multiple independent shots.

## Plan

### Phase A - Post-C1 full-layer decomposition (measurement; 1 day-scale)

Extend attribution to the pre-EP prefix: nsys kernel-class mapping of the
1.78 ms (HC-current input fill/pack vs attention proj/KV/state vs final_hc),
plus profiler stage marks for the prefix boundaries if needed (same
default-off discipline as s597; allowed surface includes
engine/hc_current.cu and engine/layer_runner.cu marks ONLY — flag-gated).
Output: a ranked post-C1 cost table that picks Phase B/C scope.

### Phase B - The structural wins, in measured-rank order

Candidates (each behind its own gate flag, each tolerance-gated, each A/B'd
at the reference shape before stacking):

1. **Overlap shared swiglu_down with the EP return** (extend
   `opt.overlap_ep_dense` coverage or re-order the stage so the dense stream
   runs swiglu_down concurrent with `ep_return_nccl` on the compute stream;
   they have no data dependence until compose).
2. **B2-D per-pair events**: replace the remaining 8×8
   `enqueue_cross_gpu_stream_barrier` at the EP sync sites with per-pair
   waits (destination waits only on its sources).
3. **route_plan_pack reduction**: fold route-plan kernels into fewer
   launches / move planning earlier under the prefix's shadow.
4. **Prefix cuts** per Phase A's ranking (HC-current structural leftovers
   from the post-MTP churn are the prime suspect — s597 follow-up #1).

Stack only what passes individually; re-measure the stack.

### Phase C - Verdict

- If ≥ ~220 decode-domain: record the confirmation, promote the passing
  flags (launcher defaults), done — the loop's performance question is
  answered YES.
- If < 220 after the candidate pool: write the floor analysis — per-stage
  measured minima, what each remaining ms is structurally bound to, and
  whether any in-scope change could close the residual. The loop's question
  is answered with the measured ceiling and what it would take to move it.

## Gates

- Tolerance ≥ 0.99 (selected-token AND sequence) vs the s597 control for
  every candidate and for the final stack; bit-exact is not required but
  report it.
- Default-off discipline: every candidate behind a flag; flag-off
  byte-identical (node counts + tolerance).
- Perf opt-in: reference shape per the fixed harness; the s598 leg
  (162.06 decode-domain / 108.50 wall) re-measured in-band as the control.
- No-SYS invariant holds (profiler classes + one nsys spot-check on the
  final stack).
- One V100 job at a time; GPUs idle-verified between runs.

## Definition of Done

1. Post-C1 full-layer decomposition table (prefix split into HC-current /
   attention / final_hc) archived.
2. Each attempted candidate: implementation behind a flag, tolerance result,
   individual A/B delta, keep/drop decision recorded.
3. Final stacked measurement at the reference shape with the same harness;
   decode-domain + wall reported.
4. **The stretch-target verdict**: either ≥ ~220 decode-domain confirmed, or
   the floor analysis with per-stage minima and an explicit
   reachable/not-reachable conclusion for 220 within B2 scope.
5. Promotions (if any) flip launcher defaults with rollback flags retained.
6. Report (SPRINT-599-REPORT.md), follow-ups, STATUS/steering/VISION updates
   by the orchestrator; commits per convention.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Overlap candidate breaks capture or ordering (dense vs compute stream inside the graph) | Med | Med | Flag-gated; capture probe first; tolerance gate catches semantic breakage |
| Prefix cost is structural (attention math, not churn leftovers) | Med | Med | Then the floor analysis documents it; partial cuts still count |
| Stacked candidates interact (win individually, flat together) | Med | Med | Stack incrementally; measure each addition; keep the per-candidate A/Bs |
| Diminishing returns vs run-band noise (~±3%) | Med | Low | Use decode-domain (less harness-sensitive) and multiple measured batches |

## Dependencies

- s598 promoted environment on gpu-01 (pack/contract/control persist;
  launcher default `nccl`).
- `DS4_V100_TP_EP_EP_STAGE_PROFILE` profiler; fixed bench harness.
- `SPRINT-598-REPORT.md` post-C1 profile; `SPRINT-597-FOLLOWUPS.md` #1
  (HC-current).
