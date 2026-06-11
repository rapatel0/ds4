# Sprint 599 Follow-Ups

## 1. Latent ordering hazard in the promoted path (CRITICAL — Sprint 600 lead)

- **What**: Every faster shared-swiglu exchange variant (5 built, including
  `batched` with bit-identical byte mechanics at 1/32 the launches) fails
  the tolerance gate at the reference shape (0.78-0.95 agreement), while
  gaining +11.5-17.9% decode-domain. Identical bytes + identical math +
  different timing = a downstream consumer is missing a dependency edge on
  the exchanged swiglu data and currently wins its race only because the
  promoted 1,792-launch exchange is slow. This is (a) the gate on ~20-30
  tok/s of demonstrated gain and (b) a latent correctness debt in the
  PROMOTED path — any future timing perturbation (driver, NCCL version,
  clock behavior) could surface it in production.
- **Why**: C-A forensics across five variants isolated timing as the only
  free variable.
- **Severity**: Critical.
- **Suggested sprint**: 600 (lead item).
- **Files**: `engine/ep_dense.cu` (`materialize_shared_swiglu_down_input`),
  `engine/decode_loop.cu` (swiglu/dense/compose ordering), evidence in
  `logs/from-cluster/sprint599/`.

## 2. C-C route-plan shadowing never attempted

- **What**: route_plan_pack is 0.517 ms/layer; the planned candidate (fold
  launches / move planning under the prefix's shadow) was not built — the
  sprint budget went to C-A forensics.
- **Severity**: Important (0.3-0.5 ms of the 220 budget).
- **Suggested sprint**: 600 (after the hazard fix).
- **Files**: `engine/router_plan.cu`, `engine/post_attention_ffn.cu`.

## 3. C-B early-return flag is tolerance-clean but perf-neutral

- **What**: The early EP-return + per-rank ordering flag collapses two 8x8
  barriers (0.075→0.005, 0.152→0.030 ms) with tolerance 1.0/1.0 but measures
  +0.5% alone, and anti-synergizes with a fast swiglu (serializes ahead of
  it). Re-evaluate as part of the post-hazard-fix stack — the barrier
  reduction may matter once the swiglu exchange is fast and the wait
  structure changes.
- **Severity**: Nice-to-have (until restacked).
- **Suggested sprint**: 600 restack.
- **Files**: `engine/decode_loop.cu` (flag retained, default off).

## 4. swiglu exchange also crosses SYS (pre-existing, now quantified)

- **What**: The promoted swiglu_down input exchange's per-slot UVA remote
  loads cross SYS pairs (same class of issue s598 fixed for the EP return).
  Fixing the hazard unlocks the `nccl` exchange variant which removes those
  SYS loads too — fold the no-SYS proof for this stage into the 600 gate.
- **Severity**: Important (folded into #1's fix).
- **Files**: `engine/ep_dense.cu`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|-----------------|-------|
| Latent ordering hazard (masked race) | Critical | 600 lead | engine/ep_dense.cu, engine/decode_loop.cu |
| Route-plan shadowing unattempted | Important | 600 | engine/router_plan.cu |
| C-B restack after hazard fix | Nice-to-have | 600 | engine/decode_loop.cu |
| swiglu exchange SYS exposure | Important | 600 (with #1) | engine/ep_dense.cu |
