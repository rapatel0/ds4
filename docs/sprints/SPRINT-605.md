# Sprint 605 - Promote edges+fix; Open the Step-Floor Reduction Campaign

Date: 2026-06-13
Status: planned

## Goal

Two parts:

1. **Promote edges+fix** as the launcher sync default. Both s603 blockers are
   now gone: correctness (the s604 DENSE_FIX) and perf (Phase E measured
   edges+fix at +15.1% over join+fix, clearing the gate s603's fix-less edges
   run failed). Gate it with a clean soak (the s604 amplifier is now the cheap
   1-run correctness check) and flip.
2. **Open the step-floor reduction campaign** — the actual path to ≥50 tok/s
   per slot. We are at ~5 tok/s/slot (step ~177 ms, edges+fix S=8). The target
   needs the step at ≤20 ms; MTP's realistic 2-3x means the base step must
   first reach the ~40-60 ms MTP-reachable floor (a 3-4x base reduction). This
   sprint does the clean-base decomposition, ranks the levers, and attacks the
   #1 feasible one. It will NOT finish the campaign — it starts it on a
   correct, fast base for the first time.

## Why now / what changed

Every prior decomposition (s581, s597, s601, s603) was measured on a base that
was either racy, mid-change, or both. s604 fixed the gating hazard and the
amplifier makes any new overlap cheap to gate for correctness. So for the
first time we can decompose the step floor on a stable correct base and trust
the lever ranking.

s601's key finding stands as the campaign's thesis: the step is **launch/wait
bound, not compute bound** — GPU busy was only ~2.94 ms of the ~4.25 ms/layer
replay (~31% pure launch/sync overhead even under full capture). The levers
are structural (overlap, fewer rendezvous, launch compaction), not kernel
speed.

## Plan

### Phase A — Promote edges+fix

1. Re-verify the pod (recreate if degraded — s604 noted ~38h degradation; the
   /workspace hostPath persists, recreation is cheap and proven in s601;
   recheck 16 Gi shm + apt provisioning).
2. Clean soak: edges+fix, un-amplified reference shape, ≥30 runs, telemetry —
   zero token events required; plus the amplifier gate (edges+fix under
   DENSE_HAZARD_AMP@20us must stay 1.0/1.0, already shown 1/1 in s604 — run ≥3
   to confirm). Also run the pre_compose-amplified fix-on gate (s604 follow-up
   #3) to close the late-step class explicitly.
3. If clean: flip `DS4_V100_TP_EP_S602_SYNC=edges` launcher default; keep join
   as rollback. Re-measure the reference-shape decode-domain as the new
   promoted baseline.

### Phase B — Clean-base step-floor decomposition + lever feasibility

1. Full per-layer stage decomposition on the promoted edges+fix base
   (DS4_V100_TP_EP_EP_STAGE_PROFILE), S=1/8/32, with the launch/wait vs
   GPU-busy split (reconfirm or update s601's ~31% overhead figure on the
   clean base).
2. Rank the step-floor levers by measured ms AND feasibility:
   - **Microbatch ping-pong** (split S into 2 half-batches, overlap microbatch
     A's cross-rank comm with B's compute — the classic comm-bound-decode 2x).
     FEASIBILITY GATE FIRST: measure free VRAM headroom (s604 showed ~28.8 GiB
     used / 32; microbatch doubles activation/staging buffers — quantify
     whether it fits at S=16+2, or only at smaller S, or needs buffer reuse).
   - **Prefix launch compaction**: the attention + HC-current prefix
     (~1.3-1.5 ms/layer combined) — fuse/reduce launches.
   - **Route-plan shadowing** (s599/603 C-C, never attempted, ~0.45 ms/layer
     pool): move route planning under the prefix's shadow.
   - **Cross-layer graph consolidation**: reduce per-layer capture/launch
     overhead across the 43-layer chain.
3. Output: a ranked, feasibility-checked lever table; pick the #1.

### Phase C — Attack the #1 lever

Implement the top-ranked feasible lever behind a flag (default off,
byte-identical), gate it with the amplifier (1-run correctness) + a soak, A/B
at the reference shape. Promote only on: correctness-clean (amplifier + soak),
tolerance 1.0/1.0, and a measured step-floor reduction. If the #1 lever is
infeasible (e.g. microbatch doesn't fit VRAM at useful S), document and take
the #2.

### Phase D — Re-measure + restate the target math

Reference-shape floors on the final config; S=1/8/16/32 per-slot tok/s; the
updated step floor and the required-MTP-multiplier for ≥50/slot at S=8 and
S=16. State honestly how much of the 3-4x base reduction this sprint captured
and what the remaining sequence is.

## Definition of Done

1. edges+fix promotion decision with the soak + amplifier gate evidence;
   launcher default flipped (or held with reason); new baseline measured.
2. Clean-base step decomposition (S=1/8/32) with the launch/wait split and the
   ranked, VRAM-feasibility-checked lever table archived.
3. The #1 feasible lever implemented, correctness-gated (amplifier + soak),
   A/B'd; promotion decision.
4. Re-measured floors + per-slot curve + updated ≥50/slot + MTP-multiplier
   statement; explicit "captured X of the needed base reduction, remaining
   sequence is Y."
5. Report, follow-ups, orchestrator docs/commits.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Microbatch overlap doesn't fit VRAM (~1.3 GiB free) | Med-High | Med | Feasibility gate FIRST in Phase B; fall to prefix compaction / route-plan shadow which add no buffers |
| New overlap reintroduces a dense↔rank or cross-rank hazard | Med | Med | The amplifier is now the cheap 1-run gate; every lever is amplifier-gated before any perf claim |
| Levers interact / don't stack | Med | Med | One lever this sprint, measured alone; stacking is later sprints |
| Diminishing per-lever returns vs the 3-4x needed | High | Med | Expected — this is a multi-sprint campaign; the DoD requires an honest remaining-sequence statement, not target attainment |
| Pod degradation corrupts measurement | Med | Low | Recreate the pod in Phase A; telemetry per run |

## Dependencies

- HEAD ecd88c37 (s604 committed, DENSE_FIX default-on; edges built; amplifier
  built; Phase E floors folded in).
- Pod environment (recreate if degraded; /workspace persists).
- s604 amplifier (DENSE_HAZARD_AMP) as the correctness gate; s597 stage
  profiler; s601 launch/wait-split methodology.
- s604 follow-ups (#1 edges promotion, #2 step-floor campaign, #3 pre_compose
  class confirmation).
