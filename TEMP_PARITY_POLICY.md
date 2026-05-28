# Steering — Parity policy for arithmetic-changing optimizations

**Decision (supersedes the bit-exact selected-token gate for any change that
reorders reductions: A2 mix/RMS all-reduce, A3 router all-reduce, A4b
row-parallel consumers, and any future one).**

## Principle

GPU0's serial reduction is **not more correct** than an all-reduce — both are
fp32 approximations of the same sum. We gate on **closeness to the reference
within tolerance**, not on reproducing the incumbent serving binary
token-for-token. We have quality margin; **tolerance matters more than exact
matching.**

## What changes

The right test is **"does any reduction order change?"** — not "does the
consumer get rank-local data?" Same end-state, different mechanism, different
gate.

- **Arithmetic-changing steps** = anything that re-orders a reduction
  (A2, A3, A4b, A1-attn / A1-ffn rank-local norms, *including the existing
  `--attention-projection-rank-local-input` path*, which recomputes the
  attention norm per rank). **Tolerance gate**, not free-running exact match.
  Free-running sequence parity *will* diverge from autoregressive drift —
  that is **not** the gate.
- **Transport-only steps** = the same arithmetic on the same device (e.g. GPU0
  still computes the norm) but the cross-rank movement switches from
  `ds4_peer_copy_async` to `ncclBroadcast` / NCCL collective. **Stay bit-exact.**
  Canonical pattern for A6/A4a: replace each `ds4_peer_copy_async(root=0, ...)`
  with `ncclBroadcast(root=0, ..., r.compose_nccl)`. Same bytes, topology-aware,
  0 SYS, graph-capturable. A mismatch here is a real bug.
- **Misclassification trap:** "rank-local consumer" is NOT a synonym for
  transport-only. If achieving rank-local-input requires a per-rank reduction
  (norm, partial GEMM, …), it's arithmetic-changing — promote separately under
  the tolerance gate (call it A6-norm / A1-attn, not "A6").

## Gate for arithmetic-changing steps

**Agreement is the gate. Numeric drift is advisory, not gating.**

1. **PRIMARY GATE (sufficient on its own to promote):**
   - **selected-token agreement ≥ 0.99** vs control on the reference shape, AND
   - **generated-sequence agreement ≥ 0.99** vs control on the reference shape.

   If both pass → **PROMOTE.** No further checks required. Numeric per-logit
   drift is *not* a reason to reject.

2. **Advisory diagnostics (report, do not gate on):**
   - max selected-logit relative error vs control. Values well above `1e-3` are
     expected — they reflect fp32 reduction-order drift compounded through 43
     layers and do not indicate a quality problem when (1) passes. The number
     is reported for situational awareness, not as a pass/fail threshold.
   - fp64 partial accumulation may reduce the drift number, but is **optional**
     if (1) already passes.

3. **Fallback (only when PRIMARY GATE fails) — triage whether the failure is
   drift or a real bug:**
   - **Coherence sanity.** Generated text on a fixed prompt is coherent
     (no repetition collapse / garbage).
   - **Authoritative-reference check (small shape, e.g. 2K ctx / 16 tok).**
     Candidate is **no further from the Python DS4-V4 reference than control
     is.** Distinguishes "different but equivalent" from "regressed toward
     garbage."

   A candidate that fails (1) but passes both fallback checks may still be
   promoted with explicit justification. A candidate that fails (1) and fails
   either fallback is a real bug — fix or reject.

## Decision rule

If selected-token agreement ≥ 0.99 AND generated-sequence agreement ≥ 0.99 →
**PROMOTE.** Do not iterate on the numeric drift number. Do not chase
bit-exactness. Do not run additional sprints on a candidate that already
satisfies the primary gate.

This is *not* license to ship anything: a candidate with token agreement
significantly below 0.99 is a real correctness problem (see A6's 1/32), not a
tolerance question.

## No-rerun rule

**Do not rerun a candidate that has already been evaluated, if its existing
evaluation artifact satisfies the current gate.** If the prior artifact shows
selected-token + generated-sequence agreement ≥ 0.99, promote it on that
existing evidence. The gate change does not require fresh A/B runs — it
requires re-classifying the existing evidence under the current rule.

## Immediate consequences under the relaxed gate

- **A3 router all-reduce** — existing s480 artifact shows token agreement 1.0
  and sequence agreement 1.0. Numeric rel-err 0.025 is advisory only and is
  **not** a rejection. **Promote on existing evidence, do not rerun.**
- **EP-compose ReduceScatter (non-compact FP32)** — existing s480 artifact
  shows agreement 1.0 and rel-err 7e-5. Already tolerance-cleared; confirm
  promotion.
- **A2 mix/RMS all-reduce** — implemented in 478 but never re-evaluated under
  the tolerance gate. Needs **one** tolerance A/B against the current gate; on
  agreement ≥ 0.99, promote. Expected pass (per-layer mix diff was ~1e-6).
- **A6 rank-local attention projection input** — STAYS rejected; 1/32 token
  agreement is a real bug, not a gate-strictness issue.

## Reporting (per candidate)

Selected-token agreement, generated-sequence agreement, decode tok/s, GPU util,
Direct-SYS bytes. Report max selected-logit rel-err as a diagnostic field.
**Do not include rel-err as a pass/fail criterion.**
