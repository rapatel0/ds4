# MTP Gate Plan (ready-to-dispatch draft)

Status: DRAFT, queued. Sequenced after the s605+ step-floor campaign reaches
the point where MTP is the binding multiplier (or sooner if the campaign
stalls — the Phase 0 re-test is cheap and high-information regardless).
Assign a sprint number at dispatch.

## Why MTP is on the critical path

≥50 tok/s per slot ⇔ step ≤ 20 ms. The clean base is ~177 ms (edges+fix,
S=8) ≈ 5 tok/s/slot. The step-floor campaign targets the ~40-60 ms
MTP-reachable floor (3-4x). MTP's accept-K-drafts mechanism is the remaining
2-3x that closes 40-60 ms → ≤20 ms. There is no ≥50/slot path that does not
go through working MTP. It has been 0/71 (broken) since s585-596.

## Phase 0 — The cheap re-test (do FIRST, before any rebuild work)

s604 fixed a cross-rank dense→rank ordering hazard at the attention-output
handoff — the exact stage the s590-595 0/71 investigation named as a prime
remaining suspect and cleared only via static (synchronized) oracles. The MTP
draft reuses that code (`run_layer(mtp_opt)` for layer 43,
`engine/token_major_loop.cu:450`). In a full-capture graph a missing edge is a
*deterministic* mis-order, which reconciles "0/71 deterministic" with "all
static checks pass." 0/71 was last measured 2026-05-30, before DENSE_FIX.

Test (no code changes — current HEAD already has DENSE_FIX default-on):
1. Run MTP serving acceptance on current HEAD (DENSE_FIX=1): record the
   accept count vs the historical 0/71. Use the s596 acceptance harness.
2. Control: DENSE_FIX=0 (the pre-s604 state) — should reproduce ~0/71.
3. Amplifier probe: DENSE_HAZARD_AMP@20us @ attn_out_a with DENSE_FIX=0 — if
   draft tokens / acceptance shift under the amplifier, the draft is
   timing-sensitive at that handoff (confirms the hazard touches the draft).
4. First-divergence: if acceptance is still 0 with the fix, capture which
   draft logit/position diverges from the main model's verified next token,
   on the clean (fixed) base — a cleaner signal than any pre-s604 run.

Outcomes:
- **Acceptance > 0 with the fix**: the hazard was (part of) the blocker. The
  s585-596 "draft math is semantically wrong" conclusion is overturned;
  proceed to the throughput loop (Phase B).
- **Still 0/71 deterministically, fix-independent, amp-independent**: the
  blocker is genuine draft semantics; the s585-596 record stands. Proceed to
  Phase A (a fresh semantic localization on the clean base, which removes the
  ordering confound that may have muddied s590-595's oracle comparisons).

## Phase A — Fresh semantic localization (only if Phase 0 stays 0/71)

The s585-596 oracles ran on a base with the (then-unknown) ordering hazard. Re-
run the same-activation comparison ladder (raw-SWA → attention-output handoff →
post-attention/FFN handoff → routed-FFN activation order → head) on the s604
clean base, this time with the amplifier as a cross-check (a stage whose oracle
passes static but whose acceptance shifts under the amplifier is ordering, not
math). Name the first stage where the live captured draft diverges from a
CPU reference computed through the *same* ordering. This is the localization
that s596 punted before completing (the post-attention residual oracle was
interrupted).

## Phase B — The throughput specdec loop (only if acceptance > 0)

This is the actual B1 throughput work, unchanged from MTP_IMPLEMENTATION.md
Phase B: the TP/EP-coordinated (K+1)-wide block-verify in
`run_token_major_serving_loop` (NOT the unreachable single-slot
`ds4_replay_verify_token_block`), draft K via the MTP forward → verify K+1 →
accept/reject, integration at the appliance generation loop. Gates: serving
acceptance rate > 0 sustained, tolerance 1.0/1.0 on accepted tokens, and the
decode-domain / per-slot uplift at the reference shape (opt-in perf per
VALIDATION_CONTROL_POLICY). Target: (K+1)x effective tokens/step → the 2-3x
that closes the ≥50/slot budget on the step-floor-reduced base.

## Phase C — Restate the verdict

With measured MTP acceptance and the step-floor-reduced base: compute the
achieved per-slot tok/s at S=8/16 and state whether ≥50/slot is reached, or the
residual gap with the specific remaining lever. This is the sprint that can
finally answer the program's exit question with a measured number rather than a
projection.

## Dependencies

- The s604 DENSE_FIX (done, default-on) — the reason to resume.
- The s605+ step-floor campaign for the base-reduction half of the budget
  (Phase B's uplift multiplies whatever base the campaign achieves).
- The s596 acceptance harness + the s590-595 oracle ladder (to re-run on the
  clean base).
- The MTP weight pack: per MTP_IMPLEMENTATION.md the EP=8 MTP weights were
  integrated (s584); confirm the current pack still carries layer-43 MTP
  tensors, else re-run the converter (the doc's steps 1-3).
- Pod, amplifier, profiler, census tooling.
