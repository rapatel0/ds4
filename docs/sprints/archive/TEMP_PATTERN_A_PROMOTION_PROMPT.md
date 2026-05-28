# Pattern-A Promotion — relaxed-gate sprint

`TEMP_PARITY_POLICY.md` has been updated. The gate for arithmetic-changing steps
is now **agreement-only**: selected-token agreement ≥ 0.99 AND
generated-sequence agreement ≥ 0.99 → **PROMOTE**. The numeric
max-selected-logit-rel-err threshold (`1e-3`) is **advisory only** — report it
as a diagnostic, do not gate on it. This is the codified version of "we have
plenty of quality margin; tolerance > exact-match."

This sprint promotes the Pattern A items that satisfy the relaxed gate. The
binding rule below is critical:

## No-rerun rule (binding)

**Do not re-run any candidate whose existing s480 (or earlier) evaluation
artifact already satisfies the relaxed gate.** The gate change is a
re-classification of existing evidence, not a justification for fresh A/B runs.
Spend the sprint on the promotion mechanics (default flips, launcher updates,
docs) — not on re-measurement.

If you are tempted to re-run something because the rel-err is "high" — stop.
Rel-err is advisory under the new policy; re-running won't change the agreement
metrics that actually gate the promotion.

## Action list

### Promote NOW on existing evidence (no rerun)

| # | Item | Existing artifact | Action |
|---|---|---|---|
| 1 | **A3 router all-reduce** | `s480-a3-router-allreduce-tolerance` — selected-token 1.0, sequence 1.0, rel-err 0.025 (advisory) | flip default: `--model-router-allreduce-logits-gate` → on. Update launcher / `run-appliance.sh`. Update VISION / status doc. |
| 2 | **EP-compose ReduceScatter (non-compact FP32)** | `s480-ep-reducescatter-tolerance` — agreement 1.0, rel-err 7e-5 | confirm `--nccl-reduce-scatter-compose-gate` defaults are aligned for the non-compact FP32 path; do not toggle the compact-route path. |

### Evaluate ONCE, then promote on agreement (no second run if it passes)

| # | Item | Status | Action |
|---|---|---|---|
| 3 | **A2 mix/RMS all-reduce** | implemented (sprint 478), gated `--tp-hc-current-allreduce-gate`, never re-judged under the tolerance gate | one A/B against the reference shape under the relaxed gate. On selected-token + sequence agreement ≥ 0.99 → flip the gate default on. **Do not re-run if it passes; do not iterate on rel-err.** |

### Stays rejected (genuine correctness failure, not a gate issue)

| # | Item | Reason |
|---|---|---|
| 4 | **A6 rank-local attention projection input** | 1/32 selected-token agreement in the s480 evaluation — that is a real bug in the rank-local-norm path, not gate strictness. Do not promote under the relaxed gate. If you want this win, fix the path (likely the per-rank norm divergence) and re-evaluate as a fresh candidate. |

### Out of scope for this sprint (separate work)

- **EP-compose ReduceScatter for the compact-route default.** Compact-route
  bypasses dense reduce-scatter by design; getting the +8.9% measured on the
  non-compact path into the served compact-route path needs a variable-shape
  collective (AlltoAllv-then-reduce or per-route-group reduce-scatter). That
  is its own sub-sprint, not this one.
- A1-attn / A1-ffn rank-local norms — not yet implemented. Future Pattern A
  sprint.
- Final RMS-norm rank-local — not yet implemented. Future Pattern A sprint.
- Anything Pattern B (row-parallel consumers) — separate sprint, FP8 dequant
  kernel re-tune required per consumer.

## Validation requirements

For #1, #2 (promotion-only, no new measurement needed):

- Verify the launcher / default-gate change correctly enables the path in the
  promoted TP/EP serving binary.
- Confirm `peer_copy_ops=0`, `peer_copy_sys_bytes=0` are unaffected (these are
  arithmetic changes, not transport changes — should be unchanged).
- Run one reference-shape sanity run (32 slots / 256K / 256 req / 64 tok) to
  confirm the now-default-on path still serves cleanly and tok/s has not
  regressed; report decode tok/s for the record. **This is a sanity run, not a
  re-gate.** If the run produces 256/256 with non-zero generated tokens and no
  HTTP errors, the promotion is complete.

For #3 (A2 evaluation):

- One A/B against control on the reference shape (32 slots / 256K / 256 req /
  64 tok).
- Report:
  - selected-token agreement (gating)
  - generated-sequence agreement (gating)
  - decode tok/s (control vs candidate; expect ~+2.7% based on sprint-478
    measured fine-bucket improvement)
  - GPU util
  - max selected-logit rel-err (advisory only; do not act on it)
- If agreement ≥ 0.99 on both → flip default on, done. No second run.
- If agreement < 0.99 → triage with coherence + small-shape authoritative
  reference per policy (rare; A2's per-layer drift is ~1e-6).

## What success looks like

- A3 default-on in the serving binary, sanity run 256/256.
- EP-compose RS confirmed default for non-compact FP32 compose, sanity run
  256/256.
- A2 evaluated once; if it passes (expected), default-on in serving binary,
  same sanity run.
- Updated TEMP_STATUS_REPORT and any sprint/VISION docs to reflect the
  promotions.
- Combined expected tok/s lift on the reference shape from #1 + #3 alone:
  **~+5% (A3 ≈ +2–3%, A2 ≈ +2.7%)**, on top of whatever EP-compose RS
  contributes in its applicable regime.

## Reporting requirements

Per promoted item, in the status report:

- The relaxed-gate metrics (agreement ≥ 0.99 — primary).
- Reference-shape decode tok/s (control vs post-promotion).
- The advisory rel-err number, labeled as advisory.
- Confirmation that `peer_copy_sys_bytes` remained 0 (no transport regression).
- The default-on path enabled and the gate flag retired (or kept as an explicit
  override).

## One-line summary

Promote A3 and EP-compose RS now using the existing s480 evidence (no rerun);
evaluate A2 once under the relaxed gate and promote if agreement passes; leave
A6 rejected (it's a real bug, not a gate issue); everything else is a separate
sprint.
