# Sprint 040 Follow-Ups

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Implement MTP draft verify/rollback against target-model state. | Critical | Sprint 041 | `tools/ds4-v100-mtp-forward-smoke.c`, `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `tools/ds4-v100-gate.sh` |
| Replace deterministic MTP forward inputs with a native prompt-token mode once verify semantics are explicit. | Important | Sprint 041 | `tools/ds4-v100-mtp-forward-smoke.c`, `ds4_v100_replay.c` |
| Extract reusable MTP composition helpers after verify/rollback defines the production API. | Important | Sprint 042 | `ds4_v100_mtp.c`, `ds4_v100_mtp.h`, `tools/ds4-v100-mtp-*.c` |
| Add candidate-margin diagnostics to the MTP forward/verify reports so tolerances can be interpreted against acceptance risk. | Nice-to-have | Sprint 041 or 042 | `tools/ds4-v100-mtp-forward-smoke.c`, future verify smoke |

## Details

### MTP Verify/Rollback

What: Add a stateful proof that the MTP draft token can be checked against the
target model and that reject/rollback restores target-model and MTP cache state.

Why: Sprint 040 proves deterministic one-token MTP composition and logits/top-k
stability, but not speculative serving semantics. The gate now correctly reports
`missing=mtp_verify`.

### Native Prompt-Token MTP Mode

What: Feed the MTP forward path from the actual just-committed target token and
HC state instead of deterministic synthetic inputs.

Why: The deterministic smoke isolates kernel composition. Production readiness
needs the same path attached to the replay/serving state machine.

### Reusable Composition Helpers

What: Move stable MTP prefix/attention/FFN/logits composition from the smoke
tool into reusable runtime helpers.

Why: Sprint 040 intentionally kept the composed surface tool-local. That avoids
locking in an API before verify/rollback defines the state ownership boundary.

### Candidate Margin Diagnostics

What: Report the top-1/top-2 margin and selected-token logit deltas in verify
smokes.

Why: Sprint 040's top-5 tokens match exactly, but selected-logit deltas are now
compound-path tolerances. Candidate margins will make later accept/reject risk
easier to reason about.
