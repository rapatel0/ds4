# Sprint 039 Follow-ups

## Items

1. **What**: Compose a full one-token resident MTP block smoke.
   **Why**: Sprint 039 proves deterministic MTP hidden-control logits/top-k,
   but the sidecar still needs one continuous resident path from prefix through
   attention, FFN, output norm, and draft logits.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 040.
   **Files**: `ds4_v100_mtp.c`, `tools/ds4-v100-mtp-logits-smoke.c`,
   `tools/ds4-v100-mtp-attn-smoke.c`, `tools/ds4-v100-mtp-ffn-smoke.c`,
   `tools/ds4-v100-gate.sh`.

2. **What**: Add draft verify/rollback semantics.
   **Why**: MTP-assisted serving must prove rejected draft tokens do not mutate
   target-model KV, MTP raw cache, or slot state incorrectly.
   **Severity**: Critical.
   **Suggested sprint**: After full one-token MTP forward.
   **Files**: `ds4_v100_replay.c`, `ds4_v100_scheduler.c`, `ds4_server.c`,
   `tools/ds4-v100-gate.sh`.

3. **What**: Promote the smoke-only output-head arena into a reusable MTP
   runtime binding if the full MTP executor also needs isolated logits tests.
   **Why**: Sprint 039 uploads `output.weight` inside the smoke to avoid
   scheduler internals. A production executor should reuse the gpu7 base
   output-head binding instead of allocating another arena.
   **Severity**: Important.
   **Suggested sprint**: Sprint 040 or runtime integration.
   **Files**: `ds4_v100_scheduler.c`, `ds4_v100_replay.c`,
   `tools/ds4-v100-mtp-logits-smoke.c`.

4. **What**: Add calibrated full-logit or margin diagnostics for MTP draft
   candidates.
   **Why**: Top-k token parity passed with tiny selected-logit deltas, but the
   next draft/verify sprint will benefit from logging top-1/top-2 margin and
   optional full-logit drift to explain speculative accept/reject behavior.
   **Severity**: Nice-to-have.
   **Suggested sprint**: During draft verify/rollback.
   **Files**: `tools/ds4-v100-mtp-logits-smoke.c`, `ds4_v100_replay.c`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Full one-token resident MTP block | Critical | Sprint 040 | `ds4_v100_mtp.c`, MTP smokes, gate |
| Draft verify/rollback | Critical | After full MTP forward | replay, scheduler, server, gate |
| Reusable output-head binding | Important | Sprint 040 or runtime integration | scheduler, replay, logits smoke |
| MTP candidate margin diagnostics | Nice-to-have | During draft verification | logits smoke, replay |
