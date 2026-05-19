# Sprint 035 Follow-ups

## Items

1. **What**: Implement resident MTP FFN block execution after prefix
   composition.
   **Why**: Q4_K routed experts now run from the sidecar arena, but the MTP
   block still needs bias-router selection, shared Q8_0 expert execution,
   routed+shared accumulation, and HC post.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 036.
   **Files**: `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4_cuda.cu`,
   `tools/ds4-v100-mtp-q4k-smoke.c`.

2. **What**: Add MTP attention/raw-cache execution for the decode-only MTP
   block.
   **Why**: Native MTP calls the same layer body with `il=1`, so it uses
   SWA/raw attention rather than compressed ratio-4/indexer attention, but it
   still must update and consume the MTP raw cache before FFN/logits.
   **Severity**: Critical.
   **Suggested sprint**: After or alongside the FFN block slice.
   **Files**: `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4_cuda.cu`.

3. **What**: Add MTP logits/top-k parity and draft verify/rollback tests.
   **Why**: The readiness blocker remains `missing=mtp_forward`; MTP cannot be
   enabled in serving until draft token selection matches a trusted oracle and
   failed drafts do not corrupt target-model state.
   **Severity**: Critical.
   **Suggested sprint**: After resident MTP block execution.
   **Files**: `ds4.c`, `ds4_v100_mtp.c`, `ds4_v100_replay.c`,
   `tools/ds4-v100-gate.sh`.

4. **What**: Generalize or benchmark the Q4_K resident primitive for active
   slot batching.
   **Why**: Sprint 035 proves the K=1 decode primitive. Aggregate throughput
   will eventually need batched route pairs or slot scheduling if MTP remains
   enabled under multi-slot operation.
   **Severity**: Important.
   **Suggested sprint**: Throughput phase after Level 3 correctness.
   **Files**: `ds4_cuda.cu`, `ds4_gpu.h`, `tools/ds4-v100-mtp-q4k-smoke.c`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Resident MTP FFN block | Critical | Sprint 036 | `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4_cuda.cu` |
| MTP attention/raw cache | Critical | Sprint 036+ | `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4_cuda.cu` |
| MTP logits/top-k and rollback | Critical | After block | `ds4.c`, `ds4_v100_mtp.c`, `ds4_v100_replay.c`, gate |
| Batched Q4_K scheduling | Important | Throughput phase | `ds4_cuda.cu`, `ds4_gpu.h` |
