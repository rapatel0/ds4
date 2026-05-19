# SPRINT-034 Follow-ups

## Items

1. **What**: Implement resident Q4_K routed expert execution for the MTP
   sidecar.
   **Why**: The prefix chain is now resident, but the MTP FFN block still
   depends on Q4_K routed expert tensors that are currently marked pending.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 035.
   **Files**: `ds4_gpu.h`, `ds4_cuda.cu`, `ds4_v100_mtp.c`,
   `tools/ds4-v100-mtp-prefix-smoke.c`.

2. **What**: Implement the dense resident MTP block path after prefix
   composition: HC attention pre/post, attention projections, raw MTP cache,
   shared Q8_0 expert path, FFN norm, and next HC output.
   **Why**: `mtp_input_hc` is now available, but no resident MTP block forward
   produces draft hidden state or logits.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 035+.
   **Files**: `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4_cuda.cu`.

3. **What**: Add MTP logits/top-k parity and draft/verify/rollback tests.
   **Why**: The readiness blocker remains `missing=mtp_forward`; MTP cannot be
   enabled for serving until the draft token matches a trusted path and failed
   drafts do not corrupt target-model state.
   **Severity**: Critical.
   **Suggested sprint**: After resident MTP block execution.
   **Files**: `ds4.c`, `ds4_v100_mtp.c`, `ds4_v100_replay.c`,
   `tools/ds4-v100-gate.sh`.

4. **What**: Harden or bypass the generic CUDA model-map range cache for tiny
   sidecar tensor-local copies.
   **Why**: Sprint 034 development exposed fragile behavior for repeated
   malloc-backed F32/Q8_0 copies. The shipped smoke avoids this for chain
   validation, but future tests should not depend on allocator accident.
   **Severity**: Important.
   **Suggested sprint**: Cache hardening or test-infra cleanup.
   **Files**: `ds4_cuda.cu`, `tools/ds4-v100-mtp-prefix-smoke.c`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Resident MTP Q4_K routed experts | Critical | Sprint 035 | `ds4_gpu.h`, `ds4_cuda.cu`, `ds4_v100_mtp.c` |
| Dense resident MTP block execution | Critical | Sprint 035+ | `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4_cuda.cu` |
| MTP logits/top-k and draft rollback | Critical | After MTP block | `ds4.c`, `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `tools/ds4-v100-gate.sh` |
| CUDA model-map cache hardening | Important | Cache hardening | `ds4_cuda.cu`, `tools/ds4-v100-mtp-prefix-smoke.c` |
