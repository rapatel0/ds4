# Sprint 038 Follow-ups

## Items

1. **What**: Add MTP output norm/logits/top-k parity.
   **Why**: Sprint 038 proves the integrated resident MTP attention slice, but
   `missing=mtp_forward` cannot clear until the MTP sidecar produces draft
   logits and a selected draft token matching a trusted oracle.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 039.
   **Files**: `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4.c`,
   `tools/ds4-v100-gate.sh`.

2. **What**: Compose a full one-token resident MTP block smoke.
   **Why**: Prefix, attention, and FFN slices now pass independently, but the
   next correctness risk is the boundary between these slices and the final
   MTP output norm/logits path.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 039 or Sprint 040 depending on logits scope.
   **Files**: `tools/ds4-v100-mtp-prefix-smoke.c`,
   `tools/ds4-v100-mtp-attn-smoke.c`, `tools/ds4-v100-mtp-ffn-smoke.c`,
   `tools/ds4-v100-gate.sh`.

3. **What**: Add draft verify/rollback semantics.
   **Why**: MTP-assisted serving must not corrupt target-model KV or slot state
   when a draft token is rejected.
   **Severity**: Critical.
   **Suggested sprint**: After logits/top-k parity.
   **Files**: `ds4_v100_replay.c`, `ds4_v100_scheduler.c`, `ds4_server.c`,
   `tools/ds4-v100-gate.sh`.

4. **What**: Replace the integrated attention smoke's CPU grouped-output
   tolerance with a CUDA reference or calibrated ULP/error envelope.
   **Why**: The current CPU reference is correct enough for residency and
   composition proof, but grouped Q8_0 output uses a wider tolerance because CPU
   serial accumulation and V100 warp reductions differ. A device-side reference
   or measured envelope would make the gate sharper.
   **Severity**: Important.
   **Suggested sprint**: After MTP logits or during performance hardening.
   **Files**: `tools/ds4-v100-mtp-attn-smoke.c`, `ds4_cuda.cu`.

## Summary

| Item | Severity | Suggested Sprint |
|------|----------|------------------|
| MTP logits/top-k parity | Critical | Sprint 039 |
| Full one-token MTP block smoke | Critical | Sprint 039 or 040 |
| Draft verify/rollback | Critical | After logits parity |
| Sharper grouped-output tolerance | Important | After logits or performance hardening |
