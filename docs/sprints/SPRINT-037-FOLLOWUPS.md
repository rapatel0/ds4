# Sprint 037 Follow-ups

## Items

1. **What**: Integrate the full MTP attention projection/output slice.
   **Why**: Sprint 037 proves raw/SWA attention and ring-cache semantics from
   resident sinks, but it uses synthetic Q/KV. Native MTP still needs resident
   HC attention control, `attn_norm`, Q/KV projections, Q head norm/RoPE, KV
   norm/RoPE, grouped attention output projection, and HC expansion.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 038.
   **Files**: `ds4_v100_mtp.c`, `ds4_cuda.cu`,
   `tools/ds4-v100-mtp-attn-smoke.c`, `tools/ds4-v100-gate.sh`.

2. **What**: Add MTP output norm/logits/top-k parity.
   **Why**: `missing=mtp_forward` cannot clear until the resident MTP path
   produces draft logits and selected draft tokens that match a trusted CPU or
   source-layout oracle.
   **Severity**: Critical.
   **Suggested sprint**: After integrated MTP attention.
   **Files**: `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4.c`,
   `tools/ds4-v100-gate.sh`.

3. **What**: Add draft verify/rollback semantics.
   **Why**: Speculative MTP serving must not corrupt target-model KV or slot
   state when a draft token is rejected.
   **Severity**: Critical.
   **Suggested sprint**: After logits parity.
   **Files**: `ds4_v100_replay.c`, `ds4_v100_scheduler.c`, `ds4_server.c`,
   `tools/ds4-v100-gate.sh`.

4. **What**: Replace focused synthetic MTP attention inputs with native prefix
   output in an integrated one-token smoke.
   **Why**: The raw attention wrapper is correct in isolation, but the next
   correctness risk is the composition boundary between prefix, attention,
   FFN, and output-head logits.
   **Severity**: Important.
   **Suggested sprint**: Sprint 038 if scope allows, otherwise the logits
   sprint.
   **Files**: `tools/ds4-v100-mtp-prefix-smoke.c`,
   `tools/ds4-v100-mtp-attn-smoke.c`, `tools/ds4-v100-mtp-ffn-smoke.c`.

## Summary

| Item | Severity | Suggested Sprint |
|------|----------|------------------|
| Integrated MTP attention projection/output | Critical | Sprint 038 |
| MTP logits/top-k parity | Critical | After attention |
| Draft verify/rollback | Critical | After logits parity |
| Native-prefix integrated smoke | Important | Sprint 038 or logits sprint |
