# Sprint 036 Follow-ups

## Items

1. **What**: Implement MTP raw/SWA attention and cache update.
   **Why**: Sprint 036 starts from a deterministic `after_attn_hc` and proves
   the resident FFN half. Native MTP still runs the layer body at `il=1`, so the
   runtime must update and consume the MTP raw attention cache before FFN.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 037.
   **Files**: `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4_cuda.cu`,
   `tools/ds4-v100-mtp-ffn-smoke.c`.

2. **What**: Add MTP logits/top-k parity.
   **Why**: `missing=mtp_forward` cannot clear until the resident MTP block
   produces draft logits and selected draft tokens that match a trusted CPU or
   source-layout oracle.
   **Severity**: Critical.
   **Suggested sprint**: After MTP attention.
   **Files**: `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `ds4.c`,
   `tools/ds4-v100-gate.sh`.

3. **What**: Add draft verify/rollback semantics.
   **Why**: Speculative MTP serving must not corrupt target-model KV or slot
   state when a draft token is rejected.
   **Severity**: Critical.
   **Suggested sprint**: After logits parity.
   **Files**: `ds4_v100_replay.c`, `ds4_server.c`, `tools/ds4-v100-gate.sh`.

4. **What**: Benchmark/fuse the MTP FFN slice.
   **Why**: Sprint 036 uses separate Q8_0 shared gate/up/down matmuls and
   standalone HC kernels. This is correct and resident, but not the final
   throughput shape.
   **Severity**: Important.
   **Suggested sprint**: Throughput phase after Level 3 correctness.
   **Files**: `ds4_cuda.cu`, `tools/ds4-v100-mtp-ffn-smoke.c`.

## Summary

| Item | Severity | Suggested Sprint |
|------|----------|------------------|
| MTP raw/SWA attention | Critical | Sprint 037 |
| MTP logits/top-k parity | Critical | After attention |
| Draft verify/rollback | Critical | After logits parity |
| MTP FFN fusion/benchmarking | Important | Throughput phase |
