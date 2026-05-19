# SPRINT-033 Follow-ups

## Items

1. **What**: Implement resident MTP F32 prefix norms and HC prefix composition:
   `enorm`, `e_proj`, HC repeat, `hnorm`, `h_proj`, and add into
   `mtp_input_hc`.
   **Why**: Sprint 033 proved resident Q8_0 projection math, but not the full
   native MTP prefix sequence.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 034.
   **Files**: `ds4_v100_mtp.c`, `ds4_gpu.h`, `ds4_cuda.cu`,
   `tools/ds4-v100-mtp-prefix-smoke.c`.

2. **What**: Add resident Q4_K routed expert execution for MTP sidecar tensors.
   **Why**: MTP routed experts are Q4_K, while the main V100 layer path uses
   MXFP4 for routed experts. Reusing the MXFP4 path would be wrong.
   **Severity**: Critical.
   **Suggested sprint**: After resident prefix composition.
   **Files**: `ds4_gpu.h`, `ds4_cuda.cu`, `ds4_v100_mtp.c`.

3. **What**: Investigate whether high-offset mmap ranges near the end of the
   MTP sidecar should be accepted by the generic CUDA model-map range cache.
   **Why**: The first Sprint 033 smoke saw the reference path fail a CUDA copy
   for `mtp.0.h_proj.weight` from the full sidecar mmap. The shipped smoke uses
   a tensor-local host copy for reference, so appliance correctness is not
   blocked.
   **Severity**: Important.
   **Suggested sprint**: When model-map cache hardening is revisited.
   **Files**: `ds4_cuda.cu`, `tools/ds4-v100-mtp-prefix-smoke.c`.

4. **What**: Extend the MTP gate from projection parity to logits/top-k parity
   against the native MTP draft path, then add draft/verify/rollback state
   tests.
   **Why**: `missing=mtp_forward` remains the readiness blocker after Sprint
   033.
   **Severity**: Critical.
   **Suggested sprint**: Sprint 034+.
   **Files**: `ds4.c`, `ds4_v100_mtp.c`, `ds4_v100_replay.c`,
   `tools/ds4-v100-gate.sh`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Resident MTP F32 prefix norms and HC composition | Critical | Sprint 034 | `ds4_v100_mtp.c`, `ds4_gpu.h`, `ds4_cuda.cu`, `tools/ds4-v100-mtp-prefix-smoke.c` |
| Resident Q4_K MTP routed experts | Critical | After prefix composition | `ds4_gpu.h`, `ds4_cuda.cu`, `ds4_v100_mtp.c` |
| High-offset mmap CUDA model-map cache investigation | Important | Cache hardening | `ds4_cuda.cu`, `tools/ds4-v100-mtp-prefix-smoke.c` |
| MTP logits/top-k parity and draft/verify/rollback | Critical | Sprint 034+ | `ds4.c`, `ds4_v100_mtp.c`, `ds4_v100_replay.c`, `tools/ds4-v100-gate.sh` |
