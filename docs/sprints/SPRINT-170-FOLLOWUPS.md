# Sprint 170 Follow-Ups

## 1. Build the persistent/fused six-route routed-FFN executor

- **What**: Implement a true DS4 persistent or fused routed-FFN executor for the
  per-request 6-route decode shape that keeps gate/up, gated-SiLU activation,
  down, and the route-weighted reduce together as one device boundary, removing
  per-layer host launch and scheduling overhead. This is the executor the
  `fixed6` probe was meant to motivate.
- **Why**: Sprint 170 showed `fixed6` dispatch bypass is flat on the real served
  6-route shape, and Sprints 154-169 ruled out epilogue fusion, stage-count
  tuning, host-orchestrated stream-per-expert pipelines, CUDA Graph replay, and
  slot/layer coalescing. The remaining cost is the gate/up (~61%) and down
  (~31%) MXFP4 GEMM execution itself, so the next lever must change the
  routed-FFN execution model, not the dispatch path.
- **Severity**: Critical (this is the main throughput path).
- **Suggested sprint**: Sprint 171 (Next).
- **Files**: `ds4_cuda.cu`,
  `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu`,
  `kernels/turbomind/ggml-turbomind/api.cc`.

## 2. Document the routed-executor flag in the appliance runbook

- **What**: Add `DS4_V100_TURBOMIND_ROUTED_EXECUTOR`
  (`off`/`auto`/`fixed96`/`fixed768`/`fixed6`) and
  `DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE` to
  `docs/operations/DS4-V100-APPLIANCE.md`, which documents the gate-up/down
  probes but not the routed-executor selector.
- **Why**: The launcher allowlist for this flag was missing `fixed6` until the
  Sprint 170 A/B failed to start (`must be off, auto, fixed96, or fixed768`).
  The runbook has no entry for the flag at all, so its supported values are not
  discoverable.
- **Severity**: Nice-to-have.
- **Suggested sprint**: When convenient.
- **Files**: `docs/operations/DS4-V100-APPLIANCE.md`.

| Item | Severity | Suggested Sprint | Files |
|------|----------|-----------------|-------|
| Persistent/fused six-route routed-FFN executor | Critical | Sprint 171 (Next) | ds4_cuda.cu, ggml-turbomind-ds4-probe.cu, api.cc |
| Document ROUTED_EXECUTOR flag in runbook | Nice-to-have | When convenient | docs/operations/DS4-V100-APPLIANCE.md |
