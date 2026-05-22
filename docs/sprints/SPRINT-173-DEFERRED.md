# Sprint 173 Deferred Items

Date: 2026-05-22

## Full Monolithic Routed-FFN Kernel

- **What**: Fuse gate/up, gated-SiLU, down, route weighting, and final
  accumulation into one persistent kernel boundary, avoiding `mid_half` and
  `down_routes` global staging in addition to `a_half`.
- **Why deferred**: Too large for the first fused-boundary sprint. Sprint 173
  must first land the descriptor and remove one real expanded intermediate.
- **Target sprint**: Sprint 174 or later if `fused6` is correct and the
  descriptor proves stable.
- **Prerequisites**: Sprint 173 executor contract, liveness logs, and replay
  correctness.
- **Files**: `ds4_cuda.cu`,
  `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu`,
  `kernels/turbomind/ggml-turbomind/api.cc`,
  `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`.

## Persistent TP/EP Scheduler Boundary

- **What**: Convert the routed-FFN executor into a persistent multi-GPU
  tensor/expert parallel boundary with explicit peer ownership, partial output,
  and overlap of compute with communication.
- **Why deferred**: The next sprint is intended to create the local primitive
  TP/EP can reuse. Prior one-layer TP overlays regressed when copy/sync work was
  bolted on after the fact.
- **Target sprint**: Sprint 174+ if Sprint 173 removes a real intermediate but
  does not clear the throughput promotion gate.
- **Prerequisites**: `PARTIAL_ACCUMULATE_F32` smoke and stable descriptor fields.
- **Files**: `ds4_cuda.cu`, `ds4_v100_layer_execute.*`,
  `ds4_v100_scheduler.*`, `tests/cuda_v100_tp_routed_ffn_smoke.c`.

## Attention And Shared-FFN Fusion

- **What**: Audit and potentially fuse attention/shared-FFN wrapper layers so
  compact dtypes stay compact in memory and conversions happen inside GPU
  execution boundaries.
- **Why deferred**: Routed FFN remains the evidenced hot path and the current
  sprint needs a single concrete implementation target.
- **Target sprint**: Future, after routed-FFN boundary decision.
- **Prerequisites**: Routed-FFN liveness instrumentation pattern from Sprint
  173.
- **Files**: `ds4_cuda.cu`, `ds4_v100_layer_execute.*`,
  `docs/architecture/DS4-V100-LAYOUT.md`.

## MTP Production Enablement

- **What**: Move MTP from diagnostic/smoke coverage to production draft commit
  with serving throughput validation.
- **Why deferred**: The user asked to optimize practical serving, but the
  immediate bottleneck is routed-FFN decode throughput and GPU utilization.
- **Target sprint**: Future practical-serving sprint.
- **Prerequisites**: Stable base decode throughput path.
- **Files**: `tools/ds4-v100-mtp-*.c`, `ds4_v100_mtp_*`,
  `tools/ds4-v100-run-appliance.sh`.

## Runbook Cleanup

- **What**: Document all routed executor modes, including `fixed6` and `fused6`,
  in the appliance runbook.
- **Why deferred**: Nice-to-have from Sprint 170; it should not distract from
  the implementation sprint unless the runbook is already being touched.
- **Target sprint**: Future documentation cleanup or Sprint 173 stretch.
- **Prerequisites**: Final decision on `fused6` after A/B.
- **Files**: `docs/operations/DS4-V100-APPLIANCE.md`.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Full monolithic routed-FFN kernel | Sprint 174+ | Sprint 173 descriptor and liveness proof |
| Persistent TP/EP scheduler boundary | Sprint 174+ | Partial-output smoke and stable executor contract |
| Attention/shared-FFN fusion | Future | Routed-FFN boundary decision |
| MTP production enablement | Future | Stable base decode throughput |
| Runbook cleanup | Future or Sprint 173 stretch | Final `fused6` decision |
