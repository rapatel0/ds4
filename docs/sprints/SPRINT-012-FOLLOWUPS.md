# SPRINT-012 Follow-Ups

These are not blockers for the Sprint 012 `SHIP` verdict, but they block a
deployable appliance.

## Full Layer/MoE Selected-Token Gate

- **What:** Wire attention output, residual/HC update, router, shared expert,
  routed expert, and output-head/top-k into a coherent bounded selected-token
  comparison on V100.
- **Why:** The gate now proves source-BF16 output-head logits, but not full
  transformer layer execution or model-quality preservation.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 013.
- **Files:** `ds4_cuda.cu`, `ds4_v100_context_cuda.cu`, future scheduler files,
  TurboMind/tc-grid integration surfaces.

## Production Output-Head Kernel

- **What:** Replace the diagnostic BF16 source reduction with a production
  output-head path, likely FP16 HMMA conversion tiles first and vocab-parallel
  or FP8/Q8 alternatives only after quality checks.
- **Why:** Sprint 012 validates correctness boundaries, not output-head
  throughput.
- **Severity:** Important.
- **Suggested sprint:** Sprint 014.
- **Files:** `ds4_cuda.cu`, output-head scheduler/kernel registry files.

## Gate-To-Deployment Integration

- **What:** Integrate `tools/ds4-v100-gate.sh` into the appliance startup or
  deployment workflow, including an operator-visible readiness policy.
- **Why:** The gate is runnable manually today; deployment needs to consume it
  consistently before any public serving unlock.
- **Severity:** Important.
- **Suggested sprint:** Sprint 014 after selected-token readiness improves.
- **Files:** `tools/ds4-v100-gate.sh`, deployment scripts/manifests.

## Throughput And MTP Benchmarks

- **What:** Add aggregate tok/s benchmarks, multi-slot scheduling, MTP, and
  tensor-parallel exceptions after single-slot correctness is real.
- **Why:** The current gate is correctness-oriented and intentionally reports
  missing throughput/MTP readiness.
- **Severity:** Important.
- **Suggested sprint:** Sprint 015+.
- **Files:** benchmark tools, scheduler files, MTP/runtime files.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Full layer/MoE selected-token gate | Critical | Sprint 013 | `ds4_cuda.cu`, V100 context/scheduler files |
| Production output-head kernel | Important | Sprint 014 | `ds4_cuda.cu`, output-head scheduler files |
| Gate-to-deployment integration | Important | Sprint 014 | gate and deployment scripts |
| Throughput and MTP benchmarks | Important | Sprint 015+ | benchmarks, scheduler, MTP/runtime files |
