# SPRINT-013 Follow-Ups

These are not blockers for the Sprint 013 `SHIP` verdict, but they block a
deployable appliance.

## Real Pack-Index Layer Integration

- **What:** Bind source-F8, source-BF16, source-MXFP4, router, attention,
  compressor, and output-head primitives to real pack-index descriptors for at
  least one source-layout layer.
- **Why:** Sprint 013 proves a bounded synthetic MoE selected-token path, not a
  real model layer.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 014.
- **Files:** scheduler/context files, `ds4_cuda.cu`, pack descriptor plumbing.

## Shared Expert In Bounded MoE Gate

- **What:** Add source-F8 shared expert gate/up/down composition to the bounded
  MoE smoke or first real layer gate.
- **Why:** DS4 FFN output includes both routed and shared expert paths. Sprint
  013 focused on the routed MXFP4 blocker.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 014.
- **Files:** `tests/cuda_v100_mxfp4_moe_smoke.c`, `ds4_cuda.cu`, scheduler
  files.

## Production Routed Expert Kernel

- **What:** Replace diagnostic MXFP4 row reductions with grouped expert kernels
  using TurboMind/tc-grid/owned low-bit paths once real-layer correctness is
  anchored.
- **Why:** The diagnostic primitive is correctness-first and will not deliver
  target throughput.
- **Severity:** Important.
- **Suggested sprint:** Sprint 015+.
- **Files:** `ds4_cuda.cu`, kernel registry, expert scheduler files.

## Appliance Gate Readiness Policy

- **What:** Change gate readiness from static `ready=false` reasons to
  descriptor-driven readiness once real layer scheduling is wired.
- **Why:** The gate now passes bounded MoE and logits smokes but should remain
  explicit about what real serving surfaces are still missing.
- **Severity:** Important.
- **Suggested sprint:** Sprint 014.
- **Files:** `tools/ds4-v100-gate.sh`, deployment scripts.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Real pack-index layer integration | Critical | Sprint 014 | scheduler/context, `ds4_cuda.cu`, pack plumbing |
| Shared expert in bounded MoE gate | Critical | Sprint 014 | MoE smoke, CUDA, scheduler files |
| Production routed expert kernel | Important | Sprint 015+ | CUDA, kernel registry, expert scheduler |
| Appliance gate readiness policy | Important | Sprint 014 | gate and deployment scripts |
