# SPRINT-013 Deferred Items

These items are intentionally outside Sprint 013.

## Full Real-Model Layer Scheduler

**What:** Wire the bounded MXFP4 MoE path into real pack-index descriptors and
the full layer-owned scheduler.

**Why deferred:** Sprint 013 first proves the source-MXFP4 expert primitive and
bounded MoE selected-token composition.

**Target sprint:** Sprint 014+.

## Public Appliance Deployment

**What:** Expose CLI/server serving with readiness checks and operational
defaults.

**Why deferred:** Deployment should follow real selected-token correctness, not
only a bounded synthetic MoE fixture.

**Target sprint:** Sprint 014+.

## Production Expert Kernel Selection

**What:** Replace diagnostic MXFP4 reductions with TurboMind/tc-grid/owned
grouped expert kernels.

**Why deferred:** Correctness and layout semantics must be anchored before
performance kernel selection.

**Target sprint:** Sprint 015+.

## Throughput, MTP, And Tensor Parallelism

**What:** Add multi-slot throughput scheduling, MTP/speculative decoding, and
tensor-parallel expert/output-head exceptions.

**Why deferred:** These can mask baseline single-slot correctness issues.

**Target sprint:** Sprint 015+.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Full real-model layer scheduler | Sprint 014+ | Needs bounded MoE selected-token gate |
| Public appliance deployment | Sprint 014+ | Needs real selected-token correctness |
| Production expert kernels | Sprint 015+ | Needs MXFP4 correctness anchor |
| Throughput/MTP/tensor parallelism | Sprint 015+ | Needs verified baseline runtime |
