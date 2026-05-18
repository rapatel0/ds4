# SPRINT-011 Deferred Items

These items are intentionally outside Sprint 011.

## Full Logits-Producing Decode

**What:** Execute the full source-layout model through output logits or a
selected token on V100.

**Why deferred:** Sprint 011 first proves source projection and bounded
attention/compressor execution. Full logits need coherent dense, attention,
router, expert, shared-expert, output-head, and scheduler integration.

**Target sprint:** Sprint 012+.

## Public Appliance Deployment

**What:** Expose a normal CLI/server path with health checks and operational
defaults.

**Why deferred:** Deployment should follow a verified logits-producing V100
source-layout path.

**Target sprint:** Sprint 013 after the logits gate.

## Production Expert Kernel Selection

**What:** Select and tune TurboMind/tc-grid/owned kernels for MXFP4/INT expert
execution.

**Why deferred:** Correctness for source-format projection and a bounded layer
slice should come before throughput-oriented expert kernel selection.

**Target sprint:** Sprint 012+.

## Throughput Scheduling

**What:** Add multi-slot batching, wavefront overlap, or aggregate tokens/sec
benchmarks.

**Why deferred:** Throughput should be optimized after baseline correctness.

**Target sprint:** Sprint 014+.

## MTP And Tensor Parallelism

**What:** Add MTP/speculative decoding or tensor-parallel exceptions.

**Why deferred:** These can mask baseline correctness issues.

**Target sprint:** Sprint 015+.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Full logits-producing decode | Sprint 012+ | Needs source projection/attention slice |
| Public appliance deployment | Sprint 013 | Needs logits-producing V100 path |
| Production expert kernel selection | Sprint 012+ | Needs correctness baseline |
| Throughput scheduling | Sprint 014+ | Needs deployed/verified baseline |
| MTP/tensor parallelism | Sprint 015+ | Needs stable baseline and bottleneck data |
