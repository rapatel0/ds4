# SPRINT-012 Deferred Items

These items are intentionally outside Sprint 012.

## Full 43-Layer Selected-Token Decode

**What:** Execute a full source-layout prompt through all layers, MoE, output
head, and selected-token comparison on V100.

**Why deferred:** Sprint 012 first lands the missing source-BF16 output-head
primitive and bounded logits/top-k gate. Full selected-token decode still needs
coherent router/shared/routed expert execution and scheduler integration.

**Target sprint:** Sprint 013+.

## Public Appliance Deployment

**What:** Expose normal CLI/server serving with startup health checks and
operational defaults.

**Why deferred:** The appliance gate must remain fail-closed until full
selected-token correctness exists.

**Target sprint:** Sprint 013 after readiness improves.

## Production MXFP4 Expert Kernel Integration

**What:** Select and wire production routed expert kernels using TurboMind,
tc-grid, or owned low-bit kernels.

**Why deferred:** Sprint 012 focuses on output-head/logits correctness and
readiness gating. Expert production integration is a larger correctness and
performance surface.

**Target sprint:** Sprint 013 or Sprint 014 depending on the logits gate result.

## Fast Output Head

**What:** Replace bounded diagnostic BF16 output-head reduction with a
production implementation such as FP16 HMMA conversion tiles, FP8/Q8
projection, or vocab-parallel output head.

**Why deferred:** The first requirement is source-faithful logits/top-k
correctness. Throughput-oriented output-head choices should follow measured
bottlenecks and quality checks.

**Target sprint:** Sprint 014.

## Throughput Scheduling, MTP, And Tensor Parallelism

**What:** Add multi-slot batching, MTP/speculative decoding, relay overlap,
wavefront scheduling, and tensor-parallel exceptions.

**Why deferred:** These optimizations can mask baseline correctness issues and
should follow a verified single-slot runtime.

**Target sprint:** Sprint 014+.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Full 43-layer selected-token decode | Sprint 013+ | Needs MoE and scheduler integration |
| Public appliance deployment | Sprint 013 | Needs readiness gate to pass real decode |
| Production MXFP4 expert kernels | Sprint 013/014 | Needs correctness surface |
| Fast output head | Sprint 014 | Needs bounded output-head correctness |
| Throughput/MTP/tensor parallelism | Sprint 014+ | Needs verified baseline runtime |
