# SPRINT-010 Deferred Items

These items are intentionally outside Sprint 010. If they become required for
the sprint to pass, stop and re-scope.

## Full Selected-Token V100 Decode

**What:** Execute all 43 layers through output logits and selected-token
generation on V100.

**Why deferred:** Sprint 010 is a bridge from existing device compressor outputs
into the F16 KV contract. Full decode needs broader dense, routed expert,
router, output-head, and scheduler integration.

**Target sprint:** Sprint 011+ after the compressor/KV bridge passes.

## Normal Source-Layout Serving Unlock

**What:** Expose native source-layout generation through normal CLI/server
paths.

**Why deferred:** Serving must remain fail-closed until V100 selected-token or
bounded-logit correctness is verified.

**Target sprint:** Deployment sprint after decode correctness.

## Production FP8/MXFP4 GEMMs

**What:** Implement broad production FP8 dense or MXFP4 expert GEMM kernels.

**Why deferred:** Sprint 010 can use bounded synthetic compressor fixtures, but
full production kernels should be gated by the source-oracle comparison path.

**Target sprint:** After single-slot correctness gates.

## Deployment And Health Checks

**What:** Package the appliance as a deployed CLI/server with health checks and
operator defaults.

**Why deferred:** Deployment should follow a verified decode path.

**Target sprint:** Sprint 011+ depending on Sprint 010 outcome.

## MTP, Tensor Parallelism, And Throughput

**What:** Add MTP/speculative decoding, tensor-parallel exceptions, multi-slot
scheduling, wavefront execution, or throughput benchmarks.

**Why deferred:** These can hide baseline correctness bugs.

**Target sprint:** Later throughput sprints.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Full selected-token V100 decode | Sprint 011+ | Needs compressor/KV bridge |
| Normal source serving unlock | Deployment sprint | Needs V100 correctness gate |
| Production FP8/MXFP4 GEMMs | Future | Needs oracle comparison path |
| Deployment and health checks | Sprint 011+ | Needs decode correctness |
| MTP/tensor parallelism/throughput | Later | Needs stable baseline |
