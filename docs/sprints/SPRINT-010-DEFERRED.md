# SPRINT-010 Deferred Items

These items are intentionally outside Sprint 010. If implementation starts
requiring them, stop and re-scope before continuing.

## Public Appliance Deployment

**What:** Expose a normal CLI/server path with health checks and operational
defaults.

**Why deferred:** Deployment should follow a verified single-slot V100
correctness slice. Sprint 010 is the integration gate before deployment.

**Target sprint:** Sprint 011.

## Full 43-Layer Logits-Producing Decode

**What:** Execute all transformer layers through output logits and selected
token generation on V100.

**Why deferred:** Sprint 010 should prove a bounded source-referenced
prefill/decode slice first. Full logits will require broader dense, attention,
router, expert, and output-head coverage.

**Target sprint:** Sprint 011+ after Sprint 010 source comparison.

## Production Routed Expert Kernels

**What:** Wire production MXFP4/INT expert kernels into full MoE execution.

**Why deferred:** Sprint 010 focuses on KV, compressor, indexer, and bounded
source comparison. Expert execution can be integrated once the layer slice has a
trusted attention/KV path.

**Target sprint:** Sprint 011+.

## Throughput Scheduling

**What:** Add multi-slot batching, wavefront scheduling, overlap, benchmarks, or
aggregate tokens/sec optimization.

**Why deferred:** Throughput should be driven by a correct baseline decode path.

**Target sprint:** Sprint 012+.

## MTP And Tensor Parallelism

**What:** Wire MTP/speculative decoding or tensor-parallel exceptions such as
vocab-parallel output head or expert TP.

**Why deferred:** These can mask correctness bugs in the baseline layer-owned
runtime.

**Target sprint:** Sprint 013+.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Public appliance deployment | Sprint 011 | Needs Sprint 010 single-slot correctness gate |
| Full logits-producing decode | Sprint 011+ | Needs bounded source-referenced V100 slice |
| Production routed expert kernels | Sprint 011+ | Needs trusted layer attention/KV path |
| Throughput scheduling | Sprint 012+ | Needs correct baseline decode |
| MTP/tensor parallelism | Sprint 013+ | Needs stable baseline and bottleneck data |
