# SPRINT-010 Deferred Items

These items are intentionally outside Sprint 010. If implementation starts
requiring them, stop and re-scope before continuing.

## Public Appliance Deployment

**What:** Expose a normal CLI/server path with health checks and operational
defaults.

**Why deferred:** Deployment should follow a verified logits-producing V100
source-layout slice. Sprint 010 proved KV ownership and compressor recurrence,
not dense, MoE, output-head, or selected-token correctness.

**Target sprint:** Sprint 012 after the Sprint 011 logits gate.

## Full 43-Layer Logits-Producing Decode

**What:** Execute all transformer layers through output logits and selected
token generation on V100.

**Why deferred:** Sprint 010 should prove a bounded source-referenced
prefill/decode slice first. Full logits will require broader dense, attention,
router, expert, and output-head coverage.

**Target sprint:** Sprint 011+ after the Sprint 010 KV/compressor gate.

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

**Target sprint:** Sprint 013+.

## MTP And Tensor Parallelism

**What:** Wire MTP/speculative decoding or tensor-parallel exceptions such as
vocab-parallel output head or expert TP.

**Why deferred:** These can mask correctness bugs in the baseline layer-owned
runtime.

**Target sprint:** Sprint 014+.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Public appliance deployment | Sprint 012 | Needs logits-producing V100 source-layout gate |
| Full logits-producing decode | Sprint 011+ | Needs dense/MoE/output-head source-layout execution |
| Production routed expert kernels | Sprint 011+ | Needs trusted layer attention/KV path |
| Throughput scheduling | Sprint 013+ | Needs correct baseline decode |
| MTP/tensor parallelism | Sprint 014+ | Needs stable baseline and bottleneck data |
