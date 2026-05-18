# SPRINT-009 Deferred Items

These items are intentionally outside Sprint 009. If implementation starts
requiring them, stop and re-scope before continuing.

## Full Logits-Producing Decode

**What:** Execute all 43 layers through output logits and selected-token
generation on V100.

**Why deferred:** Sprint 009 is the first bounded prefill/KV runtime slice. Full
decode needs broader dense, routed expert, router, output-head, and scheduler
coverage.

**Target sprint:** Sprint 010+ after bounded prefill/KV passes.

## Normal Source-Layout Serving Unlock

**What:** Expose native source-layout generation through the normal CLI/server
path.

**Why deferred:** Serving should remain fail-closed until V100 prefill/decode
correctness is proven beyond diagnostic slices.

**Target sprint:** Sprint 010 deployment or later.

## Production FP8 Dense And MXFP4 Expert Kernels

**What:** Implement broad production GEMM kernels for attention/shared FP8
weights or routed MXFP4 experts.

**Why deferred:** Sprint 009 may consume bounded source-format input tiles, but
full production kernels should follow the prefill/KV execution contract.

**Target sprint:** Sprint 010+.

## Throughput Scheduling

**What:** Add multi-slot batching, wavefront scheduling, overlap, performance
benchmarks, or aggregate tok/s optimization.

**Why deferred:** Throughput should follow a correct prefill/decode baseline.

**Target sprint:** Sprint 011+.

## MTP And Tensor Parallelism

**What:** Wire MTP/speculative decoding or tensor-parallel exceptions such as
vocab-parallel output head or routed expert TP.

**Why deferred:** These can obscure baseline correctness issues and should wait
until the layer-owned runtime path is stable.

**Target sprint:** Sprint 012+.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Full logits-producing decode | Sprint 010+ | Needs bounded prefill/KV runtime proof |
| Normal source serving unlock | Sprint 010+ | Needs V100 correctness evidence |
| Production FP8/MXFP4 kernels | Sprint 010+ | Needs runtime prefill/KV contract |
| Throughput scheduling | Sprint 011+ | Needs correct deployed baseline |
| MTP/tensor parallelism | Sprint 012+ | Needs stable baseline and bottleneck data |
