# SPRINT-008 Deferred Items

These items are intentionally outside Sprint 008. If implementation starts
requiring them, stop and re-scope before continuing.

## Full V100 Source-Layout Prefill

**What:** Execute a complete layer-owned V100 prompt prefill over the native
source model, including all attention, FFN, relay, and output-head paths.

**Why deferred:** Sprint 008 first makes the oracle, guard, KV admission, and
source-format device-anchor contracts executable. Full prefill should build on
those contracts.

**Target sprint:** Sprint 009.

**Prerequisites:** Sprint 008 `SHIP` or an `EXTEND` with KV admission and guard
automation complete.

## Compressed Attention And Indexer Device Population

**What:** Populate raw SWA, attention compressed KV, indexer compressed KV, and
compressor/indexer state tensors on the layer-owned V100 runtime for real
prompt tokens.

**Why deferred:** Sprint 008 only proves the F16 KV admission math and one
bounded source-format device anchor.

**Target sprint:** Sprint 009.

**Prerequisites:** Exact F16 KV admission and a validated source-format device
anchor.

## Production FP8 And MXFP4 Kernels

**What:** Add broad FP8 dense HMMA kernels, MXFP4 routed expert kernels,
low-bit grouped expert kernels, or output-head production kernels.

**Why deferred:** Sprint 008 may add one diagnostic CUDA source-format probe,
but production kernels need the admission and oracle harness first.

**Target sprint:** Sprint 009+.

**Prerequisites:** Source-format anchor parity and source oracle automation.

## Deployment And Serving

**What:** Expose the native source model through normal CLI/server serving,
health checks, service manifests, or production operational defaults.

**Why deferred:** Normal source-layout serving must stay fail-closed until
prefill/decode correctness is proven.

**Target sprint:** Sprint 010+.

**Prerequisites:** V100 source-layout prefill/decode correctness.

## Throughput, Multi-Slot, And Wavefront Scheduling

**What:** Optimize aggregate tok/s, slot batching, wavefront scheduling, relay
overlap, expert utilization, or context-tier admission under load.

**Why deferred:** Throughput tuning should follow correctness and deployment.

**Target sprint:** Sprint 011+.

**Prerequisites:** Correct deployed baseline.

## MTP And Tensor-Parallel Exceptions

**What:** Wire MTP/speculative decoding or selective/full tensor-parallel
exceptions such as vocab-parallel output head or routed-expert TP.

**Why deferred:** These can hide baseline correctness drift and should wait
until the layer-sharded path is stable.

**Target sprint:** Sprint 012+.

**Prerequisites:** Correct deployed baseline and measured bottlenecks.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Full V100 source-layout prefill | Sprint 009 | Needs Sprint 008 oracle/KV/source-anchor contracts |
| Compressed attention/indexer population | Sprint 009 | Needs exact F16 KV admission |
| Production FP8/MXFP4 kernels | Sprint 009+ | Needs source-format anchor parity |
| Deployment/serving | Sprint 010+ | Needs prefill/decode correctness |
| Throughput/multi-slot | Sprint 011+ | Needs correct deployed baseline |
| MTP/tensor-parallel exceptions | Sprint 012+ | Needs stable baseline and measured bottlenecks |
