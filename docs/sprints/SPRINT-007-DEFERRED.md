# SPRINT-007 Deferred Items

These items are intentionally outside Sprint 007. If implementation starts
touching them, stop and re-scope before continuing.

## Production V100 Kernels

**What:** Implement FP8 dequant-to-FP16-HMMA dense kernels, MXFP4 grouped
expert kernels, INT8/INT4 alternatives, or output-head production kernels.

**Why deferred:** Sprint 007 is a correctness-oracle sprint. Production kernels
need the oracle as a reference first.

**Target sprint:** Sprint 008+ after source semantics and first-token oracle
evidence exist.

**Prerequisites:** Sprint 007 source-format helpers and oracle report.

**Files:** `ds4_cuda.cu`, `ds4_v100_context_cuda.cu`, future kernel modules.

## Device-Side Oracle Reads

**What:** Make the oracle read directly from resident `ds4_gpu_arena` device
memory or perform device-side source-format reference math.

**Why deferred:** Sprint 006 proved residency; Sprint 007 should prove source
semantics first without coupling correctness to CUDA scheduling.

**Target sprint:** Sprint 008 or first production-kernel sprint.

**Prerequisites:** CPU/host oracle helper parity and first-token correctness
contract.

**Files:** `ds4_v100_context_cuda.cu`, `ds4_cuda.cu`, `ds4_v100_oracle.c`.

## Prefill And KV

**What:** Implement prompt prefill, SWA append beyond the first-token oracle,
compressed KV growth, indexer KV, compression-state updates, and long-context
slot admission.

**Why deferred:** Those paths are Sprint 008 scope and would make Sprint 007 too
large to serve as a clean source-layout correctness gate.

**Target sprint:** Sprint 008.

**Prerequisites:** Sprint 007 source-aware dispatch and oracle verdict.

**Files:** `ds4.c`, `ds4_cuda.cu`, future V100 scheduler/KV modules.

## Multi-Slot Scheduling And Throughput

**What:** Add slot batching, wavefront scheduling, aggregate tok/s tuning,
active-slot admission, or performance benchmarks.

**Why deferred:** Throughput optimization depends on correct single-slot decode
and prefill.

**Target sprint:** Sprint 010+.

**Prerequisites:** Decode correctness, prefill/KV correctness, and deployed
baseline.

**Files:** scheduler/runtime modules, `ds4-server.c`, benchmark tools.

## MTP And Speculative Decode

**What:** Wire MTP descriptors, MTP state, draft/verify/commit scheduling, or
speculative throughput measurements.

**Why deferred:** MTP can hide base decode drift and should wait until
single-token and prompt paths are correct.

**Target sprint:** Sprint 011+.

**Prerequisites:** Base decode, prefill/KV, deployment, and throughput baseline.

**Files:** `ds4.c`, `ds4_cuda.cu`, MTP-specific runtime files.

## Public CLI/Server Exposure

**What:** Add a public `ds4` flag or server/API path that exposes source-layout
oracle mode to normal users.

**Why deferred:** The oracle is a diagnostic guard bypass, not a supported
runtime mode. A dedicated tool/test path is safer for Sprint 007.

**Target sprint:** Future, only if manual diagnostics need it.

**Prerequisites:** Oracle safety proven and no risk of users mistaking it for
production source-model generation.

**Files:** `ds4_cli.c`, `ds4_server.c`.

## Tensor-Parallel Exceptions

**What:** Add vocab-parallel output head, routed expert TP, shared expert TP, or
2-way TP pipeline stages.

**Why deferred:** The architecture doc keeps these as later exceptions. They
should be evaluated after a correct layer-owned baseline exists.

**Target sprint:** Sprint 010+.

**Prerequisites:** Decode/prefill correctness and measured output-head/expert
bottlenecks.

**Files:** V100 scheduler/runtime/kernel modules.

## Full-Logit Oracle Capture

**What:** Capture and store local full-logit oracle artifacts for one or more
prompts.

**Why deferred:** Official API fixtures only provide top-logprob slices. Full
local logits may become useful after the first oracle runs, but they are not
required to plan Sprint 007.

**Target sprint:** Future validation-hardening sprint.

**Prerequisites:** A passing or explainable Sprint 007 source oracle.

**Files:** `tests/test-vectors/`, oracle diagnostic tools.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Production V100 FP8/MXFP4 kernels | Sprint 008+ | Needs source oracle reference |
| Device-side oracle reads | Sprint 008+ | Needs CPU/host oracle parity |
| Prefill and KV | Sprint 008 | Needs single-token correctness |
| Multi-slot and throughput | Sprint 010+ | Needs decode/prefill/deployment |
| MTP/speculative decode | Sprint 011+ | Needs stable baseline decode |
| Public CLI/server exposure | Future | Oracle is diagnostic-only |
| Tensor-parallel exceptions | Sprint 010+ | Needs measured bottlenecks |
| Full-logit oracle capture | Future | Needs source oracle path |
