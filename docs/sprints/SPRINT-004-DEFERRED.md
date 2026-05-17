# SPRINT-004 Deferred Items

This file captures work discussed during Sprint 004 planning but excluded from
the final sprint scope.

## First Source-Format Math Probe

**What:** Add the first bounded kernel path on top of resident packed bytes,
likely BF16 token embedding or another cheap non-MoE tensor family. Affected
files include `ds4_gpu.h`, `ds4_cuda.cu`, `ds4.c`, and a new correctness test.

**Why deferred:** Sprint 004 must prove structural residency first. Adding
math would force execution-context decisions before pointer ownership and
pack validation are proven.

**Target sprint:** Sprint 005.

**Prerequisites:** Sprint 004 SHIP with verified GGUF/shard provider residency.

**Files:** `ds4_gpu.h`, `ds4_cuda.cu`, `ds4.c`, `tests/*`.

## Source-Model Decode Correctness

**What:** Enable a guarded source-model decode path and compare logits/tokens
against a reference for at least one prompt.

**Why deferred:** Runtime pack loading is not enough to claim numerical
correctness. The decode graph still needs source-format kernels, attention/KV
integration, and layer scheduling.

**Target sprint:** Future, after one or more source-format math probes.

**Prerequisites:** Verified device residency and validated tensor-family
kernels.

**Files:** `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`, `tests/test-vectors/*`.

## MTP And Speculative Decoding

**What:** Load and execute the MTP path and add draft/verify/commit behavior
for higher throughput.

**Why deferred:** User direction was to defer MTP until feasibility is
understood. It should not mask basic residency or single-token correctness
problems.

**Target sprint:** Future.

**Prerequisites:** Correct base decode and stable multi-device scheduling.

**Files:** `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`, MTP-related model inputs.

## Production Multi-GPU Execution Context

**What:** Replace or wrap the existing single-runtime CUDA globals with a
proper `ds4_gpu_context[gpu]` execution model containing streams, cuBLAS
handles, model pointers, scratch, and kernel dispatch state.

**Why deferred:** Sprint 004 only needs upload-only arenas. A production
execution context should be driven by a real math path, not by the residency
smoke.

**Target sprint:** Sprint 005 or later.

**Prerequisites:** Sprint 004 arena proof and selected first math probe.

**Files:** `ds4_gpu.h`, `ds4_cuda.cu`, `ds4.c`.

## HC Relay And Layer Scheduler

**What:** Implement cross-GPU hidden-context relay, boundary synchronization,
and layer-owned execution scheduling.

**Why deferred:** Device residency must be proven first. Sprint 004 records
P2P visibility as planning signal but does not enable peer transfer.

**Target sprint:** Future.

**Prerequisites:** Production multi-GPU execution context and at least one
working layer/tensor-family compute path.

**Files:** `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`.

## Tensor Parallel Variants

**What:** Evaluate tensor-parallel alternatives such as LM-head vocab split,
expert tensor parallelism, or tensor-scheduled layers.

**Why deferred:** Baseline contiguous layer ownership is still the clearest
first appliance target. Tensor parallelism adds communication and correctness
surface before the baseline is proven.

**Target sprint:** Future.

**Prerequisites:** Baseline residency and at least partial decode correctness.

**Files:** `docs/architecture/DS4-V100-LAYOUT.md`, `ds4.c`, `ds4_cuda.cu`.

## Multi-Slot Scheduler And Context Admission

**What:** Add slot admission, active microbatch selection, context-tier memory
planning, and aggregate tok/s scheduling.

**Why deferred:** Sprint 004 does not allocate KV or run decode. Slot
admission depends on the final weight residency and KV implementation facts.

**Target sprint:** Future.

**Prerequisites:** Decode path, KV allocation, and measured per-GPU headroom.

**Files:** `ds4.c`, `ds4.h`, `ds4_cuda.cu`, `docs/architecture/DS4-V100-LAYOUT.md`.

## KV Cache Residency And F8 KV

**What:** Allocate DS4 compressed KV/cache state per layer owner and evaluate
F16 versus F8 cache storage.

**Why deferred:** Sprint 004 is weights-only residency. KV work belongs with
decode correctness and context-window validation.

**Target sprint:** Future.

**Prerequisites:** Layer scheduler and attention path integration.

**Files:** `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`.

## Full Per-Tensor Hashing

**What:** Compute full SHA-256 or equivalent hash per tensor after upload and
readback.

**Why deferred:** Whole-shard hashes plus first/last 4 KiB spot checks are
enough for structural residency. Full per-tensor hashing is expensive and can
be folded into the later correctness harness if needed.

**Target sprint:** Future.

**Prerequisites:** Sprint 004 residency smoke and a need from math validation.

**Files:** `tools/ds4-v100-residency-smoke.c`, `tests/*`.

## JSON Report Output

**What:** Emit machine-readable JSON summaries for reconciliation, residency,
memory, and P2P topology.

**Why deferred:** TSV/log artifacts are sufficient for this sprint and match
existing lightweight repo tooling.

**Target sprint:** Future.

**Prerequisites:** Stabilized artifact fields from Sprint 004.

**Files:** `ds4_pack.c`, `tools/ds4-v100-residency-smoke.c`.

## Optimized Parallel Upload

**What:** Pipeline shard reads and `cudaMemcpy` operations across GPUs, tune
chunk sizes, and reduce full-model upload wall time.

**Why deferred:** Sprint 004 only needs a correct, bounded residency proof.
Upload performance should be measured first and optimized later if it matters.

**Target sprint:** Future.

**Prerequisites:** Baseline sequential or bounded-chunk upload works and logs
I/O timing.

**Files:** `tools/ds4-v100-residency-smoke.c`, `ds4_cuda.cu`.

## Pack-Only Runtime Without GGUF Metadata

**What:** Boot the appliance from a packed directory without requiring the
source GGUF for metadata and reconciliation.

**Why deferred:** Sprint 004 intentionally keeps the source GGUF as the
metadata authority so stale or malformed pack artifacts fail closed.

**Target sprint:** Future.

**Prerequisites:** Stable pack manifest format with complete metadata and
checksums.

**Files:** `ds4_pack.c`, `ds4.c`, `tools/ds4-v100-pack.c`.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| First source-format math probe | Sprint 005 | Needs Sprint 004 residency proof |
| Source-model decode correctness | Future | Needs source-format kernels and scheduler |
| MTP and speculative decoding | Future | Needs correct base decode |
| Production multi-GPU execution context | Sprint 005+ | Needs first math path to shape API |
| HC relay and layer scheduler | Future | Needs execution context |
| Tensor parallel variants | Future | Needs baseline layer-owned decode |
| Multi-slot scheduler and context admission | Future | Needs KV/decode memory facts |
| KV cache residency and F8 KV | Future | Needs attention/decode integration |
| Full per-tensor hashing | Future | Not needed for structural residency |
| JSON report output | Future | TSV/log artifacts sufficient for now |
| Optimized parallel upload | Future | Need baseline upload timings |
| Pack-only runtime without GGUF metadata | Future | Need stable complete pack metadata |
