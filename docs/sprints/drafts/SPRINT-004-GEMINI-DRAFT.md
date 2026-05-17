---
sprint: 004
title: Runtime Pack Loading And Device Residency Smoke
status: draft
date: 2026-05-17
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
cluster_reference: /Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md
prior_sprint: SPRINT-003-REPORT.md
---

# SPRINT-004: Runtime Pack Loading And Device Residency Smoke

## Overview

Sprint 004 transitions from static planning and manifest generation into active runtime loading of packed shards. The goal is to prove the appliance can consume the `pack-index.tsv` contract, resolve tensors to their physical locations in per-GPU weight shards, and achieve full device residency for the DeepSeek V4 Flash source weights on the 8x V100 cluster.

This sprint focuses on the "plumbing" of multi-GPU weight loading. It intentionally leaves the source-model generation guard in place, as numerical correctness and decode integration are deferred until residency is verified. Success is defined by a successful "residency smoke": all shards loaded into their respective GPU arenas without exceeding the 32GB VRAM limit, with verified tensor identity and byte integrity.

## Use Cases

1. **Runtime Pack Resolution**: The engine consumes `pack-index.tsv` to map every model tensor to its (owning_gpu, shard_file, shard_offset, length) without relying on the original GGUF byte offsets.
2. **Persistent Shard Emission**: Run the full real-model shard emission using `tools/ds4-v100-pack` on the V100 cluster's persistent scratch storage.
3. **Multi-GPU Weight Upload**: Implementation of the first source-faithful GPU upload path that targets per-GPU device arenas.
4. **Device Residency Validation**: A diagnostic smoke test that allocates and populates weight arenas across all 8 GPUs, reporting VRAM usage and checksum/size integrity.

## Architecture

### Pack-Index Contract
The runtime will use the `pack-index.tsv` (generated in Sprint 003) as its primary source of truth for tensor locations. The loader in `ds4.c` must be extended to:
- Parse the TSV into a runtime lookup table.
- Validate that every tensor expected by the model metadata is present in the index.
- Cross-reference `(owning_gpu, shard_file)` to local filesystem paths.

### Device Residency Model
Unlike the current single-device or global-cache model in `ds4_cuda.cu`, Sprint 004 introduces per-GPU residency:
- **Per-Device Arenas**: Explicit allocation of one large contiguous or segmented weight buffer per GPU.
- **Direct Shard Loading**: Reading bytes directly from `gpuN.weights` files into the corresponding device memory, bypassing host-side staging where possible or minimizing it.
- **No Dequantization**: Weights are loaded in their packed source formats (`BF16`, `MXFP4`, `F8_E4M3_B128`) as dictated by the manifest.

### Multi-Device Refactor
A minimal refactor of `ds4_gpu.h` and `ds4_cuda.cu` is required to support multiple active device contexts. The focus is on *residency state* (pointers to buffers on specific PIDs/GPUs) rather than *execution state* (streams/kernels), which remains largely deferred.

## Implementation

### Phase 1: Pack-Index Runtime Reader
**Tasks:**
- [ ] Implement a `pack-index.tsv` parser in `ds4.c`.
- [ ] Extend `struct ds4_model` or equivalent to store shard-resolution metadata.
- [ ] Implement validation that matches model tensor names/dims against the index.
- [ ] **Validation**: Unit test with a synthetic small index and dummy tensor names.

### Phase 2: Cluster Shard Emission
**Tasks:**
- [ ] Recreate the `llamacpp-build-8gpu` pod on the V100 cluster.
- [ ] Run `tools/ds4-v100-pack --emit-shards` targeting persistent scratch storage.
- [ ] Verify file sizes and generate SHA-256 checksums for the resulting `gpu[0-7].weights` shards.
- [ ] **Kill Gate**: Stop if persistent storage is unavailable or insufficient (requires ~145GB).

### Phase 3: CUDA Multi-Device Arena Support
**Tasks:**
- [ ] Refactor `ds4_cuda.cu` to support a `ds4_gpu_context[8]` structure.
- [ ] Implement `ds4_gpu_alloc_weight_arena(gpu_id, size)` to provision the 32GB-limited buffers.
- [ ] Implement the upload path: `pread` from shard files followed by `cudaMemcpy` to the target device.
- [ ] **Validation**: Verify that `cudaSetDevice` is correctly managed during the upload loop.

### Phase 4: Residency Smoke Test
**Tasks:**
- [ ] Implement a dedicated smoke test (e.g., `tests/cuda_residency_smoke.c` or a flag in `ds4_cli`).
- [ ] Load the full 145GB model across the 8 V100s.
- [ ] Print per-GPU VRAM usage (used/free/reserve).
- [ ] Perform spot-checks: copy small ranges back to host and compare against source GGUF/manifest.
- [ ] **Success**: All GPUs populated, no OOMs, checksums match.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4.c` | Modify | Add pack-index parser and tensor resolution logic. |
| `ds4.h` | Modify | Update model structures to hold shard metadata. |
| `ds4_gpu.h` | Modify | Define multi-device arena and context structures. |
| `ds4_cuda.cu` | Modify | Implement per-device allocation and shard upload paths. |
| `tools/ds4-v100-pack.c` | Inspect/Use | Used for shard emission on cluster. |
| `docs/sprints/drafts/SPRINT-004-REPORT.md` | Create | Record results, VRAM usage, and checksums. |
| `tests/residency_smoke.c` | Create | New validation tool for multi-GPU loading. |

## Definition of Done

- [ ] Runtime successfully parses `pack-index.tsv` and binds all model tensors to shard locations.
- [ ] Full 145GB real-model shards emitted and verified on cluster persistent scratch.
- [ ] CUDA backend can allocate and fill 8 separate GPU arenas.
- [ ] Residency smoke test passes on the 8x V100 node without OOM or data corruption.
- [ ] Per-GPU VRAM usage is recorded and fits within the 32GB-per-GPU budget.
- [ ] Generation guard remains active (unless a math harness is explicitly added and passes).
- [ ] Validation artifacts (file sizes, checksums) are recorded in the sprint report.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| Persistent cluster storage is unavailable | Medium | High | Identify storage path early; fall back to multi-step emission if needed. |
| CUDA multi-device state management is buggy | Medium | Medium | Use strict `cudaSetDevice` wrapping and isolated allocation helpers. |
| Shard offsets or length mismatches | Low | High | Use the checksums from `ds4-v100-pack` to verify every shard after emission. |
| VRAM overfill due to unexpected overhead | Low | High | Use the `ds4-v100-plan` reserve (3-4GB) and monitor `cudaMemGetInfo`. |

## Security

- No changes to GGUF parsing; the engine remains restricted to the validated DS4 graph.
- Shard files on cluster scratch should be protected by standard filesystem permissions.
- Bounds-check all TSV parsing and shard-offset arithmetic to prevent VRAM buffer overruns.

## Dependencies

- **Hardware**: 8x V100-SXM2-32GB cluster node.
- **Storage**: ~150GB of persistent scratch on the cluster.
- **Software**: CUDA 11+ / 12+, Sm70-capable toolchain.
- **Artifacts**: `pack-index.tsv` and `DSv4-Flash-256e-fixed.gguf`.

## Open Questions

1. Should the loader support a "mixed mode" where some tensors are loaded from shards and others directly from the GGUF for faster debugging?
2. Is a full SHA-256 check of 145GB of VRAM too slow for a "smoke test"? Should we use spot-checks or CRC32?
3. Should we attempt one "micro-kernel" (e.g., BF16 norm or RMS) to prove the pointers are valid for compute, or stay strictly in residency-only?
4. How should we handle "partial" loads for local testing (e.g., loading only GPU 0's weights on a dev machine)?
