---
sprint: 006
title: Multi-GPU Execution Context And Layer Skeleton
status: draft
date: 2026-05-17
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-006-INTENT.md
---

# SPRINT-006: Multi-GPU Execution Context And Layer Skeleton

## Overview

Sprint 005 established the first resident diagnostic path for BF16 bytes on V100. However, the runtime still lacks a production-shaped execution context capable of managing the full 8-GPU cluster resources (arenas, streams, handles, and relay buffers) required for real inference.

Sprint 006 introduces the `ds4_gpu_context`, a sidecar production context that owns per-GPU state, and a minimal layer skeleton contract. This skeleton will walk the 8-stage layer map defined in `docs/architecture/DS4-V100-LAYOUT.md`, validating tensor ownership and boundary relay allocations without executing attention or MoE math.

Crucially, this sprint codifies the V100 execution-format policy, ensuring that while source bytes may be low-bit or BF16, the compute target for production activations and GEMMs is V100-native FP16. Normal source-model generation remains guarded.

## Outcome Contract

- `SHIP`: `ds4_gpu_context` supports 8-GPU initialization and metadata reporting; typed resident tensor descriptors replace raw GGUF offsets at call sites; a minimal HC relay primitive is verified for GPU-to-GPU transfers; the layer skeleton successfully walks the 8-GPU map on a V100 pod; source-model generation remains guarded.
- `EXTEND`: Context and skeleton logic land, but multi-GPU relay or real pack metadata reporting is blocked by infrastructure or complex P2P topology issues.
- `STOP`: The per-GPU ownership model is found to be fundamentally incompatible with the existing `ds4.c` engine without a broad destabilizing rewrite.

## Non-Goals

- No full source-model decode correctness.
- No attention, MoE, or routed expert math execution.
- No prefill, compressed KV, or indexer integration.
- No persistent dequantized FP16/F32 copies of source weights.
- No removal of the source-model generation guard.
- No MTP or speculative decoding.
- No broad rewrite of the Metal graph scheduler or CUDA legacy path.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | Sequence anchor: context and skeleton before decode |
| `docs/architecture/DS4-V100-LAYOUT.md` | Topology, layer map, and HC-only boundary anchor |
| `docs/sprints/drafts/SPRINT-006-INTENT.md` | Implementation constraints and V100 policy |
| `ds4_gpu.h` | Existing arena and tensor API |
| `ds4_cuda.cu` | CUDA global state and kernel implementation |
| `ds4_pack.c` | Pack index and GPU byte accounting |
| `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` | Real tensor metadata for context initialization |

## Use Cases

1. **Multi-GPU Context Init**: Initialize 8-GPU state (streams, cuBLAS handles, scratch) from a pack index and report cluster topology (P2P matrix, arena headroom).
2. **Layer Map Validation**: Walk the planned 8-stage layer map and verify that every required tensor has a valid resident descriptor on its owning GPU.
3. **HC Relay Proof**: Execute a "no-op" relay that transfers a hidden-context payload `[active_slots][4][4096]` between adjacent GPUs using the production context.
4. **Execution Policy Enforcement**: A diagnostic report confirms that every tensor family is assigned its correct V100 execution-format (FP16 for compute, BF16/FP8 for source-only).

## Architecture

### V100 Execution-Format Policy

V100 has no native BF16, FP8, or FP4 tensor-core execution. The sprint must implement the following policy:

| Format | Role | Implementation |
|---|---|---|
| **BF16** | Source / Diagnostic | Gather/expand to F32 for diagnostics; no compute. |
| **FP8 / MXFP4** | Source | Resident in packs; consumed by future low-bit kernels. |
| **FP16** | Production Compute | Activations, KV cache, and GEMM targets (Tensor Cores). |
| **FP32** | Control / Reduction | Softmax, Norm, small reductions, and high-precision logic. |

### Production Context (`ds4_gpu_context`)

The context moves away from globals to structured per-GPU ownership:

```c
typedef struct {
    int device_id;
    cudaStream_t stream;
    cublasHandle_t cublas;
    ds4_gpu_arena *arena;
    void *scratch_ptr;
    size_t scratch_size;
    // ... relay buffers and peer-access state
} ds4_gpu_device_context;

typedef struct {
    ds4_gpu_device_context devices[8];
    uint32_t active_device_count;
    // ... cluster-wide metadata
} ds4_gpu_context;
```

### HC Relay Boundary

The Hidden Context (HC) relay is the only data allowed to cross GPU boundaries in the baseline layout.

- **Shape**: `[active_slots][4][4096]` (Flash-style HC blocks).
- **Semantics**: GPU `i` writes its output HC to a relay buffer; GPU `i+1` reads it as input.
- **Implementation**: Prefer `cudaMemcpyAsync` with Peer-to-Peer (P2P) enabled; fallback to host-pinned relay if P2P is unavailable.

### Layer Skeleton

The skeleton is a non-executing "dry run" of the inference path:

```c
int ds4_gpu_layer_skeleton_walk(const ds4_gpu_context *ctx, const ds4_pack_index *idx);
```

For each layer in the 8-stage map:
1. Identify the owning GPU.
2. Resolve descriptors for every tensor (Weights, Norms, Experts).
3. Validate relay buffer availability for the HC boundary.
4. Report memory residency and execution policy for the layer's tensors.

## Implementation

### Phase 1: Context and Descriptor Definition

**Files:**
- `ds4_gpu.h`
- `ds4_gpu_arena_stub.c`
- `ds4_cuda.cu`

**Tasks:**
- [ ] Define `ds4_gpu_context` and `ds4_gpu_device_context`.
- [ ] Define `ds4_gpu_tensor_descriptor` (opaque handle or struct for resident views).
- [ ] Implement `ds4_gpu_context_create` and `ds4_gpu_context_destroy` with multi-GPU resource allocation (streams, handles, scratch).
- [ ] Implement context-aware descriptor resolution from a `ds4_pack_index`.

### Phase 2: Metadata and Policy Reporting

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `tools/ds4-v100-context-smoke.c`

**Tasks:**
- [ ] Add `ds4_gpu_context_report` to emit cluster topology, P2P status, and arena residency.
- [ ] Implement the V100 Execution-Format Policy check in the reporter.
- [ ] Create `tools/ds4-v100-context-smoke.c` to initialize the context and print the report.

### Phase 3: HC Relay Primitive

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`

**Tasks:**
- [ ] Add `ds4_gpu_relay_buffer` allocation to the device context.
- [ ] Implement `ds4_gpu_hc_relay_copy(ctx, src_gpu, dst_gpu, ...)` using P2P or fallback.
- [ ] Add a synthetic relay test to the context smoke tool: transfer a pattern between two GPUs and verify.

### Phase 4: Layer Skeleton Walk

**Files:**
- `ds4.c`
- `ds4_gpu.h`
- `tools/ds4-v100-context-smoke.c`

**Tasks:**
- [ ] Implement `ds4_gpu_layer_skeleton_walk` following the 8-stage layer map.
- [ ] Verify that every tensor required by the model (Experts, MHA, etc.) is resident and has a valid descriptor.
- [ ] Wire the skeleton walk into the context smoke tool.
- [ ] Ensure the source-model generation guard still blocks any actual math or generation.

### Phase 5: Cluster Validation

**Files:**
- `docs/sprints/drafts/SPRINT-006-CONTEXT-REPORT.log`
- `docs/sprints/drafts/SPRINT-006-SKELETON-WALK.log`
- `docs/sprints/drafts/SPRINT-006-RELAY-SMOKE.log`
- `docs/sprints/SPRINT-006-REPORT.md`

**Tasks:**
- [ ] Run the context smoke tool on the 8x V100 pod.
- [ ] Capture the cluster topology and P2P matrix.
- [ ] Capture the full layer skeleton walk output with residency facts.
- [ ] Verify the synthetic HC relay transfer on real hardware.
- [ ] Write the sprint report and update `docs/sprints/VISION.md`.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_gpu.h` | Modify | Add context, device-context, and descriptor types; add skeleton API |
| `ds4_cuda.cu` | Modify | Implement multi-GPU context management, relay, and descriptors |
| `ds4_gpu_arena_stub.c` | Modify | Provide host stubs for the new context APIs |
| `ds4.c` | Modify | Integrate layer skeleton walk and engine wiring |
| `tools/ds4-v100-context-smoke.c` | Create | Production-shaped validation tool for context and skeleton |
| `Makefile` | Modify | Add context smoke tool target |
| `docs/sprints/SPRINT-006-REPORT.md` | Create | Execution report |

## Definition of Done

- [ ] `ds4_gpu_context` initializes and destroys resources across 8 GPUs without leaks.
- [ ] Context reporter correctly identifies P2P availability and arena residency.
- [ ] V100 Execution-Format Policy is codified and verified in diagnostic output.
- [ ] Typed descriptors replace raw GGUF offsets for all tensors in the skeleton walk.
- [ ] HC relay primitive works for `[active_slots][4][4096]` payloads between at least two GPUs.
- [ ] Layer skeleton walks the full 1328-tensor model and validates ownership.
- [ ] Source-model generation remains guarded (no math executed).
- [ ] Cluster logs for topology, skeleton, and relay are archived.
- [ ] `git diff --check` passes.

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| P2P access is disabled or restricted on the pod | Medium | Medium | Implement host-pinned fallback in the relay primitive |
| Context complexity Bloats `ds4_gpu.h` | Medium | Low | Use opaque handles and internal implementation structs where possible |
| Layer map in `LAYOUT.md` has gaps for certain tensors | Low | High | Validation walk will catch gaps; update layout doc as source of truth |
| cuBLAS/Stream initialization on 8 GPUs is slow | Low | Low | Log initialization timing; defer lazy init if needed |

## Security Considerations

- Validate all tensor descriptors and arena offsets before creating resident views.
- Ensure scratch and relay buffers are strictly bounded to prevent cross-GPU or cross-layer memory leaks.
- Preserve the source-model generation guard to prevent unverified compute execution.

## Dependencies

- Sprint 004 pack residency (full 145 GiB model for real metadata reporting).
- `docs/architecture/DS4-V100-LAYOUT.md` as the authoritative layer-to-GPU mapping.
- 8x V100-SXM2-32GB cluster access.

## Open Questions

1. Should the context smoke tool allocate full 32GB arenas for all 8 GPUs, or use the existing resident arenas from a persistent process?
2. Is `[active_slots][4][4096]` the final HC shape, or should the relay primitive be more generic to support future MTP/speculative variants?
3. Should the layer skeleton be purely passive (metadata only) or should it enqueue "no-op" kernels to the streams to verify synchronization?
