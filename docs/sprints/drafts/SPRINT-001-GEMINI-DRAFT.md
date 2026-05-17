# Sprint 001: 8x V100 DS4 Appliance Spike (Feasibility Slice)

## Overview
This sprint initiates the transformation of the `ds4` private fork into a dedicated appliance for DeepSeek V4 Flash on the homelab 8x V100-SXM2-32GB stack. The goal is a "feasibility slice" to prove that DS4's narrow architectural focus can outperform or materially simplify the existing `llama.cpp` path. We prioritize multi-GPU connectivity, layer-sharding, and basic correctness over broad kernel optimization.

## Use Cases
- **Feasibility Proof:** Confirm the 81 GB q2-imatrix GGUF can fit and run across 8x 32GB V100s.
- **Hardware-Specific Baseline:** Establish a deterministic SM70 performance baseline for DS4-specific fusions.
- **Appliance Foundation:** Move from a generic single-device CUDA backend to a multi-GPU layer-sharded engine.

## Architecture
We adopt a **Layer-Sharded Appliance** model:
- **Device Plan:** Deterministic contiguous sharding of 43 layers across 8 GPUs (~5-6 layers per GPU).
- **State Ownership:** Each device owns its own `cuBLAS` handle, model weight cache, and KV cache segments.
- **Boundary Transfer:** The ~64 KiB Hidden Context (HC) activation is copied between devices at layer boundaries via `cudaMemcpyPeerAsync` (or host-staged fallback).
- **Narrow API:** Maintain the `ds4_gpu.h` contract but extend it with explicit device/stream awareness.

## Implementation

### Phase 1: Build & Infrastructure
- **SM70 Readiness:** Update `Makefile` and `ds4_cuda.cu` to build cleanly with `CUDA_ARCH=sm_70`. Verify no SM80+ intrinsics block V100 compilation.
- **Multi-GPU Init:** Refactor `ds4_gpu_init` to detect all visible GPUs and print a capability/P2P matrix.
- **Thread-Safe Backend:** Transition from global CUDA/cuBLAS state to a per-device `ds4_cuda_context` structure.

### Phase 2: Layer Sharding & Memory
- **Sharding Logic:** Implement a static `device_map` for the 43 DS4 layers.
- **Distributed Weight Cache:** Modify the weight loader to cache tensor ranges on their designated owning GPU.
- **Sharded KV Cache:** Allocate KV cache blocks on the device owning the corresponding layer to ensure local memory access during decode.
- **HC Transfer:** Implement the layer-boundary copy for the `DS4_N_HC * DS4_N_EMBD` state.

### Phase 3: Feasibility Validation
- **8-GPU Smoke Test:** Create `tests/cuda_multi_gpu_smoke.c` to verify cross-device allocation, HC copy, and basic matmul correctness.
- **Model Load Spike:** Attempt to load the published `q2-imatrix` GGUF across 8 GPUs and report exact per-device memory pressure.

## Files Summary
| File | Action | Description |
| --- | --- | --- |
| `Makefile` | Modify | Add SM70 as a primary target; update regression targets. |
| `ds4_cuda.cu` | Refactor | Move to per-device state; implement sharded weight/KV caches. |
| `ds4_gpu.h` | Modify | Extend API with device/layer awareness. |
| `ds4.c` | Modify | Update graph scheduler to handle cross-device boundaries. |
| `tests/cuda_multi_gpu_smoke.c` | **New** | Multi-device API and connectivity verification. |
| `docs/sprints/VISION.md` | **New** | Establish long-term appliance roadmap. |

## Definition of Done
- [ ] `make cuda CUDA_ARCH=sm_70` builds without errors.
- [ ] `ds4-test --cuda-multi-gpu` passes (new smoke test).
- [ ] 81 GB q2-imatrix GGUF successfully loads and shards across 8x V100.
- [ ] Per-device memory report shows <30GB usage per GPU at 64k context.
- [ ] Coherent token output verified against CPU reference for a short prompt.
- [ ] **Kill-Gate Check:** Performance or simplicity advantage over `llama.cpp` is documented.

## Risks & Mitigations
- **Risk:** P2P access may be disabled/unsupported on the V100 stack.
  - **Mitigation:** Implement host-staged `cudaMemcpy` fallback for HC transfers.
- **Risk:** q2-imatrix GGUF still doesn't fit due to scratch/KV overhead.
  - **Mitigation:** Use `managed` memory for KV cache as a fallback (already partially in `ds4_cuda.cu`).
- **Risk:** SM70 kernel performance is poor.
  - **Mitigation:** This sprint is for *fit and correctness*; performance tuning is deferred to Sprint 002.

## Security Considerations
- **No Change to Security Surface:** This sprint focuses on internal backend refactoring.
- **Isolation:** Ensure `CUDA_VISIBLE_DEVICES` is respected to prevent cross-tenant interference in the homelab.

## Dependencies
- **Hardware:** Access to `gpu-01` (8x V100 32GB).
- **Model:** Published `ds4flash-q2-imatrix.gguf`.
- **Software:** CUDA 12.x toolkit with `nvcc` and `cuBLAS`.

## Open Questions
1. Does the homelab V100 topology support full NVLink P2P, or should we assume PCIe/Host bottlenecks for the HC transfer?
2. Is the 64 KiB HC transfer frequent enough that host-staged copies will kill performance?
3. Should we shard the Output Head across GPUs or keep it on the final GPU?
