# Sprint 173: Reusable Fused Routed-FFN Boundary

## Overview
This sprint implements a reusable fused routed-FFN executor boundary for DeepSeek V4 Flash on V100. The primary goal is to eliminate global-memory staging of expanded intermediates—specifically the gathered and FP16-casted activation buffer (`a_half`)—by forming activation tiles directly inside the kernel boundary. This work establishes the architectural primitive for future Tensor Parallelism (TP) and Expert Parallelism (EP) by defining a clean contract for partial-output execution.

## Use Cases
- **High-Throughput Serving:** Reducing global memory traffic to improve continuation/decode tokens per second on 16-slot/256K workloads.
- **V100 Optimization:** Leveraging FP16 tensor cores while keeping storage in compact FP4/MXFP4 formats, performing on-the-fly expansion.
- **TP/EP Preparation:** Providing a modular executor that can participate in multi-GPU reduction via a partial-output interface.

## Architecture

### 1. Fused Executor Descriptor
A new C structure, `ds4_v100_ffn_exec_descriptor`, will encapsulate the execution state. It separates the **contract** (what to do) from the **implementation** (how to launch).

```c
typedef struct {
    // Dimensions
    uint32_t hidden;
    uint32_t mid;
    uint32_t n_experts;
    uint32_t n_routes;
    uint32_t n_tokens;

    // Weights (MXFP4 views)
    const ds4_gpu_turbomind_mxfp4_matrix_view *gate_up;
    const ds4_gpu_turbomind_mxfp4_matrix_view *down;

    // Inputs (FP32)
    const ds4_gpu_tensor *x_f32;           // [n_tokens, hidden]
    const ds4_gpu_tensor *selected_i32;    // [n_tokens, n_routes]
    const ds4_gpu_tensor *weights_f32;     // [n_tokens, n_routes]
    
    // Auxiliary (from route build)
    const int *sorted_pairs;               // [total_routes] (expert << 16 | token_pos)
    const int *expert_offsets;             // [n_experts + 1]

    // Output
    ds4_gpu_tensor *out_f32;               // [n_tokens, hidden]

    // Modes
    uint32_t flags;                        // e.g., DS4_FFN_EXEC_ACCUMULATE
    uint32_t tp_mode;                      // FULL_OUTPUT, PARTIAL_SUM, EXPERT_SHARD
} ds4_v100_ffn_exec_descriptor;
```

### 2. The Fused Kernel Boundary
The implementation will introduce a "fused" path that bypasses `tm_gather_f32_to_f16_kernel`. Instead of a global gather, the matmul kernels (or a monolithic wrapper) will:
1.  Reference the source `x_f32` (FP32) using `sorted_pairs`.
2.  Perform an on-the-fly FP32->FP16 cast during global-to-shared (G2S) or global-to-register (G2R) movement.
3.  Execute the MXFP4 matmul (gate/up) using TurboMind's tensor-core logic.
4.  Apply gated-SiLU activation.
5.  Execute the second matmul (down).
6.  Optionally reduce/scatter back to `out_f32` or stop at a partial sum for TP.

### 3. TP/EP Primitive Seed
The executor is designed as a "leaf" node in a larger parallel graph. By supporting `expert_offsets` and a `partial_sum` output mode, it can be dropped into an 8-GPU orchestrator that performs All-Reduce or All-to-All around this boundary.

## Implementation

### Phase 1: Contract and Scaffold
- Define `ds4_v100_ffn_exec_descriptor` in a new header `ds4_v100_ffn_executor.h`.
- Implement `ds4_v100_ffn_exec_fused()` host-side wrapper.
- Add `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=1` runtime flag to `ds4_cuda.cu`.

### Phase 2: Fused 6-Route Prototype
- Implement a focused CUDA kernel for the production 6-route shape (`total_routes=6`).
- This kernel will fuse:
    - FP32 input read + cast to FP16.
    - Expert gate/up matmul.
    - Activation.
    - Expert down matmul.
- Bypass the global `a_half` allocation for this path.

### Phase 3: TurboMind Integration
- Update `cuda_tm_routed_mxfp4_packed_impl` to use the new executor when the flag is enabled.
- Preserve the existing multi-kernel TurboMind path as a fallback for non-production shapes.
- Implement a liveness audit in the log output to confirm `a_half` was not materialized.

### Phase 4: Instrumentation and Benchmarking
- Add `timing_ffn_fused_ms` to the execution report.
- Report "Global Intermediates Bypassed: a_half" in the startup logs.
- Run served 16-slot/256K A/B with separate token-per-second splits for prompt/gen/cont.

## Files Summary
- `ds4_v100_ffn_executor.h`: Contract and descriptor definitions.
- `ds4_v100_ffn_executor.cu`: Fused kernel implementation and host orchestration.
- `ds4_cuda.cu`: Integration into the main TurboMind path and runtime flag handling.
- `tests/cuda_v100_fused_ffn_smoke.c`: Direct unit test for the fused executor.
- `tools/ds4-v100-replay.c`: Updated to support the new executor mode for profiling.

## Definition of Done
- [ ] V100 build passes with no regressions in standard TurboMind paths.
- [ ] `tests/cuda_v100_fused_ffn_smoke.c` passes with bit-exact or nearly-exact parity to the legacy path.
- [ ] Log evidence confirms `a_half` global staging is bypassed in the fused path.
- [ ] Replay benchmarking confirms the fused boundary is stable under production shapes.
- [ ] Served 16-slot/256K A/B results show `>10%` improvement in continuation tokens/s (or the implementation is kept as a TP seed if improvement is lower but architecture is correct).

## Risks
- **Numerical Sensitivity:** Gated-SiLU and MXFP4 dequantization are sensitive; small differences in fusion order or accumulation could lead to drift.
- **Kernel Complexity:** A full monolithic gate/up+down kernel might be large for one sprint; the "scaffold + activation tile" fallback is available to manage risk.
- **Resource Contention:** Fused kernels may use more registers or shared memory, potentially affecting occupancy compared to the discrete TurboMind kernels.

## Security
- No new security risks identified; memory access remains within existing tensor boundaries.
- Descriptor validation must ensure `sorted_pairs` and `expert_offsets` do not cause out-of-bounds reads.

## Dependencies
- Requires recent TurboMind ABI with gated-SiLU support (already present in the current branch).
- Depends on the `sorted_pairs` routing metadata produced by `cuda_tm_build_routes`.

## Open Questions
1. Should the fused kernel use a custom MXFP4 dequantization loop or try to reuse TurboMind's `MainloopSm70` templates? (Plan: Attempt template reuse for maintenance, but fallback to custom for 6-route decode optimization).
2. Is it worth removing the shared FFN intermediate as well, or is `a_half` the dominant staging bottleneck?
3. How should partial sums be represented in the `out_f32` tensor when in TP mode?
