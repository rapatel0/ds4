# Sprint 597: EP-Overhead Elimination Cycle (Instrumentation + Staged B2)

## Overview
Sprint 597 initiates the EP-Overhead Elimination Cycle for the DS4 V100 TP/EP appliance. It acts as an **instrumentation and decision gate** sprint designed to decompose the ~9.4 ms/layer Expert Parallel (EP) stage. By measuring the absolute latency of pack, NCCL collective, grouped GEMM, and barrier-waits on the promoted full-capture graph path, this sprint will empirically confirm or refute the math-vs-scaffolding hypothesis. The output will dictate the order of execution for the staged B2 implementation (fused dispatch, device-resident routing, sparse fp16 peer-write) in subsequent sprints (598+).

## Use Cases
- **Bottleneck Identification**: Pinpoint the exact sources of the ~9.4 ms/layer latency in the EP stage (32 slots / 256K / 64 tok/req) to target structural optimizations.
- **Data-Driven Steering**: Provide a concrete, measured foundation to sequence the B2 track, shifting away from theoretical overhead assumptions.

## Architecture
The architectural focus of this sprint is purely diagnostic:
- **Eager Telemetry**: Extend existing stage timers in `engine/diagnostics_support.cu` and `engine/runtime_profiler.cu` to sub-EP granularity, enabling fast, in-band analysis.
- **Production Profiling**: Leverage NVTX/nsys to capture the promoted full-capture graph path (`DS4_V100_TP_EP_DECODE_GRAPH_MODE=full`). This accounts for the true production latencies of `sync_all()` barriers and CUDA events (which differ from the eager path).
- **Steering Realignment**: Formally supersede the project's abandonment note and update the `VISION.md` and `SPIKE_B_STEERING.md` to reflect the reopened TP/EP throughput program along the B2 track.

## Implementation
1. **Instrument Eager Path**: Add sub-EP timer annotations to `engine/decode_loop.cu` surrounding the core MoE phases:
   - Route planning and host-readbacks (`router_plan.cu`).
   - EP Pack and destination shard reductions (`ep_pack_route_dest_shards_kernel`).
   - Grouped GEMM gate/up/down execution (`ep_executor.cu`).
   - NCCL ReduceScatter and return slice broadcasts (`broadcast_ep_return_slices`).
   - Global `sync_all()` barrier waits.
2. **Execute Eager Profiling**: Run the appliance in eager mode over the steady-state reference shape (32 slots / 256K / 64 tok/req) to gather the initial breakdown.
3. **Execute Graph Profiling**: Run `nsys` against the promoted full-capture graph mode to measure the exact latency of the captured critical path.
4. **Synthesize & Steer**: Produce the final decomposition report and update documentation to sequence Sprints 598+.

## Files Summary
- `engine/diagnostics_support.cu` & `engine/runtime_profiler.cu`: Add new timer contexts and logging for sub-EP stages.
- `engine/decode_loop.cu`: Insert timer boundaries around EP phases.
- `README.md`, `SPIKE_B_STEERING.md`, `docs/sprints/VISION.md`: Documentation updates superseding the MTP punt and formalizing the B2 track.
- `docs/sprints/SPRINT-597.md`: The final output report and B2 stage plan.

## Definition of Done
- A reproducible, per-rank decomposition of the ~9.4 ms/layer EP stage is documented, comparing both eager and full-capture execution.
- The ~5% math vs 95% scaffolding overhead hypothesis is either empirically confirmed or refuted.
- The B2 staged implementation (device-resident routing, sparse peer-write) is cleanly scoped into subsequent N sprints based on the measured bottlenecks.
- The `README.md`, `SPIKE_B_STEERING.md`, and `VISION.md` are updated to reflect the reopening of the TP/EP throughput program.

## Risks
- **Measurement Perturbation**: Eager mode timers rely on `cudaDeviceSynchronize()`, which significantly alters the execution latency by forcing host round-trips. We mitigate this by using the `nsys` profile of the full-capture graph as the final authoritative gate.

## Security
- No changes to network, authentication, or execution isolation. This sprint consists purely of internal performance instrumentation and planning.

## Dependencies
- Availability of `nsys` profiling tools on the V100 pod (`gpu-01`).
- The Sprint 581 promoted full-capture graph baseline (26.8 tok/s).

## Open Questions

1. **Sprint packaging**: *One umbrella cycle doc + N execution sprints, or one mega-sprint?*
   **Answer**: One umbrella cycle doc (Sprint 597) for the instrumentation and decision gate, followed by N execution sprints (598+). Repo conventions (`AGENT.md`, `SPIKE_B_STEERING.md`, and validation policies) strongly mandate focused, single-topic sprints to tightly bound correctness and validation regressions.

2. **Instrumentation approach**: *Extend eager timers vs nsys/nvtx capture of the promoted graph path?*
   **Answer**: Both will be used, but the **nsys capture of the promoted graph path is the authoritative gate**. Eager timers provide cheap in-band insight but inject synchronous host overheads. Since the full-capture graph (`DS4_V100_TP_EP_DECODE_GRAPH_MODE=full`) is the promoted default and utilizes distinct barrier mechanics (`sync_all()` as events), the true production critical path must be profiled natively.

3. **Stage order**: *Does the decomposition override the hypothesized (barrier → a2a → fusion) ordering?*
   **Answer**: **Yes**. The central purpose of Sprint 597 is to replace the ~5% / 95% theoretical hypothesis with empirical data. If the decomposition reveals that NCCL serialization out-scales the barrier-waits, the staging order will strictly follow the largest observed bottleneck.

4. **TurboMind ABI**: *Do existing grouped entry points accept device-resident `expert_offsets` without host token counts?*
   **Answer**: **No.** According to `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`, the base `ggml_turbomind_mul_mat_grouped` API takes a device `expert_offsets` pointer but performs a "synchronous device-to-host read of `expert_offsets[num_experts]`." The `_total_tokens` API variants avoid the synchronization but require a host-side `int total_tokens`. To achieve purely device-resident routing without host readbacks or synchronous blocks, B2 will require a **new entry point** (e.g., accepting `int* d_total_tokens`) or a natively persistent device kernel variant.

5. **Sparse return format**: *Row-indexed fp16 contributions vs dense shard fp16 ReduceScatter?*
   **Answer**: This decision is fully deferred to the instrumentation phase. If the `nsys` profile reveals that the actual execution time of `broadcast_ep_return_slices` is negligible but its serialization/launch overhead is massive, row-indexed fp16 contributions directly mapped to active routes may be required.

6. **One-hop forwarding**: *Static schedule computed from the cube mesh vs reusing NCCL for non-adjacent pairs?*
   **Answer**: A **static schedule computed at init from the cube mesh** must be used. Mixing NCCL collectives with direct peer transfers inside the same captured graph introduces severe ordering complexities and stream-synchronization violations against the no-SYS cube mesh constraints. A static NVLink forwarding schedule is fully deterministic and graph-capturable.

7. **README/steering "reopen" note**: *Does this cycle formally supersede the abandonment note?*
   **Answer**: **Yes.** The README's abandonment note will be explicitly superseded. `README.md`, `SPIKE_B_STEERING.md`, and `VISION.md` will be updated to document that the TP/EP throughput program has reopened via the B2 track, with the LP MTP track remaining officially punted.