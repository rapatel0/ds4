# Sprint 597: Gemini Critique of Drafts

This document synthesizes a critique of the two sprint drafts (Claude and Codex) against the original intent for Sprint 597 (EP-Overhead Elimination Cycle). The claims made in the drafts were verified against the repository's source code.

## Overall Assessment

Both drafts correctly identify that attempting to land all B2 structural changes in a single sprint is too risky and advocate for a measurement-first sprint. 

**Claude's draft** is the stronger of the two in its architectural analysis. It correctly identifies that on the *promoted full-capture path*, host readbacks of route counts are *already* avoided by using the fixed-capacity router (`upload_post_attention_fixed_capacity_route_plan_gpu`). It accurately outlines the staging of future sprints based on this reality.

**Codex's draft** provides excellent code-level nuance, particularly in correcting the Intent's line map regarding synchronization barriers. However, it fails to recognize that the production graph path already uses fixed capacities, making some of its architectural concerns moot.

## Claude Draft Analysis

### Strengths
- **Architecture Accuracy:** Correctly identifies that the promoted graph path already uses `upload_post_attention_fixed_capacity_route_plan_gpu`, meaning `rank.routes = route_capacity` (192) and no per-layer host readback occurs. 
- **ABI Understanding:** Provides a highly accurate reading of `ggml-turbomind-api.h`. It notes that while `_total_tokens` requires a host integer, the graph path satisfies this by passing the fixed worst-case capacity (192), avoiding dynamic D2H reads at the cost of a padded 3x grid. 
- **Clear Staging:** Excellent decomposition of the cycle into logical sprints (597 for measurement, 598 for barriers/executor, 599 for sparse A2A, 600 for fusion).
- **Strong DoD:** Very prescriptive, specifically calling out the tolerance gate script and the zero peer-SYS invariant.

### Weaknesses
- **Line Map Inaccuracy:** Claude claims an unconditional `sync_all()` occurs at line 918. Code verification shows line 918 is `sync_after_decode_stage("routed_ffn")`, which is an opt-in graph stage sync. The actual unconditional `sync_all()` calls occur later (e.g., lines 954, 978, 1144).

## Codex Draft Analysis

### Strengths
- **Code Nuance:** Accurately corrects the Intent's line map, identifying that `sync_after_decode_stage("routed_ffn")` is what immediately follows the routed FFN execution, not an unconditional `sync_all()`.
- **ABI Caution:** Correctly points out that because `_total_tokens` is a host integer, true dynamic device-resident execution (where the grid size dynamically shrinks without a host sync) is currently impossible without either an ABI extension or internal masking within a DS4-owned executor.

### Weaknesses
- **Missed Graph Context:** Codex states that the "production GPU route planner synchronizes every rank stream and reads... to host". This is true for the *eager* path (`upload_model_router_route_plan_gpu`), but it misses that the *promoted full-capture graph path* uses the fixed-capacity planner, which already eliminates this host sync.

## Gaps in Risk Analysis & Missing Edge Cases (Both Drafts)

1. **Zero-Active-Token Edge Case:** Neither draft explicitly addresses the edge case of a padded grid where a specific rank receives *zero* active tokens for a layer. If we implement internal device-side masking (early exit), the GEMM must handle an all-zero mask gracefully without hanging or producing NaNs.
2. **Mixed Transport Graph Capture Risks:** Both drafts propose a static one-hop forwarding schedule for direct NVLink pairs while deferring NCCL for SYS pairs, but gloss over the risk of mixing NCCL and raw peer memory writes within the same CUDA graph. NCCL has implicit stream synchronizations; mixing it with explicit `cudaEventRecord`/`cudaStreamWaitEvent` dependencies for peer writes can easily introduce deadlocks during graph replay. 
3. **Graph Topology Mutation:** If the instrumentation flags conditionally insert `cudaEventRecord` into the hot path, the graph topology changes depending on whether profiling is enabled. This risks the "Heisenbug" scenario where the profiling graph behaves differently than the production graph.

## Code-Level Claim Verification

- **`sync_all` vs `sync_after_decode_stage`:** Verified. Codex is correct. The immediate barrier after `run_down` is `sync_after_decode_stage("routed_ffn")` (line 918), while unconditional `sync_all` calls occur at lines 174-192, 954, 1144, 1373, etc.
- **Host Readback Elimination:** Verified. Claude is correct. `upload_post_attention_fixed_capacity_route_plan_gpu` in `router_plan.cu` operates without host readbacks and is used on the promoted graph path.
- **TurboMind ABI:** Verified. `ggml_turbomind_mul_mat_grouped_total_tokens` takes `total_tokens` as a host `int`. Both drafts correctly deduce that a dynamic grid size requires a host readback, and avoiding it requires either passing a fixed worst-case capacity (current behavior) or extending the ABI to take a device scalar pointer.
- **Compose Broadcasting:** Verified. `broadcast_ep_return_slices` is implemented in `engine/runtime_pack.cu` and uses eight sequential NCCL broadcasts, heavily serializing the return path.

## Recommendation

Adopt **Claude's draft** as the foundation for Sprint 597, but merge **Codex's correction** regarding the `decode_loop.cu` synchronization barriers. The sprint should proceed strictly as a measurement and B2-staging effort, ensuring the eager event timers and graph NVTX captures validate the 5% math / 95% scaffolding hypothesis before any hot-path rewrites occur.
