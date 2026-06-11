# Sprint 597 Codex Critique

## Source Verification Notes

I verified the code-level claims in the Claude and Gemini drafts against the
current source before evaluating them.

- Promoted graph mode is real and is the launcher default: `tools/ds4-v100-run-tp-ep-appliance.sh:77-86` defaults `DS4_V100_TP_EP_DECODE_GRAPH_MODE=full`, and the `full` branch passes `--decode-cudagraph-gate`, `--decode-cudagraph-replay-probe-gate`, and `--decode-cudagraph-persistent-replay-gate` (`tools/ds4-v100-run-tp-ep-appliance.sh:363-379`).
- Claude is correct that the promoted graph route plan does not use the eager host route-count readback. The eager GPU route-plan path synchronizes and copies `d_route_totals` / `d_route_offsets_all` to host at `engine/router_plan.cu:63-75`, then writes `ranks[rank].routes` from those host values at `engine/router_plan.cu:76-88`. The graph fixed-capacity path is selected when graph mode and `post_attention_fixed_capacity_route_plan_gate` are enabled (`engine/post_attention_ffn.cu:42-45`, `engine/post_attention_ffn.cu:218-226`), and it sets `r.routes = r.route_capacity`, `active_experts = kLocalExperts`, and `max_routes_per_expert = r.routes` without a D2H count readback (`engine/router_plan.cu:198-200`).
- Claude is also correct on the TurboMind ABI nuance. The bound symbols are the `_total_tokens` variants (`engine/turbomind_bindings.cu:52-55`), `routed_executor_rows()` returns `rank.routes` (`engine/turbomind_bindings.cu:372-379`), and `run_gate_selected()` / `run_down()` pass that host scalar to the ABI (`engine/ep_executor.cu:31-56`). The header says the base grouped API reads `expert_offsets[num_experts]` synchronously, while `_total_tokens` avoids that read (`kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h:188-207`). `expert_offsets` is a device pointer in the grouped APIs (`ggml-turbomind-api.h:155-160`, `193-207`, `216-251`). Gemini's "No" answer is therefore only true for shrinking the graph-visible token count below fixed capacity; it is not true for device-resident offsets on the promoted full-capture path.
- The fixed-capacity padding concern is real. Sprint 542 records `192` route rows per rank, `1536` routed rows/layer, p50 max-rank pressure `64`, and max `132`; Sprint 550 route-blocked only the compact pack kernel and found no steady-state throughput win. That supports measuring executor padding separately.
- Both drafts need to correct the promoted graph transport description. In graph mode with default `source_copy_schedule=true`, compact EP return movement does not call `broadcast_ep_return_slices()`. `engine/decode_loop.cu:1173-1194` takes the graph branch and enqueues `enqueue_graph_f32_copy_between_devices()` for every source/destination pair. That helper ignores the device IDs and launches `copy_f32_kernel` over `dst[i] = src[i]` (`engine/runtime_pack.cu:176-190`, `kernels/v100/common.cuh:33-38`). `broadcast_ep_return_slices()` and its per-source NCCL broadcasts are the non-graph branch (`engine/decode_loop.cu:1196-1233`, `engine/runtime_pack.cu:267-345`). Any final sprint must profile and validate this graph remote-copy kernel path, not just NCCL broadcasts.
- The graph barrier claim is directionally right but needs precise call sites. `sync_all()` is a full stream synchronize in eager mode and an event barrier under graph mode (`engine/decode_loop.cu:174-191`). The graph barrier records each rank's stream and dense events and makes every rank stream, dense stream, and copy stream wait on every source (`engine/output_head.cu:1726-1778`). On the default overlap/shared-FFN path, the relevant `sync_all()` calls are around the shared dense work and EP pack (`engine/decode_loop.cu:954`, `978`, `1144`), not after the graph copy branch.
- The current peer-copy accounting is not sufficient evidence for graph remote-copy no-SYS safety. The topology helper maps 16 NVLink-connected pairs and leaves the rest as SYS (`engine/runtime_types.cuh:140-167`), but `record_peer_copy()` appears unused in active source. A zero `peer_copy_sys_bytes` result does not by itself prove that `copy_f32_kernel` remote loads avoided SYS paths.
- The Sprint 396 supporting measurement is often misquoted. The table reports NCCL allreduce at `4.513 ms` vs doubling at `13.366 ms` for 32 tokens, with NCCL faster. It does not support an assumption that custom peer movement is already proven faster than NCCL for the current EP graph path.

## Claude Draft Critique

### Strengths

- Best source-grounded draft overall. It correctly narrows Sprint 597 to instrumentation, decision, and repo-status reopening instead of trying to land B2 kernels in the same sprint.
- Correctly identifies the biggest intent correction: the promoted graph path already uses fixed-capacity device-resident route offsets and avoids the eager route-count host readback. That is important because otherwise Sprint 598 would chase a bottleneck that is not present in the default serving path.
- Correctly distinguishes eager host wall-clock spans from graph replay behavior. The source confirms graph replay is measured as a whole captured region and that existing `std::chrono` splits do not represent sub-stage GPU timings under full capture.
- Strong DoD: reproducible flag, per-rank stage table, raw artifacts, self-consistency check, decision artifact, default-off behavior, tolerance gate, topology artifact, and durable docs.
- Risk analysis is much richer than Gemini's. It calls out instrumentation perturbation, eager/graph shape mismatch, overlap residuals, and the possibility that the measurement refutes the B2 thesis.

### Weaknesses

- Major transport claim is wrong for the promoted full-capture path. The draft describes `broadcast_ep_return_slices()` as the promoted compact full-capture transport and frames the residual as eight serialized NCCL broadcasts. Source shows graph mode takes the direct graph copy-kernel branch (`decode_loop.cu:1173-1194`) and only the non-graph branch calls `broadcast_ep_return_slices()` (`decode_loop.cu:1196-1233`). This affects the architecture section, Q5 answer, risk analysis, and instrumentation boundaries.
- The no-SYS discussion is too confident. Because graph EP return copies are remote-load kernels across all source/destination pairs, and peer-copy accounting does not appear to instrument those kernel loads, "peer-SYS counters stay zero" is not enough proof. The draft should require a transport-specific graph-path validation method for remote kernel loads or a code-level adjacency guard.
- The barrier narrative is directionally right but imprecise. It says three barriers fire at routed-FFN, pack, and transport. The default overlap/shared path has `sync_all()` at shared dense boundaries and after pack (`decode_loop.cu:954`, `978`, `1144`), while post-copy ordering is mostly stream-local in the graph copy branch. Instrumentation should label actual `sync_all()` call sites rather than conceptual stages.
- It overstates "device-masked executor (no ABI change)" as the recommended next step before the sprint measures whether padded GEMM is material. The idea is plausible and source-supported, but it belongs as a candidate interpretation after the decomposition, not a recommendation embedded in a measurement sprint.
- The NVTX/event plan needs more graph-capture specificity. CUDA events are capturable, but inserting per-stage event records into the captured region changes graph nodes and scheduling. The draft mentions perturbation in Risks, but the implementation should require separate `nsys` no-instrumentation runs and report node-count/topology deltas for the flag-on graph.

### Risk Analysis Gaps

- Missing risk that the current promoted graph path may already violate, or at least fail to prove, the no-SYS policy for remote-copy kernels. This is distinct from NCCL SHM/SYS and from explicit peer-copy counters.
- Missing risk that profiler-generated stage events cannot cleanly attribute overlapped rank streams; per-rank stage sums may exceed or under-run critical-path time unless the artifact distinguishes rank-local elapsed time from step critical path.
- Missing risk that graph cache behavior changes when the profile flag is on. The graph cache is keyed by layer, slots, position/final-HC state, root device, and root stream; profiler changes could cause extra captures or invalidations.
- Missing risk around artifact size and overhead of `nsys` on the 43-layer, 8-rank serving run. This matters operationally on the pod but is not a correctness blocker.

### Missing Edge Cases

- Graph remote-copy path across non-NVLink pairs, including how to detect SYS fallback for kernel remote loads.
- Zero-route or low-route ranks under fixed capacity where `r.routes` remains capacity but `d_route_totals` limits active work.
- Duplicate routes to the same slot/rank and all-tokens-to-one-rank skew, especially because compact compose uses per-source route indices/counts.
- Capture cache hit vs cache miss: the artifact should separate first capture, replay, persistent cache hit, and any eager fallback.
- `source_copy_schedule=false`, `ep_return_fp16=true`, and non-compact/ReduceScatter paths should either be explicitly out of scope or have smoke coverage to prove the new profiler does not break them.

### Definition of Done Completeness

Claude's DoD is close to complete for a planning-quality sprint. Required fixes before merge:

- Add a DoD item that the decomposition includes the actual graph EP return movement path: `enqueue_graph_f32_copy_between_devices`/`copy_f32_kernel`, NCCL broadcast only when measuring the non-graph branch, and ReduceScatter only for non-compact coverage.
- Add a no-SYS validation item that is valid for graph remote-copy kernels, not just `peer_copy_sys_bytes=0` and NCCL graph SYS edges.
- Add a graph-perturbation item: compare flag-off vs flag-on graph capture node counts/replay success/cache invalidations, and cite a no-instrumentation `nsys` run as the authority for default-path timing.
- Tighten the self-consistency language to distinguish rank-local stage sums from critical-path wall time when streams overlap.

## Gemini Draft Critique

### Strengths

- Clear and concise. It keeps Sprint 597 focused on instrumentation and a decision gate, with B2 implementation pushed to later sprints.
- Correctly makes the full-capture `nsys` profile authoritative and treats eager timers as a cheap secondary signal.
- Correctly answers that the decomposition should override the hypothesized stage order.
- Includes a useful minimum risk item: eager timers with synchronization can perturb the measured path, so graph profiling must be the final gate.

### Weaknesses

- It does not verify enough source. The draft provides few concrete file/line references and misses the most important current-code distinctions: eager route-plan host readback vs graph fixed-capacity route planning, TurboMind `_total_tokens` variants, and graph return-copy transport.
- Its TurboMind ABI answer is misleading. The base ABI does have a synchronous D2H read, but the promoted path binds and uses `_total_tokens` variants with device `expert_offsets` and a host `total_tokens` scalar. A new ABI is not required for device-resident offsets or for avoiding per-layer route-count readback on full capture. A new device-count ABI might be useful only if the sprint decides to shrink the graph-visible executor grid below fixed capacity.
- It repeats the intent's host-readback framing as if it is an active promoted-path bottleneck. Source shows the full graph path sets `routes = route_capacity` and uses device `d_route_totals` as a limit in downstream kernels, so the current cost is padded fixed capacity and extra route-plan kernels, not per-layer D2H route-count sync.
- It treats the production return path as NCCL ReduceScatter/broadcast without noting the graph `copy_f32_kernel` branch. That omission would produce an incomplete instrumentation sprint.
- The Files Summary is too narrow. It omits `engine/runtime_types.cuh` or equivalent option plumbing, `tools/ds4-v100-run-tp-ep-appliance.sh`, `docs/sprints/STATUS.md`, `docs/sprints/EXPERIMENT-STATUS.md`, and the profile parser/reporting surfaces that will likely need to consume the new TSV/artifacts.
- Minor doc accuracy issue: it cites `AGENT.md`; this repo convention is `AGENTS.md`.

### Risk Analysis Gaps

- No risk for graph-capture perturbation beyond eager timers: adding event records or NVTX-adjacent instrumentation can change graph node count and replay timing.
- No risk for source-copy graph transport and no-SYS validation. This is the most important missing risk because the source currently uses all-pairs remote-copy kernels in graph mode.
- No risk for self-consistency failure due to overlapping streams or per-rank critical-path differences.
- No risk for stale stage order if the instrumentation only measures eager or only labels NCCL branches.
- No risk for route-padding semantics: fixed graph shape is correctness-sensitive per prior sprints, so any attempt to reduce `total_tokens` or active rows must be separately gated.

### Missing Edge Cases

- Empty expert offsets and zero active rows under fixed-capacity `r.routes`.
- Max-rank route skew and all-routes-to-one-rank pressure against `route_capacity`.
- Duplicate same-slot routes and compact compose count/index correctness.
- Ramp-up/ramp-down slot counts below the reference full-slot window.
- Graph cache hit/miss, position-key behavior, and replay after capture.
- Non-default transport branches: eager NCCL broadcast, graph copy kernel, non-compact ReduceScatter, `ep_return_fp16`, and `source_copy_schedule=false`.
- Topology validation for the 12 non-NVLink undirected pairs, or 24 directed
  source/destination edges, per full all-to-all step.

### Definition of Done Completeness

Gemini's DoD is not complete enough to merge as-is.

- It does not specify the reference shape, launcher flag, artifact path, raw capture retention, or exact stage list.
- It does not require selected-token/generated-sequence tolerance, default-off behavior, graph replay success, peer/no-SYS validation, or graph node/cache-invalidation checks.
- It does not require sub-stages to sum within a stated tolerance or explain overlap residuals.
- It does not require the final decision to name measured millisecond deltas per proposed B2 stage.
- It does not require updating `STATUS.md`, `EXPERIMENT-STATUS.md`, or the sprint ledger conventions.

## Merge Recommendation

Use Claude's draft as the base because it has the right sprint shape and the best ABI/source analysis. Before merge, correct its transport model: the promoted graph path must be profiled as fixed-capacity routing plus `copy_f32_kernel` remote return copies, with NCCL broadcast treated as a separate non-graph/eager branch. Carry forward Gemini's concise "nsys graph path is authoritative" framing, but reject Gemini's TurboMind ABI conclusion and expand its DoD substantially.
