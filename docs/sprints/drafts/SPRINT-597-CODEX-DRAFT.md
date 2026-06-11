# Sprint 597 Draft: EP-Overhead Instrumentation and B2 Staging

## Overview

Sprint 597 should restart the TP/EP throughput track as a measurement-first B2
cycle. The sprint should not attempt to land fused dispatch, sparse transport,
and barrier restructuring in one implementation pass. The current code has too
many coupled hot-path contracts for that to be a reliable sprint boundary:

- `engine/decode_loop.cu` executes routed FFN, dense overlap, EP contribution
  packing, NCCL/broadcast return, and final compose as one layered sequence.
  The hard synchronization boundaries are `sync_all()` around dense overlap,
  EP pack, NCCL reduce-scatter, and compose; in graph mode `sync_all()` becomes
  a cross-rank event barrier via `enqueue_cross_gpu_stream_barrier()`.
- The line map in the intent is slightly stale: after `run_gate_selected()` and
  `run_down()`, `decode_loop.cu` calls `sync_after_decode_stage("routed_ffn")`,
  not an unconditional `sync_all()`. That helper is an opt-in graph stage sync.
  The unconditional barriers that matter for default flow are later in the EP
  sequence.
- `engine/router_plan.cu` already has GPU route count/fill kernels, but the
  production GPU route planner synchronizes every rank stream and reads
  `d_route_totals` / `d_route_offsets_all` back to host to populate
  `RankState::routes`, `active_experts`, and `max_routes_per_expert`.
- The TurboMind ABI accepts device `expert_offsets`, but the hot DS4 entry
  points currently used by `engine/ep_executor.cu` also require a host
  `total_tokens` integer. The compatibility grouped entry point without
  `total_tokens` performs its own synchronous device-to-host read of
  `expert_offsets[num_experts]`.

Therefore Sprint 597 should produce two durable outputs:

1. A per-rank, per-layer EP decomposition that separates actual MXFP4 expert
   math from route planning, packing, collective/transfer, compose, and barrier
   wait time at the reference shape.
2. A measured B2 execution order and design contract for follow-on sprints:
   device-resident actual-route execution, sparse fp16 row-indexed return, and
   pairwise event scheduling under the no-SYS topology invariant.

The target cycle result remains aggressive: reduce EP from about `9.4 ms/layer`
to about `2 ms/layer` and move aggregate decode from `26.8` tok/s toward at
least `80` tok/s at `32` slots / `256K` / `64` tok/req. Sprint 597 itself should
be considered complete when it produces a trustworthy breakdown and an ordered
implementation plan grounded in that breakdown.

## Use Cases

1. **Throughput owner deciding where to spend implementation time**
   - The sprint artifact should show whether the EP time is dominated by
     TurboMind grouped GEMM, contribution pack, NCCL/broadcast return, compose,
     or barrier wait.
   - The next sprint should start from measured rank/layer costs, not the broad
     `EP = 65.2%` bucket from Sprint 581.

2. **B2 implementer modifying the hot decode path**
   - The plan should identify which contracts must stay fixed for graph capture:
     no per-layer host readback, no dynamic allocation, fixed-capacity buffers,
     static stream/event topology, and no SYS peer paths.
   - The implementer should know whether to extend TurboMind, build a DS4-owned
     full-shape routed executor, or keep fixed host-visible shapes and move
     active-route masking inside kernels.

3. **Validation owner checking correctness and topology**
   - The sprint should use the default tolerance gate:
     selected-token agreement >= `0.99` and generated-sequence agreement >=
     `0.99` vs the promoted control.
   - Peer-SYS counters must remain zero for every candidate run.
   - Perf is explicitly in-scope for this cycle, so candidate reference-shape
     decode tok/s and EP attribution must be recorded when a B2 stage lands.

4. **Future sprint author splitting the cycle**
   - Sprint 597 should leave clear cut lines for Sprint 598+:
     route/executor contract, sparse return transport, event scheduling, then
     optional deeper fusion.

## Architecture

### Current Hot EP Flow

The promoted TP/EP decode flow is approximately:

```text
router logits/top-k
  -> GPU route plan kernels
  -> host reads route totals/offsets for RankState sizing
  -> per-rank TurboMind gate/up
  -> per-rank TurboMind down
  -> dense overlap boundary
  -> EP contribution pack
  -> sync_all/event barrier
  -> NCCL ReduceScatter OR per-source NCCL broadcast/copy return
  -> compose_next_hidden_* kernel
  -> sync_all/event barrier
```

Important source facts:

- `run_gate_selected()` and `run_down()` compute `executor_rows` from
  `routed_executor_rows()`, which currently returns `rank.routes`. In the GPU
  route planner path, `rank.routes` is populated from host copies of device
  route totals.
- `upload_post_attention_fixed_capacity_route_plan_gpu()` is already
  graph-friendly because it sets `rank.routes = rank.route_capacity`,
  `active_experts = kLocalExperts`, and avoids the route-count readback. Prior
  vision entries record that this is correctness-clean but slow because the
  routed FFN still launches the full fixed-capacity envelope.
- The current contribution path either fills a dense `[8, slots, hidden/8]`
  float grid or a compact route segment, then returns it through NCCL
  ReduceScatter or per-source broadcast. `broadcast_ep_return_slices()` loops
  source ranks, uses NCCL broadcast, and performs per-destination copies into
  `d_ep_remote[src]`.
- The no-SYS policy is already encoded in `PeerCopyAccounting` and
  `v100_nvlink_count()`. Direct peer movement must respect that adjacency or
  route through an explicit NVLink neighbor.

### Sprint 597 Instrumentation Design

Add an opt-in EP detail profiler that records CUDA events per rank and layer
for at least:

- route-plan kernels and host readback
- routed input pack/fill, if active in the selected path
- TurboMind gate/up
- TurboMind down
- dense overlap wait, if EP is overlapped with shared/attention dense work
- EP contribution pack/reduce
- NCCL collective or broadcast return
- local copy/peer copy staging
- final compose kernel
- barrier wait time at each `sync_all()` / graph event barrier boundary

The in-band event log should be emitted in a parseable format such as:

```text
tp_ep_ep_detail layer <L> rank <R> stage <name> ms <value> bytes <value> rows <value>
```

The gate should include two views:

- **Eager CUDA-event decomposition**: cheap enough to iterate and capable of
  summing sub-stages back to the Sprint 581 `~9.4 ms/layer` EP bucket.
- **Short Nsight/NVTX capture of the promoted full graph**: heavier, but needed
  to verify that full-capture replay is not hiding a different dependency
  structure than eager.

The eager decomposition is the numerical gate because the existing code only
populates detailed per-stage timers in pure eager. The graph/Nsight capture is
the default-path correlation check.

### B2 Target Shape

The staged B2 design should keep graph-visible shapes static while moving
dynamic route work behind device-side masks or a new ABI:

```text
device route plan
  -> fixed host-visible executor envelope
  -> device active-route metadata
  -> gate/up + down on active rows only, or full-shape DS4 executor with early exits
  -> row-indexed fp16 sparse return payload
  -> static no-SYS one-hop forwarding schedule
  -> pairwise event dependencies
  -> weighted compose from sparse rows
```

The sprint should not assume the existing TurboMind ABI can do true
device-resident actual-route execution. The header and implementation show:

- `ggml_turbomind_mul_mat_grouped()` accepts device `expert_offsets` but reads
  `expert_offsets[num_experts]` back to host internally.
- `ggml_turbomind_mul_mat_grouped_total_tokens()` and the gated-SiLU variants
  avoid that internal read only because the caller supplies host `total_tokens`.
- DS4 fixed-shape probes and reduce epilogues still take host `total_tokens`.
- DS4's loaded `Api` only binds `mmgt`, `mmgs`, and `mmgs_clamped`; the fixed
  probes and route-reduce ABI entries are not part of the current production
  call path.

So the B2 executor decision is explicit: either add a TurboMind/DS4 ABI that
consumes device route totals or active masks, or keep fixed host-visible shapes
and make inactive-row skipping internal to a DS4-owned full-shape executor.

## Implementation

### Phase 1: Add EP Detail Instrumentation

Files:

- `engine/runtime_types.cuh`
- `engine/runtime_options.cuh`
- `engine/runtime_profiler.cu`
- `engine/diagnostics_support.cu`
- `engine/decode_loop.cu`
- launcher/env parsing files for the V100 appliance

Tasks:

- Add an opt-in flag, for example `DS4_V100_TP_EP_EP_DETAIL_PROFILE=1`, that is
  default-off and never changes the promoted hot path.
- Add per-rank CUDA event pairs around route plan, gate/up, down, dense overlap
  boundary, EP pack, collective/broadcast return, final compose, and each
  synchronization boundary.
- Measure barrier wait by recording events before and after the stream wait or
  graph event barrier on the waiting stream. For eager `cudaStreamSynchronize`
  boundaries, record both per-rank host wait and stream-event elapsed time.
- Keep instrumentation allocation static and initialized before decode. No
  `cudaMalloc` or host readback should be inserted into the captured hot path
  unless the flag explicitly selects an instrumentation-only run.
- Emit parseable log lines with rank/layer/stage/rows/bytes/ms fields.

### Phase 2: Parse and Reconcile the Breakdown

Files:

- `tools/ds4-v100-ep-stage-breakdown.py` or equivalent parser
- `docs/sprints/SPRINT-597.md` after execution
- `docs/sprints/drafts/SPRINT-597-*` artifacts as needed

Tasks:

- Parse the detail log into per-layer and per-rank tables.
- Validate that sub-stages sum to approximately the existing EP bucket. The
  expected anchor is Sprint 581's eager attribution: `9.419 ms/layer` EP within
  a `14.445 ms` decode-domain total.
- Report p50/p95/max by stage, rank skew, and the highest wait contributor.
- Separate production cost from diagnostic-only host readbacks. In
  `router_plan.cu`, route totals/offsets are production host readbacks; compact
  counts copied for logging should not be counted as hot-path work unless the
  diagnostic flag is active.
- Archive raw logs, parsed CSV/JSON, and the summary table under the sprint
  artifact directory.

### Phase 3: B2 Decision Gate

Files:

- `docs/sprints/SPRINT-597.md`
- `SPIKE_B_STEERING.md`
- `docs/sprints/VISION.md`
- `docs/sprints/STATUS.md`
- `README.md`

Tasks:

- Decide the next implementation sprint from the measured top contributors.
- If TurboMind gate/up + down is less than roughly one third of EP, do not start
  with deeper GEMM fusion. Start with route/return/compose scaffolding.
- If NCCL/broadcast transfer dominates, prioritize sparse fp16 return and the
  no-SYS one-hop schedule.
- If barrier wait dominates, prioritize pairwise events and removing the global
  `sync_all()` fan-in/fan-out between independent source/destination pairs.
- If host route-count readback dominates more than expected, prioritize a
  device-resident executor contract. If it is only a few percent, do not sell
  host-readback removal as the main performance win; treat it as a graph-capture
  prerequisite.
- Update steering and vision with the measured stage order.
- Update README to supersede or qualify the current "abandoned / research
  archive" status for the reopened B2 investigation.

### Phase 4: Stage B2-A - Device-Resident Actual-Route Executor Contract

This is the first follow-on implementation stage unless the Phase 2 breakdown
clearly points elsewhere.

Files:

- `engine/router_plan.cu`
- `engine/ep_executor.cu`
- `engine/turbomind_bindings.cu`
- `engine/runtime_types.cuh`
- `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`
- `kernels/turbomind/ggml-turbomind/api.cc`

Tasks:

- Keep the fixed graph-visible route capacity for capture.
- Add a DS4 executor contract that carries device route totals or active-route
  masks without requiring `RankState::routes` to be updated by host readback per
  layer.
- Choose one implementation after measurement:
  - new TurboMind ABI that takes device totals/masks; or
  - DS4 full-shape executor that preserves `total_tokens = route_capacity` and
    internally early-exits inactive rows.
- Prove the contract against prior fixed-capacity route-plan correctness audits.
- Keep the old path available behind a gate and as the control.

### Phase 5: Stage B2-B - Sparse FP16 Return and Weighted Compose

Files:

- `engine/decode_loop.cu`
- `engine/ep_compose.cu`
- `engine/runtime_pack.cu`
- `kernels/v100/*.cuh` or new EP transport/compose kernels

Tasks:

- Replace dense float contribution grids with row-indexed fp16 route
  contribution payloads when the B2 gate is active.
- Preserve enough metadata for `compose_next_hidden_compact8_multi_kernel` or
  its replacement to apply route order and weights per slot.
- Keep all capacities fixed at graph-capture time and mask inactive rows on
  device.
- Report transfer bytes per source/destination pair and compare them with the
  current compact/dense return bytes.

### Phase 6: Stage B2-C - Static No-SYS One-Hop Forwarding

Files:

- `engine/runtime_types.cuh`
- `engine/runtime_pack.cu`
- `engine/runtime_resources.cu`
- optional new `engine/ep_topology.cu`

Tasks:

- Compute a static rank-pair schedule at init from the V100 cube-mesh adjacency
  already represented by `v100_nvlink_count()`.
- Directly transfer only self/NV1/NV2 pairs.
- For non-adjacent pairs, forward through a deterministic NVLink neighbor using
  fixed staging buffers and fixed event dependencies.
- Add an assert or validation summary that peer-SYS ops/bytes remain zero.
- Do not mix NCCL for only the non-adjacent minority inside the captured graph
  until a dedicated ordering proof exists. NCCL remains a fallback/control, not
  part of the first mixed transport candidate.

### Phase 7: Stage B2-D - Pairwise Event Dependencies and Optional Fusion

Files:

- `engine/decode_loop.cu`
- `engine/output_head.cu` if shared graph-event helpers need extension
- transport/compose kernels from B2-B/C

Tasks:

- Replace global `sync_all()` boundaries in the B2 return/compose path with
  per-source/per-destination events where dependency is local to a pair.
- Keep graph capture valid by allocating event slots up front and using fixed
  order.
- Only after route execution and sparse return are measured, evaluate whether
  full fused dispatch -> gate/up -> down -> weighted-combine is still necessary.

## Files Summary

- `docs/sprints/drafts/SPRINT-597-INTENT.md`: input brief.
- `docs/sprints/drafts/SPRINT-597-CODEX-DRAFT.md`: this independent draft.
- `AGENT.md`: repo conventions; keep production path narrow, do not add C++,
  preserve correctness before speed.
- `docs/sprints/VISION.md`: TP/EP appliance north star; no PP work; MTP remains
  deferred.
- `SPIKE_B_STEERING.md`: B2 backlog and structural diagnosis.
- `docs/sprints/VALIDATION_CONTROL_POLICY.md`: tolerance gate by default,
  perf opt-in only when named; this cycle opts in.
- `engine/decode_loop.cu`: primary EP sequence and synchronization boundaries.
- `engine/router_plan.cu`: GPU route plan kernels plus production host readback
  of route totals/offsets.
- `engine/ep_executor.cu`: TurboMind gate/up and down calls keyed by host
  `executor_rows`.
- `engine/turbomind_bindings.cu`: loaded TurboMind ABI and expert placement.
- `engine/ep_compose.cu`: compose alternatives and fused 8-source sum.
- `engine/runtime_pack.cu`: NCCL broadcast return and graph copy helpers.
- `engine/runtime_resources.cu`: streams, events, route buffers, NCCL init.
- `engine/runtime_profiler.cu` and `engine/diagnostics_support.cu`: existing
  timing/diagnostic plumbing to extend.
- `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`: public ABI
  proving device offsets are supported but device-resident actual counts are
  not currently enough for production DS4 calls.
- `kernels/turbomind/ggml-turbomind/api.cc`: implementation detail showing the
  compatibility grouped ABI performs a synchronous D2H read while
  `_total_tokens` variants require host `total_tokens`.
- `tools/ds4-v100-http-response-tolerance.py`: correctness gate.

## Definition of Done

- The sprint adds an opt-in EP detail profile mode that builds on the appliance
  target and is off by default.
- A reference-shape instrumentation run on gpu-01 produces per-rank,
  per-layer EP sub-stage timing with rows and bytes.
- The parsed sub-stage totals reconcile with the Sprint 581 EP bucket within an
  explicit tolerance, or the discrepancy is explained by changed graph/eager
  mode, logging overhead, or a newer baseline.
- A short promoted full-capture NVTX/Nsight capture confirms that eager
  decomposition is representative enough to choose the next B2 stage.
- The sprint document records:
  - tolerance gate and control artifact policy
  - candidate artifact path
  - decode tok/s at `32` slots / `256K` / `64` tok/req
  - EP detail table
  - peer-SYS counters
  - selected-token and generated-sequence agreement
  - B2 next-stage decision
- `SPIKE_B_STEERING.md`, `docs/sprints/VISION.md`, `docs/sprints/STATUS.md`,
  and README status language are updated to reflect the reopened B2 track and
  the measured stage order.

## Risks

- **Instrumentation perturbation**: CUDA events, NVTX ranges, and log extraction
  can change scheduling. Keep detailed profiling opt-in and use it for
  attribution, not as the only throughput measurement.
- **Eager vs full-capture mismatch**: Sprint 581 attribution is eager because
  graph replay collapses the region. The sprint must cross-check with short
  full-graph profiling before choosing a stage.
- **ABI mismatch**: assuming device offsets alone solve host-readback removal is
  wrong. Current DS4 calls still need host `total_tokens`; a real B2 executor
  needs a new ABI or fixed-shape internal masking.
- **Topology regression**: direct peer-write all-to-all can silently cross SYS
  pairs unless every pair is checked against cube-mesh adjacency.
- **Graph-capture invalidation**: any dynamic allocation, host readback, or
  variable launch shape in the hot path can invalidate the promoted full-capture
  default.
- **VRAM pressure**: the reference shape already has narrow headroom. New sparse
  staging buffers must be small, static, and preferably replace existing dense
  buffers.
- **Over-scoping**: a single sprint that tries to land all B2 stages can produce
  neither a trustworthy attribution nor a promotable implementation.

## Security

- No new external network access is required.
- Do not log prompts, generated text, model paths with secrets, or environment
  variables unrelated to the benchmark.
- Keep run artifacts under the pod/workspace artifact directory and summarize
  only metrics in repo docs.
- Maintain the global A/B/profiling lock so concurrent V100 jobs do not create
  false OOMs, polluted GPU counters, or misleading throughput.
- Preserve the no-SYS transport invariant; peer-SYS ops/bytes are a hard
  failure for any peer-write candidate.

## Dependencies

- gpu-01 with the 8x V100-SXM2-32GB topology and the current no-SYS policy.
- Current TP/EP model packs, TurboMind sidecars, and `libggml-turbomind.so`.
- The promoted full-capture launcher default:
  `DS4_V100_TP_EP_DECODE_GRAPH_MODE=full`.
- Existing validation harness:
  `tools/ds4-v100-http-response-tolerance.py`.
- Existing serving/profile wrappers and global lock.
- Nsight Systems or the existing profiler workflow for short graph-path
  correlation captures.
- Possible follow-on dependency: a TurboMind ABI extension or DS4-owned
  full-shape routed executor if the measurements point at active-route
  execution.

## Open Questions

1. **Sprint packaging**

   Answer: use one umbrella cycle note in steering/vision, but keep Sprint 597
   focused on instrumentation plus decision. Split implementation into 598+
   stages. This matches the repo convention of focused sprint docs and avoids a
   mega-sprint that couples measurement, ABI changes, transport, and graph event
   ordering.

2. **Instrumentation approach**

   Answer: do both eager in-band CUDA-event decomposition and a short full-graph
   Nsight/NVTX correlation capture. The gate is the eager decomposition because
   the code's existing detailed timers only populate in pure eager and can sum
   sub-stages back to Sprint 581's `9.419 ms/layer` EP bucket. The graph capture
   is required before acting on the result because the promoted default is full
   capture.

3. **Stage order**

   Answer: the decomposition is allowed to override the default order. Before
   measurement, the safest staged order is:

   - device-resident actual-route executor contract or full-shape active-mask
     executor
   - sparse fp16 row-indexed return and weighted compose
   - static no-SYS one-hop forwarding
   - per-pair event dependencies
   - full fused single-kernel dispatch/combine only if still needed

   I would not lead with host-readback removal as a performance claim unless the
   new breakdown contradicts Sprint 581's `~5%` host-sync bucket. Host-readback
   removal is still a graph-capture and architecture prerequisite.

4. **TurboMind ABI**

   Answer from the header and implementation: existing grouped entry points do
   accept device `expert_offsets`, but they do not provide a production-ready
   device-resident actual-route count contract for DS4. The compatibility
   `ggml_turbomind_mul_mat_grouped()` reads `expert_offsets[num_experts]` back
   to host internally. The `_total_tokens` and gated-SiLU variants avoid that
   internal read only by taking host `total_tokens`. The fixed-shape probes and
   reduce epilogues also take host `total_tokens`, and DS4 does not currently
   load those probe symbols in `Api`. B2 needs either a new ABI/persistent
   variant or a DS4 full-shape executor that keeps host-visible
   `total_tokens = route_capacity` and skips inactive work internally.

5. **Sparse return format**

   Answer: use row-indexed fp16 route contributions as the B2 target, but only
   after the decomposition quantifies pack/collective/compose cost. The current
   dense float contribution grid and per-source broadcast path move far more
   structure than the route payload needs. The sparse format must preserve slot
   route order and weights for the compact compose kernel or its replacement.
   If the decomposition shows transfer bytes are not the bottleneck and launch
   serialization is, keep NCCL/compact return for one more stage and attack
   barriers first.

6. **One-hop forwarding**

   Answer: compute a static one-hop schedule at init from the cube-mesh
   adjacency and use it for the peer-write candidate. Do not mix NCCL for only
   non-adjacent pairs inside the captured hot path in the first implementation;
   mixed NCCL plus peer writes has nontrivial ordering implications. NCCL
   remains the fallback/control path. The peer-write candidate must prove
   peer-SYS ops/bytes stay zero.

7. **README/steering reopen note**

   Answer: yes, this cycle should formally supersede the README abandonment
   note for the B2 investigation. The README should not claim an active product
   direction, but it should state that the repo is reopened for a measured
   EP-overhead elimination cycle because Sprint 581's evidence points to
   structural EP scaffolding, not only MoE compute density. Steering, vision,
   STATUS, and README should all name the new baseline and Sprint 597 artifact.
