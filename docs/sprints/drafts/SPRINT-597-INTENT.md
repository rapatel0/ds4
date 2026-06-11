# Sprint 597 Intent: EP-Overhead Elimination Cycle (Instrumentation + Staged B2)

## Seed

Plan the EP-overhead elimination cycle for the DS4 V100 TP/EP appliance:

1. An **instrumentation sprint** that decomposes the ~9.4 ms/layer EP stage
   (pack vs NCCL collective vs grouped GEMM vs barrier-wait, per rank, on
   gpu-01) to confirm the math-vs-scaffolding split.
2. A **staged B2 implementation** — fused dispatch → grouped-GEMM → weighted-
   combine with device-resident routing (no per-layer host readback of route
   counts), sparse fp16 peer-write all-to-all with a static one-hop NVLink
   forwarding schedule that respects the no-SYS cube-mesh constraint, and
   per-pair event dependencies replacing global `sync_all()` barriers.

Baseline: SPRINT-581 recorded `26.8` tok/s aggregate decode at 32 slots /
256K / 64 tok/req on the promoted full-capture default, with **EP = 65.2%**
of the decode-domain attribution. Roofline analysis (below) says the
weight-streaming ceiling at this shape is ~1,000+ tok/s aggregate, so the
gap is structural overhead, not MoE compute density.

## Context

- **Repo status**: README currently marks the repo "abandoned / research
  archive" after the MTP punt (2026-05-30). This cycle re-opens the TP/EP
  throughput program on the B2 track, which `SPIKE_B_STEERING.md` ranked
  Med-High and never executed. MTP stays punted — do not reopen it.
- **Promoted serving path**: TP/EP 8-GPU appliance, full-capture CUDA graph
  decode (`DS4_V100_TP_EP_DECODE_GRAPH_MODE=full`), NCCL transport on the
  no-SYS topology policy (Sprint 479 removed direct peer-copy hot-path
  transport; Sprint 530/531 set the no-SYS/no-SHM budget).
- **Validation**: per `docs/sprints/VALIDATION_CONTROL_POLICY.md`, tolerance
  gate is the default (selected-token AND generated-sequence agreement
  ≥ 0.99 vs control). **This cycle opts into perf measurement** — its failure
  mode is exactly "structurally landed but perf didn't transfer to serving",
  the canonical opt-in class. Builds and serving A/B runs happen on the pod
  (gpu-01, 8x V100-SXM2-32GB), not the laptop.
- **Vision**: `docs/sprints/VISION.md` North Star is the TP/EP appliance with
  every GPU participating in every layer; hard cut against PP work. This
  cycle is the continuation of the "structure, not faster kernels" diagnosis
  in `SPIKE_B_STEERING.md`.

## The Analysis Behind The Cycle (read carefully — this reframes the README)

The README's abandonment rationale ("MoE compute density is the structural
blocker") does not match the repo's own measurements:

- Sprint 581 eager attribution: EP stage = `9.419 ms`, 65.2% of a `14.445 ms`
  total. **That table is per-layer, not per-step**: ×43 layers ≈ 621 ms,
  matching the ~889 ms decode domain / ~35.9 tok/s era and the ~427 ms
  full-capture step implied by 2.34 per-req tok/s. The "per-step" label in
  SPRINT-581.md is misleading.
- The actual routed expert math inside the EP stage is ~0.2–0.3 ms/layer:
  Sprint 200 measured TurboMind MXFP4 gate/up at `0.174 ms` and down at
  `0.051 ms` (96-route compact). Low SMOCC (~0.08) on a 0.2 ms
  bandwidth-bound GEMV-like kernel is a red herring.
- Mandatory weight traffic per rank per step: 43 layers × ~17 active local
  experts (expected distinct of 32 local experts hit by 192 routes over 256)
  × 12.6 MB MXFP4 ≈ 9 GB → ~10 ms/step at ~900 GB/s HBM2. Even at 50%
  efficiency plus attention, the hardware ceiling is ~1,000+ tok/s aggregate
  at 32 slots. Measured: 26.8. The ~30–60x gap is dispatch/compose/barrier
  scaffolding (~9 ms/layer wrapped around ~0.3 ms of real work, 43×/step).
- Full graph capture only bought 1.225x because capture removes launch cost
  but replays the same serialized dependency structure: 4–6 cross-rank
  barriers per layer, per-layer host readback of route counts, dense
  contribution grids reduced over NCCL.

Supporting measurements already in the record:

- Sprint 396: NCCL small-message allreduce `4.5 GB/s` vs custom peer-doubling
  `13.4 GB/s` (2.96x) at the 32-token shape.
- Sprint 581: host-sync orchestration (route_upload/fill_pack/router_select)
  only ~5% — GPU0 skew is NOT the lever.
- Sprint 371: decode tok/s and GPU util flat from 1→32 active requests —
  fixed per-layer/per-step cost dominates.

## Recent Sprint Context

- Sprints 478–536: the "A" structural program (rank-local norms, rank-major
  consumers, host-wait removal, no-SYS NCCL transport) — done and promoted.
- Sprints 537–581: C1 graph capture program → full-capture default promoted;
  Sprint 581 is the tuning-sprint baseline (26.8 tok/s) and the gap
  attribution that names EP as the dominant cost.
- Sprints 582–596: MTP (B1) draft path — built, state-split proven, but
  deterministic acceptance 0/71; **punted 2026-05-30**. Not in scope.
- `SPIKE_B_STEERING.md` backlog relevant here: **B2** "fuse dispatch +
  grouped-GEMM + weighted-combine into 1–2 kernels with device-side offsets;
  replace the variable-size compose movement with a ring-compatible /
  statically bucketed collective (not all-pairs P2P → SHM budget, Sprint
  530)" (priority 2, Med-High). Deferred tuning levers recorded in Sprint
  581: NCCL ring/topology pinning, slots×context envelope, C4 KV spill.

## Vision Context

VISION.md exists (revision 563). North Star: TP/EP appliance at 32 slots /
256K with practical high-throughput serving. This cycle is the next entry in
the sequence after the C1/tuning sprints; it executes the steering doc's
"unifying diagnosis" (wins come from STRUCTURE: de-centralize, fuse, fewer
launches, remove host sync). Parking-lot candidates that intersect: NCCL
ring/topology pinning (deferred from 581), B2 compact EP variable-size
compose. VISION.md and SPIKE_B_STEERING.md must be updated by the merge.

## Relevant Codebase Areas

- `engine/decode_loop.cu` (~lines 900–1250) — per-layer MoE sequence: routed
  FFN → `sync_all()` (line ~918) → overlap/dense block → ep_pack
  (`ep_pack_route_dest_shards_kernel` / `ep_reduce_all_dest_shards_kernel`,
  lines ~1103–1143) → `sync_all()` (line ~1144) → NCCL ReduceScatter (lines
  ~1151–1169) or per-source broadcast (`broadcast_ep_return_slices`, lines
  ~1206–1233) → `sync_all()` (line ~1238) → final compose. `sync_all()` at
  lines ~174–192 (event barrier in graph mode, full `cudaStreamSynchronize`
  otherwise).
- `engine/router_plan.cu` — route planning; **host readback of
  `d_route_totals` / `d_route_offsets_all` at lines ~70–75** sets
  `rank.routes`, `rank.active_experts`, `rank.max_routes_per_expert` (the
  per-layer host round-trip; also what TurboMind grouped-GEMM host-side
  problem sizes consume). GPU planner kernels `gpu_route_count_all_kernel`,
  `gpu_route_fill_all_kernel` already exist (lines ~42–55).
- `engine/ep_compose.cu` — compose paths: NCCL ReduceScatter (lines ~88–118),
  compact route compose, fp16 return, `compose_next_hidden_sum8_kernel`
  fused 8-source sum (lines ~135–163).
- `engine/ep_executor.cu` — `run_gate_selected` / `run_down` TurboMind calls.
- `engine/turbomind_bindings.cu` — expert placement (rank p owns experts
  `[p*32, (p+1)*32)`), grouped-GEMM bindings (lines ~109–180).
- `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h` — grouped
  MXFP4 GEMM ABI: ragged batch via `expert_offsets`, fixed-shape probes
  (6/96/768/1536 tokens), down+route-weight+reduce epilogues (lines
  ~140–186, 253–484, 558–626). Note which entry points take host-side
  counts vs device pointers — this determines how "device-resident routing"
  must be built.
- `engine/runtime_pack.cu` (lines ~275–346) — `broadcast_ep_return_slices`
  P2P copies; `engine/runtime_resources.cu` — streams, events, NCCL init
  (`ncclCommInitAll`, line ~187), buffer allocation (`d_ep_contrib_all` =
  kGpus × slots × kHidden/kGpus floats — the dense grid).
- `engine/runtime_profiler.cu`, `engine/diagnostics_support.cu` — existing
  stage-timer plumbing the instrumentation phase should extend (the Sprint
  581 eager timers live here; they populate only in pure eager mode).
- `tools/ds4-v100-http-response-tolerance.py` — the default validation gate.
- Hardware: 8x V100-SXM2 hybrid cube mesh — each GPU has NVLink to 4 peers;
  3 of 7 peer pairs per GPU are not directly connected (the no-SYS policy
  exists because those pairs fall to SYS/SHM). Sprint 581 tracked peer-SYS
  counters; promoted path holds them at zero.

## Constraints

- **Correctness gate**: tolerance policy (≥ 0.99 selected-token AND
  generated-sequence agreement vs control at the reference shape). Bit-exact
  only if a stage explicitly justifies it.
- **Perf gate (opt-in, named here)**: decode tok/s at the reference shape
  (32 slots / 256K / 64 tok/req, deterministic, steady-state window, 128
  req), measured exactly as Sprint 581 did, full-capture default leg.
- **No-SYS topology policy**: peer-SYS counters must stay zero. Any
  peer-write transport must use only direct-NVLink pairs or explicit
  one-hop forwarding through an NVLink neighbor. No all-pairs P2P/SHM
  (Sprint 530 budget).
- **VRAM budget**: ~30.7 GiB used / 32 GiB at the reference shape (Sprint
  382); NCCL communicators already cost +848–944 MiB/GPU. New buffers must
  be small or replace existing ones (the sparse return payload is ≤ ~200 KB
  fp16 per rank per layer — capacity bounds are tiny).
- **Graph-capture compatibility**: the promoted decode is full-capture.
  Every new hot-path mechanism must be capturable (no host readbacks, no
  cudaMalloc, fixed shapes/capacities) or must explicitly justify an eager
  region. Device-side decisions replace host-side ones.
- **MXFP4 expert weights + TurboMind grouped GEMM stay** — the GEMM is not
  the problem; do not rewrite expert math except where the combine epilogue
  fuses into it.
- **MTP stays punted.** PP/layer-split stays dead (vision hard cut).
- Builds + validation on the pod per the homelab pipeline; sprint docs and
  ledger conventions per `docs/sprints/` (`STATUS.md` rollup, sprint report
  files, EXPERIMENT-STATUS.md for gates).

## Success Criteria

1. **Instrumentation phase**: a per-rank, per-stage decomposition of the
   ~9.4 ms/layer EP stage into at minimum {route-plan/host-readback, pack,
   grouped-GEMM gate/up, grouped-GEMM down, contribution pack/reduce, NCCL
   collective / peer transfer, barrier-wait, other} with absolute ms and %
   at the reference shape, reproducible from a launcher flag, archived as a
   sprint artifact. The decomposition must be self-consistent (stages sum to
   ≈ the EP stage total) and must confirm or refute the
   math-vs-scaffolding split (~5% / 95% hypothesis).
2. **Decision gate**: the decomposition picks the B2 stage order (which of
   device-resident routing / sparse peer-write a2a / barrier restructure
   lands first) based on measured, not assumed, cost.
3. **B2 staged implementation**: each stage lands behind a gate flag, passes
   the tolerance gate at the reference shape, keeps peer-SYS at zero, and
   shows a step-time reduction at the reference shape vs the immediately
   prior promoted default. Cycle-level target: EP stage ≤ ~2 ms/layer
   (from 9.4) and aggregate decode ≥ ~3x baseline (26.8 → ≥ ~80 tok/s) by
   the end of the cycle, with explicit go/no-go checkpoints between stages.
4. Steering, vision, STATUS, and README status note updated to reflect the
   reopened track and the new baseline(s).

## Verification Strategy

- **Correctness**: `tools/ds4-v100-http-response-tolerance.py` vs the
  promoted control at the reference shape; selected-token spot gates at the
  smallest shape that exercises the changed path during development.
  Existing smokes (`smokes/`) for the appliance serve path.
- **Perf**: Sprint 581's exact reference-shape methodology (de-confounded
  steady-state window, startup/warmup excluded, deterministic decode), with
  the decode-domain attribution re-run after each promoted stage. nvprof/
  nsys windows (per `gpu-profiling-guidance.md`) for kernel-level evidence;
  CUDA events for in-band stage timers; peer-SYS counters from the existing
  diagnostics.
- **Topology**: validate the one-hop forwarding schedule against the actual
  `nvidia-smi topo -m` cube-mesh adjacency on gpu-01 before any peer-write
  transport work; keep an automated assert that no transfer crosses a
  non-NVLink pair.
- **Edge cases**: zero-token experts (empty offsets), all-tokens-to-one-rank
  routing skew, slot counts below capacity (ramp-up/down windows), graph
  capture replay across steps with changing route counts (fixed-capacity
  buffers must be correct for any count ≤ capacity), fp16 return precision
  vs the ≥ 0.99 tolerance gate.

## Uncertainty Assessment

- **Correctness uncertainty: Medium** — the EP dataflow is well understood
  and reference-gated, but device-resident routing + peer-write transport
  touch the most concurrency-sensitive code in the repo (events, multi-rank
  ordering, graph capture). The tolerance gate and staged flags bound the
  risk.
- **Scope uncertainty: Medium-High** — "the cycle" spans instrumentation +
  3 structural stages; how much lands in Sprint 597 vs 598+ is an interview
  question. Per-stage go/no-go gates keep it bounded.
- **Architecture uncertainty: Medium** — the target design (device-resident
  routing, sparse a2a, pairwise events) is proven in the field (DeepEP-class
  systems) but novel on SM70 + this codebase; the TurboMind ABI's host-count
  entry points may force either a worst-case-grid GEMM launch or an ABI
  extension, which is the main design unknown.

## Open Questions

1. **Sprint packaging**: one umbrella cycle doc + N execution sprints (597 =
   instrumentation + decision, 598+ = B2 stages), or one mega-sprint with
   phases? Repo convention favors focused single-topic sprints.
2. **Instrumentation approach**: extend the existing eager stage timers to
   sub-EP granularity (cheap, in-band, but eager-only) vs nsys/nvtx capture
   of the promoted graph path (heavier, but measures the real default)?
   Probably both — which is the gate?
3. **Stage order**: hypothesis says barrier restructure + host-readback
   elimination first (pure latency, no transport risk), sparse peer-write
   a2a second, full single-kernel fusion last (and possibly unnecessary if
   the first two collapse the overhead). Does the decomposition get to
   override this ordering?
4. **TurboMind ABI**: do the existing grouped entry points accept
   device-resident `expert_offsets` without host token counts (worst-case
   grid + device early-exit), or does B2 need a new entry point / persistent
   variant? (Drafts should answer from the header, not assume.)
5. **Sparse return format**: row-indexed fp16 contributions (≤ slots×top_k
   rows × 4096) vs keeping the dense shard ReduceScatter but in fp16 with
   compact rows — what does the decomposition say the collective actually
   costs vs its launch/serialization overhead?
6. **One-hop forwarding**: static schedule computed at init from the cube
   mesh (deterministic, graph-friendly) vs reusing NCCL for the non-adjacent
   minority of pairs only? Mixing transports inside one captured graph has
   ordering implications.
7. **Where does the README/steering "reopen" note land** — does this cycle
   formally supersede the abandonment note in the README?
