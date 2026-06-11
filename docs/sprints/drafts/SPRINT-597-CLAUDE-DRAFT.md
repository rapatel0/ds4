# Sprint 597 — EP-Overhead Decomposition + B2 Decision Gate (Claude draft)

> Independent draft against `docs/sprints/drafts/SPRINT-597-INTENT.md`.
> Grounded in `engine/decode_loop.cu`, `engine/router_plan.cu`,
> `engine/ep_executor.cu`, `engine/ep_compose.cu`,
> `engine/turbomind_bindings.cu`, `engine/runtime_pack.cu`,
> `engine/runtime_resources.cu`, `engine/output_head.cu`, and
> `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`.

## Overview

This sprint **reopens the TP/EP throughput program on the B2 track** and is the
first sprint of the EP-overhead elimination cycle. It does **one** thing:
produce a per-rank, per-stage decomposition of the ~9.4 ms/layer EP stage on
the **promoted full-capture default** at the reference shape, and use that
measurement to pick the B2 stage order. It does **not** ship a structural B2
kernel; that is sprints 598+.

The framing the intent makes — that the EP stage is ~95% scaffolding wrapped
around ~0.3 ms of real expert math — is plausible and matches the recorded
attribution (Sprint 581: EP = 65.2% of decode), but it is currently an
*inference*, not a measurement. The repo's own timers cannot confirm it,
because the only fine-grained timers in `decode_loop.cu` are host wall-clock
`std::chrono` spans (`ep_ms`, `dense_ms`, `compose_ms`, `hc_current_input_ms`)
that are meaningful only in eager mode, where `sync_all()` is a real
`cudaStreamSynchronize` (`decode_loop.cu:182-191`). In the promoted full-capture
path `sync_all()` is an event barrier (`decode_loop.cu:174-181` →
`enqueue_cross_gpu_stream_barrier`, `output_head.cu:1726`), so those host spans
do not bracket GPU work at all. **We are flying on an eager-mode proxy for a
graph-mode default.** Closing that gap is the whole point of the instrumentation
sprint.

A second, code-grounded reason to measure before building: a large part of what
the intent calls "device-resident routing (no per-layer host readback of route
counts)" **is already true on the promoted path.** The promoted decode binds the
host-count-free TurboMind entry points and feeds them a *fixed* route capacity,
not a read-back count (see Architecture → ABI, and Open Question 4). So the B2
work is narrower and more specific than the intent's seed implies, and the
decomposition has to tell us *which* of the remaining costs — padded-grid GEMM,
serialized NCCL broadcast a2a, or the all-pairs barrier — actually dominates.

Per `VALIDATION_CONTROL_POLICY.md` this cycle **opts into perf measurement**:
its failure mode is exactly "structurally landed but perf didn't transfer to
serving." Sprint 597 itself is measurement-only and produces no promoted code
change, so its correctness risk is near zero; the perf-gate machinery it builds
is what the later stages depend on.

## Use Cases

- **As the cycle owner**, I need a reproducible, launcher-flag-driven
  decomposition of the EP stage into named sub-stages with absolute ms and %,
  on the *actual promoted full-capture graph path*, so the B2 stage order is
  chosen on measured cost rather than the intent's hypothesis.
- **As the cycle owner**, I need the decomposition to be self-consistent
  (sub-stages sum to ≈ the EP-stage total) and to explicitly confirm or refute
  the ~5%/95% math-vs-scaffolding split, so we know whether B2 is worth three
  sprints.
- **As a reviewer of 598+**, I need a single archived artifact (the
  decomposition table + the raw nsys/event captures) and a written decision
  ("stage X lands first because it is N ms of the 9.4") that the implementation
  sprints cite as their baseline.
- **As the release owner**, I need the reopen recorded in the durable docs
  (README status note, `VISION.md`, `SPIKE_B_STEERING.md`, `STATUS.md`) so the
  repo no longer reads as abandoned and the B2 track has a written entry point.
- **As an operator**, I need the new instrumentation to be default-off and to
  leave the promoted serving binary byte-for-byte unchanged when the flag is
  unset.

## Architecture

### Where the EP stage lives and what it is made of

The per-layer routed-MoE sequence in `run_one_step` (`decode_loop.cu`,
~890–1300) is, in execution order on the promoted compact full-capture path:

1. **Routed FFN (the real math).** Per rank, `run_gate_selected` then `run_down`
   (`decode_loop.cu:913-916` → `ep_executor.cu`). These call the TurboMind
   grouped GEMM via `api.mmgs_clamped`/`api.mmgt` with
   `executor_rows = routed_executor_rows(rank, opt) = rank.routes`
   (`ep_executor.cu:31-57`, `turbomind_bindings.cu:372-375`).
2. **`sync_all()`** (event barrier in graph mode) + the shared/dense F8 block,
   optionally overlapped (`decode_loop.cu:924-1066`).
3. **Contribution pack** — `ep_pack_route_dest_shards_kernel` (compact) or
   `zero_f32_kernel` + `ep_reduce_all_dest_shards_kernel` (dense)
   (`decode_loop.cu:1103-1143`), writing the dense contribution grid
   `r.d_ep_contrib_all` (= `kGpus × slots × (kHidden/kGpus)` floats,
   `runtime_resources.cu:442`).
4. **`sync_all()`** (`decode_loop.cu:1144`).
5. **All-to-all transport** — either `ncclReduceScatter`
   (`decode_loop.cu:1151-1169`, non-compact only) or, on the promoted compact
   path, `broadcast_ep_return_slices` (`runtime_pack.cu:275-346`): a 2D
   pre-pack of active rows + **eight sequential per-source NCCL broadcasts** +
   per-dst device extract. This is the no-SYS transport (Sprint 479/531).
6. **`sync_all()`** (`decode_loop.cu:1238`).
7. **Compose** — `compose_next_hidden_compact8_multi_kernel`
   (`decode_loop.cu:1278-1300`) folds the eight per-source slices +
   attention/shared dense outputs + carried current into `d_next_hidden`.

The intent's "~9.4 ms/layer" is the wall time around 1–7 on the eager path. The
job is to attribute it across {route-plan/host-readback, gate/up GEMM, down
GEMM, pack/reduce, NCCL collective/peer transfer, barrier-wait, compose, other}
on the **graph** path.

### The barrier structure that serializes everything

`sync_all()` in graph mode calls `enqueue_cross_gpu_stream_barrier`
(`output_head.cu:1726-1779`). That is a **full 8×8 barrier**: every rank records
a done-event on its compute and dense streams, then *every* destination stream
waits on *every* source's events. Three of these fire per routed layer
(steps 2/4/6 above) × 43 layers. This is the "global `sync_all()`" the intent
targets for replacement with per-pair event dependencies — and it is exactly the
structure full graph capture preserves (the intent's "capture removes launch
cost but replays the same serialized dependency structure"). A destination only
truly depends on the seven sources it consumes contribution slices from; on the
cube mesh, after one-hop forwarding it depends only on its direct NVLink
neighbors. The barrier over-synchronizes; the decomposition must size how much
of the 9.4 ms is wait-on-this-barrier vs real transfer.

### TurboMind grouped-GEMM ABI (grounds Open Question 4)

The promoted path binds (`turbomind_bindings.cu:52-55`):

| wrapper | ABI symbol |
|---|---|
| `api.mmgt` | `ggml_turbomind_mul_mat_grouped_total_tokens` |
| `api.mmgs` | `ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens` |
| `api.mmgs_clamped` | `ggml_turbomind_mul_mat_grouped_gated_silu_clamped_total_tokens` |

Reading the header (`ggml-turbomind-api.h:173-251`):

- `expert_offsets` is **always a device `const int*`** — token→expert tile
  boundaries are consumed device-side. Routing offsets are already device
  resident in every variant.
- The host-side input is the scalar `total_tokens` (and `N/K/group_size`). The
  `_total_tokens` family was added precisely to **avoid the base
  `ggml_turbomind_mul_mat_grouped`'s internal synchronous D2H read of
  `expert_offsets[num_experts]`** (header lines 188-192). So the promoted path
  has **no per-GEMM device→host sync** for the count.
- On the promoted full-capture path, `total_tokens = executor_rows =
  rank.routes`, and `rank.routes` is set to the **fixed worst-case capacity**
  `route_capacity = slots*top_k = 192` in the post-attention fixed-capacity
  route plan (`router_plan.cu:198`, `r.routes = r.route_capacity`), with **no
  host readback**. The per-layer host readback the intent cites
  (`router_plan.cu:63-89`, `cudaMemcpy(d_route_totals/d_route_offsets_all)` →
  `rank.routes`) lives in `upload_model_router_route_plan_gpu`, which is the
  **eager** route-plan path, not the promoted full-capture default.

**Consequence for Q4:** B2 does **not** need a new TurboMind entry point to get
device-resident routing — it already has it, at the cost of a *worst-case grid*.
The promoted GEMM tiles over 192 rows/rank every step even though Sprint 542
logged p50 max-rank route pressure of 64 and max of 132 — roughly a **3× tile
overcount**. The two ways to recover that, in increasing ABI cost:

1. **No ABI change (recommended first):** device-masked / route-blocked
   executor that keeps the graph-visible `total_tokens = 192` but makes the
   kernel early-exit on inactive padded rows. The fixed-shape reduce probes
   already accept a device `expert_offsets` and the down-reduce epilogue already
   fuses route-weight apply + atomic F32 accumulate
   (`ggml_turbomind_ds4_mxfp4_down_*_reduce`, header lines 557-626) — i.e. the
   "combine epilogue fuses into the GEMM" the intent wants partly exists. Sprint
   550 already route-blocked the compact *pack*; this extends the idea to the
   GEMM grid.
2. **ABI extension (only if the decomposition says padded-grid GEMM dominates):**
   a `_device_total_tokens` variant that reads the count from a device scalar to
   shrink the launched grid below capacity. This is the "new entry point /
   persistent variant" the intent flags as the main design unknown, and it
   should not be built speculatively.

### Sparse return format (grounds Open Question 5)

The compact promoted compose already returns **active rows only**: Sprint 531
made `broadcast_ep_return_slices` copy `routed_compose_rows(src) =
rank.routes` rows per source (`decode_loop.cu:1184-1186`,
`ep_compose.cu:41-45`), and Sprint 550 route-blocked the pack. The remaining
cost is **structure, not bytes**: the dense contribution grid
`d_ep_contrib_all` is materialized full
(`kGpus × slots × kHidden/kGpus` floats) and the transport is **eight
serialized NCCL broadcasts**, one per source rank, each a one-to-all over a
payload ≤ `slots*top_k × kHidden/kGpus` fp32 (≤ ~200 KB/rank). A row-indexed
fp16 sparse a2a (slots×top_k rows × kHidden/kGpus, with a device row→slot index)
is the target format, but **whether it wins is exactly what the decomposition
measures**: if the eight broadcasts are launch/serialization-bound (likely at
this tiny payload — Sprint 396 showed NCCL small-message allreduce at 4.5 GB/s
vs 13.4 GB/s for custom peer-doubling), the win is in replacing the *collective
shape*, not the *payload encoding*.

### Instrumentation design (grounds Open Question 2)

Two complementary mechanisms, with a clear gate:

- **Primary / gate: nsys + NVTX ranges on the promoted full-capture graph.**
  Wrap each EP sub-stage (route-plan, gate/up, down, pack, transport, barrier,
  compose) in an NVTX range and, where the work is inside the captured region,
  in paired `cudaEventRecord` on the rank-0 compute stream so the timing is
  read from device events, not host spans. This measures the **real default**
  and is the artifact the decision cites. Driven by a new launcher flag
  `DS4_V100_TP_EP_EP_STAGE_PROFILE=1` (default off; emits a TSV).
- **Secondary / cheap cross-check: extend the existing eager `std::chrono`
  sub-timers.** `decode_loop.cu` already splits `t_pre/t0/t1/t2/t_reduce_done/
  t_copy_done`; add route-plan, gate/up-vs-down, and pack-vs-reduce splits.
  These are in-band and free but **eager-only** and run a *different* executor
  shape (actual ~64 routes vs full-capture padded 192). That disagreement is
  itself a deliverable: it quantifies the worst-case-grid tax.

The gate is the nsys/NVTX capture because the policy's whole rationale for the
opt-in is "the perf didn't transfer to serving" — only the graph-path number is
serving-representative. The eager timers exist to catch gross
inconsistency and to bound the GEMM-padding delta.

## Implementation

Sprint 597 is **measurement + decision + docs**. No promoted kernel changes.

### Phase 0 — Topology ground truth (prereq for 599)
- Capture `nvidia-smi topo -m` on gpu-01 into the sprint artifact dir; derive
  the per-GPU NVLink adjacency and the 3-of-7 non-adjacent peer set. This is the
  input the static one-hop forwarding schedule (Sprint 599) needs and the assert
  that no transfer crosses a non-NVLink pair. Read-only; no code.

### Phase 1 — EP sub-stage instrumentation (the deliverable)
- `engine/diagnostics_support.cu` / `engine/runtime_profiler.cu`: add an
  EP-stage NVTX+event profiler keyed on a new `Options` field
  (`ep_stage_profile_gate`) wired from `DS4_V100_TP_EP_EP_STAGE_PROFILE`.
- `engine/decode_loop.cu`: insert NVTX ranges + `cudaEventRecord` pairs at the
  seven sub-stage boundaries (after routed_ffn, after pack `ep_pack` label
  ~1145, after `ep_copy` label ~1237, after compose), reusing the existing
  `log_rank_stage`/`sync_after_decode_stage` hook points so the captured graph
  topology is unchanged when the flag is off.
- Extend the eager `std::chrono` splits (route-plan, gate/up vs down,
  pack vs reduce, broadcast vs extract) behind the same gate.
- Emit a per-rank TSV: `layer, rank, stage, ms_event, ms_host, pct`.

### Phase 2 — Reference-shape capture (perf, opt-in)
- Run the reference shape (32 slots / 256K / 64 tok/req, deterministic,
  steady-state window, 128 req) on the pod, promoted full-capture default leg,
  with the profiler on, exactly per Sprint 581 methodology.
- Run the matching eager leg for the cross-check.
- Run nsys per `gpu-profiling-guidance.md` for one decode window for
  kernel-level corroboration of the event timings.
- Archive raw captures + the assembled decomposition table to
  `/localpool/ds4/workspace/<run-id>` and reference it in `SPRINT-597.md`.

### Phase 3 — Decision gate (the decision artifact)
- Assemble the decomposition; verify sub-stages sum to ≈ the EP-stage total
  (self-consistency check).
- Confirm or refute the ~5%/95% split with numbers.
- Write the **B2 stage order** decision into `SPRINT-597.md` and
  `SPIKE_B_STEERING.md`, choosing the lead stage by measured ms. Hypothesis to
  test, not assume (Q3): barrier restructure + device-masked executor first
  (pure latency, no transport risk; and the host-readback the intent wanted
  removed is already gone on full capture), sparse peer-write a2a second, full
  single-kernel fusion last (and possibly unnecessary).

### Phase 4 — Reopen the track in the durable docs
- README: replace/annotate the "abandoned / research archive" note with a
  reopen pointer to this cycle (Q7: yes, this supersedes it).
- `VISION.md`: bump revision; add the B2 cycle as the next sequence entry after
  C1/tuning; keep the PP hard cut and the MTP punt intact.
- `SPIKE_B_STEERING.md`: move B2 from backlog to active; record the 26.8 tok/s
  baseline and the decomposition pointer.
- `STATUS.md`: add the reopen + the new baseline rollup.

### Proposed cycle packaging (grounds Open Question 1)
Repo convention favors focused single-topic sprints, so:
- **597** = instrumentation + decision gate + reopen (this doc).
- **598** = B2 stage 1: barrier restructure (per-pair events replacing the 8×8
  `enqueue_cross_gpu_stream_barrier`) and/or device-masked executor — whichever
  the decomposition ranks first. Opts into perf.
- **599** = B2 stage 2: sparse fp16 peer-write a2a with the static one-hop
  NVLink forwarding schedule (uses Phase 0 topology). Opts into perf.
- **600** = B2 stage 3: fused dispatch→grouped-GEMM→weighted-combine, only if
  598/599 leave material overhead. Opts into perf.
Each later stage lands behind a gate flag, passes the tolerance gate, keeps
peer-SYS at zero, and shows a step-time reduction vs the prior promoted default,
with an explicit go/no-go between stages.

## Files Summary

| File | Change |
|---|---|
| `engine/runtime_types.cuh` | add `ep_stage_profile_gate` to `Options`; profiler event/handle fields on `RankState` if not reusable |
| `engine/runtime_profiler.cu` / `engine/diagnostics_support.cu` | EP sub-stage NVTX+event profiler; TSV emitter |
| `engine/decode_loop.cu` | NVTX+event markers at the 7 EP sub-stage boundaries (flag-gated, no-op when off); extended eager `std::chrono` splits |
| `tools/ds4-v100-run-tp-ep-appliance.sh` | plumb `DS4_V100_TP_EP_EP_STAGE_PROFILE` |
| `docs/sprints/SPRINT-597.md` | decomposition table, decision, artifact pointers |
| `docs/sprints/drafts/SPRINT-597-*` | this draft + merge notes |
| `README.md` | reopen note superseding the abandonment line |
| `docs/sprints/VISION.md` | revision bump; B2 cycle entry |
| `SPIKE_B_STEERING.md` | B2 → active; baseline + decomposition pointer |
| `docs/sprints/STATUS.md` | reopen + baseline rollup |
| `docs/sprints/EXPERIMENT-STATUS.md` | perf-gate entry for the cycle |

No promoted hot-path kernel, no buffer-shape, and no transport change in 597.

## Definition of Done

1. **Decomposition exists and is reproducible.** A single launcher flag
   (`DS4_V100_TP_EP_EP_STAGE_PROFILE=1`) reproduces a per-rank, per-stage table
   on the promoted full-capture default at the reference shape, archived as a
   sprint artifact with the raw nsys/event captures.
2. **Self-consistent.** Named sub-stages
   {route-plan, gate/up GEMM, down GEMM, pack/reduce, transport, barrier-wait,
   compose, other} sum to within a stated tolerance of the EP-stage total; the
   table reports absolute ms and %.
3. **Split adjudicated.** The ~5%/95% math-vs-scaffolding hypothesis is
   explicitly confirmed or refuted with numbers, including the
   eager-vs-full-capture GEMM-padding delta.
4. **Decision recorded.** `SPRINT-597.md` + `SPIKE_B_STEERING.md` name the B2
   stage order chosen by measured cost, with a one-line justification per stage.
5. **Default unchanged.** With the flag unset, the promoted binary's captured
   graph topology, peer-SYS counters (0/0), and selected-token output are
   identical to the prior promoted control. Verified by the tolerance gate
   (`tools/ds4-v100-http-response-tolerance.py`, ≥0.99 selected-token AND
   generated-sequence vs the Sprint 580/581 promoted control artifact — reused,
   not freshly produced) and `peer_copy_sys_ops/bytes = 0`.
6. **Topology ground truth archived.** `nvidia-smi topo -m` adjacency for gpu-01
   captured for the 599 forwarding schedule.
7. **Track reopened.** README, `VISION.md`, `SPIKE_B_STEERING.md`, `STATUS.md`
   updated; MTP punt and PP hard cut left intact.

## Risks

- **The instrumentation perturbs the graph it measures.** NVTX is cheap, but
  inserting `cudaEventRecord` into the captured region can change graph
  structure/timing. *Mitigation:* events recorded only when the flag is on; the
  default-off path is byte-identical (DoD 5); treat absolute ms as
  flag-on-relative and use nsys (which needs no in-graph events) as the
  arbiter for the default path's real timing.
- **Eager and graph legs measure different executor shapes.** Eager runs actual
  routes (~64), full-capture runs padded 192. *Mitigation:* this is expected and
  documented as the padding-tax finding; the gate is the graph-leg nsys/event
  number, not the eager span.
- **Self-consistency may not close** (sub-stages don't sum to the total) if there
  is hidden overlap between streams. *Mitigation:* report the residual as
  "other/overlap" and bound it; do not force-fit.
- **Decomposition could refute the thesis** (e.g. the padded-grid GEMM, not the
  transport, dominates). *Mitigation:* that is a *success* of the sprint — the
  decision gate then points B2 at the executor mask first; the cycle target
  (EP ≤ ~2 ms/layer, agg ≥ ~80 tok/s) is owned by 598+, not 597.
- **Reopening the README** is an outward-facing repo-status change. *Mitigation:*
  it is a status note, not a behavior change; phrase as "B2 throughput track
  reopened; MTP remains punted."

## Security

No new external surface. No network, auth, file-format, or input-parsing change.
The profiler reads device buffers and writes a local TSV under the run
workspace; no model weights, prompts, or response content are added to logs
beyond the existing stage-stat tags. Builds and runs occur on the trusted pod
(gpu-01) via the homelab pipeline; no secrets touched. VRAM budget unaffected
(no new persistent device buffers on the promoted path; the worst-case is a few
reused event handles).

## Dependencies

- **Pod (gpu-01, 8× V100-SXM2-32GB)** for builds and the reference-shape
  serving run — required by `VALIDATION_CONTROL_POLICY.md` and the homelab
  pipeline; the laptop cannot run this.
- **Promoted control artifact** from Sprint 580/581 (full-capture default,
  26.8 tok/s) reused as the tolerance-gate control — no fresh control run
  (none of the five invalidators apply).
- **Existing stage-timer plumbing** (`runtime_profiler.cu`,
  `diagnostics_support.cu`) and the `log_rank_stage`/`sync_after_decode_stage`
  hooks in `decode_loop.cu`.
- **nsys / NVTX** availability on the pod per `gpu-profiling-guidance.md`.
- **`tools/ds4-v100-http-response-tolerance.py`** for the correctness gate.
- No dependency on 598+; this sprint feeds them.

## Open Questions (answered)

**Q1 — Sprint packaging.** One umbrella cycle realized as **focused
single-topic sprints**, per repo convention: 597 = instrumentation + decision +
reopen; 598 = barrier/executor stage; 599 = sparse peer-write a2a + one-hop
schedule; 600 = optional full fusion. Not a mega-sprint — the per-stage go/no-go
gates are the scope control the intent's Medium-High scope uncertainty needs.

**Q2 — Instrumentation approach.** Both, with the **nsys/NVTX capture of the
promoted graph path as the gate** and the extended eager `std::chrono`
sub-timers as a cheap cross-check. Rationale is code-grounded: the existing
`ep_ms`/`dense_ms`/`compose_ms` host spans bracket `sync_all()`, which is a
real stream-sync only in eager mode (`decode_loop.cu:182-191`); in full capture
`sync_all()` is the `enqueue_cross_gpu_stream_barrier` event barrier, so host
spans don't measure GPU work. Only the graph-leg device-event/nsys number is
serving-representative, which is exactly the policy's reason for the perf opt-in.

**Q3 — Stage order / can the decomposition override the hypothesis?** **Yes —
the decomposition picks the order; the hypothesis is the prior, not the
decision.** Grounded nuance: the intent's "host-readback elimination" is
*already done* on the promoted full-capture path (`router_plan.cu:198` sets
`rank.routes = route_capacity` with no readback; the readback at
`router_plan.cu:63-89` is the eager path). So stage 1 is realistically
**barrier restructure** (replace the 8×8 `enqueue_cross_gpu_stream_barrier`,
`output_head.cu:1726-1779`, with per-pair event deps) **and/or the device-masked
executor** (recover the ~3× worst-case-grid tax), pure-latency and
transport-risk-free. Sparse a2a is stage 2; fusion is stage 3 and may be
unnecessary. Final order is whatever the measured ms rank.

**Q4 — TurboMind ABI: device-resident offsets without host counts?** **The
existing entry points already accept device-resident `expert_offsets` and need
no host count read.** The promoted path binds the `_total_tokens` variants
(`turbomind_bindings.cu:52-55`), whose entire purpose is to avoid the base
`ggml_turbomind_mul_mat_grouped`'s internal synchronous D2H read of
`expert_offsets[num_experts]` (header lines 188-192). `expert_offsets` is a
device `const int*` in every variant; the only host input is the scalar
`total_tokens`, which on full capture is the **fixed capacity 192**
(`rank.routes = route_capacity`), not a per-step readback. **So B2 needs no new
ABI entry point for worst-case-grid routing — it already runs that way.** The
cost is the ~3× padded grid (192 launched vs p50 64 active). Recover it first
with a **device-masked / route-blocked executor (no ABI change)** — the
fixed-shape down-reduce probes already fuse route-weight + atomic F32 accumulate
in the epilogue (header lines 557-626), so the combine-into-GEMM primitive
exists. Only if the decomposition shows the padded GEMM itself dominates is a
new `_device_total_tokens` (device-scalar count → smaller grid) entry point
justified — that is the real ABI unknown, and it is deferred until measured.

**Q5 — Sparse return format.** Row-indexed **fp16** contributions
(≤ `slots*top_k` rows × `kHidden/kGpus`, with a device row→slot index) is the
target, but the decomposition decides whether the win is the *encoding* or the
*collective shape*. Grounded: the compact promoted compose already transfers
**active rows only** (Sprint 531/550; `routed_compose_rows = rank.routes`,
`decode_loop.cu:1184-1186`), so bytes are already small (≤ ~200 KB/rank). The
serialization is the suspect: `broadcast_ep_return_slices`
(`runtime_pack.cu:275-346`) is **eight sequential NCCL broadcasts** + a dense
`d_ep_contrib_all` grid materialization. At this payload, NCCL small-message
throughput (Sprint 396: 4.5 GB/s) means the cost is almost certainly
launch/serialization, so 599 should replace the *collective shape* (sparse
peer-write a2a) and the fp16 encoding is a secondary halving — the
decomposition's transport-vs-barrier split confirms which.

**Q6 — One-hop forwarding: static schedule vs reuse NCCL for the minority.**
**Static schedule computed at init from the cube mesh, for the direct-NVLink
majority; reuse NCCL only for the 3-of-7 non-adjacent pairs that would otherwise
fall to SYS/SHM.** This is graph-friendly (fixed structure, deterministic) and
respects the no-SYS budget (Sprint 530). Mixing two transports inside one
captured graph has real ordering implications, which is *why* 598's per-pair
event dependencies land first: they give 599 the per-pair handles to order a
mixed-transport step correctly. The schedule must be validated against the
Phase-0 `nvidia-smi topo -m` adjacency with an automated assert that no transfer
crosses a non-NVLink pair (peer-SYS counters stay 0).

**Q7 — Does this cycle supersede the README abandonment note?** **Yes.** Sprint
597 Phase 4 replaces the "abandoned / research archive" line with a reopen
pointer to the B2 throughput cycle, updates `VISION.md`/`SPIKE_B_STEERING.md`/
`STATUS.md`, and explicitly preserves the MTP punt (2026-05-30) and the PP hard
cut. The reopen is scoped to the TP/EP B2 track only.
