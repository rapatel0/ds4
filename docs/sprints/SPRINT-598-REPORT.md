# Sprint 598 Report - B2-C: Eliminate SYS From the EP Return Transport

Date: 2026-06-11
Status: complete - C1 promoted (launcher default flipped to `nccl`)

Environment: the persistent s597 setup on pod `llm/llamacpp-build-8gpu`
(gpu-01, 8x V100-SXM2-32GB, driver 580.126.20, NCCL 2.19.3); pack
`/workspace/packs/ds4-appliance-full-tm-gated-s597`, contract
`/workspace/s597-contract/`, control `phase0-full-control/`.

## Headline

The grouped per-source NCCL broadcast EP return, captured inside the decode
graph (candidate **C1**), replaces the 56 per-pair `copy_f32_kernel` UVA
remote loads and:

| Metric | copy (s597 promoted) | nccl (C1) | gate |
|---|---:|---:|---|
| EP-return stage ms/layer | 6.92 | **0.611** (max 0.87) | <= ~1 ms PASS |
| EP window ms/layer/rank | 8.52 | **2.47** | - |
| Layer replay ms | 10.24 | **4.25** | - |
| Decode-domain tok/s (steady ref shape) | 71.21 | **162.06** | >= 1.8x PASS (**2.28x**) |
| Wall tok/s | 59.20 | **108.50** | (1.83x) |
| Tolerance vs s597 control | 1.0 (bit-exact) | **1.0 / 1.0 / logits bit-exact** | >= 0.99 PASS |
| SYS exposure | 24/56 directed pairs at ~2 ms | **zero** remote-load kernels; NVLink-ring NCCL only | no-SYS PASS |

Both controls beaten: 0.611 < 0.68 (eager NCCL control) << 6.92 (graph
copies). Stretch projection (2.3-2.6x) effectively attained at 2.28x vs the
re-measured baseline (2.20x vs the s597 73.59 anchor).

## Warm-ups

1. **Flag-off identity re-proof (closes s597 deviation #4)**: the rebuilt
   binary (transport flag default `copy`, backlog change included) at the
   reference shape: captured-graph node counts identical to the committed
   s597 binary (2697/layer + 115971 full-step); slot-indexed tolerance vs
   `phase0-full-control/`: selected-token 1.0, sequence 1.0, logits
   bit-exact. Decode-domain 71.21 / wall 59.20 tok/s = the fixed-harness
   re-statement of the s597 73.59/61.47 baseline (-3.2% run band).
2. **Harness upstreamed + backlog raised**: `appliance/http_server.cu`
   listen backlog 16 -> 256; `tools/ds4-v100-tp-ep-http-bench.sh` now carries
   the 900 s cold-load wait, `decode("utf-8","replace")`, and wave-of-32
   submission (kept for deterministic batch composition) in the repo.

## C1 capture probe (before implementation)

`tools/s598-nccl-capture-probe.cu` (new standalone): 8-rank single-process
`ncclCommInitAll`, the exact per-source grouped `ncclBroadcast` pattern of
`broadcast_ep_return_slices` at the promoted payload (3 MiB/src), captured
via the engine's origin-stream fork/join pattern. **PASS on every axis**:
eager parity, capture (200 nodes), instantiate, first replay parity,
fresh-data replay parity (graph re-reads updated buffers), 50 timed replays
at **0.7202 ms per full 8-source round**, post-timing parity. Known probe
quirk: the process hangs at exit in `ncclCommDestroy` when the captured
graph exec is still alive - probe-only teardown bug, documented, not a
transport defect.

## Implementation (flag-gated, default-off until promotion)

- `engine/runtime_options.cuh`: `Options.ep_return_nccl` from
  `DS4_V100_TP_EP_EP_RETURN_TRANSPORT=copy|nccl` (binary default `copy`).
- `engine/runtime_pack.cu`: `broadcast_ep_return_slices(...,
  skip_stream_sync)` - capture-safe mode skips the trailing host stream
  syncs (consumers are ordered by each rank's stream + the post-compose
  barrier). Default `false` keeps the eager branch and `ep_compose.cu`
  byte-identical.
- `engine/decode_loop.cu`: in the promoted graph branch, `flag=nccl` routes
  the EP return through the same broadcast primitive the eager control uses,
  captured in-graph, with s597 profiler marks (`ep_return_nccl` stage).
- `tools/ds4-v100-run-tp-ep-appliance.sh`: flag plumbing/validation;
  **default flipped to `nccl` after the gates passed; `copy` is the
  rollback flag**. The binary/Options default remains `copy`, so
  non-launcher invocations keep the prior behavior.
- No buffer changes needed: `d_ep_contrib_bcast_all` was already allocated
  unconditionally in `ensure_compose_buffers` (outside capture), and with
  the fixed-capacity route plan the broadcast path's 2D compaction branch
  never triggers (copy_elems == stride).

Appliance smoke (flag=nccl + profiler): capture succeeded (3017 nodes/layer
= 2985 + NCCL/memcpy nodes), 257 stable replays, tokens identical to the
copy transport, ep_return_nccl 0.677 ms/layer.

## Reference-shape A/B (fixed harness, 128 req x 64 tok, 4x32 batches)

Note: the harness changed this sprint (backlog + wave submission upstreamed),
so the A/B baseline is **re-stated with the same harness**: copy = 71.21
decode-domain / 59.20 wall (s597 anchor 73.59/61.47 within the run band).

| Leg | decode-domain | wall | EP-return stage | replay/layer |
|---|---:|---:|---:|---:|
| r1 copy, profiler off | 71.21 | 59.20 | (6.92, s597 table) | 10.1-10.2 ms |
| r2 nccl, profiler off | **162.06** | **108.50** | - | - |
| r3 nccl, profiler on | 160.08 (-1.2%) | 105.22 | **0.611 ms** | **4.25 ms** |

- Tolerance (slot-indexed, 128 pairs) nccl vs s597 control: selected-token
  1.0, generated-sequence 1.0, max selected-logit relative error 0.0 - the
  NCCL transport is bit-exact against the promoted copy path.
- Stage table (r3): `ep_copy_src*` stages absent; ep_window 2.47 ms
  (coverage ~101%, i.e. residual within the band; barriers now 0.30 ms
  total); remaining EP-window costs: shared_swiglu_down 0.78, ep_return_nccl
  0.61, route_plan_pack 0.53, gate_up+down 0.20.
- nsys spot-check (one 32-slot x 8-step replay window, flag=nccl):
  **0 grid-384 `copy_f32_kernel` instances** (s597 window had 19,264);
  EP return appears as `ncclDevKernel_Broadcast_RING_LL` (5.63 ms/layer-step
  summed across 8 ranks = 0.70 ms/rank, consistent with the stage table) on
  NVLink rings (`NCCL_P2P_LEVEL=NVL`; P2P over SYS disabled by policy).
  Total `copy_f32` busy time fell 18.4 s -> 2.0 s per window.

## Definition of Done

1. Warm-ups - **done** (identity re-proof recorded; harness upstreamed;
   backlog 256).
2. Gated transport alternative exists, default unchanged until promotion -
   **done** (flag default `copy` through all gating runs).
3. Tolerance >= 0.99 on the winner - **done** (1.0 / 1.0, bit-exact logits).
4. EP-return <= ~1 ms/layer + no SYS-class pair times + nsys spot-check -
   **done** (0.611 ms; per-pair copy stages gone; 0 remote-load kernels).
5. Decode-domain >= 1.8x 73.59 - **done**: 162.06 = **2.20x the s597
   anchor / 2.28x the re-measured same-harness baseline**; stretch 2.3-2.6x
   essentially reached at the lower bound.
6. Beats both controls; promoted - **done**: 0.611 < 0.68 < 6.92; launcher
   default flipped to `nccl`, `copy` kept as rollback; copy path not
   deleted.
7. Report written; STATUS/steering/VISION updates - **orchestrator's
   (per instruction)**; commits - orchestrator's.

## Deviations / incidents (honest list)

- `engine/runtime_profiler.cu` received a one-stage addition
  (`ep_return_nccl` name) outside the listed edit surface - measurement-only,
  flag-gated, required for the stage table to name the new transport.
- The C1 "capture probe before implementation" was a standalone tool (as
  ordered); note the s597 nsys capture already showed NCCL collectives
  captured in the promoted graph, which pre-validated feasibility.
- Probe teardown hang in `ncclCommDestroy` (graph exec still alive) -
  documented above; kill-after-PASS.
- One transient foreign GPU burst (host pid, 8.1 GiB on all 8 GPUs, not a
  k8s GPU pod; gemma pod is on gpu-02) crashed the first flag=nccl smoke
  mid-startup; it self-cleared and never overlapped a measured window
  (verified: only our pid on the GPUs during R1-R3). One R1 launch failed
  its own reserve check against our dying smoke server (kill+3 s race);
  every subsequent run uses a wait-for-idle preflight.
- The naive index-paired tolerance comparison remains slot-assignment-noisy
  (s597 finding); the slot-indexed comparison is the valid identity check
  and is what all numbers above use.

## Follow-ups material (for 599 planning)

- New decode profile (post-C1, per rank per layer-step): shared_swiglu_down
  0.78 ms, ep_return_nccl 0.61, route_plan_pack 0.53, barriers 0.30,
  GEMMs 0.20 - the EP window is now 2.47 ms of a 4.25 ms layer replay; the
  pre-EP prefix (hc_current 5.55 ms eager-measured + attention) is now the
  largest target, as predicted (Sprint 598 risk table: HC-current caps the
  realized speedup; observed exactly that - 2.28x not 2.6x).
- B2-D (per-pair event deps replacing the 8x8 barriers) re-bounds at
  ~0.30 ms/layer post-C1 - smaller than before; re-rank against the pre-EP
  prefix work in 599.
- The eager (non-graph) EP-return path still host-syncs after broadcasts;
  unchanged by this sprint.
- C2 (one-hop NVLink relay forwarding) not needed; relay table remains in
  `sprint597-phase01/phase1-peer-copy-analysis.txt` if NCCL ever regresses.

## Artifacts

- Pod `/workspace/s598-artifacts/`: COMMANDS.md (reproducible log),
  probe-debug.log + s598-nccl-capture-probe (binary), c1-smoke.log,
  r1-copy-baseline/ r2-nccl/ r3-nccl-prof/ (bench trees),
  nsys-insitu.nsys-rep/.sqlite + nsys-run.log, s598-nsys-insitu.sh.
- Laptop `logs/from-cluster/sprint598/`: COMMANDS.md, capture-probe-result,
  r1/r2/r3 sustained_http TSVs, r3-nccl-stage-table.txt,
  nsys-no-sys-proof.txt (no .nsys-rep in the repo).
- Source changes (uncommitted, orchestrator review): appliance/http_server.cu,
  tools/ds4-v100-tp-ep-http-bench.sh, engine/runtime_options.cuh,
  engine/runtime_pack.cu, engine/decode_loop.cu, engine/runtime_profiler.cu,
  tools/ds4-v100-run-tp-ep-appliance.sh, new tools/s598-nccl-capture-probe.cu,
  docs/sprints/SPRINT-598-REPORT.md.
