# Sprint 597 Report - EP-Overhead Decomposition, Transport Ground Truth, and B2 Decision Gate

Date: 2026-06-11
Status: phases 0-4 complete (Phase 5 docs rollup handled separately)

Environment: rebuilt from scratch on pod `llm/llamacpp-build-8gpu` (gpu-01,
8x V100-SXM2-32GB, driver 580.126.20). The May-31 workspace wipe destroyed the
s181 pack and all prior artifacts; pack and contract were regenerated from
`/models/DSv4-Flash-256e-fixed.gguf` with the s181 conventions
(`--fuse-gate-up-interleaved`) into
`/workspace/packs/ds4-appliance-full-tm-gated-s597` +
`/workspace/s597-contract/tp-ep-pack-contract.tsv`.

## Phase 0 - Anchor reproduction (re-anchored)

Reference shape: 32 slots / 256K / 64 tok/req, deterministic, 128 requests in
4 coalesced batches of 32, steady-state = batches 2-4 (batch 1 carries
capture). Harness note: requests are submitted in waves of 32 (the server's
listen backlog is 16; the old 128-at-once pattern cannot complete here and
its SYN-retry stagger depressed the s581 wall numbers).

| Leg | Agg decode-domain tok/s (steady) | Wall tok/s | per-layer-step |
|---|---:|---:|---:|
| full (promoted default) | **73.59** | 61.47 | 10.11 ms replay |
| eager | 40.97 | 37.81 | 18.17 ms |

vs the Sprint 581 anchors:

- Per-request decode 2.30 tok/s vs anchor 2.344 (**-1.8%** - the decode
  domain reproduces almost exactly).
- The "26.8 tok/s aggregate" anchor is wall-window based and is **re-anchored
  to 59-61 tok/s wall / 73.6 tok/s decode-domain**; the delta is the harness
  connection pattern, not engine behavior.
- Eager attribution (per layer-step, steady): EP 11.14 ms (61.3%),
  attention 3.40, compose 0.89, HC-current 5.55, final_hc 0.53, host-sync
  0.86; total 18.17 ms. vs anchors: EP +18.2%, total +25.8% -> re-anchored.
  The growth concentrates in HC-current (1.10 -> 5.55, post-MTP churn) and
  attention; EP remains dominant.

New tolerance-gate control artifact:
`/workspace/s597-phase01-artifacts/phase0-full-control/` (response-N.txt x128).

## Phase 1 - Topology + transport ground truth

- `nvidia-smi topo -m` archived. Hybrid cube mesh confirmed: each GPU has 4
  NVLink peers (2xNV1, 2xNV2) and 3 SYS peers; **12/28 undirected (24/56
  directed) pairs are SYS**: (0,5)(0,6)(0,7)(1,4)(1,6)(1,7)(2,4)(2,5)(2,7)
  (3,4)(3,5)(3,6). Every SYS pair has exactly two one-hop NVLink relays
  (full relay table in `phase1-peer-copy-analysis.txt`).
- Microbench (`tools/s597-peer-copy-microbench.cu`, new standalone tool;
  copy_f32_kernel-style UVA remote loads, all 64 pairs, 8K-512K): at the
  promoted 384 KiB payload (192 routes x 512 f32): self ~140 GB/s, NV2
  ~29 GB/s (13.5 us), NV1 ~17.8 GB/s (22 us), SYS ~8.3 GB/s (46-49 us).
- In-situ nsys (one 8-step replay window, unmodified promoted default;
  19,264 EP-return kernels = 56 pairs x 43 layers x 8 steps): NV2 11.1 us
  and NV1 19.7 us match the microbench; **SYS pairs average 1,990 us
  (686-3,094 us)** - ~40x the isolated value, because 24 concurrent SYS
  loads saturate PCIe/QPI. The microbench class ranking holds in situ; the
  SYS magnitude is congestion-dominated.
- **Finding: the promoted EP return crosses SYS on 24/56 directed copies and
  those copies are the single largest cost in the decode step** (~5.5-6.5
  ms/layer per rank of the 10.1-10.2 ms layer replay). `peer_copy_sys_bytes=0`
  proves nothing - `record_peer_copy()` is not wired into this path.

## Phase 2 - EP sub-stage instrumentation

`DS4_V100_TP_EP_EP_STAGE_PROFILE` (default off; launcher-plumbed env ->
`Options.ep_stage_profile`). Changes confined to: `engine/decode_loop.cu`
(stage marks, `sync_all_prof` barrier-site wrappers, collect call sites),
`engine/runtime_options.cuh` (option), `engine/runtime_profiler.cu` (stage
enum/names, marker + collector + TSV emitter), launcher.

Implementation deviation (load-bearing): CUDA rejects `cudaEventElapsedTime`
on events recorded inside a captured graph (`cudaErrorInvalidValue`,
verified). Graph-mode timing therefore uses the graph-compatible equivalent
of paired event records: **paired 1-thread `%globaltimer` stamp kernels**
writing into pre-allocated per-rank device slots (allocated once at decode
entry; no `cudaMalloc` and no D2H inside any captured region; slots are read
back only after the post-replay sync). Eager mode uses real
`cudaEventRecord` pairs. Stage set: route_plan_pack (route-plan kernels +
routed-input pack run inside `post_attention_ffn.cu` and are profiled as one
combined stage at the decode_loop boundary; nsys splits them by kernel name),
gate/up GEMM, down GEMM, dense-overlap, shared swiglu+down, contrib pack,
EP-return copy per (dst,src), compose, barrier-wait per sync_all site
(954/978/996/1006/1045/1062/1144/1170/1373), plus a synthetic `ep_window`
(route_plan begin -> barrier_1373 end) for closure. NVTX ranges emitted on
rank 0 when the flag is on. TSV: `tp_ep_ep_stage_profile` with layer, rank,
stage, ms_event, rows (actual per-rank routes read from `d_route_totals`),
bytes, pct; `tp_ep_ep_stage_routes` gives the per-step route-skew vector.

Non-perturbation verification (reference shape):

| Check | Result |
|---|---|
| Flag-off tok/s (instrumented binary) | 71.96 decode-domain vs 71.39 (Phase 0 binary) - noise band |
| Flag-off node counts | 2697/layer + 115971 full-step graph - identical to Phase 0 |
| Flag-off tolerance vs Phase 0 control | slot-indexed: selected-token 1.0, sequence 1.0, logits bit-exact (PASS; the naive request-index pairing reads 0.9375/0.9573 because request->slot assignment differs between concurrent runs - slot outputs are slot-seeded by design) |
| **Flag-on decode tok/s delta** | 70.63 vs 71.96 = **-1.85% (<= 3% gate PASS)**; wall -3.9% |
| Flag-on node delta | 2985/layer = +288 = exactly the stamp kernels (full graph +12384 = 288x43) |
| Flag-on cache behavior | unchanged: 1 capture + persistent cache hits, replay_ms 10.24 vs 10.11 |
| Flag-on eager delta | 42.33 vs 41.37 decode-domain (within eager run noise) |

## Phase 3 - Decomposition and reconciliation

**Table leg** (flag-on full capture, steady cache-hit replays, mean per rank
per layer-step; 10,965 layer-steps):

| Stage | ms | share of EP window (8.52 ms) |
|---|---:|---:|
| EP return copies (sum over 7 srcs) | **6.92** | **81%** |
|   - ep_copy by src (mean per pair) | 0.56-1.15 (SYS pairs ~2-2.6 ms, NV pairs ~15-20 us; max 3.8 ms) | |
| shared swiglu + down (dense f32) | 0.79 | 9.2% |
| barrier_1373 (post-compose 8x8 barrier) | 0.65 | 7.6% |
| route_plan_pack | 0.51 | 6.0% |
| gate/up GEMM (TurboMind, 192-row envelope) | 0.131 | 1.5% |
| down GEMM | 0.066 | 0.8% |
| barriers 954+978+1144 | 0.20 | 2.3% |
| dense_overlap (concurrent, dense stream) | 0.069 | (overlapped) |
| compose | 0.0125 | 0.15% |
| contrib_pack | 0.0079 | 0.09% |
| **ep_window total** | **8.52** | layer replay 10.24 ms (EP region = 83%) |

- **Closure: named stages cover 99.6% of the per-rank EP window (residual
  0.4% mean, p5 98.8) - far inside the <= 10% gate.** Rank-local elapsed vs
  critical path: per-rank ep_window 8.35-8.69 ms vs layer replay (root
  critical path) 10.24 ms; the ~1.7 ms remainder is the pre-EP prefix
  (hc_current + attention) + final_hc, outside the EP stage by construction.
- **Authority leg** (nsys, unmodified promoted graph,
  `phase3-authority-nsys-stages.txt`): rank-summed busy per layer-step:
  EP-return copies 48.25 ms (74% of ALL GPU busy time), other copies 5.16,
  NCCL 3.71, dense 2.78, TurboMind expert GEMM 2.39 (3.7%), everything else
  < 1 ms. Per-rank EP-return (r0:5.56 ... r2:6.53) matches the table leg
  (6.92 rank-mean) and Phase 1. Two independent instruments agree.
- **Reconciliation leg** (flag-on eager): ep_window 12.15 ms/rank/layer-step
  vs chrono buckets ep 10.41 + compose 0.92 = 11.33 (within ~7%; coverage
  residual 7.1%, also <= 10%). Key control: the eager EP return is the NCCL
  broadcast branch and costs only **0.68 ms/layer-step (compose_copy)** -
  10x cheaper than the promoted graph-copy path under SYS congestion.
- Route-skew (measured window): per-rank actual routes p50 24 / p95 52 /
  max 120 vs capacity 192; zero-route rank occurrences 4.4%. Sub-capacity
  ramp window (16 req x 16 tok): p50 12 / p95 28 / max 56, 11.2% zero-route
  ranks - and the stage profile is **identical** (ep_window 8.68 ms):
  envelope cost is load-independent.
- Capture-vs-replay separation: the serving first step is capture-only
  (replay deferred, not collected); all table rows are persistent-cache-hit
  replays; first-batch wall carries capture+instantiate (Phase 0: 11.36 vs
  10.11 ms/layer-step).

## Phase 4 - Hypothesis adjudication and B2 decision

**Math-vs-scaffolding: CONFIRMED, stronger than the prior.** Expert math
(gate/up 0.131 + down 0.066 + swiglu epilogue ~0.05) is ~0.25 ms of the
8.52 ms EP window = **~3% math / ~97% scaffolding** (authority leg: 3.7% of
all GPU busy). **Padded-GEMM tax: ~zero.** Eager actual-rows GEMM (p50 24
routes/rank) costs 0.220 ms vs the graph 192-row envelope 0.197 ms - the
grouped GEMM is launch-bound, not tile-bound, at decode shapes. The 3x
overcount has no measurable ms cost today.

Branch-table adjudication (measured ms per candidate, per rank per
layer-step, graph leg):

| Candidate dominant cost | Measured | Verdict |
|---|---|---|
| Per-pair EP return copies, esp. SYS | **6.92 ms (81% of EP window; 24/56 SYS pairs at ~2 ms each; NCCL broadcast control does the same job in 0.68 ms)** | **DOMINANT - B2-C leads 598** |
| Barrier-wait at sync_all sites | 0.85 ms (1373: 0.65; 954/978/1144: 0.20) | second - B2-D |
| Padded grouped GEMM (192 vs ~24-52 rows) | tax ~= 0 (0.197 vs 0.220 ms) | B2-A demoted to last/dropped |
| Contribution pack/reduce + compose | 0.020 ms | B2-B demoted (fold into C only if useful) |
| Expert GEMM < 1/3 of EP stage | 2.3% of EP window | B2-E stays last/optional |

**Decision - 598+ stage order:**

1. **B2-C transport (598 lead)**: eliminate SYS from the EP return. Two
   candidates, both graph-capturable, measured bars explicit: (a) the cheap
   first candidate - capture the existing NCCL broadcast return inside the
   graph (the eager control already runs the workload at 0.68 ms/layer);
   (b) static one-hop NVLink relay forwarding from the Phase 1 relay table
   (theoretical NV-class bound ~0.2 ms/layer). Promotion bar per Sprint 396:
   beat BOTH the NCCL control (0.68 ms) and the graph-copy path (6.92 ms).
   Expected step gain: EP window 8.52 -> ~1.9-2.6 ms; layer replay
   10.24 -> ~3.9-4.6 ms; decode-domain ~73.6 -> ~165-190 tok/s (~2.3-2.6x).
2. **B2-D (599)**: per-pair event dependencies replacing the 8x8
   `enqueue_cross_gpu_stream_barrier` at the EP sync_all sites - bounded by
   the measured 0.85 ms/layer barrier wait (re-measure after C; the 1373
   barrier absorbs copy skew today and will shrink with C).
3. **B2-B**: sparse fp16 row-indexed return + fused compose - only as a C
   refinement (dense-return padding is bandwidth, not kernel time; compose
   itself is 12 us).
4. **B2-A device-masked executor: dropped from the default order** - the
   measured padding tax is ~0 ms. Revisit only if C/D expose the GEMM.
5. **B2-E full fusion: last/optional** (expert GEMM 2.3% of EP stage).

Cycle target check: EP stage <= ~2 ms/layer is reachable with C(+D) alone;
aggregate >= ~3x the re-anchored 73.6 decode-domain baseline (~220 tok/s)
likely needs C+D plus part of the pre-EP prefix - C alone projects ~2.3-2.6x.

## Definition of Done checklist

1. Anchor reproduced/re-anchored with delta explained - **done** (Phase 0;
   decode-domain -1.8%, wall and eager re-anchored, causes documented).
2. Transport ground truth archived - **done** (topo dump, 56-pair
   latency/bandwidth table at 5 payloads, NVLink/SYS classification, in-situ
   nsys cross-check, written finding `PHASE1-FINDING.md`).
3. Decomposition exists and is reproducible - **done** (one launcher flag;
   per-rank per-layer TSV on the full-capture leg; covers the actual
   promoted `copy_f32_kernel` per-pair transport; NCCL appears only as the
   labeled eager control; raw nsys + TSV archived).
4. Self-consistent - **done** (residual 0.4% graph / 7.1% eager, both
   <= 10%; rank-local vs critical-path distinguished; route-skew, ramp
   window, capture-vs-replay separation recorded).
5. Non-perturbing - **done with one caveat** (flag-off: slot-indexed
   tolerance 1.0/1.0 + bit-exact logits, node counts unchanged, NCCL graph
   SYS edges remain zero by config; flag-on decode delta -1.85% <= 3%;
   node/cache deltas reported. Caveat: the naive index-paired tolerance
   reads 0.94 due to request->slot assignment nondeterminism in the harness,
   not engine divergence - slot-indexed comparison is the valid one).
6. Hypothesis adjudicated - **done** (~3%/97% confirmed; padded-GEMM tax
   measured ~= 0).
7. Decision recorded - **done** (this report; steering update is Phase 5).
8. Track reopened (README/steering/STATUS) - **deferred to orchestrator
   (Phase 5 per instruction)**.
9. Commits - handled by orchestrator; no commits made from this run.

Deviations (honest list):
- Graph-mode timing uses %globaltimer stamp kernels, not cudaEventRecord
  nodes (CUDA cannot time capture-recorded events; verified
  cudaErrorInvalidValue). Same pairing semantics, no captured-region
  alloc/D2H.
- route-plan vs routed-input-pack are one combined stage in the TSV (the
  boundary lives in `post_attention_ffn.cu`, outside the allowed edit
  surface); the nsys authority leg splits them (route_plan 0.099,
  input_pack 0.062 ms/rank/layer-step).
- The eager chrono splits were not extended stage-by-stage; the eager leg
  reuses the event-pair profiler (same stage list) and reconciles the chrono
  EP+compose buckets to within ~7%.
- The flag-off byte-identity run used the binary one collector-edit before
  final (the ep_window emitter, flag-on-only code, was added after); the
  flag-off object code path is unchanged by that edit.
- Phase 0/1 harness deviations (wave submission, UTF-8 replace, 900 s listen
  wait) are recorded in COMMANDS.md.

## Artifacts

Pod (gpu-01, hostPath-persistent):
- `/workspace/s597-phase01-artifacts/` - Phase 0/1: COMMANDS.md (full
  reproducible command log incl. Phase 2-4), build/pack logs, phase0-full
  + phase0-eager bench trees, phase0-full-control (tolerance control),
  nvidia-smi-topo.txt, peer-copy-microbench.tsv + binary,
  nsys-insitu.nsys-rep/.sqlite, PHASE1-FINDING.md.
- `/workspace/s597-phase234-artifacts/` - p2-flagoff-full (byte-identity +
  baseline), p3-flagon-full (table leg), p3-flagon-eager (reconciliation),
  p3-ramp16 (ramp window), p2-flagoff-tolerance.json, analyzers.
- Pack/contract: `/workspace/packs/ds4-appliance-full-tm-gated-s597`,
  `/workspace/s597-contract/`.

Laptop: `logs/from-cluster/sprint597-phase01/` and
`logs/from-cluster/sprint597-phase234/` (summary TSVs, analyses, finding,
microbench source, analyzers; no .nsys-rep in the repo).

Source changes (uncommitted, for orchestrator review): `engine/decode_loop.cu`,
`engine/runtime_options.cuh`, `engine/runtime_profiler.cu`,
`tools/ds4-v100-run-tp-ep-appliance.sh`; new standalone
`tools/s597-peer-copy-microbench.cu` (pod copy; source archived in
logs/from-cluster).
