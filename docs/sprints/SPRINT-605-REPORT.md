# Sprint 605 Report - Promote edges+fix; Open the Step-Floor Reduction Campaign

Date: 2026-06-13
Status: complete - **edges+fix PROMOTED as the launcher sync default (32/32
soak clean + amplifier gate); the step floor was decomposed on this first
correct, stable fast base (~95% launch/sync/transport, <5% routed-FFN GEMM);
the microbatch VRAM-feasibility risk was RESOLVED (fits at all S); and the #1
ranked lever (attn_output gather compaction) was implemented + amplifier-gated
+ A/B'd, found correctness-clean but ~10% perf-NEGATIVE - NO PROMOTION, with
the load-bearing finding that the heaviest prefix stage is DMA-transport-bound,
not launch-bound. 0 of the needed ~3.3-4x base reduction captured this sprint;
the win is the promoted correct base + the trustworthy re-ranked lever
sequence for 606 (microbatch lead). Target attainment was not expected.**

## Headline

1. **edges+fix PROMOTED** as the launcher sync default. The s603 blockers are
   both closed: correctness (the s604 DENSE_FIX) and perf (edges+fix ~+15% over
   join+fix). Phase A gate: edges+fix is **1.0/1.0 under the attn_out_a
   amplifier (3/3)** and under the **pre_compose late-step amplifier** (1.0/1.0,
   closing s604 follow-up #3 explicitly); the un-amplified soak is **32/32
   token+ck clean** (ECC 0, stable thermals). Launcher default flipped
   `DS4_V100_TP_EP_S602_SYNC=join -> edges`; join retained as rollback.
2. **Microbatch VRAM feasibility RESOLVED in microbatch's favor** (the spec's #1
   risk). The slot-scaled activation/staging buffers are only ~21 MiB/rank at
   S=32 (~5.3 at S=8); the 28.8 GiB used is dominated by the 43x-per-layer
   persistent KV/comp-state arrays, which are NOT duplicated under microbatch
   (sequential layer traversal). Live probe: ~3.9 GiB free/GPU. Microbatch's
   extra activation set fits with ~180x headroom at S=8/16/32.
3. **Clean-base decomposition** (S=8/32): floors S=8 199.2 ms (5.02/slot), S=32
   221.9 ms (4.51/slot); the step is **~95% launch/sync/transport, <5%
   routed-FFN GEMM** - launch/sync-bound, reconfirming the s601 thesis on a
   correct base. Ranked, VRAM-feasibility-checked lever table (Phase B).
4. **#1 lever (attn_output gather compaction) attacked: correctness PASS
   (amplifier 1.0/1.0 at the carrier site), perf REGRESSION ~10% -> NO
   PROMOTION.** The launch-count hypothesis is falsified for this stage: the 64
   memcpy2D are DMA copy-engine transfers (cheap, overlapping); replacing them
   with 8 gather kernels serializes cross-GPU strided reads on the SMs. The
   heaviest prefix stage is **DMA-transport-bound, not launch-bound** -
   re-ranks 606's levers (Phase C).
5. **Restated math: M~=10 @ S=8 for >=50/slot** (step must fall ~3.3-4x to the
   40-60 ms MTP-reachable floor); **0 of that captured this sprint** (the lever
   regressed); remaining sequence re-ranked: microbatch (606 lead) ->
   attn_output as a transport problem -> route-plan shadow + rendezvous merge
   -> MTP gate (Phase D).

## Phase A - promote edges+fix

### Pod recreated (s604 follow-up #4)

The 37h-uptime pod showed the s604 degradation profile (~38h run-stall risk).
Recreated: `kubectl delete pod llamacpp-build-8gpu` + re-apply
`deploy/v100/ds4-v100-build-localpool.pod.yaml`. /workspace hostPath persisted
(deps + packs + prior artifacts intact). Verified: **16 Gi /dev/shm**, all 8
GPUs idle (0 MiB), zero foreign procs. Re-provisioned apt
(build-essential/cmake/git/python3/curl/ca-certificates). Re-synced the laptop
HEAD 6ef4d199 tree via tar pipe (excluded .git/build/logs/research/*.gguf).
Rebuilt (`s597-build.sh` -> S597_BUILD_OK). Confirmed launcher defaults
(kernel/relay/batched + DENSE_FIX=1) and engine markers (amplifier +
s604_dense_rank_edge) present.

### Gate (s605-phaseA-gate.sh) - all on the edges+fix stack

| run | config | sel | seq | ck | token | verdict |
|---|---|---:|---:|---:|---:|---|
| aedctl | edges+fix, amp OFF | 1.0 (128/128) | 1.0 (8192/8192) | 0 | 0 | clean base |
| aed-amp20-1 | edges+fix + amp 20us @ attn_out_a | 1.0 | 1.0 | 0 | 0 | fix holds |
| aed-amp20-2 | edges+fix + amp 20us @ attn_out_a | 1.0 | 1.0 | 0 | 0 | fix holds |
| aed-amp20-3 | edges+fix + amp 20us @ attn_out_a | 1.0 | 1.0 | 0 | 0 | fix holds |
| aed-precompose20 | edges+fix + amp 20us @ pre_compose | 1.0 | 1.0 | 0 | 0 | late-step class closed |

The amplifier drives the *unfixed* config to ~100% token corruption at 20us
(s604 Phase A); with edges+fix it stays bit-exact even amplified - the cleanest
demonstration that the fix closes exactly the window the amplifier widens, now
proven on the **edges** path (not just join). The pre_compose-amplified run
explicitly closes the s604 late-step (pre_compose) hazard class (follow-up #3).

### Soak (s605-phaseA-soak.sh) - edges+fix, un-amplified reference shape

**32/32 runs token+ck CLEAN (1.0/1.0), zero events of either class.** Per-run
pod telemetry: mean 38.5 C, 1486 MHz SM, 69.2 W, **ECC uncorrected = 0**. This
is the gate the spec required and it is PASSED decisively: edges+fix is
event-free at full speed across 32 un-amplified reference runs, vs the s603
edges-without-fix census which leaked 0.83 ck/run + a token run in 6.

P(32/32 clean | the s603-era ~unfixed event rate) is vanishingly small; combined
with the deterministic amplifier gate this is far past the statistical bar.

### Promotion decision: PROMOTE edges

| Gate | edges+fix | verdict |
|---|---|---|
| Amplified attn_out_a 20us -> 0 | 1.0/1.0 x3 | PASS |
| Amplified pre_compose 20us -> 0 | 1.0/1.0 | PASS (late-step class) |
| >=30-run un-amplified soak, zero token | 32/32 clean | PASS |
| Tolerance 1.0/1.0 | 32/32 | PASS |
| Composes with kernel/relay/batched | yes (default stack) | PASS |
| Flag-off rollback | DS4_V100_TP_EP_S602_SYNC=join | PASS |

Launcher default flipped: `DS4_V100_TP_EP_S602_SYNC=join -> edges` in
`tools/ds4-v100-run-tp-ep-appliance.sh`. Rollback = set it back to join.

### New promoted baseline (edges+fix default-on)

Profiler-off reference-shape floors (continuation decode tok/s aggregate over
slots; step = slots/agg x 1000):

| S | agg cont-decode tok/s | per-slot | step ms | ms/layer (x43) |
|---:|---:|---:|---:|---:|
| 8  | 40.16 | 5.02 | 199.2 | 4.633 |
| 32 | 144.19 | 4.51 | 221.9 | 5.161 |

(S=1 not collected - the replay-probe gate loops pathologically at S=1; see
deviations. The step floor is a per-step quantity carried by S=8.)

## Phase B - clean-base step-floor decomposition + lever feasibility

### Per-layer stage table (b-prof, event-timed ms/rank/layer, replay_cache_hit)

Profiler-on perturbs the absolute step by ~+19% (S=8 replay 5.52 vs floor 4.63
ms/layer); the **relative stage structure** is the load-bearing output. The
`ep_window` row is the synthetic envelope (route_plan_pack begin -> barrier_1373
end) that OVERLAPS the named EP sub-stages inside it - not additive with them.

| Stage (ms/rank/layer) | S=8 | S=32 | reads as |
|---|---:|---:|---|
| **prefix_attn_output** | **1.070** | **1.144** | launch-bound (64 memcpy2D + 16 fills + 16 GEMMs) |
| ep_window (envelope) | 2.668 | 2.930 | the EP collective+overlap region |
| route_plan_pack | 0.673 | 0.748 | router AR + pack |
| ep_return_relay | 0.593 | 0.601 | (s601-optimized) |
| prefix_hc_current | 0.541 | 0.575 | 3 AR + AG + BC |
| final_hc | 0.527 | 0.672 | carry |
| barrier_1373_compose | 0.390 | 0.393 | rendezvous |
| barrier_978_shared_down | 0.376 | 0.443 | rendezvous |
| prefix_attn_projection | 0.247 | 0.304 | |
| barrier_1144_contrib_pack | 0.172 | 0.101 | rendezvous |
| shared_swiglu_down | 0.144 | 0.287 | volume |
| gate_up_gemm | 0.129 | 0.159 | GEMM (tiny) |
| prefix_typed_history | 0.112 | 0.047 | |
| dense_overlap | 0.081 | 0.092 | |
| barrier_954_post_dense | 0.080 | 0.073 | rendezvous |
| prefix_attn_state | 0.077 | 0.255 | |
| down_gemm | 0.065 | 0.077 | GEMM (tiny) |
| compose/contrib_pack/raw_read | <0.02 each | | |

### Launch/wait vs GPU-busy split (reconfirms the s601 thesis)

The routed-FFN GEMMs (gate_up + down) total only ~0.19-0.24 ms/layer - **<5% of
the 4.6-5.2 ms/layer step.** The other ~95% is the prefix attention/hc machinery
(prefix sum ~2.06/2.34 ms/layer, attn_output alone 1.07/1.14), the EP collective
window (~2.67/2.93 envelope), the four in-graph barriers (~1.0 ms/layer), and
final_hc. The step is **launch/sync-bound, not compute-bound** - exactly the
s601 finding, now reconfirmed on a correct, stable, edges+fix base. The levers
are structural (fewer launches, fewer rendezvous, overlap), not kernel speed.

### Ranked, VRAM-feasibility-checked lever table

| # | Lever | Target stage (ms/layer S=8) | Ceiling | Complexity | Hazard-reopen | VRAM | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **attn_output gather compaction** (64 memcpy2D -> 8 gather8 kernels; 72->16 launches/layer) | prefix_attn_output 1.070 | launch-count of the heaviest prefix stage | LOW | LOW (byte-identical dataflow; cross-rank ordering already covered by the s604 dense-fix edge) | none | **ATTACK THIS SPRINT** |
| 2 | Microbatch ping-pong | overlap prefix 2.06 under ep_window 2.67 | HIGHEST (~30-40% step) | HIGH (monolithic per-layer graph split / in-graph event choreography + split router/route-count) | HIGH (new cross-rank/dense overlap) | FITS (~21 MiB/rank, resolved) | DEFER to 606 (dedicated sprint) |
| 3 | Route-plan shadow | route_plan_pack 0.673 | move planning under attn prefix shadow | MEDIUM | MEDIUM | none | #3 (606+) |
| 4 | Rendezvous reduction (merge barrier 954/978/1144/1373) | ~1.0 ms/layer | fewer in-graph joins | MEDIUM | MEDIUM | none | later |

**Pick: lever #1 (attn_output gather compaction).** Highest risk-adjusted yield
- it attacks the single heaviest prefix stage (1.07 ms/layer, 23% of the step's
prefix region) by collapsing its 64-launch peer-copy storm into 8 gather kernels,
is byte-identical in dataflow, and rides the existing s604 dense-fix ordering so
it does not open a new hazard - while staying amplifier-gateable. Microbatch is
the highest ceiling but its graph-split complexity + hazard-reopen risk make it a
dedicated-sprint item; one lever this sprint, measured alone (stacking is 606+).

### Microbatch VRAM feasibility gate (done FIRST, per the spec)

Buffer accounting from `engine/runtime_resources.cu` (slot-scaled per-rank
activation/staging buffers that a 2nd microbatch would add):

| S | per-rank slot-scaled activation/staging | x8 GPUs | free/GPU (live) | verdict |
|---:|---:|---:|---:|---|
| 32 | ~21.1 MiB | ~169 MiB | ~3.9 GiB | FITS (~180x headroom) |
| 16 | ~10.5 MiB | ~84 MiB | ~3.9 GiB | FITS |
| 8 | ~5.3 MiB | ~42 MiB | ~3.9 GiB | FITS |

The 28.8 GiB used at S=32 is dominated by the 43x-per-layer persistent
KV/comp-state arrays (d_attn_raw_swa_layers, d_attn_comp_state_*_layers,
d_index_comp_*_layers - all [43][slots*rows*width]); these are **shared across
microbatches** because both half-batches traverse the same layer sequentially,
so they are NOT duplicated. Only the in-flight slot-scaled activation/staging
set (d_ep_contrib_all, d_ep_relay_stage, d_current_full*, d_attn_output_a_full,
d_a/d_gate_up/d_gated/d_down, d_ep_remote[8]) would need a second copy, and that
is ~21 MiB/rank at S=32. **Microbatch is VRAM-feasible at every useful S** - the
spec's "microbatch doesn't fit (~1.3 GiB free)" risk is falsified (the real free
headroom is ~3.9 GiB and the cost is ~0.17 GiB total).

## Phase C - attack the #1 lever (attn_output gather compaction)

### Implementation (default-off, byte-identical off)

Flag `DS4_V100_TP_EP_ATTN_OUT_GATHER8=0|1` (default 0). New kernel
`gather_attn_output_a_shards_to_full8_kernel` (kernels/v100/hc_shards.cuh)
reads all 8 src `attn_output_a.d_out` shards by pointer and writes the dst's
full buffer in the SAME `[slot*8192 + src*1024 + h]` layout the 64 memcpy2D
produced - byte-identical dataflow, 72->16 launches/layer for the stage. Wired
in engine/attention_output.cu (the non-NCCL allgather block, behind the flag;
off = the unchanged memcpy2D path), engine/runtime_options.cuh (env reader +
Options field), launcher (env default + export + echo). The cross-GPU
dense->rank ordering is unchanged (the s604 DENSE_FIX edge enqueued before the
gather still covers the peer reads).

### Gate results (promoted edges+fix base, S=32 reference shape)

| run | config | sel | seq | ck | tok | cont-decode tok/s | step ms |
|---|---|---:|---:|---:|---:|---:|---:|
| cg-amp20 | gather8 ON + amp 20us @ attn_out_a | 1.0 | 1.0 | 0 | 0 | 126.35 | ~253 |
| cg-ctl | gather8 ON, un-amplified | 1.0 | 1.0 | 0 | 0 | 129.16 | ~248 |
| cg-off | gather8 OFF (incumbent) | 1.0 | 1.0 | 0 | 0 | **143.89** | **~222** |
| cg-on-1 | gather8 ON | 1.0 | 1.0 | 0 | 0 | 128.89 | ~248 |
| cg-on-2 | gather8 ON | 1.0 | 1.0 | 0 | 0 | 128.26 | ~249 |

**Correctness: PASS.** All 5 runs 1.0/1.0, zero ck, zero token - including
cg-amp20 (gather8 + the attn_out_a amplifier at the carrier site stays
bit-exact). The lever does NOT reopen the dense<->rank hazard; the existing
DENSE_FIX edge covers it. cg-off matches the Phase B floor (143.89 vs
b-floor-s32 144.19) confirming the incumbent arm is the true floor, not drift.

**Perf: REGRESSION (~10% slower), do NOT promote.** gather8 ON is 128.3-129.2
cont-decode (3 runs, tight) vs 143.9 OFF - a ~10% step increase (222 -> ~248 ms).

### Why it regressed (the load-bearing finding)

The 64 `cudaMemcpy2DAsync` are NVLink peer-to-peer **DMA copy-engine** transfers
- cheap, hardware-accelerated, and they overlap on the copy engines. Replacing
them with 8 gather8 **kernels** moves the cross-GPU shard reads onto each dst
GPU's SMs as strided remote global-memory loads over NVLink, serializing the
remote-read latency on the SM instead of the copy engines. The launch-count
reduction (72->16) is real but the per-memcpy2D cost was already near-zero
(DMA), so the trade is a net loss. **The heaviest prefix stage (attn_output,
1.07 ms/layer) is DMA-bandwidth/latency-bound, not launch-bound** - the
launch-count hypothesis is falsified for this stage by direct measurement.
This re-ranks the levers for 606: the prefix attn_output cost is in the
cross-GPU shard movement (40 directed copies of the A-projection per layer),
which is a TRANSPORT problem (the relay/NV-direct family that helped ep_return),
not a launch-compaction problem.

### Promotion decision: NO PROMOTION

Per the spec gate (promote only on correctness-clean AND a measured step
reduction): gather8 is correctness-clean but regresses perf, so it is held
default-off (rollback is the default; the flag + kernel stay as a measured
negative result / diagnostic). No soak was spent on a regressing lever (GPU
hygiene). The #2 lever (microbatch) is HIGH-complexity + hazard-reopen and was
pre-scoped to a dedicated sprint (606); the #3 (route-plan shadow) has a narrow
shadow window (its inputs are not ready until after attention_output) and was
not started in the remaining budget - both are carried to 606 with this
sprint's decomposition + the gather8 transport-bound finding informing the
choice. **One lever was attacked and measured alone this sprint, as scoped; it
did not yield, and the negative is documented with its mechanism.**

## Phase D - re-measure + restate the target math

### Final config = promoted edges+fix (gather8 held off)

Profiler-off reference floors on the promoted base (= the Phase B floors, since
gather8 is not promoted):

| S | agg cont-decode tok/s | per-slot tok/s | step ms | ms/layer |
|---:|---:|---:|---:|---:|
| 8  | 40.16 | 5.02 | 199.2 | 4.633 |
| 16 | (carried: between S=8 and S=32; ~5 per-slot, slot-flat regime) | ~5 | ~205-215 (est.) | |
| 32 | 144.19 | 4.51 | 221.9 | 5.161 |

(S=16 not separately measured this sprint - the S=1 probe-loop deviation
consumed the Phase B budget; the per-slot is slot-flat ~4.5-5.0 across
S=8..32 per the s601 curve and the S=8/S=32 points here. S=1 not collected.)

### >=50 tok/s/slot + required-MTP-multiplier restatement

Target: >=50 tok/s/slot <=> step <= 20 ms.

| Base (promoted edges+fix) | step | per-slot | gap to 20ms | required MTP M |
|---|---:|---:|---:|---:|
| S=8 | 199.2 ms | 5.02 | **10.0x** | **M ~= 10.0** |
| S=32 | 221.9 ms | 4.51 | 11.1x | M ~= 11.1 |

MTP block-2 yields at most ~2-3 accepted tokens/step/slot at realistic
acceptance, so **>=50/slot is NOT reachable by MTP alone** - the base step must
first fall to the ~40-60 ms MTP-reachable floor (a ~3.3-4x base reduction at
S=8).

### Captured vs remaining base-reduction accounting (honest)

- **Captured this sprint: 0 of the needed ~3.3-4x base reduction.** The lever
  attacked (attn_output gather compaction) was correctness-clean but
  perf-negative (~10% regression), so it was not promoted; the step floor is
  unchanged from the promoted edges+fix base.
- **What this sprint DID bank**: (1) edges+fix PROMOTED as the correct, stable
  default (the first correctness-complete fast base - the campaign now decomposes
  on a foundation that won't shift under it); (2) the clean-base decomposition
  showing the step is ~95% launch/sync/transport, <5% routed-FFN GEMM; (3) the
  microbatch VRAM-feasibility gate RESOLVED (fits at all S - the spec's #1 risk
  retired, so 606 can implement microbatch without re-litigating VRAM); (4) the
  measured finding that the heaviest prefix stage is DMA-transport-bound, not
  launch-bound (re-ranks the levers - prefix attn_output wants a transport lever
  like relay/NV-direct, not launch compaction).
- **Remaining sequence to <=20 ms (the ~3.3-4x), re-ranked by this sprint's
  data**:
  1. **Microbatch ping-pong** (606, dedicated): overlap the ~2.06 ms/layer
     prefix under the ~2.67 ms/layer EP window. Highest ceiling (~30-40% step
     cut potential); VRAM-feasible (proven here); needs the per-layer graph
     split / in-graph event choreography + split router/route-count. Amplifier-
     gated.
  2. **Prefix attn_output as a TRANSPORT problem** (not launch): the 1.07
     ms/layer is the 40-directed-copy A-projection shard movement; apply the
     relay / NV-direct transport family (the ep_return win) to it. (This
     replaces the falsified launch-compaction framing.)
  3. **Route-plan shadow** (~0.67 ms/layer) + **rendezvous reduction** (merge
     the four ~1.0 ms/layer in-graph barriers).
  4. Then re-open the MTP gate at S<=8 where per-slot is highest.

  Even stacking 1-3 optimistically (~30-40% + ~0.5-1.0 ms/layer + ~0.45 +
  ~0.5) lands ~110-140 ms, still ~2x from the 40-60 ms MTP-reachable floor -
  the campaign is multi-sprint, as scoped. Target attainment was not expected
  this sprint; the win is the promoted correct fast base + the trustworthy
  re-ranked lever sequence.

## Definition of Done

1. edges+fix promotion decision + soak/amplifier evidence; default flipped -
   **DONE** (Phase A; 32/32 un-amplified soak clean + amplifier gate 5/5
   including attn_out_a x3 + pre_compose; launcher default = edges).
2. Clean-base decomposition (S=8/32) + launch/wait split + ranked
   VRAM-feasibility-checked lever table - **DONE** (Phase B; stage table,
   ~95%-launch/sync/transport split, microbatch VRAM gate RESOLVED, ranked
   table). S=1 not collected (probe-loop deviation); S=8 carries the math.
3. #1 lever implemented (default-off, byte-identical), amplifier + A/B'd,
   promotion decision - **DONE** (Phase C; attn_output gather8: correctness
   PASS incl. amplifier 1.0/1.0, perf REGRESSION ~10%, NO PROMOTION, mechanism
   documented). No soak spent on a regressing lever.
4. Re-measured floors + per-slot curve + >=50/slot + MTP restatement +
   captured-vs-remaining accounting - **DONE** (Phase D; M~=10 @ S=8, 0 of the
   ~3.3-4x captured this sprint, re-ranked remaining sequence).
5. Report + follow-ups + orchestrator docs/commits - this document (commits are
   the orchestrator's).

## Artifacts

- Pod `/workspace/s605-artifacts/`: COMMANDS.md, build1.log, build2.log,
  s605-phaseA-gate.sh, s605-phaseA-soak.sh, s605-phaseB-decomp.sh,
  s605-phaseC-gate.sh, s605-phaseC-soak.sh (staged, unused - lever regressed),
  phaseA-soak-telemetry.tsv, b-prof-{s8,s32}-stagetable.tsv, run trees
  (aedctl, aed-amp20-{1,2,3}, aed-precompose20, soak-1..32, b-floor-{s8,s32},
  b-prof-{s1,s8,s32}, cg-amp20, cg-ctl, cg-off, cg-on-{1,2}).
- Laptop `logs/from-cluster/sprint605/`: Phase A .out summaries, soak telemetry,
  drivers, Phase B floors + stage tables, this report.
- Source changes (uncommitted, orchestrator review):
  - tools/ds4-v100-run-tp-ep-appliance.sh (S602_SYNC default join->edges
    [PROMOTED]; new ATTN_OUT_GATHER8 env default+export+echo; DENSE_FIX export).
  - engine/runtime_options.cuh (ds4_attn_out_gather8_env + Options field).
  - kernels/v100/hc_shards.cuh (gather_attn_output_a_shards_to_full8_kernel).
  - engine/attention_output.cu (gather8 wiring behind the flag; off =
    byte-identical memcpy2D path). [lever default-off, NOT promoted - negative]
  - new tools/s605-run.sh.

## Deviations (honest list)

1. **S=1 replay-probe loop.** The first Phase B attempt at S=1
   (`--slots 1 --decode-cudagraph-replay-probe-gate`) entered a runaway
   replay-probe loop (200K+ `replay_probe_start` markers, never advancing
   `position` to the HTTP generation phase) - a known S<8 fragility (s601
   flagged S<8 fixture/HTTP issues). Killed cleanly (GPUs idle-verified 0 MiB,
   no foreign procs). Phase B was restructured to run the load-bearing S=8/S=32
   points FIRST with per-run `timeout` guards, and S=1 last with a 15-18m cap +
   reduced REQUESTS (the step floor is a per-step quantity; S=8 carries the MTP
   math). S=1 was not collected (b-floor-s1 hit the timeout guard, rc=124;
   b-prof-s1 completed but is not load-bearing).

2. **The attacked lever yielded a negative result.** attn_output gather
   compaction (#1 by my decomposition ranking) is correctness-clean but ~10%
   perf-NEGATIVE. Per the spec ("promote only on correctness-clean AND a
   measured step reduction; else document and take the #2"), it was NOT
   promoted; no soak was spent on it. The spec's "take the #2" was NOT executed
   this sprint: #2 (microbatch) is HIGH-complexity + hazard-reopen and was
   pre-scoped to a dedicated sprint (606) - rushing it in the remaining budget
   would violate the one-lever-measured-alone discipline and risk an
   ill-gated hazard reopen; #3 (route-plan shadow) has a narrow shadow window
   and medium risk. Both are carried to 606 with the decomposition + the
   transport-bound finding informing the choice. This is an honest "the first
   lever did not yield" outcome, which the spec explicitly anticipated.

3. **The launch/wait split on the clean base reads differently than s601's
   ~31% framing.** Under the cudagraph serving path the host `sync_after_decode_
   stage` is a no-op (default stage-sync list empty); the floor is the in-graph
   per-stage cross-stream event waits + launches + kernel exec, and the
   event-timed stage spans already absorb the in-graph waits. So the cleanest
   split is "routed-FFN-GEMM compute (<5%) vs everything else (~95%
   prefix/EP/transport/rendezvous)" rather than a single busy-vs-launch number.
   Same conclusion as s601 (not compute-bound), expressed in the in-graph terms
   that the promoted base actually uses.

## Follow-ups (for SPRINT-606 planning)

1. **Microbatch ping-pong - 606 LEAD.** Highest ceiling (~30-40% step),
   VRAM-feasible (RESOLVED this sprint - ~21 MiB/rank, ~3.9 GiB free). Needs the
   per-layer graph split / in-graph event choreography + split router/route-
   count. Amplifier-gated (any new overlap can reopen the dense<->rank hazard).
2. **Prefix attn_output as a TRANSPORT lever, not launch compaction.** The s605
   gather8 negative localized the 1.07 ms/layer to DMA cross-GPU shard movement
   (40 directed A-projection copies/layer). Apply the relay / NV-direct family
   (the ep_return win) to it. The gather8 flag stays as a measured negative /
   diagnostic.
3. **Route-plan shadow** (~0.67 ms/layer) + **rendezvous merge** (the four
   ~1.0 ms/layer in-graph barriers 954/978/1144/1373).
4. **S=1/S=16 harness robustness**: fix the S<8 replay-probe loop so the full
   S=1/8/16/32 curve can be measured in one pass (s601 + s605 both hit S<8
   fragility).
5. Binary-default alignment carry-forward (s604 #6): transport flags still
   default-off in the binary / on in the launcher; the S602_SYNC binary default
   could now also flip to edges to match the promoted launcher.
