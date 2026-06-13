# Sprint 603 Report - Join Reclaim on the Zero-NCCL Graph

Date: 2026-06-12
Status: complete - **edges mode implemented from a code-derived dependency
table and proven bit-identical 3/3 under the harshest Simple-stress
detector, but it reclaims only ~25% of the join pool (+6.4%, 153.4 ->
163.9 median; gate needs >= +15%) and FAILS the LL census (1 token-run in
6) - and the bigger finding is that the join default FAILS it too: the
zero-NCCL stack's residual flicker fired token-level events on BOTH syncs
this sprint (s602's "token-race-zero" was 6-run luck), the rate is
monotone in step speed and indifferent to rank-stream sync strength, and
the s601 full rank+dense barrier is the ONLY event-free configuration
(6/6 bit-exact at 2.1x cost). The hazard is a rank<->dense ordering gap;
the derived bcast-site dense-WAR guard was implemented, gated, and
FALSIFIED (events persist with it on). NO defaults change. Step floors:
edges 175.0/177.1 ms (S=1/S=8), join 186.9/188.7 - required MTP
multiplier 8.8-9.4; >=50/slot remains unreachable without closing the
rank<->dense hazard and the remaining join pool.**

## Headline

1. **Phase A**: per-collective producer->consumer edge table derived from
   the runtime_pack.cu kernel arguments BEFORE implementation (below);
   `DS4_V100_TP_EP_S602_SYNC=join|edges` + per-point bisect overrides
   landed; default join byte-identical (capture 43x971 nodes unchanged).
2. **Phase B**: edges is BIT-IDENTICAL 3/3 under Simple stress (stronger
   than the s602 rank-join's 0.997-0.999) - the derived table with its E2
   exit closure is the strongest rank-stream ordering yet measured. But
   the LL census fails: 0.83 ck-events/run and one token-run (ge-4, 9
   slots) - and the join control itself token-flipped (actl603), so the
   census gate is failed by the INCUMBENT too.
3. **Phase D (promoted to critical path)**: full-barrier control n=6 ALL
   BIT-EXACT (closes the s602 n=1 caveat); sync-strength bisect shows the
   event rate is monotone in pacing and indifferent to E0/E1 strength ->
   the hazard is OUTSIDE the rank-stream sync. The one derivable
   rank<->dense WAR (bcast site, d_current_full vs dense readers) was
   implemented as DENSE_GUARD and falsified by census (gd-6 token event,
   19 slots @ step 1; gj-1 token event with guard on join).
4. **Phase C**: NO PROMOTION - edges +6.4% (158.4-166.0 vs join
   153.8-154.9) < +15% gate, census FAIL; stage tables show the reclaim
   lands exactly in the site-hosting stages (route_plan_pack -0.12,
   prefix_hc_current -0.16 ms/layer) but transitive peer-rendezvous keeps
   ~75% of the pool. Launcher defaults unchanged (SYNC=join).
5. **Program impact**: the zero-NCCL stack is NOT token-race-zero; the
   only correctness-complete configuration known is FULL_BARRIER=1 at
   ~dd 72 (2.1x cost). The correctness-flip calculus from s602 must be
   re-weighed against this; closing the rank<->dense gap is now the
   single gating correctness item, ahead of any perf lever.

## Phase A - the dependency edge table (derived from the code BEFORE implementation)

All read/write sets below are taken from the kernel arguments in
`engine/runtime_pack.cu` (s602_copy3_kernel, s602_fold_kernel,
s602_gather8_kernel, copy_f32_kernel at the broadcast site) and the call
sites in `engine/hc_current.cu`, `engine/router_step.cu`,
`engine/post_attention_ffn.cu`, `engine/runtime_resources.cu`.

### Topology facts (code, not intuition)

- `s601_nv_adjacent(a,b) = (a>>2 == b>>2) || (b == a^4)`: each rank g has
  exactly 4 NVLink peers NV(g) = its 3 quad-mates + its mirror g^4.
- SYS sources of a destination dst = quadmates(dst^4); the relay for dst is
  R = dst^4 (NVLink-adjacent to both ends; 3 directed relays per GPU).
- The ring-order-exact fold is a SINGLE kernel per destination reading all
  8 source pointers (`S602Ptrs8`); the NCCL ring order is reproduced inside
  the kernel's accumulation loop, not across streams - so the fold itself
  imposes no cross-stream chain. The cross-stream dependencies are entirely
  the byte-movement (copy3/gather8/broadcast copies) plus the fold's direct
  remote reads.

### The 8 sites per layer (9 classes; in reference-config layer order)

| # | Site (cls) | Kind | in / shard buffers (per rank) | producer kernel (rank stream) | out | downstream consumer |
|---|---|---|---|---|---|---|
| S1 | hc_max + hc_mix | AR x2, fold_all | d_hc_reduce_max [slots], d_hc_reduce_mix [slots*kHcMix] | hc_local_max_mix_partial | d_s602_out_max/mix | hc_local_stable_sumsq, hc_apply_reduced_mix_split (own stream) |
| S2 | hc_sumsq | AR, fold_all | d_hc_reduce_sumsq [slots] | hc_local_stable_sumsq | d_s602_out_sumsq | hc_apply_reduced_mix_split (own) |
| S3 | hc_ag | AG | d_current_shard | hc_weighted_sum_shard | d_current_full_rank_major | rank_major_..._to_slot_major (own) |
| S4 | ffn_bcast | BC root 0 | src_device0 (control = rank-0 stream) | control-stream producers | d_current_full (all ranks) | downstream rank/dense kernels (pre-existing ordering) |
| S5 | post_ag | AG | d_post_attn_shard | attention output chain (rank streams) | d_post_attn_full_rank_major | ffn-norm consumers (own) |
| S6 | r_max | AR, fold_all | d_hc_reduce_max (REUSED from S1) | current_shard_max | d_s602_out_rmax | current_shard_stable_sumsq (own) |
| S7 | r_sumsq | AR, fold_all | d_hc_reduce_sumsq (REUSED from S2) | current_shard_stable_sumsq | d_s602_out_rsumsq | router_logits_allreduce_partial (own) |
| S8 | r_logits | AR, fold dst 0 ONLY (relay 4 only) | d_router_logits_rank_major | router_logits_allreduce_partial | d_s602_out_logits (rank 0) | control memcpy AFTER enqueue_control_wait_after_rank_streams (all-rank wait, unchanged) |

### Per-collective producer -> consumer edges

AR site mechanics: B0; copy3 on relay R reads `op.src[q]` for q in
quadmates(R), writes `stage(dst=R^4)[q]` (dst-resident); B1; fold on dst
reads `op.src[s]` directly for s in {dst} u NV(dst) and `stage(dst)[q]` for
the 3 SYS q, writes `out[dst]`.

| Edge | Hazard | Consumer waits | Producer set (per rank g) | Sync point |
|---|---|---|---|---|
| A1 | RAW: copy3 reads quad-mates' partials | relay g | quadmates(g) | E0 |
| A2 | RAW: fold reads NV peers' partials directly | fold dst g | NV(g) = quadmates + mirror | E0 |
| A3 | WAR: relay write into stage(g^4) vs the mirror's fold of the same cls (previous layer) | relay g | mirror g^4 | E0 (in NV(g)) |
| A4 | RAW: fold reads the staged SYS forwards written by the mirror's copy3 | fold dst g | mirror g^4 | E1 |
| A5 | WAR closure ("my in-buffer is free when my collective completes" - NCCL's contract: NCCL kernels only ever read their OWN rank's user buffers, so under NCCL a rank's buffers are free at its own stream's completion; the s602 folds remote-read peer user buffers, which introduces WARs NCCL never had) | every g before its in-buffer is overwritten | readers of in[g] = folds on NV(g) + copy3 on quadmates(g) = NV(g), recorded post-fold | E2 (site exit) |

AG site mechanics: B0; gather8 on dst reads `shard[s]` for s in {dst} u
NV(dst), writes its own out segments; copy3 on relay R reads shard[q] for q
in quadmates(R), writes the 3 SYS segments of out[R^4] (dst-resident); B1;
the downstream consumer on dst reads the full out buffer.

| Edge | Hazard | Consumer waits | Producer set | Sync point |
|---|---|---|---|---|
| G1 | RAW: gather8 direct shard reads | dst g | NV(g) | E0 |
| G2 | RAW: copy3 shard reads | relay g | quadmates(g) | E0 |
| G3 | WAR: relay writes out[g^4] SYS segments vs the mirror's previous-layer consumer of the same buffer | relay g | mirror g^4 | E0 (in NV(g)) |
| G4 | RAW: consumer on dst reads the relay-written SYS segments | dst g | mirror g^4 | E1 |
| G5 | WAR closure: readers of shard[g] = gather8 on NV(g) + copy3 on quadmates(g) = NV(g); all reads are PRE-E1, so the exit closure rides on E1 | g | NV(g) | E1 = peers (no post-E1 site work exists to hang an E2 on) |

BC site mechanics (root 0): B0; dsts 0..4 copy src_device0 -> own
d_current_full (dst streams); relays 1,2,3 copy src_device0 -> d_current_full
of dst = relay^4 in {5,6,7}; B1.

| Edge | Hazard | Consumer waits | Producer set | Sync point |
|---|---|---|---|---|
| B1e | RAW: all copies read src_device0 (produced on the rank-0/control stream) | ranks 1..4 (0 local) | rank 0 (0 in NV(g) for each) | E0 |
| B2e | WAR: relay write into dst's d_current_full vs dst's prior consumers | relays 1,2,3 | mirror (5,6,7 resp.) | E0 (in NV(g)) |
| B3e | RAW: dsts 5,6,7 consume the relay-written buffer | dsts 5,6,7 | mirror (1,2,3 resp.) | E1 |
| B4e | WAR closure: readers of src_device0 = ranks 1..4 = NV(0) | rank 0 | NV(0) | E1 = peers |

### The uniform edges mode (union of all required edges, per sync point)

- **E0 (inputs-ready)**: every rank g waits its 4 NVLink peers NV(g).
  (Union of A1+A2+A3 / G1+G2+G3 / B1e+B2e; every required producer set is
  a subset of NV(g).)
- **E1 (staged-forwards-ready)**: AR sites: g waits mirror g^4 (A4 - the
  mirror's copy3 is the only cross-stream writer into g-resident memory g
  reads after E1). AG/BC sites: g waits NV(g) (G4+G5 / B3e+B4e - E1 is
  also the site exit there).
- **E2 (AR site exit, after the fold launches)**: g waits NV(g) (A5).

### Why the s602 pairwise set failed, in this table's terms

The falsified s602 pairwise set was exactly {E0 = NV(g), E1 = mirror, no
E2} - it satisfies every RAW/WAR edge INSIDE a site but drops the exit
closure (A5/G5/B4e): a rank could leave the site and proceed while its
NVLink peers' folds were still remote-reading its live in-place buffers.
For the concrete buffer-reuse pairs in this graph (S1->S6 d_hc_reduce_max,
S2->S7 d_hc_reduce_sumsq, plus the per-layer reuse of every in buffer) the
overwrite is always >= 2 sites downstream of the last remote read, and each
intervening E0 transitively re-covers the WAR - so the closure SHOULD be
redundant by >= 1 site of margin. The s602 evidence says weak ordering
nevertheless diverged; the closure (E2/G5/B4e) is the one defensible
structural difference between the falsified set and this table, so edges
mode includes it. If edges still fails the census, the built-in per-point
overrides bisect the bracket (see flag inventory) - pairwise FAILED, join
PASSES, and {E0=join,E1=mirror,E2=none} / {E0=peers,E1=join} are the two
half-join-cost midpoints that localize the missing dependency to the
input-rendezvous side or the exit side.

### Sync cost accounting (per layer, 8 sites)

| Mode | 8-way rendezvous points | sync points | cross-stream waits |
|---|---:|---:|---:|
| join (default) | 16 | 16 | 16 x 56 = 896 |
| edges | 0 | 21 (8 E0 + 8 E1 + 5 E2) | 8x32 + 5x8 + 3x32 + 5x32 = 552 |

### Flag inventory (new)

- `DS4_V100_TP_EP_S602_SYNC=join|edges` (default join - byte-identical to
  the s602 stack; edges = the table above).
- Per-point bisect overrides (read only in edges mode):
  `DS4_V100_TP_EP_S602_SYNC_E0=join|peers`,
  `DS4_V100_TP_EP_S602_SYNC_E1=join|peers|mirror` (applies at both the AR
  E1 and the AG/BC E1; the AG/BC default is peers, AR default mirror),
  `DS4_V100_TP_EP_S602_SYNC_E2=join|peers|none`.
- `DS4_V100_TP_EP_S602_FULL_BARRIER=1` still overrides every non-none sync
  point to the s601 rank+dense barrier (rollback/diagnosis, unchanged).

### Implementation notes

- Same pre-allocated `graph_stream_done` event-slot machinery as
  s602_rank_join (fixed order, graph-capturable, no allocation at enqueue).
- mirror/peers points record on all 8 rank streams and wait 1/4 events
  respectively; none enqueues nothing and consumes no event slot, so the
  default-join capture is byte-identical to s602 (verified by gate run).
- For S8 (r_logits, fold_all=false) the uniform edge set is a strict
  superset of the minimal one (only dst 0 folds, only relay 4 forwards);
  extra edges only strengthen ordering.

### Phase A gate runs (build1, pod, same day)

- `esmoke` (SYNC=edges, 8 tok x 32 req): rc=0, no hang/crash;
  `tp_ep_s602_init ... sync edges e0 peers e1 mirror/peers e2 peers PASS`.
- `actl603` (defaults, SYNC=join): capture structure IDENTICAL to the s602
  binary (43 layers x 971 nodes; event record/wait pairs create
  dependencies, not nodes, so node-count equality + the unchanged code path
  is the byte-identity argument); decode-domain 154.84 / wall 113.34
  (inside the s602 census band 142.3-156.6). Tolerance vs the s597
  control: **ONE event, batch 2 step 42 - ck on all 32 slots AND a token
  flip on slot 0** (selected 127/128, sequence 8170/8192). This is the
  FIRST token-level event ever observed on the zero-NCCL join stack (s602:
  6/6 token-clean, flicker ck-only at steps 59-63). Join mode in this
  binary is provably the same enqueue path, so this reads as the s602
  residual flicker escalating to token level on run 7 of the
  configuration - a load-bearing new fact for Phase D and for the
  correctness narrative; join-control re-runs (actl603b/c) were added to
  estimate the rate before the A/B.

### Join-control re-runs (rate estimate)

actl603b / actl603c: BOTH BIT-EXACT (tolerance 1.0/1.0, zero ck and token
events); decode-domain 153.81 / 154.90. Join-control verdict on this
binary: 3 runs, 1 ck event (which was ALSO a token event), 2 clean - vs
the s602 baseline 6 runs / 1 ck-only / 0 token. The flicker is live on
the join default and has now crossed to token level once.

## Phase B - race gates on edges

### Simple-stress pairwise (e-sb-1/2/3, NCCL_PROTO=Simple, SYNC=edges)

ALL THREE pairwise comparisons **BIT-IDENTICAL: 1.0/1.0, first_ck/first_tok
histograms all None**. This is STRONGER than the s602 rank-join baseline
under the same detector (b5-sb: 0.9976/0.9969/0.9994 with 1-2 divergent
batches per pair). The derived edge set passes the harshest stress detector
outright.

### LL census (ge-1..6, SYNC=edges, vs s597 control)

| run | tolerance | ck events | token events | decode-domain |
|---|---|---:|---:|---:|
| ge-1 | 1.0/1.0 | 1 (s42) | 0 | 164.81 |
| ge-2 | 1.0/1.0 | 0 | 0 | 165.95 |
| ge-3 | 1.0/1.0 | 1 (s15) | 0 | 164.17 |
| ge-4 | 0.9297/0.9739 | 2 (s26, s58) | 2 (5+4 slots) | 163.66 |
| ge-5 | 1.0/1.0 | 1 (s39) | 0 | 158.38 |
| ge-6 | 1.0/1.0 | 0 | 0 | 163.26 |

- ck rate 0.83/run, token rate 1 run in 6 - **census gate FAIL** (zero
  token events required; ck <= 0.17/run baseline).
- Pairwise determinism: 10/15 pairs token-identical; the 5 divergent pairs
  are exactly the ones involving ge-4 (the single token-divergent run).
- Every ck event marks ALL 32 slots of one batch at one step (the
  corrupted quantity is batch-wide/shared, not per-slot), onsets variable
  (15/26/39/42/58 + join's 42), occasionally crossing to token flips.

### The synthesis (why this is NOT a missing site edge)

1. Under maximal Simple stress, edges is bit-identical 3/3 - if the edge
   table were missing a dependency, stress should amplify it (it amplified
   the s602 pairwise gap 30x). It does not.
2. The join default itself fired a token event the same day (actl603) -
   the hazard exists at BOTH sync strengths.
3. The rate orders with pacing: edges (faster, ~+6.4%) 0.83 ck/run >
   join (1/3 runs) > s602's slower-day join (0.17/run) - the s600
   rate-vs-spacing curve again.

Working hypothesis: the late/variable-step flicker lives OUTSIDE the s602
site synchronization (the s602 fb pair already showed the s601 full
rank+dense barrier is bit-stable) - most plausibly a rank<->dense ordering
gap somewhere in the layer. Bisect chain launched: full-barrier LL census
n=6 (the Phase D control, now on the critical path) + E0=join and E1=join
edge variants x3 each.

## Phase D - the flicker hunt (promoted to the critical path)

### Full-barrier control, n=6 (the s602 n=1 caveat closed)

fb-1..6 (`DS4_V100_TP_EP_S602_FULL_BARRIER=1`, LL, full stack): **ALL SIX
BIT-EXACT vs the s597 control (1.0/1.0, zero ck events, zero token
events)**, decode-domain 71.1-72.8 (2.1x cost). The s601 rank+dense
barrier eliminates the flicker entirely.

### Sync-strength bisect (vb = E0->join, vc = E1->join, edges otherwise)

| config | n | ck events/run | token-event runs | decode-domain |
|---|---:|---:|---:|---|
| fb (rank+dense joins) | 6 | 0 | 0 | 71.1-72.8 |
| vc (E1=join) | 3 | 0.33 (s53) | 0 | 149.7-153.3 |
| join (s602 default) | 3 | 0.33 | 1 (s42) | 153.8-154.9 |
| vb (E0=join) | 3 | 0.67 (s21 x2) | 1 (s21, 2 slots) | 157.6-159.7 |
| edges | 6 | 0.83 | 1 (s26+s58, 9 slots) | 158.4-166.0 |

The event rate is MONOTONE IN STEP SPEED and indifferent to which
rank-stream sync point is strengthened - the missing dependency is not
between rank streams at all. Combined with fb's 6/6 zero (the only config
that joins the DENSE streams at the sites), the hazard is a rank<->dense
ordering gap that every rank-stream-only sync leaves open, with exposure
scaling with pacing (the s600 rate-vs-spacing curve, third reconfirmation).

### The derived locus and the fix

Re-auditing the Phase A table for rank<->dense interactions: the ONLY s602
site whose written buffer has dense-stream consumers is the **ffn_bcast
site** - its writers overwrite `d_current_full` (ranks 0..4 own-copy from
src_device0; relays 1,2,3 cross-write the buffers of 5,6,7), and the
PREVIOUS value of `d_current_full` is consumed by dense-stream kernels
that no rank-stream join orders (WAR). The signature fits exactly:
batch-wide buffer (every event marks all 32 slots of a batch at one step),
timing-dependent variable onset, present on join/edges/NCCL alike (the
s60x promoted-path "latent event" class), absent under the full barrier.

Fix (build2): `DS4_V100_TP_EP_S602_DENSE_GUARD=1` - at the bcast site's
E0, record one pre-allocated event on every dense stream and make each
writer's rank stream wait the dense events of the GPUs whose buffers it
writes (own + mirror). 8 records + 16 waits, no joins, graph-capturable.
`=2` extends the guard to every s602 site (diagnostic superset); default 0
(byte-identical off-path).

### The fix gates - FALSIFIED

| run set | config | result |
|---|---|---|
| gd-1..6 | edges + DENSE_GUARD=1 | 4 clean; gd-2 ck-only (s21); **gd-6 TOKEN event - onset step 1, 19/32 slots flipping + ck s38** (the classic early-locus class, distinct from the late-step flicker) |
| gd-sb-1..3 | Simple stress, edges + guard | 2/3 pairs bit-identical; one ck-only batch divergence (s40), zero token |
| gj-1..3 | join + DENSE_GUARD=1 | **gj-1 TOKEN event (s42+s52, 15 flips)**; gj-2 ck-only x2; gj-3 clean |

The guard does not move the event rate on either sync mode. The bcast-site
dense-WAR is not the (only) carrier. DENSE_GUARD=2 (all sites) was NOT
census'd - the =1 falsification plus the RAW-side analysis (remote
readers of a rank's shard would need edges onto the PRODUCER's dense
stream, which guard mode 2 as built does not provide either) made it a
low-probability spend; it remains in the binary as a diagnostic.

**Flicker-hunt verdict**: localized to a rank<->dense (or copy-stream)
ordering gap that only the s601 full barrier closes (6/6 bit-exact);
batch-wide signature; rate scales with pacing; occasionally token-level
on every rank-stream-only sync mode (join 1/3 runs, join+guard 1/3,
edges 1/6, edges+guard 1/6 today). gd-6's step-1 19-slot event suggests
at least one additional early-locus event class beyond the late-step
flicker. Search space for 604: per-site full-barrier bisect (16 knobs,
the fb bracket), AG-site shard RAW vs dense producers, the checksum
collection path, and pod-state telemetry (the rates moved between s602's
day and today on identical configs - see deviations).

## Phase C - perf A/B + promotion

### Gate matrix (edges vs the join default, same binary, same day)

| Gate | Result |
|---|---|
| Race gates clean (Simple-stress pairwise) | PASS - 3/3 BIT-IDENTICAL (better than rank-join baseline) |
| Race gates clean (LL census >= 6, zero token, ck <= 0.17/run) | **FAIL** - 0.83 ck/run, 1 token-run in 6 (but the join INCUMBENT also fails: 0.33 ck/run, 1 token-run in 3) |
| Tolerance 1.0/1.0 vs s597 control | 5/6 runs PASS (ge-4 FAIL 0.9297/0.9739) |
| Perf >= +15% over join (~153.4 -> >= 176) | **FAIL** - edges 158.4-166.0, median 163.9 = +6.4% |
| VRAM | PASS - 28768 MiB unchanged |

**Verdict: NO PROMOTION. Launcher default stays `DS4_V100_TP_EP_S602_SYNC=join`**;
edges + the per-point overrides + DENSE_GUARD retained as opt-in envs
(rollback = unset).

### Where the reclaimed time went (d8ep vs d8jp, S=8, profiler-on, ms/rank/layer)

| Stage | join | edges | delta |
|---|---:|---:|---:|
| ep_window (envelope) | 2.586 | 2.483 | -0.103 |
| route_plan_pack (router ARs) | 0.798 | 0.677 | -0.121 |
| prefix_hc_current (hc sites) | 0.707 | 0.544 | -0.163 |
| prefix_attn_output | 0.642 | 0.650 | +0.008 |
| final_hc | 0.519 | 0.523 | +0.004 |
| ep_return_relay | 0.584 | 0.589 | +0.005 |

The reclaim (~0.39 ms/layer visible; step delta 188.7 -> 177.1 = 0.27
ms/layer flat) lands exactly in the stages hosting the s602 sites, but
~75% of the 1.5 ms/layer join pool survives: each peers-wait still
rendezvouses with quad+mirror, and two chained sites transitively join
all 8 rank streams - the edges win is bounded by sync DEPTH, not count.
Further reclaim needs fewer sync points (merged/fused sites) or join-free
fold dataflow, not weaker edges.

## Phase E - program restatement

- **Step floors** (256K ctx, 64 tok/req, SKIP_TOL): edges 175.0 ms (S=1,
  d1f603 5.71 tok/s) / 177.1 ms (S=8, d8f603 45.18 tok/s, 5.65/slot);
  join floors stand at s602's 186.9/188.7 (join is byte-identical to
  build5).
- **>=50 tok/s/slot budget**: target step <= 20 ms. Gap 8.8x (edges
  floor) / 9.3-9.4x (join floor). Required MTP acceptance multiplier
  M = 8.8-8.9 on edges, 9.3-9.4 on join - **>=50/slot remains
  unreachable via MTP alone**, and edges' floor cannot be banked while
  its census fails. Sequencing unchanged: (1) close the rank<->dense
  hazard (now the gating CORRECTNESS item - no config below 2.1x cost is
  event-free), (2) reclaim the remaining join pool (~1.1 ms/layer) +
  prefix-compaction (~1.1) + route-plan shadow (~0.45), (3) only then
  re-open the MTP gate at S<=8.
- **Binary-default soak recommendation**: do NOT flip any default on
  today's evidence. Before any flip: an unattended soak of >= 50
  reference runs alternating join/edges/full-barrier with pod telemetry
  (clocks/thermals), because the event rate on IDENTICAL configs moved
  ~3-5x between s602's census day and today - rate instability itself is
  part of the hazard signature and invalidates small-n promotion gates.
- **Prefix-compaction scoping**: NOT measured - the budget went to the
  fix/gate cycle (34 reference runs + 2 builds this sprint). Explicit
  deferral to 604.

## Definition of Done

1. Edge table derived from code and archived BEFORE implementation;
   `edges` behind a flag, default unchanged - **done** (table above;
   esmoke + actl603 byte-identity gates).
2. Race gates with zero token events at every stage - **executed, FAIL
   recorded with full census evidence**: edges 1 token-run/6; the join
   incumbent 1 token-run/3; flicker rate vs the 0.17/run baseline:
   join 0.33, edges 0.83 ck-events/run (pacing-ordered).
3. Perf verdict with stage tables; promotion decision per gates -
   **done: NO PROMOTION** (gate matrix above; +6.4% < +15%; census FAIL).
4. Flicker hunt - **done at n=6**: full-barrier control 6/6 bit-exact;
   bisect localizes the hazard outside the rank-stream sync; one derived
   fix (bcast dense-WAR guard) implemented, gated, and falsified honestly.
5. Updated >=50/slot statement + soak recommendation - **done** (Phase E).
6. Report + follow-ups - this document; commits are the orchestrator's.

## Deviations (honest list)

1. actl603 (the very first join control) fired the first-ever token-level
   event on the zero-NCCL stack; two re-runs were inserted before Phase B
   to establish the incumbent rate (clean 2/2).
2. The Phase B "stop on token event" rule was followed by folding Phase D
   (the flicker hunt) INTO the bisect, since the join incumbent failed
   the same gate - the spec's edges-vs-join framing dissolved into a
   single rank<->dense hazard hunt.
3. DENSE_GUARD=2 was implemented but not census'd (falsified mechanism at
   =1; budget); recorded as a diagnostic knob, not evidence.
4. Phase E floors for the RETAINED default (join) are carried from s602
   (d1f/d8f, byte-identical binary path) rather than re-measured; the
   fresh floors were measured on edges (the candidate) instead.
5. Event rates on identical configs differ ~3-5x from s602's same-pod
   census (join: 0.17 ck-only/run then vs 0.33 ck + token now) - pod
   state (uptime 20h+, thermals) is an uncontrolled variable; logged as
   part of the 604 soak design, and it weakens ALL small-n census
   comparisons across days, including s602's promotion evidence.
6. No nsys window (one-job rule + budget; no-SYS holds by construction
   for all s602/s603 transports).
7. The d8ep/d8jp stage tables are profiler-on (relative structure only),
   per the s601/s602 method.
8. ~34 reference runs consumed vs ~30 planned; the gd/gj fix-gate cycle
   was the overage; no GPU-hygiene incidents, pod left up, no foreign
   processes observed at any preflight.

## Artifacts

- Pod `/workspace/s603-artifacts/`: COMMANDS.md, build1/2.log, phaseA.sh,
  phaseB.sh, phaseCE.sh (unused as-is; superseded by chain3), chain1-3.sh,
  run trees (esmoke, actl603{,b,c}, e-sb-{1..3}, ge-{1..6}, fb-{1..6},
  vb-{1..3}, vc-{1..3}, gd-{1..6}, gd-sb-{1..3}, gj-{1..3}, d8ep, d8jp,
  d1f603, d8f603), e-sb-compare.txt, gd-sb-compare.txt, ge-pairwise.txt.
- Laptop `logs/from-cluster/sprint603/` (56 files: .out summaries,
  compares, drivers, COMMANDS.md).
- Source changes (uncommitted, orchestrator review):
  engine/runtime_options.cuh (SYNC + per-point + DENSE_GUARD envs),
  engine/runtime_pack.cu (sync-point dispatcher, edge sync, dense-WAR
  guard, site wiring, init echo), tools/ds4-v100-run-tp-ep-appliance.sh
  (S602_SYNC plumbing), new tools/s603-run.sh,
  docs/sprints/SPRINT-603-REPORT.md.

## Follow-up seeds (for SPRINT-604 planning)

1. **Close the rank<->dense hazard** - the gating correctness item. The
   bracket: full barrier (rank+dense at 16 sites) is 6/6 clean at 2.1x;
   every rank-stream-only sync leaks. Bisect levers: per-site
   FULL_BARRIER mask (16 knobs), dense-edge variants at the AG sites
   (PRODUCER-side dense edges, not the consumer-side guard falsified
   here), the checksum-collection path, copy streams.
2. **Soak protocol** before any default/promotion decision (>= 50 runs,
   alternating configs, pod telemetry) - day-to-day rate instability is
   now documented across s602/s603.
3. Join-pool reclaim beyond edges: merged/fused collective sites (share
   one sync pair across adjacent sites; the hc mm+sumsq pair and the
   router triplet are natural fusions), join-free fold dataflow.
4. Prefix-compaction scoping (deferred again; ~1.1 ms/layer pool).
5. NVIDIA escalation package unchanged; note the zero-NCCL stack's
   residual events are OURS (no NCCL in graph), so the escalation story
   for the captured-NCCL race is unaffected.
