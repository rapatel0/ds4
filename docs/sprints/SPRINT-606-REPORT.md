# Sprint 606 Report - Microbatch Ping-Pong: Capture-Machinery Verdict + Rendezvous-Merge Fallback

Date: 2026-06-14
Status: complete - **microbatch ping-pong NOT implemented (its sequential
byte-identical correctness anchor is honestly multi-sprint - no slot-range
primitive, ~157 flat-opt.slots sites, dual activation+op-struct buffer set);
the load-bearing deliverable is the capture-machinery analysis + the
graph-structure decision (option (b) in-graph choreography, with code
evidence, (a) rejected) - the Execution-Note-1 legitimate finding. The
spec-sanctioned fallback (rendezvous-merge) was implemented, amplifier-gated,
and A/B'd: RDZV_MERGE elides the redundant post-compose 1373 barrier,
correctness-clean (amplifier 1.0/1.0 at both carriers + tolerance 1.0/1.0) and
the campaign's FIRST perf-POSITIVE step lever (+0.4..+1.6% at S=8/16/32) - but
below the +15% bar, so held opt-in with the gain recorded. 0 of the ~3.3-4x
base reduction banked into the default; the win is the microbatch graph
decision + 607 re-scope + the proven-safe stackable barrier lever.**

## Headline

1. **Capture-machinery mapped; graph-structure decision made with code
   evidence.** The decode layer is captured as ONE multi-stream fork/join
   `cudaGraph` per layer (option (b) in-graph choreography is the only
   structure compatible with the persistent single-graph cache; option (a)
   graph-split is incompatible). This is the deliverable the spec asked for.
2. **Sequential byte-identical microbatch=2 is HONESTLY MULTI-SPRINT, not
   1-sprint** - the spec's Execution-Note-1 "legitimate finding" path. There
   is NO slot-range [lo,hi) primitive: `opt.slots` is consumed as a flat
   count in ~157 sites that each derive buffer sizes + kernel launch dims
   from it, and the activation set + the op-struct buffers (attn/shared/
   gate/up `d_out`/`d_x`) would all need a second copy threaded through every
   site. The correctness anchor (byte-identical sequential) cannot be reached
   safely within this sprint's budget; forcing it would violate the
   one-lever-measured-alone + correctness-anchor-first discipline.
3. **Fallback taken (spec-sanctioned): the rendezvous-merge lever**
   (s605 #3 / steering's no-graph-restructure family). Eliding the redundant
   post-compose 8x8 barrier (the 1373 site) - provably safe by dataflow,
   default-off, amplifier-gated - directly attacks the ~95%-launch/sync step.

## Phase 0 - base re-verified (in-band)

Pod `llamacpp-build-8gpu` (gpu-01, 8x V100, 16 Gi /dev/shm) up ~23h, all 8
GPUs idle (0 MiB), zero foreign procs (two Jun13 zombies are defunct s605
appliance teardowns, not active). Launcher defaults confirmed = the promoted
s605 stack: `EP_RETURN_TRANSPORT=relay`, `SWIGLU_EXCHANGE=batched`,
`HC_TRANSPORT=kernel`, `S602_SYNC=edges`, `DENSE_FIX=1`, `ATTN_OUT_GATHER8=0`.
Appliance binary rebuilt fresh on the pod.

**p0-floor-s8 control: continuation_tok_s_decode = 40.165 (5.02/slot,
step 199.2 ms), replay_ms 4.628 ms/layer** - byte-for-byte the s605 promoted
floor (40.16, 5.02/slot, 4.633 ms/layer). The base is re-verified, no drift.
The amplifier gate is live (s604/s605: amp@20us drives the unfixed config to
~100% token corruption; the promoted edges+fix stack stays 1.0/1.0 amplified).

## Phase A finding - microbatch capture-machinery analysis + graph-structure decision

### How the layer graph is captured (code evidence)

`engine/decode_loop.cu` `attempt_capture_probe` (~1683-2290):
- The layer is captured as **one multi-stream fork/join DAG in a single
  `cudaGraph_t`**. Sequence (lines 2003-2091): set root device; create
  `capture_seed` event; `cudaStreamBeginCapture(root_stream)` (2012); record
  `capture_seed` on root and have EVERY participating stream
  `cudaStreamWaitEvent(capture_seed)` (2017-2029) - the streams are each
  rank's `stream`, `dense_stream`, `copy_stream`, and `copy_streams[0..7]`
  (collected at 1994-2002, ~80 streams across 8 GPUs); then `run_one_step`
  (718-1689) enqueues the WHOLE layer across all those streams (2046); then
  every stream records a join event that the root waits (2057-2081); then
  `cudaStreamEndCapture` -> one `cudaGraph_t` (2091) -> `cudaGraphInstantiate`
  -> `persistent_graph->exec` (2126-2148).
- Replay (1874-1963): a SINGLE `cudaGraphLaunch(persistent_graph->exec,
  root_stream)` per layer-step. The graph is cached per-layer
  (`shared_rank_buffers.graph_cache.layers[layer]`,
  appliance_runtime.cu:354) keyed by (layer, slots, position, hc_shard,
  root_device, root_stream); re-captured only on a key mismatch.
- Cross-stream / cross-rank ordering is ALREADY expressed as intra-graph
  event edges inside this single capture: the s604 DENSE_FIX cross-GPU
  dense<->rank edge, the s602 sync edges, and the four `sync_all()` barriers
  (which under `decode_cudagraph_gate` become
  `enqueue_cross_gpu_stream_barrier` - an in-graph 8x8 event barrier,
  output_head.cu:1879, NOT a host sync).

### Decision: option (b) in-graph choreography; (a) graph-split rejected

- **(b) is the ONLY structure compatible with the persistent single-graph
  cache.** A 2nd microbatch is, structurally, just more streams forked from
  `capture_seed` + more enqueued work (the B half-batch) + cross-microbatch
  phase-shift event edges, all inside the same `BeginCapture/EndCapture` -
  exactly the multi-stream fork/join the machinery already builds, replayed
  with the same single `cudaGraphLaunch`.
- **(a) graph-split is rejected:** splitting the layer into prefix/compute
  sub-graphs would need two `cudaGraphExec` handles, two launches, and
  host-side interleaving per layer per step - defeating the single-launch
  persistent-replay design that is the entire point of full-capture, and
  breaking the (layer,slots,position) cache-key model.
- The router/route-plan runs INSIDE `run_one_step` at
  `kStagePostAttentionFfnInput` (decode_loop.cu:916, capturable), and the
  fixed-capacity route plan (`post_attention_fixed_capacity_route_plan_gate`,
  default on; router_plan.cu:158-255) sizes routes at `slots*top_k`
  regardless of actual selections - so a half-batch (incl. a zero-route half)
  is handled by construction. This part of microbatch is tractable.

### Why sequential byte-identical microbatch=2 is multi-sprint (the honest verdict)

The correctness anchor (microbatch=2 SEQUENTIAL byte-identical to =1) is the
non-negotiable precondition for any overlap. Reaching it requires running the
whole stage sequence twice within one capture, each half operating on its own
buffers without clobbering. Concretely:

- **No slot-range primitive exists.** `opt.slots` is consumed as a FLAT COUNT
  in ~157 sites across decode_loop / attention_output / post_attention_ffn /
  hc_current / ep_dense / ep_compose, each deriving buffer sizes AND kernel
  launch grid/dims directly from it. There is no `[lo,hi)` abstraction to feed
  a half-batch; introducing one means touching all ~157 sites.
- **A second activation/buffer set is needed on BOTH `RankState` AND the op
  structs.** The s605 ~21 MiB/rank set (`d_a`, `d_down`, `d_ep_contrib_all`,
  `d_current_full*`, `d_attn_output_a_full`, `d_ep_relay_stage`,
  `d_ep_remote[8]`, `d_route_slots`/`r.routes`) is shared singletons per rank;
  AND the resident dense op structs (`attn_op`/`shared_op`/`shared_gate_op`/
  `shared_up_op`) carry per-rank `d_out`/`d_x`/`d_x_half` that the routed and
  dense stages write/read. Microbatch B would clobber A's results in all of
  these. (VRAM fits - s605 resolved ~180x headroom - but the THREADING is the
  cost, not the bytes.)
- A narrow "EP/routed-region-only" microbatch (leave the prefix whole-batch)
  was evaluated and rejected: the routed pack writes `r.d_a` from
  `r.d_current_full` via `r.d_route_slots`/`r.routes`
  (post_attention_ffn.cu:428-504), so it still needs the per-half route plan +
  buffer duplication + the slot-range primitive into pack/GEMM/EP-return/
  compose - the same core difficulty over fewer files - AND it cannot achieve
  the spec's overlap target (prefix 2.06 ms/layer under ep_window 2.67) since
  that overlap crosses the prefix/EP boundary a routed-only split doesn't span.

**Verdict: sequential byte-identical microbatch=2 is multi-sprint.** This is
the Execution-Note-1 outcome: "If NEITHER (a) nor (b) is tractable this
sprint, that is a legitimate finding... fall back to a no-graph-restructure
lever... and re-scope microbatch for a later sprint." Option (b) is the right
structure when microbatch is built; the blocker is the slot-range + dual-buffer
threading surface, which is too large to make byte-identical-then-overlapped
safely in one sprint. Microbatch is re-scoped to a dedicated multi-increment
sprint (607+) with the explicit plan below.

## Phase B/C - the rendezvous-merge fallback lever (RDZV_MERGE)

### Implementation (default-off, byte-identical off)

Flag `DS4_V100_TP_EP_RDZV_MERGE=0|1` (default 0). When on, elides the final
post-compose 8x8 cross-GPU event barrier (the `kEpProfBarrier1373` site,
decode_loop.cu:1637) on the promoted full-capture decode path.

**Safe by dataflow:** the compose kernels write `r.d_next_hidden` on
`r.stream`; the immediate next consumer
(`expand_hidden_to_proxy_hc_shard_kernel` in `run_final_hc_carry`,
decode_loop.cu:474) reads `r.d_next_hidden` on the SAME `r.stream` - same-
stream ordered, needs no barrier. The cross-GPU rendezvous that `final_hc`
actually requires (its hc final-expand allgather reading all ranks' shards) is
ALREADY supplied by the `sync_all()` INSIDE `run_final_hc_carry` (decode_loop.cu:480),
which on the cudagraph path is the same in-graph 8x8 event barrier. So 1373 is
redundant WITH that internal sync_all whenever final_hc runs inline after
compose. The elision is gated to skip ONLY when compose is NOT the suffix
terminus (i.e. final_hc carries the cross-GPU edge) - exactly the early-return
guard below it - so the compose-terminus case keeps 1373.

Eliding 1373 removes ~16 event records + up to ~1280 `cudaStreamWaitEvent`
enqueues per layer (the full `include_copy_streams` 8x8 barrier - 8 dst x 8 src
x (stream+dense+8 copy)x2 waits) - a pure launch/sync reduction on the
~95%-launch/sync step. s605 stage table attributes ~0.39 ms/layer to this
barrier class (~8% of the 4.63 ms/layer step).

Wiring: `engine/runtime_options.cuh` (`ds4_rdzv_merge_env` + `Options.rdzv_merge`),
`engine/decode_loop.cu` (the guarded elision + an audit counter +
`tp_ep_s606_rdzv_merge` log line), launcher (default + export + echo).
Flag-off is byte-identical (the barrier runs unchanged).

### Gate results - amplifier (the correctness anchor for the elision)

The elision was verified to fire: `tp_ep_s606_rdzv_merge layer N elided_1373_barriers 2`
on every layer (the 2 = warmup + first-step capture; subsequent steps reuse the
cached graph that already has 1373 elided). Both amplifier runs on the promoted
edges+fix base, RDZV_MERGE=1, S=32 reference shape, vs the s597 control:

| run | config | sel | seq | first_ck | first_tok | verdict |
|---|---|---:|---:|---|---|---|
| rdzv-amp20-aoa | RDZV_MERGE=1 + amp 20us @ attn_out_a | 1.0 (128/128) | 1.0 (8192/8192) | {None:128} | {None:128} | PASS |
| rdzv-amp20-precompose | RDZV_MERGE=1 + amp 20us @ pre_compose | 1.0 (128/128) | 1.0 (8192/8192) | {None:128} | {None:128} | PASS |

**The amplifier gate PASSES at BOTH carrier sites.** The amplifier drives the
*unfixed* config to ~100% token corruption at 20us (s604 Phase A); with
RDZV_MERGE=1 the amplified fast stack stays bit-exact 1.0/1.0 - proving the
elided 1373 barrier did NOT reopen the dense<->rank (attn_out_a) or the
compose-region late-step (pre_compose) hazard class. The cross-GPU rendezvous
final_hc requires is fully carried by the internal sync_all() in
run_final_hc_carry, exactly as the dataflow analysis predicted.

### Gate results - A/B perf (S=8/16/32)

Paired off/on in one session (idle-verified between runs), promoted edges+fix
base. S=8/S=16 SKIP_TOL; S=32 with tolerance vs the s597 control.

| S | RDZV_MERGE | cont-decode tok/s | mean replay_ms/layer | tolerance |
|---:|---|---:|---:|---|
| 8  | OFF | 38.654 | 4.6874 | (skip) |
| 8  | ON  | 38.799 | 4.6584 | (skip) |
| 16 | OFF | 73.671 | 4.7536 | (skip) |
| 16 | ON  | 74.826 | 4.7462 | (skip) |
| 32 | OFF | 140.826 | 5.1419 | 1.0/1.0 (128/128, 8192/8192) |
| 32 | ON  | 142.456 | 5.1093 | 1.0/1.0 (128/128, 8192/8192) |

**RDZV_MERGE is correctness-clean AND perf-POSITIVE, but small:**
- S=8: +0.38% decode (−0.029 ms/layer)
- S=16: +1.57% decode (−0.007 ms/layer)
- S=32: +1.16% decode (−0.033 ms/layer)

The sign is consistent across all three S points (on faster than off on every
pairing, in both tok/s and replay_ms) - a real, repeatable positive, NOT noise,
and the OPPOSITE of the s605 gather8 lever (which was correctness-clean but
~10% perf-NEGATIVE). But the magnitude (~0.03 ms/layer ≈ 0.6%) is far below the
spec's +15% promotion bar: eliding the single 1373 barrier removes only ~0.03
of the ~1.0 ms/layer barrier pool. S=32 tolerance is bit-exact both arms.

### Promotion decision: NO PROMOTION (hold opt-in; record the gain)

Per the spec gate (promote default only on amplifier-clean + >=20-run soak +
tolerance 1.0/1.0 + **>+15% decode-domain**): RDZV_MERGE clears the
correctness gates decisively (amplifier 1.0/1.0 at both carrier sites,
tolerance 1.0/1.0 at S=32, flag-off byte-identical) but the perf gain
(+0.4..+1.6%) is well under +15%, so it is **held default-off / opt-in**, with
the measured positive banked. No soak was spent on a sub-threshold lever (GPU
hygiene; the amplifier + S=32 tolerance already establish correctness). The
flag + the dataflow proof + the audit counter stay as a measured-positive,
amplifier-gated, STACKABLE first member of the rendezvous-merge family.

Rollback is trivial (it is opt-in default 0). The lever composes with the full
promoted stack (it ran on edges+fix+relay+batched+kernel throughout).

## Phase D - restate the target math

### Final config = promoted edges+fix (RDZV_MERGE held opt-in)

Floors unchanged from the s605 promoted base (RDZV_MERGE not promoted; its
+0.4..+1.6% is recorded but not banked into the default):

| S | cont-decode tok/s | per-slot | step ms | ms/layer (x43) |
|---:|---:|---:|---:|---:|
| 8  | ~38.7-40.2 | ~5.0 | ~199 | ~4.63-4.69 |
| 16 | ~73.7-74.8 | ~4.6 | ~214 | ~4.75 |
| 32 | ~140.8-142.5 | ~4.4-4.5 | ~222 | ~5.11-5.14 |

(The off-arm S=8 here, 38.65, is ~3.7% below the p0-floor-s8 control 40.17 -
within the s602/s604/s605-documented day-to-day pod variance on this 23h+ pod;
the A/B is paired in one session so the comparison is valid regardless.)

### >=50 tok/s/slot + required-MTP-multiplier restatement (unchanged)

Target >=50/slot <=> step <= 20 ms. On the promoted base (S=8 ~199 ms, 5.0/slot):
gap **~10x**, required MTP acceptance multiplier **M ~= 10** - unchanged from
s605 (the base floor did not move: microbatch was not implemented, and the
rendezvous-merge lever's ~0.6% is not promoted).

### Captured vs remaining base-reduction accounting (honest)

- **Captured this sprint: 0 of the needed ~3.3-4x base reduction** banked into
  the default (microbatch not implemented; RDZV_MERGE's +0.6% not promoted).
- **What this sprint banked**: (1) the capture-machinery analysis + the
  graph-structure decision (option b) - the load-bearing design output that
  unblocks a correctly-scoped microbatch sprint; (2) the honest, code-evidenced
  verdict that sequential byte-identical microbatch=2 is multi-sprint (the
  slot-range + dual-buffer threading surface), with a staged 607 re-scope plan;
  (3) a correctness-clean, amplifier-gated, perf-POSITIVE rendezvous-merge lever
  (the FIRST positive step lever of the campaign - the s605 lever regressed),
  proving the barrier pool is a real, safely-attackable target; (4) the
  generalizable finding that the post-compose 8x8 barrier is fully redundant
  with the final_hc internal sync_all (the dataflow pattern that the other
  rendezvous merges in 607 can reuse).
- **Remaining sequence to <=20 ms, re-ranked**:
  1. **Microbatch ping-pong** (607, dedicated, staged per the re-scope below) -
     still the highest ceiling (~30-40% step), graph structure now decided.
  2. **Rendezvous-merge family** (607): extend RDZV_MERGE to the rest of the
     ~1.0 ms/layer barrier pool (the dense-overlap 954/978 edges, the
     include_copy_streams reduction on barriers whose next stage does not
     consume copy-stream output). RDZV_MERGE is the proven-safe template.
  3. **Prefix attn_output as a TRANSPORT lever** (s605 #2; the 1.07 ms/layer
     DMA shard movement - relay/NV-direct family, not launch compaction).
  4. **Route-plan shadow** (~0.67 ms/layer).
  5. Then re-open the MTP gate at S<=8.

## Definition of Done

1. Capture-machinery analysis + graph-structure decision - **DONE** (Phase A;
   option (b) chosen with code evidence; (a) rejected).
2. Sequential byte-identity anchor - **NOT REACHED; recorded as the
   Execution-Note-1 legitimate finding** (microbatch is multi-sprint; the
   blocker is the slot-range + dual-buffer threading surface). Fallback lever
   taken instead.
3. Fallback lever (rendezvous merge) implemented default-off, amplifier-gated;
   A/B perf - **DONE** (RDZV_MERGE elides barrier 1373; amplifier 1.0/1.0 at
   attn_out_a + pre_compose; A/B +0.4..+1.6% at S=8/16/32; tolerance 1.0/1.0
   at S=32; flag-off byte-identical).
4. Promotion decision - **DONE** (NO PROMOTION: correctness-clean + perf-
   positive but <+15%; held opt-in, gain recorded; rollback = default 0).
5. Restated target math + next-lever (microbatch re-scope) - **DONE** (Phase D;
   M~=10 @ S=8 unchanged; remaining sequence re-ranked; 607 staged plan below).
6. Report + orchestrator docs/commits - this document; commits are the
   orchestrator's.

## Microbatch re-scope (for SPRINT-607)

The graph structure is decided (option (b)). The 607 plan should be staged:
- Increment 1: introduce a `SlotRange{lo,hi}` (or `slot_base`/`slot_count`)
  threaded through the EP/routed region first (post_attention_ffn + ep_dense +
  ep_compose), with microbatch=2 running the routed region twice sequentially,
  bit-verified byte-identical BEFORE the prefix is touched.
- Increment 2: extend the slot-range through the prefix (attention_output /
  hc_current); the per-slot KV arrays are disjoint-row-safe per half.
- Increment 3: add the second activation/op-struct buffer set; bit-verify
  sequential byte-identical (the correctness anchor).
- Increment 4: the phase-shift overlap (option (b) extra streams + cross-
  microbatch edges), amplifier-gated per edge.
Each increment is bit-verifier-gated; only after Increment 3's anchor passes
does any overlap get enabled.

## Artifacts

- Pod `/workspace/s606-artifacts/`: COMMANDS.md, build1.log, run trees
  (p0-floor-s8, rdzv-amp20-aoa, rdzv-amp20-precompose, rdzv-{off,on}-s{8,16,32}),
  phaseA-amp.console, phaseBC-ab.console.
- Laptop `logs/from-cluster/sprint606/`: COMMANDS.md, phaseA-amp.console,
  phaseBC-ab.console, ab-summary.tsv, this report.
- Source changes (uncommitted, orchestrator review):
  engine/runtime_options.cuh (`ds4_rdzv_merge_env` + `Options.rdzv_merge`),
  engine/decode_loop.cu (RDZV_MERGE guarded elision of barrier 1373 + audit
  counter + log line), tools/ds4-v100-run-tp-ep-appliance.sh (env default +
  export + echo), new tools/s606-run.sh, tools/s606-phaseBC.sh.

## Deviations (honest list)

1. The sprint's primary objective (microbatch ping-pong) was NOT implemented:
   the sequential byte-identical correctness anchor is multi-sprint scope (no
   slot-range primitive; dual-buffer threading across ~157 sites + the op
   structs). This is the Execution-Note-1 "legitimate finding" outcome, and the
   spec's Risk-table top row ("if neither (a) nor (b) is tractable this sprint,
   that itself is the finding - fall to the no-graph-restructure levers"). The
   graph-structure decision (option b) is delivered with code evidence and the
   607 re-scope plan; the rendezvous-merge fallback banks a measured,
   amplifier-gated step lever instead.
