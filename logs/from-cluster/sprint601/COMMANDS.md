# Sprint 601 command log (pod llamacpp-build-8gpu, gpu-01) - 2026-06-12

## Setup
- Pod recreated with 16 Gi /dev/shm (verified: tmpfs 16G). /workspace persisted.
- Re-provisioned: apt cmake git python3 curl numactl pciutils + cuda-nsight-systems-12-2
  (provision.log).
- Laptop tree (HEAD 1ba7b528 + s601 comm-split edits) synced to /workspace/ds4
  (tar pipe, s597 convention). turbomind lib reused from persisted build dir.
- Build: CUDA_ARCH=sm_70 make -j72 appliance/ds4-v100-tp-ep-appliance (build1.log, clean).
- New code (default-off, flag-off byte-identical):
  DS4_V100_TP_EP_COMM_SPLIT=none|epret|hc|epret+hc
    epret: 8 EP-return broadcasts + swiglu nccl allgather + compose reduce-scatter
           on a dedicated comm (default buffsize)
    hc:    hc allreduces/allgather, router allreduce, full-current broadcast,
           post-attn allgather, attn-output allgather on a dedicated comm (1 MiB buffsize)
  Init prints tp_ep_s601_comm_split lines with measured min-free-VRAM cost per comm.
  Launcher plumbing for COMM_SPLIT + HEAD_COMM (defaults none/shared).
- Runner: /workspace/ds4/tools/s601-run.sh <name> (idle+foreign preflight, reference
  shape, slot-indexed tolerance + first-divergence vs phase0-full-control;
  CONTROL_DIR overridable for run-to-run comparisons).

## Phase 0
- s601-twocomm-test (rebuild of s600-twocomm-test.cu): A rc=0, B rc=0 -- TWO 8-rank
  comms now coexist; 33 MiB /dev/shm per comm, 16 Gi available. UNBLOCKED (phase0-twocomm.log).

## Phase 0 (continued)
- rctl601 (flags off, identity + in-band control): decode-domain 168.15 / wall 114.75
  (band vs s600 170.18/117.08, -1.2%); slot-indexed tolerance vs phase0-full-control:
  selected 1.0 sequence 1.0 PASS; nodes 2825/layer (= s600 copy graph).
  NOTE: batch 1 (32 slots) checksum-only divergence at step 2, zero token changes --
  the SAME ~1/256 promoted-path latent event signature as s600 rctl600 (on the record).
- smoke-split (COMM_SPLIT=epret+hc, 2 tok): both class comms init PASS;
  measured VRAM cost 490 MiB/GPU per comm (epret default buffsize, hc 1 MiB;
  NCCL_RINGS single-ring policy keeps it below the s598 ~0.9 GiB estimate);
  capture+replay OK (nodes 2833/layer, +8 vs shared-comm graph); tokens match the
  s597 probe smoke [48177,3263].

## Phase A - run-to-run divergence gate under NCCL_PROTO=Simple + SWIGLU_EXCHANGE=batched
- a0-sb-1 vs a0-sb-2 (no candidate, reproducer sanity): ALL 128 pairs diverge
  (96 at step 0, 32 at step 2), selected 0.023 / sequence 0.087 -> the s600
  reproducer fires on this binary/pod exactly as documented. Detector validated.
- a1-sb-1 vs a1-sb-2 (HEAD_COMM=dedicated): FAIL - identical signature
  (selected 0.023 / sequence 0.072; 96 pairs @ step 0, 32 @ step 2). The s600
  dedicated head comm now initializes (16 Gi shm) but does NOT kill the race.
- a3-sb-{1,2,3} (COMM_SPLIT=epret+hc - maximal class isolation: EP-return
  broadcasts on comm2, all other captured collectives on comm3, compose comm
  reduced to eager-head-only): FAIL - all 3 pairwise comparisons diverge with
  the identical signature (0.016-0.031 selected agreement; 96 @ 0, 32 @ 2).
  phaseA-stage2-compare.txt.
- a2 (COMM_SPLIT=epret alone) NOT run: strictly weaker isolation than a3;
  a3's verdict subsumes it. phaseA-stage3.sh retained unexecuted.
- PHASE A VERDICT: communicator isolation does not touch the race at any
  granularity (eager-head / EP-return class / full class split). The race is
  per-collective / NCCL-internal, exactly the spec's "intra-comm" risk row.
  Pivot to Phase B (NCCL-free EP window).

## Phase B - NCCL-free EP window (relay peer-write EP return + batched exchange)
- New transport DS4_V100_TP_EP_EP_RETURN_TRANSPORT=relay (default unchanged):
  src-side peer-WRITE copy kernels over NVLink; the 12 SYS pairs forward
  one-hop via a staging buffer on relay GPU dst^4 (NVLink-adjacent to both
  ends per the s597 relay table; each GPU relays exactly 3 directed pairs -
  the balanced schedule, expressible as relay = dst XOR 4).
  Stage W (src streams) -> 8x8 event barrier -> stage F (relay streams) ->
  8x8 event barrier -> compose. Byte moves only; graph-capturable.
  Staging: +(slots*top_k*896*4*8) bytes/GPU (~5.5 MiB at reference shape).
- Build2: rebuild with relay transport (build2.log).
- Launcher check with relay transport: config ok (build2 binary).
- b1-relay-ctl (relay return + copy exchange, LL): decode-domain 186.58 /
  wall 125.82 (+10.8% over the s600 170.18 baseline with the copy storm still
  in place); slot-indexed tolerance vs control: selected 1.0 / sequence 1.0
  PASS (transport byte-correct); capture nodes 2721/layer (vs 2825 nccl);
  replay ~3.59 ms/layer observed at startup. ONE checksum-only event
  (batch 2, step 33, zero token changes) - the latent race still fires with
  the 8 hc-class collectives captured, exactly the s600 prediction for the
  tightened-pacing experiment.
- b-sb-{1,2,3} (NCCL_PROTO=Simple + batched + relay): pairwise gate pending.
- b-sb-{1,2,3} pairwise (phaseB-compare.txt): FAIL - all 3 comparisons diverge
  with the IDENTICAL signature (96 pairs @ step 0, 32 @ step 2; selected
  agreement 0.016-0.047). Removing all 9 EP-window collectives does not move
  the divergence locus by one step.
- PHASE B VERDICT (race): the captured-NCCL race does NOT live in the EP
  window. It survives in the remaining hc-class set (hc 3 allreduces +
  allgather, router allreduce, full-current broadcast, post-attn allgather)
  - the s600 follow-up-seed-2 experiment, now answered: NCCL-free EP window
  alone is insufficient; a full NCCL-free decode graph (kernel reductions for
  hc/router) is the remaining path, out of s601 scope/budget. Note the s600
  delay-bisect already showed the masking site is the EP window's DURATION,
  not its collectives - consistent.
- Useful fact: under Simple, EVERY config's first event is batch 1 step 2 -
  the same locus as the promoted path's rare LL event (rctl600/rctl601 batch 1
  step 2; b1-relay-ctl batch 2 step 33).
- Phase B transport itself: CORRECT and FAST (b1 above); retained opt-in
  (DS4_V100_TP_EP_EP_RETURN_TRANSPORT=relay), default unchanged.

## Phase C - non-promotion + demonstrated-ceiling evidence
- Promotion gates: race-zero FAILED (above) -> no launcher default flips.
  nccl return + copy exchange + shared comm remain the promoted path.
- c1-relay-batched / c2-relay-batched-prof: demonstrated ceiling of the
  NCCL-free EP window under LL (evidence-only; tolerance recorded).
- c1-relay-batched (relay + batched, LL): decode-domain 208.18 / wall 135.27 -
  the demonstrated ceiling of the NCCL-free EP window (+23.8% over 168.15).
  Tolerance FAIL (0.9375 selected / 0.8928 sequence; checksum onsets steps
  2+13 in all batches) - the batched exchange still trips token-level
  corruption (s599 C-A5 behavior), unchanged by the relay return. batched
  stays opt-in/unpromoted.
- c2-relay-batched-prof (profiler on): 179.16 decode-domain (profiler
  overhead is large at this speed); stage table per rank/layer:
  ep_return_relay 0.24-0.26 ms (vs 0.61 nccl), shared_swiglu_down 0.41-0.50
  (batched), route_plan_pack 0.40-0.44, prefix_attn_output 0.76-0.89,
  final_hc 0.43, ep_window ~1.72-1.87, barriers 954/978/1144/1373 ~0.02-0.24.
- Phase C promotion candidate: relay + COPY exchange (b1: 186.58, 1.0/1.0).
  Gates: >=3 paired runs at promoted config (b1 + g2-relay-copy-{2,3}),
  event census vs the promoted path's own exposure (rctl601: 1 checksum-only
  event/256; rctl600: 1; rctl600b: 0), perf >= 181.6 (+8% over rctl601
  168.15), no-SYS nsys spot-check (s601-nsys.sh, relay+copy).
- Event census, LL regime, this binary (events = batches whose checksums
  diverge from the s597 control; "TOKEN" = token flips at onset):
    nccl+copy (promoted): rctl601: 1 (ck-only, b1 s2); rctl601b: 1 (TOKEN,
      8 slots b1 s2, seq 0.9395). [s600 binary: rctl600 1 ck-only; rctl600b 0]
    relay+copy: b1 1 (ck-only s33); g2-2 1 (ck-only s35); g2-3 1 (TOKEN 4
      slots s2); g2-4 2 (TOKEN s2 + TOKEN s32); g2-5 2 (s2 ck + TOKEN s27);
      g2-6 2 (TOKEN s3 + ck s23).
  => relay+copy: 9 events / 6 runs (~1.5/run), 5 token-flipping.
     promoted:   2 events / 2 runs (1.0/run), 1 token-flipping.
- PHASE C VERDICT: NO PROMOTION. relay+copy trends ~1.5x the promoted event
  rate (mechanism-consistent: the faster EP window tightens intra-replay
  pacing, raising exposure - the s600 rate-vs-spacing curve, reconfirmed).
  The "not worse" gate cannot be documented; tolerance 1.0/1.0 failed in 4/6
  candidate runs. Launcher defaults unchanged (nccl return, copy exchange,
  shared comm, early-return off). relay + batched + COMM_SPLIT all retained
  opt-in with documented rollback (unset env / =nccl / =copy / =none).
- ESCALATION FACT (new): the PROMOTED path itself fired a token-level event
  (rctl601b) - first token-flipping event ever recorded on the promoted
  config (s600 had checksum-only). The hc-collective race is now a live
  token-correctness debt at ~1 event/256 steps/batch-window, not just a
  checksum anomaly. The fully-NCCL-free decode graph (kernel reductions for
  the hc/router set, ring-order-exact for bit anchoring) is the lead item
  for the next sprint.
- no-SYS nsys spot-check for relay: MOOT with non-promotion (gate fails
  earlier); the relay path avoids SYS by construction (NV-direct + dst^4
  one-hop staging only) and the s598 no-SYS proof for the promoted NCCL
  return stands unchanged.
- C-B restack: NOT attempted - the relay return embeds two 8x8 event
  barriers; hoisting it into the dense-overlap window would fence the dense
  streams mid-overlap (the barrier waits on dense streams), structurally
  anti-synergistic; s599 already measured the restack neutral-to-negative
  with a barrier-free return. With non-promotion it is moot for defaults.
- C-C route-plan shadow: NOT attempted (budget went to the race census);
  pool re-measured at 0.40-0.44 ms/layer (c2 stage table).

## Phase D - slot-scaling curve (promoted stack, flags off)
- d32/d16/d8/d4/d1 + d8p/d1p (profiler) via phaseD.sh; wave size = slots
  (harness change); REQUESTS = 4 waves (8 waves at S=1).
- Phase D incident: S=4 and S=1 points failed on build2 - the fixture KV slot
  (Options.kv_slot default 7) is outside the configured slot count for S<8
  (tp_runtime_dense_kv_slice "slot is outside configured slot count"; HTTP
  500s). The failed d4 server lingered after its bench (held GPUs), which
  also aborted d1/d8p preflights - killed by pid; idle re-verified.
  Fix (engine, default-behavior-preserving for S>=8): clamp the fixture slot
  to slots-1 in layer_runner.cu + layer_decode.cu (build3). d4/d1/d8p/d1p
  re-run on build3; d32/d16/d8 stand from build2 (clamp is a no-op at S>=8).
- d32 (3rd promoted-path control sample): decode 169.01 / wall 114.88, and
  ANOTHER token-level event (4 slots @ b1 s2, seq 0.9697). Promoted-path
  census today: 3 runs -> 3 events, 2 token-flipping. Escalation confirmed.

## Phase D results (final)
- Scaling curve (promoted stack; S>=8 build2, S<8 build3 with the fixture
  KV-slot clamp; wave=S; 4 waves per point, 8 at S=1):
    S=1:  decode  8.11 / wall   6.13 / per-slot 8.11 / step 123.3 ms
    S=4:  decode 30.68 / wall  22.67 / per-slot 7.67 / step 130.4 ms
    S=8:  decode 63.64 / wall  45.23 / per-slot 7.96 / step 125.7 ms
    S=16: decode 111.96 / wall 77.65 / per-slot 7.00 / step 142.9 ms
    S=32: decode 169.01 / wall 114.88 / per-slot 5.28 / step 189.3 ms
  Profiler companions: d1p 6.44 (step 155.3), d8p 50.43 (step 158.6).
- Stage tables S=1 vs S=8 (stage-summary.py over the steady tail): only
  shared_swiglu_down (0.051 -> 0.241) and the routed GEMMs scale with slots;
  ep_return/route_plan/prefix stages are slot-flat latency. The step is a
  launch/wait floor (~2.9 ms/layer), slot-flat to S=8.
- >=50/slot: requires MTP acceptance multiplier 6.3 @ S=8 / 7.1 @ S=16 on
  the measured base (10.1/11.4 for the 80 ideal): not reachable by MTP
  block-2 alone; base step must fall ~2-3x further first.
- Final 32-slot promoted numbers this binary: 169.01/114.88 (d32), band
  167.5-169.0 across 3 controls.
- Report: docs/sprints/SPRINT-601-REPORT.md (laptop); summaries under
  logs/from-cluster/sprint601/.
