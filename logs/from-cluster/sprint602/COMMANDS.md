# Sprint 602 command log (pod llamacpp-build-8gpu, gpu-01) - 2026-06-12

## Setup
- Inherited partial Phase A from a prior agent (laptop working tree at HEAD d11e0f3f +
  uncommitted s602 edits): DS4_V100_TP_EP_HC_TRANSPORT=nccl|kernel + 9-class kernel
  collective set in engine/runtime_pack.cu (ring-order-exact folds, dst^4 relay byte
  moves), call sites in hc_current/router_step/post_attention_ffn/runtime_resources,
  verifier (DS4_V100_TP_EP_S602_VERIFY) + per-class mask (DS4_V100_TP_EP_S602_KERNEL_MASK),
  fold calibration envs (S602_RING/FOLD_DELTA/MIN_CHUNK/NCHANNELS), tools/s602-fold-probe.cu.
- Audit completed on laptop; gaps fixed: s602_state_init call (ensure_compose_buffers),
  s602_collect_verify at both replay-collect sites (decode_loop.cu), launcher plumbing
  for HC_TRANSPORT, tools/s602-run.sh.
- attention_output.cu allgather audited: gated by true_ds4_attention_output_nccl_allgather_gate,
  default FALSE with no setter anywhere -> dead code in the reference config (active path is
  peer cudaMemcpy2DAsync). Same for the compose reduce-scatters (nccl_reduce_scatter_compose_gate
  default false). No conversion needed.
- Laptop tree synced (tar pipe, s597 convention).

## Phase A0 - fold calibration probe
- Build: nvcc -O3 -arch=sm_70 --std=c++17 -o s602-fold-probe tools/s602-fold-probe.cu -lnccl
  (system /usr/include/nccl.h, NCCL 2.19.3).
- Actual auto rings under NCCL_P2P_LEVEL=NVL (12 channels, NCCL_DEBUG=INFO):
  ch{0,1,6,7}: 0 3 2 1 5 6 7 4 | ch{2,3,8,9}: 0 4 7 6 5 1 2 3 |
  ch{4,10}: 0 1 3 7 5 4 6 2 | ch{5,11}: 0 2 6 4 5 7 3 1.
  NOTE: differs from the s597 DS4_V100_NCCL_NO_SYS_RING "0 3 2 1 5 7 6 4" -- that env
  is only exported when ALLOW_VISIBLE_REMAP=1 (not the reference config); the engine
  default ring spec must be "0 3 2 1 5 6 7 4" (channel-0 auto ring).
- fold-probe-run1.log: default ring spec -> NO-MATCH everywhere (wrong ring; discarded).
- fold-probe-run2.log (--trials 6, true 12-ring list): per-shape verdicts:
  - single-chunk regime (count <= effective chunk): MATCH with nc=1 delta=1 rbase=0,
    fold = left fold along ch0 ring starting position 1 (rank 3). Covers
    hc_sumsq/r_sumsq at all slot counts (count<=32), hc_mix at slots<=8.
  - r_logits @ slots=32 (8192): MATCH (multi-chunk, chunks of 1024, start=c+1; 56 hyps
    incl. nc=1 minChunk in {16..512} equivalents).
  - hc_mix @ slots>=16 (384/576/768) and r_logits @ slots<=4 (256-1024): NO-MATCH in the
    entire (nc, pow2 minChunk, delta, rbase) ring-fold space -> NCCL chunking in that
    regime is outside the probe model (grouped-op aggregation / protocol switch suspected).
- Decision: the in-engine S602_VERIFY bit-verifier is ground truth; calibrate via env
  sweep in verify mode at the reference shape; fallback policy (deterministic order +
  re-anchored control) reserved for classes that cannot be matched.

## Phase A1 - flag-off byte-identity control (build1)
- rctl602 (HC_TRANSPORT default nccl, all flags off): bench rc=0,
  decode-domain 164.80 / wall 111.42 (vs s601 band 167.5-169.0 / 114.1-114.9, -1.9%);
  slot-indexed tolerance vs phase0-full-control: selected 1.0 sequence 1.0 PASS,
  first_ck/first_tok histograms all None (zero latent events this run).
  -> flag-off path byte-identical, gate PASS.
- build2: ring-spec default fixed to the measured ch0 auto ring "0 3 2 1 5 6 7 4"
  (runtime_options.cuh) + probe hypothesis grid extended with multiples-of-30
  minChunks (LL128 packs 30 floats per 128B line).

## Phase A0b - extended probe (run3) + auto min-chunk rule
- fold-probe-run3.log (extended grid incl. multiples-of-30 + 192/96/...):
  - hc_mix: GLOBAL MATCH (32 hyps) at nc=1 delta=1 ch0-ring, chunks of 192 at
    counts 384/576/768, single chunk below.
  - hc_sumsq / r_sumsq: global match (single chunk, start = ring pos 1).
  - r_logits: per-shape matches but no single global mc: 256..1024 -> 192,
    2048 -> 256, 4096 -> 512, 6144/8192 -> 1024.
  - Unifying rule (all shapes): NCCL LL picks nthreads = clamp(pow2ceil(bytes/64), 96, 512),
    minChunk = nthreads*2 floats. delta=1, nc=1, ch0 ring everywhere.
- Engine: DS4_V100_TP_EP_S602_MIN_CHUNK default now 0 = auto size rule
  (s602_min_chunk_for in runtime_pack.cu); env still overrides globally. build3.

## Phase A2 - in-engine bit-verifier (build3)
- averify1 (HC_TRANSPORT=kernel + S602_VERIFY=1, ref shape, 32 req x 8 tok):
  tp_ep_s602_init transport kernel mask 0xff verify 1 rings 1 delta 1 min_chunk 0(auto)
  nchannels 1 stage_kib 284 PASS; **ZERO tp_ep_s602_verify_mismatch lines** across
  all 9 classes x 8 ranks x 43 layers x 8 steps -- kernel transport bit-identical
  to live captured NCCL in-situ (ring-order-exact folds + byte moves all PASS).
- a2-kernel-tol: HC_TRANSPORT=kernel (NCCL skipped for the 9 classes), full
  reference shape, tolerance vs s597 control: pending.

## Phase A3 - kernel-mode full tolerance (build3) + pairwise-sync optimization
- a2-kernel-tol (HC_TRANSPORT=kernel, ref shape 128x64): bench rc=0,
  **tolerance vs s597 control: selected 1.0 / sequence 1.0 -- BIT-EXACT** (8192/8192),
  zero events. THE BIT ANCHOR SURVIVES the NCCL-free hc transport (ring-order-exact
  + auto min-chunk rule). Perf: decode-domain 67.82 / wall 57.71 -- replay ~10.9 ms/layer
  vs 4.13 control: the full 8x8+dense barriers (16/layer) destroy rank<->dense overlap.
  VRAM max 29616 MiB (control 30338).
- Optimization (build4): pairwise event ordering replaces the full barriers at the
  s602 sites: B0 = each rank waits its 4 NVLink peers; B1 = each rank waits its
  mirror (g^4). NCCL-equivalent ordering (rank streams only; dense streams untouched).
  Escape hatch DS4_V100_TP_EP_S602_FULL_BARRIER=1.

## Phase A4 - pairwise-sync kernel transport (build4): PARITY + BIT-EXACT
- averify2 (verify, pairwise sync): ZERO mismatches; replay median 4.49 ms/layer
  (verify overhead + live NCCL still included) vs 10.9 with full barriers.
- a3-kernel-tol (HC_TRANSPORT=kernel): decode-domain 162.86 / wall 112.32
  (control rctl602: 164.80/111.42 -> -1.2%, parity); tolerance vs s597 control
  **1.0/1.0 BIT-EXACT** (8192/8192); zero events; VRAM max 29616 MiB.
- PHASE A COMPLETE: all 9 hc-class collectives NCCL-free, flag-gated, bit-anchored.

## Phase B - race gates (build4) - started
- phaseB-all.sh: b-sb-{1,2,3} (Simple+batched+relay+kernel; zero NCCL in graph)
  pairwise; bsanity-{1,2} (same regime, HC on NCCL - detector must fire);
  dot dump (zero-NCCL node inventory, layer 2); g-{1..6} LL census w/ tolerance.

## Phase B1 - zero-NCCL race gate (b-sb pairwise): SIGNATURE TRANSFORMED, residual survives
- b-sb-{1,2,3} (Simple+batched+relay+kernel): all 3 pairwise comparisons still diverge,
  BUT the s600/s601 signature is GONE:
  - agreement 0.906/0.937/0.953 selected, 0.949-0.969 sequence
    (vs 0.016-0.047 with captured NCCL - a ~30x reduction in divergence mass)
  - onset moved: first_ck histograms {0,18,20,22}/{0,9,18,35}/{7,9,20,28} x32 slots
    (vs the fixed 96@step0 + 32@step2 locus); onsets VARY run-to-run
  - token flips now rare (5-14 of 128 vs nearly all)
- Reading: the captured-NCCL race is dead; a different lower-rate hazard remains.
  Suspects: batched swiglu exchange (s599 C-A5), s602 pairwise sync, relay, eager-head
  Simple NCCL. Triage pairs staged (kc = copy exchange; fb = full-barrier s602).

## Phase B2 - detector sanity: PASS (the old race reproduces on demand)
- bsanity-{1,2} (Simple+batched+relay, HC_TRANSPORT=nccl): agreement 0.047 selected /
  0.128 sequence, first_ck {0:96, 2:32} - the EXACT s600/s601 catastrophic signature.
  Detector valid on this binary/pod/day. Conclusion: replacing the hc-class NCCL
  collectives with the s602 kernel transport removes THE race (the captured-NCCL
  hazard); the residual b-sb divergence is a distinct, far weaker hazard.

## Phase B3 - zero-NCCL captured graph PROVEN (dotrun, layer 2, full stack)
- cudaGraphDebugDotPrint(layer 2, kernel+relay+batched): **0 nccl matches** in the
  verbose dot; 762 kernel nodes; s602 kernels in-graph: 57 copy3 + 41 fold +
  16 gather8. dot-summary.txt; dot kept out of the repo (pod-side only).

## Phase B4/C1 - LL census first point + the batched-exchange exoneration
- g-1 (LL, full stack kernel+relay+batched): **tolerance 1.0/1.0 BIT-EXACT vs s597
  control, zero events** - the batched exchange, which failed tolerance in every
  prior sprint (s599 C-A5, s601 c1 token flips), is CLEAN once the captured-NCCL
  race is dead. Confirms s601 reading: the racing collectives were upstream of the
  exchange; the exchange itself was innocent.
- Perf: decode-domain 162.06 / wall 114.84; replay median 4.39 ms/layer (a3 ~4.4,
  control 4.13; s601 relay-alone measured 3.59) -> the relay+batched EP-window
  savings do not currently materialize on top of the s602 site syncs (ep_ms
  envelope halved 447->234 but off the critical path). Perf gate work, not a
  correctness issue.
- Census driver bug (CENSUS_$i_DONE unbound under set -u) killed g-2..6 after g-1;
  fixed (phaseBC-census2.sh) and chained after the triage pairs.
- Queue: kc pair (copy exchange) -> fb pair (full barriers) -> census2 g-2..6 ->
  hh pair (HEAD_COMM=host under Simple: tests whether the residual divergence
  lives in the eager-head NCCL, the only NCCL left anywhere in the step).

## Phase B5 - triage kc: batched exchange EXONERATED as the residual carrier
- b-sb-kc-{1,2} (Simple, kernel+relay+COPY exchange): still diverges, same weak
  class: agreement 0.789 selected / 0.935 sequence, first_ck {4,13,20,27}x32,
  token flips rare. The residual hazard is NOT the batched exchange.

## Phase B6 - triage fb: THE RESIDUAL CARRIER IS THE PAIRWISE SYNC
- b-sb-fb-{1,2} (Simple+batched+relay+kernel + S602_FULL_BARRIER=1):
  **pairwise BIT-IDENTICAL (1.0/1.0, zero divergence)** under the harshest stress.
  -> The residual hazard is the s602 pairwise dependency set, NOT the batched
  exchange (kc), NOT the eager-head NCCL (still present in fb), NOT relay.
  fb replay 10.2 ms/layer (dense joins, not shippable for perf).
- Fix (build5): all-rank join across RANK STREAMS ONLY at both site sync points
  (8 records + 56 waits; NCCL completion-semantics-equivalent, no dense joins).
  Pairwise mode removed; FULL_BARRIER=1 escape retained.
- Queue (census2 on the pairwise build4 + hh pair) cancelled as moot; GPUs
  idle-verified; defunct zombies harmless.

## Phase B7 - build5 (rank-join) gates
- averify5: ZERO verifier mismatches (bring-up clean on rank-join).
- b5-sb-{1,2,3} (Simple stress) pairwise: residual shrinks again but not zero:
  sequence agreement 0.9976/0.9969/0.9994 (vs 0.949-0.969 pairwise, 0.128 NCCL);
  1-2 divergent batches per pair, onsets {59}/{6,63}/{6,63} - late-sequence bias.
  Rate ~1.7 events/256-step pair under Simple stress.
- Caveat recorded: fb (full-barrier) zero-divergence was a single pair; P(0|rate1.7)
  ~0.18, so "full barrier fully suppresses" is unproven. The LL census (g5 x6 +
  15 pairwise cross-compares) is the serving-regime verdict.

## Phase B8/C2 - LL census complete (build5, full stack)
- g5-{1..6}: ALL 6 token-bit-exact vs s597 control (sequence 1.0 = 8192/8192).
  ONE checksum-only event in 6 runs (g5-6 batch 3 step 62, zero token changes)
  -> 0.17 ck-events/run, ZERO token events (promoted history: 1.0/run with token
  flips in 2/3 runs; relay: 1.5/run). 6x reduction + token-exactness.
- LL pairwise determinism: 15/15 pairs token-identical (sequence 1.0); the single
  ck event visible as {62:32} ck-histogram rows in g5-6 pairs.
- Residual signature: checksum-only, LATE steps (b5-sb {6,59,63}, census {62}),
  low rate; never token-level on build5.
- Census perf: decode-domain 142.3-156.6 (median ~153.4), wall ~111.5-113.1,
  VRAM 28768 MiB.
- phaseD started (d32b kernel-only finals, d1/d8 scaling, d1p/d8p stage tables).

## Phase D1 - d32b kernel-only finals: TOKEN EVENT - kernel-only is NOT race-free
- d32b (HC_TRANSPORT=kernel only; EP return still NCCL broadcasts in-graph):
  decode 139.35 / wall 102.04 (slow tail-of-day run); tolerance FAIL:
  batch 4 first_ck step 1, 4/32 slots token-flip from step 1 - the classic
  captured-NCCL locus class (cf. s601 rctl601b/d32 escalation events).
- Reframe: kernel-only != zero-NCCL (8 EP-return broadcasts remain captured).
  With hc NCCL gone, the EP-return NCCL carries its own exposure. The FULL
  zero-NCCL stack (kernel+relay+batched) is the only correctness-complete
  config: 6/6 census token-clean. a2/a3 kernel-only cleanliness was 2-run luck.
- Queued: full-stack scaling points d1f/d8f + d8fp after phaseD.

## Phase D2 - scaling + stage tables (final)
- Kernel-only (build5): d1 5.92 (step 168.8 ms), d8 45.47 (175.9), a3 162.86 (196.5);
  d32b 139.35 - slow run AND token event (see Phase D1).
- Zero-NCCL full stack: d1f 5.35 (186.9), d8f 42.40 (188.7), census S=32 median
  ~153.4 (208.6). Slot-flat as always.
- Stage means S=8 profiler-on (d8p kernel-only / d8fp full stack):
  ep_window 2.29/2.56, route_plan_pack 0.95/0.94 (s601: 0.45), prefix_hc_current
  0.69/0.66 (s601: 0.22), final_hc 0.31/0.60, ep_return nccl 0.59 / relay 0.52.
  The 16 rank-joins/layer (~0.09 ms each) land in the router/hc/final_hc stages -
  ~1.4-1.5 ms/layer total, the 603 reclaim target.
- >=50/slot restatement: zero-NCCL stack floor 186.9 ms (S=1) / 188.7 (S=8);
  required MTP multiplier 9.3-9.4 (was 6.3-7.1 on the racing promoted base).
- GPUs idle-verified post-runs; pod left up; nothing in the repo tree but logs.
