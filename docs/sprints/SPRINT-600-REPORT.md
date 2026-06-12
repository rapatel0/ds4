# Sprint 600 Report - Root-Cause the Swiglu-Exchange Ordering Hazard

Date: 2026-06-11
Status: complete - **hazard localized far deeper than planned, but NOT to a
fixable engine edge: it is a timing-dependent corruption inside the captured
NCCL collectives themselves (V100, 8 ranks/process, full-graph capture),
reproduced across NCCL versions and protocols and with all eager NCCL
removed. No promotion; 220 verdict round 2: NOT reached and not reachable
until the EP window is NCCL-free or the platform behavior is fixed upstream.**

Environment: s598/s599 setup on pod `llm/llamacpp-build-8gpu` (gpu-01, 8x
V100-SXM2-32GB, driver 580.126.20, NCCL 2.19.3; NCCL 2.27.7 side-installed
for one experiment), pack `...-s597`, contract, s597 control, fixed harness,
wait-for-idle preflight on every run. ~22 reference-shape runs this sprint;
full command log in `logs/from-cluster/sprint600/COMMANDS.md`.

## What the hazard is NOT (each with the run that proves it)

The s599 framing ("a consumer is missing a dependency edge on the exchanged
swiglu data") is **falsified**. Eliminated, in order:

1. **Not an engine DAG edge.** Static audit: in the promoted config the
   route-plan control stream is rank 0's stream, every captured node lives on
   the 16 rank/dense streams, and the 954/978/1144/1373 sites are full 8x8
   event barriers - no unordered pair exists. Mechanical check:
   `cudaGraphDebugDotPrint` of the layer-2 graph under copy vs batched
   (verbose, archived) shows identical NCCL node sets (16 AllGather, 40
   AllReduce, 72 Broadcast - all RING_LL), identical 401-node barrier/event
   skeleton, identical root/leaf structure; the only diff is 1824 `copy_f32`
   exchange kernels vs 56 strided-seg kernels.
2. **Not stale exchanged swiglu data.** A flag-gated verifier
   (`DS4_V100_TP_EP_S600_SWIGLU_VERIFY`) bit-compares the exchanged segments
   against fresh remote reads: zero mismatches ever observed.
3. **Not cross-replay overlap.** `DS4_V100_TP_EP_S600_POSTSYNC=1`
   (cudaDeviceSynchronize on all 8 devices after every layer replay) still
   diverges (0.88/0.93, onsets 12/23/46). The race is INSIDE one replay.
4. **Not the TurboMind atomic scatter-add.** Three delay-injection legs
   (below) changed the schedule and were bit-identical to control across all
   256 steps - schedule-sensitive atomics would have shifted bits.
5. **Not NCCL-version-specific.** NCCL 2.27.7 (side-loaded) reproduces the
   identical rare-event signature (batched vs copy: 0.81/0.85, onsets
   2/5/24).
6. **Not protocol-specific.** Under `NCCL_PROTO=Simple` the system becomes
   run-to-run nondeterministic even on the copy leg (two identical copy
   +Simple runs diverge from step 0).
7. **Not eager/captured "graph mixing".** New `DS4_V100_TP_EP_HEAD_COMM=host`
   mode removes ALL eager NCCL from the communicator (the output head's five
   tiny allreduces become D2H + host reduce + H2D; its allgather becomes UVA
   peer copies). The race still fires: batched+host run-to-run diverges
   (0.836/0.901, onsets differing per run).
8. **Not comm-resource pressure.** A dedicated second communicator was also
   attempted: ncclCommInitAll allocates a fixed 8 x 4.19 MiB of /dev/shm
   proxy pools per comm; the pod's 64 MiB /dev/shm cannot host two
   (33.5 MiB each; `s600-twocomm-test.cu`, splitShare=1 identical). This is
   an environment fact, not the hazard.

## What the hazard IS (the positive characterization)

- **Locus**: within a single captured layer-graph replay, among the 16
  captured NCCL RING_LL collectives per rank per layer (hc allgather +
  3 allreduces, router allreduce, full-current broadcast, post-attn
  allgather, 8 EP-return broadcasts) on the one 8-rank-per-process
  communicator. NCCL's documentation explicitly cautions this shape
  ("using multiple GPUs per process with CUDA graph capture may result in
  deadlocks"); on this platform the failure mode is rare data corruption,
  not deadlock.
- **Trigger**: compressed intra-replay spacing. Event rate at the reference
  shape: copy (replay ~4.95 ms incl. exchange storm) ~1 event / 256 steps
  *(yes - the PROMOTED path fires too: rctl600 caught one checksum-only
  event at batch 1 step 2 with zero token changes)*; batched (~3.8 ms)
  ~3-4 / 256; nccl-exchange (~3.6 ms) ~4 / 256 with earlier onsets;
  Simple protocol: nearly every step.
- **Masking**: ANY ~600 us/layer busy-wait, at any of three different graph
  sites (pre_down / pre_return / post_return - i.e. site-INDEPENDENT),
  restores full bit-identity over 256 steps. The copy storm's only role is
  to be that delay. Adding verification kernels has the same masking effect
  (the Heisenbug result: every armed-verifier run was clean with zero
  mismatches - detection load equals protection).
- **Blast pattern**: a single firing corrupts a step-wide quantity (all 32
  slots' step checksum changes at one decode step; at token level up to
  12/32 slots flip at the onset step), then carries through KV/HC state.
  The first-divergence tool (`s600-first-divergence.py`, slot-indexed
  pairing + per-step checksum localization) is now part of the toolkit.

## Fix attempts (everything was built and measured, nothing passed the gate)

| Candidate | Result |
|---|---|
| Dedicated output-head comm (`HEAD_COMM=dedicated`) | cannot init on this pod (/dev/shm, above); code retained behind flag |
| Host-side head reductions (`HEAD_COMM=host`) | works, removes all eager NCCL, costs ~0 perf (169.4 copy leg) - but the race is not in the mixing, so it still fires; also changes sum associativity (new bit regime vs control: tokens 0.969 on the copy leg) |
| NCCL 2.27.7 upgrade | reproduces the hazard |
| NCCL_ALGO=Ring | bit-neutral (1.0/1.0, fully checksum-identical: everything already rings) but no effect on the hazard mechanism; useful fact: forcing Ring eliminates the SHM/tree connects |
| NCCL_PROTO=Simple | catastrophically worse (fires every step) |
| Delay injection (engineered spacing) | works perfectly (3 sites x 600 us: bit-exact) but is exactly the "timing-protected" debt this sprint was chartered to remove; not promoted |

## Gate matrix (DoD)

1. Hazard named with evidence - **partially met, honestly**: named to the
   captured-NCCL-collective level with an eight-way elimination matrix and
   archived artifacts (dot dumps, divergence histograms, twocomm test); the
   exact NCCL-internal mechanism (which collective's internal flag/slot path)
   could not be pinned - in-graph verification masks the race, and the
   corruption lands in the order-sensitive allreduces that have no
   independent truth to verify against.
2. Fix in the promoted path, copy 1.0/1.0 + perf in band - **no fix landed**
   (no candidate both fixed the hazard and preserved the bit anchor).
   Final binary flag-off identity: **1.0/1.0, all 8192 checksums
   bit-identical, decode-domain 170.18 / wall 117.08** (in band vs 167.19/
   112.70).
3. Fast variants tolerance-clean post-fix + jitter stress - **not achieved**
   (no fix). The jitter machinery (`DS4_V100_TP_EP_S600_JITTER`, host-retunable
   per-replay randomized delays) is built, tested, and ready for the
   follow-up sprint's stress gate.
4. Promotions - **none**. Launcher defaults unchanged
   (`EP_RETURN_TRANSPORT=nccl`, `SWIGLU_EXCHANGE=copy`, `EP_RETURN_EARLY=0`,
   `HEAD_COMM=shared`). All new flags opt-in with documented rollback.
5. Final stack + 220 verdict - **done, below**.
6. Report - this document; commits are the orchestrator's.

## Final configuration and numbers

Unchanged promoted configuration (s598). Final in-band measurement on the
final binary, flags off: **decode-domain 170.18 tok/s / wall 117.08 tok/s**,
tolerance 1.0/1.0 vs the s597 control, bit-identical step checksums.

## 220 verdict, round 2 - explicit

**NOT reached, and the s599 "reachable, gated on the hazard fix" call is
REVISED: not reachable within B2 scope on the current architecture.** The
gate is not an engine edge we can add; it is the behavior of captured NCCL
collectives on this V100 multi-GPU-per-process platform, reproduced across
two NCCL major versions and both protocols. The demonstrated 186.33 (batched)
/ 197.17 (nccl exchange) remain real but unpromotable. Paths that could
reopen 220:

1. **NCCL-free EP window** (the realistic in-repo path): batched swiglu
   exchange (0.29 ms, built) + kernel/relay-based EP return using the s597
   NVLink one-hop relay table (C2; avoids SYS; estimated 0.6-0.9 ms vs
   0.61 NCCL). Removes 9 of 16 captured collectives per rank-layer. Whether
   the remaining hc/router collectives still race at the tightened pacing is
   the first experiment of that sprint. Projected stack if clean:
   4.25 - 0.5 (exchange) - 0.26 (C-B edges) - ~0.35 (C-C shadow) + relay
   delta => ~3.4-3.5 ms/layer => ~195-205 tok/s; 220 needs the prefix
   launch-latency work on top. Call: **borderline, not safely within B2.**
2. Upstream: the evidence package (deterministic reproducer flags, event-rate
   vs spacing curve, version/protocol matrix) is escalation-ready.

## Deviations (honest list)

- C-B restack and C-C route-plan shadowing were NOT attempted: with no
  promotable fast exchange they cannot change the promoted stack, and the
  entire budget went to the root-cause matrix (~22 reference runs + 8 builds).
- The perturbation stress gate (>=5 jitter runs) was not run - it gates a fix
  that did not land. The machinery is built and validated.
- The fix DID briefly change the copy leg's token stream in the
  `HEAD_COMM=host` experiments (0.969 agreement) - this is the spec's
  contingency #5 scenario, but it was a deliberate associativity change in
  the host reduction, not race-taint; mode left opt-in, control NOT
  re-anchored.
- `NCCL_BUFFSIZE` was set globally in two diagnostic runs only; no promoted
  setting changed.
- One self-inflicted crash loop: the first two dedicated-comm runs died at
  init (the /dev/shm finding); no foreign-tenant overlap in any measured
  window; GPUs idle-verified before every run.
- Probe surface: all new probes/flags are default-off; flag-off identity was
  re-proven on the final binary (gate 2 numbers above).

## Artifacts

- Pod `/workspace/s600-artifacts/`: COMMANDS.md, build1-8 logs, s600-run.sh,
  s600-first-divergence.py, s600-dot-smoke.sh, dot/ (verbose layer-2 graph
  dumps copy+batched), s600-twocomm-test(.cu), s600-split-test(.cu), and 18
  run trees (rctl600, rctl600b, rca5v, rd-{preret,postret,predown,postsync},
  rv2-batched, rsimple-{ctl,ctl2,ca5}, rv-simple, rnccl-{ctl,ca5},
  rring-ctl, rhost-{ctl,ca5,ca5b}).
- Laptop `logs/from-cluster/sprint600/`: COMMANDS.md, dot-summary.txt, all
  sustained_http.tsv files, the analysis tools.
- Source changes (uncommitted, orchestrator review):
  `engine/runtime_options.cuh` (probe + fix flags),
  `engine/runtime_pack.cu` (s600 probe module: host-retunable delay sites,
  jitter, three verifiers), `engine/ep_dense.cu` + `engine/decode_loop.cu`
  (probe hooks, dot dump, postsync), `engine/output_head.cu` (dedicated-comm
  + host-reduction head modes), `engine/runtime_resources.cu` (startup init
  hook), `docs/sprints/SPRINT-600-REPORT.md`.
- NCCL 2.27.7 side-install kept at `/workspace/nccl-2.27.7/` (rollback: it
  is only active via explicit LD_LIBRARY_PATH).

## Follow-up seeds (for SPRINT-601 planning)

1. NCCL-free EP window: batched exchange + relay-table kernel return; then
   re-test the hazard at full speed (lead item; the only in-repo path to the
   demonstrated +11-18%).
2. If hazard persists with only hc/router collectives captured: fold those
   into kernel-based reductions too (they are small), making the decode graph
   fully NCCL-free.
3. Escalation package to NVIDIA (deterministic reproducer + matrix).
4. The promoted path's own ~1/256-step checksum-level exposure (rctl600) is
   now on the record - any future driver/NCCL/clock change can move it;
   the s600 probe flags are the standing diagnosis kit.
