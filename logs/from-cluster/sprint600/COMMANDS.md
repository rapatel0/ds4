# Sprint 600 command log (pod llamacpp-build-8gpu, gpu-01) - 2026-06-11

## Setup
- Laptop tree (HEAD 20f4458f + s600 probe edits) re-synced to /workspace/ds4 (tar pipe, s597 convention).
- Probe instrumentation (all default-off, flag-off graph byte-identical):
  engine/runtime_options.cuh  DS4_V100_TP_EP_S600_DELAY / _JITTER / _SWIGLU_VERIFY /
                              _RETURN_VERIFY / DS4_V100_TP_EP_GRAPH_DOT_DIR(_LAYER)
  engine/runtime_pack.cu      s600 probe module: host-tunable busy-wait delay sites,
                              swiglu-exchange staleness verifier, EP-return slice verifier
  engine/ep_dense.cu          xchg_tail hooks in materialize (graph path)
  engine/decode_loop.cu       delay sites pre/post down, pre_pack, pre/post return,
                              post_compose; jitter refresh per replay; verify collection;
                              cudaGraphDebugDotPrint after capture
- Build: CUDA_ARCH=sm_70 make -j72 appliance/ds4-v100-tp-ep-appliance (build.log, rc=0)
- Runner: /workspace/s600-artifacts/s600-run.sh <name> (wait-for-idle preflight,
  reference shape 32 slots/256K/64tok/128req, slot-indexed tolerance + first-divergence
  via s600-first-divergence.py against phase0-full-control)

## Phase A evidence (offline, from s599 artifacts)
- s600-first-divergence.py on s599 runs vs s597 control:
  rctl (flags off):  128 pairs, ALL decode_step_checksums bit-identical (1.0/1.0)
  rcb (early ret):   bit-identical too
  rca (nccl xchg):   whole-batch divergence at one step per batch: steps 2,4,9,24
  rca5 (batched):    whole-batch divergence at steps 4,26,27,35
  => the hazard is a RARE single event (~1 per 1000+ layer-replays) that corrupts a
     step-wide buffer (12/32 slots' tokens flip at onset), state-carried thereafter.
- Static audit: in the promoted graph config the route-plan control stream IS rank 0's
  stream; every captured node lives on the 16 rank/dense streams joined by the 8x8
  event barriers (954/978/1144/1373) => no orderable pair inside our DAG is unordered.
  Cross-rank data movement that does NOT get graph edges: NCCL collective internals
  (~11 captured LL collectives per layer: hc allreduce/allgather, router allreduce,
  post-attn allgather, 8 EP-return broadcasts).

## Runs
rctl600 (flags off, probes compiled but off, identity + in-band control):
  decode-domain 170.49 / wall 116.27 (band vs s599 rctl 167.19/112.70: +2.0%)
  slot-indexed tolerance vs phase0-full-control: selected 1.0 sequence 1.0 PASS
  NOTE (evidence): batch 1 (all 32 slots) decode_step_checksums diverge from
  the control at step 2 with ZERO token changes -> the rare corruption event
  fires (small-magnitude) on the PROMOTED copy path too: latent debt confirmed
  live, not just inferred. 1 firing in 256 steps (copy) vs 4 in 256 (batched).
rca5v (batched + SWIGLU_VERIFY + RETURN_VERIFY):
  decode-domain 67.03 (verifiers = 56+56 remote-read kernels/layer, copy-storm-like cost)
  tolerance 1.0/1.0, ALL checksums bit-identical, ZERO verifier mismatches
  -> Heisenbug: the verification load itself re-masks the race (like the copy storm).
rsimple-ctl / rsimple-ca5 / rsimple-ctl2 (NCCL_PROTO=Simple regime):
  copy+Simple vs LL control: total divergence from step 0 (expected: allreduce
    order regime change). decode-domain 117.83 (Simple is slow).
  batched+Simple vs copy+Simple: ALL 128 pairs diverge (96 at step 0, 32 at step 2).
  copy+Simple vs copy+Simple (rerun): ALSO all 128 diverge (96@0, 32@2)
  -> Simple regime is run-to-run NONDETERMINISTIC even on the promoted copy leg:
     the race fires nearly every step under Simple, rarely under LL+copy, ~4/256
     steps under LL+batched. Protocol modulates firing rate; A/B inconclusive for
     NCCL-internal vs engine-DAG. Pivot to delay bisect in the LL regime.

## Phase A continued - bisect + elimination (all reference shape, 128 req x 64 tok)
rd-preret  (batched + DELAY pre_return:600us/layer):  1.0/1.0, ALL checksums identical. 165.62
rd-postret (batched + DELAY post_return:600):         1.0/1.0, ALL identical. 164.64
rd-predown (batched + DELAY pre_down:600):            1.0/1.0, ALL identical. 165.45
  -> masking is SITE-INDEPENDENT: any +600us/layer protects; no local edge is bridged.
  -> also: the delay legs being bit-exact EXONERATES the TurboMind atomic scatter-add
     (schedule changed, bits did not).
rd-postsync (batched + cudaDeviceSynchronize on all 8 devices after EVERY replay):
  STILL diverges (0.88/0.93, onsets 12/23/46) -> no cross-replay overlap; race is
  INSIDE a single layer replay.
rv2-batched (batched + dense-stream return+allgather verifiers): verifiers cost
  pulls timing into the protected regime (decode 66.9) -> clean, 0 mismatches.
  In-graph verification cannot catch the event (detection load == protection).
rnccl-ca5/rnccl-ctl (NCCL 2.27.7 side-install, LD_LIBRARY_PATH):
  copy leg 173.39 decode-domain; batched vs copy STILL diverges (0.81/0.85,
  onsets 2/5/24) -> NOT fixed by NCCL upgrade.
rfix-ca5 attempts (dedicated output-head comm): ncclCommInitAll of a SECOND
  8-rank comm CANNOT fit this pod: NCCL allocates 8 x 4.19 MiB fixed-size
  proxy-pool shm segments per comm; /dev/shm is 64 MiB (33.5 MiB/comm).
  Verified with /workspace/s600-artifacts/s600-twocomm-test.cu (+ splitShare=1
  variant: same footprint). BUFFSIZE/LL_BUFFSIZE/SHM_DISABLE do not shrink it.
rring-ctl (copy + NCCL_ALGO=Ring): 1.0/1.0, ALL checksums identical, 166.38
  -> Ring forcing is bit-neutral (everything already runs RING_LL; confirmed by
  the graph dots: 16 NCCL kernels/rank/layer, all *_RING_LL*).
rhost-ca5/rhost-ctl (head_comm=host: ALL eager NCCL removed from the comm -
  host-side reductions + UVA-copy allgather for the output head):
  copy+host 169.37 decode-domain; batched+host vs copy+host STILL diverges
  (0.945/0.964, onsets 2/7 in 2 of 4 batches) -> eager/captured MIXING is NOT
  the racing mechanism either.

## Graph topology evidence (methodology b)
dot dumps layer 2 (verbose): copy 2825 nodes / 4534 edges; batched 1089 / 2798;
identical NCCL node sets (16 AllGather, 40 AllReduce_Sum, 72 Broadcast, all
RING_LL) and identical barrier skeleton (401 non-kernel nodes each); the ONLY
difference is 1824 copy_f32 exchange kernels vs 56 strided-seg kernels.

## Final gates
rctl600b (final binary build8, ALL flags off - identity + in-band control):
  decode-domain 170.18 / wall 117.08; tolerance 1.0/1.0; ALL 256 step
  checksums bit-identical to the s597 control. Flag-off identity PASS.
rhost-ca5b (batched+host repro vs rhost-ca5): same config, DIFFERENT onsets
  (2/7/17/17 vs 2/7/None/None patterns; 0.836/0.901) -> run-to-run
  NONDETERMINISM confirmed: the race fires inside the captured collectives
  with zero eager NCCL on the communicator.

## Disposition
- No promotion: every fast-exchange variant still trips the (now deeply
  characterized) captured-NCCL intra-replay hazard; the tolerance gate is
  unachievable until the EP window is NCCL-free or the platform behavior is
  fixed upstream. Launcher defaults unchanged (nccl return, copy exchange).
- New opt-in flags retained: DS4_V100_TP_EP_HEAD_COMM=shared|dedicated|host
  (default shared; dedicated cannot init on this pod: /dev/shm), s600 probe
  envs (DELAY/JITTER/SWIGLU_VERIFY/RETURN_VERIFY/AG_VERIFY/POSTSYNC/
  GRAPH_DOT_DIR).
- C-B restack / C-C route-plan shadow: NOT attempted (moot without a
  promotable fast exchange; budget went to the root-cause matrix).
