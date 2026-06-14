# Sprint 605 command log (pod llamacpp-build-8gpu, gpu-01) - 2026-06-13

## Setup (pod RECREATED - s604 follow-up #4, 37h degradation)
- kubectl delete pod llamacpp-build-8gpu; kubectl apply -f deploy/v100/ds4-v100-build-localpool.pod.yaml
- /workspace hostPath persisted (deps/packs/prior artifacts intact). Verified 16Gi /dev/shm, 8 GPUs idle, 0 foreign.
- apt: build-essential cmake git python3 curl ca-certificates.
- Tree synced from laptop HEAD 6ef4d199 (tar pipe, exclude .git/build/logs/research/*.gguf).
- build1: bash /workspace/s597-build.sh > build1.log (S597_BUILD_OK).
- Runner: /workspace/ds4/tools/s605-run.sh (s604-run clone, ART_DIR=/workspace/s605-artifacts).
- Launcher defaults: kernel/relay/batched + DENSE_FIX=1; S602_SYNC flipped join->edges after Phase A.

## Phase A - promote edges+fix
- s605-phaseA-gate.sh: aedctl (edges+fix amp off) 1.0/1.0 zero; aed-amp20-{1,2,3} (amp 20us @ attn_out_a)
  ALL 1.0/1.0 zero (fix holds the window on the EDGES path); aed-precompose20 (amp 20us @ pre_compose)
  1.0/1.0 zero (s604 follow-up #3 late-step class closed).
- s605-phaseA-soak.sh: 32 un-amplified edges+fix runs + telemetry. RESULT: 32/32 token+ck CLEAN (1.0/1.0),
  ECC 0, 38.5C, 1486MHz, 69.2W. PROMOTE edges. Launcher default flipped to edges (rollback=join).

## Phase B - clean-base decomposition (edges+fix base)
- VRAM feasibility (microbatch): slot-scaled activation/staging ~21 MiB/rank@S=32 (~5.3@S=8); ~3.9 GiB free/GPU.
  Microbatch FITS at all S (~180x headroom). Spec VRAM-blocker risk falsified.
- s605-phaseB-decomp.sh: b-floor-{s1,s8,s32} (profiler off, SKIP_TOL) + b-prof-{s1,s8,s32}
  (EP_STAGE_PROFILE=1) for the launch/wait vs GPU-busy split. [results below]

## Phase B floors + stage tables
- b-floor-s8 40.16 agg cont-decode (5.02/slot, step 199.2ms); b-floor-s32 144.19 (4.51/slot, 221.9ms).
- b-prof-{s8,s32}: stage tables (b-prof-*-stagetable.tsv). prefix_attn_output 1.07/1.14 ms/rank/layer
  (heaviest prefix stage); routed-FFN GEMMs <0.24 -> step ~95% launch/sync/transport, <5% GEMM.
- VRAM gate: microbatch fits at all S (~21 MiB/rank @ S=32, ~3.9 GiB free). Spec #1 risk resolved.
- b-floor-s1 rc=124 (timeout - S=1 replay-probe loop, see report deviations).

## Phase C - attn_output gather8 lever (DS4_V100_TP_EP_ATTN_OUT_GATHER8=0|1, default 0)
- build2: S597_BUILD_OK (gather8 kernel + flag).
- s605-phaseC-gate.sh: cg-amp20 (gather8+amp20@attn_out_a) 1.0/1.0 ZERO; cg-ctl 1.0/1.0; cg-off 143.89;
  cg-on-{1,2} 128.3-128.9. CORRECTNESS PASS (incl amplifier at carrier site). PERF REGRESSION ~10%.
- VERDICT: NO PROMOTION (correctness-clean but perf-negative). The 64 memcpy2D are DMA copy-engine
  transfers; 8 gather kernels serialize cross-GPU strided reads on SMs -> heaviest prefix stage is
  DMA-transport-bound, not launch-bound. gather8 stays default-off (diagnostic/negative). No soak spent.
- Phase D: M~=10 @ S=8 for >=50/slot; 0 of the ~3.3-4x base reduction captured; 606 lever sequence
  re-ranked (microbatch lead; attn_output as transport; route-plan shadow + rendezvous merge).
