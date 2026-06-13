# Sprint 604 command log (pod llamacpp-build-8gpu, gpu-01) - 2026-06-12/13

## Setup
- Tree synced from laptop HEAD 4df3f9f8 + s604 working-tree edits (tar pipe,
  s597 convention: exclude .git/build/logs/research/*.gguf).
- New code (default-off, byte-identical when off):
  - DS4_V100_TP_EP_DENSE_HAZARD_AMP=<us> + _SITE=pre_dense,post_dense,pre_down,
    post_down,pre_compose,attn_out_a,attn_out_b (Phase A amplifier: busy-wait on
    the dense stream widening the dense<->rank window; reuses s600 delay kernel).
  - DS4_V100_TP_EP_DENSE_FIX=0|1 (Phase C: cross-GPU dense<->rank edge - each
    rank waits every peer's dense completion + reverse; the fb barrier's dense
    involvement without the redundant rank<->rank 8x8 join).
  - Wired in engine/attention_output.cu (the codex candidate-1 carrier:
    attn_output_a.d_out cross-rank dense->rank gap at :48->:87), and the EP/dense
    overlap hand-offs in engine/decode_loop.cu (954/978 joins + pre/post dense).
- build1: bash /workspace/s597-build.sh > /workspace/s604-artifacts/build1.log (S597_BUILD_OK)
- Runner: /workspace/ds4/tools/s604-run.sh (s603-run clone, ART_DIR=/workspace/s604-artifacts)
- Launcher defaults = zero-NCCL stack (kernel/relay/batched, S602_SYNC=join).

## Run log

## Phase A - amplifier tuning (s604-phaseA.sh, default zero-NCCL stack)
- a0ctl (amp off): tolerance 1.0/1.0 (128/128, 8192/8192); natural ck-only event
  batch1 step22 all-32-slots, ZERO token. Byte-clean amp-off confirmed.
- a1-aoa5  (amp 5us  @ attn_out_a): sel 0.508 seq 0.759, token onsets 5..58, fires every run.
- a1-aoa20 (amp 20us @ attn_out_a): sel 0.055 seq 0.167, ~100% token, onset step 0-3.
- a1-aoa50 (amp 50us @ attn_out_a): sel 0.070 seq 0.056, ~100% token, onset step 0.
- VERDICT: deterministic dose-dependent amplifier; carrier = attn_output_a.d_out
  cross-rank dense->rank RAW (attention_output.cu:48 dense write -> :87-98 peer
  rank-stream read; per-GPU :51 edge leaves cross-rank unordered).

## Phase C - DENSE_FIX gate (s604-phaseC.sh)
- c1-fix-aoa20 (FIX=1 + amp 20us @ attn_out_a), c2-fix-aoa50 (FIX=1 + amp 50us),
  c3-fix-off (FIX=1 amp off). [results below]
- RESULTS: c1-fix-aoa20 1.0/1.0 ZERO ck ZERO token; c2-fix-aoa50 1.0/1.0 ZERO;
  c3-fix-off 1.0/1.0 ZERO (cleaner than a0ctl - the natural ck-only flicker also gone).
  DENSE_FIX kills the amplified hazard (was ~100% token-corrupt at amp 20/50us) -> ZERO.

## Phase B - other-site amplification (s604-phaseB.sh)
- b-precompose20 (amp 20us @ pre_compose): sel 0.977 seq 0.986, 3 slots token-flip
  @ step27 - the WEAK late-step class (same dense<->rank family, secondary site).
- b-predense20 / b-aob20 / b-fix-precompose20: ABORTED - the per-GEMM pre_dense amp
  is pathologically slow (>30min/run, busy-wait on every dense GEMM x44 layers x256
  steps); confirmatory only, carrier already named via attn_out_a. GPU freed for Phase D.

## Phase D - soak + telemetry (s604-phaseD.sh, 26 pairs = 52 runs alternating
##   fix-on/fix-off, un-amplified reference shape, per-run nvidia-smi telemetry)

## Phase D telemetry note
- n_concurrent_proc counts nvidia-smi compute-app ROWS (one per GPU context). Our single 8-GPU appliance shows as 8 rows sharing ONE pid (verified 1201863). A row showing 8 at soak-on-N is the PRIOR appliance still tearing down at the pre-run sample, NOT a foreign host process. The s604-run.sh wait-for-idle + foreign-pid preflight (by unique pid) passed every run.

## Phase D soak result (34 runs, 17 pairs; soak-on-18 stalled on pod degradation, stopped)
- fix ON: 17/17 token+ck CLEAN (1.0/1.0). fix OFF: 7/17 token, 11/17 ck (the s603 hazard, un-amplified).
- Telemetry: both arms ECC 0, ~38.6-38.8C; SM-clock col diff is pre-run sample timing not per-arm bias.
- GATE PASS: fix event-free, incumbent FAILS (41% token). soak-summary.txt + phaseD-telemetry.tsv saved.

## Phase E (s604-phaseE.sh): composition + floors on the clean fix base
- e-edges-fix-1 (edges+FIX=1): 1.0/1.0 ZERO ck ZERO token - fix COMPOSES with the fast stack, event-clean at full speed (resolves s603 edges census FAIL).
- e-edges-fix-{2,3}, e-edges-fix-amp20, e-floor-{s1,s8,edges-s1,edges-s8}: in flight (pod slow).

## Phase E status at report time
- e-edges-fix-1, e-edges-fix-2: BOTH 1.0/1.0 ZERO events (edges+fix composition gate PASS, 2/2).
- e-edges-fix-3, e-edges-fix-amp20, e-floor-{s1,s8,edges-s1,edges-s8}: still running (pod severely degraded over ~38h uptime, ~25-30 min/run). Floors are confirmatory for the fix-adds-~0 cost claim; the MTP restatement is reasoned from the s602/s603 measured floors (join 186.9/188.7, edges 175.0/177.1) since the fix adds no GEMMs/copies.
- GPUs left to the running Phase E; pod stays up per hygiene.
