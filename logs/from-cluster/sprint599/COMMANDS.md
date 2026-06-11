# Sprint 599 command log (pod llamacpp-build-8gpu, gpu-01) - 2026-06-11

## Phase A - post-C1 full-layer decomposition
Profiler extended with prefix stages (engine/runtime_profiler.cu stages 25-32:
prefix_hc_current/attn_projection/compressed_kv/attn_state/typed_history/
raw_read/attn_output/final_hc; marks in engine/decode_loop.cu around the
existing stage calls + run_final_hc_carry). Measurement: pa-prefix-decomp
(nccl leg, profiler on, 32 req x 64 tok). Per rank per layer-step:
  prefix_attn_output 0.723 | final_hc 0.407 | prefix_attn_projection 0.262 |
  prefix_attn_state 0.250 | prefix_hc_current 0.246 | typed_history 0.049 |
  raw_read 0.012  => prefix total ~1.95 of the 4.55 (profiler-on) replay.
Cross-check: s597-nsys-stages.py on the s598 nccl sqlite: total GPU busy
2.94 ms/rank/layer-step vs 4.25 replay -> layer is wait/latency-bound;
attention kernel math itself is tiny (0.05 ms busy).

## Phase B candidates
Flags (engine/runtime_options.cuh + launcher):
  DS4_V100_TP_EP_SWIGLU_EXCHANGE=copy|nccl|memcpy2d|batched (default copy)
  DS4_V100_TP_EP_EP_RETURN_EARLY=0|1 (default 0)
Code: engine/runtime_pack.cu swiglu_down_exchange_{nccl,memcpy2d,batched};
engine/decode_loop.cu C-B early pack+return block + per-rank
rank<->dense ordering replacing the 954/978 8x8 barriers when early.

Runs (reference shape, fixed harness; all slot-indexed tolerance vs
phase0-full-control; flags-off calibration probe = 32/32):
  rctl  (flags off):        decode 167.19 / wall 112.70; tolerance 1.0/1.0;
                            nodes 2825/layer (= s598 3017 - 192 prof stamps).
  rca   (swiglu=nccl):      decode 197.17 (+17.9%) BUT tolerance 0.781/0.935 FAIL.
  probe-ca2 (+barrier):     29/32 8-tok prefix FAIL.
  probe-ca3 (per-src bcast):14/32 FAIL (worse with more small collectives).
  probe-ca2s (+NCCL_PROTO=Simple): 0/32 (allreduce order regime change;
                            invalidates the control comparison entirely).
  probe-ca4 (memcpy2d):     26/32 FAIL (pure copies! -> not a numerics bug).
  probe-ca5 (batched kernels): 32/32 PASS at 8 tok, BUT
  rca5  (batched, 64 tok):  decode 186.33 (+11.5%); tolerance 0.922/0.954 FAIL.
  CONCLUSION: every faster swiglu exchange (5 variants, including bit-exact
  pure-copy ones) diverges at the reference shape; the promoted path is
  timing-protected by its own 1792-launch copy storm -> a LATENT downstream
  ordering hazard exists; C-A DROPPED as unpromotable; hazard is the lead
  S600 follow-up.
  rcb   (ep_return_early=1): decode 168.07 (+0.5%, noise); tolerance 1.0/1.0
                            PASS -> correct but no standalone gain; NOT
                            promoted (no measured benefit).
  rstack (batched+early+profiler): demonstrated-ceiling evidence run
                            (tolerance-failing by C-A; perf + stage minima).
