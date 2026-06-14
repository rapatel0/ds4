# Sprint 606 command log (pod llamacpp-build-8gpu, gpu-01) - 2026-06-14

## Setup
- Pod up ~23h (s605), all 8 GPUs idle, 0 foreign (two Jun13 zombies = defunct s605 teardowns).
- /workspace persisted. Launcher defaults confirmed = promoted s605 stack:
  EP_RETURN_TRANSPORT=relay, SWIGLU_EXCHANGE=batched, HC_TRANSPORT=kernel, S602_SYNC=edges,
  DENSE_FIX=1, ATTN_OUT_GATHER8=0.
- Runner: tools/s606-run.sh (s605-run clone, ART_DIR=/workspace/s606-artifacts).
- Driver: tools/s606-phaseBC.sh (amp | ab).

## Phase 0 - base re-verify
- p0-floor-s8: SLOTS=8 REQUESTS=32 SKIP_TOL=1. RESULT: continuation_tok_s_decode 40.165
  (5.02/slot, step 199.2ms, replay_ms 4.628 ms/layer) = s605 promoted floor exactly. In-band.

## Phase A - microbatch capture-machinery analysis + graph-structure decision
- Capture = ONE multi-stream fork/join cudaGraph per layer (decode_loop.cu attempt_capture_probe).
  Decision: option (b) in-graph choreography (only structure compatible with the persistent
  single-graph cache); (a) graph-split rejected.
- Sequential byte-identical microbatch=2 = MULTI-SPRINT (no slot-range primitive; opt.slots flat
  in ~157 sites; dual activation+op-struct buffer set). Execution-Note-1 legitimate finding.
  Fallback taken: rendezvous-merge lever.

## Phase B/C - RDZV_MERGE (elide redundant post-compose 1373 barrier)
- Flag DS4_V100_TP_EP_RDZV_MERGE=0|1 (default 0, byte-identical off). Built S597_BUILD_OK (build1.log).
- Amp gates (s606-phaseBC.sh amp): rdzv-amp20-aoa, rdzv-amp20-precompose (RDZV_MERGE=1 + amp 20us). [results below]
- A/B (s606-phaseBC.sh ab): rdzv-{off,on}-s{8,16,32}. [results below]

## RESULTS
- Amp gates (RDZV_MERGE=1, S=32, vs s597 control): rdzv-amp20-aoa 1.0/1.0 zero;
  rdzv-amp20-precompose 1.0/1.0 zero. Elision fires (elided_1373_barriers 2/layer x43).
  PASS - elided barrier did NOT reopen dense<->rank / compose-region hazard.
- A/B (paired one session): off->on cont-decode: S8 38.654->38.799 (+0.38%);
  S16 73.671->74.826 (+1.57%); S32 140.826->142.456 (+1.16%). replay_ms/layer drops
  on->off every S. S32 tolerance 1.0/1.0 both arms.
- DECISION: NO PROMOTION (correctness-clean + perf-positive but <+15%). Hold opt-in
  (default DS4_V100_TP_EP_RDZV_MERGE=0); gain recorded; stackable rendezvous-merge template.
- Microbatch: NOT implemented - sequential byte-identical = multi-sprint (no slot-range
  primitive; ~157 flat-opt.slots sites; dual activation+op-struct buffers). Graph decision:
  option (b) in-graph choreography. 607 staged re-scope.
