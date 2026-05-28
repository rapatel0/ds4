# Sprint 483 - A6 Rank-Major Attention Projection Promotion

## Overview

Sprint 483 resumed A6 after the Sprint 482 failure-capture work. A6 had become
servable, but the first same-shape tolerance A/B failed with selected-token
agreement `0.625`. The failure was traced to a stale rank-major source buffer:
`run_shared_hc_current_input` can overwrite `r.d_current_full` with the
FFN-normalized route input before attention projection runs, while
`r.d_current_full_rank_major` still held the pre-overwrite raw current tensor.

## Implementation

- Added generic run descriptions and server-argument pass-through to the TP/EP
  profile and A/B harnesses so profiling remains experiment-agnostic.
- Added gated A6 input-parity diagnostics under
  `--true-ds4-attention-projection-input-parity-gate`.
- Added `slot_major_current_to_rank_major_kernel` and refresh the rank-major
  attention-projection source from the actual slot-major source when the
  routed-FFN norm route-input path has overwritten `r.d_current_full`.
- Promoted
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT=1` in the
  appliance launcher and V100 env example.

## Evidence

- V100 build passed in the CUDA pod:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Before fix diagnostic:
  `/workspace/s483-a6-input-parity-debug`.
  `tp_ep_attention_rank_major_input_debug` showed `raw_mismatches=131072`
  at layers 0 and 1; HC-current full rank parity still passed, proving the
  rank-major buffer was stale relative to the current slot-major source.
- After fix diagnostic:
  `/workspace/s483-a6-input-parity-fixed`.
  Rank-major input debug passed for layers 0 and 1, and attention-projection
  input parity reported `mismatches=0`.
- Same-shape A6 promotion artifact:
  `/workspace/s483-a6-rank-major-tolerance-fixed`.
  Control and candidate both served `32/32` selected-token HTTP requests.
  Response parity matched `32/32`; tolerance passed with
  `selected_token_agreement=1.0` and `max_selected_logit_relative_error=0.0`.

## Timing

At the selected-token 32-request shape:

- Control decode: `1597.851487 ms`.
- A6 candidate decode: `1582.511255 ms`.
- Decode speedup: `1.0097x`.
- Attention projection stage speedup: `1.0973x`
  (`94.505921 ms -> 86.125679 ms`).
- Client generated tok/s speedup: `1.0114x`.

## Conclusion

A6 is promoted. The correctness blocker was a stale source-layout interaction,
not an arithmetic mismatch in the rank-major normalization kernel.
