# TEMP Status Report 447

## Current Focus

TP/EP serving correctness and performance. Sprint 447 followed the Sprint 446
finding that attention rank-local input changes tokens by itself.

## Implemented

- `run_true_ds4_attention_projection_prefix()` now uses the fresh HC-current
  source from `ranks[0].d_current_full` when HC-current peer/NCCL gather is
  active, instead of always normalizing `hc->d_current_full`.
- Added `--true-ds4-attention-projection-input-parity-gate`.
- Added `tp_ep_attention_projection_input_diff` diagnostics for the actual half
  inputs consumed by `attn_q_a` and `attn_kv_latent`.
- A/B harness now fails fast if a profile leg returns nonzero.
- Profile harness uses per-case `DS4_LOCK_FILE`.

## Results

After the current-source fix, attention-only still failed HTTP response parity:

| Leg | First token | Server decode tok/s |
|---|---:|---:|
| Control | 71302 | 20.326755 |
| Rank-local candidate | 63930 | 20.594887 |

Artifact:

- `/localpool/ds4/workspace/logs/s447-attn-source-fix-attn-only-ab`

Direct input parity audit:

- `/localpool/ds4/workspace/logs/s447-attn-input-parity-direct`
- rc: `0`
- `attn_q_a_input`: `344` lines, `10` bad lines, `325087` mismatches
- `attn_kv_latent_input`: `344` lines, `10` bad lines, `325087` mismatches

First bad rows:

```text
layer 0 rank 1  mismatches 32535  max_abs 0.135314941
layer 0 rank 6  mismatches 32535  max_abs 0.135314941
layer 0 rank 7  mismatches 32535  max_abs 0.135314941
```

## Assessment

The remaining attention rank-local bug is below the dense projections. Some
nonzero ranks have a different slot-major full-hidden buffer after HC-current
NCCL allgather/conversion, so their locally produced attention half inputs do
not match the device-0 normalized reference.

Next sprint: add a direct HC-current full-buffer parity audit immediately after
`rank_major_current_shards_to_slot_major_kernel`, compare every
`RankState::d_current_full` against rank 0, and fix the allgather/slot-major
conversion path before rerunning attention-only HTTP parity.

## Cluster State

After the Sprint 447 runs, the V100 node reported no active DS4 GPU jobs.
