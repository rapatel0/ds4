# TEMP Status Report 449

## Current Focus

TP/EP rank-major serving bundle correctness and promotion gating.

## Implemented

- Added Sprint 449 plan for the combined rank-major recheck.
- Ran the combined same-binary HTTP A/B after Sprint 448 fixed attention
  rank-local correctness.
- Candidate gates:
  - `--candidate-attention-projection-rank-local-input`
  - `--candidate-routed-ffn-rank-major-input`

## Results

Clean artifact:

- `/localpool/ds4/workspace/logs/s449-combined-rankmajor-attn-ffn-rerun`

The first attempt was invalid because an unrelated queued router-rank-major
isolate used the same port base and killed the control server with `rc=-15`.
The rerun used port base `18800` and completed both legs.

| Leg | Server generated decode tok/s | Server continuation decode tok/s | Client generated tok/s | GPU util avg | First token |
|---|---:|---:|---:|---:|---:|
| Control | 20.573850 | 20.485800 | 0.742192 | 9.651316% | 71302 |
| Attention rank-local + FFN rank-major | 20.372120 | 20.233079 | 0.745437 | 10.302632% | 71302 |

Response parity:

- `matched_pairs=4`
- `failed_pairs=0`
- `match=true`

VRAM:

- Control min free: `5698 MiB`
- Candidate min free: `5698 MiB`
- `vram_failures=0` on both legs

## Assessment

The old Sprint 445 combined rank-major token drift is fixed. Attention
rank-local input and routed-FFN rank-major input now compose correctly at the
reduced 8-slot / 256K HTTP shape.

This is still not a promotion. The combined bundle is correctness-clean but
slightly slower on server decode:

- generated decode speedup: `0.9902x`
- continuation decode speedup: `0.9877x`

The isolated routed-FFN rank-major result remains positive, so the likely next
move is not "abandon rank-major"; it is to identify the extra attention-side
cost and decide whether to promote FFN rank-major alone, add router rank-major,
or build a graph-safe route/pack bundle that produces a larger launch reduction.

## Cluster State

After Sprint 449, the V100 node reported no active DS4 GPU jobs and all eight
GPUs had `0 MiB` used by DS4 processes.
