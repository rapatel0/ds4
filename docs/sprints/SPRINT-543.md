# Sprint 543 - A5 HC Split Weighted-Sum Fusion Rejection

Date: 2026-05-29

## Goal

Test the smallest plausible A5 launch-count reduction after Sprint 542 ruled
out static route-cap tuning: fuse the promoted HC-current allreduce path's
split application with its weighted-sum current-shard kernel.

## Scope

Candidate code was intentionally narrow:

- Add a fused HC kernel that computes `d_hc_split` from the allreduced HC mix
  and writes `d_current_shard` in the same launch.
- Wire only the promoted `tp_hc_current_allreduce_gate` path.
- No launcher flags, no runtime defaults, no MTP work, no attention-projection
  A6 work.

## Build

Remote workspace:

- `/workspace/s543-hc-fused`

Build command:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`

Result:

- PASS for both candidate variants.

## Validation

Quick parity gate:

- Candidate artifact:
  `/workspace/s543-hc-fused-artifacts/none-none-s543-fused-graph8x4-p262080`
- Control artifact:
  `/workspace/s541-graph-audit-artifacts/none-s541-default-graph8x4-p262080`
- Result:
  - `http_200=8`
  - first token `29361`
  - graph cache hits `43`
  - peer-copy/SYS `0`
  - NCCL graph SYS edges `0`
  - response sequences matched control
  - decode-step checksums matched control

Warmed promotion gate, first fused attempt:

- Candidate artifact:
  `/workspace/s543-hc-fused-artifacts/none-none-s543-fused-graph32x64-p262080`
- Control artifact:
  `/workspace/s540-warmed-graph-artifacts/none-s540-graph32x64-compose-stable-p262080-serverargs-h2180dc1d`
- Result:
  - `http_200=32`
  - first token `107027`
  - graph cache hits `43`
  - position invalidations `0`
  - peer-copy/SYS `0`
  - NCCL graph SYS edges `0`
  - generated sequence multiset matched control
  - decode-step checksum multiset matched control
  - request window regressed `90.181067s -> 95.164862s`
  - client generated tok/s regressed `22.709903571 -> 21.520580950`

Bug-finder pass:

- The fused kernel collapsed the weighted-sum work from the old element-wide
  grid to a one-block-per-slot grid, reducing latency hiding. The diagnosis
  recommended either reverting or keeping an element-wide grid.

Warmed promotion gate, element-wide fused attempt:

- Candidate artifact:
  `/workspace/s543-hc-fused-artifacts/none-none-s543-fused-wide-graph32x64-p262080`
- Control artifact:
  `/workspace/s540-warmed-graph-artifacts/none-s540-graph32x64-compose-stable-p262080-serverargs-h2180dc1d`
- Result:
  - `http_200=32`
  - first token `107027`
  - graph cache hits `43`
  - position invalidations `0`
  - peer-copy/SYS `0`
  - NCCL graph SYS edges `0`
  - request window regressed `90.181067s -> 96.046732s`
  - client generated tok/s regressed `22.709903571 -> 21.322989766`
  - scaffold ms/token regressed `666.058962 -> 688.003409`

## Decision

Reject this A5 fusion shape and remove the candidate code.

The existing two-kernel sequence is better for the promoted graph-suffix
serving path. The split kernel is cheap, while the weighted-sum kernel benefits
from the old simple element-wide memory pattern. Recomputing or sharing split
inside the weighted-sum launch did not transfer to serving throughput.

## Follow-Up

- Do not retry split+weighted-sum fusion without a direct kernel microbenchmark
  showing a win at the `32` slot shape.
- True A5 should move to a different fusion target or wait for C4/ncu evidence.
- A6 remains a separate, larger attention-projection prologue project.
- The next code sprint should prefer a measured, non-HC-current launch target
  or return to C1 full-capture mechanics rather than another blind HC fusion.
