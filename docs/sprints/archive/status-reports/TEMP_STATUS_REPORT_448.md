# TEMP Status Report 448

## Current Focus

TP/EP serving correctness and performance. Sprint 448 closed the immediate
attention-rank-local correctness blocker by auditing the HC-current full buffer
and rerunning the attention-only HTTP A/B on the rebuilt binary.

## Implemented

- Added `--tp-hc-current-full-parity-gate`.
- Added `tp_ep_hc_current_full_rank_diff` diagnostics after HC-current NCCL
  allgather plus rank-major-to-slot-major conversion.
- Kept the attention projection half-input parity gate active in the same
  direct audit.
- Rebuilt `tools/ds4-v100-tp-ep-full-layer-smoke` on the V100 node at `sm_70`.

## Results

Direct audit:

- Artifact: `/localpool/ds4/workspace/logs/s448-hc-current-full-parity-direct`
- rc: `0`
- `tp_ep_hc_current_full_rank_diff`: `344` lines, `0` bad lines
- `tp_ep_attention_projection_input_diff`: `688` lines, `0` bad lines

Rebuilt HTTP attention-rank-local A/B:

- Artifact: `/localpool/ds4/workspace/logs/s447-http-attn-ranklocal-broadcast`
- Response parity: `4/4` matched
- First token: `71302` on both control and candidate

| Leg | Server generated decode tok/s | Server continuation decode tok/s | Client generated tok/s | GPU util avg |
|---|---:|---:|---:|---:|
| Control | 20.225169 | 20.265472 | 0.734972 | 9.907895% |
| Rank-local attention input | 18.383328 | 18.937739 | 0.691062 | 8.737500% |

Queued HTTP routed-FFN rank-major isolate:

- Artifact: `/localpool/ds4/workspace/logs/s447-http-ffn-rankmajor-isolate`
- Response parity: `4/4` matched
- First token: `71302` on both control and candidate

| Leg | Server generated decode tok/s | Client generated tok/s | GPU util avg |
|---|---:|---:|---:|
| Control | 18.172498 | 0.708136 | 8.593750% |
| Routed FFN rank-major input | 20.330467 | 0.739961 | 9.473684% |

## Assessment

The HC-current full-hidden buffers are not the remaining issue. The stale
source/buffer fix plus rebuild eliminates the half-input mismatch and restores
HTTP response parity for attention-rank-local input.

This gate is still not a performance promotion: at the reduced HTTP shape it is
slower despite being correct. The next TP/EP work should continue with broader
rank-major launch-reduction paths and only promote attention-rank-local input
when it is part of a net-positive serving bundle.

The FFN rank-major isolate remains a positive signal: it preserved parity and
improved server decode at the same reduced shape. That points the next sprint
toward a combined correctness-clean rank-major bundle that keeps the FFN win
without reintroducing the older attention-input token drift.

One harness issue remains: the HTTP A/B parent process sometimes stalls after a
profile leg completes and needs an explicit signal before starting the next
leg. The produced response artifacts are valid, but the handoff bug should be
fixed before using the A/B harness unattended for long runs.

## Cluster State

After the Sprint 448 runs and the queued FFN isolate, the V100 node reported no
active DS4 GPU jobs and all eight GPUs had `0 MiB` used by DS4 processes.
