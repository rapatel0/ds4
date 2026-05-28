# Sprint 480 - Pattern-A Relaxed-Gate Promotion

## Objective

Apply the relaxed arithmetic-change policy from `TEMP_PARITY_POLICY.md` to the
existing Pattern-A evidence and promote only the candidates whose existing
artifacts satisfy the agreement gate.

Primary gate:

- selected-token agreement >= `0.99`
- generated-sequence agreement >= `0.99`
- selected-logit relative error is advisory only

## Completed

- Updated `tools/ds4-v100-http-response-tolerance.py` so `pass` is governed by
  HTTP/parse/file symmetry, pair count, selected-token agreement, and
  generated-sequence agreement. The selected-logit relative-error field remains
  in the JSON as advisory diagnostics.
- Promoted A3 router all-reduce in the appliance defaults:
  `DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS=1`.
- Changed EP compose ReduceScatter default to `auto`, resolving to enabled only
  for compatible non-compact FP32 compose and disabled for compact-route
  serving.
- Regenerated policy-current summaries from existing artifacts:
  `/tmp/s481-a3-relaxed-tolerance.json`,
  `/tmp/s481-ep-rs-relaxed-tolerance.json`, and
  `/tmp/s481-a6-relaxed-tolerance.json`.

## Evidence

| Candidate | Artifact | Selected-token agreement | Sequence agreement | Advisory relerr | Decision |
|---|---|---:|---:|---:|---|
| A3 router all-reduce | `/workspace/s480-a3-router-allreduce-tolerance` | `1.0` | `1.0` | `0.025157711827123192` | Promote |
| EP compose ReduceScatter, non-compact FP32 | `/workspace/s480-ep-reducescatter-tolerance` | `1.0` | `1.0` | `7.054008547965787e-05` | Promote for non-compact FP32 |
| A6 rank-local attention projection input | `/workspace/s478-a6-tolerance` | `0.03125` | `0.03125` | `0.08766228933928177` | Reject |

## A2 Reassessment

A2 mix/RMS all-reduce remains default-on from Sprint 478 via
`DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=1`.

An attempted current A/B control with A2 disabled is not used as evidence:
the old A2-off serving path failed internally at layer 12 with `rc 5` after
serving only 64 responses. No candidate leg from that attempt is accepted and
no additional A2 variants were run after reassessing the no-rerun rule.

## Post-Promotion Sanity

The only completed full serving run in this sprint is a sanity check, not a
candidate re-gate.

- Artifact: `/workspace/s481-pattern-a-post-promotion-sanity`
- Local summary: `/tmp/s481-pattern-a-post-promotion-sanity/summary.json`
- Shape: `32` slots / `256K` / `256` requests / `64` tokens
- HTTP 200: `256/256`
- Generated-token check: `256/256`
- Total generated tokens: `16384`
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- Average generated decode tok/s: `38.179785109375`
- Average continuation decode tok/s: `38.249982609375`

## Validation

- `python3 -m py_compile tools/ds4-v100-http-response-tolerance.py tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py`
- `bash -n tools/ds4-v100-run-appliance.sh`
- `git diff --check -- tools/ds4-v100-run-appliance.sh deploy/v100/ds4-v100-appliance.env.example tools/ds4-v100-http-response-tolerance.py`
- Pod launcher checks:
  - compact-route default: A2 and A3 flags present, dense ReduceScatter absent;
  - non-compact FP32: A2, A3, and dense ReduceScatter flags present.

## Decision

A3 is default-on. EP compose ReduceScatter is default-aligned for non-compact
FP32 and remains disabled for compact-route serving. A2 remains default-on from
Sprint 478. A6 remains rejected as a real correctness failure.
