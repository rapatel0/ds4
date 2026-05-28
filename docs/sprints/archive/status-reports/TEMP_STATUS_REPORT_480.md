# TEMP Status Report 480 - Pattern-A Relaxed-Gate Promotion

## Status

Pattern-A promotion is complete under the relaxed arithmetic-change policy in
`TEMP_PARITY_POLICY.md`.

Primary gate:

- selected-token agreement >= `0.99`
- generated-sequence agreement >= `0.99`
- max selected-logit relative error is advisory only

No completed candidate re-gate is used as promotion evidence in this sprint.
The promotion decisions below reclassify existing artifacts under the current
policy.

## Promotions

### A3 router all-reduce

Promoted on existing s480 evidence.

- Existing artifact: `/workspace/s480-a3-router-allreduce-tolerance`
- Policy-current local summary: `/tmp/s481-a3-relaxed-tolerance.json`
- selected-token agreement: `32/32 = 1.0`
- generated-sequence agreement: `1.0`
- max selected-logit relative error: `0.025157711827123192` advisory only
- relaxed pass: `true`

Launcher/env default:

- `DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS=1`
- `--model-router-allreduce-logits-gate` is now present by default in the
  promoted TP/EP appliance command.

### EP compose ReduceScatter, non-compact FP32 only

Promoted/aligned on existing s480 evidence.

- Existing artifact: `/workspace/s480-ep-reducescatter-tolerance`
- Policy-current local summary: `/tmp/s481-ep-rs-relaxed-tolerance.json`
- selected-token agreement: `32/32 = 1.0`
- generated-sequence agreement: `1.0`
- max selected-logit relative error: `7.054008547965787e-05` advisory only
- relaxed pass: `true`

Launcher/env default:

- `DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE=auto`
- `auto` resolves to enabled only when final compose is non-compact FP32:
  `DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=0` and
  `DS4_V100_TP_EP_RETURN_FP16=0`.
- Compact-route serving remains off for dense reduce-scatter by design.

### A2 mix/RMS all-reduce

A2 remains the promoted default from Sprint 478:

- `DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=1`
- `--tp-hc-current-allreduce-gate` remains present by default.

Reassessment: the attempted current A/B old-control leg is not valid promotion
evidence. With `DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=0` and A3 held off, the
old control path failed internally at layer 12 with `rc 5` after serving only
64 responses. No candidate leg was accepted from that run, and no additional
A2 variants were run after reassessing the no-rerun rule.

## Rejection

### A6 rank-local attention projection input

A6 remains rejected. Existing evidence still fails the relaxed gate.

- Existing artifact: `/workspace/s478-a6-tolerance`
- Policy-current local summary: `/tmp/s481-a6-relaxed-tolerance.json`
- selected-token agreement: `1/32 = 0.03125`
- generated-sequence agreement: `0.03125`
- max selected-logit relative error: `0.08766228933928177` advisory only
- relaxed pass: `false`

Keep:

- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT=0`

## Post-Promotion Sanity

This was a serving sanity run, not a candidate re-gate.

- Artifact: `/workspace/s481-pattern-a-post-promotion-sanity`
- Local summary: `/tmp/s481-pattern-a-post-promotion-sanity/summary.json`
- Shape: `32` slots / `256K` / `256` requests / `64` tokens
- HTTP: `256/256` status 200
- Generated-token check: `256/256` emitted `64` tokens
- Total generated tokens: `16384`
- Total continuation tokens: `16128`
- Peer-copy accounting: enabled
- Peer-copy reject SYS: enabled
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- Average generated decode tok/s: `38.179785109375`
- Average continuation decode tok/s: `38.249982609375`

## Validation

- `python3 -m py_compile tools/ds4-v100-http-response-tolerance.py tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py`
- `bash -n tools/ds4-v100-run-appliance.sh`
- `git diff --check -- tools/ds4-v100-run-appliance.sh deploy/v100/ds4-v100-appliance.env.example tools/ds4-v100-http-response-tolerance.py`
- Pod launcher checks confirmed default compact route includes A2/A3 and does
  not include dense ReduceScatter, while non-compact FP32 includes
  `--nccl-reduce-scatter-compose-gate`.

## Decision

Promote A3 and non-compact FP32 EP compose ReduceScatter under the relaxed
agreement-only policy. Keep A2 default-on from Sprint 478. Keep A6 rejected.
Do not use the invalid A2-off control attempt as evidence.
