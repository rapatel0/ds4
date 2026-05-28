# TEMP Status Report 453

## Current Focus

TP/EP appliance launcher promotion for the router+FFN rank-major bundle.

## Implemented

Launcher defaults now enable:

```text
DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=1
DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=1
```

When model-router rank-major is enabled, the launcher also forces:

```text
DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=1
```

That matches the Sprint 452 validated 28-slot / 256K bundle.

The follow-up target-shape run also passed:

```text
artifact: /localpool/ds4/workspace/logs/s453-router-ffn-rankmajor-s32
shape:    32 requests / 32 slots / 256K context / 4 generated tokens
parity:   32/32 response pairs
decode:   33.891610 -> 34.708926 server generated tok/s
cont:     33.840490 -> 34.611365 server continuation tok/s
client:   5.037627 -> 5.135950 generated tok/s
gpu util: 11.76% -> 12.31%
vram:     2352 -> 2502 MiB minimum free
```

Updated files:

- `tools/ds4-v100-run-appliance.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `docs/sprints/VISION.md`
- `docs/sprints/SPRINT-453.md`

## Validation

- `bash -n tools/ds4-v100-run-appliance.sh`: pass
- `git diff --check`: pass
- Remote `--print-command` for TP/EP 28-slot serving includes:
  - `--routed-ffn-rank-major-input-gate`
  - `--model-router-rank-major-logits-gate`
  - `--post-attention-fixed-capacity-route-plan-gate`
- Explicit opt-out with both rank-major env vars set to `0` removes those gates.
- Same-binary HTTP A/B at the 32-slot target shape passed readiness and response
  parity, then improved server decode, client throughput, GPU utilization, and
  minimum free VRAM.

## Cluster State

No active DS4 GPU jobs remained after validation. All eight GPUs reported
`0 MiB` used by DS4 processes.
