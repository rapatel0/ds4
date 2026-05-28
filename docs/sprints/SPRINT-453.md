# Sprint 453: Promote Router+FFN Rank-Major Defaults

## Objective

Stay TP/EP only and make the Sprint 452 positive router+FFN rank-major bundle
the appliance launcher default, while preserving explicit opt-out controls.

Promoted defaults:

```text
DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=1
DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=1
```

## Implementation Plan

1. Change launcher default values in `tools/ds4-v100-run-appliance.sh`.
2. Update `deploy/v100/ds4-v100-appliance.env.example` so operator-visible
   defaults match the launcher.
3. Keep the existing normalization and dependency behavior:
   enabling model-router rank-major also forces routed-FFN rank-major input.
4. Validate with shell syntax, diff checks, and `--print-command` selection.

## Validation

- `bash -n tools/ds4-v100-run-appliance.sh`
- `git diff --check`
- `tools/ds4-v100-run-appliance.sh --print-command` includes both promoted
  gates by default.
- Explicit opt-out with both env vars set to `0` removes both gates.

## Decision Rule

If default selection and opt-out both work, Sprint 453 is complete.

## Outcome

Implemented:

- `tools/ds4-v100-run-appliance.sh` now defaults:

  ```text
  DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=1
  DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=1
  ```

- When model-router rank-major is enabled, the launcher also forces:

  ```text
  DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=1
  ```

  This matches the Sprint 452 validated serving bundle.

- `deploy/v100/ds4-v100-appliance.env.example` now documents the promoted
  defaults and keeps the env vars operator-visible for rollback.

Validation:

- `bash -n tools/ds4-v100-run-appliance.sh`: pass
- `git diff --check`: pass
- Remote `--print-command` with `DS4_V100_SERVE_MODE=tp-ep`,
  `DS4_V100_SLOTS=28`, and `DS4_V100_CTX=262144` includes:

  ```text
  --routed-ffn-rank-major-input-gate
  --model-router-rank-major-logits-gate
  --post-attention-fixed-capacity-route-plan-gate
  ```

- Explicit opt-out:

  ```text
  DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=0
  DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=0
  ```

  removes all three promoted gates from the printed command.

Target-shape confirmation:

- Artifact: `/localpool/ds4/workspace/logs/s453-router-ffn-rankmajor-s32`
- Shape: `32` requests / `32` slots / `256K` context / `4` generated tokens
- Readiness: control and candidate passed
- Response parity: `32/32`
- Server generated decode: `33.891610 -> 34.708926` tok/s (`1.0241x`)
- Server continuation decode: `33.840490 -> 34.611365` tok/s (`1.0228x`)
- Client generated tok/s: `5.037627 -> 5.135950` (`1.0195x`)
- Average GPU utilization: `11.76% -> 12.31%` (`1.0472x`)
- Minimum free VRAM: `2352 -> 2502 MiB`

## Decision

Sprint 453 is complete. The promoted router+FFN rank-major bundle is now the
TP/EP appliance launcher default with an explicit rollback path. The 32-slot
confirmation keeps the promotion aligned with the target serving shape, not only
the 28-slot admission tier.
