# SPRINT-025 Report

## Verdict

`EXTEND`

## Summary

Sprint 025 added the scheduler-owned gpu7 output-head path and a prompt-replay
selected-token smoke. The path is runnable on V100 and produces finite logits
and a selected token. It does not yet match the official/source oracle selected
token for `short_reasoning_plain`.

## Implementation Notes

- `ds4_v100_stage_scheduler_select_token` now performs:
  - HC RMS norm over `[4 x 4096]`
  - `hc_head_fn` F32 projection
  - `hc_head_scale`/`hc_head_base` sigmoid weights
  - HC weighted sum to `[4096]`
  - `output_norm.weight`
  - BF16 `output.weight` matmul
  - host top-1 selection with non-finite checks
- `tests/cuda_v100_selected_token_smoke.c` replays the tokenized
  `short_reasoning_plain` prompt through all 43 layers before selecting.
- `--expected-token-hex` enables explicit oracle comparison. Without it, the
  smoke validates output-head execution and reports the selected token.

## V100 Evidence

Output-head execution smoke:

```text
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=271 logit=20.014683 expected=none ok
```

Explicit oracle comparison:

```text
cuda_v100_selected_token_smoke: selected token mismatch expected=3136 got=0a0a token=271 logit=20.014683
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=271 logit=20.014683 expected=3136 FAIL
```

## Readiness

The appliance is still not ready for use. `real_model_selected_token` remains a
blocking correctness gap.

## Interpretation

The new output-head adapter is not the current blocker. It executes and emits a
plausible finite top-1 token. The mismatch is upstream numerical correctness:
prompt replay through the 43-layer body diverges before or by the output-head
comparison.
