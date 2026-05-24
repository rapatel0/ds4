# TEMP Status Report 033

Date: 2026-05-24

## Current Topline

Sprint 321 reran the official TP/EP HTTP reference parity vector with the
Sprint 320 true-attention output projection enabled.

It still fails, but the selected output changed:

```text
Reference expected:        16
Sprint 307 baseline:       ICC   / token 95933
Sprint 319 reduction fix:  )Skip / token 83480
Sprint 321 output gate:    urf   / token 64906
```

This proves the true-attention output projection reaches the live HTTP path,
but TP/EP is still not model-correct.

## V100 Evidence

Artifacts:

```text
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/
```

Parity summary:

```text
case:                  short_reasoning_plain
ctx:                   4096
steps:                 1
expected_hex:          3136
expected_text:         16
actual_hex:            757266
actual_text:           urf
match:                 false
tokenizer_ready:       1
prompt_tokens:         18
prompt_prefill_tokens: 17
generated_token_ids:   1
generated_sequence:    [64906]
slot_position:         100018
wall_tok_s:            23.926690
decode_tok_s:          25.093416
parity_exit:           1
```

Server cleanup:

```text
post-kill-pgrep: CLEAN
```

## Notes

The first Sprint 321 server attempt failed fast because
`--compact-route-compose` is incompatible with `--model-router-routes`. The
successful recorded command removes compact route compose.

## Next

Promote `attn_output_b` into the layer residual/current-hidden path before FFN
norm/router/shared/routed FFN. Then rerun this same reference parity vector.
