# TEMP Status Report 031

Date: 2026-05-24

## Current Topline

Sprint 319 reran the official TP/EP HTTP reference parity gate after the Sprint
318 block-reduction fix.

The short vector still fails, but the failure changed:

```text
Expected text: 16
Old TP/EP result before reduction fix: ICC / token 95933
Current TP/EP result: )Skip / token 83480
```

So the corrected reduction helper affects live serving output, but TP/EP is
still not model-correct.

## V100 Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/
```

Parity summary:

```text
case:                  short_reasoning_plain
ctx:                   4096
steps:                 1
expected_hex:          3136
expected_text:         16
actual_hex:            29536b6970
actual_text:           )Skip
match:                 false
tokenizer_ready:       1
prompt_tokens:         18
prompt_prefill_tokens: 17
generated_token_ids:   1
generated_sequence:    [83480]
slot_position:         100018
wall_tok_s:            193.154852
decode_tok_s:          303.200535
parity_exit:           1
```

Server cleanup:

```text
post-kill-pgrep: CLEAN
```

## Interpretation

This is a real end-to-end parity signal from the askable HTTP path, not an
isolated kernel smoke. The path can tokenize, prefill, generate, decode text,
and return an OpenAI-style chat response. It is still semantically wrong for
the reference vector.

The next sprint should continue true DS4 attention semantics:

```text
compressed KV/indexer row selection
raw + compressed attention score merge
attn_output_a -> attn_output_b
hidden-state promotion
```

No PP/layer-scheduled work should be resumed.
