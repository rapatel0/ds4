---
sprint: 319
title: TP/EP Reference Parity Rerun After Reduction Fix
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 319 - TP/EP Reference Parity Rerun After Reduction Fix

## Goal

Rerun the official TP/EP HTTP reference-vector parity gate after Sprint 318's
block-reduction broadcast fix.

## Scope

This sprint is TP/EP-only. It does not change PP/layer-split code and does
not optimize kernels. The purpose is to get current end-to-end evidence from
the askable HTTP path before choosing the next semantic implementation target.

## Why This Sprint

Sprint 318 removed an artificial true-attention saturation source:

```text
KV norm reference drift: thousands -> ~1e-6
raw-SWA max:             65504 -> ~6.29
```

The next question is whether that corrected math changes the live reference
parity result, or whether the remaining failure is still missing DS4 graph
semantics such as compressed KV/indexer selection and attention-output
promotion.

## Definition of Done

- Start the TP/EP HTTP server from the current V100 binary.
  Complete.
- Run `tools/ds4-v100-tp-ep-reference-parity.py` against the official
  `short_reasoning_plain` vector.
  Complete.
- Capture server, parity, and command artifacts under
  `logs/from-cluster/sprint319-reference-parity-after-reduction-fix/`.
  Complete.
- Record exact expected text, actual text, token IDs, and pass/fail result.
  Complete.
- Kill the HTTP server after the gate and verify no parity server remains.
  Complete.
- Update the temp status report and vision with the result.
  Complete.

## V100 Evidence

Artifacts:

```text
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/command.txt
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/startup.env
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/server.stdout
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/server.stderr
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/parity-summary.json
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/parity.stdout
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/parity.stderr
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/parity.exit
logs/from-cluster/sprint319-reference-parity-after-reduction-fix/cluster/post-kill-pgrep.txt
```

Parity result:

```text
case:                    short_reasoning_plain
ctx:                     4096
steps:                   1
expected_hex:            3136
expected_text:           16
actual_hex:              29536b6970
actual_text:             )Skip
match:                   false
tokenizer_ready:         1
prompt_tokens:           18
prompt_prefill_tokens:   17
generated_token_sequence:[83480]
slot_position:           100018
wall_tok_s:              193.154852
decode_tok_s:            303.200535
parity_exit:             1
```

The server was stopped after the parity gate. The post-kill process check is:

```text
CLEAN
```

## Outcome

The TP/EP HTTP path is still not model-correct. The reduction fix changed the
live output from Sprint 307's old `ICC` / token `95933` failure to `)Skip` /
token `83480`, but the official vector still expects `16`.

This means Sprint 318 did affect the end-to-end serving path, but the remaining
blocker is still graph semantics. The next implementation work should continue
the true DS4 attention sequence rather than kernel micro-optimization.

## Expected Outcome

Passing the short vector would make the next sprint a broader parity suite.
Failing the vector is still useful if it shows the new failure token/text after
the reduction fix; in that case the next sprint continues true DS4 attention
semantics:

```text
compressed KV/indexer row selection
raw + compressed attention merge
attn_output_a -> attn_output_b
hidden-state promotion
```

That is the actual result of this sprint.
