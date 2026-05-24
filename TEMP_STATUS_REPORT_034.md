# TEMP Status Report 034

Date: 2026-05-24

## Topline

Sprint 322 is implemented and validated on the V100 pod. The TP/EP runtime now
has an explicit post-attention FFN-input gate:

```text
--true-ds4-post-attention-ffn-input-gate
```

The gate materializes `current_hidden + attn_output_b`, recomputes FFN norm and
model-router routes from that post-attention tensor, repacks routed expert
inputs, and fills shared-FFN gate/up inputs from the same tensor.

## V100 Smoke

Command artifacts:

```text
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/smoke-command.txt
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/smoke.stdout
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/smoke.stderr
```

Result:

```text
slots:                     32
ctx:                       262144
layers:                    43
post_attention_rows:       43
pass_invocations:          43
sum_decode_ms:             1690.550157
projected_slot_step_tok/s: 18.928749
sum_hc_current_input_ms:   1111.077977
result:                    PASS
```

No finite, CUDA, bad-shape, or runtime failure rows were found in the smoke
artifacts.

## HTTP Parity

Command artifacts:

```text
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/server-command.txt
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/server.stdout
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/server.stderr
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/parity-summary.json
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/parity.stdout
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/parity.stderr
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/parity.exit
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/post-kill-pgrep.txt
```

Parity result:

```text
case:                    short_reasoning_plain
expected_text:           16
actual_text:             mere
generated_token_sequence:[88445]
match:                   false
wall_tok_s:              21.484145
decode_tok_s:            22.443315
post-kill-pgrep:         CLEAN
```

Progression:

```text
Sprint 307 baseline:       ICC   / token 95933
Sprint 319 reduction fix:  )Skip / token 83480
Sprint 321 output gate:    urf   / token 64906
Sprint 322 post-attn FFN:  mere  / token 88445
Reference expected:        16
```

## Assessment

The new path is active in live serving because the parity token changed again.
The next correctness target should be true compressed-KV/indexer attention and
raw+compressed attention merge. FFN input ordering is no longer the leading
known semantic gap.

## Current Repo State

Implemented files:

```text
tools/ds4-v100-tp-ep-full-layer-smoke.cu
docs/sprints/SPRINT-322.md
docs/sprints/VISION.md
TEMP_STATUS_REPORT_034.md
logs/from-cluster/sprint322-post-attention-ffn-input/
```
