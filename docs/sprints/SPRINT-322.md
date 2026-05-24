---
sprint: 322
title: TP/EP Post-Attention FFN Input Promotion
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 322 - TP/EP Post-Attention FFN Input Promotion

## Goal

Move the TP/EP layer semantics one step closer to DeepSeek V4 by feeding FFN
norm, router selection, shared FFN, and routed experts from the post-attention
residual:

```text
post_attn = current_hidden + attn_output_b(attention_heads)
ffn_input = rms_norm(post_attn, ffn_norm.weight)
```

## Scope

This sprint is TP/EP-only. It does not touch PP/layer-split variants, does not
add MTP, and does not optimize kernels. The work is a correctness bridge for
the current resident TP/EP HTTP path.

## Why This Sprint

Sprint 321 proved the true-attention output projection reaches the live HTTP
path, changing the official parity token from the Sprint 319 token but still
failing reference parity. Inspection shows the remaining ordering bug: FFN
norm/router/shared/routed inputs are prepared before the attention output is
available, so FFN still sees the pre-attention bridge tensor.

The next useful semantic gate is therefore to materialize the attention
residual before FFN routing and expert execution.

## Implementation

- Add a TP/EP-only option:

```text
--true-ds4-post-attention-ffn-input-gate
```

- The option implies the current true-attention output gate and the existing
FFN semantic gates.
- Allocate a per-rank `post_attn_shard` buffer.
- After `run_true_ds4_attention_output_projection`, compute rank-local:

```text
post_attn_shard[slot, local_h] =
  current_shard[slot, local_h] + attn_output_b_shard[slot, local_h]
```

- Gather those shards into the shared full hidden buffer on device 0.
- Recompute `ffn_normed` from the post-attention full hidden.
- Recompute model-router routes from the post-attention `ffn_normed`.
- Repack routed expert inputs from post-attention `ffn_normed`.
- Fill shared-FFN gate/up inputs from post-attention `ffn_normed`.
- Leave final compose algebra unchanged:

```text
next_hidden = current_hidden + attn_output_b + shared_ffn + routed_ffn
```

That keeps the residual addition equivalent while changing the FFN inputs to
the correct post-attention tensor.

## Definition of Done

- Local build passes for `tools/ds4-v100-tp-ep-full-layer-smoke`.
  Not applicable on the laptop host because the target requires CUDA; the V100
  pod build passed with the existing unused RMSNorm warning only.
- V100 32-slot / 256K smoke with the new gate completes all 43 layers with no
  finite/bad-shape/runtime failure rows.
  Complete.
- HTTP reference parity gate runs for `short_reasoning_plain` with:
  - `--model-router-routes`
  - `--routed-ffn-norm-input-gate`
  - `--true-shared-ffn-gate`
  - `--true-ds4-post-attention-ffn-input-gate`
  - `--diagnostic-output-head`
  Complete.
- Cluster artifacts are captured under
  `logs/from-cluster/sprint322-post-attention-ffn-input/`.
  Complete.
- Temp status and vision are updated with the result.
  Complete.

## Implementation Notes

The new gate is implemented in
`tools/ds4-v100-tp-ep-full-layer-smoke.cu` as
`--true-ds4-post-attention-ffn-input-gate`.

It implies the current true-attention output and FFN semantic gates, allocates
per-rank `d_post_attn_shard`, and runs:

```text
post_attn_shard = current_shard + attn_output_b_shard
gather post_attn_shard -> hc->d_current_full
hc->d_ffn_normed = rms_norm(post_attn_full, ffn_norm.weight)
router_select_topk(ffn_normed)
pack routed expert inputs from ffn_normed
fill shared gate/up inputs from ffn_normed
```

Final compose remains algebraically unchanged:

```text
next_hidden = current_hidden + attn_output_b + shared_ffn + routed_ffn
```

## V100 Evidence

Artifacts:

```text
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/smoke-command.txt
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/smoke.stdout
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/smoke.stderr
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/server-command.txt
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/server.stdout
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/server.stderr
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/parity-summary.json
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/parity.stdout
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/parity.stderr
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/parity.exit
logs/from-cluster/sprint322-post-attention-ffn-input/cluster/post-kill-pgrep.txt
```

Smoke result:

```text
post_attention_rows:      43
layers:                   43
slots:                    32
ctx:                      262144
pass_invocations:         43
sum_decode_ms:            1690.550157
projected_slot_step_tok/s:18.928749
sum_hc_current_input_ms:  1111.077977
result:                   PASS
```

HTTP parity result:

```text
case:                    short_reasoning_plain
ctx:                     4096
steps:                   1
expected_hex:            3136
expected_text:           16
actual_hex:              6d657265
actual_text:             mere
match:                   false
tokenizer_ready:         1
prompt_tokens:           18
prompt_prefill_tokens:   17
generated_token_sequence:[88445]
slot_position:           100018
wall_tok_s:              21.484145
decode_tok_s:            22.443315
parity_exit:             1
```

Server cleanup:

```text
post-kill-pgrep: CLEAN
```

## Interpretation

The reference vector still fails:

```text
Reference expected:        16
Sprint 319 reduction fix:  )Skip / token 83480
Sprint 321 output gate:    urf   / token 64906
Sprint 322 post-attn FFN:  mere  / token 88445
```

This is useful progress. The output changed again, so post-attention FFN input
promotion reaches the live serving path. The remaining highest-value semantic
gap is no longer FFN input ordering; it is true compressed-KV/indexer
attention and the raw+compressed attention merge.

## Risks

- This may change output again without closing parity, because compressed-KV
  and indexer attention are still incomplete.
- The first implementation intentionally adds copies and a second FFN input
  preparation pass. That is acceptable for correctness; optimization comes
  after semantic parity.
- Recomputing model-router routes after attention changes route plans. Any
  latent expert-binding issue may become visible again.
