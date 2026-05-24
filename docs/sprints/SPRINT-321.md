---
sprint: 321
title: TP/EP Reference Parity With True Attention Output
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 321 - TP/EP Reference Parity With True Attention Output

## Goal

Rerun the official TP/EP HTTP reference-vector parity gate with the Sprint 320
true-attention output projection gate enabled.

## Scope

This sprint is TP/EP-only. It does not optimize kernels and does not resume
PP/layer-split work. It measures the live HTTP path after adding the real
`attn_output_a -> attn_output_b` projection sequence.

## Why This Sprint

Sprint 320 proved the output projection gate is executable at the target
`32` slot / `256K` shape. Inspection shows the normal compose kernel consumes
`attn_op->d_out`, which the output gate now materializes from true-attention
heads. The next question is whether the official parity vector changes again
when the HTTP path runs with this semantic gate enabled.

This is not expected to fully pass yet because FFN routing/norm semantics are
still not strictly after the attention residual, and compressed-KV/indexer
attention is not implemented. But it is the correct end-to-end measurement
before hidden-state promotion.

## Definition of Done

- Start the TP/EP HTTP server from the current V100 binary with:
  - `--model-router-routes`
  - `--routed-ffn-norm-input-gate`
  - `--true-shared-ffn-gate`
  - `--true-ds4-attention-output-gate`
  - `--diagnostic-output-head`
  Complete. The first startup attempt also proved
  `--compact-route-compose` is incompatible with `--model-router-routes`; the
  recorded successful command removes compact route compose.
- Run `tools/ds4-v100-tp-ep-reference-parity.py` against
  `short_reasoning_plain`.
  Complete.
- Capture server, parity, startup, and command artifacts under
  `logs/from-cluster/sprint321-reference-parity-true-attention-output/`.
  Complete.
- Record expected text, actual text, token IDs, pass/fail, and throughput.
  Complete.
- Stop the HTTP server and verify no test process remains.
  Complete.
- Update temp status and vision with the result.
  Complete.

## V100 Evidence

Artifacts:

```text
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/command.txt
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/startup.env
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/server.stdout
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/server.stderr
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/parity-summary.json
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/parity.stdout
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/parity.stderr
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/parity.exit
logs/from-cluster/sprint321-reference-parity-true-attention-output/cluster/post-kill-pgrep.txt
```

Parity result:

```text
case:                    short_reasoning_plain
ctx:                     4096
steps:                   1
expected_hex:            3136
expected_text:           16
actual_hex:              757266
actual_text:             urf
match:                   false
tokenizer_ready:         1
prompt_tokens:           18
prompt_prefill_tokens:   17
generated_token_sequence:[64906]
slot_position:           100018
wall_tok_s:              23.926690
decode_tok_s:            25.093416
parity_exit:             1
```

The server was stopped after the parity gate:

```text
post-kill-pgrep: CLEAN
```

## Interpretation

The result changed again:

```text
Sprint 307 baseline:       ICC   / token 95933
Sprint 319 reduction fix:  )Skip / token 83480
Sprint 321 output gate:    urf   / token 64906
Reference expected:        16
```

That proves the true-attention output projection is active in the askable HTTP
path, but it still does not close semantic parity.

The likely next blocker is ordering: FFN norm/router/shared/routed FFN are
still fed from the pre-attention current hidden bridge, not from the
post-attention residual. The next sprint should promote `attn_output_b` into
current hidden before FFN routing/norm, then rerun this same parity gate.
