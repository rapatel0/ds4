---
sprint: 320
title: TP/EP True Attention Output Projection Gate
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 320 - TP/EP True Attention Output Projection Gate

## Goal

Add a TP/EP-only diagnostic gate that turns the true-attention head read into
real DS4 attention output projection shards.

## Scope

This sprint stays entirely on the separate TP/EP path. It does not touch
PP/layer-scheduled code. It also does not promote the attention output into
the hidden state yet; that is a follow-on step after the output tensors are
materialized and validated.

## Why This Sprint

Sprint 319 proved that the askable HTTP parity path still returns the wrong
token after the reduction fix. The remaining gap is graph semantics. The
current true-attention gates can project q/kv, write raw SWA rows, and read
attention heads, but the read result is not yet pushed through:

```text
attn_output_a.weight
attn_output_b.weight
```

Without that output projection, hidden-state promotion would still be wiring
a diagnostic tensor, not a DS4 attention result.

## Implementation

- Add `--true-ds4-attention-output-gate`.
- Make it imply raw-window/read/state/projection residency gates.
- Keep `attn_output_a` rank-local: each TP rank consumes its local
  `[slots][4096]` attention heads.
- Allocate per-rank staging for the full `attn_output_a` intermediate:
  `[slots][8192]`.
- Run real `attn_output_a.weight` through the resident dense path.
- Gather `attn_output_a` shards into the full 8192-wide intermediate on each
  rank.
- Run real `attn_output_b.weight` through the resident dense path.
- Emit per-layer PASS rows with output maxima and timing.

## Definition of Done

- The new gate parses and appears in usage.
  Complete.
- The gate has independent buffers and does not reuse undersized hidden-state
  buffers.
  Complete.
- The gate runs after raw-window attention in the all-layer token-major loop.
  Complete.
- V100 build passes.
  Complete.
- A `32` slot / `256K` / `4` step V100 gate produces 172
  `tp_ep_true_attention_output_projection` rows and zero `FAIL` rows.
  Structurally complete: the final scaffold reports `172` pass invocations and
  `PASS`, with zero `FAIL`/`bad_shape` lines. The copied log has `171`
  standalone anchored output-projection rows because one layer-2 row was lost
  as a grep-able standalone line in the high-volume stdout stream.
- Sprint doc, temp status report, and vision are updated with artifacts.
  Complete.

## V100 Evidence

Artifacts:

```text
logs/from-cluster/sprint320-true-attention-output-gate/cluster/build.log
logs/from-cluster/sprint320-true-attention-output-gate/cluster/run.log
```

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Gate command shape:

```text
slots:        32
ctx:          262144
decode_steps: 4
layers:       43
gate:         --true-ds4-attention-output-gate
```

Final scaffold:

```text
tp_ep_token_major_scaffold:
  steps:                  4
  layers:                 43
  pass_invocations:       172
  slots:                  32
  ctx:                    262144
  sum_decode_ms:          5685.874149
  ms_per_token:           1421.468537
  projected_slot_tok/s:   22.511930
  sum_ep_ms:              2122.794070
  sum_hc_current_input_ms:3269.508386
  wall_ms:                8509.907562
  checksum:               2999994000
  result:                 PASS
```

Output projection rows:

```text
standalone anchored rows: 171
structural pass rows:     172
FAIL/bad_shape rows:      0
max heads_max:            7.34393787
max out_a_max:            6.79435921
max out_b_max:            49.8094406
max output ms:            30.495427
```

Representative output row:

```text
tp_ep_true_attention_output_projection layer 42 slots 32
  head_input_cols:    4096
  out_a_cols:         8192
  out_b_shard_cols:   512
  heads_max:          4.99894285
  out_a_max:          2.40393782
  out_b_max:          26.4566479
  bad counts:         0
  ms:                 13.275515
  result:             PASS
```

## Outcome

The TP/EP true-attention path now materializes the DS4 attention output
projection sequence:

```text
rank-local raw/window heads [slots][4096]
  -> attn_output_a.weight
  -> gathered [slots][8192]
  -> attn_output_b.weight
  -> rank-local hidden shard [slots][512]
```

The first attempted topology assumed `attn_output_a.cols=32768`; the pack
proved the correct TP layout is rank-local `4096`-wide input. That is better
for TP because it avoids a pre-`attn_output_a` all-gather.

The output is still diagnostic. The next semantic step is hidden-state
promotion: consume these `attn_output_b` shards in the residual/FFN path
instead of the simplified bridge.
