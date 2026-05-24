---
sprint: 314
title: TP/EP True-Attention Raw-Window Read Gate
status: complete
started: 2026-05-24
branch: claude-takeover
---

# Sprint 314 - TP/EP True-Attention Raw-Window Read Gate

## Goal

Replace Sprint 313's one-row raw-SWA diagnostic read with an executable
sliding raw-window attention read that consumes the raw rows populated by a
resident multi-step TP/EP token-major run.

## Scope

This sprint is TP/EP-only. It does not modify PP/layer-split paths and does
not promote attention output into the hidden state. The purpose is to make the
raw-window read semantics executable and measurable before adding compressed
KV/indexer rows and `attn_output_a -> attn_output_b`.

## Implementation Plan

1. Add a separate raw-window gate:

```text
binary flag:  --true-ds4-attention-raw-window-gate
launcher env: DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
```

2. Keep the Sprint 313 one-row gate intact for comparison.
3. In token-major all-layer runs, pass the number of valid rows populated by
   the current resident process:

```text
step 0 -> valid_rows = 1
step 1 -> valid_rows = 2
step 2 -> valid_rows = 3
step 3 -> valid_rows = 4
```

4. Score the valid raw rows plus `attn_sinks`, run a stable softmax, and emit
   the per-head value read into the existing diagnostic `d_attn_heads` buffer.

## Definition of Done

- The TP/EP binary accepts `--true-ds4-attention-raw-window-gate`.
  Complete.
- The launcher validates and forwards
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1`. Complete.
- The raw-window gate implies raw-read, state, projection, residency, and HC
  carry gates. Complete.
- A V100 `32` slot / `256K` / `4` step gate completes all 43 layers.
  Complete.
- The V100 log shows raw-window pass lines with `valid_rows` reaching `4`.
  Complete.
- The run has zero `FAIL` lines. Complete.
- Outcome and next correctness gap are recorded in this sprint doc, the temp
  status report, and `docs/sprints/VISION.md`. Complete.

## V100 Evidence

Logs:

```text
logs/from-cluster/sprint314-attn-raw-window/build.log
logs/from-cluster/sprint314-attn-raw-window/run.log
```

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Gated `32` slot / `256K` / `4` step smoke:

```text
tp_ep_true_attention_projection_prefix: 172
tp_ep_true_attention_state_update:      172
tp_ep_true_attention_raw_window:        172
FAIL lines:                               0

valid_rows:
  1: 43
  2: 43
  3: 43
  4: 43
```

Token-major scaffold:

```text
steps:                4
layers:               43
pass_invocations:     172
slots:                32
ctx:                  262144
sum_decode_ms:        3602.231959
ms_per_token:         900.557990
projected_slot_tok/s: 35.533525
sum_ep_ms:            2208.283030
sum_hc_current_ms:    1077.010430
sum_final_hc_ms:      96.197564
wall_ms:              6489.082791
result:               PASS
```

## Risks

- The raw window is still diagnostic because compressed-KV rows, ratio-4
  indexer rows, RoPE/inverse-RoPE, and attention output projection are not in
  this gate.
- Early-layer saturation may remain. That is acceptable for this sprint if the
  gate is finite and structurally executes; it remains a correctness blocker
  before trusted generation.

## Outcome

The TP/EP runtime now has separate one-row and sliding raw-window attention
diagnostic gates. The raw-window gate consumes rows populated by the resident
token-major process and proves `valid_rows=1..4` execution across all 43
layers at the target `32` slot / `256K` shape.

The generated text remains untrusted. The raw-window output is finite but
inherits the early-layer `65504` saturation seen in Sprint 312 and Sprint 313.

## Next Sprint

Move from raw-window-only diagnostics to the remaining reference attention
pieces:

```text
q-head RoPE / inverse-RoPE
compressed-KV row selection and read
ratio-4 indexer selection
raw + compressed score merge
attn_output_a -> attn_output_b
```

The saturation should be isolated before feeding this attention output into the
next hidden state.
