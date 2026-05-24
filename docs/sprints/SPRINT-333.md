---
sprint: 333
title: TP/EP Raw-SWA Physical Row Kind
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 333 - TP/EP Raw-SWA Physical Row Kind

## Goal

Correct the typed raw-SWA full-layer gate so it writes the physical 128-row
raw-SWA ring, not the compressed long-attention row region on ratio layers.

## Why This Sprint

During Sprint 333 planning for compressed/indexer typed KV, the runtime row
semantics showed that `DS4_V100_TP_KV_ROW_ATTN` maps to:

- `position % 128` on SWA-only layers,
- `128 + position / ratio` on ratio layers.

That is correct for long compressed attention rows, but it is not correct for
raw-SWA on ratio layers. Sprint 332 therefore proved the device store/load
plumbing in the full-layer harness, but it used the compressed attention row
address for raw-SWA on layers with ratio `4` or `128`.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint only corrects raw-SWA
physical addressing; compressed attention and indexer row integration remains
next.

## Definition of Done

- [x] Add an explicit raw-SWA row kind to the TP runtime API.
- [x] Keep existing `ATTN` semantics unchanged for compressed attention rows.
- [x] Update the runtime smoke CLI to validate `attn_raw`.
- [x] Switch the full-layer typed raw-SWA gate to the raw row kind.
- [x] Validate that `attn_raw` maps to `position % 128` while `attn` maps to
      the compressed long-attention row on a ratio-4 layer.
- [x] Validate the all-layer typed raw-SWA gate with `32` slots / `256K`.

## Outcome

Added `DS4_V100_TP_KV_ROW_ATTN_RAW`.

Runtime row semantics are now explicit:

| Kind | Ratio-0 layer | Ratio layer |
|---|---:|---:|
| `DS4_V100_TP_KV_ROW_ATTN_RAW` | `position % 128` | `position % 128` |
| `DS4_V100_TP_KV_ROW_ATTN` | `position % 128` | `128 + position / ratio` |
| `DS4_V100_TP_KV_ROW_INDEXER` | unavailable | `position / 4` on ratio-4 layers |

The full-layer `--true-ds4-attention-typed-kv-raw-gate` now uses
`DS4_V100_TP_KV_ROW_ATTN_RAW`.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-runtime-smoke tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

Runtime row-view contrast at `32` slots / `256K`, layer `2`, slot `31`,
position `262140`:

| Kind | Physical row | Bad values | Max abs | Result |
|---|---:|---:|---:|---|
| `attn_raw` | `124` | `0` | `0.000000000` | PASS |
| `attn` | `65663` | `0` | `0.000000000` | PASS |

Corrected all-layer shared-state validation:

```text
tp_ep_true_attention_typed_kv_raw lines: 43
first typed line: physical_row 124 raw_row 124
tp_ep_all_layer_scaffold ... pass_layers 43 ... PASS
projected_slot_step_tok_s 72.313683
sum_decode_ms_per_token 442.516531
```

Artifacts:

- `logs/from-cluster/sprint333-raw-swa-row-kind/cluster/device-row-attn-raw-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint333-raw-swa-row-kind/cluster/device-row-attn-compressed-layer2-slot31-pos262140.log`
- `logs/from-cluster/sprint333-raw-swa-row-kind/cluster/alllayers-typed-raw-window-s32-pos262140.log`

## Next Step

Now that raw-SWA addressing is correct, extend typed KV integration to
compressed attention rows and ratio-4 indexer rows using the existing `ATTN`
and `INDEXER` row kinds.
