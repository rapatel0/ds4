# Sprint 293 - TP/EP HC Final-Expand Diagnostic

Date: 2026-05-23

## Goal

Replace the arbitrary TP/EP proxy HC row expansion with a diagnostic DS4-style
HC final-expand path that uses real layer `hc_ffn_*` control tensors while
staying entirely in the separate TP/EP codepath.

This is still diagnostic. It does not yet implement the full DS4 HC attention
pre/post and FFN pre/post sequence, prompt prefill, selected-token feedback, or
tokenizer text output.

## Implementation

- Added `--tp-hc-final-expand-gate` to
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The flag implies `--final-hc-carry-gate`.
- Added a resident `SharedHcControls` object:
  - loads real `blk.N.hc_ffn_fn`, `blk.N.hc_ffn_base`, and
    `blk.N.hc_ffn_scale` for all 43 layers;
  - keeps full-HC gather/norm/mix/split scratch on GPU0;
  - keeps per-rank split and scratch HC buffers resident.
- Added TP/EP-only kernels for:
  - F32 column-major HC-control projection from `[16384 x 24]`;
  - DS4 `hc4_split_one` split generation;
  - sharded HC expand over `[slots][4][512]`.
- The token-major TP/EP loop now seeds HC once, then applies real HC
  final-expand semantics per layer when the gate is enabled.
- Launcher and HTTP bench support:
  - `DS4_V100_TP_EP_HC_FINAL_EXPAND=1`
  - `--hc-final-expand`

## Definition of Done

- [x] TP/EP full-layer smoke builds on the V100 pod.
- [x] Direct 32-slot, 1-token, all-layer diagnostic output-head run passes with
  HC final-expand enabled.
- [x] Launcher-level HTTP completions run passes with HC final-expand enabled.
- [x] Full 32-concurrent launcher-level HTTP completions run forms one
  32-request coalesced batch and returns diagnostic selected-token metadata.
- [x] Status and vision record that this advances HC semantics but is not yet
  real DeepSeek text serving.

## Validation

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Direct 32-slot command:

```text
./tools/ds4-v100-tp-ep-full-layer-smoke \
  --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --contract /workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv \
  --tm-index /workspace/packs/ds4-appliance-full-tm-gated-s181/turbomind-pack-index.tsv \
  --lib /workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so \
  --slots 32 --decode-steps 1 \
  --fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose \
  --skip-descriptor-checks --skip-predecode-probes \
  --shared-expert-bindings --shared-dense-ops --overlap-ep-dense \
  --source-copy-schedule --skip-self-compose-copy --multi-copy-streams \
  --copy-event-compose --compact-route-compose \
  --token-major-all-layers --all-layers --serving-bench \
  --tp-hc-final-expand-gate --diagnostic-output-head
```

Direct result:

```text
tp_ep_hc_final_expand_shared layers=43 slots=32 control_bytes=67637796 PASS
tp_ep_token_major_scaffold sum_decode_ms=140.213387 projected_slot_step_tok_s=228.223572 sum_final_hc_ms=25.407638 PASS
tp_ep_diagnostic_output_head proxy_hc=0 total_ms=8.750574 projection_ms=7.686252 top1_ms=0.393437 first_token=122445 PASS
```

Launcher HTTP validation, 8 concurrent completion requests:

```text
endpoint=completions, tokens=1, requests=8, coalesced_batches=1,
coalesced_batch_max=8, status_200=8, generated_tok_s=26.805691,
generated_tok_s_decode=44.610688
```

Launcher HTTP validation, 32 concurrent completion requests:

```text
endpoint=completions, tokens=1, requests=32, coalesced_batches=1,
coalesced_batch_max=32, status_200=32, generated_tok_s=160.904882,
generated_tok_s_decode=271.342877
```

The 32-concurrent server run reported:

```text
tp_ep_token_major_scaffold sum_decode_ms=117.931970 projected_slot_step_tok_s=271.342877 sum_final_hc_ms=23.559238 PASS
tp_ep_diagnostic_output_head proxy_hc=0 total_ms=8.519947 projection_ms=7.564340 top1_ms=0.368505 first_token=122445 PASS
```

Evidence:

```text
logs/from-cluster/sprint293-tp-ep-hc-final-expand/cluster/
logs/from-cluster/sprint293-tp-ep-hc-final-expand-http/cluster/
logs/from-cluster/sprint293-tp-ep-hc-final-expand-http32/cluster/
```

## Decision

Keep `--tp-hc-final-expand-gate` as an opt-in diagnostic bridge. It materially
improves the TP/EP HC semantics compared with Sprint 292's arbitrary proxy row
expansion, but it is not enough to call the endpoint real DeepSeek serving.

The added cost is roughly `23-25 ms` per 43-layer token at 32 slots in the
current unfused diagnostic form, or about `0.55-0.59 ms/layer`.

## Remaining Gap

Next TP/EP work should implement the full hidden-control sequence instead of
only final expand:

- HC attention pre: exact sharded RMS, `hc_attn_fn`, split, weighted HC sum;
- attention body from the HC-selected current vector;
- HC attention expand into `after_attn_hc`;
- HC FFN pre and FFN delta from `after_attn_hc`;
- HC FFN expand into next-layer HC;
- selected-token feedback into the next decode step;
- tokenizer/prompt prefill/text output;
- MTP after the real serving loop is operational.
