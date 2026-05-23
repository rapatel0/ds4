# Sprint 292 - TP/EP Diagnostic Output-Head Serving Bridge

Date: 2026-05-23

## Goal

Wire the TP/EP sharded HC carry into the resident vocab-sharded output head and
surface diagnostic selected token IDs through the HTTP completions path.

This remains diagnostic. It uses the Sprint 291 proxy HC carry, so it does not
claim true DeepSeek text serving or correctness against the source model yet.

## Implementation

- Added `--diagnostic-output-head` to
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The flag implies `--final-hc-carry-gate`.
- Added a resident `SharedOutputHead` service object:
  - loads real replicated output controls;
  - loads real BF16 `output.weight` vocab shards once across all 8 GPUs;
  - keeps output-head scratch and top-1 buffers resident;
  - gathers per-rank `[slots][4][512]` HC shards into GPU0 logical
    `[slots][4][4096]`;
  - runs output HC collapse, output norm, vocab-sharded BF16 projection, and
    GPU-side per-shard top-1.
- `run_token_major_serving_loop()` now optionally calls the resident output
  head after the final layer.
- HTTP `/v1/completions` and `/v100/diagnostic-completions` responses now
  include diagnostic selected-token metadata when enabled:
  - `diagnostic_output_head`
  - `diagnostic_output_head_proxy_hc`
  - `selected_token`
  - `selected_logit`
  - output-head timing fields
- Added launcher env:
  - `DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD=1`
- Added HTTP bench option:
  - `--diagnostic-output-head`
- Added the env key to
  `deploy/v100/ds4-v100-appliance.env.example`, default off.

## Definition of Done

- [x] The TP/EP full-layer smoke builds on the V100 pod.
- [x] Direct 32-slot, 1-token all-layer run passes with diagnostic output head.
- [x] Launcher-level HTTP completions run passes with diagnostic output head.
- [x] Full 32-concurrent HTTP completions run forms one 32-request coalesced
  batch and returns selected-token metadata.
- [x] Status and vision record that this is still proxy-HC diagnostic output,
  not real model text serving.

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
  --diagnostic-output-head
```

Direct result:

```text
tp_ep_diagnostic_output_head
  steps=1 slots=32 proxy_hc=1
  total_ms=8.903469
  gather_ms=0.222206
  prep_ms=0.132714
  broadcast_ms=0.360320
  projection_ms=7.690283
  projection_kernel_worst_ms=7.463808
  top1_ms=0.497101
  first_token=122445
  first_logit=2819.954101562
  finite_bad=0
  PASS
```

Launcher HTTP validation, 4 concurrent completion requests:

```text
endpoint=tokens=1, requests=4, coalesced_batches=1, coalesced_batch_max=4
status_200=4, generated_tok_s=19.209160, generated_tok_s_decode=35.928856
```

Each response includes `diagnostic_output_head=1`,
`diagnostic_output_head_proxy_hc=1`, `selected_token`, `selected_logit`, and
output-head timing.

Launcher HTTP validation, 32 concurrent completion requests:

```text
endpoint=tokens=1, requests=32, coalesced_batches=1, coalesced_batch_max=32
status_200=32, generated_tok_s=158.576748, generated_tok_s_decode=294.331849
```

Server output-head timing for the 32-concurrent run:

```text
tp_ep_diagnostic_output_head
  steps=1 slots=32 proxy_hc=1
  total_ms=8.586224
  gather_ms=0.200650
  prep_ms=0.113959
  broadcast_ms=0.337343
  projection_ms=7.592902
  projection_kernel_worst_ms=7.459872
  top1_ms=0.341194
  first_token=122445
  first_logit=2819.954101562
  finite_bad=0
  PASS
```

Evidence:

```text
logs/from-cluster/sprint292-tp-ep-diagnostic-output-head/cluster/
logs/from-cluster/sprint292-tp-ep-diagnostic-output-head-http/cluster/
logs/from-cluster/sprint292-tp-ep-diagnostic-output-head-http32/cluster/
```

## Decision

Promote the resident output-head serving bridge as the next TP/EP diagnostic
surface. It proves the operational path from token-major TP/EP decode, through
sharded HC ownership, through vocab-sharded output projection, into HTTP
completion responses.

Do not mark this as real model serving. The selected token IDs come from proxy
HC rows and are only useful for wiring, timing, and endpoint integration.

## Remaining Gap

The practical serving path still needs:

- true DS4 HC row semantics in the TP/EP layer loop;
- prompt prefill and tokenizer input;
- selected-token feedback into subsequent decode steps;
- tokenizer text output and stop handling;
- then MTP.
