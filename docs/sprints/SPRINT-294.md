# Sprint 294 - TP/EP HC Current-Input Bridge

Date: 2026-05-23

## Goal

Move the TP/EP diagnostic serving path one step closer to real DS4 layer
semantics by replacing synthetic routed-expert activations with current vectors
derived from the resident sharded HC state.

This remains diagnostic. The endpoint still does not implement prompt prefill,
tokenizer text output, selected-token feedback, or the full DS4 attention/FFN
pre/post hidden-control sequence.

## Implementation

- Added `--tp-hc-current-input-gate` to
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- The flag implies:
  - `--tp-hc-final-expand-gate`
  - `--final-hc-carry-gate`
- Extended `SharedHcControls` to load real per-layer:
  - `blk.N.hc_attn_fn`
  - `blk.N.hc_attn_base`
  - `blk.N.hc_attn_scale`
  - existing `blk.N.hc_ffn_*` controls
- Added TP/EP-only kernels for:
  - deterministic initial sharded HC seed for first diagnostic layer;
  - sharded HC weighted sum from real `hc_attn_*` split weights;
  - current-shard gather into a full `[slots,4096]` current vector;
  - route activation packing from the HC-derived current vector;
  - dense diagnostic input fill from the HC-derived current vector.
- Added launcher and HTTP bench support:
  - `DS4_V100_TP_EP_HC_CURRENT_INPUT=1`
  - `--hc-current-input`

## Important Limitation

The routed expert input now comes from the HC-derived current vector. The dense
diagnostic inputs are still a bridge, not exact DS4 attention/FFN semantics:
the current `[slots,4096]` vector is repeated/truncated into the dense op input
widths expected by the existing diagnostic dense tensors.

That limitation is deliberate for this sprint. It lets the serving harness run
real resident TP/EP dataflow through all 43 layers without claiming that the
attention body, FFN pre path, or token feedback are complete.

## Definition of Done

- [x] TP/EP full-layer smoke builds on the V100 pod.
- [x] Direct 32-slot, 256K-context, 1-token, all-layer run passes with HC
  current input, HC final expand, and diagnostic output head enabled.
- [x] Launcher-level `/v1/completions` run passes with 32 concurrent requests
  coalesced into one 32-slot batch.
- [x] Evidence is copied back from the cluster.
- [x] Status and vision record the remaining gap to real DS4 serving.

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
  --tp-hc-current-input-gate --tp-hc-final-expand-gate \
  --diagnostic-output-head
```

Direct result:

```text
tp_ep_token_major_scaffold pass_invocations=43 slots=32 ctx=262144
sum_decode_ms=134.008975 projected_slot_step_tok_s=238.789977
sum_ep_ms=51.465883 sum_compose_ms=19.072492
sum_hc_current_input_ms=40.646652 sum_final_hc_ms=22.678353 PASS

tp_ep_diagnostic_output_head proxy_hc=0 total_ms=8.530776
projection_ms=7.555476 top1_ms=0.363674 first_token=117160 PASS
```

Launcher HTTP validation:

```text
tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir /workspace/logs/sprint294-tp-ep-hc-current-input-http32 \
  --tokens-cases 1 --requests 32 --endpoint completions \
  --diagnostic-output-head --hc-current-input --concurrent-requests
```

HTTP result:

```text
endpoint=completions tokens=1 ctx=262144 slots=32 generation_requests=32
coalesced_batches=1 coalesced_batch_max=32 status_200=32
generated_tokens=32 generated_tok_s=145.914985
generated_tok_s_decode=225.722945
ep_ms=59.899706 compose_ms=18.852835 compose_copy_ms=10.605672
gpu_util_avg=4.500000 gpu_util_max=31.000000
```

Server-side HTTP summary:

```text
tp_ep_token_major_scaffold sum_decode_ms=141.766713
projected_slot_step_tok_s=225.722945
sum_hc_current_input_ms=40.406994
sum_final_hc_ms=22.594013 PASS
tp_ep_diagnostic_output_head proxy_hc=0 total_ms=8.476833 first_token=117160 PASS
```

Evidence:

```text
logs/from-cluster/sprint294-tp-ep-hc-current-input/cluster/
logs/from-cluster/sprint294-tp-ep-hc-current-input-http32/cluster/
```

## Decision

Keep `--tp-hc-current-input-gate` as an opt-in diagnostic bridge. It is a real
serving-dataflow improvement because the routed expert path no longer consumes
fixed synthetic activations. It is not yet production DS4 serving because dense
diagnostic inputs and recurrent token state are still incomplete.

## Remaining Gap

Next TP/EP work should prioritize real prototype serving semantics over
additional kernel tuning:

- feed prompt/prefill embeddings into the initial HC state;
- implement the true HC attention pre path and attention body input layout;
- implement the true HC attention expand, HC FFN pre, and HC FFN expand
  sequence rather than using dense diagnostic input repeat/truncate;
- feed the selected output token back into the next decode step;
- return tokenizer text rather than diagnostic metadata only;
- run multi-token HTTP serving once token feedback is wired;
- add MTP only after the base TP/EP serving loop is prompt-driven and recurrent.
