# Sprint 295 - TP/EP KV And Resident State Guardrails

Date: 2026-05-23

## Goal

Make the TP/EP serving scaffold honest about resident downstream state. The
previous HTTP path allocated the sharded KV runtime, but only exercised one
diagnostic KV slot and reset HC state at the start of each serving call. That
is not acceptable for downstream tasks, where rerunning from scratch would
erase the point of KV caching.

## Implementation

- Added `--tp-kv-all-slots-gate`.
  - In resident TP/EP layer decode, this updates and verifies sharded KV rows
    for every active slot instead of only the old diagnostic `kv_slot=7`.
  - The runtime KV arena is still the existing sharded F8 layout from
    `ds4_v100_tp_runtime`.
- Added `--tp-hc-persist-state-gate`.
  - This stops the token-major serving loop from resetting resident sharded HC
    state at every serving call.
  - The first call still seeds HC when needed; later calls can continue from
    resident state.
- Added launcher and bench flags:
  - `DS4_V100_TP_EP_KV_ALL_SLOTS=1`
  - `DS4_V100_TP_EP_HC_PERSIST_STATE=1`
  - `--kv-all-slots`
  - `--hc-persist-state`
- Added HTTP `/status`, `/metrics`, and per-response metadata for:
  - `kv_runtime_resident`
  - `kv_all_slots_gate`
  - `hc_persist_state_gate`

## Important Limitation

This sprint validates resident KV allocation and all-active-slot KV row
updates. It does not yet implement a full prompt/session KV cache with
per-client slot assignment, eviction, prefix reuse, or tokenizer-driven
prefill. The attention body is still diagnostic, so KV is verified as a
runtime-resident state surface rather than consumed by real attention.

## Validation

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
  --tp-hc-persist-state-gate --tp-kv-all-slots-gate \
  --diagnostic-output-head
```

Direct result:

```text
tp_ep_token_major_scaffold pass_invocations=43 slots=32 ctx=262144
sum_decode_ms=131.638876 projected_slot_step_tok_s=243.089283
sum_hc_current_input_ms=40.082554 sum_final_hc_ms=22.508169 PASS
tp_ep_serving_bench aggregate_generated_tok_s_decode=243.089283
aggregate_generated_tok_s_wall=71.431217 PASS
```

The wall metric is intentionally much lower than decode because the all-slot
KV verifier writes and reads every active slot outside the timed decode stage.
That makes it a correctness guardrail, not an optimized serving mode.

HTTP validation:

```text
tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir /workspace/logs/sprint295-tp-ep-kv-session-state-http32 \
  --tokens-cases 1 --requests 32 --endpoint completions \
  --diagnostic-output-head --hc-current-input \
  --hc-persist-state --kv-all-slots --concurrent-requests
```

HTTP result:

```text
endpoint=completions tokens=1 ctx=262144 slots=32 generation_requests=32
coalesced_batches=1 coalesced_batch_max=32 status_200=32
generated_tok_s=58.791255 generated_tok_s_decode=206.196887
ep_ms=68.284903 compose_ms=21.215404 compose_copy_ms=11.879992
gpu_util_avg=2.750000 gpu_util_max=15.000000
```

Status evidence:

```json
{
  "kv_runtime_resident": 1,
  "kv_all_slots_gate": 1,
  "hc_persist_state_gate": 1,
  "generation_batches": 1,
  "coalesced_requests": 32,
  "next_position": 100001
}
```

Evidence:

```text
logs/from-cluster/sprint295-tp-ep-kv-session-state/cluster/
logs/from-cluster/sprint295-tp-ep-kv-session-state-http32/cluster/
```

## Decision

Keep both gates opt-in until real prompt/session ownership exists. They are
now the correctness mode for downstream-serving work because they prevent us
from accidentally benchmarking a scratch-style request loop.

## Remaining Gap

Next TP/EP serving work should add real session ownership:

- parse a request/session key;
- assign stable slots to sessions;
- persist per-session position and prompt length;
- reuse slot KV/HC state for continuation requests;
- define eviction/reset semantics;
- make prompt prefill populate the same resident KV/HC state;
- only then optimize the KV update path and remove diagnostic readback.
