---
sprint: 337
title: TP/EP Typed KV HTTP Serving Gate
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 337 - TP/EP Typed KV HTTP Serving Gate

## Goal

Promote the typed raw, compressed-attention, indexer, and compressed-history
KV path into the TP/EP tokenizer-enabled HTTP serving appliance path.

## Why This Sprint

Sprint 336 proved typed KV current-row storage and visible-history reload in
the all-layer smoke harness. The vision's next operational gap is proving that
same state path under the serving loop: tokenizer input, prompt prefill,
multi-token decode, output-head sample/feed, and resident session/KV reuse.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint does not optimize the
typed-KV path; it makes it selectable and observable in the appliance serving
path and validates a small HTTP smoke before scaling to longer serving runs.

## Implementation Plan

- Add first-class appliance env toggles for the four typed KV gates.
- Cascade typed-history serving mode to the true raw-window dependencies.
- Pass the existing typed KV binary flags through the TP/EP appliance launcher.
- Expose typed KV gate state in `/status`, `/metrics`, and response metadata.
- Build the TP/EP HTTP binary on the V100 pod.
- Run a tokenizer-enabled HTTP serving smoke with resident sessions and typed
  KV gates enabled.
- Capture server logs, responses, status, and metrics under
  `logs/from-cluster/sprint337-typed-kv-http-serving/`.

## Definition of Done

- [x] `tools/ds4-v100-run-appliance.sh --print-command` emits all four typed
      KV serving flags when `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY=1`.
- [x] The TP/EP HTTP binary builds for `sm_70` on the V100 pod.
- [x] A tokenizer-enabled `/v1/chat/completions` request returns HTTP 200 with
      typed KV gate state visible in response metadata.
- [x] A repeated request using the same `session_id` shows resident-session
      reuse rather than full reset.
- [x] Server evidence includes typed KV raw, compressed, indexer, and history
      PASS lines during the serving decode path.
- [x] Artifacts and status are recorded and committed.

## Outcome

Added appliance-level TP/EP env toggles:

- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW`
- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED`
- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER`
- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY`

`DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY=1` cascades on the raw,
compressed, indexer, raw-window, projection/state/residency, and HC input/final
dependencies. The TP/EP HTTP server now exposes those typed KV gate states in
`/status`, `/metrics`, and each generation response's `ds4_v100` metadata.

## Validation

Local launcher checks:

```text
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
tools/ds4-v100-run-appliance.sh --print-command --allow-missing
```

Result: PASS. The printed TP/EP command included all four typed KV flags.

V100 build:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

Tokenizer-enabled HTTP smoke:

```text
slots=32
ctx=262144
tokens=2
typed raw/compressed/indexer/history KV gates enabled
diagnostic output head enabled
HC persistent state enabled
```

Two `/v1/chat/completions` requests used the same `session_id`.

```text
chat-response-1.txt http 200 cache_hit 0 slot 0 pos_out 100014 typed_history 1 tok_s 62.781503
chat-response-2.txt http 200 cache_hit 1 slot 0 pos_out 100016 typed_history 1 tok_s 57.725287
status cache_hits 1 cache_misses 1 typed_history 1 served 4
```

Server typed KV evidence:

```text
typed_raw 685
typed_compressed 83
typed_indexer 83
typed_history 653
history_loaded_attn_rows_2 84
history_loaded_indexer_rows_2 84
```

Representative serving PASS lines:

```text
tp_ep_true_attention_typed_kv_raw layer 0 slots 32 position 100000 physical_row 32 raw_row 32 ... PASS
tp_ep_true_attention_typed_kv_compressed layer 2 slots 32 ratio 4 position 100003 bounded_row 0 physical_row 25128 ... PASS
tp_ep_true_attention_typed_kv_indexer layer 2 slots 32 ratio 4 position 100003 bounded_row 0 visible_rows 1 physical_row 25000 ... PASS
tp_ep_true_attention_typed_kv_history layer 2 slots 32 ratio 4 visible_attn_rows 2 loaded_attn_rows 2 loaded_indexer_rows 2 PASS
```

Artifacts:

- `logs/from-cluster/sprint337-typed-kv-http-serving/cluster/command.txt`
- `logs/from-cluster/sprint337-typed-kv-http-serving/cluster/chat-response-1.txt`
- `logs/from-cluster/sprint337-typed-kv-http-serving/cluster/chat-response-2.txt`
- `logs/from-cluster/sprint337-typed-kv-http-serving/cluster/status.json`
- `logs/from-cluster/sprint337-typed-kv-http-serving/cluster/metrics.txt`
- `logs/from-cluster/sprint337-typed-kv-http-serving/cluster/server.out`
- `logs/from-cluster/sprint337-typed-kv-http-serving/cluster/server.err`
- `logs/from-cluster/sprint337-typed-kv-http-serving/cluster/typed-kv-counts.txt`

## Current Gap

This proves the typed KV gates run through tokenizer-enabled HTTP serving,
resident session reuse, and visible compressed-history reload. It is still a
two-request smoke, so the next sprint should scale this into a longer
multi-request serving run and compare throughput against the no-typed-KV HTTP
baseline.

## Risks

- Typed history reload currently stages bounded visible rows. This is enough
  for the serving smoke but not the final production allocator/read kernel.
- The diagnostic output head is still enabled for visibility. A production
  output-head/quality gate remains separate from typed KV serving plumbing.
