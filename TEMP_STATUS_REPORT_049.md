# TEMP_STATUS_REPORT_049

Date: 2026-05-24

## Topline

Sprint 337 promoted the typed KV gates into the TP/EP HTTP appliance path and
proved a tokenizer-enabled serving smoke with resident session reuse.

This is still a diagnostic serving path, not production-ready, but typed raw
SWA, compressed attention, ratio-4 indexer, and typed-history gates now run
from the HTTP server command surface instead of only the full-layer smoke.

## What Changed

Updated:

- `tools/ds4-v100-run-appliance.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`

New appliance env gates:

- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW`
- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED`
- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER`
- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY`

`DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY=1` cascades on the other
typed KV gates plus true raw-window dependencies. The HTTP server now exposes
typed KV gate state in `/status`, `/metrics`, and generation response metadata.

## V100 Validation

Build:

- Command: `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS

HTTP serving smoke:

```text
slots=32
ctx=262144
tokens=2
typed raw/compressed/indexer/history KV gates enabled
diagnostic output head enabled
HC persistent state enabled
```

Results:

```text
chat-response-1.txt http 200 cache_hit 0 slot 0 pos_out 100014 typed_history 1 tok_s 62.781503
chat-response-2.txt http 200 cache_hit 1 slot 0 pos_out 100016 typed_history 1 tok_s 57.725287
status cache_hits 1 cache_misses 1 typed_history 1 served 4
```

Server PASS-line counts:

```text
typed_raw 685
typed_compressed 83
typed_indexer 83
typed_history 653
history_loaded_attn_rows_2 84
history_loaded_indexer_rows_2 84
```

GPUs were released after the smoke:

```text
0, 0, 32495, 0
1, 0, 32495, 0
2, 0, 32495, 0
3, 0, 32495, 0
4, 0, 32495, 0
5, 0, 32495, 0
6, 0, 32495, 0
7, 0, 32495, 0
```

## Current Gap

The HTTP smoke demonstrated visible compressed-history reload in serving, but
it was still only two requests with two generated tokens each. The next serving
sprint should run a longer multi-request typed-KV case and an A/B against the
no-typed-KV HTTP baseline so we know the serving throughput cost of the typed
production KV path.

## Artifact

- `logs/from-cluster/sprint337-typed-kv-http-serving/cluster/`
