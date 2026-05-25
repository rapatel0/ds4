# TEMP Status Report 080 - Sprint 368

Date: 2026-05-25

## Current Focus

TP/EP production-serving correctness for long-context chat. Sprint 368 fixed
the Sprint 367 failure mode where chat prompt prefill plus generation could
cross the 256K context boundary and fail as HTTP 500 during GPU decode.

## Implemented

- Added explicit TP/EP HTTP context admission.
- Admission checks:

```text
start_position + prompt_prefill_steps + requested_decode_steps <= 262144
```

- Cache hits use the cached slot position and zero prompt prefill.
- Cache misses include `prompt_tokens - 1` prefill steps.
- Over-context requests now return HTTP 400 with structured JSON.

## Invalid Shape Test

Shape:

```text
endpoint: /v1/chat/completions
ctx: 262144
position: 262112
requests: 1
tokens/request: 32
prompt_prefill_steps: 16
```

Response:

```json
{"error":"context_window_exceeded","ctx":262144,"start_position":262112,"prompt_prefill_steps":16,"requested_decode_steps":32,"final_position":262160,"cache_hit":0}
```

Result:

- HTTP status: `400`
- compressed-KV lines: `0`
- server log: `tp_ep_http_context_rejected`
- no `tp_ep_http_decode_failed`

## Valid Shape Test

Shape:

```text
endpoint: /v1/chat/completions
ctx: 262144
position: 262080
requests: 32
tokens/request: 32
```

Result:

| HTTP 200 | Coalesced batch | First token | Bad | Client tok/s | Server wall tok/s | Server decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 32/32 | 32 | 89340 | 0 | 51.069220 | 82.089657 | 98.301727 |

## Validation

Local checks passed:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
./ds4_test --server
./ds4_test --metal-kernels
```

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

## Artifacts

```text
logs/from-cluster/sprint368-chat-context-admission-invalid/
logs/from-cluster/sprint368-chat-context-admission-valid/
```
