# Sprint 278 - TP/EP Sustained HTTP Matrix

Date: 2026-05-23

## Goal

Add repeatable sustained HTTP matrix tooling for the TP/EP launcher path and
run the current 32-slot / 256K target cases.

## Implementation

Added `tools/ds4-v100-tp-ep-http-bench.sh`.

The script:

- Starts `tools/ds4-v100-run-appliance.sh` with `DS4_V100_SERVE_MODE=tp-ep`.
- Uses Python stdlib HTTP clients, so it does not require `curl` in the pod.
- Runs one resident server per token case.
- Exercises:
  - `GET /health`
  - `GET /v100/status`
  - `POST /v100/selected-token`
  - `GET /metrics`
- Writes:
  - `sustained_http.tsv`
  - `sustained_http.json`
  - per-case `result.json`, HTTP responses, server logs, and launcher runtime
    command/startup env.

## Cluster Matrix

Shape:

```text
ctx=262144
slots=32
active_microbatch=32
serve_mode=tp-ep
```

Results:

| Tokens/request | Generated tokens | Continuation tokens | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Decode continuation tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1024 | 992 | 737.091414 | 766.964251 | 952.404358 | 995.431373 |
| 64 | 2048 | 2016 | 739.774102 | 755.504630 | 960.711973 | 984.032395 |

Both cases returned `32/32` token match and `0` mismatches.

## Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint278-tp-ep-http-matrix/cluster/
```

Primary files:

- `sustained_http.tsv`
- `sustained_http.json`
- `cases/case_0_ctx262144_s32_tok32/result.json`
- `cases/case_1_ctx262144_s32_tok64/result.json`

## Decision

The launcher-backed TP/EP HTTP path now has repeatable sustained-serving
metrology. Current practical topline is about `737-740` wall generated tok/s
and `755-767` wall continuation tok/s for 32 slots at 256K.

## Next

The remaining operational gap is deployment polish rather than proof of a
server path:

- Wire `DS4_V100_SERVE_MODE=tp-ep` into the Kubernetes deployment example.
- Add GPU utilization capture around the HTTP matrix.
- Decide the next optimization target from current stage timings: compose copy
  remains the largest measured stage.
