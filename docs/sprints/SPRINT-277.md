# Sprint 277 - TP/EP Appliance Launcher Path

Date: 2026-05-23

## Goal

Wire the TP/EP resident HTTP harness into the appliance launcher so it can be
started by deployment configuration, not only by a hand-written benchmark
command.

## Implementation

Updated `tools/ds4-v100-run-appliance.sh`:

- Added `DS4_V100_SERVE_MODE=tp-ep`.
- Added TP/EP launcher variables:
  - `DS4_V100_TP_EP_BIN`
  - `DS4_V100_TP_EP_CONTRACT`
  - `DS4_V100_TP_EP_TM_INDEX`
  - `DS4_V100_TP_EP_TOP_K`
  - `DS4_V100_TP_EP_KV_SLOT`
  - `DS4_V100_TP_EP_POSITION`
- Builds the TP/EP command with the promoted resident settings:
  - shared TP runtime
  - shared expert bindings
  - shared dense ops
  - dense FP16 cache compose
  - source-scheduled peer copies
  - skip-self compose
  - multi-copy streams
- Keeps PP replay-server flags off the TP/EP command.
- Fails closed for the current TP/EP production target:
  - `ctx=262144`
  - `slots=32`
  - `active_microbatch == slots`
  - MTP off

Updated `deploy/v100/ds4-v100-appliance.env.example` with the TP/EP variables.

## Validation

Local:

- `bash -n tools/ds4-v100-run-appliance.sh`
- `DS4_V100_SERVE_MODE=tp-ep ... --check --allow-missing`
- `DS4_V100_SERVE_MODE=tp-ep ... --print-command --allow-missing`

V100 pod launcher smoke:

```text
DS4_V100_SERVE_MODE=tp-ep
DS4_V100_CTX=262144
DS4_V100_SLOTS=32
DS4_V100_ACTIVE_MICROBATCH=32
DS4_V100_TOKENS=32
DS4_V100_MAX_REQUESTS=4
```

HTTP sequence:

1. `GET /health`
2. `GET /v100/status`
3. `POST /v100/selected-token`
4. `GET /metrics`

Topline from the POST response:

| Metric | Value |
|---|---:|
| Generated tokens | 1024 |
| Continuation tokens | 992 |
| Token match | 32/32 |
| Wall generated tok/s | 728.744669 |
| Wall continuation tok/s | 753.022651 |
| Decode-only generated tok/s | 939.787471 |
| Decode-only continuation tok/s | 976.290858 |
| Total wall time | 1.405156 s |
| Total decode time | 1.089608 s |

## Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint277-tp-ep-launcher-smoke/cluster/
```

Primary files:

- `health.json`
- `status_before.json`
- `response.json`
- `metrics.txt`
- `server.log`
- `server.err`
- `runtime/command.txt`
- `runtime/startup.env`

## Decision

`DS4_V100_SERVE_MODE=tp-ep` is now the operational launcher path for the
current TP/EP resident server. It is still bounded to the current target shape
and serialized request handling.

## Next

Build a TP/EP sustained HTTP bench script that drives this launcher path and
records a matrix without relying on `curl` in the pod image.
