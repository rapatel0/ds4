# Sprint 276 - TP/EP Resident HTTP Harness

Date: 2026-05-23

## Goal

Expose the resident TP/EP backend through an in-process HTTP harness without
returning to the PP replay server.

## Rationale

Sprint 275 produced repeatable tool-level sustained-serving artifacts, but it
still invoked the backend through a benchmark wrapper. Practical metrology and
deployment need a server surface that keeps TP/EP state resident across
requests and exposes health, status, metrics, and generation endpoints.

This sprint intentionally keeps the implementation TP/EP-only. It does not
extend the PP scheduler or the old replay HTTP server.

## Implementation

Updated `tools/ds4-v100-tp-ep-full-layer-smoke.cu`:

- Extracted the token-major resident serving loop into
  `run_token_major_serving_loop`.
- Added `ServingBenchResult` so generated/continuation timing can be returned
  to callers instead of only printed as TSV rows.
- Added `--serve-http`, `--host`, `--port`, and `--max-requests`.
- Added a minimal TP/EP-only HTTP server with:
  - `GET /health`
  - `GET /v100/status`
  - `GET /metrics`
  - `POST /v100/selected-token`
- The server borrows resident TP runtime, rank buffers, expert bindings, dense
  FP16 cache, and shared dense ops. Requests are serialized because rank buffers
  are mutable during decode.

## Cluster Result

Smoke shape:

```text
slots=32
ctx=262144
tokens_per_request=32
position=100000
top_k=6
max_requests=4
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
| Wall generated tok/s | 719.275018 |
| Wall continuation tok/s | 751.645517 |
| Decode-only generated tok/s | 926.497242 |
| Decode-only continuation tok/s | 974.020201 |
| Total wall time | 1.423656 s |
| Total decode time | 1.105238 s |

Stage summary from the server log:

| Stage | Time |
|---|---:|
| Sum decode | 1105.238045 ms |
| Sum EP | 487.118601 ms |
| Sum compose | 617.780674 ms |
| Compose reduce | 93.107302 ms |
| Compose copy | 457.066582 ms |
| Compose final | 67.606790 ms |

## Evidence

Cluster artifacts are saved under:

```text
logs/from-cluster/sprint276-tp-ep-http-smoke/cluster/
```

Primary files:

- `health.json`
- `status_before.json`
- `response.json`
- `metrics.txt`
- `server.log`
- `server.err`

## Decision

The TP/EP resident HTTP harness is operational as a smoke-tested server path.
It is not yet the final production appliance:

- Requests are serialized.
- The endpoint synthesizes the current 32-slot resident decode shape rather
  than accepting arbitrary user prompt/token routing.
- It is not yet wired into `tools/ds4-v100-run-appliance.sh` or the Kubernetes
  deployment.

The next sprint should turn this into the operational TP/EP appliance launcher
path and run a sustained HTTP matrix from the same server surface.

## Next

- Add launcher/env support for `DS4_V100_SERVE_MODE=tp-ep`.
- Add a TP/EP HTTP sustained-decode bench script that drives the server without
  requiring `curl`.
- Run at least two HTTP cases:
  - `32` slots / `256K` / `32` tokens
  - `32` slots / `256K` / `64` tokens
- Preserve generated and continuation tok/s separately.
