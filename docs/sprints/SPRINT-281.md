# Sprint 281 - TP/EP HTTP Stage Metrics

Date: 2026-05-23

## Goal

Expose stage timing through the TP/EP HTTP serving artifacts so operational
benchmarks show where time is going without requiring manual `server.log`
inspection.

## Implementation

Updated `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

- Added stage totals to `ServingBenchResult`:
  - EP time
  - dense time
  - compose time
  - compose reduce time
  - compose copy time
  - compose final time
- Added the stage fields to each `/v100/selected-token` response under
  `timing_ms`.
- Added last/cumulative stage fields to `/v100/status`.
- Added matching Prometheus-style stage counters to `/metrics`.

Updated `tools/ds4-v100-tp-ep-http-bench.sh`.

- Bumped the sustained HTTP schema to
  `ds4_v100_tp_ep_sustained_http.v3`.
- Aggregates stage timings across all resident generation requests.
- Adds `ep_ms`, `dense_ms`, `compose_ms`, `compose_reduce_ms`,
  `compose_copy_ms`, and `compose_final_ms` to `sustained_http.tsv` and
  per-case `result.json`.

## Validation

Local validation:

```text
bash -n tools/ds4-v100-tp-ep-http-bench.sh
git diff --check
```

V100 pod validation:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Target matrix:

```text
tokens_cases=32,64
generation_requests=3
ctx=262144
slots=32
active_microbatch=32
serve_mode=tp-ep
```

Results:

| Tokens/request | Generated tokens | Wall generated tok/s | Decode generated tok/s | EP ms | Compose ms | Compose copy ms | Avg GPU util | Max GPU util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3072 | 742.897231 | 963.120273 | 1345.860853 | 1842.727542 | 1365.843009 | 18.000000 | 40.000000 |
| 64 | 6144 | 739.612937 | 976.387282 | 2663.985462 | 3626.650073 | 2569.208878 | 16.214286 | 33.000000 |

Both cases returned aggregate `96/96` token match and `0` mismatches.

The 64-token case reports:

```text
compose_copy_ms / compose_ms = 2569.208878 / 3626.650073 = 70.8%
compose_ms / (ep_ms + compose_ms) = 3626.650073 / 6290.635535 = 57.7%
```

## Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint281-tp-ep-http-stage-metrics/cluster/
```

Primary files:

- `sustained_http.tsv`
- `sustained_http.json`
- Per-case `result.json`
- Per-case `responses.json`
- Per-case `status_after.json`
- Per-case `metrics.txt`
- Per-case `gpu_util.csv`

## Decision

The operational HTTP artifact now confirms the same bottleneck seen in the
lower-level logs: compose-copy is the largest individual stage. For the
64-token, three-request serving case, compose-copy is `70.8%` of compose time,
and compose as a whole is `57.7%` of EP plus compose time.

This makes the next performance sprint concrete: target compose-copy movement
and synchronization in the TP/EP path. Request coalescing remains necessary
for appliance semantics, but it will not by itself raise GPU utilization above
the fixed 32-slot synthetic shape because that shape already fills the active
slot count.

## Next

- Optimize TP/EP compose-copy scheduling or representation.
- Keep the stage metrics in every sustained serving run.
- After compose-copy improves, add true HTTP request coalescing/admission for
  independent client requests.
