# Sprint 280 - TP/EP Multi-Request HTTP Metrology

Date: 2026-05-23

## Goal

Make the TP/EP HTTP harness measure resident sustained serving across multiple
generation requests instead of only one selected-token POST per server start.

## Implementation

Updated `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

- Added cumulative counters to the TP/EP HTTP server:
  - prompt tokens
  - generated tokens
  - continuation tokens
  - total decode/wall time
  - continuation decode/wall time
  - next logical position
- Exposed cumulative throughput fields through `/v100/status`.
- Exposed matching Prometheus-style cumulative metrics through `/metrics`.

Updated `tools/ds4-v100-tp-ep-http-bench.sh`.

- Added `--requests N`, default `1`.
- Keeps one resident TP/EP server alive per token case.
- Sends `N` generation POSTs to `/v100/selected-token`.
- Writes per-request `response_NNN.json`, aggregate `responses.json`, and a
  compatibility `response.json` containing the last response.
- Aggregates generated/continuation tok/s across all generation requests.
- Keeps GPU-utilization sampling active across the full generation-request
  sequence.
- Bumped the sustained HTTP artifact schema to
  `ds4_v100_tp_ep_sustained_http.v2`.

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

Short smoke:

```text
tokens_cases=8
generation_requests=3
ctx=262144
slots=32
```

Smoke result:

| Tokens/request | Generation requests | Generated tokens | Continuation tokens | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Decode continuation tok/s | Avg GPU util | Max GPU util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 768 | 672 | 682.649279 | 745.069189 | 885.928031 | 977.582777 | 11.062500 | 32.000000 |

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

| Tokens/request | Generation requests | Generated tokens | Continuation tokens | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Decode continuation tok/s | Avg GPU util | Max GPU util | Max GPU mem MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3 | 3072 | 2976 | 751.114404 | 760.078310 | 972.107940 | 985.591696 | 18.875000 | 40.000000 | 13184 |
| 64 | 3 | 6144 | 6048 | 762.277426 | 766.925593 | 988.565789 | 995.281608 | 19.870536 | 41.000000 | 13184 |

Both target cases returned aggregate `96/96` token match and `0` mismatches.

The `status_after.json` artifact confirms cumulative server accounting. For
the 32-token case it reports `generation_requests=3`,
`total_generated_tokens=3072`, `total_continuation_tokens=2976`, and
`next_position=100096`.

## Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint280-tp-ep-http-multirequest/cluster/
```

Primary files:

- `sustained_http.tsv`
- `sustained_http.json`
- Per-case `responses.json`
- Per-request `response_000.json`, `response_001.json`, `response_002.json`
- Per-case `status_after.json`
- Per-case `metrics.txt`
- Per-case `gpu_util.csv`

## Decision

The HTTP harness now supports resident multi-request metrology. This is still
not full prompt routing or continuous batching, but it is a better operational
measurement surface: one loaded TP/EP server can serve repeated generation
requests, accumulate counters, and report sustained generated and continuation
tok/s with GPU utilization.

The measured topline improves slightly over the one-request Sprint 279 matrix:
`751-762` wall generated tok/s versus `745-754`. GPU utilization remains low,
peaking at `40-41%`, which keeps the next bottleneck decision unchanged:
compose/copy and request coalescing are the practical serving gaps.

## Next

- Add an actual coalescing/admission layer so independent HTTP requests can
  fill the 32 active slots instead of relying on one synthetic request to
  occupy all slots.
- Preserve the new cumulative `/v100/status` and `/metrics` counters as the
  primary serving metrology.
- Keep optimizing toward higher GPU utilization at `32` slots / `256K` before
  enabling MTP.
