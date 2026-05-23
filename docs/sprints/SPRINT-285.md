# Sprint 285 - TP/EP Promoted Serving Topline

Date: 2026-05-23

## Goal

Re-establish the normal promoted TP/EP HTTP serving topline after enabling
event-wait compose and compact route-compose by default.

## Configuration

Promoted defaults:

```text
DS4_V100_SERVE_MODE=tp-ep
DS4_V100_CTX=262144
DS4_V100_SLOTS=32
DS4_V100_ACTIVE_MICROBATCH=32
DS4_V100_TP_EP_COPY_EVENT_COMPOSE=1
DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1
DS4_V100_TP_EP_RETURN_FP16=0
```

Benchmark:

```text
tokens_cases=32,64
generation_requests=3
```

## Validation

Ran the promoted HTTP bench through the normal launcher defaults on the V100
pod:

```text
./tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir /workspace/logs/sprint285-tp-ep-promoted-topline \
  --tokens-cases 32,64 \
  --requests 3 \
  --port-base 18350
```

Results:

| Tokens/request | Generated tokens | Continuation tokens | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Decode continuation tok/s | EP ms | Compose ms | Compose copy ms | Avg GPU util | Max GPU util | Match |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3072 | 2976 | 771.036527 | 781.922821 | 1017.437412 | 1033.741359 | 1334.051119 | 1684.325071 | 930.176638 | 11.214286 | 31.000000 | 96/96 |
| 64 | 6144 | 6048 | 794.694599 | 799.391755 | 1044.457435 | 1051.722519 | 2613.815141 | 3266.704197 | 1808.001118 | 11.834821 | 32.000000 | 96/96 |

## Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint285-tp-ep-promoted-topline/cluster/
```

Primary files:

- `sustained_http.tsv`
- `sustained_http.json`
- Per-case `responses.json`
- Per-case `status_after.json`
- Per-case `metrics.txt`
- Per-case `gpu_util.csv`

## Decision

The promoted TP/EP serving topline is now:

```text
32 slots / 256K / 3 resident requests:
  32 tokens/request: 771.036527 wall generated tok/s
  64 tokens/request: 794.694599 wall generated tok/s
```

This is a substantial improvement over the pre-TP/EP serving baseline, but it
is still below the practical target for high-throughput serving. GPU
utilization remains low, so the next stage is request coalescing/admission and
additional compose/EP fusion, not MTP yet.

## Next

- Add true request coalescing/admission so independent HTTP clients can fill
  the 32-slot active microbatch.
- Keep promoted stage metrics in every topline run.
- After request coalescing is operational, revisit MTP as a decode multiplier.
