# Sprint 273 - TP/EP Serving Metric Bridge

Date: 2026-05-23
Status: Complete

## Overview

Sprint 273 adds the first TP/EP serving-shaped metric bridge. It does not yet
run the HTTP appliance path, but it reports generated-token and continuation
token rates from the resident token-major TP/EP loop using the same target
shape we want for serving.

This sprint is intentionally a bridge from scaffold metrology toward serving
metrology. It also exposes the remaining operational gap clearly.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now supports:

```text
--serving-bench
```

When used with `--token-major-all-layers`, it emits:

- generated tokens
- continuation tokens
- first-token decode time
- continuation decode time
- decode-only generated tok/s
- decode-only continuation tok/s
- wall-time generated tok/s
- wall-time continuation tok/s

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint273-tp-ep-serving-bench/cluster/serving-bench-32req-16tok.log`
- `logs/from-cluster/sprint273-tp-ep-serving-bench/cluster/serving-bench-32req-16tok-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 32768
--decode-steps 16
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--shared-expert-bindings --overlap-ep-dense --source-copy-schedule
--skip-self-compose-copy --multi-copy-streams
--token-major-all-layers --all-layers --serving-bench
```

Result:

| Metric | Value |
|---|---:|
| Requests / slots | 32 |
| Generated tokens/request | 16 |
| Generated tokens | 512 |
| Continuation tokens | 480 |
| First-token decode ms | 69.547361 |
| Continuation decode ms | 515.270515 |
| Total decode ms | 584.817876 |
| Total wall ms | 48245.815783 |
| Aggregate generated tok/s, decode-only | 875.486234 |
| Aggregate continuation tok/s, decode-only | 931.549518 |
| Aggregate generated tok/s, wall | 10.612319 |
| Aggregate continuation tok/s, wall | 10.616412 |
| Checksum | 8244145680 |
| Result | PASS |

## Decision

The decode-only TP/EP path is now in the expected operating range for the first
practical milestone, but the current scaffold is not an operational serving
harness. Wall throughput is poor because token-major execution still calls the
heavy per-layer `run_layer()` scaffold for every token/layer invocation.

The next sprint should stop adding scaffold micro-optimizations and build a
resident TP/EP serving loop that precomputes per-layer descriptors, binds
resident expert/dense/runtime state once, and calls the decode body directly
without per-layer scaffold setup.
