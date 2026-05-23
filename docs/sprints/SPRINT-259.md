# Sprint 259 - TP Runtime A/B Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 259 adds a same-binary toggle for TP runtime sharing and runs a direct
A/B between local per-layer TP runtime and shared all-layer TP runtime. Sprint
258 showed the shared runtime decode regression persisted in a longer gate, but
that compared separate runs across evolving commits. This sprint isolates the
choice in one executable.

This is a measurement sprint, not generated-token serving throughput.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add a same-binary TP runtime sharing toggle.
- Compare local per-layer TP runtime versus shared all-layer TP runtime at
  `32` slots / `256K`.
- Preserve checksum and all-layer pass/fail reporting.
- Set the default to the faster decode-speed base.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now supports:

```text
--share-tp-runtime
--local-tp-runtime
```

The default is local per-layer TP runtime because the A/B shows it is faster
for decode. Shared TP runtime remains available as an opt-in diagnostic and
residency experiment.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint259-tp-runtime-ab/cluster/local-runtime-50step.log`
- `logs/from-cluster/sprint259-tp-runtime-ab/cluster/local-runtime-50step-summary.log`
- `logs/from-cluster/sprint259-tp-runtime-ab/cluster/shared-runtime-50step.log`
- `logs/from-cluster/sprint259-tp-runtime-ab/cluster/shared-runtime-50step-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 50
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes --all-layers
```

A/B result:

| Metric | Local TP runtime | Shared TP runtime |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Shared TP runtime | 0 | 1 |
| Sum decode ms/token | 42.723359 | 46.972659 |
| Projected slot-step tok/s | 749.004771 | 681.247356 |
| Sum EP ms | 11.688230 | 13.327484 |
| Sum dense ms | 7.390219 | 8.766676 |
| Sum compose ms | 23.639029 | 24.874004 |
| Wall ms | 35558.446077 | 30847.018046 |
| Checksum | 204721433 | 204721433 |
| Result | PASS | PASS |

## Decision

Use local per-layer TP runtime as the current decode-speed base. Shared TP
runtime remains correct and reduces setup wall time, but it loses `9.05%`
projected decode throughput in the same-binary A/B. Do not build the next
performance sprint on shared TP runtime until the EP/dense timing interaction
is understood.

Next implementation should hoist expert descriptor bindings or collapse more
of the EP/dense/compose boundary while keeping the local TP runtime mode as the
benchmark default.
