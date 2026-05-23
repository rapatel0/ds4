# Sprint 254 - TP/EP Pre-Decode Probe Bypass

Date: 2026-05-23
Status: Complete

## Overview

Sprint 253 restored the decode-only all-layer scaffold benchmark, but every
layer still ran pre-decode TurboMind warmup, isolated gate/down timing, and
repeat checks before the actual decode loop. Those probes are useful for strict
validation, but they are not serving work. Sprint 254 adds an opt-in
`--skip-predecode-probes` mode for serving-shaped benchmark runs after strict
validation has already passed.

This remains a scaffold gate, not generated-token serving throughput.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add an opt-in pre-decode probe bypass.
- Keep strict pre-decode probes enabled by default.
- Preserve shared dense cache and descriptor-bypass modes.
- Run the all-layer decode-only gate on the V100 pod at `32` slots / `256K`.

## Non-Goals

- No PP scheduler changes.
- No production server integration.
- No logits-equivalent output.
- No MTP.
- No TurboMind/API handle hoist yet.

## Design

The new flag:

```text
--skip-predecode-probes
```

skips only the validation/timing probes before `run_decode_loop()`:

```text
default:
  warmup gate/down
  isolated gate timing
  isolated down timing
  repeat check
  decode loop

skip mode:
  decode loop only
```

The decode loop still runs its own warmup and timed steps, updates the EP
remote contribution path, runs cache-backed dense compose, and checks finite
output.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Adds `--skip-predecode-probes` |
| `docs/sprints/SPRINT-254.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint254-tp-ep-skip-predecode-probes/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in separate TP/EP code.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Default strict probe behavior is preserved.
- [x] `--skip-predecode-probes` passes the all-layer decode-only gate with
      shared dense cache and descriptor checks disabled.
- [x] Evidence records `predecode_probes=0`.
- [x] Evidence is copied to
      `logs/from-cluster/sprint254-tp-ep-skip-predecode-probes/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint254-tp-ep-skip-predecode-probes/cluster/all-layer-10step-skip-predecode.log`
- `logs/from-cluster/sprint254-tp-ep-skip-predecode-probes/cluster/all-layer-10step-skip-predecode-summary.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --decode-steps 10
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes --all-layers
```

10-step all-layer decode-only gate:

| Metric | Sprint 253 probes on | Sprint 254 probes off |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Descriptor checks | 0 | 0 |
| Predecode probes | 1 | 0 |
| Sum decode ms/token | 44.035733 | 44.848746 |
| Projected slot-step tok/s | 726.682578 | 713.509362 |
| Per-layer avg decode ms | 1.024087 | 1.042994 |
| Per-layer min decode ms | 0.982757 | 0.998564 |
| Per-layer max decode ms | 1.155619 | 1.134957 |
| Sum EP ms | 11.804094 | 11.806264 |
| Sum dense ms | 7.744769 | 8.126768 |
| Sum compose ms | 24.482197 | 24.910200 |
| Wall ms | 39951.007721 | 37819.503379 |
| Result | PASS | PASS |

The bypass reduces wall time another `5.3%` by removing non-serving validation
work. The decode proxy is slightly lower than Sprint 253, but still in the same
scaffold band; use strict gates for correctness and this mode for lightweight
serving-shaped scaffold timing.

## Decision

`--skip-predecode-probes` is promoted only for benchmark runs after strict
validation has passed. The next implementation step should hoist real runtime
state, not add more probe bypasses: TurboMind/API handles, route buffers,
expert bindings, stream/event lifecycle, and eventually TP runtime state must
move outside the per-layer runner.
