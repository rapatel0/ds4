# Sprint 252 - TP/EP Descriptor Check Bypass

Date: 2026-05-23
Status: Complete

## Overview

Sprint 251 hoisted dense cache residency, but the all-layer scaffold still did
per-layer dense/control descriptor byte checks. Those checks are useful for
validation and bad for a serving-shaped loop. Sprint 252 adds an opt-in
`--skip-descriptor-checks` mode so validated packs can run the all-layer
scaffold without rereading/checksumming dense and control rows every layer.

This is still not generated-token serving throughput. It is a cleaner
serving-shaped scaffold gate.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add an opt-in descriptor checksum bypass.
- Keep strict descriptor checks as the default.
- Preserve the shared dense cache path from Sprint 251.
- Run a controlled all-layer A/B against Sprint 251 at `32` slots / `256K`.

## Non-Goals

- No PP scheduler changes.
- No production server integration.
- No logits-equivalent output.
- No MTP.
- No TurboMind/API handle hoist yet.

## Design

The new flag:

```text
--skip-descriptor-checks
```

changes only the dense/control descriptor validation phase:

```text
default:
  device-read dense/control descriptor bytes
  checksum them on the owning GPU
  require nonzero descriptor checksum

skip mode:
  count descriptor bytes from the contract
  do not reread/checksum descriptor bytes
  allow descriptor checksum to be zero
```

The full scaffold still parses the contract, verifies row presence, checks KV
slice behavior, runs real TurboMind MXFP4 EP experts, runs cache-backed dense
compose, and checks finite deterministic decode output.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Adds `--skip-descriptor-checks` |
| `docs/sprints/SPRINT-252.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint252-tp-ep-descriptor-bypass/` | V100 evidence |

## Definition Of Done

- [x] Implementation stays in separate TP/EP code.
- [x] No PP scheduler files are modified.
- [x] Tool builds on the V100 pod with `CUDA_ARCH=sm_70`.
- [x] Descriptor checks remain enabled by default.
- [x] `--skip-descriptor-checks` passes the all-layer 10-step gate with
      shared dense cache.
- [x] Evidence records `descriptor_checks=0`.
- [x] Evidence is copied to
      `logs/from-cluster/sprint252-tp-ep-descriptor-bypass/`.
- [x] Status and vision docs are updated.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint252-tp-ep-descriptor-bypass/cluster/all-layer-10step-skip-descriptor-compose.log`
- `logs/from-cluster/sprint252-tp-ep-descriptor-bypass/cluster/all-layer-10step-skip-descriptor-compose-summary.log`
- `logs/from-cluster/sprint252-tp-ep-descriptor-bypass/cluster/all-layer-10step-skip-descriptor.log`

Command shape:

```text
--slots 32 --top-k 6 --kv-slot 7 --position 1024
--warmup 2 --iters 5 --compose-next-hidden --decode-steps 10
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --all-layers
```

10-step all-layer gate:

| Metric | Sprint 251 checks on | Sprint 252 checks off |
|---|---:|---:|
| Passing layers | 43 / 43 | 43 / 43 |
| Descriptor checks | 1 | 0 |
| Sum decode ms/token | 43.753529 | 44.383590 |
| Projected slot-step tok/s | 731.369579 | 720.987187 |
| Per-layer avg decode ms | 1.017524 | 1.032177 |
| Per-layer min decode ms | 0.981924 | 1.000416 |
| Per-layer max decode ms | 1.054658 | 1.221799 |
| Sum EP ms | 11.837140 | 11.784128 |
| Sum dense ms | 7.613544 | 7.933067 |
| Sum compose ms | 24.296721 | 24.659710 |
| Wall ms | 74382.064295 | 46990.435640 |
| Result | PASS | PASS |

The bypass cuts wall time by about `36.8%` versus Sprint 251 because it removes
per-layer descriptor reread/checksum work. The decode proxy is within expected
run-to-run variance and remains dominated by compose/synchronization.

## Follow-Up

A decode-only attempt with descriptor checks off and shared dense cache failed
early with:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:1660: invalid resource handle
```

The controlled A/B kept `--compose-next-hidden` enabled and passed. The
decode-only failure is a smoke-harness path issue to fix before using
decode-only as the standard all-layer benchmark.

## Decision

`--skip-descriptor-checks` is promoted for serving-shaped TP/EP scaffold
measurements after descriptor validation has passed. It should not replace the
strict validation gate. The next residency target is hoisting TurboMind/API
handles and rank buffers so the all-layer loop stops recreating per-layer
expert execution state.
