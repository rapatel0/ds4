# Sprint 232 - One-Layer TP/EP Correctness Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 230 proved sharded TP KV row ownership. Sprint 231 proved bounded EP8
routed expert execution with real TurboMind MXFP4 kernels. Sprint 232 combines
those primitives into the first one-layer TP/EP fixture gate.

This sprint should still avoid the frozen PP scheduler. The purpose is to make
the per-layer TP/EP execution boundary concrete enough that the following
sprint can scale from a fixture layer to DS4 layer descriptors and then to all
43 layers.

## Goals

- Add a new TP/EP-only one-layer smoke tool.
- Own all eight V100s in one process.
- Allocate/use the separate TP runtime at the target `32` slots / `256K` /
  F8-KV shape.
- Execute a ratio-4 dense/KV slice for one layer.
- Execute an EP8 routed expert slice for the same active-slot shape:
  - `32` slots;
  - `top_k=6`;
  - `192` aggregate routes;
  - `256` global experts;
  - `32` local experts per GPU.
- Produce a deterministic next-hidden fixture per rank.
- Validate:
  - finite output;
  - deterministic repeat;
  - KV row update correctness;
  - EP route distribution accounting;
  - per-rank latency visibility.
- Report the one-layer timing envelope:
  - dense/KV slice time;
  - EP gate/up time;
  - EP down time;
  - EP worst-rank time;
  - modeled dispatch/return bytes.

## Non-Goals

- No PP scheduler changes.
- No generic PP/TP scheduler abstraction.
- No full DS4 layer descriptor loading.
- No logits comparison yet.
- No serving integration.
- No MTP.
- No production throughput claim.

## Implementation

1. Add `tools/ds4-v100-tp-ep-layer-smoke.cu`.
2. Link it with `ds4_v100_tp_runtime.cu`.
3. Reuse the public TurboMind C ABI for MXFP4 gated-SiLU/down.
4. Reuse the Sprint 230 TP runtime dense/KV API for the ratio-4 layer slice.
5. Reuse/adapt the Sprint 231 EP route plan and kernel invocation.
6. Add a Makefile target and local CUDA guard.
7. Build on the V100 pod with `CUDA_ARCH=sm_70`.
8. Run at `32` slots / `256K` / `top_k=6`.
9. Copy evidence to
   `logs/from-cluster/sprint232-tp-ep-layer-smoke/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-layer-smoke.cu` | one-layer TP/EP fixture gate |
| `ds4_v100_tp_runtime.h` | only if additional TP runtime reporting is needed |
| `ds4_v100_tp_runtime.cu` | only if additional TP runtime reporting is needed |
| `Makefile` | build target |
| `docs/sprints/SPRINT-232.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint232-tp-ep-layer-smoke/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] New smoke tool is TP/EP-only and does not modify PP scheduler files.
- [x] Tool opens the separate TP runtime at `32` slots / `256K` / F8 KV.
- [x] Tool updates/verifies a ratio-4 dense/KV row through the TP runtime.
- [x] Tool runs real TurboMind MXFP4 grouped gated-SiLU and down kernels on all
      eight GPUs.
- [x] Tool reports route distribution, dispatch bytes, return bytes, and
      per-rank latency.
- [x] Tool validates finite deterministic repeat output.
- [x] V100 build passes with `CUDA_ARCH=sm_70`.
- [x] V100 smoke passes at `32` slots / `256K` / `top_k=6`.
- [x] Evidence is copied to
      `logs/from-cluster/sprint232-tp-ep-layer-smoke/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

Build on V100:

```text
cd /workspace/ds4-sprint181
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-layer-smoke
```

Target one-layer fixture:

```text
./tools/ds4-v100-tp-ep-layer-smoke \
  --lib ./build/turbomind-v100/libggml-turbomind.so \
  --slots 32 --top-k 6 --layer 2 --kv-slot 7 --position 1024 \
  --warmup 5 --iters 30
```

Result:

```text
runtime_bytes_per_gpu hidden 524288 kv 3707940864 comp_state 1803550720 scratch 1610612736 total 7122628608
dense_kv_slice layer 2 ratio 4 slot 7 position 1024 attn_row 384 indexer_row 256 attn_row_bytes 65 indexer_row_bytes 17 max_abs 0.000000000 dense_kv_ms 1.078032
tp_ep_layer_smoke slots 32 ctx 262144 top_k 6 aggregate_routes 192 dispatch_bytes 1572864 return_bytes 1572864 route_imbalance 1.000000 worst_gate_ms 0.161143 worst_down_ms 0.082637 worst_ep_ms 0.243780 one_layer_ms 1.321812 repeat_max_abs 0.000000000 repeat_bad 0 repeat_nan 0 PASS
```

Per-rank EP timing:

```text
rank 0 routes 24 ep_ms 0.068164
rank 1 routes 24 ep_ms 0.067891
rank 2 routes 24 ep_ms 0.067482
rank 3 routes 24 ep_ms 0.067072
rank 4 routes 24 ep_ms 0.066731
rank 5 routes 24 ep_ms 0.066423
rank 6 routes 24 ep_ms 0.066219
rank 7 routes 24 ep_ms 0.243780
```

Logs:

- `logs/from-cluster/sprint232-tp-ep-layer-smoke/layer2-ratio4-32slot-top6.log`

## Risks

- This is still a fixture layer, not a DS4 descriptor-driven layer. It proves
  the execution boundary and lifecycle, not model equivalence.
- If rank-7 skew persists from Sprint 231, it may dominate the one-layer timing
  envelope. That should be recorded rather than averaged away.
- If linking the TP runtime and TurboMind ABI in one tool exposes lifecycle
  issues, the sprint should record the exact failure and fix the lifecycle
  boundary before adding more DS4 math.

## Decision

Sprint 232 passes. The separate TP/EP path can now open the target TP runtime,
verify a ratio-4 DS4-style sharded KV row, and run the real TurboMind MXFP4
EP routed expert kernels in the same process at `32` slots / `256K` /
`top_k=6`. This is still a fixture layer, not DS4 descriptor equivalence or
serving, but it proves the combined lifecycle and accounting boundary needed
for the next sprint.

The one-layer envelope is `1.321812 ms`, dominated by the dense/KV fixture API
at `1.078032 ms`; the EP worst rank is `0.243780 ms`. Rank `7` remains the
slow EP rank. Sprint 233 should replace fixture weights/routes with
descriptor-driven layer data for one real DS4 layer while preserving per-rank
timing and the TP/EP-only codepath.
