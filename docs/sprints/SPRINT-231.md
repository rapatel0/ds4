# Sprint 231 - TP/EP Routed Expert Slice

Date: 2026-05-23
Status: Complete

## Overview

Sprint 230 proved that the separate TP runtime can own all eight GPUs and
update sharded DS4 KV rows at the target `32` slot / `256K` shape. Sprint 231
adds the next missing bounded primitive: expert-parallel routed FFN execution
using the real low-bit TurboMind MXFP4 kernels, still outside the frozen
PP/layer scheduler.

This sprint is a TP/EP runtime gate, not a serving integration. It should make
the EP route distribution, local expert ownership, dispatch/return byte model,
and kernel behavior concrete at the production slot target.

## Goals

- Add a new TP/EP-only routed expert smoke tool.
- Own all eight V100s in one process and enable peer access.
- Model EP8 ownership as `32` experts per GPU from `256` total experts.
- Model `32` active slots with `top_k=6`, or `192` aggregate routes.
- Distribute routes across GPUs and report:
  - routes per GPU;
  - active local experts per GPU;
  - max routes per local expert;
  - worst-rank imbalance;
  - aggregate dispatch bytes and return bytes.
- Run the real TurboMind MXFP4 grouped gated-SiLU and down kernels on every
  GPU for the local routed rows.
- Validate finite output and deterministic repeat output on all GPUs.
- Measure per-rank gate/up latency, down latency, total local EP latency, and
  worst-rank latency.

## Non-Goals

- No PP scheduler changes.
- No generic PP/TP scheduler abstraction.
- No serving integration.
- No full DS4 layer correctness against logits.
- No dense TP attention implementation.
- No MTP.
- No production throughput claim.

## Implementation

1. Add `tools/ds4-v100-tp-ep-expert-smoke.cu`.
2. Reuse the public TurboMind C ABI from
   `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`.
3. Generate deterministic MXFP4 fixtures for local experts.
4. Use `ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens` for fused
   gate/up.
5. Use `ggml_turbomind_mul_mat_grouped_total_tokens` for down projection.
6. Use a fixed EP route plan by default:
   - `256` global experts;
   - `8` GPUs;
   - `32` experts per GPU;
   - `32` slots;
   - `top_k=6`;
   - balanced aggregate routes but non-uniform local expert density.
7. Add a Makefile target for the new tool.
8. Build on the V100 pod with `CUDA_ARCH=sm_70`.
9. Run the smoke against `./libggml-turbomind.so` for:
   - `32` slots / `top_k=6`;
   - a denser diagnostic case if time allows.
10. Copy evidence to
    `logs/from-cluster/sprint231-tp-ep-expert-slice/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-expert-smoke.cu` | TP/EP routed expert slice smoke |
| `Makefile` | build target |
| `docs/sprints/SPRINT-231.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint231-tp-ep-expert-slice/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] New smoke tool is TP/EP-only and does not modify PP scheduler files.
- [x] Smoke uses the real TurboMind MXFP4 grouped gated-SiLU ABI.
- [x] Smoke uses the real TurboMind MXFP4 grouped down ABI.
- [x] Smoke reports route distribution and imbalance across all eight GPUs.
- [x] Smoke reports dispatch bytes and return bytes for `32` slots / `top_k=6`.
- [x] V100 build passes with `CUDA_ARCH=sm_70`.
- [x] V100 smoke passes finite and deterministic repeat checks.
- [x] Smoke reports per-rank and worst-rank latency.
- [x] Evidence is copied to
      `logs/from-cluster/sprint231-tp-ep-expert-slice/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

Build on V100:

```text
cd /workspace/ds4-sprint181
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-expert-smoke
```

Target 32-slot/top-6 run:

```text
./tools/ds4-v100-tp-ep-expert-smoke \
  --lib ./build/turbomind-v100/libggml-turbomind.so \
  --slots 32 --top-k 6 --warmup 5 --iters 30
```

Result:

```text
rank 0 routes 24 active_local_experts 6 max_routes_per_expert 6 total_ms 0.059529
rank 1 routes 24 active_local_experts 6 max_routes_per_expert 6 total_ms 0.059255
rank 2 routes 24 active_local_experts 6 max_routes_per_expert 6 total_ms 0.058982
rank 3 routes 24 active_local_experts 6 max_routes_per_expert 6 total_ms 0.058880
rank 4 routes 24 active_local_experts 6 max_routes_per_expert 6 total_ms 0.058709
rank 5 routes 24 active_local_experts 6 max_routes_per_expert 6 total_ms 0.058607
rank 6 routes 24 active_local_experts 6 max_routes_per_expert 6 total_ms 0.058948
rank 7 routes 24 active_local_experts 6 max_routes_per_expert 6 total_ms 0.249378
tp_ep_expert_smoke slots 32 top_k 6 aggregate_routes 192 dispatch_bytes 1572864 return_bytes 1572864 route_imbalance 1.000000 worst_total_ms 0.249378 repeat_max_abs 0 repeat_bad 0 repeat_nan 0 PASS
```

Denser diagnostic:

```text
./tools/ds4-v100-tp-ep-expert-smoke \
  --lib ./build/turbomind-v100/libggml-turbomind.so \
  --slots 64 --top-k 6 --warmup 5 --iters 30
```

Result:

```text
tp_ep_expert_smoke slots 64 top_k 6 aggregate_routes 384 dispatch_bytes 3145728 return_bytes 3145728 route_imbalance 1.000000 worst_total_ms 0.268049 repeat_max_abs 0 repeat_bad 0 repeat_nan 0 PASS
```

Logs:

- `logs/from-cluster/sprint231-tp-ep-expert-slice/ep32-top6.log`
- `logs/from-cluster/sprint231-tp-ep-expert-slice/ep64-top6-diagnostic.log`

## Risks

- The current TurboMind ABI is process-global enough that repeated `init` /
  `shutdown` semantics may not be safe across all devices. If that appears,
  this sprint should record the exact failure and narrow the next work to
  making the ABI multi-device safe.
- This validates EP kernel behavior and route density, not complete TP/EP
  serving. Full correctness still requires a one-layer TP/EP gate.
- A deterministic repeat check is weaker than a DS4 logits comparison, but it
  is enough for this bounded kernel/lifecycle sprint.

## Decision

Sprint 231 passes as a bounded EP kernel/lifecycle gate. The separate TP/EP
path can run the real TurboMind MXFP4 grouped gated-SiLU and grouped down
kernels on all eight V100s with an EP8 route distribution at the target
`32` slot / `top_k=6` shape. The modeled wire volume is small:
`1.5 MiB` dispatch and `1.5 MiB` return for `192` aggregate routes. The
deterministic repeat check is exact and finite on all ranks.

The main observation is rank skew: ranks `0-6` finish the local EP slice in
about `0.059 ms`, while rank `7` reports `0.249 ms` at the target shape and
`0.268 ms` in the denser diagnostic. The next sprint should carry this
per-rank timing forward into the one-layer TP/EP correctness gate rather than
averaging it away.
