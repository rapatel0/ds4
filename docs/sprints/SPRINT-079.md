# Sprint 079: Routed MXFP4 Row-Pair Occupancy Probe

## Status

Complete. Row-pair routed MXFP4 kernels are correct and opt-in, but they remain
disabled by default because paired V100 throughput evidence showed a slight
regression against the current one-row routed MXFP4 kernels.

## Overview

Sprint 078 showed that event-ordered stage handoff removes a measured
synchronization bucket but does not materially improve aggregate throughput. The
next practical serving lever is the routed MXFP4 expert hot path. Current
sustained 1M/4-slot decode is about `9.16` generated tok/s with average GPU
utilization around `20%`, far below the hardware target range.

The routed expert kernels still execute scalar source-MXFP4 row reductions: one
CTA computes one gate/up row for one selected route/token, and one CTA computes
one down row for one token. That preserves source layout and correctness, but
it underuses Volta because each CTA performs small dequantized dot products
without tensor-core or packed-integer tiling.

Sprint 079 adds a bounded row-pair variant for the existing grouped routed
MXFP4 batch kernels. Each CTA computes two adjacent output rows when possible,
reusing the same input vector load across two rows and reducing CTA scheduling
overhead. The experiment stays opt-in so the default appliance path remains the
known-good Sprint 076/078 combination unless V100 evidence shows a real win.

## Goals

1. Add row-pair CUDA variants for grouped routed MXFP4 gate/up/SwiGLU and
   down-sum kernels.
2. Wire the variants into the existing batch and pointer-input batch APIs behind
   `DS4_CUDA_MXFP4_ROUTE_ROWS2=1`.
3. Preserve current public APIs, source MXFP4 layout, route selection, and
   default kernel behavior.
4. Extend focused MXFP4 smoke coverage to exercise the opt-in row-pair path.
5. Validate selected-token correctness on V100 for default and opt-in paths.
6. Run sustained V100 A/B for the current practical fixture:
   `ctx=1048576`, `slots=4`, `tokens=16`, `requests=4`,
   `async_pipeline_mode=per-step`.

## Non-Goals

- Native FP4/FP8 tensor-core execution on V100.
- Full HMMA/DP4A rewrite of MXFP4 expert math.
- Persistent MoE kernels or expert bucketing by expert id.
- Shared F8 batching changes.
- Scheduler, MTP, output-head, or stage handoff behavior changes.
- Making the path default without same-fixture throughput evidence.

## Implementation

1. Extend `ds4_cuda.cu`:
   - add row-pair gate/up/SwiGLU kernels for contiguous and pointer-input
     batched routed MXFP4 execution;
   - add a row-pair down-sum kernel;
   - dispatch row-pair kernels only when `DS4_CUDA_MXFP4_ROUTE_ROWS2=1`;
   - fall back to the current one-row kernels for odd tail rows and default
     execution.
2. Keep `ds4_gpu.h` stable:
   - no new public runtime API is required;
   - existing fallback backends remain unchanged because the switch is internal
     to the CUDA implementation.
3. Extend `tests/cuda_v100_mxfp4_moe_smoke.c`:
   - run the pointer-input grouped route path with
     `DS4_CUDA_MXFP4_ROUTE_ROWS2=1`;
   - compare both slots against the existing CPU/per-route reference.
4. Validate:
   - local compile for changed objects and smoke targets where possible;
   - V100 build for `ds4_cuda.o`, replay, selected-token smoke, full scheduler
     smoke, and focused MXFP4 smoke;
   - V100 focused smoke with and without row-pair enabled;
   - V100 selected-token or short replay correctness with row-pair enabled;
   - sustained benchmark default vs row-pair opt-in on the same fixture.

## Definition of Done

- [x] `ds4_cuda.cu` compiles with `CUDA_ARCH=sm_70`.
- [x] Focused MXFP4 MoE smoke passes on V100 for default kernels.
- [x] Focused MXFP4 MoE smoke passes on V100 with
  `DS4_CUDA_MXFP4_ROUTE_ROWS2=1`.
- [x] V100 selected-token or short replay smoke passes with
  `DS4_CUDA_MXFP4_ROUTE_ROWS2=1` and expected token hex `3136`.
- [x] Sustained V100 A/B records generated tok/s, continuation tok/s, latency,
  routed/stage timing where available, token matches, and GPU utilization for:
  - default routed MXFP4 kernels;
  - `DS4_CUDA_MXFP4_ROUTE_ROWS2=1`.
- [x] Sprint report records whether row-pair remains opt-in or becomes default.
- [x] Vision document is updated with the measured result.
- [x] Artifacts are committed.

## Results

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Avg GPU util | Max GPU util | Token match |
|---|---:|---:|---:|---:|---:|---:|
| Default routed MXFP4 | `9.055694` | `8.489713` | `7065.558` | `19.911%` | `40.000%` | `4/4` |
| `DS4_CUDA_MXFP4_ROUTE_ROWS2=1` | `9.035946` | `8.471200` | `7081.063` | `19.790%` | `41.000%` | `4/4` |

The row-pair path preserved correctness but regressed generated tok/s by
`0.22%` and continuation tok/s by `0.22%`. The result confirms that halving CTA
row count is not enough to fix the MXFP4 hot path; the next useful kernel sprint
needs a more substantial execution-shape change, such as route/expert tiling or
a packed low-bit dot-product path.

## Decision Rule

- Make row-pair routed MXFP4 default only if it improves generated tok/s by at
  least `3%` on the 1M/4-slot per-step fixture without selected-token or
  sustained-token correctness regression.
- Keep it opt-in if it is correct but improves less than `3%`.
- Disable or remove it if it regresses correctness or causes unstable V100
  execution.

## Risks

- Computing two rows per CTA may reduce parallelism enough to offset input-load
  reuse.
- The main bottleneck may be source-MXFP4 dequant math rather than CTA launch or
  input load overhead.
- Odd row tails and route/token indexing must preserve exact existing layout.
- The row-pair path may help the focused smoke but not the full 43-layer replay
  if another stage bucket dominates.

## Security

No new external serving surface. The experiment is an internal CUDA kernel
selection switch.
