# TEMP Status Report 002

Date: 2026-05-21

## Topline

- Latest committed sprint: `b363afb` (`replay: test layer wavefront diagnostic`).
- Worktree was clean before this report was created.
- The DS4-Flash V100 appliance path is correct enough to run production-shaped
  replay and HTTP serving benchmarks with the copied TurboMind kernels.
- MTP verify/commit exists, but current performance work is measuring MTP off
  because exact verification still computes the target token and is not a
  throughput win.
- The latest work closed the layer-parallel slot-scheduling line. Layer-span
  execution is correct, but in-stage layer-wavefront scheduling is materially
  slower than the existing per-step event-handoff path.

## Current Throughput Picture

There are two useful categories of numbers:

1. Latest clean same-prompt A/B harness, which is good for comparing a local
   scheduler change.
2. Historical best served throughput by context tier, which is better for
   understanding the current ceiling.

Latest clean Sprint 168 A/B at `ctx=256K`, `slots=16`, `tokens/request=16`,
MTP off:

| Mode | Status | Generated tok/s | Continuation tok/s | Decision |
|---|---:|---:|---:|---|
| per-step event-handoff control | `16/16` OK | `32.906564` | `30.849903` | keep |
| layer-wavefront chunk 2 | `16/16` OK | `26.126248` | `24.493358` | reject |
| layer-wavefront chunk 4 | `16/16` OK | `19.175887` | `17.977394` | reject |

Best observed served tiers from the sprint history:

| Context | Slots | Generated tok/s | Continuation tok/s | Notes |
|---:|---:|---:|---:|---|
| 16K | 256 | `61.223893` | `57.397400` | Sprint 146 control repeat |
| 32K | 128 | `60.130047` | about `56.37` | Sprint 139 best 128-slot run |
| 64K | 64 | `57.322945` | about `53.74` | Sprint 136 |
| 128K | 32 | `52.840889` | about `49.54` | Sprint 135 |
| 256K | 16-32 | `63-65` | `62-64` | Sprint 159 sustained/cap tests; not the same harness as Sprint 168 |
| 1M | 4 | `21.771077` | `20.410385` | Sprint 119 |
| 1M | 1 | `3.600787` | `3.375738` | older sustained single-slot result; needs a fresh rerun |

Interpretation: the appliance is usable and correct, but the measured ceiling is
still tens of tok/s, not the `1k-2k` aggregate tok/s practical-serving target.
The gap is not from VRAM fit; it is from the execution shape failing to keep the
V100 tensor cores busy during routed MoE decode.

## What Has Been Tested

- Copied TurboMind kernels are in the repo and are the production expert hot
  path. TC-grid remains useful prior art, but it is not the current production
  serving path.
- FP16/BF16 issue is handled by converting BF16 storage to V100-compatible
  compute paths. V100 does not do native BF16 math.
- KV caching is implemented and used through the stage-owned KV/compressed-KV
  runtime. Long-context runs are not recomputing prompt state from scratch per
  decode token.
- MTP verify and exact commit have shipped as correctness features. They are not
  currently throughput-positive.
- Software-pipelined fused MXFP4 gate/up variants were explored. They improved
  isolated kernels in places but did not move served throughput.
- Host stream-per-expert software pipelining was tested and rejected. The host
  orchestration/readback cost erased the potential gain.
- Down-reduce epilogue fusion and route-row reduce variants were correct but
  flat at served level.
- Fixed-shape routed executors proved the kernel can be faster when the runtime
  presents dense route shapes, but real HTTP serving often reaches the FFN as
  tiny six-route shapes.
- CUDA Graph wrapping failed because the current routed-FFN path still uses
  legacy default-stream behavior that cannot be captured cleanly.
- A one-layer TP2 scheduler overlay is correct and fits in 32 GB V100 memory,
  but the synchronous overlay is slower. TP remains promising only if redesigned
  as persistent peer ownership / broader TP-EP scheduling, not per-layer copy
  overlays.

## Latest Sprint Results

Sprint 167 added layer-span scheduler APIs:

- `ds4_v100_stage_scheduler_decode_token_layer_span()`
- `ds4_v100_stage_scheduler_decode_hc_layer_span()`

V100 validation passed with segmented stage execution matching the full-stage
path inside the observed TurboMind repeat-run drift envelope:

- `max_abs_slot0=0.01612854`
- `max_abs_slot1=0.0221862793`
- threshold `0.03`

Sprint 168 used those APIs for an opt-in in-stage layer-wavefront worker:

- `DS4_V100_ASYNC_LAYER_WAVEFRONT=1`
- `DS4_V100_ASYNC_LAYER_WAVEFRONT_CHUNK=N`

It was correct but slower, so it stays diagnostic-only.

## Current Assessment

The next useful try is not another queue-level coalescing variant. Those have
been tested enough to show the tradeoff: they can create denser route shapes,
but they destroy stage overlap or add host scheduling overhead.

The best remaining candidates are:

1. A DS4-specific persistent routed-FFN executor boundary that keeps packed
   MXFP4 dequant, gate/up HMMA, gated activation, down HMMA, and weighted
   reduce inside one long-lived execution boundary.
2. A broader TP/EP topology experiment that creates dense fused-kernel shapes
   without per-layer synchronous payload copies.

The next implementation should produce hardware evidence against one of those
two paths and continue reporting prompt/prefill and continuation/decode tok/s
separately.
